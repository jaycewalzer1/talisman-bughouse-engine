"""Out-of-step self-play for the custom engine (docs/custom_engine_spec.md section 3).

    python -m arena.selfplay --games 200 --workers 1 --out runs/selfplay/it01 \
        --model third_party/hivemind/engine/models/hivemind-clock-v2-fp32.onnx --binary hivemind-clock

Both teams are one free-mode ``HivemindTeamController`` (real clocks, FreePass, no sit rules, think time =
remaining clock / k with k drawn per game and team from ``--k-range``, a jitter factor per search from
``--jitter``, one machine per team) on the same network. Every search records a training sample; at game end the
samples get their game id, outcome (+1/0/-1 for the sample's team, flags included, draws and move-limit games 0)
and moves left (moves landed after the sample's search started), and go into chunks under ``<out>/training_data``
(what ``train_loop.py --mode rl --selfplay-dir <out>`` and ``convert_selfplay_data.py`` read). Games that crash are
dropped from the data.

Spec v2 (docs/custom_engine_spec_v2.md): chunks are HVM6 by default (``--shard-version 5`` writes the v1 format):
each sample carries piece-arrival counts (pieces landing in each pocket within 5 s after its search started) and
masked think_gain labels (filled later by ``tools/think_labels.py``); each chunk gets a ``.pos.jsonl`` sidecar with
the positions. Every ``position`` carries the per-board elapsed times (``--no-elapsed`` turns that off). At most one
sample per seat per unchanged position in a row is kept; the dropped repeats are counted per game.
``--think-head`` scales each move's node budget by the network's think_gain (off by default).

Spec v3 (docs/custom_engine_spec_v3.md, audit WP-C), ``--regime v3`` (the default is ``v2``, the behaviour
above; cluster/selfplay.sbatch runs v3):
* base time per game from ``--base-mix`` (60/120/180/300 s at 0.25/0.25/0.35/0.15), increment 0, and the node
  budget decoupled from it: per team a think target T is drawn log-uniformly from ``--think-target-ms`` and
  k = base / T, so the move-1 budget does not depend on the base. ``max_plies`` scales with base / 60 s.
* ``--time-odds``: with p 0.25 one team (either, either sign) starts with base + one of +-5/15/30/60 s, with p 0.1
  both teams start low (independent draws from -5/15/30/60 s); start clocks never below a quarter of base.
* time accounting: searched nodes charged, a ``--min-think-ms`` floor, a per-seat pace mixture with an additive
  lag (``--lag-ms``) in place of the uniform jitter, the ``pace`` suffix and ``TreeMoveMs`` from the pace EMA.
* exploration: root noise on recorded searches, a move temperature for the first ``--temperature-plies`` plies,
  playout-cap randomisation (``--pcr-p`` of searches at ``--pcr-nodes`` recorded, the rest unrecorded).
* samples: one per seat per (FENs, ``--clock-bucket-ms`` clock bucket), the sit->move search always kept, each
  with the sit length that followed (sidecar ``sit_ms``) and the spec v3 sidecar fields.
* a sitting seat re-looks only every ``--free-sit-ms`` or when its own board changes, with a small budget.
Each ``v3`` setting has its own flag; a flag given explicitly wins over the regime default.

Engine processes persist across games (one per team in this process, or per worker with ``--workers``).

Writes ``<out>/training_data/chunk_<run id>_NNNNNN.hvm``, ``<out>/games/game_NNNN.jsonl`` (unless
``--no-events``) and ``<out>/summary.json``, and prints a results table.
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import random
import sys
import time
import traceback
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed

from arena.interfaces import SEATS, TEAM_OF
from arena.openings import OPENING_SETS, paired_schedule
from arena.recorder import DEFAULT_SAMPLES_PER_CHUNK, ShardWriter, finish_game
from arena.stats import summarize_game


# What each v3 setting is under --regime v2 (the controller's own defaults).
_V2_VALUES = {"time_odds": False, "charge": None, "min_think_ms": None, "pace": None, "pace_mixture": False,
              "lag_ms": None, "noise": None, "temperature": None, "pcr_p": None, "dedupe": None,
              "sit_recheck": None, "scale_max_plies": False}


def _onoff(text: str) -> bool:
    t = text.strip().lower()
    if t in ("on", "1", "true", "yes"):
        return True
    if t in ("off", "0", "false", "no"):
        return False
    raise argparse.ArgumentTypeError(f"expected on/off, got {text!r}")


def parse_mix(text: str):
    """"60000:0.25,120000:0.75" -> ((60000, 0.25), (120000, 0.75)); "off" -> "off"."""
    if text.strip().lower() in ("off", "none", ""):
        return "off"
    out = []
    for part in text.split(","):
        b, _, w = part.partition(":")
        try:
            out.append((int(b), float(w or 1)))
        except ValueError:
            raise argparse.ArgumentTypeError(f"--base-mix needs MS:WEIGHT,..., got {text!r}") from None
    if not out or any(b <= 0 or w < 0 for b, w in out) or sum(w for _, w in out) <= 0:
        raise argparse.ArgumentTypeError(f"--base-mix needs positive times and weights, got {text!r}")
    return tuple(out)


def parse_range(text: str, name: str, cast=float) -> tuple | None:
    """"lo,hi" -> (lo, hi); "off" -> None."""
    if text is None or text.strip().lower() in ("off", "none", ""):
        return None
    parts = [p for p in text.replace(":", ",").split(",") if p.strip()]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"{name} must be LO,HI, got {text!r}")
    lo, hi = cast(parts[0]), cast(parts[1])
    if not 0 < lo <= hi:
        raise argparse.ArgumentTypeError(f"{name} must have 0 < LO <= HI, got {text!r}")
    return lo, hi


# Spec v3 D2: self-play base times and weights; time odds (audit self-play 1).
BASE_MIX = ((60000, 0.25), (120000, 0.25), (180000, 0.35), (300000, 0.15))
ODDS_OFFSETS_MS = (5000, 15000, 30000, 60000)
ODDS_ONE_TEAM_P = 0.25
ODDS_BOTH_LOW_P = 0.10
MIN_START_FRACTION = 0.25  # a start clock is never below this share of base

# Defaults of the v3 regime: (dest, value). A flag the user gives explicitly keeps its own value.
V3_DEFAULTS = {
    "base_mix": BASE_MIX, "think_target_ms": (200.0, 1000.0), "time_odds": True, "charge": "searched",
    "min_think_ms": 100.0, "pace": True, "pace_mixture": True, "lag_ms": 50.0, "noise": True,
    "temperature": 1.0, "pcr_p": 0.25, "dedupe": "bucket", "sit_recheck": True, "scale_max_plies": True,
}
V3_CONTROLLER = ("charge", "min_think_ms", "pace", "lag_ms", "noise", "temperature", "temperature_plies", "pcr_p",
                 "pcr_nodes", "dedupe", "clock_bucket_ms", "sit_recheck", "recheck_nodes")


def draw_base_ms(rng: random.Random, mix) -> int:
    return int(rng.choices([b for b, _ in mix], [w for _, w in mix])[0])


def draw_time_odds(rng: random.Random, base_ms: int) -> dict:
    """{team: start offset ms} for one game: p 0.25 one team +-5/15/30/60 s, p 0.1 both teams low, else none.
    Offsets are clamped so a start clock is at least ``MIN_START_FRACTION`` of base."""
    floor = -int(base_ms * (1 - MIN_START_FRACTION))
    u = rng.random()
    if u < ODDS_ONE_TEAM_P:
        team = rng.choice(("T1", "T2"))
        off = rng.choice(ODDS_OFFSETS_MS) * rng.choice((1, -1))
        offsets = {team: off, ("T2" if team == "T1" else "T1"): 0}
    elif u < ODDS_ONE_TEAM_P + ODDS_BOTH_LOW_P:
        offsets = {t: -rng.choice(ODDS_OFFSETS_MS) for t in ("T1", "T2")}
    else:
        offsets = {"T1": 0, "T2": 0}
    return {t: max(floor, o) for t, o in offsets.items()}


def controller_kw(spec: dict, team: str | None = None) -> dict:
    kw = {"free": True, "record": True, "k_range": spec["k_range"], "jitter": spec["jitter"],
          "free_sit_ms": spec["free_sit_ms"], "min_nodes": spec["nodes_range"][0],
          "max_nodes": spec["nodes_range"][1], "ms_per_node": spec["ms_per_node"],
          "think_head": spec.get("think_head", False), "send_elapsed": spec.get("send_elapsed", True)}
    v3 = spec.get("v3") or {}
    kw.update({k: v3[k] for k in V3_CONTROLLER if v3.get(k) is not None})
    if v3.get("pace_mixture"):
        kw["pace_mixture"] = v3["pace_mixture"]
    if spec.get("think_head") and spec.get("regime") == "v3":
        kw["think_head_norm"] = True
    if team is not None:
        if spec.get("k"):
            k = spec["k"][team]
            kw["k_range"] = (k, k)
        if spec.get("start_offsets"):
            kw["start_offset_ms"] = spec["start_offsets"][team]
    return kw


# Engine processes kept across games: {(team, binary, model): process}, one set per Python process (so one per
# pool worker). The cache holds one reference; each game takes one more for its controller.
_PROCS: dict = {}


def _engine(team: str, spec: dict, process_factory=None):
    key = (team, spec.get("binary"), spec.get("model"))
    proc = _PROCS.get(key)
    if proc is None or getattr(proc, "closed", False):
        if process_factory is not None:
            proc = process_factory(team)
        else:
            from arena.players.hivemind import HivemindProcess, resolve_binary
            proc = HivemindProcess(resolve_binary(spec["binary"]), spec["model"])
        _PROCS[key] = proc
    proc.acquire()
    return proc


def close_engines() -> None:
    """Release the cached engine processes (end of a run, or after a game that crashed)."""
    for proc in list(_PROCS.values()):
        try:
            proc.release()
        except Exception:  # noqa: BLE001
            pass
    _PROCS.clear()


def build_players(spec: dict, process_factory=None) -> tuple[dict, list]:
    """{seat: player} for two free-mode controllers; also the controllers. ``process_factory(team)`` replaces the
    engine process (tests pass a stub). Processes come from the per-process cache, so they outlive the game."""
    from arena.players.hivemind import HivemindTeamController

    players, ctls = {}, []
    for i, team in enumerate(("T1", "T2")):
        proc = _engine(team, spec, process_factory)
        ctl = HivemindTeamController(team, proc, seed=spec["seed"] + i, **controller_kw(spec, team))
        ctls.append(ctl)
        for s in SEATS:
            if TEAM_OF[s] == team:
                players[s] = ctl.player(s)
    return players, ctls


def make_state(spec: dict):
    """The game's BughouseState, with per-seat start clocks under time odds."""
    from arena.state import BughouseState

    offsets = spec.get("start_offsets") or {}
    starts = {s: spec["base_ms"] + offsets.get(TEAM_OF[s], 0) for s in SEATS}
    if all(v == spec["base_ms"] for v in starts.values()):
        return BughouseState(spec["base_ms"], 0, fens=spec["fens"])
    if "start_clocks_ms" in inspect.signature(BughouseState).parameters:
        return BughouseState(spec["base_ms"], 0, fens=spec["fens"], start_clocks_ms=starts)
    st = BughouseState(spec["base_ms"], 0, fens=spec["fens"])  # older state.py: set the clocks directly
    st.clocks_ms.update(starts)
    return st


def play_one(spec: dict, process_factory=None) -> tuple[dict, list[bytes], list[dict]]:
    """Play one self-play game. Returns its summary, its finished samples (empty when the game crashed) and their
    sidecar position dicts."""
    from arena.sim import Simulator

    t0 = time.perf_counter()
    meta = {k: spec[k] for k in ("game_id", "game_num", "opening_id", "opening")}
    sim = None
    try:
        players, ctls = build_players(spec, process_factory)
        sim = Simulator(make_state(spec), players, game_id=spec["game_id"],
                        setup="selfplay-free", max_plies=spec["max_plies"])
        result = sim.run()
    except Exception as exc:  # noqa: BLE001 - one bad game should not kill the batch
        close_engines()  # an engine in an unknown state is not reused
        return {**meta, "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}, [], []
    for c in ctls:
        c.finish(sim.now)
    events = list(sim.events)
    samples = [s for c in ctls for s in c.samples]
    finished, metas = ([], []) if result.reason == "aborted" else finish_game(
        samples, result.result, spec["game_num"], events, version=spec.get("shard_version", 6), with_meta=True)
    if spec.get("out") and not spec.get("no_events"):
        gdir = os.path.join(spec["out"], "games")
        os.makedirs(gdir, exist_ok=True)
        with open(os.path.join(gdir, f"game_{spec['index']:04d}.jsonl"), "w") as f:
            for e in events:
                f.write(json.dumps(e, separators=(",", ":")) + "\n")
    summary = summarize_game(events)
    summary.pop("seats", None)
    summary.update(meta)
    summary["k"] = {c.team: round(c.clock_k, 1) for c in ctls}
    summary["searches"] = {c.team: c.searches for c in ctls}
    summary["samples"] = len(finished)
    summary["repeats_dropped"] = sum(c.repeats_dropped for c in ctls)
    if spec.get("think_head"):
        looks = sum(c.quick_looks for c in ctls)
        missing = sum(c.think_head_missing for c in ctls)
        summary["think_head"] = {"looks": looks, "missing": missing, "mean_p": round(
            sum(c.think_gain_sum for c in ctls) / (looks - missing), 3) if looks > missing else None}
    summary["sit_spells"] = sum(1 for e in events if e["type"] == "stall" and e.get("active"))
    summary["no_root"] = sum(c.no_root_results for c in ctls)
    if spec.get("regime") == "v3":
        summary["base_ms"] = spec["base_ms"]
        summary["start_offsets"] = spec["start_offsets"]
        summary["think_target_ms"] = spec["think_target_ms"]
        for k in ("unrecorded_fast", "transitions", "rechecks", "rechecks_skipped", "nodes_requested",
                  "nodes_searched"):
            summary[k] = sum(getattr(c, k) for c in ctls)
        summary["stalls"] = [st for c in ctls for st in c.stalls]
    summary["wall_s"] = round(time.perf_counter() - t0, 2)
    return summary, finished, metas


def make_specs(args) -> list[dict]:
    specs = []
    for i, (pair, _swap) in enumerate(paired_schedule(args.games, args.seed, OPENING_SETS[args.openings])):
        spec = {
            "index": i + 1, "game_id": f"s{args.seed}g{i + 1:05d}", "game_num": args.game_id_offset + i + 1,
            "opening_id": pair.id, "opening": pair.name, "fens": pair.fens, "seed": args.seed * 100003 + 2 * i,
            "base_ms": args.base_ms, "max_plies": args.max_plies, "k_range": args.k_range, "jitter": args.jitter,
            "free_sit_ms": args.free_sit_ms, "nodes_range": args.nodes_range, "ms_per_node": args.ms_per_node,
            "binary": args.binary, "model": args.model, "out": args.out, "no_events": args.no_events,
            "shard_version": args.shard_version, "think_head": args.think_head, "send_elapsed": not args.no_elapsed,
            "regime": args.regime,
        }
        spec.update(v3_spec(args, i))
        specs.append(spec)
    return specs


def v3_spec(args, i: int) -> dict:
    """The per-game spec v3 draws (base time, think targets and k, time odds) from their own seeded stream, and
    the controller settings. Under ``--regime v2`` with no v3 flag given this changes nothing: base and
    ``max_plies`` as given, k drawn by the controller from ``--k-range``, no offsets, controller defaults."""
    rng = random.Random(f"selfplay-v3:{args.seed}:{i}")
    base = draw_base_ms(rng, args.base_mix) if args.base_mix else args.base_ms
    out: dict = {"base_ms": base, "start_offsets": {"T1": 0, "T2": 0}, "think_target_ms": None, "k": None}
    if args.think_target_ms:
        lo, hi = args.think_target_ms
        targets = {t: math.exp(rng.uniform(math.log(lo), math.log(hi))) for t in ("T1", "T2")}
        out["think_target_ms"] = {t: round(v, 1) for t, v in targets.items()}
        out["k"] = {t: base / targets[t] for t in targets}
    if args.time_odds:
        out["start_offsets"] = draw_time_odds(rng, base)
    if args.scale_max_plies:
        out["max_plies"] = max(1, int(round(args.max_plies * base / 60000)))
    ctl = {k: getattr(args, k) for k in V3_CONTROLLER if getattr(args, k) is not None}
    if args.pace_mixture:
        from arena.players.hivemind import DEFAULT_PACE_MIXTURE
        ctl["pace_mixture"] = DEFAULT_PACE_MIXTURE
    out["v3"] = ctl
    return out


def results(summaries: list[dict], writer: ShardWriter | None = None) -> dict:
    ok = [s for s in summaries if "error" not in s]
    n = len(ok) or 1
    searches = sum(sum(s["searches"].values()) for s in ok)
    out = {
        "games": len(ok), "errors": len(summaries) - len(ok),
        "results": dict(Counter(s["result"] for s in ok)),
        "reasons": dict(Counter(s["reason"] for s in ok)),
        "flag_rate": round(sum(s["flag"] for s in ok) / n, 3),
        "mean_plies": round(sum(s["plies"] for s in ok) / n, 1),
        "mean_duration_s": round(sum(s["duration_ms"] for s in ok) / n / 1000, 1),
        "searches": searches,
        "sit_spells_per_search": round(sum(s["sit_spells"] for s in ok) / searches, 3) if searches else 0.0,
        "samples": sum(s["samples"] for s in ok),
        "repeats_dropped": sum(s.get("repeats_dropped", 0) for s in ok),
        "no_root": sum(s.get("no_root", 0) for s in ok),
        "chunks": [os.path.basename(p) for p in writer.paths] if writer else [],
        "channels": writer.channels if writer else None,
    }
    if any("nodes_requested" in s for s in ok):
        for k in ("unrecorded_fast", "transitions", "rechecks", "rechecks_skipped", "nodes_requested",
                  "nodes_searched"):
            out[k] = sum(s.get(k, 0) for s in ok)
        out["base_ms"] = dict(Counter(str(s["base_ms"]) for s in ok))
        out["time_odds_games"] = sum(1 for s in ok if any(s["start_offsets"].values()))
        out["stalls"] = stall_stats([st for s in ok for st in s.get("stalls", [])])
    return out


def stall_stats(stalls: list[dict]) -> dict:
    """Per sit spell (audit self-play 4(d)): how many, share behind on the diagonal, share with the partner's
    opponent on turn, share in check, mean legal move count."""
    n = len(stalls)
    if not n:
        return {"n": 0}
    return {"n": n,
            "diagonal_behind": round(sum(st["diagonal_ms"] < 0 for st in stalls) / n, 3),
            "mean_diagonal_ms": round(sum(st["diagonal_ms"] for st in stalls) / n),
            "partop_on_turn": round(sum(bool(st["partop_on_turn"]) for st in stalls) / n, 3),
            "in_check": round(sum(bool(st["in_check"]) for st in stalls) / n, 3),
            "mean_legal": round(sum(st["legal"] for st in stalls) / n, 1)}


def format_results(r: dict) -> str:
    lines = [f"games {r['games']} ({r['errors']} errors)   samples {r['samples']}   channels {r['channels']}",
             f"results {r['results']}   reasons {r['reasons']}",
             f"flag rate {r['flag_rate']:.3f}   mean plies {r['mean_plies']}   mean length {r['mean_duration_s']} s",
             f"searches {r['searches']}   sit spells per search {r['sit_spells_per_search']:.3f}   chunks {len(r['chunks'])}",
             f"repeated-position samples dropped {r.get('repeats_dropped', 0)}   no-root answers {r.get('no_root', 0)}"]
    if "nodes_requested" in r:
        ratio = r["nodes_searched"] / r["nodes_requested"] if r["nodes_requested"] else 0.0
        lines += [f"base times {r['base_ms']}   time-odds games {r['time_odds_games']}",
                  f"unrecorded fast searches {r['unrecorded_fast']}   sit->move samples {r['transitions']}   "
                  f"re-looks {r['rechecks']} (skipped wakes {r['rechecks_skipped']})",
                  f"nodes searched / requested {ratio:.2f}   stalls {r['stalls']}"]
    return "\n".join(lines)


def run(args, process_factory=None) -> dict:
    os.makedirs(args.out, exist_ok=True)
    run_id = args.run_id or f"{os.path.basename(os.path.normpath(args.out))}-s{args.seed}"
    writer = ShardWriter(os.path.join(args.out, "training_data"), run_id, samples_per_chunk=args.samples_per_chunk,
                         version=args.shard_version)
    specs = make_specs(args)
    summaries: list[dict] = []
    t0 = time.perf_counter()

    def done(summary: dict, samples: list[bytes], metas: list[dict]) -> None:
        summaries.append(summary)
        writer.append(samples, None if args.no_positions else metas)
        tag = summary.get("error") or (f"{summary['result']} by {summary['reason']} in {summary['plies']} plies, "
                                       f"{summary['samples']} samples ({summary['repeats_dropped']} repeats "
                                       f"dropped), k {summary['k']}")
        print(f"[{len(summaries)}/{len(specs)}] {summary['game_id']} {tag}", flush=True)

    try:
        if args.workers <= 1 or process_factory is not None:
            for s in specs:
                done(*play_one(s, process_factory))
        else:
            with ProcessPoolExecutor(max_workers=args.workers) as ex:
                for fut in as_completed([ex.submit(play_one, s) for s in specs]):
                    done(*fut.result())
    finally:
        close_engines()
    writer.close()
    summaries.sort(key=lambda s: s["game_id"])
    cfg = {k: v for k, v in vars(args).items()}
    report = {"config": cfg, "run_id": run_id, "wall_s": round(time.perf_counter() - t0, 1),
              "results": results(summaries, writer), "games": summaries}
    with open(args.summary or os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(report, f, indent=2)
    return report


def parse_args(argv=None):
    from arena.players.hivemind import BINARY_NAMES, ENGINE_DIR, MS_PER_NODE

    ap = argparse.ArgumentParser(prog="python -m arena.selfplay", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--workers", type=int, default=1, help="parallel games (processes); 1 on the shared Mac")
    ap.add_argument("--out", required=True, help="output dir, e.g. runs/selfplay/it01")
    ap.add_argument("--model", default=os.path.join(ENGINE_DIR, "models", "hivemind-clock-v2-fp32.onnx"),
                    help="ONNX network (the 43-plane spec v2 clock network; a 41-plane v1 network also loads with the ONNX Runtime backend only, its elapsed planes dropped)")
    ap.add_argument("--binary", default="hivemind-clock",
                    help=f"engine: one of {', '.join(BINARY_NAMES)} or a path (default hivemind-clock)")
    ap.add_argument("--k-range", type=lambda t: parse_range(t, "--k-range"), default=(60.0, 300.0),
                    help="think time = remaining clock / k, k drawn per game and team from LO,HI (default 60,300)")
    ap.add_argument("--nodes-range", type=lambda t: parse_range(t, "--nodes-range", int), default=(16, 1600),
                    help="clamp on nodes per search, LO,HI (default 16,1600)")
    ap.add_argument("--jitter", type=lambda t: parse_range(t, "--jitter"), default=(0.5, 2.0),
                    help="think-time factor per search drawn from LO,HI, both teams (default 0.5,2.0; 'off' = 1)")
    ap.add_argument("--free-sit-ms", type=int, default=500, help="longest wait after a pass (default 500)")
    ap.add_argument("--ms-per-node", type=float, default=MS_PER_NODE, help="cost model, ms of clock per node")
    ap.add_argument("--base-ms", type=int, default=60000)
    ap.add_argument("--max-plies", type=int, default=400)
    ap.add_argument("--openings", default="train", choices=tuple(OPENING_SETS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--game-id-offset", type=int, default=0, help="added to game numbers (gameId in the samples)")
    ap.add_argument("--run-id", default=None, help="chunk name tag (default <out basename>-s<seed>)")
    ap.add_argument("--samples-per-chunk", type=int, default=DEFAULT_SAMPLES_PER_CHUNK)
    ap.add_argument("--no-events", action="store_true", help="do not write per-game event logs")
    ap.add_argument("--summary", default=None, help="summary file (default <out>/summary.json)")
    ap.add_argument("--shard-version", type=int, choices=(5, 6), default=6,
                    help="chunk format: 6 = HVM6 with arrival / think_gain block (spec v2, default), 5 = v1 HVM5")
    ap.add_argument("--no-positions", action="store_true",
                    help="do not write the <chunk>.pos.jsonl sidecars (tools/think_labels.py needs them)")
    ap.add_argument("--think-head", action="store_true",
                    help="scale each move's budget by 0.5 + 1.5 p, p = think_gain from a 16-node look (spec v2 B)")
    ap.add_argument("--no-elapsed", action="store_true",
                    help="do not send per-board elapsed times in 'position' (v1 behaviour)")
    v3 = ap.add_argument_group("spec v3 regime (defaults apply with --regime v3; an explicit flag wins)")
    v3.add_argument("--regime", choices=("v2", "v3"), default="v2",
                    help="v2: the behaviour before spec v3 (default); v3: the audit's self-play regime")
    v3.add_argument("--base-mix", type=parse_mix, default=None,
                    help="base times and weights per game, e.g. 60000:0.25,120000:0.25,180000:0.35,300000:0.15 "
                         "('off' = --base-ms)")
    v3.add_argument("--think-target-ms", type=lambda t: parse_range(t, "--think-target-ms"), default=None,
                    help="move-1 think time per team, log-uniform LO,HI; k = base / target (v3: 200,1000)")
    v3.add_argument("--time-odds", type=_onoff, default=None, help="on/off (v3: on)")
    v3.add_argument("--charge", choices=("requested", "searched"), default=None, help="v3: searched")
    v3.add_argument("--min-think-ms", type=float, default=None, help="think-time floor in clock ms (v3: 100)")
    v3.add_argument("--pace", type=_onoff, default=None, help="pace EMA suffix and TreeMoveMs from it (v3: on)")
    v3.add_argument("--pace-mixture", type=_onoff, default=None,
                    help="per-seat pace mixture instead of --jitter (v3: on)")
    v3.add_argument("--lag-ms", type=float, default=None, help="mean additive lag per search (v3: 50)")
    v3.add_argument("--noise", type=_onoff, default=None, help="root noise on recorded searches (v3: on)")
    v3.add_argument("--temperature", type=float, default=None, help="move temperature (v3: 1.0)")
    v3.add_argument("--temperature-plies", type=int, default=20, help="temperature while plies < N (default 20)")
    v3.add_argument("--pcr-p", type=float, default=None, help="playout-cap share of full searches (v3: 0.25)")
    v3.add_argument("--pcr-nodes", type=int, default=400, help="nodes of a full search (default 400)")
    v3.add_argument("--dedupe", choices=("position", "bucket"), default=None, help="v3: bucket")
    v3.add_argument("--clock-bucket-ms", type=int, default=5000, help="dedupe clock bucket (default 5000)")
    v3.add_argument("--sit-recheck", type=_onoff, default=None, help="throttled sitter re-looks (v3: on)")
    v3.add_argument("--recheck-nodes", type=int, default=32, help="nodes of a sitter re-look (default 32)")
    v3.add_argument("--scale-max-plies", type=_onoff, default=None,
                    help="max plies x base / 60 s (v3: on)")
    args = ap.parse_args(argv)
    for k, v in V3_DEFAULTS.items():
        if getattr(args, k) is None:
            setattr(args, k, v if args.regime == "v3" else (None if k in ("base_mix", "think_target_ms") else
                                                             _V2_VALUES.get(k)))
    if args.base_mix == "off":
        args.base_mix = None
    if args.jitter is None:
        args.jitter = (1.0, 1.0)
    if args.k_range is None or args.nodes_range is None:
        ap.error("--k-range and --nodes-range cannot be off")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    report = run(args)
    print()
    print(format_results(report["results"]))
    print(f"\nwrote {args.summary or args.out + '/summary.json'} and {report['results']['samples']} samples ({report['wall_s']:.0f}s)")
    return 1 if report["results"]["games"] == 0 else 0


if __name__ == "__main__":
    sys.exit(main())
