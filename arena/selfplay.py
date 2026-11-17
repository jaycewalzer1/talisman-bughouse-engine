"""Out-of-step self-play for the custom engine (docs/custom_engine_spec.md section 3).

    python -m arena.selfplay --games 200 --workers 1 --out runs/selfplay/it01 \
        --model third_party/hivemind/engine/models/hivemind-clock-fp32.onnx --binary hivemind-clock

Both teams are one free-mode ``HivemindTeamController`` (real clocks, FreePass, no sit rules, think time =
remaining clock / k with k drawn per game and team from ``--k-range``, a jitter factor per search from
``--jitter``, one machine per team) on the same network. Every search records a training sample; at game end the
samples get their game id, outcome (+1/0/-1 for the sample's team, flags included, draws and move-limit games 0)
and moves left, and go into HVM5 chunks under ``<out>/training_data`` (what ``train_loop.py --mode rl
--selfplay-dir <out>`` and ``convert_selfplay_data.py`` read). Games that crash are dropped from the data.

Writes ``<out>/training_data/chunk_<run id>_NNNNNN.hvm``, ``<out>/games/game_NNNN.jsonl`` (unless
``--no-events``) and ``<out>/summary.json``, and prints a results table.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed

from arena.interfaces import SEATS, TEAM_OF
from arena.openings import OPENING_SETS, paired_schedule
from arena.recorder import DEFAULT_SAMPLES_PER_CHUNK, ShardWriter, finish_game
from arena.stats import summarize_game


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


def controller_kw(spec: dict) -> dict:
    return {"free": True, "record": True, "k_range": spec["k_range"], "jitter": spec["jitter"],
            "free_sit_ms": spec["free_sit_ms"], "min_nodes": spec["nodes_range"][0],
            "max_nodes": spec["nodes_range"][1], "ms_per_node": spec["ms_per_node"]}


def build_players(spec: dict, process_factory=None) -> tuple[dict, list]:
    """{seat: player} for two free-mode controllers; also the controllers. ``process_factory(team)`` replaces the
    engine process (tests pass a stub)."""
    from arena.players.hivemind import HivemindProcess, HivemindTeamController, resolve_binary

    players, ctls = {}, []
    for i, team in enumerate(("T1", "T2")):
        proc = process_factory(team) if process_factory else HivemindProcess(resolve_binary(spec["binary"]),
                                                                             spec["model"])
        ctl = HivemindTeamController(team, proc, seed=spec["seed"] + i, **controller_kw(spec))
        ctls.append(ctl)
        for s in SEATS:
            if TEAM_OF[s] == team:
                players[s] = ctl.player(s)
    return players, ctls


def play_one(spec: dict, process_factory=None) -> tuple[dict, list[bytes]]:
    """Play one self-play game. Returns its summary and its finished samples (empty when the game crashed)."""
    from arena.sim import Simulator
    from arena.state import BughouseState

    t0 = time.perf_counter()
    meta = {k: spec[k] for k in ("game_id", "game_num", "opening_id", "opening")}
    sim = None
    try:
        players, ctls = build_players(spec, process_factory)
        sim = Simulator(BughouseState(spec["base_ms"], 0, fens=spec["fens"]), players, game_id=spec["game_id"],
                        setup="selfplay-free", max_plies=spec["max_plies"])
        result = sim.run()
    except Exception as exc:  # noqa: BLE001 - one bad game should not kill the batch
        return {**meta, "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}, []
    events = list(sim.events)
    samples = [s for c in ctls for s in c.samples]
    finished = [] if result.reason == "aborted" else finish_game(samples, result.result, spec["game_num"])
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
    summary["sit_spells"] = sum(1 for e in events if e["type"] == "stall" and e.get("active"))
    summary["wall_s"] = round(time.perf_counter() - t0, 2)
    return summary, finished


def make_specs(args) -> list[dict]:
    specs = []
    for i, (pair, _swap) in enumerate(paired_schedule(args.games, args.seed, OPENING_SETS[args.openings])):
        specs.append({
            "index": i + 1, "game_id": f"s{args.seed}g{i + 1:05d}", "game_num": args.game_id_offset + i + 1,
            "opening_id": pair.id, "opening": pair.name, "fens": pair.fens, "seed": args.seed * 100003 + 2 * i,
            "base_ms": args.base_ms, "max_plies": args.max_plies, "k_range": args.k_range, "jitter": args.jitter,
            "free_sit_ms": args.free_sit_ms, "nodes_range": args.nodes_range, "ms_per_node": args.ms_per_node,
            "binary": args.binary, "model": args.model, "out": args.out, "no_events": args.no_events,
        })
    return specs


def results(summaries: list[dict], writer: ShardWriter | None = None) -> dict:
    ok = [s for s in summaries if "error" not in s]
    n = len(ok) or 1
    searches = sum(sum(s["searches"].values()) for s in ok)
    return {
        "games": len(ok), "errors": len(summaries) - len(ok),
        "results": dict(Counter(s["result"] for s in ok)),
        "reasons": dict(Counter(s["reason"] for s in ok)),
        "flag_rate": round(sum(s["flag"] for s in ok) / n, 3),
        "mean_plies": round(sum(s["plies"] for s in ok) / n, 1),
        "mean_duration_s": round(sum(s["duration_ms"] for s in ok) / n / 1000, 1),
        "searches": searches,
        "sit_spells_per_search": round(sum(s["sit_spells"] for s in ok) / searches, 3) if searches else 0.0,
        "samples": sum(s["samples"] for s in ok),
        "chunks": [os.path.basename(p) for p in writer.paths] if writer else [],
        "channels": writer.channels if writer else None,
    }


def format_results(r: dict) -> str:
    lines = [f"games {r['games']} ({r['errors']} errors)   samples {r['samples']}   channels {r['channels']}",
             f"results {r['results']}   reasons {r['reasons']}",
             f"flag rate {r['flag_rate']:.3f}   mean plies {r['mean_plies']}   mean length {r['mean_duration_s']} s",
             f"searches {r['searches']}   sit spells per search {r['sit_spells_per_search']:.3f}   chunks {len(r['chunks'])}"]
    return "\n".join(lines)


def run(args, process_factory=None) -> dict:
    os.makedirs(args.out, exist_ok=True)
    run_id = args.run_id or f"{os.path.basename(os.path.normpath(args.out))}-s{args.seed}"
    writer = ShardWriter(os.path.join(args.out, "training_data"), run_id, samples_per_chunk=args.samples_per_chunk)
    specs = make_specs(args)
    summaries: list[dict] = []
    t0 = time.perf_counter()

    def done(summary: dict, samples: list[bytes]) -> None:
        summaries.append(summary)
        writer.append(samples)
        tag = summary.get("error") or (f"{summary['result']} by {summary['reason']} in {summary['plies']} plies, "
                                       f"{summary['samples']} samples, k {summary['k']}")
        print(f"[{len(summaries)}/{len(specs)}] {summary['game_id']} {tag}", flush=True)

    if args.workers <= 1 or process_factory is not None:
        for s in specs:
            done(*play_one(s, process_factory))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for fut in as_completed([ex.submit(play_one, s) for s in specs]):
                done(*fut.result())
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
    ap.add_argument("--model", default=os.path.join(ENGINE_DIR, "models", "hivemind-clock-fp32.onnx"),
                    help="ONNX network (the 41-plane clock network)")
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
    ap.add_argument("--openings", default="train", choices=("train", "all", "eval"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--game-id-offset", type=int, default=0, help="added to game numbers (gameId in the samples)")
    ap.add_argument("--run-id", default=None, help="chunk name tag (default <out basename>-s<seed>)")
    ap.add_argument("--samples-per-chunk", type=int, default=DEFAULT_SAMPLES_PER_CHUNK)
    ap.add_argument("--no-events", action="store_true", help="do not write per-game event logs")
    ap.add_argument("--summary", default=None, help="summary file (default <out>/summary.json)")
    args = ap.parse_args(argv)
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
