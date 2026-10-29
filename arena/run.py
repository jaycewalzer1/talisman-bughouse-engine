"""Batch match runner: contender setup vs baseline setup over paired openings.

    python -m arena.run --contender sharing --baseline silent --kind heuristic --games 20 \
        --base-ms 60000 --channel-delay-ms 0 --nodes 200 --seed 0 --out runs/sharing_vs_silent

Writes runs/<name>/game_XXXX.jsonl (one event per line) and runs/<name>/summary.json, and prints a
results table (arena/stats.py).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, is_dataclass

from arena.interfaces import SEATS, TEAM_OF, GameResult
from arena.openings import paired_schedule
from arena.stats import compute_stats, format_table, summarize_game

log = logging.getLogger("arena.run")

SETUPS = ("centralized", "silent", "sharing")
KINDS = ("random", "heuristic", "hivemind")


def setup_label(t1_setup: str, t2_setup: str) -> str:
    """The `setup` string passed to the Simulator (shown in game_start) when teams differ."""
    return t1_setup if t1_setup == t2_setup else f"T1={t1_setup},T2={t2_setup}"


def event_to_dict(e) -> dict:
    if isinstance(e, dict):
        return e
    if is_dataclass(e):
        return asdict(e)
    if hasattr(e, "to_dict"):
        return e.to_dict()
    return dict(e)


def build_players(t1_setup: str, t2_setup: str, kind: str, *, seed: int, nodes: int) -> tuple[dict, list]:
    """Seat -> Player with T1 on t1_setup and T2 on t2_setup; also the list of players to close."""
    from arena.players import make_players

    players = make_players(t1_setup, kind, seed=seed, nodes=nodes, opponent_setup=t2_setup)
    return players, list({id(p): p for p in players.values()}.values())


def _close(player) -> None:
    try:
        player.close()
    except Exception:  # noqa: BLE001 - best effort
        pass


def play_game(spec: dict, on_event=None) -> tuple[list[dict], object]:
    """Build players and run one game. spec keys: game_id, t1_setup, t2_setup, kind, seed, nodes,
    base_ms, increment_ms, fens, channel_delay_ms, max_plies, charge_wall_time."""
    from arena.sim import Simulator
    from arena.state import BughouseState

    players, created = build_players(spec["t1_setup"], spec["t2_setup"], spec["kind"], seed=spec["seed"],
                            nodes=spec["nodes"])
    try:
        state = BughouseState(spec["base_ms"], spec.get("increment_ms", 0), fens=spec.get("fens"))
        sim = Simulator(state, players, game_id=spec["game_id"],
                        setup=setup_label(spec["t1_setup"], spec["t2_setup"]),
                        channel_delay_ms=spec.get("channel_delay_ms", 0),
                        max_plies=spec.get("max_plies", 400),
                        charge_wall_time=spec.get("charge_wall_time", False), on_event=on_event)
        try:
            result = sim.run()
        except Exception as exc:  # noqa: BLE001 - a player crashed: close the game as aborted
            log.error("game %s aborted: %s", spec["game_id"], traceback.format_exc())
            events = [event_to_dict(e) for e in sim.events]
            if events and events[-1].get("type") == "game_end":
                return events, GameResult("draw", "aborted")
            end = {"type": "game_end", "seq": (events[-1]["seq"] + 1) if events else 0,
                   "ts": events[-1]["ts"] if events else 0, "result": "draw", "reason": "aborted",
                   "board": None, "loserSeat": None, "error": f"{type(exc).__name__}: {exc}"}
            events.append(end)
            if on_event is not None:
                on_event(end)
            return events, GameResult("draw", "aborted")
        return [event_to_dict(e) for e in sim.events], result
    finally:
        for p in created:
            _close(p)


def make_specs(args) -> list[dict]:
    specs = []
    for i, (pair, swap) in enumerate(paired_schedule(args.games, args.seed)):
        contender_team = "T2" if swap else "T1"
        t1, t2 = (args.baseline, args.contender) if swap else (args.contender, args.baseline)
        specs.append({
            "index": i + 1, "game_id": f"g{i + 1:04d}", "t1_setup": t1, "t2_setup": t2,
            "contender_team": contender_team, "opening_id": pair.id, "opening": pair.name,
            "fens": pair.fens, "swap": swap, "kind": args.kind, "seed": args.seed * 100003 + i,
            "nodes": args.nodes, "base_ms": args.base_ms, "increment_ms": args.increment_ms,
            "channel_delay_ms": args.channel_delay_ms, "max_plies": args.max_plies,
            "charge_wall_time": args.charge_wall_time, "out": args.out,
        })
    return specs


def run_one(spec: dict) -> dict:
    """Worker: play one game, write its jsonl, return its annotated summary (or an error record)."""
    t0 = time.perf_counter()
    meta = {k: spec[k] for k in ("game_id", "contender_team", "opening_id", "opening", "swap",
                                 "t1_setup", "t2_setup")}
    try:
        events, _result = play_game(spec)
    except Exception as exc:  # noqa: BLE001 - one bad game should not kill the batch
        return {**meta, "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}
    path = os.path.join(spec["out"], f"game_{spec['index']:04d}.jsonl")
    with open(path, "w") as f:
        for e in events:
            f.write(json.dumps(e, separators=(",", ":")) + "\n")
    summary = summarize_game(events)
    summary.update(meta)
    if events and events[-1].get("error"):  # a player crashed; game recorded as aborted
        summary["error"] = events[-1]["error"]
        summary["traceback"] = "(aborted mid-game; see the jsonl and the log above)"
    summary["file"] = os.path.basename(path)
    summary["wall_s"] = round(time.perf_counter() - t0, 3)
    return summary


def parse_args(argv=None):
    ap = argparse.ArgumentParser(prog="python -m arena.run", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--contender", default="sharing", choices=SETUPS)
    ap.add_argument("--baseline", default="silent", choices=SETUPS)
    ap.add_argument("--kind", default="heuristic", choices=KINDS)
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--base-ms", type=int, default=60000)
    ap.add_argument("--increment-ms", type=int, default=0)
    ap.add_argument("--channel-delay-ms", type=int, default=0)
    ap.add_argument("--nodes", type=int, default=200)
    ap.add_argument("--max-plies", type=int, default=400)
    ap.add_argument("--charge-wall-time", action="store_true",
                    help="charge real search wall time on the clock instead of the player's think_ms")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1, help="parallel games (processes)")
    ap.add_argument("--boot", type=int, default=2000, help="bootstrap resamples")
    ap.add_argument("--out", default=None, help="output dir (default runs/<contender>_vs_<baseline>_<kind>)")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.out is None:
        args.out = os.path.join("runs", f"{args.contender}_vs_{args.baseline}_{args.kind}")
    os.makedirs(args.out, exist_ok=True)
    specs = make_specs(args)
    summaries: list[dict] = []
    t0 = time.perf_counter()

    def report(s: dict) -> None:
        summaries.append(s)
        if "error" in s:
            print(f"[{len(summaries)}/{len(specs)}] {s['game_id']} ERROR {s['error']}", file=sys.stderr)
            print(s["traceback"], file=sys.stderr)
        else:
            who = "contender" if s["result"] == s["contender_team"] else "draw" if s["result"] == "draw" else "baseline"
            print(f"[{len(summaries)}/{len(specs)}] {s['game_id']} {s['opening']:<40} contender={s['contender_team']} "
                  f"-> {s['result']} ({who}) by {s['reason']} in {s['plies']} plies, {s['wall_s']:.1f}s", flush=True)

    if args.workers <= 1:
        for spec in specs:
            report(run_one(spec))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for fut in as_completed([ex.submit(run_one, s) for s in specs]):
                report(fut.result())
    summaries.sort(key=lambda s: s["game_id"])

    stats = compute_stats(summaries, n_boot=args.boot, seed=args.seed)
    out = {"config": vars(args), "wall_s": round(time.perf_counter() - t0, 2), "stats": stats,
           "errors": sum(1 for s in summaries if "error" in s), "games": summaries}
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(out, f, indent=2)

    print()
    print(format_table(stats, args.contender, args.baseline))
    if out["errors"]:
        print(f"\n{out['errors']} game(s) failed; see summary.json")
    print(f"\nwrote {args.out}/summary.json and {len(summaries) - out['errors']} game files "
          f"({out['wall_s']:.1f}s)")
    return 1 if out["errors"] == len(summaries) and summaries else 0


if __name__ == "__main__":
    sys.exit(main())
