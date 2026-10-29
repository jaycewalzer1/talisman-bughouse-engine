"""Throughput benchmark: how much wall time and compute a game costs, to budget runs before committing
cluster hours.

    python -m arena.bench --setups centralized silent --games 4 --nodes 200 --workers 2

Plays ``--games`` games per setup pair (a setup against itself unless ``--vs`` is given) with the same
options as ``arena.run`` and reports per game: wall seconds, plies, searches, requested nodes, wall ms
per requested node, and the simulated game length. The projection at the end scales the measured wall
time to ``--plan-games`` games on ``--workers`` parallel workers. Run it on the target machine (e.g. a PACE
GPU node with the TensorRT build): numbers measured on one machine do not transfer to another.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor

from arena.openings import paired_schedule
from arena.run import play_game


def game_metrics(events: list[dict], wall_s: float) -> dict:
    starts = [e for e in events if e["type"] == "thinking" and e["phase"] == "start"]
    end = events[-1]
    return {
        "wall_s": round(wall_s, 2),
        "plies": sum(1 for e in events if e["type"] == "move"),
        "searches": sum(1 for e in starts if (e.get("nodes") or 0) > 0),
        "nodes": sum(e.get("nodes") or 0 for e in starts),
        "borrowed": sum(1 for e in starts if e.get("borrowed")),
        "sim_s": round(end["ts"] / 1000, 1),
        "result": end.get("result"),
        "reason": end.get("reason"),
    }


def bench_one(spec: dict) -> dict:
    t0 = time.perf_counter()
    events, _ = play_game(spec)
    return {"setup": spec["label"], **game_metrics(events, time.perf_counter() - t0)}


def make_specs(args) -> list[dict]:
    specs = []
    for setup in args.setups:
        opp = args.vs or setup
        for i, (pair, swap) in enumerate(paired_schedule(args.games, args.seed)):
            t1, t2 = (opp, setup) if swap else (setup, opp)
            specs.append({"label": setup if opp == setup else f"{setup}_vs_{opp}", "game_id": f"bench{i:03d}",
                          "t1_setup": t1, "t2_setup": t2, "kind": args.kind, "seed": args.seed * 100003 + i,
                          "nodes": args.nodes, "base_ms": args.base_ms, "increment_ms": 0, "fens": pair.fens,
                          "channel_delay_ms": 0, "max_plies": args.max_plies, "charge_wall_time": False,
                          "player_kw": {"compute": args.compute, "lend": args.lend}})
    return specs


def report(rows: list[dict], args, total_wall: float) -> dict:
    out = {}
    for setup in dict.fromkeys(r["setup"] for r in rows):
        rs = [r for r in rows if r["setup"] == setup]
        nodes = sum(r["nodes"] for r in rs)
        wall = sum(r["wall_s"] for r in rs)
        out[setup] = {
            "games": len(rs),
            "wall_s_per_game": round(statistics.mean(r["wall_s"] for r in rs), 1),
            "plies_per_game": round(statistics.mean(r["plies"] for r in rs), 1),
            "searches_per_game": round(statistics.mean(r["searches"] for r in rs), 1),
            "nodes_per_game": round(nodes / len(rs)),
            "wall_ms_per_node": round(1000 * wall / nodes, 2) if nodes else None,
            "sim_s_per_game": round(statistics.mean(r["sim_s"] for r in rs), 1),
            "flags": sum(r["reason"] == "flag" for r in rs),
        }
    per_game = statistics.mean(r["wall_s"] for r in rows)
    out["projection"] = {
        "plan_games": args.plan_games, "workers": args.workers,
        "wall_hours": round(args.plan_games * per_game / args.workers / 3600, 1),
        "worker_hours": round(args.plan_games * per_game / 3600, 1),
        "measured_parallel_speedup": round(sum(r["wall_s"] for r in rows) / total_wall, 2) if total_wall else None,
    }
    return out


def parse_args(argv=None):
    ap = argparse.ArgumentParser(prog="python -m arena.bench", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--setups", nargs="+", default=["centralized", "silent"],
                    choices=("centralized", "silent", "sharing"))
    ap.add_argument("--vs", default=None, choices=("centralized", "silent", "sharing"),
                    help="opponent setup for every game (default: each setup plays itself)")
    ap.add_argument("--kind", default="hivemind", choices=("random", "heuristic", "hivemind"))
    ap.add_argument("--games", type=int, default=4, help="games per setup")
    ap.add_argument("--nodes", type=int, default=200)
    ap.add_argument("--compute", default="throughput", choices=("throughput", "nodes"))
    ap.add_argument("--lend", action="store_true")
    ap.add_argument("--base-ms", type=int, default=60000)
    ap.add_argument("--max-plies", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--plan-games", type=int, default=5000, help="games to project the cost of")
    ap.add_argument("--out", default=None, help="write the report as JSON here")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    specs = make_specs(args)
    t0 = time.perf_counter()
    if args.workers <= 1:
        rows = [bench_one(s) for s in specs]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            rows = list(ex.map(bench_one, specs))
    total = time.perf_counter() - t0
    for r in rows:
        print(json.dumps(r), flush=True)
    rep = report(rows, args, total)
    print(json.dumps(rep, indent=2))
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"config": vars(args), "games": rows, "report": rep}, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
