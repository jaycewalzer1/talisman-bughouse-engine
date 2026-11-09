"""Does this machine's Hivemind build read positions the same way as the reference build?

    python cluster/engine_parity.py record runs/<run>/game_*.jsonl   # on the reference machine
    python cluster/engine_parity.py check                            # on the new machine

Each position gets a minimal search with settings that make it nearly repeatable (the default batch of 8 makes
the eval depend on thread timing). The reference keeps only positions that came back identical every time, and a
check takes the median of several searches, so results can be compared across builds. FP16 and FP32 differ slightly,
so evals only have to agree within --tolerance. Even so, 1 or 2 positions in 200 still jump between identical runs on
one machine, so up to --max-outliers may miss it; a build that really differs misses on many. A pass writes
runs/cluster/parity_passed, which cluster/bench.sbatch requires.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from arena.players.hivemind import HivemindProcess  # noqa: E402

REFERENCE = os.path.join(os.path.dirname(__file__), "engine_parity_ref.json")
PASSED = os.path.join("runs", "cluster", "parity_passed")
REPEATABLE = {"BatchSize": 1, "MateProbe": "false", "CertifySelectedMove": "false"}
REPEATS = 3


def positions_from_games(paths: list[str], seed: int) -> list[dict]:
    """Every position in the games, shuffled, with the team to move on board A (so the engine has a move to make)."""
    found = []
    for path in paths:
        with open(path) as f:
            events = [json.loads(line) for line in f]
        fens = dict(events[0]["fens"])
        for e in events:
            if e["type"] == "move":
                fens[e["board"]] = e["fen"]
                found.append({"fens": dict(fens), "team": "T1" if fens["A"].split()[1] == "w" else "T2"})
    random.Random(seed).shuffle(found)
    return found


def evaluations(process: HivemindProcess, p: dict) -> list[float] | None:
    """The eval from each of REPEATS searches, or None for a position with nothing to search (the game is over)."""
    qs = [process.search(p["fens"]["A"], p["fens"]["B"], team=p["team"], nodes=1).q for _ in range(REPEATS)]
    return None if None in qs else qs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    record = sub.add_parser("record")
    record.add_argument("games", nargs="+")
    record.add_argument("--positions", type=int, default=200)
    record.add_argument("--seed", type=int, default=0)
    check = sub.add_parser("check")
    check.add_argument("--tolerance", type=float, default=0.02)
    check.add_argument("--max-outliers", type=int, default=5)
    args = ap.parse_args(argv)

    process = HivemindProcess(options=REPEATABLE)
    try:
        if args.cmd == "record":
            ref = []
            for p in positions_from_games(args.games, args.seed):
                qs = evaluations(process, p)
                if qs is not None and max(qs) - min(qs) < 1e-6:  # a few positions still vary with thread timing
                    ref.append({**p, "q": qs[0]})
                if len(ref) == args.positions:
                    break
            with open(REFERENCE, "w") as f:
                json.dump(ref, f, indent=0)
            print(f"recorded {len(ref)} positions")
            return 0

        with open(REFERENCE) as f:
            ref = json.load(f)
        diffs = []
        for p in ref:
            qs = evaluations(process, p)
            diffs.append(float("inf") if qs is None else abs(statistics.median(qs) - p["q"]))
    finally:
        process.close()

    bad = sum(d > args.tolerance for d in diffs)
    print(f"{len(diffs)} positions: mean diff {statistics.mean(diffs):.5f}, max {max(diffs):.5f}, "
          f"{bad} over {args.tolerance}")
    if os.path.exists(PASSED):
        os.remove(PASSED)
    if bad > args.max_outliers:
        print("PARITY FAILED: do not benchmark this build")
        return 1
    os.makedirs(os.path.dirname(PASSED), exist_ok=True)
    with open(PASSED, "w") as f:
        f.write(f"{statistics.mean(diffs):.5f} {max(diffs):.5f}\n")
    print("parity ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
