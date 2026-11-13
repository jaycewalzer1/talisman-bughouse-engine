"""Tables from a training run's recorded games: waiting and results by opponent, how losses ended, won vs lost.

    python tools/train_tables.py runs/phase1 --split 20

Reads every <run>/itNNNN/summary.json. --split is the last iteration of the "early" half. Waits are counted per
100 of the contender's decisions, because a longer game has more chances to wait.
"""

import argparse
import glob
import json
import os
from collections import defaultdict


def load(run: str) -> list[dict]:
    rows = []
    for path in sorted(glob.glob(os.path.join(run, "it[0-9]*", "summary.json"))):
        iteration = int(os.path.basename(os.path.dirname(path))[2:])
        with open(path) as f:
            for g in json.load(f)["games"]:
                if "error" in g:
                    continue
                me = g["teams"][g["contender_team"]]
                won = g["result"] == g["contender_team"]
                rows.append({"iteration": iteration, "opponent": g["opponent"], "won": won,
                             "score": 1.0 if won else 0.5 if g["result"] not in ("T1", "T2") else 0.0,
                             "lost_on_time": bool(g["flag"]) and g["loser_team"] == g["contender_team"],
                             "won_on_time": bool(g["flag"]) and g["loser_team"] != g["contender_team"],
                             "stalls": me["stalls"], "messages": me["messages"], "decisions": me["decisions"],
                             "plies": g["plies"]})
    return rows


def line(name: str, rows: list[dict]) -> str:
    if not rows:
        return f"{name:<26} {0:>5}"
    n, decisions = len(rows), sum(r["decisions"] for r in rows) or 1
    losses = [r for r in rows if r["score"] == 0.0]
    on_time = f"{sum(r['lost_on_time'] for r in losses)}/{len(losses)}"
    return (f"{name:<26} {n:>5} {sum(r['score'] for r in rows) / n:>6.3f} "
            f"{100 * sum(r['stalls'] for r in rows) / decisions:>9.2f} "
            f"{100 * sum(r['messages'] for r in rows) / decisions:>9.2f} "
            f"{sum(r['plies'] for r in rows) / n:>6.0f} {on_time:>13} {sum(r['won_on_time'] for r in rows):>12}")


HEADER = (f"{'':<26} {'games':>5} {'score':>6} {'waits/100':>9} {'talks/100':>9} {'plies':>6} "
          f"{'losses on time':>13} {'wins on time':>12}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run")
    ap.add_argument("--split", type=int, default=20)
    args = ap.parse_args()
    rows = load(args.run)
    last = max(r["iteration"] for r in rows)
    halves = ((f"iterations 1-{args.split}", [r for r in rows if r["iteration"] <= args.split]),
              (f"iterations {args.split + 1}-{last}", [r for r in rows if r["iteration"] > args.split]))

    for title, part in halves:
        by_opponent = defaultdict(list)
        for r in part:
            by_opponent[r["opponent"]].append(r)
        print(f"\n== {title}, by opponent\n{HEADER}")
        for name in sorted(by_opponent):
            print(line(name, by_opponent[name]))
        print(line("ALL", part))

    print(f"\n== won games against lost games\n{HEADER}")
    for title, part in halves:
        print(line(f"won,  {title}", [r for r in part if r["won"]]))
        print(line(f"lost, {title}", [r for r in part if r["score"] == 0.0]))

    stock = [r for r in rows if r["opponent"] == "stock-centralized"]
    print(f"\n== against stock-centralized only, blocks of 10 iterations\n{HEADER}")
    for start in range(1, last + 1, 10):
        print(line(f"iterations {start}-{start + 9}", [r for r in stock if start <= r["iteration"] < start + 10]))


if __name__ == "__main__":
    main()
