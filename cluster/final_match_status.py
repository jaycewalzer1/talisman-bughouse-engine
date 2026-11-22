#!/usr/bin/env python3
"""The final match's table (cluster/launch_final_match.sh): each pairing's games, score of the first engine with its 95%
range (parts merged, cluster bootstrap over openings), and each engine's total over its two pairings.
    .venv/bin/python cluster/final_match_status.py runs/final_match/final1
"""

import csv
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from decide_full_run import merged_row_stats  # noqa: E402


def elo(score):
    return 0.0 if score in (0, 1) else -400 * math.log10(1 / score - 1)


def main(out):
    totals = {}
    print(f"{'pairing':40s} {'games':>6s}  score of the first [95%]     Elo")
    with open(os.path.join(out, "pairings.tsv")) as f:
        for label, *_ in csv.reader(f, delimiter="\t"):
            a, b = label.split("_vs_")
            r = merged_row_stats(os.path.join(out, label))
            if r is None:
                print(f"{label:40s} {0:6d}  (no games yet)")
                continue
            lo, hi = r["ci95"]
            print(f"{label:40s} {r['games']:6d}  {r['score']:.3f} [{lo:.3f}, {hi:.3f}]  {elo(r['score']):+6.0f}"
                  + ("" if r["finished"] else "  running") + (f"  {r['errors']} errors" if r["errors"] else ""))
            for name, s in ((a, r["score"]), (b, 1 - r["score"])):
                g, t = totals.get(name, (0, 0.0))
                totals[name] = (g + r["games"], t + s * r["games"])
    print("\nengine totals over both pairings")
    for name, (g, t) in sorted(totals.items(), key=lambda kv: -kv[1][1] / max(kv[1][0], 1)):
        print(f"  {name:30s} {g:5d} games  {t / g:.3f}")


if __name__ == "__main__":
    main(sys.argv[1])
