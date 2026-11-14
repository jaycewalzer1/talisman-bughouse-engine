"""The wait-regret table from one or more probe folders, including a probe that was cut off before it finished.

    python tools/probe_summary.py runs/probe/*

Reads each folder's rollouts.jsonl and pools them. A position only counts once it has at least one finished rollout
on the play branch and on a wait branch. "wait-play" is the probed team's score after a forced wait minus its score
after playing at once, averaged over positions: above 0 means waiting helped.
"""

import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from arena.probe_wait import format_summary, summarize  # noqa: E402


def main() -> None:
    rows = []
    for folder in sys.argv[1:]:
        path = os.path.join(folder, "rollouts.jsonl")
        if not os.path.exists(path):
            print(f"{folder}: no rollouts yet")
            continue
        with open(path) as f:
            mine = [json.loads(line) for line in f if line.strip()]
        for r in mine:  # two folders can probe files with the same name
            r["file"] = os.path.join(folder, os.path.basename(r["file"]))
        rows += mine
        print(f"{folder}: {len(mine)} rollouts")
    if not rows:
        return
    waits = sorted({r["wait_ms"] for r in rows if r["wait_ms"] > 0})
    summary = summarize(rows, waits)
    print()
    print(format_summary(summary))
    # how a position's own result is spread: a mean near 0 can hide positions where waiting clearly wins
    for w in waits:
        diffs = sorted(p[f"wait_{w}"] - p["play"] for p in summary["positions"].values()
                       if f"wait_{w}" in p and "play" in p)
        if diffs:
            quarter = len(diffs) // 4
            print(f"wait_{w}: lowest quarter of positions {sum(diffs[:quarter or 1]) / (quarter or 1):+.3f}, "
                  f"highest quarter {sum(diffs[-(quarter or 1):]) / (quarter or 1):+.3f}")


if __name__ == "__main__":
    main()
