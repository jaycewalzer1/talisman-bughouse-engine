#!/usr/bin/env python3
"""Training curves for the dev panel (viewer durative.html, Training step): one JSON object per lineage with, for every
iteration, its gate games cut into blocks of --block games (win rate per block, in play order), the gate totals, and
what changed in that iteration (self-play games and positions, flag games, training steps and replay share, whether
the generator was replaced, the rl-loop settings in force).
    .venv/bin/python cluster/training_curve.py runs/full/hybrid1 ../talisman-ab2/runs/wave2/ab2/narrow
A lineage folder is a wave2_iterate.sbatch ROOT (it holds runs/rl-itN). Reads only; prints JSON on stdout.
"""

import argparse
import json
import os
import sys


def load(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def games(path):
    """Contender scores (1, 0.5, 0) in play order from an engine tournament's games.tsv (result column: +1/-1/0)."""
    out = []
    try:
        with open(path) as f:
            next(f)  # the config line
            for line in f:
                cols = line.rstrip("\n").split("\t")
                if len(cols) > 1 and cols[1] in ("1", "-1", "0"):
                    out.append((int(cols[1]) + 1) / 2)
    except (OSError, StopIteration):
        pass
    return out


def blocks(scores, size):
    return [{"games": len(chunk), "score": sum(chunk) / len(chunk)}
            for chunk in (scores[i:i + size] for i in range(0, len(scores), size)) if chunk]


def lineage(root, size):
    runs = os.path.join(root, "runs")
    its = sorted((d for d in os.listdir(runs) if d.startswith("rl-it") and d[5:].isdigit()), key=lambda d: int(d[5:])) \
        if os.path.isdir(runs) else []
    settings = None
    log = os.path.join(runs, "rl-loop.log")
    if os.path.isfile(log):
        with open(log, errors="replace") as f:
            for line in f:
                if " loop: iterations " in line:
                    settings = line.split(" loop: ", 1)[1].strip()
    launch = os.path.join(root, "launch_settings.txt")
    out = {"root": os.path.abspath(root), "name": os.path.basename(os.path.normpath(root)),
           "launch": open(launch).read().strip() if os.path.isfile(launch) else None,
           "loop_settings": settings, "iterations": []}
    for it in its:
        stages, arms = os.path.join(runs, it, "stages"), os.path.join(runs, it, "arms")
        segs = [load(os.path.join(stages, f)) or {} for f in sorted(os.listdir(stages))
                if f.startswith("seg") and f.endswith(".done")] if os.path.isdir(stages) else []
        labels = [s.get("labels", {}) for s in segs]
        train = load(os.path.join(stages, "train.done")) or {}
        promote = load(os.path.join(stages, "promote.done"))
        entry = {"iteration": int(it[5:]), "done": os.path.isfile(os.path.join(stages, "iteration.done")),
                 "selfplay_games": sum(s.get("games", 0) for s in segs),
                 "positions": sum(s.get("positions", 0) for s in segs),
                 "segments": len(segs),
                 "flag_games": sum(l.get("flag_games", 0) for l in labels),
                 "repeat_records": sum(l.get("repeat_records", 0) for l in labels),
                 "train": {k: train.get(k) for k in ("steps", "replay_fraction", "new_positions", "replay_positions")}
                 if train else None,
                 "promoted": None if promote is None else promote.get("promoted"),
                 "generator": (open(os.path.join(stages, "generator")).read().strip()[:12]
                               if os.path.isfile(os.path.join(stages, "generator")) else None),
                 "matches": {}}
        for name in ("generator", "it0"):
            marker = load(os.path.join(stages, f"match_{name}.done"))
            source = "generator" if marker and marker.get("same_as") == "generator" else name
            scores = games(os.path.join(arms, f"match_{source}", "games.tsv"))
            if not scores and not marker:
                continue
            entry["matches"][name] = {
                "games": len(scores), "score": marker.get("score") if marker else (sum(scores) / len(scores)),
                "elo": marker.get("elo") if marker else None, "ci": marker.get("ci") if marker else None,
                "finished": marker is not None, "same_as_generator": source != name, "blocks": blocks(scores, size)}
        out["iterations"].append(entry)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("roots", nargs="+")
    ap.add_argument("--block", type=int, default=100)
    args = ap.parse_args()
    json.dump({"block": args.block, "lineages": [lineage(r, args.block) for r in args.roots if os.path.isdir(r)]},
              sys.stdout)


if __name__ == "__main__":
    main()
