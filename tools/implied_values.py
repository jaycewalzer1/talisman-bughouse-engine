"""Implied piece values per pocket from a network's value head (audit 2026-11-17 WP-H, theory item 4).

For a sample of recorded positions (the ``.pos.jsonl`` sidecars next to self-play chunks: both FENs, clocks,
elapsed times, the searching team and the TimeAdvantage input it was given), one piece of each type P N B R Q is
added to each of the four pockets, from the searching team's view:

    ours_A    our seat on board A (T1: A_white)        theirs_A  the opponent on board A
    ours_B    our seat on board B (T1: B_black)        theirs_B  the opponent on board B

and dV = V(with the piece) - V(position) is read from the raw network (no search). Everything else, the
TimeAdvantage plane included, stays as recorded. Reported per pocket and piece: mean and median dV, the same
restricted to |V| < 0.5 (where the value head is not saturated), and dV relative to the pawn of the same pocket.
Bughouse folklore (theory 02 2.5) is P 1, N = B = R 2, Q 4.

    .venv/bin/python tools/implied_values.py --shards runs/custom/local1h/it01/selfplay/training_data \\
        --model third_party/hivemind/engine/models/hivemind-fp32.onnx --positions 300 [--json out.json]

Single process, CPU ONNX Runtime.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import statistics
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from regression import MODELS_DIR, NetEval  # noqa: E402

POCKETS = ("ours_A", "theirs_A", "ours_B", "theirs_B")
PIECES = "PNBRQ"
FOLKLORE = {"P": 1, "N": 2, "B": 2, "R": 2, "Q": 4}


def sample_positions(dirs: list[str], n: int, seed: int = 0) -> list[dict]:
    """``n`` sidecar lines drawn at random (without replacement) from the chunks in ``dirs``."""
    metas = []
    for d in dirs:
        for path in sorted(glob.glob(os.path.join(d, "*.pos.jsonl"))):
            with open(path) as f:
                metas += [json.loads(line) for line in f if line.strip()]
    if not metas:
        raise SystemExit(f"no .pos.jsonl sidecars under {dirs}")
    rng = random.Random(seed)
    return metas if n <= 0 or n >= len(metas) else rng.sample(metas, n)


def pocket_color(team: str, pocket: str) -> str:
    """'w' or 'b': the colour whose pocket ``pocket`` is on its board, for ``team``."""
    ours = pocket.startswith("ours")
    board = pocket[-1]
    # T1 plays white on A and black on B
    t1_color = "w" if board == "A" else "b"
    color = t1_color if team == "T1" else ("b" if t1_color == "w" else "w")
    return color if ours else ("b" if color == "w" else "w")


def add_to_pocket(fen: str, piece: str, color: str) -> str:
    board, rest = fen.split(" ", 1)
    sym = piece.upper() if color == "w" else piece.lower()
    if "[" in board:
        head, pocket = board.split("[", 1)
        pocket = pocket.rstrip("]")
    else:
        head, pocket = board, ""
    return f"{head}[{pocket}{sym}] {rest}"


def run(args) -> dict:
    net = NetEval(args.model, threads=args.threads)
    metas = sample_positions(args.shards, args.positions, args.seed)
    planes, keys = [], []
    for i, m in enumerate(metas):
        team, ta = m["team"], bool(m.get("timeAdvantage", False))
        planes.append(net.planes(m["fenA"], m["fenB"], team, m["clocks"], m.get("elapsed"), ta))
        keys.append((i, None, None))
        for pocket in POCKETS:
            color = pocket_color(team, pocket)
            for piece in PIECES:
                fa, fb = m["fenA"], m["fenB"]
                if pocket.endswith("A"):
                    fa = add_to_pocket(fa, piece, color)
                else:
                    fb = add_to_pocket(fb, piece, color)
                planes.append(net.planes(fa, fb, team, m["clocks"], m.get("elapsed"), ta))
                keys.append((i, pocket, piece))
    v = net.run(np.stack(planes))["value"][:, 0]
    base = {i: float(v[j]) for j, (i, p, _) in enumerate(keys) if p is None}
    d: dict = {(p, x): [] for p in POCKETS for x in PIECES}
    d_mid: dict = {(p, x): [] for p in POCKETS for x in PIECES}
    for j, (i, p, x) in enumerate(keys):
        if p is None:
            continue
        dv = float(v[j]) - base[i]
        d[(p, x)].append(dv)
        if abs(base[i]) < 0.5:
            d_mid[(p, x)].append(dv)
    table = {}
    for p in POCKETS:
        row = {}
        pawn = statistics.fmean(d[(p, "P")])
        pawn_mid = statistics.fmean(d_mid[(p, "P")]) if d_mid[(p, "P")] else float("nan")
        for x in PIECES:
            vals, mid = d[(p, x)], d_mid[(p, x)]
            mean = statistics.fmean(vals)
            mean_mid = statistics.fmean(mid) if mid else float("nan")
            row[x] = {"mean_dv": round(mean, 4), "median_dv": round(statistics.median(vals), 4),
                      "mean_dv_unsaturated": round(mean_mid, 4), "n_unsaturated": len(mid),
                      "rel_pawn": round(mean / pawn, 2) if abs(pawn) > 1e-6 else None,
                      "rel_pawn_unsaturated": round(mean_mid / pawn_mid, 2)
                      if mid and abs(pawn_mid) > 1e-6 else None}
        table[p] = row
    return {"model": args.model, "channels": net.channels, "positions": len(metas),
            "mean_base_value": round(statistics.fmean(base.values()), 4),
            "share_unsaturated": round(sum(abs(b) < 0.5 for b in base.values()) / len(base), 4),
            "dv": table, "folklore": FOLKLORE}


def print_report(rep: dict) -> None:
    print(f"model {rep['model']} ({rep['channels']} channels), {rep['positions']} positions, "
          f"mean V {rep['mean_base_value']}, {rep['share_unsaturated']:.0%} with |V| < 0.5")
    print(f"{'pocket':<10}" + "".join(f"{x:>22}" for x in PIECES))
    for p, row in rep["dv"].items():
        print(f"{p:<10}" + "".join(f"{row[x]['mean_dv']:>+10.4f} ({row[x]['rel_pawn']!s:>5}P)    " for x in PIECES))
    print("unsaturated (|V| < 0.5):")
    for p, row in rep["dv"].items():
        print(f"{p:<10}" + "".join(f"{row[x]['mean_dv_unsaturated']:>+10.4f} "
                                   f"({row[x]['rel_pawn_unsaturated']!s:>5}P)    " for x in PIECES))


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shards", action="append", required=True, help="directory with chunk .pos.jsonl sidecars")
    ap.add_argument("--model", default=os.path.join(MODELS_DIR, "hivemind-clock-v2-fp32.onnx"))
    ap.add_argument("--positions", type=int, default=300, help="positions sampled (0 = all)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--json", default=None)
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    rep = run(args)
    print_report(rep)
    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w") as f:
            json.dump(rep, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
