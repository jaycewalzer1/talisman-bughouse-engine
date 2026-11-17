"""How a network's raw outputs respond to the clocks (audit 2026-11-17 WP-H, theory item 4).

Two measurements on a sample of recorded positions (chunk ``.pos.jsonl`` sidecars), raw network, no search:

1. **dV per +-10 s per clock, by material density.** Each of the four clocks (by role from the searching team's
   view: ``ours_A``, ``theirs_A``, ``ours_B``, ``theirs_B``) is moved by +10 s and by -10 s (floored at 100 ms) and
   dV = V(changed) - V(recorded) is averaged per role and per material bucket (men on both boards, kings and
   pockets excluded; the start position has 60).
2. **Pass rate vs diagonal.** On every board where the team is on turn, the seat's diagonal (own clock minus the
   partner's opponent's) is set to each value of ``--diagonals`` by moving those two clocks symmetrically about
   their mean; reported per diagonal: mean raw p(pass) over the legal moves plus the pass slot, and the share of
   positions where the pass is the network's top-1.

The TimeAdvantage plane follows the changed clocks as the arena sets it (``team_time_edge`` > 0), so for the stock
74-channel it04, whose only time input is that plane, every response goes through its sign; ``--hold-ta`` keeps
it at the recorded value instead, which isolates the clock planes of a widened network.

    .venv/bin/python tools/clock_response.py --shards runs/custom/local1h/it01/selfplay/training_data \\
        --model third_party/hivemind/engine/models/hivemind-fp32.onnx --positions 300 [--json out.json]

Single process, CPU ONNX Runtime.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from arena.interfaces import OPPONENT, PARTNER  # noqa: E402
from arena.players.heuristic import team_time_edge  # noqa: E402
from implied_values import sample_positions  # noqa: E402
from pass_stats import MATERIAL_BINS, bin_label, material, on_turn_seats  # noqa: E402
from regression import MODELS_DIR, NetEval  # noqa: E402

STEP_MS = 10000
FLOOR_MS = 100
DIAGONALS_S = (-30, -20, -10, -5, -2, -1, 1, 2, 5, 10, 20, 30)


def role_seat(team: str, role: str) -> str:
    """Seat of ``role`` (ours_A / theirs_A / ours_B / theirs_B) for ``team``."""
    t1 = {"ours_A": "A_white", "theirs_A": "A_black", "ours_B": "B_black", "theirs_B": "B_white"}[role]
    return t1 if team == "T1" else OPPONENT[t1]


def _ta(net_hold: bool, meta: dict, clocks: dict, team: str) -> bool:
    return bool(meta.get("timeAdvantage", False)) if net_hold else team_time_edge(clocks, team) > 0


def run(args) -> dict:
    net = NetEval(args.model, threads=args.threads)
    metas = sample_positions(args.shards, args.positions, args.seed)
    planes, keys = [], []
    roles = ("ours_A", "theirs_A", "ours_B", "theirs_B")
    for i, m in enumerate(metas):
        team, clocks = m["team"], {k: int(v) for k, v in m["clocks"].items()}
        planes.append(net.planes(m["fenA"], m["fenB"], team, clocks, m.get("elapsed"),
                                 _ta(args.hold_ta, m, clocks, team)))
        keys.append(("base", i, None, None))
        for role in roles:
            seat = role_seat(team, role)
            for sgn in (+1, -1):
                c = dict(clocks)
                c[seat] = max(FLOOR_MS, c[seat] + sgn * STEP_MS)
                planes.append(net.planes(m["fenA"], m["fenB"], team, c, m.get("elapsed"),
                                         _ta(args.hold_ta, m, c, team)))
                keys.append(("step", i, role, sgn))
        for board, seat in on_turn_seats(m["fenA"], m["fenB"], team).items():
            other = OPPONENT[PARTNER[seat]]
            mid = (clocks[seat] + clocks[other]) / 2
            for d in args.diagonals:
                c = dict(clocks)
                c[seat] = int(max(FLOOR_MS, mid + d * 500))
                c[other] = int(max(FLOOR_MS, mid - d * 500))
                planes.append(net.planes(m["fenA"], m["fenB"], team, c, m.get("elapsed"),
                                         _ta(args.hold_ta, m, c, team)))
                keys.append(("diag", i, board, d))
    out = net.run(np.stack(planes))
    v = out["value"][:, 0]
    base = {k[1]: float(v[j]) for j, k in enumerate(keys) if k[0] == "base"}
    steps: dict = defaultdict(list)
    sweep: dict = defaultdict(list)
    for j, (kind, i, a, b) in enumerate(keys):
        m = metas[i]
        if kind == "step":
            bucket = bin_label(material(m["fenA"], m["fenB"]), MATERIAL_BINS)
            dv = float(v[j]) - base[i]
            steps[(a, b, bucket)].append(dv)
            steps[(a, b, "all")].append(dv)
        elif kind == "diag":
            logits = out["pi_a"][j] if a == "A" else out["pi_b"][j]
            pol = net.board_policy(logits, m["fenA"] if a == "A" else m["fenB"])
            sweep[b].append((pol["p_pass"], pol["top"] == "pass", float(v[j])))
    buckets = sorted({k[2] for k in steps}, key=lambda s: (s == "all", s))
    dv_table = {}
    for bucket in buckets:
        row = {}
        for role in roles:
            plus, minus = steps.get((role, 1, bucket), []), steps.get((role, -1, bucket), [])
            if plus:
                row[role] = {"n": len(plus), "dv_plus10": round(statistics.fmean(plus), 4),
                             "dv_minus10": round(statistics.fmean(minus), 4),
                             "abs_dv_plus10": round(statistics.fmean(abs(x) for x in plus), 4),
                             "abs_dv_minus10": round(statistics.fmean(abs(x) for x in minus), 4)}
        dv_table[bucket] = row
    sweep_table = {str(d): {"n": len(sweep[d]), "mean_p_pass": round(statistics.fmean(x[0] for x in sweep[d]), 4),
                            "share_pass_top": round(statistics.fmean(x[1] for x in sweep[d]), 4),
                            "mean_value": round(statistics.fmean(x[2] for x in sweep[d]), 4)}
                   for d in args.diagonals if sweep[d]}
    return {"model": args.model, "channels": net.channels, "positions": len(metas), "hold_ta": args.hold_ta,
            "dv_per_10s": dv_table, "pass_vs_diagonal": sweep_table}


def print_report(rep: dict) -> None:
    print(f"model {rep['model']} ({rep['channels']} channels), {rep['positions']} positions, "
          f"TimeAdvantage {'held' if rep['hold_ta'] else 'follows the clocks'}")
    print("\ndV for +10 s / -10 s on one clock (mean, and mean |dV|), by material bucket")
    roles = ("ours_A", "theirs_A", "ours_B", "theirs_B")
    print(f"  {'bucket':<12}" + "".join(f"{r:>30}" for r in roles))
    for bucket, row in rep["dv_per_10s"].items():
        cells = []
        for r in roles:
            c = row.get(r)
            cells.append(f"{c['dv_plus10']:+.3f}/{c['dv_minus10']:+.3f} |{c['abs_dv_plus10']:.3f}| n={c['n']}"
                         if c else "-")
        print(f"  {bucket:<12}" + "".join(f"{x:>30}" for x in cells))
    print("\npass vs the on-turn seat's diagonal (s)")
    print(f"  {'diag':>6} {'n':>6} {'p(pass)':>9} {'pass top-1':>11} {'V':>8}")
    for d, s in rep["pass_vs_diagonal"].items():
        print(f"  {d:>6} {s['n']:>6} {s['mean_p_pass']:>9.4f} {s['share_pass_top']:>11.4f} {s['mean_value']:>+8.4f}")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shards", action="append", required=True, help="directory with chunk .pos.jsonl sidecars")
    ap.add_argument("--model", default=os.path.join(MODELS_DIR, "hivemind-clock-v2-fp32.onnx"))
    ap.add_argument("--positions", type=int, default=300, help="positions sampled (0 = all)")
    ap.add_argument("--diagonals", type=lambda s: [int(x) for x in s.split(",")], default=list(DIAGONALS_S),
                    help="diagonals in s for the pass sweep, comma list (write --diagonals=-10,10 when it starts with -)")
    ap.add_argument("--hold-ta", action="store_true", help="keep the TimeAdvantage plane at its recorded value")
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
