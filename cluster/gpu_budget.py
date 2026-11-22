#!/usr/bin/env python3
"""What the PACE account can still afford, in GPU-hours, and what to give the next run.

    .venv/bin/python cluster/gpu_budget.py                      # the budget and a plan for half of what is left
    .venv/bin/python cluster/gpu_budget.py --share 1.0 --run final1   # the final run: everything but the reserve

The account (paceship-bughouse) is charged in dollars: `pace-quota` shows its balance, `sshare` its GPU-minutes used.
The dollar rate of an RTX 6000 GPU-hour is measured, not assumed: every call appends a snapshot (balance, GPU-minutes)
to runs/budget/snapshots.tsv, and the rate is the balance drop over the GPU-hours used between the oldest and newest
snapshots once 20 or more GPU-hours lie between them. Until then it uses --rate (default 0.1491 $/GPU-h, PACE's
published 2022 internal RTX 6000 rate, https://pace.gatech.edu/rate-study) and says so.

The plan keeps --reserve GPU-hours back (the final 200-game round robin of three engines, cluster/launch_final_match.sh:
3 pairs x 200 games x ~1.6 min a game at 120 s, about 20 GPU-h, plus slack), gives --share of the rest to the next
run, and prints the launch line (cluster/launch_full_run.sh with WORKERS and GPU_HOURS), with the iterations that cap
buys at the measured self-play rate R (games per GPU-hour, cluster/wave2_selfplay_bench.py) plus about 3 GPU-hours per
iteration for validation, training and the gate.
"""

import argparse
import csv
import datetime as dt
import math
import os
import re
import subprocess
import sys

ACCOUNT = "paceship-bughouse"
SNAPSHOTS = "runs/budget/snapshots.tsv"
FIELDS = ["time", "balance", "reserved", "available", "gpu_minutes"]


def balance():
    """(balance, reserved, available) in dollars from pace-quota's account table."""
    out = subprocess.run(["pace-quota"], capture_output=True, text=True, timeout=120).stdout
    out = re.sub(r"\x1b\[[0-9;]*m", "", out)
    for line in out.splitlines():
        parts = line.split()
        if parts and parts[0] == ACCOUNT and len(parts) >= 4:
            return tuple(float(p.replace(",", "")) for p in parts[1:4])
    sys.exit(f"pace-quota shows no {ACCOUNT} balance")


def gpu_minutes():
    out = subprocess.run(["sshare", "-A", ACCOUNT, "-l", "-P", "-n", "-o", "Account,User,GrpTRESRaw%400"],
                         capture_output=True, text=True, timeout=120).stdout
    for line in out.splitlines():
        m = re.search(r"gres/gpu=(\d+)", line)
        if m:
            return int(m.group(1))
    sys.exit(f"sshare shows no GPU usage for {ACCOUNT}")


def snapshots(path, new):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rows = []
    if os.path.isfile(path):
        with open(path) as f:
            rows = list(csv.DictReader(f, delimiter="\t"))
    rows.append(new)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, FIELDS, delimiter="\t")
        w.writeheader()
        w.writerows(rows)
    return rows


def measured_rate(rows, min_gpu_hours=20.0):
    """$ per GPU-hour from the oldest and newest snapshots, or None while too little lies between them. The balance
    (not available) is compared: a running job's reservation is not a charge yet."""
    if len(rows) < 2:
        return None
    a, b = rows[0], rows[-1]
    hours = (int(b["gpu_minutes"]) - int(a["gpu_minutes"])) / 60
    spent = float(a["balance"]) - float(b["balance"])
    if hours < min_gpu_hours or spent <= 0:
        return None
    return spent / hours, hours


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--share", type=float, default=0.5, help="share of the budget after the reserve for the next run")
    ap.add_argument("--reserve", type=float, default=30.0, help="GPU-hours kept for the final match (default 30)")
    ap.add_argument("--rate", type=float, default=0.1491, help="$ per GPU-hour until a rate is measured")
    ap.add_argument("--rate-per-hour-r", type=float, default=726.0,
                    help="self-play games per GPU-hour (hybrid narrow bench, round r4)")
    ap.add_argument("--games", type=int, default=20000, help="self-play games per iteration in the plan")
    ap.add_argument("--workers", type=int, default=7, help="self-play workers in the plan")
    ap.add_argument("--run", default="par1", help="run name in the printed launch line")
    ap.add_argument("--config", default="hybrid")
    args = ap.parse_args()

    bal, reserved, available = balance()
    used = gpu_minutes()
    rows = snapshots(SNAPSHOTS, {"time": dt.datetime.now().isoformat(timespec="seconds"), "balance": bal,
                                 "reserved": reserved, "available": available, "gpu_minutes": used})
    m = measured_rate(rows)
    if m:
        rate, over = m
        how = f"measured over {over:.0f} GPU-h between {rows[0]['time']} and {rows[-1]['time']}"
    else:
        rate, how = args.rate, f"ASSUMED (--rate; PACE 2022 rate study) until {SNAPSHOTS} spans 20+ GPU-h"
    left = available / rate
    spendable = max(0.0, left - args.reserve)
    cap = math.floor(spendable * args.share)
    per_iter = args.games / args.rate_per_hour_r + 3.0
    iters = max(1, int(cap // per_iter))
    gpus = args.workers + 1
    wall = per_iter / gpus + 3.0  # self-play spread over the GPUs, then training and the gate on the main job
    print(f"account {ACCOUNT}: balance ${bal:.2f}, reserved ${reserved:.2f}, available ${available:.2f}")
    print(f"GPU-hours used so far: {used / 60:.0f}")
    print(f"rate: ${rate:.4f} per GPU-hour ({how})")
    print(f"left: {left:.0f} GPU-hours; reserve {args.reserve:.0f} for the final match; {spendable:.0f} to spend")
    print(f"next run: {args.share:.0%} of that = {cap} GPU-hours")
    print(f"  at R = {args.rate_per_hour_r:.0f} games/GPU-h and {args.games} games per iteration: about "
          f"{per_iter:.1f} GPU-h per iteration -> {iters} iterations, about {iters * wall / 24:.1f} days on "
          f"{gpus} GPUs (if the queue gives them)")
    print(f"launch on PACE:\n  RUN={args.run} CONFIG={args.config} WIDE=0 WORKERS={args.workers} "
          f"GAMES_PER_ITER={args.games} ITERS={iters} CHAIN={math.ceil(iters / 2) + 1} GPU_HOURS={cap} "
          f"bash cluster/launch_full_run.sh")


if __name__ == "__main__":
    main()
