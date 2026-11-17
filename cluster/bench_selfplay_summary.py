"""Summary of a cluster/bench_selfplay.sbatch run (WP-I, docs/measurements_bench.md).

    .venv/bin/python cluster/bench_selfplay_summary.py runs/cluster/bench_selfplay_<job> [--cpus 8]

Per stage (one arena.selfplay run): games per GPU-hour two ways (end to end over the stage's wall time, and steady
state = workers x 3600 / mean game wall time, which leaves out the tail where fewer than `workers` games run), mean
game wall and clock length, searches per second, GPU utilisation (mean, median, p90 of the nvidia-smi samples),
peak GPU memory, and CPU use of the job as cores busy and as a share of the allocated cores. Then the three decision
lines the doc names: C++ cross-game batching server (only if GPU utilisation < 30%), CPU-bound or not, and the
training-round count (halve it if under 50 games per GPU-hour at the self-play base-time mix). Writes
<run>/report.json with everything printed.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys

# Spec v3 D2: self-play base time drawn from {60, 120, 180, 300} s with these weights.
BASE_MIX = {60: 0.25, 120: 0.25, 180: 0.35, 300: 0.15}
UTIL_BATCHING = 30.0     # % GPU utilisation under which a C++ batching server is worth building (audit WP-I)
CPU_BOUND = 0.85         # share of allocated cores busy above which the stage is CPU-bound
MIN_GAMES_PER_HOUR = 50  # audit section 4: under this, halve the training-round count
ROUND_GAMES = 10_000     # games per training round (audit section 4)


def pct(xs: list[float], q: float) -> float | None:
    if not xs:
        return None
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * (len(xs) - 1) + 0.5))]


def read_gpu(path: str) -> dict | None:
    if not os.path.isfile(path):
        return None
    util, mem, power = [], [], []
    with open(path) as f:
        for row in csv.reader(f):
            if len(row) < 5:
                continue
            try:
                util.append(float(row[1]))
                mem.append(float(row[3]))
                power.append(float(row[4]))
            except ValueError:
                continue  # "[N/A]" or a cut-off last line
    if not util:
        return None
    return {"samples": len(util), "util_mean": round(statistics.fmean(util), 1),
            "util_median": pct(util, 0.5), "util_p90": pct(util, 0.9),
            "share_idle": round(sum(u == 0 for u in util) / len(util), 3),
            "mem_mib_max": max(mem), "power_w_mean": round(statistics.fmean(power), 1)}


def read_cpu(path: str) -> dict | None:
    """Cores busy from the counter samples: cgroup usage_usec (exact, this job only) or /proc/stat jiffies (whole
    machine, so an upper bound on a shared node)."""
    if not os.path.isfile(path):
        return None
    rows = []
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) == 3:
                try:
                    rows.append((float(parts[0]), parts[1], float(parts[2])))
                except ValueError:
                    pass
    if len(rows) < 2:
        return None
    (t0, kind, c0), (t1, _, c1) = rows[0], rows[-1]
    if t1 <= t0:
        return None
    per_s = 1e6 if kind == "cgroup_usec" else os.sysconf("SC_CLK_TCK")
    return {"source": kind, "cores_busy": round((c1 - c0) / per_s / (t1 - t0), 2)}


def stage_report(run: str, row: dict, cpus: int | None) -> dict:
    name, workers = row["stage"], int(row["workers"])
    wall = max(1.0, float(row["end"]) - float(row["start"]))
    base_s = None if row["base_ms"] == "mix" else int(row["base_ms"]) / 1000  # None = the production mix
    rep = {"stage": name, "base_s": base_s, "workers": workers, "games_asked": int(row["games"]),
           "exit": int(row["exit"]), "wall_s": wall}
    summary_path = os.path.join(run, name, "summary.json")
    if os.path.isfile(summary_path):
        s = json.load(open(summary_path))
        games = [g for g in s["games"] if "error" not in g]
        r = s["results"]
        game_wall = [g["wall_s"] for g in games]
        searches = r["searches"]
        rep.update({
            "games": r["games"], "errors": r["errors"], "flag_rate": r["flag_rate"], "reasons": r["reasons"],
            "mean_clock_len_s": r["mean_duration_s"], "samples": r["samples"],
            "mean_game_wall_s": round(statistics.fmean(game_wall), 1) if game_wall else None,
            "games_per_gpu_hour": round(r["games"] * 3600 / wall, 1),
            "games_per_gpu_hour_steady": round(workers * 3600 / statistics.fmean(game_wall), 1) if game_wall else None,
            "searches_per_s": round(searches / wall, 1),
            "samples_per_gpu_hour": round(r["samples"] * 3600 / wall),
        })
    rep["gpu"] = read_gpu(os.path.join(run, name, "gpu.csv"))
    rep["cpu"] = read_cpu(os.path.join(run, name, "cpu.tsv"))
    if rep["cpu"] and cpus:
        rep["cpu"]["cpus"] = cpus
        rep["cpu"]["share_busy"] = round(rep["cpu"]["cores_busy"] / cpus, 2)
    return rep


def mix_rate(stages: list[dict]) -> tuple[float | None, str]:
    """Games per GPU-hour at the self-play base-time mix and how it was found. A "mix" stage (base time drawn from the
    production mix) is used as measured (6 workers preferred). Otherwise from the two stages with the most base times
    at one worker count (6 preferred): per-game GPU time is taken as linear in base time (a + b x base) and averaged
    over the mix. With one base time only, that stage's steady rate scaled by base time (a rough guess)."""
    mixed = [s for s in stages if s["base_s"] is None and s.get("games_per_gpu_hour_steady")]
    if mixed:
        best = max(mixed, key=lambda s: (s["workers"] == 6, s.get("games") or 0))
        return best["games_per_gpu_hour_steady"], f"measured on stage {best['stage']}"
    by_workers: dict[int, dict[float, float]] = {}
    for s in stages:
        if s.get("games_per_gpu_hour_steady") and s["base_s"] is not None:
            by_workers.setdefault(s["workers"], {})[s["base_s"]] = 3600 / s["games_per_gpu_hour_steady"]
    if not by_workers:
        return None, ""
    workers = max(by_workers, key=lambda w: (len(by_workers[w]), w == 6))
    pts = sorted(by_workers[workers].items())
    if len(pts) >= 2:
        (x0, y0), (x1, y1) = pts[0], pts[-1]
        b = (y1 - y0) / (x1 - x0)
        a = y0 - b * x0
        cost = sum(w * (a + b * base) for base, w in BASE_MIX.items())
        how = f"interpolated from {pts[0][0]:.0f} s and {pts[-1][0]:.0f} s at {workers} workers"
    else:
        x0, y0 = pts[0]
        cost = sum(w * y0 * base / x0 for base, w in BASE_MIX.items())
        how = f"ROUGH: one base time only ({x0:.0f} s), scaled by base time"
    return (round(3600 / cost, 1), how) if cost > 0 else (None, "")


def fmt(v, unit="") -> str:
    return "n/a" if v is None else f"{v}{unit}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run", help="runs/cluster/bench_selfplay_<job>")
    ap.add_argument("--cpus", type=int, default=None, help="cores allocated to the job")
    ap.add_argument("--model", default=None, help="network used (recorded in report.json)")
    ap.add_argument("--regime", default=None, help="self-play regime used (recorded in report.json)")
    args = ap.parse_args(argv)

    with open(os.path.join(args.run, "stages.tsv")) as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    if not rows:
        print("no stages finished", file=sys.stderr)
        return 1
    stages = [stage_report(args.run, r, args.cpus) for r in rows]

    print(f"{'stage':<16}{'games':>6}{'err':>4}{'g/GPU-h':>9}{'steady':>8}{'game wall':>10}{'clock':>7}"
          f"{'srch/s':>8}{'flag':>6}{'GPU% mean/p50/p90':>20}{'idle':>6}{'mem MiB':>8}{'CPU busy':>10}")
    for s in stages:
        g, c = s.get("gpu") or {}, s.get("cpu") or {}
        gpu = (f"{g['util_mean']}/{g['util_median']:.0f}/{g['util_p90']:.0f}" if g else "n/a")
        cpu = (f"{c['cores_busy']}" + (f" ({c['share_busy']:.0%})" if "share_busy" in c else "")) if c else "n/a"
        print(f"{s['stage']:<16}{fmt(s.get('games')):>6}{fmt(s.get('errors')):>4}{fmt(s.get('games_per_gpu_hour')):>9}"
              f"{fmt(s.get('games_per_gpu_hour_steady')):>8}{fmt(s.get('mean_game_wall_s'), 's'):>10}"
              f"{fmt(s.get('mean_clock_len_s'), 's'):>7}{fmt(s.get('searches_per_s')):>8}"
              f"{fmt(s.get('flag_rate')):>6}{gpu:>20}{fmt(g.get('share_idle')):>6}{fmt(g.get('mem_mib_max')):>8}"
              f"{cpu:>10}")
        if s["exit"] != 0:
            print(f"  ! {s['stage']} exited {s['exit']}")

    decisions = []
    utils = [s["gpu"]["util_mean"] for s in stages if s.get("gpu") and s["workers"] == 6] or \
            [s["gpu"]["util_mean"] for s in stages if s.get("gpu")]
    if utils:
        u = max(utils)
        decisions.append(f"GPU utilisation at production settings {u}% (best stage) -> " + (
            f"under {UTIL_BATCHING:.0f}%: a C++ cross-game batching server is worth building, after WP-C's protocol "
            "fixes are in and this bench is rerun on them" if u < UTIL_BATCHING else
            f"at least {UTIL_BATCHING:.0f}%: no batching server; the GPU is already busy"))
    else:
        decisions.append("GPU utilisation: no nvidia-smi samples (not a GPU run), no batching decision")
    busy = [s["cpu"]["share_busy"] for s in stages if s.get("cpu") and "share_busy" in s["cpu"]]
    if busy:
        b = max(busy)
        src = {s["cpu"]["source"] for s in stages if s.get("cpu")}
        decisions.append(f"CPU busy up to {b:.0%} of the allocated cores ({', '.join(sorted(src))}) -> " + (
            "CPU-bound: ask for more cores (sbatch -n16) or fewer search threads before any GPU-side work"
            if b >= CPU_BOUND else "not CPU-bound"))
        if b >= CPU_BOUND and utils and max(utils) < UTIL_BATCHING:
            decisions.append("low GPU use on CPU-bound stages says little about batching: rerun with sbatch -n16 and "
                             "decide on the batching server from that run")
    rate, how = mix_rate(stages)
    if rate:
        hours = round(ROUND_GAMES / rate)
        decisions.append(f"games per GPU-hour at the base-time mix {BASE_MIX} ({how}): {rate} -> {ROUND_GAMES} games cost "
                         f"~{hours} GPU-hours per round" + (
                             f"; under {MIN_GAMES_PER_HOUR}: halve the training-round count, not the ablations"
                             if rate < MIN_GAMES_PER_HOUR else ""))
    w = {s["workers"]: s.get("games_per_gpu_hour_steady") for s in stages if s["base_s"] == 60}
    if len(w) >= 2:
        lo, hi = min(w), max(w)
        if w[lo] and w[hi]:
            decisions.append(f"60 s games: {lo} -> {hi} workers changes steady games/GPU-hour {w[lo]} -> {w[hi]} "
                             f"(x{w[hi] / w[lo]:.2f}); near x{hi / lo:.1f} means just run more workers")
    print("\n" + "\n".join("* " + d for d in decisions))

    report = {"run": args.run, "model": args.model, "regime": args.regime, "cpus": args.cpus, "stages": stages,
              "games_per_gpu_hour_mix": rate, "games_per_gpu_hour_mix_how": how, "decisions": decisions}
    with open(os.path.join(args.run, "report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nwrote {os.path.join(args.run, 'report.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
