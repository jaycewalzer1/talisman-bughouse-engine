"""GO or NO-GO for the full wave 2 training run, from the wave 2c readiness round (cluster/launch_wave2c.sh) and the
previous round's narrow/wide A/B. On GO it prints the one-line launch command (cluster/launch_full_run.sh) and its
GPU-hour estimate from the measured self-play rate.

    .venv/bin/python cluster/decide_full_run.py                  # the latest round (runs/wave2c/LATEST)
    .venv/bin/python cluster/decide_full_run.py --round r2 --ab runs/wave2/ab2 --run full1

Reads runs/wave2c/<ROUND>/: build.ok (written by the build job on success), gpu_check.json, rows/<row>/ (arena rows,
scored with tools/compare_evals.py), bench/<config>_<width>*/bench.json, smoke_<config>/{narrow,wide}/ (wave2_iterate
lineages); and <AB>/results_ab.tsv (default runs/wave2/ab2). The queue (squeue) says whether the round is still running.

Blockers (any one is NO-GO):
  B1 build      build.ok missing: cluster/build.sh did not pass
  B2 gpu        gpu_check.json missing or "overall" not PASS (each failed check is named)
  B3 smoke      a smoke lineage did not finish an iteration (self-play -> train -> gate): smoke_nodur and smoke_cont, each
                narrow and wide, need runs/rl-it1/stages/iteration.done and a vs_it13 gate score in results.tsv
  B4 control    c_same_120 score outside [0.46, 0.54]
  B5 rows       a row of the round (11: cluster/wave2c_configs.py ROWS) missing, under 250 games, with errors in more than
                5% of its games, or a w2c_* job still in the queue
  B6 rate       no self-play rate R (bench.json with every game played) for the chosen config and width
  B7 budget     the full run's estimate is over the 10,000 GPU-hour budget

Config choice (all against c_nodur_120 of this round): the candidates are c_all_120 (cont), c_all_nod1_120 (cont_nod1)
and the best-scoring single-switch row (c_reuse/joint/carry/ownpace/waiteval/mateblock_120). A candidate qualifies when
  its 95% range lies wholly above c_nodur_120's (lo > nodur hi), or
  its score beats c_nodur_120's by at least 0.05 with at least 500 games in both rows.
The qualifying candidate with the highest score is chosen; with none, nodur (durative tree off, sit gate on). A chosen
config other than nodur and cont was not benched or smoke-trained: it borrows cont's rate and smoke (said in the output).

Narrow vs wide, from <AB>/results_ab.tsv (each iteration's gate vs it13, 400 games, about +-0.05): wide when at least
2 iterations have both arms' vs_it13, the wide arm's mean over those iterations is at least 0.03 above the narrow
arm's, and the wide arm's last score is not below the narrow arm's. Otherwise narrow, also when the file is missing.

Full run: ITERS 10 x GAMES_PER_ITER 50,000 (+ 500 validation games per iteration), the plan's ~1,700 GPU-hours at
R = 300. Estimate = ITERS x (GAMES_PER_ITER + 500) / R + ITERS x 1.3 GPU-hours (training and two gate matches per
iteration; cluster/README.md). R = the bench's steady games per GPU-hour (end to end when steady is missing).
CHAIN = ceil(1.1 x estimate / 16) + ITERS chained 16 h jobs (each job stops at an iteration's end, and jobs left over
after iteration ITERS exit at once). Exit code 0 on GO, 1 on NO-GO.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass, field

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from wave2c_configs import BENCHED, BENCHED_AS, CONFIGS, ROWS, SINGLE_SWITCH  # noqa: E402

SAME_RANGE = (0.46, 0.54)
MIN_ROW_GAMES = 250
MAX_ERROR_SHARE = 0.05
MARGIN = 0.05
MARGIN_MIN_GAMES = 500
WIDE_MARGIN = 0.03
WIDE_MIN_ITERS = 2
FULL_ITERS = 10
FULL_GAMES_PER_ITER = 50_000
VAL_GAMES = 500
ITER_OVERHEAD_H = 1.3
JOB_HOURS = 16
PLAN_GPU_H = 1_700
BUDGET_GPU_H = 10_000
SMOKES = tuple((config, width) for config in BENCHED for width in ("narrow", "wide"))


@dataclass
class Decision:
    go: bool = False
    blockers: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    config: str = "nodur"
    wide: int = 0
    rate: float | None = None
    gpu_hours: float | None = None
    chain: int | None = None
    run: str = "full1"
    command: str | None = None


# ---------------------------------------------------------------------- readers

def row_stats(path: str, n_boot: int = 500) -> dict | None:
    """games, score, ci95, errors of one arena row (tools/compare_evals.py's scorer), or None without games."""
    from tools.compare_evals import row

    r = row(path, None, n_boot=n_boot)
    if r is None:
        return None
    errors = r.get("errors", 0)
    if not r["finished"]:
        res = os.path.join(path, "results.jsonl")
        if os.path.exists(res):
            with open(res) as f:
                errors = sum('"error"' in line for line in f)
    return {"games": r["games"], "score": r["score"], "ci95": list(r["ci95"]), "errors": errors,
            "finished": r["finished"]}


def bench_rate(round_dir: str, config: str, width: str) -> tuple[float | None, str]:
    """(games per GPU-hour, where from) from the newest complete bench of config and width."""
    found = []
    for path in glob.glob(os.path.join(round_dir, "bench", f"{config}_{width}*", "bench.json")):
        try:
            with open(path) as f:
                b = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        if b.get("exit") == 0 and b.get("games") and b.get("games") == b.get("games_asked"):
            found.append((os.path.getmtime(path), path, b))
    if not found:
        return None, f"no complete bench of {config} {width} under {round_dir}/bench"
    _, path, b = max(found)
    rate = b.get("games_per_gpu_hour_steady") or b.get("games_per_gpu_hour")
    how = "steady" if b.get("games_per_gpu_hour_steady") else "end to end"
    return rate, f"{path}: {b['games']} games, {how}"


def smoke_status(round_dir: str, config: str, width: str) -> tuple[bool, str]:
    """Did smoke_<config>/<width> finish iteration 1 (self-play, train, gate)?"""
    arm = os.path.join(round_dir, f"smoke_{config}", width)
    done = os.path.join(arm, "runs", "rl-it1", "stages", "iteration.done")
    tsv = os.path.join(arm, "results.tsv")
    if not os.path.isfile(done):
        return False, f"{arm}: no runs/rl-it1/stages/iteration.done"
    if not os.path.isfile(tsv):
        return False, f"{arm}: no results.tsv"
    with open(tsv) as f:
        rows = [r for r in csv.DictReader(f, delimiter="\t") if r.get("iteration") == "rl-it1"]
    if not rows or not rows[0].get("vs_it13"):
        return False, f"{arm}: rl-it1 has no gate score vs it13 in results.tsv"
    return True, f"{arm}: rl-it1 vs_it13 {rows[0]['vs_it13']}, promoted {rows[0].get('promoted', '')}"


def queued_round_jobs() -> list[str]:
    """Names of this user's w2c_* jobs still in the queue ([] when squeue is not there, e.g. on the Mac)."""
    try:
        out = subprocess.run(["squeue", "-h", "-u", os.environ.get("USER", ""), "-o", "%j"],
                             capture_output=True, text=True, timeout=60).stdout
    except (OSError, subprocess.TimeoutExpired):
        return []
    return sorted({n for n in out.split() if n.startswith("w2c_") and n != "w2c_watchdog"})


# ---------------------------------------------------------------------- rules

def choose_config(rows: dict[str, dict | None]) -> tuple[str, list[str]]:
    """The config to train (a CONFIGS name) and the reasons, from the round's rows (rule in the module docstring)."""
    by_row = {row: name for name, (row, _) in CONFIGS.items()}
    nodur = rows.get(CONFIGS["nodur"][0])
    if not nodur:
        return "nodur", ["c_nodur_120 has no games: nodur by default"]
    singles = [CONFIGS[n][0] for n in SINGLE_SWITCH if rows.get(CONFIGS[n][0])]
    candidates = [CONFIGS["cont"][0], CONFIGS["cont_nod1"][0]]
    if singles:
        candidates.append(max(singles, key=lambda r: rows[r]["score"]))
    reasons, qualified = [], []
    for row in candidates:
        r = rows.get(row)
        if not r:
            reasons.append(f"{row}: no games")
            continue
        diff = r["score"] - nodur["score"]
        disjoint = r["ci95"][0] > nodur["ci95"][1]
        margin = diff >= MARGIN and r["games"] >= MARGIN_MIN_GAMES and nodur["games"] >= MARGIN_MIN_GAMES
        ok = disjoint or margin
        reasons.append(f"{row} {r['score']:.3f} [{r['ci95'][0]:.3f}, {r['ci95'][1]:.3f}] {r['games']} games vs "
                       f"c_nodur_120 {nodur['score']:.3f} [{nodur['ci95'][0]:.3f}, {nodur['ci95'][1]:.3f}] "
                       f"{nodur['games']} games: {diff:+.3f}, "
                       + ("QUALIFIES (" + ("ranges disjoint" if disjoint else f">= {MARGIN} with >= "
                                           f"{MARGIN_MIN_GAMES} games each") + ")" if ok else "does not qualify"))
        if ok:
            qualified.append(row)
    if not qualified:
        return "nodur", reasons + ["no continuity candidate beats c_nodur_120 by the rule: nodur"]
    best = max(qualified, key=lambda r: rows[r]["score"])
    return by_row[best], reasons + [f"chosen: {best} ({by_row[best]})"]


def choose_width(ab_tsv: str) -> tuple[int, list[str]]:
    """1 (wide) or 0 (narrow) and the reasons, from the A/B's results_ab.tsv (rule in the module docstring)."""
    if not os.path.isfile(ab_tsv):
        return 0, [f"{ab_tsv} not found: narrow"]
    with open(ab_tsv) as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    both = []
    for r in rows:
        try:
            both.append((r["iteration"], float(r["narrow_vs_it13"]), float(r["wide_vs_it13"])))
        except (KeyError, TypeError, ValueError):
            continue
    if len(both) < WIDE_MIN_ITERS:
        return 0, [f"{ab_tsv}: {len(both)} iteration(s) with both arms gated (< {WIDE_MIN_ITERS}): narrow"]
    narrow = sum(n for _, n, _ in both) / len(both)
    wide = sum(w for _, _, w in both) / len(both)
    last_n, last_w = both[-1][1], both[-1][2]
    table = ", ".join(f"{it} {n:.3f}/{w:.3f}" for it, n, w in both)
    why = (f"{ab_tsv}: narrow/wide vs it13 by iteration {table}; means {narrow:.3f}/{wide:.3f} "
           f"({wide - narrow:+.3f}), last {last_n:.3f}/{last_w:.3f}")
    if wide - narrow >= WIDE_MARGIN and last_w >= last_n:
        return 1, [why, f"wide: mean ahead by >= {WIDE_MARGIN} and last not behind"]
    return 0, [why, f"narrow: wide not ahead by >= {WIDE_MARGIN} on the mean with its last not behind"]


def estimate_gpu_hours(rate: float, iters: int = FULL_ITERS, games: int = FULL_GAMES_PER_ITER,
                       val: int = VAL_GAMES) -> float:
    return iters * (games + val) / rate + iters * ITER_OVERHEAD_H


def chain_jobs(gpu_hours: float, iters: int = FULL_ITERS) -> int:
    return math.ceil(1.1 * gpu_hours / JOB_HOURS) + iters


def free_run_name(runs_full: str) -> str:
    n = 1
    while os.path.exists(os.path.join(runs_full, f"full{n}")):
        n += 1
    return f"full{n}"


def decide(round_dir: str, ab_dir: str, run: str | None = None, queue: list[str] | None = None,
           runs_full: str = os.path.join("runs", "full")) -> Decision:
    d = Decision(run=run or free_run_name(runs_full))
    # B1, B2
    if not os.path.isfile(os.path.join(round_dir, "build.ok")):
        d.blockers.append(f"B1 build: {round_dir}/build.ok missing (cluster/build.sh did not pass; build_w2c_<job>.log)")
    gpu_path = os.path.join(round_dir, "gpu_check.json")
    try:
        with open(gpu_path) as f:
            gpu = json.load(f)
    except (OSError, json.JSONDecodeError):
        gpu = None
    if gpu is None:
        d.blockers.append(f"B2 gpu: {gpu_path} missing (the GPU check did not run or did not finish)")
    elif gpu.get("overall") != "PASS":
        failed = [c["check"] for c in gpu.get("checks", []) if c.get("status") == "FAIL"]
        d.blockers.append(f"B2 gpu: GPU check {gpu.get('overall')}: failed {', '.join(failed) or '?'} "
                          f"({round_dir}/gpu_check.tsv)")
    # B3
    for config, width in SMOKES:
        ok, why = smoke_status(round_dir, config, width)
        (d.notes if ok else d.blockers).append(("smoke ok: " if ok else "B3 smoke: ") + why)
    # B4, B5
    rows = {}
    for row in ROWS:
        path = os.path.join(round_dir, "rows", row)
        rows[row] = row_stats(path) if os.path.isdir(path) else None
    incomplete = []
    for row, r in rows.items():
        if r is None:
            incomplete.append(f"{row} no games")
        elif r["games"] < MIN_ROW_GAMES:
            incomplete.append(f"{row} {r['games']} games (< {MIN_ROW_GAMES})")
        elif r["errors"] > MAX_ERROR_SHARE * (r["games"] + r["errors"]):
            incomplete.append(f"{row} errors {r['errors']} in {r['games'] + r['errors']} games (> 5%)")
    if incomplete:
        d.blockers.append("B5 rows: " + "; ".join(incomplete))
    queue = queued_round_jobs() if queue is None else queue
    if queue:
        d.blockers.append(f"B5 rows: round still running ({len(queue)} w2c_* jobs queued: {', '.join(queue[:6])}"
                          f"{' ...' if len(queue) > 6 else ''})")
    same = rows.get("c_same_120")
    if same and not SAME_RANGE[0] <= same["score"] <= SAME_RANGE[1]:
        d.blockers.append(f"B4 control: c_same_120 {same['score']:.3f} over {same['games']} games is outside "
                          f"[{SAME_RANGE[0]}, {SAME_RANGE[1]}]: the harness is not fair, no row can be read")
    elif same:
        d.notes.append(f"control ok: c_same_120 {same['score']:.3f} over {same['games']} games")
    # config and width
    d.config, why = choose_config(rows)
    d.notes += ["config: " + w for w in why]
    d.wide, why = choose_width(os.path.join(ab_dir, "results_ab.tsv"))
    d.notes += ["width: " + w for w in why]
    measured_as = BENCHED_AS[d.config]
    if measured_as != d.config:
        d.notes.append(f"{d.config} was not benched or smoke-trained in this round: its rate and smoke are {measured_as}'s")
    # B6, B7
    width = "wide" if d.wide else "narrow"
    d.rate, why = bench_rate(round_dir, measured_as, width)
    if d.rate is None:
        d.blockers.append(f"B6 rate: {why}")
    else:
        d.notes.append(f"rate: R = {d.rate:.0f} games per GPU-hour ({why})")
        d.gpu_hours = estimate_gpu_hours(d.rate)
        d.chain = chain_jobs(d.gpu_hours)
        if d.gpu_hours > BUDGET_GPU_H:
            fit = int((BUDGET_GPU_H / FULL_ITERS - ITER_OVERHEAD_H) * d.rate - VAL_GAMES) // 1000 * 1000
            d.blockers.append(f"B7 budget: the full run would take about {d.gpu_hours:,.0f} GPU-hours (> "
                              f"{BUDGET_GPU_H:,}); GAMES_PER_ITER about {max(fit, 0):,} fits")
    if d.gpu_hours is not None:
        d.command = (f"RUN={d.run} CONFIG={d.config} WIDE={d.wide} CHAIN={d.chain} "
                     f"GAMES_PER_ITER={FULL_GAMES_PER_ITER} bash cluster/launch_full_run.sh")
    d.go = not d.blockers
    return d


def report(d: Decision) -> str:
    out = ["GO" if d.go else "NO-GO", ""]
    if d.blockers:
        out += ["Blockers:"] + [f"  - {b}" for b in d.blockers] + [""]
    out += ["Readings:"] + [f"  - {n}" for n in d.notes] + [""]
    out.append(f"Config: {d.config}  WAVE2_EXTRA=\"{CONFIGS[d.config][1]}\"  WIDE={d.wide} "
               f"({'wide' if d.wide else 'narrow'})")
    if d.gpu_hours is not None:
        out.append(f"Estimate: {FULL_ITERS} iterations x {FULL_GAMES_PER_ITER:,} games at R = {d.rate:.0f}: about "
                   f"{d.gpu_hours:,.0f} GPU-hours (plan {PLAN_GPU_H:,} at R = 300, budget {BUDGET_GPU_H:,}); "
                   f"CHAIN={d.chain} jobs of {JOB_HOURS} h, one GPU at a time: about "
                   f"{d.gpu_hours / 24:,.0f} days of wall clock")
    if d.command:
        out += ["", ("Launch (on PACE, from the repo dir):" if d.go else "Would launch, once the blockers are cleared:"),
                f"  {d.command}"]
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--round", default=None, help="round name under runs/wave2c (default: runs/wave2c/LATEST)")
    ap.add_argument("--ab", default=os.path.join("runs", "wave2", "ab2"), help="the A/B run folder")
    ap.add_argument("--run", default=None, help="the full run's RUN name (default: the first free fullN)")
    args = ap.parse_args(argv)
    os.chdir(REPO)
    name = args.round
    if name is None:
        try:
            with open(os.path.join("runs", "wave2c", "LATEST")) as f:
                name = f.read().strip()
        except OSError:
            print("NO-GO\n\nno round: runs/wave2c/LATEST missing (bash cluster/launch_wave2c.sh first)")
            return 1
    d = decide(os.path.join("runs", "wave2c", name), args.ab, args.run)
    print(f"round runs/wave2c/{name}   A/B {args.ab}\n")
    print(report(d))
    return 0 if d.go else 1


if __name__ == "__main__":
    sys.exit(main())
