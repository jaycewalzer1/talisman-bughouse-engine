#!/usr/bin/env python3
"""Leaderboard of one tuning round (cluster/tune_round.py, cluster/tune_match.sh): every candidate against
upstream-it13, its parts merged, and against the round's baseline on the games both played. Standard library only.

    python3 cluster/tune_status.py runs/tune/r1
    python3 cluster/tune_status.py runs/tune/r1 --elo0 0 --elo1 20 --delo0 0 --delo1 15 --json
    python3 cluster/tune_status.py runs/tune/r1 --cancel        # print scancel lines for decided candidates
    python3 cluster/tune_status.py runs/tune/r1 --engine clock_k_220   # its ENGINES entry for the final match

A round folder holds one folder per candidate (<cand>/part<k>/, each an arena row of cluster/eval.sbatch with its
results.jsonl) and round.json (written by tune_round.py: mode, baseline, each candidate's parameters). Every candidate
of a round plays the same schedule (part k: seed SEED_BASE + k, pair p: the same opening and colours), so a game pair
(part, pair_index) of one candidate meets the same opening as that pair of every other.

Per candidate, against upstream-it13:
  games, complete pairs, W-D-L, pentanomial [0, 0.5, 1, 1.5, 2] pair points, score with a 95% range (normal over game
  pairs), logistic Elo with the range's ends converted, and two GSPRTs of H0: Elo = elo0 against H1: Elo = elo1 in
  logistic Elo (fishtest's LLR_logistic, as rl_loop.py's sprt_llr): "trinomial" (per game W/D/L) and "pentanomial"
  (per colour-swapped pair; the honest one when a pair's two games are correlated).
Against the baseline (paired over the game pairs both have finished):
  shared pairs, difference of score and of Elo with a 95% range, the correlation of the two candidates' pair scores
  (how much pairing on openings helps), and a GSPRT on the paired differences of H0: dElo = delo0 against H1: dElo =
  delo1 (the Elo difference converted to a score difference around the baseline's own score on those pairs).

Decisions: H1 when LLR >= ln((1 - beta) / alpha), H0 when LLR <= ln(beta / (1 - alpha)), else "continue"; none before
MIN_GAMES games (the variance of a handful of games is meaningless). The tests look at the data many times, which is
what a GSPRT allows; the score ranges are fixed-sample ranges and are too narrow if read only when they look good.
In an SPSA round (mode "spsa") no candidate is stopped early: the gradient needs every candidate's games.
Each candidate's test against the baseline is at its own alpha: in a round of m candidates a false H1 somewhere is
likely (about 1 - 0.95^m when none is better). A pick from such a round is a lead, not evidence; the evidence is the
combine round's own paired test against the baseline (cluster/tune_round.py combine, more parts).

Part state: a part is over when it wrote summary.json or its slurm job (from <cand>/jobs.tsv, asked through squeue)
is gone; a part that hits its 8 h limit writes no summary and shows as "ended without summary (timed out?)". Off the
cluster (no squeue) such a part shows as "no summary (running or timed out ...)". --cancel prints one scancel line
per decided candidate with its job ids from jobs.tsv (or one --name line per part when jobs.tsv has none).
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import subprocess
import sys

MIN_GAMES = 100
Z95 = 1.959963984540054
BUCKETS = (0.0, 0.5, 1.0, 1.5, 2.0)


# ----------------------------------------------------------------------------------------------- Elo and GSPRT
def expected_score(elo: float) -> float:
    return 1.0 / (1.0 + 10.0 ** (-elo / 400.0))


def score_elo(score: float) -> float:
    """Logistic Elo of a score; +-inf outside (0, 1) is clamped to +-1000 for printing."""
    if score <= 0.0:
        return -1000.0
    if score >= 1.0:
        return 1000.0
    return 400.0 * math.log10(score / (1.0 - score))


def sprt_bounds(alpha: float = 0.05, beta: float = 0.05) -> tuple[float, float]:
    return math.log(beta / (1.0 - alpha)), math.log((1.0 - beta) / alpha)


def llr_normal(n: float, mean: float, var: float, s0: float, s1: float) -> float:
    """fishtest's LLR_logistic: the GSPRT log-likelihood ratio of mean s1 against s0 for n samples with this mean and
    variance, n (s1 - s0) (2 mean - s0 - s1) / (2 var)."""
    if n <= 0 or var <= 0:
        return 0.0
    return n * (s1 - s0) * (2.0 * mean - s0 - s1) / (2.0 * var)


def llr_trinomial(wins: int, draws: int, losses: int, elo0: float, elo1: float) -> float:
    """Per-game GSPRT (rl_loop.py sprt_llr): a zero count is taken as 1e-3, as fishtest regularises."""
    if wins + draws + losses == 0:
        return 0.0
    c = [max(x, 1e-3) for x in (wins, draws, losses)]
    n = sum(c)
    w, d = c[0] / n, c[1] / n
    mean = w + d / 2.0
    var = w + d / 4.0 - mean ** 2
    return llr_normal(n, mean, var, expected_score(elo0), expected_score(elo1))


def pent_moments(counts: list[int]) -> tuple[int, float, float]:
    """(pairs, mean per-game score, variance of a pair's mean score), each bucket regularised by 1e-3."""
    n = sum(counts)
    reg = [c + 1e-3 for c in counts]
    tot = sum(reg)
    xs = [b / 2.0 for b in BUCKETS]
    mu = sum(x * c for x, c in zip(xs, reg)) / tot
    var = sum(c * (x - mu) ** 2 for x, c in zip(xs, reg)) / tot
    return n, mu, var


def llr_pentanomial(counts: list[int], elo0: float, elo1: float) -> float:
    n, mu, var = pent_moments(counts)
    return llr_normal(n, mu, var, expected_score(elo0), expected_score(elo1)) if n else 0.0


def decide(llr: float, games: int, alpha: float, beta: float) -> str:
    lo, hi = sprt_bounds(alpha, beta)
    if games < MIN_GAMES:
        return "continue"
    return "H1" if llr >= hi else "H0" if llr <= lo else "continue"


# ----------------------------------------------------------------------------------------------- reading rows
def read_games(part_dir: str) -> tuple[list[dict], bool]:
    """(game summaries, finished?) of one arena row: results.jsonl (written as games finish), else summary.json's
    games. A line still being written is skipped."""
    games: list[dict] = []
    path = os.path.join(part_dir, "results.jsonl")
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    games.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    summary = os.path.join(part_dir, "summary.json")
    finished = os.path.exists(summary)
    if not games and finished:
        with open(summary) as f:
            games = list(json.load(f).get("games") or [])
    return games, finished


def game_points(g: dict) -> float:
    res, ct = g.get("result"), g["contender_team"]
    return 1.0 if res == ct else 0.5 if res == "draw" else 0.0


def pair_index(g: dict) -> int:
    if g.get("pair_index") is not None:
        return int(g["pair_index"])
    digits = "".join(ch for ch in str(g.get("game_id") or g.get("file") or "") if ch.isdigit())
    if not digits:
        raise ValueError(f"cannot tell which pair game {g.get('game_id')!r} belongs to")
    return (int(digits) - 1) // 2


def read_jobs(cand_dir: str) -> dict:
    """{"part<k>": slurm job id} from the candidate's jobs.tsv (cluster/tune_match.sh writes one line per part)."""
    jobs = {}
    path = os.path.join(cand_dir, "jobs.tsv")
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                cols = line.rstrip("\n").split("\t")
                if len(cols) >= 4 and cols[1].startswith("part") and cols[3].strip():
                    jobs[cols[1]] = cols[3].strip().split(";")[0]
    return jobs


def queued_jobs(ids) -> set | None:
    """The ids among `ids` still pending or running in slurm (squeue), or None when squeue cannot be asked (not on
    the cluster): the caller then cannot tell a running part from one that hit its time limit."""
    ids = sorted({str(i) for i in ids if str(i).isdigit()})
    if not ids:
        return set()
    try:
        r = subprocess.run(["squeue", "-h", "-o", "%i", "-j", ",".join(ids)], capture_output=True, text=True,
                           timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    if r.returncode != 0 and "Invalid job id" not in r.stderr:
        return None
    return {x.strip().split("_")[0] for x in r.stdout.split()} & set(ids)


def load_candidate(cand_dir: str, queued=None) -> dict:
    """{"pairs": {(part, pair): (points of game 1 + game 2)}, "games", "errors", "wdl", "parts", "finished", "state",
    "jobs"} of one candidate; only pairs with both games finished and neither in error count as pairs. A part is over
    when it wrote summary.json or its slurm job (jobs.tsv) is no longer queued: a part that hits its time limit
    writes no summary. "state": "finished" (all parts wrote a summary), "ended" (all over, some without a summary:
    timed out or failed), "running", or "unknown" (no summary everywhere and squeue not available).
    queued: the set of job ids still in slurm, or None (ask squeue here)."""
    parts = sorted((d for d in glob.glob(os.path.join(cand_dir, "part*")) if os.path.isdir(d)),
                   key=lambda d: int(os.path.basename(d)[4:]) if os.path.basename(d)[4:].isdigit() else 0)
    by_pair: dict = {}
    errors = 0
    wdl = [0, 0, 0]
    jobs = read_jobs(cand_dir)
    finished = bool(parts)
    no_summary = []
    for d in parts:
        part = os.path.basename(d)
        games, done = read_games(d)
        finished = finished and done
        if not done:
            no_summary.append(part)
        for g in games:
            if "error" in g:
                errors += 1
            else:
                p = game_points(g)
                wdl[0 if p == 1.0 else 1 if p == 0.5 else 2] += 1
            by_pair.setdefault((part, pair_index(g)), []).append(g)
    pairs = {}
    for key, gs in by_pair.items():
        if len(gs) == 2 and not any("error" in g for g in gs):
            pairs[key] = sum(game_points(g) for g in gs)
    if finished or not parts:
        state = "finished" if finished else "running"
    else:
        if queued is None:
            queued = queued_jobs(jobs.get(p) for p in no_summary)
        if queued is None or any(p not in jobs for p in no_summary):
            state = "unknown"
        else:
            state = "running" if any(jobs[p] in queued for p in no_summary) else "ended"
    return {"pairs": pairs, "games": sum(wdl), "errors": errors, "wdl": wdl, "parts": len(parts),
            "finished": finished, "state": state, "jobs": jobs}


# ----------------------------------------------------------------------------------------------- statistics
def mean_var(xs: list[float]) -> tuple[float, float]:
    if not xs:
        return float("nan"), float("nan")
    m = sum(xs) / len(xs)
    v = sum((x - m) ** 2 for x in xs) / (len(xs) - 1) if len(xs) > 1 else 0.0
    return m, v


def vs_opponent(c: dict, elo0: float, elo1: float, alpha: float, beta: float) -> dict:
    counts = [0] * 5
    for pts in c["pairs"].values():
        counts[BUCKETS.index(pts)] += 1
    pair_scores = [pts / 2.0 for pts in c["pairs"].values()]
    w, d, l = c["wdl"]
    games = w + d + l
    score = (w + d / 2.0) / games if games else float("nan")
    m, v = mean_var(pair_scores)
    half = Z95 * math.sqrt(v / len(pair_scores)) if len(pair_scores) > 1 else float("nan")
    lo, hi = (max(0.0, m - half), min(1.0, m + half)) if pair_scores else (float("nan"), float("nan"))
    llr3 = llr_trinomial(w, d, l, elo0, elo1)
    llr5 = llr_pentanomial(counts, elo0, elo1)
    return {"games": games, "pairs": len(pair_scores), "wdl": [w, d, l], "pentanomial": counts, "score": score,
            "ci95": [lo, hi], "elo": score_elo(score) if games else float("nan"),
            "elo_ci95": [score_elo(lo), score_elo(hi)] if pair_scores else [float("nan")] * 2,
            "llr_trinomial": llr3, "llr_pentanomial": llr5,
            "decision": decide(llr5, games, alpha, beta), "decision_trinomial": decide(llr3, games, alpha, beta)}


def vs_baseline(c: dict, base: dict, delo0: float, delo1: float, alpha: float, beta: float) -> dict:
    """Paired comparison on the game pairs both candidates finished."""
    shared = sorted(set(c["pairs"]) & set(base["pairs"]))
    n = len(shared)
    out = {"shared_pairs": n, "shared_games": 2 * n}
    if n < 2:
        return {**out, "dscore": float("nan"), "dscore_se": float("nan"), "dscore_ci95": [float("nan")] * 2,
                "delo": float("nan"),
                "delo_ci95": [float("nan")] * 2, "rho": float("nan"), "llr": 0.0, "decision": "continue"}
    a = [c["pairs"][k] / 2.0 for k in shared]
    b = [base["pairs"][k] / 2.0 for k in shared]
    diffs = [x - y for x, y in zip(a, b)]
    md, vd = mean_var(diffs)
    ma, va = mean_var(a)
    mb, vb = mean_var(b)
    cov = sum((x - ma) * (y - mb) for x, y in zip(a, b)) / (n - 1)
    rho = cov / math.sqrt(va * vb) if va > 0 and vb > 0 else float("nan")
    half = Z95 * math.sqrt(vd / n)
    eb = score_elo(min(max(mb, 1e-3), 1 - 1e-3))
    to_elo = lambda s: score_elo(min(max(s, 1e-3), 1 - 1e-3)) - eb
    d0 = expected_score(eb + delo0) - expected_score(eb)
    d1 = expected_score(eb + delo1) - expected_score(eb)
    llr = llr_normal(n, md, max(vd, 1e-9), d0, d1)
    clip = lambda x: min(max(x, 0.0), 1.0)  # the candidate's score on the shared pairs stays a score
    return {**out, "dscore": md, "dscore_se": math.sqrt(vd / n), "dscore_ci95": [md - half, md + half], "delo": to_elo(mb + md),
            "delo_ci95": [to_elo(clip(mb + md - half)), to_elo(clip(mb + md + half))], "rho": rho, "llr": llr,
            "decision": decide(llr, 2 * n, alpha, beta), "base_score_shared": mb}


# ----------------------------------------------------------------------------------------------- the round
def load_round(round_dir: str) -> dict:
    path = os.path.join(round_dir, "round.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    names = sorted(os.path.basename(os.path.dirname(p)) for p in glob.glob(os.path.join(round_dir, "*", "part*")))
    return {"name": os.path.basename(round_dir.rstrip("/")), "mode": "unknown",
            "baseline": "base" if "base" in names else None,
            "candidates": [{"name": n, "params": {}} for n in dict.fromkeys(names)]}


def final_match_engine(round_dir: str, cand: str) -> dict:
    """The candidate as an entry of cluster/launch_final_match.sh's ENGINES: kind wave2, its contender options without
    ms_per_node (the final match measures both arms, MEASURE=1), the it13 model (null)."""
    with open(os.path.join(round_dir, "candidates", f"{cand}.json")) as f:
        c = json.load(f)
    opts = {k: v for k, v in c["contender_kw"].items() if k != "ms_per_node"}
    return {"name": f"tuned-{c.get('round', '')}-{cand}", "kind": "wave2", "options": opts, "model": None}


def status(round_dir: str, elo0: float = 0.0, elo1: float = 20.0, delo0: float = 0.0, delo1: float = 30.0,
           alpha: float = 0.05, beta: float = 0.05, baseline: str | None = None) -> dict:
    rnd = load_round(round_dir)
    base_name = baseline or rnd.get("baseline")
    dirs = {c["name"]: os.path.join(round_dir, c["name"]) for c in rnd["candidates"]}
    all_ids = [i for d in dirs.values() for i in read_jobs(d).values()]
    queued = queued_jobs(all_ids) if all_ids else set()
    loaded = {n: load_candidate(d, queued) for n, d in dirs.items()}
    base = loaded.get(base_name) if base_name else None
    rows = []
    for c in rnd["candidates"]:
        data = loaded[c["name"]]
        row = {"name": c["name"], "params": c.get("changed", c.get("params", {})), "parts": data["parts"],
               "errors": data["errors"], "finished": data["finished"],
               "state": data["state"], "jobs": data["jobs"],
               "vs_opponent": vs_opponent(data, elo0, elo1, alpha, beta)}
        if base is not None and c["name"] != base_name:
            row["vs_baseline"] = vs_baseline(data, base, delo0, delo1, alpha, beta)
        rows.append(row)
    return {"round": rnd.get("name"), "mode": rnd.get("mode"), "baseline": base_name, "dir": round_dir,
            "elo": [elo0, elo1], "delo": [delo0, delo1], "alpha": alpha, "beta": beta,
            "bounds": list(sprt_bounds(alpha, beta)), "candidates": rows}


def stoppable(st: dict) -> list[str]:
    """Candidates whose jobs may be cancelled now: decided against the baseline (either way), in a round that is
    not SPSA, never the baseline itself (every other candidate is read against its games)."""
    if st.get("mode") == "spsa":
        return []
    return [r["name"] for r in st["candidates"]
            if r["name"] != st["baseline"] and r.get("state", "running") not in ("finished", "ended")
            and r.get("vs_baseline", {}).get("decision") in ("H0", "H1")]


def cancel_line(st: dict, name: str) -> str:
    """One scancel line for a candidate: by its job ids from jobs.tsv (scancel takes several ids), else one
    --name line per part (scancel's --name takes a single job name)."""
    row = next(r for r in st["candidates"] if r["name"] == name)
    ids = [j for _, j in sorted(row.get("jobs", {}).items(), key=lambda kv: int(kv[0][4:] or 0))
           if str(j).isdigit()]
    if ids and len(ids) >= row["parts"]:
        return "scancel " + " ".join(ids) + f"   # {name}"
    return "\n".join(f"scancel -u \"$USER\" --name=tune_{st['round']}_{name}_p{k}" for k in range(1, 1 + row["parts"]))


def _f(x: float, fmt: str) -> str:
    return "nan" if x != x else format(x, fmt)


def format_table(st: dict) -> str:
    lines = [f"round {st['round']} ({st['mode']}) in {st['dir']}; baseline {st['baseline']}; "
             f"GSPRT vs it13 Elo [{st['elo'][0]:g}, {st['elo'][1]:g}], vs baseline dElo [{st['delo'][0]:g}, "
             f"{st['delo'][1]:g}], LLR bounds [{st['bounds'][0]:.2f}, {st['bounds'][1]:.2f}]",
             f"{'candidate':28s} {'games':>5s}  {'score [95%]':22s} {'Elo':>5s} {'LLR5':>6s} {'LLR3':>6s} dec  | "
             f"{'pairs':>5s} {'dElo [95%]':20s} {'rho':>5s} {'LLR':>6s} dec"]
    def key(r):
        vb = r.get("vs_baseline")
        d = vb["delo"] if vb and vb["delo"] == vb["delo"] else (0.0 if r["name"] == st["baseline"] else -1e9)
        return -d
    for r in sorted(st["candidates"], key=key):
        o = r["vs_opponent"]
        lo, hi = o["ci95"]
        s = (f"{r['name'][:28]:28s} {o['games']:5d}  {_f(o['score'], '.3f')} [{_f(lo, '.3f')}, {_f(hi, '.3f')}]  "
             f"{_f(o['elo'], '+5.0f')} {o['llr_pentanomial']:+6.2f} {o['llr_trinomial']:+6.2f} {o['decision']:4s} | ")
        vb = r.get("vs_baseline")
        if vb:
            dl, dh = vb["delo_ci95"]
            s += (f"{vb['shared_pairs']:5d} {_f(vb['delo'], '+5.1f')} [{_f(dl, '+5.1f')}, {_f(dh, '+5.1f')}]  "
                  f"{_f(vb['rho'], '+5.2f')} {vb['llr']:+6.2f} {vb['decision']}")
        else:
            s += "  (baseline)" if r["name"] == st["baseline"] else ""
        state = r.get("state", "finished" if r["finished"] else "running")
        if state == "ended":
            s += "  ended without summary (timed out?)"
        elif state == "unknown":
            s += "  no summary (running or timed out: squeue unavailable)"
        elif state == "running":
            s += "  running"
        if r["errors"]:
            s += f"  {r['errors']} errors"
        if r["params"]:
            s += "  " + " ".join(f"{k}={v}" for k, v in r["params"].items())
        lines.append(s)
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("round_dir")
    ap.add_argument("--elo0", type=float, default=0.0, help="vs it13: H0 logistic Elo (default 0)")
    ap.add_argument("--elo1", type=float, default=20.0, help="vs it13: H1 logistic Elo (default 20)")
    ap.add_argument("--delo0", type=float, default=0.0, help="vs baseline: H0 Elo difference (default 0)")
    ap.add_argument("--delo1", type=float, default=30.0,
                    help="vs baseline: H1 Elo difference (default 30: ~1400 games a candidate; 15 needs ~5700)")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--beta", type=float, default=0.05)
    ap.add_argument("--baseline", default=None, help="baseline candidate (default: round.json's)")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--engine", metavar="CAND", help="print CAND's entry for the final match's ENGINES and exit")
    ap.add_argument("--cancel", action="store_true",
                    help="also print scancel lines for candidates decided against the baseline (not run)")
    args = ap.parse_args(argv)
    if args.engine:
        print(json.dumps(final_match_engine(args.round_dir, args.engine)))
        return 0
    st = status(args.round_dir, args.elo0, args.elo1, args.delo0, args.delo1, args.alpha, args.beta, args.baseline)
    if args.json:
        print(json.dumps(st, indent=1))
    else:
        print(format_table(st))
    if args.cancel:
        names = stoppable(st)
        if st.get("mode") == "spsa":
            print("# SPSA round: no candidate is stopped early")
        for n in names:
            print(cancel_line(st, n))
    return 0


if __name__ == "__main__":
    sys.exit(main())
