"""One row per evaluation, to compare fixed policies scored by cluster/eval.sbatch or the iterate.sh gates.

    python tools/compare_evals.py runs/eval/*
    python tools/compare_evals.py runs/custom/main/it00/gate_180 runs/custom/main/it0*/gate_180   # vs the it00 row
    python tools/compare_evals.py --json runs/beat/same_*

score is the contender's share of points against the one opponent, with the 95% range from a cluster bootstrap
over opening pairs (the row's own summary.json range). "vs first" is the score minus the first row's, with its
95% range from a bootstrap too: paired over opening pairs when both rows played the same schedule (same openings
and colours, as the gate rows of one RUN do), else the two rows resampled independently. A difference whose range
includes 0 is not a result. nElo is normalised Elo from the pentanomial pair counts (tools/sprt.py), and the
nodes columns are each side's mean nodes searched per game. A row that is still running (no summary.json yet) is
read from its results.jsonl or game files.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import defaultdict

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from arena.stats import contender_row  # noqa: E402
from tools.sprt import nelo, pentanomial  # noqa: E402


def load(run: str) -> tuple[list[dict], dict | None, bool]:
    """(finished game summaries, summary.json or None, finished?)."""
    path = os.path.join(run, "summary.json")
    if os.path.exists(path):
        with open(path) as f:
            report = json.load(f)
        return [g for g in report["games"] if "error" not in g], report, True
    from tools.sprt import load_row

    try:
        return [g for g in load_row(run) if "error" not in g], None, False
    except FileNotFoundError:
        return [], None, False


def _key(g: dict):
    return (g.get("opening_id"), g.get("contender_team"), g.get("game_id"))


def diff_ci(a: list[dict], b: list[dict], n_boot: int = 2000, seed: int = 0) -> tuple[float, float, float, bool]:
    """(score(b) - score(a), lo, hi, paired). Paired when both rows hold the same games by (opening, side, id):
    resample opening clusters and take the mean difference over the shared games; else independent cluster
    bootstraps of each row."""
    rows_a = {_key(g): contender_row(g) for g in a}
    rows_b = {_key(g): contender_row(g) for g in b}
    shared = sorted(set(rows_a) & set(rows_b), key=str)
    rng = random.Random(seed)
    mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")
    paired = len(shared) >= 0.9 * max(len(rows_a), len(rows_b)) and len(shared) > 1
    if paired:
        clusters: dict = defaultdict(list)
        for k in shared:
            clusters[rows_a[k]["cluster"]].append(rows_b[k]["score"] - rows_a[k]["score"])
        keys = list(clusters)
        est = mean([d for k in keys for d in clusters[k]])
        boots = []
        for _ in range(n_boot):
            sample = [d for _k in keys for d in clusters[keys[rng.randrange(len(keys))]]]
            boots.append(mean(sample))
    else:
        def groups(rows):
            g: dict = defaultdict(list)
            for r in rows.values():
                g[r["cluster"]].append(r["score"])
            return list(g.values())
        ga, gb = groups(rows_a), groups(rows_b)
        if not ga or not gb:
            return float("nan"), float("nan"), float("nan"), False
        est = mean([s for c in gb for s in c]) - mean([s for c in ga for s in c])
        boots = []
        for _ in range(n_boot):
            sa = [s for _k in ga for s in ga[rng.randrange(len(ga))]]
            sb = [s for _k in gb for s in gb[rng.randrange(len(gb))]]
            boots.append(mean(sb) - mean(sa))
    boots.sort()
    return est, boots[int(0.025 * (len(boots) - 1))], boots[int(round(0.975 * (len(boots) - 1)))], paired


def row(run: str, first: list[dict] | None, n_boot: int = 2000) -> dict | None:
    games, report, finished = load(run)
    if not games:
        return None
    rows = [contender_row(g) for g in games]
    n = len(rows)
    score = sum(r["score"] for r in rows) / n
    if report:
        (_name, r), = list(report["results"]["by_opponent"].items())[:1]
        ci = r["ci95"]
    else:
        from arena.stats import cluster_bootstrap_ci, mean_of

        c = cluster_bootstrap_ci(rows, mean_of("score"), n_boot=n_boot)
        ci = [c["lo"], c["hi"]]
    mine = [g["teams"][g["contender_team"]] for g in games]
    decisions = sum(t["decisions"] for t in mine) or 1
    losses = [g for g in games if g["result"] in ("T1", "T2") and g["result"] != g["contender_team"]]
    reasons: dict = defaultdict(int)
    for g in games:
        reasons[g.get("reason")] += 1
    try:
        penta = pentanomial(games)
    except ValueError:
        penta = [0] * 5
    out = {"run": run, "games": n, "finished": finished, "score": score, "ci95": ci, "nelo": nelo(penta),
           "c_nodes_per_game": sum(r["c_nodes"] for r in rows) / n,
           "b_nodes_per_game": sum(r["b_nodes"] for r in rows) / n,
           "waits_per_100": 100 * sum(t["stalls"] for t in mine) / decisions,
           "lost_on_time": [sum(bool(g.get("flag")) for g in losses), len(losses)],
           "reasons": dict(sorted(reasons.items(), key=lambda kv: str(kv[0]))),
           "errors": report["errors"] if report else 0}
    if report and "sprt" in report:
        out["sprt"] = report["sprt"]["decision"]
    if first is not None:
        d, lo, hi, paired = diff_ci(first, games, n_boot=n_boot)
        out.update({"vs_first": d, "vs_first_ci95": [lo, hi], "paired": paired})
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--json", action="store_true", help="one JSON object per row")
    args = ap.parse_args(argv)
    if not args.json:
        print(f"{'eval':<22} {'games':>6} {'score':>6} {'95% range':>16} {'nElo':>6} {'vs first':>9} "
              f"{'95% range':>17} {'nodes/g c':>9} {'nodes/g b':>9} {'waits/100':>9} {'lost on time':>12} "
              f"{'errors':>6}  reasons")
    first = None
    for run in args.runs:
        r = row(run, first[0] if first else None, n_boot=args.boot)
        if r is None:
            print(json.dumps({"run": run, "games": 0}) if args.json else f"{os.path.basename(run):<22} no result yet")
            continue
        if first is None:
            games, _, _ = load(run)
            first = (games,)
        if args.json:
            print(json.dumps(r))
            continue
        name = "/".join(run.rstrip("/").split("/")[-2:])[-22:] + ("" if r["finished"] else "*")
        ci = f"[{r['ci95'][0]:.3f}, {r['ci95'][1]:.3f}]"
        if "vs_first" in r:
            versus = f"{r['vs_first']:>+9.3f}"
            vci = f"[{r['vs_first_ci95'][0]:+.3f}, {r['vs_first_ci95'][1]:+.3f}]" + ("p" if r["paired"] else " ")
        else:
            versus, vci = f"{'-':>9}", "-"
        reasons = " ".join(f"{k}={v}" for k, v in r["reasons"].items())
        print(f"{name:<22} {r['games']:>6} {r['score']:>6.3f} {ci:>16} {max(-999.0, min(999.0, r['nelo'])):>+6.0f} {versus} {vci:>17} "
              f"{r['c_nodes_per_game']:>9.0f} {r['b_nodes_per_game']:>9.0f} {r['waits_per_100']:>9.2f} "
              f"{'%d/%d' % tuple(r['lost_on_time']):>12} {r['errors']:>6}  {reasons}")
    if not args.json:
        print("(* still running; p = paired over the same games as the first row)")


if __name__ == "__main__":
    main()
