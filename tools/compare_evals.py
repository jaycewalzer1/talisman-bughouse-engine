"""One row per evaluation, to compare fixed policies scored by cluster/eval.sbatch.

    python tools/compare_evals.py runs/eval/*

score is the contender's share of points against the one opponent, with the 95% range from the run's own
bootstrap over opening pairs. "vs first" is the score minus the first row's, with a rough 95% half-width for
that difference; a difference smaller than its half-width is not a result.
"""

import json
import os
import sys


def main() -> None:
    print(f"{'eval':<22} {'games':>6} {'score':>6} {'95% range':>16} {'vs first':>9} {'±':>6} "
          f"{'waits/100':>9} {'lost on time':>12} {'errors':>6}")
    first = None
    for run in sys.argv[1:]:
        path = os.path.join(run, "summary.json")
        if not os.path.exists(path):
            print(f"{os.path.basename(run):<22} no result yet")
            continue
        with open(path) as f:
            report = json.load(f)
        games = [g for g in report["games"] if "error" not in g]
        (name, r), = list(report["results"]["by_opponent"].items())[:1]
        mine = [g["teams"][g["contender_team"]] for g in games]
        decisions = sum(t["decisions"] for t in mine) or 1
        losses = [g for g in games if g["result"] in ("T1", "T2") and g["result"] != g["contender_team"]]
        on_time = sum(bool(g["flag"]) for g in losses)
        n, score = r["games"], r["score"]
        if first is None:
            first = (score, n)
            versus, spread = f"{'-':>9}", f"{'-':>6}"
        else:
            versus = f"{score - first[0]:>+9.3f}"
            spread = f"{1.96 * (0.25 / n + 0.25 / first[1]) ** 0.5:>6.3f}"
        ci = f"[{r['ci95'][0]:.3f}, {r['ci95'][1]:.3f}]"
        print(f"{os.path.basename(run):<22} {n:>6} {score:>6.3f} {ci:>16} {versus} {spread} "
              f"{100 * sum(t['stalls'] for t in mine) / decisions:>9.2f} {f'{on_time}/{len(losses)}':>12} "
              f"{report['errors']:>6}")


if __name__ == "__main__":
    main()
