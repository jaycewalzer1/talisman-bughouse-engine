#!/bin/bash
# The wave 2c readiness round at a glance, from the repo root on PACE:
#   bash cluster/status_wave2c.sh            # the latest round (runs/wave2c/LATEST)
#   ROUND=r2 bash cluster/status_wave2c.sh
# queue; the round's build and GPU check; each row's contender score so far with its 95% range (status_wave2.sh's
# scorer, arena.pool over rows/<row>/results.jsonl; "final" once summary.json exists); the long test's rows, each merged
# over its parts (cluster/launch_long_test.sh; decide_full_run.merged_row_stats), with each part's games; the bench's
# self-play rate R; the smoke lineages' iteration rows. Then the previous round: runs/wave2 rows, runs/wave2/ab2/results_ab.tsv and the
# ab2 arms' log tails. The verdict is cluster/decide_full_run.py's; read docs/wave2_tests.md before the numbers.
set -uo pipefail
cd "$(git rev-parse --show-toplevel)"
ROUND="${ROUND:-$(cat runs/wave2c/LATEST 2>/dev/null)}"
OUT="runs/wave2c/$ROUND"
AB="${AB:-runs/wave2/ab2}"

squeue -u "$USER" -o "%.9i %.20j %.3t %.10M %.24E %R"
echo
if [ -z "$ROUND" ] || [ ! -d "$OUT" ]; then
  echo "no wave 2c round yet (runs/wave2c/LATEST): bash cluster/launch_wave2c.sh"
else
  echo "== round $OUT    jobs: $( { tr '\t\n' ' ;' < "$OUT/jobs.tsv"; } 2>/dev/null)"
  echo "build: $([ -f "$OUT/build.ok" ] && echo "ok ($(cat "$OUT/build.ok"))" || echo "not passed (yet): build_w2c_<job>.log")"
  if [ -f "$OUT/gpu_check.tsv" ]; then
    echo "gpu check: $(grep -o '"overall": "[A-Z]*"' "$OUT/gpu_check.json" 2>/dev/null || echo running)"
    column -t -s $'\t' "$OUT/gpu_check.tsv" | sed 's/^/  /'
  else
    echo "gpu check: not started"
  fi
  [ -f "$OUT/watchdog.log" ] && { echo "watchdog:"; tail -n 3 "$OUT/watchdog.log" | sed 's/^/  /'; }
  echo
fi

.venv/bin/python - "$OUT" "$AB" <<'EOF'
import csv, glob, json, os, sys
sys.path.insert(0, "cluster")
from arena.pool import results_by_opponent
from decide_full_run import merged_row_stats
from wave2c_configs import BENCHED, LONG_ROWS

def rows_table(parent, title):
    print(f"== {title}")
    print(f"{'row':<20} {'games':>6} {'score':>7} {'95% CI':>17} {'flag':>6}  state")
    any_row = False
    for d in sorted(glob.glob(os.path.join(parent, "*/"))):
        f = os.path.join(d, "results.jsonl")
        if not os.path.exists(f):
            continue
        rows = []
        for line in open(f):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
        if not rows:
            continue
        errors = sum("error" in r for r in rows)
        res = results_by_opponent(rows, n_boot=200, seed=0)
        for opp, r in res["by_opponent"].items():
            any_row = True
            state = "final" if os.path.exists(os.path.join(d, "summary.json")) else "running"
            err = f", {errors} errors" if errors else ""
            print(f"{os.path.basename(d.rstrip('/')):<20} {r['games']:>6} {r['score']:>7.3f} "
                  f"[{r['ci95'][0]:.3f}, {r['ci95'][1]:.3f}] {r.get('flag_rate', float('nan')):>6.2f}  "
                  f"{state} vs {opp}{err}")
    if not any_row:
        print("(no games yet)")
    print()

out, ab = sys.argv[1], sys.argv[2]
if os.path.isdir(out):
    rows_table(os.path.join(out, "rows"), f"{out}/rows (decide reads c_same_120 first, then c_all / c_all_nod1 / "
                                          f"the best single switch / h_best_120 (hybrid) against c_nodur_120)")
    print(f"== {out}/long: the long test, parts merged (decide reads a merged row of 250+ games instead of its round row)")
    print(f"{'row':<14} {'games':>6} {'score':>7} {'95% CI':>17} {'parts':>6}  stands in for / parts' games")
    any_long = False
    for lrow, stands_for in LONG_ROWS.items():
        d = os.path.join(out, "long", lrow)
        if not os.path.isdir(d):
            continue
        any_long = True
        r = merged_row_stats(d, n_boot=200)
        per = []
        for p in sorted(glob.glob(os.path.join(d, "part*")), key=lambda p: int(p.rsplit("part", 1)[1] or 0)):
            f = os.path.join(p, "results.jsonl")
            n = sum(1 for line in open(f) if '"error"' not in line) if os.path.exists(f) else 0
            per.append(f"{os.path.basename(p)[4:]}:{n}{'' if os.path.exists(os.path.join(p, 'summary.json')) else '+'}")
        if r is None:
            print(f"{lrow:<14} {0:>6} {'':>7} {'':>17} {len(per):>6}  {stands_for}; no games yet")
            continue
        err = f", {r['errors']} errors" if r["errors"] else ""
        print(f"{lrow:<14} {r['games']:>6} {r['score']:>7.3f} [{r['ci95'][0]:.3f}, {r['ci95'][1]:.3f}] {r['parts']:>6}  "
              f"{stands_for}; {' '.join(per)}{err}")
    if not any_long:
        print("(no long test: bash cluster/launch_long_test.sh)")
    else:
        print("(part:games, + = still running or cut off by the time limit)")
    print()
    print("== self-play rate R (games per GPU-hour, cluster/wave2_selfplay_bench.py)")
    benches = sorted(glob.glob(os.path.join(out, "bench", "*", "bench.json")))
    for path in benches:
        b = json.load(open(path))
        print(f"{os.path.basename(os.path.dirname(path)):<22} games {b.get('games')}/{b.get('games_asked')}  "
              f"exit {b.get('exit')}  R end-to-end {b.get('games_per_gpu_hour')}  "
              f"steady {b.get('games_per_gpu_hour_steady')}  wall {b.get('wall_s')} s")
    if not benches:
        print("(no bench finished yet)")
    print()
    print("== smoke lineages (one iteration: self-play -> train -> gate)")
    for config in BENCHED:
        for width in ("narrow", "wide"):
            arm = os.path.join(out, f"smoke_{config}", width)
            tsv = os.path.join(arm, "results.tsv")
            done = os.path.isfile(os.path.join(arm, "runs", "rl-it1", "stages", "iteration.done"))
            if os.path.isfile(tsv):
                last = list(csv.DictReader(open(tsv), delimiter="\t"))
                last = last[-1] if last else {}
                print(f"smoke_{config:<6} {width:<7} {'finished' if done else 'running'}  games {last.get('games', '')}  "
                      f"vs_generator {last.get('vs_generator', '')}  vs_it13 {last.get('vs_it13', '')}  "
                      f"flag_games {last.get('flag_games', '')}")
            else:
                stages = sorted(os.path.basename(p) for p in glob.glob(os.path.join(arm, "runs", "rl-it1", "stages", "*.done")))
                print(f"smoke_{config:<6} {width:<7} {'stages ' + ' '.join(stages) if stages else 'not started'}")
    print()

rows_table("runs/wave2", "previous round: runs/wave2 rows (it13 at 120 s; ref_120 0.227, nodur 0.436 there)")
tsv = os.path.join(ab, "results_ab.tsv")
print(f"== {tsv}")
print(open(tsv).read().rstrip() if os.path.isfile(tsv) else "(not there)")
EOF
echo
for log in $(ls -t wave2_ab2_w*_*.log 2>/dev/null | head -2); do echo "== $log"; tail -n 4 "$log"; done
