#!/bin/bash
# The wave 2c readiness round at a glance, from the repo root on PACE:
#   bash cluster/status_wave2c.sh            # the latest round (runs/wave2c/LATEST)
#   ROUND=r2 bash cluster/status_wave2c.sh
# queue; the round's build and GPU check; each row's contender score so far with its 95% range (status_wave2.sh's
# scorer, arena.pool over rows/<row>/results.jsonl; "final" once summary.json exists); the bench's self-play rate R;
# the smoke lineages' iteration rows. Then the previous round: runs/wave2 rows, runs/wave2/ab2/results_ab.tsv and the
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
  echo "== round $OUT    jobs: $(tr '\t\n' ' ;' < "$OUT/jobs.tsv" 2>/dev/null)"
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
from arena.pool import results_by_opponent

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
                                          f"the best single switch against c_nodur_120)")
    print("== self-play rate R (games per GPU-hour, cluster/wave2_selfplay_bench.py)")
    benches = sorted(glob.glob(os.path.join(out, "bench", "*", "bench.json")))
    for path in benches:
        b = json.load(open(path))
        print(f"{os.path.basename(os.path.dirname(path)):<22} games {b['games']}/{b['games_asked']}  exit {b['exit']}  "
              f"R end-to-end {b['games_per_gpu_hour']}  steady {b['games_per_gpu_hour_steady']}  wall {b['wall_s']} s")
    if not benches:
        print("(no bench finished yet)")
    print()
    print("== smoke lineages (one iteration: self-play -> train -> gate)")
    for config in ("nodur", "cont"):
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
