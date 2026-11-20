#!/bin/bash
# The wave 2 round at a glance, from the repo root on PACE:
#   bash cluster/status_wave2.sh
# queue, then each test row's contender score so far (arena.pool's own scorer over runs/wave2/<row>/results.jsonl;
# "final" once summary.json exists), then each A/B arm's last log lines. Read docs/wave2_tests.md before the numbers.
set -uo pipefail
cd "$(git rev-parse --show-toplevel)"

squeue -u "$USER" -o "%.9i %.18j %.3t %.10M %R"
echo
.venv/bin/python - <<'EOF'
import glob, json, os
from arena.pool import results_by_opponent
print(f"{'row':<20} {'games':>6} {'score':>7} {'95% CI':>17} {'flag':>6}  state")
for d in sorted(glob.glob("runs/wave2/*/")):
    f = os.path.join(d, "results.jsonl")
    if not os.path.exists(f):
        continue
    rows = [json.loads(l) for l in open(f) if l.strip()]
    if not rows:
        continue
    res = results_by_opponent(rows, n_boot=200, seed=0)
    for opp, r in res["by_opponent"].items():
        state = "final" if os.path.exists(os.path.join(d, "summary.json")) else "running"
        print(f"{os.path.basename(d.rstrip('/')):<20} {r['games']:>6} {r['score']:>7.3f} "
              f"[{r['ci95'][0]:.3f}, {r['ci95'][1]:.3f}] {r.get('flag_rate', float('nan')):>6.2f}  {state} vs {opp}")
EOF
echo
for log in $(ls -t wave2_ab*_w*_*.log 2>/dev/null | head -2); do echo "== $log"; tail -n 4 "$log"; done
for r in runs/wave2/ab*/; do [ -f "$r/results_ab.tsv" ] && { echo "== $r results_ab.tsv"; cat "$r/results_ab.tsv"; }; done
