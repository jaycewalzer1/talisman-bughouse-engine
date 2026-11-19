#!/bin/bash
# Prints the queue and every result of the checks so far. Safe to run at any time: a row with no result says so.
#   bash cluster/results.sh
# A match or eval row appears when its job finishes (about 4 hours after it starts). Learner rows and the probe
# table fill in while their jobs run.
cd "$(git rev-parse --show-toplevel)"
shopt -s nullglob
PY=.venv/bin/python
M=runs/match
E=runs/eval

echo "== queue"
squeue -u "$USER" -o "%.10i %.16j %.3t %.10M %R"

echo; echo "== team setups at 200 nodes, against one program at 200 nodes (first row should be 0.50)"
$PY tools/compare_evals.py $M/central_200 $M/silent_200 $M/sharing_200 $M/half_200 $M/lend_200

echo; echo "== think budgets, against one program at 200 nodes"
$PY tools/compare_evals.py $M/central_200 $M/central_50 $M/central_100 $M/central_400 $M/lend_100

echo; echo "== against one program at 100 nodes"
$PY tools/compare_evals.py $M/central_50_v100 $M/lend_100_v100

echo; echo "== untrained scheduler, feature groups on and off"
$PY tools/compare_evals.py $E/untrained_none $E/untrained_tempo $E/untrained_mate $E/untrained_serve $E/untrained_all

echo; echo "== hand-written mate rules (first row: same mate probes, no rule)"
$PY tools/compare_evals.py $E/rule_none $E/rule_wait $E/rule_ask

echo; echo "== untrained scheduler: always borrows the whole idle machine, or picks none, half or all itself"
$PY tools/compare_evals.py $E/untrained_none $E/untrained_lend

echo; echo "== learner check (wave1 is yesterday's run at the old step size)"
learner=(runs/learner/*)
if [ "${#learner[@]}" -gt 0 ]; then $PY tools/compare_runs.py "${learner[@]}" runs/waves/wave1; else echo "not started"; fi

echo; echo "== wait-regret probe"
probe=(runs/probe/*)
if [ "${#probe[@]}" -gt 0 ]; then $PY tools/probe_summary.py "${probe[@]}"; else echo "not started"; fi

echo; echo "== beat stock: one machine per team (decision rule: docs/beat_stock.md)"
Bt=runs/beat
echo "-- controls, should be 0.50"
$PY tools/compare_evals.py $Bt/same_100_v100 $Bt/same_50_v50
echo "-- fixed budgets against stock at 100 nodes, then at 50"
$PY tools/compare_evals.py $Bt/same_100_v100 $Bt/c25_v100 $Bt/c50_v100 $Bt/c200_v100
$PY tools/compare_evals.py $Bt/same_50_v50 $Bt/c25_v50 $Bt/c100_v50
echo "-- the time flag"
$PY tools/compare_evals.py $Bt/same_100_v100 $Bt/flag_99_v100 $Bt/nosit_100_v100 $Bt/flagnosit_99_v100
$PY tools/compare_evals.py $Bt/same_50_v50 $Bt/flag_49_v50
echo "-- clock-based think time against stock at 100 nodes, then at 50"
$PY tools/compare_evals.py $Bt/same_100_v100 $Bt/clock55_v100 $Bt/clock110_v100 $Bt/clock220_v100 $Bt/clock110_v100_open
$PY tools/compare_evals.py $Bt/same_50_v50 $Bt/clock55_v50 $Bt/clock110_v50 $Bt/clock220_v50
echo "-- 3 minute clock"
$PY tools/compare_evals.py $Bt/long_c100_v300 $Bt/long_clock110_v300

echo; echo "== custom engine (cluster/iterate.sh, docs/custom_engine_spec.md): self-play and gate per iteration"
loops=(runs/custom/*/results.tsv)
if [ "${#loops[@]}" -eq 0 ]; then echo "not started"; fi
for r in "${loops[@]}"; do echo "-- $r"; column -t -s $'\t' "$r"; done
gates=(runs/custom/*/it*/gate)
if [ "${#gates[@]}" -gt 0 ]; then
  echo "-- gates: new network (free mode) against stock Hivemind, both jittered, held-out openings"
  $PY tools/compare_evals.py "${gates[@]}"
fi

echo; echo "== wave 2: durative tree on upstream v3 vs stock it13 (docs/wave2_tests.md; same_120 must read 0.50 first)"
W2=runs/wave2
if [ -d "$W2" ]; then
  w2=()
  for r in same_120 vs_it13_120 vs_it13_180 vs_it13_800n_120 ref_120 nodur sitgate instant racewin noflag vs_it04_120; do
    [ -d "$W2/$r" ] && w2+=("$W2/$r")
  done
  [ "${#w2[@]}" -gt 0 ] && $PY tools/compare_evals.py "${w2[@]}"
  for r in "${w2[@]}"; do $PY tools/sprt.py "$r" 2>/dev/null | tail -1; done
else
  echo "not started"
fi
