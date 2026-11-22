#!/bin/bash
# The long test of a wave 2c round: the hybrid, nodur and a same-engine control against stock it13 at 120 s with enough
# games for a tight 95% range (about +-0.03 at 1300 games, against +-0.055 for one 8 h row). Run it on PACE from the
# repo root after cluster/launch_wave2c.sh (it waits for that round's build), or later on a round whose build passed:
#   bash cluster/launch_long_test.sh               # the latest round (runs/wave2c/LATEST), 5 parts a row
#   ROUND=r2 PARTS=3 bash cluster/launch_long_test.sh
# Each row is split into PARTS parallel 8 h eval jobs (cluster/eval.sbatch, -J w2c_<row>_p<k>), each in its own folder
# runs/wave2c/<ROUND>/long/<row>/part<k> with its own seed (SEED_BASE + k: other openings in another order). eval
# resumes whatever partial games a folder holds, so a folder is never reused: a second launch adds parts k+1.. next to
# the old ones. One 8 h job plays about 250-300 games at 120 s (the 2026-11-19 rows), so 5 parts give about 1300 games
# a row. cluster/status_wave2c.sh and cluster/decide_full_run.py merge the parts (one row, a cluster bootstrap over
# openings); decide prefers a merged row over the round row it stands in for (cluster/wave2c_configs.py LONG_ROWS).
#   l_hybrid_120  the hybrid (h_best_120's options) vs upstream-it13             stands in for h_best_120
#   l_nodur_120   nodur (c_nodur_120's options) vs upstream-it13                  stands in for c_nodur_120
#   l_same_120    the hybrid vs itself (pool opponent wave2-self, hybrid on): must read 0.50   for c_same_120
# Every part of every row shares the seed of that part number, so the hybrid and nodur play the same opening schedule.
# Dependency: afterok on the round's w2c_build when it is queued, else BUILD=<job id>, else none when the round's
# build.ok exists (the build passed and left the engines at the pin). The jobs are named w2c_*, so the round's watchdog
# cancels them with the round if the build or the GPU check fails, and decide counts them as the round still running.
# Settings: ROUND (runs/wave2c/LATEST), PARTS (5), PAIRS_PER_PART (200: more than 8 h plays, the job ends at its time
# limit), SEED_BASE (1000), BUILD, W2C_RUNS (runs/wave2c, for dry runs). Cost: rows x PARTS x 8 GPU-hours (3 x 5 x 8 =
# 120 at the defaults), printed with the job ids.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
W2C_RUNS="${W2C_RUNS:-runs/wave2c}"
ROUND="${ROUND:-$(cat "$W2C_RUNS/LATEST" 2>/dev/null || true)}"
PARTS="${PARTS:-5}"
PAIRS_PER_PART="${PAIRS_PER_PART:-200}"
SEED_BASE="${SEED_BASE:-1000}"
JOB_HOURS=8   # cluster/eval.sbatch's time limit

if [ -z "$ROUND" ]; then
  echo "no round: $W2C_RUNS/LATEST missing (bash cluster/launch_wave2c.sh first, or ROUND=rN)" >&2; exit 1
fi
OUT="$W2C_RUNS/$ROUND"
[ -d "$OUT" ] || { echo "$OUT does not exist" >&2; exit 1; }
case "$PARTS" in ''|*[!0-9]*|0) echo "PARTS must be a positive number" >&2; exit 1 ;; esac

B=$(squeue -h -u "$USER" -n w2c_build -o "%i" | head -1)
B="${B:-${BUILD:-}}"
if [ -n "$B" ]; then
  DEP=(--dependency="afterok:$B")
  echo "round $OUT: the parts wait for build job $B"
elif [ -f "$OUT/build.ok" ]; then
  DEP=()
  echo "round $OUT: build passed ($(cat "$OUT/build.ok")), no dependency"
else
  echo "no w2c_build job in the queue and no $OUT/build.ok: launch the round first (bash cluster/launch_wave2c.sh)," >&2
  echo "or BUILD=<job id> of a build that is still running" >&2
  exit 1
fi

# The same common settings as the wave2c rows (cluster/launch_checks.sh wave2c): pool contender, equal clock, MEASURE=1.
W2=(POOL=arena/pools/wave2.json SETUP=pool OPENINGS=bughouse RULES=v3 MEASURE=1 MEASURE_NODES=800
    "WARM_KINDS=upstream wave2")
HYB='{"durative_tree": false, "sit_gate": true, "hybrid": true}'
NODUR='{"durative_tree": false, "sit_gate": true}'
LONG="$OUT/long"
mkdir -p "$LONG"
ids=()
part() {  # row, opponent, contender options, opponent options, k
  local dir="$LONG/$1/part$5"
  mkdir -p "$dir"   # taken now, so a second launch before this job starts picks the next number
  # A subshell with exports, as launch_checks.sh's w2c: OPPONENT_KW stays unset (not empty) when a row has none.
  (export NAME="${dir#runs/}" OPPONENT="$2" CONTENDER_KW="$3" BASE_MS=120000 PAIRS="$PAIRS_PER_PART" \
     SEED=$((SEED_BASE + $5)) "${W2[@]}"
   [ -z "$4" ] || export OPPONENT_KW="$4"
   sbatch --parsable ${DEP[@]+"${DEP[@]}"} -J "w2c_$1_p$5" cluster/eval.sbatch)
}
row() {  # row, opponent, contender options, opponent options
  local k=1 i id mine=()
  while [ -e "$LONG/$1/part$k" ]; do k=$((k + 1)); done
  for ((i = 0; i < PARTS; i++)); do
    id=$(part "$1" "$2" "$3" "$4" $((k + i)))
    ids+=("$id"); mine+=("$id")
    printf '%s\tpart%s\tseed %s\t%s\n' "$1" $((k + i)) $((SEED_BASE + k + i)) "$id" >> "$LONG/jobs.tsv"
  done
  echo "$1: parts $k-$((k + PARTS - 1)), jobs ${mine[*]}"
}
#    row            opponent       contender options  opponent options
row  l_hybrid_120   upstream-it13  "$HYB"             ""
row  l_nodur_120    upstream-it13  "$NODUR"           ""
row  l_same_120     wave2-self     "$HYB"             "$HYB"

echo "long test $LONG: ${#ids[@]} jobs (${ids[*]})"
echo "GPU-hours: at most $((${#ids[@]} * JOB_HOURS)) by the time limits (3 rows x $PARTS parts x $JOB_HOURS h);"
echo "each part plays about 250-300 games, so about $((PARTS * 250))-$((PARTS * 300)) games a row"
echo "status: bash cluster/status_wave2c.sh    decide: .venv/bin/python cluster/decide_full_run.py"
