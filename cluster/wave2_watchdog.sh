#!/bin/bash
#SBATCH -J w2_watchdog -A paceship-bughouse -q inferno
#SBATCH -N1 -n1 --mem=2G -t 24:00:00
#SBATCH -o w2_watchdog_%j.log
# Stops the wave 2 round on failure, so GPU hours are not spent past a bad result. CPU only. Every 15 minutes:
#   build    BUILD (the cluster/build.sh job) did not complete         -> cancel every w2_* row and ab1_* training job
#   control  same_120 off 0.50 (95% range excludes 0.50, |score - 0.50| > 0.03, at least 400 games)
#                                                                       -> the harness is unfair: cancel everything
#   errors   any row with errors in more than 5% of its games (at least 100 games)   -> cancel everything
#   headline vs_it13_120 or ref_120 with its 95% range below 0.45 (at least 400 games)
#                                                                       -> wave 2 search is broken: cancel the training
#   crash    an ab1_w<arm> job FAILED or ran out of memory              -> cancel that arm's queued jobs
#   gate     an arm's iteration scores under 0.40 against it13         -> cancel that arm's queued jobs
# Every decision goes to runs/wave2/watchdog.log. It resubmits itself before its 24 h limit while jobs remain.
#   BUILD=<job id> sbatch --dependency=afterany:<job id> cluster/wave2_watchdog.sh
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(git rev-parse --show-toplevel)}"
BUILD="${BUILD:?set BUILD, the cluster/build.sh job id}"
RUN="${RUN:-ab1}"
POLL_S="${POLL_S:-900}"
LIFE_S="${LIFE_S:-84600}"   # 23.5 h, then hand over to a fresh copy
PY=.venv/bin/python
LOG=runs/wave2/watchdog.log
mkdir -p runs/wave2

say() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }
jobs_named() { squeue -h -u "$USER" -o "%j" | grep -E "$1" | grep -vx w2_watchdog | sort -u; }
cancel() {  # regex of job names, reason
  local names; names=$(jobs_named "$1")
  [ -z "$names" ] && return 0
  say "CANCEL $(echo $names) : $2"
  for n in $names; do scancel -u "$USER" -n "$n"; done
}
row() {  # row name -> "games score lo hi errors" (empty if no games yet)
  $PY tools/compare_evals.py --json "runs/wave2/$1" 2>/dev/null | $PY -c '
import json, sys
d = json.loads(sys.stdin.readline() or "{}")
if d.get("games"):
    print(d["games"], d["score"], d["ci95"][0], d["ci95"][1], d.get("errors", 0))'
}

state=$(sacct -n -X -j "$BUILD" -o State | head -1 | tr -d ' ')
if [ "$state" != COMPLETED ]; then
  cancel "^(w2_|${RUN}_w)" "build $BUILD ended $state"
  exit 0
fi
say "watching (build $BUILD COMPLETED, run $RUN)"

start=$SECONDS
while [ -n "$(jobs_named "^(w2_|${RUN}_w)")" ]; do
  r=$(row same_120)
  if [ -n "$r" ] && $PY -c 'import sys; g,s,lo,hi,e=map(float,sys.argv[1:]); sys.exit(0 if g>=400 and (hi<0.5 or lo>0.5) and abs(s-0.5)>0.03 else 1)' $r; then
    cancel "^(w2_|${RUN}_w)" "control same_120 off 0.50: games score lo hi errors = $r"
  fi
  for d in runs/wave2/*/; do
    n=$(basename "$d"); r=$(row "$n")
    if [ -n "$r" ] && $PY -c 'import sys; g,s,lo,hi,e=map(float,sys.argv[1:]); sys.exit(0 if g>=100 and e>0.05*g else 1)' $r; then
      cancel "^(w2_|${RUN}_w)" "row $n has errors: games score lo hi errors = $r"
    fi
  done
  for n in vs_it13_120 ref_120; do
    r=$(row $n)
    if [ -n "$r" ] && $PY -c 'import sys; g,s,lo,hi,e=map(float,sys.argv[1:]); sys.exit(0 if g>=400 and hi<0.45 else 1)' $r; then
      cancel "^${RUN}_w" "headline $n below 0.45: games score lo hi errors = $r"
    fi
  done
  for arm in 0 1; do
    bad=$(sacct -n -X -u "$USER" -S now-7days --name "${RUN}_w$arm" -o JobID,State | grep -E 'FAILED|OUT_OF_ME' | head -1)
    [ -n "$bad" ] && cancel "^${RUN}_w$arm\$" "training arm $arm job failed: $bad"
    lineage=$([ $arm = 0 ] && echo narrow || echo wide)
    tsv="runs/wave2/$RUN/$lineage/results.tsv"
    if [ -f "$tsv" ] && $PY -c '
import csv, sys
rows = [r for r in csv.DictReader(open(sys.argv[1]), delimiter="\t") if r.get("vs_it13")]
sys.exit(0 if rows and float(rows[-1]["vs_it13"]) < 0.40 else 1)' "$tsv"; then
      cancel "^${RUN}_w$arm\$" "training arm $arm ($lineage) scores under 0.40 vs it13: $(tail -1 "$tsv")"
    fi
  done
  if [ $((SECONDS - start)) -ge "$LIFE_S" ]; then
    say "handing over to a fresh watchdog"
    BUILD=$BUILD RUN=$RUN sbatch cluster/wave2_watchdog.sh >/dev/null
    exit 0
  fi
  sleep "$POLL_S"
done
say "no wave 2 jobs left; done"
