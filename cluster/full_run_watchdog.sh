#!/bin/bash
#SBATCH -J full_watchdog -A paceship-bughouse -q inferno
#SBATCH -N1 -n1 --mem=2G -t 24:00:00
#SBATCH -o full_watchdog_%j.log
# Stops the full run (cluster/launch_full_run.sh) on failure, so GPU hours are not spent past a bad result. CPU only.
# Every 30 minutes while a full_<RUN> job is left:
#   crash    a full_<RUN> job FAILED or ran out of memory (the next job would repeat it)  -> cancel the chain
#   gate     the last finished iteration scores under 0.40 against it13 (results.tsv vs_it13) -> cancel the chain
# Every decision goes to RUNS_DIR/RUN/watchdog.log. It resubmits itself before its 24 h limit while jobs remain.
#   RUN=full1 RUNS_DIR=runs/full sbatch cluster/full_run_watchdog.sh
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(git rev-parse --show-toplevel)}"
RUN="${RUN:?set RUN}"
RUNS_DIR="${RUNS_DIR:-runs/full}"
POLL_S="${POLL_S:-1800}"
LIFE_S="${LIFE_S:-84600}"   # 23.5 h, then hand over to a fresh copy
# Only jobs that ended after the first watchdog started count, so a failure before a RESUME=1 relaunch does not.
SINCE="${SINCE:-$(date +%Y-%m-%dT%H:%M:%S)}"
PY=.venv/bin/python
LOG="$RUNS_DIR/$RUN/watchdog.log"
mkdir -p "$RUNS_DIR/$RUN"

say() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }
queued() { squeue -h -u "$USER" -n "full_$RUN" -o "%i" | head -1; }
cancel() { say "CANCEL full_$RUN : $1"; scancel -u "$USER" -n "full_$RUN"; }

say "watching full_$RUN"
start=$SECONDS
while [ -n "$(queued)" ]; do
  bad=$(sacct -n -X -u "$USER" -S "$SINCE" --name "full_$RUN" -o JobID,State | grep -E 'FAILED|OUT_OF_ME' | head -1)
  if [ -n "$bad" ]; then cancel "a job failed: $bad"; break; fi
  tsv="$RUNS_DIR/$RUN/results.tsv"
  if [ -f "$tsv" ] && $PY -c '
import csv, sys
rows = [r for r in csv.DictReader(open(sys.argv[1]), delimiter="\t") if r.get("vs_it13")]
sys.exit(0 if rows and float(rows[-1]["vs_it13"]) < 0.40 else 1)' "$tsv"; then
    cancel "last iteration under 0.40 vs it13: $(tail -1 "$tsv")"; break
  fi
  if [ $((SECONDS - start)) -ge "$LIFE_S" ]; then
    say "handing over to a fresh watchdog"
    RUN=$RUN RUNS_DIR=$RUNS_DIR SINCE=$SINCE sbatch cluster/full_run_watchdog.sh >/dev/null
    exit 0
  fi
  sleep "$POLL_S"
done
say "no full_$RUN jobs left; done"
