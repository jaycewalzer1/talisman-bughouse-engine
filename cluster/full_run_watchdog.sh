#!/bin/bash
#SBATCH -J full_watchdog -A paceship-bughouse -q inferno
#SBATCH -N1 -n1 --mem=2G -t 24:00:00
#SBATCH -o full_watchdog_%j.log
# Stops the full run (cluster/launch_full_run.sh) on failure, so GPU hours are not spent past a bad result. CPU only.
# Every 30 minutes while a full_<RUN> job is left:
#   crash    a full_<RUN> job FAILED or ran out of memory (the next job would repeat it)  -> cancel the chain
#   gate     the last finished iteration scores under 0.40 against it13 (results.tsv vs_it13) -> cancel the chain
#   workers  3 or more self-play workers (sp_<RUN>, launch_full_run.sh WORKERS=K) FAILED or ran out of memory -> cancel
#   cap      with GPU_HOURS set, the GPU-hours of full_<RUN> and sp_<RUN> jobs since the run began reach it -> stop:
#            STOP_WORKERS (no new workers), rl-loop.stop (the main job exits before its next stage), pending jobs
#            cancelled; running jobs finish their segment or stage
# Every decision goes to RUNS_DIR/RUN/watchdog.log. It resubmits itself before its 24 h limit while jobs remain.
#   RUN=full1 RUNS_DIR=runs/full [GPU_HOURS=900] sbatch cluster/full_run_watchdog.sh
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(git rev-parse --show-toplevel)}"
RUN="${RUN:?set RUN}"
RUNS_DIR="${RUNS_DIR:-runs/full}"
POLL_S="${POLL_S:-1800}"
LIFE_S="${LIFE_S:-84600}"   # 23.5 h, then hand over to a fresh copy
# Only jobs that ended after the first watchdog started count, so a failure before a RESUME=1 relaunch does not.
SINCE="${SINCE:-$(date +%Y-%m-%dT%H:%M:%S)}"
GPU_HOURS="${GPU_HOURS:-}"
PY=.venv/bin/python
LOG="$RUNS_DIR/$RUN/watchdog.log"
mkdir -p "$RUNS_DIR/$RUN"

say() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }
queued() { squeue -h -u "$USER" -n "full_$RUN,sp_$RUN" -o "%i" | head -1; }
cancel() { say "CANCEL full_$RUN and sp_$RUN : $1"; scancel -u "$USER" -n "full_$RUN"; scancel -u "$USER" -n "sp_$RUN"; }
gpu_hours() {  # GPU-hours of this run's jobs since SINCE (running jobs counted so far)
  sacct -n -X -P -u "$USER" -S "$SINCE" --name "full_$RUN,sp_$RUN" -o ElapsedRaw,AllocTRES \
    | awk -F'|' '{ if (match($2, /gres\/gpu=[0-9]+/)) s += $1 * substr($2, RSTART + 9, RLENGTH - 9) } END { printf "%.1f", s / 3600 }'
}
stop_run() {
  say "STOP at the cap: $1"
  echo "$1" > "$RUNS_DIR/$RUN/STOP_WORKERS"
  mkdir -p "$RUNS_DIR/$RUN/runs" && touch "$RUNS_DIR/$RUN/runs/rl-loop.stop"
  scancel -u "$USER" -t PENDING -n "full_$RUN"; scancel -u "$USER" -t PENDING -n "sp_$RUN"
}

say "watching full_$RUN"
start=$SECONDS
while [ -n "$(queued)" ]; do
  bad=$(sacct -n -X -u "$USER" -S "$SINCE" --name "full_$RUN" -o JobID,State | grep -E 'FAILED|OUT_OF_ME' | head -1)
  if [ -n "$bad" ]; then cancel "a job failed: $bad"; break; fi
  wbad=$(sacct -n -X -u "$USER" -S "$SINCE" --name "sp_$RUN" -o JobID,State | grep -cE 'FAILED|OUT_OF_ME' || true)
  if [ "$wbad" -ge 3 ]; then cancel "$wbad self-play workers failed"; break; fi
  if [ -n "$GPU_HOURS" ] && [ ! -e "$RUNS_DIR/$RUN/STOP_WORKERS" ]; then
    used=$(gpu_hours)
    say "GPU-hours $used of $GPU_HOURS"
    if awk -v u="$used" -v c="$GPU_HOURS" 'BEGIN { exit !(u >= c) }'; then stop_run "$used of $GPU_HOURS GPU-hours used"; fi
  fi
  tsv="$RUNS_DIR/$RUN/results.tsv"
  if [ -f "$tsv" ] && $PY -c '
import csv, sys
rows = [r for r in csv.DictReader(open(sys.argv[1]), delimiter="\t") if r.get("vs_it13")]
sys.exit(0 if rows and float(rows[-1]["vs_it13"]) < 0.40 else 1)' "$tsv"; then
    cancel "last iteration under 0.40 vs it13: $(tail -1 "$tsv")"; break
  fi
  if [ $((SECONDS - start)) -ge "$LIFE_S" ]; then
    say "handing over to a fresh watchdog"
    RUN=$RUN RUNS_DIR=$RUNS_DIR SINCE=$SINCE GPU_HOURS=$GPU_HOURS sbatch cluster/full_run_watchdog.sh >/dev/null
    exit 0
  fi
  sleep "$POLL_S"
done
say "no full_$RUN jobs left; done"
