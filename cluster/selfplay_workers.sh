#!/bin/bash
# Tops up the self-play workers of a concurrent run (cluster/wave2_iterate.sbatch WORKERS=K). rl-loop runs it as its
# --selfplay-hook at the start of each iteration's self-play, inside the main job, with the main job's settings in the
# environment: it submits WORKERS minus the sp_<RUN> jobs already queued or running, each a copy of
# cluster/wave2_iterate.sbatch with WORKER=1 that plays claimed segments of that iteration and exits when none is left
# (so a worker never sits on a GPU through training and the gate). Not fatal: rl-loop plays on alone if this fails.
# Skipped when RUNS_DIR/RUN/STOP_WORKERS exists (the watchdog writes it at the run's GPU-hour cap).
# Every submission goes to RUNS_DIR/RUN/workers.tsv (time, iteration, job id).
set -uo pipefail
cd "$(dirname "$0")/.."
RUN="${RUN:?the main job exports RUN}"
RUNS_DIR="${RUNS_DIR:-runs/wave2}"
WORKERS="${WORKERS:?the main job exports WORKERS}"
WORKER_TIME="${WORKER_TIME:-16:00:00}"
OUT="$RUNS_DIR/$RUN"
it="${RL_ITERATION:-?}"

if [ -e "$OUT/STOP_WORKERS" ]; then
  echo "it$it: $OUT/STOP_WORKERS exists ($(cat "$OUT/STOP_WORKERS")): no workers"
  exit 0
fi
have=$(squeue -h -u "$USER" -n "sp_$RUN" -t PENDING,RUNNING -o "%i" | wc -l | tr -d ' ')
need=$((WORKERS - have))
if [ "$need" -le 0 ]; then
  echo "it$it: $have sp_$RUN workers already queued or running"
  exit 0
fi
# A job's own memory settings would override the sbatch file's in the child job.
unset SLURM_MEM_PER_NODE SLURM_MEM_PER_CPU SLURM_MEM_PER_GPU
ids=()
for _ in $(seq "$need"); do
  id=$(WORKER=1 WORKERS=0 sbatch --parsable -J "sp_$RUN" -t "$WORKER_TIME" cluster/wave2_iterate.sbatch) || break
  ids+=("$id")
  printf '%s\tit%s\t%s\n' "$(date '+%F %T')" "$it" "$id" >> "$OUT/workers.tsv"
done
echo "it$it: $have workers were queued, submitted ${#ids[@]}: ${ids[*]:-none}"
[ "${#ids[@]}" -eq "$need" ]
