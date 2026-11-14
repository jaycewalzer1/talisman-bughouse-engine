#!/bin/bash
#SBATCH -J train -A paceship-bughouse -q inferno
#SBATCH -N1 -n6 --mem=32G --gres=gpu:rtx_6000:1 -t 12:00:00
#SBATCH -o train_%x_%j.log
# One scheduler training run in runs/<NAME>. Settings come from the environment:
#   NAME=waves/wave1 FEATURES=tempo BASELINE=eval EXPLORATION=0.15 THINK_LOGIT=2 sbatch -J wave1 cluster/train.sbatch
# LR is the optimizer's step size and SNAPSHOT_EVERY how often a frozen copy joins the opponents (0 = never).
# It continues runs/<NAME> if that run already has saved progress. It stops itself before the time limit and
# submits its own continuation, so a run longer than 12 hours needs nothing from you.
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(git rev-parse --show-toplevel)}"
source cluster/env.sh

NAME="${NAME:?set NAME, the run folder under runs/}"
OUT="runs/$NAME"
ITERATIONS="${ITERATIONS:-30}"
PY=.venv/bin/python
RUN_SECONDS="${RUN_SECONDS:-41400}"  # 11.5 hours, inside the 12 hour limit

echo "commit $(git rev-parse HEAD)  run $OUT  features '${FEATURES:-}'  baseline ${BASELINE:-mean}"
echo "lr ${LR:-3e-4}  think logit ${THINK_LOGIT:-6}  exploration ${EXPLORATION:-0.05}  pairs ${PAIRS:-50}  seed ${SEED:-0}"
$PY -c "from arena.players.hivemind import DEFAULT_MODEL; print('engine model', DEFAULT_MODEL)"
$PY cluster/warm_engine.py || exit 1
$PY cluster/engine_parity.py check || exit 1

RESUME=()
[ -f "$OUT/state.pt" ] && RESUME=(--resume)
timeout "$RUN_SECONDS" $PY -m arena.train_scheduler --out "$OUT" --iterations "$ITERATIONS" \
  --pairs "${PAIRS:-50}" --workers 3 --seed "${SEED:-0}" --features "${FEATURES:-}" \
  --baseline "${BASELINE:-mean}" --exploration "${EXPLORATION:-0.05}" --think-logit "${THINK_LOGIT:-6}" \
  --lr "${LR:-3e-4}" --snapshot-every "${SNAPSHOT_EVERY:-20}" \
  ${RESUME[@]+"${RESUME[@]}"}
status=$?

if [ "$status" -eq 124 ]; then
  echo "time is nearly up: submitting a continuation of $OUT"
  sbatch -J "${SLURM_JOB_NAME:-train}" cluster/train.sbatch
elif [ "$status" -eq 0 ]; then
  $PY tools/train_tables.py "$OUT" --split "$((ITERATIONS / 2))"
fi
exit "$status"
