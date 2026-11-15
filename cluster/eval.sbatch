#!/bin/bash
#SBATCH -J eval -A paceship-bughouse -q inferno
#SBATCH -N1 -n6 --mem=32G --gres=gpu:rtx_6000:1 -t 08:00:00
#SBATCH -o eval_%x_%j.log
# Scores one fixed team against one opponent on the held-out openings, into runs/<NAME>. No forced exploration and
# no learning, so two teams scored this way can be compared directly (tools/compare_evals.py).
# Scheduler: without POLICY it is untrained, which plays like stock plus whatever the feature groups do on their own.
#   NAME=eval/untrained_none sbatch -J none cluster/eval.sbatch
#   NAME=eval/wave2 FEATURES=mate,danger POLICY=runs/waves/wave2/latest.pt sbatch -J wave2 cluster/eval.sbatch
# RULE=mate_wait or mate_ask adds a fixed hand-written rule to the scheduler (it needs FEATURES to include mate).
# Stock Hivemind in a team setup (centralized, silent or sharing; LEND=1 lends idle compute), no scheduler:
#   NAME=match/central_100 SETUP=centralized NODES=100 sbatch -J c100 cluster/eval.sbatch
# NODES and OPPONENT_NODES replace the node budgets (200 each). PAIRS=800 is 1600 games, about 4 hours. A job that
# is cut off leaves no result, so keep it inside the time limit and use another SEED and NAME for more games.
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(git rev-parse --show-toplevel)}"
source cluster/env.sh

NAME="${NAME:?set NAME, the result folder under runs/}"
OUT="runs/$NAME"
PY=.venv/bin/python
if [ -f "$OUT/summary.json" ]; then
  echo "$OUT already has a result: pick another NAME" >&2
  exit 1
fi

ARGS=()
if [ -n "${SETUP:-}" ]; then
  ARGS+=(--setup "$SETUP")
  [ "${LEND:-0}" = 1 ] && ARGS+=(--lend)
  WHO="stock $SETUP lend ${LEND:-0}"
else
  ARGS+=(--agent --features "${FEATURES:-}")
  [ -n "${POLICY:-}" ] && ARGS+=(--policy "$POLICY")
  [ -n "${RULE:-}" ] && ARGS+=(--rule "$RULE")
  WHO="scheduler features '${FEATURES:-}' policy ${POLICY:-untrained} rule ${RULE:-none}"
fi
[ -n "${NODES:-}" ] && ARGS+=(--nodes "$NODES")
[ -n "${OPPONENT_NODES:-}" ] && ARGS+=(--opponent-nodes "$OPPONENT_NODES")

echo "commit $(git rev-parse HEAD)  eval $OUT  $WHO  nodes ${NODES:-200}"
echo "opponent ${OPPONENT:-stock-centralized} nodes ${OPPONENT_NODES:-as in the pool}  pairs ${PAIRS:-800}  seed ${SEED:-0}"
$PY -c "from arena.players.hivemind import DEFAULT_MODEL; print('engine model', DEFAULT_MODEL)"
$PY cluster/warm_engine.py || exit 1
$PY cluster/engine_parity.py check || exit 1

$PY -m arena.pool run "${ARGS[@]}" --opponent "${OPPONENT:-stock-centralized}" --openings eval \
  --pairs "${PAIRS:-800}" --workers 3 --seed "${SEED:-0}" --out "$OUT"
