#!/bin/bash
#SBATCH -J probe -A paceship-bughouse -q inferno
#SBATCH -N1 -n6 --mem=32G --gres=gpu:rtx_6000:1 -t 06:00:00
#SBATCH -o probe_%j.log
# Wait-regret probe (arena/probe_wait.py) on recorded games: from moments where a player had asked for a piece, play
# the game out several times moving at once and several times waiting first, and compare the team's score.
# Only games against one opponent are used, because the probe rebuilds every player with one node budget.
#   RUN=runs/baseline_split OPPONENT=stock-centralized POSITIONS=3 ROLLOUTS=4 sbatch cluster/probe.sbatch
# RUN may name several run folders, or a pattern such as "runs/waves/signal/it000[12]". The probe stops itself
# before the time limit and prints the table for the rollouts it finished.
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(git rev-parse --show-toplevel)}"
source cluster/env.sh

RUN="${RUN:-runs/baseline_split}"
OPPONENT="${OPPONENT:-stock-centralized}"
OUT="${OUT:-runs/probe_wait_${SLURM_JOB_ID:-local}}"
PY=.venv/bin/python

echo "commit $(git rev-parse HEAD)"
$PY cluster/warm_engine.py || exit 1
$PY cluster/engine_parity.py check || exit 1

# $RUN is left unquoted on purpose, so a pattern or a list becomes one argument per run folder.
mapfile -t GAMES < <($PY - "$OPPONENT" $RUN <<'PYEOF'
import json, os, sys
opponent, runs = sys.argv[1], sys.argv[2:]
for run in runs:
    for g in json.load(open(os.path.join(run, "summary.json")))["games"]:
        if g.get("opponent") == opponent and "error" not in g:
            print(os.path.join(run, g["file"]))
PYEOF
)
echo "${#GAMES[@]} games against $OPPONENT from $RUN"
[ "${#GAMES[@]}" -gt 0 ] || exit 1

timeout "${RUN_SECONDS:-19800}" $PY -m arena.probe_wait "${GAMES[@]}" --kind hivemind --nodes 200 --lend \
  --waits 2000 5000 --positions "${POSITIONS:-3}" --rollouts "${ROLLOUTS:-4}" --workers 3 --seed "${SEED:-0}" \
  --out "$OUT"
status=$?
if [ "$status" -eq 124 ]; then
  echo "time is nearly up: the table below covers the rollouts that finished"
  $PY tools/probe_summary.py "$OUT"
fi
exit "$status"
