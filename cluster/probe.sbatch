#!/bin/bash
#SBATCH -J probe -A paceship-bughouse -q inferno
#SBATCH -N1 -n6 --mem=32G --gres=gpu:rtx_6000:1 -t 06:00:00
#SBATCH -o probe_%j.log
# Wait-regret probe (arena/probe_wait.py) on recorded games: from moments where a player had asked for a piece, play
# the game out several times moving at once and several times waiting first, and compare the team's score.
# Only games against one opponent are used, because the probe rebuilds every player with one node budget.
#   RUN=runs/baseline_split OPPONENT=stock-centralized POSITIONS=3 ROLLOUTS=4 sbatch cluster/probe.sbatch
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(git rev-parse --show-toplevel)}"
source cluster/env.sh

RUN="${RUN:-runs/baseline_split}"
OPPONENT="${OPPONENT:-stock-centralized}"
OUT="${OUT:-runs/probe_wait_${SLURM_JOB_ID:-local}}"
PY=.venv/bin/python

echo "commit $(git rev-parse HEAD)"
$PY cluster/warm_engine.py
$PY cluster/engine_parity.py check

mapfile -t GAMES < <($PY - "$RUN" "$OPPONENT" <<'PYEOF'
import json, os, sys
run, opponent = sys.argv[1:]
for g in json.load(open(os.path.join(run, "summary.json")))["games"]:
    if g.get("opponent") == opponent and "error" not in g:
        print(os.path.join(run, g["file"]))
PYEOF
)
echo "${#GAMES[@]} games against $OPPONENT from $RUN"

$PY -m arena.probe_wait "${GAMES[@]}" --kind hivemind --nodes 200 --lend --waits 2000 5000 \
  --positions "${POSITIONS:-3}" --rollouts "${ROLLOUTS:-4}" --workers 3 --out "$OUT"
