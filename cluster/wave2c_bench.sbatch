#!/bin/bash
#SBATCH -J w2c_bench -A paceship-bughouse -q inferno
#SBATCH -N1 -n6 --mem=32G --gres=gpu:rtx_6000:1 -t 03:00:00
#SBATCH -o w2c_bench_%x_%j.log
# Wave 2c readiness round, job (c): wave 2 self-play throughput R (games per GPU-hour) of one candidate full-run config,
# on the narrow network (it13) and then the wide one (it13 widened), BENCH_GAMES games each, on the allocation a
# wave2_iterate.sbatch segment gets (one RTX 6000, 6 cores, 6 games at once on build-sp1 with TensorRT). The command is
# the one rl-loop runs for a self-play segment (cluster/wave2_selfplay_bench.py), so R here is R in the full run.
# Writes OUT/bench/<CONFIG>_<width>[_<job>]/bench.json (end-to-end and steady games per GPU-hour, games, exit);
# cluster/decide_full_run.py reads the steady rate (end to end when steady is missing) of the chosen config and width.
#   OUT=runs/wave2c/<ROUND> CONFIG=nodur sbatch --dependency=afterok:<gpu check> cluster/wave2c_bench.sbatch
# Settings: OUT (required), CONFIG (required: a name in cluster/wave2c_configs.py), BENCH_GAMES (150 per width: about
# 30 min at R = 300), WIDTHS ("narrow wide"), PARALLEL_GAMES (6), SEED (7001).
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(git rev-parse --show-toplevel)}"
[ -n "${SLURM_JOB_ID:-}" ] && source cluster/env.sh

OUT="${OUT:?set OUT, the round folder (runs/wave2c/<ROUND>)}"
CONFIG="${CONFIG:?set CONFIG, nodur, cont or hybrid (cluster/wave2c_configs.py)}"
W2="$PWD/third_party/wave2-hivemind"
W2PY="$W2/.venv/bin/python"
echo "commit $(git rev-parse HEAD)  wave2 $(git -C "$W2" rev-parse HEAD)  pin $(cat cluster/wave2_engine.ref)"
echo "config $CONFIG: $("$W2PY" cluster/wave2c_configs.py "$CONFIG")"
command -v nvidia-smi >/dev/null && nvidia-smi -L

code=0
for width in ${WIDTHS:-narrow wide}; do
  dir="$OUT/bench/${CONFIG}_$width"
  # a resubmitted bench keeps the earlier folder and measures again next to it (decide reads the newest)
  [ -e "$dir" ] && dir="${dir}_${SLURM_JOB_ID:-local_$$}"
  echo; echo "=== $CONFIG, $width: ${BENCH_GAMES:-150} games into $dir"
  "$W2PY" cluster/wave2_selfplay_bench.py --config "$CONFIG" --width "$width" --games "${BENCH_GAMES:-150}" \
    --parallel-games "${PARALLEL_GAMES:-6}" --seed "${SEED:-7001}" --out "$dir" || code=1
done
exit $code
