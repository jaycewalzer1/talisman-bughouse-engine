#!/bin/bash
#SBATCH -J selfplay -A paceship-bughouse -q inferno
#SBATCH -N1 -n8 --mem=32G --gres=gpu:rtx_6000:1 -t 08:00:00
#SBATCH -o selfplay_%x_%j.log
# Out-of-step self-play for the custom engine (docs/custom_engine_spec.md section 3) into runs/<NAME>.
# Both teams are the free-mode controller on the same network: real clocks, FreePass, no sit rules, think time =
# clock / k (k per game and team from K_RANGE), a jitter factor per search (JITTER), one machine per team.
# Samples land in runs/<NAME>/training_data as HVM6 chunks with .pos.jsonl sidecars (spec v2: arrival labels,
# masked think_gain labels for tools/think_labels.py; SHARD_VERSION=5 writes v1 HVM5 chunks).
#   NAME=custom/main/it01/selfplay MODEL=third_party/hivemind/engine/models/hivemind-clock-v2-fp32.onnx \
#     GAMES=400 sbatch cluster/selfplay.sbatch
# Several jobs can share one NAME as long as each has its own SEED (chunks and game ids are tagged by seed):
#   NAME=... MODEL=... sbatch --array=0-7 cluster/selfplay.sbatch      # SEED = array index
# Other settings: WORKERS (6), K_RANGE (60,300), JITTER (0.5,2.0), NODES_RANGE (16,1600), FREE_SIT_MS (500),
# BASE_MS (60000), MAX_PLIES (400), BINARY (hivemind-clock; or a path; HIVEMIND_CLOCK_BINARY also works),
# SHARD_VERSION (6), THINK_HEAD (0; 1 = scale budgets by the network's think_gain, needs a network with that head),
# NO_ELAPSED (0; 1 = leave the per-board elapsed times out of 'position', as in v1).
# Spec v3 (WP-C): SP_REGIME (v3; v2 = the old regime: fixed BASE_MS, k from K_RANGE, uniform JITTER). Under v3 the
# base time is drawn per game from a mix (60000:0.25,120000:0.25,180000:0.35,300000:0.15), k = base / think target
# (200-1000 ms), and BASE_MS / K_RANGE / JITTER are not used. Optional overrides, each passed only when set (SP_ prefix
# so nothing in the cluster environment collides): SP_BASE_MIX, SP_THINK_TARGET_MS, SP_TIME_ODDS (on|off), SP_CHARGE
# (searched|requested), SP_MIN_THINK_MS, SP_PACE (on|off), SP_PACE_MIXTURE (on|off), SP_LAG_MS, SP_NOISE (on|off),
# SP_TEMPERATURE, SP_TEMPERATURE_PLIES, SP_PCR_P, SP_PCR_NODES, SP_DEDUPE (bucket|position), SP_CLOCK_BUCKET_MS,
# SP_SIT_RECHECK (on|off), SP_RECHECK_NODES, SP_SCALE_MAX_PLIES (on|off), SP_OPENINGS (train), SP_MS_PER_NODE.
# SP_MS_PER_NODE (default: the arena's 5.5, the Mac CPU's) should be the value measured for this binary and network
# on the GPU (arena.pool's measured ms_per_node), or every budget is the Mac's.
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(git rev-parse --show-toplevel)}"
source cluster/env.sh

NAME="${NAME:?set NAME, the output folder under runs/}"
MODEL="${MODEL:?set MODEL, the clock network ONNX (fp32)}"
SEED="${SEED:-${SLURM_ARRAY_TASK_ID:-0}}"
OUT="runs/$NAME"
PY=.venv/bin/python

echo "commit $(git rev-parse HEAD)  selfplay $OUT  seed $SEED  games ${GAMES:-400}"
echo "model $MODEL  binary ${BINARY:-hivemind-clock}  regime ${SP_REGIME:-v3}  k ${K_RANGE:-60,300}  jitter ${JITTER:-0.5,2.0}"

# Optional spec v3 overrides: "--flag value" for every variable that is set.
V3_ARGS=()
for pair in BASE_MIX:--base-mix THINK_TARGET_MS:--think-target-ms TIME_ODDS:--time-odds CHARGE:--charge \
    MIN_THINK_MS:--min-think-ms PACE:--pace PACE_MIXTURE:--pace-mixture LAG_MS:--lag-ms NOISE:--noise \
    TEMPERATURE:--temperature TEMPERATURE_PLIES:--temperature-plies PCR_P:--pcr-p PCR_NODES:--pcr-nodes \
    DEDUPE:--dedupe CLOCK_BUCKET_MS:--clock-bucket-ms SIT_RECHECK:--sit-recheck RECHECK_NODES:--recheck-nodes \
    SCALE_MAX_PLIES:--scale-max-plies MS_PER_NODE:--ms-per-node OPENINGS:--openings; do
  var="SP_${pair%%:*}"; flag="${pair#*:}"
  if [ -n "${!var:-}" ]; then V3_ARGS+=("$flag" "${!var}"); fi
done
echo "v3 overrides: ${V3_ARGS[*]:-none}"
$PY -c "from arena.players.hivemind import resolve_binary; print('engine binary', resolve_binary('${BINARY:-hivemind-clock}'))"

# A network's first start on a GPU type builds its TensorRT engine file (~200 s), longer than a player's 120 s wait.
$PY cluster/warm_engine.py --binary "${BINARY:-hivemind-clock}" --model "$MODEL" || exit 1

$PY -m arena.selfplay --out "$OUT" --model "$MODEL" --binary "${BINARY:-hivemind-clock}" \
  --games "${GAMES:-400}" --workers "${WORKERS:-6}" --seed "$SEED" --game-id-offset "$((SEED * 1000000))" \
  --k-range "${K_RANGE:-60,300}" --jitter "${JITTER:-0.5,2.0}" --nodes-range "${NODES_RANGE:-16,1600}" \
  --free-sit-ms "${FREE_SIT_MS:-500}" --base-ms "${BASE_MS:-60000}" --max-plies "${MAX_PLIES:-400}" \
  --summary "$OUT/summary_s$SEED.json" --no-events --shard-version "${SHARD_VERSION:-6}" \
  --regime "${SP_REGIME:-v3}" ${V3_ARGS[@]+"${V3_ARGS[@]}"} \
  $([ "${THINK_HEAD:-0}" = 1 ] && echo --think-head) $([ "${NO_ELAPSED:-0}" = 1 ] && echo --no-elapsed)
