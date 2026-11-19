#!/bin/bash
#SBATCH -J bench_selfplay -A paceship-bughouse -q inferno
#SBATCH -N1 -n6 --mem=32G --gres=gpu:rtx_6000:1 -t 01:30:00
#SBATCH -o bench_selfplay_%j.log
# WP-I throughput bench (docs/audit_2026-11-17.md WP-I, docs/measurements_bench.md): games per GPU-hour and GPU
# utilisation of arena.selfplay at production settings, on the same allocation as cluster/selfplay.sbatch (one
# RTX 6000, 6 cores (the node class allows 6 per GPU), 6 workers = 6 games at once, 2 engine processes per game, 4 search threads each).
#   sbatch cluster/bench_selfplay.sbatch                     # default stages, ~1 GPU-hour
#   sbatch -n16 cluster/bench_selfplay.sbatch                # same stages with 16 cores: is the 8-core run CPU-bound?
# Local dry run of the script path (no GPU; one short game):
#   STAGES="60000:1:1" bash cluster/bench_selfplay.sbatch
# Settings:
#   STAGES  space-separated BASE:WORKERS:GAMES, one arena.selfplay run each (default
#           "60000:6:60 180000:6:30 60000:12:72": the two production base times at 6 workers, then 60 s at 12 workers
#           to see whether more games at once raise games/hour or just share the same GPU/CPU). BASE is a base time
#           in ms, or "mix" for the production base-time mix itself (v3 only; e.g. "mix:6:60" measures the mix
#           directly instead of interpolating it from the 60 s and 180 s stages)
#   SP_REGIME (v3, as cluster/selfplay.sbatch: base time fixed per stage via --base-mix BASE:1, k = base / think
#           target 200-1000 ms, i.e. k 60-300 at 60 s, pace mixture instead of jitter, time odds, PCR, searched-node
#           charging; v2 = the old regime with fixed BASE_MS, K_RANGE and JITTER)
#   SP_* overrides as in cluster/selfplay.sbatch (SP_MS_PER_NODE above all: set it to the GPU-measured value or every
#           budget is the Mac's 5.5 ms/node), passed only when set
#   MODEL (engine/models/hivemind-clock-v3-fp32.onnx), BINARY (hivemind-clock), K_RANGE (60,300, v2 only),
#   NODES_RANGE (16,1600), JITTER (0.5,2.0, v2 only), FREE_SIT_MS (500), MAX_PLIES (400; v3 scales it by base/60 s),
#   SEED (0), SAMPLE_S (1, GPU sampling period), EXTRA_ARGS (passed to every arena.selfplay run, for flags added
#   later; record them in the doc with the numbers)
# Writes runs/cluster/bench_selfplay_<job id, or local_<timestamp>>/: stages.tsv, <stage>/summary.json, <stage>/gpu.csv,
# <stage>/cpu.tsv, <stage>/training_data (written because production writes it; its cost is part of
# the measurement), report.json, and prints the summary of cluster/bench_selfplay_summary.py.
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(git rev-parse --show-toplevel)}"

STAGES="${STAGES:-60000:6:60 180000:6:30 60000:12:72}"
MODEL="${MODEL:-third_party/hivemind/engine/models/hivemind-clock-v3-fp32.onnx}"
BINARY="${BINARY:-hivemind-clock}"
SAMPLE_S="${SAMPLE_S:-1}"
SEED="${SEED:-0}"
REGIME="${SP_REGIME:-v3}"
# A local run gets a timestamped folder so a rerun never mixes with (or overwrites) an earlier one.
OUT="runs/cluster/bench_selfplay_${SLURM_JOB_ID:-local_$(date +%Y%m%d_%H%M%S)}"
PY=.venv/bin/python

if [ -n "${SLURM_JOB_ID:-}" ]; then
  source cluster/env.sh || exit 1
fi
[ -f "$MODEL" ] || { echo "model $MODEL missing" >&2; exit 1; }

mkdir -p "$OUT"
printf 'stage\tbase_ms\tworkers\tgames\tstart\tend\texit\n' > "$OUT/stages.tsv"
echo "commit $(git rev-parse HEAD), hivemind $(git -C third_party/hivemind rev-parse HEAD)"
git status --short | grep -v '^??' || true
echo "model $MODEL ($(shasum -a 256 "$MODEL" | cut -c1-16))  binary $BINARY  regime $REGIME  stages $STAGES"

# Spec v3 overrides, the same SP_* variables and flags as cluster/selfplay.sbatch (except SP_BASE_MIX: the stage
# sets the base time). Production self-play is v3, so the bench is too unless SP_REGIME=v2.
V3_ARGS=()
for pair in THINK_TARGET_MS:--think-target-ms TIME_ODDS:--time-odds CHARGE:--charge \
    MIN_THINK_MS:--min-think-ms PACE:--pace PACE_MIXTURE:--pace-mixture LAG_MS:--lag-ms NOISE:--noise \
    TEMPERATURE:--temperature TEMPERATURE_PLIES:--temperature-plies PCR_P:--pcr-p PCR_NODES:--pcr-nodes \
    DEDUPE:--dedupe CLOCK_BUCKET_MS:--clock-bucket-ms SIT_RECHECK:--sit-recheck RECHECK_NODES:--recheck-nodes \
    SCALE_MAX_PLIES:--scale-max-plies MS_PER_NODE:--ms-per-node OPENINGS:--openings; do
  var="SP_${pair%%:*}"; flag="${pair#*:}"
  if [ -n "${!var:-}" ]; then V3_ARGS+=("$flag" "${!var}"); fi
done
echo "v3 overrides: ${V3_ARGS[*]:-none}"
echo "host $(hostname)  cores $(nproc 2>/dev/null || sysctl -n hw.ncpu)  slurm cpus ${SLURM_CPUS_ON_NODE:-none}"
$PY -c "from arena.players.hivemind import resolve_binary; print('engine binary', resolve_binary('$BINARY'))"
command -v nvidia-smi >/dev/null && nvidia-smi -L

# The first start of a network on a GPU type builds its TensorRT engine file (~200 s); keep that out of the stages.
$PY cluster/warm_engine.py --binary "$BINARY" --model "$MODEL" || exit 1

# CPU used by this job: the Slurm cgroup's usage_usec when there is one (counts only our processes), otherwise the
# machine-wide busy jiffies from /proc/stat (Linux) or nothing (Mac). One line per period: epoch, kind, counter.
cpu_counter() {
  local cg
  cg=$(awk -F: '$1=="0"{print $3}' /proc/self/cgroup 2>/dev/null)
  if [ -n "$cg" ] && [ -f "/sys/fs/cgroup$cg/cpu.stat" ]; then
    echo "cgroup_usec $(awk '$1=="usage_usec"{print $2}' "/sys/fs/cgroup$cg/cpu.stat")"
  elif [ -f /proc/stat ]; then
    awk '$1=="cpu"{print "procstat_busy_jiffies", $2+$3+$4+$7+$8}' /proc/stat
  fi
}

SAMPLERS=()
stop_samplers() {
  if [ ${#SAMPLERS[@]} -gt 0 ]; then kill "${SAMPLERS[@]}" 2>/dev/null; wait "${SAMPLERS[@]}" 2>/dev/null; fi
  SAMPLERS=()
}
trap stop_samplers EXIT

i=0
for stage in $STAGES; do
  IFS=: read -r base workers games <<< "$stage"
  i=$((i + 1))
  if [ "$base" = mix ]; then
    [ "$REGIME" = v3 ] || { echo "stage $stage: a mix stage needs SP_REGIME=v3" >&2; exit 1; }
    name="s${i}_bmix_w${workers}"; base_args=(--base-ms 60000)
  elif [ "$REGIME" = v3 ]; then
    name="s${i}_b$((base / 1000))_w${workers}"; base_args=(--base-mix "$base:1" --base-ms "$base")
  else
    name="s${i}_b$((base / 1000))_w${workers}"; base_args=(--base-ms "$base")
  fi
  dir="$OUT/$name"
  mkdir -p "$dir"
  label="$base ms"; [ "$base" = mix ] && label="production mix"
  echo; echo "=== $name: base $label, regime $REGIME, $workers games at once, $games games"
  if command -v nvidia-smi >/dev/null; then
    nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,power.draw \
      --format=csv,noheader,nounits -lms "$((SAMPLE_S * 1000))" > "$dir/gpu.csv" &
    SAMPLERS+=($!)
  fi
  (while true; do c=$(cpu_counter); [ -n "$c" ] && echo "$(date +%s.%N | sed 's/N$//') $c"; sleep 2; done) \
    > "$dir/cpu.tsv" &
  SAMPLERS+=($!)
  start=$(date +%s)
  # shellcheck disable=SC2086
  $PY -m arena.selfplay --out "$dir" --model "$MODEL" --binary "$BINARY" --games "$games" --workers "$workers" \
    --seed "$SEED" --game-id-offset "$(( (SEED * 100 + i) * 1000000 ))" --run-id "bench-$name" \
    --k-range "${K_RANGE:-60,300}" --jitter "${JITTER:-0.5,2.0}" --nodes-range "${NODES_RANGE:-16,1600}" \
    --free-sit-ms "${FREE_SIT_MS:-500}" "${base_args[@]}" --max-plies "${MAX_PLIES:-400}" \
    --summary "$dir/summary.json" --no-events --regime "$REGIME" ${V3_ARGS[@]+"${V3_ARGS[@]}"} ${EXTRA_ARGS:-}
  code=$?
  end=$(date +%s)
  stop_samplers
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$name" "$base" "$workers" "$games" "$start" "$end" "$code" >> "$OUT/stages.tsv"
done

echo; echo "=== summary"
$PY cluster/bench_selfplay_summary.py "$OUT" --cpus "${SLURM_CPUS_ON_NODE:-$(nproc 2>/dev/null || sysctl -n hw.ncpu)}" \
  --model "$MODEL" --regime "$REGIME"
