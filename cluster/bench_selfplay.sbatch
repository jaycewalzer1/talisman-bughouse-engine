#!/bin/bash
#SBATCH -J bench_selfplay -A paceship-bughouse -q inferno
#SBATCH -N1 -n8 --mem=32G --gres=gpu:rtx_6000:1 -t 01:30:00
#SBATCH -o bench_selfplay_%j.log
# WP-I throughput bench (docs/audit_2026-11-17.md WP-I, docs/measurements_bench.md): games per GPU-hour and GPU
# utilisation of arena.selfplay at production settings, on the same allocation as cluster/selfplay.sbatch (one
# RTX 6000, 8 cores, 6 workers = 6 games at once, 2 engine processes per game, 4 search threads each).
#   sbatch cluster/bench_selfplay.sbatch                     # default stages, ~1 GPU-hour
#   sbatch -n16 cluster/bench_selfplay.sbatch                # same stages with 16 cores: is the 8-core run CPU-bound?
# Local dry run of the script path (no GPU; one short game):
#   STAGES="60000:1:1" bash cluster/bench_selfplay.sbatch
# Settings:
#   STAGES  space-separated BASE_MS:WORKERS:GAMES, one arena.selfplay run each (default
#           "60000:6:60 180000:6:30 60000:12:72": the two production base times at 6 workers, then 60 s at 12 workers
#           to see whether more games at once raise games/hour or just share the same GPU/CPU)
#   MODEL (engine/models/hivemind-clock-v2-fp32.onnx), BINARY (hivemind-clock), K_RANGE (60,300),
#   NODES_RANGE (16,1600), JITTER (0.5,2.0), FREE_SIT_MS (500), MAX_PLIES (400), SEED (0),
#   SAMPLE_S (1, GPU sampling period), EXTRA_ARGS (passed to every arena.selfplay run, for flags added later,
#   e.g. WP-C's min-move or base-time-mix options; record them in the doc with the numbers)
# Writes runs/cluster/bench_selfplay_<job id or local>/: stages.tsv, <stage>/summary.json, <stage>/gpu.csv,
# <stage>/cpu.tsv, <stage>/training_data (written because production writes it; its cost is part of
# the measurement), report.json, and prints the summary of cluster/bench_selfplay_summary.py.
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(git rev-parse --show-toplevel)}"

STAGES="${STAGES:-60000:6:60 180000:6:30 60000:12:72}"
MODEL="${MODEL:-third_party/hivemind/engine/models/hivemind-clock-v2-fp32.onnx}"
BINARY="${BINARY:-hivemind-clock}"
SAMPLE_S="${SAMPLE_S:-1}"
SEED="${SEED:-0}"
OUT="runs/cluster/bench_selfplay_${SLURM_JOB_ID:-local}"
PY=.venv/bin/python

if [ -n "${SLURM_JOB_ID:-}" ]; then
  source cluster/env.sh || exit 1
fi
[ -f "$MODEL" ] || { echo "model $MODEL missing" >&2; exit 1; }

mkdir -p "$OUT"
printf 'stage\tbase_ms\tworkers\tgames\tstart\tend\texit\n' > "$OUT/stages.tsv"
echo "commit $(git rev-parse HEAD), hivemind $(git -C third_party/hivemind rev-parse HEAD)"
git status --short | grep -v '^??' || true
echo "model $MODEL ($(shasum -a 256 "$MODEL" | cut -c1-16))  binary $BINARY  stages $STAGES"
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
  name="s${i}_b$((base / 1000))_w${workers}"
  dir="$OUT/$name"
  mkdir -p "$dir"
  echo; echo "=== $name: base ${base} ms, $workers games at once, $games games"
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
    --free-sit-ms "${FREE_SIT_MS:-500}" --base-ms "$base" --max-plies "${MAX_PLIES:-400}" \
    --summary "$dir/summary.json" --no-events ${EXTRA_ARGS:-}
  code=$?
  end=$(date +%s)
  stop_samplers
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$name" "$base" "$workers" "$games" "$start" "$end" "$code" >> "$OUT/stages.tsv"
done

echo; echo "=== summary"
$PY cluster/bench_selfplay_summary.py "$OUT" --cpus "${SLURM_CPUS_ON_NODE:-$(nproc 2>/dev/null || sysctl -n hw.ncpu)}" \
  --model "$MODEL"
