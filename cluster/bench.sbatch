#!/bin/bash
#SBATCH -J bench -A paceship-bughouse -q inferno
#SBATCH -N1 -n32 --mem=64G --gres=gpu:1 -t 01:00:00
#SBATCH -o bench_%j.log
# Measures games per GPU-hour on Phoenix. See cluster/README.md. The defaults fit inside the 1 hour limit, because
# no stage starts a new game after STAGE_SECONDS. Override them to test locally:
#   NODES=40 STAGE_SECONDS=20 WORKERS_LIST="1 2" AGENT_WORKERS=2 bash cluster/bench.sbatch
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(git rev-parse --show-toplevel)}"

NODES="${NODES:-200}"
STAGE_SECONDS="${STAGE_SECONDS:-540}"
WORKERS_LIST="${WORKERS_LIST:-1 4 8}"
AGENT_WORKERS="${AGENT_WORKERS:-4}"
OUT="runs/cluster/bench_${SLURM_JOB_ID:-local}"
PY=.venv/bin/python

if [ -n "${SLURM_JOB_ID:-}" ]; then
  source cluster/env.sh
  # PARITY_OPTIONAL=1 is for measuring speed and cost only. Game results from a build that failed parity don't count.
  if [ ! -f runs/cluster/parity_passed ] && [ -z "${PARITY_OPTIONAL:-}" ]; then
    echo "parity has not passed on this build, so not spending GPU time" >&2
    exit 1
  fi
fi

mkdir -p "$OUT"
echo "commit $(git rev-parse HEAD), hivemind $(git -C third_party/hivemind rev-parse HEAD)"
git status --short | grep -v '^??' || true
command -v pace-quota >/dev/null && pace-quota | grep -A3 "Job Charge Account"

SAMPLERS=()
if command -v nvidia-smi >/dev/null; then
  nvidia-smi -L
  nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv -l 5 > "$OUT/gpu.csv" &
  SAMPLERS+=($!)
fi
(while sleep 30; do uptime; done) > "$OUT/load.log" &
SAMPLERS+=($!)
trap 'kill "${SAMPLERS[@]}" 2>/dev/null || true' EXIT

for workers in $WORKERS_LIST; do
  echo "=== stock Hivemind, centralized, $workers games at once"
  $PY -m arena.bench --setups centralized --games 400 --nodes "$NODES" --workers "$workers" \
    --max-seconds "$STAGE_SECONDS" --out "$OUT/stock_w$workers.json"
done

echo "=== untrained scheduler, $AGENT_WORKERS games at once"
$PY -m arena.bench --kind agent --setups sharing --lend --games 20 --nodes "$NODES" --workers "$AGENT_WORKERS" \
  --max-seconds "$STAGE_SECONDS" --out "$OUT/agent_w$AGENT_WORKERS.json"

echo "=== summary"
$PY - "$OUT" <<'EOF'
import glob, json, os, sys
for path in sorted(glob.glob(os.path.join(sys.argv[1], "*.json"))):
    report = json.load(open(path))["report"]
    per_hour = report["projection"]["games_per_hour"] if report["projection"] else None
    warn = "  <-- under 50 games per GPU-hour" if per_hour is None or per_hour < 50 else ""
    print(f"{os.path.basename(path):<20} games/hour {per_hour}  aborted {report['aborted']}  "
          f"invalid moves {report['invalid_moves']}{warn}")
EOF
command -v pace-quota >/dev/null && pace-quota | grep -A3 "Job Charge Account"
exit 0
