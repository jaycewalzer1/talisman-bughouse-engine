#!/bin/bash
#SBATCH -J elohunt_speed -A paceship-bughouse -q inferno
#SBATCH -N1 -n6 --mem=32G --gres=gpu:rtx_6000:1 -t 00:55:00
#SBATCH -o /storage/project/ps-bughouse-0/jwalzer3/elohunt-speed/elohunt_speed_%j.log
# Speed-gap diagnosis (read-only use of the main checkout's binaries; every file written is under elohunt-speed).
set -uo pipefail
REPO=/storage/project/ps-bughouse-0/jwalzer3/talisman-bughouse-engine
D=/storage/project/ps-bughouse-0/jwalzer3/elohunt-speed
cd "$REPO" && source cluster/env.sh && cd "$D"
PY="$REPO/.venv/bin/python"
BU=$REPO/third_party/upstream-hivemind/engine/build-ninja/hivemind.bin
BW=$REPO/third_party/wave2-hivemind/engine/build-ninja/hivemind.bin
POS=$REPO/third_party/wave2-hivemind/engine/tests/data/stock_parity_positions.txt
OUT=$D/results_${SLURM_JOB_ID:-local}.jsonl
nvidia-smi -L; nvidia-smi --query-gpu=name,clocks.sm,clocks.max.sm,power.draw,temperature.gpu,utilization.gpu --format=csv
lscpu | grep -E "Model name|^CPU\(s\)|MHz" ; nproc
run() { $PY $D/gpu_bench.py "$@" | tee -a "$OUT"; }
# Fresh engine builds first (each ~3-4 min) so the rounds below can use them.
for m in mF1 mF2 mD; do
  echo "=== raw inference $m (builds its engine)"; (cd $(dirname $BU)/..; $BU bench --model $D/$m/$( [ $m = mD ] && echo dummy.onnx || echo twin-s-clock-fp32.onnx) 3000) 2>&1 | tail -4
done
for m in mU mW; do echo "=== raw inference $m"; (cd $(dirname $BU)/..; $BU bench --model $D/$m/twin-s-clock-fp32.onnx 3000) 2>&1 | tail -3; done
for m in mU mW; do echo "=== raw inference $m with BW"; (cd $(dirname $BW)/..; $BW bench --model $D/$m/twin-s-clock-fp32.onnx 3000) 2>&1 | tail -3; done
for round in 1 2 3; do
  echo "=== round $round"
  if [ $((round % 2)) = 1 ]; then order="A B C D"; else order="D C B A"; fi
  for c in $order; do
    case $c in
      A) run --label "BU+mU" --binary $BU --model $D/mU/twin-s-clock-fp32.onnx --positions $POS ;;
      B) run --label "BW+mW hybrid" --binary $BW --model $D/mW/twin-s-clock-fp32.onnx --flags hybrid --positions $POS ;;
      C) run --label "BU+mW" --binary $BU --model $D/mW/twin-s-clock-fp32.onnx --positions $POS ;;
      D) run --label "BW+mU hybrid" --binary $BW --model $D/mU/twin-s-clock-fp32.onnx --flags hybrid --positions $POS ;;
    esac
  done
done
echo "=== flags on our binary (own engine file)"
run --label "BW+mW stock" --binary $BW --model $D/mW/twin-s-clock-fp32.onnx --flags stock
run --label "BW+mW nodur" --binary $BW --model $D/mW/twin-s-clock-fp32.onnx --flags nodur
echo "=== fresh engines"
for m in mF1 mF2; do
  run --label "BU+$m" --binary $BU --model $D/$m/twin-s-clock-fp32.onnx
  run --label "BW+$m hybrid" --binary $BW --model $D/$m/twin-s-clock-fp32.onnx --flags hybrid
done
echo "=== CPU side: dummy network (near-free inference)"
for r in 1 2; do
  run --label "BU+dummy" --binary $BU --model $D/mD/dummy.onnx --positions $POS
  run --label "BW+dummy hybrid" --binary $BW --model $D/mD/dummy.onnx --flags hybrid --positions $POS
done
echo "=== node sweep"
for n in 200 3200; do
  run --label "BU+mU n$n" --binary $BU --model $D/mU/twin-s-clock-fp32.onnx --nodes $n --repeats 6
  run --label "BW+mW hybrid n$n" --binary $BW --model $D/mW/twin-s-clock-fp32.onnx --flags hybrid --nodes $n --repeats 6
done
nvidia-smi --query-gpu=name,clocks.sm,power.draw,temperature.gpu --format=csv
echo done
