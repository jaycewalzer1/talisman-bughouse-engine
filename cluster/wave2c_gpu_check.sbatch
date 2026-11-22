#!/bin/bash
#SBATCH -J w2c_gpu -A paceship-bughouse -q inferno
#SBATCH -N1 -n6 --mem=48G --gres=gpu:rtx_6000:1 -t 04:00:00
#SBATCH -o w2c_gpu_%j.log
# Wave 2c readiness round, job (a): the GPU engine checks the wave 2c round and the full run stand on, after
# cluster/build.sh. Each check is one line of OUT/gpu_check.tsv (name, PASS / FAIL / SKIP, seconds, detail) and the whole
# goes to OUT/gpu_check.json ("overall": PASS only when no check FAILED). Exit 1 on any FAIL: the bench and smoke jobs
# (afterok on this job) then never start and cluster/wave2c_watchdog.sh cancels the round.
#   pin            third_party/wave2-hivemind HEAD equals cluster/wave2_engine.ref
#   w2_env         the wave2-hivemind Python env (uv sync, or venv + pip without uv)
#   sp1_build      the one-worker self-play engine at that commit (cluster/build_sp1.sh: TensorRT, rebuilt when stale)
#   options        build-sp1 and build-ninja both advertise the wave 2c options (TeamTreeReuse, JointPlans, PlanCarry,
#                  OwnPaceBelowRoot, WaitNodeActingEval, BlockableCheckNotMate) and the hybrid's (HybridMode, spec v6):
#                  a binary without them would refuse every continuity or hybrid row and self-play run
#   narrow_parity  it13 (narrow, 74 planes) on the TensorRT match engine against Python ONNX Runtime, wave 2 flags and
#                  seat clocks (tools/upstream_check.py parity --engine wave2, tolerance 2e-2: TensorRT runs FP16)
#   wide_net       it13 widened (twin-s-clock-wide, 130 planes) made by `python -m hivemind.wave2 widen` when missing
#   wide_parity    the same parity on the wide network (WAVE2_MODEL=...-wide-fp32.onnx): the wide TensorRT path, never
#                  compiled on PACE before this round
#   stock_parity   stock upstream v3.0.3 parity on TensorRT (tools/upstream_check.py parity --engine upstream)
#   stock_binary   engine/tests/stock_parity.sh (wave 2 binary with every option off = stock v3.0.3 bit for bit): needs an
#                  ONNX Runtime C++ install (ONNXRuntime_ROOT), so SKIP when there is none
#   sp_<config>_<width>  6 self-play games on TensorRT through the rl-loop segment command (cluster/wave2_selfplay_bench.py)
#                  for nodur, cont and hybrid, narrow and wide: the self-play TensorRT path of each candidate and width
# cluster/check_clock_engine.py is not here: it checks our fork's hivemind-clock build (a think_gain head it randomises),
# which it13 and the wide network do not have; cluster/build.sh runs it on the clock network and fails the build if it fails.
#   OUT=runs/wave2c/<ROUND> sbatch --dependency=afterok:<build> cluster/wave2c_gpu_check.sbatch   (launch_wave2c.sh does this)
# Local dry run of the script path (no GPU, ONNX Runtime engines): OUT=runs/wave2c/local bash cluster/wave2c_gpu_check.sbatch
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(git rev-parse --show-toplevel)}"
[ -n "${SLURM_JOB_ID:-}" ] && source cluster/env.sh

OUT="${OUT:?set OUT, the round folder (runs/wave2c/<ROUND>)}"
W2="${W2:-$PWD/third_party/wave2-hivemind}"
W2PY="$W2/.venv/bin/python"
PY=.venv/bin/python
TOL="${WAVE2_TOL:-2e-2}"
SP_GAMES="${SP_GAMES:-6}"
mkdir -p "$OUT/gpu_check"
TSV="$OUT/gpu_check.tsv"
printf 'check\tstatus\tseconds\tdetail\n' > "$TSV"
echo "commit $(git rev-parse HEAD)  wave2 $(git -C "$W2" rev-parse HEAD)  pin $(cat cluster/wave2_engine.ref)"
command -v nvidia-smi >/dev/null && nvidia-smi -L

check() {  # name, detail, command...: runs the command, its output to OUT/gpu_check/<name>.log, one row in the tsv
  local name="$1" detail="$2" start=$SECONDS status
  shift 2
  echo; echo "== $name: $*"
  if "$@" > "$OUT/gpu_check/$name.log" 2>&1; then status=PASS; else status=FAIL; fi
  tail -n 3 "$OUT/gpu_check/$name.log"
  printf '%s\t%s\t%s\t%s\n' "$name" "$status" "$((SECONDS - start))" "$detail" >> "$TSV"
  echo "$name: $status"
}
skip() { printf '%s\tSKIP\t0\t%s\n' "$1" "$2" >> "$TSV"; echo "$1: SKIP ($2)"; }

check pin "wave2-hivemind HEAD = cluster/wave2_engine.ref" \
  test "$(git -C "$W2" rev-parse HEAD)" = "$(cat cluster/wave2_engine.ref)"
w2_env() {
  if command -v uv >/dev/null 2>&1; then (cd "$W2" && uv sync)
  else { [ -x "$W2PY" ] || "${PYTHON:-python3}" -m venv "$W2/.venv"; } && "$W2PY" -m pip install --quiet -e "$W2"; fi
}
check w2_env "uv sync in third_party/wave2-hivemind" w2_env
check sp1_build "build-sp1 at the pinned commit" bash cluster/build_sp1.sh "$W2"
options() {
  local e h o
  for e in "$W2/engine/build-sp1/hivemind" "$W2/engine/build-ninja/hivemind"; do
    h=$("$e" --help 2>&1)   # not `--help | grep -q`: SIGPIPE under pipefail (wave2_iterate.sbatch)
    for o in team-tree-reuse joint-plans plan-carry own-pace-below-root wait-node-acting-eval blockable-check-not-mate \
             hybrid-mode; do
      grep -q -- "--$o" <<<"$h" || { echo "$e lacks --$o"; return 1; }
    done
    echo "$e: all wave 2c and hybrid options"
  done
}
check options "both engines advertise the wave 2c and hybrid options" options
check narrow_parity "it13 TensorRT vs ONNX Runtime, tol $TOL" \
  $PY tools/upstream_check.py parity --engine wave2 --coreml default --positions 40 --tol "$TOL"
WIDE_FP32="$W2/engine/models/twin-s-clock-wide-fp32.onnx"
if [ -f "$WIDE_FP32" ] && [ -f "$W2/engine/models/twin-s-clock-wide.onnx" ]; then
  skip wide_net "already in engine/models"
else
  check wide_net "python -m hivemind.wave2 widen" bash -c "cd '$W2' && '$W2PY' -m hivemind.wave2 widen"
fi
check wide_parity "wide it13 TensorRT vs ONNX Runtime, tol $TOL" \
  env WAVE2_MODEL="$WIDE_FP32" $PY tools/upstream_check.py parity --engine wave2 --coreml default --positions 40 --tol "$TOL"
check stock_parity "stock upstream TensorRT vs ONNX Runtime, tol $TOL" \
  $PY tools/upstream_check.py parity --engine upstream --coreml default --positions 40 --tol "$TOL"
if [ -n "${ONNXRuntime_ROOT:-}" ] && [ -d "${ONNXRuntime_ROOT:-}" ]; then
  check stock_binary "engine/tests/stock_parity.sh (ONNX Runtime builds)" bash -c "cd '$W2' && JOBS=2 sh engine/tests/stock_parity.sh"
else
  skip stock_binary "no ONNX Runtime C++ install (ONNXRuntime_ROOT): the bit-for-bit stock check runs on the Mac"
fi
for config in nodur cont hybrid; do
  for width in narrow wide; do
    # one folder per job: a resubmitted check plays its games again in a fresh folder (the bench refuses a used one)
    check "sp_${config}_$width" "$SP_GAMES self-play games, $config, $width, TensorRT build-sp1" \
      "$W2PY" cluster/wave2_selfplay_bench.py --config "$config" --width "$width" --games "$SP_GAMES" \
        --out "$OUT/gpu_check/sp_${config}_${width}_${SLURM_JOB_ID:-local_$$}"
  done
done

$PY - "$TSV" "$OUT/gpu_check.json" "$(git -C "$W2" rev-parse HEAD)" "$(cat cluster/wave2_engine.ref)" <<'EOF'
import csv, json, sys
rows = list(csv.DictReader(open(sys.argv[1]), delimiter="\t"))
overall = "FAIL" if any(r["status"] == "FAIL" for r in rows) or not rows else "PASS"
json.dump({"overall": overall, "engine_commit": sys.argv[3], "pin": sys.argv[4], "checks": rows},
          open(sys.argv[2], "w"), indent=1)
print()
for r in rows:
    print(f"{r['check']:<18} {r['status']:<5} {r['seconds']:>6}s  {r['detail']}")
print(f"gpu check: {overall}")
sys.exit(0 if overall == "PASS" else 1)
EOF
