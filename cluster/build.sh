#!/bin/bash
# Builds the TensorRT engines (stock build-ninja/hivemind and the clock build build-clock/hivemind-clock), the Python
# env and the spec v2 clock network, then runs the C++ and arena tests and the clock engine's think_gain check.
# Run on a GPU node.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
source cluster/env.sh

git submodule update --init third_party/hivemind
echo "commit $(git rev-parse HEAD), hivemind $(git -C third_party/hivemind rev-parse HEAD)"

[ -x .venv/bin/python ] || "$PYTHON" -m venv .venv
.venv/bin/python -m pip install --quiet torch chess einops onnx onnxruntime onnxscript onnxsim pandas polars pytest \
  timm tensorboard rtpt numpy matplotlib websockets
.venv/bin/python -m pip install --quiet --no-deps -e third_party/hivemind

MODELS=third_party/hivemind/engine/models
FP16="$MODELS/hivemind-it04-crossboard-risev33-loss1.556-p82.0.onnx"
.venv/bin/python third_party/hivemind/tools/fetch_network.py
[ -f "$MODELS/hivemind-fp32.onnx" ] || \
  .venv/bin/python third_party/hivemind/engine/scripts/convert_onnx_fp32.py "$FP16" "$MODELS/hivemind-fp32.onnx"

# Native-arch code would crash on a node with an older CPU than the one that built it.
(cd third_party/hivemind/engine
 cmake --preset ninja-release -DHIVEMIND_BACKEND=tensorrt -DTensorRT_DIR="$TensorRT_DIR" \
   -DCUDA_TOOLKIT_ROOT_DIR="$CUDA_TOOLKIT_ROOT_DIR" -DHIVEMIND_USE_CCACHE=OFF -DHIVEMIND_NATIVE_ARCH=OFF
 # Two compiles at a time: the unity files are heavy, and five at once passed 32 GB (job 15455397). Two peak at 20 GB.
 cmake --build build-ninja --parallel "${CMAKE_BUILD_PARALLEL_LEVEL:-2}"
 # RootMateScanStopsAtItsDeadline is a stopwatch check (return within 200 ms). It took 249 ms on an RTX 6000 node's
 # CPU with native-arch code off, and passes on the login nodes, so it is skipped here.
 # Other tests give a search a wall-clock deadline (120 to 800 ms) and expect it to finish its proof. Which of those
 # miss varies with the node (two missed in job 15488611 that passed in 15456719), so a failed test gets three tries.
 ctest --test-dir build-ninja --output-on-failure --repeat until-pass:3 \
   -E '^EngineTest\.RootMateScanStopsAtItsDeadline$')

# The custom-engine loop (iterate.sh, selfplay.sbatch, think_labels.py: --binary hivemind-clock) runs the clock build:
# 44 planes per board (spec v3: v2 plus the diagonal clock plane), think_gain output. Under TensorRT it loads only
# 88-channel (v3) networks.
(cd third_party/hivemind/engine
 cmake --preset ninja-release -B build-clock -DHIVEMIND_CLOCK_PLANES=ON -DHIVEMIND_BACKEND=tensorrt \
   -DTensorRT_DIR="$TensorRT_DIR" -DCUDA_TOOLKIT_ROOT_DIR="$CUDA_TOOLKIT_ROOT_DIR" -DHIVEMIND_USE_CCACHE=OFF \
   -DHIVEMIND_NATIVE_ARCH=OFF
 cmake --build build-clock --parallel "${CMAKE_BUILD_PARALLEL_LEVEL:-2}"
 ctest --test-dir build-clock --output-on-failure --repeat until-pass:3 \
   -E '^EngineTest\.RootMateScanStopsAtItsDeadline$')

# The v3 clock network (models/ is not in git): it04's checkpoint widened by make_clock_net.py (default --planes 44).
CLOCK_V3="$MODELS/hivemind-clock-v3-fp32.onnx"
if [ ! -f "$CLOCK_V3" ]; then
  .venv/bin/python third_party/hivemind/tools/fetch_network.py --variant checkpoint
  .venv/bin/python third_party/hivemind/tools/make_clock_net.py --check
fi

# The first start on a new GPU type builds the TensorRT engine file and saves it. That took longer than the 120 s a
# player waits for the engine (job 15489984), so it is built here with a long wait.
.venv/bin/python cluster/warm_engine.py
.venv/bin/python cluster/warm_engine.py --binary hivemind-clock --model "$CLOCK_V3"
# think_gain on the GPU path: every position must print 'info string think_gain' and match Python ONNX Runtime.
.venv/bin/python cluster/check_clock_engine.py --model "$CLOCK_V3"

# Upstream v3 (the target engine, arena kind "upstream"): built with TensorRT next to our fork. UPSTREAM=0 skips it.
if [ "${UPSTREAM:-1}" = 1 ]; then
  BACKEND=tensorrt JOBS="${CMAKE_BUILD_PARALLEL_LEVEL:-2}" sh tools/build_upstream.sh
  .venv/bin/python cluster/warm_engine.py --kind upstream
fi

# Wave 2 (arena kind "wave2", docs/wave2_tests.md): branch talisman/wave2 of the fork (upstream v3.0.3 + the spec v4
# durative tree) into third_party/wave2-hivemind, then the wave 2 check: the engine's own tests and raw-policy parity
# against Python ONNX Runtime on twin-s-clock with the wave 2 flags and seat clocks. TensorRT runs the network in
# FP16, so parity is read at WAVE2_TOL (default 2e-2) instead of the Mac's 1e-4. WAVE2=0 skips it. The branch must be
# on GitHub (git push fork talisman/wave2), or set WAVE2_REPO to a reachable copy.
WAVE2_CHECK=skipped
if [ "${WAVE2:-1}" = 1 ]; then
  WAVE2=1 REPO="${WAVE2_REPO:-https://github.com/jaycewalzer1/hivemind}" BACKEND=tensorrt \
    JOBS="${CMAKE_BUILD_PARALLEL_LEVEL:-2}" sh tools/build_upstream.sh
  .venv/bin/python cluster/warm_engine.py --kind wave2
  WAVE2_CHECK=PASS
  (cd third_party/wave2-hivemind/engine
   cmake --build build-ninja --target hivemind_tests --parallel "${CMAKE_BUILD_PARALLEL_LEVEL:-2}"
   ctest --test-dir build-ninja --output-on-failure --repeat until-pass:3) || WAVE2_CHECK=FAIL
  .venv/bin/python tools/upstream_check.py parity --engine wave2 --coreml default --positions 40 \
    --tol "${WAVE2_TOL:-2e-2}" || WAVE2_CHECK=FAIL
  .venv/bin/python tools/upstream_check.py parity --engine upstream --coreml default --positions 40 \
    --tol "${WAVE2_TOL:-2e-2}" || WAVE2_CHECK=FAIL
  echo "wave2 check: $WAVE2_CHECK (engine tests, wave2 parity, stock upstream parity)"
fi

.venv/bin/python -m pytest arena/tests arena/tests_phase1 -q
echo "wave2 check: $WAVE2_CHECK"
[ "$WAVE2_CHECK" != FAIL ] || { echo "build done but the wave2 check FAILED: see above" >&2; exit 1; }
echo "build ok. Next: .venv/bin/python cluster/engine_parity.py check"
