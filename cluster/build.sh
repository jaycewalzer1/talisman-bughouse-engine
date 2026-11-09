#!/bin/bash
# Builds the TensorRT engine and the Python env, then runs the C++ and arena tests. Run on a GPU node.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
source cluster/env.sh

git submodule update --init third_party/hivemind
echo "commit $(git rev-parse HEAD), hivemind $(git -C third_party/hivemind rev-parse HEAD)"

[ -x .venv/bin/python ] || "$PYTHON" -m venv .venv
.venv/bin/python -m pip install --quiet torch chess einops onnx onnxruntime onnxscript onnxsim pandas polars pytest \
  timm tensorboard rtpt numpy matplotlib
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
 cmake --build build-ninja
 ctest --test-dir build-ninja --output-on-failure)

.venv/bin/python -m pytest arena/tests arena/tests_phase1 -q
echo "build ok. Next: .venv/bin/python cluster/engine_parity.py check"
