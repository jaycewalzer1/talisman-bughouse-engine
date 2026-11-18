#!/bin/sh
# Build upstream aminwoo/hivemind (the v3 clock model, "upstream-v3" in the arena) next to our fork and fetch its
# latest promoted twin-s-clock network. Our submodule third_party/hivemind is not touched.
#
#   sh tools/build_upstream.sh            # clone/checkout TAG, build engine, fetch + convert the network
#   TAG=v3.0.3 JOBS=2 sh tools/build_upstream.sh
#   BACKEND=tensorrt sh tools/build_upstream.sh   # on PACE (after source cluster/env.sh: TensorRT_DIR, CUDA root)
#
# Result:
#   third_party/upstream-hivemind/engine/build-ninja/hivemind          (ONNX Runtime CPU backend, -O3 release preset)
#   third_party/upstream-hivemind/engine/models/twin-s-clock.onnx      (HF main, SHA-256 checked below)
#   third_party/upstream-hivemind/engine/models/twin-s-clock-fp32.onnx (FP32 for ORT CPU: it has no FP16 kernels)
# The arena player is arena/players/upstream.py; docs/upstream/v3_arm.md records versions and checks.
set -eu

ROOT=$(cd "$(dirname "$0")/.." && pwd)
DEST="$ROOT/third_party/upstream-hivemind"
TAG=${TAG:-v3.0.3}                 # latest v3.0.x tag on 2026-11-17
JOBS=${JOBS:-2}                    # several agents/builds share this Mac
ORT_ROOT=${ONNXRuntime_ROOT:-/opt/homebrew/opt/onnxruntime}
BACKEND=${BACKEND:-onnxruntime}    # onnxruntime (Mac, CPU) or tensorrt (PACE GPU nodes)
PY=${PY:-$ROOT/.venv/bin/python}
# twin-s-clock iteration 13 ("latest promoted network" per the model card) at HF revision edaa246 (2026-10-26).
HF_REPO=aminwoo/bughouse-twin-s-clock
HF_REV=${HF_REV:-edaa246cdb86c43bb3f18f0836501912c080e7f1}
NET_SHA256=${NET_SHA256:-f4ac25e32a993776f26b42d79cc5c1e48b77a16a13d1e2a2db30528f43c9af73}  # = twin-s-clock-it13.onnx

if [ ! -d "$DEST/.git" ]; then
  git clone --depth 1 --branch "$TAG" https://github.com/aminwoo/hivemind "$DEST"
else
  git -C "$DEST" fetch --depth 1 origin "refs/tags/$TAG:refs/tags/$TAG"
  git -C "$DEST" -c advice.detachedHead=false checkout "$TAG"
fi
echo "upstream: $(git -C "$DEST" describe --tags) $(git -C "$DEST" rev-parse --short HEAD)"

cd "$DEST/engine"
if [ "$BACKEND" = tensorrt ]; then
  cmake --preset ninja-release -DHIVEMIND_BACKEND=tensorrt -DTensorRT_DIR="$TensorRT_DIR" \
    -DCUDA_TOOLKIT_ROOT_DIR="$CUDA_TOOLKIT_ROOT_DIR" -DHIVEMIND_USE_CCACHE=OFF -DHIVEMIND_NATIVE_ARCH=OFF
else
  cmake --preset ninja-release -DHIVEMIND_BACKEND=onnxruntime -DONNXRuntime_ROOT="$ORT_ROOT" \
    -DHIVEMIND_USE_CCACHE=OFF
fi
cmake --build build-ninja --target hivemind -j "$JOBS"

mkdir -p models
NET=models/twin-s-clock.onnx
if [ ! -f "$NET" ]; then
  curl -fL --retry 3 -o "$NET.part" "https://huggingface.co/$HF_REPO/resolve/$HF_REV/twin-s-clock.onnx"
  mv "$NET.part" "$NET"
fi
if command -v sha256sum >/dev/null 2>&1; then SUM=$(sha256sum "$NET" | cut -d' ' -f1); else SUM=$(shasum -a 256 "$NET" | cut -d' ' -f1); fi
echo "network sha256: $SUM"
if [ -n "$NET_SHA256" ] && [ "$SUM" != "$NET_SHA256" ]; then
  echo "sha256 mismatch: expected $NET_SHA256" >&2
  exit 1
fi
if [ ! -f models/twin-s-clock-fp32.onnx ]; then
  "$PY" -I scripts/convert_onnx_fp32.py "$NET" models/twin-s-clock-fp32.onnx
fi
echo "ok: $DEST/engine/build-ninja/hivemind with $DEST/engine/models/twin-s-clock-fp32.onnx"
