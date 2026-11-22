#!/bin/bash
# Builds the one-worker wave 2 self-play engine (third_party/wave2-hivemind/engine/build-sp1, -DHIVEMIND_SEARCH_WORKERS=1;
# TensorRT under Slurm, ONNX Runtime on the Mac) when it is missing or was built from another wave2-hivemind commit
# than the checkout's HEAD. The commit it was built from goes to build-sp1/.talisman_commit. Before this stamp the
# engine was built only when missing, so a checkout moved to a new pin (cluster/wave2_engine.ref) kept self-playing on
# the old binary: a wave 2c option would then be refused, or a run would quietly measure the old engine.
#   bash cluster/build_sp1.sh [W2 dir, default third_party/wave2-hivemind]
# Called by cluster/wave2_iterate.sbatch (engines step) and cluster/wave2c_gpu_check.sbatch. Jobs that start together
# take turns on a lock, so one builds while the others wait instead of testing a half-linked binary (ab1_w1
# 15877463-65 died on that). Run with cluster/env.sh sourced on PACE (TensorRT_DIR, CUDA_TOOLKIT_ROOT_DIR).
set -uo pipefail
W2="${1:-third_party/wave2-hivemind}"
JOBS="${CMAKE_BUILD_PARALLEL_LEVEL:-2}"
SP="$W2/engine/build-sp1"
STAMP="$SP/.talisman_commit"
want=$(git -C "$W2" rev-parse HEAD) || { echo "build_sp1: $W2 is not a git checkout" >&2; exit 1; }

exec 9>"$W2/engine/.build-sp1.lock"
command -v flock >/dev/null && flock 9   # PACE has flock; the Mac does not (one job at a time there)
if [ -x "$SP/hivemind" ] && [ "$(cat "$STAMP" 2>/dev/null)" = "$want" ]; then
  echo "build-sp1 up to date at $want"
  exit 0
fi
# Built from another commit with the same engine/ tree (a Python-only change, e.g. rl-loop): the binary is the same,
# so restamp instead of relinking a binary a running run may be using.
built=$(cat "$STAMP" 2>/dev/null || true)
if [ -x "$SP/hivemind" ] && [ -n "$built" ] && git -C "$W2" diff --quiet "$built" "$want" -- engine 2>/dev/null; then
  echo "build-sp1 built at $built has the same engine/ as $want: restamped, no rebuild"
  echo "$want" > "$STAMP"
  exit 0
fi
echo "== building the one-worker self-play engine at $want ($JOBS jobs; was $(cat "$STAMP" 2>/dev/null || echo 'unstamped or missing'))"
if [ -n "${SLURM_JOB_ID:-}" ]; then
  BACKEND=(-DHIVEMIND_BACKEND=tensorrt -DTensorRT_DIR="$TensorRT_DIR" -DCUDA_TOOLKIT_ROOT_DIR="$CUDA_TOOLKIT_ROOT_DIR"
           -DHIVEMIND_NATIVE_ARCH=OFF)
else
  BACKEND=(-DHIVEMIND_BACKEND=onnxruntime -DONNXRuntime_ROOT="${ONNXRuntime_ROOT:-/opt/homebrew/opt/onnxruntime}")
fi
(cd "$W2/engine" && cmake --preset ninja-release -B build-sp1 -DHIVEMIND_SEARCH_WORKERS=1 \
   -DHIVEMIND_USE_CCACHE=OFF "${BACKEND[@]}" && cmake --build build-sp1 --target hivemind -j "$JOBS") \
  || { echo "build_sp1: self-play engine build failed" >&2; exit 1; }
echo "$want" > "$STAMP"
