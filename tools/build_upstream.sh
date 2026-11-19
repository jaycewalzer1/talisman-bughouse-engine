#!/bin/sh
# Build upstream aminwoo/hivemind (the v3 clock model, "upstream-v3" in the arena) next to our fork and fetch its
# latest promoted twin-s-clock network. Our submodule third_party/hivemind is not touched.
#
#   sh tools/build_upstream.sh            # clone/checkout TAG, build engine, fetch + convert the network
#   TAG=v3.0.3 JOBS=2 sh tools/build_upstream.sh
#   BACKEND=tensorrt sh tools/build_upstream.sh   # on PACE (after source cluster/env.sh: TensorRT_DIR, CUDA root)
#   WAVE2=1 sh tools/build_upstream.sh            # the wave 2 engine (spec v4 durative tree on upstream v3.0.3):
#                                                 # branch talisman/wave2 of github.com/jaycewalzer1/hivemind, full
#                                                 # clone, into third_party/wave2-hivemind (arena kind "wave2")
#   REPO=<git url> REF=<tag or branch> DEST=<dir> sh tools/build_upstream.sh   # any other ref
#   PIN=<commit> ...                              # build exactly this commit of REF (detached); fails if REPO
#                                                 # does not have it (cluster/build.sh pins wave 2 to
#                                                 # cluster/wave2_engine.ref)
#   TESTS=1 ...                                   # also build hivemind_tests and run ctest
#
# A tag is cloned --depth 1 (stock upstream). A branch is cloned in full (FULL=1 forces that for a tag too); an
# existing checkout that is on that branch is fast-forwarded from REPO (local commits are never discarded; a checkout
# ahead of REPO stays as it is: this is how the Mac builds its working copy of talisman/wave2). A checkout that has
# diverged from REPO is an error (ALLOW_STALE=1 builds it as it is, with a warning).
#
# Result:
#   third_party/upstream-hivemind/engine/build-ninja/hivemind          (ONNX Runtime CPU backend, -O3 release preset)
#   third_party/upstream-hivemind/engine/models/twin-s-clock.onnx      (HF main, SHA-256 checked below)
#   third_party/upstream-hivemind/engine/models/twin-s-clock-fp32.onnx (FP32 for ORT CPU: it has no FP16 kernels)
# The arena player is arena/players/upstream.py; docs/upstream/v3_arm.md records versions and checks.
set -eu

ROOT=$(cd "$(dirname "$0")/.." && pwd)
if [ "${WAVE2:-0}" = 1 ]; then
  REPO=${REPO:-https://github.com/jaycewalzer1/hivemind}
  REF=${REF:-talisman/wave2}
  DEST=${DEST:-$ROOT/third_party/wave2-hivemind}
fi
REPO=${REPO:-https://github.com/aminwoo/hivemind}
REF=${REF:-${TAG:-v3.0.3}}         # TAG kept for old callers; v3.0.3 = latest v3.0.x tag on 2026-11-17
DEST=${DEST:-$ROOT/third_party/upstream-hivemind}
TESTS=${TESTS:-0}
JOBS=${JOBS:-2}                    # several agents/builds share this Mac
ORT_ROOT=${ONNXRuntime_ROOT:-/opt/homebrew/opt/onnxruntime}
BACKEND=${BACKEND:-onnxruntime}    # onnxruntime (Mac, CPU) or tensorrt (PACE GPU nodes)
PY=${PY:-$ROOT/.venv/bin/python}
# twin-s-clock iteration 13 ("latest promoted network" per the model card) at HF revision edaa246 (2026-10-26).
HF_REPO=aminwoo/bughouse-twin-s-clock
HF_REV=${HF_REV:-edaa246cdb86c43bb3f18f0836501912c080e7f1}
NET_SHA256=${NET_SHA256:-f4ac25e32a993776f26b42d79cc5c1e48b77a16a13d1e2a2db30528f43c9af73}  # = twin-s-clock-it13.onnx

IS_TAG=0
if git ls-remote --exit-code --tags "$REPO" "refs/tags/$REF" >/dev/null 2>&1; then IS_TAG=1; fi
FULL=${FULL:-$((1 - IS_TAG))}
if [ ! -d "$DEST/.git" ]; then
  if [ "$FULL" = 1 ]; then
    git clone --branch "$REF" "$REPO" "$DEST"
  else
    git clone --depth 1 --branch "$REF" "$REPO" "$DEST"
  fi
elif [ "$IS_TAG" = 1 ]; then
  git -C "$DEST" fetch --depth 1 "$REPO" "refs/tags/$REF:refs/tags/$REF"
  git -C "$DEST" -c advice.detachedHead=false checkout "$REF"
elif [ "$(git -C "$DEST" symbolic-ref -q --short HEAD || true)" = "$REF" ]; then
  if ! git -C "$DEST" pull --ff-only "$REPO" "$REF"; then
    if [ "${ALLOW_STALE:-0}" = 1 ]; then
      echo "warning: $DEST not fast-forwarded from $REPO $REF; building it as it is (ALLOW_STALE=1)"
    else
      echo "$DEST could not be fast-forwarded from $REPO $REF (diverged or unreachable); ALLOW_STALE=1 builds it as it is" >&2
      exit 1
    fi
  fi
elif [ -z "$(git -C "$DEST" status --porcelain --untracked-files=no)" ]; then
  git -C "$DEST" fetch "$REPO" "$REF"
  git -C "$DEST" checkout -B "$REF" FETCH_HEAD
else
  echo "$DEST has uncommitted changes and is not on $REF: commit or move them first" >&2
  exit 1
fi
if [ -n "${PIN:-}" ]; then
  git -C "$DEST" cat-file -e "$PIN^{commit}" 2>/dev/null || git -C "$DEST" fetch "$REPO" "$REF" || true
  if ! git -C "$DEST" cat-file -e "$PIN^{commit}" 2>/dev/null; then
    echo "pinned commit $PIN is not in $REPO $REF: push it (git -C $DEST push fork $REF) or change the pin" >&2
    exit 1
  fi
  git -C "$DEST" -c advice.detachedHead=false checkout --detach "$PIN"
fi
echo "engine: $REPO $REF -> $(git -C "$DEST" describe --tags --always) $(git -C "$DEST" rev-parse HEAD)${PIN:+ (pinned)}"

cd "$DEST/engine"
if [ "$BACKEND" = tensorrt ]; then
  cmake --preset ninja-release -DHIVEMIND_BACKEND=tensorrt -DTensorRT_DIR="$TensorRT_DIR" \
    -DCUDA_TOOLKIT_ROOT_DIR="$CUDA_TOOLKIT_ROOT_DIR" -DHIVEMIND_USE_CCACHE=OFF -DHIVEMIND_NATIVE_ARCH=OFF
else
  cmake --preset ninja-release -DHIVEMIND_BACKEND=onnxruntime -DONNXRuntime_ROOT="$ORT_ROOT" \
    -DHIVEMIND_USE_CCACHE=OFF
fi
cmake --build build-ninja --target hivemind -j "$JOBS"
if [ "$TESTS" = 1 ]; then
  cmake --build build-ninja --target hivemind_tests -j "$JOBS"
  # Stopwatch tests (a search must return within a wall deadline) miss on loaded or slow nodes: three tries each.
  ctest --test-dir build-ninja --output-on-failure --repeat until-pass:3
fi

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
