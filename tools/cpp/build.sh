#!/bin/sh
# Builds tools/cpp/policy_dump against the engine's release build (build-ninja).
set -e
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
E="$ROOT/third_party/hivemind/engine"
ORT="${ONNXRuntime_ROOT:-$(brew --prefix onnxruntime)}"
clang++ -std=gnu++2b -Ofast -fno-finite-math-only -march=native -flto \
  -DHIVEMIND_BACKEND_ONNXRUNTIME -DNDEBUG -DNNUE_EMBEDDING_OFF \
  -I"$E/src" -I"$ORT/include/onnxruntime" \
  "$ROOT/tools/cpp/policy_dump.cc" \
  "$E/build-ninja/libhivemind_lib.a" "$E/build-ninja/src/Fairy-Stockfish/libFairy-Stockfish.a" \
  -L"$ORT/lib" -lonnxruntime -Wl,-rpath,"$ORT/lib" \
  -o "$ROOT/tools/cpp/policy_dump"
