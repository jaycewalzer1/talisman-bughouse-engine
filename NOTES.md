# talisman-bughouse-engine — working notes

## Layout
- `third_party/hivemind` — submodule, pinned at `1d01d6d` (the commit cited in the proposal). Do not edit in place; fixes go on a branch/fork.
- `third_party/Fairy-Stockfish` — submodule at `9f778da` (upstream master, 2026-09-23). Note Hivemind also vendors its own trimmed Fairy-Stockfish copy under `engine/src/Fairy-Stockfish`.
- `.venv` — project Python env (3.13, macOS PyPI wheels; torch 2.14.1 with MPS). Upstream `uv.lock` pins CUDA torch and cannot resolve on macOS, so we don't use `uv run` inside the submodule.

## Setup (macOS, M4 Max)
```bash
brew install onnxruntime ninja
uv venv --python 3.13 .venv
uv pip install --python .venv/bin/python torch torchvision chess einops matplotlib onnx onnxruntime onnxscript onnxsim pandas polars pytest timm tensorboard rtpt numpy
uv pip install --python .venv/bin/python --no-deps -e third_party/hivemind

# network (stdlib-only downloader, SHA-256 verified)
python3.13 third_party/hivemind/tools/fetch_network.py
python3.13 third_party/hivemind/tools/fetch_network.py --variant checkpoint
.venv/bin/python third_party/hivemind/engine/scripts/convert_onnx_fp32.py \
  third_party/hivemind/engine/models/hivemind-it04-crossboard-risev33-loss1.556-p82.0.onnx \
  third_party/hivemind/engine/models/hivemind-fp32.onnx

# engine (CPU / ONNX Runtime backend)
cd third_party/hivemind/engine
cmake --preset ninja-release -DHIVEMIND_BACKEND=onnxruntime \
  -DONNXRuntime_ROOT=/opt/homebrew/opt/onnxruntime -DHIVEMIND_USE_CCACHE=OFF
cmake --build build-ninja
```

## Baseline status (2026-10-18)
- Network: `aminwoo/bughouse-rise-v3` @ `eac7b7a`, `hivemind-it04-crossboard-risev33-loss1.556-p82.0` (RISEv3.3, cross-board).
  Input `[B, 74, 8, 8]`; outputs `value`, `pi_a`/`pi_b` (4672 each, one per board), `wdl_out`, `moves_left`. ONNX 27 MB, checkpoint 111 MB. **14.18M params**: conv body 11.75M, policy heads 0.92M, cross-board attention (2 layers, dim 192, 6 heads, 8 memory tokens) 0.59M, value head 0.54M, value fusion 0.15M. Checkpoint at training_iteration 13315 (RL iteration it04); includes optimizer state.
- CPU inference (ORT, FP32, batch 8): ~48 ms/batch, ~165 pos/s.
- Python tests: 85 pass, 1 skip, 1 fail (`test_default_validation_fraction_is_ten_percent`: constant is 0.05, test expects 0.10 — stale test).
- C++ tests (upstream flags, macOS/Clang): 283/296; 13 MCTS-selection failures, 2 segfaults; engine starved (24 nodes/3 s, 3.3M rejected selections).
- **Root cause: `-Ofast` ⇒ `-ffinite-math-only`.** Selection seeds `bestScore = -infinity` (node.cc) and the solver uses infinity sentinels; Clang folds those comparisons, so selection returns nullptr.
  Evidence: Linux GCC 14 (Docker, Ubuntu 24.04 arm64) passes 294/296 with the same flags; Mac at `-O2` passes (UBSan clean); Mac `-Ofast -fno-finite-math-only` passes 294/296.
  Fix: commit `d80c337` on branch `talisman/baseline-fixes` in the hivemind submodule — `-fno-finite-math-only` + `#error` guard on `__FINITE_MATH_ONLY__` in search_params.h. **Local-only commit**: push to a fork before cloning this repo elsewhere (PACE).
  After fix: 657 nodes/3 s (~213 nps, network-bound), 16 rejected selections.
  GCC passing is luck, not correctness — keep the flag on all platforms (PACE too).
- The 2 stale tests (PW exponent retune; FPU-dependent tie-break) are fixed in `d80c337`. **296/296 on macOS Clang 17 and Linux GCC 14.**
- Shell gotcha: zsh does not word-split `$f`; use `${=f}` when looping over flag sets.
- macOS ASan is unusable here (runtime hangs in InitializeShadowMemory before main); use UBSan or Linux ASan in Docker.
- Linux test image: `docker/Dockerfile.linux-test` (ubuntu:24.04 + g++-14 + ORT 1.23.2 aarch64); build: `docker build -t talisman-linux -f docker/Dockerfile.linux-test docker`.

## Policy-index parity (2026-10-18)
Tools: `tools/cpp/build.sh` → `tools/cpp/policy_dump` (engine indices per legal move);
`.venv/bin/python tools/policy_parity.py --random 2000` (diff vs training labels);
`.venv/bin/python tools/probe_policy_conventions.py` (which index the net learned). Run from repo root.

Parity over 2008 positions (~107k legal moves): normal, drops, en passant, N/Q promotions all match. Mismatches, 100% of cases:
- **Castling**: Python labels `e1g1`/`e1c1` (1348/3140); engine `get_fast_policy_index` uses Fairy's raw king→rook to_sq = `e1h1`/`e1a1` (1412/3268).
- **Pass**: Python `label_to_index['pass']` = 63 (make_map has 64 duplicate 'pass' slots; dict keeps last); engine = 0.
- R/B underpromotions: absent from both (by design; no policy planes).

Training data is internally inconsistent: supervised labels (generate_planes, Python) use 63/e1g1; self-play targets (selfplay.cc marginal_policy) use 0/e1h1.
**The shipped it04 net learned the engine convention**: pass@0 mean legal-rank 0.17 vs pass@63 1.00 (logit gap ≈ −11);
O-O@e1h1 0.59 vs @e1g1 0.97. O-O-O recheck on standard openings (`tools/probe_castling_openings.py`): engine index e1a1 ranks 44–51 of 47–52 legal moves; old Python slot e1c1 ranks 33–42. **The net never learned O-O-O at the index the engine reads → engine almost never castles long.** Supervised data put O-O-O on e1c1; self-play rarely castles long so never taught e1a1. Fixed labels only help if the net is retrained/fine-tuned. Frozen-evaluator experiments: symmetric handicap for both teams; document as limitation.
**Fixed in hivemind commit after d80c337 (see `git -C third_party/hivemind log`)**: `move2planes.policy_label` + `make_label_index`; parity now 100% on representable moves (R/B underpromotions reported as `unrepresented`, by design).
Note: `tests/data/test_generate_planes.py` checked pass at index 0 while data used 63, so its both-boards-pass check was vacuous before this fix.
⇒ Engine inference is consistent with this net. Fix belonged on the **Python side** (generate_planes + make_map consumers: infer_from_fen, search_training_fen, inspect_data_loader, analyze_training_data, verify_augmentation) → canonicalize to pass=0, castling=king→rook.

## Input-plane + value parity (2026-10-18)
Tool: `tools/cpp/planes_dump.cc` (replays moves on engine Board, dumps board_to_planes + C++ Engine outputs);
`.venv/bin/python tools/planes_parity.py --games 500` (replays the same games through BughouseBoard/board2planes, compares channels, values, top-1 legal move).
- C++ ORT engine vs Python ORT on identical planes: value Δ ≤ 2e-6, policy Δ ≤ 3e-5 → inference path is sound.
- Before fix: 277/505 positions differed in 3 channels/board: ep_square (python-chess sets after any double push; Fairy only if ep legal), last_to after castling (g1 vs h1), halfmove_clock (pawn drops reset in Fairy, not python-chess). End-to-end value Δ up to 0.17; 18/998 top-1 moves differed.
- Fixed in hivemind (board2planes.py, board.py) following the engine; 3005 positions (seed 1): 0 channel mismatches, 5981/5981 top-1 agree.
- **Not covered**: time-advantage *computation* (plane is fed from Python's clocks to both sides; the C++ clock→flag logic lives in UCI/search — check in arena work), board-swap/flip augmentation paths, FP16/TensorRT backend (PACE).

## Move generation: upstream Fairy-Stockfish vs Hivemind vendored copy (2026-10-18)
`.venv/bin/python tools/movegen_parity.py --games 1000` (needs `make -C third_party/Fairy-Stockfish/src build ARCH=apple-silicon`).
1982 board positions: 82,034 legal moves agree, 0 differ; 224 R/B underpromotions exist only upstream (Hivemind omits them by design).
Gotcha: upstream `go perft` runs on a search thread — readyok can precede its output; delimit on `Nodes searched`.

## Upstream reports
Drafts in `docs/upstream/` (not filed). Need fork links filled in.

## Week 1–2 TODO
- [x] Reproduce C++ test failures on Linux (Docker) to classify mac-only vs upstream
- [x] Root-cause selection starvation (finite-math) — fixed + compile-time guard (d80c337)
- [x] Fix/update the 2 stale tests
- [ ] Fork hivemind on GitHub, push `talisman/baseline-fixes`, repoint .gitmodules
- [ ] File upstream reports (drafts in docs/upstream/) — needs user OK
- [x] Count params / document architecture from checkpoint
- [x] Policy-index parity harness (castling, pass, drops, promotions, ep)
- [x] Input-plane + value parity (board2planes ↔ board_to_planes, Python ORT ↔ C++ Engine)
- [ ] Time-advantage flag: C++ clock→flag logic vs Python `time_advantage` (do with 4-clock runner)
- [ ] Re-run parity on PACE with TensorRT FP16 backend
- [x] Recheck O-O-O on standard openings — net never learned it at engine index (see above)
- [x] Python tests for pass/castling labels + fix Python side to engine convention
- [x] Stale `test_default_validation_fraction` (0.10→0.05 in 4614b55). Python suite: 91 pass, 1 skip
- [ ] Regenerate any supervised planes made with the old labels (none exist locally)
- [x] Fairy-Stockfish vs Hivemind legal-move diff (0 differences)
