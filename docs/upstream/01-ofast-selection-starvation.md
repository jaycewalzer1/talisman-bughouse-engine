# MCTS selection starves under -Ofast on Clang (macOS): 13 failing tests, ~8 nps

**Commit:** 1d01d6d · **Platform:** macOS 26 (arm64), Apple Clang 17, ONNX Runtime backend

## Symptom
- `hivemind_tests`: 283/296 pass; 13 NodeTest/EngineTest/MctsSolverTest selection tests fail and two segfault
  (`NodeTest.PendingEvaluationDivertsSelectionToAvailableSibling`, `NodeTest.BlockedBatchLeafDivertsSelectionToAvailableSibling`).
- `go movetime 3000` from startpos: 24 nodes, `rejected selection attempts 3325224 (same batch 3325220 ...)`.

## Cause
The release flags use `-Ofast`, which implies `-ffinite-math-only`. `Node::select_child_and_apply_virtual_loss`
seeds its argmax with `-std::numeric_limits<float>::infinity()` and the solver ranks proven losses as `+infinity`.
Under finite-math Clang may fold those comparisons, so selection returns no child. GCC 14 (Ubuntu 24.04) happens
to pass with the same flags, so Linux CI would not catch it.

## Evidence
| Build | Tests | startpos 3 s |
|---|---|---|
| Mac `-Ofast` | 283/296, 2 segfaults | 24 nodes |
| Mac `-O2` (+UBSan, no reports) | matches Linux | – |
| Mac `-Ofast -fno-finite-math-only` | 294/296 (= Linux) | 657 nodes, 16 rejected selections |

## Proposed fix
Add `-fno-finite-math-only` after `-Ofast` in `engine/CMakeLists.txt`, plus a guard so it cannot regress:
```cpp
#if defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__
#error "Search requires IEEE infinities: build with -fno-finite-math-only"
#endif
```
Two remaining failures are stale tests: `ProgressiveWideningScheduleIsExplicit` (PW_EXPONENT 0.4→0.3 in 4614b55)
and `AtomicVirtualLossDivertsNextSelection` (asserts FPU-dependent order instead of diversion).
Patch available: <fork link> (commit d80c337).
