#!/bin/bash
#SBATCH -J wave2 -A paceship-bughouse -q inferno
#SBATCH -N1 -n6 --mem=48G --gres=gpu:rtx_6000:1 -t 16:00:00
#SBATCH -o wave2_%x_%j.log
# Wave 2 training loop (docs/custom_engine_spec_v4_durative.md; third_party/wave2-hivemind/docs/wave2_training.md):
# twin-s-clock it13 continued by upstream's own RL loop (hivemind rl-loop --wave2) on four real seat clocks, the
# durative tree, flag loss and no hard sit gate, into runs/wave2/<RUN>. Steps, each leaving <RUN>/.done_<step>:
#   engines   the wave 2 match engine (build-ninja, from cluster/build.sh) must exist; the one-worker self-play engine
#             (build-sp1, -DHIVEMIND_SEARCH_WORKERS=1, TensorRT) is built here when missing. Both must list the wave 2
#             options in --help (ALLOW_STOCK=1 plays stock v3 games instead: a plumbing check, not wave 2)
#   env       the wave2-hivemind Python env (uv sync, or venv + pip without uv)
#   init      python -m hivemind.wave2 init: it13 ONNX and training checkpoint, SHA-256 checked, as this RUN's lineage
#             (<RUN>/models/<LINEAGE>.onnx generator, <LINEAGE>-it0.onnx fixed baseline = it13,
#             <RUN>/artifacts/<LINEAGE>/best.pt); later promotions replace the generator, never the it0 copy
#   loop      hivemind rl-loop --wave2 for ITERS more iterations: self-play (GAMES_PER_ITER), validation, label check,
#             train from the generator (trunk TRUNK_LR, heads LR), match vs the generator and vs it13 at equal clock
#             (MATCH_GAMES at MATCH_MS per move, both under the wave 2 rules), promote only above PROMOTE_ABOVE.
#             rl-loop keeps its own marker per stage (<RUN>/runs/rl-itN/stages), so a cut-off job resumes mid-
#             iteration; this step's marker is per job (.done_loop_<last iteration>)
#   results   <RUN>/results.tsv: one row per finished iteration from the stage markers
#
#   RUN=w2a ITERS=2 sbatch cluster/wave2_iterate.sbatch
#   RUN=w2a ITERS=2 sbatch cluster/wave2_iterate.sbatch                     # resubmit: continues with the next two
#   RUN=w2-nogate ITERS=1 WAVE2_EXTRA="--sit-gate true" sbatch cluster/wave2_iterate.sbatch   # an ablation lineage
#   RUN=smoke ITERS=1 GAMES_PER_ITER=4 SEGMENT_GAMES=4 VAL_GAMES=2 MATCH_GAMES=2 PARALLEL_GAMES=2 BATCH_SIZE=1024 \
#     NO_GATE=1 bash cluster/wave2_iterate.sbatch                           # tiny check (ALLOW_STOCK=1 before the
#                                                                           # wave 2 self-play options exist)
#
# Settings: RUN (required), ITERS (1), GAMES_PER_ITER (20000), SEGMENT_GAMES (2000), VAL_GAMES (500),
# PARALLEL_GAMES (6 per GPU, the cores), WINDOW (3 replayed iterations), NEW_PASSES (1.5), REPLAY_PASSES (0.5),
# BATCH_SIZE (1024), BASE_MIX (empty = the preset's 60/120/180/300 s at 0.25/0.25/0.35/0.15; MS:W,...),
# WAVE2_EXTRA (further rl-loop wave 2 flags, e.g. "--instant-transfer true"), LR (1e-4, heads), TRUNK_LR (5e-5),
# PROMOTE_ABOVE (0.5: a tie keeps the generator), MATCH_GAMES (400, colour-swapped pairs), MATCH_MS (100),
# NO_GATE (0; 1 = no matches, smoke only), LINEAGE (twin-s-clock-w2), ALLOW_STOCK (0), W2 (third_party/wave2-hivemind).
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(git rev-parse --show-toplevel)}"
[ -n "${SLURM_JOB_ID:-}" ] && source cluster/env.sh

RUN="${RUN:?set RUN, the loop folder under runs/wave2/}"
REPO="$PWD"
ROOT="$REPO/runs/wave2/$RUN"
W2="${W2:-$REPO/third_party/wave2-hivemind}"
case "$W2" in /*) ;; *) W2="$REPO/$W2" ;; esac
ITERS="${ITERS:-1}"
GAMES_PER_ITER="${GAMES_PER_ITER:-20000}"
SEGMENT_GAMES="${SEGMENT_GAMES:-2000}"
VAL_GAMES="${VAL_GAMES:-500}"
PARALLEL_GAMES="${PARALLEL_GAMES:-6}"
WINDOW="${WINDOW:-3}"
NEW_PASSES="${NEW_PASSES:-1.5}"
REPLAY_PASSES="${REPLAY_PASSES:-0.5}"
BATCH_SIZE="${BATCH_SIZE:-1024}"
BASE_MIX="${BASE_MIX:-}"
WAVE2_EXTRA="${WAVE2_EXTRA:-}"
LR="${LR:-1e-4}"
TRUNK_LR="${TRUNK_LR:-5e-5}"
PROMOTE_ABOVE="${PROMOTE_ABOVE:-0.5}"
MATCH_GAMES="${MATCH_GAMES:-400}"
MATCH_MS="${MATCH_MS:-100}"
NO_GATE="${NO_GATE:-0}"
LINEAGE="${LINEAGE:-twin-s-clock-w2}"
ALLOW_STOCK="${ALLOW_STOCK:-0}"
JOBS="${CMAKE_BUILD_PARALLEL_LEVEL:-2}"

die() { echo "wave2_iterate: $*" >&2; exit 1; }
done_step() { echo "${2:-}" > "$ROOT/.done_$1"; }
is_done() { [ -f "$ROOT/.done_$1" ]; }

[ -d "$W2/.git" ] || die "$W2 missing: run cluster/build.sh (it clones branch talisman/wave2 and builds it)"
mkdir -p "$ROOT"
MATCH_ENGINE="$W2/engine/build-ninja/hivemind"
SP_ENGINE="$W2/engine/build-sp1/hivemind"
PY="$W2/.venv/bin/python"
echo "commit $(git rev-parse HEAD)  wave2 $(git -C "$W2" rev-parse --short HEAD) ($(git -C "$W2" rev-parse --abbrev-ref HEAD))"
echo "run $ROOT  iterations +$ITERS  games $GAMES_PER_ITER ($SEGMENT_GAMES per segment, val $VAL_GAMES)  window $WINDOW"
echo "train lr $LR trunk $TRUNK_LR  batch $BATCH_SIZE  passes $NEW_PASSES/$REPLAY_PASSES  promote above $PROMOTE_ABOVE"
echo "gate: $( [ "$NO_GATE" = 1 ] && echo off || echo "$MATCH_GAMES games at $MATCH_MS ms vs the generator and vs it13" )"
echo "wave 2: base mix ${BASE_MIX:-preset}  extra ${WAVE2_EXTRA:-none}  allow stock $ALLOW_STOCK"

# ------------------------------------------------------------------ engines
if ! is_done engines; then
  [ -x "$MATCH_ENGINE" ] || die "match engine $MATCH_ENGINE missing: run cluster/build.sh (WAVE2=1, the default)"
  if [ ! -x "$SP_ENGINE" ]; then
    echo "== building the one-worker self-play engine ($JOBS jobs)"
    if [ -n "${SLURM_JOB_ID:-}" ]; then
      BACKEND=(-DHIVEMIND_BACKEND=tensorrt -DTensorRT_DIR="$TensorRT_DIR" -DCUDA_TOOLKIT_ROOT_DIR="$CUDA_TOOLKIT_ROOT_DIR"
               -DHIVEMIND_NATIVE_ARCH=OFF)
    else
      BACKEND=(-DHIVEMIND_BACKEND=onnxruntime -DONNXRuntime_ROOT="${ONNXRuntime_ROOT:-/opt/homebrew/opt/onnxruntime}")
    fi
    (cd "$W2/engine" && cmake --preset ninja-release -B build-sp1 -DHIVEMIND_SEARCH_WORKERS=1 \
       -DHIVEMIND_USE_CCACHE=OFF "${BACKEND[@]}" && cmake --build build-sp1 --target hivemind -j "$JOBS") \
      || die "self-play engine build failed"
  fi
  for e in "$SP_ENGINE" "$MATCH_ENGINE"; do
    if ! "$e" --help 2>&1 | grep -q -- "--durative-tree"; then
      [ "$ALLOW_STOCK" = 1 ] || die "$e lists no wave 2 self-play options (--durative-tree ...): rebuild talisman/wave2 \
with engine part 2, or ALLOW_STOCK=1 for a plumbing check on stock games"
      echo "warning: $e has no wave 2 self-play options: ALLOW_STOCK=1, the games are stock v3"
    fi
  done
  done_step engines "$(git -C "$W2" rev-parse HEAD)"
fi

# ------------------------------------------------------------------ env
if ! is_done env; then
  if command -v uv >/dev/null 2>&1; then
    (cd "$W2" && uv sync) || die "uv sync failed"
  else
    [ -x "$PY" ] || "${PYTHON:-python3}" -m venv "$W2/.venv" || die "venv failed"
    "$PY" -m pip install --quiet -e "$W2" || die "pip install failed"
  fi
  "$PY" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
  done_step env
fi

# ------------------------------------------------------------------ init
if ! is_done init; then
  (cd "$W2" && "$PY" -m hivemind.wave2 init --name "$LINEAGE" --models "$ROOT/models" --artifacts "$ROOT/artifacts") \
    > "$ROOT/init.txt" || die "init failed (network download or SHA-256)"
  cat "$ROOT/init.txt"
  done_step init "$(grep sha256 "$ROOT/init.txt" | tr '\n' ' ')"
fi

# ------------------------------------------------------------------ loop
finished() { find "$ROOT/runs" -path '*/rl-it*/stages/iteration.done' 2>/dev/null | wc -l | tr -d ' '; }
FIRST=$(( $(finished) + 1 ))
LAST=$(( FIRST + ITERS - 1 ))
if ! is_done "loop_$LAST"; then
  echo; echo "== rl-loop it$FIRST..it$LAST"; date
  ARGS=(--wave2 --no-upload --keep-selfplay --pause-for "" --last "$LAST" --runs "$ROOT/runs"
        --games "$GAMES_PER_ITER" --segment-games "$SEGMENT_GAMES" --val-games "$VAL_GAMES"
        --parallel-games "$PARALLEL_GAMES" --window "$WINDOW" --new-passes "$NEW_PASSES"
        --replay-passes "$REPLAY_PASSES" --batch-size "$BATCH_SIZE" --lr "$LR" --trunk-lr "$TRUNK_LR"
        --promote-above "$PROMOTE_ABOVE" --match-games "$MATCH_GAMES" --match-movetime "$MATCH_MS"
        --generator "$ROOT/models/$LINEAGE.onnx" --generator-checkpoint "$ROOT/artifacts/$LINEAGE/best.pt"
        --selfplay-engine "$SP_ENGINE" --match-engine "$MATCH_ENGINE" --openings "$W2/engine/openings.tsv")
  [ -n "$BASE_MIX" ] && ARGS+=(--base-mix "$BASE_MIX")
  [ "$NO_GATE" = 1 ] && ARGS+=(--no-gate)
  [ "$ALLOW_STOCK" = 1 ] && ARGS+=(--allow-stock-engine)
  # shellcheck disable=SC2086  # WAVE2_EXTRA is split on purpose
  (cd "$W2" && "$PY" -m hivemind rl-loop "${ARGS[@]}" $WAVE2_EXTRA) || die "rl-loop failed (see $ROOT/runs/rl-loop.log)"
  done_step "loop_$LAST"
fi

# ------------------------------------------------------------------ results
"$PY" - "$ROOT" <<'EOF'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
rows = ["iteration\tgames\tpositions\tflag_games\trepeat_records\tvs_generator\tvs_it13\telo_vs_it13\tpromoted"]
def load(path):
    return json.loads(path.read_text()) if path.is_file() else {}
for it in sorted((root / "runs").glob("rl-it*"), key=lambda p: int(p.name[5:])):
    stages = it / "stages"
    if not (stages / "iteration.done").is_file():
        continue
    segs = [load(p) for p in sorted(stages.glob("seg*.done"))]
    labels = [s.get("labels", {}) for s in segs]
    gen, fixed, promote = load(stages / "match_generator.done"), load(stages / "match_it0.done"), load(stages / "promote.done")
    rows.append("\t".join(str(v) for v in (
        it.name, sum(s.get("games", 0) for s in segs), sum(s.get("positions", 0) for s in segs),
        sum(l.get("flag_games", 0) for l in labels), sum(l.get("repeat_records", 0) for l in labels),
        gen.get("score", ""), fixed.get("score", ""), fixed.get("elo", ""), promote.get("promoted", ""))))
(root / "results.tsv").write_text("\n".join(rows) + "\n")
print("\n".join(rows))
EOF
echo "wave2_iterate done: $ROOT/results.tsv, log $ROOT/runs/rl-loop.log"
