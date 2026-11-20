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
# Wave 2b (third_party/wave2-hivemind/docs/wave2_training.md, "Wave 2b"):
#   WIDE=1    the lineage starts from it13 widened (twin-s-clock-wide: 26 more input planes per board, arrivals head,
#             zero-initialised, so it plays exactly as it13 at the start); init runs `hivemind.wave2 init --wide`
#             (widening it13 with its parity check when the files are missing), rl-loop runs with --wide (seat timing
#             and arrivals labels into the chunks, arrivals loss at ARRIVALS_WEIGHT). The fixed baseline stays it13.
#   AB=1      the A/B: two lineages under runs/wave2/<RUN>/narrow and .../wide, the same settings and the same
#             self-play seeds (rl-loop seeds segment k of iteration n with 1000 n + k), each gated vs it13 at equal
#             clock. With WIDE unset this job runs both, one iteration each in turn (narrow first), so a cut-off job
#             leaves them at most one iteration apart; WIDE=0 or WIDE=1 runs that arm alone (two jobs, two GPUs:
#             half the wall clock). <RUN>/<arm>/results.tsv per lineage and <RUN>/results_ab.tsv side by side.
#
#   RUN=w2a ITERS=2 sbatch cluster/wave2_iterate.sbatch
#   RUN=w2a ITERS=2 sbatch cluster/wave2_iterate.sbatch                     # resubmit: continues with the next two
#   RUN=w2-nogate ITERS=1 WAVE2_EXTRA="--sit-gate true" sbatch cluster/wave2_iterate.sbatch   # an ablation lineage
#   RUN=w2b-wide WIDE=1 ITERS=2 sbatch cluster/wave2_iterate.sbatch         # a wide lineage on its own
#   RUN=ab1 AB=1 ITERS=3 sbatch cluster/wave2_iterate.sbatch                # the A/B, both arms in one job
#   RUN=ab1 AB=1 WIDE=0 ITERS=3 sbatch cluster/wave2_iterate.sbatch; \
#   RUN=ab1 AB=1 WIDE=1 ITERS=3 sbatch cluster/wave2_iterate.sbatch         # the A/B as two parallel jobs
#   RUN=smoke ITERS=1 GAMES_PER_ITER=4 SEGMENT_GAMES=4 VAL_GAMES=2 MATCH_GAMES=2 PARALLEL_GAMES=2 BATCH_SIZE=256 \
#     NO_GATE=1 bash cluster/wave2_iterate.sbatch                           # tiny check (ALLOW_STOCK=1 before the
#                                                                           # wave 2 self-play options exist)
#
# Settings: RUN (required), ITERS (1), GAMES_PER_ITER (20000), SEGMENT_GAMES (2000), VAL_GAMES (500),
# PARALLEL_GAMES (6 per GPU, the cores), WINDOW (3 replayed iterations), NEW_PASSES (1.5), REPLAY_PASSES (0.5),
# BATCH_SIZE (1024), BASE_MIX (empty = the preset's 60/120/180/300 s at 0.25/0.25/0.35/0.15; MS:W,...),
# WAVE2_EXTRA (further rl-loop wave 2 flags, e.g. "--instant-transfer true"), LR (1e-4, heads), TRUNK_LR (5e-5),
# PROMOTE_ABOVE (0.5: a tie keeps the generator), MATCH_GAMES (400, colour-swapped pairs), MATCH_MS (100),
# NO_GATE (0; 1 = no matches, smoke only), LINEAGE (twin-s-clock-w2, twin-s-clock-w2w with WIDE=1), ALLOW_STOCK (0),
# W2 (third_party/wave2-hivemind), WIDE (0), AB (0), ARRIVALS_WEIGHT (0.05).
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(git rev-parse --show-toplevel)}"
[ -n "${SLURM_JOB_ID:-}" ] && source cluster/env.sh

RUN="${RUN:?set RUN, the loop folder under runs/wave2/}"
REPO="$PWD"
BASE="$REPO/runs/wave2/$RUN"
ROOT="$BASE"
AB="${AB:-0}"
WIDE_SET="${WIDE:-}"
WIDE="${WIDE:-0}"
ARRIVALS_WEIGHT="${ARRIVALS_WEIGHT:-0.05}"
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
LINEAGE_SET="${LINEAGE:-}"
ALLOW_STOCK="${ALLOW_STOCK:-0}"
JOBS="${CMAKE_BUILD_PARALLEL_LEVEL:-2}"

die() { echo "wave2_iterate: $*" >&2; exit 1; }
done_step() { echo "${2:-}" > "$ROOT/.done_$1"; }
is_done() { [ -f "$ROOT/.done_$1" ]; }
case "$WIDE" in 0|1) ;; *) die "WIDE is 0 or 1" ;; esac
case "$AB" in 0|1) ;; *) die "AB is 0 or 1" ;; esac

[ -d "$W2/.git" ] || die "$W2 missing: run cluster/build.sh (it clones branch talisman/wave2 and builds it)"
mkdir -p "$ROOT"
MATCH_ENGINE="${MATCH_ENGINE:-$W2/engine/build-ninja/hivemind}"
SP_ENGINE="${SP_ENGINE:-$W2/engine/build-sp1/hivemind}"
PY="$W2/.venv/bin/python"
echo "commit $(git rev-parse HEAD)  wave2 $(git -C "$W2" rev-parse HEAD) ($(git -C "$W2" rev-parse --abbrev-ref HEAD))  pin $(cat cluster/wave2_engine.ref 2>/dev/null || echo none)"
echo "run $ROOT  iterations +$ITERS  games $GAMES_PER_ITER ($SEGMENT_GAMES per segment, val $VAL_GAMES)  window $WINDOW"
echo "train lr $LR trunk $TRUNK_LR  batch $BATCH_SIZE  passes $NEW_PASSES/$REPLAY_PASSES  promote above $PROMOTE_ABOVE"
echo "gate: $( [ "$NO_GATE" = 1 ] && echo off || echo "$MATCH_GAMES games at $MATCH_MS ms vs the generator and vs it13" )"
echo "wave 2: base mix ${BASE_MIX:-preset}  extra ${WAVE2_EXTRA:-none}  allow stock $ALLOW_STOCK"
echo "wave 2b: AB $AB  WIDE ${WIDE_SET:-unset}  arrivals weight $ARRIVALS_WEIGHT"

# ------------------------------------------------------------------ engines
if ! is_done engines; then
  [ -x "$MATCH_ENGINE" ] || die "match engine $MATCH_ENGINE missing: run cluster/build.sh (WAVE2=1, the default)"
  # The A/B arms start together on two GPUs: one builds build-sp1 while the other waits here, instead of testing a
  # half-linked binary (ab1_w1 15877463-65 died on that). The lock is released when this shell exits.
  exec 9>"$W2/engine/.build-sp1.lock"
  flock 9
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
    # Not `--help | grep -q`: grep exits at the first match, the engine gets SIGPIPE and pipefail fails the test.
    if ! grep -q -- "--durative-tree" <<<"$("$e" --help 2>&1)"; then
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

# One lineage: init, ITERS (or $1) iterations, results. ROOT, LINEAGE and ARM_WIDE say which.
run_arm() {
# ------------------------------------------------------------------ init
if ! is_done init; then
  INIT_WIDE=(); [ "$ARM_WIDE" = 1 ] && INIT_WIDE=(--wide)
  (cd "$W2" && "$PY" -m hivemind.wave2 init --name "$LINEAGE" --models "$ROOT/models" --artifacts "$ROOT/artifacts" \
     ${INIT_WIDE[@]+"${INIT_WIDE[@]}"}) \
    > "$ROOT/init.txt" || die "init failed (network download or SHA-256)"
  cat "$ROOT/init.txt"
  done_step init "$(grep sha256 "$ROOT/init.txt" | tr '\n' ' ')"
fi

# ------------------------------------------------------------------ loop
finished() { find "$ROOT/runs" -path '*/rl-it*/stages/iteration.done' 2>/dev/null | wc -l | tr -d ' '; }
FIRST=$(( $(finished) + 1 ))
LAST=$(( FIRST + ${1:-$ITERS} - 1 ))
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
  [ "$ARM_WIDE" = 1 ] && ARGS+=(--wide --arrivals-weight "$ARRIVALS_WEIGHT")
  # shellcheck disable=SC2086  # WAVE2_EXTRA is split on purpose
  (cd "$W2" && "$PY" -m hivemind rl-loop "${ARGS[@]}" $WAVE2_EXTRA) || die "rl-loop failed (see $ROOT/runs/rl-loop.log)"
  done_step "loop_$LAST"
fi

# ------------------------------------------------------------------ results
"$PY" - "$ROOT" <<'EOF'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
rows = ["iteration\tgames\tpositions\tflag_games\trepeat_records\tarrival_labels\tvs_generator\tvs_it13\telo_vs_it13\tpromoted"]
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
        sum(s.get("timing", {}).get("labelled", 0) for s in segs),
        gen.get("score", ""), fixed.get("score", ""), fixed.get("elo", ""), promote.get("promoted", ""))))
(root / "results.tsv").write_text("\n".join(rows) + "\n")
print("\n".join(rows))
EOF
}

arm_setup() {
  ARM_WIDE="$1"
  if [ "$AB" = 1 ]; then
    if [ "$ARM_WIDE" = 1 ]; then ROOT="$BASE/wide"; else ROOT="$BASE/narrow"; fi
  else
    ROOT="$BASE"
  fi
  if [ -n "$LINEAGE_SET" ]; then LINEAGE="$LINEAGE_SET"
  elif [ "$ARM_WIDE" = 1 ]; then LINEAGE=twin-s-clock-w2w
  else LINEAGE=twin-s-clock-w2; fi
  mkdir -p "$ROOT"
  echo; echo "==== lineage $LINEAGE in $ROOT (wide $ARM_WIDE)"
}

if [ "$AB" = 1 ] && [ -z "$WIDE_SET" ]; then
  # Both arms in one job: one iteration each in turn, narrow first.
  for _ in $(seq 1 "$ITERS"); do
    for arm in 0 1; do
      arm_setup "$arm"
      run_arm 1
    done
  done
else
  arm_setup "$WIDE"
  run_arm "$ITERS"
fi

if [ "$AB" = 1 ]; then
  "$PY" - "$BASE" <<'EOF'
import sys
from pathlib import Path
base = Path(sys.argv[1])
def table(arm):
    path = base / arm / "results.tsv"
    if not path.is_file():
        return {}
    lines = path.read_text().splitlines()
    header = lines[0].split("\t")
    return {row[0]: dict(zip(header, row)) for row in (line.split("\t") for line in lines[1:])}
narrow, wide = table("narrow"), table("wide")
rows = ["iteration\tnarrow_vs_it13\tnarrow_elo\tnarrow_promoted\twide_vs_it13\twide_elo\twide_promoted\twide_arrival_labels"]
for it in sorted(set(narrow) | set(wide), key=lambda name: int(name[5:])):
    n, w = narrow.get(it, {}), wide.get(it, {})
    rows.append("\t".join([it, n.get("vs_it13", ""), n.get("elo_vs_it13", ""), n.get("promoted", ""),
                           w.get("vs_it13", ""), w.get("elo_vs_it13", ""), w.get("promoted", ""),
                           w.get("arrival_labels", "")]))
(base / "results_ab.tsv").write_text("\n".join(rows) + "\n")
print("\n".join(rows))
EOF
fi
if [ "$AB" = 1 ]; then echo "wave2_iterate done: $BASE/results_ab.tsv, per arm $BASE/<arm>/results.tsv"
else echo "wave2_iterate done: $ROOT/results.tsv, log $ROOT/runs/rl-loop.log"; fi
