#!/bin/bash
#SBATCH -J reanalyse -A paceship-bughouse -q inferno
#SBATCH -N1 -n6 --mem=48G --gres=gpu:rtx_6000:1 -t 12:00:00
#SBATCH -o reanalyse_%j.log
# Deep search targets by reanalysis for one iteration of an existing run (docs/search_balance.md sections 6 and 8;
# third_party/wave2-hivemind/src/hivemind/cli/reanalyse.py). First use: the src = reanalyse A/B on hybrid1 it1.
#
#   RUN=hybrid1 IT=1 SEGMENTS=seg00 sbatch cluster/reanalyse.sbatch                 # 1/8 of seg00 at 4 x 800 nodes
#   RUN=hybrid1 IT=1 K=2 FRACTION=0.25 ARM=k2f4 sbatch cluster/reanalyse.sbatch      # another arm, all segments
#   RUN=hybrid1 IT=1 DRY_RUN=1 sbatch cluster/reanalyse.sbatch                       # counts, GPU-h estimate, commands
#   RUN=hybrid1 PROBE=1 MODEL=runs/full/hybrid1/models/twin-s-clock-w2.onnx sbatch cluster/reanalyse.sbatch
#                                                                                   # the frozen probe set instead
#
# The run is READ ONLY here: output goes to OUT (default runs/scratch/reanalyse_<RUN>_it<IT>, markers in OUT/stages),
# never into runs/full/<RUN>. Pass OUT=runs/full/<RUN>/runs/rl-it<IT>/search/deep only for a run that owns it
# (rl-loop --balance on). A rerun of the same command resumes: finished shards and segments are kept.
#
# Needs an engine with the reanalyse command and selfplay --record-state-output (spec section 6), built into its own
# directory (ENGINE, default build-balance-sp1) so build-sp1, which live runs use, is never relinked. The engine
# change must first be merged with its flags off and pass the byte-identical self-play test (spec section 9,
# deployment safety): hybrid1's chained jobs rebuild build-sp1 when the fork HEAD changes.
#
# hybrid1 it1 was played without --record-state-output, so it has no state dumps: REPLAY=1 (the default) first plays
# each requested segment again (same seed, games, generator and wave 2 options) with a state dump, into OUT; that
# costs the segment's self-play again (about 2.6 GPU-h per 2000-game segment). REPLAY_GAMES=N replays only N games.
# The replay's marker says whether its chunks are byte-identical to the original's; either way the deep targets and
# selected.tsv refer to the replay's records, which then are that arm's search data.
#
# Settings: RUN (required), IT (1), ARM (reanalyse), K (4), FRACTION (0.125), SEGMENTS (all finished), MAX_STATES,
# SHARD_STATES (4096), PARALLEL_ENGINES (6), PARALLEL_GAMES (6), REPLAY (1), REPLAY_GAMES, ENGINE, MODEL, OUT,
# DRY_RUN (0), PROBE (0), NODES (probe only, 12800), EXTRA (more reanalyse.py options).
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(git rev-parse --show-toplevel)}"
source cluster/env.sh

: "${RUN:?RUN is required (a run under runs/full)}"
IT="${IT:-1}"
ARM="${ARM:-reanalyse}"
K="${K:-4}"
FRACTION="${FRACTION:-0.125}"
W2=third_party/wave2-hivemind
ROOT="runs/full/$RUN"
ENGINE="${ENGINE:-$W2/engine/build-balance-sp1/hivemind}"
OUT="${OUT:-runs/scratch/reanalyse_${RUN}_it${IT}}"
PY="$W2/.venv/bin/python"
die() { echo "reanalyse.sbatch: $*" >&2; exit 1; }

[ -d "$ROOT/runs/rl-it$IT" ] || [ "${PROBE:-0}" = 1 ] || die "$ROOT/runs/rl-it$IT missing"
[ -x "$ENGINE" ] || die "engine $ENGINE missing: build the search-balance engine there (cmake --preset ninja-release \
-B build-balance-sp1 -DHIVEMIND_SEARCH_WORKERS=1 with the TensorRT options of cluster/build_sp1.sh), never into build-sp1"
"$ENGINE" --help 2>&1 | grep -q reanalyse || die "$ENGINE has no reanalyse command"
echo "commit $(git rev-parse HEAD)  wave2 $(git -C "$W2" rev-parse HEAD)  engine $ENGINE  host $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
date

COMMON=(--engine "$(realpath "$ENGINE")" --parallel-engines "${PARALLEL_ENGINES:-6}")
[ "${DRY_RUN:-0}" = 1 ] && COMMON+=(--dry-run)

if [ "${PROBE:-0}" = 1 ]; then
  : "${MODEL:?PROBE=1 needs MODEL (the current best network ONNX)}"
  "$PY" cluster/probe_deep_targets.py --run "$ROOT" --model "$MODEL" --nodes "${NODES:-12800}" \
    --out "$OUT/probe" "${COMMON[@]}" ${EXTRA:-}
  status=$?
else
  ARGS=(run --run "$(realpath "$ROOT")" --iteration "$IT" --arm "$ARM" -k "$K" -f "$FRACTION"
        --deep-root "$(realpath -m "$OUT")" --shard-states "${SHARD_STATES:-4096}"
        --parallel-games "${PARALLEL_GAMES:-6}")
  [ -n "${SEGMENTS:-}" ] && ARGS+=(--segments $SEGMENTS)
  [ -n "${MAX_STATES:-}" ] && ARGS+=(--max-states "$MAX_STATES")
  [ -n "${MODEL:-}" ] && ARGS+=(--model "$(realpath "$MODEL")")
  [ "${REPLAY:-1}" = 1 ] && ARGS+=(--replay-missing-states)
  [ -n "${REPLAY_GAMES:-}" ] && ARGS+=(--replay-games "$REPLAY_GAMES")
  (cd "$W2" && .venv/bin/python -m hivemind.cli.reanalyse "${ARGS[@]}" "${COMMON[@]}" ${EXTRA:-})
  status=$?
  [ "${DRY_RUN:-0}" = 1 ] || (cd "$W2" && .venv/bin/python -m hivemind.cli.reanalyse status \
    --run "$(realpath "../../$ROOT")" --iteration "$IT" --arm "$ARM" --deep-root "$(realpath -m "../../$OUT")")
fi
date
exit "$status"
