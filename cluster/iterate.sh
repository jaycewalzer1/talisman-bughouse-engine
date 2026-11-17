#!/bin/bash
#SBATCH -J iterate -A paceship-bughouse -q inferno
#SBATCH -N1 -n8 --mem=64G --gres=gpu:rtx_6000:1 -t 16:00:00
#SBATCH -o iterate_%x_%j.log
# Custom-engine training loop (docs/custom_engine_spec.md section 3), one or more iterations into runs/custom/<RUN>:
#   generate  arena.selfplay: out-of-step free-mode self-play with the current network -> <it>/selfplay/training_data
#   think     (optional, THINK_LABELS=1) tools/think_labels.py: think_gain labels by searching sampled positions at
#             THINK_NODES and 4x that -> <it>/selfplay/training_data_think, which convert then reads instead
#   convert   convert_selfplay_data.convert_to_split_parquet (game-level train/val split, replay from the last
#             iteration's chunks) -> <it>/rl_data
#   train     hivemind train_loop --mode rl from the current checkpoint -> <it>/workspace/artifacts/training/weights/rl
#   export    the trained ONNX converted to FP32 for the engine -> <it>/model-fp32.onnx (+ <it>/model.tar)
#   gate      arena.pool run: the new network (free mode, hivemind-clock) against stock Hivemind (stock binary,
#             stock rules, STOCK_NODES fixed budget, one machine), both teams jittered, held-out openings -> <it>/gate
# Each finished step leaves a marker (<it>/.done_<step>), so a rerun with the same RUN continues where it stopped.
# Every iteration trains from the previous one's checkpoint; the gate is a measurement, not a filter, unless
# GATE_MIN is set (the loop stops when a gate score falls below it).
#
#   RUN=main ITERS=3 sbatch cluster/iterate.sh
#   RUN=main START=4 ITERS=2 sbatch cluster/iterate.sh            # continue with it04, it05
#   RUN=smoke ITERS=1 GAMES=4 GATE_PAIRS=2 WORKERS=1 bash cluster/iterate.sh   # tiny local check (on the Mac:
#             only with the hivemind-clock build present, one worker, nothing else heavy running)
#
# Settings: MODEL0 / CKPT0 (start network, default engine/models/hivemind-clock-v2-fp32.onnx / hivemind-clock-v2.tar from
# tools/make_clock_net.py), GAMES (2000 per iteration), WORKERS (6 under Slurm, 1 elsewhere), K_RANGE, JITTER,
# NODES_RANGE, BASE_MS, ARCH (crossboard-risev33), BATCH_SIZE, STOCK_NODES (100: set it to stock's best fixed
# budget from docs/beat_stock.md), GATE_PAIRS (400 = 800 games), GATE_JITTER (0.5,2.0), GATE_MIN (unset),
# VAL_FRACTION (game-level validation share for convert; unset = the converter's default 0.05; a tiny smoke run
# with 2 games needs 0.5). Spec v2: SHARD_VERSION (6), THINK_HEAD (0; 1 = self-play and gate scale budgets by the
# network's think_gain), THINK_LABELS (0 = off), THINK_NODES (200; below ~200 the labels are mostly search noise), THINK_FRACTION (0.25),
# THINK_MAX (0 = no cap), THINK_REPEAT (0.05: share of labelled positions searched twice at N, so think_labels.json
# logs noise_rate next to the label rates).
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(git rev-parse --show-toplevel)}"
[ -n "${SLURM_JOB_ID:-}" ] && source cluster/env.sh

RUN="${RUN:?set RUN, the loop folder under runs/custom/}"
REPO="$PWD"
ROOT="runs/custom/$RUN"
PY="$REPO/.venv/bin/python"
HM="$REPO/third_party/hivemind"
MODELS="$HM/engine/models"
START="${START:-1}"
ITERS="${ITERS:-1}"
GAMES="${GAMES:-2000}"
if [ -n "${SLURM_JOB_ID:-}" ]; then WORKERS="${WORKERS:-6}"; else WORKERS="${WORKERS:-1}"; fi
K_RANGE="${K_RANGE:-60,300}"
JITTER="${JITTER:-0.5,2.0}"
NODES_RANGE="${NODES_RANGE:-16,1600}"
BASE_MS="${BASE_MS:-60000}"
ARCH="${ARCH:-crossboard-risev33}"
STOCK_NODES="${STOCK_NODES:-100}"
GATE_PAIRS="${GATE_PAIRS:-400}"
GATE_JITTER="${GATE_JITTER:-0.5,2.0}"
SHARD_VERSION="${SHARD_VERSION:-6}"
THINK_HEAD="${THINK_HEAD:-0}"
THINK_LABELS="${THINK_LABELS:-0}"
THINK_NODES="${THINK_NODES:-200}"
THINK_FRACTION="${THINK_FRACTION:-0.25}"
THINK_MAX="${THINK_MAX:-0}"
THINK_REPEAT="${THINK_REPEAT:-0.05}"
THINK_FLAG=""; THINK_KW=""
[ "$THINK_HEAD" = 1 ] && THINK_FLAG="--think-head" && THINK_KW=', "think_head": true'
mkdir -p "$ROOT"
RESULTS="$ROOT/results.tsv"
[ -f "$RESULTS" ] || printf 'iter\tgames\tsamples\tflag_rate\tsit_spells_per_search\tgate_games\tgate_score\tgate_lo\tgate_hi\tmodel\n' > "$RESULTS"

label() { printf 'it%02d' "$1"; }
# The chunks an iteration trains on: the think-labelled copy when that step ran, else the raw self-play chunks.
data_of() { if [ -f "$1/.done_think" ]; then echo "$1/selfplay/training_data_think"; else echo "$1/selfplay/training_data"; fi; }
done_step() { touch "$1/.done_$2"; }
is_done() { [ -f "$1/.done_$2" ]; }
die() { echo "iterate: $*" >&2; exit 1; }

# The network and checkpoint an iteration starts from: the previous iteration's, or MODEL0 / CKPT0.
start_of() {
  local prev="$ROOT/$(label $(($1 - 1)))"
  if [ "$1" -gt 1 ] && [ -f "$prev/model-fp32.onnx" ]; then
    echo "$prev/model-fp32.onnx $prev/model.tar"
  else
    echo "${MODEL0:-$MODELS/hivemind-clock-v2-fp32.onnx} ${CKPT0:-$MODELS/hivemind-clock-v2.tar}"
  fi
}

echo "commit $(git rev-parse HEAD)  loop $ROOT  iterations $START..$((START + ITERS - 1))  games $GAMES  workers $WORKERS"
echo "k $K_RANGE  jitter $JITTER  nodes $NODES_RANGE  clock $BASE_MS ms  gate: stock $STOCK_NODES nodes, $GATE_PAIRS pairs, jitter $GATE_JITTER"
echo "shards v$SHARD_VERSION  think head $THINK_HEAD  think labels $THINK_LABELS (N $THINK_NODES, fraction $THINK_FRACTION, cap $THINK_MAX, repeat $THINK_REPEAT)"
$PY -c "from arena.players.hivemind import resolve_binary as r; print('binaries', r('hivemind'), r('hivemind-clock'))"

for ((i = START; i < START + ITERS; i++)); do
  IT="$(label "$i")"
  DIR="$ROOT/$IT"
  mkdir -p "$DIR"
  read -r MODEL CKPT <<< "$(start_of "$i")"
  case "$MODEL" in /*) ;; *) MODEL="$REPO/$MODEL" ;; esac
  case "$CKPT" in /*) ;; *) CKPT="$REPO/$CKPT" ;; esac
  [ -f "$MODEL" ] || die "$IT: start network $MODEL missing (run tools/make_clock_net.py in the hivemind repo)"
  [ -f "$CKPT" ] || die "$IT: start checkpoint $CKPT missing"
  echo; echo "== $IT from $MODEL ($CKPT)"
  date

  # ------------------------------------------------------------------ generate
  if ! is_done "$DIR" generate || { [ "$THINK_LABELS" = 1 ] && ! is_done "$DIR" think; }; then
    # A new network on a new GPU type builds its TensorRT engine file on first start (~200 s), longer than the
    # 120 s a player waits for the engine: build it here with the long wait.
    $PY cluster/warm_engine.py --binary hivemind-clock --model "$MODEL" || die "$IT: engine warm-up failed"
  fi
  if ! is_done "$DIR" generate; then
    rm -rf "$DIR/selfplay/training_data"  # a cut-off run leaves partial chunks: start the iteration's data again
    $PY -m arena.selfplay --out "$DIR/selfplay" --model "$MODEL" --binary hivemind-clock --games "$GAMES" \
      --workers "$WORKERS" --seed "$i" --game-id-offset "$((i * 1000000))" --run-id "$RUN-$IT" \
      --k-range "$K_RANGE" --jitter "$JITTER" --nodes-range "$NODES_RANGE" --base-ms "$BASE_MS" --no-events \
      --shard-version "$SHARD_VERSION" $THINK_FLAG || die "$IT: self-play failed"
    done_step "$DIR" generate
  fi

  # ------------------------------------------------------------------ think labels (optional)
  if [ "$THINK_LABELS" = 1 ] && ! is_done "$DIR" think; then
    rm -rf "$DIR/selfplay/training_data_think"
    $PY tools/think_labels.py --in "$DIR/selfplay/training_data" --out "$DIR/selfplay/training_data_think" \
      --model "$MODEL" --binary hivemind-clock --nodes "$THINK_NODES" --fraction "$THINK_FRACTION" \
      --max-positions "$THINK_MAX" --repeat-fraction "$THINK_REPEAT" --seed "$i" || die "$IT: think labels failed"
    done_step "$DIR" think
  fi

  # ------------------------------------------------------------------ convert
  if ! is_done "$DIR" convert; then
    PREV_DATA=""
    [ "$i" -gt 1 ] && [ -d "$(data_of "$ROOT/$(label $((i - 1)))")" ] && \
      PREV_DATA="$(data_of "$ROOT/$(label $((i - 1)))")"
    rm -rf "$DIR/rl_data"
    # Resolve the chunk dir here: inside the subshell below the cwd is $HM, where data_of's relative marker test
    # would always miss .done_think and fall back to the unlabelled chunks.
    DATA="$(data_of "$DIR")"
    echo "convert reads $DATA"
    (cd "$HM" && "$PY" - "$REPO/$DIR" "${PREV_DATA:+$REPO/$PREV_DATA}" "${VAL_FRACTION:-}" "$REPO/$DATA" <<'EOF'
import sys
from pathlib import Path
from hivemind.data.convert_selfplay_data import convert_to_split_parquet
it = Path(sys.argv[1]).resolve()
replay = sys.argv[2] or None
kw = {"validation_fraction": float(sys.argv[3])} if sys.argv[3] else {}
train, val, n_train, n_val = convert_to_split_parquet(Path(sys.argv[4]).resolve(), it / "rl_data",
                                                      replay_input_dir=replay, **kw)
print(f"converted: train {train} ({n_train}), validation {val} ({n_val})")
EOF
    ) || die "$IT: conversion failed"
    done_step "$DIR" convert
  fi

  # ------------------------------------------------------------------ train
  WS="$REPO/$DIR/workspace"
  if ! is_done "$DIR" train; then
    mkdir -p "$WS"
    (cd "$HM" && HIVEMIND_WORKSPACE="$WS" "$PY" -m hivemind.training.train_loop --mode rl --architecture "$ARCH" \
      --checkpoint "$CKPT" --rl-data-dir "$REPO/$DIR/rl_data/train" --val-data-dir "$REPO/$DIR/rl_data/val" \
      --iteration "$IT" ${BATCH_SIZE:+--batch-size "$BATCH_SIZE"}) || die "$IT: training failed"
    done_step "$DIR" train
  fi

  # ------------------------------------------------------------------ export
  if ! is_done "$DIR" export; then
    W="$WS/artifacts/training/weights/rl"
    ONNX="$(ls -t "$W"/*.onnx 2>/dev/null | grep -v -- '-fp16' | head -1)"
    TAR="$(ls -t "$W"/*.tar 2>/dev/null | head -1)"
    [ -n "$ONNX" ] && [ -n "$TAR" ] || die "$IT: no trained ONNX/checkpoint under $W"
    "$PY" "$HM/engine/scripts/convert_onnx_fp32.py" "$ONNX" "$DIR/model-fp32.onnx" || die "$IT: FP32 export failed"
    cp "$TAR" "$DIR/model.tar"
    echo "exported $ONNX -> $DIR/model-fp32.onnx"
    done_step "$DIR" export
  fi

  # ------------------------------------------------------------------ gate
  if ! is_done "$DIR" gate; then
    rm -rf "$DIR/gate"
    NEW="$REPO/$DIR/model-fp32.onnx"
    $PY cluster/warm_engine.py --binary hivemind-clock --model "$NEW" && $PY cluster/warm_engine.py \
      || die "$IT: engine warm-up failed"
    $PY -m arena.pool run --setup centralized --opponent stock-centralized --opponent-nodes "$STOCK_NODES" \
      --contender-kw "{\"free\": true, \"binary\": \"hivemind-clock\", \"model\": \"$NEW\", \"k_range\": [${K_RANGE}]${THINK_KW}}" \
      --opponent-kw '{"serial": true}' --jitter "$GATE_JITTER" --openings eval --pairs "$GATE_PAIRS" \
      --workers "$WORKERS" --seed "$i" --base-ms "$BASE_MS" --out "$DIR/gate" || die "$IT: gate failed"
    done_step "$DIR" gate
  fi

  # ------------------------------------------------------------------ results
  ROW="$($PY - "$DIR" "$IT" <<'EOF'
import json, sys
d, it = sys.argv[1], sys.argv[2]
sp = json.load(open(f"{d}/selfplay/summary.json"))["results"]
g = json.load(open(f"{d}/gate/summary.json"))["results"]["by_opponent"]["stock-centralized"]
print("\t".join(map(str, [it, sp["games"], sp["samples"], sp["flag_rate"], sp["sit_spells_per_search"], g["games"],
                          g["score"], g["ci95"][0], g["ci95"][1], f"{d}/model-fp32.onnx"])))
EOF
  )" || die "$IT: could not read results"
  grep -q "^$IT	" "$RESULTS" || echo "$ROW" >> "$RESULTS"
  echo "$ROW"
  SCORE="$(echo "$ROW" | cut -f7)"
  if [ -n "${GATE_MIN:-}" ] && "$PY" -c "import sys; sys.exit(0 if float('$SCORE') < float('$GATE_MIN') else 1)"; then
    echo "gate score $SCORE is below GATE_MIN $GATE_MIN: stopping after $IT"
    break
  fi
done

echo; column -t -s $'\t' "$RESULTS"
