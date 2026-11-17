#!/bin/bash
#SBATCH -J iterate -A paceship-bughouse -q inferno
#SBATCH -N1 -n8 --mem=64G --gres=gpu:rtx_6000:1 -t 16:00:00
#SBATCH -o iterate_%x_%j.log
# Custom-engine training loop (docs/custom_engine_spec.md section 3, spec v3), one or more iterations into
# runs/custom/<RUN>:
#   it00      (once per RUN, GATE_IT00=1) the gate rows for the start network itself (MODEL0, untrained), with
#             exactly the contender's settings: the baseline every later gate row is reported against
#   generate  arena.selfplay: out-of-step free-mode self-play with the current network -> <it>/selfplay/training_data
#   think     (optional, THINK_LABELS=1) tools/think_labels.py: think_gain labels by searching sampled positions at
#             THINK_NODES and 4x that -> <it>/selfplay/training_data_think, which convert then reads instead
#   convert   convert_selfplay_data.convert_to_split_parquet (game-level train/val split, replay from the last
#             iteration's chunks) -> <it>/rl_data
#   train     hivemind train_loop --mode rl from the current checkpoint -> <it>/workspace/artifacts/training/weights/rl
#   export    the trained ONNX converted to FP32 for the engine -> <it>/model-fp32.onnx (+ <it>/model.tar)
#   gate      arena.pool rows, the new network (free mode, hivemind-clock) against stock Hivemind (stock binary),
#             at equal clock (spec v3 contract 5, docs/beat_stock.md "Headline metric"): one row per GATE_BASES
#             entry (120 s and 180 s) -> <it>/gate_120, <it>/gate_180, plus the equal-nodes diagnostic row
#             -> <it>/gate_eqn_120. Contender unjittered; each arm charged the nodes it searched x its own
#             ms_per_node, measured on this GPU once per binary+network (<RUN>/ms_per_node.json); the opponent
#             under its own time manager (remaining clock / GATE_STOCK_K, default: the k that spends the same
#             mean clock as the contender's k range). Every row of a RUN plays the same schedule (GATE_SEED), so
#             the it00 comparison is paired; rows resume (--skip-done) when the job is cut off.
#   promote   (PROMOTE=1) the next iteration starts from the better of (this iteration's start, the new network)
#             by the gate rows: new score - start score, paired, averaged over GATE_BASES, must exceed
#             PROMOTE_MARGIN (default 0). PROMOTE=0: always continue from the new network (the old behaviour).
# Each finished step leaves a marker (<it>/.done_<step>), so a rerun with the same RUN continues where it stopped.
# GATE_MIN (unset) stops the loop when the headline gate score (the last GATE_BASES row) falls below it.
#
#   RUN=main ITERS=3 sbatch cluster/iterate.sh
#   RUN=main START=4 ITERS=2 sbatch cluster/iterate.sh            # continue with it04, it05
#   RUN=smoke ITERS=1 GAMES=4 GATE_PAIRS=2 WORKERS=1 bash cluster/iterate.sh   # tiny local check (on the Mac:
#             only with the hivemind-clock build present, one worker, nothing else heavy running)
#
# Settings: MODEL0 / CKPT0 (start network, default engine/models/hivemind-clock-v2-fp32.onnx / hivemind-clock-v2.tar from
# tools/make_clock_net.py), GAMES (2000 per iteration), WORKERS (6 under Slurm, 1 elsewhere), K_RANGE, JITTER,
# NODES_RANGE, BASE_MS, ARCH (crossboard-risev33), BATCH_SIZE, VAL_FRACTION (game-level validation share for
# convert; unset = the converter's default 0.05; a tiny smoke run with 2 games needs 0.5). Spec v2: SHARD_VERSION
# (6), THINK_HEAD (0; 1 = self-play and gate scale budgets by the network's think_gain), THINK_LABELS (0 = off),
# THINK_NODES (200; below ~200 the labels are mostly search noise), THINK_FRACTION (0.25), THINK_MAX (0 = no cap),
# THINK_REPEAT (0.05: share of labelled positions searched twice at N, so think_labels.json logs noise_rate).
#
# Gate (WP-A): GATE_BASES (120000,180000), GATE_PAIRS (400 = 800 games a row), GATE_SEED (0, the same for every
# row of a RUN), GATE_OPENINGS (eval), GATE_K_RANGE (= K_RANGE), GATE_STOCK_K (auto; 0 = fixed STOCK_NODES, 100),
# GATE_MEASURE (1 = measured per-arm ms_per_node; 0 = the player default), GATE_CHARGE (searched; "" = the
# player default), GATE_EQN (1 = the equal-nodes row at GATE_EQN_BASE, default the first GATE_BASES entry),
# GATE_IT00 (1), GATE_SPRT (off; e.g. 0,20 stops a row once the GSPRT over normalised Elo is decided, which
# unpairs later comparisons), GATE_OPP_JITTER (off; LO,HI jitters the opponent only), GATE_CONTENDER_KW and
# GATE_OPPONENT_KW (extra JSON members, e.g. '"max_nodes": 3200'), PROMOTE (1), PROMOTE_MARGIN (0),
# GATE_POOL / GATE_OPPONENT (arena/pools/phase1.json / stock-centralized; for the target engine
# GATE_POOL=arena/pools/upstream_v3.json GATE_OPPONENT=upstream-v3: any opponent other than stock-centralized
# plays with its pool options plus GATE_OPPONENT_KW only, under its own time manager).
#
# Self-play (WP-C) and convert/train (WP-D) options. Each is passed only when the program lists the flag in its
# --help (or, for convert, takes the keyword); a set option the program does not have is reported and skipped.
# Empty = the program's own default (for self-play: the REGIME's).
#   REGIME          --regime            self-play regime (v3 = spec v3: base-time mix 60/120/180/300 s at
#                                       0.25/0.25/0.35/0.15 (D2), think target, time odds, searched charge, pace,
#                                       noise, temperature, playout caps, bucket dedupe; v2 = the old regime). v3
#   BASE_MIX        --base-mix          e.g. 60000:0.25,120000:0.25,180000:0.35,300000:0.15 ('off' = BASE_MS)
#   TIME_ODDS       --time-odds         on/off
#   THINK_TARGET_MS --think-target-ms   LO,HI move-1 think time; k = base / target
#   SP_CHARGE       --charge            requested/searched
#   MIN_THINK_MS    --min-think-ms      think-time floor
#   PACE_MIX        --pace-mixture      on/off (replaces JITTER)
#   LAG_MS          --lag-ms            mean additive lag
#   EXPLORE_NOISE   --noise             root noise on recorded searches, on/off
#   MOVE_TEMP       --temperature       move temperature
#   TEMP_PLIES      --temperature-plies
#   FULL_SEARCH_P   --pcr-p             playout-cap share of full searches
#   PCR_NODES       --pcr-nodes
#   DEDUPE          --dedupe            position/bucket
#   SP_MS_PER_NODE  --ms-per-node       self-play cost model
#   SP_EXTRA        appended to the self-play command as is
#   Q_VALUE_RATIO   convert keyword q_value_ratio (value target = (1-r) outcome + r root_q)
#   WEIGHT_NODES_REF convert keyword weight_nodes_ref (sample weight min(1, nodes / N))
#   SPLIT_BY        convert keyword split_by (auto/game/opening)
#   REPLAY_ITERS    replay chunks from the last N iterations (1 = the previous one only, the old behaviour; the
#                   audit asks for 3)
#   CONVERT_KW      JSON object of further convert_to_split_parquet keywords
#   RECIPE          train_loop --recipe (auto/clock/stock; clock = two learning rates for trunk and new columns)
#   TRUNK_LR        train_loop --trunk-lr  (not a train_loop flag yet: reported and skipped)
#   HEAD_LR         train_loop --head-lr   (same)
#   DRIFT_SHARD     train_loop --drift-shard (frozen SL validation shard) and --drift-out <it>/drift.json
#   DRIFT_REFERENCE train_loop --drift-reference (default: the start checkpoint; use it04)
#   TRAIN_EXTRA     appended to the train_loop command as is
# A <it>/drift.json written by training (WP-D drift metric) is copied into results.tsv as one JSON column.
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
SHARD_VERSION="${SHARD_VERSION:-6}"
THINK_HEAD="${THINK_HEAD:-0}"
THINK_LABELS="${THINK_LABELS:-0}"
THINK_NODES="${THINK_NODES:-200}"
THINK_FRACTION="${THINK_FRACTION:-0.25}"
THINK_MAX="${THINK_MAX:-0}"
THINK_REPEAT="${THINK_REPEAT:-0.05}"
THINK_FLAG=""; THINK_KW=""
[ "$THINK_HEAD" = 1 ] && THINK_FLAG="--think-head" && THINK_KW=', "think_head": true'
MODEL0="${MODEL0:-$MODELS/hivemind-clock-v2-fp32.onnx}"
CKPT0="${CKPT0:-$MODELS/hivemind-clock-v2.tar}"
case "$MODEL0" in /*) ;; *) MODEL0="$REPO/$MODEL0" ;; esac
case "$CKPT0" in /*) ;; *) CKPT0="$REPO/$CKPT0" ;; esac

# ---- gate (WP-A)
GATE_BASES="${GATE_BASES:-120000,180000}"
GATE_PAIRS="${GATE_PAIRS:-400}"
GATE_SEED="${GATE_SEED:-0}"
GATE_OPENINGS="${GATE_OPENINGS:-eval}"
GATE_K_RANGE="${GATE_K_RANGE:-$K_RANGE}"
GATE_MEASURE="${GATE_MEASURE:-1}"
GATE_CHARGE="${GATE_CHARGE-searched}"
GATE_EQN="${GATE_EQN:-1}"
GATE_EQN_BASE="${GATE_EQN_BASE:-${GATE_BASES%%,*}}"
GATE_IT00="${GATE_IT00:-1}"
GATE_SPRT="${GATE_SPRT:-}"
GATE_OPP_JITTER="${GATE_OPP_JITTER:-}"
GATE_CONTENDER_KW="${GATE_CONTENDER_KW:-}"
GATE_OPPONENT_KW="${GATE_OPPONENT_KW:-}"
GATE_POOL="${GATE_POOL:-arena/pools/phase1.json}"
GATE_OPPONENT="${GATE_OPPONENT:-stock-centralized}"
PROMOTE="${PROMOTE:-1}"
PROMOTE_MARGIN="${PROMOTE_MARGIN:-0}"
[ -n "${GATE_JITTER:-}" ] && echo "iterate: GATE_JITTER is gone (spec v3: the contender is unjittered in the gate); use GATE_OPP_JITTER for the opponent" >&2
# The stock opponent's k: remaining clock / k per search, with k = 1 / E[1/k] over the contender's k range, so both
# sides spend the same mean share of their clock per search. GATE_STOCK_K=0 keeps the old fixed STOCK_NODES.
if [ -z "${GATE_STOCK_K:-}" ]; then
  GATE_STOCK_K="$("$PY" -c "import math,sys; lo,hi=map(float,sys.argv[1].split(',')); print(round(lo if hi<=lo else (hi-lo)/math.log(hi/lo),1))" "$GATE_K_RANGE")"
fi

# ---- self-play / convert / train options (WP-C, WP-D)
REGIME="${REGIME:-v3}"
BASE_MIX="${BASE_MIX:-}"
TIME_ODDS="${TIME_ODDS:-}"
THINK_TARGET_MS="${THINK_TARGET_MS:-}"
SP_CHARGE="${SP_CHARGE:-}"
MIN_THINK_MS="${MIN_THINK_MS:-}"
PACE_MIX="${PACE_MIX:-}"
LAG_MS="${LAG_MS:-}"
EXPLORE_NOISE="${EXPLORE_NOISE:-}"
MOVE_TEMP="${MOVE_TEMP:-}"
TEMP_PLIES="${TEMP_PLIES:-}"
FULL_SEARCH_P="${FULL_SEARCH_P:-}"
PCR_NODES="${PCR_NODES:-}"
DEDUPE="${DEDUPE:-}"
SP_MS_PER_NODE="${SP_MS_PER_NODE:-}"
SP_EXTRA="${SP_EXTRA:-}"
Q_VALUE_RATIO="${Q_VALUE_RATIO:-}"
WEIGHT_NODES_REF="${WEIGHT_NODES_REF:-}"
SPLIT_BY="${SPLIT_BY:-}"
REPLAY_ITERS="${REPLAY_ITERS:-1}"
CONVERT_KW="${CONVERT_KW:-}"
RECIPE="${RECIPE:-}"
TRUNK_LR="${TRUNK_LR:-}"
HEAD_LR="${HEAD_LR:-}"
DRIFT_SHARD="${DRIFT_SHARD:-}"
DRIFT_REFERENCE="${DRIFT_REFERENCE:-}"
TRAIN_EXTRA="${TRAIN_EXTRA:-}"

mkdir -p "$ROOT"
RESULTS="$ROOT/results.tsv"

label() { printf 'it%02d' "$1"; }
# The chunks an iteration trains on: the think-labelled copy when that step ran, else the raw self-play chunks.
data_of() { if [ -f "$1/.done_think" ]; then echo "$1/selfplay/training_data_think"; else echo "$1/selfplay/training_data"; fi; }
done_step() { touch "$1/.done_$2"; }
is_done() { [ -f "$1/.done_$2" ]; }
die() { echo "iterate: $*" >&2; exit 1; }

# The gate rows of one network, in order: one per GATE_BASES entry, then the equal-nodes row.
GATE_TAGS=()
IFS=',' read -r -a _BASES <<< "$GATE_BASES"
for b in "${_BASES[@]}"; do GATE_TAGS+=("gate_$((b / 1000)):$b:"); done
[ "$GATE_EQN" = 1 ] && GATE_TAGS+=("gate_eqn_$((GATE_EQN_BASE / 1000)):$GATE_EQN_BASE:eqn")
GATE_TAG_LIST="$(IFS=','; echo "${GATE_TAGS[*]}")"

# The start of an iteration as "label model checkpoint": what the previous iteration chose (<prev>/.next_start,
# the promotion rule), else the previous iteration's network, else it00 = MODEL0 / CKPT0.
start_of() {
  local prev="$ROOT/$(label $(($1 - 1)))"
  if [ "$1" -gt 1 ] && [ -f "$prev/.next_start" ]; then
    cat "$prev/.next_start"
  elif [ "$1" -gt 1 ] && [ -f "$prev/model-fp32.onnx" ]; then
    echo "$(label $(($1 - 1))) $REPO/$prev/model-fp32.onnx $REPO/$prev/model.tar"
  else
    echo "it00 $MODEL0 $CKPT0"
  fi
}

# One gate row: gate_row <out dir> <network> <base ms> <mode: "" | eqn>. Resumes a row that was cut off.
gate_row() {
  local out="$1" model="$2" base="$3" mode="$4"
  local charge="" stock_k="" opp_jitter="" extra=()
  [ -n "$GATE_CHARGE" ] && charge=", \"charge\": \"$GATE_CHARGE\""
  [ "$GATE_STOCK_K" != 0 ] && stock_k=", \"clock_k\": $GATE_STOCK_K"
  [ -n "$GATE_OPP_JITTER" ] && opp_jitter=", \"jitter\": [$GATE_OPP_JITTER]"
  local ckw="{\"free\": true, \"binary\": \"hivemind-clock\", \"model\": \"$model\", \"k_range\": [$GATE_K_RANGE], \"jitter\": [1.0, 1.0]${charge}${THINK_KW}${GATE_CONTENDER_KW:+, $GATE_CONTENDER_KW}}"
  local okw="{\"serial\": true${charge}${stock_k}${opp_jitter}${GATE_OPPONENT_KW:+, $GATE_OPPONENT_KW}}"
  local onodes=(--opponent-nodes "$STOCK_NODES")
  if [ "$GATE_OPPONENT" != stock-centralized ]; then
    okw="{${GATE_OPPONENT_KW}}"; onodes=()
  fi
  [ "$GATE_MEASURE" = 1 ] && extra+=(--measure-ms-per-node --ms-per-node-cache "$ROOT/ms_per_node.json")
  [ "$mode" = eqn ] && extra+=(--equal-ms-per-node)
  [ -n "$GATE_SPRT" ] && extra+=(--sprt "$GATE_SPRT")
  echo "gate row $out: clock $base ms ${mode:+($mode) }contender $ckw opponent $okw"
  $PY -m arena.pool run --pool "$GATE_POOL" --setup centralized --opponent "$GATE_OPPONENT" \
    ${onodes[@]+"${onodes[@]}"} --contender-kw "$ckw" --opponent-kw "$okw" --openings "$GATE_OPENINGS" --pairs "$GATE_PAIRS" \
    --workers "$WORKERS" --seed "$GATE_SEED" --base-ms "$base" --skip-done ${extra[@]+"${extra[@]}"} --out "$out"
}

# All gate rows of one network into <dir>/gate_*; marker .done_gate when every row has its summary.json.
gate_all() {
  local dir="$1" model="$2" t tag base mode
  is_done "$dir" gate && return 0
  $PY cluster/warm_engine.py --binary hivemind-clock --model "$model" && $PY cluster/warm_engine.py || return 1
  for t in "${GATE_TAGS[@]}"; do
    IFS=':' read -r tag base mode <<< "$t"
    [ -f "$dir/$tag/summary.json" ] && continue
    gate_row "$dir/$tag" "$model" "$base" "$mode" || return 1
  done
  done_step "$dir" gate
}

# Programs' flags, read once: options WP-C / WP-D add are passed only once the program has them.
SP_HELP="$($PY -m arena.selfplay --help 2>/dev/null)"
TRAIN_HELP=""
if [ -n "$RECIPE$TRUNK_LR$HEAD_LR$DRIFT_SHARD$DRIFT_REFERENCE" ]; then
  TRAIN_HELP="$(cd "$HM" && "$PY" -m hivemind.training.train_loop --help 2>/dev/null)"
fi
has_flag() { grep -qE -- "$1([] =,]|\$)" <<< "$2"; }
SP_OPTS=()
sp_opt() {  # flag value env-name
  [ -z "$2" ] && return 0
  if has_flag "$1" "$SP_HELP"; then SP_OPTS+=("$1" "$2"); else echo "iterate: arena.selfplay has no $1: $3=$2 not used" >&2; fi
}
sp_opt --regime "$REGIME" REGIME
sp_opt --base-mix "$BASE_MIX" BASE_MIX
sp_opt --time-odds "$TIME_ODDS" TIME_ODDS
sp_opt --think-target-ms "$THINK_TARGET_MS" THINK_TARGET_MS
sp_opt --charge "$SP_CHARGE" SP_CHARGE
sp_opt --min-think-ms "$MIN_THINK_MS" MIN_THINK_MS
sp_opt --pace-mixture "$PACE_MIX" PACE_MIX
sp_opt --lag-ms "$LAG_MS" LAG_MS
sp_opt --noise "$EXPLORE_NOISE" EXPLORE_NOISE
sp_opt --temperature "$MOVE_TEMP" MOVE_TEMP
sp_opt --temperature-plies "$TEMP_PLIES" TEMP_PLIES
sp_opt --pcr-p "$FULL_SEARCH_P" FULL_SEARCH_P
sp_opt --pcr-nodes "$PCR_NODES" PCR_NODES
sp_opt --dedupe "$DEDUPE" DEDUPE
sp_opt --ms-per-node "$SP_MS_PER_NODE" SP_MS_PER_NODE
TRAIN_OPTS=()
train_opt() {  # flag value env-name
  [ -z "$2" ] && return 0
  if has_flag "$1" "$TRAIN_HELP"; then TRAIN_OPTS+=("$1" "$2"); else echo "iterate: train_loop has no $1: $3=$2 not used" >&2; fi
}
train_opt --recipe "$RECIPE" RECIPE
train_opt --trunk-lr "$TRUNK_LR" TRUNK_LR
train_opt --head-lr "$HEAD_LR" HEAD_LR
train_opt --drift-shard "$DRIFT_SHARD" DRIFT_SHARD
train_opt --drift-reference "$DRIFT_REFERENCE" DRIFT_REFERENCE
CONVERT_ALL="$("$PY" -c "
import json, sys
kw = json.loads(sys.argv[4]) if sys.argv[4] else {}
for k, v, f in (('q_value_ratio', sys.argv[1], float), ('weight_nodes_ref', sys.argv[2], float), ('split_by', sys.argv[3], str)):
    if v:
        kw[k] = f(v)
print(json.dumps(kw) if kw else '')" "$Q_VALUE_RATIO" "$WEIGHT_NODES_REF" "$SPLIT_BY" "$CONVERT_KW")" || die "bad CONVERT_KW"

echo "commit $(git rev-parse HEAD)  loop $ROOT  iterations $START..$((START + ITERS - 1))  games $GAMES  workers $WORKERS"
echo "self-play: k $K_RANGE  jitter $JITTER  nodes $NODES_RANGE  clock $BASE_MS ms  options: ${SP_OPTS[*]:-none} ${SP_EXTRA}"
echo "gate rows: $GATE_TAG_LIST  pairs $GATE_PAIRS  seed $GATE_SEED  openings $GATE_OPENINGS  contender k $GATE_K_RANGE unjittered"
echo "gate opponent: $GATE_OPPONENT ($GATE_POOL); if stock: $( [ "$GATE_STOCK_K" = 0 ] && echo "$STOCK_NODES fixed nodes" || echo "clock / $GATE_STOCK_K" )  charge ${GATE_CHARGE:-default}  measured ms/node $GATE_MEASURE  sprt ${GATE_SPRT:-off}  it00 row $GATE_IT00  promote $PROMOTE (margin $PROMOTE_MARGIN)"
echo "convert: replay from the last $REPLAY_ITERS iteration(s)  keywords ${CONVERT_ALL:-none}  train options: ${TRAIN_OPTS[*]:-none} ${TRAIN_EXTRA}"
echo "shards v$SHARD_VERSION  think head $THINK_HEAD  think labels $THINK_LABELS (N $THINK_NODES, fraction $THINK_FRACTION, cap $THINK_MAX, repeat $THINK_REPEAT)"
$PY -c "from arena.players.hivemind import resolve_binary as r; print('binaries', r('hivemind'), r('hivemind-clock'))"

# ------------------------------------------------------------------ it00 baseline rows (once per RUN)
if [ "$GATE_IT00" = 1 ]; then
  mkdir -p "$ROOT/it00"
  [ -f "$MODEL0" ] || die "it00: start network $MODEL0 missing (run tools/make_clock_net.py in the hivemind repo)"
  if ! is_done "$ROOT/it00" gate; then
    echo; echo "== it00: gate rows for the start network $MODEL0"; date
    gate_all "$ROOT/it00" "$MODEL0" || die "it00: gate failed"
  fi
fi

for ((i = START; i < START + ITERS; i++)); do
  IT="$(label "$i")"
  DIR="$ROOT/$IT"
  mkdir -p "$DIR"
  read -r FROM MODEL CKPT <<< "$(start_of "$i")"
  case "$MODEL" in /*) ;; *) MODEL="$REPO/$MODEL" ;; esac
  case "$CKPT" in /*) ;; *) CKPT="$REPO/$CKPT" ;; esac
  [ -f "$MODEL" ] || die "$IT: start network $MODEL missing (run tools/make_clock_net.py in the hivemind repo)"
  [ -f "$CKPT" ] || die "$IT: start checkpoint $CKPT missing"
  [ -f "$DIR/.start" ] || echo "$FROM $MODEL $CKPT" > "$DIR/.start"
  echo; echo "== $IT from $FROM: $MODEL ($CKPT)"
  date

  # ------------------------------------------------------------------ generate
  if ! is_done "$DIR" generate || { [ "$THINK_LABELS" = 1 ] && ! is_done "$DIR" think; }; then
    # A new network on a new GPU type builds its TensorRT engine file on first start (~200 s), longer than the
    # 120 s a player waits for the engine: build it here with the long wait.
    $PY cluster/warm_engine.py --binary hivemind-clock --model "$MODEL" || die "$IT: engine warm-up failed"
  fi
  if ! is_done "$DIR" generate; then
    rm -rf "$DIR/selfplay/training_data"  # a cut-off run leaves partial chunks: start the iteration's data again
    # shellcheck disable=SC2086  # SP_EXTRA is split on purpose
    $PY -m arena.selfplay --out "$DIR/selfplay" --model "$MODEL" --binary hivemind-clock --games "$GAMES" \
      --workers "$WORKERS" --seed "$i" --game-id-offset "$((i * 1000000))" --run-id "$RUN-$IT" \
      --k-range "$K_RANGE" --jitter "$JITTER" --nodes-range "$NODES_RANGE" --base-ms "$BASE_MS" --no-events \
      --shard-version "$SHARD_VERSION" $THINK_FLAG ${SP_OPTS[@]+"${SP_OPTS[@]}"} $SP_EXTRA \
      || die "$IT: self-play failed"
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
    # Replay chunks from the last REPLAY_ITERS iterations that have data (newest first), as a JSON list.
    PREV_DATA=""
    for ((j = i - 1; j >= 1 && j >= i - REPLAY_ITERS; j--)); do
      P="$(data_of "$ROOT/$(label "$j")")"
      [ -d "$P" ] && PREV_DATA="${PREV_DATA:+$PREV_DATA,}\"$REPO/$P\""
    done
    [ -n "$PREV_DATA" ] && PREV_DATA="[$PREV_DATA]"
    rm -rf "$DIR/rl_data"
    # Resolve the chunk dir here: inside the subshell below the cwd is $HM, where data_of's relative marker test
    # would always miss .done_think and fall back to the unlabelled chunks.
    DATA="$(data_of "$DIR")"
    echo "convert reads $DATA"
    (cd "$HM" && "$PY" - "$REPO/$DIR" "$PREV_DATA" "${VAL_FRACTION:-}" "$REPO/$DATA" "$CONVERT_ALL" <<'EOF'
import inspect, json, sys
from pathlib import Path
from hivemind.data.convert_selfplay_data import convert_to_split_parquet
it = Path(sys.argv[1]).resolve()
replay = json.loads(sys.argv[2]) if sys.argv[2] else None
if replay and len(replay) == 1:
    replay = replay[0]  # one directory: the form every converter version takes
kw = {"validation_fraction": float(sys.argv[3])} if sys.argv[3] else {}
if sys.argv[5]:
    kw.update(json.loads(sys.argv[5]))
params = inspect.signature(convert_to_split_parquet).parameters
takes_any = any(p.kind == p.VAR_KEYWORD for p in params.values())
for k in [k for k in kw if k not in params and not takes_any]:
    print(f"iterate: convert_to_split_parquet has no {k} yet: {k}={kw.pop(k)} not used", file=sys.stderr)
train, val, n_train, n_val = convert_to_split_parquet(Path(sys.argv[4]).resolve(), it / "rl_data",
                                                      replay_input_dir=replay, **kw)
print(f"converted: train {train} ({n_train}), validation {val} ({n_val})  options {kw}")
EOF
    ) || die "$IT: conversion failed"
    done_step "$DIR" convert
  fi

  # ------------------------------------------------------------------ train
  WS="$REPO/$DIR/workspace"
  if ! is_done "$DIR" train; then
    mkdir -p "$WS"
    # shellcheck disable=SC2086  # TRAIN_EXTRA is split on purpose
    (cd "$HM" && HIVEMIND_WORKSPACE="$WS" "$PY" -m hivemind.training.train_loop --mode rl --architecture "$ARCH" \
      --checkpoint "$CKPT" --rl-data-dir "$REPO/$DIR/rl_data/train" --val-data-dir "$REPO/$DIR/rl_data/val" \
      --iteration "$IT" ${BATCH_SIZE:+--batch-size "$BATCH_SIZE"} ${TRAIN_OPTS[@]+"${TRAIN_OPTS[@]}"} \
      ${DRIFT_SHARD:+--drift-out "$REPO/$DIR/drift.json"} $TRAIN_EXTRA) \
      || die "$IT: training failed"
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
  NEW="$REPO/$DIR/model-fp32.onnx"
  gate_all "$DIR" "$NEW" || die "$IT: gate failed"

  # ------------------------------------------------------------------ results and promotion
  OUT="$($PY - "$REPO" "$DIR" "$IT" "$ROOT" "$GATE_TAG_LIST" "$FROM" "$PROMOTE" "$PROMOTE_MARGIN" <<'EOF'
import json, os, sys
repo, d, it, root, tags, start, promote, margin = sys.argv[1:9]
sys.path.insert(0, repo)
from tools.compare_evals import diff_ci, load
from tools.sprt import nelo, pentanomial

tags = [t.split(":") for t in tags.split(",") if t]
sp = json.load(open(f"{d}/selfplay/summary.json"))["results"]
head = ["iter", "games", "samples", "flag_rate", "sit_spells_per_search"]
row = [it, sp.get("games"), sp.get("samples"), sp.get("flag_rate"), sp.get("sit_spells_per_search")]
gains = []
fmt = lambda x: "-" if x is None or x == "" else (f"{x:.4f}" if isinstance(x, float) else str(x))
for tag, base, mode in tags:
    games, rep, _ = load(f"{d}/{tag}")
    r = next(iter(rep["results"]["by_opponent"].values()))
    base_games = load(f"{root}/it00/{tag}")[0] if os.path.isdir(f"{root}/it00/{tag}") else []
    vs = diff_ci(base_games, games) if base_games else (None, None, None, None)
    head += [f"{tag}_{k}" for k in ("games", "score", "lo", "hi", "vs_it00", "vs_it00_lo", "vs_it00_hi", "nelo",
                                    "c_nodes", "b_nodes", "reasons")]
    row += [r["games"], r["score"], r["ci95"][0], r["ci95"][1], vs[0], vs[1], vs[2], nelo(pentanomial(games)),
            r.get("contender_nodes_per_game"), r.get("baseline_nodes_per_game"),
            json.dumps(r.get("reasons", {}), separators=(",", ":"))]
    if mode != "eqn":
        start_games = load(f"{root}/{start}/{tag}")[0] if os.path.isdir(f"{root}/{start}/{tag}") else []
        if start_games:
            gains.append(diff_ci(start_games, games)[0])
drift = ""
if os.path.exists(f"{d}/drift.json"):
    drift = json.dumps(json.load(open(f"{d}/drift.json")), separators=(",", ":"))
if promote != "1":
    promoted, why = 1, "PROMOTE=0"
elif not gains:
    promoted, why = 1, f"no gate rows for the start {start}"
else:
    gain = sum(gains) / len(gains)
    promoted, why = int(gain > float(margin)), f"{gain:+.4f} vs {start} (margin {margin})"
head += ["drift", "start", "promoted", "promotion", "model"]
row += [drift, start, promoted, why, f"{d}/model-fp32.onnx"]
print("\t".join(head))
print("\t".join(fmt(x) for x in row))
EOF
  )" || die "$IT: could not read results"
  HEADER="$(echo "$OUT" | sed -n 1p)"
  ROW="$(echo "$OUT" | sed -n 2p)"
  if [ -f "$RESULTS" ] && [ "$(head -1 "$RESULTS")" != "$HEADER" ]; then
    OLD="$ROOT/results.$(date +%Y%m%d-%H%M%S).tsv"
    mv "$RESULTS" "$OLD" && echo "results.tsv had other columns: moved to $OLD"
  fi
  [ -f "$RESULTS" ] || echo "$HEADER" > "$RESULTS"
  grep -q "^$IT	" "$RESULTS" || echo "$ROW" >> "$RESULTS"
  paste <(echo "$HEADER" | tr '\t' '\n') <(echo "$ROW" | tr '\t' '\n') | column -t -s $'\t'

  # Promotion: the next iteration starts from the new network, or again from this iteration's start.
  PROMOTED="$(echo "$ROW" | awk -F'\t' '{print $(NF-2)}')"
  if [ "$PROMOTED" = 1 ]; then
    echo "$IT $REPO/$DIR/model-fp32.onnx $REPO/$DIR/model.tar" > "$DIR/.next_start"
    echo "promoted $IT: $(echo "$ROW" | awk -F'\t' '{print $(NF-1)}')"
  else
    echo "$FROM $MODEL $CKPT" > "$DIR/.next_start"
    echo "not promoted: next iteration starts again from $FROM ($(echo "$ROW" | awk -F'\t' '{print $(NF-1)}'))"
  fi

  HEADLINE="${GATE_TAGS[$((${#_BASES[@]} - 1))]%%:*}"
  SCORE="$("$PY" -c "import json,sys; print(next(iter(json.load(open(sys.argv[1]))['results']['by_opponent'].values()))['score'])" "$DIR/$HEADLINE/summary.json")"
  if [ -n "${GATE_MIN:-}" ] && "$PY" -c "import sys; sys.exit(0 if float('$SCORE') < float('$GATE_MIN') else 1)"; then
    echo "headline gate score $SCORE ($HEADLINE) is below GATE_MIN $GATE_MIN: stopping after $IT"
    break
  fi
done

echo; column -t -s $'\t' "$RESULTS"
