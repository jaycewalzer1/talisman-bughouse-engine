#!/bin/bash
#SBATCH -J eval -A paceship-bughouse -q inferno
#SBATCH -N1 -n6 --mem=32G --gres=gpu:rtx_6000:1 -t 08:00:00
#SBATCH -o eval_%x_%j.log
# Scores one fixed team against one opponent on the held-out openings, into runs/<NAME>. No forced exploration and
# no learning, so two teams scored this way can be compared directly (tools/compare_evals.py).
# Scheduler: without POLICY it is untrained, which plays like stock plus whatever the feature groups do on their own.
#   NAME=eval/untrained_none sbatch -J none cluster/eval.sbatch
#   NAME=eval/wave2 FEATURES=mate,danger POLICY=runs/waves/wave2/latest.pt sbatch -J wave2 cluster/eval.sbatch
# RULE=mate_wait or mate_ask adds a fixed hand-written rule to the scheduler (it needs FEATURES to include mate).
# Stock Hivemind in a team setup (centralized, silent or sharing; LEND=1 lends idle compute), no scheduler:
#   NAME=match/central_100 SETUP=centralized NODES=100 sbatch -J c100 cluster/eval.sbatch
# With LEND=1, LEND_SHARE=0.5 borrows only half of the partner's idle machine (default: all of it).
# CONTENDER_KW and OPPONENT_KW are JSON player options, e.g. CONTENDER_KW='{"clock_k": 110, "serial": true}'.
# BASE_MS is each player's starting clock (default 60000).
# NODES and OPPONENT_NODES replace the node budgets (200 each). PAIRS=800 is 1600 games, about 4 hours at 60 s.
# Resumable: every finished game is kept (runs/<NAME>/game_NNNN.jsonl, results.jsonl), so a job cut off by the time
# limit is finished by submitting it again with the same NAME and settings (the row refuses a different schedule).
# Long rows (180 s, 1600+ games) are run that way over several jobs. Read a running row with
#   .venv/bin/python tools/sprt.py runs/<NAME>   or   .venv/bin/python tools/compare_evals.py runs/<NAME>
# Audit 2026-11-17 WP-A, all off unless set:
#   MEASURE=1     measure ms per searched node per binary+network for every engine team without an ms_per_node
#                 in its options (kept in runs/<NAME>/ms_per_node.json; MEASURE_NODES, default 400)
#                 MEASURE_WARM_MS (default 0 = off): wave 2 / upstream arms search that long before they are
#                 measured (a cold first arm reads ~20% slow on PACE, docs/speed_gap.md); MEASURE_REPEATS (3)
#   SPRT=0,20     GSPRT over normalised Elo [ELO0,ELO1(,ALPHA,BETA)] after every game pair; stops when decided
#   JITTER=lo,hi  think-time factor per search for every centralized Hivemind team (both sides)
#   WORKERS       parallel games (default 3)
#   POOL          pool file (default arena/pools/phase1.json; arena/pools/upstream_v3.json has upstream-v3, fairy-sf;
#                 arena/pools/wave2.json has the wave 2 contender and upstream-it13, wave2-self, it04)
#   SETUP=pool    the contender exactly as the pool file has it (e.g. kind wave2), not stock Hivemind or the agent
#   OPENINGS      opening set (default eval; bughouse, bughouse_canon, chess_controls, bare, train, all)
#   RULES         arena rules profile: v3 (spec v3 D4, partners flagging together) or v2 (the default)
#   WARM_KINDS    extra engines to warm before the row, e.g. "upstream wave2" (TensorRT engine files)
#   ENGINE_REF    refuse to start unless third_party/wave2-hivemind is at this commit (the long test sets its round's pin)
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(git rev-parse --show-toplevel)}"
source cluster/env.sh

NAME="${NAME:?set NAME, the result folder under runs/}"
OUT="runs/$NAME"
PY=.venv/bin/python
if [ -f "$OUT/summary.json" ]; then
  echo "$OUT already has a result: pick another NAME (tools/rescore.py rebuilds a result from the game files)" >&2
  exit 1
fi
[ -d "$OUT" ] && echo "resuming $OUT: $(ls "$OUT"/game_*.jsonl 2>/dev/null | wc -l | tr -d ' ') game files already there"

ARGS=()
if [ "${SETUP:-}" = pool ]; then
  WHO="the pool's contender ($(.venv/bin/python -c "import json,sys; c=json.load(open(sys.argv[1]))['contender']; print(c['name'], c['kind'])" "${POOL:-arena/pools/phase1.json}"))"
elif [ -n "${SETUP:-}" ]; then
  ARGS+=(--setup "$SETUP")
  [ "${LEND:-0}" = 1 ] && ARGS+=(--lend)
  [ -n "${LEND_SHARE:-}" ] && ARGS+=(--lend-share "$LEND_SHARE")
  WHO="stock $SETUP lend ${LEND:-0} share ${LEND_SHARE:-1}"
else
  ARGS+=(--agent --features "${FEATURES:-}")
  [ -n "${POLICY:-}" ] && ARGS+=(--policy "$POLICY")
  [ -n "${RULE:-}" ] && ARGS+=(--rule "$RULE")
  WHO="scheduler features '${FEATURES:-}' policy ${POLICY:-untrained} rule ${RULE:-none}"
fi
[ -n "${NODES:-}" ] && ARGS+=(--nodes "$NODES")
[ -n "${OPPONENT_NODES:-}" ] && ARGS+=(--opponent-nodes "$OPPONENT_NODES")
[ -n "${CONTENDER_KW:-}" ] && ARGS+=(--contender-kw "$CONTENDER_KW")
[ -n "${OPPONENT_KW:-}" ] && ARGS+=(--opponent-kw "$OPPONENT_KW")
[ "${MEASURE:-0}" = 1 ] && ARGS+=(--measure-ms-per-node --measure-nodes "${MEASURE_NODES:-400}")
[ "${MEASURE:-0}" = 1 ] && [ -n "${MEASURE_WARM_MS:-}" ] && ARGS+=(--measure-warm-ms "$MEASURE_WARM_MS")
[ "${MEASURE:-0}" = 1 ] && [ -n "${MEASURE_REPEATS:-}" ] && ARGS+=(--measure-repeats "$MEASURE_REPEATS")
[ -n "${SPRT:-}" ] && ARGS+=(--sprt "$SPRT")
[ -n "${JITTER:-}" ] && ARGS+=(--jitter "$JITTER")
[ -n "${RULES:-}" ] && ARGS+=(--rules "$RULES")

echo "commit $(git rev-parse HEAD)  eval $OUT  $WHO  nodes ${NODES:-200}"
if [ -d third_party/wave2-hivemind/.git ]; then
  echo "wave2 engine $(git -C third_party/wave2-hivemind rev-parse HEAD)  pin $(cat cluster/wave2_engine.ref 2>/dev/null || echo none)"
fi
# ENGINE_REF (cluster/launch_long_test.sh sets it): the wave2 engine commit this row must play; any other is refused.
if [ -n "${ENGINE_REF:-}" ]; then
  have=$(git -C third_party/wave2-hivemind rev-parse HEAD 2>/dev/null || echo none)
  if [ "$have" != "$ENGINE_REF" ]; then
    echo "wave2 engine is at $have, not ENGINE_REF $ENGINE_REF: this row would play another engine; refused" >&2
    exit 1
  fi
fi
echo "opponent ${OPPONENT:-stock-centralized} nodes ${OPPONENT_NODES:-as in the pool}  pairs ${PAIRS:-800}  seed ${SEED:-0}"
echo "contender options ${CONTENDER_KW:-none}  opponent options ${OPPONENT_KW:-none}  clock ${BASE_MS:-60000} ms"
echo "measure ms/node ${MEASURE:-0} warm ${MEASURE_WARM_MS:-0} ms repeats ${MEASURE_REPEATS:-3}  sprt ${SPRT:-off}  jitter ${JITTER:-off}  workers ${WORKERS:-3}"
echo "pool ${POOL:-arena/pools/phase1.json}  openings ${OPENINGS:-eval}  rules ${RULES:-v2}"
$PY -c "from arena.players.hivemind import DEFAULT_MODEL; print('engine model', DEFAULT_MODEL)"
$PY cluster/warm_engine.py || exit 1
$PY cluster/engine_parity.py check || exit 1
for kind in ${WARM_KINDS:-}; do $PY cluster/warm_engine.py --kind "$kind" || exit 1; done

$PY -m arena.pool run --pool "${POOL:-arena/pools/phase1.json}" "${ARGS[@]}" --opponent "${OPPONENT:-stock-centralized}" --openings "${OPENINGS:-eval}" \
  --pairs "${PAIRS:-800}" --workers "${WORKERS:-3}" --seed "${SEED:-0}" --base-ms "${BASE_MS:-60000}" \
  --skip-done --out "$OUT"
