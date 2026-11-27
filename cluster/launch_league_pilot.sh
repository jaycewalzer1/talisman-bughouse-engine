#!/bin/bash
# Launches the league pilot (docs/plan_league_2026-11-25.md phase 2, docs/custom_engine_spec_v7_league.md 7.1) on PACE
# from the repo root:
#   RUN=league1 bash cluster/launch_league_pilot.sh            # DRY_RUN=1 prints everything instead of submitting
#   BUILD=1 RUN=league1 bash cluster/launch_league_pilot.sh    # first launch after a pin bump: build, then the run
# Two iterations of twin-s it13 (narrow, the hybrid config) through cluster/launch_full_run.sh with pace randomization,
# league opponents, think_gain labels and the learned time policy, and the real-clock two-score gate:
#   WIDE=0 ITERS=2 GAMES_PER_ITER=10000 WORKERS=3 SEGMENT_GAMES=500 GPU_HOURS=100 CHAIN=2 (each overridable)
#   WAVE2_EXTRA = cluster/wave2c_configs.py hybrid + --pace --league --think-labels --think-policy --think-weight 0.1
#                 --gate2 (PACE=0 LEAGUE=0 THINK=0 GATE2=0 drop a switch; EXTRA appends further rl-loop flags).
#   launch_settings.txt records it as "config custom" with that WAVE2_EXTRA (launch_full_run.sh takes CONFIG or
#   WAVE2_EXTRA, not both).
# Preflight, before anything is submitted:
#   pin      cluster/wave2_engine.ref must have d5802ee (the raw-policy opening crash fix) as an ancestor, else refused.
#   engines  the wave2-hivemind checkout at the pin, build-sp1 and build-ninja listing --pace-k-min (PACE),
#            --opponent-model (LEAGUE), --think-policy (THINK) in --help, rl-loop --help listing --pace, --league,
#            --think-labels, --gate2 (for the switches on). Otherwise the build commands are printed and nothing is
#            submitted. BUILD=1 instead moves the checkout to the pin (clean checkout, no other jobs of yours queued:
#            ALLOW_BUSY=1 overrides) and submits the build (lb_<RUN>: cluster/build.sh for build-ninja, then
#            cluster/build_sp1.sh for build-sp1, as cluster/launch_wave2c.sh does); everything below waits on it.
# Then, in this order:
#   THINK=1  cluster/think_gain_check.sbatch (-J tg_<RUN>): the headed it13 on TensorRT vs ONNX Runtime.
#   chain    cluster/launch_full_run.sh (unchanged). Its chain jobs wait (afterok) on the build and the think check:
#            launch_full_run.sh has no dependency setting, so an sbatch shim put first on its PATH turns the chain's
#            --dependency=singleton into --dependency=afterok:<ids>,singleton. Its watchdog counts the pilot's jobs.
#   GATE2=1  runs/full/<RUN>/gate2/wave2.json (python -m hivemind.gate2 wave2-json --wave2 <WAVE2_EXTRA>)
#            and the incumbent's gate arms (it13 = the lineage's first generator, same sha) through
#            cluster/gate2_match.sh with the GATE2_* settings (spec v7 3.3 defaults), so iteration 1 waits only for
#            the candidate's arms. A GATE2_* setting off its default is also passed to rl-loop (--gate2-*), so both
#            sides use the same settings. A failure here is a warning: rl-loop's gate2 stage submits it itself.
# Cost: about 41 GPU-hours for iteration 1 and 30 for iteration 2 (spec v7 7.1); the watchdog stops the run at
# GPU_HOURS (100). Status: python3 cluster/league_pilot_status.py runs/full/<RUN>.
# Local checks (tests/cluster/test_league_pilot.py) only: PILOT_REF (pin file), PILOT_W2, PILOT_SP_ENGINE,
# PILOT_MATCH_ENGINE, PILOT_PY (wave2-hivemind python), PILOT_LAUNCH, PILOT_GATE2_SCRIPT, PILOT_THINK_CHECK, IT13_ONNX.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
RUN="${RUN:?set RUN, the pilot run folder under runs/full (e.g. RUN=league1)}"
DRY_RUN="${DRY_RUN:-0}"
BUILD="${BUILD:-0}"
PACE="${PACE:-1}"; LEAGUE="${LEAGUE:-1}"; THINK="${THINK:-1}"; GATE2="${GATE2:-1}"
EXTRA="${EXTRA:-}"
RESUME="${RESUME:-0}"
for v in DRY_RUN BUILD PACE LEAGUE THINK GATE2 RESUME; do
  case "${!v}" in 0|1) ;; *) echo "$v is 0 or 1" >&2; exit 1 ;; esac
done
RUNS_DIR="${FULL_RUNS:-runs/full}"   # FULL_RUNS only for dry runs (as launch_full_run.sh)
NEED=d5802ee8f962f33bd211be499b1b9b9e5010ec57   # raw-policy opening sample fix (hybrid1 it3 seg04 crash)

# The pull first, then this script again from the pulled copy (bash reads a script as it runs).
if [ "$DRY_RUN" != 1 ] && [ "${NO_PULL:-0}" != 1 ] && [ -z "${PILOT_PULLED:-}" ]; then
  git fetch -q origin
  git checkout -q custom-engine
  git pull -q --ff-only
  PILOT_PULLED=1 exec bash cluster/launch_league_pilot.sh
fi
echo "code: $(git log --oneline -1)"

REF="${PILOT_REF:-cluster/wave2_engine.ref}"
W2="${PILOT_W2:-third_party/wave2-hivemind}"
SP_ENGINE_BIN="${PILOT_SP_ENGINE:-$W2/engine/build-sp1/hivemind}"
MATCH_ENGINE_BIN="${PILOT_MATCH_ENGINE:-$W2/engine/build-ninja/hivemind}"
W2PY="${PILOT_PY:-$W2/.venv/bin/python}"
case "$W2PY" in /*) ;; *) W2PY="$PWD/$W2PY" ;; esac   # run from inside $W2
LAUNCH="${PILOT_LAUNCH:-cluster/launch_full_run.sh}"
GATE2_SCRIPT="${PILOT_GATE2_SCRIPT:-cluster/gate2_match.sh}"
THINK_CHECK="${PILOT_THINK_CHECK:-cluster/think_gain_check.sbatch}"
IT13="${IT13_ONNX:-$W2/engine/models/twin-s-clock-it13.onnx}"

sb() { if [ "$DRY_RUN" = 1 ]; then echo "DRY_RUN sbatch $*" >&2; echo "dry$RANDOM"; else sbatch "$@"; fi; }

# Pilot settings (spec v7 7.1); the env overrides them, launch_full_run.sh checks them.
WIDE="${WIDE:-0}"; ITERS="${ITERS:-2}"; GAMES_PER_ITER="${GAMES_PER_ITER:-10000}"; WORKERS="${WORKERS:-3}"
SEGMENT_GAMES="${SEGMENT_GAMES:-500}"; GPU_HOURS="${GPU_HOURS:-100}"; CHAIN="${CHAIN:-2}"
# Gate settings (spec v7 2 and 3.3). Off-default values also go to rl-loop as --gate2-* flags.
GATE2_GAMES="${GATE2_GAMES:-400}"; GATE2_PARTS="${GATE2_PARTS:-4}"; GATE2_BASE_MS="${GATE2_BASE_MS:-60000}"
GATE2_FREE_K="${GATE2_FREE_K:-160}"; GATE2_EQUAL_K="${GATE2_EQUAL_K:-110}"; GATE2_SEED_BASE="${GATE2_SEED_BASE:-7000}"
GATE2_OPP_SIT="${GATE2_OPP_SIT:-observed}"; GATE2_ARMS="${GATE2_ARMS:-equal free}"

echo "league pilot $RUNS_DIR/$RUN: $ITERS iterations of $GAMES_PER_ITER games, $WORKERS workers, cap $GPU_HOURS GPU-hours"
echo "cost: about 41 GPU-hours for iteration 1 (self-play ~13, val 0.7, think labels 1.5, train 1.7, 100 ms gate 1.2,"
echo "      gate2 candidate 10.7 + incumbent 10.7, main job idle while the gate runs ~1.5) and about 30 for iteration 2;"
echo "      the watchdog stops the run at $GPU_HOURS"

# ------------------------------------------------------------------ pin
pin=$(tr -d '[:space:]' < "$REF")
[ -n "$pin" ] || { echo "$REF is empty" >&2; exit 1; }
if ! git -C "$W2" cat-file -e "$pin^{commit}" 2>/dev/null && [ "$DRY_RUN" != 1 ]; then
  git -C "$W2" fetch -q fork talisman/wave2 2>/dev/null || git -C "$W2" fetch -q origin 2>/dev/null || true
fi
git -C "$W2" cat-file -e "$pin^{commit}" 2>/dev/null \
  || { echo "the pin $pin ($REF) is not in $W2: git -C $W2 fetch fork talisman/wave2" >&2; exit 1; }
if ! git -C "$W2" merge-base --is-ancestor "$NEED" "$pin" 2>/dev/null; then
  echo "refused: the pin $pin ($REF) does not contain d5802ee (raw-policy opening crash fix); the pilot needs it" >&2
  exit 1
fi
echo "pin: $pin (contains d5802ee)"
if [ -e "$RUNS_DIR/$RUN" ] && [ "$RESUME" != 1 ]; then
  echo "$RUNS_DIR/$RUN exists: pick a new RUN (RESUME=1 passes on to launch_full_run.sh)" >&2
  exit 1
fi

# ------------------------------------------------------------------ rl-loop flags
config_flags=$(.venv/bin/python cluster/wave2c_configs.py hybrid)
flags="$config_flags"
[ "$PACE" = 1 ] && flags="$flags --pace"
[ "$LEAGUE" = 1 ] && flags="$flags --league"
[ "$THINK" = 1 ] && flags="$flags --think-labels --think-policy --think-weight 0.1"
if [ "$GATE2" = 1 ]; then
  flags="$flags --gate2"
  [ "$GATE2_ARMS" = "equal free" ] || { echo "GATE2_ARMS: rl-loop's gate needs both arms (equal free)" >&2; exit 1; }
  for kv in GAMES:400 PARTS:4 BASE_MS:60000 FREE_K:160 EQUAL_K:110 SEED_BASE:7000 OPP_SIT:observed; do
    v="GATE2_${kv%%:*}"
    if [ "${!v}" != "${kv#*:}" ]; then
      flag=$(tr 'A-Z_' 'a-z-' <<<"${kv%%:*}")
      flags="$flags --gate2-$flag ${!v}"
    fi
  done
fi
[ -n "$EXTRA" ] && flags="$flags $EXTRA"
echo "WAVE2_EXTRA: $flags"

# ------------------------------------------------------------------ engines
need_engine=(); need_loop=()
[ "$PACE" = 1 ] && { need_engine+=(--pace-k-min); need_loop+=(--pace); }
[ "$LEAGUE" = 1 ] && { need_engine+=(--opponent-model); need_loop+=(--league); }
[ "$THINK" = 1 ] && { need_engine+=(--think-policy); need_loop+=(--think-labels); }
[ "$GATE2" = 1 ] && need_loop+=(--gate2)
engine_help() {  # PACE's modules (TensorRT libraries) when there; not `| grep -q` (SIGPIPE under pipefail)
  ( [ -f cluster/modules.sh ] && source cluster/modules.sh >/dev/null 2>&1; "$1" --help 2>&1 ) || true
}
check() {  # prints one line per thing the switches on need and the checkout lacks
  local have help e o
  have=$(git -C "$W2" rev-parse HEAD)
  [ "$have" = "$pin" ] || echo "$W2 is at $have, not the pin $pin"
  for e in "$SP_ENGINE_BIN" "$MATCH_ENGINE_BIN"; do
    if [ ! -x "$e" ]; then echo "$e missing"; continue; fi
    help=$(engine_help "$e")
    for o in ${need_engine[@]+"${need_engine[@]}"}; do grep -qF -- "$o" <<<"$help" || echo "$e lists no $o"; done
  done
  if [ "${#need_loop[@]}" -gt 0 ]; then
    if [ ! -x "$W2PY" ]; then echo "$W2PY missing (wave2-hivemind env)"
    else
      help=$(cd "$W2" && "$W2PY" -m hivemind rl-loop --help 2>&1 || true)
      for o in "${need_loop[@]}"; do grep -qF -- "$o" <<<"$help" || echo "rl-loop lists no $o"; done
    fi
  fi
  return 0
}
missing=$(check)
build_wrap="cd $PWD && CMAKE_BUILD_PARALLEL_LEVEL=2 bash cluster/build.sh && bash -c '. cluster/env.sh && bash cluster/build_sp1.sh'"
if [ -n "$missing" ] && [ "$BUILD" != 1 ]; then
  echo "not ready for the switches on:" >&2
  sed 's/^/  /' <<<"$missing" >&2
  echo "build at the pin first (GPU node; as cluster/launch_wave2c.sh): cluster/build.sh builds build-ninja," \
       "cluster/build_sp1.sh build-sp1:" >&2
  echo "  BUILD=1 RUN=$RUN bash cluster/launch_league_pilot.sh   (moves $W2 to the pin, submits the build, the run waits on it)" >&2
  echo "  or by hand: git -C $W2 checkout --detach $pin; sbatch -A paceship-bughouse -q inferno -N1 -n6 --mem=64G" \
       "--gres=gpu:rtx_6000:1 -t 4:00:00 -J lb_$RUN --wrap \"$build_wrap\"" >&2
  exit 1
fi
[ -z "$missing" ] && echo "engines: build-sp1, build-ninja and rl-loop list the options the switches need"

deps=()
B=""
if [ "$BUILD" = 1 ]; then
  [ -n "$missing" ] && sed 's/^/build for: /' <<<"$missing"
  if [ "$DRY_RUN" != 1 ]; then
    names=$(squeue -h -u "$USER" -o "%j") || { echo "squeue failed" >&2; exit 1; }
    busy=$(grep -v -e watchdog -e "^$" <<<"$names" | sort -u | tr "\n" " " || true)
    if [ -n "$busy" ] && [ "${ALLOW_BUSY:-0}" != 1 ]; then
      echo "jobs of yours still queued: $busy" >&2
      echo "The build rebuilds the engines in $W2 that they may run: wait for them, or ALLOW_BUSY=1" >&2
      exit 1
    fi
  fi
  if [ "$(git -C "$W2" rev-parse HEAD)" != "$pin" ]; then
    if [ "$DRY_RUN" = 1 ]; then echo "DRY_RUN: would move $W2 to the pin $pin" >&2
    else
      [ -z "$(git -C "$W2" status --porcelain --untracked-files=no)" ] \
        || { echo "$W2 has uncommitted changes and is not at the pin $pin: commit or move them first" >&2; exit 1; }
      git -C "$W2" -c advice.detachedHead=false checkout -q --detach "$pin"
      echo "moved $W2 to the pin $pin"
      left=$(check | grep '^rl-loop' || true)
      [ -z "$left" ] || { echo "the pin's rl-loop itself lacks what the switches need (a build does not help):" >&2
                          echo "$left" >&2; exit 1; }
    fi
  fi
  B=$(sb --parsable -A paceship-bughouse -q inferno -N1 -n6 --mem=64G --gres=gpu:rtx_6000:1 -t 4:00:00 \
        -J "lb_$RUN" -o "lb_${RUN}_%j.log" --wrap "$build_wrap")
  deps+=("$B")
  echo "build: $B (lb_$RUN, log lb_${RUN}_$B.log)"
fi

# ------------------------------------------------------------------ think check
T=""
if [ "$THINK" = 1 ]; then
  if [ ! -f "$THINK_CHECK" ]; then
    if [ "$DRY_RUN" = 1 ]; then echo "DRY_RUN warning: $THINK_CHECK missing (spec v7 6.3, WP4)" >&2
    else echo "$THINK_CHECK missing (spec v7 6.3): THINK=1 needs it, THINK=0 drops the think switches" >&2; exit 1; fi
  fi
  dep=(); [ -n "$B" ] && dep=("--dependency=afterok:$B")
  T=$(sb --parsable -J "tg_$RUN" -o "tg_${RUN}_%j.log" ${dep[@]+"${dep[@]}"} "$THINK_CHECK")
  deps+=("$T")
  echo "think check: $T (tg_$RUN, log tg_${RUN}_$T.log)"
fi

# ------------------------------------------------------------------ chain
shim=""
if [ "${#deps[@]}" -gt 0 ]; then
  after="afterok:$(IFS=:; echo "${deps[*]}")"
  if [ "$DRY_RUN" = 1 ]; then echo "DRY_RUN: the chain's sbatch gets --dependency=$after,singleton (sbatch shim)" >&2
  else
    real=$(command -v sbatch) || { echo "no sbatch on PATH" >&2; exit 1; }
    shim=$(mktemp -d "${TMPDIR:-/tmp}/league_pilot.XXXXXX")
    cat > "$shim/sbatch" <<EOF
#!/bin/bash
# cluster/launch_league_pilot.sh: the chain (--dependency=singleton) also waits for the build and the think check.
args=()
for a in "\$@"; do
  [ "\$a" = --dependency=singleton ] && a="--dependency=$after,singleton"
  args+=("\$a")
done
exec "$real" "\${args[@]}"
EOF
    chmod +x "$shim/sbatch"
  fi
fi
rc=0
env -u CONFIG -u BALANCE -u PACE -u LEAGUE -u THINK -u GATE2 -u BUILD -u EXTRA \
  PATH="${shim:+$shim:}$PATH" RUN="$RUN" WAVE2_EXTRA="$flags" WIDE="$WIDE" ITERS="$ITERS" \
  GAMES_PER_ITER="$GAMES_PER_ITER" WORKERS="$WORKERS" SEGMENT_GAMES="$SEGMENT_GAMES" GPU_HOURS="$GPU_HOURS" \
  CHAIN="$CHAIN" RESUME="$RESUME" NO_PULL=1 DRY_RUN="$DRY_RUN" bash "$LAUNCH" || rc=$?
[ -n "$shim" ] && rm -r "$shim"
if [ "$rc" != 0 ]; then
  echo "launch_full_run.sh failed ($rc)${B:+; build $B}${T:+; think check $T}: cancel them if the run is not relaunched" >&2
  exit "$rc"
fi
ROOT="$RUNS_DIR/$RUN"
if [ "$DRY_RUN" != 1 ]; then
  { [ -n "$B" ] && printf 'build\t%s\n' "$B"; [ -n "$T" ] && printf 'think_check\t%s\n' "$T"; true; } >> "$ROOT/pilot_jobs.tsv"
fi

# ------------------------------------------------------------------ gate2: wave2.json + the incumbent's arms
if [ "$GATE2" = 1 ]; then
  g2root="$ROOT"
  # A dry run writes wave2.json and lets gate2_match.sh (DRY_RUN=1) make its net dir in a scratch copy under runs/
  # (the script refuses a folder outside runs/), removed at the end.
  [ "$DRY_RUN" = 1 ] && { mkdir -p runs; g2root="$(mktemp -d "$PWD/runs/.league_pilot_dry.XXXXXX")/$RUN"; }
  mkdir -p "$g2root/gate2"
  # gate2_match.sh reads the run's engine from launch_settings.txt (launch_full_run.sh wrote it; not in a dry run)
  [ -f "$g2root/launch_settings.txt" ] || echo "config pilot-dry-run  engine $pin" > "$g2root/launch_settings.txt"
  # shellcheck disable=SC2086  # the flags are split on purpose
  wave2_args=(--wave2 $flags)   # wave2-json ignores the non-wave-2 flags (named on stderr)
  [ -n "${BASE_MIX:-}" ] && wave2_args+=(--base-mix "$BASE_MIX")
  if (cd "$W2" && "$W2PY" -m hivemind.gate2 wave2-json "${wave2_args[@]}") > "$g2root/gate2/wave2.json.tmp"; then
    mv "$g2root/gate2/wave2.json.tmp" "$g2root/gate2/wave2.json"
    echo "gate2: $ROOT/gate2/wave2.json from wave2-json ${wave2_args[*]}"
    [ "$DRY_RUN" = 1 ] && echo "DRY_RUN: wave2.json written to the scratch copy $g2root/gate2 only" >&2
  else
    rm -f "$g2root/gate2/wave2.json.tmp"
    echo "warning: python -m hivemind.gate2 wave2-json failed: no wave2.json; rl-loop writes it at its first gate2" \
         "stage and the incumbent's arms are submitted then" >&2
  fi
  if [ -f "$g2root/gate2/wave2.json" ]; then
    if [ ! -f "$IT13" ] && [ "$DRY_RUN" != 1 ]; then
      (cd "$W2" && "$W2PY" -c 'import sys; from hivemind.cli.fetch_network import fetch_network as f; f("clock-it13", sys.argv[1])' \
        "$(cd "$(dirname "$IT13")" && pwd)") || true
    fi
    if [ ! -f "$GATE2_SCRIPT" ]; then
      echo "warning: $GATE2_SCRIPT missing (spec v7 3.3): the incumbent's arms are submitted by rl-loop" >&2
    elif [ ! -f "$IT13" ]; then
      echo "warning: $IT13 missing: the incumbent's arms are submitted by rl-loop at iteration 1" >&2
    else
      g2env=(GATE2_GAMES="$GATE2_GAMES" GATE2_PARTS="$GATE2_PARTS" GATE2_BASE_MS="$GATE2_BASE_MS"
             GATE2_FREE_K="$GATE2_FREE_K" GATE2_EQUAL_K="$GATE2_EQUAL_K" GATE2_SEED_BASE="$GATE2_SEED_BASE"
             GATE2_OPP_SIT="$GATE2_OPP_SIT" GATE2_ARMS="$GATE2_ARMS" DRY_RUN="$DRY_RUN")
      [ -n "$B" ] && g2env+=(SBATCH_DEPENDENCY="afterok:$B")   # eval jobs wait for the engines too
      echo "gate2 incumbent: bash $GATE2_SCRIPT $g2root incumbent $IT13${B:+ (after the build $B)}"
      env "${g2env[@]}" bash "$GATE2_SCRIPT" "$g2root" incumbent "$IT13" \
        || echo "warning: $GATE2_SCRIPT failed for the incumbent: rl-loop's gate2 stage submits it" >&2
    fi
  fi
  [ "$DRY_RUN" = 1 ] && rm -r "$(dirname "$g2root")"
fi
echo "status: python3 cluster/league_pilot_status.py $ROOT    cap: $GPU_HOURS GPU-hours (full_, sp_, g2_, tg_, lb_ jobs)"
