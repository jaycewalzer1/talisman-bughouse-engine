#!/bin/bash
# Launches the full wave 2 training run on PACE from the repo root, as cluster/decide_full_run.py prints it:
#   RUN=full1 CONFIG=nodur WIDE=0 CHAIN=80 GAMES_PER_ITER=50000 bash cluster/launch_full_run.sh
# One lineage in runs/full/<RUN> (cluster/wave2_iterate.sbatch, RUNS_DIR=runs/full), CHAIN chained 16 h jobs of one
# iteration each (-J full_<RUN>, --dependency=singleton: one at a time, in order; a job cut off mid-iteration is
# continued by the next from rl-loop's stage markers), capped at ITERS iterations (MAX_ITER: jobs left after the
# last iteration exit at once), and cluster/full_run_watchdog.sh, which cancels the chain on a crash or a gate under 0.40.
# Settings: RUN (required), CONFIG (a name in cluster/wave2c_configs.py, e.g. nodur or cont; sets WAVE2_EXTRA) or
# WAVE2_EXTRA itself, WIDE (0 narrow, 1 wide), CHAIN (number of chained jobs), GAMES_PER_ITER (50000), ITERS (10),
# RESUME (0; 1 adds CHAIN more jobs to an existing runs/full/<RUN> with its recorded settings checked), NO_PULL (0;
# 1 skips the git pull, for dry runs). Refuses an existing run folder unless RESUME=1, and a wave2-hivemind checkout
# that is not at cluster/wave2_engine.ref (the engines the readiness round checked).
# Concurrent self-play: WORKERS=K adds K self-play worker jobs (-J sp_<RUN>) to every iteration (rl-loop
# --shared-selfplay, cluster/selfplay_workers.sh); the main job trains and gates alone, the workers exit when the
# iteration has its games. SEGMENT_GAMES (500 with workers, 2000 with workers and BALANCE=1, else wave2_iterate's 2000) is what one claim plays;
# PER_JOB (1; 2 with workers) iterations per chained job. GPU_HOURS=N caps the run: the watchdog sums the GPU-hours of
# full_<RUN> and sp_<RUN> jobs and stops the run (no new workers, rl-loop.stop, pending jobs cancelled) at N.
# WORKERS=0 is the old single-GPU chain, bit for bit. cluster/gpu_budget.py prints what N the account can afford.
# A wave2-hivemind checkout off the pin is moved to the pin here when only Python differs between them (the engines
# stay as built); an engine change still needs the readiness round's build.
#
# Search balance (cluster/README.md "Search balance"; docs/search_balance.md). BALANCE=1 runs the chain with
# wave2_iterate.sbatch BALANCE=1 (rl-loop --search-balance; build-sp1 rebuilt with the playout-cap flags at each main
# job's start); BALANCE_PRIOR / BALANCE_ARGS are passed on. BALANCE unset or 0: everything below is as before, and
# "balance 1" is added to launch_settings.txt only when it is on. With BALANCE=1 the checkout may also be moved to a
# pin whose engine/ differs (fast-forward only): build-sp1 is rebuilt by the next main job, never under a running one.
# Switching a RUNNING run (its running job and the watchdog are kept):
#   RESUME=1 REPLACE_PENDING=1 RUN=hybrid1 BALANCE=1 bash cluster/launch_full_run.sh
#   REPLACE_PENDING=1 (needs RESUME=1) submits the same number of jobs again (CHAIN defaults to the number of PENDING
#   full_<RUN> jobs) with the new environment and only then cancels the old PENDING ones (scancel -t PENDING, by id);
#   singleton keeps the new jobs behind them, and a failed sbatch leaves the old chain uncancelled. The recorded settings fill
#   CONFIG/WAVE2_EXTRA, WIDE, GAMES_PER_ITER, ITERS, WORKERS, SEGMENT_GAMES and PER_JOB when unset, and must match apart
#   from "engine" and "balance"; the old line goes to launch_settings.history. No new watchdog (NEW_WATCHDOG=1 adds
#   one). The running job finishes its iteration as it started; the first replaced job continues from the markers.
#   Switch back the same way with BALANCE=0 (BALANCE left unset keeps the recorded "balance 1").
# Switching a RUNNING run to concurrent self-play, more iterations or a GPU-hour cap (same switch, at an iteration
# boundary for the workers; cluster/README.md "Switching a running run"):
#   RESUME=1 REPLACE_PENDING=1 NEW_WATCHDOG=1 RUN=hybrid1 WORKERS=5 SEGMENT_GAMES=2000 ITERS=20 GPU_HOURS=400 CHAIN=10 bash cluster/launch_full_run.sh
#   With REPLACE_PENDING=1, WORKERS, SEGMENT_GAMES, PER_JOB, ITERS (the MAX_ITER of the run) and GPU_HOURS may change
#   too (config/wave2_extra, wide and games_per_iter may not); the change goes to launch_settings.history. rl-loop
#   finishes the iteration in progress the way it began (an iteration begun without workers is finished by the main
#   job alone, with its own segment size) and starts the next one with the new settings. With BALANCE=1 the games per
#   iteration must be a multiple of SEGMENT_GAMES (rl-loop refuses otherwise), and SEGMENT_GAMES defaults to 2000
#   (another size needs BALANCE_ANY_SEGMENT=1: the balance prior's noise terms are calibrated on 2000-game segments). GPU_HOURS lives in the watchdog, so
#   changing it needs NEW_WATCHDOG=1, which then REPLACES the run's watchdog: the new one is submitted first with
#   --dependency=afterany on the old one (found by the "watching full_<RUN>" line of its Slurm log, or OLD_WATCHDOG=
#   ids|none), then the old one is cancelled, so the two never run together. It keeps the old one's monitor and
#   anchors settings (its last "watching" line in watchdog.log; MONITOR/ANCHORS override) and counts the cap from
#   the run's first watchdog line (CAP_SINCE overrides). If the old watchdog cannot be cancelled, the new one is.
# DRY_RUN=1: no git pull or checkout, nothing written, every sbatch/scancel printed instead of run (squeue still read;
# a fake squeue/sbatch on PATH works for local checks).
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
RUN="${RUN:?set RUN, the run folder under runs/full (decide_full_run.py prints it)}"
RESUME="${RESUME:-0}"
REPLACE_PENDING="${REPLACE_PENDING:-0}"
DRY_RUN="${DRY_RUN:-0}"
RUNS_DIR="${FULL_RUNS:-runs/full}"   # FULL_RUNS only for dry runs
case "$REPLACE_PENDING" in 0|1) ;; *) echo "REPLACE_PENDING is 0 or 1" >&2; exit 1 ;; esac
case "$DRY_RUN" in 0|1) ;; *) echo "DRY_RUN is 0 or 1" >&2; exit 1 ;; esac
if [ "$REPLACE_PENDING" = 1 ] && [ "$RESUME" != 1 ]; then echo "REPLACE_PENDING=1 needs RESUME=1" >&2; exit 1; fi
if [ "$DRY_RUN" = 1 ]; then NO_PULL=1; fi

sb() { if [ "$DRY_RUN" = 1 ]; then echo "DRY_RUN sbatch $*" >&2; echo "dry$RANDOM"; else sbatch "$@"; fi; }
sc() { if [ "$DRY_RUN" = 1 ]; then echo "DRY_RUN scancel $*" >&2; else scancel "$@"; fi; }

PENDING_IDS=()
if [ "$REPLACE_PENDING" = 1 ]; then
  [ -f "$RUNS_DIR/$RUN/launch_settings.txt" ] || { echo "$RUNS_DIR/$RUN/launch_settings.txt missing: not a launched run" >&2; exit 1; }
  # The recorded settings are the defaults: fields are separated by two spaces, "key value".
  eval "$(python3 - "$RUNS_DIR/$RUN/launch_settings.txt" <<'EOF'
import shlex, sys
line = open(sys.argv[1]).read().strip()
for field in line.split("  "):
    key, _, value = field.partition(" ")
    if key.isidentifier():
        print(f"REC_{key}={shlex.quote(value)}")
EOF
)"
  # A failed squeue is not an empty queue.
  pend=$(squeue -h -u "$USER" -n "full_$RUN" -t PENDING -o "%i") || { echo "squeue failed" >&2; exit 1; }
  for id in $pend; do PENDING_IDS+=("$id"); done
  [ "${#PENDING_IDS[@]}" -gt 0 ] || { echo "no PENDING full_$RUN jobs to replace (RESUME=1 without REPLACE_PENDING adds jobs)" >&2; exit 1; }
  if [ -z "${CONFIG:-}" ] && [ -z "${WAVE2_EXTRA:-}" ]; then
    if [ "${REC_config:-custom}" != custom ]; then CONFIG="$REC_config"; else WAVE2_EXTRA="${REC_wave2_extra:-}"; fi
  fi
  WIDE="${WIDE:-${REC_wide:-}}"
  GAMES_PER_ITER="${GAMES_PER_ITER:-${REC_games_per_iter:-}}"
  ITERS="${ITERS:-${REC_iters:-}}"
  if [ -n "${REC_workers:-}" ]; then
    WORKERS="${WORKERS:-$REC_workers}"; SEGMENT_GAMES="${SEGMENT_GAMES:-${REC_segment_games:-}}"
    PER_JOB="${PER_JOB:-${REC_per_job:-}}"
  fi
  CHAIN="${CHAIN:-${#PENDING_IDS[@]}}"
  GPU_HOURS="${GPU_HOURS-${REC_gpu_hours:-}}"
  # Search balance stays as recorded unless BALANCE is given (BALANCE=0 switches it off): a switch made for other
  # settings must not turn it off by leaving BALANCE out.
  if [ -z "${BALANCE:-}" ] && [ "${REC_balance:-0}" = 1 ]; then BALANCE=1; echo "BALANCE=1 kept from launch_settings.txt (BALANCE=0 turns it off)"; fi
fi

WIDE="${WIDE:-0}"
CHAIN="${CHAIN:?set CHAIN, the number of chained 16 h jobs (decide_full_run.py prints it)}"
GAMES_PER_ITER="${GAMES_PER_ITER:-50000}"
ITERS="${ITERS:-10}"
WORKERS="${WORKERS:-0}"
GPU_HOURS="${GPU_HOURS:-}"
BALANCE="${BALANCE:-0}"
NEW_WATCHDOG="${NEW_WATCHDOG:-0}"
case "$WORKERS" in ''|*[!0-9]*) echo "WORKERS is a number" >&2; exit 1 ;; esac
case "$BALANCE" in 0|1) ;; *) echo "BALANCE is 0 or 1" >&2; exit 1 ;; esac
case "$NEW_WATCHDOG" in 0|1) ;; *) echo "NEW_WATCHDOG is 0 or 1" >&2; exit 1 ;; esac
# Search balance's noise terms (the prior's sigma_hat/tau_hat from cluster/balance_prior.sbatch and the pooled
# sigma_samples) are measured on 2000-game segments and are not rescaled by game count, so with BALANCE=1 the
# workers' segment defaults to 2000 too, and another size is refused unless BALANCE_ANY_SEGMENT=1.
if [ "$WORKERS" -gt 0 ]; then
  if [ "$BALANCE" = 1 ]; then SEGMENT_GAMES="${SEGMENT_GAMES:-2000}"; else SEGMENT_GAMES="${SEGMENT_GAMES:-500}"; fi
  PER_JOB="${PER_JOB:-2}"
else PER_JOB="${PER_JOB:-1}"; fi
case "$WIDE" in 0|1) ;; *) echo "WIDE is 0 or 1" >&2; exit 1 ;; esac
for v in ITERS PER_JOB CHAIN GAMES_PER_ITER ${SEGMENT_GAMES:+SEGMENT_GAMES}; do
  case "${!v}" in ''|*[!0-9]*|0) echo "$v is a positive number (got '${!v}')" >&2; exit 1 ;; esac
done
case "$GPU_HOURS" in ''|[0-9]*) ;; *) echo "GPU_HOURS is a number of GPU-hours" >&2; exit 1 ;; esac
if [ "$BALANCE" = 1 ] && [ "${SEGMENT_GAMES:-2000}" != 2000 ] && [ "${BALANCE_ANY_SEGMENT:-0}" != 1 ]; then
  echo "BALANCE=1 with SEGMENT_GAMES $SEGMENT_GAMES: search balance's noise terms are calibrated on 2000-game segments" \
       "and are not rescaled, so a smaller segment overstates Z. Use SEGMENT_GAMES=2000, or BALANCE_ANY_SEGMENT=1 to accept it" >&2
  exit 1
fi
# rl-loop --search-balance plans whole segments: a remainder would fail the first job (and the watchdog the chain).
if [ "$BALANCE" = 1 ] && [ $((GAMES_PER_ITER % ${SEGMENT_GAMES:-2000})) -ne 0 ]; then
  echo "BALANCE=1 needs GAMES_PER_ITER ($GAMES_PER_ITER) to be a multiple of SEGMENT_GAMES (${SEGMENT_GAMES:-2000})" >&2
  exit 1
fi

if [ "${NO_PULL:-0}" != 1 ]; then
  git fetch -q origin
  git checkout -q custom-engine
  git pull -q --ff-only
fi
echo "code: $(git log --oneline -1)"
if [ -n "${CONFIG:-}" ]; then
  WAVE2_EXTRA=$(.venv/bin/python cluster/wave2c_configs.py "$CONFIG")
fi
WAVE2_EXTRA="${WAVE2_EXTRA:?set CONFIG (cluster/wave2c_configs.py) or WAVE2_EXTRA}"
W2=third_party/wave2-hivemind
pin=$(cat cluster/wave2_engine.ref)
have=$(git -C "$W2" rev-parse HEAD)
if [ "$have" != "$pin" ] && [ "${NO_PULL:-0}" != 1 ]; then
  git -C "$W2" fetch -q fork talisman/wave2 2>/dev/null || git -C "$W2" fetch -q origin 2>/dev/null || true
  if git -C "$W2" merge-base --is-ancestor "$have" "$pin" 2>/dev/null && git -C "$W2" diff --quiet "$have" "$pin" -- engine; then
    echo "moving $W2 from $have to the pin $pin (Python only: no engine rebuild)"
    git -C "$W2" checkout -q "$pin" && have=$pin
  elif [ "$BALANCE" = 1 ] && git -C "$W2" merge-base --is-ancestor "$have" "$pin" 2>/dev/null; then
    # The search-balance engine change: build-sp1 is rebuilt by wave2_iterate.sbatch (BALANCE=1) at the next main
    # job's start, never under a running job; build-ninja (gates, anchors) is not touched.
    echo "moving $W2 from $have to the pin $pin (engine/ differs: BALANCE=1 rebuilds build-sp1 at the next job's start)"
    git -C "$W2" checkout -q "$pin" && have=$pin
  fi
fi
if [ "$have" != "$pin" ]; then
  if [ "$DRY_RUN" = 1 ]; then
    echo "DRY_RUN: $W2 is at $have, not the pin $pin (a real run would move it or stop here)" >&2
  else
    echo "$W2 is at $have, not the pin $pin: run the readiness round's build first (bash cluster/launch_wave2c.sh)" >&2
    exit 1
  fi
fi

settings="config ${CONFIG:-custom}  wave2_extra $WAVE2_EXTRA  wide $WIDE  games_per_iter $GAMES_PER_ITER  iters $ITERS  engine $pin"
if [ "$WORKERS" -gt 0 ]; then settings="$settings  workers $WORKERS  segment_games $SEGMENT_GAMES  per_job $PER_JOB"; fi
if [ -n "$GPU_HOURS" ]; then settings="$settings  gpu_hours $GPU_HOURS"; fi
if [ "$BALANCE" = 1 ]; then settings="$settings  balance 1"; fi
# The settings without the fields a switch may change (REPLACE_PENDING=1 compares these): what the lineage is
# (config, wave2_extra, wide, games_per_iter) stays; how it is run (engine pin, balance, iterations, workers,
# segment size, iterations per job, GPU-hour cap) may change.
core() { sed -e 's/  engine [0-9a-f]*//' -e 's/  balance [01]//' -e 's/  iters [0-9]*//' -e 's/  workers [0-9]*//' \
             -e 's/  segment_games [0-9]*//' -e 's/  per_job [0-9]*//' -e 's/  gpu_hours [0-9.]*//'; }
if [ -e "$RUNS_DIR/$RUN" ]; then
  if [ "$RESUME" != 1 ]; then
    echo "$RUNS_DIR/$RUN exists: pick a new RUN, or RESUME=1 to add CHAIN jobs to it" >&2
    exit 1
  fi
  was=$(cat "$RUNS_DIR/$RUN/launch_settings.txt" 2>/dev/null || true)
  if [ "$REPLACE_PENDING" = 1 ]; then
    if [ "$(core <<<"$was")" != "$(core <<<"$settings")" ]; then
      echo "REPLACE_PENDING=1 may change engine, balance, iters, workers, segment_games, per_job and gpu_hours only;" \
           "other settings differ from launch_settings.txt:" >&2
      echo "  was: $was" >&2
      echo "  now: $settings" >&2
      exit 1
    fi
    if [ "$GPU_HOURS" != "${REC_gpu_hours:-}" ] && [ "$NEW_WATCHDOG" != 1 ]; then
      echo "GPU_HOURS ${GPU_HOURS:-none} differs from the run's ${REC_gpu_hours:-none}: the cap is the watchdog's," \
           "so this needs NEW_WATCHDOG=1 (it replaces the run's watchdog)" >&2
      exit 1
    fi
    done_its=$( (find "$RUNS_DIR/$RUN/runs" -path '*/rl-it*/stages/iteration.done' 2>/dev/null || true) | wc -l | tr -d ' ')
    if [ "$ITERS" -le "$done_its" ]; then
      echo "warning: ITERS $ITERS: the run has $done_its finished iterations; the new jobs will exit at once" >&2
    fi
  elif [ "$was" != "$settings" ]; then
    echo "RESUME=1 with other settings than $RUNS_DIR/$RUN/launch_settings.txt:" >&2
    echo "  was: $was" >&2
    echo "  now: $settings" >&2
    exit 1
  fi
else
  if [ "$DRY_RUN" = 1 ]; then echo "DRY_RUN: would create $RUNS_DIR/$RUN with launch_settings.txt" >&2
  else
    mkdir -p "$RUNS_DIR/$RUN"
    echo "$settings" > "$RUNS_DIR/$RUN/launch_settings.txt"
  fi
fi
echo "run $RUNS_DIR/$RUN: $settings, $CHAIN chained jobs"

# NEW_WATCHDOG=1 with REPLACE_PENDING=1 replaces the run's watchdog (see the header). Found before anything is submitted.
REPLACE_WD=0; OLD_WD=()
if [ "$REPLACE_PENDING" = 1 ] && [ "$NEW_WATCHDOG" = 1 ]; then
  REPLACE_WD=1
  if [ -n "${OLD_WATCHDOG:-}" ]; then
    [ "$OLD_WATCHDOG" = none ] || for id in ${OLD_WATCHDOG//,/ }; do OLD_WD+=("$id"); done
  else
    wds=$(squeue -h -u "$USER" -n full_watchdog -o "%i") || { echo "squeue failed" >&2; exit 1; }
    unknown=()
    for id in $wds; do
      out=$(scontrol show job "$id" 2>/dev/null | sed -n 's/.*StdOut=\([^ ]*\).*/\1/p' | head -1)
      if [ -n "$out" ] && [ -f "$out" ]; then
        if grep -qE "watching full_${RUN}( |\$)" "$out"; then OLD_WD+=("$id"); fi
      else
        unknown+=("$id")
      fi
    done
    if [ "${#unknown[@]}" -gt 0 ]; then
      echo "cannot tell which run full_watchdog ${unknown[*]} watches (no Slurm log yet): name the run's watchdog(s)" \
           "with OLD_WATCHDOG=id[,id] (or OLD_WATCHDOG=none)" >&2
      exit 1
    fi
  fi
  # The old watchdog's monitor and anchor settings (its "watching" line), unless given.
  last=$(grep -E "watching full_${RUN} \(monitor" "$RUNS_DIR/$RUN/watchdog.log" 2>/dev/null | tail -1 || true)
  rec_mon=$(sed -n 's/.*(monitor \([01]\).*/\1/p' <<<"$last")
  rec_anc=$(sed -n 's/.*, anchors \([^,)]*\).*/\1/p' <<<"$last"); [ "$rec_anc" = off ] && rec_anc=""
  WD_MONITOR="${MONITOR:-${rec_mon:-1}}"
  WD_ANCHORS="${ANCHORS-$rec_anc}"
  first=$(grep -m1 -E "watching full_${RUN}( |\$)" "$RUNS_DIR/$RUN/watchdog.log" 2>/dev/null | cut -c1-19 | tr ' ' T || true)
  WD_CAP_SINCE="${CAP_SINCE:-${first:-$(date +%Y-%m-%dT%H:%M:%S)}}"
  dep=(); [ "${#OLD_WD[@]}" -gt 0 ] && dep=("--dependency=afterany:$(IFS=:; echo "${OLD_WD[*]}")")
  echo "watchdog: replacing ${OLD_WD[*]:-none found} (monitor $WD_MONITOR, anchors ${WD_ANCHORS:-off}, cap ${GPU_HOURS:-none} from $WD_CAP_SINCE)"
  W=$(RUN="$RUN" RUNS_DIR="$RUNS_DIR" GPU_HOURS="$GPU_HOURS" CAP_SINCE="$WD_CAP_SINCE" MONITOR="$WD_MONITOR" \
      ANCHORS="$WD_ANCHORS" sb --parsable ${dep[@]+"${dep[@]}"} cluster/full_run_watchdog.sh) \
    || { echo "watchdog sbatch failed: nothing submitted or cancelled" >&2; exit 1; }
fi

# BALANCE and its settings reach the jobs only when on, so BALANCE=0 submits exactly the old environment.
bal=()
if [ "$BALANCE" = 1 ]; then
  bal=(BALANCE=1)
  [ -n "${BALANCE_PRIOR:-}" ] && bal+=("BALANCE_PRIOR=$BALANCE_PRIOR")
  [ -n "${BALANCE_ARGS:-}" ] && bal+=("BALANCE_ARGS=$BALANCE_ARGS")
fi
if [ "$REPLACE_PENDING" = 1 ] && [ "$BALANCE" = 0 ]; then bal=(BALANCE=0); fi  # switching back: override an inherited 1
ids=()
for _ in $(seq "$CHAIN"); do
  job=(env MAX_ITER="$ITERS" RUN="$RUN" RUNS_DIR="$RUNS_DIR" WIDE="$WIDE" ITERS="$PER_JOB" GAMES_PER_ITER="$GAMES_PER_ITER"
       WAVE2_EXTRA="$WAVE2_EXTRA" WORKERS="$WORKERS" ${SEGMENT_GAMES:+"SEGMENT_GAMES=$SEGMENT_GAMES"} ${bal[@]+"${bal[@]}"}
       sbatch --parsable -J "full_$RUN" --dependency=singleton cluster/wave2_iterate.sbatch)
  if [ "$DRY_RUN" = 1 ]; then echo "DRY_RUN ${job[*]}" >&2; ids+=("dry$RANDOM")
  else
    id=$("${job[@]}") || { echo "sbatch failed after submitting: ${ids[*]:-none}; nothing was cancelled (REPLACE_PENDING: the old pending jobs stay, the new ones wait behind them)" >&2
                           [ "${#ids[@]}" -gt 0 ] && printf '%s\n' "${ids[@]}" >> "$RUNS_DIR/$RUN/jobs.txt"
                           if [ "$REPLACE_WD" = 1 ]; then sc "$W" || echo "cancel the new watchdog $W by hand" >&2; fi
                           exit 1; }
    ids+=("$id")
  fi
done
if [ "$REPLACE_WD" = 1 ]; then
  :  # submitted above, before the chain
elif [ "$REPLACE_PENDING" = 1 ]; then
  W="none submitted (the run's existing watchdog is kept; NEW_WATCHDOG=1 replaces it)"
else
  W=$(RUN="$RUN" RUNS_DIR="$RUNS_DIR" GPU_HOURS="$GPU_HOURS" sb --parsable cluster/full_run_watchdog.sh)
fi
[ "$DRY_RUN" = 1 ] || printf '%s\n' "${ids[@]}" >> "$RUNS_DIR/$RUN/jobs.txt"
if [ "$REPLACE_PENDING" = 1 ]; then
  # Submitted first (above), cancelled after: with --dependency=singleton the new jobs wait behind the old pending ones,
  # so no old job can start after a new one, and a failed squeue/sbatch before this point leaves the old chain intact.
  # Only jobs still PENDING are cancelled; one that started meanwhile is left alone and reported.
  sc -t PENDING "${PENDING_IDS[@]}" || echo "warning: scancel failed; cancel ${PENDING_IDS[*]} by hand (the new chain waits behind them)" >&2
  if [ "$DRY_RUN" != 1 ]; then
    started=$(squeue -h -u "$USER" -j "$(IFS=,; echo "${PENDING_IDS[*]}")" -t RUNNING -o "%i" 2>/dev/null | tr '\n' ' ') || started=""
    [ -n "$started" ] && echo "warning: $started started before it could be cancelled; it runs with the old settings" >&2
    if [ "$was" != "$settings" ]; then
      printf '%s\treplaced %s\t%s\n' "$(date +%FT%T)" "${PENDING_IDS[*]}" "$was" >> "$RUNS_DIR/$RUN/launch_settings.history"
      echo "$settings" > "$RUNS_DIR/$RUN/launch_settings.txt"
    fi
  fi
  echo "cancelled pending: ${PENDING_IDS[*]}"
  if [ "$REPLACE_WD" = 1 ] && [ "${#OLD_WD[@]}" -gt 0 ]; then
    # The new watchdog waits (afterany) on the old one: cancelling the old one starts it. If that fails, the new one
    # would start only when the old one ends, and the old one hands over to a copy of itself before that: cancel the
    # new one instead, so the run never has two.
    if sc "${OLD_WD[@]}"; then
      echo "cancelled old watchdog: ${OLD_WD[*]} (new watchdog $W starts now)"
      [ "$DRY_RUN" = 1 ] || printf '%s\twatchdog %s replaced by %s\tgpu_hours %s monitor %s anchors %s cap_since %s\n' "$(date +%FT%T)" \
        "${OLD_WD[*]}" "$W" "${GPU_HOURS:-none}" "$WD_MONITOR" "${WD_ANCHORS:-off}" "$WD_CAP_SINCE" >> "$RUNS_DIR/$RUN/launch_settings.history"
    else
      echo "warning: scancel of the old watchdog ${OLD_WD[*]} failed: cancelling the new one ($W); the old one keeps" \
           "watching WITHOUT the new GPU-hour cap" >&2
      sc "$W" || echo "warning: cancel the new watchdog $W by hand" >&2
      W="$W (cancelled)"
    fi
    if [ "$DRY_RUN" != 1 ]; then
      # An old watchdog that handed over to a copy of itself between the lookup and the cancel leaves that copy.
      others=$(squeue -h -u "$USER" -n full_watchdog -o "%i" 2>/dev/null | grep -vxF -e "${W%% *}" $(printf -- '-e %s ' "${OLD_WD[@]}") | tr '\n' ' ' || true)
      if [ -n "$others" ]; then
        echo "note: other full_watchdog jobs queued: $others(check that none watches full_$RUN: their Slurm logs)" >&2
      fi
    fi
  fi
fi
echo "chain: ${ids[0]} .. ${ids[${#ids[@]}-1]} (${#ids[@]} jobs, full_$RUN)    watchdog: $W"
[ "$WORKERS" -gt 0 ] && echo "workers: $WORKERS per iteration (sp_$RUN, $RUNS_DIR/$RUN/workers.tsv), $SEGMENT_GAMES games per segment, $PER_JOB iterations per job"
[ -n "$GPU_HOURS" ] && echo "cap: $GPU_HOURS GPU-hours over full_$RUN + sp_$RUN (the watchdog stops the run there)"
[ "$BALANCE" = 1 ] && echo "search balance: on (prior ${BALANCE_PRIOR:-$RUNS_DIR/$RUN/balance/prior.json}; none = holds theta0; cluster/search_balance.py status $RUNS_DIR/$RUN)"
echo "results: $RUNS_DIR/$RUN/results.tsv    log: $RUNS_DIR/$RUN/runs/rl-loop.log    watchdog: $RUNS_DIR/$RUN/watchdog.log"
