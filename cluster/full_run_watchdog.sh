#!/bin/bash
#SBATCH -J full_watchdog -A paceship-bughouse -q inferno
#SBATCH -N1 -n1 --mem=2G -t 24:00:00
#SBATCH -o full_watchdog_%j.log
# Stops the full run (cluster/launch_full_run.sh) on failure, so GPU hours are not spent past a bad result. CPU only.
# Every 30 minutes while a full_<RUN> job is left:
#   crash    a full_<RUN> job FAILED or ran out of memory (the next job would repeat it)  -> cancel the chain
#   gate     the last finished iteration scores under 0.40 against it13 (results.tsv vs_it13) -> cancel the chain
#   workers  3 or more self-play workers (sp_<RUN>, launch_full_run.sh WORKERS=K) FAILED or ran out of memory -> cancel
#   cap      with GPU_HOURS set, the GPU-hours of full_<RUN>, sp_<RUN>, mon_<RUN>_* and am_<RUN>_* jobs since the run
#            began reach it -> stop: STOP_WORKERS (no new workers, monitor or anchor jobs), rl-loop.stop (the main job
#            exits before its next stage), pending jobs cancelled; running jobs finish their segment or stage.
#            The league pilot's jobs count too (cluster/launch_league_pilot.sh, spec v7 7.3): gate2 arms g2_<RUN>_*
#            (cluster/gate2_match.sh, which also stops submitting on STOP_WORKERS), the think check tg_<RUN> and the
#            engine build lb_<RUN>; their pending jobs are cancelled at the cap. A run without them counts as before.
# The training monitor (cluster/README.md, "Training monitor"; MONITOR=0 switches it off), every pass:
#   probe    an iteration with stages/train.done and no metrics/itN.probe.json gets cluster/monitor_iteration.sbatch
#            (-J mon_<RUN>_itN; at most MONITOR_TRIES (2) jobs an iteration, listed in metrics/monitor_jobs.tsv)
#   anchors  only with ANCHORS set (opt-in: about 21 GPU-hours per anchored iteration, e.g. ANCHORS="hybrid-it13
#            upstream-it13"): a finished iteration due for anchors (cluster/launch_anchor_match.sh, ANCHOR_EVERY) gets
#            its 120 s anchor matches, and finished parts are merged; only iterations from anchors/FROM on (written at
#            the first pass: the newest due finished iteration then, so turning anchors on mid-run does not backfill;
#            ANCHOR_FROM=N overrides)
#   metrics  cluster/run_metrics.py on the login node when a stage marker, probe or anchor result is newer than the
#            last collection, then --alerts; new lines of alerts.tsv are copied into watchdog.log. Alerts never cancel
#            anything: a stop-level alert is a recommendation for a person.
# While monitor or anchor jobs remain after the chain has ended, the watchdog stays to submit and merge them.
# Every decision goes to RUNS_DIR/RUN/watchdog.log. It resubmits itself before its 24 h limit while jobs remain.
#   RUN=full1 RUNS_DIR=runs/full [GPU_HOURS=900 [CAP_SINCE=2026-11-22T12:28:44]] sbatch cluster/full_run_watchdog.sh
# DRY_RUN=1 prints every sbatch/scancel instead of running it (local checks with fake squeue/sacct on PATH).
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(git rev-parse --show-toplevel)}" || exit 1
RUN="${RUN:?set RUN}"
RUNS_DIR="${RUNS_DIR:-runs/full}"
POLL_S="${POLL_S:-1800}"
LIFE_S="${LIFE_S:-84600}"   # 23.5 h, then hand over to a fresh copy
# Only jobs that ended after the first watchdog started count, so a failure before a RESUME=1 relaunch does not.
SINCE="${SINCE:-$(date +%Y-%m-%dT%H:%M:%S)}"
GPU_HOURS="${GPU_HOURS:-}"
# The GPU-hour cap counts jobs from CAP_SINCE (default SINCE). A watchdog that replaces another one mid-run
# (launch_full_run.sh REPLACE_PENDING=1 NEW_WATCHDOG=1) gets the run's start here, so the cap covers the whole run,
# while its crash checks (SINCE) start now.
CAP_SINCE="${CAP_SINCE:-$SINCE}"
MONITOR="${MONITOR:-1}"
# Anchors are opt-in: a watchdog that picks this code up by its self-handover (no ANCHORS in its environment) must
# not start spending ~21 GPU-hours per anchored iteration on the run's account without someone asking for it.
ANCHORS="${ANCHORS:-}"
MONITOR_TRIES="${MONITOR_TRIES:-2}"
DRY_RUN="${DRY_RUN:-0}"
PY=.venv/bin/python
ROOT="$RUNS_DIR/$RUN"
LOG="$ROOT/watchdog.log"
mkdir -p "$ROOT"

say() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }
sb() { if [ "$DRY_RUN" = 1 ]; then echo "DRY_RUN sbatch $*" >&2; echo "dry$RANDOM"; else sbatch "$@"; fi; }
sc() { if [ "$DRY_RUN" = 1 ]; then echo "DRY_RUN scancel $*" >&2; else scancel "$@"; fi; }
queued() { squeue -h -u "$USER" -n "full_$RUN,sp_$RUN" -o "%i" | head -1; }
# Monitor and anchor jobs have one name per iteration (and part), so they are matched by prefix.
side_jobs() { squeue -h -u "$USER" -o "%i %j %T" "$@" | awk -v m="mon_${RUN}_it" -v a="am_${RUN}_it" \
  'index($2, m) == 1 || index($2, a) == 1 { print $1 }'; }
# The league pilot's gate2 arms (one name per network, arm and part), think check and build (cancelled at the cap only).
pilot_jobs() { squeue -h -u "$USER" -o "%i %j %T" "$@" | awk -v g="g2_${RUN}_" -v t="tg_${RUN}" -v b="lb_${RUN}" \
  'index($2, g) == 1 || $2 == t || $2 == b { print $1 }'; }
cancel() { say "CANCEL full_$RUN and sp_$RUN : $1"; sc -u "$USER" -n "full_$RUN"; sc -u "$USER" -n "sp_$RUN"; }
gpu_hours() {  # GPU-hours of this run's jobs since SINCE (running jobs counted so far); monitor, anchors, pilot included
  sacct -n -X -P -u "$USER" -S "$CAP_SINCE" -o JobName,ElapsedRaw,AllocTRES \
    | awk -F'|' -v r="$RUN" '$1 == "full_" r || $1 == "sp_" r || index($1, "mon_" r "_it") == 1 || index($1, "am_" r "_it") == 1 ||
                             index($1, "g2_" r "_") == 1 || $1 == "tg_" r || $1 == "lb_" r {
        if (match($3, /gres\/gpu=[0-9]+/)) s += $2 * substr($3, RSTART + 9, RLENGTH - 9) } END { printf "%.1f", s / 3600 }'
}
stop_run() {
  say "STOP at the cap: $1"
  echo "$1" > "$ROOT/STOP_WORKERS"
  mkdir -p "$ROOT/runs" && touch "$ROOT/runs/rl-loop.stop"
  sc -u "$USER" -t PENDING -n "full_$RUN"; sc -u "$USER" -t PENDING -n "sp_$RUN"
  pend=$( { side_jobs -t PENDING; pilot_jobs -t PENDING; } | tr '\n' ' ')
  # shellcheck disable=SC2086  # word splitting: one id each
  [ -n "$pend" ] && sc $pend
}

iters() {  # iteration numbers present, ascending
  for d in "$ROOT"/runs/rl-it*; do n=${d##*rl-it}; case "$n" in ''|*[!0-9]*) ;; *) echo "$n" ;; esac; done | sort -n
}
anchor_from() {  # first iteration that gets anchors (see the header); fixed at the first pass
  if [ -n "${ANCHOR_FROM:-}" ]; then echo "$ANCHOR_FROM"; return; fi
  if [ ! -f "$ROOT/anchors/FROM" ]; then
    mkdir -p "$ROOT/anchors"
    from=1
    for n in $(iters); do
      [ -f "$ROOT/runs/rl-it$n/stages/iteration.done" ] && [ $((n % ${ANCHOR_EVERY:-2})) -eq 0 ] && from=$n
    done
    echo "$from" > "$ROOT/anchors/FROM"
    say "anchors from it$from on (anchors/FROM; ANCHOR_FROM=N or launch_anchor_match.sh by hand for earlier ones)" >/dev/null
  fi
  cat "$ROOT/anchors/FROM"
}
monitor_pass() {
  [ "$MONITOR" = 1 ] || return 0
  local names stopped=0 n st tries id from stamp before
  [ -e "$ROOT/STOP_WORKERS" ] && stopped=1
  # A failed squeue is not an empty queue: submitting then would start a second job on an iteration still running.
  names=$(squeue -h -u "$USER" -o "%j") || { say "squeue failed; monitor submits nothing this pass"; stopped=1; }
  from=0; [ -n "$ANCHORS" ] && from=$(anchor_from)
  mkdir -p "$ROOT/metrics/logs"
  for n in $(iters); do
    st="$ROOT/runs/rl-it$n/stages"
    if [ "$stopped" = 0 ] && [ -f "$st/train.done" ] && [ ! -f "$ROOT/metrics/it$n.probe.json" ] \
       && ! grep -qxF "mon_${RUN}_it$n" <<<"$names"; then
      tries=$(awk -F'\t' -v n="$n" '$1 == n' "$ROOT/metrics/monitor_jobs.tsv" 2>/dev/null | wc -l | tr -d ' ')
      if [ "$tries" -lt "$MONITOR_TRIES" ]; then
        id=$(sb --parsable -J "mon_${RUN}_it$n" -o "$ROOT/metrics/logs/mon_it${n}_%j.log" \
               cluster/monitor_iteration.sbatch "$ROOT" "$n")
        [ "$DRY_RUN" = 1 ] || printf '%s\t%s\t%s\n' "$n" "$id" "$(date +%FT%T)" >> "$ROOT/metrics/monitor_jobs.tsv"
        say "monitor it$n: submitted $id"
      elif [ "$tries" -eq "$MONITOR_TRIES" ]; then
        printf '%s\tgave-up\t%s\n' "$n" "$(date +%FT%T)" >> "$ROOT/metrics/monitor_jobs.tsv"
        say "monitor it$n: $tries jobs and no probe; not resubmitted (metrics/logs)"
      fi
    fi
    if [ -n "$ANCHORS" ] && [ -f "$st/iteration.done" ] && [ "$n" -ge "$from" ] && [ ! -e "$ROOT/anchors/it$n/done" ]; then
      # at the cap (or squeue down): merge what has finished, submit nothing
      ANCHORS="$ANCHORS" NO_SUBMIT=$stopped DRY_RUN=$DRY_RUN bash cluster/launch_anchor_match.sh "$ROOT" "$n" 2>&1 \
        | while IFS= read -r line; do say "anchors: $line"; done
    fi
  done
  [ -f cluster/run_metrics.py ] || return 0
  stamp="$ROOT/metrics/.collected"
  if [ ! -f "$stamp" ] || [ -n "$(find "$ROOT"/runs/rl-it*/stages "$ROOT/metrics" "$ROOT/anchors" -newer "$stamp" \
        \( -name '*.done' -o -name '*.probe.json' -o -name summary.json \) 2>/dev/null | head -1)" ]; then
    touch "$stamp"
    before=0; [ -f "$ROOT/alerts.tsv" ] && before=$(wc -l < "$ROOT/alerts.tsv")
    $PY cluster/run_metrics.py "$ROOT" >/dev/null 2>>"$LOG" || say "run_metrics.py failed (stderr above)"
    $PY cluster/run_metrics.py --alerts "$ROOT" >/dev/null 2>>"$LOG" || say "run_metrics.py --alerts failed (stderr above)"
    tail -n +"$((before + 1))" "$ROOT/alerts.tsv" 2>/dev/null | while IFS= read -r line; do say "ALERT $line"; done
  fi
}

if [ -n "$GPU_HOURS" ]; then
  say "watching full_$RUN (monitor $MONITOR, anchors ${ANCHORS:-off}, cap $GPU_HOURS GPU-hours from $CAP_SINCE, job ${SLURM_JOB_ID:-none})"
else
  say "watching full_$RUN (monitor $MONITOR, anchors ${ANCHORS:-off})"
fi
start=$SECONDS
while :; do
  monitor_pass
  # The cap comes before the cancelling checks, so a pass that also cancels still leaves the stop files behind.
  if [ -n "$GPU_HOURS" ] && [ ! -e "$ROOT/STOP_WORKERS" ]; then
    used=$(gpu_hours)
    say "GPU-hours $used of $GPU_HOURS"
    if awk -v u="$used" -v c="$GPU_HOURS" 'BEGIN { exit !(u >= c) }'; then stop_run "$used of $GPU_HOURS GPU-hours used"; fi
  fi
  if [ -z "$(queued)" ]; then
    [ "$MONITOR" = 1 ] && [ -n "$(side_jobs)" ] || break
  else
    bad=$(sacct -n -X -u "$USER" -S "$SINCE" --name "full_$RUN" -o JobID,State | grep -E 'FAILED|OUT_OF_ME' | head -1)
    if [ -n "$bad" ]; then cancel "a job failed: $bad"; break; fi
    wbad=$(sacct -n -X -u "$USER" -S "$SINCE" --name "sp_$RUN" -o JobID,State | grep -cE 'FAILED|OUT_OF_ME' || true)
    if [ "$wbad" -ge 3 ]; then cancel "$wbad self-play workers failed"; break; fi
    tsv="$ROOT/results.tsv"
    if [ -f "$tsv" ] && $PY -c '
import csv, sys
rows = [r for r in csv.DictReader(open(sys.argv[1]), delimiter="\t") if r.get("vs_it13")]
sys.exit(0 if rows and float(rows[-1]["vs_it13"]) < 0.40 else 1)' "$tsv"; then
      cancel "last iteration under 0.40 vs it13: $(tail -1 "$tsv")"; break
    fi
  fi
  if [ $((SECONDS - start)) -ge "$LIFE_S" ]; then
    say "handing over to a fresh watchdog"
    RUN=$RUN RUNS_DIR=$RUNS_DIR SINCE=$SINCE CAP_SINCE=$CAP_SINCE GPU_HOURS=$GPU_HOURS MONITOR=$MONITOR ANCHORS="$ANCHORS" \
      sb cluster/full_run_watchdog.sh >/dev/null
    exit 0
  fi
  sleep "$POLL_S"
done
say "no full_$RUN jobs left; done"
