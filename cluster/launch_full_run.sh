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
# iteration has its games. SEGMENT_GAMES (500 with workers, else wave2_iterate's 2000) is what one claim plays;
# PER_JOB (1; 2 with workers) iterations per chained job. GPU_HOURS=N caps the run: the watchdog sums the GPU-hours of
# full_<RUN> and sp_<RUN> jobs and stops the run (no new workers, rl-loop.stop, pending jobs cancelled) at N.
# WORKERS=0 is the old single-GPU chain, bit for bit. cluster/gpu_budget.py prints what N the account can afford.
# A wave2-hivemind checkout off the pin is moved to the pin here when only Python differs between them (the engines
# stay as built); an engine change still needs the readiness round's build.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
RUN="${RUN:?set RUN, the run folder under runs/full (decide_full_run.py prints it)}"
WIDE="${WIDE:-0}"
CHAIN="${CHAIN:?set CHAIN, the number of chained 16 h jobs (decide_full_run.py prints it)}"
GAMES_PER_ITER="${GAMES_PER_ITER:-50000}"
ITERS="${ITERS:-10}"
RESUME="${RESUME:-0}"
WORKERS="${WORKERS:-0}"
GPU_HOURS="${GPU_HOURS:-}"
case "$WORKERS" in ''|*[!0-9]*) echo "WORKERS is a number" >&2; exit 1 ;; esac
if [ "$WORKERS" -gt 0 ]; then SEGMENT_GAMES="${SEGMENT_GAMES:-500}"; PER_JOB="${PER_JOB:-2}"; else PER_JOB="${PER_JOB:-1}"; fi
RUNS_DIR="${FULL_RUNS:-runs/full}"   # FULL_RUNS only for dry runs
case "$WIDE" in 0|1) ;; *) echo "WIDE is 0 or 1" >&2; exit 1 ;; esac

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
  fi
fi
if [ "$have" != "$pin" ]; then
  echo "$W2 is at $have, not the pin $pin: run the readiness round's build first (bash cluster/launch_wave2c.sh)" >&2
  exit 1
fi

settings="config ${CONFIG:-custom}  wave2_extra $WAVE2_EXTRA  wide $WIDE  games_per_iter $GAMES_PER_ITER  iters $ITERS  engine $pin"
if [ "$WORKERS" -gt 0 ]; then settings="$settings  workers $WORKERS  segment_games $SEGMENT_GAMES  per_job $PER_JOB"; fi
if [ -e "$RUNS_DIR/$RUN" ]; then
  if [ "$RESUME" != 1 ]; then
    echo "$RUNS_DIR/$RUN exists: pick a new RUN, or RESUME=1 to add CHAIN jobs to it" >&2
    exit 1
  fi
  if [ "$(cat "$RUNS_DIR/$RUN/launch_settings.txt" 2>/dev/null)" != "$settings" ]; then
    echo "RESUME=1 with other settings than $RUNS_DIR/$RUN/launch_settings.txt:" >&2
    echo "  was: $(cat "$RUNS_DIR/$RUN/launch_settings.txt" 2>/dev/null)" >&2
    echo "  now: $settings" >&2
    exit 1
  fi
else
  mkdir -p "$RUNS_DIR/$RUN"
  echo "$settings" > "$RUNS_DIR/$RUN/launch_settings.txt"
fi
echo "run $RUNS_DIR/$RUN: $settings, $CHAIN chained jobs"

ids=()
for _ in $(seq "$CHAIN"); do
  ids+=("$(env MAX_ITER="$ITERS" RUN="$RUN" RUNS_DIR="$RUNS_DIR" WIDE="$WIDE" ITERS="$PER_JOB" GAMES_PER_ITER="$GAMES_PER_ITER" \
    WAVE2_EXTRA="$WAVE2_EXTRA" WORKERS="$WORKERS" ${SEGMENT_GAMES:+"SEGMENT_GAMES=$SEGMENT_GAMES"} \
    sbatch --parsable -J "full_$RUN" --dependency=singleton cluster/wave2_iterate.sbatch)")
done
W=$(RUN="$RUN" RUNS_DIR="$RUNS_DIR" GPU_HOURS="$GPU_HOURS" sbatch --parsable cluster/full_run_watchdog.sh)
printf '%s\n' "${ids[@]}" >> "$RUNS_DIR/$RUN/jobs.txt"
echo "chain: ${ids[0]} .. ${ids[${#ids[@]}-1]} (${#ids[@]} jobs, full_$RUN)    watchdog: $W"
[ "$WORKERS" -gt 0 ] && echo "workers: $WORKERS per iteration (sp_$RUN, $RUNS_DIR/$RUN/workers.tsv), $SEGMENT_GAMES games per segment, $PER_JOB iterations per job"
[ -n "$GPU_HOURS" ] && echo "cap: $GPU_HOURS GPU-hours over full_$RUN + sp_$RUN (the watchdog stops the run there)"
echo "results: $RUNS_DIR/$RUN/results.tsv    log: $RUNS_DIR/$RUN/runs/rl-loop.log    watchdog: $RUNS_DIR/$RUN/watchdog.log"
