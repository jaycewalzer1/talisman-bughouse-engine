#!/bin/bash
#SBATCH -J monitor -A paceship-bughouse -q inferno
#SBATCH -N1 -n4 --mem=24G --gres=gpu:rtx_6000:1 -t 00:30:00
#SBATCH -o monitor_%x_%j.log
# The training monitor's GPU step for one iteration (cluster/README.md, "Training monitor"): builds the run's frozen
# probe set when it is missing (cluster/probe_net.py build, once per run, RUN_ROOT/probe), probes the iteration's
# network on it (probe_net.py eval -> RUN_ROOT/metrics/itN.probe.json), then collects every iteration's record
# (cluster/run_metrics.py -> metrics/itN.json) and its alerts (alerts.tsv). Needs stages/train.done of the iteration.
# The watchdog submits it as -J mon_<RUN>_itN; by hand, from the repo root:
#   sbatch -J mon_hybrid1_it3 cluster/monitor_iteration.sbatch runs/full/hybrid1 3
#   RUN_ROOT=runs/full/hybrid1 ITER=3 bash cluster/monitor_iteration.sbatch     # no GPU: DEVICE=cpu
# Settings: RUN_ROOT and ITER (arguments or environment), DEVICE (cuda; cpu off the cluster).
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(git rev-parse --show-toplevel)}" || exit 1
[ -n "${SLURM_JOB_ID:-}" ] && source cluster/env.sh

RUN_ROOT="${1:-${RUN_ROOT:?set RUN_ROOT, a run folder such as runs/full/hybrid1}}"
ITER="${2:-${ITER:?set ITER, the iteration number}}"
DEVICE="${DEVICE:-$([ -n "${SLURM_JOB_ID:-}" ] && echo cuda || echo cpu)}"
PY=.venv/bin/python                               # stdlib only: run_metrics.py
TPY=third_party/wave2-hivemind/.venv/bin/python   # torch + hivemind: probe_net.py
[ -f "$RUN_ROOT/runs/rl-it$ITER/stages/train.done" ] || { echo "$RUN_ROOT it$ITER: not trained yet" >&2; exit 1; }
echo "commit $(git rev-parse HEAD)  monitor $RUN_ROOT it$ITER  device $DEVICE"
echo "wave2 engine $(git -C third_party/wave2-hivemind rev-parse HEAD 2>/dev/null || echo none)"
status=0

# The probe set is built once and frozen. Monitor jobs for several iterations can start together, so one builds
# under a mkdir lock (atomic on the shared filesystem) and the others wait for it; a lock older than 40 min is a dead
# job's. probe_net.py build itself does nothing when the set is complete.
lock="$RUN_ROOT/probe.lock"
for _ in $(seq 120); do
  if mkdir "$lock" 2>/dev/null; then
    trap 'rmdir "$lock" 2>/dev/null' EXIT
    $TPY cluster/probe_net.py build "$RUN_ROOT" || { echo "probe set build failed" >&2; exit 1; }
    rmdir "$lock"; trap - EXIT
    break
  fi
  if [ -n "$(find "$lock" -maxdepth 0 -mmin +40 2>/dev/null)" ]; then rmdir "$lock" 2>/dev/null; continue; fi
  sleep 10
done
[ -d "$lock" ] && { echo "$lock still held after 20 min: probe set not built" >&2; exit 1; }

$TPY cluster/probe_net.py eval "$RUN_ROOT" "$ITER" --device "$DEVICE" || { echo "probe eval failed" >&2; status=1; }
# The record is written even without a probe (its network field is then null, with the reason under unavailable).
$PY cluster/run_metrics.py "$RUN_ROOT" || status=1
$PY cluster/run_metrics.py --alerts "$RUN_ROOT" || status=1
exit $status
