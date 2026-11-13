#!/bin/bash
# Submits the comparison runs as separate jobs, one GPU each, so they train side by side. One GPU is already full
# with a single run's three games at once (bench job 15539375), so sharing a job would only slow them all down.
#   bash cluster/launch_waves.sh            # 30 updates of 100 games each, per run
#   ITERATIONS=10 PAIRS=20 bash cluster/launch_waves.sh
# Running it again continues any run that has not finished.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
export ITERATIONS="${ITERATIONS:-30}" PAIRS="${PAIRS:-50}" SEED="${SEED:-0}"
GROUP="${GROUP:-waves}"

submit() {  # name, features, baseline, exploration, think logit
  NAME="$GROUP/$1" FEATURES="$2" BASELINE="$3" EXPLORATION="$4" THINK_LOGIT="$5" sbatch -J "$1" cluster/train.sbatch
}

#       name     features                  baseline  exploration  think
submit  control  ""                        mean      0.05         6   # the setup of the first 40-iteration run
submit  signal   ""                        eval      0.15         6   # only the stronger learning signal
submit  wave1    tempo                     eval      0.15         2   # thinking time: how forced is the move
submit  wave2    mate,danger               eval      0.15         6   # waiting: what a piece is worth to us or them
submit  wave3    serve                     eval      0.15         6   # acting on the partner's request
squeue -u "$USER"
