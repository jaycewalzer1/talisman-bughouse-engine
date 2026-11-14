#!/bin/bash
# Submits the checks that come before the next long training run, one GPU per job. Each answers one question:
#   bash cluster/launch_checks.sh learner    # can the learner unlearn a known bad habit, and at which step size?
#   bash cluster/launch_checks.sh probe      # does waiting ever beat moving, with no learning involved?
#   bash cluster/launch_checks.sh features   # what do the feature groups do to an untrained policy's score?
#   bash cluster/launch_checks.sh all
# Running "learner" again continues any run that has not finished. "features" refuses to overwrite a result.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
WHAT="${1:?say which check: learner, probe, features or all}"

learner() {  # name, step size, pairs per update, updates, seed
  NAME="learner/$1" LR="$2" PAIRS="$3" ITERATIONS="$4" SEED="$5" FEATURES="" BASELINE=eval EXPLORATION=0.15 \
    THINK_LOGIT=2 SNAPSHOT_EVERY=0 sbatch -J "$1" cluster/train.sbatch
}

if [ "$WHAT" = learner ] || [ "$WHAT" = all ]; then
  # Every run starts where wave1 did: 35% of moves off the normal think budget, which loses on time. Yesterday's
  # wave1 (step size 3e-4, think time flat at 1.23) is the comparison. 3000 games per run, about 7.5 hours.
  #        name         step   pairs  updates  seed
  learner  lr1e-3_s0    1e-3   50     30       0
  learner  lr1e-3_s1    1e-3   50     30       1
  learner  lr3e-3_s0    3e-3   50     30       0
  learner  lr3e-3_s1    3e-3   50     30       1
  learner  small_s0     1e-3   10     150      0   # the same games in five times as many, smaller updates
  learner  small_s1     1e-3   10     150      1
fi

if [ "$WHAT" = probe ] || [ "$WHAT" = all ]; then
  # Positions come from yesterday's featureless run, 100 recorded games against stock per job.
  for part in 12 34 56 78; do
    RUN="runs/waves/signal/it000[$part]" OUT="runs/probe/signal_$part" POSITIONS=2 ROLLOUTS=4 SEED="$part" \
      sbatch -J "probe$part" cluster/probe.sbatch
  done
fi

if [ "$WHAT" = features ] || [ "$WHAT" = all ]; then
  # Untrained policy, 1600 games against stock each, same openings and seeds for every row. About 4 hours a job.
  evaluate() { NAME="eval/untrained_$1" FEATURES="$2" sbatch -J "$1" cluster/eval.sbatch; }
  evaluate none   ""
  evaluate tempo  tempo
  evaluate mate   mate,danger
  evaluate serve  serve
  evaluate all    tempo,mate,danger,serve
fi
squeue -u "$USER"
