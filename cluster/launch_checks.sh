#!/bin/bash
# Submits the checks that come before the next long training run, one GPU per job. Each answers one question:
#   bash cluster/launch_checks.sh learner    # can the learner unlearn a known bad habit, and at which step size?
#   bash cluster/launch_checks.sh probe      # does waiting ever beat moving, with no learning involved?
#   bash cluster/launch_checks.sh features   # what do the feature groups do to an untrained policy's score?
#   bash cluster/launch_checks.sh matches    # stock only: what is each team setup worth, and which think budget wins?
#   bash cluster/launch_checks.sh rules      # hand-written: does waiting for a mating piece win, alone or when asked for?
#   bash cluster/launch_checks.sh lend       # how much of the partner's idle machine is it best to borrow?
#   bash cluster/launch_checks.sh beat       # can one Hivemind that manages its clock beat stock Hivemind? (20 jobs)
#   bash cluster/launch_checks.sh all
# Running "learner" again continues any run that has not finished. "features" refuses to overwrite a result.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
WHAT="${1:?say which check: learner, probe, features, matches, rules, lend, beat or all}"

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
if [ "$WHAT" = matches ] || [ "$WHAT" = all ]; then
  # Stock Hivemind against stock Hivemind, no scheduler. The opponent is the one-program team at 200 nodes unless
  # the last column says otherwise. 1600 games a row; the 400-node row is shorter because its games are slow.
  match() {  # name, setup, lend, nodes, opponent, pairs
    NAME="match/$1" SETUP="$2" LEND="$3" NODES="$4" OPPONENT="$5" PAIRS="$6" sbatch -J "$1" cluster/eval.sbatch
  }
  #      name               setup        lend  nodes  opponent                 pairs
  # what each team setup is worth at equal compute; the first row should come out at 0.50
  match  central_200        centralized  0     200    stock-centralized        800
  match  silent_200         silent       0     200    stock-centralized        800
  match  sharing_200        sharing      0     200    stock-centralized        800
  match  lend_200           sharing      1     200    stock-centralized        800
  # which fixed think budget wins on a 60 second clock (yesterday the 100-node opponent beat us 67 to 33)
  match  central_50         centralized  0     50     stock-centralized        800
  match  central_100        centralized  0     100    stock-centralized        800
  match  central_400        centralized  0     400    stock-centralized        500
  match  lend_100           sharing      1     100    stock-centralized        800
  match  central_50_v100    centralized  0     50     stock-centralized-100n   800
  match  lend_100_v100      sharing      1     100    stock-centralized-100n   800
fi
if [ "$WHAT" = rules ] || [ "$WHAT" = all ]; then
  # Untrained scheduler with the mate inputs on, 1600 games against stock each. The first row is the comparison:
  # it pays for the same mate probes and does nothing with them.
  rule() { NAME="eval/rule_$1" FEATURES=mate RULE="$2" sbatch -J "rule_$1" cluster/eval.sbatch; }
  rule none  ""
  rule wait  mate_wait   # wait up to 5 seconds for the mating piece, in silence
  rule ask   mate_ask    # the same, and ask the partner, who captures the piece if it can
fi
if [ "$WHAT" = lend ] || [ "$WHAT" = all ]; then
  # Borrowing none or all of the partner's idle machine are the sharing_200 and lend_200 rows of "matches". This
  # adds half: the partner is slowed while its machine is out instead of being made to wait for it.
  NAME=match/half_200 SETUP=sharing LEND=1 LEND_SHARE=0.5 NODES=200 OPPONENT=stock-centralized PAIRS=800 \
    sbatch -J half_200 cluster/eval.sbatch
  # Untrained scheduler choosing the amount itself on every move: all about 4 times in 5, else half or none.
  # Compare with untrained_none from "features", which always borrows all.
  NAME=eval/untrained_lend FEATURES=lend sbatch -J lend cluster/eval.sbatch
fi
if [ "$WHAT" = beat ]; then  # not part of "all": 20 jobs
  # One centralized Hivemind against stock centralized Hivemind, both on ONE machine each ("serial": a search
  # asked for while the other seat's search runs waits for it), which is the fair accounting once the boards
  # stop moving in step. The decision rule for these results is in docs/beat_stock.md: read it before the numbers.
  SER='{"serial": true}'
  beat() {  # name, nodes, opponent nodes, contender options, pairs, clock ms, opponent options
    NAME="beat/$1" SETUP=centralized LEND=0 NODES="$2" OPPONENT=stock-centralized OPPONENT_NODES="$3" \
      CONTENDER_KW="$4" OPPONENT_KW="${7:-$SER}" PAIRS="$5" BASE_MS="$6" sbatch -J "$1" cluster/eval.sbatch
  }
  #     name                 nodes opp  contender options                                  pairs clock
  # controls: identical teams, should come out at 0.50
  beat  same_100_v100        100   100  "$SER"                                             800   60000
  beat  same_50_v50          50    50   "$SER"                                             800   60000
  # fixed budgets against each other: which one is stock's best on a 60 second clock?
  beat  c25_v100             25    100  "$SER"                                             800   60000
  beat  c50_v100             50    100  "$SER"                                             800   60000
  beat  c200_v100            200   100  "$SER"                                             800   60000
  beat  c25_v50              25    50   "$SER"                                             800   60000
  beat  c100_v50             100   50   "$SER"                                             800   60000
  # the time flag: 1% faster owns "ahead on time" all game; with and without the right to sit
  beat  flag_99_v100         99    100  "$SER"                                             800   60000
  beat  flag_49_v50          49    50   "$SER"                                             800   60000
  beat  nosit_100_v100       100   100  '{"serial": true, "max_consecutive_sits": 0}'      800   60000
  beat  flagnosit_99_v100    99    100  '{"serial": true, "max_consecutive_sits": 0}'      800   60000
  # clock-based think time: remaining clock / k. k 55, 110, 220 start at about 200, 100, 50 nodes
  beat  clock55_v100         100   100  '{"serial": true, "clock_k": 55}'                  800   60000
  beat  clock110_v100        100   100  '{"serial": true, "clock_k": 110}'                 800   60000
  beat  clock220_v100        100   100  '{"serial": true, "clock_k": 220}'                 800   60000
  beat  clock55_v50          50    50   '{"serial": true, "clock_k": 55}'                  800   60000
  beat  clock110_v50         50    50   '{"serial": true, "clock_k": 110}'                 800   60000
  beat  clock220_v50         50    50   '{"serial": true, "clock_k": 220}'                 800   60000
  # how much the old accounting (no machine queue, both teams) changes the clock result
  beat  clock110_v100_open   100   100  '{"clock_k": 110}'                                 800   60000 '{}'
  # a 3 minute clock: does the speed effect and the clock rule hold with more time? (fewer games: slower)
  beat  long_c100_v300       100   300  "$SER"                                             300   180000
  beat  long_clock110_v300   300   300  '{"serial": true, "clock_k": 110}'                 300   180000
fi
squeue -u "$USER"
