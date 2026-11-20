#!/bin/bash
# Submits the checks that come before the next long training run, one GPU per job. Each answers one question:
#   bash cluster/launch_checks.sh learner    # can the learner unlearn a known bad habit, and at which step size?
#   bash cluster/launch_checks.sh probe      # does waiting ever beat moving, with no learning involved?
#   bash cluster/launch_checks.sh features   # what do the feature groups do to an untrained policy's score?
#   bash cluster/launch_checks.sh matches    # stock only: what is each team setup worth, and which think budget wins?
#   bash cluster/launch_checks.sh rules      # hand-written: does waiting for a mating piece win, alone or when asked for?
#   bash cluster/launch_checks.sh lend       # how much of the partner's idle machine is it best to borrow?
#   bash cluster/launch_checks.sh beat       # can one Hivemind that manages its clock beat stock Hivemind? (20 jobs)
#   bash cluster/launch_checks.sh controls   # three more same_* seeds: the noise floor of a 1600-game row (3 jobs)
#   bash cluster/launch_checks.sh harness    # equal-clock harness checks with measured per-arm ms_per_node (8 jobs)
#   bash cluster/launch_checks.sh wave2      # wave 2 (durative tree on upstream v3) vs it13, ablations (11 jobs)
#   bash cluster/launch_checks.sh wave2c     # wave 2c continuity fixes vs it13, one fix at a time (6 jobs, 7 with WAVE2C_REF=1)
#   bash cluster/launch_checks.sh all
# Running "learner" again continues any run that has not finished. "features" refuses to overwrite a result.
set -euo pipefail
# sbatch has no SBATCH_DEPENDENCY input variable, so pass it on the command line: SBATCH_DEPENDENCY=afterok:<job> holds every job below.
sbatch() { command sbatch ${SBATCH_DEPENDENCY:+--dependency=$SBATCH_DEPENDENCY} "$@"; }
cd "$(git rev-parse --show-toplevel)"
WHAT="${1:?say which check: learner, probe, features, matches, rules, lend, beat, controls, harness, wave2, wave2c or all}"

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
  beat() {  # name, nodes, opponent nodes, contender options, pairs, clock ms, opponent options, seed
    NAME="beat/$1" SETUP=centralized LEND=0 NODES="$2" OPPONENT=stock-centralized OPPONENT_NODES="$3" \
      CONTENDER_KW="$4" OPPONENT_KW="${7:-$SER}" PAIRS="$5" BASE_MS="$6" SEED="${8:-0}" \
      sbatch -J "$1" cluster/eval.sbatch
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
SER='{"serial": true}'
if [ "$WHAT" = controls ]; then  # not part of "all": 3 jobs
  # Three more seeds of the identical-teams control (audit D3 item 5). Their pooled spread with the two existing
  # same_* rows is the noise floor every 1600-game row is read against; publish it in docs/beat_stock.md.
  #     name                 nodes opp  contender options  pairs clock  opponent options  seed
  for seed in 1 2 3; do
    NAME="beat/same_100_v100_s$seed" SETUP=centralized LEND=0 NODES=100 OPPONENT=stock-centralized \
      OPPONENT_NODES=100 CONTENDER_KW="$SER" OPPONENT_KW="$SER" PAIRS=800 BASE_MS=60000 SEED="$seed" \
      sbatch -J "same_s$seed" cluster/eval.sbatch
  done
fi
if [ "$WHAT" = harness ]; then  # not part of "all": 8 jobs
  # The equal-clock harness (docs/beat_stock.md "Headline metric from 2026-11-17", spec v3 contract 5): each arm is
  # charged the nodes it searched x its own ms_per_node, measured on the job's GPU (MEASURE=1). Needs WP-C's
  # charge="searched" in arena/players/hivemind.py. SPRT=0,20 stops a row once it is decided.
  CHG='"serial": true, "charge": "searched"'
  harness() {  # name, contender options, opponent options, clock ms, pairs, extra env
    # Exports in a subshell, not "env ... sbatch", which would skip the sbatch function (SBATCH_DEPENDENCY).
    (export NAME="harness/$1" SETUP=centralized LEND=0 NODES=100 OPPONENT=stock-centralized OPPONENT_NODES=100 \
       CONTENDER_KW="$2" OPPONENT_KW="$3" BASE_MS="$4" PAIRS="$5" MEASURE=1
     [ -z "${6:-}" ] || export ${6}
     sbatch -J "h_$1" cluster/eval.sbatch)
  }
  #        name               contender options                                 opponent options                    clock   pairs
  # identical arms, ms_per_node measured once for the shared binary+network: must read 0.50
  harness  same_meas_120      "{$CHG, \"clock_k\": 110}"                          "{$CHG, \"clock_k\": 110}"            120000  800
  # identical network, one arm modelled 55x faster (0.1 vs 5.5 ms/node, explicit values): the fast side must win
  harness  speed_01_v55       "{$CHG, \"clock_k\": 110, \"ms_per_node\": 0.1}"    "{$CHG, \"clock_k\": 110, \"ms_per_node\": 5.5}" 60000 200 "SPRT=0,20"
  # the c25/c50 rows of "beat" again, now charged the nodes actually searched (audit eval 3)
  harness  c25_v100_searched  "{$CHG}"                                          "{$CHG}"                            60000   800 "NODES=25"
  harness  c50_v100_searched  "{$CHG}"                                          "{$CHG}"                            60000   800 "NODES=50"
  # M0 parity arm: the widened network (clock build, stock rules, zero clock weights) vs stock it04 at equal clock
  # must read 0.50 +- the row's range before any training (audit D1; doubles as the parity gate)
  W="third_party/hivemind/engine/models/hivemind-clock-v3-fp32.onnx"
  harness  m0_widened_120     "{$CHG, \"clock_k\": 110, \"binary\": \"hivemind-clock\", \"model\": \"$PWD/$W\"}" "{$CHG, \"clock_k\": 110}" 120000 800
  harness  m0_widened_180     "{$CHG, \"clock_k\": 110, \"binary\": \"hivemind-clock\", \"model\": \"$PWD/$W\"}" "{$CHG, \"clock_k\": 110}" 180000 800
  # M0 third arm: stock it04 at equal clock against upstream v3 twin-s-clock under its own time manager (WP-G's
  # player kind "upstream"; its ms_per_node is the explicit value in arena/pools/upstream_v3.json, not measured
  # here because the measurement speaks our fork's UCI dialect). Needs the upstream build and setups.py hook.
  harness  m0_upstream_120    "{$CHG, \"clock_k\": 110}"                          "{}"                                120000  800 "POOL=arena/pools/upstream_v3.json OPPONENT=upstream-v3 OPPONENT_NODES="
  harness  m0_upstream_180    "{$CHG, \"clock_k\": 110}"                          "{}"                                180000  800 "POOL=arena/pools/upstream_v3.json OPPONENT=upstream-v3 OPPONENT_NODES="
fi
if [ "$WHAT" = wave2 ]; then  # not part of "all": 11 jobs. Read docs/wave2_tests.md before the numbers.
  # Every row: the pool's wave2 contender (arena/pools/wave2.json: clock_k 110, serial, charge "searched") against
  # one opponent at equal clock, each arm charged its searched nodes x its own ms_per_node measured on the job's GPU
  # (MEASURE=1, 800 nodes, per binary + flags), bughouse openings, arena rules v3. Needs cluster/build.sh's wave2
  # build ("wave2 check: PASS"); SBATCH_DEPENDENCY=afterok:<build job> holds the rows behind it.
  W2=(POOL=arena/pools/wave2.json SETUP=pool OPENINGS=bughouse RULES=v3 MEASURE=1 MEASURE_NODES=800
      "WARM_KINDS=upstream wave2")
  w2() {  # name, opponent, contender options (wave 2 flag overrides), clock ms, pairs, extra env
    # A subshell with exports, not "env ... sbatch": env would run the sbatch binary and skip the function above
    # that adds SBATCH_DEPENDENCY.
    (export NAME="wave2/$1" OPPONENT="$2" CONTENDER_KW="$3" BASE_MS="$4" PAIRS="$5" "${W2[@]}"
     [ -z "${6:-}" ] || export ${6}
     sbatch -J "w2_$1" cluster/eval.sbatch)
  }
  #   name             opponent        contender options                                clock   pairs  extra
  # the headline: wave 2 (every flag on but MateRaceByLanding) against stock upstream it13 (D1 measured it04 at 0.165 / 0.139 here)
  w2  vs_it13_120      upstream-it13   '{}'                                             120000  800    SPRT=0,20
  w2  vs_it13_180      upstream-it13   '{}'                                             180000  800    SPRT=0,20
  # the ablations' reference: the headline's settings without SPRT. vs_it13_120 stops once decided and an SPRT-stopped
  # score is biased by the stopping, so the ablations are read against this fixed-length row instead (review finding)
  w2  ref_120          upstream-it13   '{}'                                             120000  600
  # ablations at 120 s, one switch each (racewin switches one on), read as the difference from ref_120. No SPRT: it would stop a row on
  # "beats it13 or not", which is not the question, and leave too few games to compare. 600 pairs fit one 8 h job.
  # nodur keeps the gate: without the durative clock the engine always gates (v3's lockstep tree has no sit-off
  # rule, and with both teams free to pass the search never finished; integration 2026-11-19)
  w2  nodur            upstream-it13   '{"durative_tree": false, "sit_gate": true}'     120000  600
  w2  sitgate          upstream-it13   '{"sit_gate": true}'                             120000  600
  w2  instant          upstream-it13   '{"instant_transfer": true}'                     120000  600
  # racewin = landing-time mate races AND no root mate provers (the option switches them off; agent.cc run_search):
  # it does not isolate the landing rule
  w2  racewin          upstream-it13   '{"mate_race_by_landing": true}'                 120000  600
  w2  noflag           upstream-it13   '{"flag_loss": false}'                           120000  600
  # control: wave 2 against itself must read 0.50 (no SPRT: the full 1600 games are the noise floor)
  w2  same_120         wave2-self      '{}'                                             120000  800
  # sanity: wave 2 against stock it04 (our fork) must win clearly
  w2  vs_it04_120      it04            '{}'                                             120000  200    SPRT=0,20
  # stock it13 at its own 800 fixed nodes (D1's opponent setting) instead of clock_k 110
  w2  vs_it13_800n_120 upstream-it13-800n '{}'                                          120000  800    SPRT=0,20
fi
if [ "$WHAT" = wave2c ]; then  # not part of "all": 6 jobs (7 with WAVE2C_REF=1). Read docs/wave2_tests.md "Wave 2c rows".
  # Spec v5 (docs/custom_engine_spec_v5_continuity.md): team tree reuse (fix 1), joint team actions (fix 2) and plan
  # carry (fix 3), each a player option of the wave 2 arm that is off by default. Same common settings as "wave2"
  # (pool contender, equal clock, MEASURE=1 per binary + flags, so a fix that changes node cost is charged its own
  # ms_per_node), against stock it13 at 120 s, no SPRT. Needs the wave 2c build (cluster/build.sh, "wave2 check: PASS"
  # on the new engine commit); SBATCH_DEPENDENCY=afterok:<build job> holds the rows behind it. Fixes 1 and 3 keep the
  # engine's tree and plans between searches (no ucinewgame per search, one at each game's first search); fix 2 alone
  # keeps the per-search ucinewgame of ref_120 (arena/players/upstream.py WAVE2_STATE_FLAGS).
  W2=(POOL=arena/pools/wave2.json SETUP=pool OPENINGS=bughouse RULES=v3 MEASURE=1 MEASURE_NODES=800
      "WARM_KINDS=upstream wave2")
  w2c() {  # name, opponent, contender options (wave 2 flag overrides), clock ms, pairs, extra env
    # A subshell with exports, as w2 above: "env ... sbatch" would skip the sbatch function (SBATCH_DEPENDENCY).
    (export NAME="wave2c/$1" OPPONENT="$2" CONTENDER_KW="$3" BASE_MS="$4" PAIRS="$5" "${W2[@]}"
     [ -z "${6:-}" ] || export ${6}
     sbatch -J "w2c_$1" cluster/eval.sbatch)
  }
  #    name          opponent       contender options                                clock   pairs
  # all three fixes on top of the headline settings; the round's success mark is this row at or above c_nodur_120
  w2c  c_all_120     upstream-it13  '{"continuity": true}'                           120000  600
  # one fix each, read as the difference from ref_120 (0.227 in the 2026-11-19 PACE round) or c_ref_120
  w2c  c_reuse_120   upstream-it13  '{"team_tree_reuse": true}'                      120000  600
  w2c  c_joint_120   upstream-it13  '{"joint_plans": true}'                          120000  600
  w2c  c_carry_120   upstream-it13  '{"plan_carry": true}'                           120000  600
  # the durative-off, gate-on reference (PACE nodur 0.436), rerun on the new binary in the same round
  w2c  c_nodur_120   upstream-it13  '{"durative_tree": false, "sit_gate": true}'     120000  600
  # control: continuity against itself (pool opponent wave2c-self) must read 0.50; 800 pairs, the noise floor
  w2c  c_same_120    wave2c-self    '{"continuity": true}'                           120000  800
  # optional same-week durative-on reference: with every fix off the new binary is bit-identical to the one ref_120
  # ran (spec v5 tests 0.4, 0.6), so ref_120 stands unless the round wants it again
  if [ "${WAVE2C_REF:-0}" = 1 ]; then
    w2c  c_ref_120   upstream-it13  '{}'                                             120000  600
  fi
fi
squeue -u "$USER"
