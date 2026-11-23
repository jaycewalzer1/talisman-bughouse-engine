# Elo hunt: where the hybrid wins and loses against upstream-it13 at 120 s

Date: 2026-11-23. GPU-free, read-only analysis of the wave 2c r4 long test on PACE.
Network in every game on both sides: upstream's twin-s-clock it13 (HF aminwoo/bughouse-twin-s-clock@edaa246, FP32 ONNX,
TensorRT on the RTX 6000). Our side: the wave2-hivemind engine with `durative_tree false, sit_gate true, hybrid true`
(HybridMode gate). Opponent: stock upstream aminwoo/hivemind v3.0.3.

Goal reminder: the final match is 200 games and needs a score **strictly above 0.6**. Today the hybrid is at 0.514.

## Bottom line

1. **No game is decided on time.** 0 flags in 3,956 games. Both sides end the game with about **89 s of 120 s still on
   the clock** (mean of a team's two seats; the lowest seat of ours averaged 88 s, never below 10 s in any game).
   The harness time manager (`clock_k` 110: each search gets remaining clock / 110, about 1.0 s) spends a quarter of
   the clock. Time is not lost; it is never used.
2. **The speed gap costs real Elo, and the 0.514 headline is partly measurement luck.** The arena charges
   `searched nodes x ms_per_node`, measured once per part from three 800-node searches. Our engine measured
   0.050-0.054 ms/node in every part; upstream measured 0.042 in three parts and 0.052-0.053 in two. In the two parts
   where upstream happened to measure as slow as us the hybrid scored **0.541** (n=579); in the three normal parts it
   scored **0.493** (n=766). At the true speed ratio the hybrid is about even with upstream, not +10 Elo.
3. **Sitting and the clock lead go with winning more strongly than anything else in the data.** Score by sits
   (ours minus theirs) runs from 0.15 at -3 to 0.84 at +3; score by share of turns spent ahead on the team clock runs
   from 0.27 to 0.70. These are correlations (winning positions make sitting attractive), but an older row (`runs/wave2/`) points to
   the causal side: upstream at a fixed 800 nodes (about 40 ms a move, 25 times fewer nodes) scored **the same**
   against our engine as upstream at clock_k 110, because it was ahead on the clock in 56 of 57 turns and sat 9.7
   times a game.
4. **There is no phase where we collapse.** Mean eval of both engines (same network) stays at 0.00 for the first 15 s
   of game time; the hybrid drifts slightly positive late (+0.02..+0.06 after 50 s), the nodur arm drifts negative
   from 20 s on (to -0.11 at 60 s). No opening, colour, board, or game length is out of line beyond noise.
5. **Ranked levers** (details at the end): (1) our own time allocation, i.e. playing faster to hold the clock
   lead, which switches off upstream's sitting (`can_sit = lead > 0`): unknown sign, the only lever with a plausible
   +50..+100 Elo, cheapest to test (a player option, no engine change); (2) engine speed: closing the 18% ms/node gap is
   worth about +20..+40 Elo; (3) sit policy (hybrid knobs), +10..+30, and it interacts with (1); (4) search
   parameters, 0..+20; (5) the network (RL), unknown and the slowest.

To pass the final with 87% probability the true score has to be about **0.64 (+100 Elo)**; at a true 0.60 the
chance is 47%:

| true score | Elo | P(final > 0.6 in 200 games) |
|---|---|---|
| 0.55 | +35 | 0.07 |
| 0.58 | +56 | 0.26 |
| 0.60 | +70 | 0.47 |
| 0.62 | +85 | 0.70 |
| 0.64 | +100 | 0.87 |
| 0.65 | +108 | 0.92 |
| 0.67 | +123 | 0.98 |

## Data and method

Rows in `runs/wave2c/r4/long/` on PACE (five parts each, seeds 1001-1005, `arena/pools/wave2.json`, bughouse
openings, v3 rules, BASE_MS 120000, MEASURE=1 at 800 nodes; both teams `clock_k` 110, `max_nodes` 100000, serial,
charge searched; ours also `root_min_ms` 100, `root_lag_ms` 75):

| row | opponent | games | score | Elo |
|---|---|---|---|---|
| l_hybrid_120 | upstream-it13 | 1345 | 0.514 +- 0.014 | +10 |
| l_nodur_120 | upstream-it13 | 1244 | 0.451 +- 0.014 | -34 |
| l_same_120 | hybrid vs hybrid (control) | 1367 | 0.503 +- 0.013 | +2 |

(The brief quotes 1252 games / 0.511 for l_hybrid_120; the files on PACE on 2026-11-23 hold 1345 games.)

Copied with rsync to the scratchpad; parsed with `tools/elo_hunt_parse.py <dir with l_*> games.pkl`, reports from
`tools/elo_hunt_report.py games.pkl` and `tools/elo_hunt_report_eval.py games.pkl`. Per game the files hold every
search (`thinking` end: seat, charged think ms, searched nodes, root eval in the searching team's perspective), every
move with all four clocks, pass/stall events with reasons, hybrid triggers, and the result record (reason, board,
loser seat, `clock_teams` lead counters, hybrid counters).

Same part number = same seed = same openings in every row, so hybrid and nodur games can be paired by opening.

## 1. Results by termination, side, board, length

| row | checkmate | repetition | flag | other |
|---|---|---|---|---|
| hybrid | 1341 | 4 | 0 | 0 |
| nodur | 1237 | 7 | 0 | 0 |
| same | 1360 | 6 | 0 | 1 sit-off adjudication |

By our colours (T1 = A white + B black):

| row | us T1 | us T2 |
|---|---|---|
| hybrid | 0.496 (674) | 0.532 (671) |
| nodur | 0.429 (623) | 0.472 (621) |
| same | 0.515 (683) | 0.490 (684) |

The colour gap (about 0.04) is 1.4 sigma and flips sign in the control row: noise.

Decisive games, by the board that ended them and the mated seat (hybrid row):

| board | we lost: mated seat | n | we won: mated seat | n |
|---|---|---|---|---|
| A | A_black | 169 | A_black | 169 |
| A | A_white | 149 | A_white | 169 |
| B | B_black | 190 | B_black | 187 |
| B | B_white | 144 | B_white | 164 |

Board A: 338 wins / 318 losses; board B: 351 / 334. No board or seat is a leak.

By game length (plies; hybrid / nodur / same):

| plies | hybrid | nodur | same |
|---|---|---|---|
| 40-79 | 0.503 (198) | 0.420 (194) | 0.464 (196) |
| 80-119 | 0.521 (627) | 0.466 (553) | 0.500 (598) |
| 120-159 | 0.514 (391) | 0.436 (389) | 0.517 (460) |
| 160-199 | 0.505 (109) | 0.450 (80) | 0.542 (96) |

Flat: no length band where we specifically lose. Mean game: about 60 s of game time, 112 plies, 56 decisions a team.

Colour-swapped pairs (hybrid): totals 0 / 1 / 2 in 159 / 331 / 177 pairs, which is the binomial split for p=0.514
(158 / 335 / 177): the two games of a pair are close to independent, openings do not decide games.

## 2. Time

Per search (thinking end events):

| row | side | searches | think ms mean / median | nodes mean / median |
|---|---|---|---|---|
| hybrid | us | 75,287 | 1023 / 1046 | 16,478 / 18,161 |
| hybrid | them | 75,270 | 1022 / 1045 | 18,672 / 20,049 |
| nodur | us | 70,164 | 1024 / 1045 | 16,501 / 18,104 |
| nodur | them | 70,178 | 1020 / 1044 | 20,434 / 22,561 |
| same | both | 77,683 | 1024 / 1045 | 16,317 / 17,961 |

Equal time, fewer nodes: we search 12% (hybrid row) to 19% (nodur row) fewer nodes per decision, which is exactly the
measured ms/node ratio.

Remaining clock (mean of a team's two seats, s) at the k-th move of the game, hybrid row:

| move | games | us | them |
|---|---|---|---|
| 10 | 1345 | 116.5 | 116.5 |
| 30 | 1343 | 110.8 | 110.8 |
| 60 | 1285 | 102.4 | 102.5 |
| 100 | 814 | 91.6 | 91.7 |
| 150 | 154 | 79.5 | 79.5 |
| 200 | 9 | 69.4 | 69.4 |

At game end: won games us 90.1 / them 89.6 s; lost games us 89.3 / them 89.9 s. The nodur and same rows are the same
to the tenth of a second. We neither over- nor under-spend relative to upstream (same time manager), nobody gets
flagged, nobody flags anybody, and **about 75% of the clock is unused in every game.**

Spending more of it is not obviously good. Older rows (`runs/wave2/`, durative wave2 engine against upstream):

| row | our clock_k | upstream budget | upstream nodes/decision | upstream sits ahead/game | upstream turns ahead / behind | score (ours) |
|---|---|---|---|---|---|---|
| ref_120 | 110 | clock_k 110 | 20,836 | 4.06 | 44.3 / 8.6 | 0.227 (300) |
| vs_it13_800n_120 | 110 | fixed 800 nodes (~40 ms/move) | 370 | 9.65 | 55.8 / 1.0 | 0.203 (106) |
| vs_it13_180 | 110 (180 s clock) | clock_k 110 | 31,876 | 3.76 | 44.0 / 8.3 | 0.257 (111) |

Upstream with 56 times fewer nodes per decision but always ahead on the clock and sitting 9.7 times a game for 37 s was
at least as strong as upstream at 1 s a move. Giving both sides 1.5x the clock (180 s) changed nothing. In this
regime nodes are cheap and clock lead is expensive. (Caveat: that was the durative engine, which was 0.23 anyway;
the hybrid has not been tested against either setting.)

## 3. Speed (ms per node) and the luck of the measurement

ms/node measured at the start of each part (three 800-node searches from the start position):

| row / part | ours | upstream | ratio them/us | score |
|---|---|---|---|---|
| hybrid p1 | 0.0503 | 0.0425 | 0.845 | 0.460 (239) |
| hybrid p2 | 0.0517 | 0.0420 | 0.811 | 0.509 (269) |
| hybrid p3 | 0.0525 | 0.0523 | 0.995 | **0.579** (297) |
| hybrid p4 | 0.0529 | 0.0416 | 0.785 | 0.508 (258) |
| hybrid p5 | 0.0537 | 0.0534 | 0.995 | 0.500 (282) |
| nodur p1 | 0.0507 | 0.0435 | 0.859 | 0.472 (265) |
| nodur p2 | 0.0559 | 0.0416 | 0.744 | 0.456 (259) |
| nodur p3 | 0.0495 | 0.0420 | 0.848 | 0.425 (228) |
| nodur p4 | 0.0527 | 0.0412 | 0.782 | 0.451 (255) |
| nodur p5 | 0.0523 | 0.0416 | 0.795 | 0.445 (237) |

- Our engine never measured below 0.0495; upstream is bimodal (0.041-0.044 in 8 of 10 parts, 0.052-0.053 in 2). The
  two slow upstream measurements both fell in the hybrid row, parts 3 and 5.
- Fit across all ten parts (score on log2 of the ratio, a separate intercept per row): **+73 +- 57 Elo per doubling
  of relative nodes**. Paired by opening (same part = same openings): hybrid minus nodur is +0.029 +- 0.023 in parts
  1, 2, 4 (both rows measured at similar ratios) and +0.104 +- 0.030 in parts 3, 5 (hybrid measured at parity,
  nodur at 0.80-0.85). The difference, about +0.075 for 0.25-0.33 doublings, points to the upper end of the fit.
- Read: the hybrid's real edge over nodur is about **+20 Elo** (0.029), not +45; at the usual measured ratio (0.81)
  the hybrid scores **0.493**, i.e. about -5 Elo against upstream.
- In the final match each part measures once more the same way, so the result can move by roughly +-30 Elo on a
  three-search measurement. That is variance we do not control unless the measurement changes (more nodes, more
  repeats, interleaved, or the median of several), which affects both engines equally and is a harness decision.
- Measurement detail that matters for engineering: the measurement is wall time / searched nodes for three
  800-node searches (about 40 ms each) after one 64-node warm-up, while games search about 18,000 nodes. Any fixed
  per-search cost of our engine (option parsing of SeatClocksMs etc., tree set-up, the hybrid gate, mate probes
  started per search) is charged as if it scaled with nodes. Profiling an 800-node search of each binary is the first
  speed task; it can be done on one GPU in an hour.

## 4. Sits and passes

Per game (stall active to inactive):

| row | side | engine passes (`upstream_pass`) | hybrid sits | mean sit |
|---|---|---|---|---|
| hybrid | us | 3.56 | 0.25 | 1.7 s |
| hybrid | them | 3.73 | - | 1.8 s |
| nodur | us | 3.72 | - | 1.8 s |
| nodur | them | 3.84 | - | 1.7 s |

Hybrid gate counters (hybrid row, 1345 games): 6,949 triggers (5.2/game), 373 sits (0.28/game), 142 missed.
Sit durations (both sides) have quartiles 1.1 / 1.8 / 2.1 s.

From `clock_teams`: we sit while ahead 3.20/game and while behind 0.62/game; upstream 3.36 and 0.37.

Score by sit difference (ours minus theirs):

| sit diff | hybrid | nodur | same |
|---|---|---|---|
| -3 | 0.146 (79) | 0.148 (81) | 0.211 (95) |
| -2 | 0.214 (140) | 0.213 (150) | 0.282 (172) |
| -1 | 0.299 (226) | 0.340 (228) | 0.372 (227) |
| 0 | 0.489 (268) | 0.463 (260) | 0.478 (255) |
| +1 | 0.726 (234) | 0.554 (196) | 0.657 (220) |
| +2 | 0.809 (162) | 0.719 (137) | 0.798 (163) |
| +3 | 0.837 (92) | 0.761 (69) | 0.785 (79) |

Score by the share of our turns spent ahead on the team clock lead (hybrid): 0-20% 0.273, 20-40% 0.388,
40-60% 0.416, 60-80% 0.558, 80-100% 0.697. Games with 0 / 1 / 2+ hybrid sits of ours: 0.499 (1079) / 0.545 (199) /
0.657 (67).

The hybrid row's +1 sit bucket (0.726 against nodur's 0.554) is where the hybrid's gain shows. All of this is
correlation; the causal test is lever 1 below.

## 5. Eval trajectories

Mean root eval in our perspective by game time (5 s windows), from our searches, theirs, and the mean of both
(consensus; both use it13, so the scales match; each side reads about +0.08 for itself at the start in every row,
including the control):

| t (s) | hybrid consensus | nodur consensus | same consensus |
|---|---|---|---|
| 0 | -0.000 | -0.000 | +0.002 |
| 10 | -0.001 | -0.004 | +0.003 |
| 20 | -0.004 | -0.021 | +0.003 |
| 30 | -0.012 | -0.043 | -0.005 |
| 40 | -0.017 | -0.025 | +0.006 |
| 50 | +0.011 | -0.043 | +0.030 |
| 60 | +0.021 | -0.111 | +0.010 |
| 75 | +0.059 | -0.016 | +0.038 |

Score given the consensus eval at 10 s is near 0 (the opening is over, nothing decided yet): hybrid 0.529 (535),
nodur 0.461 (498), same 0.493 (543). The nodur arm loses ground through the middlegame (20-60 s); the hybrid holds
level until about 45 s and gains slightly after, which is when sits and piece waits pile up. There is no single phase
to fix; the deficit is spread thin.

Our own eval between our consecutive searches drifts by -0.003 per search over the first 20 searches in every row and
for both engines: the same network optimism, not a defect of ours.

## 6. Openings

230 distinct openings, median 4 games each. The spread of per-opening scores (sd 0.19 for openings with 6+ games) is
the binomial expectation (0.20): no detectable opening effect. "Bare start" (26% of games): hybrid 0.486 (348),
nodur 0.454 (326). Largest families (hybrid): e6-Nc6 0.622 (74), Fortress 0.571 (70), Two Knights 0.597 (62),
e6-Nf6 ...d5 0.431 (65), d4 d5 Nc3 0.460 (62), all within 2 sigma. Not a lever: the final match plays the pool's
openings anyway.

## 7. Other within-game signals (not causal)

Score by our mean nodes per search divided by theirs within a game (hybrid): 0.7 -> 0.96 (53), 0.8 -> 0.55 (529),
0.9 -> 0.41 (264), 1.0 -> 0.52 (438), 1.1 -> 0.17 (60). The control row shows the same slope (0.9 -> 0.82,
1.1 -> 0.18), so this is game state (the side that is winning gets more, shorter forced or sitting decisions), not
speed. Mentioned so nobody reads it as a speed result.

## Ranked levers

Elo figures are against upstream-it13 at 120 s from today's hybrid (true level about 0.49-0.51).

**1. Our time allocation (play faster, hold the clock lead). Estimated -30..+100 Elo, sign unknown; test first.**
Evidence: 75% of the clock unused; clock lead and sit difference are the strongest correlates of winning; upstream
lost nothing by moving 25 times faster with a permanent lead; +50% clock for both changed nothing. Upstream's sitting
is hard-gated on `can_sit = lead > 0`, so if our searches are shorter than its 1 s, our lead stays positive and
upstream can almost never sit, while our sit gate can. Cost: fewer nodes (the fit says about -70 Elo per halving),
which the 800n row suggests the tempo pays back. Experiment: contender `clock_k` in {220, 440, 1000} plus a lower
value (55) as the other direction, against upstream at its fixed 110, 400 games each (about 11 GPU-h a row, 5 parts
in parallel, a few hours of wall clock). Then a finer step and the hybrid knobs at the best value. A smarter rule
(keep the lead, spend the margin in sharp positions) is a follow-up; the first sweep is only player options.
Rules question for the user: `clock_k` is a contender player option and `launch_final_match.sh` passes engine
`options` through to the player (`arena/pool.py` merges `contender_kw`), so a different time manager for our engine
is technically supported; whether that counts as "our engine" for the final is the user's call. The opponent's
`clock_k` 110 is fixed by the pool.

**2. Engine speed (ms/node). Estimated +20..+40 Elo for closing the gap.** We are charged 18-20% more per node than
upstream on the same GPU and network (0.050-0.054 vs 0.042). +73 +- 57 Elo per doubling from the part fit, more from
the paired part comparison; the gap is 0.25-0.3 doublings. First find out where it comes from: profile an
800-node search of both binaries (fixed per-search cost versus per-node cost); if it is fixed cost, the 800-node
measurement overstates our real ms/node and a larger MEASURE_NODES would be the fair fix (harness decision, applies
to both). Engine-side candidates: anything the wave 2 build does per search or per node that stock does not
(SeatClocksMs parsing, hybrid gate, extra mate-probe or certification work, a different TensorRT build or batch
size). Also, `measure_wave2_ms_per_node` measures with the flags only, not `engine_options`, so any speed-relevant
engine option set through `engine_options` is not measured.

**3. Sit policy (hybrid knobs). Estimated +10..+30 Elo, larger if lever 1 gives us the lead.** The hybrid is worth
about +20 Elo over nodur so far; it sits only 0.28 times a game out of 5.2 triggers, with 0.1 missed. With clocks never
below 88 s, `HybridMinClockMs` (3000) never binds; `HybridMarginPermille` (20) and `HybridPassPriorPermille` (150)
control how often it sits; `HybridMaxWaitMs` (1000) caps waits while real sits last 1.1-2.1 s. Sweep margin {0, 10,
20, 40} and pass prior {150, 300, 500} after lever 1.

**4. Search parameters. Estimated 0..+20 Elo.** CPUCT 3.0, FPU reduction 1.0, policy temperature 1.0, PW and moves-
left discount are upstream's own defaults on the same network, so upstream has already been tuned near them; gains
would come from settings that suit 16k-node searches with sitting. Low prior, cheap to test via `engine_options`
(but see the measurement caveat in lever 2).

**5. Network (RL on the hybrid). Unknown, slowest.** hybrid1 it1 gated 0.49 against it13 (no gain). The biggest
ceiling, but each iteration costs days of wall clock and the gain per iteration so far is zero; keep it running in
the background, do not count on it for the deadline.

Not levers: openings, colour, board, game length, flag handling (FlagLoss, MateRaceByLanding): no game reaches a low
clock at clock_k 110. That changes if lever 1 moves toward slower play.

## Tunable options (current values in the final-match hybrid)

### Arena player options (contender `player_kw`, i.e. the `options` of a wave2 engine in `ENGINES`)

| option | current | meaning |
|---|---|---|
| `clock_k` | 110 | budget = own seat clock / k, converted to nodes with ms_per_node |
| `min_nodes` / `max_nodes` | 16 / 100000 | clamp of the budget |
| `serial` | true | one engine per team; the second seat queues |
| `charge` | searched | clock charged with searched nodes x ms_per_node |
| `sit_ms` | unset (wave2: min(PassCapK, 1.5) x this search's budget) | how long a pass waits unless an event wakes it |
| `ema_alpha` | 0.3 | move-time tracker smoothing (TeamMoveTimeMs / OpponentMoveTimeMs) |
| `root_min_ms` / `root_lag_ms` | 100 / 75 | RootMoveTimeMs = max(budget, min) + lag |
| `durative_tree` | false | DurativeTree |
| `sit_gate` | true | SitGate (upstream's can_sit = lead > 0) |
| `hybrid` | true | sets HybridMode gate |
| `pass_cap_k` | 8 | PassCapK (1..64) |
| `flag_loss`, `instant_transfer`, `mate_race_by_landing` | wave 2 defaults (true, false, false) | FlagLoss, InstantTransfer, MateRaceByLanding |
| `hybrid_event_lands` | all | HybridEventLands (all/sit/none) |
| `hybrid_max_wait_ms` | 1000 | HybridMaxWaitMs (0..10000) |
| `hybrid_min_clock_ms` | 3000 | HybridMinClockMs (0..600000) |
| `hybrid_margin_permille` | 20 | HybridMarginPermille (0..1000) |
| `hybrid_pass_prior_permille` | 150 | HybridPassPriorPermille (0..1000) |
| `team_tree_reuse`, `joint_plans`, `plan_carry`, `own_pace_below_root`, `wait_node_acting_eval`, `blockable_check_not_mate` | false | wave 2c / diagnosis switches (knobs: TeamTreeReuseToleranceMs 250, JointPlanFreePermille 500, PlanCarryPermille 300) |
| `engine_options` | {} | raw UCI options, sent after the flags (not used by the ms/node measurement) |

### Engine UCI options (`third_party/wave2-hivemind/engine/src/interface/uci.cc`, defaults from `search/search_params.h`)

| UCI option | default | range |
|---|---|---|
| CPUCTInitPermille | 3000 (CPUCT_INIT 3.0; CPUCT_BASE 19652 compile-time) | 1..10000 |
| PolicyTemperaturePermille | 1000 | 100..10000 |
| MovesLeftDiscountPermille | 50 | 0..1000 |
| PUCTWidening | true | check |
| PWCoefficientPermille / RootPWCoefficientPermille | 1000 / 4000 | 1..10000 |
| PWExponentPermille | 300 | 1..1000 |
| PWMassStartPermille / RootPWMassStartPermille | 0 / -1 (inherit) | 0..999 / -1..999 |
| PWMassExponentPermille / PWMassCapPermille | 175 / 2000 | 1..10000 |
| PWMassNormalize | false | check |
| BatchSize | 8 | 1..1024 |
| MateProbe | true | check |
| FrozenRace | true | check |
| ConditionalMateProbe | false | check |
| MateThreads | 4 | 1..max |
| CertifySelectedMove | true | check |
| InternalMateProbe | off | off/telemetry/bias/certify/certifyonly |
| Transpositions | true | check |
| GumbelRootSearch | false | check (RootGumbelPoolSize 2, InitialCandidates 1, Replenishment 1, ValueScalePermille 2000, MaxRoundVisits 2) |
| DrawContemptPermille | 0 | 0..1000 |
| OpeningNoise | false | check (Plies 16, AlphaPermille 100, EpsilonPermille 600) |
| MaxMoveTimeMs | 0 | 0..3600000 |
| TimeModel | clock | flag/clock |
| Ponder / BackgroundSearch | true / true (the arena sends false) | check |

Compile-time only (`search_params.h`, needs a rebuild): FPU_REDUCTION 1.0 (dynamic FPU on), Q_INIT -1.0, virtual
loss style VIRTUAL_LOSS, NUM_SEARCH_THREADS 4 (`-DHIVEMIND_SEARCH_WORKERS`), SUBMIT_BEFORE_BACKUP true, TT sizes,
mate-probe and certification budgets (e.g. SELECTED_MOVE_CERT_TIME_PERCENT 3 / MAX_MS 100, FROZEN_RACE_TIME_PERCENT
20 / MAX_MS 250). The mate-probe and certification time slices are a candidate source of per-search fixed cost
(lever 2).

## What was not done

- No new games; everything above is from existing files. The causal size of levers 1-3 is unmeasured.
- No profiling of the two binaries (GPU work, outside this track).
- The `l_*` rows have no per-search clock lead or tree statistics beyond what is listed; eval is the root value of
  the searching team only, so "blunder" attribution per move was not attempted.
- The older `runs/wave2/` comparisons (section 2) use the durative engine and 100-300 games; treat them as direction,
  not size.
