# Wave 2 test round: the durative tree in upstream v3

Written 2026-11-19. Wave 2 = spec v4 (`docs/custom_engine_spec_v4_durative.md`) built into upstream v3.0.3 with the
twin-s-clock it13 network (decision D1: at equal clock it04 scored 0.165 at 120 s and 0.139 at 180 s against it13,
`runs/cluster_check_final.txt`). The headline it has to earn: **trained on real clocks with no hard-coded rules,
beat the best**. This round answers the first half of that for the untrained engine: does the durative tree with
the sit gate removed beat stock it13 at equal clock, and which part of it does the work?

## 1. What runs where

| Piece | Where |
|---|---|
| Engine | `third_party/wave2-hivemind`, branch `talisman/wave2` of `github.com/jaycewalzer1/hivemind` (based on upstream tag v3.0.3) |
| Stock comparison | `third_party/upstream-hivemind` (tag v3.0.3, untouched) |
| Network (both) | twin-s-clock it13, sha256 `f4ac25e3...`, inputs unchanged (planes 37/38 = lead, speed) |
| Arena kind | `wave2` (`arena/players/upstream.py`: `Wave2TeamController`, `wave2_team`) |
| Pool | `arena/pools/wave2.json`: contender `wave2`; opponents `upstream-it13`, `wave2-self`, `it04`, `upstream-it13-800n`, `wave2c-self` (section 7) |
| Build | `WAVE2=1 sh tools/build_upstream.sh` (Mac) / `cluster/build.sh` (PACE, TensorRT) |
| Rows | `bash cluster/launch_checks.sh wave2`; wave 2c: `bash cluster/launch_checks.sh wave2c` (section 7) |

## 2. UCI contract (what the arena sends)

Every option defaults to stock v3.0.3 behaviour in the engine; the `wave2` arm sends the wave 2 value of each flag
once at engine start, and the seat options before every search. Player options (pool `player_kw`, or
`CONTENDER_KW` on a row) switch single flags for ablations.

| Engine option | Player option | Stock default | Wave 2 sends |
|---|---|---|---|
| `DurativeTree` | `durative_tree` | false | true |
| `SitGate` (upstream's `can_sit = lead > 0`) | `sit_gate` | true | false |
| `FlagLoss` | `flag_loss` | false | true |
| `InstantTransfer` | `instant_transfer` | true | false |
| `MateRaceByLanding` | `mate_race_by_landing` | false | false (integration 2026-11-19: on, it costs drop mates; row `racewin` tests it on) |
| `PassCapK` | `pass_cap_k` | 8 | 8 |
| `SeatClocksMs` `"<Aw> <Ab> <Bw> <Bb>"` | - | empty | the arena's four clocks at `now` (the on-turn seats already charged their thinking so far) |
| `SeatPaceMs` | - | empty | per-seat EMA (alpha 0.3) of observed action time, read off the clocks between off-turn observations, sits included; a seat without a sample uses its team's EMA, then this search's budget |
| `SeatElapsedMs` | - | empty | `now - last landing` on that board for the seat on turn, 0 for the others (all 0 when the arena gives no landing times). Not quite self-play's meaning: self-play sends the time since the seat's current decision began, which restarts when it comes back from a pass; the arena cannot see that, so for a sitting opponent the tree here lands it sooner (`max(1, pace - elapsed)`, often 1 ms) than in training. Planes are not affected (both feed board elapsed to the wide planes) |
| `SeatSitting` | - | empty | added in integration: 1 for an opponent seat on turn whose elapsed exceeds 1.5 x min(its pace, its own budget = its clock at decision start / clock_k, doubled when its team is on turn on both boards) (`seat_sitting`, `SIT_INFER_FACTOR`), so the tree's sit-off rule applies; 0 otherwise. (Before review fix `36d92bf` the budget was ours, which marked opponents sitting exactly when we were behind on the clock) |
| `SeatArrivalsMs` `"<pocket>:<piece>:<ms_ago>,..."` | - | empty | wave 2b: pieces more in a pocket than at our last search, dated by the other board's last landing, kept 5 s (`PocketLandingTracker`); the wide planes read them only with a wide network (`WAVE2_MODEL=.../twin-s-clock-wide-fp32.onnx`); with `InstantTransfer` off any network's tree also holds an opponent seat's landings younger than its `SeatElapsedMs` as not yet droppable |
| `Team`, `TimeModel clock`, `TimeLeadMs`, `TeamMoveTimeMs`, `OpponentMoveTimeMs` | - | as v3 | as the stock `upstream` arm (`docs/upstream/v3_arm.md`) |
| `RootMoveTimeMs` | - | as v3 | `max(budget, root_min_ms) + root_lag_ms`; the wave 2 pool sets 100 and 75 (rules v3: every landing costs at least 100 ms plus 0-150 ms lag), so the tree prices our own actions as the arena charges them, like the opponents' clock-read paces (review fix `36d92bf`; before it the bare budget, about half the real cost at 10 s left) |

Seat order is always A_white, A_black, B_white, B_black; T1 = A_white + B_black is upstream's white team. Unit tests:
`arena/tests/test_upstream.py` (`test_seat_values_*`, `test_wave2_*`, `test_parity_seat_options_*`).

A pass in the arena waits `PassCapK x` this search's budget unless an event (any landing) wakes the team first: the
arena's version of the tree's "a pass lasts until the next event on the other board, capped at k x RootMoveTimeMs".
`sit_ms` overrides it. Stock `upstream` keeps its old wait (the team's move time).

## 3. Before the rows: build and check

```sh
# PACE, on a GPU node (pushes are the user's call: the branch has to be on GitHub first)
git -C third_party/wave2-hivemind push fork talisman/wave2
bash cluster/build.sh            # ends with "wave2 check: PASS" or "FAIL"
```

The wave 2 check is: the engine's own gtest suite (`hivemind_tests`, stopwatch tests get three tries), raw-policy
parity of the wave 2 engine against Python ONNX Runtime with the wave 2 flags on and the time given only through
`SeatClocksMs`/`SeatPaceMs` (chosen so that lead and speeds equal the reference's: this checks the seat -> plane
mapping and its signs), and the same parity for stock upstream. TensorRT runs FP16, so parity is read at
`WAVE2_TOL` = 2e-2 on PACE; on the Mac (CPU provider) the tolerance is 1e-4.

Mac versions of the same checks:

```sh
WAVE2=1 TESTS=1 JOBS=2 sh tools/build_upstream.sh
.venv/bin/python tools/upstream_check.py parity --engine wave2 --positions 20
.venv/bin/python tools/regression.py run --engine wave2 --engine-nodes 200 --out runs/regression/wave2.json
.venv/bin/python tools/regression.py run --engine wave2 --engine-nodes 200 --wave2-flags '{"sit_gate": true}'
```

**Acceptance test 8** (regression sit-off family `sitoff_for_piece`: at least 18/20 sits when 10 s ahead, 0/20 when
10 s behind, at 200 nodes) is printed by `tools/regression.py` as `spec v4 test 8 (engine): PASS/FAIL`, for the raw
net and the engine. Reference numbers measured 2026-11-19 on the Mac:

| Engine | raw net ahead / behind | 200 nodes ahead / behind | test 8 |
|---|---|---|---|
| stock upstream it13 (`--engine upstream`) | 20/20, 0/20 | 20/20, 0/20 | PASS (the gate forbids sitting when behind) |
| it04 raw net (`--engine fork`, `hivemind-fp32.onnx`) | 18/20, 0/20 | - | PASS (net) |
| wave 2, it13 (`--engine wave2`, integration 2026-11-19) | 20/20, 0/20 | 12/20, 0/20 (800 nodes: 20/20, 0/20) | **FAIL at 200 nodes**, PASS at 800 (the same with `SitGate` on: it is the durative tree at 200 nodes, not the gate; `docs/wave2_local_check.md`) |
| wave 2, widened it13 (`WAVE2_MODEL=.../twin-s-clock-wide-fp32.onnx`) | 20/20, 0/20 | 12/20, 0/20 | FAIL at 200 nodes, as it13 |

For stock it13 the 0/20 when behind is the hard gate's work; for wave 2 (SitGate false) it has to come from the
tree pricing the pass. Stock it13 also scores drop_mates 80/80 (net, value sign and engine) and sit_move 192/208 net,
176/208 engine.

Wave 2 ms per searched node on the Mac (CPU provider, 400 nodes, startpos, flags on, under other agents' load):
0.66 against stock upstream's 0.52 in the same minute, **measured on the binary built at 09:25, before the
engine's durative changes were in**: it only shows the plumbing works. Every row measures both arms on its own GPU.

## 4. The rows (`bash cluster/launch_checks.sh wave2`, 10 jobs)

Common to every row: the pool's `wave2` contender against one opponent, both arms `clock_k 110`, `max_nodes
100000`, `serial`, `charge "searched"`; each arm charged its searched nodes x its own ms per node, measured on the
job's GPU at 800 nodes (`MEASURE=1`; wave 2 is measured per flag set, since the flags change the tree); equal
clock; `bughouse` openings; arena rules v3 (min move time and lag, partners flagging together lose, 5 s sit-off
adjudication). `SBATCH_DEPENDENCY=afterok:<build job>` holds them behind a build.

| Row | Opponent | Contender change | Clock | Pairs | SPRT | Question | Expected reading |
|---|---|---|---|---|---|---|---|
| `vs_it13_120` | upstream-it13 | none (wave 2 flags; MateRaceByLanding off) | 120 s | 800 | 0,20 | headline: does wave 2 beat the best at equal clock? | unknown. The net was trained under the lockstep tree with the gate, so the new tree may first cost strength; > 0.50 with SPRT H1 is the success, 0.45-0.50 means "train it", < 0.45 means a bug or a value mismatch |
| `vs_it13_180` | upstream-it13 | none | 180 s | 800 | 0,20 | does it hold with more time? | within 0.03 of the 120 s row |
| `ref_120` | upstream-it13 | none | 120 s | 600 | - | the ablations' reference: the headline's settings at a fixed length (an SPRT-stopped row is short and its score biased by the stopping) | within the noise of `vs_it13_120` |
| `nodur` | upstream-it13 | `DurativeTree` off, `SitGate` on | 120 s | 600 | - | v3's lockstep tree inside the wave 2 binary: what the durative tree and the gate removal add together. (The planned "lockstep, no gate" cannot run: v3's tree has no sit-off rule, both teams passed forever and the search hung, so the engine now keeps the gate whenever the tree is not durative; docs/wave2_local_check.md) | near 0.50 (it is close to stock it13); `ref_120` minus this row is the durative tree **and** the gate removal together; `sitgate` minus this row is the durative tree alone (both gated) |
| `sitgate` | upstream-it13 | `SitGate` on | 120 s | 600 | - | durative tree with the old gate: the gate's cost or value inside the new tree | if close to the headline, removing the gate is free (the headline's "no hard-coded rules" costs nothing) |
| `instant` | upstream-it13 | `InstantTransfer` on | 120 s | 600 | - | does modelling transfer delay matter? | small difference (< 0.03) |
| `racewin` | upstream-it13 | `MateRaceByLanding` on, which **also switches off the root mate provers** (`enableRootMateSearch`, `enableMateProbe`, `certifySelectedMove`; agent.cc `run_search`) | 120 s | 600 | - | landing-time mate races *without root provers*: the row does not isolate the landing rule (two changes at once) | at or below `ref_120`: locally it lost drop mates (68/80 vs 80/80 at 800 nodes), which is the provers, not the races. Isolating the races needs landing-aware provers (not built) |
| `noflag` | upstream-it13 | `FlagLoss` off | 120 s | 600 | - | does seeing flags in the tree matter? | lower than the headline, more flag losses (`reasons` column) |
| `same_120` | wave2-self | none | 120 s | 800 | - | control | **0.50**, inside about +-0.025 (1600 games) |
| `vs_it04_120` | it04 (our fork, 100 nodes base, clock_k 110) | none | 120 s | 200 | 0,20 | sanity: the wave 2 arm is wired right | at least 0.80 (it13 scored 0.835 against it04) |
| `vs_it13_800n_120` | upstream-it13-800n (it13 at 800 fixed nodes, as D1 and upstream's own self-play) | none | 120 s | 800 | 0,20 | headline against it13 as D1 played it | near or above `vs_it13_120` |

Reading an ablation: the difference from `ref_120` (same settings as the headline, fixed 1200 games, no SPRT; the
headline `vs_it13_120` stops when decided, so it can be short and its score is biased by the stopping). With 1200
games the 95% range of one row is about +-0.03,
so the difference between two rows needs about 0.04 before it means anything. Read `same_120` first: if it is not
0.50, nothing else in the round is read.

Not comparable with D1 directly: D1's rows used the `eval` openings and arena rules v2. `vs_it04_120` links the two
(it measures the it04 gap again under these settings).

Reading the rows:

```sh
.venv/bin/python tools/sprt.py runs/wave2/vs_it13_120
.venv/bin/python tools/compare_evals.py runs/wave2/ref_120 runs/wave2/sitgate
bash cluster/results.sh
```

## 5. Cost

The D1 rows (`h_m0_upstream_*`, it04 vs it13, equal clock) played about 1185 games at 120 s and 1170 at 180 s in one
8 h job on one RTX 6000 with 3 workers, about **150 games per GPU-hour**. Wave 2 against it13 has two fast arms, so
the same rate is assumed.

| Rows | Games (max) | GPU-hours (max) |
|---|---|---|
| `vs_it13_120`, `vs_it13_180`, `vs_it13_800n_120` | 3 x 1600 | 32 |
| `ref_120` and 5 ablations | 6 x 1200 | 48 |
| `same_120` | 1600 | 11 |
| `vs_it04_120` | 400 | 3 |
| **Total** | 15,200 | **about 94** (11 GPUs in parallel: one 8 h job each, plus a second job for the three 1600-game rows if their SPRT has not stopped them) |

SPRT rows stop as soon as they are decided; a clear result (`vs_it04_120`, or a large gap either way) takes a few
hundred games. A row cut off at 8 h is finished by submitting it again with the same `NAME` and settings (the arena
resumes from its game files). The build job is about 1 GPU-hour on top.

## 6. Open

* `talisman/wave2` is local only. PACE builds from GitHub, so someone has to push it (`git -C
  third_party/wave2-hivemind push fork talisman/wave2`); agents never push. `WAVE2_REPO` points the build at another
  copy.
* The Mac checks of section 3 were re-run on the finished branch (clean build, 2026-11-19): numbers in
  `docs/wave2_local_check.md`. The ms/node above (0.66 vs 0.52) predates the engine changes; the clean build measured
  0.20-0.24 ms per searched node for wave 2 and 0.21 for stock (Mac CPU, 800 nodes, under load).
* No row uses the widened wave 2b network yet: untrained it plays as it13 (zero weights), so a row would only measure
  the transposition split of wide node keys. Once a wide lineage has promoted, a row runs it with
  `WAVE2_MODEL=<its fp32 onnx>` exported before `bash cluster/launch_checks.sh wave2` (sbatch passes the environment).
* Acceptance tests 1-7 of spec v4 belong to the engine's own gtest suite (run by the wave 2 check); test 8 is the
  regression run above.
* With TensorRT the wrapper `build-ninja/hivemind` sets `LD_LIBRARY_PATH`; the arena runs `hivemind.bin` directly, as
  it already does for stock upstream (the cluster modules provide the libraries).

## 7. Wave 2c rows (continuity, spec v5)

Added 2026-11-20. Spec: `docs/custom_engine_spec_v5_continuity.md`. The PACE round of 2026-11-19 put the durative
tree at 0.227 (`ref_120`) against 0.436 with it off (`nodur`), at nearly the same nodes/ms and the same eval
calibration, so the loss is move choice. Wave 2c adds three engine fixes, each off by default: **team tree reuse**
(fix 1: keep the team's tree between decisions and adopt the node the game reached, re-timed to the real clocks),
**joint team actions below the root** (fix 2: our two seats choose a pair, the half whose seat is not ready lands at
its ready time) and **plan carry** (fix 3: the next search of the partner is biased toward the half the last search
chose for it). These rows need the wave 2c engine build (`cluster/wave2_engine.ref` at the new commit, then
`bash cluster/build.sh`, "wave2 check: PASS").

### UCI contract additions

| Engine option | Player option | Engine default | Sent |
|---|---|---|---|
| `TeamTreeReuse` | `team_tree_reuse` | false | only when true |
| `TeamTreeReuseToleranceMs` | `team_tree_reuse_tolerance_ms` (0..5000) | 250 | only when not 250 |
| `JointPlans` | `joint_plans` | false | only when true |
| `JointPlanFreePermille` | `joint_plan_free_permille` (0..1000) | 500 | only when not 500 |
| `PlanCarry` | `plan_carry` | false | only when true |
| `PlanCarryPermille` | `plan_carry_permille` (0..1000) | 300 | only when not 300 |
| (none) | `continuity` | - | expands in Python to the three switches; a switch the row sets itself wins; no `Continuity` option is sent |

Unlike the six wave 2 flags, these are sent only off the engine default. A row without them therefore sends exactly
the options it sent before and still runs on a pre-2c binary. A row that sets one refuses to start on a binary that
does not advertise it (`RuntimeError ... build the wave 2c engine`), so a fix cannot silently do nothing.

`team_tree_reuse` or `plan_carry` on: the engine keeps its tree and plans between searches, so the arena sends no
`ucinewgame` before each search (as `reuse_tree`), and the controller sends one at its first search of each game.
`joint_plans` alone keeps the per-search `ucinewgame` of `ref_120`, so `c_joint_120` differs from it only by fix 2.
The engine side clears the rest of its cross-search state under fixes 1 and 3 (spec v5 fix 1 step 7).

Telemetry: the engine prints `info string Team tree reuse: <visits> visits recovered, ...` or `... none (<reason>)`
and `info string Plan carry: <board> <move> weight <w>` or `... dropped (<reason>)`. The arena records them per
search (`SearchResult.reused_visits`, `reuse_refusal`, `plan_carried`; the `thinking` start event's
`reused_visits` and `plan_carried`), per game (`continuity` in the summary: searches, reuse hits, reused visits,
plan carries per team) and per row (`contender_reuse_per_search`, `contender_plan_carries_per_search` and the
baseline's in `summary.json`; the printed table appends `c_reuse/search=` and `c_carry/search=` where they are
nonzero). Unit tests: `arena/tests/test_upstream.py` (`test_wave2_continuity_*`, `test_wave2_parses_reuse_line`,
`test_wave2_controller_*continuity*`, `test_continuity_telemetry_*`, `test_sim_records_continuity_*`).

### The rows (`bash cluster/launch_checks.sh wave2c`, 6 jobs, 7 with `WAVE2C_REF=1`)

Common settings as section 4 (pool `wave2` contender, `clock_k 110`, `serial`, `charge "searched"`, `MEASURE=1` at
800 nodes per binary and flag set, so fix 2's node cost is charged as measured; `bughouse` openings; rules v3), all
against `upstream-it13` at 120 s, no SPRT. `SBATCH_DEPENDENCY=afterok:<build job>` holds every row behind the build.

| Row | Opponent | `CONTENDER_KW` | Pairs | Question | Expected reading |
|---|---|---|---|---|---|
| `c_all_120` | upstream-it13 | `{"continuity": true}` | 600 | all three fixes on top of the headline settings | **success mark: at or above `c_nodur_120`** (the PACE gap between durative on and off was 0.21) |
| `c_reuse_120` | upstream-it13 | `{"team_tree_reuse": true}` | 600 | fix 1 alone | above `ref_120` (0.227) if memory between decisions is what was missing; reuse per search well above 0 (the gate run of the engine work sets the expected hit rate) |
| `c_joint_120` | upstream-it13 | `{"joint_plans": true}` | 600 | fix 2 alone | above `ref_120` if joint planning below the root is what was missing; reuse and carries per search exactly 0 |
| `c_carry_120` | upstream-it13 | `{"plan_carry": true}` | 600 | fix 3 alone | at or above `ref_120`; carries per search above 0 |
| `c_nodur_120` | upstream-it13 | `{"durative_tree": false, "sit_gate": true}` | 600 | the durative-off, gate-on reference, rerun on the new binary in the same round | near the PACE `nodur` (0.436) |
| `c_same_120` | wave2c-self | `{"continuity": true}` | 800 | control: continuity against itself | **0.50**, inside about +-0.025 (1600 games) |
| `c_ref_120` (only with `WAVE2C_REF=1`) | upstream-it13 | `{}` | 600 | the durative-on reference on the new binary | within the noise of `ref_120` (0.227). Optional: with every fix off the new binary matches the one `ref_120` ran in one-search (0.4, `wave2_parity.sh`) and whole-game and multi-search (0.6, `wave2_game_parity.sh`) checks; self-play is compared less node counts, which are not reproducible run to run on either binary |

How to read them:

1. Read `c_same_120` first. If it is not 0.50, nothing else in the round is read.
2. Check that each fix fired: `c_reuse/search` above 0 in `c_reuse_120` and `c_all_120`, `c_carry/search` above 0 in
   `c_carry_120` and `c_all_120`, both 0 in `c_joint_120` and `c_nodur_120` (`summary.json`, `results` ->
   `by_opponent`). A fix row whose counter is 0 measured nothing: report it, do not read its score.
3. Each fix-alone row against `ref_120` (or `c_ref_120`); `c_all_120` against `c_nodur_120`. With about 1200 games a
   row's 95% range is about +-0.03, so a difference needs about 0.04 before it means anything.
4. Fix 2 changes node cost; the per-arm `ms_per_node` measured for the row is in its `ms_per_node.json`. A row that
   scores better only because it is charged less per node is a speed result, not a move-choice result.

```sh
SBATCH_DEPENDENCY=afterok:<build job> bash cluster/launch_checks.sh wave2c
.venv/bin/python tools/compare_evals.py runs/wave2/ref_120 runs/wave2c/c_reuse_120
.venv/bin/python tools/compare_evals.py runs/wave2c/c_nodur_120 runs/wave2c/c_all_120
bash cluster/results.sh
```

Cost: 5 rows x 1200 games and `c_same_120` at 1600, about 8 GPU-hours each at section 5's rate: about 48
GPU-hours, plus 8 for `c_ref_120`.
