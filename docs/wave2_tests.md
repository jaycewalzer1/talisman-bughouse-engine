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
| Pool | `arena/pools/wave2.json`: contender `wave2`; opponents `upstream-it13`, `wave2-self`, `it04`, `upstream-it13-800n` |
| Build | `WAVE2=1 sh tools/build_upstream.sh` (Mac) / `cluster/build.sh` (PACE, TensorRT) |
| Rows | `bash cluster/launch_checks.sh wave2` |

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
| `SeatElapsedMs` | - | empty | `now - last landing` on that board for the seat on turn, 0 for the others (all 0 when the arena gives no landing times) |
| `SeatSitting` | - | empty | added in integration: 1 for an opponent seat on turn whose elapsed exceeds 2x its pace (`seat_sitting`, `SIT_INFER_FACTOR`), so the tree's sit-off rule applies; 0 otherwise |
| `SeatArrivalsMs` `"<pocket>:<piece>:<ms_ago>,..."` | - | empty | wave 2b: pieces more in a pocket than at our last search, dated by the other board's last landing, kept 5 s (`PocketLandingTracker`); read by the engine only with a wide network (`WAVE2_MODEL=.../twin-s-clock-wide-fp32.onnx`), ignored with it13 |
| `Team`, `TimeModel clock`, `TimeLeadMs`, `TeamMoveTimeMs`, `OpponentMoveTimeMs`, `RootMoveTimeMs` | - | as v3 | as the stock `upstream` arm (`docs/upstream/v3_arm.md`) |

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
| `nodur` | upstream-it13 | `DurativeTree` off, `SitGate` on | 120 s | 600 | - | v3's lockstep tree inside the wave 2 binary: what the durative tree and the gate removal add together. (The planned "lockstep, no gate" cannot run: v3's tree has no sit-off rule, both teams passed forever and the search hung, so the engine now keeps the gate whenever the tree is not durative; docs/wave2_local_check.md) | near 0.50 (it is close to stock it13); `vs_it13_120` minus this row is the durative tree's contribution |
| `sitgate` | upstream-it13 | `SitGate` on | 120 s | 600 | - | durative tree with the old gate: the gate's cost or value inside the new tree | if close to the headline, removing the gate is free (the headline's "no hard-coded rules" costs nothing) |
| `instant` | upstream-it13 | `InstantTransfer` on | 120 s | 600 | - | does modelling transfer delay matter? | small difference (< 0.03) |
| `racewin` | upstream-it13 | `MateRaceByLanding` on | 120 s | 600 | - | do mate races decided by landing time help in play, given that the option switches off the root mate provers? | at or below `vs_it13_120`: locally it lost drop mates (68/80 vs 80/80 at 800 nodes) |
| `noflag` | upstream-it13 | `FlagLoss` off | 120 s | 600 | - | does seeing flags in the tree matter? | lower than the headline, more flag losses (`reasons` column) |
| `same_120` | wave2-self | none | 120 s | 800 | - | control | **0.50**, inside about +-0.025 (1600 games) |
| `vs_it04_120` | it04 (our fork, 100 nodes base, clock_k 110) | none | 120 s | 200 | 0,20 | sanity: the wave 2 arm is wired right | at least 0.80 (it13 scored 0.835 against it04) |
| `vs_it13_800n_120` | upstream-it13-800n (it13 at 800 fixed nodes, as D1 and upstream's own self-play) | none | 120 s | 800 | 0,20 | headline against it13 as D1 played it | near or above `vs_it13_120` |

Reading an ablation: the difference from `vs_it13_120`. With 1200 games the 95% range of one row is about +-0.03,
so the difference between two rows needs about 0.04 before it means anything. Read `same_120` first: if it is not
0.50, nothing else in the round is read.

Not comparable with D1 directly: D1's rows used the `eval` openings and arena rules v2. `vs_it04_120` links the two
(it measures the it04 gap again under these settings).

Reading the rows:

```sh
.venv/bin/python tools/sprt.py runs/wave2/vs_it13_120
.venv/bin/python tools/compare_evals.py runs/wave2/vs_it13_120 runs/wave2/sitgate
bash cluster/results.sh
```

## 5. Cost

The D1 rows (`h_m0_upstream_*`, it04 vs it13, equal clock) played about 1185 games at 120 s and 1170 at 180 s in one
8 h job on one RTX 6000 with 3 workers, about **150 games per GPU-hour**. Wave 2 against it13 has two fast arms, so
the same rate is assumed.

| Rows | Games (max) | GPU-hours (max) |
|---|---|---|
| `vs_it13_120`, `vs_it13_180`, `vs_it13_800n_120` | 3 x 1600 | 32 |
| 5 ablations | 5 x 1200 | 40 |
| `same_120` | 1600 | 11 |
| `vs_it04_120` | 400 | 3 |
| **Total** | 14,000 | **about 86** (10 GPUs in parallel: one 8 h job each, plus a second job for the three 1600-game rows if their SPRT has not stopped them) |

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
