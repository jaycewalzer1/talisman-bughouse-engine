# Wave 2 + 2b local check (Mac CPU, 2026-11-19)

Integration of the four wave 2 packages (engine parts 1-2, arena, training) and wave 2b (wide planes, arrivals
head, `SeatArrivalsMs`, landing history, HDST v3) on this Mac, before anything runs on PACE.

**These are plumbing checks, not strength results.** Every game count below is 2 to 4 games; a 4-game score has a
95% range of about +-0.5. Nothing here says whether wave 2 beats it13. That is the job of `bash cluster/launch_checks.sh
wave2` (`docs/wave2_tests.md`).

Versions (sections 1-6): main repo `custom-engine`, wave2-hivemind `talisman/wave2` at `8c22eb7` (engine 2b `2b5d378`, training 2b
`8c22eb7`). Networks: twin-s-clock it13 (HF `aminwoo/bughouse-twin-s-clock@edaa246`, ONNX sha256 `f4ac25e3...`,
FP32 copy `twin-s-clock-it13-fp32.onnx`; checkpoint sha256 `5bb43c7f...`) and its widened copy
`twin-s-clock-wide-fp32.onnx` (sha256 `942606d5...`, zero-initialised new weights). Inference: ONNX Runtime on the
Mac (the engine's default providers; self-play logs show the CoreML provider taking 122 of 124 nodes).

## 1. Build and suites

| Check | Result |
|---|---|
| Clean build (old `build-ninja`, `build-sp1` moved to the Trash; `ninja-release` preset, ONNX Runtime, 2 jobs) | OK, `build-ninja` (4 search workers, tests) and `build-sp1` (1 worker) |
| C++ suite `build-ninja/tests/hivemind_tests` | 514 tests: 513 passed, 1 skipped (`MateRaceTest.MeasureFrozenRaceOnTheGame`, upstream's perf test, skipped on stock too), 0 failed |
| wave2-hivemind Python (`uv run pytest`) | 195 passed, 8 failed. The 8 are the ones that fail on pristine v3.0.3 here: 6 in `tests/data/test_archive.py`, 2 in `test_cli_rl_loop.py`; macOS bsdtar has no `tar --compare` ("Option --compare is not supported"); no GNU tar on this Mac to confirm the other way |
| Main repo `arena/tests` | 292 passed (291 + 1 new for the parity fix below) |
| Main repo `arena/tests_phase1` | 105 passed |
| UCI option names: every option the arena, `upstream_check`, `regression`, `warm_engine` send | all advertised by the engine's `uci` handshake |
| Training flags vs engine: `hivemind.wave2.missing_engine_options` with every flag on (`--mate-race-by-landing`, `--pass-cap-k` included) | none missing, for `selfplay --help` and `tournament --help` |

## 2. Wave 2b cross-checks

| Check | Result |
|---|---|
| C++ wide planes (`dumpplanes --wide-planes true`) vs Python `board2planes_wide`, 400 random positions (200 with seat clocks, thinking times and landings around the 5 s edge) | 800 board blocks, 0 differ, max abs diff 0; 2966 non-zero extra planes in the engine output (the check is not vacuous) |
| it13 vs widened it13 through one engine (`build-sp1`, one worker), 16 positions replayed from a wave 2 arena UCI log, 400 nodes, stock options (no seat timing) | bestmove 16/16, final report (nodes, score, PV) 16/16 identical |
| the same with the wave 2 flags, seat clocks, elapsed and a `SeatArrivalsMs` (the new planes non-zero) | bestmove 16/16; final report 14/16. Of the 2 that differ, one has the same score and nodes (409) and another PV tail; the other scores -8870 vs -8911 cp at 415 vs 408 nodes: wide node keys include landing history and thinking time, so wide trees split transpositions narrow trees merge (`wave2_engine.md`, Wave 2b). it13 against itself: 16/16 identical in both modes, so this is not noise |
| widened net outputs = it13's whatever the new planes hold | engine gtest `Wave2bTest` (FP32 and FP16 pairs, random extras: max diff 0, arrivals exactly 0); Python 300-position parity (4 comparisons, 0) |
| Sample round trip C++ -> Python: `selfplay` (3 games, 48 nodes, 15 s base, all wave 2 flags) -> HDST v2 + `seat_clocks.tsv` -> `attach_timing` (v3) -> `read_chunk` narrow / wide | wide net 131 records, it13 134: narrow fields unchanged by the v3 rewrite; timing block equals the tsv (clocks, elapsed, recent, arrivals on the labelled rows); 131/131 and 134/134 labelled; the wide layout holds every narrow plane in place |
| Do the tsv clocks equal the clocks the engine's root encoded? (open item from the network package) | lead plane 37 recomputed from the tsv clocks and paces with the engine's formula equals the record's plane on 131/131 and 134/134 records |
| Trainer path: it13 on a real v3 chunk read narrow vs it13 widened on the same chunk read wide (`DistillDataset.planes`) | 555 and 332 records (the A/B smoke's wide arm): every shared output identical (max diff 0), arrivals 0, with about 1,150 non-zero extra plane values per record |

## 3. Arena games (wave2 vs stock upstream it13)

`arena.pool run --pool arena/pools/wave2.json --opponent upstream-it13 --openings bughouse --rules v3 --base-ms 60000
--measure-ms-per-node --measure-nodes 800`, both arms `clock_k 110`, `charge "searched"`. The wave 2 engine ran
through a logging shim (stdin tee) so the options that reached it are counted. Output: `runs/wave2_local/int_*`.

| Row | Contender change | Narrow it13: games, score, sits wave2 / it13 | Widened: games, score, sits wave2 / it13 | Flags seen by the engine |
|---|---|---|---|---|
| headline | none (DurativeTree, no SitGate, FlagLoss, delayed transfer, MateRaceByLanding off, PassCapK 8) | 4, 0.75, 45 / 39 | 4, 0.25, 39 / 29 | as set, at every engine start |
| nodur | DurativeTree off, SitGate on | 2, 0.50 | 2, 0.50 | DurativeTree false, SitGate true |
| sitgate | SitGate on | 2, 0.00 | 2, 0.50 | SitGate true |
| instant | InstantTransfer on | 2, 0.50 | 2, 0.00 | InstantTransfer true |
| racewin | MateRaceByLanding on | 2, 0.50 | 2, 0.50 | MateRaceByLanding true |
| noflag | FlagLoss off | 2, 0.00 | 2, 0.00 | FlagLoss false |
| passcap | PassCapK 2 | 2, 0.50 | 2, 0.50 | PassCapK 2 |

30 games, 0 errors, 0 unplayed, all ended by checkmate (no flag, no sit-off adjudication at 60 s). Each row's switch
reached the engine and the other flags kept their wave 2 values. `SeatClocksMs` went before every search;
`SeatArrivalsMs` was non-empty before about 85% of searches, and all 1,147 distinct values sent were accepted by the
engine's parser (`info string SeatArrivalsMs set to N landings`, no refusal). The wave 2 arm sat (stall events) more
than stock it13 in every row, as expected with the gate off.

ms per searched node, measured per binary + network + flag set at 800 nodes: wave 2 0.17-0.29, stock upstream 0.21.
The spread is load, not the flags: the training smoke (section 5) ran at the same time as the narrow ablation rows.
So the equal-clock charge of these games is approximate; on PACE each row measures on its own GPU.

## 4. Regression and parity (`tools/regression.py`, `tools/upstream_check.py`)

| Run | drop_mates net / value / engine | sit_move net / engine | sit-off family ahead / behind (engine) | spec v4 test 8 |
|---|---|---|---|---|
| wave2, it13, 200 nodes | 80/80, 80/80, 79/80 | 192/208, 176/208 | 12/20 (+10 s), 12/20 (+3 s); 0/20, 0/20 | net PASS, **engine FAIL** (12/20 when ahead; needs 18) |
| wave2, widened, 200 nodes | 80/80, 80/80, 79/80 | 192/208, 175/208 | 12/20, 11/20; 0/20, 0/20 | net PASS, **engine FAIL** |
| wave2, it13, 200 nodes, `SitGate` on | | | 12/20, 12/20; 0/20, 0/20 | engine FAIL |
| wave2, it13, 800 nodes | 80/80, 80/80, 80/80 | 192/208, 192/208 | 20/20, 20/20; 0/20, 0/20 | engine PASS |
| stock upstream it13, 200 nodes (reference, `docs/wave2_tests.md`) | 80/80 | 192/208, 176/208 | 20/20; 0/20 | PASS |

Reading: when 10 s behind, wave 2 never sits (0/20) although nothing forbids it: the tree prices the pass. When ahead
it sits only 12/20 at 200 nodes, and the same with the gate back on, so the cause is the durative tree at a small
budget (waiting nodes cost an evaluation each, `wave2_engine.md`), not the gate's removal. At 800 nodes it is 20/20.
Test 8 is specified at 200 nodes, so it fails as written. Arena rows search far more than 200 nodes per move.
The earlier pass of this check (pre-2b binary, 10:25-10:54) gave the same picture (11/20 at 200 nodes, PASS at 800).
It also found that MateRaceByLanding on costs drop mates (68/80 at 800 nodes, because it switches off the root mate
provers), which is why the headline and the training preset have it off and row `racewin` tests it.

Raw-policy parity (`upstream_check.py parity --positions 20`): stock upstream PASS, wave2 with it13 PASS, wave2
with the widened net PASS (26/26 top-1, max |dp| 1.65e-6). The widened run first crashed: the tool's Python
reference fed 78 planes to a 130-input network. Fixed in `tools/upstream_check.py` (`_wide_extras`: the reference
builds the 26 extra planes per block from the seat clocks it gives the engine). With zero-initialised new weights
this parity cannot see a mistake in those planes; it starts to mean something with a trained wide network.

## 5. Training smoke (both arms, through the cluster script)

`cluster/wave2_iterate.sbatch` submitted through a fake `sbatch` that runs it locally (`RUN=int-ab AB=1 ITERS=1
GAMES_PER_ITER=4 SEGMENT_GAMES=4 VAL_GAMES=2 MATCH_GAMES=2 PARALLEL_GAMES=2 BATCH_SIZE=256`, gate on), self-play
on the fresh `build-sp1`, matches on `build-ninja`. Output: `runs/wave2/int-ab`.

| Arm | Self-play | Labels | Training | Export | Gate (2 games, 100 ms) |
|---|---|---|---|---|---|
| narrow (`twin-s-clock-w2`, from it13) | 4 games, 958 records; val 2 games, 265 | clean: 3 checkmates, 1 macro-ply limit, 0 flag games | 6 steps, trunk 5e-5 / heads 1e-4 | `C.onnx` 78 inputs, 5 outputs | 0.50 vs generator (= it13), not promoted |
| wide (`twin-s-clock-w2w`, from widened it13) | 4 games, 555 records; val 2 games, 332 | clean, 4 checkmates; 555/555 and 332/332 records to HDST v3 with arrivals labels | 4 steps; arrivals loss 0.1991 -> 0.1979 | `C.onnx` 130 inputs, 6 outputs (`arrivals` last) | 0.50 vs generator, 0.50 vs it13, not promoted |

Both arms ran init (SHA-256 checked), self-play, label check, (wide) timing attach, training, export, gate and
results; `results_ab.tsv` was written. The two arms used the same seeds but played different games (958 vs 555
records): the wide tree's node keys split transpositions, so equal seeds do not give equal games even with one search
worker. The narrow val WDL loss (0.06) and the wide one (1.47) are on different validation games; the trainer path
check in section 2 shows the wide data path gives it13's outputs exactly, so the gap is the games, not the pipeline.

## 6. Cluster dry runs (fake `sbatch`)

* `bash cluster/launch_checks.sh wave2`: 10 submissions, each with its NAME, OPPONENT, CONTENDER_KW, BASE_MS, PAIRS,
  SPRT and the common settings (pool, `SETUP=pool`, bughouse openings, rules v3, MEASURE=1 at 800 nodes,
  `WARM_KINDS=upstream wave2`). With `SBATCH_DEPENDENCY=afterok:12345`: 10/10 carry `--dependency=afterok:12345`.
* `cluster/wave2_iterate.sbatch`: section 5 (the fake sbatch printed the `#SBATCH` header and ran the script).
* `bash -n` on every `cluster/*.sh` and `cluster/*.sbatch`: OK.

Not run anywhere yet: TensorRT (the wide-input copy and `arrivals` binding in `engine/src/nn/engine.cc` are unbuilt),
`cluster/build.sh`, anything on PACE. `talisman/wave2` is local only; PACE builds it from GitHub, so it has to be
pushed first (`git -C third_party/wave2-hivemind push fork talisman/wave2`).

## 7. Review fixes and their rerun (2026-11-19, afternoon)

Two reviewers read wave 2 + 2b after the integration above. Every finding was re-checked against the code and
fixed; none was skipped. Engine commits on `talisman/wave2` (`d538648`..`261081b`), main repo on `custom-engine`.

| Finding | Fix | Commit |
|---|---|---|
| HIGH: at the root the other team's seat that lands before our budget was charged the whole budget and flag-checked against it (false flag wins in every scramble, opponent clocks overcharged, e.g. 4500 instead of 4900) | each seat on turn runs `min(dt, its ready time)`; its action is dated back (the seat its move hands the board to is charged the rest; a pass runs from the earlier time) | wave2 `d538648` |
| MEDIUM: one team on turn on both boards could pass forever in 1 ms steps (the pass woke at the partner's ready time) | a pass wakes only on a landing on the other board, as the self-play game does: a pass on the other board sends a sitting seat on to its own cap, a move wakes it 1 ms later | wave2 `d538648` |
| LOW: root never seeded delayed-transfer pieces from `SeatArrivalsMs` | an opponent seat on turn starts with its landings younger than its `SeatElapsedMs` pending; self-play sends its landing log for this with narrow nets too | wave2 `d538648`, `79d21fa` |
| LOW: pass cap in the tree (k x root budget for every seat) differed from the game (k x the seat's own move time) | k x the seat's own action time (root budget for our seats, pace for theirs) | wave2 `d538648` |
| LOW: `seat_clocks.tsv` `recent` counts had no cap while the search keeps 12 | the tsv keeps the newest 12 too (`pocket_landing_counts(..., MAX_RECENT_LANDINGS)`); arrivals labels still count all | wave2 `79d21fa` |
| HIGH: PACE build clones an unpushed branch; no record of the engine commit | build pinned to `cluster/wave2_engine.ref` (now `261081b`), a diverged checkout is an error (`ALLOW_STALE=1` to override), `eval.sbatch` and `wave2_iterate.sbatch` print the engine commit and pin. **The push itself is still owed** (below) | main `d2f732a` |
| MEDIUM: arena `seat_sitting` compared opponents with our own budget (opponents read as sitting exactly when we were behind) | the opponent seat's own budget, (its clock + elapsed) / clock_k, doubled when its team is on turn on both boards (serial second seat) | main `36d92bf` |
| MEDIUM: our own actions priced at the bare budget under rules v3 (about half the real cost at 10 s left) | `RootMoveTimeMs = max(budget, root_min_ms) + root_lag_ms`; the wave 2 pool sets 100 and 75 | main `36d92bf` |
| MEDIUM: ablations read against the SPRT-stopped headline | new row `ref_120` (headline settings, 600 pairs, no SPRT) is the ablations' reference; `launch_checks.sh wave2` now 11 jobs, about 94 GPU-hours | main `6d8fdb3` |
| MEDIUM: `racewin` and `nodur` each change two things | `racewin` documented and commented as "landing races without root provers"; `nodur` read as tree + gate removal, `sitgate - nodur` as the tree alone. Landing-aware root provers (the real fix for racewin) are not built | main `6d8fdb3` |
| LOW: arena `SeatElapsedMs` = since last landing vs self-play's since decision start | documented in the contract table (`docs/wave2_tests.md`); the arena cannot see a sitting opponent re-decide | main `6d8fdb3` |
| LOW: the wide stem's 26 new input columns trained at the trunk rate (5e-5) | `NewInputRate`: their AdamW update is scaled by lr / trunk_lr, exactly a heads-rate step; `--new-inputs-lr trunk` restores the old way. Gate columns documented as wave 2 search vs it13, not comparable with the arena rows | wave2 `261081b` |

Rerun after the fixes (same Mac, `build-ninja` and `build-sp1` rebuilt at `261081b` with 2 jobs):

| Check | Result |
|---|---|
| C++ suite `build-ninja/tests/hivemind_tests` | 519 tests: 518 passed, 1 skipped (`MeasureFrozenRaceOnTheGame`, as before), 0 failed. 5 new: root early landing is not a flag (the reviewer's 300/100/500 case) and a 5000 ms clock reads 4900; no same-team 1 ms leapfrog (20 more passes cost at least 2 s); opponent pass capped at its own pace; root pending from `SeatArrivalsMs`; recent counts capped at 12. One old test (`RootTeamDecidesAtTheRoot...`) rewritten for the new charging (Bw 59700 not 59000, Bb charged the 700 ms it was on turn) |
| `engine/tests/stock_parity.sh` (defaults vs stock v3.0.3, one worker) | 8/8 positions identical |
| wave2-hivemind Python (`uv run pytest`) | 196 passed (195 + 1 new, `NewInputRate`), 8 failed: the same 8 bsdtar `tar --compare` failures as before |
| Main repo `arena/tests` | 294 passed (292 + 2 new: opponent's own budget for `SeatSitting`, rules v3 root cost) |
| Main repo `arena/tests_phase1` | 105 passed |
| Regression, wave2 + it13, 200 nodes | drop_mates 80/80 net, value, engine (79/80 before); sit_move engine 175/208; sit-off ahead 11/20 (+10 s), 12/20 (+3 s), behind 0/20, 0/20: test 8 engine still FAIL at 200 nodes, as before (12/20) |
| Regression, wave2 + it13, 800 nodes | drop_mates 80/80 all; sit_move engine 192/208; sit-off 20/20, 20/20 ahead, 0/20 behind: test 8 **PASS** |
| Self-play, 3 games each, 48 nodes, 15 s base, wave 2 flags (it13 and widened it13, `build-sp1`) | 130 and 132 rows; recent pieces per row at most 9 and 11 (cap 12 never reached at 15 s); 2 flag games and 1 resignation each; 215 s per 3 games. Not rerun: `attach_timing` round trip (the tsv format did not change, only the cap) |
| Trainer, widened it13, 4 steps on the A/B smoke's wide chunk, trunk 5e-5 / heads 1e-4 | runs; new-plane stem weights 0 -> max 2.5e-4, old stem columns moved at most 1.3e-4 (about half), as intended |
| Arena headline, narrow it13, rules v3, 60 s, 3 pairs (`runs/wave2_local/fix_narrow_headline`) | 6 games, 0 errors, all checkmates, wave 2 scored 0.167 (95% range 0.00-0.50; the earlier 4-game run read 0.75). Plumbing only: 6 games say nothing about strength |
| `bash cluster/launch_checks.sh wave2` with a fake `sbatch` and `SBATCH_DEPENDENCY=afterok:12345` | 11 submissions, all with the dependency; `ref_120` 600 pairs without SPRT |
| `bash -n` / `sh -n` on the changed cluster and tools scripts; pin logic on a scratch clone | OK; a pinned commit missing from the repo fails with a message naming the push |

Still owed before PACE: push `talisman/wave2` (`git -C third_party/wave2-hivemind push fork talisman/wave2`) and
`custom-engine`; `cluster/build.sh` now refuses to build anything but the pinned `261081b`, so the pin has to be on
GitHub. Update `cluster/wave2_engine.ref` with every engine commit that should run.

## 8. Continuity round (spec v5)

The final local check of wave 2c (`docs/custom_engine_spec_v5_continuity.md`): team tree reuse (fix 1), joint team
actions in the durative tree (fix 2), plan carry (fix 3) and the `continuity` preset that turns on all three. Run on
2026-11-21 on this Mac. Engine: `talisman/wave2` at `8fe9b37` (all three fixes and the review fixes, clean tree).
Main repo: `custom-engine` at `799a0dc`, plus `b279075` (probe mode `cont`, below). Network everywhere:
twin-s-clock it13 (`engine/models/twin-s-clock-fp32.onnx`, HF `aminwoo/bughouse-twin-s-clock@edaa246`) on ONNX
Runtime. The self-play smoke also used the FP32 copy `twin-s-clock-it13-fp32.onnx` and the widened copy
`twin-s-clock-wide-fp32.onnx`. Output: `runs/wave2_local/continuity_v5/` and `runs/continuity_v5/`.

**These are plumbing checks, not strength results.** The arena smoke is 8 games, and the probe is 4 games plus
40 positions.

### 8.1 Build and suites

| Check | Result |
|---|---|
| Clean build: old `build-ninja` and `build-sp1` moved to the Trash; `build-ninja` with the `ninja-release` preset, ONNX Runtime, 4 workers, tests on; `build-sp1` configured as before (1 worker, fast build, ccache, tests off); 2 jobs each, one at a time | OK, 46/46 and 11/11 steps (6 and 4 compiler warnings, unused constants) |
| Snag on the way: the first `build-ninja` configure hung for about 50 min at 0% CPU in FetchContent's GoogleTest download (no network here) | stopped; the half-made dir went to the Trash. The old cache had `FETCHCONTENT_SOURCE_DIR_GOOGLETEST` pointing at `third_party/upstream-hivemind/engine/build-ninja/_deps/googletest-src` (v1.14.0, the pinned version), so the rebuild sets it again. A clean configure on a Mac with no network needs that flag |
| C++ suite `build-ninja/tests/hivemind_tests` | 589 tests: 588 passed, 1 skipped (`MateRaceTest.MeasureFrozenRaceOnTheGame`, as before), 0 failed; 39 s |
| `engine/tests/stock_parity.sh` (defaults vs stock v3.0.3, one worker, `JOBS=2`) | 8/8 positions identical |
| `engine/tests/wave2_parity.sh` (spec v5 0.4: every wave 2c option off vs `261081b`, headline flags, seat clocks) | 8/8 identical |
| `engine/tests/wave2_game_parity.sh` (0.6: whole games and a 10-search UCI session vs `261081b`) | 5/5 identical (self-play pgn and clocks, tournament pgn and tsv, UCI session) |
| wave2-hivemind Python (`uv run pytest`), run in two parts | without `tests/distill`: 183 passed, 8 failed (the same 8 bsdtar `tar --compare` failures: 6 in `tests/data/test_archive.py`, 2 in `test_cli_rl_loop.py`, "Option --compare is not supported"). `tests/distill` on its own: 19 passed in 134 s (`test_a_training_step_reduces_the_loss` 112 s; it ran over 30 min and was stopped last round). Total 202 passed, 8 failed |
| Main repo `arena/tests` | 305 passed |
| Main repo `arena/tests_phase1` | 105 passed |

### 8.2 Regression sets (`tools/regression.py run --engine wave2`, `build-ninja`)

`--wave2-flags '{}'` (wave 2 defaults: headline flags, wave 2c off) and `'{"continuity": true}'`. Regression cases
are single positions searched cold (`ucinewgame` before each one clears the team tree and the plans), so the
continuity rows test fix 2 only. Spec v5 4.3 says the same; fixes 1 and 3 are covered by the sequenced C++ test 4.1.

| Run | drop_mates net / value / engine | sit_move net / engine | sit-off family ahead +10 / +3 s; behind -10 / -3 s (engine) | spec v4 test 8 |
|---|---|---|---|---|
| wave 2 defaults, 200 nodes | 80/80, 80/80, 80/80 | 192/208, 176/208 | 12/20, 12/20; 0/20, 0/20 | net PASS, **engine FAIL** (12/20 when ahead) |
| continuity, 200 nodes | 80/80, 80/80, 80/80 | 192/208, 176/208 | 12/20, 13/20; **1/20**, 0/20 | net PASS, **engine FAIL** (12/20 when ahead, 1/20 when behind) |
| wave 2 defaults, 800 nodes | 80/80, 80/80, 80/80 | 192/208, 192/208 | 20/20, 20/20; 0/20, 0/20 | **PASS** |
| continuity, 800 nodes | 80/80, 80/80, 80/80 | 192/208, 191/208 | 20/20, 20/20; 0/20, **1/20** | **PASS** (test 8 reads the +-10 s rows) |

The defaults match section 7 exactly: drop mates, sit_move and test 8 are the same, and test 8 fails at 200 nodes
and passes at 800 as before. The continuity runs differ from the defaults in only these engine actions: 2 of 288 at
200 nodes and 1 of 288 at 800 nodes, all in `sitoff_for_piece`:

| Nodes | Case | Expected | Defaults | Continuity |
|---|---|---|---|---|
| 200 | `sitoff-minor-pinBe4@B_black/diag+3` | sit | move | sit (now right) |
| 200 | `sitoff-smoth-pinNe4@B_white/diag-10` | move | move | sit (now wrong) |
| 800 | `sitoff-smoth-pinNe4@A_white/diag-3` | move | move | sit (now wrong) |

Spec v5 4.3 asks for at least the `261081b` numbers: 80/80 drop mates, 20/20 sits when ahead, 0/20 when behind.
Read on the +-10 s rows that test 8 uses, it **passes** at 800 nodes. Read on every behind row, it does not quite:
the same smothered-mate case now sits once when 3 s behind (1/20, against 0/20). One case cannot say whether
JointPlans is biased toward sitting. The PACE rows `c_joint_120` / `c_all_120` measure sit rates over whole games.

### 8.3 Step 0 rerun with the continuity preset (`tools/continuity_probe.py`, mode `cont`)

The probe got a mode `cont` (`{"continuity": true}` over the headline flags) in `b279075`. In `reuse` it counts a
hit from the engine's `Team tree reuse: N visits recovered` line and logs the refusal reasons and the carried plans.
The summary tables gain a continuity column. The baseline numbers below are from `docs/continuity_baseline.md` (the
`261081b` binaries, which wave2_parity shows are identical to this build with the options off).

Reuse (local arena, `arena/pools/wave2.json`, wave 2 vs upstream it13, 60 s, `reuse_tree` on for both, seed 0, 2
pairs; `runs/continuity_v5/reuse_cont`, 223 s):

| Run | Wave 2 searches | Started from a reused subtree | Mean visits recovered (hits) | Recovered / searched nodes (hits) | Mean nodes searched |
|---|---|---|---|---|---|
| baseline, durative on (exact-hash reuse) | 316 | 0 (0.0%) | 0 | 0 | about 2050 |
| baseline, probe ceiling (same board, clocks ignored) | 316 | 104 (32.9%) | 583 | about 28% | about 2050 |
| **continuity** (fix 1, tolerance 250 ms) | 271 | **19 (7.0%)** | 328 | about 15% | 1892 |
| upstream it13 in the same games | 260 | 0 (0.0%) | 0 | 0 | 1932 |

Refusals in the continuity run (each of the 271 searches printed a `Team tree reuse` line): `shape` 97, `no node` 92,
`clock` 46, `committed` 8, `solved` 5, `no tree` 4. A carried plan was applied in 96 of 271 searches (35%). Fix 1
reaches about a fifth of the probe's ceiling. The main losses are `shape` (the node with the live boards was
expanded under another expansion key) and `no node` (no node holds the live boards: the other team's landings
arrived in another order than the tree's paces predicted, `wave2_engine.md` fix 1). `clock` (46) is the part a wider
tolerance could recover; self-play at 1000 and 3000 ms (`wave2_engine.md`) gained only a few points.

Node scaling (the same 40 positions, `go nodes N` on a fresh tree, `build-sp1`, one worker). Here every search starts
after `ucinewgame`, so **only fix 2 acts**; fixes 1 and 3 need a previous search of the same game. Rows:
`runs/continuity_v5/scaling.jsonl` (the baseline rows copied, plus `cont`).

| Nodes | on agrees with it13@64k (baseline) | off agrees (baseline) | **continuity agrees** |
|---|---|---|---|
| 1k | 23/40 (57%) | 23/40 (57%) | 20/40 (50%) |
| 4k | 20/40 (50%) | 25/40 (62%) | 22/40 (55%) |
| 16k | 22/40 (55%) | 27/40 (68%) | 26/40 (65%) |

| Nodes | continuity: per board vs it13@64k | root passes on an on-turn board (on / off / cont) | mean root q (on / off / cont) | ms per node (on / off / cont) |
|---|---|---|---|---|
| 1k | 57% | 4 / 3 / 4 | -0.083 / -0.127 / -0.100 | 0.514 / 0.518 / 0.517 |
| 4k | 61% | 4 / 4 / 5 | -0.091 / -0.131 / -0.115 | 0.511 / 0.523 / 0.519 |
| 16k | 71% | 7 / 4 / 5 | -0.116 / -0.137 / -0.123 | 0.604 / 0.518 / 0.518 |

Noise floor: continuity at 4k repeated gives the same joint action 40/40. Continuity agrees with durative on at 16k
on 32/40 and with off at 16k on 24/40.

Reading: unlike durative on, the continuity column **rises with nodes** (50%, 55%, 65%). At 16k it is about where off
is (26 against 27 of 40) and where it13's own 16k search is (28/40). It does not pass 16k on, which holds at 55%.
It starts lower at 1k, where planned pairs take visits from the immediate moves. The 16k wall time per node is
off's, not on's slower 0.60 ms. With 40 positions a difference of 4 is within noise (about +-6 at 95%), so this is
direction, not proof. It is the first sign that fix 2 moves the tree toward the "more nodes help" behaviour the
baseline found missing.

### 8.4 Self-play smoke and cluster dry run

`build-sp1/hivemind.bin selfplay --games 3 --nodes 48 --seed 11 --base-mix 15000:1`, wave 2 flags (`--durative-tree
true --sit-gate false --flag-loss true --instant-transfer false --mate-race-by-landing false --pass-cap-k 8`) and
`--continuity true`:

| Network | Games | Samples | Team tree reuse | Plan carry | Wall, peak memory |
|---|---|---|---|---|---|
| it13 fp32 | 3: a flag (Bb), 2 resignations; no error | 127 | 14.2% of searches (18/127), 52.8 mean visits, by seat Aw 9 Ab 8 Bw 0 Bb 1; refused `no node` 46, `shape` 38, `clock` 12, `no tree` 6, `solved` 5, `committed` 2 | 6 recorded, 0 applied (6 dropped `old root`) | 195 s, 232 MB |
| widened it13 fp32 | 3: a flag (Bb), 2 resignations; no error | 93 | 12.9% (12/93), 39.5 mean visits, Aw 2 Ab 6 Bw 3 Bb 1; `no node` 39, `shape` 21, `clock` 12, `no tree` 6, `solved` 2, `committed` 1 | 2 recorded, 0 applied (2 `old root`) | 155 s, 229 MB |

0 applied is expected with fix 1 on: the carried half is met by an old-root re-adoption, whose priors already hold
it (`wave2_engine.md`, review fixes). Both seats of both teams adopted trees.

`bash cluster/launch_checks.sh wave2c` with a fake `sbatch`/`squeue` on PATH (the script has no dry-run switch) and
`SBATCH_DEPENDENCY=afterok:12345`: 6 submissions (`c_all_120`, `c_reuse_120`, `c_joint_120`, `c_carry_120`,
`c_nodur_120`, `c_same_120`), 7 with `WAVE2C_REF=1` (`c_ref_120`). All carry `--dependency=afterok:12345`. Each has
its NAME, OPPONENT (`upstream-it13`, `wave2c-self` for `c_same_120`), CONTENDER_KW, 120000 ms and pairs (600, 800
for `c_same_120`), plus the common settings (pool `arena/pools/wave2.json`, `SETUP=pool`, bughouse openings, rules
v3, MEASURE=1 at 800 nodes, `WARM_KINDS=upstream wave2`). `bash -n` on every `cluster/*.sh` and `cluster/*.sbatch`:
OK.

**Not ready for PACE as is:** `cluster/wave2_engine.ref` still pins `261081b`, and `cluster/build.sh` builds only the
pinned commit. A `261081b` binary does not advertise the wave 2c options, so the arena would refuse every wave 2c
row at start-up (`UpstreamHivemindProcess.start`). Before `launch_checks.sh wave2c`: push `talisman/wave2` with
`8fe9b37`, set the pin to it, and push `custom-engine`. None of this was done here.

### 8.5 Arena smoke: continuity vs durative off + gate on

`arena.pool run --pool arena/pools/wave2.json --opponent wave2-self --contender-kw '{"continuity": true}'
--opponent-kw '{"durative_tree": false, "sit_gate": true}' --openings bughouse --rules v3 --base-ms 30000 --pairs 4
--seed 3 --measure-ms-per-node --measure-nodes 800`. Both sides run the same `build-ninja` binary (`runs/wave2_local/continuity_v5/arena_cont_vs_nodur`, 207 s).

| Games | Continuity score | Results | Measured ms/node (cont / nodur) | Continuity telemetry |
|---|---|---|---|---|
| 8 (0 errors, 0 unplayed) | 0.188 (1 win, 6 losses, 1 draw; 95% range 0.00-0.25) | 7 checkmates, 1 repetition draw, no flags | 0.207 / 0.213 | reuse 0.094 per search, plan carries 0.40 per search; opponent 0 and 0 |

`specs.jsonl` shows each side got its options: the contender `continuity: true` (the measurement key lists
`team_tree_reuse`, `joint_plans`, `plan_carry` true), the opponent `durative_tree: false, sit_gate: true`. **Smoke
only:** 8 games at 30 s say nothing about strength. The question of whether continuity closes the 0.227 vs 0.436 gap
is for `c_all_120` against `c_nodur_120` on PACE.

### 8.6 Not done or still open

* PACE needs the push and the new pin (8.4). Nothing was pushed.
* The Step 0 probe build (`ReuseProbe` print) was not rebuilt; the ceiling numbers are the baseline's.
* Wave 2 at 64k nodes was not run (as in the baseline); the deep references are it13 at 64k and each mode at 16k.
* Spec v5 4.3 on the behind rows: one new sit at -3 s at 800 nodes (8.2).
* Test 2.4b stays unbuilt (`wave2_engine.md`).

## 9. Review of the durative diagnosis fixes (2026-11-21)

Three findings on the diagnosis-fix work (D1-D7), fixed in the main repo only; no engine source changed, so the
wave2-hivemind binaries are the ones of section 8 plus the diagnosis fixes (`ec81b9a`).

| Finding | Fix | Regression test |
|---|---|---|
| A row passing the engine's `Continuity` preset raw through `engine_options` passed the start-up check on a continuity-round binary from before `426339a`, whose preset sets only three switches, and ran without D1 and D5 | start-up also requires `OwnPaceBelowRoot` and `BlockableCheckNotMate` (`CONTINUITY_PRESET_SWITCHES`) whenever `Continuity` is sent (`344aa80`) | `test_wave2_raw_continuity_preset_refuses_a_binary_from_before_the_diagnosis_fixes` |
| `seat_decisions` reset an inferred sitter to sitting 0 when a landing on the other board woke it; self-play's `landed` leaves `sitting_` set (only `commit_move` clears it), so at the landing instant it sends 1, elapsed 0 | a seat inferred sitting stays at 1, elapsed 0, until its own board lands (`9f98a7e`). Left as is: a woken seat that thinks about a move (self-play 0, elapsed since the wake) also reads 1, since the arena cannot tell it from a re-pass | `test_a_landing_that_wakes_a_sitting_opponent_keeps_it_sitting_as_self_play_does` (no hand-set mirror; checks the landing instant and a re-pass) |
| `wave2c` kept `c_ref_120` optional and read the fix rows against the PACE `ref_120`, which ran on the pre-D4 arena inputs | `c_ref_120` always runs (10 jobs, `WAVE2C_REF` gone); every fix-alone row is read against it (`1c714f8`) | fake-`sbatch` run of `launch_checks.sh wave2c`: 10 submissions, `c_ref_120` last |

Checks after the fixes: `build-ninja` and `build-sp1` rebuilt with 2 jobs (no work to do). C++ suite 606 tests: 605
passed, 1 skipped (`MateRaceTest.MeasureFrozenRaceOnTheGame`). `stock_parity.sh` 8/8 identical, `wave2_parity.sh`
8/8 identical. wave2-hivemind Python: without `tests/distill` 184 passed, 8 failed (the same 8 bsdtar
`--compare` failures, environment only); `tests/distill` 19 passed. Main repo `arena/tests` 312 passed,
`arena/tests_phase1` 105 passed. No measurement was made, so no network was used beyond the parity scripts' own
twin-s-clock network.

## 10. Diagnosis fixes (2026-11-21, final local check)

The final local check of the durative diagnosis fixes (`docs/durative_diagnosis.md`): D1 `OwnPaceBelowRoot`, D3
`WaitNodeActingEval` and D5 `BlockableCheckNotMate` in the engine, D4 and D7 in the arena, D6 in the wave2-hivemind
tests. The `continuity` preset now sets five switches (fixes 1-3 plus D1 and D5); D3 is not in it. Engine:
`talisman/wave2` at `ec81b9a` (clean tree). Main repo: `custom-engine` at `9a73bed`, plus the probe modes below.
Network for every measurement: twin-s-clock it13 (`engine/models/twin-s-clock-fp32.onnx`, HF
`aminwoo/bughouse-twin-s-clock@edaa246`) on ONNX Runtime (Core ML EP); the self-play smoke used its FP32 copy
`twin-s-clock-it13-fp32.onnx`. Output: `runs/continuity_diag/` and `runs/wave2_local/diag_fixes/`.

**These are plumbing checks, not strength results.** Another job (a reels script) ran on the Mac during the
1k/4k scaling runs (load average about 15), so the wall times there are noisy; node counts, moves and q are not
affected.

### 10.1 Build and suites

| Check | Result |
|---|---|
| Clean build: old `build-ninja` and `build-sp1` moved to the Trash; `build-ninja` with the `ninja-release` preset, `HIVEMIND_BACKEND=onnxruntime`, 4 workers, tests on, GoogleTest from the local source dir (8.1); `build-sp1` 1 worker, fast build, tests off; 2 jobs each, one at a time | OK, 47/47 steps in 45 s and 11/11 in 11 s (6 and 4 compiler warnings, as before). The preset alone configures the TensorRT backend and fails on this Mac; `-DHIVEMIND_BACKEND=onnxruntime` is needed |
| C++ suite `build-ninja/tests/hivemind_tests` | 606 tests: 605 passed, 1 skipped (`MateRaceTest.MeasureFrozenRaceOnTheGame`, as before), 0 failed; 40 s |
| `engine/tests/stock_parity.sh` (`JOBS=2`) | 8/8 positions identical |
| `engine/tests/wave2_parity.sh` (every wave 2c option off vs `261081b`, `JOBS=2`) | 8/8 identical |
| wave2-hivemind Python (`uv run --offline pytest`), in two parts | without `tests/distill`: 184 passed, 8 failed, all macOS tar "Option --compare is not supported" (6 in `tests/data/test_archive.py`, 2 in `tests/test_cli_rl_loop.py`). With the diagnosis' GNU-tar shim on PATH, those two files pass 32/32, so D6 holds. `tests/distill`: 19 passed (153 s). Total 203 passed, 8 failed (environment only) |
| Main repo `arena/tests` | 312 passed |
| Main repo `arena/tests_phase1` | 105 passed |

`wave2_game_parity.sh` was not rerun (no engine source changed since the engine agent ran it: 5/5).

### 10.2 Node scaling, same 40 positions (`tools/continuity_probe.py scaling`)

`go nodes N` on a fresh tree (`ucinewgame`), `build-sp1`, one worker, the 40 positions of
`runs/continuity_baseline/positions.jsonl`; agreement is the joint action against upstream it13 at 64k (the
baseline's rows, copied into `runs/continuity_diag/scaling.jsonl`). Only JointPlans, D1, D3 and D5 act here (fixes 1
and 3 need a previous search of the same game). New probe modes: `cont_d3` (preset + `wait_node_acting_eval`),
`cont_fp200` (preset, `joint_plan_free_permille` 200), `cont_nod1` (preset with `own_pace_below_root` false, added to
split D1 from D5), `cont_tol1000` / `cont_tol3000` (reuse tolerance).

Identity check: the rerun of wave 2 defaults (`on`) and durative off (`off`) gives the baseline's joint action on
40/40, 39/40, 40/40 (on, 1k/4k/16k; 39/40 is the baseline's own repeat floor) and 40/40 at every budget (off).

| Arm | Nodes | Joint vs it13@64k (40) | Per board | Mean root q | Root passes on an on-turn board | One board on turn (25): joint | (25): q | Both boards (15): joint |
|---|---|---|---|---|---|---|---|---|
| baseline on (`261081b`) | 1k / 4k / 16k | 23 / 20 / 22 | 64 / 59 / 61% | -0.083 / -0.091 / -0.116 | 4 / 4 / 7 | 15 / 13 / 13 | -0.100 / -0.107 / -0.131 | 8 / 7 / 9 |
| baseline off | 1k / 4k / 16k | 23 / 25 / 27 | 65 / 69 / 70% | -0.127 / -0.131 / -0.137 | 3 / 4 / 4 | 15 / 17 / 17 | -0.112 / -0.121 / -0.133 | 8 / 8 / 10 |
| continuity round preset (`8fe9b37`, fixes 1-3) | 1k / 4k / 16k | 20 / 22 / 26 | 57 / 61 / 71% | -0.100 / -0.115 / -0.123 | 4 / 5 / 5 | 12 / 14 / 17 | -0.107 / -0.123 / -0.142 | 8 / 8 / 9 |
| **wave 2 defaults now** (`on`) | 1k / 4k / 16k | 23 / 20 / 22 | 64 / 59 / 61% | -0.083 / -0.093 / -0.116 | 4 / 5 / 7 | 15 / 13 / 13 | -0.100 / -0.107 / -0.131 | 8 / 7 / 9 |
| **preset now** (`cont`: fixes 1-3, D1, D5) | 1k / 4k / 16k | 20 / 22 / **23** | 59 / 60 / 64% | -0.135 / -0.143 / **-0.170** | 6 / 6 / 7 | 13 / 13 / **14** | -0.140 / -0.147 / -0.192 | 7 / 9 / 9 |
| **preset + D3** (`cont_d3`) | 1k / 4k / 16k | 21 / 20 / 24 | 60 / 57 / 65% | -0.148 / -0.168 / -0.178 | 5 / 7 / 7 | 13 / 13 / 15 | -0.154 / -0.171 / -0.194 | 8 / 7 / 9 |
| **preset, free share 200** (`cont_fp200`) | 1k / 4k / 16k | 19 / 21 / 20 | 56 / 59 / 56% | -0.131 / -0.159 / -0.182 | 4 / 5 / 7 | 12 / 13 / 12 | -0.132 / -0.158 / -0.191 | 7 / 8 / 8 |
| **preset without D1** (`cont_nod1`) | 4k / 16k | 22 / 26 | 61 / 71% | -0.115 / -0.123 | 5 / 5 | 14 / 17 | -0.123 / -0.142 | 8 / 9 |
| **durative off + gate now** (`off`) | 1k / 4k / 16k | 23 / 25 / 27 | 65 / 69 / 70% | -0.127 / -0.131 / -0.137 | 3 / 4 / 4 | 15 / 17 / 17 | -0.112 / -0.121 / -0.133 | 8 / 8 / 10 |

it13 at 64k: mean q -0.128; it13 at 16k agrees with it13 at 64k on 28/40.

Same joint action as the preset (`cont`): `cont_d3` 36 / 32 / 31 of 40, `cont_fp200` 38 / 37 / 29, the continuity
round's preset 36 / 28 / 31 (1k / 4k / 16k). `cont_nod1` equals the continuity round's preset on 40/40 at 4k and 16k
(q within 0.001).

Reading:
1. **D5 changes nothing on these positions**; the whole difference between the preset now and the continuity round's
   preset is D1. `cont_nod1` reproduces the continuity round's moves exactly.
2. **D1 lowers agreement at 16k and overshoots on q.** With D1 the preset agrees 23/40 at 16k against 26/40 without
   it, and per board 64% against 71%; at 1k and 4k it is level (20 and 22 both ways). Mean root q moves from -0.123 to
   -0.170, below off (-0.137) and it13 at 64k (-0.128), and it still drifts down with nodes (-0.135, -0.143, -0.170).
   The diagnosis predicted the level shift (it removed about +0.05 of optimism); here it goes past the reference by
   about 0.04. A difference of 3 of 40 is within noise (about +-6 at 95%), so this does not show D1 is wrong, but it
   gives no support for keeping D1 in the preset either. A likely cause: the arena's own-seat `SeatPaceMs` is clock
   drop per move, which counts sits and the serial brain's queueing, so charging it below the root may make our team
   too slow. `c_ownpace_120` (D1 alone) and `c_all_120` on PACE are the test.
3. **D3** (`cont_d3`): 21 / 20 / 24 against 20 / 22 / 23; no gain shown, as in the diagnosis. q moves down a further
   0.01-0.03. It stays out of the preset.
4. **JointPlanFreePermille 200** is no better at low nodes (19 / 21 against 20 / 22) and clearly worse at 16k (20
   against 23; 12 of 25 one-board positions against 14). **Use the default 500 for the PACE rows.**
5. **D2 (does JointPlans close the one-board gap?)** Not with the preset as it now stands: on the 25 positions where
   we are on turn on one board, the preset agrees 13 / 13 / 14 of 25 against off's 15 / 17 / 17. Without D1
   (`cont_nod1`, the continuity round's preset) it reaches off at 16k (17/25) but not at 4k (14 against 17). On
   the 15 both-board positions every arm is within 7-10, so the remaining gap is in the one-board positions.

### 10.3 Fix 1 hit-rate gate at 250 / 1000 / 3000 ms (`tools/continuity_probe.py reuse`)

The continuity round ran the arena gate only at 250 ms (8.3). Here: the preset now (D1, D5 on, arena with D4), local
arena, `arena/pools/wave2.json`, wave 2 vs upstream it13, 60 s, `reuse_tree` on for both, seed 0, 2 pairs (4 games
each), `build-ninja`. Different tolerances play different games, so the denominators differ.

| Run | Tolerance | Wave 2 searches | Adopted | Mean visits recovered (hits) | Recovered / searched (hits) | Refused: shape / no node / clock / committed / solved / no tree | Carried plan applied | Score vs it13 (4 games) |
|---|---|---|---|---|---|---|---|---|
| continuity round (8.3, before D1/D4/D5) | 250 ms | 271 | 19 (7.0%) | 328 | about 15% | 97 / 92 / 46 / 8 / 5 / 4 | 96 | |
| `reuse_cont` | 250 ms | 229 | 9 (3.9%) | 377 | 18% | 86 / 80 / **24** / 19 / 7 / 4 | 81 | 0.500 |
| `reuse_cont_tol1000` | 1000 ms | 208 | 10 (4.8%) | 347 | 15% | 80 / 73 / **4** / 29 / 8 / 4 | 77 | 0.500 |
| `reuse_cont_tol3000` | 3000 ms | 240 | 20 (8.3%) | 291 | 14% | 69 / 93 / **6** / 34 / 14 / 4 | 91 | 0.250 |

Upstream it13 reused nothing in any of these games (0 of 204-231 searches each). All 12 games ended in checkmate.

Reading: the tolerance acts only on the `clock` refusals, and 1000 ms removes almost all of them (24 to 4, about 10%
to 2% of searches); 3000 ms removes no more (6). The rest of the 3000 ms run's higher hit rate (8.3%) comes from its
games, not from the clock: its `no node` and `shape` counts are as large as at 250 ms, and 4 games per run cannot
separate 3.9%, 4.8% and 8.3%. Recovered subtrees shrink as the tolerance grows (377, 347, 291 visits), as expected
from re-timing older nodes further. **Recommendation: default tolerance 1000 ms** (it recovers what the clock
refusals cost at the smallest re-timing; 3000 ms adds nothing measurable). This is a recommendation only: the
engine and arena defaults stay at 250 ms, and changing them is a separate decision. The hit rate is also lower than
in the continuity round (3.9% against 7.0% at 250 ms), with fewer `clock` refusals; whether D1's slower own seats,
D4's arena inputs or simply other games cause it, 4 games cannot say.

### 10.4 Self-play smoke and cluster dry run

`build-sp1/hivemind.bin selfplay --model models/twin-s-clock-it13-fp32.onnx --games 3 --nodes 48 --seed 11
--base-mix 15000:1`, wave 2 flags (`--durative-tree true --sit-gate false --flag-loss true --instant-transfer false
--mate-race-by-landing false --pass-cap-k 8`) and `--continuity true`:

| Network | Games | Samples | Team tree reuse | Plan carry | Wall, peak memory |
|---|---|---|---|---|---|
| it13 fp32 | 3, no error: flag (Bb), resignation, flag (Bw) | 158 | 14.6% of searches (23/158), 53.5 mean visits, by seat Aw 7 Ab 8 Bw 1 Bb 7; refused `no node` 57, `shape` 43, `clock` 15, `committed` 8, `no tree` 6, `solved` 6 | 9 recorded, 0 applied (9 dropped `old root`, as in 8.4) | 252 s, 201 MB |

The self-play log does not print which switches the preset set, so this smoke shows only that the preset with D1 and
D5 runs whole games without error; that `--continuity` sets D1 and D5 is covered by the C++ tests
(`test_diag_fixes.cc`).

`bash cluster/launch_checks.sh wave2c` with a fake `sbatch`/`squeue` on PATH and `SBATCH_DEPENDENCY=afterok:12345`:
10 submissions, all with `--dependency=afterok:12345`, 120000 ms: `c_all_120` (`{"continuity": true}`),
`c_reuse_120`, `c_joint_120`, `c_carry_120`, `c_ownpace_120`, `c_waiteval_120`, `c_mateblock_120`, `c_nodur_120`
(600 pairs each, against `upstream-it13`), `c_same_120` (800 pairs, `wave2c-self`) and `c_ref_120` (`{}`, 600
pairs). `bash -n cluster/launch_checks.sh`: OK.

**Still not ready for PACE as is:** `cluster/wave2_engine.ref` still pins `261081b`, and `talisman/wave2` is 35
commits ahead of `fork/talisman/wave2` (nothing pushed). A `261081b` binary does not advertise the wave 2c or
diagnosis options, so the arena would refuse every row except `c_ref_120` and `c_nodur_120` at start-up. Push
`talisman/wave2` at `ec81b9a`, set the pin and push `custom-engine` first. None of this was done here.

### 10.5 Arena smoke: preset vs durative off + gate on

`arena.pool run --pool arena/pools/wave2.json --opponent wave2-self --contender-kw '{"continuity": true}'
--opponent-kw '{"durative_tree": false, "sit_gate": true}' --openings bughouse --rules v3 --base-ms 30000 --pairs 4
--seed 3 --measure-ms-per-node --measure-nodes 800`, both sides `build-ninja`
(`runs/wave2_local/diag_fixes/arena_cont_vs_nodur`, 251 s).

| Games | Preset score | Results | Measured ms/node (preset / nodur) | Preset telemetry |
|---|---|---|---|---|
| 8 (0 errors, 0 unplayed) | 0.125 (1 win, 7 losses; 95% range 0.00-0.25) | 8 checkmates | 0.202 / 0.211 | reuse 0.06 per search, plan carries 0.44 per search |

`specs.jsonl` shows the contender got `continuity: true` and the opponent `durative_tree: false, sit_gate: true`.
**Smoke only:** 8 games at 30 s say nothing about strength (the continuity round's smoke was 0.188 on the same
seeds). Whether the preset closes the gap to nodur is for `c_all_120` against `c_nodur_120` and `c_ref_120` on PACE.

### 10.6 Not done or still open

* D1 in the preset is not supported by this check: 23/40 against 26/40 without it at 16k, and q overshoots the it13
  reference by about 0.04 (10.2). Within noise, so the preset is unchanged; `c_ownpace_120` and `c_all_120` decide.
  If D1 loses there, the preset should drop it (or the arena should send our seats a thinking-only pace).
* D2 is not closed: the preset with D1 is at 13-14 of 25 one-board positions against off's 17 at 4k and 16k.
* The reuse tolerance recommendation (1000 ms) is from 3 x 4 games; the defaults were not changed.
* `wave2_game_parity.sh` was not rerun (no engine source change since the engine agent's 5/5).
* Wave 2 at 64k nodes was not run, as in the baseline.
* PACE needs the push and the new pin (10.4). Nothing was pushed.

## 11. Hybrid round (spec v6)

The final local check of the hybrid (`docs/custom_engine_spec_v6_hybrid.md`, H3 as built plus the review fixes) before
the wave 2c round and the long test go to PACE. Engine: `talisman/wave2` at `4b45129` (clean tree; the review fixes
`a1c3a09..4b45129` on top of `e5f3aae`, the engine the Measurements section used). Main repo: `custom-engine` at
`98e69a8`, whose `cluster/wave2_engine.ref` already pins `4b45129`. Network for every measurement: twin-s-clock it13
(`engine/models/twin-s-clock-fp32.onnx`, HF `aminwoo/bughouse-twin-s-clock@edaa246`) on ONNX Runtime (Core ML EP);
the self-play smoke used its FP32 copy `twin-s-clock-it13-fp32.onnx`. The Mac was quiet at the start (load 6-10, 86%
free memory); the harness scoring, the Python suites and the self-play smoke then ran side by side (load up to about
28), which changes wall times only. Output: `runs/strength/scores_v6fix.jsonl`, `runs/strength/report_v6fix.md`,
`runs/wave2_local/hybrid_v6/` (git-ignored).

### 11.1 Build and suites

| Check | Result |
|---|---|
| Clean build: old `build-ninja` and `build-sp1` moved to the Trash; `build-ninja` with the `ninja-release` preset, `-DHIVEMIND_BACKEND=onnxruntime`, 4 workers, tests on, `FETCHCONTENT_SOURCE_DIR_GOOGLETEST` = `third_party/upstream-hivemind/engine/build-ninja/_deps/googletest-src`; `build-sp1` 1 worker, fast build, ccache, tests off; 2 jobs each, one at a time | OK, 48/48 steps in 47 s and 11/11 in 11 s (6 and 4 compiler warnings, as before) |
| C++ suite `build-ninja/tests/hivemind_tests` | 629 tests: 628 passed, 1 skipped (`MateRaceTest.MeasureFrozenRaceOnTheGame`, as before), 0 failed; 57 s. Includes the 8 `HybridIt13` tests and the review-fix regressions |
| `engine/tests/stock_parity.sh` (`JOBS=2`) | 8/8 positions identical |
| `engine/tests/wave2_parity.sh` (every wave 2c / v6 option off vs `261081b`, `JOBS=2`) | 8/8 identical |
| `engine/tests/wave2_game_parity.sh` (`JOBS=2`) | 5/5 identical (selfplay pgn and clocks, tournament pgn and tsv, a 10-search UCI session) |
| `engine/tests/hybrid_parity.sh` (spec v6 test 0.3, `JOBS=2`) | 8/8 identical with no trigger; positive control fired on 3 of 3 one-board positions (5 skipped: on turn on 2 boards). Position 7 matched on its second attempt (the script reruns a differing pair up to twice; the script names position 8 as the known run-to-run case, so position 7 needing a rerun is new and noted here) |
| wave2-hivemind Python (`uv run --offline pytest`), in two parts | without `tests/distill`: 186 passed, 8 failed, all macOS bsdtar "Option --compare is not supported" (6 in `tests/data/test_archive.py`, 2 in `tests/test_cli_rl_loop.py`), as in 10.1. With a GNU-tar shim on PATH (scratchpad; `-I` to `--use-compress-program`, `--format` dropped outside create, `--compare` emulated), those two files pass 33/33 (the rl-loop file gained a test since 10.1). `tests/distill`: 19 passed (515 s under load). Total 205 passed, 8 failed (environment only) |
| Main repo `arena/tests` | 354 passed (includes the decide / launch dry-run tests of `test_decide_full_run.py`) |
| Main repo `arena/tests_phase1` | 105 passed |

### 11.2 Harness rerun: nodur and the chosen hybrid on the fixed engine

`tools/strength_harness.py score nodur` and `score hybrid` (`{"durative_tree": false, "sit_gate": true, "hybrid": true}`,
the H3 defaults) at 1k / 4k / 16k, `--jobs 2`, into `runs/strength/scores_v6fix.jsonl`; the same 300 positions and
it13 @ 32k labels as the baselines and the Measurements; `report --scores runs/strength/scores_v6fix.jsonl --configs
nodur hybrid`. "Measure" = the spec v6 Measurements table (engine `e5f3aae`, before the review fixes; nodur from the
baselines at `72e79eb`). Wall: 243 + 212 s (1k+4k) and 636 + 635 s (16k).

All positions (n 300), joint agreement with it13 @ 32k:

| Config | 1k now | 1k Measure | 4k now | 4k Measure | 16k now | 16k Measure | 16k board now / Measure |
|---|---|---|---|---|---|---|---|
| nodur | 59% [54, 65] | 61% [56, 67] | 76% [71, 81] | 76% [71, 81] | 88% [84, 91] | 87% [83, 91] | 92% / 91% |
| hybrid | 59% [53, 64] | 60% [54, 66] | 74% [69, 79] | 75% [71, 80] | 87% [83, 91] | 88% [84, 91] | 90% / 90% |

Root q, Brier and log-loss: both configs 0.054-0.060, 0.119-0.122, 0.375-0.384, as in the Measurements.

Triggered subset (the same 30 positions fire as before, 10%):

| Nodes | Kept sits now / Measure | Triggered: joint agree now / Measure | nodur same ids now / Measure | Act-vs-sit agree now / Measure | Differs from nodur: triggered / rest, now / Measure |
|---|---|---|---|---|---|
| 1000 | 1 / 1 | 60% / 67% | 70% / 70% | 97% / 97% | 3/30, 16/270 / 2/30, 17/270 |
| 4000 | 3 / 2 | 73% / 77% | 83% / 83% | 90% / 93% | 3/30, 17/270 / 2/30, 12/270 |
| 16000 | 2 / 2 | 90% / 90% | 97% / 97% | 93% / 93% | 2/30, 12/270 / 2/30, 8/270 |

Sit rate when behind on clock (sit-or-act subset, n 35): hybrid 6% / 6% / 3%, nodur 3% / 0% / 0% (Measure: hybrid
3-6%, nodur 0-3%). Kept sits by position: 167 at every budget, 222 and 292 at 4k, 42 at 16k (Measure: 167 every
budget, 222 at 4k, 42 at 16k; 292 is new).

Reading: **the review fixes did not change the hybrid's picture on positions.** Every difference from the Measurements
is 1-2 positions of 30, inside the 4-worker noise floor (nodur against itself differs on 1/30 triggered and 11/270
other positions, spec v6 4.1). The hybrid still agrees with it13 @ 32k less than nodur where it sits, and is level
elsewhere. **Position 167 is still a conditionally proven sit** (`choice_reason` proven, q_sit 0.972 against q_act
0.987, 163 nodes at the 1k budget): the review fixes stop proofs that lean on the event below *moving* children
(`a1c3a09`), not the sit's own proof through the event, so the known issue in the Measurements stays open.

### 11.3 Self-play smoke with the chosen hybrid

`build-sp1/hivemind.bin selfplay --model models/twin-s-clock-it13-fp32.onnx --games 3 --nodes 48 --seed 11
--base-mix 15000:1 --durative-tree false --sit-gate true --flag-loss true --instant-transfer false
--mate-race-by-landing false --pass-cap-k 8 --hybrid true` (`runs/wave2_local/hybrid_v6/selfplay`):

| Network | Games | Samples | Hybrid | Wall, peak memory |
|---|---|---|---|---|
| it13 fp32 | 3, no error: flag (Bb), resignation, flag (Ab) | 147 | 146 searches, 17 triggered, 0 sits kept (3 refused by the margin, the rest no sit chosen) | 211 s, 197 MB |

The hybrid runs whole self-play games without error and writes training data (`chunk_11_000000.hvm`), but kept no sit
in these 3 games at 48 nodes, so the smoke does not exercise the "sit never the policy target" fix (`a4a6c0a`); that
is covered by the C++ tests and `tests/test_wave2_training.py`.

### 11.4 Cluster dry runs (fake `sbatch` / `squeue` on PATH)

* `launch_wave2c.sh` (`NO_PULL=1`, `W2C_RUNS` in the scratchpad): checkout at the pin `4b45129`; 21 submissions: the
  build (`-t 4:00:00`, writes `build.ok`), the GPU check (`afterok:build`), 12 rows (`afterok:build`, names
  `w2c_<row>`, `NAME` `<round>/rows/<row>`) including `w2c_h_best_120`, `w2c_bench_{nodur,cont,hybrid}` and
  `w2c_smoke_{nodur,cont,hybrid}` (`afterok:gpu`; smoke `WAVE2_EXTRA` for the hybrid `--durative-tree false
  --sit-gate true --hybrid`), the watchdog (`afterany:build`). Prints "GPU-hours: at most 137". `h_best_120` in
  `launch_checks.sh`: `upstream-it13`, `{"durative_tree": false, "sit_gate": true, "hybrid": true}`, 120000 ms, 600
  pairs.
* `launch_long_test.sh` (`W2C_RUNS=runs/dryrun_v6_long`, round `r1` with `engine.ref` = the pin), build 4242 queued:
  15 parts `w2c_l_{hybrid,nodur,same}_120_p1..p5`, all `--dependency=afterok:4242`, folders
  `<round>/long/<row>/part1..5`, seeds 1001-1005 in every row, watchdog `w2c_watchdog_long` `afterany:4242`, "at most
  120" GPU-hours. Then with `build.ok` and `PARTS=1`: parts 6 (no folder reused), seed 1006, no dependency; the
  options each part gets: `l_hybrid_120` `upstream-it13` with the hybrid's flags, `l_nodur_120` `upstream-it13` with
  `{"durative_tree": false, "sit_gate": true}`, `l_same_120` `wave2-self` with the hybrid on both sides, all 120000 ms,
  200 pairs, `ENGINE_REF=4b45129...`. Dry-run folders moved to the Trash afterwards.
* `resume_ab2.sh` (queue holding build 4242; a made-up `runs/wave2/ab2` with a narrow `.done_loop_1` without
  `iteration.done` and a wide one with it): the narrow marker set aside as `.done_loop_1.skipped-<stamp>`, the wide one
  kept; 3 submissions, `ab2_w0` twice (WIDE 0) and `ab2_w1` (WIDE 1), all `afterok:4242,singleton`, MAX_ITER 2, AB 1,
  10000 games. With no build queued: exit 1; with `BUILD=77`: the same 3 on `afterok:77,singleton` (it still prints
  the "no w2c_build job" line to stderr first, cosmetic). The made-up folder went to the Trash. The ab2 jobs will run
  nodur on the new pin; with the hybrid off it plays bit-for-bit as before (11.1 parity).
* `bash -n` on `launch_wave2c.sh`, `launch_long_test.sh`, `resume_ab2.sh`, `launch_checks.sh`: OK.

### 11.5 Engine head for the pin, and what PACE needs

**The pin should be `4b451290d2417acadaaa111748bfdef14d0ba9b3`** (`talisman/wave2` HEAD, the hybrid with the review
fixes), which `cluster/wave2_engine.ref` already holds. Remote state (read with `git ls-remote`, nothing pushed):
`fork/talisman/wave2` is at `72e79eb`, 12 commits behind `4b45129`; `origin/custom-engine` is at `ca33b42`, behind
the local branch. Before `launch_wave2c.sh` on PACE: push `talisman/wave2` to the fork (the build fetches the pin from
there) and push `custom-engine`. None of this was done here.

### 11.6 Not done or still open

* No strength result: this check only shows the fixed engine scores like `e5f3aae` on positions. The local match
  verdict (0.502 vs nodur, a tie) stands; `h_best_120` and the long test decide.
* Position 167's conditionally proven sit is still open (11.2).
* `hybrid_parity.sh` position 7 needed a second attempt (11.1).
* The self-play smoke kept no hybrid sit (11.3).
* The 8 tar failures are the Mac's bsdtar; the shim used here is not in the repo.

## 12. League round (spec v7)

The integration check of `docs/custom_engine_spec_v7_league.md` (WP1-WP5 as built, plan phases 1-2) before the league
pilot goes to PACE. Engine: `talisman/wave2` at `c45be01` (WP1/3/4 on top of `d5802ee`, plus the integration's test
fix); `cluster/wave2_engine.ref` bumped to it. Main repo: `custom-engine` (WP1/2/4/5 plus the integration fixes below).
Network for every run: twin-s-clock it13 (`engine/models/twin-s-clock-it13.onnx`, sha256 `f4ac25e3...`, its checkpoint
`artifacts/distill/twin-s-clock-it13/twin-s-clock-it13.pt`) on ONNX Runtime (Core ML EP), and the networks the smoke
trained from it. The Mac was busy throughout: other agents' manim renders and old-stack hivemind engines, load 20 at the
start and 100-190 during the parity scripts. Nothing ran on PACE (not reachable).

### 12.1 Integration fixes

| Fix | Where | Why |
|---|---|---|
| `think_head_missing` and the other v7 counters reach game files and summaries | `arena/run.py` (`controller_telemetry`, `V7_TELEMETRY_KEYS`), `arena/stats.py`, `arena/tests/test_controller_telemetry.py` (3 tests) | WP2 kept them on the controller only, so `gate2_match.sh` could never report `think_head_missing`. Now a `game_end` event carries `controller_telemetry` `{team: {...}}`, and so the summary and `results.jsonl`, **only when a team controller has `think_head` on or `opp_sit_pace` other than `own_k`**: rows without v7 options write exactly the files they wrote before |
| `arena/think_policy` guards removed from `arena/tests/test_opp_pace.py` | WP2's test | the module exists since `9fd13f8` |
| `gate2_match.sh` without `launch_settings.txt` | `cluster/gate2_match.sh`, new test | the `sed ... \| head` read exited the script silently under `pipefail`; it now falls back to the pin |
| `gate2_match.sh` merge globbed `part*` | same, test extended | a file named `part1.log` next to the part folders crashed the merge (`int('1.log')`) in the smoke below; only `part<k>` folders count now |
| `league_pilot_status.py` keys | `cluster/league_pilot_status.py` | reads `hivemind.league.segment_summary`'s `mean_k_learner` and `mean_s` (WP5 had guessed other names) |
| `LeaguePaceSelfPlay.OffModeKeepsBudgetsOutputsAndColumns` | `W2:engine/tests/test_league_pace.cc` | its second run must write the same record count, which only a one-worker build guarantees (120 vs 104 records in the 4-worker build); the comparison now runs only when `SearchParams::NUM_SEARCH_THREADS == 1` |
| pin | `cluster/wave2_engine.ref` = `c45be018f15b64d7b087be379a9b39e172a13c52` | contains `d5802ee`; with the fork at the pin the dry-run tests that refused (fork off the pin) run again |

### 12.2 Build and suites

| Check | Result |
|---|---|
| Clean build `W2:engine/build-int` (`-G Ninja -DCMAKE_BUILD_TYPE=Release -DHIVEMIND_BACKEND=onnxruntime -DHIVEMIND_USE_CCACHE=OFF`, 4 search workers, tests on), `nice`, `-j6` | OK, 51/51 steps |
| C++ suite `build-int/tests/hivemind_tests` (full, at `fcc5ac7`) | 653 tests: 650 passed, 1 skipped (`MateRaceTest.MeasureFrozenRaceOnTheGame`), 2 failed, 238 s: `SearchBalanceSelfPlay.StateDumpDoesNotChangeTheRunAndReanalyseReproducesIt` (fails in every 4-worker build, `build-ninja` included; passes in one-worker builds, as WP3/WP4 found) and `LeaguePaceSelfPlay.OffModeKeepsBudgetsOutputsAndColumns` (fixed above; after the fix `Pace*:LeaguePace*` 12/12) |
| Fork Python `uv run --offline pytest tests` (`c45be01`) | 395 tests: 387 passed, 8 failed, 489 s: all 8 the macOS bsdtar `--compare` (6 in `tests/data/test_archive.py`, 2 in `tests/test_cli_rl_loop.py`), as in 11.1. `test_rl_loop_search_balance.py::test_a_challenger_plan_runs_end_to_end`, which WP1 saw fail once, passed (also alone) |
| Main `arena/tests tests/cluster` | 493 passed, 1 skipped, 0 failed, 254 s (the dry-run tests that refused with the fork off the pin pass now) |
| Main `arena/tests_phase1` | 105 passed |
| `bash -n` on `launch_league_pilot.sh`, `gate2_match.sh`, `phase0_chk1.sh`, `full_run_watchdog.sh` | OK |

### 12.3 Parity

| Check | Result |
|---|---|
| Off-mode self-play identity vs `d5802ee` (both one-worker builds; it13, the hybrid flags, base mix 60/120 s, seat move times, 4 games, 50 nodes, seed 4242, state dump on) | **byte-identical**: HDST chunk `search_4242_00000.dst`, `states.jsonl`, `seat_clocks.tsv`, `games.pgn` (less its Date tag), every `selfplay game` line. The `d5802ee` source was checked file by file against `git show d5802ee:` |
| `engine/tests/stock_parity.sh` (`JOBS=4`) | 8/8 identical |
| `engine/tests/wave2_parity.sh` (`JOBS=4`) | 6/8, then 5/8 on a rerun; **not conclusive**: the differing positions changed between runs (5 and 8, then 2, 3 and 5), and the differences are node counts of the last progress line (a search stopped early, 242 vs 314 nodes at a 300 budget), the mark of a starved machine, not of code |
| `engine/tests/wave2_game_parity.sh` (`JOBS=4`) | 4/5: selfplay pgn and clocks, tournament pgn and tsv identical; the 10-search UCI session differed (root width 15 vs 13, 74 vs 42 nodes: an early stop again) |
| `engine/tests/hybrid_parity.sh` (`JOBS=4`) | 4/8 identical with no trigger, positive control fired on 3 of 3. This script compares **one binary with itself** (`HybridMode gate` vs `off`), so its differences cannot come from the v7 code; they are the same early stops |

The v7 commits do not touch the search (`agent.cc` only prints `think_gain` after a verbose search when the network has
the head; it13 has none), the stock parity and the self-play identity are exact, and the same-binary script differs as
much as the cross-binary ones. **The three search-parity scripts should be rerun on a quiet machine** before reading
them as a pass; they were 8/8, 8/8 and 5/5 at load 28 in 11.1.

### 12.4 League smoke on the Mac (rl-loop end to end, gate through the fake Slurm)

`runs/smoke_league/e2e1` (git-ignored). rl-loop from the fork's `.venv`, engine `build-int/hivemind` for self-play,
reanalyse and the 100 ms match: `--wave2 --durative-tree false --sit-gate true --hybrid --base-mix 20000:1 --pace
--pace-nodes 32:96 --league --think-labels --think-label-fraction 0.5 --think-policy --think-weight 0.1 --gate2
--gate2-games 4 --gate2-parts 1 --gate2-base-ms 15000 --gate2-min-pairs 2 --gate2-poll-s 5`, 8 games of 2 per segment,
2 validation games, `--parallel-games 2 --batch-size 64 --match-games 2 --match-movetime 100`. The gate script was a
local wrapper: it calls the real `cluster/gate2_match.sh` with `tests/cluster/fake_slurm` on PATH, then plays each
queued `g2_` job here with the `arena.pool run` line of `eval.sbatch` and the job's saved environment (wave 2 binary
`build-int/hivemind.bin`, upstream `third_party/upstream-hivemind/engine/build-ninja/hivemind.bin`, `MEASURE=1` with 2
repeats and no warm-up, 2 workers; `modules.sh`, warm-up and the parity check are PACE-only and skipped) and marks the
job done. Two iterations, 12 and 17 minutes.

| Stage | it1 | it2 |
|---|---|---|
| league plan (frozen) | 4 segments: self 2, fixed 2 (`fixed:k80`, `fixed:k110`, weight 1), past 0 (no promoted generator yet) | the same: it1's network is promoted but is the current generator, so it is not a past opponent (spec 5.3); past appears from it3 |
| self-play | seg00 self 289 records, seg01 fixed:k80 222 (learner 0-0-2), seg02 self 189, seg03 fixed:k110 153 (learner 1-0-1); val 371 (stock, no pace/league/think flags) | 394, 217 (1-0-1), 346, 146 (0-0-2) |
| pace and think scale (segment markers) | mean k self 213/216, league learner 116/226, opponents 80/110 as fixed; mean s 1.000 (it13 has no head: `think_missing` = every searched decision) | mean s 0.990-0.991, `think_missing 0`: the promoted generator carries the head and self-play scales by it |
| league-only records | `states.jsonl` = header + 2 game lines + 222 records, chunk 222 records: only the learner's decisions | |
| think labels | 853 records, 441 labelled, rates A 0.087 / B 0.132, noise 0.091, 44 s | 1103, 551, 0.150 / 0.112, noise 0.123, 186 s |
| train | 20 steps, `--think-weight 0.1`, head added to it13's checkpoint; `C.onnx` outputs `value, pi_a, pi_b, wdl_out, moves_left, think_gain` | 33 steps from it1's headed checkpoint, replay share 0.20 |
| 100 ms gate | 0.500 (2 games) | 0.500 |
| gate2 (4 games = 2 pairs per arm, 15 s base) | candidate `7cc48e04`: equal 1.000, free 1.000, `think_head_missing` 0 (mean think_gain 0.327, mean scale 0.99 over 44 looks in game 1); incumbent it13 `f4ac25e3`: equal 1.000, free 1.000, `think_head_missing` 208 (no head: clock budget). d equal 0, free 0, sum 0: **promoted** | candidate `fb7059dc`: equal 1.000, free 0.750; incumbent `7cc48e04` **reused from it1's cache** (no games played for it). d equal 0, free -0.25 (SE 0.25, no drop at 2 pairs), sum -0.25: **not promoted**, generator stays |

Also checked on the smoke's files:

* `RUN_ROOT/gate2/wave2.json` written by rl-loop equals `python -m hivemind.gate2 wave2-json` on the same flags (what
  the launcher pre-writes), so the launcher's file and rl-loop's never conflict.
* `promote.done` keeps the 100 ms `score` and gains `gate2`; `stages/gate2.done` has both networks' summaries, arms,
  settings and hours (0.135 and 0.134).
* `cluster/league_pilot_status.py runs/smoke_league/e2e1` prints every stage: league plan with PFSP weights,
  self-play per kind with learner W-D-L, mean k and scale, labels, train steps, both gates and the decision.
* In the gate games the observed opponent pace was used (`opp_k_levels`: mostly `seat`, a few `team`/`own_k`
  fallbacks early in a game).

The smoke's own setup mistake, kept here because PACE runs avoid it by construction: rl-loop's league needs
`<lineage>-it0.onnx` before iteration 1 (the fixed opponent); `wave2_iterate.sbatch`'s init writes it, a hand-made run
root must copy it (the first attempt here failed its seg01 with "league opponent network ... not found").

### 12.5 Cluster dry runs (fake Slurm)

* `DRY_RUN=1 RUN=league1 bash cluster/launch_league_pilot.sh` with the stock local engines: stops with the build
  message (build-sp1 and build-ninja list none of `--pace-k-min`, `--opponent-model`, `--think-policy`), as expected.
* The same with `PILOT_SP_ENGINE`/`PILOT_MATCH_ENGINE` = `build-int/hivemind`: pin `c45be01` (contains `d5802ee`),
  `WAVE2_EXTRA` = `--durative-tree false --sit-gate true --hybrid --pace --league --think-labels --think-policy
  --think-weight 0.1 --gate2`, think check `tg_league1`, chain of 2 `full_league1` jobs with the afterok shim,
  watchdog, `wave2.json` from wave2-json, and the incumbent's 8 gate parts (equal and free, parts 1-4, 100 games each
  at 60000 ms, `ENGINE_REF` = the pin). With `BUILD=1`: the build `lb_league1` first, the think check
  `--dependency=afterok:<build>`, the incumbent parts after the build. The scratch copy under `runs/` was removed.
* `DRY_RUN=1 bash cluster/think_gain_check.sbatch`: prints build, add-head, probe, compare and UCI search steps.

### 12.6 Review fixes (2026-11-27)

Two reviews of the integrated round; each finding checked against the code, fixed with a test or rejected with a
reason. Engine pin bumped to the fork's `60e94fa` (contains `c45be01`); build `W2:engine/build-fixer` (same flags as
`build-int`, `-j4`, `nice`).

| Finding | Verdict | Fix |
|---|---|---|
| R2.1 `think_gain_check.sbatch` with its default relative `OUT`: add-head and the probe run inside the fork checkout, the engine runs from its build folder, so the headed networks land in, and are looked up from, the wrong folder; compare and uci_search FAIL and the chain never starts | confirmed | `0872d38`: `OUT`, `W2`, `ENGINE`, `IT13_*` and the pythons made absolute. A **non-dry local run** from the repo root with no `OUT`, relative `ENGINE` (`build-int`) and relative `IT13_PT`, 10 positions: build SKIP, headed PASS, probe PASS, compare PASS (headed vs it13 0.0, probe think 3.0e-3, value 3.0e-3 vs ONNX Runtime), uci_search PASS (`think_gain 0.3333 0.3333` headed, none it13). New dry-run test on the absolute paths |
| R2.2 sbatch has no `SBATCH_DEPENDENCY` input variable: with `BUILD=1` the incumbent's 8 gate parts started at once on the old binary | confirmed (the repo said so in `launch_checks.sh`) | `bb911b6`: `gate2_match.sh` passes it as `--dependency`; tests check every fake `sbatch` call gets it (and none without). Launcher dry run with `BUILD=1`: all 8 incumbent parts `--dependency=afterok:<lb_ job>` |
| R1.2 an arm with no finished pairs (all games errors, or every part `GAVE_UP`) never wrote `done`: rl-loop polled for `--gate2-timeout-h` (10 h of an idle main GPU job) | confirmed | `bb911b6`: merged as `pairs 0` (summary with `score` null, header-only `pairs.tsv`, a line on stderr) so the net dir is done and `hivemind.gate2.decide` returns "incomplete" at once; test. Note: if it is the *incumbent* whose arm has 0 pairs, its cached result makes every later candidate "incomplete" too; that is now visible at once in the status instead of costing 10 h per iteration |
| R2.3 the watchdog did not notice a failed `lb_`/`tg_` (the chain stays PENDING on DependencyNeverSatisfied, the queue never empties) | confirmed | `6c94d4a`: `sacct --name lb_<RUN>,tg_<RUN>` FAILED/OUT_OF_MEMORY/TIMEOUT cancels the run; tests for a failed build and a timed-out think check |
| R2.4 crash/gate/workers cancels left the run's `g2_` jobs running outside the cap | confirmed | `6c94d4a`: `cancel()` also scancels the run's gate2 arms, think check and build, running or pending (other runs' jobs untouched; test). A gate2 stage that times out in rl-loop still leaves the candidate's parts queued; they stay under the watchdog and its cap |
| R1.1 think labels measured the gain against the budget the head itself had scaled (from it2 on: high-p positions labelled 2B vs 8B, low-p 0.5B vs 2B, pulling the targets toward the head's own mean) | confirmed | fork `60e94fa`: under `--think-policy` each state line also carries `think_ref_nodes` (pace, clock and cap at think scale 1; the field is absent without `--think-policy`, so off-mode state dumps are byte-identical) and `hivemind.think labels` searches base and repeat at it and deep at 4x it. C++ tests (headed `think_probe.onnx`: written references equal `pace_node_budget(..., 1.0, ...)` per search and differ from `target_nodes` where the scale is 1.625; no field without the policy) and Python tests; spec v7 6.2 amended. The pace clock the search saw is still the scaled one (second order) |
| R1.3 `print_think_gain` (one extra batch inference before `bestmove`) inflates a headed network's measured ms/node, so at it1 the candidate gets slightly fewer nodes in both arms | real, **deferred** | the cost is real in play too (the engine runs that inference every verbose search), so only the size differs: about 1 of ~100 batches in the 800-node measurement vs about 1 of ~900 in a 60 s game move, i.e. roughly 1% fewer nodes for the candidate in the equal arm at it1 only (from it2 both networks have the head). The direction is against promotion (conservative). The fix needs an engine UCI option plus controller and measurement plumbing on both repos; left for after the pilot |
| R2.5 test gaps | addressed by the tests above | |

The reviewer's calibration note on R1.1 stays open: labels come from paced self-play budgets (pace nodes 200-3200 at
the pilot defaults) while the free arm applies p at clock budgets of roughly 7,000+ nodes.

Suites after the fixes (fork `60e94fa`, main with the pin bump; load 20-90 from other agents throughout):

| Check | Result |
|---|---|
| C++ `build-fixer/tests/hivemind_tests` (full, 4 workers) | 655 tests: 651 passed, 1 skipped, 3 failed, 223 s. `SearchBalanceSelfPlay.StateDumpDoesNotChangeTheRunAndReanalyseReproducesIt` as before (4-worker builds). `RootMateSearchTest.CertifiedConditionalMateWinsBehindOnTime` and `EngineTest.DetectsKnightCaptureFeedAfterBc6` are new in this run but fail the same way (the first every time, the second on and off) in the integrator's `build-int` binary, which has none of these changes: timing under load, not the fix. `LeaguePace*`, `Pace*`, `ThinkPolicy*`, `ThinkGain*` all pass (2 new tests) |
| Fork Python `uv run --offline pytest tests` | 397 tests: 388 passed, 9 failed, 1594 s: the 8 bsdtar `--compare` as before, and `test_rl_loop_search_balance.py::test_a_challenger_plan_runs_end_to_end` (`--games` 2 where 1 is expected), which fails the same way at `c45be01` in a scratch worktree with the same controller: older than this round (WP1 saw it fail once) |
| Main `arena/tests tests/cluster` | 497 passed, 1 skipped on the first run, plus 1 failure and 3 setup errors that were the fork being off the old pin; after the pin bump those (`test_tune.py`, `test_launch_switch.py`, 27 tests) pass. 7 new cluster tests |
| Main `arena/tests_phase1` | 105 passed |
| `bash -n` on the four changed or launched scripts | OK |
| Launcher dry run `DRY_RUN=1 BUILD=1 RUN=league1` (fake Slurm, engines `build-fixer`) | pin `60e94fa` (contains `d5802ee`), build `lb_`, think check after it, chain shim, and the 8 incumbent gate parts each with `--dependency=afterok:<lb_ job>` |
| Headed self-play (it1 candidate of the smoke below, `--think-policy true`, pace 16-4096 nodes, 2 games) | 68 records, all with `think_ref_nodes`; 21 differ from `target_nodes` by the head's scale (ratio 0.957-0.984; mean scale 0.989) |

Smoke rerun (`runs/smoke_league/e2e_fix`, the 12.4 setup on `build-fixer`, 2 iterations, 39 min): it1 self-play,
labels 401 of 790 (rates 0.158/0.093, noise 0.087), 19 train steps, head added; gate2 candidate equal 1.000, free 0.750,
incumbent it13 1.000/1.000: d free -0.25, **not promoted**. it2 (generator still it13: scale 1.000, every state line has
`think_ref_nodes` equal to `target_nodes`): labels 381 of 777, 25 steps, gate2 candidate 1.000/1.000, incumbent reused
from cache: d sum 0, **promoted**. `league_pilot_status.py` prints both iterations. As in 12.4 the counts show the
plumbing only; this run's promoted network came at it2, so the headed-generator path (labels at `think_ref_nodes` !=
`target_nodes`) is covered by the self-play check above and the unit tests, not by this smoke.

### 12.7 Not done or still open

* R1.3 (think_gain print counted in the measured ms/node) deferred, see 12.6; the label-vs-arena budget calibration.

* Nothing on PACE: the build at the pin, the think_gain check on TensorRT (the TensorRT binding has never been
  compiled), the pilot launch and the existing hybrid1 run's status all wait for PACE access. Before `BUILD=1 RUN=league1
  bash cluster/launch_league_pilot.sh` there: push `talisman/wave2` (the build fetches the pin `60e94fa` from the
  fork) and `custom-engine`. Nothing was pushed here.
* `wave2_parity.sh`, `wave2_game_parity.sh` and `hybrid_parity.sh` need a rerun on a quiet machine (12.3).
* `SearchBalanceSelfPlay.StateDumpDoesNotChangeTheRunAndReanalyseReproducesIt` still fails in 4-worker builds (older
  than v7).
* The smoke's gate counts are far too small to say anything about strength; it shows the plumbing, both decisions and
  the cache, not a result.
* `afterok:<ids>,singleton` through the launcher's sbatch shim is untested on real Slurm (WP5).
* WP2's item 6 on it2's network and reading hybrid1's self-play value targets are still PACE-side
  (`docs/hoarding_diagnosis.md`).
* The 8 tar failures are the Mac's bsdtar.
