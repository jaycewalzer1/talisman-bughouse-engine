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
