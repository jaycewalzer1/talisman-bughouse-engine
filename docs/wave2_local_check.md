# Wave 2 + 2b local check (Mac CPU, 2026-11-19)

Integration of the four wave 2 packages (engine parts 1-2, arena, training) and wave 2b (wide planes, arrivals
head, `SeatArrivalsMs`, landing history, HDST v3) on this Mac, before anything runs on PACE.

**These are plumbing checks, not strength results.** Every game count below is 2 to 4 games; a 4-game score has a
95% range of about +-0.5. Nothing here says whether wave 2 beats it13. That is the job of `bash cluster/launch_checks.sh
wave2` (`docs/wave2_tests.md`).

Versions: main repo `custom-engine`, wave2-hivemind `talisman/wave2` at `8c22eb7` (engine 2b `2b5d378`, training 2b
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
