# Running on PACE Phoenix

Clone into project storage, not home (home has a 20 GB limit). Run each step from the repo root.

| Step | Where | Command |
|---|---|---|
| 1. Survey modules | Login node (free) | `bash cluster/survey.sh` |
| 2. Write `cluster/modules.sh` | Login node | From the survey: load modules and set `PYTHON`, `TensorRT_DIR`, `CUDA_TOOLKIT_ROOT_DIR` |
| 3. Build and test | GPU node on embers (free) | `bash cluster/build.sh` |
| 4. Parity against the Mac | Same GPU node | `.venv/bin/python cluster/engine_parity.py check` |
| 5. Benchmark | `sbatch` on inferno (billed) | `sbatch cluster/bench.sbatch` |

Step 3 needs an interactive GPU node. Confirm the QoS name and GPU flag against the current PACE docs:

```bash
salloc -A paceship-bughouse -q embers -N1 -n8 --mem=32G --gres=gpu:1 -t 1:00:00
```

The benchmark refuses to start until the parity check has passed. Run `pace-quota` before and after, or read the
copies the job prints, to learn what an hour costs.

Results are in `runs/cluster/bench_<job id>/`. The job measures stock Hivemind at 1, 4 and 8 games at once, then
about 20 games of the untrained scheduler playing itself. That second number is an upper bound on training cost:
it has four engines per game, where a training game against centralized stock has three.

## When to stop

Fix the problem before spending more if any game crashes or plays an illegal move, if a stage gets under 50 games
per GPU-hour, or if GPU utilization stays low.

## Updating the parity reference

`cluster/engine_parity_ref.json` was recorded on the Mac's CPU ONNX Runtime build. To rebuild it there:

```bash
.venv/bin/python cluster/engine_parity.py record runs/<some run>/game_*.jsonl
```

## Custom engine loop

`docs/custom_engine_spec.md`. Needs the `hivemind-clock` build (set `HIVEMIND_CLOCK_BINARY` if it is not at
`third_party/hivemind/engine/build-clock/hivemind-clock`) and the widened network from `tools/make_clock_net.py` in the
hivemind repo (spec v2: `engine/models/hivemind-clock-v2-fp32.onnx` and `hivemind-clock-v2.tar`). `cluster/build.sh`
builds both (TensorRT, `-B build-clock -DHIVEMIND_CLOCK_PLANES=ON`), warms the clock engine and runs
`cluster/check_clock_engine.py`, which needs every position to print `info string think_gain` and to match Python
ONNX Runtime. The v1 files `hivemind-clock*` (41 channels per board) load only with the ONNX Runtime backend (the Mac,
elapsed planes dropped); the TensorRT build rejects any input width but 86, so the cluster needs the v2 files.

```bash
NAME=custom/sp/it01 MODEL=third_party/hivemind/engine/models/hivemind-clock-v2-fp32.onnx sbatch --array=0-7 cluster/selfplay.sbatch
RUN=main ITERS=3 STOCK_NODES=100 sbatch cluster/iterate.sh    # generate -> convert -> train -> export -> gate
RUN=main THINK_LABELS=1 sbatch cluster/iterate.sh             # + think_gain labels (N = THINK_NODES, 200 default)
```

`iterate.sh` keeps a marker per finished step, so resubmitting with the same `RUN` continues. Set `STOCK_NODES` to
stock's best fixed budget from the beat-stock checks. Per-iteration results go to `runs/custom/<RUN>/results.tsv`
and show up in `bash cluster/results.sh`.

## Wave 2 loop

`cluster/wave2_iterate.sbatch` trains wave 2: twin-s-clock iteration 13 continued by upstream's own RL loop
(`hivemind rl-loop --wave2` in `third_party/wave2-hivemind`, branch `talisman/wave2`) on four real seat clocks, the
durative tree, flag loss and no hard sit gate. The design and the engine interface are in
`third_party/wave2-hivemind/docs/wave2_training.md`.

Before the first job: `bash cluster/build.sh` (WAVE2=1, the default) clones `talisman/wave2` from the fork and
builds its TensorRT match engine; the job builds the one-worker self-play engine (`build-sp1`) itself, with 2
compiles at a time. The branch must be on GitHub for the clone (`git push fork talisman/wave2`), or set `WAVE2_REPO`.

```bash
RUN=w2a ITERS=2 sbatch cluster/wave2_iterate.sbatch            # it1, it2 into runs/wave2/w2a
RUN=w2a ITERS=2 sbatch cluster/wave2_iterate.sbatch            # resubmit: it3, it4 (or finishes a cut-off it2 first)
RUN=w2-sitgate WAVE2_EXTRA="--sit-gate true" sbatch cluster/wave2_iterate.sbatch   # an ablation, its own lineage
```

Each RUN is its own lineage: `init` copies the SHA-256-checked it13 ONNX and training checkpoint into
`runs/wave2/<RUN>/models` and `artifacts` (generator, fixed it13 baseline `-it0`, generator checkpoint). Steps leave
`runs/wave2/<RUN>/.done_<step>`; inside the loop, rl-loop keeps a marker per stage (`runs/rl-itN/stages`), so a job
cut off mid-iteration continues at the stage it stopped in. Settings (environment): `RUN`, `ITERS` (1),
`GAMES_PER_ITER` (20000), `SEGMENT_GAMES` (2000), `VAL_GAMES` (500), `PARALLEL_GAMES` (6), `BASE_MIX` (empty = the
preset 60/120/180/300 s at 0.25/0.25/0.35/0.15), `WAVE2_EXTRA` (ablation flags), `LR` (1e-4 heads), `TRUNK_LR` (5e-5),
`PROMOTE_ABOVE` (0.5), `MATCH_GAMES` (400), `MATCH_MS` (100), `WINDOW` (3), `BATCH_SIZE` (1024), `NO_GATE` (0),
`ALLOW_STOCK` (0: refuse engines without the wave 2 options).

Per iteration: self-play, validation games, a value-label check per segment (flags labelled as losses, no records
dropped, unfinished games unlabelled; the stage fails otherwise), training from the generator's checkpoint, a match
against the generator and one against it13 at equal clock (`MATCH_MS` per move for both, both under the wave 2
rules), and promotion only when the candidate scores above `PROMOTE_ABOVE` against the generator. Results:
`runs/wave2/<RUN>/results.tsv` (games, positions, flag games, repeated positions, score vs the generator and vs it13,
Elo vs it13, promoted) and `runs/wave2/<RUN>/runs/rl-loop.log`. The headline against stock v3 at 120/180 s is a
separate arena row (`cluster/launch_checks.sh`, the wave2 group), not part of this job.

### Wave 2b: the wide network, and the narrow/wide A/B

`WIDE=1` starts the lineage from it13 widened (`twin-s-clock-wide`: 26 more input planes per board for the seat
clocks, thinking time, pocket presence and recent landings, plus an arrivals head; all zero-initialised, so it plays
exactly as it13 at first). `init` runs `hivemind.wave2 init --wide` (it widens it13 and checks parity when the files
are missing), the loop runs `rl-loop --wide` (seat timing and arrivals labels go into the chunks, the arrivals loss
at `ARRIVALS_WEIGHT`, 0.05). The fixed baseline stays it13. Details:
`third_party/wave2-hivemind/docs/wave2_training.md`, section "Wave 2b".

`AB=1` is the medium training run: two lineages, `runs/wave2/<RUN>/narrow` (it13) and `.../wide` (it13 widened),
same settings and same self-play seeds (segment k of iteration n plays seed 1000 n + k), each gated against it13 at
equal movetime under the wave 2 rules. Without `WIDE` the job runs both, one iteration each in turn; with `WIDE=0` or
`WIDE=1` it runs one arm, so two jobs on two GPUs halve the wall clock. Results per arm in `<arm>/results.tsv`
(now with an `arrival_labels` column) and side by side in `runs/wave2/<RUN>/results_ab.tsv`.

```bash
RUN=ab1 AB=1 ITERS=4 GAMES_PER_ITER=10000 sbatch cluster/wave2_iterate.sbatch          # both arms, one job
RUN=ab1 AB=1 WIDE=0 ITERS=4 GAMES_PER_ITER=10000 sbatch cluster/wave2_iterate.sbatch   # or one job per arm
RUN=ab1 AB=1 WIDE=1 ITERS=4 GAMES_PER_ITER=10000 sbatch cluster/wave2_iterate.sbatch
```

The wide arm needs a wave 2b engine (it reads a 130-plane network and writes the wave 2b `seat_clocks.tsv`
columns); `SP_ENGINE` / `MATCH_ENGINE` override the engine paths.

Cost (no PACE self-play rate measured yet; GPU-hours ~= games / R + iterations x ~1.3, R = games per GPU-hour at 800
nodes, 6 parallel games): at a guessed R = 300, the medium A/B above (2 arms x 4 iterations x 10,500 games) is about
**290 GPU-hours** (16 h jobs: about 18 job-slots), the big run (the winner, 10 iterations x 50,000 games) about
**1,700**. At R = 150 both double. Read R off the first segment (`runs/rl-it1/search/logs/seg00.log` start and end)
and rescale before committing to the rest.

Checked locally 2026-11-19 (Mac CPU, `AB=1 ITERS=1 GAMES_PER_ITER=4 VAL_GAMES=2 BATCH_SIZE=128 NO_GATE=1`, engine
`build-ninja` for both): both arms ran init, self-play, label check, training and results; the wide arm's chunks went
to v3 with arrivals labels on all 676 + 386 records; `results_ab.tsv` written. 4 min 16 s. No matches, no GPU.

## Wave 2c readiness round

One round on PACE after which one script says GO or NO-GO for the full training run and prints its launch command.
Background and the rows' meaning: `docs/wave2_tests.md` sections 7 and 8.

```bash
bash cluster/launch_wave2c.sh                 # pull, pin check, submit everything (prints job ids and GPU-hours)
bash cluster/launch_long_test.sh              # the long test: hybrid, nodur, control vs it13, ~1300 games a row
bash cluster/status_wave2c.sh                 # queue, build, GPU check, rows, long test, R, smokes; old round, ab2
.venv/bin/python cluster/decide_full_run.py   # GO / NO-GO with reasons, and the full run's command on GO
```

The round needs the wave 2d (hybrid) engine: `cluster/wave2_engine.ref` pins wave2-hivemind `e5f3aae`, so that
commit must be on the fork first (`git -C third_party/wave2-hivemind push fork talisman/wave2`), and custom-engine
pushed, before `launch_wave2c.sh` runs on PACE.

`launch_wave2c.sh` refuses while wave 2 jobs of an earlier round (`w2_*`, `w2c_*`, `ab*_w*`) are queued, because they run
from the same `third_party/wave2-hivemind` checkout, which the round moves to `cluster/wave2_engine.ref` and rebuilds
(`ALLOW_BUSY=1` goes ahead). Each round gets a new folder `runs/wave2c/rN` (named in `runs/wave2c/LATEST`), never
an old one: eval resumes any partial games it finds.

| Job | Waits for | What it measures | Output | GPU-h (limit) |
|---|---|---|---|---|
| `w2c_build` | - | `cluster/build.sh`: TensorRT builds (stock, clock, upstream, wave 2 at the pin), engine tests, parity | `build_w2c_<job>.log`, `rN/build.ok` | 4 |
| `w2c_gpu` (a) | build ok | `cluster/wave2c_gpu_check.sbatch`: pin, wave2 env, `build-sp1` at the pin (`cluster/build_sp1.sh`), wave 2c and hybrid options in both engines, it13 and wide parity on TensorRT, stock parity, 6 self-play games per candidate config (nodur, cont, hybrid) and width on TensorRT | `rN/gpu_check.tsv`, `.json` | 4 |
| `w2c_<row>` (b) | build ok | the 12 `launch_checks.sh wave2c` rows, one 8 h job each (`h_best_120` is the hybrid) | `rN/rows/<row>` | 12 x 8 |
| `w2c_bench_nodur`, `_cont`, `_hybrid` (c) | gpu ok | self-play rate R on the rl-loop segment command, narrow then wide, 150 games each (`cluster/wave2c_bench.sbatch`) | `rN/bench/<config>_<width>/bench.json` | 3 x 3 |
| `w2c_smoke_nodur`, `_cont`, `_hybrid` (d) | gpu ok | one `wave2_iterate.sbatch` iteration per config, narrow and wide (`AB=1`): 200 self-play games, train, gate (60 games) | `rN/smoke_<config>/{narrow,wide}` | 3 x 8 |
| `w2c_watchdog` (e) | build ends | cancels every `w2c_*` job if the build or the GPU check fails, and a row with errors in over 5% of its games | `rN/watchdog.log` | CPU |

**Cost: at most 137 GPU-hours** by the time limits, about 115 expected (the rows use their whole 8 h, the rest
end sooner). An 8 h row plays about 270-300 games at 120 s (the 2026-11-19 round), a 95% range of about +-0.055.

Decision rules (`cluster/decide_full_run.py` docstring, `docs/wave2_tests.md` section 8): NO-GO on any of B1 build
not passed, B2 GPU check not PASS, B3 a smoke arm (nodur, cont, hybrid x narrow, wide) without a finished, gated
iteration, B4 `c_same_120` (or the long test's `l_same_120`) outside [0.46, 0.54], B5 a row missing, under 250 games,
over 5% errors or still queued (long-test parts included), B6 no self-play rate for the chosen config and width, B7
the full run over 10,000 GPU-hours. Config: `cont` (`c_all_120`), `cont_nod1` (`c_all_nod1_120`), the best single
switch or `hybrid` (`h_best_120`) only when it beats nodur with disjoint 95% ranges, or by 0.05 with 500 games in both
rows; otherwise `nodur`. A merged long-test row of 250+ games is read instead of its round row (`l_hybrid_120` for
`h_best_120`, `l_nodur_120` for `c_nodur_120`). Width: wide only when `runs/wave2/ab2/results_ab.tsv` has at least
2 gated iterations with the wide mean 0.03 above narrow and the wide last not behind; otherwise narrow.

With one 8 h job per row the 0.05 rule cannot fire (under 500 games); only a gap of about 0.11 clears disjoint ranges.
To give it the games, resubmit the rows into the same folders (eval resumes each; about 88 GPU-hours more):
`W2C_PREFIX=wave2c/r1/rows bash cluster/launch_checks.sh wave2c`. For the hybrid and nodur the long test does this.

### The long test

`bash cluster/launch_long_test.sh` (after `launch_wave2c.sh`; `ROUND=rN` for another round than `LATEST`) runs three
rows against stock it13 at 120 s with the round's settings (pool contender, equal clock, `MEASURE=1`, bughouse
openings, rules v3), each split into `PARTS` (5) parallel 8 h `eval.sbatch` jobs:

| Row | Contender | Opponent | Stands in for |
|---|---|---|---|
| `l_hybrid_120` | `{"durative_tree": false, "sit_gate": true, "hybrid": true}` | `upstream-it13` | `h_best_120` |
| `l_nodur_120` | `{"durative_tree": false, "sit_gate": true}` | `upstream-it13` | `c_nodur_120` |
| `l_same_120` | the hybrid | `wave2-self` with the hybrid on (control, must read 0.50) | `c_same_120` |

Part k of every row goes to `runs/wave2c/<ROUND>/long/<row>/part<k>` with seed `SEED_BASE + k` (1000 + k), job name
`w2c_<row>_p<k>`; the folder is made at submission and never reused (eval resumes partial folders), so launching
again adds parts k+1... The parts wait for the round's `w2c_build` (or `BUILD=<job id>`); with no build queued and
the round's `build.ok` present they start at once. Being `w2c_*` jobs, the watchdog cancels them with the round, and
decide reads the round as still running while they are queued. `status_wave2c.sh` and `decide_full_run.py` merge each
row's parts (`merged_row_stats`: all games, cluster bootstrap over openings). **Cost: 3 x 5 x 8 = 120 GPU-hours** at
most; at 250-300 games an 8 h job that is about 1250-1500 games a row, a 95% range of about +-0.03.

### The full run

`decide_full_run.py` prints, on GO, one line such as

```bash
RUN=full1 CONFIG=nodur WIDE=0 CHAIN=127 GAMES_PER_ITER=50000 bash cluster/launch_full_run.sh
```

`launch_full_run.sh` checks the pin, makes `runs/full/<RUN>` (refused if it exists; `RESUME=1` adds jobs with the
same settings), submits CHAIN chained 16 h `wave2_iterate.sbatch` jobs (`-J full_<RUN>`, `--dependency=singleton`,
one iteration per job, `MAX_ITER` 10 so leftover jobs exit at once) and `cluster/full_run_watchdog.sh` (cancels the
chain on a failed job or a gate under 0.40 vs it13). `CONFIG` sets `WAVE2_EXTRA` from `cluster/wave2c_configs.py`.
The estimate is 10 x (50,000 + 500) / R + 10 x 1.3 GPU-hours (about 1,700 at R = 300). The chain uses one GPU at a
time: rl-loop's `--distributed` self-play refuses wave 2 options, so wall clock is about the GPU-hour total (1,700
GPU-hours is about 71 days). Splitting self-play over GPUs is not built.
