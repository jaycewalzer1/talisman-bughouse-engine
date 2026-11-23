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

The round needs the wave 2d (hybrid) engine: `cluster/wave2_engine.ref` pins wave2-hivemind `ff835c8` (e5f3aae plus the 2026-11-22 hybrid review fixes), so that
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
| `w2c_watchdog` (e) | build ends | cancels every `w2c_*` job if the build or the GPU check fails, and a row with errors in over 5% of its games (a long-test row merged over its parts) | `rN/watchdog.log` | CPU |

**Cost: at most 137 GPU-hours** by the time limits, about 115 expected (the rows use their whole 8 h, the rest
end sooner). An 8 h row plays about 270-300 games at 120 s (the 2026-11-19 round), a 95% range of about +-0.055.

Decision rules (`cluster/decide_full_run.py` docstring, `docs/wave2_tests.md` section 8): NO-GO on any of B1 build
not passed, B2 GPU check not PASS, B3 a smoke arm (nodur, cont, hybrid x narrow, wide) without a finished, gated
iteration, B4 `c_same_120` (or the long test's `l_same_120`, at any game count) outside [0.46, 0.54] or long rows read without any
`l_same_120` games, B5 a row missing, under 250 games,
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
again adds parts k+1... The parts play the round's engine only: they wait for the round's own build (the `build`
line of `rN/jobs.tsv`, or `BUILD=<job id>`); with it gone they start at once only when the round's `build.ok` exists
and `third_party/wave2-hivemind` is at the round's pin (`rN/engine.ref`, written by `launch_wave2c.sh`; older rounds:
`cluster/wave2_engine.ref`). A queued `w2c_build` of another round is refused (it moves the checkout), and every part
carries `ENGINE_REF=<pin>`, which `eval.sbatch` checks before it plays. Being `w2c_*` jobs, the round's watchdog
cancels them if the build or the GPU check fails; the long test also submits its own `w2c_watchdog_long`
(`wave2c_watchdog.sh` with `LONG=1`), which cancels every part of a long row whose merged parts have errors in over 5%
of at least 100 games, whether or not the round's watchdog is still alive. Decide reads the round as still running
while parts are queued. `status_wave2c.sh` and `decide_full_run.py` merge each
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
The estimate is 10 x (50,000 + 500) / R + 10 x 1.3 GPU-hours (about 1,700 at R = 300). Without workers the chain
uses one GPU at a time, so wall clock is about the GPU-hour total.

### Concurrent self-play, budgets and the final match

`WORKERS=K` spreads each iteration's self-play over K more GPUs on the same lineage (rl-loop `--shared-selfplay` in
`third_party/wave2-hivemind`): the main `full_<RUN>` job claims self-play segments through files on the shared
filesystem (`rl-itN/stages/segNN.claim`, heartbeat every minute, taken over after 15 minutes without one), and at the
start of each iteration's self-play its hook (`cluster/selfplay_workers.sh`) tops up K `sp_<RUN>` worker jobs that
claim and play the other segments, then exit. The main job plays validation while they finish, then trains and gates
alone, so no worker holds a GPU through training. `GPU_HOURS=N` caps the run: the watchdog sums the GPU-hours of
`full_<RUN>` and `sp_<RUN>` and at N writes `STOP_WORKERS`, touches `rl-loop.stop` (the main job exits before its
next stage) and cancels pending jobs. A worker killed mid-segment loses nothing: its claim goes stale and the segment
resumes from its checkpoint in whoever takes it.

```bash
.venv/bin/python cluster/gpu_budget.py           # balance, measured $/GPU-h, GPU-h left, a plan for half of it
RUN=par1 CONFIG=hybrid WIDE=0 WORKERS=7 GAMES_PER_ITER=20000 ITERS=6 CHAIN=4 GPU_HOURS=900 bash cluster/launch_full_run.sh
.venv/bin/python cluster/training_curve.py runs/full/par1   # gate win rate per 100 games per iteration (dev panel)
ENGINES=my_engines.json NAME=final1 bash cluster/launch_final_match.sh   # 3-engine round robin, 200 games a pair
.venv/bin/python cluster/final_match_status.py runs/final_match/final1
```

**Switching a running run to workers, more iterations or a cap.** `RESUME=1 REPLACE_PENDING=1` may also change
`WORKERS`, `SEGMENT_GAMES`, `PER_JOB`, `ITERS` (the run's last iteration) and `GPU_HOURS`; `CONFIG`/`WAVE2_EXTRA`,
`WIDE` and `GAMES_PER_ITER` stay (they are what the lineage is). The old `launch_settings.txt` line goes to
`launch_settings.history`, and `balance 1` is kept unless `BALANCE=0` is given. The switch is safe at any point of an
iteration: rl-loop plays each iteration's self-play the way it began. An iteration begun with workers writes
`stages/selfplay_mode.done` (`{"shared": true, "segment_games": S}`) and is finished shared by any later job; one begun
without them (no such marker, e.g. hybrid1's it2) is finished by the main job alone with the segment size it began
with, no hook runs and workers leave it, even if the job that resumes it (in self-play, training or the gate) has
workers. The next iteration then starts with the new settings. The cap is the watchdog's, so a `GPU_HOURS` change needs
`NEW_WATCHDOG=1`, which replaces the run's watchdog: it finds the old one by the `watching full_<RUN>` line in its
Slurm log (`OLD_WATCHDOG=id[,id]|none` names it instead; a `full_watchdog` job without a log yet must be named), submits
the new one with `--dependency=afterany:<old>`, then the chain, cancels the old pending jobs and only then the old
watchdog, so two never run together (if that cancel fails, the new one is cancelled instead and the old one keeps
watching without the cap). The new one keeps the old one's `monitor`/`anchors` (its last `watching` line in
`watchdog.log`; `MONITOR`/`ANCHORS` override) and counts the cap from the run's first `watching` line (`CAP_SINCE`
overrides; its crash checks start now). With `BALANCE=1`, `GAMES_PER_ITER` must be a multiple of `SEGMENT_GAMES`.
Tested locally end to end on a copy of `runs/wave2/int-shared` with the fake Slurm in `tests/cluster/fake_slurm`
(`tests/cluster/test_launch_switch.py` covers the launcher alone).

```bash
# hybrid1: 5 workers per iteration from it3 on (it2 finishes single-GPU), 20 iterations, a 400 GPU-hour cap
DRY_RUN=1 RESUME=1 REPLACE_PENDING=1 NEW_WATCHDOG=1 RUN=hybrid1 BALANCE=1 WORKERS=5 ITERS=20 CHAIN=10 GPU_HOURS=400 \
  bash cluster/launch_full_run.sh
RESUME=1 REPLACE_PENDING=1 NEW_WATCHDOG=1 RUN=hybrid1 BALANCE=1 WORKERS=5 ITERS=20 CHAIN=10 GPU_HOURS=400 \
  bash cluster/launch_full_run.sh
```

`gpu_budget.py` measures the account's dollar rate per GPU-hour from its own snapshots (`runs/budget/snapshots.tsv`,
one per call; it needs 20 GPU-hours between the first and last) and until then assumes PACE's 2022 RTX 6000 rate,
saying so. It keeps 30 GPU-hours back for the final match. `launch_final_match.sh` takes a JSON list of three engines
(`cluster/final_match_engines.example.json`) and copies each model into the match folder at launch.

## Training monitor

Every finished iteration of a run gets one record with every KPI and with how the network's use of its inputs
changes, so what moves Elo can be read off the iterations. Elo against fixed anchors is the target; the rest are
diagnostics and alarms. A number that cannot be measured from what exists is `null` and listed under `unavailable`
with the reason, never estimated.

| File (under `runs/full/<RUN>/`) | Written by | What |
|---|---|---|
| `metrics/itN.json` | `cluster/run_metrics.py RUN_ROOT` | the iteration's record (schema 1): outcome (gates, promotion, anchors, Elo vs it13), cost (GPU-hours, Elo per GPU-hour), self-play stats, training curve, search, network (the probe), settings changed, alerts, `unavailable` |
| `metrics/itN.probe.json` | `cluster/probe_net.py eval RUN_ROOT N` | the network on the frozen probe set: held-out loss/top-1/value MAE, calibration, input-group ablations (policy KL and value change per plane group), agreement with the previous iteration and with search, per-layer change and dead units, worst positions and biggest swings |
| `probe/` | `cluster/probe_net.py build RUN_ROOT` | the probe set: built once per run from its validation games and reused by every iteration, so iterations are probed on the same positions |
| `metrics/summary.json` | `run_metrics.py --summary` | every run's iterations in one file (the dev panel reads it) |
| `alerts.tsv` | `run_metrics.py --alerts RUN_ROOT` | one line per new alert: time, iteration, level (`warn`, `stop`), code, message |
| `anchors/itN/<anchor>/summary.json`, `games.tsv` | `cluster/launch_anchor_match.sh` | the iteration's network at 120 s against a fixed anchor (below) |
| `metrics/monitor_jobs.tsv`, `metrics/logs/` | the watchdog | monitor jobs submitted per iteration and their logs |

**Monitor job.** `cluster/monitor_iteration.sbatch RUN_ROOT N` (30 min, one RTX 6000, `-J mon_<RUN>_itN`) builds the
probe set when it is missing (one job builds under the `probe.lock` directory, others wait), runs `probe_net.py eval`
on cuda with the wave2-hivemind venv (torch), then `run_metrics.py RUN_ROOT` and `--alerts` with the repo venv. It needs
`stages/train.done` only, so the probe is ready before the gate ends; the record is rewritten when the gate finishes.

**Anchors.** `cluster/launch_anchor_match.sh RUN_ROOT N` plays `rl-itN/models/C.onnx` (copied to `anchors/itN/C.onnx`
with its SHA-256) at the real 120 s clock against fixed anchors, on the long test's arena setup (pool
`arena/pools/wave2.json`, bughouse openings, v3 rules, `MEASURE=1`, equal clocks) with the gate's engine options
(the `wave2` dict of `stages/match_generator.done`, reduced to the arena player options that differ from the wave 2
values and checked to send the same engine options; for hybrid1 that is
`{"durative_tree": false, "sit_gate": true, "hybrid_mode": "gate"}`). Anchors (`ANCHORS`): `hybrid-it13` (it13 under
the same options: what training added to the network) and `upstream-it13` (stock upstream it13: the headline).
`ANCHOR_GAMES` (400) per anchor in `ANCHOR_PARTS` (2) `eval.sbatch` jobs (`-J am_<RUN>_itN_<anchor>_p<k>`, seed
`SEED_BASE` + k, 5000 + k, so every iteration plays the same openings), every `ANCHOR_EVERY` (2) iterations
(`FORCE=1` any). It plays the engine commit in the run's `launch_settings.txt` (the checkout may differ only outside
`engine/`). Each call does what is missing: parts with a `summary.json` are done, queued parts are left alone, others
are submitted again (eval resumes their games) up to `MAX_TRIES` (3); when every part of an anchor has finished it
writes `<anchor>/summary.json` (`games, score, elo, ci` (Elo, from the cluster bootstrap over openings), `score_ci,
wins, losses, draws, errors, movetime: null, clock, blocks` of 100 games, and the options played) and `games.tsv`
(one game a line, the contender's result +1/-1/0 in the second column). `anchors/itN/done` marks the iteration done.
`run_metrics.py` reads each `<anchor>/summary.json` into `outcome.anchors` of `itN.json` (an anchor still playing is
listed under `unavailable`; the gate's it0 match is `gate_it13` only), and re-reads a finished iteration when one lands.
**Cost: about 21 GPU-hours** per anchored iteration with the defaults (1.6 GPU-minutes a game at 120 s), so about
10 GPU-hours per iteration at `ANCHOR_EVERY=2`.

**Watchdog.** `cluster/full_run_watchdog.sh` drives all of it on every pass (`MONITOR=0` switches it off): a monitor
job for each iteration with `train.done` and no probe (at most `MONITOR_TRIES`, 2, per iteration), anchor matches
for each due finished iteration from `anchors/FROM` on **only when `ANCHORS` is set** (opt-in, for their GPU-hours:
e.g. `ANCHORS="hybrid-it13 upstream-it13"`; a watchdog that takes this code over by its self-handover has none), and `run_metrics.py` plus `--alerts` on the login node when
a marker, probe or anchor result is newer than the last collection, with new alerts copied into `watchdog.log`.
`anchors/FROM` is written at the first pass with the newest due finished iteration, so a watchdog that picks the
monitor up mid-run does not backfill old anchors; `ANCHOR_FROM=N`
overrides. Alerts never cancel anything: a `stop` alert is a recommendation. `GPU_HOURS` counts `mon_` and `am_` jobs
with `full_` and `sp_`; at the cap no new monitor or anchor jobs start, pending ones are cancelled, and finished
anchor parts are still merged. The watchdog stays alive while monitor or anchor jobs are queued after the chain ends.
When `squeue` fails, neither submits anything that pass (finished parts are still merged). `DRY_RUN=1` (watchdog and anchor launcher) prints each `sbatch`/`scancel` instead of running it.

**Panel.** `.venv/bin/python cluster/run_metrics.py --summary runs/full/<RUN>` writes `metrics/summary.json`, which
the dev panel reads (copy the run's `metrics/` and `anchors/*/*/summary.json` to the Mac, or run it on a read-only
copy with `--out`).

**Backfill** an old run (any `wave2_iterate.sbatch` root, e.g. `runs/wave2/ab2/narrow`):

```bash
.venv/bin/python cluster/run_metrics.py runs/full/hybrid1          # records from what exists, probes null
for n in 1 2 3; do sbatch -J mon_hybrid1_it$n cluster/monitor_iteration.sbatch runs/full/hybrid1 $n; done
FORCE=1 bash cluster/launch_anchor_match.sh runs/full/hybrid1 1    # anchors for an iteration the watchdog skipped
bash cluster/launch_anchor_match.sh runs/full/hybrid1 1            # again once the parts end: merges summary.json
```

## Search balance

`cluster/search_balance.py` (spec and simulation: `docs/search_balance.md`) decides, iteration by iteration, how a
run spends self-play GPU time: deep searches on a share f of decisions at k x 800 nodes (the playout cap), cheaper
b-node searches on the rest, and a Q-vs-outcome value blend lambda in the trainer. Today's recipe is theta0 =
(f 0, b 800, lambda 0). Each iteration it plans the segments (`balance/itN.plan.json`), one challenger segment plays
a neighbouring recipe on the same GPU time as its twin seg00 (fewer games when it costs more per game), paired
mini-trains on the two score the difference d on a probe, and `decide` promotes, continues or rejects
(`balance/itN.json`). Validation self-play always stays stock. **Off by default**: without `BALANCE=1` no command,
marker, log line or metrics field changes.

**What it does with no prior.** Until `runs/full/<RUN>/balance/prior.json` exists *and* says the mini-train/score
stage is built (`score_stage`), every plan is mode `frozen` at theta0: no challenger, no overhead, only the realised
costs logged. The panel then shows "holding theta0: no measured prior". f > 0 arms stay masked until the probe bias
was measured small (`probe_bias_measured`), lambda > 0 until the trainer has `--search-q-blend`, k = 8 until the
bench saw it fit (`fits_bench`), reanalyse until its fidelity test passed.

**Flags.** `wave2_iterate.sbatch BALANCE=1` adds `--search-balance --search-balance-controller
<repo>/cluster/search_balance.py` (and `--search-balance-prior BALANCE_PRIOR` when set) to rl-loop for the main job
and its workers; `BALANCE_ARGS` replaces those flags verbatim. With `BALANCE=1` every main job first brings
`build-sp1` to the checkout's commit (`cluster/build_sp1.sh`, a no-op when current). If that build fails, the
engine has no `--deep-fraction`, the controller file is missing or rl-loop has no `--search-balance`, the main job
logs `WARNING search balance OFF` and runs today's recipe (as `BALANCE=0`) instead of failing, since a FAILED job
makes the watchdog cancel the chain; a broken engine binary still fails it (a worker always dies on these). That rebuild is safe in a chain: `--dependency=singleton` means no other main job of the run is
running, workers start after it, and nothing else uses `build-sp1` (monitor jobs are Python; gates and anchors play
`build-ninja`, which is never rebuilt here). `launch_full_run.sh BALANCE=1` passes it on and records `balance 1` in
`launch_settings.txt`; with it, the wave2-hivemind checkout may also fast-forward to a pin whose `engine/` differs
(the rebuild above happens at the next job, never under a running one).

**Turning it on for a running run** (hybrid1: the running job and the watchdog stay; the next job starts with it):

```bash
# after the engine + rl-loop change is pinned in cluster/wave2_engine.ref (byte-identical with the cap off)
DRY_RUN=1 RESUME=1 REPLACE_PENDING=1 RUN=hybrid1 BALANCE=1 bash cluster/launch_full_run.sh   # prints scancel/sbatch
RESUME=1 REPLACE_PENDING=1 RUN=hybrid1 BALANCE=1 bash cluster/launch_full_run.sh
```

`REPLACE_PENDING=1` submits the same number of jobs as there are PENDING `full_<RUN>` jobs (`CHAIN` defaults to
it) with the new environment, and only after every sbatch succeeded cancels those PENDING jobs (by id, `scancel -t
PENDING`); singleton keeps the new jobs behind them, so a failed sbatch or squeue never leaves the run without a
chain, and `launch_settings.txt` is rewritten only after the cancel; the other settings come from `launch_settings.txt` and must
match apart from `engine` and `balance` (the old line goes to `launch_settings.history`). No new watchdog unless
`NEW_WATCHDOG=1`. Switch back the same way with `BALANCE=0`. Note that moving the checkout also gives the running
job's later subprocesses (training, gates) the new Python, which must be identical with balance off.

**The prior** (`cluster/balance_prior.sbatch`, one GPU job array on finished iterations of an existing run; it writes
only `balance/prior.json`, `balance/prior_work/` and, if missing, the run's frozen `probe/`):

```bash
RUN_ROOT=runs/full/hybrid1 ITERS="1" bash cluster/balance_prior.sbatch plan     # pairs, GPU-h estimate, sbatch line
RUN_ROOT=runs/full/hybrid1 ITERS="1" BUILD=1 sbatch --array=0-3 cluster/balance_prior.sbatch
```

Task 0 is the budget sweep (gamma per node count, `fits_bench`), task 1 the probe-bias measurement (a f=1/8,k=4
segment on seg00's GPU time and seed, probe-mode self-play at k = 16 for a deep probe, mini-trains inc vs chal), tasks
2.. one theta0-vs-theta0 replicate each on two disjoint finished segments (sigma_hat from seed pairs, tau_hat from the
spread across replicates). The last task to finish scores every mini net with `probe_net.py eval` and writes
`prior.json`. It plays on `engine/build-balance-sp1` (`BUILD=1` builds it), never `build-sp1`. **Cost: about
4.7 + 0.75 R GPU-hours** (R replicates: 2 per iteration of 5 segments, at most `REPLICATES`, 10): ~6.2 GPU-h on hybrid1
after it1, ~7.7 after it2. It is the user's call to submit. Its GPU-hours (`cost_gpu_h`) are added once to the cost
of the first balance iteration in `run_metrics.py`.

**Costs during the run.** Planned overhead per controlled iteration ~1.0-1.3 GPU-h (mini-trains 0.65, probe refresh
0.55 every 2nd iteration, scoring 0.05, mini-gate 0.3 when it can decide), capped at 10% of the run's GPU-hours; the
challenger segment itself is not overhead (its games train the main net). In frozen mode the overhead is 0.

**Monitor and panel.** `run_metrics.py` merges `balance/itN.json` (and the plan) into `metrics/itN.json` as
`"balance"`: incumbent and challenger, decision with Z, d and its 95% range, evidence, overhead (frac vs the 0.10 cap),
per-arm games/GPU-h, visits/s, deep and masked shares, sit and flag rates, main mix, probe version and the prior's
status. Cost gets `balance_overhead_gpu_hours` (inside the iteration's window, so already in `gpu_hours`),
`selfplay_gpu_h_by_arm`, and on the first balance iteration `balance_prior_gpu_hours` / `gpu_hours_total` (what
`gpu_hours_cum` and `elo_per_gpu_hour` use). The dev panel's Metrics step has a "Search balance" section. Status by
hand: `.venv/bin/python cluster/search_balance.py status runs/full/<RUN>`.
