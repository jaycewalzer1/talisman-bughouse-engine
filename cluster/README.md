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
hivemind repo (`engine/models/hivemind-clock-fp32.onnx` and `hivemind-clock.tar`).

```bash
NAME=custom/sp/it01 MODEL=third_party/hivemind/engine/models/hivemind-clock-fp32.onnx sbatch --array=0-7 cluster/selfplay.sbatch
RUN=main ITERS=3 STOCK_NODES=100 sbatch cluster/iterate.sh    # generate -> convert -> train -> export -> gate
```

`iterate.sh` keeps a marker per finished step, so resubmitting with the same `RUN` continues. Set `STOCK_NODES` to
stock's best fixed budget from the beat-stock checks. Per-iteration results go to `runs/custom/<RUN>/results.tsv`
and show up in `bash cluster/results.sh`.
