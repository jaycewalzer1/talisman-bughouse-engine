# Self-play throughput bench (WP-I)

Audit `docs/audit_2026-11-17.md` WP-I, self-play finding 9. One job measures how many self-play games one PACE GPU
produces per hour at production settings, and how busy the GPU and the CPUs are while it does.

## How to run

```
sbatch cluster/bench_selfplay.sbatch            # ~1 GPU-hour: 60 s x 60 games, 180 s x 30 games (6 workers), 60 s x 72 games (12 workers)
sbatch -n16 cluster/bench_selfplay.sbatch       # same stages with 16 cores, if the first run says CPU-bound
STAGES="60000:6:60 180000:6:30 mix:6:60" sbatch cluster/bench_selfplay.sbatch   # also measure the base-time mix directly
STAGES="60000:1:1" bash cluster/bench_selfplay.sbatch     # local dry run of the script path, no GPU
.venv/bin/python cluster/bench_selfplay_summary.py runs/cluster/bench_selfplay_<job> --cpus 8   # re-summarise
```

Settings are those of `cluster/selfplay.sbatch`: the spec v3 regime (`SP_REGIME=v3`, the default since 5f21253;
`SP_REGIME=v2` gives the old fixed-k regime), `hivemind-clock` TensorRT build, `hivemind-clock-v2-fp32.onnx` (the job
prints its sha256 prefix), 16-1600 nodes, free sit 500 ms, one RTX 6000, 8 cores, 32 GB, and 6 workers = 6 games at
once x 2 engine processes (one per team) x 4 search threads = 48 search threads. Under v3 each stage fixes the base
time with `--base-mix BASE:1`; k = base / think target (200-1000 ms), which is the audit's k 60-300 at 60 s and
k 180-900 at 180 s; the pace mixture replaces the jitter, and time odds, PCR, searched-node charging and the
base-scaled ply cap are on, as in production. A `mix` stage draws the base from the production mix and measures the
mix rate directly. The same `SP_*` overrides as `selfplay.sbatch` are passed through; **set `SP_MS_PER_NODE` to the
GPU-measured value**, or every budget uses the Mac's 5.5 ms/node and the games are not production games. The engine is warmed before the first stage so the TensorRT build (~200 s) is not counted. Samples
are written as in production, since that cost is part of the rate. Flags added later by other packages (WP-C's
`min_move_ms`, base-time mix) go in `EXTRA_ARGS`; write them next to the numbers below.

Output in `runs/cluster/bench_selfplay_<job>/`: `stages.tsv`, per stage `summary.json`, `gpu.csv` (nvidia-smi every
1 s), `cpu.tsv` (the job's Slurm cgroup `usage_usec` every 2 s, else `/proc/stat`), and `report.json`.

## What each number decides

| Number | Decides |
|---|---|
| GPU utilisation (mean of 1 s samples) at 6 workers | Build a C++ cross-game batching server **only if under ~30%**, and only after WP-C's protocol fixes (persistent engines, no `stop`+`isready` after each `bestmove`) are in and the bench is rerun on them. At 30% or more, no batching server. Never port the Python simulator for its own sake. |
| CPU busy share of the allocated cores | At 85% or more the run is CPU-bound, and low GPU use says nothing about batching: rerun with `-n16` (or fewer search threads) and decide from that run. |
| Games per GPU-hour at the self-play base-time mix (spec v3 D2: 60/120/180/300 s at 0.25/0.25/0.35/0.15, interpolated linearly from the 60 s and 180 s stages) | GPU-hours per 10,000-game training round. **Under 50 games per GPU-hour: halve the training-round count, not the ablations** (audit section 4). |
| Steady games per GPU-hour, 6 vs 12 workers at 60 s | Close to x2: run more workers per GPU in `selfplay.sbatch` / `iterate.sh`, the cheapest fix. Close to x1: the GPU or CPU is already saturated. |
| Errors, flag rate, mean game length | Sanity only: the bench games must look like production games (local1h: 58% flags at 60 s). |

The mix rate comes from a `mix` stage when there is one; otherwise it is interpolated from the 60 s and 180 s stages
(the summary line says which).

"Games per GPU-hour" is shown two ways: end to end (games / stage wall time, which includes the tail where fewer
than `workers` games run) and steady (workers x 3600 / mean game wall time). The decisions use the steady figure.

## Results

| Date | Job | Commit | Cores | Stage | Games | g/GPU-h steady | GPU % mean | CPU busy | Notes |
|---|---|---|---|---|---|---|---|---|---|
| 2026-11-17 | local dry run (M4 Max, CPU ONNX, no GPU) | e54418f | 14 | 60 s, 1 worker, v2 | 1 | 41 (one CPU game of 87 s wall, not a GPU figure; an earlier 40 s checkmate gave 89) | n/a | n/a | Script path check only. |
| 2026-11-17 | local dry run, v3 and v2 | 7add543 + this change | 14 | 60 s and mix, 1 worker | 0 | n/a | n/a | n/a | Script path ran end to end (warm, v3 flags accepted, summary on failed stages), but every game died in `arena/recorder.py` `infer_channels`: the clock engine rebuilt at 15:06 from hivemind a413293 (diagonal clock plane, 44 channels per board) emits 88-channel samples and the recorder knows only (86, 82, 74). Self-play is broken on this build until that is fixed; rerun the dry run then. |

No PACE figure yet; the first `sbatch cluster/bench_selfplay.sbatch` fills this table.
