# The ms/node "speed gap" is in the measurement, not the engine (2026-11-23)

The arena charges each arm `searched nodes x ms_per_node`, with `ms_per_node` measured once per eval part
(`MEASURE=1`: one 64-node warm-up, then three 800-node searches from the start position; contender first). In the
r4 long test our wave 2 hybrid read 0.050-0.054 ms/node in every part and upstream it13 0.042 in most parts
(0.052-0.053 in two), so we searched 12-19% fewer nodes per decision (`docs/elo_hunt_analysis.md` section 3).

## What the engines really cost on the GPU

PACE job 16047669 (`-J elohunt_speed`, one Quadro RTX 6000, atl1-1-03-007-17-0, 3 min 45 s): the main checkout's
binaries (upstream v3.0.3 `upstream-hivemind/engine/build-ninja`, wave 2 `wave2-hivemind/engine/build-ninja` at
0d6b72e) on twin-s-clock it13 (`twin-s-clock-fp32.onnx`, sha256 efe731c0...), TensorRT FP16 batch 8, 4 search
workers, each config a fresh process with 64 + 2 x 800 nodes of warm-up, then 10 x 800 nodes from the start position
plus 8 positions of `stock_parity_positions.txt` x 2. Engine files: `mU` = upstream's cached TensorRT engine, `mW` =
wave 2's (different bytes, same signature), `mF1`/`mF2` = built fresh in the job. Script and raw results:
`tools/speed_gap/` (`elohunt_speed.sbatch`, `gpu_bench.py`, `results_16047669.jsonl`).

| config (3 rounds, alternating order) | ms/node, start position | ms/node, 8 positions |
|---|---|---|
| upstream binary + upstream engine file | 0.0428 / 0.0435 / 0.0434 | 0.0417 / 0.0475 / 0.0422 |
| wave 2 hybrid + wave 2 engine file | 0.0431 / 0.0431 / 0.0433 | 0.0416 / 0.0425 / 0.0425 |
| upstream binary + wave 2 engine file | 0.0428 / 0.0434 / 0.0431 | 0.0438 / 0.0485 / 0.0437 |
| wave 2 hybrid + upstream engine file | 0.0431 / 0.0436 / 0.0432 | 0.0441 / 0.0423 / 0.0423 |

Single runs: wave 2 stock flags 0.0433, nodur 0.0431; fresh engine files upstream 0.0435 / 0.0430, hybrid 0.0431 /
0.0431. Node sweep: 200 nodes 0.0557 (upstream) vs 0.0568 (hybrid), 3200 nodes 0.0416 vs 0.0420. Raw inference
(`hivemind.bin bench`, batch 8): 1874 (upstream file) vs 1870 (wave 2 file) batches/s. CPU side alone (a near-free
dummy network with it13's inputs and outputs, `tools/speed_gap/dummy_net.py`): 0.0107-0.0110 (upstream) vs
0.0110-0.0114 (hybrid) ms/node, i.e. about 3% more tree work per node, which is under 1% of the real per-node cost.

Locally (M4 Max, ONNX Runtime) the two binaries also match: 0.216 vs 0.217 ms/node (Core ML), the same NN batch
count per 800-node search (105-106 batches of 8), and with the dummy network 5.9 vs 6.2 ms per 800-node search.

So on the same GPU, warm, the engines run at the same speed (within 1%); the engine file, the binary and the wave 2
flags make no difference. There is no engine-side per-node cost to remove, so the fork is unchanged.

## Where the 20% comes from

The arena's own measurement. In every eval part the wave 2 contender is measured first, right after the job's
warm-up processes and with the GPU idle before it (the idle RTX 6000 sits at 300 MHz, 1845 MHz under load); upstream
is measured about a second later on a busy GPU. The first-measured arm reads ~0.050-0.054, the second ~0.042 (and
upstream, too, read 0.052 in the parts where it presumably met a cooled GPU). The same binaries measured warm read
0.043 both. This is an inference from the table above plus the ms_per_node.json files of every PACE row since
2026-11-18; it has not been tested directly (a cold-start run on PACE), so the first row that uses the fix below
should confirm it from its own `ms_per_node.json`.

## The fix (opt-in)

`arena.pool run --measure-warm-ms W --measure-repeats R` (eval.sbatch: `MEASURE_WARM_MS`, `MEASURE_REPEATS`; with
`MEASURE=1`): every wave 2 / upstream arm runs `MEASURE_NODES`-node searches for W ms of search wall time in its own
measuring process before the R measured searches. Off by default (W = 0 is the old procedure, byte for byte); the
cache record gets `warm_ms` when set. Nothing in the engine or the search changes.

Suggested for the final match and any row whose score is meant to be compared with it:
`MEASURE_WARM_MS=3000 MEASURE_REPEATS=10` (about 4 s more per part). Expected effect: our charge drops from ~0.052
to ~0.043 while upstream's stays ~0.042-0.043, i.e. about 20% more nodes per decision for us at equal clocks. In the
r4 long test the two parts where upstream happened to measure as slow as us scored 0.541 (n=579) against 0.493
(n=766) in the other three. That split is weak evidence, not a predicted gain: the per-part scores are
0.460 / 0.509 / 0.579 / 0.508 / 0.500, one of the two equal-charge parts (part 5) scored only 0.500, and the gap
rests mostly on part 3. The difference of ~0.047 has a binomial SE of ~0.027 (z ~ 1.7, not significant at 0.05,
and less so once the correlation within colour-swapped game pairs is allowed for). Even at face value 0.54 is well
short of the 0.6 target, so equal charge is not expected to reach 0.6 on its own. Confirm the effect with the
first row run with `MEASURE_WARM_MS=3000 MEASURE_REPEATS=10`: both arms should read ~0.043 in its
`ms_per_node.json`.

A shared `ms_per_node.json` (iterate.sh gates, `--ms-per-node-cache`) never mixes procedures: a cached wave 2 /
upstream record whose `warm_ms` (absent = 0) differs from the requested one is re-measured rather than reused;
`--ms-per-node` overrides are left alone.
