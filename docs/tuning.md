# Tuning our engine against upstream-it13 at 120 s

The goal is the final match (`cluster/launch_final_match.sh`): our engine (the wave2 engine in hybrid mode on
twin-s-clock it13) must score above 0.6 against stock upstream v3.0.3 on it13 over 200 games at 120 s. Today it scores
about 0.49-0.51 (`docs/elo_hunt_analysis.md`), so it needs about +70 Elo for even odds and +100 Elo for 87%. This page
is the machinery for tuning the engine's play parameters directly in that match's setting: many candidates in
parallel on PACE, sequential stopping, and a leaderboard.

| file | what it does |
|---|---|
| `cluster/tune_space.json` | the tunable parameters: name, where it goes (player option or raw UCI option), current value, range, step, scale, whether it changes node cost (`speed`), a prior, and why |
| `cluster/tune_round.py` | writes one round of candidates (`oat`, `grid`, `spsa`, `combine`), prints the launch lines and the GPU cost; submits nothing |
| `cluster/tune_match.sh` | launches one candidate against upstream-it13, split into PARTS parallel 8 h `cluster/eval.sbatch` jobs named `tune_<round>_<cand>_p<k>` |
| `cluster/tune_status.py` | merges the parts, scores each candidate against it13 and against the round's baseline (paired), runs the GSPRTs, prints the leaderboard and `scancel` lines for decided candidates |
| `tests/cluster/test_tune.py` | tests of the three tools and that every candidate's options are accepted by the arena |

Everything is standard-library Python and bash; `tune_status.py` and `tune_round.py` run with `python3` anywhere.

## What a candidate plays

Exactly the final match's arena: pool `arena/pools/wave2.json`, the pool's wave2 contender with the candidate's options,
opponent `upstream-it13` (stock upstream, clock_k 110, untouched), bughouse openings, rules v3, 120 s equal clocks,
`MEASURE=1`, the engine pin as `ENGINE_REF`. Part k of every candidate in a round plays seed `SEED_BASE + k` with the
same pairs per part, so candidate A's pair p of part k is the same opening in the same colours as candidate B's.
`tune_status.py` pairs candidates on those shared game pairs. `SEED_BASE` 5000 keeps tuning off the final match's
(3000) and the long test's (1000) schedules.

A candidate's options are the space's `base` (`durative_tree false, sit_gate true, hybrid true`) plus every parameter:
player options by name (`clock_k`, `root_lag_ms`, `hybrid_margin_permille`, ...), engine options inside
`engine_options` (sent after the flags, and only when they differ from the engine's default, so the baseline sends
exactly the hybrid's options).

### Charge: fixed or measured

The arena charges each side `searched nodes x ms_per_node`. In the long test each part measured ms_per_node once
from three 800-node searches, and the measurements were noisy: ours 0.0495-0.0559, upstream's 0.0412-0.0534 (two of
ten parts measured upstream as slow as us, and the hybrid scored 0.54 in those against 0.49 in the others). In a
tuning round that noise would land on every candidate differently. So by default (`"charge": {"mode": "fixed"}` in the
space) every candidate is charged the same fixed values, the medians of those ten parts: **ours 0.0525, upstream
0.0420** (ratio 0.80, the usual one). `--charge measure` instead lets each job measure as the final match does.

Parameters marked `"speed": true` (mate provers, certification, transpositions, mate threads) change what a node
costs in wall time, so a fixed charge would give them their time for free. They are left out of `oat` and `spsa`
unless `--include-speed`, and should be run in a round with `--charge measure`, compared against a baseline measured
the same way. `arena/pool.py` now keys and measures a wave2 arm with its `engine_options` (before this change the
measurement ignored them; rows without `engine_options` keep their keys and values). Note: the mate provers run on
CPU in wall-time slices, while the measurement is three searches from the start position, where they find nothing;
the arena may underprice them in play either way. Treat a "speed" win as provisional until it holds under
`--charge measure` at a larger `MEASURE_NODES`.

## Running a round (on PACE, from the repo root)

```bash
# 1. write the round (nothing is submitted); prints one launch line per candidate and the cost
python3 cluster/tune_round.py grid --name r1 --set clock_k=55,160,220,440,1000 --parts 4
# 2. look at the lines, then launch them all (or a subset)
bash runs/tune/r1/launch.sh
#    a single candidate, dry: prints the sbatch lines, creates nothing
DRY_RUN=1 NAME=r1 CAND=runs/tune/r1/candidates/clock_k_220.json PARTS=4 bash cluster/tune_match.sh
# 3. read it while it runs
python3 cluster/tune_status.py runs/tune/r1
python3 cluster/tune_status.py runs/tune/r1 --cancel      # also prints scancel lines for decided candidates
# 4. take a winner into the final match's ENGINES file (drops the fixed ms_per_node)
python3 cluster/tune_status.py runs/tune/r1 --engine clock_k_220
```

Round kinds:

- `oat`: the baseline plus every chosen parameter one step up and one down (`--mult` steps; bools flipped, each other
  choice). `--params a,b,c` picks parameters (default: all without `speed`).
- `grid`: the baseline plus one candidate per listed value, `--set name=v1,v2,...` (repeatable). Use it for big moves.
- `spsa`: K antithetic pairs `theta +- c*step*Delta` (Delta random +-1 per parameter), every candidate on the same
  schedule. With `--prev <spsa round>` it first steps theta by `gain x (score per step)` from that round's paired
  plus/minus differences, clipped to `--max-move` steps, then writes the next pairs around the new theta. Numeric
  parameters only; `clock_k` moves on a log2 scale (one step = x1.41). SPSA rounds are never stopped early.
- `combine --prev <oat or grid round>`: one candidate with every single-parameter change that beat the baseline
  (`--rule positive`: dElo > 0 and LLR > 0; `--rule ci`: the 95% range above 0), against the baseline, with more parts
  (default 6). Writes `space_next.json` with those values as the next round's baseline (`--space` it next time).

Every round folder holds `round.json`, `space.json` (the space it was written from), `candidates/*.json`, `launch.sh`,
and per candidate `<cand>/part<k>/` (arena rows) plus `jobs.tsv`. A candidate folder is never reused: a new attempt is
a new round name.

## Reading the leaderboard

Per candidate against it13: games, score with a 95% range (normal over game pairs), logistic Elo, and two GSPRTs of
`H0: Elo = elo0` against `H1: Elo = elo1` (default `[0, 20]`): `LLR5` on the pentanomial pair counts (the one the
decision uses) and `LLR3` on per-game W/D/L (rl_loop.py's `sprt_llr`). Against the baseline: shared game pairs, dElo
with a 95% range, `rho` (correlation of the two candidates' pair scores on the same openings: how much the pairing
helps), and a GSPRT on the paired differences of `H0: dElo = delo0` against `H1: dElo = delo1` (default `[0, 30]`),
alpha = beta = 0.05, no decision before 100 games. `--cancel` prints `scancel` lines for candidates decided against
the baseline (either way), never the baseline, never in an SPSA round. The 95% ranges are fixed-sample ranges: read
them at the end, not every hour.

## How many games

Measured on the long test (`runs/wave2c/r4/long`, 671 hybrid pairs: pentanomial [159, 1, 331, 3, 177]): draws are
under 1%, and a pair's two games are uncorrelated, so a game is worth one coin-flip-like sample with standard
deviation 0.5, and one Elo near a score of 0.5 is 0.00144 of score.

| question | games per candidate |
|---|---|
| one candidate vs it13, 95% range +-20 Elo | ~1,160 |
| one candidate vs it13, +-30 Elo | ~520 |
| difference of two candidates on the same openings, 95% range +-20 Elo | ~2,300 x (1 - rho) each |
| GSPRT dElo [0, 15], expected games when the truth is 0 or 15 | ~5,700 (rho 0), ~4,600 (rho 0.2) |
| GSPRT dElo [0, 30] | ~1,400 (rho 0), ~1,150 (rho 0.2) |
| GSPRT dElo [0, 40] | ~800 |
| GSPRT dElo [0, 60] | ~360 |

Throughput: one 8 h job plays 250-300 games at 120 s (about 1.6 GPU-minutes a game). The qos `inferno` caps a user at
**32 GPUs at once**, shared with the training jobs (`full_hybrid1`, `sp_hybrid1`, `balance_prior`), so a tuning wave
is about 28 jobs, **about 8,000 games per 8 hours of wall clock**. GPU-hours are not the limit; the 32-GPU cap is.

What this means: +-20 Elo resolution on a difference costs ~2,300 games a candidate, three candidates per 8 h wave.
Small tweaks (+5..+15 Elo) cannot be resolved before the deadline and are not worth a round. Tuning here can only find
and confirm large effects (30 Elo and up), so test big moves first and use small steps only to polish a winner.

## Recommended first round

`docs/elo_hunt_analysis.md` ranks our own time allocation first: 75% of the clock is never used, clock lead and sits
go with winning, and upstream lost nothing when it played 25 times faster with a permanent clock lead. `clock_k` is a
player option (no engine change) and the only lever with a plausible +50..+100 Elo, sign unknown. So:

```bash
python3 cluster/tune_round.py grid --name r1 --set clock_k=55,160,220,440,1000 --parts 4
bash runs/tune/r1/launch.sh        # 6 candidates x 4 parts = 24 jobs, ~1,150 games each, one 8 h wave, ~180 GPU-h
python3 cluster/tune_status.py runs/tune/r1 --cancel
```

At ~1,150 games a candidate, a difference of 40 Elo from the baseline is resolved (dElo range about +-28). Then:

1. Second round at the best `clock_k` (edit its `value` in a copy of the space, or `combine --prev runs/tune/r1`):
   the sit policy and the search, big moves only:
   `grid --set hybrid_margin_permille=0,40 --set hybrid_pass_prior_permille=300,500 --set CPUCTInitPermille=2250,4000
   --set PolicyTemperaturePermille=800,1250 --set root_lag_ms=0,150` (11 candidates; with 2 parts each, 22 jobs).
2. `combine` the winners, then confirm the combined engine against the baseline with 6-8 parts (2,000+ games).
3. Only then SPSA over the numeric parameters around the winner (`spsa --k 6 --c 2`, 1 part each, several rounds),
   and a `--charge measure --include-speed` round for the mate-prover switches.
4. Before the final, a long check of the chosen engine with `--charge measure` (as the final match charges), e.g.
   a `combine`-style round with 8 parts.

Open question for the user (from `docs/elo_hunt_analysis.md`): `clock_k` and the other controller options are arena
player options of our wave2 arm, passed through `ENGINES` in the final match. Whether a different time manager counts
as "our engine" for the final is the user's call; the harness supports it either way.

## Caveats

- The fixed charge removes measurement noise between candidates but is not what the final match does; a winner is
  confirmed with `--charge measure` before it goes into `ENGINES`.
- A candidate that plays much faster (`clock_k` 1000) finishes its schedule early; the paired comparison uses the
  pairs both candidates finished, so the baseline's job length sets how many pairs a comparison has.
- `BatchSize`, the number of search threads, CPUCT base, FPU reduction and virtual loss are compile-time or need a
  TensorRT rebuild; they are not in the space (see its `fixed` notes).
- Old arena rows without `results.jsonl` are not read by `tune_status.py` (every eval row since 2026-11-17 has one).
