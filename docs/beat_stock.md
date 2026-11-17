# Can one Hivemind that manages its clock beat stock Hivemind?

Written 2026-11-16, before any of these results existed. Launched with `bash cluster/launch_checks.sh beat`
(20 jobs), read with `bash cluster/results.sh` (section "beat stock"). Results land in `runs/beat/`.

## What is being tested

Both teams are one centralized Hivemind (`HivemindTeamController`, the same it04 network) on one machine each.
`serial` makes a search that is asked for while the other seat's search is still running wait for it; without
it, a team gets up to two machines once the boards stop moving in step. Only the contender changes:

| Arms | Contender | What it answers |
|---|---|---|
| `same_*` | identical to stock | controls: must be 0.50 |
| `c25/c50/c200/c100_v*` | another fixed node budget | which fixed budget is stock's best on a 60 s clock |
| `flag_*`, `nosit_*`, `flagnosit_*` | 1% fewer nodes, so it is ahead on time all game; sits on or off | what the yes/no time flag is worth |
| `clock*_v*` | think time = own remaining clock / k (55, 110, 220: about 200, 100, 50 nodes at the start) | does a plain clock-based time manager beat a fixed budget |
| `clock110_v100_open` | as `clock110_v100` without the machine queue | how much the old accounting moves the result |
| `long_*` | 3 minute clock, 600 games | does it hold with more time |

Each row is 1,600 games (800 opening pairs) except the long ones; one row's 95% range is about ±0.035.

## Decision rule

1. **Valid run.** Both `same_*` controls fall inside 0.465–0.535. If not, stop: something in the arena is
   unfair, and nothing else in the table can be read.
2. **Stock's best budget, B\*.** The fixed budget that does best in the `c*` rows. Every claim below is against
   B\*, not against a slow setting.
3. **Replicate before believing.** There are 16 non-control rows, so about one will clear 0.50 by luck. Any row
   that clears it is rerun with `SEED=1` and a new NAME before it counts.
4. **Go: time management is the direction.** A clock or flag arm's 95% range is above 0.50 against B\*, in both
   seeds, and the 3 minute arm is not below 0.50. Then the clock-aware network (real clock values in place of
   the yes/no flag, learned sitting) is the main project, and the headline is in reach.
   **Strong go:** the lower end of the range is above 0.53.
5. **Flag, not clock.** If the flag arms win and the clock arms don't, the gain comes from owning the yes/no
   flag. That still points to the clock-aware network (the flag is what it replaces), but say so plainly: stock
   is beaten through a design flaw in how it sees time.
6. **No go.** Nothing beats B\* after replication. Once stock's budget is chosen well, time management adds
   little, and beating stock needs something other than time.

Model for every row: `hivemind-fp32.onnx` (aminwoo/bughouse-rise-v3, it04-crossboard-risev33), run by the cluster's TensorRT
build of the engine.

## Headline metric from 2026-11-17 (spec v3 D2; read this before any new row)

Written before any row under these rules existed. It replaces "fixed nodes at 60 s against stock it04" as the
number that says whether the project is winning.

**The headline is the score at equal wall-clock on identical hardware, modelled as nodes-as-time.** Each arm (one
binary + one network) has its own `ms_per_node`, measured on the gate GPU (`arena.pool run --measure-ms-per-node`,
or `python -m arena.pool measure --name X` to store it in the pool file): wall ms per *searched* node at 400
requested nodes from the start position. Every search is charged the nodes it actually searched times its own
arm's `ms_per_node` (spec v3 contract 5; the controller side is WP-C's `charge="searched"`). A faster network
therefore gets more nodes per second of clock, which is what "beats the best engine" means. The old fixed
`MS_PER_NODE = 5.5` (Mac CPU, per requested node) is not used for any headline row.

- **Gate rows**: 120 s and 180 s, increment 0, the held-out openings, 800 games each (`GATE_PAIRS=400`). The
  contender plays unjittered; the opponent runs under its own time manager (stock: remaining clock / k with k
  chosen so it spends the same mean share of its clock per search as the contender). Every row of one training
  run uses the same schedule (`GATE_SEED`), so rows are compared paired over the same games.
- **Baseline row `it00`**: the start network itself, untrained, with exactly the contender's settings. An
  iteration's result is its score minus it00's on the same row, with a paired bootstrap 95% range
  (`results.tsv` columns `gate_*_vs_it00*`, or `tools/compare_evals.py <it00 row> <rows>`). A rising score with a
  range that includes 0 is not a training result.
- **Equal-nodes diagnostic** (one row per iteration, `gate_eqn_120`): the same row with both arms charged the
  opponent's `ms_per_node`, so both get the same nodes per second of clock. Headline minus this row is what the
  network's speed is worth; this row alone is network quality plus time management.
- **Statistics**: the unit is the game pair (same opening, teams swapped). Ranges are cluster bootstraps over
  opening pairs (56 clusters with the 8-opening eval set, not 800 games). Accept/reject decisions use the GSPRT
  over normalised Elo with bounds [0, 20], alpha = beta = 0.05 (`tools/sprt.py ROW`, or `--sprt 0,20` on a row to
  stop it once decided). The noise floor is the pooled spread of the five `same_*` controls
  (`launch_checks.sh beat` + `controls`); no row is read as positive inside it.
- **Real wall-clock** only once, for the final match against upstream and Fairy-Stockfish (cluster contention leaks
  into real-time results).
- **Long rows** finish over several jobs: rows resume from their per-game files when resubmitted with the same NAME
  (`eval.sbatch`) or RUN (`iterate.sh`); `tools/rescore.py` rebuilds a cut-off row's result.

Opponents, in order of what each proves (D3): upstream v3 twin-s-clock latest under its own time manager (the
target; WP-G's player kind, not in the arena yet); `it00`; stock it04 at equal clock (continuity with the table
above); Fairy-Stockfish (the floor, later). Rows in the table above were all run under the old accounting
(`requested` nodes charged, 5.5 ms/node, mostly 60 s) and are kept for history only; the c25/c50 rows are rerun with
searched-node charging in `launch_checks.sh harness`.

Harness checks that must pass before a gate row is believed (`bash cluster/launch_checks.sh harness`): identical
arms with measured `ms_per_node` read 0.50; it04 against itself at 0.1 vs 5.5 ms/node shows the fast side winning;
the unchanged widened network (clock build, zero clock weights) against stock it04 at equal clock reads 0.50
within its range at 120 s and 180 s (the M0 parity arm).
