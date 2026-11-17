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
