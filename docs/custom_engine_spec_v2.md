# Custom engine spec, addendum v2: thinking-time input, think-time head, piece-arrival head

Written 2026-11-17, to be built after `docs/custom_engine_spec.md` (v1) lands, and folded into it then. Same rules:
the stock binary, stock network and stock options stay unchanged; new behaviour is behind the clock build or an
option that is off by default. All three must be in place before the first training round, so the network is
widened once.

## A. Input: how long the side to move has been thinking (43 channels per board)

Each board block grows from 41 to 43 channels (board A 0..42, board B 43..85); the clock build's
`NB_INPUT_CHANNELS_PER_BOARD` becomes 43. With `e` = seconds since the last move landed on that board (or since the
game started), for whichever side is on turn there:

| Channel in the board block | Value (constant plane) |
|---|---|
| 41 | `min(e / 10, 1)` |
| 42 | `1 - exp(-e / 2)` |

Whose turn it is stays in the existing on-turn plane (25), so these two planes say how long that side has been
thinking. No timing given: both 0.

- UCI: the clocks suffix gains an optional `elapsed <ea> <eb>`: ms since the last move landed on board A and on
  board B. `position fen <A> | <B> clocks <aw> <ab> <bw> <bb> elapsed <ea> <eb>`.
- In the tree with `TreeMoveMs`: a board's elapsed time resets to 0 when a move is played there, and grows by
  `TreeMoveMs` when that board passes.
- Arena: the controller sends each board's time since its last landed move.
- Network: the shared per-board stem widens from 41 (v1) to 43 inputs, with the new weights at exactly 0. Parity
  with it04 must still hold whatever the new planes hold.

## B. Output: think-time head ("does thinking longer change my move?")

A new network output `think_gain`: one logit per board, the chance that a search 4 times longer would play a
different move on that board. Head: small (global pool of the shared board features, one hidden layer), last layer
initialised to 0, so policy, value and everything else are unchanged at the start.

- Labels: `tools/think_labels.py` takes positions from recorded self-play shards, searches each at N and 4N nodes
  with the clock engine, and writes label 1 if the chosen move on a board differs, else 0 (boards not on turn are
  masked). Output: shards readable by the converter, label column `think_gain_a`, `think_gain_b`, with a mask.
- Training: a masked binary cross-entropy term, weight 0.1 by default, only on samples that have labels.
- Engine: reads the extra output when the network has it and prints `info string think_gain <pa> <pb>` with each
  search's root evaluation; ignores it otherwise. Stock network: no such output, no line.
- Arena (free mode, option `think_head`, off by default): this move's node budget = clock rule budget x
  `(0.5 + 1.5 * p)` for the seat's board, where `p` is the root `think_gain` from a 16-node quick look charged to the
  same budget. Off: clock rule only, as in v1.

## C. Output: piece-arrival head (auxiliary training target)

A new output `arrivals`: 20 values, for each of the 4 pockets (ours on A, theirs on A, ours on B, theirs on B,
team-relative as in the clock planes) and each piece type (P, N, B, R, Q), the predicted number of pieces of that
type arriving in that pocket within the next 5 simulated seconds. Head: small, last layer at 0.

- Labels: the recorder knows every capture and when it lands, so for each sample it counts pieces arriving in each
  pocket within 5 s after the sample's search started. Stored with the sample (see shard format below).
- Training: mean squared error on the counts, weight 0.1 by default. The engine never reads this output.

## Shard format

Samples written by the arena recorder gain a trailing block: `think_gain` labels (2 floats + 2 mask bytes, masked
when absent) and `arrivals` (20 floats). Shards carrying it start with a version marker so
`convert_selfplay_data.py` reads both the old format and this one; the converter writes the new columns to parquet,
missing values masked. Round-trip tests for both formats.

## Done means

- Clock build: 43-channel planes in C++ and Python agree (0 mismatches with clocks and elapsed times on a few
  hundred positions); the widened network equals it04 on every output it04 has, whatever the new planes hold.
- The engine loads a network with and without the new outputs; the stock binary and stock network are unchanged and
  all existing tests pass.
- End to end on the Mac: 2 recorded games carry arrival labels; `think_labels.py` labels a few positions; a few
  training steps use all three losses; the exported network loads in the clock engine and prints `think_gain`.

## D. Fixes carried over from the v1 build and review

1. **Sit-offs resolve in the tree.** In v1 games both teams sat in check on their boards until one flagged, and
   the search could not see it: with `TreeMoveMs` about 200 ms the flag is hundreds of macro-plies deep. Rule: in
   the clock build with `TreeMoveMs > 0`, when a full round passes in the tree with no move on either board (every
   side on turn passed, so the position repeats with only clocks changed), the node is terminal and is decided by
   the clocks: the seat on turn whose clock runs out first loses for its team, as if nobody ever moves again
   (equal: draw). C++ test with a hand-built sit-off.
2. **No flood of repeated samples.** The recorder keeps at most one sample per seat per unchanged position (same
   FENs, ignoring clocks) in a row; repeated searches while sitting are dropped, and the count dropped is logged.
3. **`movesLeft` counts moves.** Set from the number of moves landed after the sample's search started, not from
   the number of searches.
4. **Waiting-board mate check under FreePass.** `has_unavoidable_waiting_board_mate` (`searchthread.cc`) adds the
   pass reply whenever FreePass is on, not only with TimeAdvantage.
5. **Joint-policy loss.** Find why `train_joint_policy_loss` reports 0.0000 on samples that carry joint policies
   (the reviewer's lead: the loss needs at least 7 model outputs and `joint_policy_loss_factor` defaults to 0).
   Turn it on if the model supports it, else document why it stays off.

## Layout note from v1

it04 (crossboard-risev33) has no separate per-board stem; its first conv sees both boards. The widening keeps
the per-board block layout of v1 (board A 0..40, board B 41..81) and for v2 extends each block to 43 (A 0..42,
B 43..85), touching the stem conv and the board-token projections, new weights at 0.
