# Custom engine: shared spec

Written 2026-11-17. The contract every part of the custom engine builds against. Branches: `custom-engine` here,
`talisman/custom-engine` in `third_party/hivemind`. Change this file first if a contract has to change.

## Goal

Start from the it04 weights (`hivemind-it04-crossboard-risev33-loss1.556-p82.0`). Remove the hand-written time
and stalling rules, give the network the real clocks, and train it on games where the two boards run
independently on running clocks. Stock Hivemind must stay bit-for-bit unchanged: every new behaviour is behind
an option or a separate build that is off by default.

## Scope

The main change is **real clocks plus removing the hard-coded time and stalling rules**. One engine controls
both of its team's boards, so there are no teammates and no communication: no messages, requests, lending or
scheduler. The lockstep tests showed a split team at best ties one program (`lend_200` 0.501, `docs/beat_stock.md`
and `cluster/results.sh`), and a single agent already sees every board, pocket and clock. Communication would only
come back for a different setting, such as the engine partnering a human.

## 1. Engine (C++, `third_party/hivemind/engine`)

Build option `-DHIVEMIND_CLOCK_PLANES=ON` builds a separate binary `hivemind-clock` with
`NB_INPUT_CHANNELS_PER_BOARD = 41` (37 + 4 clock planes, below). The default build stays at 37 and is
unchanged. Everything else is UCI options, all default off, so stock behaviour is identical unless set.

| UCI | Meaning |
|---|---|
| `position fen <fenA> \| <fenB> clocks <aw> <ab> <bw> <bb>` | Optional clock suffix: remaining ms for A_white, A_black, B_white, B_black. Without it there are no clocks (stock behaviour). |
| `setoption name FreePass value true` | A pass on any board whose side is on turn is legal, alone or on both boards, regardless of `TimeAdvantage` or partner captures (replaces `is_single_pass_legal` / `is_double_sit_legal` in `joint_action.h`). A side not on turn still has only "pass". Mate under FreePass is the arena rule (`arena.state.is_bughouse_mate`): in check, no legal move, and double check or no empty square between king and checker; a stuck board or a blockable check is never a loss, whatever `TimeAdvantage` or the partner board. |
| `setoption name MateProbeAnyTime value true` | The root mate probe also runs without `TimeAdvantage` (`agent.cc` probe gate). |
| `setoption name TreeMoveMs value <ms>` | With clocks given and ms > 0, every macro-ply in the search tree lasts `ms`: on each board, the seat on turn loses `ms` from its clock, whether it moves or passes. A clock at or below 0 is a terminal loss for that seat's team (a flag), scored like checkmate. 0 (default) = clocks frozen in the tree. |
| `go nodes N sample` | As `go nodes N`, and before `bestmove` prints one line `info string sample <base64>`: one `TrainingSample` (selfplay.cc) serialized exactly as selfplay.cc writes samples to its shards, with `gameId=0`, `outcome=0`, `movesLeft=0`, `wdl=1`, `nodes` = nodes searched, `rootQ`, the root's `planes` (quantized as selfplay.cc does), and `policyA` / `policyB` / `jointPolicy` from root visit counts as selfplay.cc builds them. |

Clock planes (only in the 41-channel build; channels 37..40 of EACH board's block, so board A is 0..40 and board
B is 41..81). For a board, "our seat" is the searching team's seat on that board. With `t` = that seat's
remaining seconds:

| Channel in the board block | Value (constant over the 8x8 plane) |
|---|---|
| 37 | our seat: `min(t / 300, 1)` |
| 38 | our seat: `exp(-t / 10)` |
| 39 | opponent's seat on this board: `min(t / 300, 1)` |
| 40 | opponent's seat on this board: `exp(-t / 10)` |

No clocks given: all four planes are 0. Inside the tree with `TreeMoveMs`, planes use the clocks at that node.
The legacy `TimeAdvantage` plane (channel 31 of each board) keeps its current meaning and is still set by the
`TimeAdvantage` option.

## 2. Network (Python, `third_party/hivemind/src`)

- `NUM_BUGHOUSE_CHANNELS_PER_BOARD` becomes configurable (37 default, 41 with clock planes) without changing
  default behaviour; `board2planes` fills channels 37..40 exactly as above when asked to, from given clocks.
- `tools/make_clock_net.py` (in the hivemind repo): loads the it04 checkpoint, widens the shared per-board input
  stem from 37 to 41 input channels with the new weights set to exactly 0, saves a checkpoint, and exports ONNX
  (fp32, and fp16 via the existing converter). Parity test: on random and real positions the new network's
  outputs equal it04's (max abs difference under 1e-5) whatever values the clock planes hold.
- Training: the existing `train_loop.py --mode rl` path must accept 41-channel data and the widened checkpoint;
  `convert_selfplay_data.py` must read shards whose planes have 82 channels.

## 3. Out-of-step self-play (Python, this repo)

- `HivemindTeamController(free=True)`: sends clocks with every `position`, sets `FreePass`, `MateProbeAnyTime`
  and `TreeMoveMs` (= this search's own think time), never sets `RequireMoveOn`, no sit cap. A pass becomes a
  wait that lasts until the next move or piece lands on either board, or `free_sit_ms` (default 500), whichever
  is first; then the controller decides again.
- Think time: remaining clock / k, with k drawn per game and team from a range (default 60..300), clamped as in
  `clock_k`. Every search is charged as `serial` (one machine per team).
- Out of step by default: every search's charged think time is multiplied by a jitter drawn per search from
  `jitter` (default 0.5..2.0), for both teams.
- Recording: with `record=True` every search uses `go nodes N sample`; at game end the arena fills in `gameId`,
  `outcome` (+1 / 0 / -1 from that sample's team's view, flags included) and `movesLeft`, and appends the samples
  to a binary shard in the exact format `convert_selfplay_data.read_binary_shard` reads.
- `python -m arena.selfplay --games N --workers W --nodes ... --out runs/selfplay/<iter>`; cluster job
  `cluster/selfplay.sbatch`; loop script `cluster/iterate.sh` (generate, convert, train, export, gate).
- Gate: the new network (free mode) against stock Hivemind (stock binary, stock rules, its best fixed budget) in
  the jittered arena, using the beat-stock eval path.

## 4. C++ out-of-step self-play (later replacement, built alongside)

`hivemind-clock` tool `async_selfplay`: the same event-driven game loop as `arena/sim.py` (clocks, landing
times, wakes, pocket transfer, FICS mate rule, flags) driving many games per process. It replaces the Python
runner only after it reproduces the arena's event logs exactly for the same openings, seeds and a
deterministic stub player.

## Rules for everyone

- Off by default: the stock binary, stock options and every existing test must behave exactly as before.
- This Mac runs other GPU work and a memory guard: build with at most 4 jobs (`cmake --build ... -j4`), run single
  in-process games, never worker pools, and never touch processes named kokopelli or memguard.
- Tests: `.venv/bin/python -m pytest arena/tests -q`, the hivemind Python tests, and the engine's C++ tests.
