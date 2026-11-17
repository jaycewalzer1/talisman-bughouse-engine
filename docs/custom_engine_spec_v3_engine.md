# Custom engine spec v3, engine notes (WP-E)

Written 2026-11-17 with E1 and E3 landed on the fork branch `talisman/custom-engine` (commits `687261b` code,
`2e57af8` tests). This file records what the engine now does, the choices made where the audit left room, and the
hooks the durative tree (E2) should build on. Spec v3 (`docs/custom_engine_spec_v3.md`) is the contract; this is
the engine-side detail.

## What landed

### Protocol

`position <fen|startpos> [moves ...] [clocks aw ab bw bb] [elapsed ea eb] [pace paw pab pbw pbb] [reps ra rb]`

- The sections may come in any order; the FEN and the move list end at any of `moves clocks elapsed pace reps`.
- `pace` (shared contract 1): each seat's expected move time in ms. Stored on the `Board` (`has_pace()`,
  `pace_ms(board, color)`), copied with it, cleared by every new `position`. Without `clocks` it is ignored with
  `info string pace needs a clocks suffix; ignored`. Nothing in the search reads it yet; E2 will.
- `reps` (new): occurrences of the current position on board A and on board B, as the game history counts them
  (>= 1). It fills the repetition planes 35-36 and the repetition part of the node key when the controller sends a
  bare FEN, which the arena does today. Sending the game's `moves` from its start FEN works too (it always did); `reps`
  is the cheap path. **WP-C**: send `reps` with the arena's FICS count (board only, reset by any drop).

### UCI options (all new behaviour off by default; existing options and defaults unchanged)

| Option | Default | Effect |
|---|---|---|
| `MateRacePessimism` | `true` | `false`: the root team no longer loses every simultaneous waiting-board mate race in the tree, and the root's race vetoes on claimed wins are skipped. Ablation arm only. |
| `PassExploration` | `0` | `1` = root, `2` = root and depth 1. Expands (pass, best other board) and (best own board, pass) first, then forces `max(1, floor(sqrt(2 P N)))` visits on chosen-pass edges; the `go ... sample` policy target prunes them back (KataGo policy-target pruning). Depth-1 nodes start after 8 visits. |
| `RootNoiseAllGame` | `false` | With `OpeningNoise`, root Dirichlet noise at every ply (lifts the `OpeningNoisePlies` 200 cap). |
| `MoveTemperaturePermille` | `0` | Played root action drawn with probability `visits^(1/T)` when the root is unsolved; never a solver-proven loss when an alternative exists. Fresh seed per search. |
| `MoveTemperaturePlies` | `0` | `0` = temperature at every ply; `n` = only while both boards' game ply is below `n`. |

### Rules in the tree (D4), only under tree clocks (`clocks` given and `TreeMoveMs` > 0)

- **Double flag**: `Board::flag_losers()`. One team flagged loses; both teams on one macro-ply: the team whose most
  negative seat is lower loses; an exact tie is a draw.
- **Repetition and rule50**: `is_draw_on_board` and `is_repetition_draw` return false. A repeated line goes on and
  is scored by its clocks. The repetition planes still show the counts.
- **Time advantage per node**: `Board::node_time_advantage(team, rootTeam, rootAdvantage)`. Under tree clocks it is
  `team_time_edge(team) > 0`, where the edge is the sum over the team's seats of (own - same-board opponent), which is
  also the sum of its two diagonals and is the arena's `team_time_edge`. It is zero-sum: a tie is an advantage for
  neither team. It drives plane 31, the node key's time-advantage bit, the pass rules and the mate rules at each node,
  and at the root too (the `TimeAdvantage` option is overridden by the clocks under tree clocks; the arena sets it the
  same way, `> 0`). Without tree clocks it is the old alternation, so stock and `TreeMoveMs 0` planes are bit for bit
  the old ones.

  Note for WP-F: the diagonal plane (channel 43) is `tanh((t_ourA - t_oppB) / s)`, one diagonal. Plane 31 uses the
  sum of both diagonals. They agree in sign whenever the two diagonals do; when they disagree, plane 31 follows the
  arena's rule. If the audit meant the single diagonal for plane 31, change `team_time_edge` in `board.h` (one line)
  and both teams can then be "ahead" at once.

### Tree reuse (latent until the arena stops sending `ucinewgame`)

- Retained candidates are filed under `Board::reuse_key`: the search key itself when clocks do not affect the
  search (stock: unchanged), otherwise the position, repetition fingerprint, last moves and team, without clocks,
  elapsed times, pass streak or time-advantage bit. The FEN signature still has to match.
- A reused subtree is walked from the real root clocks and every keyed node is re-keyed
  (`Agent::refresh_reused_subtree`) before `reindex_reused_subtree` files it in the transposition table. Under tree
  clocks, when the root key changed, solver verdicts in the subtree are dropped (`Node::reset_solver_state`): a flag
  or sit-off proved on predicted clocks need not hold. Visit counts and values stay.
- With a move temperature the subtree kept is the one below the move actually played.

## Tests (engine/tests)

`test_custom_engine.cc`: `DoubleFlagMostNegativeSeatLosesTieDraws` (100 and 600 ms, `TreeMoveMs` 1000, single
loser; mirror; tie draw), `TimeAdvantageFollowsTreeClocksAtEveryNode` (a double-sit line past the crossover flips
plane 31 and the node key; plane 31 agrees with the sign of planes 37-40), `FrozenTreeClocksKeepTheLegacyTimeAdvantage`
(`TreeMoveMs 0` planes bit for bit), `RepetitionAndRule50AreNotTerminalUnderTreeClocks`, `PositionParsesRepsAndPace`,
`MateRaceAndExplorationOptionsDefaultOff`, `JointGeneratorQueuesBothPassActionsFirst`,
`PassExplorationVisitsAPassAtEveryRecordedRoot` (100 self-play roots from `runs/custom/local1h` in
`tests/data/pass_exploration_positions.tsv`, 100 nodes, FreePass; set `HIVEMIND_PASS_TEST_NETWORK` to run it on a
real network), `MoveTemperatureSpreadsThePlayedMove`, `TreeReuseSurvivesJitteredClocks`.
`test_mate_race.cc`: `MateRacePessimismOffMakesTheRaceSymmetric`.

Not covered here (they need E2): "perpetual on A with mate-in-2 on B evaluates near +1" at search level (only the
classifier is tested: a repetition is never a terminal under tree clocks), "sit for the knight" at depth <= 2, and "a
seat with 3 s makes at most 3000/minMoveMs moves in any line".

## Notes for E2 (durative tree)

Where the lockstep model lives, and what to change:

1. **Charging**: `Board::make_moves` / `unmake_moves` (board.cc) charge `treeMoveMs` to the seat on turn on each board
   every macro-ply. E2 replaces this with per-seat ready times: ours = this search's budget (`TreeMoveMs`), theirs =
   `pace_ms` (default ours). Keep the undo record (`TreeClockUndo`) the single place that restores everything
   make_moves changed; add the ready times to it.
2. **Who acts**: children strictly alternate teams (`Node(~teamToPlay)` in `node.h` `expand_next_joint_child` and
   `try_init_and_expand`; `teamToPlay` is fixed per node). Under E2 the team to act at a child is whoever is ready
   first, which can be the same team again. That breaks the backup sign flip in `SearchThread::backup` (it negates
   every level) and every place that derives a child's team as `~parent`. Store the acting team on the node (already
   there) and make backup negate only when the team changes between levels.
3. **Keys**: `Board::hash_key` mixes raw clocks, elapsed times and the pass streak. Add the ready times. Keep
   `reuse_key` free of all of them.
4. **Flags and sit-offs**: `flag_losers` already scores simultaneous flags by the most negative seat; with landing
   times a flag happens at an exact time, so the earliest flag wins outright and the tie rule only covers equal times.
   `sit_off_losers` (spec v2 D1) becomes "the lower seat must move" (search 8), with D1 kept as the fallback for
   constant `TreeMoveMs`.
5. **Mate races**: `classify_terminal_position` takes `mateRacePessimism`; E2 replaces the root-team-loses rule with
   the landing order (the team whose mate lands first wins). The flag can stay as the ablation switch.
6. **Time advantage**: `node_time_advantage` already reads the node's clocks; under E2 it should read clocks at the
   node's current time (clock minus time already spent thinking by the on-turn seats).
7. **Captured pieces** (search 9): `push_move` adds the captured piece to the partner's hand at once; E2 delays it to
   the receiving seat's next decision after the capture lands. Ablate it.
8. **Pass exploration** stays useful under E2 but its "pass" becomes "wait until the next event"; the forced-playout
   edge list (`Node::passEdges`) is filled from `JointCandidateGenerator::isChoicePass`, which only looks at the
   action, so it needs no change.
9. Things that alternate the time advantage by team and were left as they were (hint lookups only, not correctness):
   the joint mate solver's transposition lookups in `agent.cc` (`search_hash_key(teamToPlay,
   teamToPlayHasTimeAdvantage)` around the solver's tree hints). Under tree clocks they can miss a node whose key uses
   the per-node advantage; they fall back to searching without the hint.
