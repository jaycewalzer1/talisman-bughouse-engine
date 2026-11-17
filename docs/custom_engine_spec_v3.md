# Custom engine spec v3: audit fixes

Written 2026-11-17 from `docs/audit_2026-11-17.md`. Contract for the work packages WP-A..WP-I there; read the audit
for the findings and file:line evidence behind each item. v1 and v2 stay in force except where this file changes them.
Same rules: stock binary, stock network and stock options unchanged; everything new is off by default or only in the
clock build, and every existing test keeps passing.

## Defaults chosen (the audit's recommendations; the user can override)

| Decision | Choice |
|---|---|
| D1 starting network | Undecided until measured: build the upstream v3 arm and the M0 three-arm measurement (WP-A, WP-G). Code stays on widened it04 meanwhile. |
| D2 headline metric | Equal wall-clock on identical hardware, modelled as nodes-as-time with a per-arm `ms_per_node` measured on the gate GPU. One equal-nodes diagnostic row per iteration. Real wall-clock only for the final match. Gates at 120 s and 180 s; self-play base time drawn from {60, 120, 180, 300} s with weights 0.25/0.25/0.35/0.15, increment 0. |
| D3 opponents | upstream v3 twin-s-clock latest (target); `it00` (widened net, untrained) baseline row; stock it04 at equal clock; Fairy-Stockfish floor (later); three more `same_*` control seeds. |
| D4 rules (one profile, arena and tree) | double flag on an exact tie = draw; FICS repetition (board only, counter reset by any drop, threefold) = draw in the arena; in the tree repetition is non-terminal under tree clocks and rule50 is off; `min_move_ms` 100 plus an additive lag draw; flags immediate; `max_plies` scaled by base time, aborts scored by the on-turn clocks and kept out of training; stalemate = wait. |
| D5 durative tree | Lands before the first kept training round (WP-E E2). |
| D6 heads in round 1 | `think_head` off; `arrivals` loss on only after its fixes (WP-D); `movesLeft` discount 0 in the clock build. |

## Shared contracts

1. **`pace` suffix.** `position fen <A> | <B> clocks <aw> <ab> <bw> <bb> elapsed <ea> <eb> pace <paw> <pab> <pbw> <pbb>`:
   each seat's expected move time in ms (an EMA of that seat's observed landings, default = this search's own
   budget). WP-C sends it, WP-E parses it (pace without clocks: ignored with an info string), WP-F may read it for
   pace planes later.
2. **Recorder sidecar fields** (WP-C writes into each sample's meta, WP-D persists and converts): `nodes_requested`,
   `nodes_searched`, `sit_ms` (how long the seat then sat, 0 if it moved), `start_offset_ms` (time-odds offset of the
   sample's team), `pace` (the four values sent), `clocks_ms` and `elapsed_ms` exact (float, not quantised).
3. **`GameResult.reason` vocabulary** (WP-B defines; WP-A and WP-D consume): `checkmate`, `flag`, `double_flag`
   (draw), `repetition` (draw), `sitoff_adjudicated`, `max_plies` (abort), `aborted`.
4. **File ownership.** `cluster/iterate.sh` belongs to WP-A only; WP-C and WP-D list the lines they need there in
   their reports and integration applies them. `arena/players/hivemind.py` belongs to WP-C only. `arena/sim.py` and
   `arena/state.py` to WP-B only. Engine search/board/uci files to WP-E only; `planes.cc` and `board2planes.py` to
   WP-F only. Trainer, converter and loaders to WP-D only.
5. **Equal-clock gate.** Contender unjittered in the gate, each arm charged `nodes_searched x its own ms_per_node`,
   the opponent under its own time manager; the it00 row runs with exactly the contender's settings.
