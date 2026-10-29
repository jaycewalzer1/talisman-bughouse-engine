# Arena event stream

The arena (`arena/`) emits one JSON object per event. The same stream feeds:
- replay files: `runs/<name>/game_<n>.jsonl`, one event per line, in `seq` order;
- the live WebSocket server (`python -m arena.serve`), one event per text frame;
- both viewers (2D and 3D) in `viewer/`.

## Conventions

- `ts`: simulated game time in **milliseconds since game start** (integer). Clocks run on this timeline.
- `seq`: integer, strictly increasing from 0 within a game.
- Seats: `A_white`, `A_black`, `B_white`, `B_black`. Boards: `A`, `B`.
- Teams: `T1` = `A_white` + `B_black`; `T2` = `A_black` + `B_white`. Partners sit on different boards
  with opposite colors.
- `clocks`: `{ "A_white": ms, "A_black": ms, "B_white": ms, "B_black": ms }`, remaining time at `ts`.
  Between events, the clock of the side to move on each board is running (derive side to move from that board's FEN).
- FEN: crazyhouse bracket form with the pocket (reserve) in brackets, e.g.
  `rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR[Nn] b KQkq - 0 1`. Uppercase pocket letters are
  White's reserve on that board, lowercase Black's. A `~` after a piece letter marks a promoted piece
  (e.g. `Q~`; it becomes a pawn when captured) — parsers should strip `~`.
- Moves are UCI. Drops are `N@f3` (uppercase piece letter, always). Castling is king-to-target (`e1g1`).
- Piece letters elsewhere (`piece`, `capture`) use FEN case for the color they have **in the receiving pocket**.

## Events

```jsonc
// first event of every game
{ "type": "game_start", "seq": 0, "ts": 0, "game_id": "g0001", "setup": "silent",
  "seats": { "A_white": { "name": "Hivemind-1", "team": "T1", "kind": "hivemind" }, ... },
  "fens": { "A": "<fen>", "B": "<fen>" }, "clocks": { ... }, "base_ms": 60000, "increment_ms": 0,
  "channel_delay_ms": 0 }

// a player starts / finishes thinking. think_ms is the clock time the search will cost (start)
// or did cost (end). nodes/eval/depth optional. eval is from the thinker's perspective, [-1, 1].
{ "type": "thinking", "seq": 7, "ts": 1200, "seat": "A_white", "phase": "start", "think_ms": 800,
  "nodes": 400, "eval": null, "depth": null }

// a move landed on a board
{ "type": "move", "seq": 8, "ts": 2000, "board": "A", "seat": "A_white", "uci": "e2e4", "san": "e4",
  "isDrop": false, "capture": null, "fen": "<fen of this board after the move>", "clocks": { ... } }

// a captured piece goes to the capturer's partner's reserve (always right after the capturing move)
{ "type": "pass", "seq": 9, "ts": 2000, "fromBoard": "A", "fromSeat": "A_white", "toSeat": "B_black",
  "piece": "n" }

// team message. deliverTs = ts + channel delay. tokens below; payload is token-specific
{ "type": "message", "seq": 10, "ts": 2100, "deliverTs": 2100, "fromSeat": "A_white", "toSeat": "B_black",
  "token": "need_N", "text": "need a knight in 5s", "payload": { "piece": "N", "deadlineTs": 7100, "value": 0.4 } }

// a player deliberately waits instead of moving (bughouse "sitting"), or stops waiting
{ "type": "stall", "seq": 11, "ts": 2100, "seat": "A_white", "active": true, "reason": "need_piece" }

// last event of every game
{ "type": "game_end", "seq": 120, "ts": 54000, "result": "T1", "reason": "checkmate",
  "board": "B", "loserSeat": "B_white" }
```

Message tokens: `need_P|N|B|R|Q` (request a piece), `have_P|N|B|R|Q` (offer), `stall` (I am sitting),
`go` (move now, stop waiting), `sit` (please wait), `ok`, `no` (reply to a request).

`game_end.result`: `T1`, `T2`, or `draw`. `reason`: `checkmate`, `flag`, `stalemate`, `max_plies`, `aborted`.
`board`/`loserSeat` are null when not applicable.
