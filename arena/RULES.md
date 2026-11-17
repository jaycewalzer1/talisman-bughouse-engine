# Arena rules profile (FICS-style bughouse)

Implemented in `arena/state.py` (`BughouseState`) and, for the timing rules, `arena/sim.py` (`Simulator`).
Tests: `arena/tests/test_state.py`, `arena/tests/test_sim.py`. This is the spec v3 D4 profile
(`docs/custom_engine_spec_v3.md`); `third_party/hivemind/engine/src/tools/async_selfplay.cc` mirrors it and
`tools/async_parity.py` checks the two agree.

## Boards and pieces
- Two boards, A and B, each a python-chess `CrazyhouseBoard` (standard chess + drops).
- Teams: T1 = A_white + B_black, T2 = A_black + B_white. Partners play opposite colours on different boards.
- **Transfer:** a captured piece goes to the capturer's partner's pocket on the other board. It keeps its
  colour (A_white takes a black knight, so B_black receives `n`). The capturer's own pocket is never credited.
- A captured promoted piece transfers as a pawn.
- **Drops** use the usual crazyhouse rules: pieces may go only on empty squares, and pawns may not be
  dropped on rank 1 or 8. Dropped pawns on rank 2 or 7 may still make a double step. Dropping with check or mate is allowed.
- Pawns promote to Q, R, B or N.

## Clocks
- Each seat has its own clock (`base_ms`, or a per-seat `start_clocks_ms` for time odds), plus `increment_ms`
  added after each move. Only the side to move on each board loses time. A move is charged up to its landing
  time before it is applied.
- A clock that reaches 0 loses for that seat's team (`reason="flag"`), at once (no claim delay). If both
  boards run out in the same `advance_clocks` interval, the clock that had less time left at the start of the
  interval ran out first and loses, whichever seat it is. If both ran out at the same millisecond the game is a
  **draw** (`reason="double_flag"`). Remaining time is clamped at 0.
- **Minimum move time and lag** (Simulator `min_move_ms`, `lag_max_ms`; `RULES_V3` uses 100 and 150): every
  landing costs at least `min_move_ms` of clock, also an instant or cached move, plus an additive lag drawn
  uniformly from `[0, lag_max_ms]` per move from the game's seed. So a seat with 3 s left makes at most about
  30 more moves; a seat that keeps moving can flag. Both are 0 by default (old behaviour).

## Game end
- **Checkmate on either board ends the game for both boards.** A position is mate when the side to move is
  in check, has no legal move (drops from its current pocket count as moves), **and** no drop could block the
  check if a piece were available:
  - a contact check (adjacent checker), a knight or pawn check, or a double check can't be blocked, so it is mate;
  - a single check from a rook, bishop or queen with at least one empty square between it and the king is
    **not** mate, because the partner may still supply a piece to block with.
- A side that has no legal move but isn't mated has to wait. This covers the drop-blockable check above
  and stalemate. Its clock keeps running, and the game ends only if a piece arrives or the side flags.
  There are no stalemate draws.
- **Repetition (FICS style): the third occurrence of the same board position on one board is a draw**
  (`reason="repetition"`, `board` set). The position is that board only: placement, side to move, castling and
  en passant square; the pocket and the other board are ignored. Any drop on that board resets its count.
  `BughouseState(repetition=False)` turns it off. (Engine trees treat repetition differently under tree clocks;
  spec v3 D4.)
- **Sit-off adjudication** (Simulator `sitoff_ms`; `RULES_V3` uses 5 s): when the seats on turn on both boards
  have both been sitting (waiting, not thinking) for `sitoff_ms` at once, nothing can change until one moves,
  so the game ends as `sitoff_adjudicated`: the lower of the two running clocks loses (it would flag first),
  equal clocks are a draw. A seeded share of games (`sitoff_unadjudicated`, 10%) is never adjudicated and runs
  to the flag, so the data still shows what happens when nobody breaks. Off by default.
- **Ply cap**: `max_plies` (total over both boards) is a safety abort. `max_plies=None` scales it with the base
  time (400 at 60 s and below, 1200 at 180 s). Without `score_aborts` it is a draw (old behaviour); with it the
  game is scored by the on-turn clocks like a sit-off, reason `max_plies`. Such games stay out of training.
- Results are sticky: once the game is over, `legal_moves` returns `[]` and `apply_move` raises `ValueError`.

## End reasons

`GameResult.reason` is one of `checkmate`, `flag`, `double_flag` (draw), `repetition` (draw),
`sitoff_adjudicated`, `max_plies`, `aborted` (draw; error or empty queue). Defined in `arena/interfaces.py`
(`REASONS`).

## Simplifications and notes
- "Blockable" is a geometric test: an empty square between the checker and the king. The state does not
  ask whether the partner can actually win a suitable piece. Hivemind's own rules are more refined
  (`third_party/hivemind/RULES.md`). In the arena, a stuck side simply sits until it gets a piece or flags.
- There are no draws by the 50-move rule, insufficient material or agreement. Draws come only from a double
  flag, repetition, an equal-clock adjudication or abort, or `aborted`.
- FEN keeps python-chess's `~` marker on promoted pieces (e.g. `Q~`). The marker round-trips through
  `BughouseState(fens=...)` and is needed so that a captured promoted piece transfers as a pawn.
  A consumer that can't parse it can strip the `~` (`fen.replace("~", "")`), but loses that information.
