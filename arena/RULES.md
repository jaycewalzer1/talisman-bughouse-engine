# Arena rules profile (FICS-style bughouse)

Implemented in `arena/state.py` (`BughouseState`). Tests: `arena/tests/test_state.py`.

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
- Each seat has its own clock (`base_ms`), plus `increment_ms` added after each move. Only the side to
  move on each board loses time. A move is charged up to its landing time before it is applied.
- A clock that reaches 0 loses for that seat's team (`reason="flag"`). If both boards run out in the same
  `advance_clocks` interval, the clock that had less time left at the start of the interval loses.
  Remaining time is clamped at 0.

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
- Results are sticky: once the game is over, `legal_moves` returns `[]` and `apply_move` raises `ValueError`.

## Simplifications and notes
- "Blockable" is a geometric test: an empty square between the checker and the king. The state does not
  ask whether the partner can actually win a suitable piece. Hivemind's own rules are more refined
  (`third_party/hivemind/RULES.md`). In the arena, a stuck side simply sits until it gets a piece or flags.
- There are no draws by repetition, the 50-move rule, insufficient material or agreement.
  (`max_plies`/`aborted` are the simulator's job.)
- FEN keeps python-chess's `~` marker on promoted pieces (e.g. `Q~`). The marker round-trips through
  `BughouseState(fens=...)` and is needed so that a captured promoted piece transfers as a pawn.
  A consumer that can't parse it can strip the `~` (`fen.replace("~", "")`), but loses that information.
