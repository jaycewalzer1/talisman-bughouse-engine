"""BughouseState: two crazyhouse boards, partner pocket transfer, four clocks, FICS-style terminal rules
(checkmate, flag, double flag on an exact tie = draw, threefold board repetition = draw).

Rules profile: arena/RULES.md. Contract: arena/interfaces.py (GameState).
"""

from __future__ import annotations

import copy

import chess
import chess.variant

from arena.interfaces import BOARD_OF, PARTNER, SEATS, TEAM_OF, GameResult, MoveResult, seat_of

BOARDS = ("A", "B")
OTHER_BOARD = {"A": "B", "B": "A"}
OTHER_TEAM = {"T1": "T2", "T2": "T1"}


def _color_name(color: chess.Color) -> str:
    return "white" if color == chess.WHITE else "black"


class BughouseState:
    """Implements arena.interfaces.GameState.

    fens: optional {"A": fen, "B": fen} starting positions (bracket form, `~` promoted markers allowed).
    start_clocks_ms: optional per-seat starting clocks (e.g. time odds); seats not named start at base_ms.
    repetition: FICS-style repetition draw (RULES.md). On by default; False restores the old no-draw profile."""

    def __init__(self, base_ms: int, increment_ms: int = 0, fens: dict | None = None,
                 start_clocks_ms: dict | None = None, repetition: bool = True):
        fens = fens or {}
        self.boards: dict[str, chess.variant.CrazyhouseBoard] = {
            b: chess.variant.CrazyhouseBoard(fens[b]) if b in fens else chess.variant.CrazyhouseBoard()
            for b in BOARDS
        }
        self.base_ms = base_ms
        self.increment_ms = increment_ms
        self.clocks_ms: dict[str, int] = {s: base_ms for s in SEATS}
        for seat, ms in (start_clocks_ms or {}).items():
            if seat not in self.clocks_ms:
                raise ValueError(f"unknown seat {seat!r} in start_clocks_ms")
            self.clocks_ms[seat] = int(ms)
        self.start_clocks_ms: dict[str, int] = dict(self.clocks_ms)
        self.repetition = repetition
        # Per board: how often each board-only position (placement, side to move, castling, ep; pockets
        # ignored) occurred since the last drop on that board.
        self._seen: dict[str, dict[str, int]] = {b: {repetition_key(self.boards[b]): 1} for b in BOARDS}
        self.last_ms = 0  # simulated time of the last advance_clocks call
        self._result: GameResult | None = None

    # --- queries -----------------------------------------------------------------------------------

    def fen(self, board: str) -> str:
        return self.boards[board].fen()

    def to_move(self, board: str) -> str:
        return seat_of(board, _color_name(self.boards[board].turn))

    def legal_moves(self, seat: str) -> list[str]:
        board = BOARD_OF[seat]
        if self._result is not None or self.to_move(board) != seat:
            return []
        return [m.uci() for m in self.boards[board].legal_moves]

    def result(self) -> GameResult | None:
        return self._result

    def copy(self) -> "BughouseState":
        return copy.deepcopy(self)

    # --- clocks ------------------------------------------------------------------------------------

    def advance_clocks(self, now_ms: int) -> GameResult | None:
        """Charge elapsed time to the side to move on each board. Returns the flag result the first time
        a clock reaches 0. If both boards flag in the same interval, the clock that had less time left at the
        start of the interval ran out first and loses; on an exact tie both flagged at the same ms and the game
        is a draw (reason "double_flag")."""
        if self._result is not None:
            return None
        elapsed = max(0, now_ms - self.last_ms)
        self.last_ms = max(self.last_ms, now_ms)
        flagged: list[tuple[int, str]] = []  # (remaining before charge, seat)
        for board in BOARDS:
            seat = self.to_move(board)
            before = self.clocks_ms[seat]
            self.clocks_ms[seat] = max(0, before - elapsed)
            if self.clocks_ms[seat] == 0:
                flagged.append((before, seat))
        if not flagged:
            return None
        if len(flagged) == 2 and flagged[0][0] == flagged[1][0]:
            self._result = GameResult(result="draw", reason="double_flag")
            return self._result
        _, loser = min(flagged)
        self._result = GameResult(result=OTHER_TEAM[TEAM_OF[loser]], reason="flag",
                                  board=BOARD_OF[loser], loser_seat=loser)
        return self._result

    # --- moves -------------------------------------------------------------------------------------

    def apply_move(self, seat: str, uci: str, now_ms: int) -> MoveResult:
        if self._result is not None:
            raise ValueError("game is over")
        board_name = BOARD_OF[seat]
        if self.to_move(board_name) != seat:
            raise ValueError(f"not {seat}'s turn")
        board = self.boards[board_name]
        try:
            move = chess.Move.from_uci(uci)
        except ValueError:
            raise ValueError(f"bad uci {uci!r}") from None
        if move not in board.legal_moves:
            raise ValueError(f"illegal move {uci} for {seat}")
        if self.advance_clocks(now_ms) is not None:
            raise ValueError(f"{self._result.loser_seat} flagged before the move landed")

        mover = board.turn
        san = board.san(move)
        pocket = board.pockets[mover]
        before = {pt: pocket.count(pt) for pt in chess.PIECE_TYPES}
        board.push(move)
        # python-chess credits the capture to the mover's own pocket: redirect it to the partner.
        capture = None
        for pt in chess.PIECE_TYPES:
            if pocket.count(pt) > before[pt]:
                pocket.remove(pt)
                partner_board = self.boards[OTHER_BOARD[board_name]]
                partner_board.pockets[not mover].add(pt)
                capture = chess.Piece(pt, not mover).symbol()
        self.clocks_ms[seat] += self.increment_ms

        if is_bughouse_mate(board):
            loser = seat_of(board_name, _color_name(board.turn))
            self._result = GameResult(result=TEAM_OF[seat], reason="checkmate", board=board_name, loser_seat=loser)
        else:
            seen = self._seen[board_name]
            if move.drop is not None:
                seen.clear()  # FICS: any drop resets the repetition count on that board
            key = repetition_key(board)
            seen[key] = seen.get(key, 0) + 1
            if self.repetition and seen[key] >= 3:
                self._result = GameResult(result="draw", reason="repetition", board=board_name)
        return MoveResult(san=san, capture=capture, fen=board.fen())


def repetition_key(board: chess.Board) -> str:
    """Board-only position for the repetition rule: placement (with `~` promoted markers), side to move,
    castling and legal en passant square, as in the FEN; the pocket is ignored."""
    parts = board.fen().split(" ")
    return " ".join([parts[0].split("[", 1)[0]] + parts[1:4])


def is_bughouse_mate(board: chess.Board) -> bool:
    """Checkmate under bughouse rules: in check, no legal move, and the check cannot be blocked by a
    hypothetical drop (a single sliding checker with at least one empty square between it and the king)."""
    if not board.is_check() or any(board.legal_moves):
        return False
    checkers = board.checkers()
    if len(checkers) > 1:
        return True  # double check: only a king move helps, and there is none
    checker = checkers.pop()
    king = board.king(board.turn)
    return not chess.between(checker, king)  # empty interposition squares => partner could supply a blocker
