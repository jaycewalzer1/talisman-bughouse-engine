"""Search-free reads of the two boards, for the scheduler's extra inputs.

They answer three cheap questions that decide which expensive engine probes are worth running:

* ``checking_drops``: which piece types could be dropped with check on a king right now.
* ``capturable``: which piece types a side could capture on a board, so they would land in a partner's pocket.
* ``add_to_pocket``: the FEN for "what if this pocket held one more of that piece".

Piece letters are the upper-case ones in ``arena.players.scheduler.PIECES``.
"""

from __future__ import annotations

import chess
import chess.variant

from arena.players.heuristic import LETTER, as_side

_BACK_RANKS = chess.BB_RANK_1 | chess.BB_RANK_8


def add_to_pocket(fen: str, color: bool, letter: str) -> str:
    """The bracket-form FEN with one more ``letter`` in ``color``'s pocket."""
    start, end = fen.index("["), fen.index("]")
    piece = letter.upper() if color == chess.WHITE else letter.lower()
    return fen[:end] + piece + fen[end:] if start < end else fen


def checking_drops(board: chess.variant.CrazyhouseBoard, attacker: bool) -> set[str]:
    """Piece letters ``attacker`` could drop with check on the other king, if it held them."""
    king = board.king(not attacker)
    if king is None:
        return set()
    occupied = board.occupied
    empty = ~occupied & chess.BB_ALL
    diagonal = chess.BB_DIAG_ATTACKS[king][chess.BB_DIAG_MASKS[king] & occupied]
    straight = (chess.BB_RANK_ATTACKS[king][chess.BB_RANK_MASKS[king] & occupied]
                | chess.BB_FILE_ATTACKS[king][chess.BB_FILE_MASKS[king] & occupied])
    # A piece on the king's square attacks exactly the squares that piece type would give check from.
    # For pawns that is the other colour's attack pattern, and pawns cannot be dropped on a back rank.
    squares = {
        "P": chess.BB_PAWN_ATTACKS[not attacker][king] & ~_BACK_RANKS,
        "N": chess.BB_KNIGHT_ATTACKS[king],
        "B": diagonal,
        "R": straight,
        "Q": diagonal | straight,
    }
    return {letter for letter, mask in squares.items() if mask & empty}


def capturable(board: chess.variant.CrazyhouseBoard, color: bool) -> dict[str, float]:
    """{piece letter: how soon} for the pieces ``color`` could capture on ``board``: 1.0 when it is to move and
    the capture is legal now, 0.5 when it would have to wait for the other side to move first. A captured promoted
    piece counts as a pawn, as it does in the pocket."""
    to_move = board.turn == color
    b = board if to_move else as_side(board, color)
    out: dict[str, float] = {}
    for move in b.generate_pseudo_legal_captures():
        if to_move and not b.is_legal(move):
            continue
        if b.is_en_passant(move) or b.promoted & chess.BB_SQUARES[move.to_square]:
            letter = "P"
        else:
            letter = LETTER.get(b.piece_type_at(move.to_square))
        if letter:
            out[letter] = 1.0 if to_move else 0.5
    return out


def captures_of(board: chess.variant.CrazyhouseBoard, legal_moves: list[str], letter: str) -> list[str]:
    """The moves in ``legal_moves`` (UCI) that capture a ``letter`` on ``board``."""
    out = []
    for uci in legal_moves:
        if "@" in uci:
            continue
        try:
            move = chess.Move.from_uci(uci)
        except ValueError:
            continue
        if board.is_en_passant(move):
            taken = "P"
        elif board.piece_at(move.to_square) is None:
            continue
        elif board.promoted & chess.BB_SQUARES[move.to_square]:
            taken = "P"
        else:
            taken = LETTER.get(board.piece_type_at(move.to_square))
        if taken == letter:
            out.append(uci)
    return out
