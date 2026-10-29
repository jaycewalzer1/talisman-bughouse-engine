import pytest

from arena.interfaces import GameResult
from arena.state import BughouseState


def pocket(fen: str) -> str:
    return fen[fen.index("[") + 1:fen.index("]")]


# --- transfer ---------------------------------------------------------------------------------------

def test_capture_goes_to_partner_with_color_kept():
    s = BughouseState(60000)
    s.apply_move("A_white", "e2e4", 0)
    s.apply_move("A_black", "d7d5", 0)
    r = s.apply_move("A_white", "e4d5", 0)
    assert r.san == "exd5" and r.capture == "p"
    assert pocket(s.fen("A")) == ""  # capturer's own pocket not credited
    assert pocket(s.fen("B")) == "p"  # B_black (A_white's partner) gets a black pawn
    assert "P@e5" not in s.legal_moves("A_black")


def test_black_capture_goes_to_b_white_uppercase():
    s = BughouseState(60000, fens={"A": "4r2k/8/8/8/8/8/8/K3N3[] b - - 0 1"})
    r = s.apply_move("A_black", "e8e1", 0)
    assert r.capture == "N"
    assert pocket(s.fen("A")) == "" and pocket(s.fen("B")) == "N"


def test_promoted_piece_capture_transfers_as_pawn():
    s = BughouseState(60000, fens={"A": "4r2k/8/8/8/8/8/8/K3Q~3[] b - - 0 1"})
    r = s.apply_move("A_black", "e8e1", 0)
    assert r.capture == "P"
    assert pocket(s.fen("B")) == "P" and pocket(s.fen("A")) == ""


def test_promotion_then_capture_round_trip():
    s = BughouseState(60000, fens={"A": "k6r/4P3/8/8/8/8/8/K7[] w - - 0 1"})
    s.apply_move("A_white", "e7e8q", 0)
    assert "Q~" in s.fen("A")
    r = s.apply_move("A_black", "h8e8", 0)
    assert r.capture == "P"  # promoted queen goes to A_black's partner (B_white) as a white pawn
    assert pocket(s.fen("B")) == "P" and pocket(s.fen("A")) == ""


def test_drop_consumes_pocket():
    s = BughouseState(60000, fens={"B": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR[N] w KQkq - 0 1"})
    assert "N@e4" in s.legal_moves("B_white")
    r = s.apply_move("B_white", "N@e4", 0)
    assert r.san == "N@e4" and r.capture is None
    assert pocket(s.fen("B")) == ""
    with pytest.raises(ValueError):
        s.apply_move("B_black", "N@e5", 0)  # black has no knight in pocket


def test_partner_capture_unlocks_drop():
    s = BughouseState(60000)
    for seat, uci in [("B_white", "e2e4"), ("B_black", "d7d5"), ("B_white", "e4d5")]:
        s.apply_move(seat, uci, 0)
    assert pocket(s.fen("A")) == "p"  # B_white's partner is A_black
    s.apply_move("A_white", "e2e4", 0)
    assert "P@e5" in s.legal_moves("A_black")
    s.apply_move("A_black", "P@e5", 0)
    assert pocket(s.fen("A")) == ""


# --- turns / legality -------------------------------------------------------------------------------

def test_turn_validation():
    s = BughouseState(60000)
    assert s.legal_moves("A_black") == []
    assert len(s.legal_moves("A_white")) == 20
    with pytest.raises(ValueError):
        s.apply_move("A_black", "e7e5", 0)
    with pytest.raises(ValueError):
        s.apply_move("A_white", "e2e5", 0)
    with pytest.raises(ValueError):
        s.apply_move("A_white", "garbage", 0)
    assert s.to_move("A") == "A_white" and s.to_move("B") == "B_white"
    s.apply_move("A_white", "e2e4", 0)
    assert s.to_move("A") == "A_black"


def test_castling_uci():
    s = BughouseState(60000, fens={"A": "4k3/8/8/8/8/8/8/4K2R[] w K - 0 1"})
    assert s.apply_move("A_white", "e1g1", 0).san == "O-O"


# --- clocks -----------------------------------------------------------------------------------------

def test_clocks_run_for_side_to_move_only():
    s = BughouseState(1000)
    assert s.advance_clocks(300) is None
    assert s.clocks_ms == {"A_white": 700, "A_black": 1000, "B_white": 700, "B_black": 1000}
    s.apply_move("A_white", "e2e4", 400)  # mover charged up to now first
    assert s.clocks_ms["A_white"] == 600 and s.clocks_ms["B_white"] == 600
    s.advance_clocks(500)
    assert s.clocks_ms["A_black"] == 900 and s.clocks_ms["A_white"] == 600


def test_increment():
    s = BughouseState(1000, increment_ms=200)
    s.apply_move("A_white", "e2e4", 100)
    assert s.clocks_ms["A_white"] == 1100


def test_flag_is_sticky_and_clamped():
    s = BughouseState(1000)
    s.apply_move("A_white", "e2e4", 400)
    r = s.advance_clocks(1500)
    assert r == GameResult(result="T1", reason="flag", board="B", loser_seat="B_white")
    assert s.clocks_ms["B_white"] == 0 and s.clocks_ms["A_black"] == 0
    assert s.result() == r
    assert s.advance_clocks(2000) is None and s.result() == r
    assert s.legal_moves("A_black") == []
    with pytest.raises(ValueError):
        s.apply_move("A_black", "e7e5", 2000)


def test_move_after_flag_time_rejected():
    s = BughouseState(1000)
    with pytest.raises(ValueError):
        s.apply_move("A_white", "e2e4", 1000)
    assert s.result().reason == "flag" and s.result().loser_seat == "A_white"


# --- terminal rules ---------------------------------------------------------------------------------

def test_smothered_knight_mate():
    s = BughouseState(60000, fens={"A": "6rk/6pp/7N/8/8/8/8/K7[] w - - 0 1"})
    s.apply_move("A_white", "h6f7", 0)
    assert s.result() == GameResult(result="T1", reason="checkmate", board="A", loser_seat="A_black")
    assert s.legal_moves("B_white") == []  # game over on both boards


def test_contact_queen_mate():
    s = BughouseState(60000, fens={"B": "7k/8/5K2/6Q1/8/8/8/8[] w - - 0 1"})
    s.apply_move("B_white", "g5g7", 0)
    assert s.result() == GameResult(result="T2", reason="checkmate", board="B", loser_seat="B_black")


def test_rook_check_on_open_file_is_not_mate():
    s = BughouseState(60000, fens={"A": "6rk/6p1/8/8/8/8/R7/K7[] w - - 0 1"})
    s.apply_move("A_white", "a2h2", 0)
    assert s.result() is None
    assert s.legal_moves("A_black") == []  # must wait for a piece; clock keeps running
    s.advance_clocks(1000)
    assert s.clocks_ms["A_black"] == 59000
    # partner B_white captures a black piece -> A_black can now block with a drop
    for seat, uci in [("B_white", "e2e4"), ("B_black", "d7d5"), ("B_white", "e4d5")]:
        s.apply_move(seat, uci, 1000)
    assert "P@h5" in s.legal_moves("A_black")
    s.apply_move("A_black", "P@h5", 1000)
    assert s.result() is None


def test_back_rank_check_with_empty_square_is_not_mate():
    s = BughouseState(60000, fens={"A": "6k1/5ppp/8/8/8/8/8/K3R3[] w - - 0 1"})
    s.apply_move("A_white", "e1e8", 0)
    assert s.result() is None


def test_protected_contact_rook_check_is_mate():
    s = BughouseState(60000, fens={"A": "R6k/6pp/5N2/8/8/8/8/K7[] w - - 0 1"})
    s.apply_move("A_white", "a8g8", 0)
    assert s.result() == GameResult(result="T1", reason="checkmate", board="A", loser_seat="A_black")


def test_double_check_without_king_move_is_mate():
    # Nh4-g6 discovers Rh1 along the open h-file: a drop can block only one of the two checks
    s = BughouseState(60000, fens={"A": "6rk/6p1/8/8/7N/8/8/K6R[qrbnp] w - - 0 1"})
    s.apply_move("A_white", "h4g6", 0)
    assert s.result() == GameResult(result="T1", reason="checkmate", board="A", loser_seat="A_black")


def test_stalemate_is_not_terminal():
    s = BughouseState(60000, fens={"A": "k7/8/1K6/8/8/8/8/2Q5[] w - - 0 1"})
    s.apply_move("A_white", "c1c7", 0)
    assert s.result() is None and s.legal_moves("A_black") == []
    assert s.to_move("A") == "A_black"


# --- fen / copy -------------------------------------------------------------------------------------

def test_fen_round_trip():
    s = BughouseState(60000)
    for seat, uci in [("A_white", "e2e4"), ("A_black", "d7d5"), ("A_white", "e4d5"), ("B_black", "e7e5")]:
        if seat == "B_black":
            s.apply_move("B_white", "g1f3", 0)
        s.apply_move(seat, uci, 0)
    s.apply_move("B_white", "f3e5", 0)
    fens = {"A": s.fen("A"), "B": s.fen("B")}
    assert pocket(fens["A"]) == "p" and pocket(fens["B"]) == "p"
    t = BughouseState(60000, fens=fens)
    assert t.fen("A") == fens["A"] and t.fen("B") == fens["B"]
    assert t.legal_moves("A_black") == s.legal_moves("A_black")
    assert s.fen("A").startswith("rnbqkbnr/ppp1pppp/8/3P4/8/8/PPPP1PPP/RNBQKBNR[p] b KQkq")


def test_copy_is_independent():
    s = BughouseState(60000)
    t = s.copy()
    t.apply_move("A_white", "e2e4", 500)
    assert s.to_move("A") == "A_white" and s.clocks_ms["A_white"] == 60000
    assert t.to_move("A") == "A_black"
