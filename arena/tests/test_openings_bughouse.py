"""Bughouse opening sets (arena/openings_bughouse.py): registration, legality, pocket transfer, schedule size."""

from __future__ import annotations

import chess
import chess.variant

from arena.openings import OPENING_SETS, paired_schedule
from arena.openings_bughouse import (
    BARE_ID,
    BARE_SHARE,
    BUGHOUSE_SETS,
    CANON,
    CONTROLS,
    bughouse_set,
    clusters,
    pair_fens,
)
from arena.state import BughouseState


def test_sets_registered_and_old_sets_unchanged():
    for name in ("bughouse", "bughouse_canon", "chess_controls", "bare"):
        assert OPENING_SETS[name] is BUGHOUSE_SETS[name]
    assert len(OPENING_SETS["all"]) == 24 and len(OPENING_SETS["eval"]) == 56 and len(OPENING_SETS["train"]) == 240


def test_gate_set_has_100_clusters_and_bare_share():
    s = OPENING_SETS["bughouse"]
    assert clusters(s) >= 100
    bare = sum(p.id == BARE_ID for p in s) / len(s)
    assert abs(bare - BARE_SHARE) < 0.02
    ids = {p.id for p in s}
    assert len(ids & {p.id for p in OPENING_SETS["all"] + OPENING_SETS["train"] + OPENING_SETS["eval"]}) == 0
    # A schedule over one full cycle visits every cluster; the repeats of the bare start share one id.
    sched = list(paired_schedule(2 * len(s), seed=3, pairs=s))
    assert len({p.id for p, _ in sched}) == clusters(s)
    assert sum(swap for _, swap in sched) == len(s)
    assert len(bughouse_set(bare_share=0.0)) == clusters(bughouse_set(bare_share=0.0))


def test_controls_are_two_or_three_chess_lines():
    assert 2 <= len(CONTROLS) <= 3 and len(OPENING_SETS["chess_controls"]) == len(CONTROLS) * (len(CONTROLS) - 1)
    assert len(CANON) >= 10


def _material(fen: str) -> int:
    b = chess.variant.CrazyhouseBoard(fen)
    return chess.popcount(b.occupied) + sum(len(b.pockets[c]) for c in (chess.WHITE, chess.BLACK))


def test_every_pair_is_a_legal_bughouse_start():
    for name in ("bughouse_canon", "chess_controls", "bare"):
        for p in OPENING_SETS[name]:
            boards = [chess.variant.CrazyhouseBoard(f) for f in (p.fen_a, p.fen_b)]
            assert all(b.turn == chess.WHITE and b.is_valid() and not b.is_game_over() for b in boards), p.name
            assert _material(p.fen_a) + _material(p.fen_b) == 64, p.name  # captures move to the other board
            st = BughouseState(60_000, fens=p.fens)
            assert st.legal_moves("A_white") and st.legal_moves("B_white")


def test_captures_feed_the_partner_on_the_other_board():
    # Scandinavian retreat: White takes a pawn on A (-> Black's pocket on B), Black takes one (-> White's on B).
    scand = "e4 d5 exd5 Qxd5 Nc3 Qd8"
    f = pair_fens(scand, "e4 e6 d4 d6")
    b = chess.variant.CrazyhouseBoard(f["B"])
    assert len(b.pockets[chess.WHITE]) == 1 and len(b.pockets[chess.BLACK]) == 1
    assert len(chess.variant.CrazyhouseBoard(f["A"]).pockets[chess.WHITE]) == 0
    # Leaf Gambit: only White captures on A -> one pawn for Black on B.
    f = pair_fens("e4 e6 d4 d6", "e4 d5 exd5 e6")
    a = chess.variant.CrazyhouseBoard(f["A"])
    assert len(a.pockets[chess.BLACK]) == 1 and len(a.pockets[chess.WHITE]) == 0
    assert "[p]" in f["A"]
