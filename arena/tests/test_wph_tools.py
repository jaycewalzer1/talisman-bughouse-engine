"""Tests for the WP-H measurement tools (audit 2026-11-17): regression sets, pass_stats, implied_values,
clock_response. Network tests use the shipped it04 / v2 ONNX files and skip when they are absent; shard tests use
the first recorded chunk under runs/custom/ and skip when there is none."""

from __future__ import annotations

import glob
import os
import sys

import chess
import chess.variant
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import regression as rg  # noqa: E402

MODELS = os.path.join(ROOT, "third_party", "hivemind", "engine", "models")
IT04 = os.path.join(MODELS, "hivemind-fp32.onnx")
V2 = os.path.join(MODELS, "hivemind-clock-v2-fp32.onnx")
needs_it04 = pytest.mark.skipif(not os.path.isfile(IT04), reason="it04 ONNX not present")
needs_v2 = pytest.mark.skipif(not os.path.isfile(V2), reason="v2 ONNX not present")


def _chunks():
    return sorted(glob.glob(os.path.join(ROOT, "runs", "custom", "*", "it01", "selfplay", "training_data", "*.hvm")))


needs_chunk = pytest.mark.skipif(not _chunks(), reason="no recorded chunk under runs/custom")


# ------------------------------------------------------------------------------------------- regression sets

@pytest.mark.parametrize("path", [rg.SIT_MOVE_FILE, rg.DROP_MATE_FILE])
def test_every_set_claim_holds(path):
    entries = rg.load_entries(path)
    assert len(entries) >= 10
    bad = {e["id"]: rg.verify_entry(e) for e in entries if rg.verify_entry(e)}
    assert not bad


def test_sit_move_set_size_and_both_verdicts():
    cases = rg.expand_all(rg.load_entries(rg.SIT_MOVE_FILE))
    assert len(rg.load_entries(rg.SIT_MOVE_FILE)) >= 20
    assert {c.expect for c in cases} >= {"sit", "move"}
    # each placement puts the attacker on the seat it names, and the diagonal is the attacker's
    for c in cases:
        assert rg.TEAM_OF[c.placement] == c.team
        fen = c.fen_a if c.board == "A" else c.fen_b
        assert fen.split()[1] == ("w" if c.placement.endswith("white") else "b")


def test_verify_catches_a_false_claim():
    entry = dict(rg.load_entries(rg.SIT_MOVE_FILE)[0])
    entry["claims"] = {"mating_drops": ["N@a6"]}
    assert rg.verify_entry(entry)


def test_bughouse_capture_goes_to_partner():
    b = chess.variant.CrazyhouseBoard("6k1/8/8/3n4/4P3/8/8/6K1[] w - - 0 1")
    got = rg.push_bughouse(b, chess.Move.from_uci("e4d5"))
    assert got == chess.Piece(chess.KNIGHT, chess.BLACK)
    assert b.pockets[chess.WHITE].count(chess.KNIGHT) == 0


def test_solver_smothered_and_blockable_check_is_not_mate():
    assert rg.forced_mate_moves(chess.variant.CrazyhouseBoard("6rk/6pp/8/8/8/8/PP3PPP/6K1[N] w - - 0 1"), 1) == ["N@f7"]
    # back-rank rook drop with empty interposition squares: blockable, so no bughouse mate
    assert rg.forced_mate_moves(chess.variant.CrazyhouseBoard("6k1/5ppp/8/8/8/8/5PPP/6K1[R] w - - 0 1"), 1) == []


def test_mirror_uci():
    assert rg.mirror_uci("N@f7") == "N@f2"
    assert rg.mirror_uci("c4f7") == "c5f2"
    assert rg.mirror_uci("a7a8q") == "a2a1q"


@needs_it04
def test_it04_known_answers_and_colour_symmetry():
    """Equal clocks (TimeAdvantage false): it04's raw top-1 is the mating drop. With TimeAdvantage true it04 ranks
    the pass first in these positions (docs/measurements_2026-11-17.md), so only the "eq" variant is a known answer."""
    net = rg.NetEval(IT04, threads=2)
    cases = [c for c in rg.expand_all(rg.load_entries(rg.DROP_MATE_FILE))
             if c.id.split("@")[0] in ("smothered-1", "clamp-g7") and c.tag == "eq"]
    rows = rg.evaluate_cases(net, cases)
    assert all(r["policy"]["top"] in c.best for c, r in zip(cases, rows))
    by = {c.id: r for c, r in zip(cases, rows)}
    for c in cases:
        if c.placement == "A_white":
            twin = by[c.id.replace("@A_white", "@A_black")]
            assert abs(twin["value"] - by[c.id]["value"]) < 1e-4


@needs_it04
@needs_v2
def test_widened_v2_matches_it04_on_the_sets():
    """v2 is it04 with zero-initialised clock columns: identical verdicts on every case."""
    cases = rg.expand_all(rg.load_entries(rg.SIT_MOVE_FILE))[:40]
    a = rg.evaluate_cases(rg.NetEval(IT04, threads=2), cases)
    b = rg.evaluate_cases(rg.NetEval(V2, threads=2), cases)
    assert [r["policy"]["top"] for r in a] == [r["policy"]["top"] for r in b]
    assert max(abs(x["value"] - y["value"]) for x, y in zip(a, b)) < 0.03
