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


# ------------------------------------------------------------------------------------------- pass_stats

import pass_stats as ps  # noqa: E402


def test_seat_diagonal_is_own_minus_partners_opponent():
    clocks = {"A_white": 40000, "A_black": 30000, "B_white": 25000, "B_black": 35000}
    assert ps.seat_diagonal(clocks, "A_white") == 15000  # vs B_white
    assert ps.seat_diagonal(clocks, "B_black") == 5000  # vs A_black
    assert ps.seat_diagonal(clocks, "A_black") == -5000  # vs B_black


def test_event_stats_counts_sits_and_flag_state(tmp_path):
    fen = "6k1/5ppp/8/8/8/8/5PPP/6K1[] w - - 0 1"
    seats = {"A_white": {"name": "x-free"}, "A_black": {"name": "y"}, "B_white": {"name": "y"},
             "B_black": {"name": "x-free"}}
    clocks = {s: 10000 for s in seats}
    events = [
        {"type": "game_start", "ts": 0, "seats": seats, "fens": {"A": fen, "B": fen}, "clocks": clocks},
        {"type": "move", "ts": 500, "board": "B", "seat": "B_white", "fen": fen.replace(" w ", " b "),
         "clocks": dict(clocks, A_white=9500, B_white=9500)},  # the arena charges every on-turn seat
        {"type": "stall", "ts": 600, "seat": "A_white", "active": True, "reason": "free_pass"},
        {"type": "stall", "ts": 900, "seat": "A_white", "active": False},
        {"type": "stall", "ts": 950, "seat": "A_white", "active": True, "reason": "free_pass"},
        {"type": "game_end", "ts": 10500, "reason": "flag", "loserSeat": "A_white"},
    ]
    path = tmp_path / "game_0001.jsonl"
    path.write_text("\n".join(__import__("json").dumps(e) for e in events))
    rep = ps.event_stats([str(path)])
    assert rep["decisions"]["contender"]["n"] == 1  # one sit spell, not two
    assert rep["decisions"]["contender"]["sit_rate"] == 1.0
    assert rep["sit_lengths"]["contender"]["all"]["max_ms"] == 9900
    assert rep["flags"] == {"contender": {"sitting": 1}}
    # A_white at t=600: 10000 - 600 = 9400 vs B_white 9500 (B_black on turn since 500): behind by 100 ms
    assert "behind" in rep["decisions"]["contender"]["by_diagonal"]


@needs_chunk
def test_pass_stats_runs_on_a_recorded_shard():
    d = os.path.dirname(_chunks()[0])
    rep = ps.run(ps.parse_args(["--shards", d]))
    assert rep["samples"] > 0
    allg = rep["pass_all"]["all"]
    assert 0.0 <= allg["share_pass_top"] <= allg["share_pass_visited"] <= 1.0
    assert sum(g["n"] for g in rep["pass_by_diag_sign"].values()) == rep["shard_rows"]
    assert set(rep["value_labels"]["by_reason"]) <= {"checkmate", "flag", "?", "double_flag", "repetition",
                                                     "sitoff_adjudicated", "max_plies", "aborted"}


# ------------------------------------------------------------------------------------------- implied_values, clock_response

import clock_response as cr  # noqa: E402
import implied_values as iv  # noqa: E402


def test_pocket_colours_and_roles():
    assert [iv.pocket_color("T1", p) for p in iv.POCKETS] == ["w", "b", "b", "w"]
    assert [iv.pocket_color("T2", p) for p in iv.POCKETS] == ["b", "w", "w", "b"]
    assert [cr.role_seat("T1", r) for r in iv.POCKETS] == ["A_white", "A_black", "B_black", "B_white"]
    assert [cr.role_seat("T2", r) for r in iv.POCKETS] == ["A_black", "A_white", "B_white", "B_black"]
    assert iv.add_to_pocket("8/8/8/8/8/8/8/K6k[Pq] w - - 0 1", "N", "b") == "8/8/8/8/8/8/8/K6k[Pqn] w - - 0 1"
    assert iv.add_to_pocket("8/8/8/8/8/8/8/K6k w - - 0 1", "Q", "w") == "8/8/8/8/8/8/8/K6k[Q] w - - 0 1"


@needs_chunk
@needs_it04
def test_implied_values_signs_on_a_shard():
    d = os.path.dirname(_chunks()[0])
    rep = iv.run(iv.parse_args(["--shards", d, "--model", IT04, "--positions", "12", "--threads", "2"]))
    assert rep["dv"]["ours_A"]["Q"]["mean_dv"] > 0 > rep["dv"]["theirs_A"]["Q"]["mean_dv"]
    assert rep["dv"]["ours_B"]["Q"]["mean_dv"] > 0 > rep["dv"]["theirs_B"]["Q"]["mean_dv"]


@needs_chunk
@needs_it04
@needs_v2
@pytest.mark.parametrize("model", [IT04, V2])
def test_clock_response_is_zero_without_clock_planes(model):
    """it04 has no clock planes and v2's clock columns are zero-initialised: with TimeAdvantage held, no clock
    change may move the value or the pass prior."""
    d = os.path.dirname(_chunks()[0])
    rep = cr.run(cr.parse_args(["--shards", d, "--model", model, "--positions", "8", "--hold-ta",
                                "--diagonals=-10,10", "--threads", "2"]))
    for row in rep["dv_per_10s"].values():
        for c in row.values():
            assert c["abs_dv_plus10"] < 2e-3 and c["abs_dv_minus10"] < 2e-3
    sweep = rep["pass_vs_diagonal"]
    assert abs(sweep["-10"]["mean_p_pass"] - sweep["10"]["mean_p_pass"]) < 2e-3


def test_net_calibration_scores_value_and_wdl_heads():
    # WDL head order is (loss, draw, win), as hivemind trains it on y_wdl = value + 1
    rows = [{"net_value": 0.8, "outcome": 1, "net_wdl": [0.05, 0.05, 0.9]},
            {"net_value": 0.2, "outcome": -1, "net_wdl": [0.3, 0.1, 0.6]},
            {"net_value": -0.5, "outcome": 0, "net_wdl": [0.2, 0.6, 0.2]}]
    c = ps.net_calibration(rows)
    assert c["samples"] == 3
    assert c["value_sign_agree"] == 0.5  # draws excluded; one of the two decisive signs right
    assert c["wdl_argmax_acc"] == round(2 / 3, 4)
    assert c["value_mse"] == round((0.2 ** 2 + 1.2 ** 2 + 0.5 ** 2) / 3, 4)
    brier = ((0.1 ** 2 + 0.05 ** 2 + 0.05 ** 2) + (0.6 ** 2 + 0.1 ** 2 + 0.7 ** 2) + (0.2 ** 2 + 0.4 ** 2 + 0.2 ** 2)) / 3
    assert c["wdl_brier"] == round(brier, 4)
    assert c["wdl_score_mse"] == round(((0.85 - 1) ** 2 + (0.3 + 1) ** 2 + 0.0) / 3, 4)
