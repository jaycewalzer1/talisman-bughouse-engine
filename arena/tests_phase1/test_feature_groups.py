"""The optional feature groups (tempo, mate, danger, serve), the hold message, the think setting and the eval
baseline. The engine is faked except in the last test."""

from __future__ import annotations

import chess
import pytest
import torch

import arena.players.scheduler_player as sp
from arena.interfaces import Message, Observation
from arena.players.context import add_to_pocket, capturable, captures_of, checking_drops
from arena.players.heuristic import parse_board
from arena.players.hivemind import HivemindProcess, SearchResult, hivemind_available
from arena.players.scheduler import (
    BASE_THINK_INDEX,
    FEATURE_SIZE,
    MOVE_INDEX,
    PIECES,
    QUIET_INDEX,
    SERVE_OPTIONS,
    Action,
    CommState,
    Features,
    SchedulerPolicy,
    exploration_prob,
    serve_log_prob,
    to_tensor,
)
from arena.train_scheduler import behavior, eval_baseline_loss, features_from_dict, fit_eval_baseline, learn, step_log_prob

START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR[] w KQkq - 0 1"
SMOTHERED = "6rk/6pp/8/8/8/8/8/K7[] w - - 0 1"  # white mates with N@f7 if it had a knight
BLACK_TAKES_KNIGHT = "rnbqkbnr/ppp1pppp/8/3p4/4N3/8/PPPPPPPP/RNBQKB1R[] b KQkq - 0 1"  # d5xe4
BACK_RANK = "6k1/8/8/8/8/8/5PPP/6K1[] w - - 0 1"  # a black rook dropped on the first rank would check
ROOKS_FACE = "r3k3/8/8/8/8/8/8/R3K3[] b - - 0 1"  # each side can take the other's rook
KNIGHT_HANGS = "rnbqkb1r/pppppppp/8/8/4n3/3P4/PPP1PPPP/RNBQKBNR[] w KQkq - 0 1"  # d3xe4 takes a knight
N = PIECES.index("N")
R = PIECES.index("R")


def features(**kw) -> Features:
    base = dict(
        eval=0.0, eval_change=0.0, search_passed=False, wants=(), can_offer=False,
        own_clock_ms=60_000, partner_clock_ms=60_000, own_opponent_clock_ms=60_000,
        partner_opponent_clock_ms=60_000, partner_waiting=False, offered_piece=None, offered_plies=0,
        requested_piece=None, holding=(), sits_in_row=0, nodes_used_fraction=0.0,
    )
    return Features(**{**base, **kw})


def obs(fen_a=START, fen_b=START, legal=("e2e4", "g1f3"), inbox=(), now=1000) -> Observation:
    return Observation(
        now_ms=now, seat="A_white", fens={"A": fen_a, "B": fen_b},
        clocks_ms={"A_white": 60_000, "A_black": 60_000, "B_white": 60_000, "B_black": 60_000},
        to_move={"A": "A_white", "B": "B_black"}, legal_moves=list(legal), inbox=inbox,
    )


def msg(token: str, **payload) -> Message:
    return Message("B_black", "A_white", token, 0, None, payload)


class FakeProcess:
    """Answers each search with ``answer(fen_a, multipv)`` and remembers the calls."""

    def __init__(self, answer=None):
        self.answer = answer or (lambda fen_a, multipv: SearchResult("e2e4", None, 0.1, 10, None, 1, 1))
        self.calls = []

    def search(self, fen_a, fen_b, *, team, nodes, time_advantage=False, require=None, multipv=1):
        self.calls.append({"fen_a": fen_a, "nodes": nodes, "multipv": multipv})
        return self.answer(fen_a, multipv)

    def release(self):
        pass


def player(process, **kw) -> sp.SchedulerPlayer:
    return sp.SchedulerPlayer("A_white", process, SchedulerPolicy(), nodes=100, ms_per_node=10.0, **kw)


def force_move(monkeypatch):
    monkeypatch.setattr(sp, "sample_action", lambda *a: Action(MOVE_INDEX, QUIET_INDEX, BASE_THINK_INDEX,
                                                               torch.tensor(0.0)))


# ------------------------------------------------------------------------------------------ board reads

def test_add_to_pocket_uses_the_colours_case():
    assert add_to_pocket(START, chess.WHITE, "N").split()[0].endswith("[N]")
    assert add_to_pocket(add_to_pocket(START, chess.WHITE, "N"), chess.BLACK, "Q").split()[0].endswith("[Nq]")


def test_checking_drops_only_lists_pieces_with_a_free_checking_square():
    assert checking_drops(parse_board(SMOTHERED), chess.WHITE) == {"N"}
    assert {"R", "Q", "N"} <= checking_drops(parse_board(BACK_RANK), chess.BLACK)
    assert "P" not in checking_drops(parse_board(BACK_RANK), chess.BLACK)  # f2 and h2 are taken


def test_capturable_is_sooner_for_the_side_to_move():
    board = parse_board(ROOKS_FACE)
    assert capturable(board, chess.BLACK) == {"R": 1.0}
    assert capturable(board, chess.WHITE) == {"R": 0.5}
    assert capturable(parse_board(START), chess.WHITE) == {}


def test_captures_of_finds_the_capturing_moves():
    assert captures_of(parse_board(KNIGHT_HANGS), ["e2e3", "d3e4", "N@f3"], "N") == ["d3e4"]
    assert captures_of(parse_board(KNIGHT_HANGS), ["e2e3", "d3e4"], "Q") == []


# ------------------------------------------------------------------------------------------ inputs and policy

def test_optional_inputs_are_zero_until_filled():
    plain = to_tensor(features())
    assert plain.shape == (FEATURE_SIZE,) and FEATURE_SIZE == 61
    assert torch.all(plain[32:] == 0)
    full = to_tensor(features(n_legal=20, in_check=True, top_gap=0.3, gain=(0, 0.9, 0, 0, 0), mate_with=("N",),
                              supply=(0, 1.0, 0, 0, 0), danger=(0, 0, 0, 0.6, 0), hold_piece="R", serve_cost=0.1))
    assert torch.equal(full[:32], plain[:32])
    assert full[32:].abs().sum() > 0 and full.abs().max() <= 3


def test_comm_state_remembers_a_hold_request_until_it_expires():
    c = CommState("A_white")
    c.update(obs(inbox=(msg("hold_R", piece="R"),), now=1000))
    assert c.hold_piece == "R"
    c.update(obs(now=1000 + 15_001))
    assert c.hold_piece is None


def test_think_logit_loosens_only_the_think_head_and_is_saved():
    x = to_tensor(features())
    tight, loose = SchedulerPolicy(), SchedulerPolicy(think_logit=2.0)
    assert torch.softmax(tight(x).think, -1)[BASE_THINK_INDEX] > 0.98
    assert torch.softmax(loose(x).think, -1)[BASE_THINK_INDEX].item() == pytest.approx(0.649, abs=0.01)
    assert torch.equal(tight(x).gate, loose(x).gate)
    reloaded = SchedulerPolicy()
    reloaded.load_state_dict(loose.state_dict())
    assert float(reloaded.think_logit) == 2.0


def test_serve_head_starts_undecided():
    assert torch.softmax(SchedulerPolicy()(to_tensor(features())).serve, -1).tolist() == [0.5, 0.5]


def test_exploration_start_can_be_raised():
    assert exploration_prob(0, 100, True, 0.2) == pytest.approx(0.2)
    assert exploration_prob(50, 100, True, 0.2) == pytest.approx(0.1)
    assert exploration_prob(0, 100, False, 0.2) == 0.0


def test_unknown_feature_group_is_refused():
    with pytest.raises(ValueError):
        player(FakeProcess(), features=("telepathy",))


# ------------------------------------------------------------------------------------------ the player

def test_without_features_no_extra_search_runs(monkeypatch):
    force_move(monkeypatch)
    process = FakeProcess()
    d = player(process).decide(obs(SMOTHERED, BLACK_TAKES_KNIGHT, legal=("a1b1",)))
    assert [c["nodes"] for c in process.calls] == [16, 84] and d.nodes == 100


def test_tempo_reads_the_gap_to_the_second_choice(monkeypatch):
    force_move(monkeypatch)

    def answer(fen_a, multipv):
        multi = [("e2e4", None, 0.1), ("g1f3", None, -0.2)] if multipv > 1 else []
        return SearchResult("e2e4", None, 0.1, 10, None, 1, 1, multi=multi)

    process = FakeProcess(answer)
    p = player(process, features=("tempo",))
    p.decide(obs())
    f = p.steps[0]["features"]
    assert process.calls[0]["multipv"] == 2 and process.calls[1]["multipv"] == 1
    assert (f["n_legal"], f["in_check"]) == (2, False) and f["top_gap"] == pytest.approx(0.3)


def test_mate_probe_sees_the_mate_and_pays_for_itself(monkeypatch):
    force_move(monkeypatch)

    def answer(fen_a, multipv):
        if "[N]" in fen_a:
            return SearchResult("N@f7", None, 1.0, None, 1, 1, 1)
        return SearchResult("a1b1", None, 0.1, 10, None, 1, 1)

    process = FakeProcess(answer)
    p = player(process, features=("mate",))
    d = p.decide(obs(SMOTHERED, BLACK_TAKES_KNIGHT, legal=("a1b1",)))
    f = p.steps[0]["features"]
    assert f["mate_with"] == ("N",) and f["gain"][N] == pytest.approx(0.9) and f["supply"][N] == 1.0
    # quick look 16, one probe 16, and the long search gets what is left of the 100
    assert [c["nodes"] for c in process.calls] == [16, 16, 68] and d.nodes == 100


def test_mate_probe_skips_pieces_that_cannot_arrive(monkeypatch):
    force_move(monkeypatch)
    process = FakeProcess()
    p = player(process, features=("mate",))
    p.decide(obs(SMOTHERED, START, legal=("a1b1",)))  # nobody can capture a knight on the other board
    assert len(process.calls) == 2 and p.steps[0]["features"]["mate_with"] == ()


def test_danger_probe_tells_the_partner_to_hold_the_piece(monkeypatch):
    force_move(monkeypatch)

    def answer(fen_a, multipv):
        return SearchResult("h2h3", None, -0.5 if "[r]" in fen_a else 0.1, 10, None, 1, 1)

    p = player(FakeProcess(answer), features=("danger",))
    d = p.decide(obs(BACK_RANK, ROOKS_FACE, legal=("h2h3",)))
    assert p.steps[0]["features"]["danger"][R] == pytest.approx(0.6)
    assert [m.token for m in d.messages] == ["hold_R"]
    again = p.decide(obs(BACK_RANK, ROOKS_FACE, legal=("h2h3",), now=2000))
    assert again.messages == ()  # not repeated within HOLD_RESEND_MS


def serving(monkeypatch, choice: str, gap: float = 0.1):
    force_move(monkeypatch)
    monkeypatch.setattr(sp, "sample_serve", lambda *a: (SERVE_OPTIONS.index(choice), torch.tensor(0.0)))

    def answer(fen_a, multipv):
        multi = [("e2e3", None, 0.1), ("d3e4", None, 0.1 - gap)] if multipv > 1 else []
        return SearchResult("e2e3", None, 0.1, 10, None, 1, 1, multi=multi)

    p = player(FakeProcess(answer), features=("serve",))
    return p, p.decide(obs(KNIGHT_HANGS, legal=("e2e3", "d3e4"), inbox=(msg("need_N", piece="N"),)))


def test_serve_plays_the_capture_the_partner_asked_for(monkeypatch):
    p, d = serving(monkeypatch, "serve")
    assert d.uci == "d3e4" and [m.token for m in d.messages] == ["have_N"]
    assert p.steps[-1]["type"] == "serve" and p.steps[-1]["features"]["serve_cost"] == pytest.approx(0.1)
    assert p.comm.requested_piece is None


def test_serve_can_be_declined(monkeypatch):
    p, d = serving(monkeypatch, "own_move")
    assert d.uci == "e2e3" and d.messages == () and p.steps[-1]["type"] == "serve"


def test_a_capture_that_costs_too_much_is_never_offered(monkeypatch):
    p, d = serving(monkeypatch, "serve", gap=sp.SERVE_MAX_COST + 0.1)
    assert d.uci == "e2e3" and all(s["type"] != "serve" for s in p.steps)


# ------------------------------------------------------------------------------------------ training

def step(eval_: float, **kw) -> dict:
    f = features(eval=eval_, **kw)
    from dataclasses import asdict
    return {"type": "decide", "features": asdict(f), "gate": MOVE_INDEX, "talk": QUIET_INDEX,
            "think": BASE_THINK_INDEX, "epsilon": 0.0}


def test_recorded_features_survive_json(tmp_path):
    import json
    d = json.loads(json.dumps(step(0.2, gain=(0, 0.9, 0, 0, 0), mate_with=("N",))["features"]))
    assert features_from_dict(d) == features(eval=0.2, gain=(0, 0.9, 0, 0, 0), mate_with=("N",))


def test_eval_baseline_fits_the_expected_result():
    games = [([step(0.5), step(0.5)], 1.0), ([step(-0.5), step(-0.5)], -1.0)]
    a, k = fit_eval_baseline(games)
    assert a == pytest.approx(0.0) and k == pytest.approx(2.0)
    assert fit_eval_baseline([([step(0.3)], 1.0)]) == (1.0, 0.0)


def test_eval_baseline_credits_only_the_surprise():
    policy = SchedulerPolicy()
    expected = eval_baseline_loss(policy, [step(0.5)], 1.0, 0.0, 2.0)  # the engine already expected the win
    surprise = eval_baseline_loss(policy, [step(-0.5)], 1.0, 0.0, 2.0)  # it expected a loss
    assert expected.item() == pytest.approx(0.0, abs=1e-6)
    assert surprise.item() > 0  # minimising it makes the taken action more likely


def test_learn_with_eval_baseline_takes_a_step():
    policy = SchedulerPolicy(think_logit=2.0)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-2)
    before = policy.think.weight.clone()
    games = [([step(0.0, top_gap=1.0), step(0.4)], 1.0), ([step(0.1), step(-0.3)], -1.0)]
    stats = learn(policy, optimizer, games, 0.0, eval_baseline=True)
    assert "baseline_slope" in stats and not torch.equal(before, policy.think.weight)


def test_serve_steps_have_a_log_probability_and_show_in_the_log():
    from dataclasses import asdict
    serve = {"type": "serve", "features": asdict(features(serve_cost=0.1)), "choice": 1}
    policy = SchedulerPolicy()
    assert step_log_prob(policy, serve).item() == pytest.approx(serve_log_prob(policy, features(serve_cost=0.1), 1).item())
    stats = behavior([([step(0.0, mate_with=("N",)), serve], 1.0)])
    assert (stats["serve_offers"], stats["served"], stats["mate_seen"], stats["mate_waited"]) == (1, 1, 1, 0)
    assert stats["think_mean"] == 1.0


# ------------------------------------------------------------------------------------------ real engine

@pytest.mark.skipif(not hivemind_available(), reason="hivemind binary or model missing")
def test_engine_reports_the_mate_once_the_piece_is_in_the_pocket():
    process = HivemindProcess()
    try:
        without = process.search(SMOTHERED, START, team="T1", nodes=16, require="A")
        with_knight = process.search(add_to_pocket(SMOTHERED, chess.WHITE, "N"), START, team="T1", nodes=16,
                                     require="A")
        several = process.search(KNIGHT_HANGS, START, team="T1", nodes=32, require="A", multipv=3)
    finally:
        process.close()
    assert without.mate is None
    assert with_knight.mate == 1 and with_knight.move_a == "N@f7"
    assert len(several.multi) >= 2 and several.multi[0][0] == several.move_a
