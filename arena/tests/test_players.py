"""Tests for arena.players: random/heuristic legality, determinism, messaging; setups; Hivemind (skipped if
the engine binary or model is missing)."""

from __future__ import annotations

import chess
import chess.variant
import pytest

from arena.interfaces import BOARD_OF, PARTNER, SEATS, TEAM_OF, Message, Observation
from arena.players import (
    MS_PER_NODE,
    HeuristicPlayer,
    HeuristicTeam,
    HivemindPlayer,
    HivemindProcess,
    HivemindTeamController,
    RandomPlayer,
    SearchResult,
    hivemind_available,
    make_players,
    team_time_edge,
)

START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR[] w KQkq - 0 1"


def make_obs(seat: str, fens: dict, *, clocks=None, inbox=(), now_ms=1000) -> Observation:
    """Minimal fake Observation built with python-chess (no arena.state needed)."""
    to_move = {}
    for b in ("A", "B"):
        color = "white" if chess.variant.CrazyhouseBoard(fens[b]).turn else "black"
        to_move[b] = f"{b}_{color}"
    legal = []
    if to_move[BOARD_OF[seat]] == seat:
        legal = [m.uci() for m in chess.variant.CrazyhouseBoard(fens[BOARD_OF[seat]]).legal_moves]
    return Observation(now_ms=now_ms, seat=seat, fens=dict(fens), clocks_ms=clocks or {s: 60000 for s in SEATS},
                       to_move=to_move, legal_moves=legal, inbox=tuple(inbox))


def random_positions(n=25, seed=3):
    import random
    rng = random.Random(seed)
    out = []
    b = chess.variant.CrazyhouseBoard()
    for _ in range(n * 4):
        moves = list(b.legal_moves)
        if not moves or b.is_game_over():
            b = chess.variant.CrazyhouseBoard()
            continue
        b.push(rng.choice(moves))
        if rng.random() < 0.3:  # sprinkle pocket pieces, as partner captures would
            b.pockets[rng.choice([chess.WHITE, chess.BLACK])].add(rng.choice([1, 2, 3, 4, 5]))
        out.append(b.fen())
    return out[:n]


# ---------------------------------------------------------------------------------------------- random

def test_random_player_legal_and_deterministic():
    for fen in random_positions():
        seat = "A_white" if chess.variant.CrazyhouseBoard(fen).turn else "A_black"
        obs = make_obs(seat, {"A": fen, "B": START})
        d1 = RandomPlayer(seat, seed=1).decide(obs)
        d2 = RandomPlayer(seat, seed=1).decide(obs)
        assert d1 == d2
        assert d1.kind == "move" and d1.uci in obs.legal_moves and d1.think_ms == 300


def test_random_player_waits_without_moves():
    obs = make_obs("A_black", {"A": START, "B": START})  # not A_black's turn
    d = RandomPlayer("A_black", seed=0).decide(obs)
    assert d.kind == "wait" and d.wait_ms == 200


# ---------------------------------------------------------------------------------------------- heuristic

@pytest.mark.parametrize("comm", ["none", "request", "share"])
def test_heuristic_legal_moves_only(comm):
    for fen in random_positions():
        seat = "A_white" if chess.variant.CrazyhouseBoard(fen).turn else "A_black"
        obs = make_obs(seat, {"A": fen, "B": START})
        d = HeuristicPlayer(seat, seed=0, comm=comm).decide(obs)
        if d.kind == "move":
            assert d.uci in obs.legal_moves
            assert 50 <= d.think_ms <= 2500
        else:
            assert d.stall_reason


def test_heuristic_takes_mate_in_one():
    fen = "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR[] w KQkq - 4 4"
    d = HeuristicPlayer("A_white", seed=0).decide(make_obs("A_white", {"A": fen, "B": START}))
    assert d.kind == "move" and d.uci == "h5f7" and d.eval == 1.0


def test_heuristic_requests_missing_piece_and_sits():
    # Black king smothered on h8: a knight drop on f7 mates, but white has no knight.
    fen_a = "6rk/6pp/8/8/8/8/8/4K3[] w - - 0 1"
    obs = make_obs("A_white", {"A": fen_a, "B": START})
    d = HeuristicPlayer("A_white", seed=0, comm="request").decide(obs)
    tokens = [m.token for m in d.messages]
    assert "need_N" in tokens
    need = next(m for m in d.messages if m.token == "need_N")
    assert need.to_seat == PARTNER["A_white"] and need.payload["piece"] == "N"
    assert need.payload["deadlineTs"] > obs.now_ms and 0 < need.payload["value"] <= 1
    # team not behind on time -> it sits for the piece and says so
    assert d.kind == "wait" and d.stall_reason == "need_piece" and "stall" in tokens

    # behind on time: still asks, but moves instead of sitting
    clocks = {"A_white": 20000, "A_black": 50000, "B_white": 50000, "B_black": 20000}
    assert team_time_edge(clocks, "T1") < 0
    d2 = HeuristicPlayer("A_white", seed=0, comm="request").decide(make_obs("A_white", {"A": fen_a, "B": START},
                                                                            clocks=clocks))
    assert d2.kind == "move" and any(m.token == "need_N" for m in d2.messages)

    # once the knight is in the pocket, it mates
    d3 = HeuristicPlayer("A_white", seed=0).decide(make_obs("A_white", {"A": fen_a.replace("[]", "[N]"),
                                                                          "B": START}))
    assert d3.kind == "move" and d3.uci == "N@f7"

    # silent players never ask
    d4 = HeuristicPlayer("A_white", seed=0, comm="none").decide(obs)
    assert d4.kind == "move" and not d4.messages


def test_heuristic_partner_request_prefers_capture_and_replies():
    # B_black can take a bishop or a knight with the d6 pawn; partner (A_white) asked for a knight.
    fen_b = "4k3/8/3p4/2B1N3/8/8/8/4K3[] b - - 0 1"
    req = Message(from_seat="A_white", to_seat="B_black", token="need_N", sent_ms=900,
                  payload={"piece": "N", "deadlineTs": 5000, "value": 0.9})
    d = HeuristicPlayer("B_black", seed=0).decide(make_obs("B_black", {"A": START, "B": fen_b}, inbox=(req,)))
    assert d.kind == "move" and d.uci == "d6e5"
    assert "have_N" in [m.token for m in d.messages]
    # silent ignores the request
    d2 = HeuristicPlayer("B_black", seed=0, comm="none").decide(
        make_obs("B_black", {"A": START, "B": fen_b}, inbox=(req,)))
    assert not d2.messages


def test_heuristic_sends_go_when_partner_sits_and_team_behind():
    clocks = {"A_white": 20000, "A_black": 50000, "B_white": 50000, "B_black": 20000}
    fen_a = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR[] b KQkq - 0 1"  # A_black (T2) to move
    fen_b = START  # B_white (T2) to move; A_black's partner
    # A_black's team is T2, clocks favour T2 -> no go. Flip: B_white sitting, T2 behind.
    clocks_t2_behind = {s: (20000 if TEAM_OF[s] == "T2" else 50000) for s in SEATS}
    stall = Message(from_seat="B_white", to_seat="A_black", token="stall", sent_ms=900, payload={})
    d = HeuristicPlayer("A_black", seed=0).decide(make_obs("A_black", {"A": fen_a, "B": fen_b},
                                                           clocks=clocks_t2_behind, inbox=(stall,)))
    assert d.kind == "move" and "go" in [m.token for m in d.messages]
    d2 = HeuristicPlayer("A_black", seed=0).decide(make_obs("A_black", {"A": fen_a, "B": fen_b},
                                                            clocks=clocks, inbox=(stall,)))
    assert "go" not in [m.token for m in d2.messages]


def test_heuristic_go_ends_sitting():
    fen_a = "6rk/6pp/8/8/8/8/8/4K3[] w - - 0 1"
    p = HeuristicPlayer("A_white", seed=0)
    assert p.decide(make_obs("A_white", {"A": fen_a, "B": START})).kind == "wait"
    go = Message(from_seat="B_black", to_seat="A_white", token="go", sent_ms=1100)
    d = p.decide(make_obs("A_white", {"A": fen_a, "B": START}, inbox=(go,), now_ms=1200))
    assert d.kind == "move"


def test_heuristic_team_brain_coordinates():
    fen_a = "6rk/6pp/8/8/8/8/8/4K3[] w - - 0 1"  # A_white needs a knight
    fen_b = "4k3/8/3p4/2B1N3/8/8/8/4K3[] b - - 0 1"  # B_black (partner) can capture one
    team = HeuristicTeam("T1", seed=0)
    fens = {"A": fen_a, "B": fen_b}
    da = team.player("A_white").decide(make_obs("A_white", fens))
    assert da.kind == "wait" and da.stall_reason == "need_piece" and not da.messages
    db = team.player("B_black").decide(make_obs("B_black", fens))
    assert db.kind == "move" and db.uci == "d6e5" and not db.messages
    # if the partner cannot get a knight, the brain does not sit
    team2 = HeuristicTeam("T1", seed=0)
    d = team2.player("A_white").decide(make_obs("A_white", {"A": fen_a, "B": START.replace(" w ", " b ")}))
    assert d.kind == "move"


def _play(setup, kind="heuristic", seed=5, opponent_setup=None, max_plies=400):
    sim_mod = pytest.importorskip("arena.sim")
    state_mod = pytest.importorskip("arena.state")
    players = make_players(setup, kind, seed=seed, opponent_setup=opponent_setup)
    sim = sim_mod.Simulator(state_mod.BughouseState(60000), players, game_id="t", setup=setup, max_plies=max_plies)
    res = sim.run()
    return res, sim.events


@pytest.mark.parametrize("setup", ["centralized", "silent", "sharing"])
def test_heuristic_games_deterministic_and_legal(setup):
    r1, e1 = _play(setup)
    r2, e2 = _play(setup)
    assert r1 == r2 and e1 == e2
    assert not [e for e in e1 if e["type"] == "stall" and e.get("reason") in ("invalid_move", "no_moves")
                and e["active"]]
    msgs = [e for e in e1 if e["type"] == "message"]
    if setup == "sharing":
        assert msgs and all(TEAM_OF[m["fromSeat"]] == TEAM_OF[m["toSeat"]] for m in msgs)
    else:
        assert not msgs


def test_random_games_legal():
    _, events = _play("silent", kind="random", max_plies=200)
    assert not [e for e in events if e["type"] == "stall" and e.get("reason") == "invalid_move"]


# ---------------------------------------------------------------------------------------------- setups

def test_make_players_setups():
    for setup in ("centralized", "silent", "sharing"):
        ps = make_players(setup, "heuristic", seed=1)
        assert set(ps) == set(SEATS) and all(p.seat == s for s, p in ps.items())
    ps = make_players("centralized", "heuristic", opponent_setup="silent")
    assert "central" in ps["A_white"].name and ps["A_black"].comm == "none"
    with pytest.raises(ValueError):
        make_players("telepathy")


def test_make_players_hivemind_compute_split():
    ps = make_players("centralized", "hivemind", nodes=400, opponent_setup="silent")
    try:
        assert ps["A_white"].controller is ps["B_black"].controller
        assert ps["A_white"].controller.nodes == 400
        assert ps["A_black"].nodes == 200 and ps["B_white"].nodes == 200
        assert ps["A_black"].process is not ps["B_white"].process
    finally:
        for p in ps.values():
            p.close()  # processes are lazy: nothing was started


# ---------------------------------------------------------------------------------------------- hivemind (stubbed)

class StubProcess:
    def __init__(self, result: SearchResult):
        self.result = result
        self.calls = []

    def search(self, fen_a, fen_b, **kw):
        self.calls.append(kw)
        return self.result

    def acquire(self):
        return self

    def release(self):
        pass


def test_hivemind_player_translates_pass_to_wait():
    fen_b = START.replace(" w ", " b ")  # B_black to move as well
    stub = StubProcess(SearchResult("e2e4", None, 0.1, 20, None, 120, 3))
    p = HivemindPlayer("B_black", stub, nodes=100, ms_per_node=5.0)
    clocks = {s: (70000 if TEAM_OF[s] == "T1" else 50000) for s in SEATS}  # T1 ahead -> may sit
    d = p.decide(make_obs("B_black", {"A": START, "B": fen_b}, clocks=clocks))
    assert d.kind == "wait" and d.think_ms == 500 and d.wait_ms > 500
    assert stub.calls[-1]["require"] is None and stub.calls[-1]["time_advantage"] is True
    # behind on time: must move on its board
    d = p.decide(make_obs("B_black", {"A": START, "B": fen_b}))
    assert stub.calls[-1]["require"] == "B"


def test_hivemind_controller_reuses_joint_search():
    fen_b = START.replace(" w ", " b ")
    stub = StubProcess(SearchResult("e2e4", "e7e5", 0.2, 40, None, 241, 3))
    ctl = HivemindTeamController("T1", stub, nodes=200, ms_per_node=5.0)
    fens = {"A": START, "B": fen_b}
    da = ctl.player("A_white").decide(make_obs("A_white", fens, now_ms=1000))
    db = ctl.player("B_black").decide(make_obs("B_black", fens, now_ms=1000))
    assert (da.uci, db.uci) == ("e2e4", "e7e5")
    assert len(stub.calls) == 1 and da.think_ms == db.think_ms == 1000 and db.nodes == 0
    # asked later at the same position: only the remaining search time is charged
    dc = ctl.player("B_black").decide(make_obs("B_black", fens, now_ms=1600))
    assert len(stub.calls) == 1 and dc.think_ms == 400


# ---------------------------------------------------------------------------------------------- hivemind (real)

needs_engine = pytest.mark.skipif(not hivemind_available(), reason="hivemind binary or model missing")


@needs_engine
def test_hivemind_one_decision_from_startpos():
    proc = HivemindProcess()
    p = HivemindPlayer("A_white", proc, nodes=50)
    try:
        obs = make_obs("A_white", {"A": START, "B": START})
        d = p.decide(obs)
        assert d.kind == "move" and d.uci in obs.legal_moves
        assert d.think_ms == round(50 * MS_PER_NODE) and d.nodes == 50
        assert d.eval is not None and -1.0 <= d.eval <= 1.0
        # raw joint result: the team (white on A, black on B) is not on turn on B at startpos
        res = proc.search(START, START, team="T1", nodes=20)
        assert res.move_a in obs.legal_moves and res.move_b is None and res.nodes >= 20
    finally:
        p.close()
    assert proc.closed
