"""upstream-v3 arm (arena/players/upstream.py): lead sign, move-time EMAs, policy parsing, options sent per search.
Engine tests run only when tools/build_upstream.sh has built the binary and fetched the network."""

from __future__ import annotations

import pytest

from arena.interfaces import Observation
from arena.players.heuristic import team_time_edge
from arena.players.hivemind import SearchResult
from arena.players.upstream import (
    MoveTimeTracker,
    UpstreamHivemindProcess,
    UpstreamTeamController,
    lead_ms,
    parse_policy,
    upstream_available,
    upstream_team,
)

START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR[] w KQkq - 0 1"


def clocks(aw, ab, bw, bb):
    return {"A_white": aw, "A_black": ab, "B_white": bw, "B_black": bb}


def test_lead_sign_matches_upstream_time_state():
    # Upstream time_state.h: the white team's (T1 = A_white + B_black) lead is Aw - Bw = Bb - Ab.
    c = clocks(50_000, 45_000, 40_000, 55_000)  # sums 95k on both boards; Aw - Bw = 10k = Bb - Ab
    assert lead_ms(c, "T1") == 10_000
    assert lead_ms(c, "T2") == -10_000
    # Against the arena's own edge: the lead is half of team_time_edge, with the same sign, for every team.
    for c in (clocks(60_000, 58_000, 61_000, 57_000), clocks(1_000, 59_000, 30_000, 30_000), c):
        for team in ("T1", "T2"):
            assert lead_ms(c, team) == round(team_time_edge(c, team) / 2)
    assert lead_ms(clocks(60_000, 60_000, 60_000, 60_000), "T1") == 0


def test_lead_sign_when_partner_sits():
    # T1's B_black sits (its clock runs) while A moves quickly: T1 falls behind on time -> negative lead.
    c = clocks(59_000, 59_000, 60_000, 50_000)
    assert lead_ms(c, "T1") < 0 < lead_ms(c, "T2")


def _fen_after(moves: str) -> str:
    import chess.variant

    b = chess.variant.CrazyhouseBoard()
    for m in moves.split():
        b.push_uci(m)
    return b.fen()


def test_move_time_tracker_reads_clocks():
    tr = MoveTimeTracker(alpha=1.0)  # alpha 1: the EMA is the last sample
    fens = {"A": START, "B": START}
    to_move = {"A": "A_white", "B": "B_white"}
    tr.observe(fens, clocks(60_000, 60_000, 60_000, 60_000), to_move)
    # A_white moved spending 2 s; B_white moved spending 500 ms; now black is on turn on both boards.
    fens = {"A": _fen_after("e2e4"), "B": _fen_after("d2d4")}
    to_move = {"A": "A_black", "B": "B_black"}
    tr.observe(fens, clocks(58_000, 60_000, 59_500, 60_000), to_move)
    # The first observation saw A_white/B_white on turn, so they had no off-turn reading yet: no sample.
    assert tr.samples == {"T1": 0, "T2": 0}
    # Black replies on both boards (A_black 1 s, B_black 3 s), then white moves again (A_white 400 ms, B_white 0.7 s).
    fens = {"A": _fen_after("e2e4 e7e5 g1f3"), "B": _fen_after("d2d4 d7d5 c2c4")}
    to_move = {"A": "A_black", "B": "B_black"}
    tr.observe(fens, clocks(57_600, 59_000, 58_800, 57_000), to_move)
    assert tr.samples == {"T1": 1, "T2": 1}  # A_white (T1) and B_white (T2): one move each since the last reading
    assert tr.move_ms("T1", 9) == 400 and tr.move_ms("T2", 9) == 700
    assert MoveTimeTracker().move_ms("T1", 123) == 123  # no sample: the fallback


def test_parse_policy():
    lines = ["info string livenodes 0", "Value: 0.5", "WDL: 0.6 0.1 0.3", "Predicted plies to end: 70", "",
             "Board A (fen):", "  e2e4: 0.7", "  pass: 0.3", "", "Board B (fen):", "  (not our turn)", "readyok"]
    p = parse_policy(lines)
    assert p.value == 0.5 and p.wdl == (0.6, 0.1, 0.3)
    assert p.board_a == {"e2e4": 0.7, "pass": 0.3} and p.board_b == {}


class FakeProcess:
    def __init__(self, move_a="e2e4", move_b=None, nodes=150):
        self.calls = []
        self.move_a, self.move_b, self.nodes = move_a, move_b, nodes
        self.refs = 1

    def search_clock(self, fen_a, fen_b, **kw):
        self.calls.append(kw)
        return SearchResult(self.move_a, self.move_b, 0.1, 20, None, self.nodes, 3)

    def acquire(self):
        self.refs += 1
        return self

    def release(self):
        self.refs -= 1


def _obs(seat, c, legal, now=0):
    return Observation(now_ms=now, seat=seat, fens={"A": START, "B": START}, clocks_ms=c,
                       to_move={"A": "A_white", "B": "B_white"}, legal_moves=legal)


def test_controller_sends_clock_options_and_charges_searched_nodes():
    proc = FakeProcess(nodes=150)
    ctl = UpstreamTeamController("T1", proc, nodes=100, ms_per_node=2.0)
    dec = ctl.player("A_white").decide(_obs("A_white", clocks(61_000, 60_000, 59_000, 60_000), ["e2e4", "d2d4"]))
    assert dec.kind == "move" and dec.uci == "e2e4"
    assert dec.think_ms == 300  # 150 searched nodes x 2 ms, not the 100 requested
    kw = proc.calls[0]
    assert kw["team"] == "T1" and kw["nodes"] == 100
    assert kw["lead"] == 1_000  # ((61k - 59k) + (60k - 60k)) / 2
    assert kw["root_move_ms"] == 200  # this search's budget
    assert kw["team_move_ms"] == 200 and kw["opp_move_ms"] == 200  # no samples yet: equal speeds at our budget
    req = UpstreamTeamController("T1", FakeProcess(nodes=150), nodes=100, ms_per_node=2.0, charge="requested")
    assert req.player("A_white").decide(_obs("A_white", clocks(1, 1, 1, 1), ["e2e4"])).think_ms == 200


def test_controller_pass_becomes_wait_and_t2_board_b():
    proc = FakeProcess(move_a=None, move_b=None)
    ctl = UpstreamTeamController("T2", proc, nodes=100, ms_per_node=1.0, sit_ms=300)
    dec = ctl.player("B_white").decide(_obs("B_white", clocks(60_000, 60_000, 62_000, 60_000), ["d2d4"]))
    assert dec.kind == "wait" and dec.wait_ms == dec.think_ms + 300 and ctl.passes == 1
    assert proc.calls[0]["lead"] == 1_000  # T2 = -((Aw - Bw) + (Bb - Ab)) / 2 = -((60k - 62k) + 0) / 2
    with pytest.raises(ValueError):
        UpstreamTeamController("T1", proc, clock_k=0)


def test_upstream_process_refuses_fork_dialect():
    with pytest.raises(NotImplementedError):
        UpstreamHivemindProcess().search(START, START, team="T1", nodes=1)


needs_engine = pytest.mark.skipif(not upstream_available(), reason="run sh tools/build_upstream.sh")


@needs_engine
def test_engine_policy_and_search():
    proc = UpstreamHivemindProcess(coreml="off")
    try:
        pol = proc.raw_policy(START, START, team="T1", lead=3000, team_move_ms=100, opp_move_ms=100)
        assert "pass" in pol.board_a and pol.board_b == {}  # T1 is not on turn on B (white to move there)
        assert abs(sum(pol.board_a.values()) - 1.0) < 1e-3
        behind = proc.raw_policy(START, START, team="T1", lead=-3000, team_move_ms=100, opp_move_ms=100)
        assert behind.board_a != pol.board_a  # the clock planes reach the network
        res = proc.search_clock(START, START, team="T1", nodes=64, lead=0, team_move_ms=100, opp_move_ms=100)
        assert res.move_a is not None and res.move_b is None and res.nodes >= 64
    finally:
        proc.close()


@needs_engine
def test_upstream_team_players():
    players = upstream_team("T2", nodes=32, coreml="off")
    assert set(players) == {"A_black", "B_white"}
    dec = players["B_white"].decide(_obs("B_white", clocks(60_000, 60_000, 60_000, 60_000), ["e2e4", "d2d4", "g1f3"]))
    assert dec.kind in ("move", "wait") and dec.nodes >= 32
    for p in players.values():
        p.close()
    assert players["B_white"].controller.process.closed
