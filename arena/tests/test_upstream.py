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


# ---------------------------------------------------------------------------------------------- wave 2

from arena.players.upstream import (  # noqa: E402
    WAVE2_ENGINE_DIR,
    Wave2TeamController,
    seat_elapsed_ms,
    seat_sitting,
    seat_values,
    wave2_flag_options,
    wave2_team,
)


def test_seat_values_order_is_aw_ab_bw_bb():
    assert seat_values(clocks(1, 2, 3, 4)) == "1 2 3 4"
    assert seat_values({"B_black": 4.6, "A_white": -5, "B_white": 3, "A_black": 2}) == "0 2 3 5"


def test_wave2_flags_default_ablation_and_validation():
    assert wave2_flag_options() == {"DurativeTree": True, "SitGate": False, "FlagLoss": True,
                                    "InstantTransfer": False, "MateRaceByLanding": False, "PassCapK": 8}
    nodur = wave2_flag_options(durative_tree=False)
    assert nodur["DurativeTree"] is False and nodur["SitGate"] is False  # only the named flag moves
    assert wave2_flag_options(sit_gate=True, pass_cap_k=None)["SitGate"] is True
    assert wave2_flag_options(pass_cap_k=3)["PassCapK"] == 3
    for bad in ({"pass_cap_k": 0}, {"pass_cap_k": 65}, {"flag_loss": "yes"}, {"durativ_tree": True}):
        with pytest.raises(ValueError):
            wave2_flag_options(**bad)


def test_seat_elapsed_only_for_the_seats_on_turn():
    obs = Observation(now_ms=5_000, seat="A_white", fens={"A": START, "B": START}, clocks_ms=clocks(1, 1, 1, 1),
                      to_move={"A": "A_black", "B": "B_white"}, legal_moves=[], last_move_ms={"A": 4_200, "B": 1_000})
    assert seat_elapsed_ms(obs) == {"A_white": 0, "A_black": 800, "B_white": 4_000, "B_black": 0}
    no_landings = Observation(now_ms=5_000, seat="A_white", fens={"A": START, "B": START},
                              clocks_ms=clocks(1, 1, 1, 1), to_move={"A": "A_white", "B": "B_white"}, legal_moves=[])
    assert seat_elapsed_ms(no_landings) == {s: 0 for s in ("A_white", "A_black", "B_white", "B_black")}


def _w2_obs(seat, c, fens, to_move, legal, now=0, last=None):
    return Observation(now_ms=now, seat=seat, fens=fens, clocks_ms=c, to_move=to_move, legal_moves=legal,
                       last_move_ms=last)


def test_wave2_controller_sends_seat_clocks_pace_and_elapsed():
    proc = FakeProcess(move_a=None, move_b="g8f6", nodes=100)
    ctl = Wave2TeamController("T1", proc, nodes=100, ms_per_node=2.0)  # budget 200 ms
    p = ctl.player("B_black")
    ctl.tracker.alpha = 1.0  # the EMA is the last sample
    # Readings as in test_move_time_tracker_reads_clocks: start, then after 1.e4 (2 s) and 1.d4 (500 ms).
    ctl.time_options(_w2_obs("B_black", clocks(60_000, 60_000, 60_000, 60_000), {"A": START, "B": START},
                             {"A": "A_white", "B": "B_white"}, []), 200)
    ctl.time_options(_w2_obs("B_black", clocks(58_000, 60_000, 59_500, 60_000),
                             {"A": _fen_after("e2e4"), "B": _fen_after("d2d4")},
                             {"A": "A_black", "B": "B_black"}, []), 200)
    # A_black replies in 1 s and A_white moves again in 400 ms (A_black on turn, 300 ms in); B_black is still on
    # its first move, 1.9 s in.
    fens = {"A": _fen_after("e2e4 e7e5 g1f3"), "B": _fen_after("d2d4")}
    c = clocks(57_600, 58_700, 59_500, 58_100)
    obs = _w2_obs("B_black", c, fens, {"A": "A_black", "B": "B_black"}, ["g8f6", "d7d5"], now=3_900,
                  last={"A": 3_600, "B": 2_000})
    dec = p.decide(obs)
    assert dec.kind == "move" and dec.uci == "g8f6"
    kw = proc.calls[0]
    assert kw["team"] == "T1" and kw["lead"] == round(((57_600 - 59_500) + (58_100 - 58_700)) / 2)
    opts = kw["options"]
    assert opts["SeatClocksMs"] == "57600 58700 59500 58100"
    # Only A_white has a clean sample (400 ms for its second move). A_black and B_black are on turn and B_white has
    # not moved again: no samples. B_black falls back to its team's EMA (T1: A_white's 400); A_black and B_white
    # (T2, no sample at all) to this search's budget, 200 ms.
    assert opts["SeatPaceMs"] == "400 200 200 400"
    assert opts["SeatElapsedMs"] == "0 300 0 1900"
    # A_black (T2, on turn) is 300 ms into a 200 ms pace: under 2x, still thinking. B_black is ours: never marked.
    assert opts["SeatSitting"] == "0 0 0 0"
    assert kw["root_move_ms"] == 200


def test_seat_sitting_marks_opponent_seats_on_turn_well_past_their_pace():
    obs = Observation(now_ms=5_000, seat="A_white", fens={"A": START, "B": START}, clocks_ms=clocks(1, 1, 1, 1),
                      to_move={"A": "A_white", "B": "B_white"}, legal_moves=[], last_move_ms={"A": 0, "B": 4_000})
    paces = {"A_white": 100, "A_black": 100, "B_white": 400, "B_black": 100}
    # B_white (T2) is 1000 ms in at a 400 ms pace: sitting. A_white is ours (T1): 5 s in, still never marked.
    assert seat_sitting(obs, "T1", paces) == {"A_white": 0, "A_black": 0, "B_white": 1, "B_black": 0}
    # At exactly 2x it is still thinking; from T2's side its own seat is never marked, A_white is.
    assert seat_sitting(obs, "T1", {**paces, "B_white": 500})["B_white"] == 0
    assert seat_sitting(obs, "T2", paces) == {"A_white": 1, "A_black": 0, "B_white": 0, "B_black": 0}


def test_wave2_pace_falls_back_to_budget_and_t2_lead():
    proc = FakeProcess(move_a=None, move_b=None)
    ctl = Wave2TeamController("T2", proc, nodes=50, ms_per_node=2.0, pass_cap_k=3)
    dec = ctl.player("B_white").decide(_obs("B_white", clocks(60_000, 60_000, 62_000, 60_000), ["d2d4"]))
    kw = proc.calls[0]
    assert kw["lead"] == 1_000 and kw["options"]["SeatPaceMs"] == "100 100 100 100"
    assert kw["options"]["SeatElapsedMs"] == "0 0 0 0"  # the observation carries no landing times
    assert kw["options"]["SeatSitting"] == "0 0 0 0"
    # A pass waits PassCapK x this search's budget unless an event wakes it.
    assert dec.kind == "wait" and dec.wait_ms == dec.think_ms + 3 * 100 and dec.stall_reason == "upstream_pass"
    assert ctl.player("B_white").kind == "wave2" and ctl.player("B_white").name.startswith("Wave2-B_white")


def test_wave2_team_sets_flags_at_start_and_runs_in_its_engine_dir():
    players = wave2_team("T1", nodes=64, sit_gate=True, engine_options={"PassCapK": 5}, ms_per_node=0.3)
    ctl = players["A_white"].controller
    assert isinstance(ctl, Wave2TeamController) and ctl.ms_per_node == 0.3 and ctl.pass_cap_k == 8
    assert ctl.process.cwd == WAVE2_ENGINE_DIR
    assert ctl.process.extra_options == {"DurativeTree": True, "SitGate": True, "FlagLoss": True,
                                         "InstantTransfer": False, "MateRaceByLanding": False, "PassCapK": 5}
    assert ctl.process._proc is None  # lazy: nothing started
    for p in players.values():
        p.close()


def test_wave2_kind_in_setups_and_pool_measures_it_with_its_flags(tmp_path):
    from arena.players.setups import KINDS, team_players
    from arena.pool import Pool, apply_ms_per_node, arm_key, for_evaluation, make_schedule, unbuildable

    assert "wave2" in KINDS
    assert team_players("T2", "centralized", "wave2", seed=0, nodes=8)["B_white"].kind == "wave2"
    pool = Pool.load("arena/pools/wave2.json")
    for opp in ("upstream-it13", "wave2-self", "it04"):
        specs = make_schedule(for_evaluation(pool, opponent=opp), 2, openings="bughouse", rules="v3")
        assert unbuildable(specs) == []
    abl = for_evaluation(pool, opponent="wave2-self", contender_kw={"sit_gate": True})
    assert arm_key(abl.contender).startswith("wave2|") and arm_key(abl.contender).endswith("|sit_gate=True")
    assert arm_key(abl.contender) != arm_key(abl.opponents[0])
    calls = []

    def fake(b, m, nodes, repeats, coreml=None, wave2=None):
        calls.append((b, wave2))
        return {"ms_per_node": 0.1 if wave2 is None else 0.2}

    new, used = apply_ms_per_node(for_evaluation(pool, opponent="upstream-it13",
                                                 contender_kw={"ms_per_node": "measure"},
                                                 opponent_kw={"ms_per_node": "measure"}),
                                  str(tmp_path / "ms.json"), measure_upstream=fake)
    assert new.contender.player_kw["ms_per_node"] == 0.2 and new.opponents[0].player_kw["ms_per_node"] == 0.1
    assert sorted(c[1] is None for c in calls) == [False, True]
    assert [c[1] for c in calls if c[1] is not None] == [{}]  # the wave 2 arm, no flag overrides


def test_parity_seat_options_encode_the_reference_lead_and_speeds():
    from tools.upstream_check import seat_time_options

    for team in ("T1", "T2"):
        for lead in (0, 750, -12_000, 70_000):
            o = seat_time_options(team, lead, team_ms=300, opp_ms=1_100)
            c = dict(zip(("A_white", "A_black", "B_white", "B_black"), map(int, o["SeatClocksMs"].split())))
            assert min(c.values()) >= 0 and lead_ms(c, team) == lead
            pace = o["SeatPaceMs"].split()
            assert pace == (["300", "1100", "1100", "300"] if team == "T1" else ["1100", "300", "300", "1100"])
