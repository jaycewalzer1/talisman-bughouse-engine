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
    PocketLandingTracker,
    Wave2TeamController,
    fen_pockets,
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
    # B_white (T2) is 1000 ms in at a 400 ms pace (budget 2000): past 1.5 x 400, sitting. A_white is ours (T1).
    assert seat_sitting(obs, "T1", paces, 2_000) == {"A_white": 0, "A_black": 0, "B_white": 1, "B_black": 0}
    # At a 700 ms pace it is still within 1.5x, unless this search's budget is shorter: a pace inflated by earlier
    # sits does not hide a sit (game 2 of the local check: 3.5 s in at a 2 s pace, 444 ms budget).
    assert seat_sitting(obs, "T1", {**paces, "B_white": 700}, 2_000)["B_white"] == 0
    assert seat_sitting(obs, "T1", {**paces, "B_white": 2_034}, 444)["B_white"] == 1
    # From T2's side its own seat is never marked; A_white (5 s at 100 ms) is.
    assert seat_sitting(obs, "T2", paces, 2_000) == {"A_white": 1, "A_black": 0, "B_white": 0, "B_black": 0}


def test_seat_sitting_uses_the_opponents_own_budget_not_ours():
    # Review finding: we have 40 s, the opponents 70 s, clock_k 110: our budget is 364 ms, A_black's own about
    # (70 000 + 600) / 110 = 642 ms. 600 ms in, it is still thinking, not sitting.
    obs = Observation(now_ms=10_600, seat="B_black", fens={"A": START, "B": START},
                      clocks_ms=clocks(40_000, 70_000, 70_000, 40_000), to_move={"A": "A_black", "B": "B_black"},
                      legal_moves=[], last_move_ms={"A": 10_000, "B": 10_000})
    paces = {s: 2_000 for s in ("A_white", "A_black", "B_white", "B_black")}
    assert seat_sitting(obs, "T1", paces, 364)["A_black"] == 1  # without clock_k: our budget, as before
    assert seat_sitting(obs, "T1", paces, 364, clock_k=110)["A_black"] == 0
    late = Observation(**{**obs.__dict__, "now_ms": 11_000})
    assert seat_sitting(late, "T1", paces, 364, clock_k=110)["A_black"] == 1  # 1000 ms > 1.5 x 645
    # The other team on turn on both boards: a serial brain's second seat waits for the first: twice the time.
    both = Observation(**{**obs.__dict__, "to_move": {"A": "A_black", "B": "B_white"}, "now_ms": 11_200})
    assert seat_sitting(both, "T1", paces, 364, clock_k=110)["A_black"] == 0


def test_wave2_root_move_time_adds_the_rules_v3_floor_and_lag():
    proc = FakeProcess(move_a="e2e4", move_b=None)
    ctl = Wave2TeamController("T1", proc, nodes=40, ms_per_node=1.0, root_min_ms=100, root_lag_ms=75)
    ctl.player("A_white").decide(_obs("A_white", clocks(60_000, 60_000, 60_000, 60_000), ["e2e4"]))
    assert proc.calls[0]["root_move_ms"] == 100 + 75  # a 40 ms budget still costs the 100 ms floor + mean lag
    proc2 = FakeProcess(move_a="e2e4", move_b=None)
    plain = Wave2TeamController("T1", proc2, nodes=40, ms_per_node=1.0)
    plain.player("A_white").decide(_obs("A_white", clocks(60_000, 60_000, 60_000, 60_000), ["e2e4"]))
    assert proc2.calls[0]["root_move_ms"] == 40
    with pytest.raises(ValueError):
        Wave2TeamController("T1", proc2, root_lag_ms=-1)
    players = wave2_team("T1", nodes=8, root_min_ms=100, root_lag_ms=75)
    assert players["A_white"].controller.root_min_ms == 100 and players["A_white"].controller.root_lag_ms == 75
    for p in players.values():
        p.close()


def test_wave2_pace_falls_back_to_budget_and_t2_lead():
    proc = FakeProcess(move_a=None, move_b=None)
    ctl = Wave2TeamController("T2", proc, nodes=50, ms_per_node=2.0, pass_cap_k=3)
    dec = ctl.player("B_white").decide(_obs("B_white", clocks(60_000, 60_000, 62_000, 60_000), ["d2d4"]))
    kw = proc.calls[0]
    assert kw["lead"] == 1_000 and kw["options"]["SeatPaceMs"] == "100 100 100 100"
    assert kw["options"]["SeatElapsedMs"] == "0 0 0 0"  # the observation carries no landing times
    assert kw["options"]["SeatSitting"] == "0 0 0 0"
    # A pass waits min(PassCapK, 1.5) x this search's budget (+1 ms) unless an event wakes it: a sitting opponent
    # lands nothing, so the pass re-checks before the arena's 5 s sit-off adjudication.
    assert dec.kind == "wait" and dec.wait_ms == dec.think_ms + 151 and dec.stall_reason == "upstream_pass"
    assert ctl.default_sit_ms({"root_move_ms": 1_000}) == 1_501
    assert Wave2TeamController("T2", proc, pass_cap_k=1).default_sit_ms({"root_move_ms": 1_000}) == 1_000
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


def test_parity_reference_wide_extras_follow_the_team_view():
    import math

    import numpy as np

    from tools.upstream_check import _wide_extras, seat_clocks_for

    p74 = np.zeros((74, 8, 8), np.float32)
    p74[12 + 1] = 1 / 16       # block A: our N
    p74[37 + 17 + 4] = 2 / 16  # block B: their Q
    clocks = seat_clocks_for("T1", 30_000)  # Aw = Bb = 90 s, Ab = Bw = 60 s
    a, b = _wide_extras(p74, 0, True, clocks), _wide_extras(p74, 1, True, clocks)
    assert a[0, 0, 0] == np.float32(90 / 300) and a[2, 0, 0] == np.float32(60 / 300)
    assert b[1, 0, 0] == np.float32(math.exp(-9.0)) and b[3, 0, 0] == np.float32(math.exp(-6.0))
    assert a[6 + 1, 0, 0] == 1 and a[6:16, 0, 0].sum() == 1 and b[11 + 4, 0, 0] == 1 and b[6:16, 0, 0].sum() == 1
    t2 = _wide_extras(p74, 0, False, clocks)  # the black team: ours is Ab
    assert t2[0, 0, 0] == np.float32(60 / 300) and t2[2, 0, 0] == np.float32(90 / 300)
    assert not _wide_extras(p74, 0, True, None)[:6].any() and not a[4:6].any() and not a[16:].any()


# ---------------------------------------------------------------------------------------------- wave 2b

def _pocket_fen(pocket):
    return f"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR[{pocket}] w KQkq - 0 1"


def test_fen_pockets_reads_the_bracket_by_colour():
    assert fen_pockets(_pocket_fen("NQpp"))["white"] == {"P": 0, "N": 1, "B": 0, "R": 0, "Q": 1}
    assert fen_pockets(_pocket_fen("NQpp"))["black"]["P"] == 2
    assert fen_pockets(START)["white"] == {p: 0 for p in "PNBRQ"}
    assert fen_pockets("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")["black"]["P"] == 0


def test_pocket_landing_tracker_dates_new_pieces_by_the_other_boards_last_landing():
    t = PocketLandingTracker()
    # The first look only records the pockets: what was there before is not a landing.
    t.observe({"A": _pocket_fen("N"), "B": START}, 1_000, {"A": 0, "B": 0})
    assert t.value(1_000) == ""
    # A knight more for A_white (captured on B, whose last move landed at 1 800) and a pawn for B_white (A's last
    # landing at 1 500). A_black's pocket shrank (a drop): nothing.
    t.observe({"A": _pocket_fen("NN"), "B": _pocket_fen("P")}, 2_000, {"A": 1_500, "B": 1_800})
    assert t.value(2_000) == "Aw:N:200,Bw:P:500"
    # Without landing times the pieces landed now; landings 5 s old or more drop out.
    t.observe({"A": _pocket_fen("NNq"), "B": _pocket_fen("P")}, 3_000)
    assert t.value(3_000) == "Ab:Q:0,Aw:N:1200,Bw:P:1500"
    t.observe({"A": _pocket_fen("NNq"), "B": _pocket_fen("P")}, 6_900)
    assert t.value(6_900) == "Ab:Q:3900"
    # A clock going back is a new game: start over.
    t.observe({"A": _pocket_fen("R"), "B": START}, 100, {"A": 0, "B": 0})
    assert t.value(100) == ""


def test_wave2_controller_sends_seat_arrivals():
    proc = FakeProcess(move_a=None, move_b="g8f6", nodes=100)
    ctl = Wave2TeamController("T1", proc, nodes=100, ms_per_node=2.0)
    p = ctl.player("B_black")
    to_move = {"A": "A_white", "B": "B_black"}
    p.decide(_w2_obs("B_black", clocks(60_000, 60_000, 60_000, 60_000), {"A": START, "B": START}, to_move,
                     ["g8f6"], now=1_000, last={"A": 0, "B": 0}))
    assert proc.calls[0]["options"]["SeatArrivalsMs"] == ""
    # B_black's pocket gained a bishop from a capture on A that landed at 1 700.
    p.decide(_w2_obs("B_black", clocks(59_000, 60_000, 60_000, 59_000), {"A": START, "B": _pocket_fen("b")},
                     to_move, ["g8f6", "d7d5"], now=2_000, last={"A": 1_700, "B": 900}))
    assert proc.calls[-1]["options"]["SeatArrivalsMs"] == "Bb:B:300"



# ---------------------------------------------------------------------------------------------- wave 2c (spec v5)

import stat  # noqa: E402
import sys  # noqa: E402

from arena.players.upstream import (  # noqa: E402
    WAVE2_CONTINUITY_FLAGS,
    WAVE2_FLAGS,
    expand_wave2_presets,
    parse_continuity,
)

WAVE2_BASE = {"DurativeTree": True, "SitGate": False, "FlagLoss": True, "InstantTransfer": False,
              "MateRaceByLanding": False, "PassCapK": 8}


def test_wave2_continuity_flags():
    # Defaults: the six wave 2 flags only, exactly as before wave 2c (rows without a fix send nothing new).
    assert wave2_flag_options() == WAVE2_BASE
    assert wave2_flag_options(team_tree_reuse=False, plan_carry_permille=300) == WAVE2_BASE  # engine defaults
    # Each switch alone maps to its UCI option; the knobs are sent only off their defaults.
    assert wave2_flag_options(team_tree_reuse=True) == {**WAVE2_BASE, "TeamTreeReuse": True}
    assert wave2_flag_options(joint_plans=True) == {**WAVE2_BASE, "JointPlans": True}
    assert wave2_flag_options(plan_carry=True) == {**WAVE2_BASE, "PlanCarry": True}
    knobs = wave2_flag_options(team_tree_reuse_tolerance_ms=1000, joint_plan_free_permille=0, plan_carry_permille=1000)
    assert knobs == {**WAVE2_BASE, "TeamTreeReuseToleranceMs": 1000, "JointPlanFreePermille": 0,
                     "PlanCarryPermille": 1000}
    # continuity expands to the three switches (no Continuity option is ever sent); a switch set by the row wins.
    all3 = {**WAVE2_BASE, "TeamTreeReuse": True, "JointPlans": True, "PlanCarry": True}
    assert wave2_flag_options(continuity=True) == all3
    assert wave2_flag_options(continuity=True, joint_plans=False) == {**WAVE2_BASE, "TeamTreeReuse": True,
                                                                      "PlanCarry": True}
    assert wave2_flag_options(continuity=False) == WAVE2_BASE
    assert expand_wave2_presets({"continuity": True, "sit_gate": True}) == {
        "sit_gate": True, "team_tree_reuse": True, "joint_plans": True, "plan_carry": True}
    assert "Continuity" not in wave2_flag_options(continuity=True)
    # Ablation flags and continuity combine (c_nodur-style overrides keep working).
    assert wave2_flag_options(continuity=True, durative_tree=False)["DurativeTree"] is False
    for bad in ({"team_tree_reuse": "yes"}, {"continuity": 1}, {"continuity": "true"},
                {"team_tree_reuse_tolerance_ms": -1}, {"team_tree_reuse_tolerance_ms": 5001},
                {"joint_plan_free_permille": 1001}, {"plan_carry_permille": True}, {"plan_carry_permille": "x"},
                {"team_tree_resue": True}, {"Continuity": True}):
        with pytest.raises(ValueError):
            wave2_flag_options(**bad)
    assert set(WAVE2_CONTINUITY_FLAGS) <= set(WAVE2_FLAGS)


def test_wave2_continuity_in_the_pool_and_its_measurement_key(tmp_path):
    from arena.pool import Pool, apply_ms_per_node, arm_key, for_evaluation, make_schedule, unbuildable

    pool = Pool.load("arena/pools/wave2.json")
    self_ = next(o for o in pool.opponents if o.name == "wave2c-self")
    assert self_.kind == "wave2" and self_.player_kw == {**pool.contender.player_kw, "continuity": True}
    specs = make_schedule(for_evaluation(pool, opponent="wave2c-self", contender_kw={"continuity": True}), 2,
                          openings="bughouse", rules="v3")
    assert unbuildable(specs) == []
    # The measurement key holds the expanded switches: c_all and c_joint measure their own node cost, c_same's two
    # arms share one, and rows without a fix keep the keys (and the cache entries) they had before.
    ref = for_evaluation(pool, opponent="upstream-it13")
    c_all = for_evaluation(pool, opponent="wave2c-self", contender_kw={"continuity": True})
    c_joint = for_evaluation(pool, opponent="upstream-it13", contender_kw={"joint_plans": True})
    assert arm_key(ref.contender).endswith("|")
    assert arm_key(c_all.contender).endswith("|team_tree_reuse=True,joint_plans=True,plan_carry=True")
    assert arm_key(c_all.contender) == arm_key(c_all.opponents[0])
    assert arm_key(c_joint.contender).endswith("|joint_plans=True")
    calls = []

    def fake(b, m, nodes, repeats, coreml=None, wave2=None):
        calls.append(wave2)
        return {"ms_per_node": 0.2}

    apply_ms_per_node(for_evaluation(pool, opponent="upstream-it13",
                                     contender_kw={"ms_per_node": "measure", "continuity": True}),
                      str(tmp_path / "ms.json"), measure_upstream=fake)
    assert {"team_tree_reuse": True, "joint_plans": True, "plan_carry": True} in calls
    assert wave2_flag_options(**calls[calls.index({"team_tree_reuse": True, "joint_plans": True,
                                                   "plan_carry": True})]) == {
        **WAVE2_BASE, "TeamTreeReuse": True, "JointPlans": True, "PlanCarry": True}


def test_wave2_team_continuity_implies_reuse_tree():
    def ctl_of(**kw):
        players = wave2_team("T1", nodes=8, **kw)
        ctl = players["A_white"].controller
        for p in players.values():
            p.close()
        return ctl

    for kw in ({"team_tree_reuse": True}, {"plan_carry": True}, {"continuity": True},
               {"engine_options": {"TeamTreeReuse": "true"}}):
        ctl = ctl_of(**kw)
        assert ctl.process.reset_tree is False and ctl.new_game_at_start, kw
    for kw in ({}, {"joint_plans": True}, {"continuity": True, "team_tree_reuse": False, "plan_carry": False}):
        ctl = ctl_of(**kw)
        assert ctl.process.reset_tree is True and not ctl.new_game_at_start, kw
    assert ctl_of(continuity=True).process.extra_options == {**WAVE2_BASE, "TeamTreeReuse": True,
                                                             "JointPlans": True, "PlanCarry": True}


def test_wave2_controller_new_game_only_at_its_first_search():
    proc = FakeProcess(move_a="e2e4", move_b=None)
    ctl = Wave2TeamController("T1", proc, nodes=40, ms_per_node=1.0, new_game_at_start=True)
    p = ctl.player("A_white")
    p.decide(_obs("A_white", clocks(60_000, 60_000, 60_000, 60_000), ["e2e4"]))
    fens = {"A": _fen_after("e2e4 e7e5"), "B": START}
    p.decide(_w2_obs("A_white", clocks(59_000, 59_000, 60_000, 60_000), fens, {"A": "A_white", "B": "B_white"},
                     ["g1f3"], now=2_000))
    assert len(proc.calls) == 2 and proc.calls[0].get("new_game") is True and "new_game" not in proc.calls[1]
    plain = FakeProcess(move_a="e2e4", move_b=None)
    Wave2TeamController("T1", plain, nodes=40).player("A_white").decide(
        _obs("A_white", clocks(60_000, 60_000, 60_000, 60_000), ["e2e4"]))
    assert "new_game" not in plain.calls[0]


def test_wave2_parses_reuse_line():
    def parse(*lines):
        return parse_continuity(list(lines), SearchResult("e2e4", None, 0.1, 20, None, 100, 3))

    r = parse("info string Team tree reuse: 580 visits recovered, clock diff 12 ms, 3 pruned",
              "info string Plan carry: B e7e5 weight 0.3", "info depth 3 score cp 20 nodes 100")
    assert (r.reused_visits, r.reuse_refusal, r.plan_carried) == (580, None, 1)
    r = parse("info string Team tree reuse: none (clock)", "info string Plan carry: A dropped (board changed)")
    assert (r.reused_visits, r.reuse_refusal, r.plan_carried) == (0, "clock", 0)
    r = parse("info string Plan carry: A d2d4 weight 0.3", "info string Plan carry: B e7e5 weight 0.3")
    assert (r.reused_visits, r.reuse_refusal, r.plan_carried) == (None, None, 2)
    r = parse("info string Tree reuse: 40 visits recovered")  # the exact (old) path is not the team tree
    assert (r.reused_visits, r.plan_carried) == (None, 0)
    # The process parses them into SearchResult.
    proc = UpstreamHivemindProcess()
    r = proc._parse(["info string Team tree reuse: 7 visits recovered, clock diff 0 ms, 0 pruned",
                     "info depth 1 score cp 5 nodes 9 pv (e2e4,pass)"], "bestmove (e2e4,pass)", 3)
    assert r.reused_visits == 7 and r.nodes == 9 and r.move_a == "e2e4"


class ContinuityProcess(FakeProcess):
    def __init__(self, results):
        super().__init__()
        self.results = list(results)

    def search_clock(self, fen_a, fen_b, **kw):
        self.calls.append(kw)
        return self.results.pop(0)


def test_wave2_controller_counts_continuity_and_marks_decisions():
    def res(reused=None, refusal=None, carried=0):
        return SearchResult("e2e4", "e7e5", 0.1, 20, None, 100, 3, reused_visits=reused, reuse_refusal=refusal,
                            plan_carried=carried)

    proc = ContinuityProcess([res(), res(0, "clock"), res(400, carried=1)])
    ctl = Wave2TeamController("T1", proc, nodes=40, ms_per_node=1.0)
    obs = [_w2_obs("A_white", clocks(60_000 - i, 60_000, 60_000, 60_000), {"A": _fen_after(m), "B": START},
                   {"A": "A_white", "B": "B_white"}, ["e2e4"], now=i) for i, m in enumerate(("", "g1f3 g8f6",
                                                                                              "b1c3 b8c6"))]
    obs[0] = _obs("A_white", clocks(60_000, 60_000, 60_000, 60_000), ["e2e4"])
    d = [ctl.player("A_white").decide(o) for o in obs]
    assert [x.reused_visits for x in d] == [0, 0, 400] and [x.plan_carried for x in d] == [0, 0, 1]
    assert ctl.reuse_hits == 1 and ctl.reused_visits == 400 and ctl.reuse_refusals == {"clock": 1}
    assert ctl.plan_carries == 1 and ctl.searches == 3
    # B_black served from the same search (cache): no second count.
    third = obs[2]
    ctl.player("B_black").decide(Observation(**{**third.__dict__, "seat": "B_black", "legal_moves": ["e7e5"]}))
    assert ctl.reuse_hits == 1 and ctl.searches == 3


def test_continuity_telemetry_reaches_the_summary_and_results():
    from arena.interfaces import SEATS
    from arena.pool import format_results, results_by_opponent
    from arena.stats import compute_stats, summarize_game

    def think(seq, seat, nodes, **extra):
        return {"type": "thinking", "seq": seq, "ts": 0, "seat": seat, "phase": "start", "think_ms": 5,
                "nodes": nodes, **extra}

    events = [{"type": "game_start", "seq": 0, "ts": 0, "game_id": "g", "setup": "s", "seats": {s: s for s in SEATS}},
              think(1, "A_white", 50, reused_visits=300, plan_carried=1), think(2, "B_black", 50),
              think(3, "B_black", 0), think(4, "A_black", 50), think(5, "A_white", 50, reused_visits=100),
              {"type": "game_end", "seq": 6, "ts": 5, "result": "T1", "reason": "flag", "board": "A",
               "loserSeat": "A_black"}]
    s = summarize_game(events)
    assert s["continuity"]["T1"] == {"searches": 3, "reuse_hits": 2, "reused_visits": 400, "plan_carries": 1}
    assert s["continuity"]["T2"] == {"searches": 1, "reuse_hits": 0, "reused_visits": 0, "plan_carries": 0}
    m = compute_stats([{**s, "contender_team": "T1", "opening_id": "o"}], n_boot=10)["metrics"]
    assert m["contender_reuse_per_search"]["est"] == pytest.approx(2 / 3)
    assert m["contender_reused_visits_per_hit"]["est"] == 200
    assert m["contender_plan_carries_per_search"]["est"] == pytest.approx(1 / 3)
    assert m["baseline_reuse_per_search"]["est"] == 0
    old = {k: v for k, v in s.items() if k != "continuity"}  # a summary from before wave 2c: n/a (NaN), no error
    est = compute_stats([{**old, "contender_team": "T1", "opening_id": "o"}],
                        n_boot=10)["metrics"]["contender_reuse_per_search"]["est"]
    assert est != est
    res = results_by_opponent([{**s, "contender_team": "T1", "opening_id": "o", "opponent": "x", "group": "g"}],
                              n_boot=10)
    assert res["by_opponent"]["x"]["contender_reuse_per_search"] == pytest.approx(2 / 3)
    assert "c_reuse/search=0.67" in format_results(res)


def test_sim_records_continuity_on_thinking_start():
    from arena.interfaces import SEATS, Decision
    from arena.sim import Simulator
    from arena.state import BughouseState

    class P:
        kind = "stub"

        def __init__(self, seat):
            self.seat, self.name, self.n = seat, seat, 0

        def decide(self, obs):
            self.n += 1
            if not obs.legal_moves:
                return Decision(kind="wait", wait_ms=1000)
            extra = {"reused_visits": 5, "plan_carried": 1} if self.seat == "A_white" and self.n == 1 else {}
            return Decision(kind="move", uci=sorted(obs.legal_moves)[0], think_ms=100, nodes=10, **extra)

        def close(self):
            pass

    sim = Simulator(BughouseState(5_000, 0), {s: P(s) for s in SEATS}, game_id="g", setup="test", max_plies=8)
    sim.run()
    starts = [e for e in sim.events if e["type"] == "thinking" and e["phase"] == "start"]
    marked = [e for e in starts if "reused_visits" in e]
    assert len(marked) == 1 and marked[0]["seat"] == "A_white" and marked[0]["plan_carried"] == 1
    assert all("plan_carried" not in e for e in starts if e is not marked[0])


FAKE_ENGINE = r'''
import os, sys
log = open(os.environ["FAKE_ENGINE_LOG"], "a")
opts = os.environ.get("FAKE_ENGINE_OPTIONS", "").split()
searches = 0
for line in sys.stdin:
    cmd = line.strip()
    log.write(cmd + "\n"); log.flush()
    if cmd == "uci":
        for o in opts:
            print(f"option name {o} type check default false")
        print("uciok", flush=True)
    elif cmd == "isready":
        print("readyok", flush=True)
    elif cmd.startswith("go"):
        searches += 1
        if searches > 1:
            print("info string Team tree reuse: 120 visits recovered, clock diff 3 ms, 0 pruned")
        print("info depth 1 score cp 10 nodes 50 pv (e2e4,pass)")
        print("bestmove (e2e4,pass)", flush=True)
    elif cmd == "quit":
        break
'''


def _fake_engine(tmp_path, monkeypatch, options):
    eng = tmp_path / "fake_engine"
    eng.write_text(f"#!{sys.executable}\n" + FAKE_ENGINE)
    eng.chmod(eng.stat().st_mode | stat.S_IXUSR)
    model = tmp_path / "net.onnx"
    model.write_text("")
    log = tmp_path / "log.txt"
    monkeypatch.setenv("FAKE_ENGINE_LOG", str(log))
    monkeypatch.setenv("FAKE_ENGINE_OPTIONS", " ".join(options))
    return str(eng), str(model), log


def _play_two_searches(binary, model, **kw):
    players = wave2_team("T1", nodes=50, ms_per_node=1.0, binary=binary, model=model, **kw)
    try:
        p = players["A_white"]
        d1 = p.decide(_obs("A_white", clocks(60_000, 60_000, 60_000, 60_000), ["e2e4"]))
        fens = {"A": _fen_after("e2e4 e7e5"), "B": START}
        d2 = p.decide(_w2_obs("A_white", clocks(59_000, 59_000, 60_000, 60_000), fens,
                              {"A": "A_white", "B": "B_white"}, ["e2e4"], now=2_000))
        return p.controller, d1, d2
    finally:
        for q in players.values():
            q.close()


def _commands_before_each_go(log):
    out, cur = [], []
    for c in log.read_text().splitlines():
        if c.startswith("go"):
            out.append(cur)
            cur = []
        else:
            cur.append(c)
    return out


def test_wave2_continuity_ucinewgame_sequence_with_an_engine(tmp_path, monkeypatch):
    all_opts = [u for u, _ in WAVE2_CONTINUITY_FLAGS.values()]
    binary, model, log = _fake_engine(tmp_path, monkeypatch, all_opts)
    # team_tree_reuse: one ucinewgame, at the controller's first search, and the option set once at start-up.
    ctl, d1, d2 = _play_two_searches(binary, model, team_tree_reuse=True)
    before = _commands_before_each_go(log)
    assert len(before) == 2 and before[0].count("ucinewgame") == 1 and "ucinewgame" not in before[1]
    assert "setoption name TeamTreeReuse value true" in before[0]
    assert not any("JointPlans" in c or "PlanCarry" in c for c in before[0])
    assert d1.reused_visits == 0 and d2.reused_visits == 120 and ctl.reuse_hits == 1
    # joint_plans alone: ucinewgame before every search, as in ref_120.
    log.write_text("")
    _play_two_searches(binary, model, joint_plans=True)
    before = _commands_before_each_go(log)
    assert [b.count("ucinewgame") for b in before] == [1, 1]
    assert "setoption name JointPlans value true" in before[0]


def test_wave2_continuity_refuses_a_binary_without_the_options(tmp_path, monkeypatch):
    binary, model, log = _fake_engine(tmp_path, monkeypatch, [])  # a pre-2c engine
    with pytest.raises(RuntimeError, match="TeamTreeReuse"):
        _play_two_searches(binary, model, continuity=True)
    log.write_text("")
    ctl, d1, d2 = _play_two_searches(binary, model)  # rows without a fix still run on it
    assert d1.kind == "move" and not any("TeamTreeReuse" in c for c in log.read_text().splitlines())
