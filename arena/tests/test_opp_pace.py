"""Spec v7 4.1-4.2 (WP2): the opponents' pace in the wave 2 controller's sit threshold (``opp_sit_pace``) and the
arena call site of the learned time policy (``think_head``)."""

from __future__ import annotations

import hashlib
import json
import random

import chess
import chess.variant
import pytest

from arena.interfaces import SEATS, Decision, Observation
from arena.players.hivemind import SearchResult
from arena.players.upstream import (
    OPP_PACE_LEVELS,
    OppPaceEstimator,
    UpstreamHivemindProcess,
    Wave2TeamController,
    seat_decisions,
    wave2_team,
)
from arena.sim import RULES_V3, Simulator
from arena.state import BughouseState

START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR[] w KQkq - 0 1"


def clocks(aw, ab, bw, bb):
    return {"A_white": aw, "A_black": ab, "B_white": bw, "B_black": bb}


# ---------------------------------------------------------------------------------------------- own_k: unchanged

def _golden_cases():
    """48 seeded observations around the sit thresholds (both teams, with and without clock_k)."""
    rng = random.Random(20261127)
    out = []
    for _ in range(48):
        now = rng.randint(5_000, 60_000)
        la, lb = max(0, now - rng.randint(0, 5000)), max(0, now - rng.randint(0, 5000))
        to_move = {"A": rng.choice(["A_white", "A_black"]), "B": rng.choice(["B_white", "B_black"])}
        c = {s: rng.randint(500, 120_000) for s in SEATS}
        paces = {s: rng.randint(50, 2000) for s in SEATS}
        team = rng.choice(["T1", "T2"])
        budget = rng.randint(50, 3000)
        ck = rng.choice([None, 80.0, 110.0, 160.0, 400.0])
        obs = Observation(now_ms=now, seat="A_white", fens={"A": START, "B": START}, clocks_ms=c, to_move=to_move,
                          legal_moves=[], last_move_ms={"A": la, "B": lb})
        out.append((obs, team, paces, budget, ck))
    return out


# seat_decisions' output on _golden_cases, captured from the function before spec v7 ([elapsed, sitting] in SEATS
# order).
GOLDEN = json.loads(
    "[[[0,0,2143,0],[0,1,0,0]],[[0,1671,334,0],[0,0,0,0]],[[256,0,0,0],[0,0,1,0]],[[4171,0,0,4315],[0,0,0,0]]"
    ",[[0,4912,333,0],[0,0,0,0]],[[0,4355,0,0],[0,0,0,1]],[[341,0,4834,0],[0,0,0,0]],[[0,468,0,0],[0,0,1,0]],"
    "[[0,2589,1364,0],[0,0,0,0]],[[243,0,0,4034],[0,0,0,0]],[[0,1178,0,0],[0,0,0,1]],[[0,0,0,0],[0,1,1,0]],[["
    "0,0,1209,0],[1,0,0,0]],[[1313,0,42,0],[0,0,0,0]],[[0,0,0,0],[1,0,0,1]],[[4424,0,0,0],[0,0,1,0]],[[0,0,91"
    "6,0],[1,0,0,0]],[[2111,0,0,0],[0,0,1,0]],[[0,2201,1959,0],[0,0,0,0]],[[0,2854,4966,0],[0,0,0,0]],[[201,0"
    ",0,0],[0,0,0,1]],[[0,0,0,0],[1,0,0,1]],[[3466,0,0,1405],[0,0,0,0]],[[0,0,0,0],[1,0,0,1]],[[1038,0,0,0],["
    "0,0,1,0]],[[0,0,0,0],[1,0,0,1]],[[0,4396,1900,0],[0,0,0,0]],[[0,0,0,164],[0,1,0,0]],[[4336,0,74,0],[0,0,"
    "0,0]],[[3802,0,0,4878],[0,0,0,0]],[[0,0,0,0],[0,1,1,0]],[[0,0,0,0],[0,1,1,0]],[[413,0,4019,0],[0,0,0,0]]"
    ",[[0,0,0,455],[1,0,0,0]],[[0,0,2680,0],[1,0,0,0]],[[0,2737,4016,0],[0,0,0,0]],[[0,631,726,0],[0,0,0,0]],"
    "[[0,0,4803,0],[1,0,0,0]],[[0,0,1796,0],[0,1,0,0]],[[0,0,0,0],[1,0,0,1]],[[0,0,390,0],[1,0,0,0]],[[1,0,21"
    "11,0],[0,0,0,0]],[[2462,0,0,4500],[0,0,0,0]],[[648,0,4718,0],[0,0,0,0]],[[0,0,0,4101],[0,1,0,0]],[[0,133"
    ",182,0],[0,0,0,0]],[[4729,0,0,0],[0,0,1,0]],[[0,0,0,1180],[0,1,0,0]]]"
)


def test_own_k_equals_the_old_seat_decisions_on_recorded_observations():
    cases = _golden_cases()
    assert len(cases) == len(GOLDEN)
    assert sum(sum(g[1]) for g in GOLDEN) > 10 and sum(1 for g in GOLDEN if not any(g[1])) > 10  # both outcomes
    for (obs, team, paces, budget, ck), want in zip(cases, GOLDEN):
        for kw in ({}, {"opp_k": None}, {"opp_k": {}}):
            e, s = seat_decisions(obs, team, paces, budget, clock_k=ck, **kw)
            assert [[e[x] for x in SEATS], [s[x] for x in SEATS]] == want


def test_opp_k_replaces_clock_k_for_its_seat_only():
    # T2's B_white has had board B since 1000 with 60 s on its clock and still thinks at 2000 (1000 ms: k 60). With
    # our clock_k 110 its threshold is 1.5 x 60000 / 110 = 818 ms, so the old rule calls it sitting (audit #3).
    obs = Observation(now_ms=2_000, seat="A_white", fens={"A": START, "B": START},
                      clocks_ms=clocks(60_000, 60_000, 59_000, 60_000), to_move={"A": "A_white", "B": "B_white"},
                      legal_moves=[], last_move_ms={"A": 1_900, "B": 1_000})
    paces = {s: 5_000 for s in SEATS}
    assert seat_decisions(obs, "T1", paces, 400, clock_k=110.0)[1]["B_white"] == 1
    e, s = seat_decisions(obs, "T1", paces, 400, clock_k=110.0, opp_k={"B_white": 60.0})
    assert s["B_white"] == 0 and e["B_white"] == 1_000  # 1.5 x 60000 / 60 = 1500 ms
    assert seat_decisions(obs, "T1", paces, 400, clock_k=110.0, opp_k={"A_black": 60.0})[1]["B_white"] == 1
    # Without clock_k (fixed nodes) a seat in opp_k still gets its own pace: 1.5 x 400 = 600 < 1000, but 1500 > 1000.
    assert seat_decisions(obs, "T1", paces, 400)[1]["B_white"] == 1
    assert seat_decisions(obs, "T1", paces, 400, opp_k={"B_white": 60.0})[1]["B_white"] == 0


# ---------------------------------------------------------------------------------------------- a little game

class LegalProcess:
    """A fake wave 2 engine: the middle legal move (python-chess, sorted) on each board where the team is on turn.
    ``gain`` (pa, pb): quick looks (QUICK_LOOK_NODES nodes) print think_gain; ``nodes`` overrides searched nodes."""

    def __init__(self, gain=None, nodes=None, look_nodes=None):
        self.calls, self.refs, self.gain, self.nodes, self.look_nodes = [], 1, gain, nodes, look_nodes

    def search_clock(self, fen_a, fen_b, *, team, nodes, **kw):
        self.calls.append(dict(kw, nodes=nodes))
        moves = []
        for fen, white_team in ((fen_a, "T1"), (fen_b, "T2")):
            b = chess.variant.CrazyhouseBoard(fen)
            on = (b.turn == chess.WHITE) == (team == white_team)
            ms = sorted(m.uci() for m in b.legal_moves) if on else []
            moves.append(ms[len(ms) // 2] if ms else None)
        look = nodes == 16
        searched = (self.look_nodes if look else self.nodes) or nodes
        res = SearchResult(moves[0], moves[1], 0.0, 1, None, searched, 1)
        if look and self.gain is not None:
            res.think_gain = self.gain
        return res

    def acquire(self):
        self.refs += 1
        return self

    def release(self):
        self.refs -= 1


class PacedOpponent:
    """Thinks clock / k (the sim adds the minimum move time and lag); with ``sit_every`` every n-th decision first
    sits ``sit_ms`` (a wait), then moves."""

    kind = "scripted"

    def __init__(self, seat, k, sit_every=0, sit_ms=4000):
        self.seat, self.name, self.k = seat, f"paced-{seat}", k
        self.sit_every, self.sit_ms, self.n, self._sat = sit_every, sit_ms, 0, False

    def decide(self, obs):
        if not obs.legal_moves:
            return Decision(kind="wait", wait_ms=500)
        if self.sit_every and not self._sat and self.n % self.sit_every == self.sit_every - 1:
            self._sat = True
            return Decision(kind="wait", wait_ms=self.sit_ms)
        self._sat = False
        self.n += 1
        ms = sorted(obs.legal_moves)
        return Decision(kind="move", uci=ms[(obs.ply * 7) % len(ms)],
                        think_ms=int(obs.clocks_ms[self.seat] / self.k))

    def close(self):
        pass


def play(ctl_kw=None, opp_k=(250, 60), sit_every=0, proc=None, max_plies=80):
    """One rules-v3 game at 60 s: our T1 is a Wave2TeamController (clock_k 110) on a fake engine; T2's A_black
    thinks at k 250, B_white at k 60 (slower than us: the audit #3 case)."""
    proc = proc or LegalProcess()
    ctl = Wave2TeamController("T1", proc, nodes=100, ms_per_node=1.0, clock_k=110.0, root_min_ms=100,
                              root_lag_ms=75, **(ctl_kw or {}))
    players = {s: ctl.player(s) for s in ("A_white", "B_black")}
    players["A_black"] = PacedOpponent("A_black", opp_k[0], sit_every)
    players["B_white"] = PacedOpponent("B_white", opp_k[1], sit_every)
    Simulator(BughouseState(60_000), players, game_id="g", setup="test", seed=3,
              **{**RULES_V3, "max_plies": max_plies}).run()
    return ctl, proc


def _sat_marks(proc):
    return sum(int(c["options"]["SeatSitting"].split()[1]) + int(c["options"]["SeatSitting"].split()[2])
               for c in proc.calls)


def _digest(proc):
    return hashlib.sha256(json.dumps([c["options"] for c in proc.calls], sort_keys=True).encode()).hexdigest()[:16]


# Searches, options digest and opponent sitting marks of these games, captured before spec v7.
GOLDEN_GAMES = {0: (40, "32af437c33b45cbe", 9), 4: (41, "283fb02a5c524f75", 10)}


@pytest.mark.parametrize("sit_every", [0, 4])
@pytest.mark.parametrize("kw", [{}, {"opp_sit_pace": "own_k"}, {"opp_sit_pace": "own_k", "think_head": False}])
def test_own_k_controller_sends_what_it_sent_before(kw, sit_every):
    ctl, proc = play(kw, sit_every=sit_every)
    assert (len(proc.calls), _digest(proc), _sat_marks(proc)) == GOLDEN_GAMES[sit_every]
    t = ctl.telemetry()
    assert t["opp_sit_pace"] == "own_k" and t["opp_k_estimates"] == {} and not any(t["opp_k_levels"].values())
    assert t["think_looks"] == 0 and ctl.opp_pace.samples == {"A_black": [], "B_white": []}


def test_observed_reads_the_slow_seat_and_stops_calling_it_sitting():
    ctl, proc = play({"opp_sit_pace": "observed"})
    est = ctl.telemetry()["opp_k_estimates"]
    assert 45 < est["B_white"] < 62  # thinks clock / 60, plus 100 ms minimum and lag: reads a little below 60
    assert _sat_marks(proc) < GOLDEN_GAMES[0][2]
    assert ctl.opp_k_levels["seat"] > 0 and set(ctl.opp_k_levels) == set(OPP_PACE_LEVELS)
    a = [c / max(1, t) for c, t, _ in ctl.opp_pace.samples["A_black"]]
    assert a and 150 < sorted(a)[len(a) // 2] < 260  # k 250 with a 100 ms floor at about 240 ms


def test_fixed_uses_opp_clock_k_for_the_opponents_on_turn():
    ctl, proc = play({"opp_sit_pace": "fixed", "opp_clock_k": 60.0})
    assert set(ctl.opp_k_estimates.values()) == {60.0} and not any(ctl.opp_k_levels.values())
    assert _sat_marks(proc) < GOLDEN_GAMES[0][2]


# ---------------------------------------------------------------------------------------------- the estimator

FEN_W = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR[] w KQkq - 0 {n}"
FEN_B = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR[] b KQkq - 0 {n}"


def _feed(est, seat, readings, sit_at=()):
    """``readings``: B_white's clock at successive off-turn readings, one move apart (made 0, 1, 2, ...). During each
    decision d in ``sit_at`` (from made d to d + 1) the seat is seen on turn and called sitting."""
    board = "B"
    for i, clock in enumerate(readings):
        if i - 1 in sit_at:  # on turn during decision i - 1 (i - 1 moves made: white to move at fullmove i + 1)
            fens = {"A": START, board: FEN_W.format(n=i + 1)}
            est.mark_sitting(fens, {"A": "A_white", board: seat}, {seat: 1})
        fens = {"A": START, board: FEN_B.format(n=i + 1)}  # i moves made, black to move
        c = clocks(60_000, 60_000, 60_000, 60_000)
        c[seat] = clock
        est.observe(fens, c, {"A": "A_white", board: "B_black"})


def test_estimator_median_of_untagged_and_sits_skipped():
    est = OppPaceEstimator("T1")
    # Thinks of 600 ms from 60 s (k 100), a 6 s sit at the 3rd decision (tagged: it was seen sitting).
    readings = [60_000, 59_400, 58_800, 52_800, 52_200, 51_600]
    _feed(est, "B_white", readings, sit_at={2})
    s = est.samples["B_white"]
    assert [x[2] for x in s] == [False, False, True, False, False]  # decision 2 (made 2 -> 3) was tagged
    assert s[2] == (58_800, 6_000, True)
    k, level = est.estimate("B_white", None, 110.0)
    assert level == "seat" and k == pytest.approx(statistics_median([60_000 / 600, 59_400 / 600, 52_800 / 600,
                                                                     52_200 / 600]))


def statistics_median(xs):
    xs = sorted(xs)
    n = len(xs)
    return xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2


def test_estimator_window_is_the_last_12_untagged():
    est = OppPaceEstimator("T1")
    readings = [100_000]
    for i in range(20):  # 8 slow thinks (k 10) then 12 fast ones (k ~ 1000)
        readings.append(readings[-1] - (int(readings[-1] / 10) if i < 8 else 100))
    _feed(est, "B_white", readings)
    k, level = est.estimate("B_white")
    assert level == "seat" and k > 300  # only the last 12 count


def test_estimator_fallback_levels_in_order():
    # < 3 untagged but >= 3 samples: the 75th percentile of all of them.
    est = OppPaceEstimator("T1")
    _feed(est, "B_white", [60_000, 59_000, 53_000, 52_000], sit_at={0, 1, 2})
    ratios = sorted(c / t for c, t, _ in est.samples["B_white"])
    assert len(ratios) == 3 and all(x[2] for x in est.samples["B_white"])
    k, level = est.estimate("B_white", 80.0, 110.0)
    assert level == "seat_q75" and k == pytest.approx(ratios[1] + 0.5 * (ratios[2] - ratios[1]))
    # < 3 samples of its own: the team's pooled samples (A_black has none, B_white three).
    assert est.estimate("A_black", 80.0, 110.0) == (k, "team")
    # Nothing at all: opp_clock_k, then our clock_k, then None (fixed nodes: the old budget rule).
    empty = OppPaceEstimator("T1")
    assert empty.estimate("B_white", 80.0, 110.0) == (80.0, "opp_clock_k")
    assert empty.estimate("B_white", None, 110.0) == (110.0, "own_k")
    assert empty.estimate("B_white", None, None) == (None, "own_k")
    # Two samples: still not enough for the seat or the team.
    two = OppPaceEstimator("T1")
    _feed(two, "B_white", [60_000, 59_000, 58_000])
    assert two.estimate("B_white", None, 110.0) == (110.0, "own_k")


def test_estimator_skips_gaps_of_more_than_one_move_and_on_turn_readings():
    est = OppPaceEstimator("T1")
    c = clocks(60_000, 60_000, 60_000, 60_000)
    est.observe({"A": START, "B": FEN_B.format(n=1)}, c, {"A": "A_white", "B": "B_black"})  # made 0
    c2 = dict(c, B_white=59_000)
    est.observe({"A": START, "B": FEN_W.format(n=2)}, c2, {"A": "A_white", "B": "B_white"})  # on turn: no reading
    c3 = dict(c, B_white=57_000)
    est.observe({"A": START, "B": FEN_B.format(n=3)}, c3, {"A": "A_white", "B": "B_black"})  # made 2: two moves
    assert est.samples["B_white"] == []
    c4 = dict(c, B_white=56_500)
    est.observe({"A": START, "B": FEN_B.format(n=4)}, c4, {"A": "A_white", "B": "B_black"})  # made 3
    assert est.samples["B_white"] == [(57_000, 500, False)]


# ---------------------------------------------------------------------------------------------- options

def test_controller_keyword_validation():
    proc = LegalProcess()
    for bad in ({"opp_sit_pace": "mine"}, {"opp_sit_pace": "fixed"}, {"opp_clock_k": 80.0},
                {"opp_sit_pace": "observed", "opp_clock_k": 0}, {"opp_sit_pace": "observed", "opp_clock_k": True},
                {"opp_sit_pace": "fixed", "opp_clock_k": "80"}, {"think_head": "yes"}, {"think_head": 1}):
        with pytest.raises(ValueError):
            Wave2TeamController("T1", proc, **bad)
    ctl = Wave2TeamController("T1", proc, opp_sit_pace="observed", opp_clock_k=80)
    assert ctl.opp_clock_k == 80.0 and ctl.opp_sit_pace == "observed"


def test_wave2_team_passes_the_new_keywords():
    players = wave2_team("T2", nodes=8, opp_sit_pace="fixed", opp_clock_k=95.5, root_min_ms=100)
    ctl = players["A_black"].controller
    assert (ctl.opp_sit_pace, ctl.opp_clock_k, ctl.think_head, ctl.root_min_ms) == ("fixed", 95.5, False, 100)
    default = wave2_team("T1", nodes=8)["A_white"].controller
    assert (default.opp_sit_pace, default.opp_clock_k, default.think_head) == ("own_k", None, False)
    with pytest.raises(ValueError):
        wave2_team("T1", nodes=8, opp_sit_pace="observd")


# ---------------------------------------------------------------------------------------------- think head

def test_upstream_parser_reads_think_gain():
    res = UpstreamHivemindProcess._parse(None, ["info depth 1 score cp 10 nodes 16 pv (e2e4,pass)",
                                                "info string think_gain 0.25 0.75"], "bestmove (e2e4,pass)", 3)
    assert res.think_gain == (0.25, 0.75) and res.nodes == 16


def _first_decision(proc, **kw):
    ctl = Wave2TeamController("T1", proc, nodes=100, ms_per_node=2.0, clock_k=110.0, max_nodes=2000, **kw)
    p = ctl.player("A_white")
    dec = p.decide(Observation(now_ms=0, seat="A_white", fens={"A": START, "B": START},
                               clocks_ms=clocks(60_000, 60_000, 60_000, 60_000),
                               to_move={"A": "A_white", "B": "B_white"}, legal_moves=["e2e4", "d2d4"],
                               last_move_ms={"A": 0, "B": 0}))
    return ctl, dec


def test_think_head_quick_look_then_scaled_budget_both_charged():
    tp = pytest.importorskip("arena.think_policy")
    proc = LegalProcess(gain=(0.6, 0.1), look_nodes=20)
    ctl, dec = _first_decision(proc, think_head=True, new_game_at_start=True)
    clock_budget = round(60_000 / 110 / 2.0)  # 273 nodes
    assert [c["nodes"] for c in proc.calls] == [tp.QUICK_LOOK_NODES, tp.budget(clock_budget, 0.6, 16, 2000)]
    main = proc.calls[1]["nodes"]
    assert main == round(clock_budget * 1.4) - 16
    look_opts, main_opts = ({k: v for k, v in c.items() if k not in ("nodes", "new_game")} for c in proc.calls)
    assert look_opts == main_opts  # same options; ucinewgame only before the look
    assert proc.calls[0].get("new_game") is True and "new_game" not in proc.calls[1]
    assert dec.nodes == 20 + main and dec.think_ms == (20 + main) * 2  # both searches charged as searched
    assert ctl.nodes_requested == 16 + main and ctl.nodes_searched == 20 + main and ctl.searches == 1
    t = ctl.telemetry()
    assert (t["think_looks"], t["think_head_missing"]) == (1, 0)
    assert t["think_gain_sum"] == pytest.approx(0.6) and t["think_scale_sum"] == pytest.approx(1.4)


def test_think_head_missing_line_keeps_the_clock_budget():
    tp = pytest.importorskip("arena.think_policy")
    proc = LegalProcess(gain=None)
    ctl, dec = _first_decision(proc, think_head=True)
    clock_budget = round(60_000 / 110 / 2.0)
    assert [c["nodes"] for c in proc.calls] == [tp.QUICK_LOOK_NODES, clock_budget]
    assert dec.nodes == 16 + clock_budget and dec.think_ms == (16 + clock_budget) * 2
    assert (ctl.think_looks, ctl.think_head_missing, ctl.think_gain_sum) == (1, 1, 0.0)


def test_think_head_off_runs_one_search():
    proc = LegalProcess(gain=(0.9, 0.9))
    ctl, dec = _first_decision(proc)
    assert [c["nodes"] for c in proc.calls] == [273] and ctl.think_looks == 0 and dec.nodes == 273


def test_think_head_in_a_game_with_opp_pace_observed():
    pytest.importorskip("arena.think_policy")
    ctl, proc = play({"opp_sit_pace": "observed", "think_head": True}, proc=LegalProcess(gain=(0.2, 0.2)))
    t = ctl.telemetry()
    assert t["think_looks"] == t["searches"] > 10 and t["think_head_missing"] == 0
    assert t["think_scale_sum"] == pytest.approx(0.8 * t["think_looks"])
    assert len(proc.calls) == 2 * t["searches"]
