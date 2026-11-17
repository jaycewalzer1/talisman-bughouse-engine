"""Spec v3 WP-C (docs/custom_engine_spec_v3.md, docs/audit_2026-11-17.md section 3): controller time accounting
(searched-node charge, clock floor, pace mixture, pace suffix, TreeMoveMs from pace), exploration options,
playout-cap randomisation, the clock-bucket dedupe with sit->move transitions and sit lengths, the sitter re-look,
the stock no-root count and the normalised think head. All against stub engines."""

from __future__ import annotations

import math
from dataclasses import replace

import pytest

from arena.interfaces import SEATS
from arena.players import HivemindTeamController, SearchResult
from arena.players.hivemind import (
    NOISE_ALL_GAME_OPTION,
    TEMPERATURE_OPTION,
    HivemindPlayer,
    fen_plies,
    position_command,
)
from arena.tests.test_custom_engine import CLOCKS, FEN_B_BLACK, fake_sample
from arena.tests.test_players import START, StubProcess, make_obs

FEN_A_BLACK = START.replace(" w ", " b ")


def obs_at(seat, fens, *, now_ms=1000, clocks=None, last=None, ply=0):
    return replace(make_obs(seat, fens, clocks=clocks or CLOCKS, now_ms=now_ms), last_move_ms=last, ply=ply)


class ScriptStub(StubProcess):
    """Answers from a list of (move_a, move_b) in turn (the last one repeats), reports ``searched`` nodes (or the
    request when None) and a sample with the right team byte when asked."""

    def __init__(self, script, searched=None):
        super().__init__(None)
        self.script = list(script)
        self.searched = searched

    def search(self, fen_a, fen_b, **kw):
        self.calls.append(kw)
        ma, mb = self.script[min(len(self.calls) - 1, len(self.script) - 1)]
        nodes = self.searched if self.searched is not None else kw["nodes"]
        res = SearchResult(ma, mb, 0.0, 0, None, nodes, 2)
        if kw.get("sample"):
            res.sample = fake_sample(0 if kw["team"] == "T1" else 1)
        return res


def free_ctl(stub, **kw):
    base = dict(ms_per_node=5.0, free=True, k_range=(100, 100), jitter=(1.0, 1.0))
    base.update(kw)
    return HivemindTeamController("T1", stub, **base)


# ---------------------------------------------------------------------------------------------- charging

def test_stock_controller_charges_searched_nodes_when_asked():
    stub = StubProcess(SearchResult("e2e4", None, 0.1, 20, None, 146, 3))
    obs = make_obs("A_white", {"A": START, "B": START})
    old = HivemindTeamController("T1", stub, nodes=100, ms_per_node=5.0)
    d = old.player("A_white").decide(obs)
    assert d.think_ms == 500 and d.nodes == 100  # default unchanged: the requested budget
    new = HivemindTeamController("T1", stub, nodes=100, ms_per_node=5.0, charge="searched")
    d = new.player("A_white").decide(obs)
    assert d.think_ms == 146 * 5 and d.nodes == 146
    assert new.nodes_requested == 100 and new.nodes_searched == 146
    with pytest.raises(ValueError, match="charge"):
        HivemindTeamController("T1", stub, charge="wall")


def test_split_player_charges_searched_nodes_when_asked():
    stub = StubProcess(SearchResult("e2e4", None, 0.1, 20, None, 146, 3))
    p = HivemindPlayer("A_white", stub, nodes=100, ms_per_node=5.0, charge="searched")
    clocks = {s: 50000 for s in SEATS}  # equal clocks: must move
    d = p.decide(make_obs("A_white", {"A": START, "B": START}, clocks=clocks))
    assert d.kind == "move" and d.think_ms == 730 and d.nodes == 146


def test_free_controller_charges_searched_nodes():
    stub = ScriptStub([("e2e4", None)], searched=146)
    ctl = free_ctl(stub, k_range=(590, 590), charge="searched")  # 59 s / 590 = 100 ms = 20 nodes
    d = ctl.player("A_white").decide(obs_at("A_white", {"A": START, "B": FEN_B_BLACK}))
    assert stub.calls[0]["nodes"] == 20 and d.nodes == 146 and d.think_ms == 730


def test_min_think_ms_floors_the_budget_in_clock_time():
    stub = ScriptStub([("e2e4", None)])
    low = {**CLOCKS, "A_white": 3000}  # 3 s / 100 = 30 ms, under the floor
    ctl = free_ctl(stub, min_think_ms=150, min_nodes=1)
    d = ctl.player("A_white").decide(obs_at("A_white", {"A": START, "B": FEN_B_BLACK}, clocks=low))
    assert stub.calls[0]["nodes"] == 30 and d.think_ms == 150  # 150 ms of clock: a seat can flag by moving
    plain = free_ctl(ScriptStub([("e2e4", None)]), min_nodes=1)
    assert plain.player("A_white").decide(obs_at("A_white", {"A": START, "B": FEN_B_BLACK}, clocks=low)).think_ms == 30


# ---------------------------------------------------------------------------------------------- pace

def test_pace_mixture_is_per_seat_with_lag():
    ctl = free_ctl(ScriptStub([("e2e4", None)]), pace_mixture=((1, 0.5), (1, 2.0)), pace_sigma=0.0, lag_ms=0.0,
                   seed=11)
    assert ctl.jitter is None and set(ctl.seat_factor) == {"A_white", "B_black"}
    assert all(f in (0.5, 2.0) for f in ctl.seat_factor.values())
    factors = {free_ctl(ScriptStub([]), pace_mixture=((1, 0.5), (1, 2.0)), seed=s).seat_factor["A_white"]
               for s in range(30)}
    assert factors == {0.5, 2.0}  # both components drawn across games
    lagged = free_ctl(ScriptStub([]), pace_mixture=((1, 1.0),), pace_sigma=0.0, lag_ms=80.0, seed=3)
    draws = [lagged._think_ms(100, *lagged._time_noise("A_white")) - 500 for _ in range(400)]
    assert min(draws) >= 0 and 60 <= sum(draws) / len(draws) <= 100  # additive, mean ~80 ms


def test_position_command_pace_suffix_only_with_clocks():
    pace = {"A_white": 400.4, "A_black": 900, "B_white": 0, "B_black": 1200}
    assert position_command("a", "b", CLOCKS, {"A": 1, "B": 2}, pace).endswith(
        "clocks 59000 58000 57000 56000 elapsed 1 2 pace 400 900 0 1200")
    assert position_command("a", "b", None, None, pace) == "position fen a | b"
    assert position_command("a", "b", CLOCKS, None, pace).endswith("56000 pace 400 900 0 1200")


def test_tree_move_ms_comes_from_pace_not_the_jitter_draw():
    stub = ScriptStub([("e2e4", None)])
    ctl = free_ctl(stub, jitter=(0.5, 2.0), pace=True, seed=4)
    d = ctl.player("A_white").decide(obs_at("A_white", {"A": START, "B": FEN_B_BLACK}, last={"A": 0, "B": 0}))
    call = stub.calls[0]
    assert call["options"]["TreeMoveMs"] == 590 != d.think_ms  # own unjittered budget, not the 0.5-2x charge
    assert call["pace"] == {s: 590.0 for s in SEATS}  # nobody seen yet: this search's own budget


def test_pace_ema_follows_observed_landings():
    stub = ScriptStub([("e2e4", "e7e5")])
    ctl = free_ctl(stub, pace=True, pace_alpha=0.5)
    clocks = {s: 60000 for s in SEATS}
    fens = {"A": START, "B": FEN_B_BLACK}
    ctl.player("A_white").decide(obs_at("A_white", fens, now_ms=0, clocks=clocks, last={"A": 0, "B": 0}))
    # A_white landed at 2000 after 2 s; A_black has been thinking for 500 ms since
    fens2 = {"A": FEN_A_BLACK.replace(" 0 1", " 0 1"), "B": FEN_B_BLACK}
    clocks2 = {**clocks, "A_white": 58000, "A_black": 59500}
    ctl._observe_pace(replace(obs_at("A_black", fens2, now_ms=2500, clocks=clocks2, last={"A": 2000, "B": 0})))
    assert ctl._pace_ema == {"A_white": 2000.0}
    # A_black lands at 3000 (1 s), then A_white again at 7000 (4 s): EMA 0.5 * 4000 + 0.5 * 2000
    fens3 = {"A": START.replace(" 0 1", " 0 2"), "B": FEN_B_BLACK}
    ctl._observe_pace(obs_at("A_white", fens3, now_ms=3000, clocks={**clocks2, "A_black": 59000},
                             last={"A": 3000, "B": 0}))
    fens4 = {"A": FEN_A_BLACK.replace(" 0 1", " 0 2"), "B": FEN_B_BLACK}
    ctl._observe_pace(obs_at("A_black", fens4, now_ms=7000, clocks={**clocks2, "A_black": 59000, "A_white": 54000},
                             last={"A": 7000, "B": 0}))
    assert ctl._pace_ema["A_black"] == 1000.0 and ctl._pace_ema["A_white"] == 3000.0
    assert fen_plies(START) == 0 and fen_plies(FEN_A_BLACK) == 1 and fen_plies("x") is None


# ---------------------------------------------------------------------------------------------- exploration

def test_noise_on_recorded_searches_and_temperature_early_only():
    stub = ScriptStub([("e2e4", None)])
    ctl = free_ctl(stub, record=True, noise=True, temperature=1.0, temperature_plies=20, dedupe="bucket")
    ctl.player("A_white").decide(obs_at("A_white", {"A": START, "B": FEN_B_BLACK}, ply=3))
    opts = stub.calls[-1]["options"]
    assert opts["OpeningNoise"] is True and opts[NOISE_ALL_GAME_OPTION] is True
    assert opts["OpeningNoiseAlphaPermille"] == 300 and opts[TEMPERATURE_OPTION] == 1000
    ctl.player("A_white").decide(obs_at("A_white", {"A": START, "B": START}, ply=25, now_ms=9000))
    assert stub.calls[-1]["options"][TEMPERATURE_OPTION] == 0
    plain = free_ctl(ScriptStub([("e2e4", None)]))
    plain.player("A_white").decide(obs_at("A_white", {"A": START, "B": FEN_B_BLACK}))
    assert set(plain.process.calls[0]["options"]) == {"FreePass", "MateProbeAnyTime", "TreeMoveMs"}
    with pytest.raises(ValueError, match="free"):
        HivemindTeamController("T1", StubProcess(None), noise=True)


def test_playout_cap_randomisation():
    stub = ScriptStub([("e2e4", None)])
    ctl = free_ctl(stub, record=True, noise=True, pcr_p=0.25, pcr_nodes=400, dedupe="bucket", seed=9)
    n = 200
    for i in range(n):  # a new position every time: nothing is a repeat
        fens = {"A": START, "B": FEN_B_BLACK if i % 2 else START}
        clocks = {**CLOCKS, "A_white": 59000 - 37 * i}
        d = ctl.player("A_white").decide(obs_at("A_white", fens, now_ms=10_000 * (i + 1), clocks=clocks))
        full = stub.calls[-1]["nodes"] == 400
        assert d.think_ms == d.nodes * 5  # charged as the clock-budget search either way
        assert stub.calls[-1]["options"]["OpeningNoise"] is full
    kept = len(ctl.samples)
    assert kept + ctl.unrecorded_fast == n and 0.15 * n < kept < 0.35 * n
    assert all(s.extra["search_kind"] == "full" and s.nodes_requested == 400 for s in ctl.samples)


# ---------------------------------------------------------------------------------------------- dedupe, sits

def test_pass_then_move_records_both_with_sit_length():
    """The audit's test: a stub that passes 10 times then moves keeps the first pass and the move search, whose
    clock planes differ, and the pass sample carries how long the seat then sat."""
    stub = ScriptStub([(None, None)] * 10 + [("e2e4", None)])
    ctl = free_ctl(stub, record=True, dedupe="bucket", clock_bucket_ms=60_000)
    fens = {"A": START, "B": START}
    for i in range(11):
        clocks = {**CLOCKS, "A_white": 59000 - 600 * i, "B_white": 57000 - 600 * i}
        d = ctl.player("A_white").decide(obs_at("A_white", fens, now_ms=1000 + 600 * i, clocks=clocks))
    assert d.kind == "move" and len(stub.calls) == 11
    first, last = ctl.samples
    assert first.extra["search_kind"] == "full" and not first.extra["transition"]
    assert last.extra["transition"] and ctl.transitions == 1 and ctl.repeats_dropped == 9
    assert first.clocks_ms[0] == 59000.0 and last.clocks_ms[0] == 53000.0  # A_white first
    assert first.extra["sit_ms"] == 6000.0 and last.extra["sit_ms"] == 0.0
    meta = first.meta(7)
    for k in ("nodes_requested", "nodes_searched", "sit_ms", "start_offset_ms", "clocks_ms"):
        assert k in meta
    assert meta["clocks_ms"] == [59000.0, 58000.0, 57000.0, 56000.0] and meta["sit_ms"] == 6000.0
    assert meta["gameId"] == 7 and meta["fenA"] == START and meta["nodes_requested"] == 118


def test_clock_bucket_dedupe_keeps_one_per_bucket_and_finish_fills_sits():
    stub = ScriptStub([(None, None)])
    ctl = free_ctl(stub, record=True, dedupe="bucket", clock_bucket_ms=2000, start_offset_ms=-15000)
    fens = {"A": START, "B": START}
    for i in range(8):  # 600 ms apart: the clocks cross a 2 s bucket edge every few searches
        clocks = {**CLOCKS, "A_white": 59000 - 600 * i}
        ctl.player("A_white").decide(obs_at("A_white", fens, now_ms=1000 + 600 * i, clocks=clocks))
    buckets = {(59000 - 600 * i) // 2000 for i in range(8)}
    assert len(ctl.samples) == len(buckets) and ctl.repeats_dropped == 8 - len(buckets)
    assert all(s.extra["sit_ms"] is None for s in ctl.samples)
    ctl.finish(10_000)
    assert [s.extra["sit_ms"] for s in ctl.samples][0] == 9000.0
    assert all(s.start_offset_ms == -15000 and s.meta(1)["start_offset_ms"] == -15000 for s in ctl.samples)


def test_sit_recheck_skips_unchanged_wakes_and_does_not_block_partner():
    stub = ScriptStub([(None, None), (None, None), ("e2e4", None)])
    ctl = free_ctl(stub, sit_recheck=True, recheck_nodes=20, free_sit_ms=500)
    seat = ctl.player("A_white")
    fens = {"A": START, "B": START}
    d = seat.decide(obs_at("A_white", fens, now_ms=1000))  # 590 ms search, pass: next look at 2090
    assert d.kind == "wait" and len(stub.calls) == 1 and len(ctl.stalls) == 1
    busy = ctl._busy_until
    # woken at 1800 by a move on board B without a capture: board A's FEN unchanged -> no search
    d = seat.decide(obs_at("A_white", {"A": START, "B": FEN_B_BLACK}, now_ms=1800))
    assert d.kind == "wait" and d.wait_ms == 290 and len(stub.calls) == 1 and ctl.rechecks_skipped == 1
    # due: a 20-node re-look that passes; it neither queues behind nor moves the partner's machine
    d = seat.decide(obs_at("A_white", fens, now_ms=2090))
    assert len(stub.calls) == 2 and stub.calls[-1]["nodes"] == 20 and d.think_ms == 100
    assert ctl._busy_until == busy and ctl.rechecks == 1 and len(ctl.stalls) == 1  # same sit spell
    # a piece arrives in A_white's pocket before the next look is due: re-look now, it wants to move, full search
    pocket = START.replace("[]", "[N]")
    d = seat.decide(obs_at("A_white", {"A": pocket, "B": START}, now_ms=2300))
    assert len(stub.calls) == 4 and stub.calls[2]["nodes"] == 20 and stub.calls[3]["nodes"] == 118
    assert d.kind == "move" and d.think_ms == 100 + 590


def test_recheck_off_by_default_searches_every_wake():
    stub = ScriptStub([(None, None)])
    ctl = free_ctl(stub)
    for i in range(3):
        ctl.player("A_white").decide(obs_at("A_white", {"A": START, "B": START}, now_ms=1000 + 10 * i))
    assert len(stub.calls) == 3 and ctl.rechecks_skipped == 0


def test_stall_log_fields():
    stub = ScriptStub([(None, None)])
    ctl = free_ctl(stub)
    ctl.player("A_white").decide(obs_at("A_white", {"A": START, "B": START}))
    st = ctl.stalls[0]
    # A_white 59000 vs the partner's opponent B_white 57000; B_white is on turn on board B
    assert st == {"seat": "A_white", "t": 1000, "diagonal_ms": 2000, "partop_on_turn": True, "in_check": False,
                  "legal": 20}


# ---------------------------------------------------------------------------------------------- no root, head

def test_stock_no_root_is_counted_and_optionally_played_through():
    from arena.players.hivemind import HivemindProcess

    class NoRoot(StubProcess):
        def search(self, fen_a, fen_b, **kw):
            self.calls.append(kw)
            return HivemindProcess._parse([], "bestmove (none)", 5)

    obs = make_obs("A_white", {"A": START, "B": START})
    ctl = HivemindTeamController("T1", NoRoot(None), nodes=50, ms_per_node=5.0)
    assert ctl.player("A_white").decide(obs).kind == "wait" and ctl.no_root_results == 1
    ctl2 = HivemindTeamController("T1", NoRoot(None), nodes=50, ms_per_node=5.0, stock_no_root="heuristic")
    d = ctl2.player("A_white").decide(obs)
    assert d.kind == "move" and d.uci in obs.legal_moves and ctl2.no_root_results == 1


def test_think_head_multiplier_normalised_to_mean_one():
    ps = [0.0, 1.0, 0.0, 1.0]
    stub = ScriptStub([("e2e4", None)])
    it = iter(ps)
    orig = stub.search

    def search(fen_a, fen_b, **kw):
        res = orig(fen_a, fen_b, **kw)
        if kw["nodes"] == 16:
            p = next(it)
            res.think_gain = (p, p)
        return res

    stub.search = search
    ctl = free_ctl(stub, think_head=True, think_head_norm=True)
    totals = []
    for i, _ in enumerate(ps):
        fens = {"A": START, "B": FEN_B_BLACK if i % 2 else START}
        totals.append(ctl.player("A_white").decide(obs_at("A_white", fens, now_ms=10_000 * (i + 1))).nodes)
    # raw multipliers 0.5, 2, 0.5, 2 over a budget of 118, each divided by the running mean of the raw ones so far:
    # 0.5 / 0.5, 2 / 1.25, 0.5 / 1, 2 / 1.25 (unnormalised the second move would take 236 nodes)
    assert totals == [118, 189, 59, 189]
    long = HivemindTeamController("T1", ScriptStub([]), think_head=True, think_head_norm=True, free=True)
    mults = []
    for i in range(2000):  # in the long run the mean multiplier is 1 whatever the head's mean p
        long._th_mult_sum += 0.5 + 1.5 * (0.2 if i % 3 else 0.9)
        long._th_mult_n += 1
        mults.append((0.5 + 1.5 * (0.2 if i % 3 else 0.9)) / (long._th_mult_sum / long._th_mult_n))
    assert sum(mults[200:]) / len(mults[200:]) == pytest.approx(1.0, abs=0.01)


def test_setups_pass_v3_options():
    from arena.players import make_players

    ps = make_players("centralized", "hivemind", free=True, binary="hivemind-clock", charge="searched",
                      pace=True, pcr_p=0.25, dedupe="bucket", sit_recheck=True, min_think_ms=100,
                      pace_mixture=[[1, 1.0]], stock_no_root="heuristic")
    try:
        ctl = ps["A_white"].controller
        assert ctl.charge == "searched" and ctl.pace and ctl.pcr_p == 0.25 and ctl.dedupe == "bucket"
        assert ctl.sit_recheck and ctl.min_think_ms == 100 and ctl.pace_mixture == ((1.0, 1.0),)
    finally:
        for p in ps.values():
            p.close()
    split = make_players("silent", "hivemind", charge="searched")
    try:
        assert all(p.charge == "searched" for p in split.values())
    finally:
        for p in split.values():
            p.close()


# ---------------------------------------------------------------------------------------------- self-play specs

def _args(*extra):
    from arena.selfplay import parse_args
    return parse_args(["--games", "200", "--out", "/nonexistent/x", "--seed", "3", *extra])


def test_v2_regime_is_unchanged_by_default():
    from arena.selfplay import controller_kw, make_specs

    args = _args()
    assert args.regime == "v2" and args.base_mix is None and not args.time_odds
    spec = make_specs(args)[0]
    assert spec["base_ms"] == 60000 and spec["max_plies"] == 400 and spec["k"] is None
    kw = controller_kw(spec, "T1")
    assert kw["k_range"] == (60.0, 300.0) and kw["jitter"] == (0.5, 2.0)
    assert not {"charge", "pace", "pcr_p", "dedupe", "sit_recheck", "noise", "pace_mixture"} & set(kw)


def test_time_odds_share_and_sign_balanced_over_200_specs():
    from arena.selfplay import make_specs

    specs = make_specs(_args("--regime", "v3"))
    one = [s for s in specs if sum(1 for v in s["start_offsets"].values() if v) == 1]
    both = [s for s in specs if all(v < 0 for v in s["start_offsets"].values())]
    assert 0.15 < len(one) / 200 < 0.35 and 0.03 < len(both) / 200 < 0.18
    signs = [v > 0 for s in one for v in s["start_offsets"].values() if v]
    teams = [t for s in one for t, v in s["start_offsets"].items() if v]
    assert 0.3 < sum(signs) / len(signs) < 0.7 and 0.3 < teams.count("T1") / len(teams) < 0.7
    for s in specs:  # a start clock is never below a quarter of base
        assert all(s["base_ms"] + v >= s["base_ms"] // 4 for v in s["start_offsets"].values())


def test_base_time_mix_and_move1_budget_independent():
    from arena.selfplay import controller_kw, make_specs

    specs = make_specs(_args("--regime", "v3", "--games", "600"))
    bases = [s["base_ms"] for s in specs]
    share = {b: bases.count(b) / len(bases) for b in set(bases)}
    assert set(share) == {60000, 120000, 180000, 300000} and 0.27 < share[180000] < 0.43
    # move-1 budget = base / k = the think target, whatever the base
    budgets = [s["base_ms"] / controller_kw(s, "T1")["k_range"][0] for s in specs]
    assert all(200 - 1e-6 <= b <= 1000 + 1e-6 for b in budgets)
    mean = sum(budgets) / len(budgets)
    by_base = {b: [x for x, bb in zip(budgets, bases) if bb == b] for b in share}
    assert all(abs(sum(v) / len(v) - mean) < 0.15 * mean for v in by_base.values())
    xb, xm = sum(bases) / len(bases), mean
    cov = sum((b - xb) * (x - xm) for b, x in zip(bases, budgets))
    corr = cov / math.sqrt(sum((b - xb) ** 2 for b in bases) * sum((x - xm) ** 2 for x in budgets))
    assert abs(corr) < 0.1
    assert {s["max_plies"] for s in specs} == {400, 800, 1200, 2000}  # scaled by base / 60 s


def test_explicit_flag_wins_over_regime_default():
    from arena.selfplay import controller_kw, make_specs

    spec = make_specs(_args("--regime", "v3", "--charge", "requested", "--time-odds", "off", "--pcr-p", "0.5"))[0]
    kw = controller_kw(spec, "T2")
    assert kw["charge"] == "requested" and kw["pcr_p"] == 0.5 and spec["start_offsets"] == {"T1": 0, "T2": 0}
    assert kw["pace"] and kw["dedupe"] == "bucket" and kw["sit_recheck"] and kw["pace_mixture"]
    assert make_specs(_args("--pace", "on"))[0]["v3"]["pace"] is True  # also under v2


class CountingStub(StubProcess):
    """LegalStub-like engine counting its own lifetime (acquire/release)."""

    def __init__(self, seed):
        from arena.tests.test_custom_engine import LegalStub
        super().__init__(None)
        self.inner = LegalStub(seed)
        self.refs = 1
        self.closed = False

    def search(self, fen_a, fen_b, **kw):
        self.calls.append(kw)
        return self.inner.search(fen_a, fen_b, **kw)

    def acquire(self):
        self.refs += 1
        return self

    def release(self):
        self.refs -= 1
        if self.refs <= 0:
            self.closed = True


def test_v3_selfplay_end_to_end_persistent_engines(tmp_path):
    from arena.recorder import read_sidecar
    from arena.selfplay import parse_args, run

    args = parse_args(["--games", "2", "--out", str(tmp_path / "v3"), "--regime", "v3", "--base-mix",
                       "20000:1", "--max-plies", "240", "--seed", "5", "--no-events"])
    made = []

    def factory(team):
        made.append(CountingStub(len(made) + 20))
        return made[-1]

    report = run(args, process_factory=factory)
    r = report["results"]
    assert r["games"] == 2 and r["errors"] == 0, report["games"]
    assert len(made) == 2  # one engine per team for both games
    assert all(s.closed for s in made)  # released at the end of the run
    assert all(len(s.calls) > 10 for s in made)
    calls = [c for s in made for c in s.calls]
    assert all("pace" in c and c["options"]["FreePass"] for c in calls)
    assert r["nodes_requested"] > 0 and "stalls" in r and r["base_ms"] == {"20000": 2}
    metas = [m for c in sorted((tmp_path / "v3" / "training_data").glob("*.hvm")) for m in read_sidecar(str(c))]
    assert len(metas) == r["samples"] > 0
    for m in metas:
        assert m["sit_ms"] is not None and m["sit_ms"] >= 0 and len(m["pace"]) == 4
        assert m["search_kind"] in ("full", "fast", "recheck") and m["nodes_searched"] >= 1
    assert any(m["search_kind"] == "full" for m in metas)
    assert all(m["search_kind"] != "fast" or m["transition"] for m in metas)  # fast ones only as transitions
