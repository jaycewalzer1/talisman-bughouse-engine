"""Custom engine, arena side (docs/custom_engine_spec.md section 3): free-mode controller, clocks and sample
plumbing in HivemindProcess, the HVM5 recorder (round trip through hivemind's own reader) and self-play, all
against stub engines (the hivemind-clock binary is not needed)."""

from __future__ import annotations

import base64
import os
import random

import chess
import chess.variant
import pytest

from arena.interfaces import SEATS, TEAM_OF
from arena.players import HivemindProcess, HivemindTeamController, SearchResult, make_players
from arena.players.hivemind import position_command, resolve_binary, DEFAULT_BINARY
from arena.recorder import (
    HEADER,
    ShardWriter,
    encode_sample,
    finish_game,
    infer_channels,
    outcome_for,
    patch_sample,
    read_chunk,
    read_sidecar,
    view_sample,
)
from arena.tests.test_players import START, StubProcess, make_obs

FEN_B_BLACK = START.replace(" w ", " b ")
CLOCKS = {"A_white": 59000, "A_black": 58000, "B_white": 57000, "B_black": 56000}


def fake_sample(team: int, channels: int = 82, *, q: float = 0.25, nodes: int = 100, ply: int = 3) -> bytes:
    planes = bytes((i * 7 + team) % 256 for i in range(channels * 64))
    return encode_sample(planes=planes, policy_a=[(5, 0.75), (4672 - 1, 0.25)], policy_b=[(10, 1.0)],
                         joint=[(5, 4672, 0.6), (4671, 10, 0.4)], nodes=nodes, macro_ply=ply, team=team,
                         has_time_advantage=1, root_q=q)


class SampleStub(StubProcess):
    """StubProcess that also returns a sample with the right team byte."""

    def search(self, fen_a, fen_b, **kw):
        self.calls.append(kw)
        res = SearchResult(self.result.move_a, self.result.move_b, 0.1, 20, None, 120, 3)
        if kw.get("sample"):
            res.sample = fake_sample(0 if kw["team"] == "T1" else 1)
        return res


# ---------------------------------------------------------------------------------------------- process plumbing

def test_position_command_clock_suffix_and_stock_form():
    assert position_command("a", "b") == "position fen a | b"
    assert position_command("a", "b", CLOCKS) == "position fen a | b clocks 59000 58000 57000 56000"
    assert position_command("a", "b", {**CLOCKS, "B_black": -12}).endswith("clocks 59000 58000 57000 0")


def test_parse_reads_base64_sample_line():
    blob = fake_sample(1)
    lines = ["info depth 3 score cp 20 nodes 120 nps 1 time 1 pv (e2e4,pass)",
             "info string sample " + base64.b64encode(blob).decode()]
    res = HivemindProcess._parse(lines, "bestmove (e2e4,pass)", 5)
    assert res.sample == blob and res.move_a == "e2e4" and res.move_b is None and res.nodes == 120
    assert HivemindProcess._parse(lines[:1], "bestmove (e2e4,pass)", 5).sample is None


def test_binary_choice():
    assert resolve_binary(None) == resolve_binary("hivemind") == DEFAULT_BINARY
    assert resolve_binary("hivemind-clock").endswith("hivemind-clock")
    assert resolve_binary("/x/y") == "/x/y"


# ---------------------------------------------------------------------------------------------- free controller

def test_free_controller_sends_clocks_and_free_options():
    stub = SampleStub(SearchResult("e2e4", "e7e5", 0.2, 40, None, 241, 3))
    ctl = HivemindTeamController("T1", stub, ms_per_node=5.0, free=True, record=True, k_range=(100, 100),
                                 jitter=(1.0, 1.0), seed=3)
    assert ctl.serial and ctl.clock_k == 100
    d = ctl.player("A_white").decide(make_obs("A_white", {"A": START, "B": FEN_B_BLACK}, clocks=CLOCKS))
    call = stub.calls[-1]
    assert call["clocks"] == CLOCKS and call["sample"] is True and call["require_option"] is False
    assert call.get("require") is None
    # 59 s / 100 = 590 ms = 118 nodes at 5 ms
    assert call["nodes"] == 118 and d.think_ms == 590 and d.kind == "move" and d.uci == "e2e4"
    assert call["options"] == {"FreePass": True, "MateProbeAnyTime": True, "TreeMoveMs": 590}
    assert len(ctl.samples) == 1 and ctl.samples[0][0] == 1000
    # the partner at the same position reuses the joint result: no new search, no new sample
    db = ctl.player("B_black").decide(make_obs("B_black", {"A": START, "B": FEN_B_BLACK}, clocks=CLOCKS))
    assert db.uci == "e7e5" and len(stub.calls) == 1 and len(ctl.samples) == 1


def test_free_controller_pass_waits_then_searches_again():
    stub = SampleStub(SearchResult(None, None, 0.0, 0, None, 100, 2))  # passes on both boards
    ctl = HivemindTeamController("T1", stub, ms_per_node=5.0, free=True, k_range=(100, 100), jitter=(1.0, 1.0),
                                 free_sit_ms=500, max_consecutive_sits=0)
    obs = make_obs("A_white", {"A": START, "B": START}, clocks=CLOCKS)
    for i in range(4):  # no sit cap, even with max_consecutive_sits=0; a pass is never reused from the cache
        d = ctl.player("A_white").decide(obs)
        assert d.kind == "wait" and d.stall_reason == "free_pass"
        assert d.wait_ms == d.think_ms + 500
        assert len(stub.calls) == i + 1
    assert "RequireMoveOn" not in str(stub.calls)


class NoRootStub(StubProcess):
    """Engine answer ``bestmove (none)``: no root, so no sample either."""

    def search(self, fen_a, fen_b, **kw):
        self.calls.append(kw)
        return HivemindProcess._parse([], "bestmove (none)", 5)


def test_parse_marks_bestmove_none_as_no_root():
    assert HivemindProcess._parse([], "bestmove (none)", 5).no_root
    assert not HivemindProcess._parse([], "bestmove (pass,pass)", 5).no_root
    assert not HivemindProcess._parse([], "bestmove (e2e4,pass)", 5).no_root


def test_free_controller_no_root_with_legal_moves_logs_and_moves(caplog):
    stub = NoRootStub(None)
    ctl = HivemindTeamController("T1", stub, ms_per_node=5.0, free=True, record=True, k_range=(100, 100),
                                 jitter=(1.0, 1.0))
    obs = make_obs("B_black", {"A": START, "B": FEN_B_BLACK}, clocks=CLOCKS)
    with caplog.at_level("ERROR", logger="arena.players.hivemind"):
        d = ctl.player("B_black").decide(obs)
    assert d.kind == "move" and d.uci in obs.legal_moves  # heuristic fallback, not a silent wait or a crash
    assert ctl.no_root_results == 1 and ctl.samples == []  # nothing recorded for that search
    assert "bestmove (none)" in caplog.text and FEN_B_BLACK in caplog.text


# Same table as engine/tests/test_custom_engine.cc FreePassMateFollowsArenaRule: under FreePass the engine's
# is_checkmate must give exactly these verdicts, whatever TimeAdvantage or the partner board.
ARENA_MATE_CASES = [
    ("stalemate", "7k/8/8/8/8/8/5q2/7K[] w - - 0 1", False),
    ("blockable_no_moves", "k7/8/8/8/5n2/5n2/8/r6K[] w - - 0 1", False),
    ("back_rank_blockable", "R5k1/5ppp/8/8/8/8/8/6K1[] b - - 0 1", False),
    ("smothered", "6rk/5Npp/8/8/8/8/8/K7[] b - - 0 1", True),
    ("contact_queen", "k7/1Q6/1K6/8/8/8/8/8[] b - - 0 1", True),
    ("double_check", "4R2k/5Npp/8/8/8/8/8/K7[] b - - 0 1", True),
    ("quiet", "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR[] b KQkq - 0 1", False),
]


@pytest.mark.parametrize("name,fen,mate", ARENA_MATE_CASES)
def test_arena_mate_table_matches_engine_test(name, fen, mate):
    from arena.state import is_bughouse_mate
    assert is_bughouse_mate(chess.variant.CrazyhouseBoard(fen)) is mate


@pytest.mark.skipif(not os.path.exists(resolve_binary("hivemind-clock")), reason="needs the hivemind-clock build")
def test_engine_free_mode_does_not_score_stuck_board_as_mate():
    """The review repro: board A stalemated (white, not in check), board B quiet. Under FreePass the engine
    must not report mate for either TimeAdvantage value; the arena does not end this game."""
    from arena.players.hivemind import ENGINE_DIR
    model = os.path.join(ENGINE_DIR, "models", "hivemind-clock-fp32.onnx")
    if not os.path.exists(model):
        pytest.skip("needs hivemind-clock-fp32.onnx")
    proc = HivemindProcess(binary=resolve_binary("hivemind-clock"), model=model)
    try:
        for fen_a in ("7k/8/8/8/8/8/5q2/7K[] w - - 0 1", "k7/8/8/8/5n2/5n2/8/r6K[] w - - 0 1"):
            for ta in (False, True):
                r = proc.search(fen_a, FEN_B_BLACK, team="T1", nodes=32, time_advantage=ta,
                                clocks={s: 60000 for s in SEATS}, sample=True, require_option=False,
                                options={"FreePass": True, "MateProbeAnyTime": True, "TreeMoveMs": 300})
                assert not r.no_root and r.sample is not None and r.mate is None, (fen_a, ta, r.lines[-3:])
                assert r.move_a is None and r.move_b is not None  # waits on A, moves on B
    finally:
        proc.close()


def test_free_controller_k_per_game_and_jitter_per_search():
    ks = {HivemindTeamController("T1", StubProcess(None), free=True, seed=s).clock_k for s in range(40)}
    assert len(ks) == 40 and all(60 <= k <= 300 for k in ks)
    t1 = HivemindTeamController("T1", StubProcess(None), free=True, seed=5)
    t2 = HivemindTeamController("T2", StubProcess(None), free=True, seed=5)
    assert t1.clock_k != t2.clock_k  # drawn per team
    stub = SampleStub(SearchResult("e2e4", None, 0.1, 20, None, 120, 3))
    ctl = HivemindTeamController("T1", stub, ms_per_node=5.0, free=True, k_range=(100, 100), seed=1)
    assert ctl.jitter == (0.5, 2.0)
    thinks = []
    for i in range(30):  # far apart in time, so no queueing: think = 590 ms * jitter
        fens = {"A": START, "B": START if i % 2 else FEN_B_BLACK}
        d = ctl.player("A_white").decide(make_obs("A_white", fens, clocks=CLOCKS, now_ms=10_000 * (i + 1)))
        thinks.append(d.think_ms)
        assert stub.calls[-1]["options"]["TreeMoveMs"] == d.think_ms
    assert all(295 <= t <= 1180 for t in thinks) and len(set(thinks)) > 20


def test_free_controller_is_serial():
    stub = SampleStub(SearchResult("e2e4", "e7e5", 0.2, 40, None, 241, 3))
    ctl = HivemindTeamController("T1", stub, ms_per_node=5.0, free=True, k_range=(100, 100), jitter=(1.0, 1.0))
    da = ctl.player("A_white").decide(make_obs("A_white", {"A": START, "B": FEN_B_BLACK}, clocks=CLOCKS,
                                               now_ms=1000))
    other = {"A": START.replace(" w ", " b "), "B": FEN_B_BLACK}
    db = ctl.player("B_black").decide(make_obs("B_black", other, clocks=CLOCKS, now_ms=1200))
    assert da.think_ms == 590 and db.think_ms == 390 + round(56000 / 100 / 5) * 5


def test_record_needs_free_and_checks_team_byte():
    with pytest.raises(ValueError, match="record"):
        HivemindTeamController("T1", StubProcess(None), record=True)
    stub = SampleStub(SearchResult("e2e4", None, 0.1, 20, None, 120, 3))
    ctl = HivemindTeamController("T2", stub, free=True, record=True)
    stub.search = lambda *a, **kw: SearchResult("e7e5", None, 0, 0, None, 1, 1, sample=fake_sample(0))  # T1's byte
    with pytest.raises(ValueError, match="team byte"):
        ctl.player("A_black").decide(make_obs("A_black", {"A": FEN_B_BLACK, "B": START}, clocks=CLOCKS))


def test_stock_controller_jitter_off_by_default_and_optional():
    stub = StubProcess(SearchResult("e2e4", None, 0.1, 20, None, 120, 3))
    ctl = HivemindTeamController("T1", stub, nodes=200, ms_per_node=5.0)
    assert ctl.jitter is None and not ctl.free
    d = ctl.player("A_white").decide(make_obs("A_white", {"A": START, "B": START}))
    assert d.think_ms == 1000 and "clocks" not in stub.calls[-1] and "options" not in stub.calls[-1]
    jit = HivemindTeamController("T1", stub, nodes=200, ms_per_node=5.0, jitter=(0.5, 2.0), seed=2)
    thinks = {jit.player("A_white").decide(make_obs("A_white", {"A": START, "B": START if i % 2 else FEN_B_BLACK},
                                                    now_ms=10_000 * (i + 1))).think_ms for i in range(10)}
    assert len(thinks) > 5 and all(500 <= t <= 2000 for t in thinks)


def test_setups_pass_free_options_and_refuse_split():
    ps = make_players("centralized", "hivemind", nodes=200, free=True, k_range=[80, 90], jitter=[0.9, 1.1],
                      binary="hivemind-clock", record=True)
    try:
        ctl = ps["A_white"].controller
        assert ctl.free and ctl.record and ctl.jitter == (0.9, 1.1) and 80 <= ctl.clock_k <= 90
        assert ctl.process.binary.endswith("hivemind-clock")
        assert ps["A_black"].controller.clock_k != ctl.clock_k
    finally:
        for p in ps.values():
            p.close()
    with pytest.raises(ValueError, match="centralized"):
        make_players("silent", "hivemind", jitter=[0.5, 2.0])


def test_pool_jitter_reaches_both_centralized_teams():
    from arena.pool import Pool, for_evaluation

    pool = for_evaluation(Pool.load("arena/pools/phase1.json"), setup="centralized", opponent="stock-centralized",
                          contender_kw={"free": True}, jitter=(0.5, 2.0))
    assert pool.contender.player_kw == {"free": True, "jitter": [0.5, 2.0]}
    assert [o.player_kw for o in pool.opponents] == [{"jitter": [0.5, 2.0]}]
    plain = for_evaluation(Pool.load("arena/pools/phase1.json"), setup="centralized", opponent="stock-centralized")
    assert "jitter" not in plain.contender.player_kw and plain.opponents[0].player_kw == {}


# ---------------------------------------------------------------------------------------------- recorder

def test_patch_and_outcome():
    b = fake_sample(1)
    v = view_sample(patch_sample(b, game_id=77, outcome=-1, moves_left=9))
    assert (v.game_id, v.outcome, v.wdl, v.moves_left, v.team, v.nodes, v.macro_ply) == (77, -1, 0, 9, 1, 100, 3)
    assert v.root_q == pytest.approx(0.25) and v.channels == 82
    assert outcome_for(0, "T1") == 1 and outcome_for(1, "T1") == -1 and outcome_for(1, "T2") == 1
    assert outcome_for(0, "draw") == 0
    assert infer_channels(fake_sample(0, 74)) == 74


def test_finish_game_orders_by_time_and_fills_fields():
    samples = [(500, 2, fake_sample(1)), (100, 9, fake_sample(0)), (500, 1, fake_sample(0))]
    out = [view_sample(b) for b in finish_game(samples, "T2", 42)]
    assert [v.team for v in out] == [0, 0, 1]
    assert [v.outcome for v in out] == [-1, -1, 1] and [v.wdl for v in out] == [0, 0, 2]
    assert [v.moves_left for v in out] == [3, 2, 1] and {v.game_id for v in out} == {42}
    assert all(v.outcome == 0 for v in map(view_sample, finish_game(samples, "draw", 1)))


@pytest.mark.parametrize("channels", [74, 82])
def test_shard_round_trip_through_hivemind_reader(tmp_path, monkeypatch, channels):
    np = pytest.importorskip("numpy")
    conv = pytest.importorskip("hivemind.data.convert_selfplay_data")
    if channels not in getattr(conv, "SUPPORTED_INPUT_CHANNELS", (conv.NB_INPUT_CHANNELS,)):
        # an older reader built for one width only: point it at this one
        monkeypatch.setattr(conv, "NB_INPUT_CHANNELS", channels)
        monkeypatch.setattr(conv, "NB_INPUT_VALUES", channels * 64)
    raw = [(i * 10, i, fake_sample(i % 2, channels, q=0.1 * i, nodes=50 + i, ply=i)) for i in range(5)]
    finished = finish_game(raw, "T1", 1234)
    w = ShardWriter(str(tmp_path), "it01-s0", samples_per_chunk=3)
    w.append(finished)
    w.close()
    assert [os.path.basename(p) for p in w.paths] == ["chunk_it01-s0_000000.hvm", "chunk_it01-s0_000001.hvm"]
    assert w.written == 5 and w.channels == channels
    got = [s for p in w.paths for s in conv.read_binary_shard(p)]
    assert len(got) == 5
    for i, s in enumerate(got):
        team = i % 2
        assert s["game_id"] == 1234 and s["team"] == team and s["macro_ply"] == i and s["search_nodes"] == 50 + i
        assert s["outcome"] == (1 if team == 0 else -1) and s["y_wdl"] == s["outcome"] + 1
        assert s["y_moves_left"] == 5 - i and s["has_time_advantage"] is True
        assert s["root_q"] == pytest.approx(0.1 * i, abs=1e-6)
        assert s["x"] == view_sample(finished[i]).planes and len(s["x"]) == channels * 64
        pa = np.frombuffer(s["policy_a"], dtype=np.float32)
        assert pa[5] == pytest.approx(0.75) and pa[4671] == pytest.approx(0.25) and pa.sum() == pytest.approx(1)
        assert np.frombuffer(s["policy_b"], dtype=np.float32)[10] == pytest.approx(1.0)
        assert s["joint_policy_count"] == 2
        ja = np.frombuffer(s["joint_policy_a"], dtype=np.uint16)
        assert list(ja[:2]) == [5, 4671]
    # and our own light reader agrees byte for byte
    assert [b for p in w.paths for b in read_chunk(p)[1]] == finished
    with open(w.paths[0], "rb") as f:
        assert HEADER.unpack(f.read(HEADER.size)) == (b"HVM5", 5, channels, 4672, 3)
    w2 = ShardWriter(str(tmp_path), "it01-s0")  # a second writer never overwrites the first one's chunks
    w2.append(finished[:1])
    w2.close()
    assert os.path.basename(w2.paths[0]) == "chunk_it01-s0_000002.hvm"


def test_writer_refuses_mixed_widths(tmp_path):
    w = ShardWriter(str(tmp_path), "x")
    w.append([fake_sample(0, 82)])
    with pytest.raises(ValueError):
        w.append([fake_sample(0, 74)])


# ---------------------------------------------------------------------------------------------- self-play

class LegalStub:
    """An engine that plays a random legal move for its team on each board where it is on turn, passes
    sometimes, and returns a sample with the right team byte."""

    def __init__(self, seed: int, pass_rate: float = 0.15):
        self.rng = random.Random(seed)
        self.pass_rate = pass_rate
        self.calls = []

    def search(self, fen_a, fen_b, *, team, nodes, **kw):
        self.calls.append(dict(kw, team=team, nodes=nodes))
        moves = []
        for board, fen in (("A", fen_a), ("B", fen_b)):
            b = chess.variant.CrazyhouseBoard(fen)
            seat = f"{board}_{'white' if b.turn else 'black'}"
            legal = [m.uci() for m in b.legal_moves]
            ours = TEAM_OF[seat] == team
            moves.append(self.rng.choice(legal) if ours and legal and self.rng.random() > self.pass_rate else None)
        return SearchResult(moves[0], moves[1], 0.0, 0, None, nodes, 1,
                            sample=fake_sample(0 if team == "T1" else 1, ply=len(self.calls)))

    def acquire(self):
        return self

    def release(self):
        pass


def _hivemind_reads(conv, path) -> list[dict] | None:
    """hivemind's reader on one chunk, or None when that reader does not know the chunk's format/width yet."""
    try:
        return conv.read_binary_shard(str(path))
    except ValueError as exc:
        if "Unsupported self-play chunk" in str(exc) or "channels" in str(exc):
            return None
        raise


@pytest.mark.parametrize("version", [5, 6])
def test_selfplay_end_to_end_with_stub_engine(tmp_path, version):
    from arena.selfplay import parse_args, run

    conv = pytest.importorskip("hivemind.data.convert_selfplay_data")
    out = tmp_path / "it00"
    args = parse_args(["--games", "2", "--out", str(out), "--base-ms", "20000", "--max-plies", "120",
                       "--seed", "4", "--samples-per-chunk", "50", "--shard-version", str(version)])
    stubs = []

    def factory(team):
        stubs.append(LegalStub(len(stubs) + 10))
        return stubs[-1]

    report = run(args, process_factory=factory)
    r = report["results"]
    assert r["games"] == 2 and r["errors"] == 0 and r["samples"] > 0 and r["channels"] == 82
    # every search either kept its sample or was a counted repeat of the seat's last position (spec v2 D2)
    assert r["samples"] + r["repeats_dropped"] == sum(len(s.calls) for s in stubs)
    for s in stubs:  # free mode on every call, with per-board elapsed times (spec v2 A)
        assert all(c["clocks"] is not None and c["options"]["FreePass"] and c["elapsed"] is not None
                   for c in s.calls)
    chunks = sorted((out / "training_data").glob("*.hvm"))
    assert [c.name for c in chunks] == r["chunks"] and len(chunks) >= 2
    own = [(view_sample(b, version=version), m) for c in chunks
           for b, m in zip(read_chunk(str(c))[1], read_sidecar(str(c)))]
    assert len(own) == r["samples"]
    got = None
    if all(_hivemind_reads(conv, c) is not None for c in chunks[:1]):
        got = [s for c in chunks for s in conv.read_binary_shard(str(c))]
        assert len(got) == r["samples"]
    for g in report["games"]:
        mine = [(v, m) for v, m in own if v.game_id == g["game_num"]]
        assert len(mine) == g["samples"]
        for v, m in mine:
            assert v.outcome == outcome_for(v.team, g["result"]) and m["gameId"] == g["game_num"]
            assert v.moves_left == g["plies"] - m["ply"]  # spec v2 D3: moves landed after the search started
            if version == 6:
                assert v.think_mask == (0, 0) and len(v.arrivals) == 20 and all(x >= 0 for x in v.arrivals)
            else:
                assert v.arrivals is None
        if version == 6:
            assert sum(sum(v.arrivals) for v, _ in mine) > 0  # random play captures within 5 s
    if got is not None:
        for (v, _), s in zip(own, got):
            assert s["game_id"] == v.game_id and s["y_moves_left"] == v.moves_left and s["x"] == v.planes
    assert (out / "summary.json").exists() and len(list((out / "games").glob("*.jsonl"))) == 2
