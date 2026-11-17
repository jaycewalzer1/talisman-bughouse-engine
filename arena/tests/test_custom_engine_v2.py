"""Custom engine spec v2, arena side (docs/custom_engine_spec_v2.md): elapsed times in ``position`` (A), think_gain
parsing and the free-mode think head (B), arrival labels (C), the HVM6 trailing block, the repeated-position filter
(D2), movesLeft from landed moves (D3) and tools/think_labels.py, all against stub engines."""

from __future__ import annotations

import base64
import math
import os
import struct
import sys

import pytest

from arena.interfaces import SEATS
from arena.players import HivemindProcess, HivemindTeamController, SearchResult
from arena.players.hivemind import parse_think_gain, position_command
from arena.recorder import (
    HEADER,
    TRAILER,
    RecordedSample,
    ShardWriter,
    arrival_counts,
    base_sample,
    captures,
    decode_trailer,
    encode_trailer,
    finish_game,
    infer_format,
    pocket_index,
    read_chunk,
    read_sidecar,
    sample_size,
    view_sample,
)
from arena.tests.test_custom_engine import CLOCKS, FEN_B_BLACK, LegalStub, fake_sample
from arena.tests.test_players import START, StubProcess, make_obs

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def obs_at(seat, fens, *, now_ms=1000, last=None, ply=0):
    from dataclasses import replace
    return replace(make_obs(seat, fens, clocks=CLOCKS, now_ms=now_ms), last_move_ms=last, ply=ply)


# ---------------------------------------------------------------------------------------------- A: elapsed

def test_position_command_elapsed_suffix():
    assert position_command("a", "b", CLOCKS, {"A": 1234, "B": -5}) == \
        "position fen a | b clocks 59000 58000 57000 56000 elapsed 1234 0"
    assert position_command("a", "b", None, {"A": 1, "B": 2}) == "position fen a | b"  # needs the clocks suffix
    assert position_command("a", "b", CLOCKS) == "position fen a | b clocks 59000 58000 57000 56000"


class SampleStub(StubProcess):
    """Returns ``self.result`` (or ``self.results`` in turn) with a sample of the right team byte when asked."""

    def __init__(self, result, results=None):
        super().__init__(result)
        self.results = list(results or [])

    def search(self, fen_a, fen_b, **kw):
        self.calls.append(kw)
        base = self.results.pop(0) if self.results else self.result
        res = SearchResult(base.move_a, base.move_b, 0.1, 20, None, kw["nodes"], 3, think_gain=base.think_gain,
                           no_root=base.no_root)
        if kw.get("sample"):
            res.sample = fake_sample(0 if kw["team"] == "T1" else 1)
        return res


def test_free_controller_sends_elapsed_since_last_landed_move():
    stub = SampleStub(SearchResult("e2e4", None, 0, 0, None, 0, 0))
    ctl = HivemindTeamController("T1", stub, ms_per_node=5.0, free=True, k_range=(100, 100), jitter=(1.0, 1.0))
    ctl.player("A_white").decide(obs_at("A_white", {"A": START, "B": FEN_B_BLACK}, now_ms=7000,
                                        last={"A": 6500, "B": 2000}))
    assert stub.calls[-1]["elapsed"] == {"A": 500, "B": 5000}
    ctl.player("A_white").decide(obs_at("A_white", {"A": START, "B": START}, now_ms=9000))  # no timing known
    assert stub.calls[-1]["elapsed"] is None
    off = HivemindTeamController("T1", stub, free=True, send_elapsed=False)
    off.player("A_white").decide(obs_at("A_white", {"A": START, "B": FEN_B_BLACK}, last={"A": 0, "B": 0}))
    assert stub.calls[-1]["elapsed"] is None


def test_simulator_observation_carries_last_move_times():
    from arena.sim import Simulator
    from arena.state import BughouseState
    from arena.interfaces import Decision

    seen = []

    class Mover:
        kind = "test"

        def __init__(self, seat):
            self.seat, self.name, self.n = seat, seat, 0

        def decide(self, obs):
            seen.append((obs.seat, obs.now_ms, dict(obs.last_move_ms)))
            return Decision(kind="move", uci=obs.legal_moves[0], think_ms=300 if obs.seat[0] == "A" else 700)

        def close(self):
            pass

    sim = Simulator(BughouseState(60000, 0), {s: Mover(s) for s in SEATS}, game_id="t", setup="t", max_plies=6)
    sim.run()
    assert seen[0][2] == {"A": 0, "B": 0}
    a_black = next(x for x in seen if x[0] == "A_black")
    assert a_black[1] == 300 and a_black[2]["A"] == 300 and a_black[2]["B"] == 0
    later = [x for x in seen if x[1] >= 700 and x[0] != "A_white"]
    assert any(x[2]["B"] == 700 for x in later)


# ---------------------------------------------------------------------------------------------- B: think head

def test_parse_think_gain_line():
    assert parse_think_gain("info string think_gain 0.25 0.75") == (0.25, 0.75)
    pa, pb = parse_think_gain("info string think_gain 2.0 -2.0")  # logits
    assert pa == pytest.approx(1 / (1 + math.exp(-2))) and pb == pytest.approx(1 / (1 + math.exp(2)))
    assert parse_think_gain("info string think_gain x 1") is None
    assert parse_think_gain("info string think_gain 0.5") is None
    lines = ["info string think_gain 0.1 0.2", "info depth 2 score cp 10 nodes 30 pv (e2e4,pass)",
             "info string think_gain 0.3 0.4"]
    assert HivemindProcess._parse(lines, "bestmove (e2e4,pass)", 1).think_gain == (0.3, 0.4)
    assert HivemindProcess._parse(lines[1:2], "bestmove (e2e4,pass)", 1).think_gain is None  # stock network


@pytest.mark.parametrize("p,total", [(0.0, 59), (1.0, 236), (0.5, 148)])
def test_think_head_scales_budget_and_charges_quick_look(p, total):
    stub = SampleStub(SearchResult("e2e4", None, 0, 0, None, 0, 0, think_gain=(p, 0.9)))
    ctl = HivemindTeamController("T1", stub, ms_per_node=5.0, free=True, record=True, k_range=(100, 100),
                                 jitter=(1.0, 1.0), think_head=True)
    d = ctl.player("A_white").decide(obs_at("A_white", {"A": START, "B": FEN_B_BLACK}))
    look, main = stub.calls
    # clock budget: 59 s / 100 / 5 ms = 118 nodes; the look is 16 nodes, charged to the same budget, no sample
    assert look["nodes"] == 16 and not look.get("sample") and look["options"]["TreeMoveMs"] == 590
    assert main["nodes"] == total - 16 and main["sample"] is True
    assert d.nodes == total and d.think_ms == total * 5 and main["options"]["TreeMoveMs"] == total * 5
    assert ctl.quick_looks == 1 and ctl.think_head_missing == 0 and ctl.think_gain_sum == pytest.approx(p)


def test_think_head_uses_the_seats_board_and_falls_back_without_head():
    stub = SampleStub(SearchResult(None, "e7e5", 0, 0, None, 0, 0, think_gain=(0.0, 1.0)))
    ctl = HivemindTeamController("T1", stub, ms_per_node=5.0, free=True, k_range=(100, 100), jitter=(1.0, 1.0),
                                 think_head=True)
    d = ctl.player("B_black").decide(obs_at("B_black", {"A": START, "B": FEN_B_BLACK}))
    assert d.nodes == 2 * 112  # B_black's 56 s / 100 / 5 = 112 nodes, p = 1 on board B
    stub2 = SampleStub(SearchResult("e2e4", None, 0, 0, None, 0, 0))  # no think_gain line: stock-like network
    ctl2 = HivemindTeamController("T1", stub2, ms_per_node=5.0, free=True, k_range=(100, 100), jitter=(1.0, 1.0),
                                  think_head=True)
    d2 = ctl2.player("A_white").decide(obs_at("A_white", {"A": START, "B": FEN_B_BLACK}))
    assert d2.nodes == 118 and stub2.calls[1]["nodes"] == 118 - 16 and ctl2.think_head_missing == 1


def test_think_head_off_by_default_and_needs_free():
    stub = SampleStub(SearchResult("e2e4", None, 0, 0, None, 0, 0, think_gain=(1.0, 1.0)))
    ctl = HivemindTeamController("T1", stub, ms_per_node=5.0, free=True, k_range=(100, 100), jitter=(1.0, 1.0))
    assert ctl.player("A_white").decide(obs_at("A_white", {"A": START, "B": FEN_B_BLACK})).nodes == 118
    assert len(stub.calls) == 1
    with pytest.raises(ValueError, match="think_head"):
        HivemindTeamController("T1", stub, think_head=True)


def test_think_head_respects_node_clamp():
    stub = SampleStub(SearchResult("e2e4", None, 0, 0, None, 0, 0, think_gain=(1.0, 1.0)))
    ctl = HivemindTeamController("T1", stub, ms_per_node=5.0, free=True, k_range=(100, 100), jitter=(1.0, 1.0),
                                 think_head=True, max_nodes=150)
    assert ctl.player("A_white").decide(obs_at("A_white", {"A": START, "B": FEN_B_BLACK})).nodes == 150


# ---------------------------------------------------------------------------------------------- D2: repeats

def test_repeated_position_keeps_one_sample_per_seat():
    stub = SampleStub(SearchResult(None, None, 0, 0, None, 0, 0))  # always passes
    ctl = HivemindTeamController("T1", stub, ms_per_node=5.0, free=True, record=True, k_range=(100, 100),
                                 jitter=(1.0, 1.0))
    fens = {"A": START, "B": START}
    for i in range(4):  # sitting: the same position, only the clocks move
        ctl.player("A_white").decide(obs_at("A_white", fens, now_ms=1000 + 600 * i))
    assert len(ctl.samples) == 1 and ctl.repeats_dropped == 3
    assert [c["sample"] for c in stub.calls] == [True, False, False, False]
    moved = {"A": START, "B": FEN_B_BLACK}
    ctl.player("A_white").decide(obs_at("A_white", moved, now_ms=4000))  # the other board changed
    ctl.player("A_white").decide(obs_at("A_white", fens, now_ms=4600))  # back again: not "in a row" any more
    assert len(ctl.samples) == 3 and ctl.repeats_dropped == 3
    ctl.player("B_black").decide(obs_at("B_black", moved, now_ms=5000))  # per seat
    assert len(ctl.samples) == 4


# ---------------------------------------------------------------------------------------------- HVM6 block

def test_trailer_layout_is_exactly_90_bytes():
    assert TRAILER.format == "<ffBB20f" and TRAILER.size == 90
    arr = tuple(float(i) for i in range(20))
    blk = encode_trailer((1.0, None), arr)
    assert blk[:10] == struct.pack("<ffBB", 1.0, 0.0, 1, 0) and blk[10:] == struct.pack("<20f", *arr)
    assert decode_trailer(blk) == ((1.0, 0.0), (1, 0), arr)
    tg, mask, nan = decode_trailer(encode_trailer())
    assert tg == (0.0, 0.0) and mask == (0, 0) and all(math.isnan(x) for x in nan)
    with pytest.raises(ValueError):
        encode_trailer(None, (1.0,) * 19)


@pytest.mark.parametrize("channels", [74, 82, 86])
def test_v6_sample_parse_and_infer(channels):
    raw = fake_sample(1, channels)
    full = raw + encode_trailer((0.0, 1.0), [2.0] + [0.0] * 19)
    assert infer_format(raw) == (channels, 5) and infer_format(full) == (channels, 6)
    assert sample_size(full, channels, version=6) == len(raw) + 90 == len(full)
    v = view_sample(full)
    assert v.channels == channels and v.think_gain == (0.0, 1.0) and v.think_mask == (1, 1) and v.arrivals[0] == 2
    assert base_sample(full, channels, 6) == raw and view_sample(raw).arrivals is None


def _events_with_captures():
    """move/pass event dicts: three moves; captures at moves 0 (B_black takes, A_white receives a knight at 1000)
    and 2 (A_black takes a promoted piece -> pawn to B_white at 4000), and a queen to A_black at 9000 (move 3)."""
    return [
        {"type": "game_start", "ts": 0},
        {"type": "move", "ts": 1000, "board": "B", "seat": "B_black", "capture": "N"},
        {"type": "pass", "ts": 1000, "fromSeat": "B_black", "toSeat": "A_white", "piece": "N"},
        {"type": "move", "ts": 2000, "board": "A", "seat": "A_white", "capture": None},
        {"type": "move", "ts": 4000, "board": "A", "seat": "A_black", "capture": "p"},
        {"type": "pass", "ts": 4000, "fromSeat": "A_black", "toSeat": "B_white", "piece": "p"},
        {"type": "move", "ts": 9000, "board": "B", "seat": "B_white", "capture": "q"},
        {"type": "pass", "ts": 9000, "fromSeat": "B_white", "toSeat": "A_black", "piece": "q"},
        {"type": "game_end", "ts": 9500},
    ]


def test_pocket_order_is_team_relative():
    assert [pocket_index("T1", s) for s in ("A_white", "A_black", "B_black", "B_white")] == [0, 1, 2, 3]
    assert [pocket_index("T2", s) for s in ("A_black", "A_white", "B_white", "B_black")] == [0, 1, 2, 3]


def test_arrival_counts_window_and_ply():
    caps = captures(_events_with_captures())
    assert caps == [(0, 1000, "A_white", "N"), (2, 4000, "B_white", "p"), (3, 9000, "A_black", "q")]
    t1 = arrival_counts(caps, "T1", ply=0, start_ms=500)  # window to 5500: N to ours-A, pawn to theirs-B
    assert t1[0 * 5 + 1] == 1 and t1[3 * 5 + 0] == 1 and sum(t1) == 2
    t2 = arrival_counts(caps, "T2", ply=0, start_ms=500)  # the same pieces, seen from T2
    assert t2[1 * 5 + 1] == 1 and t2[2 * 5 + 0] == 1 and sum(t2) == 2
    # a search that started after move 0 landed (ply 1) does not count it, even at the same ms
    assert sum(arrival_counts(caps, "T1", ply=1, start_ms=1000)) == 1
    late = arrival_counts(caps, "T2", ply=3, start_ms=4500)  # queen at 9000 is within 4500 + 5000
    assert late[0 * 5 + 4] == 1 and sum(late) == 1
    assert sum(arrival_counts(caps, "T2", ply=2, start_ms=3999)) == 1  # 9000 > 8999: only the pawn


def test_finish_game_v6_moves_left_and_arrivals():
    ev = _events_with_captures()
    recs = [RecordedSample(500, 1, fake_sample(0), seat="A_white", team="T1", ply=0, fen_a="a", fen_b="b"),
            RecordedSample(1500, 2, fake_sample(1), seat="A_black", team="T2", ply=1, fen_a="a2", fen_b="b"),
            RecordedSample(9200, 3, fake_sample(1), seat="A_black", team="T2", ply=4, fen_a="a3", fen_b="b")]
    out, metas = finish_game(recs, "T2", 9, ev, version=6, with_meta=True)
    vs = [view_sample(b, version=6) for b in out]
    assert [v.moves_left for v in vs] == [4, 3, 0]  # 4 moves in the game minus moves landed before each search
    assert [v.outcome for v in vs] == [-1, 1, 1] and all(v.think_mask == (0, 0) for v in vs)
    assert vs[0].arrivals[1] == 1 and sum(vs[0].arrivals) == 2
    assert sum(vs[1].arrivals) == 1 and vs[1].arrivals[2 * 5 + 0] == 1  # T2: pawn to B_white = ours on B
    assert sum(vs[2].arrivals) == 0
    assert [m["fenA"] for m in metas] == ["a", "a2", "a3"] and metas[0]["gameId"] == 9
    # v5 output of the same records: the HVM5 bytes only, movesLeft still from moves
    v5 = [view_sample(b) for b in finish_game(recs, "T2", 9, ev)]
    assert [v.moves_left for v in v5] == [4, 3, 0] and all(v.arrivals is None for v in v5)


def test_v6_chunk_round_trip_with_sidecar(tmp_path):
    recs = [RecordedSample(100 * i, i, fake_sample(i % 2, 86), seat="A_white", team="T1" if i % 2 == 0 else "T2",
                           ply=i, fen_a=f"fa{i}", fen_b="fb", clocks=dict(CLOCKS), elapsed={"A": i, "B": 0},
                           tree_ms=300, nodes=50) for i in range(5)]
    out, metas = finish_game(recs, "T1", 3, _events_with_captures(), version=6, with_meta=True)
    w = ShardWriter(str(tmp_path), "r", samples_per_chunk=3, version=6)
    w.append(out, metas)
    w.close()
    assert len(w.paths) == 2 and w.channels == 86
    with open(w.paths[0], "rb") as f:
        assert HEADER.unpack(f.read(HEADER.size)) == (b"HVM6", 6, 86, 4672, 3)
    got, side = [], []
    for p in w.paths:
        ch, bufs, ver = read_chunk(p, with_version=True)
        assert (ch, ver) == (86, 6)
        got += bufs
        side += read_sidecar(p)
    assert got == out and [m["fenA"] for m in side] == [f"fa{i}" for i in range(5)]
    assert side[2]["elapsed"] == {"A": 2, "B": 0} and side[0]["treeMs"] == 300
    with pytest.raises(ValueError):  # a v6 writer refuses v5 samples
        ShardWriter(str(tmp_path / "x"), "r", version=6).append([fake_sample(0, 86)])


def test_v6_chunk_through_hivemind_reader_when_supported(tmp_path):
    conv = pytest.importorskip("hivemind.data.convert_selfplay_data")
    recs = [RecordedSample(i, i, fake_sample(0, 82), team="T1", ply=0) for i in range(3)]
    out = finish_game(recs, "T1", 5, _events_with_captures(), version=6)
    out[1] = base_sample(out[1], 82, 6) + encode_trailer((1.0, None), view_sample(out[1], version=6).arrivals)
    w = ShardWriter(str(tmp_path), "r", version=6)
    w.append(out)
    w.close()
    try:
        got = conv.read_binary_shard(w.paths[0])
    except ValueError as exc:
        pytest.skip(f"hivemind reader does not read HVM6 yet: {exc}")
    assert len(got) == 3 and all(g["x"] == view_sample(b, version=6).planes for g, b in zip(got, out))
    if "think_gain_a" in got[1]:
        assert got[1]["think_gain_a"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------------------------- think_labels.py

sys.path.insert(0, os.path.join(REPO, "tools"))


class LabelStub:
    """Chooses a move from the node count: <= 100 nodes plays e2e4 on A and pass on B, more plays d2d4 / pass."""

    def __init__(self):
        self.calls = []

    def search(self, fen_a, fen_b, *, team, nodes, **kw):
        self.calls.append(dict(kw, team=team, nodes=nodes, fens=(fen_a, fen_b)))
        return SearchResult("e2e4" if nodes <= 100 else "d2d4", None, 0, 0, None, nodes, 1)


def test_think_labels_end_to_end_on_stub_selfplay(tmp_path):
    import think_labels
    from arena.selfplay import parse_args, run

    out = tmp_path / "sp"
    args = parse_args(["--games", "1", "--out", str(out), "--base-ms", "10000", "--max-plies", "40", "--seed", "2"])
    run(args, process_factory=lambda team: LegalStub(7 if team == "T1" else 8))
    src = out / "training_data"
    dst = tmp_path / "think"
    stub = LabelStub()
    targs = think_labels.parse_args(["--in", str(src), "--out", str(dst), "--nodes", "100", "--fraction", "1",
                                     "--repeat-fraction", "0"])
    report = think_labels.run(targs, process=stub)
    s = report["stats"]
    before = [b for c in sorted(src.glob("*.hvm")) for b in read_chunk(str(c))[1]]
    metas = [m for c in sorted(src.glob("*.hvm")) for m in read_sidecar(str(c))]
    after = [b for c in sorted(dst.glob("*.hvm")) for b in read_chunk(str(c))[1]]
    assert s["labelled"] == len(before) == len(after) and len(stub.calls) == 2 * len(before)
    assert {c["nodes"] for c in stub.calls} == {100, 400}
    assert all(c["options"]["TreeMoveMs"] == m["treeMs"] for c, m in zip(stub.calls[::2], metas))
    for b0, b1, m in zip(before, after, metas):
        v0, v1 = view_sample(b0, version=6), view_sample(b1, version=6)
        assert base_sample(b0, 82, 6) == base_sample(b1, 82, 6)
        assert v0.arrivals == v1.arrivals  # arrival labels copied
        on_turn = think_labels.teams_on_turn(m["fenA"], m["fenB"])
        assert v1.think_mask == (int(on_turn["A"] == m["team"]), int(on_turn["B"] == m["team"]))
        if v1.think_mask[0]:
            assert v1.think_gain[0] == 1.0  # e2e4 at 100 nodes, d2d4 at 400: changed
        if v1.think_mask[1]:
            assert v1.think_gain[1] == 0.0  # pass both times: unchanged
    assert s["n_a"] + s["n_b"] > 0 and (dst / "think_labels.json").exists()
    assert sorted(p.name for p in dst.glob("*.pos.jsonl")) == sorted(p.name for p in src.glob("*.pos.jsonl"))


def test_think_labels_defaults_and_repeat_sample(tmp_path):
    import think_labels
    from arena.selfplay import parse_args, run

    defaults = think_labels.parse_args(["--in", "x", "--out", "y"])
    assert defaults.nodes >= 200 and defaults.repeat_fraction > 0  # N below ~200 is mostly search noise
    out = tmp_path / "sp"
    run(parse_args(["--games", "1", "--out", str(out), "--base-ms", "10000", "--max-plies", "40", "--seed", "2"]),
        process_factory=lambda team: LegalStub(7 if team == "T1" else 8))
    stub = LabelStub()
    report = think_labels.run(think_labels.parse_args(
        ["--in", str(out / "training_data"), "--out", str(tmp_path / "t"), "--nodes", "100", "--fraction", "1",
         "--repeat-fraction", "0.5", "--seed", "3"]), process=stub)
    s = report["stats"]
    repeats = len(stub.calls) - 2 * s["labelled"]  # every labelled position: N and 4N; the sample: one more N
    assert 0 < repeats < s["labelled"] and s["noise_n"] > 0 and s["noise_rate"] == 0.0  # the stub is deterministic
    full = think_labels.run(think_labels.parse_args(
        ["--in", str(out / "training_data"), "--out", str(tmp_path / "f"), "--nodes", "100", "--fraction", "1",
         "--repeat-check"]), process=LabelStub())
    assert full["stats"]["noise_n"] >= s["noise_n"]


def test_think_labels_masks_no_root_and_converts_v5(tmp_path):
    import think_labels

    class NoRoot:
        def search(self, *a, **kw):
            return HivemindProcess._parse([], "bestmove (none)", 1)

    meta = {"fenA": START, "fenB": FEN_B_BLACK, "team": "T1", "clocks": CLOCKS, "treeMs": 100}
    assert think_labels.label_position(NoRoot(), meta, 10) == (None, None)
    assert think_labels.label_position(LabelStub(), {**meta, "team": "T2"}, 10) == (None, None)  # T2 not on turn
    # an HVM5 chunk without a sidecar comes out as HVM6, all masked, arrivals NaN
    src = tmp_path / "v5"
    w = ShardWriter(str(src), "old")
    w.append([fake_sample(0), fake_sample(1)])
    w.close()
    think_labels.run(think_labels.parse_args(["--in", str(src), "--out", str(tmp_path / "o")]), process=LabelStub())
    ch, bufs, ver = read_chunk(str(next((tmp_path / "o").glob("*.hvm"))), with_version=True)
    assert ver == 6 and len(bufs) == 2
    v = view_sample(bufs[0], version=6)
    assert v.think_mask == (0, 0) and all(math.isnan(x) for x in v.arrivals)


def test_hivemind_process_sends_elapsed_with_stub_engine(tmp_path):
    """A stub UCI engine (a Python script) echoes the position line back; HivemindProcess must send the elapsed
    suffix and parse a think_gain line."""
    script = tmp_path / "engine.py"
    script.write_text(
        "import sys\n"
        "pos = ''\n"
        "for line in sys.stdin:\n"
        "    line = line.strip()\n"
        "    if line == 'uci': print('uciok', flush=True)\n"
        "    elif line == 'isready': print('readyok', flush=True)\n"
        "    elif line.startswith('position'): pos = line\n"
        "    elif line.startswith('go'):\n"
        "        print('info string echo ' + pos, flush=True)\n"
        "        print('info string think_gain 0.2 0.6', flush=True)\n"
        "        print('info depth 1 score cp 5 nodes 16 nps 1 time 1 pv (e2e4,pass)', flush=True)\n"
        "        print('bestmove (e2e4,pass)', flush=True)\n"
        "    elif line == 'quit': break\n")
    exe = tmp_path / "engine"
    exe.write_text(f"#!/bin/sh\nexec {sys.executable} {script} \"$@\"\n")
    exe.chmod(0o755)
    model = tmp_path / "m.onnx"
    model.write_text("x")
    proc = HivemindProcess(str(exe), str(model), timeout_s=10)
    try:
        r = proc.search(START, FEN_B_BLACK, team="T1", nodes=16, clocks=CLOCKS, elapsed={"A": 1500, "B": 30},
                        require_option=False, options={"FreePass": True})
    finally:
        proc.close()
    echo = next(l for l in r.lines if l.startswith("info string echo"))
    assert echo.endswith("clocks 59000 58000 57000 56000 elapsed 1500 30")
    assert r.think_gain == (0.2, 0.6) and r.move_a == "e2e4"


def test_setups_pass_think_head_and_refuse_split():
    from arena.players import make_players

    ps = make_players("centralized", "hivemind", free=True, think_head=True, quick_nodes=8, send_elapsed=False,
                      binary="hivemind-clock")
    try:
        ctl = ps["A_white"].controller
        assert ctl.think_head and ctl.quick_nodes == 8 and not ctl.send_elapsed
    finally:
        for p in ps.values():
            p.close()
    with pytest.raises(ValueError, match="centralized"):
        make_players("silent", "hivemind", think_head=True)
