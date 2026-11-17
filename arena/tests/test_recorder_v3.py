"""Recorder, spec v3 WP-D: sidecar fields, arrivals end-of-game mask, pace-scaled window, one movesLeft definition,
aborted games kept out of training."""

from __future__ import annotations

import math

from arena.recorder import (
    ARRIVAL_WINDOW_MAX_MS,
    ARRIVAL_WINDOW_MS,
    RecordedSample,
    arrival_window_ms,
    finish_game,
    game_end_info,
    opening_key,
    view_sample,
)
from arena.tests.test_custom_engine import fake_sample
from arena.tests.test_custom_engine_v2 import _events_with_captures


def _events(reason="checkmate", end_ts=9500):
    ev = [dict(e) for e in _events_with_captures()]
    ev[0]["fens"] = {"A": "fa", "B": "fb"}
    ev[-1].update({"reason": reason, "ts": end_ts})
    return ev


def _recs():
    return [RecordedSample(500, 1, fake_sample(0), seat="A_white", team="T1", ply=0),
            RecordedSample(4600, 2, fake_sample(1), seat="A_black", team="T2", ply=2),
            RecordedSample(9200, 3, fake_sample(1), seat="A_black", team="T2", ply=4)]


def test_game_end_info_and_opening_key():
    assert game_end_info(_events("flag", 9500)) == ("flag", 9500)
    assert game_end_info([]) == (None, None)
    assert opening_key(_events()) == "fa|fb"


def test_arrivals_masked_when_window_passes_game_end():
    out, metas = finish_game(_recs(), "T2", 9, _events(end_ts=9500), version=6, with_meta=True,
                             mask_arrivals_past_end=True)
    vs = [view_sample(b, version=6) for b in out]
    assert not any(math.isnan(x) for x in vs[0].arrivals)  # 500 + 5000 <= 9500
    assert all(math.isnan(x) for x in vs[1].arrivals)  # 4600 + 5000 > 9500: label truncated by the end
    assert all(math.isnan(x) for x in vs[2].arrivals)
    assert [m["gameEndMs"] for m in metas] == [9500] * 3 and metas[0]["arrivalWindowMs"] == ARRIVAL_WINDOW_MS
    # off (the default): the same bytes as before spec v3
    plain = finish_game(_recs(), "T2", 9, _events(end_ts=9500), version=6)
    assert not any(math.isnan(x) for x in view_sample(plain[2], version=6).arrivals)


def test_pace_scaled_window():
    r = RecordedSample(0, 0, b"", pace=(200.0, 300.0, 400.0, 300.0))
    assert arrival_window_ms(r, "pace") == 3000  # 10 x mean pace 300 ms
    assert arrival_window_ms(r, "fixed") == ARRIVAL_WINDOW_MS
    assert arrival_window_ms(RecordedSample(0, 0, b""), "pace") == ARRIVAL_WINDOW_MS  # no pace: fixed
    assert arrival_window_ms(RecordedSample(0, 0, b"", pace=(9e9,) * 4), "pace") == ARRIVAL_WINDOW_MAX_MS
    # a 3 s window at 500 ms does not reach the knight... it lands at 1000, inside; the pawn at 4000 is outside
    recs = [RecordedSample(500, 1, fake_sample(0), seat="A_white", team="T2", ply=0, pace=(300.0,) * 4)]
    out = finish_game(recs, "T1", 1, _events(), version=6, arrival_window="pace")
    arr = view_sample(out[0], version=6).arrivals
    assert sum(arr) == 1  # only the knight (to A_white = theirs on A for T2)


def test_moves_left_is_moves_after_the_search_started():
    out = finish_game(_recs(), "T1", 9, _events())
    assert [view_sample(b).moves_left for b in out] == [4, 2, 0]  # 4 moves in the game minus ply
    # bare tuples (no positions): the stock count of samples to the end
    bare = finish_game([(0, 0, fake_sample(0)), (1, 1, fake_sample(1))], "T1", 9)
    assert [view_sample(b).moves_left for b in bare] == [2, 1]


def test_sidecar_carries_v3_fields_and_reason():
    rec = RecordedSample(500, 1, fake_sample(0), seat="A_white", team="T1", ply=0, nodes_requested=100,
                         nodes_searched=146, sit_ms=0.0, start_offset_ms=-2000.0, pace=(300, 300, 250, 250),
                         clocks_ms=(59999.5, 60000, 60000, 58000.25), elapsed_ms=(499.5, 0))
    _, metas = finish_game([rec], "T1", 4, _events("flag"), version=6, with_meta=True, opening_id=1007)
    m = metas[0]
    assert m["nodes_requested"] == 100 and m["nodes_searched"] == 146 and m["start_offset_ms"] == -2000.0
    assert m["clocks_ms"] == [59999.5, 60000.0, 60000.0, 58000.25] and m["elapsed_ms"] == [499.5, 0.0]
    assert m["pace"] == [300.0, 300.0, 250.0, 250.0] and m["reason"] == "flag" and m["opening"] == "1007"
    # fields not recorded stay out of the sidecar
    _, metas = finish_game(_recs()[:1], "T1", 4, _events(), version=6, with_meta=True)
    assert "clocks_ms" not in metas[0] and metas[0]["opening"] == "fa|fb"


def test_aborted_and_max_plies_games_dropped_on_request():
    for reason in ("aborted", "max_plies"):
        assert finish_game(_recs(), "draw", 1, _events(reason), version=6, with_meta=True,
                           drop_non_training=True) == ([], [])
    assert len(finish_game(_recs(), "T1", 1, _events("flag"), drop_non_training=True)) == 3
