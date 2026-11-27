"""Spec v7 controller counters on the game_end event and in game summaries (arena.run.controller_telemetry), as
cluster/gate2_match.sh's merge reads them (think_head_missing for our team)."""

from arena.run import V7_TELEMETRY_KEYS, controller_telemetry
from arena.stats import summarize_game


class FakeController:
    def __init__(self, team, think_head=False, opp_sit_pace="own_k", missing=0):
        self.team, self.think_head, self.opp_sit_pace, self.missing = team, think_head, opp_sit_pace, missing

    def telemetry(self):
        return {"searches": 9, "opp_sit_pace": self.opp_sit_pace, "opp_k_levels": {"median": 3}, "think_looks": 4,
                "think_head_missing": self.missing, "think_gain_sum": 1.0, "think_scale_sum": 4.0}


class FakeSeat:
    def __init__(self, controller):
        self.controller = controller


def test_defaults_add_nothing():
    ctl = FakeController("T1")
    assert controller_telemetry([FakeSeat(ctl), FakeSeat(ctl), object()]) == {}


def test_v7_controller_reports_once_per_team():
    a = FakeController("T1", think_head=True, missing=3)
    b = FakeController("T2", opp_sit_pace="observed")
    tel = controller_telemetry([FakeSeat(a), FakeSeat(a), FakeSeat(b), FakeSeat(b)])
    assert set(tel) == {"T1", "T2"}
    assert set(tel["T1"]) == set(V7_TELEMETRY_KEYS)
    assert tel["T1"]["think_head_missing"] == 3 and "searches" not in tel["T1"]


def test_summary_carries_it_only_when_present():
    end = {"type": "game_end", "seq": 1, "ts": 10, "result": "T1", "reason": "checkmate", "board": "A",
           "loserSeat": "A_black"}
    start = {"type": "game_start", "seq": 0, "ts": 0, "game_id": "g1", "setup": "x", "seats": {}}
    assert "controller_telemetry" not in summarize_game([start, end])
    tel = {"T2": {"think_head_missing": 2}}
    s = summarize_game([start, {**end, "controller_telemetry": tel}])
    assert s["controller_telemetry"] == tel
