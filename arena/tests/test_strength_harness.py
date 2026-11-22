"""tools/strength_harness.py: the event-log replay behind ``positions`` and the scoring arithmetic (no engine)."""

from __future__ import annotations

import json

from arena.state import BughouseState
from tools import strength_harness as h

START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR[] w KQkq - 0 1"


def _spec(kind_t1="wave2", kind_t2="upstream"):
    kw = {"clock_k": 110, "max_nodes": 100000, "serial": True, "charge": "searched", "ms_per_node": 0.2}
    return {"index": 1, "rules": "v3", "teams": {
        "T1": {"setup": "centralized", "kind": kind_t1, "nodes": 800,
               "player_kw": {**kw, "root_min_ms": 100, "root_lag_ms": 75, "durative_tree": False, "sit_gate": True}},
        "T2": {"setup": "centralized", "kind": kind_t2, "nodes": 800, "player_kw": dict(kw)}}}


class _Log:
    """Writes a consistent event log (fens and clocks from BughouseState, as the simulator emits them)."""

    def __init__(self):
        self.st = BughouseState(30000, fens={"A": START, "B": START})
        self.ev = [{"type": "game_start", "seq": 0, "ts": 0, "fens": {"A": START, "B": START},
                    "clocks": dict(self.st.clocks_ms), "base_ms": 30000, "increment_ms": 0}]

    def add(self, **e):
        e["seq"] = len(self.ev)
        self.ev.append(e)

    def move(self, seat, uci, t0, ts, nodes=300):
        self.add(type="thinking", ts=t0, seat=seat, phase="start", think_ms=ts - t0, nodes=nodes, eval=None)
        self.add(type="thinking", ts=ts, seat=seat, phase="end", think_ms=ts - t0, nodes=nodes, eval=0.1)
        mr = self.st.apply_move(seat, uci, ts)
        self.add(type="move", ts=ts, board=seat[0], seat=seat, uci=uci, capture=mr.capture, fen=mr.fen,
                 clocks={s: int(v) for s, v in self.st.clocks_ms.items()})

    def write(self, path, result="T1"):
        self.add(type="game_end", ts=self.ev[-1]["ts"], result=result, reason="checkmate")
        with open(path, "w") as f:
            for e in self.ev:
                f.write(json.dumps(e) + "\n")


def test_replay_finds_moves_passes_and_hidden_repasses(tmp_path):
    g = _Log()
    g.add(type="stall", ts=0, seat="B_white", active=True, reason="upstream_pass")  # T2 sits on B
    g.move("A_white", "e2e4", 0, 300)  # T1 moves on A; the landing wakes B_white, which passes again (no event)
    g.move("A_black", "e7e5", 300, 600)  # B_white still sits: another hidden re-pass
    g.move("B_white", "d2d4", 900, 1000)
    path = tmp_path / "game_0001.jsonl"
    g.write(path)
    decs = h.replay_game(str(path), _spec())
    played = [(d["seat"], d["played"], d["ts"], bool(d.get("hidden"))) for d in decs]
    assert ("B_white", "pass", 0, False) in played
    assert ("A_white", "e2e4", 0, False) in played
    assert ("B_white", "pass", 300, True) in played and ("B_white", "pass", 600, True) in played
    assert ("B_white", "d2d4", 900, False) in played
    first = next(d for d in decs if d["seat"] == "A_white")
    assert first["result"] == 1.0 and first["team"] == "T1" and first["played_by"] == "wave2"
    assert first["root_move_ms"] == max(first["budget_ms"], 100) + 75
    assert first["seat_options"]["SeatClocksMs"] == "30000 30000 30000 30000"
    late = next(d for d in decs if d["seat"] == "B_white" and d["ts"] == 900)
    assert late["flags"]["after_sit"] and h.sit_or_act({**late})
    assert late["result"] == 0.0


def test_replay_rejects_inconsistent_log(tmp_path):
    g = _Log()
    g.move("A_white", "e2e4", 0, 300)
    g.ev[-1]["clocks"]["A_white"] += 5
    path = tmp_path / "game_0001.jsonl"
    g.write(path)
    try:
        h.replay_game(str(path), _spec())
    except h.ReplayError as exc:
        assert "clock" in str(exc)
    else:
        raise AssertionError("a clock mismatch must be refused")


def test_enrich_scores_agreement_sits_and_outcome_losses():
    p = {"game": "g", "id": 0, "board": "A", "on_turn": ["A", "B"], "sit_or_act": True, "lead": 500,
         "played": "pass", "result": 1.0}
    r = {"move_a": None, "move_b": "e2e4", "q": 0.5}
    ref = {"move_a": "d2d4", "move_b": "e2e4"}
    row = h.enrich(p, r, ref)
    assert row["sits"] == 1.0 and row["agree_played"] == 1.0
    assert row["agree_board"] == 0.0 and row["agree_joint"] == 0.0 and row["agree_actsit"] == 0.0
    assert abs(row["brier"] - 0.0625) < 1e-9


def test_cluster_bootstrap_is_deterministic_and_brackets_the_estimate():
    rows = [{"game": f"g{i // 3}", "x": float(i % 2)} for i in range(60)]
    est, lo, hi = h._boot(rows, h._mean("x"), n_boot=300)
    assert lo <= est <= hi
    assert h._boot(rows, h._mean("x"), n_boot=300) == (est, lo, hi)


def test_report_hybrid_section_counts_triggers_kept_sits_and_the_triggered_subset():
    """Spec v6 4.1: trigger and kept-sit counts from the rows' Hybrid lines, agreement on the triggered subset beside
    nodur on the same ids, and how often the joint action differs from nodur (triggered / rest)."""
    pos = [{"id": i, "game": f"g{i}", "board": "A", "on_turn": ["A"], "sit_or_act": True, "lead": -300,
            "played": "e2e4", "result": 1.0} for i in range(4)]
    labels = [{"id": i, "nodes": 32000, "move_a": "e2e4", "move_b": None, "q": 0.1} for i in range(4)]
    fired = {"sit_board": "A", "event_ms": 200, "chose": "act", "choice_reason": "no sit chosen"}
    nod = [{"id": i, "config": "nodur", "nodes": 4000, "move_a": "e2e4", "move_b": None, "q": 0.1} for i in range(4)]
    hyb = [{"id": 0, "config": "hybrid", "nodes": 4000, "move_a": None, "move_b": None, "q": 0.2,
            "hybrid": {**fired, "chose": "sit A", "choice_reason": "margin"}},
           {"id": 1, "config": "hybrid", "nodes": 4000, "move_a": "e2e4", "move_b": None, "q": 0.1, "hybrid": fired},
           {"id": 2, "config": "hybrid", "nodes": 4000, "move_a": "d2d4", "move_b": None, "q": 0.1,
            "hybrid": {"sit_board": None, "reason": "gate allows"}},
           {"id": 3, "config": "hybrid", "nodes": 4000, "move_a": "e2e4", "move_b": None, "q": 0.1,
            "hybrid": {"sit_board": None, "reason": "both boards"}}]
    lines = h.report_lines(pos, labels, nod + hyb, None, n_boot=50)
    row = next(l for l in lines if l.startswith("| hybrid | 4000 | 4 | 2 (50%)"))
    cells = [c.strip() for c in row.strip("|").split("|")]
    assert cells[4] == "1" and cells[5] == "1"  # one kept sit, behind on clock
    assert cells[6].startswith("50%") and cells[7].startswith("100%")  # triggered joint agree: hybrid / nodur
    assert cells[10] == "1/2 / 1/2"  # differs from nodur: triggered 1 of 2, rest 1 of 2
    assert not any(l.startswith("| nodur | 4000 | 4 | 0") for l in lines)  # nodur has no Hybrid lines
