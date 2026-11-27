"""tools/hoarding_diagnosis.py (spec v7 4.3) on two tiny synthetic game logs."""

from __future__ import annotations

import json

import pytest

from tools import hoarding_diagnosis as hd

START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR[] w KQkq - 0 1"
C = {"A_white": 60_000, "A_black": 60_000, "B_white": 60_000, "B_black": 60_000}


def _start():
    seats = {s: {"name": f"Wave2-{s}-k110", "team": hd.TEAM_OF[s], "kind": "wave2"} for s in hd.SEATS}
    return {"type": "game_start", "ts": 0, "seats": seats, "fens": {"A": START, "B": START}, "clocks": dict(C)}


def _think(seat, ts, ev):
    return [{"type": "thinking", "ts": ts - 100, "seat": seat, "phase": "start", "think_ms": 100, "nodes": 10},
            {"type": "thinking", "ts": ts, "seat": seat, "phase": "end", "think_ms": 100, "nodes": 10, "eval": ev}]


def _move(seat, ts, uci, fen, drop=False):
    return {"type": "move", "ts": ts, "board": seat[0], "seat": seat, "uci": uci, "san": uci, "isDrop": drop,
            "capture": None, "fen": fen, "clocks": dict(C)}


def game_one():
    """Contender T1 wins by mate on A_black. B_black gets a pawn at 150 and drops it at 300; A_black gets a knight at
    400 and never drops it (it moves with it in hand)."""
    return [_start(),
            *_think("A_white", 100, 0.2),
            _move("A_white", 100, "e2e4", "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR[] b KQkq - 0 1"),
            {"type": "pass", "ts": 150, "fromBoard": "A", "fromSeat": "A_white", "toSeat": "B_black", "piece": "P"},
            *_think("B_white", 200, 0.0),
            _move("B_white", 200, "e2e4", "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR[p] b KQkq - 0 1"),
            *_think("B_black", 300, 0.5),
            _move("B_black", 300, "P@e5", "rnbqkbnr/pppppppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR[] w KQkq - 0 2",
                  drop=True),
            {"type": "pass", "ts": 400, "fromBoard": "B", "fromSeat": "B_white", "toSeat": "A_black", "piece": "N"},
            *_think("A_black", 500, -0.3),
            _move("A_black", 500, "e7e5", "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR[n] w KQkq - 0 2"),
            {"type": "game_end", "ts": 600, "result": "T1", "reason": "checkmate", "board": "A",
             "loserSeat": "A_black"}]


def game_two():
    """Contender T2 loses: A_white gets a queen and mates A_black with a queen drop."""
    return [_start(),
            {"type": "pass", "ts": 50, "fromBoard": "B", "fromSeat": "B_black", "toSeat": "A_white", "piece": "Q"},
            *_think("A_white", 200, 0.9),
            _move("A_white", 200, "Q@h5", "rnbqkbnr/pppppppp/8/7Q/8/8/PPPPPPPP/RNBQKBNR[] b KQkq - 0 1", drop=True),
            {"type": "game_end", "ts": 200, "result": "T1", "reason": "checkmate", "board": "A",
             "loserSeat": "A_black"}]


@pytest.fixture
def match(tmp_path):
    part = tmp_path / "part1"
    part.mkdir()
    rows = []
    for i, (events, ct) in enumerate(((game_one(), "T1"), (game_two(), "T2")), 1):
        name = f"game_{i:04d}.jsonl"
        (part / name).write_text("".join(json.dumps(e) + "\n" for e in events))
        rows.append({"game_id": f"p{i}", "contender_team": ct, "file": name})
    (part / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return tmp_path


def _row(rows, **kw):
    return next(r for r in rows if all(r[k] == v for k, v in kw.items()))


def test_helpers():
    assert hd.fen_pockets("8/8/8/8/8/8/8/8[QNp] w - - 0 1") == (
        {"P": 0, "N": 1, "B": 0, "R": 0, "Q": 1}, {"P": 1, "N": 0, "B": 0, "R": 0, "Q": 0})
    assert [hd.pocket_bin(v) for v in (0, 1, 4, 5, 9, 10, 30)] == ["0", "1-4", "1-4", "5-9", "5-9", "10+", "10+"]
    assert hd.on_board(START, True) == 39 and hd.on_board("8/8/8/8/8/8/8/4K2Q[] w - - 0 1", True) == 9


def test_one_game_records():
    g = hd.game_records(game_one(), "T1")
    assert g["outcome"] == {"us": 1, "them": -1} and g["mismatch"] == 0 and g["checked"] == 4
    assert [(m["seat"], m["drop"], m["pocket_before"]) for m in g["moves"]] == [
        ("A_white", False, 0), ("B_white", False, 0), ("B_black", True, 1), ("A_black", False, 3)]
    holds = sorted((h["seat"], h["piece"], h["ms"], h["moves"], h["dropped"]) for h in g["holds"])
    assert holds == [("A_black", "N", 200, 1, False), ("B_black", "P", 150, 1, True)]
    assert [(t["seat"], t["pocket"], t["eval"], t["next_drop"]) for t in g["thinks"]] == [
        ("A_white", 0, 0.2, False), ("B_white", 0, 0.0, False), ("B_black", 1, 0.5, True), ("A_black", 3, -0.3, False)]
    assert g["end"] == {"reason": "checkmate", "loser": "A_black", "loser_side": "them", "drop_mate": False,
                        "mated_pocket": 3, "partner_pocket": 0}
    assert g["balance"][-1]["us"] == (1, -3)  # our dropped pawn (the synthetic pass took it from nowhere)


def test_a_pocket_out_of_step_with_the_fen_is_counted_and_corrected():
    ev = game_one()
    i = next(i for i, e in enumerate(ev) if e["type"] == "pass")
    ev[i] = dict(ev[i], piece="N")  # the log says a knight landed; the next FEN on B shows a pawn
    g = hd.game_records(ev, "T1")
    assert g["mismatch"] == 1 and g["checked"] == 4
    assert [m["pocket_before"] for m in g["moves"]][2] == 1  # corrected from the FEN before B_black's drop


def test_tables_on_two_games(match):
    rep = hd.analyse(str(match))
    assert rep["n"] == 2 and rep["score"][0] == 0.5 and rep["mismatch"] == 0
    d = rep["drop"]
    assert _row(d, side="us", outcome="all")["rate"][0] == 1.0  # B_black's only move with a pocket dropped
    assert _row(d, side="them", outcome="all")["rate"][0] == 0.5  # A_black kept its knight; A_white dropped a queen
    assert _row(d, side="us", outcome="won")["games"] == 1 and _row(d, side="us", outcome="lost")["drops"] == 0
    h = rep["holding"]
    assert _row(h, side="them", piece="N")["never"][0] == 1.0 and _row(h, side="us", piece="P")["ms_median"][0] == 150
    assert _row(h, side="them", piece="Q")["ms_median"][0] == 150 and _row(h, side="them", piece="all")["landed"] == 2
    p = rep["pocket"]
    assert _row(p, side="us", outcome="won")["seat_pocket"][0] == 0.5  # A_white 0, B_black 1
    assert _row(p, side="them", outcome="all")["seat_pocket"][0] == pytest.approx((0 + 3 + 9) / 3)
    cal = rep["calibration"]
    assert _row(cal, side="us", bin="1-4")["gap"][0] == pytest.approx(0.5 - 1)
    e = {r["side"]: r for r in rep["endings"]}
    assert e["us"]["mated"][0] == 0.5 and e["them"]["mated"][0] == 0.5
    assert e["us"]["drop_mates_received"][0] == 0.5 and e["them"]["drop_mates_given"][0] == 0.5
    assert e["them"]["mated_seat_pocket"][0] == 3 and e["us"]["flagged"][0] == 0.0
    text = hd.render(rep, "it2", "it0")
    assert "**5. Endings**" in text and "| it2 | all |" in text


def test_network_table_and_cli(match, tmp_path):
    samples = [{"side": "us", "bin": "5-9", "value": 0.2, "drop_mass": 0.3, "pass": 0.0, "dropped": True,
                "result": -1, "game": 1},
               {"side": "us", "bin": "5-9", "value": 0.4, "drop_mass": 0.1, "pass": 0.0, "dropped": False,
                "result": 1, "game": 2}]
    row = hd.network_table(samples)[0]
    assert row["positions"] == 2 and row["value"][0] == pytest.approx(0.3) and row["dropped"][0] == 0.5
    out, js = tmp_path / "t.md", tmp_path / "t.json"
    assert hd.main([str(match), "--us", "it2", "--them", "it0", "--ref", f"self={match}", "--out", str(out),
                    "--json", str(js)]) == 0
    assert "**References**" in out.read_text() and json.loads(js.read_text())["match"]["n"] == 2
