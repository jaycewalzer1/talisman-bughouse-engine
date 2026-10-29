"""Tests for arena.probe_wait: replaying a recorded game to a fork point and probing forced waits."""

from __future__ import annotations

import json

from arena.interfaces import Decision, Observation, SEATS
from arena.openings import paired_schedule
from arena.probe_wait import (
    ForcedWait,
    candidate_points,
    main,
    own_pocket,
    replay,
    summarize,
    team_setups,
)
from arena.run import play_game


def record(tmp_path, t1="sharing", t2="silent", seed=3):
    pair, _ = next(iter(paired_schedule(2, seed)))
    spec = {"game_id": "g0001", "t1_setup": t1, "t2_setup": t2, "kind": "heuristic", "seed": seed,
            "nodes": 200, "base_ms": 60000, "increment_ms": 0, "fens": pair.fens, "channel_delay_ms": 0,
            "max_plies": 400, "charge_wall_time": False}
    events, _ = play_game(spec)
    path = tmp_path / "game_0001.jsonl"
    path.write_text("".join(json.dumps(e) + "\n" for e in events))
    return path, events


def test_replay_matches_recorded_clocks_and_positions(tmp_path):
    _, events = record(tmp_path)
    moves = [e for e in events if e["type"] == "move"]
    last = moves[len(moves) // 2]
    state = replay(events, last["seq"] + 1)  # replay itself checks every recorded move's clocks
    assert state.fen(last["board"]) == last["fen"]
    assert state.last_ms == events[last["seq"] + 1]["ts"]
    assert team_setups(events[0]) == ("sharing", "silent")


def test_candidate_points_follow_need_messages(tmp_path):
    for seed in range(20):  # find a recorded game where a seat asked for a piece and then moved again
        _, events = record(tmp_path, seed=seed)
        need_pts = candidate_points(events, need_window_ms=60000, any_point=False, min_ply=0)
        if need_pts:
            break
    any_pts = candidate_points(events, need_window_ms=60000, any_point=True, min_ply=0)
    assert need_pts and set(need_pts) <= set(any_pts)
    for seq in need_pts:
        e = events[seq]
        assert any(m["type"] == "message" and m["token"].startswith("need_") and m["fromSeat"] == e["seat"]
                   and 0 <= e["ts"] - m["ts"] <= 60000 for m in events[:seq])


class Mover:
    seat, name, kind = "A_white", "mover", "test"

    def decide(self, obs):
        return Decision(kind="move", uci=obs.legal_moves[0], think_ms=10)

    def close(self):
        pass


def obs(now, fen_a="8/8/8/8/8/8/8/K6k[] w - - 0 1"):
    return Observation(now_ms=now, seat="A_white", fens={"A": fen_a, "B": fen_a}, clocks_ms={s: 1 for s in SEATS},
                       to_move={"A": "A_white", "B": "B_white"}, legal_moves=["a1a2"])


def test_forced_wait_sits_until_deadline_or_piece():
    fw = ForcedWait(Mover(), until_ms=3000, pocket="")
    d = fw.decide(obs(1000))
    assert d.kind == "wait" and d.wait_ms == 2000 and d.stall_reason == "probe_wait"
    assert fw.decide(obs(3000)).kind == "move"
    fw = ForcedWait(Mover(), until_ms=3000, pocket="")
    assert fw.decide(obs(1500, "8/8/8/8/8/8/8/K6k[Nq] w - - 0 1")).kind == "move"  # a knight arrived
    assert own_pocket("8/8/8/8/8/8/8/K6k[NqP] w - - 0 1", "A_white") == "NP"
    assert own_pocket("8/8/8/8/8/8/8/K6k[NqP] w - - 0 1", "A_black") == "q"


def test_summarize_differences():
    rows = [{"file": "g.jsonl", "seq": 5, "branch": "play", "score": 0.0},
            {"file": "g.jsonl", "seq": 5, "branch": "wait_2000", "score": 1.0},
            {"file": "g.jsonl", "seq": 9, "branch": "play", "score": 1.0},
            {"file": "g.jsonl", "seq": 9, "branch": "wait_2000", "score": 1.0}]
    s = summarize(rows, [2000], n_boot=200)
    o = s["overall"]["wait_2000"]
    assert o["positions"] == 2 and o["mean_diff"] == 0.5 and o["wait_better"] == 1 and o["play_better"] == 0


def test_probe_end_to_end_heuristic(tmp_path):
    record(tmp_path)
    out = tmp_path / "probe"
    assert main([str(tmp_path), "--kind", "heuristic", "--any", "--positions", "2", "--rollouts", "1",
                 "--waits", "1500", "--boot", "50", "--out", str(out)]) == 0
    rows = [json.loads(l) for l in (out / "rollouts.jsonl").read_text().splitlines()]
    assert len(rows) == 4 and {r["branch"] for r in rows} == {"play", "wait_1500"}
    assert all(r["end_ts"] >= r["fork_ts"] for r in rows)
    summary = json.loads((out / "summary.json").read_text())
    assert summary["overall"]["wait_1500"]["positions"] == 2
