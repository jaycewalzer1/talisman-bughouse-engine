"""Tests for arena.openings, arena.stats, and arena.serve (replay mode)."""

from __future__ import annotations

import asyncio
import json
import os
import socket
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import chess.variant
import pytest

from arena.openings import OPENING_PAIRS, paired_schedule
from arena.stats import cluster_bootstrap_ci, compute_stats, mean_of, summarize_game

REPO = Path(__file__).resolve().parents[2]

FEN0 = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR[] w KQkq - 0 1"
CLOCKS = {"A_white": 60000, "A_black": 60000, "B_white": 60000, "B_black": 60000}

EVENTS = [
    {"type": "game_start", "seq": 0, "ts": 0, "game_id": "g0001", "setup": "T1=sharing,T2=silent",
     "seats": {s: {"name": s, "team": t, "kind": "heuristic"} for s, t in
               (("A_white", "T1"), ("A_black", "T2"), ("B_white", "T2"), ("B_black", "T1"))},
     "fens": {"A": FEN0, "B": FEN0}, "clocks": CLOCKS, "base_ms": 60000, "increment_ms": 0,
     "channel_delay_ms": 0},
    {"type": "thinking", "seq": 1, "ts": 0, "seat": "A_white", "phase": "start", "think_ms": 300, "nodes": 100},
    {"type": "thinking", "seq": 2, "ts": 0, "seat": "B_white", "phase": "start", "think_ms": 500, "nodes": 50},
    {"type": "thinking", "seq": 3, "ts": 300, "seat": "A_white", "phase": "end", "think_ms": 300, "nodes": 100},
    {"type": "move", "seq": 4, "ts": 300, "board": "A", "seat": "A_white", "uci": "e2e4", "san": "e4",
     "isDrop": False, "capture": None, "fen": FEN0, "clocks": CLOCKS},
    {"type": "thinking", "seq": 5, "ts": 500, "seat": "B_white", "phase": "end", "think_ms": 500, "nodes": 50},
    {"type": "move", "seq": 6, "ts": 500, "board": "B", "seat": "B_white", "uci": "d2d4", "san": "d4",
     "isDrop": False, "capture": None, "fen": FEN0, "clocks": CLOCKS},
    {"type": "message", "seq": 7, "ts": 600, "deliverTs": 600, "fromSeat": "B_black", "toSeat": "A_white",
     "token": "need_N", "text": "need a knight", "payload": {"piece": "N"}},
    {"type": "stall", "seq": 8, "ts": 600, "seat": "B_black", "active": True, "reason": "need_piece"},
    {"type": "thinking", "seq": 9, "ts": 700, "seat": "A_black", "phase": "end", "think_ms": 200, "nodes": 10},
    {"type": "move", "seq": 10, "ts": 700, "board": "A", "seat": "A_black", "uci": "d7d5", "san": "d5",
     "isDrop": False, "capture": None, "fen": FEN0, "clocks": CLOCKS},
    {"type": "game_end", "seq": 11, "ts": 60600, "result": "T2", "reason": "flag", "board": "B",
     "loserSeat": "B_black"},
]


# ---------------------------------------------------------------- openings

def test_opening_pairs_are_valid_bracket_fens():
    assert len(OPENING_PAIRS) >= 20
    for p in OPENING_PAIRS:
        for fen in (p.fen_a, p.fen_b):
            assert "[] w " in fen
            board = chess.variant.CrazyhouseBoard(fen)
            assert board.is_valid()
        assert p.fen_a != p.fen_b


def test_paired_schedule_plays_each_pair_twice_with_swap():
    sched = list(paired_schedule(20, seed=3))
    assert len(sched) == 20
    for k in range(0, 20, 2):
        (pa, sa), (pb, sb) = sched[k], sched[k + 1]
        assert pa == pb and sa is False and sb is True
    counts = Counter(p.id for p, _ in sched)
    assert set(counts.values()) == {2}  # 10 distinct pairs, each twice
    assert sched == list(paired_schedule(20, seed=3))  # deterministic
    assert [p.id for p, _ in sched] != [p.id for p, _ in paired_schedule(20, seed=4)]


def test_paired_schedule_wraps_after_all_pairs():
    n = 2 * len(OPENING_PAIRS) + 4
    counts = Counter(p.id for p, _ in paired_schedule(n, 0))
    assert sum(counts.values()) == n and len(counts) == len(OPENING_PAIRS)


# ---------------------------------------------------------------- summarize_game

def test_summarize_game_hand_written():
    s = summarize_game(EVENTS)
    assert s["game_id"] == "g0001" and s["result"] == "T2" and s["reason"] == "flag"
    assert s["flag"] is True and s["loser_seat"] == "B_black" and s["loser_team"] == "T1"
    assert s["plies"] == 3 and s["duration_ms"] == 60600
    t1, t2 = s["teams"]["T1"], s["teams"]["T2"]
    # only "end" thinking events count as decisions
    assert (t1["decisions"], t1["think_ms"], t1["nodes"]) == (1, 300, 100)
    assert (t2["decisions"], t2["think_ms"], t2["nodes"]) == (2, 700, 60)
    assert (t1["moves"], t2["moves"]) == (1, 2)
    assert (t1["messages"], t1["stalls"], t2["messages"]) == (1, 1, 0)


def test_compute_stats_from_summary():
    s = summarize_game(EVENTS)
    s.update(contender_team="T2", opening_id=0)
    st = compute_stats([s])
    m = st["metrics"]
    assert st["wins"] == 1 and m["score"]["est"] == 1.0
    assert m["contender_latency_ms"]["est"] == 350 and m["baseline_latency_ms"]["est"] == 300
    assert m["baseline_flagged"]["est"] == 1.0 and m["contender_flagged"]["est"] == 0.0
    assert m["contender_nodes_per_game"]["est"] == 60


# ---------------------------------------------------------------- bootstrap

def _synthetic(n_pairs: int, seed: int = 0) -> list[dict]:
    """Two games per opening pair with identical scores within a pair (strong cluster dependence)."""
    import random
    rng = random.Random(seed)
    rows = []
    for c in range(n_pairs):
        v = rng.choice([0.0, 1.0])
        rows += [{"cluster": c, "score": v}, {"cluster": c, "score": v}]
    return rows


def test_ci_contains_mean_and_is_reproducible():
    rows = _synthetic(15)
    ci = cluster_bootstrap_ci(rows, mean_of("score"), seed=1)
    assert ci["lo"] <= ci["est"] <= ci["hi"] and ci["lo"] < ci["hi"]
    assert ci["clusters"] == 15 and ci["n"] == 30
    assert ci == cluster_bootstrap_ci(rows, mean_of("score"), seed=1)


def test_cluster_bootstrap_is_wider_than_iid_when_games_are_correlated():
    rows = _synthetic(30)
    clustered = cluster_bootstrap_ci(rows, mean_of("score"), seed=0)
    for i, r in enumerate(rows):
        r["game"] = i
    iid = cluster_bootstrap_ci(rows, mean_of("score"), cluster_key="game", seed=0)
    w_c, w_i = clustered["hi"] - clustered["lo"], iid["hi"] - iid["lo"]
    assert w_c > 1.2 * w_i  # ~sqrt(2) wider for perfectly correlated pairs


def test_single_cluster_ci_is_degenerate():
    ci = cluster_bootstrap_ci([{"cluster": 0, "score": 1.0}], mean_of("score"))
    assert ci["lo"] == ci["est"] == ci["hi"] == 1.0


# ---------------------------------------------------------------- serve (replay)

def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


async def _receive_game(port: int, timeout: float = 15.0) -> list[dict]:
    from websockets.asyncio.client import connect

    deadline = time.monotonic() + timeout
    while True:
        try:
            ws = await connect(f"ws://127.0.0.1:{port}", origin="http://localhost:5173")
            break
        except OSError:
            if time.monotonic() > deadline:
                raise
            await asyncio.sleep(0.1)
    got: list[dict] = []
    async with ws:
        while True:
            ev = json.loads(await asyncio.wait_for(ws.recv(), timeout=timeout))
            if ev["type"] == "game_start":
                got = []  # joined mid-replay: start from the next full game
            got.append(ev)
            if ev["type"] == "game_end" and got[0]["type"] == "game_start":
                return got


def test_serve_replay_streams_events_in_order(tmp_path):
    path = tmp_path / "game_0001.jsonl"
    path.write_text("".join(json.dumps(e) + "\n" for e in EVENTS))
    port = _free_port()
    proc = subprocess.Popen(
        [sys.executable, "-m", "arena.serve", "--host", "127.0.0.1", "--port", str(port), "--replay", str(path),
         "--speed", "200", "--pause-s", "0.2"],
        cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env={**os.environ, "PYTHONUNBUFFERED": "1"})
    try:
        got = asyncio.run(_receive_game(port))
    finally:
        proc.terminate()
        out, _ = proc.communicate(timeout=10)
    assert [e["seq"] for e in got] == [e["seq"] for e in EVENTS]
    assert got == EVENTS
    assert b"client connected" in out


def test_serve_late_joiner_gets_snapshot_first(tmp_path):
    """A client joining after the game finished (during the pause) receives the whole game at once."""
    path = tmp_path / "game_0001.jsonl"
    path.write_text("".join(json.dumps(e) + "\n" for e in EVENTS))
    port = _free_port()
    proc = subprocess.Popen(
        [sys.executable, "-m", "arena.serve", "--host", "127.0.0.1", "--port", str(port), "--replay", str(path),
         "--speed", "1000", "--once"], cwd=REPO, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    async def late() -> tuple[list[dict], float]:
        from websockets.asyncio.client import connect
        await asyncio.sleep(1.5)  # replay at speed 1000 takes ~0.06 s
        for _ in range(100):
            try:
                ws = await connect(f"ws://127.0.0.1:{port}")
                break
            except OSError:
                await asyncio.sleep(0.1)
        t0 = time.monotonic()
        async with ws:
            got = [json.loads(await asyncio.wait_for(ws.recv(), 5)) for _ in EVENTS]
        return got, time.monotonic() - t0

    try:
        got, dt = asyncio.run(late())
    finally:
        proc.terminate()
        proc.wait(timeout=10)
    assert got == EVENTS and dt < 1.0
