"""Tests for arena.sim.Simulator using a tiny fake GameState and scripted players."""

from __future__ import annotations

import copy

import pytest

from arena.interfaces import (
    BOARD_OF,
    OPPONENT,
    PARTNER,
    SEATS,
    TEAM_OF,
    Decision,
    GameResult,
    Message,
    MoveResult,
    Observation,
)
from arena.sim import Simulator

# ---------------------------------------------------------------------------------------------- fakes

DEFAULT_MOVES = ["a2a3", "b2b3", "c2c3"]


class FakeState:
    """Minimal two-board state. Every seat always has DEFAULT_MOVES (+ drops of pocket pieces, + any
    `extra` moves). `captures` maps uci -> piece letter passed to the mover's partner. `mate_at` maps
    board -> board-ply count after which that board is checkmate (loser = side to move)."""

    def __init__(self, base_ms=60000, increment_ms=0, captures=None, extra=None, mate_at=None,
                 legal_fn=None, no_moves=()):
        self.base_ms = base_ms
        self.increment_ms = increment_ms
        self.clocks_ms = {s: base_ms for s in SEATS}
        self.board_ply = {"A": 0, "B": 0}
        self.pockets = {s: [] for s in SEATS}
        self.captures = captures or {}
        self.extra = extra or {}
        self.mate_at = mate_at or {}
        self.legal_fn = legal_fn
        self.no_moves = set(no_moves)
        self.last_ms = 0
        self.flag: GameResult | None = None
        self.mate: GameResult | None = None

    def fen(self, board):
        pocket = "".join(self.pockets[f"{board}_white"]) + "".join(self.pockets[f"{board}_black"]).lower()
        return f"{board}{self.board_ply[board]}[{pocket}]"

    def to_move(self, board):
        return f"{board}_{'white' if self.board_ply[board] % 2 == 0 else 'black'}"

    def legal_moves(self, seat):
        if self.to_move(BOARD_OF[seat]) != seat or seat in self.no_moves:
            return []
        moves = DEFAULT_MOVES + list(self.extra.get(seat, [])) + [f"{p.upper()}@e4" for p in self.pockets[seat]]
        if self.legal_fn is not None:
            moves = [m for m in moves if self.legal_fn(self, seat, m)]
        return moves

    def advance_clocks(self, now_ms):
        if self.flag is not None:
            return self.flag
        dt = now_ms - self.last_ms
        assert dt >= 0, "time went backwards"
        self.last_ms = now_ms
        for b in "AB":
            s = self.to_move(b)
            self.clocks_ms[s] -= dt
            if self.clocks_ms[s] <= 0:
                self.clocks_ms[s] = 0
                if self.flag is None:
                    self.flag = GameResult(TEAM_OF[OPPONENT[s]], "flag", b, s)
        return self.flag

    def apply_move(self, seat, uci, now_ms):
        if uci not in self.legal_moves(seat):
            raise ValueError(uci)
        self.advance_clocks(now_ms)
        board = BOARD_OF[seat]
        if "@" in uci:
            self.pockets[seat].remove(uci[0])
        cap = self.captures.get(uci)
        if cap:
            self.pockets[PARTNER[seat]].append(cap.upper())
        self.clocks_ms[seat] += self.increment_ms
        self.board_ply[board] += 1
        if self.mate_at.get(board) == self.board_ply[board]:
            loser = self.to_move(board)
            self.mate = GameResult(TEAM_OF[OPPONENT[loser]], "checkmate", board, loser)
        return MoveResult(san=uci, capture=cap, fen=self.fen(board))

    def result(self):
        return self.mate or self.flag

    def copy(self):
        return copy.deepcopy(self)


class Scripted:
    """Player driven by a function obs -> Decision (or a list of Decisions consumed in order, then the
    fallback `default`)."""

    def __init__(self, seat, script=None, default=None):
        self.seat = seat
        self.name = f"scripted-{seat}"
        self.kind = "scripted"
        self.script = list(script) if isinstance(script, (list, tuple)) else script
        self.default = default or (lambda obs: Decision(kind="move", uci=obs.legal_moves[0], think_ms=100)
                                   if obs.legal_moves else Decision(kind="wait", wait_ms=1000))
        self.observations: list[Observation] = []
        self.closed = False

    def decide(self, obs):
        self.observations.append(obs)
        if callable(self.script):
            return self.script(obs)
        if self.script:
            d = self.script.pop(0)
            return d(obs) if callable(d) else d
        return self.default(obs)

    def close(self):
        self.closed = True


def make(state=None, scripts=None, **kw):
    state = state or FakeState()
    scripts = scripts or {}
    players = {s: Scripted(s, scripts.get(s)) for s in SEATS}
    kw.setdefault("game_id", "g0001")
    kw.setdefault("setup", "test")
    sim = Simulator(state, players, **kw)
    return sim, players, state


def of(events, type_, **match):
    return [e for e in events if e["type"] == type_ and all(e.get(k) == v for k, v in match.items())]


def mv(uci, think=100, **kw):
    return Decision(kind="move", uci=uci, think_ms=think, **kw)


def wait(ms, reason="need_piece", **kw):
    return Decision(kind="wait", wait_ms=ms, stall_reason=reason, **kw)


def check_stream(events):
    assert events[0]["type"] == "game_start"
    assert events[-1]["type"] == "game_end"
    assert len(of(events, "game_end")) == 1
    assert [e["seq"] for e in events] == list(range(len(events)))
    ts = [e["ts"] for e in events]
    assert ts == sorted(ts)


# ---------------------------------------------------------------------------------------------- tests

def test_basic_stream_ordering_and_clock_charging():
    seen = []
    sim, players, _ = make(max_plies=6, on_event=seen.append)
    res = sim.run()
    ev = sim.events
    check_stream(ev)
    assert seen == ev
    assert res == GameResult("draw", "max_plies")
    end = ev[-1]
    assert end["reason"] == "max_plies" and end["board"] is None and end["loserSeat"] is None

    gs = ev[0]
    assert gs["game_id"] == "g0001" and gs["setup"] == "test"
    assert gs["seats"]["A_white"] == {"name": "scripted-A_white", "team": "T1", "kind": "scripted"}
    assert gs["seats"]["A_black"]["team"] == "T2"
    assert gs["base_ms"] == 60000 and gs["increment_ms"] == 0 and gs["channel_delay_ms"] == 0
    assert gs["fens"] == {"A": "A0[]", "B": "B0[]"}

    moves = of(ev, "move")
    assert len(moves) == 6
    # both boards in parallel: A_white and B_white both land at 100
    first = moves[0]
    assert first["seat"] == "A_white" and first["ts"] == 100 and moves[1]["seat"] == "B_white"
    assert moves[1]["ts"] == 100
    assert first["clocks"]["A_white"] == 59900 and first["clocks"]["A_black"] == 60000
    # A_black thinks 100 from t=100 -> lands at 200, charged 100
    ab = of(ev, "move", seat="A_black")[0]
    assert ab["ts"] == 200 and ab["clocks"]["A_black"] == 59900 and ab["clocks"]["A_white"] == 59900
    assert first["isDrop"] is False and first["capture"] is None and first["fen"] == "A1[]"

    # every move preceded by thinking start/end of the same seat
    for m in moves:
        i = ev.index(m)
        assert ev[i - 1]["type"] == "thinking" and ev[i - 1]["phase"] == "end" and ev[i - 1]["seat"] == m["seat"]
    starts = of(ev, "thinking", phase="start", seat="A_white")
    assert starts[0]["ts"] == 0 and starts[0]["think_ms"] == 100
    assert all(p.closed for p in players.values())


def test_decide_inputs_observation():
    sim, players, _ = make(max_plies=3)
    sim.run()
    obs = players["A_white"].observations[0]
    assert obs.now_ms == 0 and obs.seat == "A_white"
    assert obs.to_move == {"A": "A_white", "B": "B_white"}
    assert obs.legal_moves == DEFAULT_MOVES
    assert set(obs.clocks_ms) == set(SEATS)
    assert players["A_black"].observations[0].now_ms == 100


def test_move_pass_message_ordering_and_drop():
    state = FakeState(captures={"x1x1": "n"}, extra={"A_white": ["x1x1"]})
    msg = Message("A_white", "B_black", "have_N", sent_ms=-1, text="knight coming", payload={"piece": "N"})
    scripts = {"A_white": [mv("x1x1", 50, messages=(msg,), eval=0.3, nodes=42)],
               "B_white": [mv("a2a3", 10)],
               "B_black": [lambda obs: mv("N@e4", 100)]}
    sim, players, _ = make(state, scripts, max_plies=4)
    sim.run()
    ev = sim.events
    check_stream(ev)
    i = ev.index(of(ev, "move", seat="A_white")[0])
    assert [e["type"] for e in ev[i - 1:i + 3]] == ["thinking", "move", "pass", "message"]
    assert ev[i - 1]["eval"] == 0.3 and ev[i - 1]["nodes"] == 42 and ev[i - 1]["think_ms"] == 50
    assert ev[i]["capture"] == "n"
    p = ev[i + 1]
    assert p == {"type": "pass", "seq": p["seq"], "ts": 50, "fromBoard": "A", "fromSeat": "A_white",
                 "toSeat": "B_black", "piece": "n"}
    m = ev[i + 2]
    assert m["ts"] == 50 and m["deliverTs"] == 50 and m["fromSeat"] == "A_white" and m["toSeat"] == "B_black"
    assert m["token"] == "have_N" and m["text"] == "knight coming" and m["payload"] == {"piece": "N"}
    # B_black decides at 10 (before capture) -> no drop yet; then... it gets its first decision at 10
    bb_obs = players["B_black"].observations[0]
    assert bb_obs.now_ms == 10 and "N@e4" not in bb_obs.legal_moves


def test_channel_delay_delivery_into_inbox():
    msg = Message("A_white", "B_black", "need_N", sent_ms=0)
    scripts = {
        "A_white": [mv("a2a3", 100, messages=(msg,))],
        "B_white": [mv("a2a3", 300)],
        # B_black to move at 300; message (sent 100, delivered 600) not there yet -> waits, woken at 600
        "B_black": [wait(5000), mv("b2b3", 100)],
        "A_black": [mv("a2a3", 10000)],
    }
    sim, players, _ = make(scripts=scripts, channel_delay_ms=500, max_plies=4)
    sim.run()
    ev = sim.events
    check_stream(ev)
    m = of(ev, "message")[0]
    assert m["ts"] == 100 and m["deliverTs"] == 600
    obs = players["B_black"].observations
    assert obs[0].now_ms == 300 and obs[0].inbox == ()
    assert obs[1].now_ms == 600 and len(obs[1].inbox) == 1
    got = obs[1].inbox[0]
    assert got.token == "need_N" and got.sent_ms == 100 and got.from_seat == "A_white"
    # inbox is cleared after a decision
    assert all(o.inbox == () for o in obs[2:])
    assert of(ev, "move", seat="B_black")[0]["ts"] == 700


def test_wait_woken_early_by_partner_move_and_stall_events():
    scripts = {
        "A_white": [wait(5000, "need_piece"), mv("a2a3", 100)],
        "B_white": [mv("a2a3", 300)],
        "B_black": [mv("a2a3", 10000)],
        "A_black": [mv("a2a3", 10000)],
    }
    sim, players, _ = make(scripts=scripts, max_plies=2)
    sim.run()
    ev = sim.events
    check_stream(ev)
    obs = players["A_white"].observations
    assert [o.now_ms for o in obs] == [0, 300]
    stalls = of(ev, "stall", seat="A_white")
    assert [(s["active"], s["ts"]) for s in stalls] == [(True, 0), (False, 400)]
    assert stalls[0]["reason"] == "need_piece"
    i = ev.index(stalls[1])
    assert ev[i + 1]["type"] == "thinking" and ev[i + 1]["phase"] == "end"
    assert ev[i + 2]["type"] == "move" and ev[i + 2]["seat"] == "A_white" and ev[i + 2]["ts"] == 400
    # waiting cost clock time: A_white ran 400ms
    assert ev[i + 2]["clocks"]["A_white"] == 59600


def test_wait_expires_without_events_and_stall_emitted_once():
    scripts = {
        "A_white": [wait(1000), wait(1000), mv("a2a3", 0)],
        "B_white": [mv("a2a3", 50000)],
    }
    sim, players, _ = make(scripts=scripts, max_plies=1)
    sim.run()
    ev = sim.events
    assert [o.now_ms for o in players["A_white"].observations] == [0, 1000, 2000]
    assert len(of(ev, "stall", seat="A_white", active=True)) == 1
    assert len(of(ev, "stall", seat="A_white", active=False)) == 1


def test_flag_at_exact_ms_while_waiting():
    state = FakeState(base_ms=1000)
    scripts = {
        "A_white": lambda obs: wait(10_000),
        "B_white": lambda obs: mv("a2a3", 5000),
    }
    sim, players, _ = make(state, scripts)
    res = sim.run()
    ev = sim.events
    check_stream(ev)
    assert res == GameResult("T2", "flag", "A", "A_white")
    end = ev[-1]
    assert end["ts"] == 1000 and end["result"] == "T2" and end["reason"] == "flag"
    assert end["board"] == "A" and end["loserSeat"] == "A_white"
    assert not of(ev, "move")


def test_flag_while_thinking_beats_landing_at_same_ms():
    state = FakeState(base_ms=500)
    scripts = {"A_white": [mv("a2a3", 500)], "B_white": [mv("a2a3", 100)], "B_black": [mv("a2a3", 600)]}
    sim, _, _ = make(state, scripts)
    res = sim.run()
    ev = sim.events
    check_stream(ev)
    assert res.reason == "flag" and res.loser_seat == "A_white" and ev[-1]["ts"] == 500
    assert not of(ev, "move", seat="A_white")


def test_flag_without_any_decisions_pending_increment_rescheduling():
    # with increment, flag times are recomputed after each move
    state = FakeState(base_ms=1000, increment_ms=500)
    scripts = {"A_white": [mv("a2a3", 900)], "A_black": lambda obs: wait(100_000),
               "B_white": lambda obs: wait(100_000)}
    sim, _, _ = make(state, scripts)
    res = sim.run()
    # B_white flags at 1000 (A_black would flag at 900+1000=1900)
    assert res == GameResult("T1", "flag", "B", "B_white") and sim.events[-1]["ts"] == 1000
    assert of(sim.events, "move", seat="A_white")[0]["clocks"]["A_white"] == 600


def test_stale_move_is_reasked_and_time_lost():
    # "z9z9" is only legal for A_white while board B has no moves
    state = FakeState(extra={"A_white": ["z9z9"]},
                      legal_fn=lambda st, seat, m: not (m == "z9z9" and st.board_ply["B"] > 0))
    scripts = {"A_white": [mv("z9z9", 500), mv("a2a3", 100)], "B_white": [mv("a2a3", 100)],
               "A_black": [mv("a2a3", 10000)], "B_black": [mv("a2a3", 10000)]}
    sim, players, _ = make(state, scripts, max_plies=2)
    sim.run()
    ev = sim.events
    check_stream(ev)
    th = of(ev, "thinking", seat="A_white")
    assert [(t["phase"], t["ts"], t["think_ms"]) for t in th] == [
        ("start", 0, 500), ("end", 500, 500), ("start", 500, 100), ("end", 600, 100)]
    assert [o.now_ms for o in players["A_white"].observations] == [0, 500]
    m = of(ev, "move", seat="A_white")[0]
    assert m["uci"] == "a2a3" and m["ts"] == 600 and m["clocks"]["A_white"] == 60000 - 600


def test_checkmate_single_game_end():
    state = FakeState(mate_at={"B": 3})
    sim, _, _ = make(state)
    res = sim.run()
    ev = sim.events
    check_stream(ev)
    assert res == GameResult("T2", "checkmate", "B", "B_black")
    assert ev[-1]["loserSeat"] == "B_black" and ev[-2]["type"] == "move"
    assert len(of(ev, "move", board="B")) == 3


def test_max_plies_counts_both_boards():
    sim, _, _ = make(max_plies=5)
    res = sim.run()
    assert res.reason == "max_plies" and res.result == "draw"
    assert len(of(sim.events, "move")) == 5
    check_stream(sim.events)


def test_zero_legal_moves_guard_turns_move_into_wait():
    state = FakeState(no_moves={"A_white"})
    calls = []

    def bad(obs):
        calls.append(obs.now_ms)
        return Decision(kind="move", uci="a2a3", think_ms=0)

    scripts = {"A_white": bad, "B_white": [mv("a2a3", 1000)], "B_black": [mv("a2a3", 10_000)]}
    sim, _, _ = make(state, scripts, max_plies=10)
    sim.run()
    # re-asked every 200ms, but woken early by B_white's move at 1000
    assert calls[:6] == [0, 200, 400, 600, 800, 1000]
    st = of(sim.events, "stall", seat="A_white", active=True)
    assert len(st) == 1 and st[0]["reason"] == "no_moves"


def test_wait_messages_emitted_and_delivered_wake_partner():
    msg = Message("A_white", "B_black", "sit", sent_ms=0)
    scripts = {
        "A_white": [wait(10_000, "time_advantage", messages=(msg,)), wait(10_000), mv("a2a3", 0)],
        "B_white": [mv("a2a3", 20)],
        "B_black": [wait(10_000), mv("a2a3", 0)],
    }
    sim, players, _ = make(scripts=scripts, channel_delay_ms=50, max_plies=2)
    sim.run()
    ev = sim.events
    check_stream(ev)
    st = of(ev, "stall", seat="A_white", active=True)[0]
    i = ev.index(st)
    assert ev[i + 1]["type"] == "message" and ev[i + 1]["token"] == "sit" and ev[i + 1]["deliverTs"] == 50
    obs = players["B_black"].observations
    assert obs[0].now_ms == 20 and obs[0].inbox == ()
    assert obs[1].now_ms == 50 and obs[1].inbox[0].token == "sit"
    # A_white was woken by B_white's move at 20 but still sitting: no second stall-start
    assert len(of(ev, "stall", seat="A_white", active=True)) == 1


def test_determinism():
    def run_once():
        sim, _, _ = make(FakeState(captures={"b2b3": "p"}), max_plies=40, channel_delay_ms=30)
        sim.run()
        return sim.events

    assert run_once() == run_once()


def test_close_called_even_on_player_error():
    def boom(obs):
        raise RuntimeError("boom")

    sim, players, _ = make(scripts={"B_white": boom})
    with pytest.raises(RuntimeError):
        sim.run()
    assert all(p.closed for p in players.values())


def test_events_match_documented_schema():
    state = FakeState(captures={"b2b3": "p"})
    msg = Message("A_white", "B_black", "need_P", sent_ms=0, payload={"piece": "P"})
    scripts = {"A_white": [wait(200, messages=(msg,))]}
    sim, _, _ = make(state, scripts, max_plies=12)
    sim.run()
    for e in sim.events:
        assert set(e) == EVENT_KEYS[e["type"]], e


EVENT_KEYS = {
    "game_start": {"type", "seq", "ts", "game_id", "setup", "seats", "fens", "clocks", "base_ms",
                   "increment_ms", "channel_delay_ms"},
    "thinking": {"type", "seq", "ts", "seat", "phase", "think_ms", "nodes", "eval", "depth"},
    "move": {"type", "seq", "ts", "board", "seat", "uci", "san", "isDrop", "capture", "fen", "clocks"},
    "pass": {"type", "seq", "ts", "fromBoard", "fromSeat", "toSeat", "piece"},
    "message": {"type", "seq", "ts", "deliverTs", "fromSeat", "toSeat", "token", "text", "payload"},
    "stall": {"type", "seq", "ts", "seat", "active", "reason"},
    "game_end": {"type", "seq", "ts", "result", "reason", "board", "loserSeat"},
}
