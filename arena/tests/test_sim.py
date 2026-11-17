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
    "thinking": {"type", "seq", "ts", "seat", "phase", "think_ms", "nodes", "eval", "depth", "borrowed"},
    "move": {"type", "seq", "ts", "board", "seat", "uci", "san", "isDrop", "capture", "fen", "clocks"},
    "pass": {"type", "seq", "ts", "fromBoard", "fromSeat", "toSeat", "piece"},
    "message": {"type", "seq", "ts", "deliverTs", "fromSeat", "toSeat", "token", "text", "payload"},
    "stall": {"type", "seq", "ts", "seat", "active", "reason"},
    "game_end": {"type", "seq", "ts", "result", "reason", "board", "loserSeat"},
}


# ---------------------------------------------------------------------------------------------- machine lending

def test_borrowed_machine_defers_partner_until_returned():
    # t=0: A_white borrows idle B_black's machine for 300 ms; B_white moves at 100, so B_black is to
    # move at 100 but its machine is busy until 300.
    sim, players, _ = make(max_plies=3, scripts={
        "A_white": lambda obs: mv(obs.legal_moves[0], think=300, borrow=obs.partner_machine_free),
    })
    sim.run()
    ev = sim.events
    check_stream(ev)
    assert players["A_white"].observations[0].partner_machine_free is True
    start = of(ev, "thinking", phase="start", seat="A_white")[0]
    assert start["borrowed"] is True and start["think_ms"] == 300
    assert players["B_black"].observations[0].now_ms == 300
    assert of(ev, "thinking", phase="start", seat="B_black")[0]["ts"] == 300
    assert of(ev, "move", seat="B_black")[0]["ts"] == 400  # its clock ran while it waited for the machine
    assert of(ev, "move", seat="B_black")[0]["clocks"]["B_black"] == 60000 - 300


def test_part_of_a_machine_lent_slows_the_partner_instead_of_deferring_it():
    # t=0: A_white borrows half of B_black's machine for 300 ms. B_white moves at 100, so B_black starts its
    # 100 ms search at 100 on the half it kept: that takes 200 ms.
    sim, players, _ = make(max_plies=3, scripts={
        "A_white": lambda obs: mv(obs.legal_moves[0], think=300, borrow=obs.partner_machine_free, borrow_share=0.5),
    })
    sim.run()
    check_stream(sim.events)
    assert players["B_black"].observations[0].now_ms == 100
    start = of(sim.events, "thinking", phase="start", seat="B_black")[0]
    assert (start["ts"], start["think_ms"]) == (100, 200)
    assert of(sim.events, "move", seat="B_black")[0]["ts"] == 300


def test_a_search_that_outlasts_the_loan_speeds_up_when_the_machine_returns():
    # As above, but B_black wants 150 ms of search: 100 of it done at half speed by 300, the last 50 at full speed.
    sim, _, _ = make(max_plies=3, scripts={
        "A_white": lambda obs: mv(obs.legal_moves[0], think=300, borrow=obs.partner_machine_free, borrow_share=0.5),
        "B_black": lambda obs: mv(obs.legal_moves[0], think=150),
    })
    sim.run()
    assert of(sim.events, "thinking", phase="start", seat="B_black")[0]["think_ms"] == 250
    assert of(sim.events, "move", seat="B_black")[0]["ts"] == 350


def test_borrowing_none_of_a_machine_raises():
    sim, _, _ = make(max_plies=3, scripts={
        "A_white": lambda obs: mv(obs.legal_moves[0], borrow=True, borrow_share=0.0)})
    with pytest.raises(ValueError, match="borrowed 0.0"):
        sim.run()


def test_borrowing_a_busy_machine_raises():
    # A_white thinks 100 on its own machine; B_black is to move at 50 and tries to borrow it.
    sim, players, _ = make(max_plies=10, scripts={
        "B_white": [mv("a2a3", think=50)],
        "B_black": lambda obs: mv(obs.legal_moves[0], think=100, borrow=True),
    })
    with pytest.raises(ValueError, match="borrowed"):
        sim.run()
    assert players["B_black"].observations[0].partner_machine_free is False
    assert all(p.closed for p in players.values())


def test_search_behind_a_wait_keeps_own_machine_busy():
    # A_white sits after a 400 ms search; its partner B_black (to move at 50) cannot borrow until 400.
    sim, players, _ = make(max_plies=4, scripts={
        "A_white": [wait(1000, think_ms=400)],
        "B_white": [mv("a2a3", think=50)],
    })
    sim.run()
    frees = [(o.now_ms, o.partner_machine_free) for o in players["B_black"].observations]
    assert frees[0] == (50, False)


def test_simulator_starts_at_forked_state_time():
    state = FakeState()
    state.advance_clocks(5000)  # both white clocks ran 5 s before the fork
    sim, _, _ = make(state=state, max_plies=1)
    sim.run()
    ev = sim.events
    check_stream(ev)
    assert ev[0]["ts"] == 5000 and ev[0]["clocks"]["A_white"] == 55000
    first = of(ev, "move")[0]
    assert first["ts"] == 5100 and first["clocks"]["A_white"] == 54900


# ---------------------------------------------------------------------------------------------- phase 1 waits

def test_open_ended_wait_is_asked_again_every_5s():
    scripts = {"A_white": lambda obs: wait(5000), "A_black": lambda obs: wait(5000),
               "B_white": lambda obs: wait(5000), "B_black": lambda obs: wait(5000)}
    sim, players, _ = make(FakeState(base_ms=16_000), scripts)
    sim.run()
    assert [o.now_ms for o in players["A_white"].observations] == [0, 5000, 10000, 15000]


def test_piece_arrival_ends_the_wait_right_away():
    # A_white waits for a knight. B_black captures one at 2000 ms: A_white is asked again at that moment,
    # with the knight already in its pocket, and does not sit out the rest of its 5 s.
    def a_white(obs):
        if "N@e4" in obs.legal_moves:
            return mv("N@e4", 0)
        return wait(5000)

    scripts = {"A_white": a_white, "B_white": [mv("a2a3", 1000)], "B_black": [mv("xn", 1000)]}
    sim, players, _ = make(FakeState(captures={"xn": "n"}, extra={"B_black": ["xn"]}), scripts, max_plies=3)
    sim.run()
    assert of(sim.events, "pass", toSeat="A_white")[0]["ts"] == 2000
    assert of(sim.events, "move", seat="A_white")[0]["uci"] == "N@e4"
    assert of(sim.events, "move", seat="A_white")[0]["ts"] == 2000


def test_partner_message_interrupts_an_open_ended_wait():
    msg = Message("B_black", "A_white", "have_N", 0, payload={"piece": "N", "plies": 2})
    scripts = {"A_white": lambda obs: wait(5000), "B_white": [mv("a2a3", 1000)],
               "B_black": [mv("a2a3", 500, messages=(msg,))], "A_black": lambda obs: wait(5000)}
    sim, players, _ = make(FakeState(base_ms=3000), scripts)
    sim.run()
    asked = players["A_white"].observations
    delivered = [o for o in asked if o.inbox]
    assert delivered and delivered[0].now_ms == 1500
    assert delivered[0].inbox[0].payload["plies"] == 2


# ---------------------------------------------------------------------------------------------- spec v3 rules

from arena.sim import RULES_V3, scaled_max_plies  # noqa: E402
from arena.state import BughouseState  # noqa: E402


def _landings(events, seat):
    return [e["ts"] for e in of(events, "move", seat=seat)]


def test_min_move_ms_floors_every_landing_even_instant_moves():
    sim, _, _ = make(scripts={s: (lambda obs: mv(obs.legal_moves[0], 0)) for s in SEATS}, max_plies=40,
                     min_move_ms=100)
    sim.run()
    for seat in SEATS:
        t = _landings(sim.events, seat)
        assert t and all(b - a >= 100 for a, b in zip(t, t[1:]))
    assert all(e["think_ms"] >= 100 for e in of(sim.events, "thinking", phase="start"))


def test_three_seconds_of_clock_allow_at_most_30_landings():
    sim, _, _ = make(FakeState(base_ms=3000), {s: (lambda obs: mv(obs.legal_moves[0], 0)) for s in SEATS},
                     max_plies=10_000, min_move_ms=100)
    res = sim.run()
    assert res.reason == "flag"
    assert max(len(_landings(sim.events, s)) for s in SEATS) <= 30


def test_additive_lag_is_seeded_and_bounded():
    def costs(seed):
        sim, _, _ = make(scripts={s: (lambda obs: mv(obs.legal_moves[0], 10)) for s in SEATS}, max_plies=60,
                         min_move_ms=100, lag_max_ms=50, seed=seed)
        sim.run()
        return [e["think_ms"] for e in of(sim.events, "thinking", phase="start")]

    a = costs(1)
    assert all(100 <= c <= 150 for c in a) and len(set(a)) > 5
    assert a == costs(1) and a != costs(2)


def test_lag_and_floor_off_by_default():
    sim, _, _ = make(scripts={s: (lambda obs: mv(obs.legal_moves[0], 7)) for s in SEATS}, max_plies=8)
    sim.run()
    assert {e["think_ms"] for e in of(sim.events, "thinking", phase="start")} == {7}


def test_max_plies_scales_with_base_time():
    assert scaled_max_plies(60000) == 400 and scaled_max_plies(180000) == 1200 and scaled_max_plies(2000) == 400
    sim, _, _ = make(FakeState(base_ms=300_000), max_plies=None)
    assert sim.max_plies == 2000
    assert make(max_plies=7)[0].max_plies == 7


def test_aborted_game_scored_by_the_on_turn_clocks():
    sim, _, _ = make(max_plies=5, score_aborts=True)
    res = sim.run()
    # after 5 plies (100 ms each) A_black is on turn with 59900 and B_white with 59800: B_white would flag first
    assert res == GameResult("T1", "max_plies", "B", "B_white")
    sim, _, _ = make(max_plies=6, score_aborts=True)
    assert sim.run() == GameResult("draw", "max_plies")  # equal on-turn clocks


def _all_wait(obs):
    return wait(1000)


def test_mutual_pass_is_adjudicated_with_the_predicted_loser():
    state = FakeState()
    state.clocks_ms["B_white"] = 50_000  # B_white would flag first
    sim, _, _ = make(state, {s: _all_wait for s in SEATS}, sitoff_ms=3000, sitoff_unadjudicated=0.0)
    res = sim.run()
    assert res == GameResult("T1", "sitoff_adjudicated", "B", "B_white")
    assert sim.events[-1]["ts"] == 3000 and sim.events[-1]["reason"] == "sitoff_adjudicated"


def test_sitoff_clock_starts_when_the_second_seat_sits():
    scripts = {"A_white": _all_wait, "B_white": [mv("a2a3", 500)], "B_black": _all_wait}
    sim, _, _ = make(scripts=scripts, sitoff_ms=3000, sitoff_unadjudicated=0.0)
    res = sim.run()
    # A_white sits from 0, B_black from 500 (its turn starts when B_white lands): adjudicated at 3500
    assert res.reason == "sitoff_adjudicated" and sim.events[-1]["ts"] == 3500
    assert res.loser_seat == "A_white"  # 56500 left vs B_black's 57000


def test_one_sided_sit_is_not_adjudicated():
    scripts = {"A_white": _all_wait, "B_white": lambda obs: mv(obs.legal_moves[0], 100),
               "B_black": lambda obs: mv(obs.legal_moves[0], 100)}
    sim, _, _ = make(FakeState(base_ms=10_000), scripts, sitoff_ms=1000, sitoff_unadjudicated=0.0,
                     max_plies=10_000)
    res = sim.run()
    assert res.reason == "flag" and res.loser_seat == "A_white"


def test_unadjudicated_games_run_to_the_flag():
    sim, _, _ = make(FakeState(base_ms=10_000), {s: _all_wait for s in SEATS}, sitoff_ms=1000,
                     sitoff_unadjudicated=1.0)
    assert not sim.adjudicates_sitoffs
    res = sim.run()
    assert res.reason == "double_flag" or res.reason == "flag"


def test_unadjudicated_share_is_about_ten_percent():
    share = sum(not make(sitoff_ms=1000, seed=k)[0].adjudicates_sitoffs for k in range(2000)) / 2000
    assert 0.08 < share < 0.12
    assert not make(seed=3)[0].adjudicates_sitoffs  # off unless sitoff_ms is set


def test_frozen_defender_flags_without_the_attacker_moving():
    # A_black is in a rook check that a drop could block, so it is not mate, but its hand is empty: no legal
    # move. Nobody on board B moves (no piece can arrive), so A_black sits until it flags.
    frozen = "R5k1/5ppp/8/8/8/8/8/K7[] b - - 0 1"
    start = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR[] w KQkq - 0 1"
    state = BughouseState(60000, fens={"A": frozen, "B": start}, start_clocks_ms={"A_black": 2000})
    assert state.legal_moves("A_black") == [] and state.result() is None
    players = {s: Scripted(s, lambda obs: mv(obs.legal_moves[0], 100) if obs.legal_moves else wait(500))
               for s in SEATS}
    players["B_white"] = Scripted("B_white", lambda obs: wait(500))
    sim = Simulator(state, players, game_id="frozen", setup="test")
    res = sim.run()
    assert res == GameResult("T1", "flag", "A", "A_black") and sim.events[-1]["ts"] == 2000
    assert not of(sim.events, "move")
    assert of(sim.events, "stall", seat="A_black", active=True)[0]["reason"] == "need_piece"


def test_rules_v3_profile_runs_a_real_game():
    state = BughouseState(20_000)
    players = {s: Scripted(s, lambda obs: mv(sorted(obs.legal_moves)[0], 0) if obs.legal_moves else wait(500))
               for s in SEATS}
    sim = Simulator(state, players, game_id="v3", setup="test", seed=5, **RULES_V3)
    res = sim.run()
    assert res.reason in {"checkmate", "flag", "double_flag", "repetition", "sitoff_adjudicated", "max_plies"}
    check_stream(sim.events)
    for seat in SEATS:
        t = _landings(sim.events, seat)
        assert all(b - a >= 100 for a, b in zip(t, t[1:]))


# ---------------------------------------------------------------------------------------------- stats metrics

from arena.stats import METRICS, compute_stats, format_table, summarize_game  # noqa: E402


def test_stats_sitting_flag_and_pass_rate_by_diagonal_sign():
    state = FakeState()
    state.clocks_ms["B_white"] = 50_000  # A_white's diagonal (A_white - B_white) is +10 s: ahead
    scripts = {"A_white": _all_wait, "B_white": lambda obs: mv(obs.legal_moves[0], 100),
               "B_black": lambda obs: mv(obs.legal_moves[0], 100)}
    sim, _, _ = make(state, scripts, max_plies=10_000)
    res = sim.run()
    assert res.loser_seat == "A_white" and res.reason == "flag"
    s = summarize_game(sim.events)
    assert s["flag_while"] == "sitting" and s["flag_other_check"] is False  # FakeState FENs: no check known
    t1, t2 = s["clock_teams"]["T1"], s["clock_teams"]["T2"]
    assert (t1["turns_ahead"] >= 1, t1["sits_ahead"], t1["sit_ms_ahead"]) == (True, 1, 60_000)
    # B_white starts 10 s behind A_white and never sits
    assert t2["turns_behind"] >= 1 and t2["sits_behind"] == 0 and t2["sits_ahead"] == 0
    s.update(contender_team="T2", opening_id=0)
    m = compute_stats([s], n_boot=10)["metrics"]
    assert m["baseline_flagged_sitting"]["est"] == 1.0 and m["baseline_flagged_moving"]["est"] == 0.0
    assert m["baseline_pass_rate_ahead"]["est"] == 1.0 and m["baseline_sit_ms_ahead"]["est"] == 60_000
    assert m["contender_pass_rate_behind"]["est"] == 0.0
    assert set(METRICS) <= set(m) and "pass rate, ahead" in format_table(compute_stats([s], n_boot=10))


def test_stats_flag_while_moving():
    sim, _, _ = make(FakeState(base_ms=500), {"A_white": [mv("a2a3", 600)], "B_white": [mv("a2a3", 100)],
                                              "B_black": [mv("a2a3", 1000)]})
    sim.run()
    s = summarize_game(sim.events)
    assert s["loser_seat"] == "A_white" and s["flag_while"] == "moving"


def test_stats_flag_win_while_the_other_board_is_in_check():
    start = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR[] w KQkq - 0 1"
    checked = "rnbqk1nr/pppp1ppp/8/4p3/1b1P4/8/PPP1PPPP/RNBQKBNR[] w KQkq - 1 3"  # white in check on B
    clocks = {s: 60000 for s in SEATS}
    ev = [
        {"type": "game_start", "seq": 0, "ts": 0, "game_id": "g", "setup": "t", "seats": {},
         "fens": {"A": start, "B": checked}, "clocks": clocks},
        {"type": "stall", "seq": 1, "ts": 0, "seat": "B_white", "active": True, "reason": "need_piece"},
        {"type": "game_end", "seq": 2, "ts": 60000, "result": "T2", "reason": "flag", "board": "A",
         "loserSeat": "A_white"},
    ]
    s = summarize_game(ev)
    assert s["flag_other_check"] is True and s["flag_while"] == "moving"
    assert s["clock_teams"]["T2"]["sit_ms_behind"] == 60000  # B_white level on the diagonal: "behind"
    s.update(contender_team="T2", opening_id=0)
    m = compute_stats([s], n_boot=10)["metrics"]
    assert m["contender_flag_win_other_check"]["est"] == 1.0 and m["baseline_flag_win_other_check"]["est"] == 0.0


def test_stats_reads_older_summaries_without_clock_fields():
    s = summarize_game([{"type": "game_start", "ts": 0},
                        {"type": "game_end", "ts": 5, "result": "draw", "reason": "double_flag"}])
    s.pop("clock_teams")
    s.update(contender_team="T1", opening_id=0)
    m = compute_stats([s], n_boot=10)["metrics"]
    assert m["draw_rate"]["est"] == 1.0 and m["score"]["est"] == 0.5
