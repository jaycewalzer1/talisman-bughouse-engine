"""Event-driven bughouse simulator.

Simulated time is integer milliseconds. Both boards run in parallel; on each board the seat to move is
asked to decide when its turn starts (and again after a wait ends or a stale move bounces). All timed
work goes through one heap of internal events ordered by (time, insertion counter), so a run is fully
deterministic given deterministic players (and charge_wall_time=False).

Emitted events follow docs/events.md exactly.
"""

from __future__ import annotations

import heapq
import time
from dataclasses import replace
from typing import Callable

from arena.interfaces import (
    BOARD_OF,
    OPPONENT,
    PARTNER,
    SEATS,
    TEAM_OF,
    Decision,
    GameResult,
    GameState,
    Message,
    Observation,
    Player,
)

BOARDS = ("A", "B")
FALLBACK_WAIT_MS = 200

# internal event kinds
_FLAG = "flag_check"
_DECIDE = "decide"
_LAND = "land"
_WAKE = "wake"
_DELIVER = "deliver"

# seat statuses
_IDLE = "idle"  # not to move (or game over)
_PENDING = "pending"  # a decide event is queued
_THINKING = "thinking"  # a move is in flight, lands later
_WAITING = "waiting"  # sitting until wake time or an interesting event


class Simulator:
    def __init__(
        self,
        state: GameState,
        players: dict[str, Player],
        *,
        game_id: str,
        setup: str,
        channel_delay_ms: int = 0,
        max_plies: int = 400,
        charge_wall_time: bool = False,
        on_event: Callable[[dict], None] | None = None,
    ) -> None:
        missing = [s for s in SEATS if s not in players]
        if missing:
            raise ValueError(f"missing players for seats: {missing}")
        self.state = state
        self.players = players
        self.game_id = game_id
        self.setup = setup
        self.channel_delay_ms = int(channel_delay_ms)
        self.max_plies = int(max_plies)
        self.charge_wall_time = charge_wall_time
        self.on_event = on_event

        self.events: list[dict] = []
        self.now = 0
        self.plies = 0
        self.result: GameResult | None = None

        self._heap: list[tuple[int, int, str, tuple]] = []
        self._counter = 0
        self._status: dict[str, str] = {s: _IDLE for s in SEATS}
        self._gen: dict[str, int] = {s: 0 for s in SEATS}  # invalidates stale decide/land/wake entries
        self._inbox: dict[str, list[Message]] = {s: [] for s in SEATS}
        self._sitting: dict[str, bool] = {s: False for s in SEATS}
        self._ran = False

    # ------------------------------------------------------------------ public

    def run(self) -> GameResult:
        if self._ran:
            raise RuntimeError("Simulator.run() can only be called once")
        self._ran = True
        try:
            self._start()
            while self.result is None and self._heap:
                t, _, kind, data = heapq.heappop(self._heap)
                if t < self.now:  # never move time backwards
                    t = self.now
                self.now = t
                if self._advance():
                    break
                getattr(self, "_on_" + kind)(*data)
            if self.result is None:  # queue drained without a result: should not happen
                self._end(GameResult(result="draw", reason="aborted"))
            return self.result
        finally:
            for p in self.players.values():
                try:
                    p.close()
                except Exception:
                    pass

    # ------------------------------------------------------------------ plumbing

    def _push(self, t: int, kind: str, *data) -> None:
        heapq.heappush(self._heap, (int(t), self._counter, kind, data))
        self._counter += 1

    def _emit(self, type_: str, **fields) -> dict:
        ev = {"type": type_, "seq": len(self.events), "ts": self.now}
        ev.update(fields)
        self.events.append(ev)
        if self.on_event is not None:
            self.on_event(ev)
        return ev

    def _clocks(self) -> dict[str, int]:
        return {s: int(self.state.clocks_ms[s]) for s in SEATS}

    def _advance(self) -> bool:
        """Run clocks to now. Ends the game and returns True on a flag."""
        res = self.state.advance_clocks(self.now)
        if res is None:
            # Fallback in case the state only flags strictly below zero: a running clock at 0 is a flag.
            for b in BOARDS:
                seat = self.state.to_move(b)
                if self.state.clocks_ms[seat] <= 0:
                    res = GameResult(result=TEAM_OF[OPPONENT[seat]], reason="flag", board=b, loser_seat=seat)
                    break
        if res is not None:
            self._end(res)
            return True
        return False

    def _end(self, res: GameResult) -> None:
        if self.result is not None:
            return
        self.result = res
        for s in SEATS:
            self._status[s] = _IDLE
        self._emit("game_end", result=res.result, reason=res.reason, board=res.board, loserSeat=res.loser_seat)

    def _schedule_flag_check(self, board: str) -> None:
        seat = self.state.to_move(board)
        self._push(self.now + max(0, int(self.state.clocks_ms[seat])), _FLAG, board)

    def _start_turn(self, seat: str) -> None:
        self._gen[seat] += 1
        self._status[seat] = _PENDING
        self._push(self.now, _DECIDE, seat, self._gen[seat])

    def _wake(self, seat: str) -> None:
        if self._status[seat] == _WAITING:
            self._start_turn(seat)

    def _wake_all_waiting(self) -> None:
        for s in SEATS:
            self._wake(s)

    def _send(self, seat: str, msgs: tuple[Message, ...]) -> None:
        for m in msgs:
            deliver = self.now + self.channel_delay_ms
            m = replace(m, from_seat=seat, sent_ms=self.now)
            self._emit(
                "message",
                deliverTs=deliver,
                fromSeat=m.from_seat,
                toSeat=m.to_seat,
                token=m.token,
                text=m.text,
                payload=dict(m.payload),
            )
            if deliver <= self.now:
                self._deliver(m)
            else:
                self._push(deliver, _DELIVER, m)

    def _deliver(self, m: Message) -> None:
        if m.to_seat not in self._inbox:
            return
        self._inbox[m.to_seat].append(m)
        self._wake(m.to_seat)

    def _stop_sitting(self, seat: str) -> None:
        if self._sitting[seat]:
            self._sitting[seat] = False
            self._emit("stall", seat=seat, active=False, reason=None)

    # ------------------------------------------------------------------ lifecycle

    def _start(self) -> None:
        st = self.state
        self.now = 0
        st.advance_clocks(0)
        self._emit(
            "game_start",
            game_id=self.game_id,
            setup=self.setup,
            seats={
                s: {
                    "name": getattr(self.players[s], "name", s),
                    "team": TEAM_OF[s],
                    "kind": getattr(self.players[s], "kind", "unknown"),
                }
                for s in SEATS
            },
            fens={b: st.fen(b) for b in BOARDS},
            clocks=self._clocks(),
            base_ms=getattr(st, "base_ms", None),
            increment_ms=getattr(st, "increment_ms", 0),
            channel_delay_ms=self.channel_delay_ms,
        )
        res = st.result()
        if res is not None:
            self._end(res)
            return
        for b in BOARDS:
            self._schedule_flag_check(b)
        for b in BOARDS:
            self._start_turn(st.to_move(b))

    # ------------------------------------------------------------------ handlers

    def _on_flag_check(self, board: str) -> None:
        # _advance() already ran for this timestamp; a still-running game means this check was stale.
        pass

    def _on_deliver(self, m: Message) -> None:
        self._deliver(m)

    def _on_wake(self, seat: str, gen: int) -> None:
        if gen != self._gen[seat] or self._status[seat] != _WAITING:
            return
        self._start_turn(seat)

    def _observe(self, seat: str) -> Observation:
        st = self.state
        inbox = tuple(self._inbox[seat])
        self._inbox[seat] = []
        return Observation(
            now_ms=self.now,
            seat=seat,
            fens={b: st.fen(b) for b in BOARDS},
            clocks_ms=self._clocks(),
            to_move={b: st.to_move(b) for b in BOARDS},
            legal_moves=list(st.legal_moves(seat)),
            inbox=inbox,
            ply=self.plies,
        )

    def _on_decide(self, seat: str, gen: int) -> None:
        if gen != self._gen[seat] or self._status[seat] != _PENDING:
            return
        if self.state.to_move(BOARD_OF[seat]) != seat:  # defensive; should not happen
            self._status[seat] = _IDLE
            return
        obs = self._observe(seat)
        t0 = time.perf_counter()
        dec = self.players[seat].decide(obs)
        wall_ms = int(round((time.perf_counter() - t0) * 1000))

        if dec.kind == "move" and (not obs.legal_moves or dec.uci not in obs.legal_moves):
            # Guard: no legal moves (or a move that is not legal right now) -> sit briefly and re-ask.
            reason = "no_moves" if not obs.legal_moves else "invalid_move"
            dec = Decision(kind="wait", wait_ms=FALLBACK_WAIT_MS, messages=dec.messages, stall_reason=reason)

        self._gen[seat] += 1
        g = self._gen[seat]
        if dec.kind == "move":
            cost = wall_ms if self.charge_wall_time else max(0, int(dec.think_ms))
            self._status[seat] = _THINKING
            self._emit(
                "thinking",
                seat=seat,
                phase="start",
                think_ms=cost,
                nodes=dec.nodes,
                eval=None,
                depth=getattr(dec, "depth", None),
            )
            self._push(self.now + cost, _LAND, seat, g, dec, self.now)
        else:
            self._status[seat] = _WAITING
            if not self._sitting[seat]:
                self._sitting[seat] = True
                self._emit("stall", seat=seat, active=True, reason=dec.stall_reason)
            self._send(seat, dec.messages)
            # a delivery inside _send (delay 0) may have already woken us; only schedule if still waiting
            if self._status[seat] == _WAITING and self._gen[seat] == g:
                self._push(self.now + max(1, int(dec.wait_ms)), _WAKE, seat, g)

    def _on_land(self, seat: str, gen: int, dec: Decision, started: int) -> None:
        if gen != self._gen[seat] or self._status[seat] != _THINKING:
            return
        st = self.state
        board = BOARD_OF[seat]
        spent = self.now - started
        end_fields = dict(seat=seat, phase="end", think_ms=spent, nodes=dec.nodes,
                          eval=dec.eval, depth=getattr(dec, "depth", None))
        if dec.uci not in st.legal_moves(seat):
            self._emit("thinking", **end_fields)
            self._start_turn(seat)
            return
        try:
            mr = st.apply_move(seat, dec.uci, self.now)
        except ValueError:
            # BughouseState raises if the mover flagged by now_ms (and records the flag on the state).
            res = st.result()
            if res is not None:
                self._end(res)
                return
            self._emit("thinking", **end_fields)  # treat as stale: re-ask, time spent is lost
            self._start_turn(seat)
            return
        self._stop_sitting(seat)
        self._status[seat] = _IDLE
        self.plies += 1
        self._emit("thinking", **end_fields)
        self._emit(
            "move",
            board=board,
            seat=seat,
            uci=dec.uci,
            san=mr.san,
            isDrop="@" in dec.uci,
            capture=mr.capture,
            fen=mr.fen,
            clocks=self._clocks(),
        )
        if mr.capture:
            self._emit("pass", fromBoard=board, fromSeat=seat, toSeat=PARTNER[seat], piece=mr.capture)
        self._send(seat, dec.messages)

        res = st.result()
        if res is not None:
            self._end(res)
            return
        if self.plies >= self.max_plies:
            self._end(GameResult(result="draw", reason="max_plies"))
            return
        self._schedule_flag_check(board)
        self._wake_all_waiting()
        nxt = st.to_move(board)
        if self._status[nxt] == _IDLE:
            self._start_turn(nxt)
