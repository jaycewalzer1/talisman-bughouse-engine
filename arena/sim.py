"""Event-driven bughouse simulator.

Simulated time is integer milliseconds. Both boards run in parallel; on each board the seat to move is
asked to decide when its turn starts (and again after a wait ends or a stale move bounces). All timed
work goes through one heap of internal events ordered by (time, insertion counter), so a run is fully
deterministic given deterministic players (and charge_wall_time=False).

Emitted events follow docs/events.md exactly.

Clock-realism and rule options (spec v3 D4, arena/RULES.md), all off by default so existing callers and the
C++ parity stub behave as before; RULES_V3 holds the profile the spec chose:
- min_move_ms: every landing costs at least this much clock (also a cached or instant move);
- lag_max_ms: an additive lag drawn uniformly from [0, lag_max_ms] per move (SplitMix64 seeded from `seed`,
  the same stream as async_selfplay.cc), charged to the mover's clock like think time;
- max_plies=None scales the safety cap with the base time (scaled_max_plies); score_aborts scores a game
  that hits it by the on-turn clocks (the lower one loses, equal is a draw), reason "max_plies";
- sitoff_ms: when both on-turn seats have been sitting (passing) for this long at once, the game ends as
  "sitoff_adjudicated" with the lower of their two clocks losing (it would flag first if nobody moved; equal
  is a draw). A seeded share `sitoff_unadjudicated` of games is never adjudicated and runs to the flag.
"""

from __future__ import annotations

import heapq
import math
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
M64 = (1 << 64) - 1
_LAG_STREAM = 0x6C61675F  # "lag_": seed ^ this seeds the lag draws (async_selfplay.cc uses the same)
_SITOFF_STREAM = 0x7369746F  # "sito": seed ^ this decides whether this game adjudicates sit-offs

# Spec v3 D4 rules profile for the clock build and free-mode self-play: Simulator(..., **RULES_V3).
RULES_V3 = {
    "min_move_ms": 100,
    "lag_max_ms": 150,
    "max_plies": None,
    "score_aborts": True,
    "sitoff_ms": 5000,
    "sitoff_unadjudicated": 0.1,
}


def scaled_max_plies(base_ms: int | None) -> int:
    """Safety cap on total plies (both boards) scaled with the base time: 400 at 60 s and below, 1200 at
    180 s. Every engine game seen so far ended well before it (audit section 5, rules 1)."""
    return max(400, round(400 * (base_ms or 60000) / 60000))


class SplitMix64:
    def __init__(self, seed: int):
        self.state = seed & M64

    def next(self) -> int:
        self.state = (self.state + 0x9E3779B97F4A7C15) & M64
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & M64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & M64
        return z ^ (z >> 31)

# internal event kinds
_FLAG = "flag_check"
_DECIDE = "decide"
_LAND = "land"
_WAKE = "wake"
_DELIVER = "deliver"
_SITOFF = "sitoff"

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
        max_plies: int | None = 400,
        charge_wall_time: bool = False,
        on_event: Callable[[dict], None] | None = None,
        min_move_ms: int = 0,
        lag_max_ms: int = 0,
        seed: int = 0,
        score_aborts: bool = False,
        sitoff_ms: int | None = None,
        sitoff_unadjudicated: float = 0.1,
    ) -> None:
        missing = [s for s in SEATS if s not in players]
        if missing:
            raise ValueError(f"missing players for seats: {missing}")
        self.state = state
        self.players = players
        self.game_id = game_id
        self.setup = setup
        self.channel_delay_ms = int(channel_delay_ms)
        self.max_plies = scaled_max_plies(getattr(state, "base_ms", None)) if max_plies is None else int(max_plies)
        self.min_move_ms = max(0, int(min_move_ms))
        self.lag_max_ms = max(0, int(lag_max_ms))
        self.seed = int(seed)
        self.score_aborts = score_aborts
        self.sitoff_ms = None if sitoff_ms is None else max(1, int(sitoff_ms))
        self._lag_rng = SplitMix64(self.seed ^ _LAG_STREAM)
        # Whether this game adjudicates sit-offs: a seeded per-game draw keeps a share running to the flag.
        self.adjudicates_sitoffs = (self.sitoff_ms is not None and
                                    SplitMix64(self.seed ^ _SITOFF_STREAM).next() % 1_000_000
                                    >= round(sitoff_unadjudicated * 1_000_000))
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
        self._lent_until: dict[str, int] = {s: 0 for s in SEATS}  # seat's machine lent to its partner
        self._lent_share: dict[str, float] = {s: 0.0 for s in SEATS}  # how much of it, while lent
        self._sit_search_until: dict[str, int] = {s: 0 for s in SEATS}  # search behind a wait decision
        self._last_move_ms: dict[str, int] = {b: 0 for b in BOARDS}  # when the last move landed on each board
        self._pass_since: dict[str, int | None] = {s: None for s in SEATS}  # sitting (not thinking) since
        self._sitoff_gen = 0
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
            out = [b for b in BOARDS if self.state.clocks_ms[self.state.to_move(b)] <= 0]
            teams = {TEAM_OF[self.state.to_move(b)] for b in out}
            if len(out) == 2 and len(teams) == 2:
                res = GameResult(result="draw", reason="double_flag")
            elif out:  # one seat, or two partners: their team flagged
                seat = self.state.to_move(out[0])
                res = GameResult(result=TEAM_OF[OPPONENT[seat]], reason="flag", board=out[0], loser_seat=seat)
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

    def _on_turn_result(self, reason: str) -> GameResult:
        """Score by the two on-turn clocks: the lower would flag first if nobody moved; equal is a draw when the
        two on-turn seats are on different teams. If both are partners (same team), that team flags whichever
        clock runs out first, so it loses (board A's seat is named)."""
        a, b = (self.state.to_move(x) for x in BOARDS)
        ca, cb = int(self.state.clocks_ms[a]), int(self.state.clocks_ms[b])
        if TEAM_OF[a] == TEAM_OF[b]:
            loser = a if ca <= cb else b
            return GameResult(result=TEAM_OF[OPPONENT[loser]], reason=reason, board=BOARD_OF[loser],
                              loser_seat=loser)
        if ca == cb:
            return GameResult(result="draw", reason=reason)
        loser = a if ca < cb else b
        return GameResult(result=TEAM_OF[OPPONENT[loser]], reason=reason, board=BOARD_OF[loser], loser_seat=loser)

    def _arm_sitoff(self) -> None:
        """If both on-turn seats are sitting, schedule the adjudication check for when the mutual pass has
        lasted sitoff_ms. Older checks are invalidated by the generation counter."""
        if not self.adjudicates_sitoffs or self.result is not None:
            return
        self._sitoff_gen += 1
        since = [self._pass_since[self.state.to_move(b)] for b in BOARDS]
        if None in since:
            return
        self._push(max(since) + self.sitoff_ms, _SITOFF, self._sitoff_gen)

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
        # A state forked from a running game (arena.probe_wait) starts at its own clock time.
        self.now = int(getattr(st, "last_ms", 0))
        st.advance_clocks(self.now)
        self._last_move_ms = {b: self.now for b in BOARDS}
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

    def _on_sitoff(self, gen: int) -> None:
        if gen != self._sitoff_gen:
            return
        since = [self._pass_since[self.state.to_move(b)] for b in BOARDS]
        if None in since or self.now < max(since) + self.sitoff_ms:
            return
        self._end(self._on_turn_result("sitoff_adjudicated"))

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
            partner_machine_free=self._machine_free(PARTNER[seat]),
            last_move_ms=dict(self._last_move_ms),
            reps=({b: st.repetition_count(b) for b in BOARDS} if hasattr(st, "repetition_count") else None),
        )

    def _machine_free(self, seat: str) -> bool:
        return (self._status[seat] != _THINKING and self._lent_until[seat] <= self.now
                and self._sit_search_until[seat] <= self.now)

    def _lend(self, seat: str, until: int, share: float) -> None:
        self._lent_until[seat], self._lent_share[seat] = until, share

    def _own_machine_ms(self, seat: str, work_ms: int) -> int:
        """Clock time for work_ms of search on seat's own machine. While part of the machine is lent to the
        partner the rest runs that much slower. A machine lent whole never gets here: the decision is deferred."""
        left = self._lent_until[seat] - self.now
        keep = 1.0 - self._lent_share[seat]
        if left <= 0 or keep >= 1.0 or work_ms <= 0:
            return work_ms
        slow_work = left * keep  # the search done by the time the lent part comes back
        if work_ms <= slow_work:
            return math.ceil(work_ms / keep)
        return left + math.ceil(work_ms - slow_work)

    def _on_decide(self, seat: str, gen: int) -> None:
        if gen != self._gen[seat] or self._status[seat] != _PENDING:
            return
        if self.state.to_move(BOARD_OF[seat]) != seat:  # defensive; should not happen
            self._status[seat] = _IDLE
            return
        if self._lent_until[seat] > self.now and self._lent_share[seat] >= 1.0:
            # our whole machine is searching for the partner: wait for it
            self._push(self._lent_until[seat], _DECIDE, seat, gen)
            return
        obs = self._observe(seat)
        t0 = time.perf_counter()
        dec = self.players[seat].decide(obs)
        wall_ms = int(round((time.perf_counter() - t0) * 1000))

        if dec.kind == "move" and (not obs.legal_moves or dec.uci not in obs.legal_moves):
            # Guard: no legal moves (or a move that is not legal right now) -> sit briefly and re-ask.
            reason = "no_moves" if not obs.legal_moves else "invalid_move"
            dec = Decision(kind="wait", wait_ms=FALLBACK_WAIT_MS, messages=dec.messages, stall_reason=reason)
        if dec.borrow and not obs.partner_machine_free:
            raise ValueError(f"{seat} borrowed its partner's machine while it was busy")
        if dec.borrow and not 0.0 < dec.borrow_share <= 1.0:
            raise ValueError(f"{seat} borrowed {dec.borrow_share} of its partner's machine; must be in (0, 1]")

        self._gen[seat] += 1
        g = self._gen[seat]
        hybrid = {k: int(getattr(dec, k, 0) or 0) for k in ("hybrid_trigger", "hybrid_sit", "hybrid_missed")}
        if any(hybrid.values()):  # spec v6: only hybrid players produce it, so other event files are unchanged
            self._emit("hybrid", seat=seat, **{k[len("hybrid_"):]: v for k, v in hybrid.items() if v})
        if dec.kind == "move":
            cost = self._own_machine_ms(seat, wall_ms if self.charge_wall_time else max(0, int(dec.think_ms)))
            cost = max(cost, self.min_move_ms)
            if self.lag_max_ms:
                cost += self._lag_rng.next() % (self.lag_max_ms + 1)
            self._status[seat] = _THINKING
            self._pass_since[seat] = None
            if dec.borrow:
                self._lend(PARTNER[seat], self.now + cost, dec.borrow_share)
            self._emit(
                "thinking",
                seat=seat,
                phase="start",
                think_ms=cost,
                nodes=dec.nodes,
                eval=None,
                **({"capped": True} if getattr(dec, "capped", False) else {}),
                **({"reused_visits": dec.reused_visits} if getattr(dec, "reused_visits", 0) else {}),
                **({"plan_carried": dec.plan_carried} if getattr(dec, "plan_carried", 0) else {}),
                depth=getattr(dec, "depth", None),
                borrowed=dec.borrow,
            )
            self._push(self.now + cost, _LAND, seat, g, dec, self.now)
            self._arm_sitoff()
        else:
            planned = max(0, int(dec.think_ms))
            slowed = self._own_machine_ms(seat, planned) - planned
            self._sit_search_until[seat] = self.now + planned + slowed
            if dec.borrow:
                self._lend(PARTNER[seat], self._sit_search_until[seat], dec.borrow_share)
            self._status[seat] = _WAITING
            if self._pass_since[seat] is None:
                self._pass_since[seat] = self.now
            if not self._sitting[seat]:
                self._sitting[seat] = True
                self._emit("stall", seat=seat, active=True, reason=dec.stall_reason)
            self._send(seat, dec.messages)
            # a delivery inside _send (delay 0) may have already woken us; only schedule if still waiting
            if self._status[seat] == _WAITING and self._gen[seat] == g:
                self._push(self.now + max(1, int(dec.wait_ms) + slowed), _WAKE, seat, g)
            self._arm_sitoff()

    def _on_land(self, seat: str, gen: int, dec: Decision, started: int) -> None:
        if gen != self._gen[seat] or self._status[seat] != _THINKING:
            return
        st = self.state
        board = BOARD_OF[seat]
        spent = self.now - started
        end_fields = dict(seat=seat, phase="end", think_ms=spent, nodes=dec.nodes,
                          eval=dec.eval, depth=getattr(dec, "depth", None), borrowed=dec.borrow)
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
        self._pass_since[seat] = None
        self.plies += 1
        self._last_move_ms[board] = self.now
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
            self._end(self._on_turn_result("max_plies") if self.score_aborts
                      else GameResult(result="draw", reason="max_plies"))
            return
        self._schedule_flag_check(board)
        self._wake_all_waiting()
        nxt = st.to_move(board)
        if self._status[nxt] == _IDLE:
            self._start_turn(nxt)
        self._arm_sitoff()
