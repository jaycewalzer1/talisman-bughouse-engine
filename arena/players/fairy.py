"""Fairy-Stockfish bughouse over CECP (XBoard) as an arena opponent: the external floor (audit eval 9, D3 item 4).

Binary: ``third_party/Fairy-Stockfish/src/stockfish`` (``make -C third_party/Fairy-Stockfish/src build
ARCH=apple-silicon``; NOTES.md). One process per seat: Fairy plays one board and talks to its partner with
``tellics ptell <word>``, as on FICS. Probed 2026-11-17 on this build (Fairy-Stockfish 181026):

* Handshake ``xboard`` / ``protover 2`` (features, ``done=1``), ``variant bughouse``, ``easy`` (no pondering),
  ``partner fairy`` (so it ptells in its engine dialect and does not chat), ``cores 1``.
* Each decision: ``new`` (clears the hash and the partner state, so nothing carries over between searches; the
  partner state we know is re-sent below), ``variant bughouse``, ``force``, ``setboard <bracket FEN>`` (the
  arena's FEN, holdings in ``[...]``), ``level 0 <base min> 0``, ``time``/``otim`` (our and our opponent's clocks,
  centiseconds), ``ptell time``/``ptell otim`` (the partner's and the partner's opponent's clocks, which Fairy's
  sitting rules compare), ``ptell dead``/``ptell fast`` when the partner told us so, then ``go`` -> ``move <m>``.
  Thinking lines are ``depth score time(cs) nodes seldepth nps tbhits <tab> pv``.
* Time: Fairy's own time manager on wall time. The charged think time is the engine's wall time for the search
  (``charge="wall"``, default) or ``nodes x ms_per_node`` (``charge="nodes"``, a fixed cost model). Fairy's
  ``nodestime`` option (nodes as time) was tried and stops after a few hundred nodes on this build, so it is not used.
* Sitting is never left to the engine (its sit loops busy-wait on the wall clock, which the simulated arena cannot
  give it). Instead the wrapper reads Fairy's own ptells from each search: ``dead`` (it is mated without help) ->
  this seat sits (a wait until the next move lands, at most ``sit_ms``) while its clock has more than 1 s, as Fairy
  would; ``sit`` -> the PARTNER seat sits until told ``x``/``go`` or its clock falls under 1 s (Fairy's ``sit`` when it
  mates by force and the opponent is shorter on time). Every ptell is relayed to the partner as an arena message
  (token ``fairy_<word>``); the partner re-sends ``dead``/``fast`` to its engine before its next search.
"""

from __future__ import annotations

import os
import queue
import random
import re
import subprocess
import threading
import time

from arena.interfaces import BOARD_OF, OPPONENT, PARTNER, Decision, Message, Observation
from arena.players.heuristic import evaluate_moves, parse_board, team_seats

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FAIRY_BINARY = os.environ.get("FAIRY_BINARY",
                              os.path.join(REPO_ROOT, "third_party", "Fairy-Stockfish", "src", "stockfish"))
FAIRY_MS_PER_NODE = 0.0005  # ~2 M nodes/s single thread on the M4 Max (bench 1.8 M, bughouse startpos 2.0 M)
SIT_FLOOR_MS = 1000  # Fairy sits only while its clock has more than this left (search.cpp sit loop)
_THINK_RE = re.compile(r"^\s*(\d+)\s+(-?\d+)\s+(\d+)\s+(\d+)")
XBOARD_MATE = 100000  # Fairy prints mate scores as +-(XBOARD_VALUE_MATE + moves), far above any cp score


def fairy_available(binary: str = FAIRY_BINARY) -> bool:
    return os.path.isfile(binary) and os.access(binary, os.X_OK)


class FairyResult:
    def __init__(self) -> None:
        self.move: str | None = None
        self.score: int | None = None  # centipawns from the mover's side; mates beyond +-XBOARD_MATE
        self.depth = 0
        self.nodes = 0
        self.wall_ms = 0
        self.ptells: list[str] = []
        self.lines: list[str] = []
        self.wants_piece = False  # it waited for a virtual drop: the wrapper sits instead


class FairyProcess:
    """One Fairy-Stockfish process in XBoard mode, playing one bughouse board."""

    def __init__(self, binary: str = FAIRY_BINARY, *, threads: int = 1, hash_mb: int = 64,
                 timeout_s: float = 120.0) -> None:
        self.binary = binary
        self.threads = int(threads)
        self.hash_mb = int(hash_mb)
        self.timeout_s = timeout_s
        self._proc: subprocess.Popen | None = None
        self._q: queue.Queue[str | None] = queue.Queue()
        self.closed = False

    def start(self) -> None:
        if self._proc is not None:
            return
        if self.closed:
            raise RuntimeError("FairyProcess is closed")
        if not fairy_available(self.binary):
            raise FileNotFoundError(f"Fairy-Stockfish binary missing: {self.binary}")
        self._proc = subprocess.Popen([self.binary], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                      stderr=subprocess.DEVNULL, text=True, bufsize=1)
        threading.Thread(target=self._reader, daemon=True).start()
        self._send("xboard")
        self._send("protover 2")
        self._wait_for(lambda l: "done=1" in l)
        for cmd in ("variant bughouse", "easy", "partner fairy", f"cores {self.threads}", f"memory {self.hash_mb}"):
            self._send(cmd)
        self._ping()

    def _reader(self) -> None:
        assert self._proc is not None and self._proc.stdout is not None
        for line in self._proc.stdout:
            self._q.put(line.rstrip("\n"))
        self._q.put(None)

    def _send(self, cmd: str) -> None:
        assert self._proc is not None and self._proc.stdin is not None
        self._proc.stdin.write(cmd + "\n")
        self._proc.stdin.flush()

    def _wait_for(self, pred, collect: list[str] | None = None, timeout_s: float | None = None) -> str:
        deadline = time.monotonic() + (timeout_s or self.timeout_s)
        while True:
            left = deadline - time.monotonic()
            if left <= 0:
                raise TimeoutError("fairy-stockfish did not answer in time")
            try:
                line = self._q.get(timeout=left)
            except queue.Empty:
                continue
            if line is None:
                raise RuntimeError("fairy-stockfish exited")
            if collect is not None:
                collect.append(line)
            if pred(line):
                return line

    def _ping(self, n: int = 1) -> None:
        self._send(f"ping {n}")
        self._wait_for(lambda l: l.strip() == f"pong {n}")

    def think(self, fen: str, *, base_ms: int, own_ms: int, opp_ms: int, partner_ms: int, partner_opp_ms: int,
              partner_tells: tuple[str, ...] = ()) -> FairyResult:
        """Search ``fen`` (side to move = this seat) under Fairy's own time manager; return its move and ptells."""
        self.start()
        self._send("new")
        self._send("variant bughouse")
        self._send("force")
        self._send(f"setboard {fen}")
        self._send(f"level 0 {max(1, round(base_ms / 60000))} 0")
        self._send(f"time {max(1, own_ms // 10)}")
        self._send(f"otim {max(1, opp_ms // 10)}")
        self._send(f"ptell time {max(0, partner_ms // 10)}")
        self._send(f"ptell otim {max(0, partner_opp_ms // 10)}")
        for word in partner_tells:
            self._send(f"ptell {word}")
        self._ping(2)  # all of the above applied before go
        res = FairyResult()
        t0 = time.perf_counter()
        self._send("go")
        stopped = []

        def done(l: str) -> bool:
            # Fairy busy-waits on the wall clock after announcing "dead" (it sits for its partner) and while waiting
            # for a virtual drop. The wrapper models both as arena waits, so release the engine at once: "ptell dead"
            # (partner dead) ends both loops. Not "?": that blocks Fairy's input thread until the search returns.
            if l.startswith("tellics ptell dead") and not stopped:
                stopped.append(True)
                self._send("ptell dead")
            # "fast" makes its wait for a virtual drop (a piece it does not hold yet) end at once; the wrapper turns
            # that wait into an arena sit (FairyResult.wants_piece) instead.
            if l.startswith("tellics ptell fast"):
                self._send("ptell fast")
            return l.startswith(("move ", "resign", "1-0", "0-1", "1/2"))

        try:
            line = self._wait_for(done, collect=res.lines, timeout_s=max(5.0, own_ms / 1000 + 5))
        except TimeoutError:
            self._send("?")  # move now (stops a sit loop, if one ever starts)
            line = self._wait_for(lambda l: l.startswith("move "), collect=res.lines, timeout_s=10)
        res.wall_ms = int((time.perf_counter() - t0) * 1000)
        self._send("force")
        if line.startswith("move "):
            res.move = line.split()[1]
        after_last_think: list[str] = []
        for l in res.lines:
            if l.startswith("tellics ptell "):
                word = l[len("tellics ptell "):].strip()
                res.ptells.append(word)
                after_last_think.append(word.split()[0] if word else "")
                continue
            m = _THINK_RE.match(l)
            if m:
                res.depth, res.score, res.nodes = int(m.group(1)), int(m.group(2)), int(m.group(4))
                after_last_think = []
        # search.cpp: after the search, a best move that drops a piece not in hand makes Fairy ptell "fast", wait for
        # the piece, then ptell "x" and play its best real move.
        res.wants_piece = "fast" in after_last_think
        return res

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        proc, self._proc = self._proc, None
        if proc is None:
            return
        try:
            proc.stdin.write("quit\n")
            proc.stdin.flush()
            proc.wait(timeout=5)
        except Exception:
            proc.kill()


def holdings_fen(fen: str) -> str:
    """The arena's bracket FEN is what Fairy's setboard reads for bughouse; promoted markers ``~`` included."""
    return fen


class FairyPlayer:
    """One seat played by its own Fairy-Stockfish process (an independent, ptell-talking seat)."""

    kind = "fairy"

    def __init__(self, seat: str, process: FairyProcess, *, base_ms: int = 60000, sit_ms: int = 1000,
                 charge: str = "wall", ms_per_node: float = FAIRY_MS_PER_NODE) -> None:
        if charge not in ("wall", "nodes"):
            raise ValueError(f"charge must be 'wall' or 'nodes', got {charge!r}")
        self.seat = seat
        self.process = process
        self.base_ms = int(base_ms)
        self.sit_ms = int(sit_ms)
        self.charge = charge
        self.ms_per_node = float(ms_per_node)
        self.name = f"Fairy-{seat}"
        self.searches = self.sits = self.fallback_moves = 0
        self.nodes_total = self.wall_ms_total = 0
        self._dead = False  # our last search said we are mated without help
        self._partner_sit = False  # partner asked us to sit
        self._partner_state: dict[str, bool] = {"dead": False, "fast": False}

    def _msg(self, word: str) -> Message:
        return Message(from_seat=self.seat, to_seat=PARTNER[self.seat], token=f"fairy_{word.split()[0]}", sent_ms=0,
                       text=f"ptell {word}", payload={"ptell": word})

    def _read_inbox(self, obs: Observation) -> None:
        for m in obs.inbox:
            word = (m.payload or {}).get("ptell") or ""
            w = word.split()[0] if word else ""
            if w == "sit":
                self._partner_sit = True
            elif w in ("x", "go"):
                self._partner_sit = False
                self._partner_state["dead"] = False
            elif w == "dead":
                self._partner_state["dead"] = True
            elif w == "fast":
                self._partner_state["fast"] = True
            elif w == "slow":
                self._partner_state["fast"] = False

    def decide(self, obs: Observation) -> Decision:
        self._read_inbox(obs)
        if not obs.legal_moves:
            return Decision(kind="wait", wait_ms=200, stall_reason="no_moves")
        own = int(obs.clocks_ms[self.seat])
        if self._partner_sit and own > SIT_FLOOR_MS:
            self.sits += 1
            return Decision(kind="wait", wait_ms=min(self.sit_ms, own - SIT_FLOOR_MS), stall_reason="partner_sit")
        board = BOARD_OF[self.seat]
        partner = PARTNER[self.seat]
        tells = tuple(w for w, on in self._partner_state.items() if on)
        res = self.process.think(holdings_fen(obs.fens[board]), base_ms=self.base_ms, own_ms=own,
                                 opp_ms=int(obs.clocks_ms[OPPONENT[self.seat]]),
                                 partner_ms=int(obs.clocks_ms[partner]),
                                 partner_opp_ms=int(obs.clocks_ms[OPPONENT[partner]]), partner_tells=tells)
        self.searches += 1
        self.nodes_total += res.nodes
        self.wall_ms_total += res.wall_ms
        think = res.wall_ms if self.charge == "wall" else int(round(res.nodes * self.ms_per_node))
        msgs = tuple(self._msg(w) for w in res.ptells if w.split()[0] in ("sit", "x", "dead", "fast", "slow", "go"))
        words = [w.split()[0] for w in res.ptells]
        if "dead" in words:
            self._dead = True
        if "x" in words:
            self._dead = False
        if res.wants_piece and own - think > int(obs.clocks_ms[OPPONENT[partner]]):
            # Fairy waits for the piece while its clock exceeds the partner's opponent's (search.cpp)
            self.sits += 1
            return Decision(kind="wait", think_ms=think, wait_ms=think + self.sit_ms, nodes=res.nodes,
                            messages=msgs, stall_reason="fairy_wants_piece")
        if self._dead and own - think > SIT_FLOOR_MS:
            self.sits += 1
            return Decision(kind="wait", think_ms=think, wait_ms=think + self.sit_ms, nodes=res.nodes,
                            messages=msgs, stall_reason="fairy_dead")
        move = res.move
        if move not in obs.legal_moves:
            alt = {"e1h1": "e1g1", "e1a1": "e1c1", "e8h8": "e8g8", "e8a8": "e8c8"}.get(move or "")
            move = alt if alt in obs.legal_moves else None
        if move is None:
            self.fallback_moves += 1
            b = parse_board(obs.fens[board])
            move = evaluate_moves(b, list(obs.legal_moves), wanted=None, rng=random.Random(obs.ply))[0].uci
        q = None
        if res.score is not None:
            q = 1.0 if res.score >= XBOARD_MATE else -1.0 if res.score <= -XBOARD_MATE else \
                max(-1.0, min(1.0, res.score / 1000.0))
        return Decision(kind="move", uci=move, think_ms=think, nodes=res.nodes, eval=q, messages=msgs)

    def close(self) -> None:
        self.process.close()


def fairy_team(team: str, *, seed: int = 0, nodes: int = 0, **kw) -> dict:
    """{seat: FairyPlayer} for one team (``kind="fairy"``): two processes, each seat on its own board. ``nodes`` is
    ignored (Fairy runs its own time manager); options ``binary``, ``threads``, ``hash_mb``, ``base_ms``,
    ``sit_ms``, ``charge``, ``ms_per_node``."""
    pkw = {k: kw[k] for k in ("threads", "hash_mb") if kw.get(k) is not None}
    fkw = {k: kw[k] for k in ("base_ms", "sit_ms", "charge", "ms_per_node") if kw.get(k) is not None}
    return {s: FairyPlayer(s, FairyProcess(kw.get("binary") or FAIRY_BINARY, **pkw), **fkw) for s in team_seats(team)}


__all__ = ["FairyProcess", "FairyPlayer", "FairyResult", "fairy_team", "fairy_available", "FAIRY_BINARY"]
