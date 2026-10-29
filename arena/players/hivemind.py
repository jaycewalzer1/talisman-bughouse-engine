"""Hivemind (C++ MCTS + RISEv3.3 network) over UCI.

Spike findings (2026-10-29, macOS M4 Max, ONNX Runtime CPU backend, ``build-ninja/hivemind``,
``models/hivemind-fp32.onnx``)
-----------------------------------------------------------------------------------------------------
* Launch: ``hivemind --model <onnx>`` (no subcommand = UCI loop). ``uci`` -> option list + ``uciok``
  (~0.2 s warm, ~1.3 s cold); ``isready`` -> ``livenodes N`` + ``readyok``. 4 search threads, batch 8.
* Position: ``position fen <fenA> | <fenB>`` -- both boards in ONE command separated by ``|``. Our bracket
  form (``...RNBQKBNR[Nn] b KQkq - 0 1``, ``[]`` for empty pockets, ``~`` promoted markers) is accepted
  as-is by the vendored Fairy-Stockfish ``bughouse`` variant; no conversion to ``placement/pocket`` is
  needed. ``position startpos`` sets both boards. Optional ``moves 1e2e4 2d7d5`` (board-prefixed) exists
  but we always send full FENs.
* The engine plays for a TEAM, not a seat: ``setoption name Team value white`` = board-A white + board-B
  black (our T1); ``black`` = board-A black + board-B white (our T2).
* ``go nodes N`` returns a JOINT action: ``bestmove (<moveA>,<moveB>) [ponder (...)]`` where an entry is
  ``pass`` when the team does not move on that board (not its turn, or it chooses to sit).
  ``bestmove (none)`` when the team is not on turn on either board / no root. Moves are UCI; drops
  ``N@f3``; castling ``e1g1``. ``info depth D score cp X|mate M nodes N nps .. time T pv (a,b) (a,b) ...``
  precedes it; cp = 180*tan(1.56*Q), Q in [-1,1] from the team's perspective, so Q = atan(cp/180)/1.56.
* Seat mapping: a seat plays the entry for its own board. ``pass`` there while the seat is to move
  means the engine wants to sit -> we return a ``wait`` Decision; a move for the partner's board is
  ignored by an independent seat (the partner decides for itself).
* ``RequireMoveOn A|B|none`` removes joint actions that pass on that board when a move exists there,
  i.e. it forces a move for one seat (the other board's entry may still be a move or pass). It does NOT
  restrict the search to one board: the engine always searches joint actions of both boards.
  A forced mate on the other board can still be returned with ``pass`` on the required board.
* ``TimeAdvantage true|false`` is a network input plane (team may sit); it swings the root eval a lot
  (startpos-like: about -270 cp with false, +250 cp with true), so we set it every search from the clocks.
* After ``bestmove`` the engine keeps searching in the background ("permanent brain") until the next
  ``position``/``go``/``stop``. We send ``stop`` right after ``bestmove`` so no free compute leaks
  between decisions, and by default ``ucinewgame`` before every search (clears tree reuse and the
  transposition table) so every decision gets exactly its node budget.
* Timing: ``go nodes N`` overshoots by about one batch per thread (100 -> ~146 nodes, 200 -> ~241,
  400 -> ~441). ``measure_ms_per_node`` (startpos, incl. ucinewgame/isready round trips), wall ms per
  *requested* node: 50: 11.3, 100: 7.4, 200: 5.8, 400: 5.2, 800: 4.7 -- i.e. roughly
  ``400 ms + 4.2 ms * N`` (~215-240 searched nodes/s, network-bound). ``MS_PER_NODE = 5.5`` below is the
  fixed proportional cost model, fitted to the 200-400 node operating range; ``go nodes 100`` takes
  ~0.65-0.75 s of wall time.
* Searches are multi-threaded and therefore not bit-for-bit deterministic.
"""

from __future__ import annotations

import math
import os
import queue
import re
import subprocess
import threading
import time
import random
from dataclasses import dataclass, field

import chess

from arena.interfaces import BOARD_OF, PARTNER, TEAM_OF, Decision, Message, Observation
from arena.players.heuristic import (
    LETTER,
    PTYPE,
    evaluate_moves,
    parse_board,
    team_time_edge,
)

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ENGINE_DIR = os.path.join(REPO_ROOT, "third_party", "hivemind", "engine")
DEFAULT_BINARY = os.environ.get("HIVEMIND_BINARY", os.path.join(ENGINE_DIR, "build-ninja", "hivemind"))
DEFAULT_MODEL = os.environ.get("HIVEMIND_MODEL", os.path.join(ENGINE_DIR, "models", "hivemind-fp32.onnx"))

# Fixed cost model: simulated think time per *requested* node (ms). Measured with measure_ms_per_node
# at 200-400 nodes on the M4 Max CPU build (5.8 / 5.2); see module docstring.
MS_PER_NODE = 5.5

TEAM_OPTION = {"T1": "white", "T2": "black"}


def hivemind_available(binary: str = DEFAULT_BINARY, model: str = DEFAULT_MODEL) -> bool:
    return os.path.isfile(binary) and os.access(binary, os.X_OK) and os.path.isfile(model)


def cp_to_q(cp: float) -> float:
    return max(-1.0, min(1.0, math.atan(cp / 180.0) / 1.56))


@dataclass
class SearchResult:
    move_a: str | None  # None = pass / no move on board A
    move_b: str | None
    q: float | None  # team perspective, [-1, 1]
    cp: int | None
    mate: int | None
    nodes: int  # nodes the engine reports having searched
    depth: int | None
    pv: list[tuple[str, str]] = field(default_factory=list)
    wall_ms: int = 0
    lines: list[str] = field(default_factory=list)

    def move_for(self, board: str) -> str | None:
        return self.move_a if board == "A" else self.move_b


_INFO_RE = re.compile(r"^info depth (\d+) score (cp|mate) (-?\d+) nodes (\d+)")


def _parse_joint(tok: str) -> tuple[str | None, str | None] | None:
    tok = tok.strip()
    if not (tok.startswith("(") and tok.endswith(")")) or "," not in tok:
        return None
    a, b = tok[1:-1].split(",", 1)
    return (None if a == "pass" else a), (None if b == "pass" else b)


class HivemindProcess:
    """One Hivemind engine subprocess. Not thread-safe across concurrent searches (guarded by a lock).
    Started lazily on first use. ``close()`` is idempotent; shared users call ``release()``."""

    def __init__(self, binary: str = DEFAULT_BINARY, model: str = DEFAULT_MODEL, *, options: dict | None = None,
                 reset_tree: bool = True, timeout_s: float = 120.0, refs: int = 1) -> None:
        self.binary = binary
        self.model = model
        self.extra_options = dict(options or {})
        self.reset_tree = reset_tree
        self.timeout_s = timeout_s
        self._proc: subprocess.Popen | None = None
        self._q: queue.Queue[str | None] = queue.Queue()
        self._lock = threading.Lock()
        self._opts: dict[str, str] = {}
        self._refs = refs
        self.closed = False

    # ------------------------------------------------------------------ process plumbing

    def start(self) -> None:
        if self._proc is not None:
            return
        if self.closed:
            raise RuntimeError("HivemindProcess is closed")
        if not hivemind_available(self.binary, self.model):
            raise FileNotFoundError(f"hivemind binary/model missing: {self.binary} / {self.model}")
        self._proc = subprocess.Popen([self.binary, "--model", self.model], stdin=subprocess.PIPE,
                                      stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1,
                                      cwd=ENGINE_DIR)
        threading.Thread(target=self._reader, daemon=True).start()
        self._send("uci")
        self._wait_for(lambda l: l == "uciok")
        self._opts = {}
        for k, v in {"Ponder": "false", **self.extra_options}.items():
            self.set_option(k, v)
        self._sync()

    def _reader(self) -> None:
        assert self._proc is not None and self._proc.stdout is not None
        for line in self._proc.stdout:
            self._q.put(line.rstrip("\n"))
        self._q.put(None)

    def _send(self, cmd: str) -> None:
        assert self._proc is not None and self._proc.stdin is not None
        self._proc.stdin.write(cmd + "\n")
        self._proc.stdin.flush()

    def _wait_for(self, pred, collect: list[str] | None = None) -> str:
        deadline = time.monotonic() + self.timeout_s
        while True:
            left = deadline - time.monotonic()
            if left <= 0:
                raise TimeoutError("hivemind did not answer in time")
            try:
                line = self._q.get(timeout=left)
            except queue.Empty:
                continue
            if line is None:
                raise RuntimeError("hivemind process exited")
            if collect is not None:
                collect.append(line)
            if pred(line):
                return line

    def _sync(self) -> None:
        self._send("isready")
        self._wait_for(lambda l: l == "readyok")

    def set_option(self, name: str, value) -> None:
        value = str(value).lower() if isinstance(value, bool) else str(value)
        if self._opts.get(name) == value:
            return
        self._send(f"setoption name {name} value {value}")
        self._opts[name] = value

    # ------------------------------------------------------------------ search

    def search(self, fen_a: str, fen_b: str, *, team: str, nodes: int, time_advantage: bool = False,
               require: str | None = None) -> SearchResult:
        """Joint search for ``team`` ("T1"/"T2") from the two bracket-form FENs."""
        with self._lock:
            self.start()
            self.set_option("Team", TEAM_OPTION[team])
            self.set_option("TimeAdvantage", bool(time_advantage))
            self.set_option("RequireMoveOn", require or "none")
            if self.reset_tree:
                self._send("ucinewgame")
            self._send(f"position fen {fen_a} | {fen_b}")
            self._sync()  # drains stale output (late info strings) before the search
            lines: list[str] = []
            t0 = time.perf_counter()
            self._send(f"go nodes {max(1, int(nodes))}")
            best = self._wait_for(lambda l: l.startswith("bestmove"), collect=lines)
            wall_ms = int((time.perf_counter() - t0) * 1000)
            self._send("stop")  # end the permanent-brain background search
            self._sync()
        return self._parse(lines, best, wall_ms)

    @staticmethod
    def _parse(lines: list[str], best: str, wall_ms: int) -> SearchResult:
        res = SearchResult(None, None, None, None, None, 0, None, wall_ms=wall_ms, lines=lines)
        parts = best.split()
        joint = _parse_joint(parts[1]) if len(parts) > 1 else None
        if joint:
            res.move_a, res.move_b = joint
        for line in reversed(lines):
            m = _INFO_RE.match(line)
            if not m:
                continue
            res.depth = int(m.group(1))
            res.nodes = int(m.group(4))
            val = int(m.group(3))
            if m.group(2) == "cp":
                res.cp, res.q = val, cp_to_q(val)
            else:
                res.mate, res.q = val, (1.0 if val > 0 else -1.0)
            if " pv " in line:
                res.pv = [j for j in (_parse_joint(t) for t in line.split(" pv ", 1)[1].split()) if j]
            break
        return res

    # ------------------------------------------------------------------ lifecycle

    def acquire(self) -> "HivemindProcess":
        self._refs += 1
        return self

    def release(self) -> None:
        self._refs -= 1
        if self._refs <= 0:
            self.close()

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        proc, self._proc = self._proc, None
        if proc is None:
            return
        try:
            self._send_raw(proc, "stop")
            self._send_raw(proc, "quit")
            proc.wait(timeout=5)
        except Exception:
            proc.kill()
            try:
                proc.wait(timeout=5)
            except Exception:
                pass

    @staticmethod
    def _send_raw(proc: subprocess.Popen, cmd: str) -> None:
        if proc.stdin and not proc.stdin.closed:
            proc.stdin.write(cmd + "\n")
            proc.stdin.flush()

    def __del__(self):  # best effort
        try:
            self.close()
        except Exception:
            pass


def measure_ms_per_node(nodes: int = 200, repeats: int = 3, **kw) -> float:
    """Wall ms per requested node from startpos (both boards white to move, team T1)."""
    start = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR[] w KQkq - 0 1"
    p = HivemindProcess(**kw)
    try:
        p.search(start, start, team="T1", nodes=8)  # warm-up
        walls = [p.search(start, start, team="T1", nodes=nodes).wall_ms for _ in range(repeats)]
    finally:
        p.close()
    return sum(walls) / len(walls) / nodes


# ---------------------------------------------------------------------------------------------- players

def _legalize(uci: str | None, legal: list[str]) -> str | None:
    """Map an engine move onto the observation's legal list (castling king->rook to king->target)."""
    if uci is None:
        return None
    if uci in legal:
        return uci
    alt = {"e1h1": "e1g1", "e1a1": "e1c1", "e8h8": "e8g8", "e8a8": "e8c8"}.get(uci)
    if alt in legal:
        return alt
    return None


def _pv_needs(pv: list[tuple[str, str]], board: str, fen: str) -> list[str]:
    """Piece letters our side drops on ``board`` later in the PV (our plies: 0, 2, 4, ...) but does
    not hold now."""
    b = parse_board(fen)
    color = b.turn
    have = {LETTER[pt] for pt in LETTER if b.pockets[color].count(pt) > 0}
    out: list[str] = []
    for i, (ma, mb) in enumerate(pv):
        if i % 2:
            continue
        mv = ma if board == "A" else mb
        if mv and "@" in mv:
            letter = mv.split("@")[0].upper()
            if letter in PTYPE and letter not in have and letter not in out:
                out.append(letter)
    return out


class HivemindPlayer:
    """One seat driven by one engine process (independent setups). The engine searches joint actions
    over both boards for the seat's team; we play only the entry for our own board. A ``pass`` there
    becomes a ``wait`` Decision. ``think_ms = nodes * ms_per_node`` (fixed cost model).

    Sitting: the engine may pass on our board only while the team is ahead on time (``TimeAdvantage``)
    and we have sat fewer than ``max_consecutive_sits`` times in a row; otherwise ``RequireMoveOn`` is
    set to our board. ``comm="share"`` sends the search result to the partner on every decision and
    honours partner ``go`` (forces a move) / ``sit`` (allows sitting) messages; ``comm="none"`` is silent.

    Lending (``lend=True``): when the partner's machine is idle (``Observation.partner_machine_free``),
    the search runs on both machines: twice the nodes at twice the rate, so the same think time. Only
    meaningful when ``ms_per_node`` models half a machine (``compute="throughput"`` in setups).
    """

    kind = "hivemind"

    def __init__(self, seat: str, process: HivemindProcess, nodes: int = 200, ms_per_node: float = MS_PER_NODE,
                 *, comm: str = "none", max_consecutive_sits: int = 2, sit_ms: int = 1000,
                 request_window_ms: int = 3000, lend: bool = False) -> None:
        self.seat = seat
        self.process = process
        self.nodes = int(nodes)
        self.ms_per_node = float(ms_per_node)
        self.comm = comm
        self.max_consecutive_sits = max_consecutive_sits
        self.sit_ms = sit_ms
        self.request_window_ms = request_window_ms
        self.lend = lend
        self.name = f"Hivemind-{seat}-{comm}-{self.nodes}n" + ("-lend" if lend else "")
        self._sits = 0
        self._partner_requests: dict[str, int] = {}
        self._partner_sitting = False
        self._go_sent = False
        self._force_move = False
        self._sit_until = 0
        self._closed = False

    def _msg(self, token: str, text: str | None = None, **payload) -> Message:
        return Message(from_seat=self.seat, to_seat=PARTNER[self.seat], token=token, sent_ms=0, text=text,
                       payload=payload)

    def _read_inbox(self, obs: Observation) -> None:
        if self.comm == "none":
            return
        for m in obs.inbox:
            if m.token.startswith("need_"):
                self._partner_requests[m.token[5:]] = int(m.payload.get("deadlineTs", obs.now_ms + 3000))
            elif m.token == "stall":
                self._partner_sitting, self._go_sent = True, False
            elif m.token == "go":
                self._force_move = True
            elif m.token == "sit":
                self._sit_until = int(m.payload.get("deadlineTs", obs.now_ms + 2000))
        partner = PARTNER[self.seat]
        if obs.to_move.get(BOARD_OF[partner]) != partner:
            self._partner_sitting = False
        self._partner_requests = {k: v for k, v in self._partner_requests.items() if v >= obs.now_ms}

    def decide(self, obs: Observation) -> Decision:
        self._read_inbox(obs)
        if not obs.legal_moves:
            return Decision(kind="wait", wait_ms=200, stall_reason="no_moves")
        team = TEAM_OF[self.seat]
        board = BOARD_OF[self.seat]
        edge = team_time_edge(obs.clocks_ms, team)
        ahead = edge > 0
        may_sit = (ahead or self._sit_until > obs.now_ms) and not self._force_move \
            and self._sits < self.max_consecutive_sits
        borrow = self.lend and obs.partner_machine_free
        nodes = self.nodes * 2 if borrow else self.nodes
        res = self.process.search(obs.fens["A"], obs.fens["B"], team=team, nodes=nodes,
                                  time_advantage=ahead, require=None if may_sit else board)
        think = int(round(nodes * self.ms_per_node / (2 if borrow else 1)))
        return self._finish(obs, res, think, edge, nodes, borrow)

    def _finish(self, obs: Observation, res: SearchResult, think: int, edge: int, nodes: int,
                borrow: bool = False) -> Decision:
        board = BOARD_OF[self.seat]
        move = _legalize(res.move_for(board), list(obs.legal_moves))
        engine_passed = res.move_for(board) is None
        msgs: list[Message] = []
        needs = _pv_needs(res.pv, board, obs.fens[board]) if self.comm != "none" else []
        if self.comm != "none":
            for letter in needs[:1]:
                msgs.append(self._msg(f"need_{letter}", f"need {letter}", piece=letter,
                                      deadlineTs=obs.now_ms + think + self.request_window_ms,
                                      value=round(abs(res.q or 0.0), 3)))
            if move is not None and self._partner_sitting and edge < 0 and not self._go_sent:
                msgs.append(self._msg("go", "we're behind on time, move"))
                self._go_sent = True
            if move is not None:
                b = parse_board(obs.fens[board])
                mv = chess.Move.from_uci(move)
                if not mv.drop and b.piece_at(mv.to_square) is not None:
                    letter = LETTER.get(b.piece_type_at(mv.to_square))
                    if letter in self._partner_requests:
                        msgs.append(self._msg(f"have_{letter}", f"{letter} coming", piece=letter))
                        self._partner_requests.pop(letter, None)
        if move is None and engine_passed:
            self._sits += 1
            if self.comm != "none":
                msgs.append(self._msg("stall", "sitting", deadlineTs=obs.now_ms + think + self.sit_ms))
                if self.comm == "share":
                    msgs.append(self._status(None, res, needs))
            return Decision(kind="wait", think_ms=think, wait_ms=think + self.sit_ms, nodes=nodes, eval=res.q,
                            messages=tuple(msgs), stall_reason="time_advantage" if edge > 0 else "engine_pass",
                            borrow=borrow)
        if move is None:  # engine move not legal in the arena (e.g. rules mismatch): 1-ply fallback
            b = parse_board(obs.fens[board])
            move = evaluate_moves(b, list(obs.legal_moves), wanted=None, rng=random.Random(obs.ply))[0].uci
        self._sits = 0
        self._force_move = False
        if self.comm == "share":
            msgs.append(self._status(move, res, needs))
        return Decision(kind="move", uci=move, think_ms=think, nodes=nodes, eval=res.q, messages=tuple(msgs),
                        borrow=borrow)

    def _status(self, move: str | None, res: SearchResult, needs: list[str]) -> Message:
        q = res.q if res.q is not None else 0.0
        pv = [f"({a or 'pass'},{b or 'pass'})" for a, b in res.pv[:4]]
        return self._msg("ok", f"plan {move or 'sit'} ({q:+.2f})", kind="status", best=move, eval=res.q,
                         pv=pv, needs=needs)

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self.process.release()


class HivemindTeamController:
    """Centralized controller: ONE engine process per team searching joint actions with the team's full
    node budget. Each seat's ``decide`` asks the controller, which searches (or reuses the joint result it
    computed for exactly the same two positions, e.g. when both seats are asked at the same instant) and
    hands back the entry for that seat's board. A reused result costs only the remaining part of the
    original search time (``ready_ms - now``), so both seats' moves land when the one search finishes."""

    def __init__(self, team: str, process: HivemindProcess, nodes: int = 400, ms_per_node: float = MS_PER_NODE,
                 *, max_consecutive_sits: int = 2, sit_ms: int = 1000) -> None:
        self.team = team
        self.process = process
        self.nodes = int(nodes)
        self.ms_per_node = float(ms_per_node)
        self.max_consecutive_sits = max_consecutive_sits
        self.sit_ms = sit_ms
        self._cache: tuple[tuple[str, str, bool], SearchResult, int] | None = None
        self._sits: dict[str, int] = {}
        self._seats: dict[str, "CentralizedHivemindSeat"] = {}

    def player(self, seat: str) -> "CentralizedHivemindSeat":
        if seat not in self._seats:
            if self._seats:  # the process starts with one reference; each further seat view adds one
                self.process.acquire()
            self._seats[seat] = CentralizedHivemindSeat(seat, self)
        return self._seats[seat]

    def decide(self, seat: str, obs: Observation) -> Decision:
        if not obs.legal_moves:
            return Decision(kind="wait", wait_ms=200, stall_reason="no_moves")
        board = BOARD_OF[seat]
        edge = team_time_edge(obs.clocks_ms, self.team)
        ahead = edge > 0
        sits = self._sits.get(seat, 0)
        require = None if (ahead and sits < self.max_consecutive_sits) else board
        key = (obs.fens["A"], obs.fens["B"], ahead)
        full = int(round(self.nodes * self.ms_per_node))
        cached = self._cache
        if cached and cached[0] == key and _legalize(cached[1].move_for(board), list(obs.legal_moves)) \
                and (require is None or cached[1].move_for(board) is not None):
            res, ready = cached[1], cached[2]
            think, nodes = max(0, ready - obs.now_ms), 0
        else:
            res = self.process.search(obs.fens["A"], obs.fens["B"], team=self.team, nodes=self.nodes,
                                      time_advantage=ahead, require=require)
            think, nodes = full, self.nodes
            self._cache = (key, res, obs.now_ms + full)
        move = _legalize(res.move_for(board), list(obs.legal_moves))
        if move is None and res.move_for(board) is None:
            self._sits[seat] = sits + 1
            return Decision(kind="wait", think_ms=think, wait_ms=think + self.sit_ms, nodes=nodes, eval=res.q,
                            stall_reason="time_advantage" if ahead else "engine_pass")
        if move is None:
            b = parse_board(obs.fens[board])
            move = evaluate_moves(b, list(obs.legal_moves), wanted=None, rng=random.Random(obs.ply))[0].uci
        self._sits[seat] = 0
        return Decision(kind="move", uci=move, think_ms=think, nodes=nodes, eval=res.q)


class CentralizedHivemindSeat:
    kind = "hivemind"

    def __init__(self, seat: str, controller: HivemindTeamController) -> None:
        self.seat = seat
        self.controller = controller
        self.name = f"Hivemind-{seat}-central-{controller.nodes}n"
        self._closed = False

    def decide(self, obs: Observation) -> Decision:
        return self.controller.decide(self.seat, obs)

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self.controller.process.release()
