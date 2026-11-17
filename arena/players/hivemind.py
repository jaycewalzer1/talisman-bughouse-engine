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
  ``position``/``go``/``stop``. We send ``stop`` right after ``bestmove`` (without waiting for it) so no free
  compute leaks between decisions, and by default ``ucinewgame`` before every search (clears tree reuse and the
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

import base64
import itertools
import logging
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

from arena.interfaces import BOARD_OF, OPPONENT, PARTNER, SEATS, TEAM_OF, Decision, Message, Observation
from arena.recorder import RecordedSample
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
# The clock-plane build (docs/custom_engine_spec.md section 1): a separate binary, ``-DHIVEMIND_CLOCK_PLANES=ON``.
CLOCK_BINARY_CANDIDATES = tuple(p for p in (
    os.environ.get("HIVEMIND_CLOCK_BINARY"),
    os.path.join(ENGINE_DIR, "build-clock", "hivemind-clock"),
    os.path.join(ENGINE_DIR, "build-ninja-clock", "hivemind-clock"),
    os.path.join(ENGINE_DIR, "build-ninja", "hivemind-clock"),
) if p)
BINARY_NAMES = ("hivemind", "hivemind-clock")


def resolve_binary(binary: str | None) -> str:
    """``None`` or ``"hivemind"``: the stock binary. ``"hivemind-clock"``: the clock-plane build (the first of
    ``CLOCK_BINARY_CANDIDATES`` that exists, else the first candidate). Anything else is a path, used as is."""
    if binary is None or binary == "hivemind":
        return DEFAULT_BINARY
    if binary == "hivemind-clock":
        return next((p for p in CLOCK_BINARY_CANDIDATES if os.path.isfile(p)), CLOCK_BINARY_CANDIDATES[0])
    return binary

# Fixed cost model: simulated think time per *requested* node (ms). Measured with measure_ms_per_node
# at 200-400 nodes on the M4 Max CPU build (5.8 / 5.2); see module docstring.
MS_PER_NODE = 5.5

TEAM_OPTION = {"T1": "white", "T2": "black"}


def hivemind_available(binary: str = DEFAULT_BINARY, model: str = DEFAULT_MODEL) -> bool:
    return os.path.isfile(binary) and os.access(binary, os.X_OK) and os.path.isfile(model)


def cp_to_q(cp: float) -> float:
    return max(-1.0, min(1.0, math.atan(cp / 180.0) / 1.56))


_LOG = logging.getLogger(__name__)


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
    # With MultiPV > 1: (move_a, move_b, q) per reported line, the engine's choice first.
    multi: list[tuple[str | None, str | None, float]] = field(default_factory=list)
    # With ``search(sample=True)``: one serialized TrainingSample (``info string sample <base64>``), decoded.
    sample: bytes | None = None
    # ``bestmove (none)``: the engine found no root (it judged the team mated or not on turn on either board).
    no_root: bool = False
    # ``info string think_gain <pa> <pb>`` (spec v2 B, networks with a think-time head): per board, the chance that
    # a 4x longer search plays a different move there. None when the engine printed no such line.
    think_gain: tuple[float, float] | None = None

    def move_for(self, board: str) -> str | None:
        return self.move_a if board == "A" else self.move_b


_INFO_RE = re.compile(r"^info depth (\d+)(?: multipv (\d+))? score (cp|mate) (-?\d+) nodes (\d+)")
SAMPLE_PREFIX = "info string sample "
THINK_GAIN_PREFIX = "info string think_gain "
CLOCK_SEATS = ("A_white", "A_black", "B_white", "B_black")  # order of the ``clocks`` suffix


def position_command(fen_a: str, fen_b: str, clocks: dict | None = None, elapsed: dict | None = None,
                     pace: dict | None = None) -> str:
    """``position fen <A> | <B>``, with ``clocks <aw> <ab> <bw> <bb>`` (remaining ms, >= 0) when clocks are given,
    and then ``elapsed <ea> <eb>`` (ms since the last move landed on board A / B, >= 0; spec v2 A) when elapsed
    times are given too. Elapsed times without clocks are not sent (the suffix extends the clocks suffix).
    ``pace`` ({seat: ms}, spec v3 contract 1) appends ``pace <paw> <pab> <pbw> <pbb>``, each seat's expected move
    time, again only with clocks. An engine that does not parse it skips the unknown tokens."""
    cmd = f"position fen {fen_a} | {fen_b}"
    if clocks is not None:
        cmd += " clocks " + " ".join(str(max(0, int(clocks[s]))) for s in CLOCK_SEATS)
        if elapsed is not None:
            cmd += " elapsed " + " ".join(str(max(0, int(elapsed[b]))) for b in ("A", "B"))
        if pace is not None:
            cmd += " pace " + " ".join(str(max(0, int(round(pace[s])))) for s in CLOCK_SEATS)
    return cmd


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x)) if x >= 0 else math.exp(x) / (1.0 + math.exp(x))


def parse_think_gain(line: str) -> tuple[float, float] | None:
    """``info string think_gain <pa> <pb>`` -> (pa, pb) as probabilities. Values already in [0, 1] are taken as
    probabilities; a value outside [0, 1] is read as a logit and passed through a sigmoid. Malformed: None."""
    parts = line[len(THINK_GAIN_PREFIX):].split()
    if len(parts) < 2:
        return None
    try:
        vals = [float(parts[0]), float(parts[1])]
    except ValueError:
        return None
    if not all(math.isfinite(v) for v in vals):
        return None
    return tuple(v if 0.0 <= v <= 1.0 else _sigmoid(v) for v in vals)  # type: ignore[return-value]


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
        # The engine runs with cwd=ENGINE_DIR, so a model path relative to the caller's cwd must be made absolute.
        self.model = os.path.abspath(model)
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
               require: str | None = None, multipv: int = 1, clocks: dict | None = None, sample: bool = False,
               options: dict | None = None, require_option: bool = True,
               elapsed: dict | None = None, pace: dict | None = None) -> SearchResult:
        """Joint search for ``team`` ("T1"/"T2") from the two bracket-form FENs. ``multipv`` > 1 also fills
        ``SearchResult.multi`` with that many candidate lines.

        Custom-engine extras, all off by default: ``clocks`` ({seat: ms}) adds the ``clocks`` suffix to
        ``position``; ``sample`` searches with ``go nodes N sample`` and fills ``SearchResult.sample``; ``options``
        are set before the search (e.g. FreePass, TreeMoveMs); ``require_option=False`` never sends RequireMoveOn;
        ``elapsed`` ({"A": ms, "B": ms}, with ``clocks`` only) adds ``elapsed <ea> <eb>`` (spec v2 A); ``pace``
        ({seat: ms}, with ``clocks`` only) adds ``pace <4 ms>`` (spec v3)."""
        with self._lock:
            self.start()
            self.set_option("Team", TEAM_OPTION[team])
            self.set_option("TimeAdvantage", bool(time_advantage))
            if require_option:
                self.set_option("RequireMoveOn", require or "none")
            self.set_option("MultiPV", int(multipv))
            for k, v in (options or {}).items():
                self.set_option(k, v)
            if self.reset_tree:
                self._send("ucinewgame")
            self._send(position_command(fen_a, fen_b, clocks, elapsed, pace))
            self._sync()  # drains stale output (late info strings) before the search
            lines: list[str] = []
            t0 = time.perf_counter()
            self._send(f"go nodes {max(1, int(nodes))}" + (" sample" if sample else ""))
            best = self._wait_for(lambda l: l.startswith("bestmove"), collect=lines)
            wall_ms = int((time.perf_counter() - t0) * 1000)
            # End the permanent-brain background search. No isready round trip here (spec v3 WP-C): the engine
            # handles commands in order, so the stop is done before the next setoption, and the isready before
            # the next 'go' drains any late output.
            self._send("stop")
        res = self._parse(lines, best, wall_ms)
        if sample and res.sample is None and not res.no_root:
            raise RuntimeError("hivemind did not print 'info string sample' (needs the hivemind-clock build)")
        return res

    @staticmethod
    def _parse(lines: list[str], best: str, wall_ms: int) -> SearchResult:
        res = SearchResult(None, None, None, None, None, 0, None, wall_ms=wall_ms, lines=lines)
        parts = best.split()
        joint = _parse_joint(parts[1]) if len(parts) > 1 else None
        res.no_root = joint is None
        if joint:
            res.move_a, res.move_b = joint
        last: dict[int, tuple[re.Match, str]] = {}  # the latest info line per MultiPV index
        for line in lines:
            if line.startswith(SAMPLE_PREFIX):
                res.sample = base64.b64decode(line[len(SAMPLE_PREFIX):].strip(), validate=True)
                continue
            if line.startswith(THINK_GAIN_PREFIX):
                res.think_gain = parse_think_gain(line) or res.think_gain  # the last well-formed line wins
                continue
            m = _INFO_RE.match(line)
            if m:
                last[int(m.group(2) or 1)] = (m, line)
        for index in sorted(last):
            m, line = last[index]
            val = int(m.group(4))
            q = cp_to_q(val) if m.group(3) == "cp" else (1.0 if val > 0 else -1.0)
            pv = [j for j in (_parse_joint(t) for t in line.split(" pv ", 1)[1].split()) if j] if " pv " in line else []
            if pv:
                res.multi.append((pv[0][0], pv[0][1], q))
            if index == min(last):
                res.depth, res.nodes, res.q, res.pv = int(m.group(1)), int(m.group(5)), q, pv
                if m.group(3) == "cp":
                    res.cp = val
                else:
                    res.mate = val
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

CHARGE_MODES = ("requested", "searched")


def _check_charge(charge: str) -> str:
    if charge not in CHARGE_MODES:
        raise ValueError(f"charge must be one of {CHARGE_MODES}, got {charge!r}")
    return charge


def charged_nodes(charge: str, requested: int, res: SearchResult) -> int:
    """Nodes to charge for one search: the requested budget, or with ``charge="searched"`` the nodes the engine
    reports (``res.nodes``; the requested budget when it reported none)."""
    if charge == "searched" and res.nodes > 0:
        return int(res.nodes)
    return int(requested)

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


def pv_offers(pv: list[tuple[str, str]], board: str, fen: str) -> list[tuple[str, int]]:
    """Pieces our side captures on ``board`` in the PV, as (letter, plies from now), earliest first and
    one entry per letter. These can be offered to the partner. A captured promoted piece counts as a
    pawn, as it does in the pocket. Drops are replayed even if the piece is not in the pocket yet."""
    b = parse_board(fen)
    color = b.turn
    out: dict[str, int] = {}
    for i, (ma, mb) in enumerate(pv):
        mv = ma if board == "A" else mb
        if not mv:
            continue
        b.turn = color if i % 2 == 0 else not color
        move = chess.Move.from_uci(mv)
        if i % 2 == 0 and not move.drop:
            target = chess.square(chess.square_file(move.to_square), chess.square_rank(move.from_square)) \
                if b.is_en_passant(move) else move.to_square
            pt = b.piece_type_at(target)
            if pt is not None and b.color_at(target) != color:
                out.setdefault("P" if b.promoted & chess.BB_SQUARES[target] else LETTER.get(pt, "K"), i)
        if move.drop:
            b.pockets[b.turn].add(move.drop)
        b.push(move)
    return sorted(out.items(), key=lambda kv: kv[1])


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
    ``lend_share`` below 1 borrows only that part of the partner's machine (0.5: one and a half times the nodes).

    ``charge="searched"`` (spec v3 WP-C; default ``"requested"``) charges the nodes the engine reports having
    searched instead of the requested budget (``go nodes N`` overshoots by about a batch per thread).
    """

    kind = "hivemind"

    def __init__(self, seat: str, process: HivemindProcess, nodes: int = 200, ms_per_node: float = MS_PER_NODE,
                 *, comm: str = "none", max_consecutive_sits: int = 2, sit_ms: int = 1000,
                 request_window_ms: int = 3000, lend: bool = False, lend_share: float = 1.0,
                 charge: str = "requested") -> None:
        if not 0.0 < lend_share <= 1.0:
            raise ValueError(f"lend_share must be in (0, 1], got {lend_share}")
        self.charge = _check_charge(charge)
        self.seat = seat
        self.process = process
        self.nodes = int(nodes)
        self.ms_per_node = float(ms_per_node)
        self.comm = comm
        self.max_consecutive_sits = max_consecutive_sits
        self.sit_ms = sit_ms
        self.request_window_ms = request_window_ms
        self.lend = lend
        self.lend_share = float(lend_share)
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
        machines = 1 + self.lend_share if borrow else 1  # ours, plus the borrowed part of the partner's
        nodes = int(round(self.nodes * machines))
        res = self.process.search(obs.fens["A"], obs.fens["B"], team=team, nodes=nodes,
                                  time_advantage=ahead, require=None if may_sit else board)
        charged = charged_nodes(self.charge, nodes, res)
        think = int(round(charged * self.ms_per_node / machines))
        return self._finish(obs, res, think, edge, charged, borrow)

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
                            borrow=borrow, borrow_share=self.lend_share)
        if move is None:  # engine move not legal in the arena (e.g. rules mismatch): 1-ply fallback
            b = parse_board(obs.fens[board])
            move = evaluate_moves(b, list(obs.legal_moves), wanted=None, rng=random.Random(obs.ply))[0].uci
        self._sits = 0
        self._force_move = False
        if self.comm == "share":
            msgs.append(self._status(move, res, needs))
        return Decision(kind="move", uci=move, think_ms=think, nodes=nodes, eval=res.q, messages=tuple(msgs),
                        borrow=borrow, borrow_share=self.lend_share)

    def _status(self, move: str | None, res: SearchResult, needs: list[str]) -> Message:
        q = res.q if res.q is not None else 0.0
        pv = [f"({a or 'pass'},{b or 'pass'})" for a, b in res.pv[:4]]
        return self._msg("ok", f"plan {move or 'sit'} ({q:+.2f})", kind="status", best=move, eval=res.q,
                         pv=pv, needs=needs)

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self.process.release()


_SAMPLE_SEQ = itertools.count()  # orders recorded samples that share a timestamp
SAMPLE_TEAM_OFFSET = 16  # TrainingSample: gameId u64, nodes u32, macroPly u16, movesLeft u16, then team u8
SAMPLE_TEAM_BYTE = {"T1": 0, "T2": 1}  # engine Team white = our T1 = selfplay.cc team 0

# Exploration options (spec v3 WP-C, engine side WP-E E3). The OpeningNoise* options exist in the fork's uci.cc
# (OpeningNoisePlies is capped at 200 board plies there). The all-game noise switch and the move temperature are
# WP-E's; their names live here so integration can change them in one place. uci.cc ignores unknown option
# names silently, so sending them to an engine without them is harmless.
NOISE_ALL_GAME_OPTION = "OpeningNoiseAllGame"
TEMPERATURE_OPTION = "MoveTemperaturePermille"
NOISE_ALPHA_PERMILLE = 300  # stock selfplay.h: Dirichlet alpha 0.3, epsilon 0.25
NOISE_EPSILON_PERMILLE = 250
NOISE_PLIES = 200

# Per-seat pace mixture (spec v3 WP-C): (weight, factor). A seat's factor is drawn once per game; every search
# then multiplies it by a mean-1 log-normal draw and adds an exponential lag.
DEFAULT_PACE_MIXTURE = ((0.25, 0.6), (0.5, 1.0), (0.25, 1.6))
DEDUPE_MODES = ("position", "bucket")
NO_ROOT_MODES = ("wait", "heuristic")


def _pair(value, name: str) -> tuple[float, float] | None:
    if value is None:
        return None
    lo, hi = (float(v) for v in value)
    if not 0 < lo <= hi:
        raise ValueError(f"{name} must be (lo, hi) with 0 < lo <= hi, got {value}")
    return lo, hi


def _mixture(value) -> tuple[tuple[float, float], ...] | None:
    if value is None:
        return None
    mix = tuple((float(w), float(f)) for w, f in value)
    if not mix or any(w < 0 or f <= 0 for w, f in mix) or sum(w for w, _ in mix) <= 0:
        raise ValueError(f"pace_mixture must be (weight >= 0, factor > 0) pairs with a positive total, got {value}")
    return mix


def fen_plies(fen: str) -> int | None:
    """Half-moves played on one board, from the FEN's fullmove number and side to move (None if unreadable)."""
    parts = fen.split()
    try:
        return 2 * (int(parts[5]) - 1) + (1 if parts[1] == "b" else 0)
    except (IndexError, ValueError):
        return None


class ControllerSample(RecordedSample):
    """A ``RecordedSample`` whose ``extra`` dict is filled in place after the search: ``sit_ms`` (how long the seat
    then sat, known only once it moves or the game ends; the NamedTuple field cannot change) and the controller's
    own keys ``search_kind`` / ``transition``. ``meta`` merges them into the sidecar line; every other spec v3 key
    is a ``RecordedSample`` field in the recorder's format (seat-ordered float lists)."""

    def meta(self, game_id: int) -> dict:
        out = super().meta(game_id)
        out.update(getattr(self, "extra", None) or {})
        return out


_V3_FIELDS = set(getattr(RecordedSample, "_fields", ()))


def make_sample(extra: dict, **fields) -> ControllerSample:
    """A ControllerSample; spec v3 fields the recorder's RecordedSample does not have go to ``extra`` instead."""
    for k in [k for k in fields if k not in _V3_FIELDS]:
        v = fields.pop(k)
        extra[k] = list(v) if isinstance(v, tuple) else v
    s = ControllerSample(**fields)
    s.extra = extra
    return s


class HivemindTeamController:
    """Centralized controller: ONE engine process per team searching joint actions with the team's full
    node budget. Each seat's ``decide`` asks the controller, which searches (or reuses the joint result it
    computed for exactly the same two positions, e.g. when both seats are asked at the same instant) and
    hands back the entry for that seat's board. A reused result costs only the remaining part of the
    original search time (``ready_ms - now``), so both seats' moves land when the one search finishes.

    ``clock_k``: instead of a fixed ``nodes``, each search gets the seat's own remaining clock divided by
    ``clock_k`` (converted to nodes, clamped to ``min_nodes..max_nodes``): a plain clock-based time manager.
    ``serial``: the team has ONE machine, so a search asked for while the other seat's search is still running
    queues behind it and its think time includes the wait. Off by default: then the two seats can search at
    the same time when their positions differ, which gives the team up to twice its compute once the boards
    stop moving in step.

    ``jitter`` (lo, hi): every new search's charged think time is multiplied by a factor drawn uniformly from
    that range (``seed`` seeds the draws), so two teams do not move in lock step. Off (None) by default.

    ``free=True`` (custom engine, docs/custom_engine_spec.md section 3; needs the ``hivemind-clock`` build):
    no hand-written time or sitting rules. Every ``position`` carries the four clocks; ``FreePass`` and
    ``MateProbeAnyTime`` are set; ``TreeMoveMs`` is this search's own think time; ``RequireMoveOn`` is never
    sent and there is no sit cap. Think time is the seat's remaining clock / k, with k drawn once per
    controller (one game, one team) uniformly from ``k_range`` and clamped like ``clock_k``; searches are
    always ``serial``; ``jitter`` defaults to (0.5, 2.0). A pass on the seat's board becomes a wait that ends
    when the next move (and so any captured piece) lands on either board, or after ``free_sit_ms``, whichever
    is first; then the controller searches again. ``record=True`` searches with ``go nodes N sample`` and keeps
    the samples in ``samples`` as ``arena.recorder.RecordedSample`` (time, order, bytes and the position it came
    from) for arena.recorder.

    Spec v2, free mode only: every ``position`` also carries ``elapsed <ea> <eb>`` (ms since the last move landed
    on each board, from ``Observation.last_move_ms``; ``send_elapsed=False`` or an observation without it sends
    none). With ``record``, at most one sample per seat per unchanged position in a row (same two FENs, clocks
    ignored): a repeated search while sitting runs without ``sample`` and is counted in ``repeats_dropped``.
    ``think_head=True`` (off by default): before each new search a ``quick_nodes`` (16) look reads the root
    ``think_gain`` p for the seat's board, and the move's node budget becomes clock budget x (0.5 + 1.5 p),
    clamped to ``min_nodes..max_nodes``, with the quick look's nodes charged to that same budget. A network
    without the head prints no think_gain: then the clock budget is used unchanged (``think_head_missing``).

    Spec v3 (docs/custom_engine_spec_v3.md, audit WP-C). Every option below is off by default, so a controller
    built as before behaves as before; ``arena.selfplay --regime v3`` turns them on.

    * ``charge="searched"``: charge the nodes the engine reports having searched (``go nodes N`` overshoots by
      about a batch per thread), not the requested N. Both stock and free mode. Samples record both.
    * ``min_think_ms``: a floor on think time in clock ms (the budget is max(clock / k, floor), then nodes), in
      place of the bare ``min_nodes`` floor that let a seat move every 44-88 ms and so never flag by moving.
    * ``pace_mixture`` ((weight, factor), ...): replaces ``jitter``. Each seat draws one factor per game from the
      mixture; each search's charge is budget x that factor x a mean-1 log-normal (``pace_sigma``) plus an
      exponential lag of mean ``lag_ms``.
    * ``pace=True``: the controller keeps an EMA (``pace_alpha``) of every seat's observed move times (from the
      clocks at each landing, read off the FENs' ply counts), sends them as the ``pace <aw> <ab> <bw> <bb>``
      suffix (seats not yet seen: this search's own unjittered budget), and sets ``TreeMoveMs`` to the seat's own
      pace instead of this search's jittered charge.
    * ``noise=True``: Dirichlet root noise (``OpeningNoise*`` and the all-game switch) on every search that may
      be recorded, off on the others. ``temperature`` > 0: the move temperature option while the game has
      fewer than ``temperature_plies`` landed moves (0 after).
    * ``pcr_p``: playout-cap randomisation. With probability ``pcr_p`` a search runs at least ``pcr_nodes`` and
      may be recorded; the rest run at the clock budget and are not recorded (except the sit->move
      transition). The clock is charged as for the clock-budget search either way (times the overshoot with
      ``charge="searched"``), so the game's time course does not depend on the draw.
    * ``dedupe="bucket"``: at most one sample per seat per (both FENs, all four clocks // ``clock_bucket_ms``),
      instead of one per unchanged position, and the search where a sitting seat decides to move is always kept.
      Every kept sample carries ``sit_ms``: how long the seat then sat (0 when it moved).
    * ``sit_recheck=True``: a sitting seat woken before ``free_sit_ms`` has passed since its last look, with its
      own board's FEN (pieces and both pockets there) unchanged, waits on without a search. A due re-look is a
      ``recheck_nodes`` search that neither queues behind nor blocks the partner's search; only when it wants
      to move does a full-budget search follow (its time added after the re-look's).
    * ``think_head_norm=True``: the think-head multiplier is divided by its running mean in the game, so it
      moves time between moves without cutting the mean.
    * ``start_offset_ms``: the team's time-odds offset, written into every sample's sidecar.
    * ``stock_no_root``: what a stock-mode seat does when the engine answers ``bestmove (none)`` while it has
      legal moves: ``"wait"`` (as before) or ``"heuristic"`` (a 1-ply move). Counted in ``no_root_results``
      in both modes.
    ``stalls`` lists one dict per sit spell in free mode (seat, time, diagonal clock difference, whether the
    partner's opponent is on turn, in check, legal move count)."""

    def __init__(self, team: str, process: HivemindProcess, nodes: int = 400, ms_per_node: float = MS_PER_NODE,
                 *, max_consecutive_sits: int = 2, sit_ms: int = 1000, clock_k: float | None = None,
                 min_nodes: int = 16, max_nodes: int = 1600, serial: bool = False, free: bool = False,
                 free_sit_ms: int = 500, k_range=(60.0, 300.0), jitter=None, record: bool = False,
                 seed: int = 0, think_head: bool = False, quick_nodes: int = 16, send_elapsed: bool = True,
                 charge: str = "requested", min_think_ms: float | None = None, pace: bool = False,
                 pace_alpha: float = 0.3, pace_mixture=None, pace_sigma: float = 0.25, lag_ms: float = 0.0,
                 noise: bool = False, temperature: float = 0.0, temperature_plies: int = 20,
                 pcr_p: float | None = None, pcr_nodes: int = 400, dedupe: str = "position",
                 clock_bucket_ms: int = 5000, sit_recheck: bool = False, recheck_nodes: int = 32,
                 think_head_norm: bool = False, start_offset_ms: int = 0, stock_no_root: str = "wait") -> None:
        if clock_k is not None and clock_k <= 0:
            raise ValueError(f"clock_k must be > 0, got {clock_k}")
        if record and not free:
            raise ValueError("record needs free=True (samples come from the hivemind-clock build)")
        if think_head and not free:
            raise ValueError("think_head needs free=True (the think_gain output comes from the clock network)")
        v3_free_only = {"pace": pace, "noise": noise, "temperature": temperature > 0, "pcr_p": pcr_p is not None,
                        "sit_recheck": sit_recheck, "dedupe": dedupe != "position"}
        if not free and any(v3_free_only.values()):
            raise ValueError(f"{[k for k, v in v3_free_only.items() if v]} need free=True")
        if dedupe not in DEDUPE_MODES:
            raise ValueError(f"dedupe must be one of {DEDUPE_MODES}, got {dedupe!r}")
        if stock_no_root not in NO_ROOT_MODES:
            raise ValueError(f"stock_no_root must be one of {NO_ROOT_MODES}, got {stock_no_root!r}")
        if pcr_p is not None and not 0.0 <= pcr_p <= 1.0:
            raise ValueError(f"pcr_p must be in [0, 1], got {pcr_p}")
        if not 0.0 < pace_alpha <= 1.0:
            raise ValueError(f"pace_alpha must be in (0, 1], got {pace_alpha}")
        self.team = team
        self.process = process
        self.nodes = int(nodes)
        self.ms_per_node = float(ms_per_node)
        self.max_consecutive_sits = max_consecutive_sits
        self.sit_ms = sit_ms
        self.clock_k = clock_k
        self.min_nodes, self.max_nodes = int(min_nodes), int(max_nodes)
        self.serial = serial
        self.free = bool(free)
        self.free_sit_ms = int(free_sit_ms)
        self.record = bool(record)
        self._rng = random.Random(f"{seed}:{team}")
        self.k_range = _pair(k_range, "k_range")
        self.jitter = _pair(jitter, "jitter")
        self.pace_mixture = _mixture(pace_mixture)
        if self.free:
            self.serial = True
            if self.jitter is None and self.pace_mixture is None:
                self.jitter = (0.5, 2.0)
            self.clock_k = self._rng.uniform(*self.k_range)
        if self.pace_mixture is not None:
            self.jitter = None  # the mixture replaces the uniform jitter
        self.pace_sigma = float(pace_sigma)
        self.lag_ms = max(0.0, float(lag_ms))
        self.seat_factor: dict[str, float] = {}
        if self.pace_mixture is not None:
            weights = [w for w, _ in self.pace_mixture]
            for s in SEATS:
                if TEAM_OF[s] == team:
                    self.seat_factor[s] = self._rng.choices([f for _, f in self.pace_mixture], weights)[0]
        self.think_head = bool(think_head)
        self.quick_nodes = max(1, int(quick_nodes))
        self.send_elapsed = bool(send_elapsed)
        self.charge = _check_charge(charge)
        self.min_think_ms = None if min_think_ms is None else max(0.0, float(min_think_ms))
        self.pace = bool(pace)
        self.pace_alpha = float(pace_alpha)
        self.noise = bool(noise)
        self.temperature = max(0.0, float(temperature))
        self.temperature_plies = int(temperature_plies)
        self.pcr_p = None if pcr_p is None else float(pcr_p)
        self.pcr_nodes = int(pcr_nodes)
        self.dedupe = dedupe
        self.clock_bucket_ms = max(1, int(clock_bucket_ms))
        self.sit_recheck = bool(sit_recheck)
        self.recheck_nodes = max(1, int(recheck_nodes))
        self.think_head_norm = bool(think_head_norm)
        self.start_offset_ms = int(start_offset_ms)
        self.stock_no_root = stock_no_root
        self.samples: list[RecordedSample] = []
        self.searches = 0
        self.repeats_dropped = 0  # record: searches whose sample was dropped as a repeat (dedupe)
        self.unrecorded_fast = 0  # record + pcr_p: clock-budget searches not recorded
        self.transitions = 0  # dedupe="bucket": sit->move searches kept
        self.rechecks = 0  # sit_recheck: small re-look searches run
        self.rechecks_skipped = 0  # sit_recheck: wakes answered without a search
        self.nodes_requested = 0  # every search, requested nodes
        self.nodes_searched = 0  # every search, nodes the engine reported (requested when it reported none)
        self.quick_looks = 0  # think_head: 16-node looks run
        self.think_head_missing = 0  # think_head: looks that gave no think_gain (clock budget used unchanged)
        self.think_gain_sum = 0.0  # think_head: sum of the p values used (mean = sum / (looks - missing))
        self._th_mult_sum, self._th_mult_n = 0.0, 0
        self._last_sample_pos: dict[str, tuple[str, str]] = {}
        self._seen_pos: dict[str, tuple[str, str]] = {}
        self._seen: dict[str, set] = {}
        self._pending_sit: dict[str, list[ControllerSample]] = {}
        self._sitting: dict[str, dict] = {}
        self._pace_ema: dict[str, float] = {}
        self._pace_clock: dict[str, float] = {}
        self._pace_plies: dict[str, int] = {}
        self.stalls: list[dict] = []
        self.no_root_results = 0  # searches that returned bestmove (none) although the seat could move
        self._busy_until = 0  # serial: the one machine is searching until then
        self._cache: tuple[tuple[str, str, bool], SearchResult, int] | None = None
        self._sits: dict[str, int] = {}
        self._seats: dict[str, "CentralizedHivemindSeat"] = {}

    def budget(self, clock_ms: int) -> int:
        """Nodes for one search: the fixed ``nodes``, or with ``clock_k`` the remaining clock / clock_k (at least
        ``min_think_ms`` of clock when set), clamped to ``min_nodes..max_nodes``."""
        if self.clock_k is None:
            return self.nodes
        ms = max(0, clock_ms) / self.clock_k
        if self.min_think_ms is not None:
            ms = max(ms, self.min_think_ms)
        nodes = int(round(ms / self.ms_per_node))
        return min(self.max_nodes, max(self.min_nodes, nodes))

    def player(self, seat: str) -> "CentralizedHivemindSeat":
        if seat not in self._seats:
            if self._seats:  # the process starts with one reference; each further seat view adds one
                self.process.acquire()
            self._seats[seat] = CentralizedHivemindSeat(seat, self)
        return self._seats[seat]

    # ------------------------------------------------------------------ think-time model

    def _time_noise(self, seat: str | None = None) -> tuple[float, float]:
        """(factor, lag ms) for one new search: the seat's pace-mixture factor x a mean-1 log-normal draw, or the
        uniform ``jitter`` draw, or 1; plus an exponential lag when ``lag_ms`` > 0."""
        if self.pace_mixture is not None and seat in self.seat_factor:
            s = self.pace_sigma
            factor = self.seat_factor[seat] * math.exp(self._rng.gauss(0.0, s) - s * s / 2) if s > 0 \
                else self.seat_factor[seat]
        elif self.jitter:
            factor = self._rng.uniform(*self.jitter)
        else:
            factor = 1.0
        lag = self._rng.expovariate(1.0 / self.lag_ms) if self.lag_ms > 0 else 0.0
        return factor, lag

    def _think_ms(self, nodes: float, factor: float = 1.0, lag: float = 0.0) -> int:
        ms = nodes * self.ms_per_node
        if self.min_think_ms is not None:
            ms = max(ms, self.min_think_ms)
        return int(round(ms * factor + lag))

    def _charged_ms(self, budget: int, seat: str | None = None) -> int:
        """Think time of one new search: budget * ms_per_node (floored at ``min_think_ms``), times this search's
        jitter or pace draw, plus lag."""
        return self._think_ms(budget, *self._time_noise(seat))

    def _expected_ms(self, nodes: int) -> float:
        """This search's own unjittered think time: the pace default."""
        ms = nodes * self.ms_per_node
        return max(ms, self.min_think_ms) if self.min_think_ms is not None else ms

    # ------------------------------------------------------------------ pace (spec v3 contract 1)

    def _observe_pace(self, obs: Observation) -> None:
        """Update every seat's move-time EMA from the moves that landed since the last observation. A seat's move
        time is its clock at its previous landing minus its clock at this one; the on-turn seat's clock has run
        since the board's last landing, so that part is added back. Several moves by one seat between two
        observations are averaged."""
        for b in ("A", "B"):
            plies = fen_plies(obs.fens[b])
            if plies is None:
                continue
            last = obs.last_move_ms[b] if obs.last_move_ms else obs.now_ms
            on_turn = obs.to_move.get(b)

            def at_landing(s: str) -> float:
                c = float(obs.clocks_ms[s])
                return c + max(0, obs.now_ms - int(last)) if s == on_turn else c

            prev = self._pace_plies.get(b)
            if prev is not None and plies > prev:
                for s in (f"{b}_white", f"{b}_black"):
                    white = s.endswith("white")
                    n = sum(1 for p in range(prev, plies) if (p % 2 == 0) == white)
                    if n and s in self._pace_clock:
                        spent = (self._pace_clock[s] - at_landing(s)) / n
                        if spent >= 0:
                            old = self._pace_ema.get(s)
                            self._pace_ema[s] = spent if old is None else \
                                self.pace_alpha * spent + (1 - self.pace_alpha) * old
            if prev is None or plies != prev:
                for s in (f"{b}_white", f"{b}_black"):
                    self._pace_clock[s] = at_landing(s)
                self._pace_plies[b] = plies

    def pace_values(self, default_ms: float) -> dict:
        """{seat: expected move ms} for the ``pace`` suffix: each seat's EMA, ``default_ms`` until it has one."""
        return {s: self._pace_ema.get(s, default_ms) for s in CLOCK_SEATS}

    # ------------------------------------------------------------------ stock mode

    def decide(self, seat: str, obs: Observation) -> Decision:
        if not obs.legal_moves:
            return Decision(kind="wait", wait_ms=200, stall_reason="no_moves")
        if self.free:
            return self._decide_free(seat, obs)
        board = BOARD_OF[seat]
        edge = team_time_edge(obs.clocks_ms, self.team)
        ahead = edge > 0
        sits = self._sits.get(seat, 0)
        require = None if (ahead and sits < self.max_consecutive_sits) else board
        key = (obs.fens["A"], obs.fens["B"], ahead)
        budget = self.budget(obs.clocks_ms[seat])
        cached = self._cache
        if cached and cached[0] == key and _legalize(cached[1].move_for(board), list(obs.legal_moves)) \
                and (require is None or cached[1].move_for(board) is not None):
            res, ready = cached[1], cached[2]
            think, nodes = max(0, ready - obs.now_ms), 0
        else:
            res = self.process.search(obs.fens["A"], obs.fens["B"], team=self.team, nodes=budget,
                                      time_advantage=ahead, require=require)
            self.searches += 1
            if res.no_root:
                self.no_root_results += 1
            charged = charged_nodes(self.charge, budget, res)
            self.nodes_requested += budget
            self.nodes_searched += res.nodes if res.nodes > 0 else budget
            full = self._charged_ms(charged)
            queued = max(0, self._busy_until - obs.now_ms) if self.serial else 0
            think, nodes = queued + full, charged
            self._busy_until = obs.now_ms + think
            self._cache = (key, res, obs.now_ms + think)
        move = _legalize(res.move_for(board), list(obs.legal_moves))
        if move is None and res.move_for(board) is None and not (res.no_root and self.stock_no_root == "heuristic"):
            self._sits[seat] = sits + 1
            return Decision(kind="wait", think_ms=think, wait_ms=think + self.sit_ms, nodes=nodes, eval=res.q,
                            stall_reason="time_advantage" if ahead else "engine_pass")
        if move is None:
            b = parse_board(obs.fens[board])
            move = evaluate_moves(b, list(obs.legal_moves), wanted=None, rng=random.Random(obs.ply))[0].uci
        self._sits[seat] = 0
        return Decision(kind="move", uci=move, think_ms=think, nodes=nodes, eval=res.q)

    # ------------------------------------------------------------------ free mode

    def _elapsed(self, obs: Observation) -> dict | None:
        if not self.send_elapsed or obs.last_move_ms is None:
            return None
        return {b: max(0, obs.now_ms - int(obs.last_move_ms[b])) for b in ("A", "B")}

    def _think_head_nodes(self, seat: str, obs: Observation, budget: int, elapsed: dict | None,
                          ahead: bool) -> tuple[int, int]:
        """(total nodes for this move, nodes already spent on the quick look). Spec v2 B."""
        quick = min(self.quick_nodes, budget)
        look = self.process.search(obs.fens["A"], obs.fens["B"], team=self.team, nodes=quick, time_advantage=ahead,
                                   clocks=obs.clocks_ms, elapsed=elapsed, require_option=False,
                                   options={"FreePass": True, "MateProbeAnyTime": True,
                                            "TreeMoveMs": int(round(budget * self.ms_per_node))})
        self.quick_looks += 1
        if look.think_gain is None or look.no_root:
            self.think_head_missing += 1
            return budget, quick
        p = min(1.0, max(0.0, float(look.think_gain[0 if BOARD_OF[seat] == "A" else 1])))
        self.think_gain_sum += p
        mult = 0.5 + 1.5 * p
        if self.think_head_norm:  # mean 1 over the game's looks so far (audit inputs 5)
            self._th_mult_sum += mult
            self._th_mult_n += 1
            mult /= self._th_mult_sum / self._th_mult_n
        total = min(self.max_nodes, max(self.min_nodes, int(round(budget * mult))))
        return max(total, quick + 1), quick

    def _stall_info(self, seat: str, obs: Observation) -> dict:
        """Per sit spell (audit self-play 4(d)): the diagonal (our clock minus the partner's opponent's), whether
        the partner's opponent is on turn on the other board, in check, legal move count."""
        partner = PARTNER[seat]
        partop = OPPONENT[partner]
        try:
            in_check = parse_board(obs.fens[BOARD_OF[seat]]).is_check()
        except ValueError:
            in_check = None
        return {"seat": seat, "t": obs.now_ms, "diagonal_ms": int(obs.clocks_ms[seat] - obs.clocks_ms[partop]),
                "partop_on_turn": obs.to_move.get(BOARD_OF[partner]) == partop, "in_check": in_check,
                "legal": len(obs.legal_moves)}

    def _exploration_options(self, recorded: bool, ply: int) -> dict:
        opts: dict = {}
        if self.noise:
            opts["OpeningNoise"] = bool(recorded)
            if recorded:
                opts.update({"OpeningNoisePlies": NOISE_PLIES, "OpeningNoiseAlphaPermille": NOISE_ALPHA_PERMILLE,
                             "OpeningNoiseEpsilonPermille": NOISE_EPSILON_PERMILLE, NOISE_ALL_GAME_OPTION: True})
        if self.temperature > 0:
            on = recorded is not None and ply < self.temperature_plies
            opts[TEMPERATURE_OPTION] = int(round(1000 * self.temperature)) if on else 0
        return opts

    def _free_search(self, seat: str, obs: Observation, *, recheck: bool = False,
                     pre_ms: int = 0) -> tuple[SearchResult, int, int]:
        """One free-mode search for ``seat``: (result, think ms for the Decision, nodes charged). Records the
        sample when the recording rules keep it."""
        board = BOARD_OF[seat]
        fen_a, fen_b = obs.fens["A"], obs.fens["B"]
        ahead = team_time_edge(obs.clocks_ms, self.team) > 0  # the legacy TimeAdvantage input plane only
        elapsed = self._elapsed(obs)
        budget = self.budget(obs.clocks_ms[seat])
        if recheck:
            total, spent = max(1, min(budget, self.recheck_nodes)), 0
            full = self.pcr_p is None  # a re-look is never a playout-cap full search
        else:
            total, spent = (self._think_head_nodes(seat, obs, budget, elapsed, ahead) if self.think_head
                            else (budget, 0))
            full = self.pcr_p is None or self._rng.random() < self.pcr_p
        request = total - spent
        if self.pcr_p is not None and full and not recheck:
            request = max(request, self.pcr_nodes - spent)
        factor, lag = self._time_noise(seat)
        pace = self.pace_values(self._expected_ms(total)) if self.pace else None
        tree_ms = int(round(pace[seat])) if pace is not None else self._think_ms(total, factor, lag)
        pos = (fen_a, fen_b)
        legacy_repeat = self.record and self.dedupe == "position" and self._last_sample_pos.get(seat) == pos
        want_sample = self.record and (not legacy_repeat if self.dedupe == "position" else True)
        options = {"FreePass": True, "MateProbeAnyTime": True, "TreeMoveMs": tree_ms}
        options.update(self._exploration_options(None if recheck else full, obs.ply))
        kw = {"pace": pace} if pace is not None else {}
        res = self.process.search(fen_a, fen_b, team=self.team, nodes=request, time_advantage=ahead,
                                  clocks=obs.clocks_ms, sample=want_sample, options=options,
                                  require_option=False, elapsed=elapsed, **kw)
        self.searches += 1
        searched = (res.nodes if res.nodes > 0 else request) + spent
        self.nodes_requested += request + spent
        self.nodes_searched += searched
        # Clock charge: the clock-budget search's nodes, times the overshoot when charging searched nodes.
        charged = total if self.charge == "requested" else max(1, int(round(total * searched / (request + spent))))
        own = self._think_ms(charged, factor, lag)
        passed = not res.no_root and res.move_for(board) is None
        if res.no_root:
            # The seat has legal moves (decide() returned early otherwise), so the engine should have a
            # root. Log it with the position, record no sample for it, and fall back to a heuristic move
            # in the caller, as the illegal-move path does, instead of crashing the game or waiting silently.
            self.no_root_results += 1
            _LOG.error("hivemind returned bestmove (none) for %s (%s) with %d legal moves; "
                       "fenA=%s fenB=%s clocks=%s time_advantage=%s", seat, self.team,
                       len(obs.legal_moves), fen_a, fen_b, obs.clocks_ms, ahead)
        elif self.record:
            transition = False
            if self.dedupe == "position":
                keep = not legacy_repeat
                if legacy_repeat:
                    self.repeats_dropped += 1
            else:
                bkey = pos + tuple(int(obs.clocks_ms[s]) // self.clock_bucket_ms for s in CLOCK_SEATS)
                if self._seen_pos.get(seat) != pos:
                    self._seen_pos[seat], self._seen[seat] = pos, set()
                transition = seat in self._sitting and not passed
                keep = transition or (full and bkey not in self._seen[seat])
                if keep:
                    self._seen[seat].add(bkey)
                    self.transitions += transition
                elif not full:
                    self.unrecorded_fast += 1
                else:
                    self.repeats_dropped += 1
            if keep:
                if res.sample is None:
                    raise RuntimeError("record=True but the search returned no sample")
                if len(res.sample) <= SAMPLE_TEAM_OFFSET or \
                        res.sample[SAMPLE_TEAM_OFFSET] != SAMPLE_TEAM_BYTE[self.team]:
                    raise ValueError(f"sample team byte does not match team {self.team}")
                # spec v3 contract 2: seat-ordered (A_white A_black B_white B_black) and (A, B) float tuples
                extra = {"sit_ms": None if passed else 0.0,
                         "search_kind": "recheck" if recheck else ("full" if full else "fast"),
                         "transition": transition}
                sample = make_sample(
                    extra, now_ms=obs.now_ms, seq=next(_SAMPLE_SEQ), buf=res.sample, seat=seat, team=self.team,
                    ply=obs.ply, fen_a=fen_a, fen_b=fen_b, clocks=dict(obs.clocks_ms), elapsed=elapsed,
                    tree_ms=tree_ms, nodes=total, time_advantage=ahead,
                    nodes_requested=request + spent, nodes_searched=searched,
                    start_offset_ms=float(self.start_offset_ms),
                    pace=None if pace is None else tuple(float(pace[s]) for s in CLOCK_SEATS),
                    clocks_ms=tuple(float(obs.clocks_ms[s]) for s in CLOCK_SEATS),
                    elapsed_ms=None if elapsed is None else (float(elapsed["A"]), float(elapsed["B"])))
                self.samples.append(sample)
                self._last_sample_pos[seat] = pos
                if passed:
                    self._pending_sit.setdefault(seat, []).append(sample)
        if recheck:  # a sitter's re-look: does not queue behind or block the partner's search
            return res, own, charged
        queued = max(0, self._busy_until - obs.now_ms - pre_ms)
        think = pre_ms + queued + own
        self._busy_until = obs.now_ms + think
        self._cache = ((fen_a, fen_b, ahead), res, obs.now_ms + think)
        return res, think, charged

    def _free_wait(self, seat: str, obs: Observation, res: SearchResult, think: int, nodes: int) -> Decision:
        if seat not in self._sitting:
            self.stalls.append(self._stall_info(seat, obs))
        self._sitting[seat] = {"own_fen": obs.fens[BOARD_OF[seat]], "next_ms": obs.now_ms + think + self.free_sit_ms}
        return Decision(kind="wait", think_ms=think, wait_ms=think + self.free_sit_ms, nodes=nodes, eval=res.q,
                        stall_reason="free_pass")

    def _moved(self, seat: str, obs: Observation) -> None:
        """The seat decided to move: its sit spell (if any) ends now; fill ``sit_ms`` of the samples it left."""
        self._sitting.pop(seat, None)
        for s in self._pending_sit.pop(seat, []):
            s.extra["sit_ms"] = float(max(0, obs.now_ms - s.now_ms))

    def finish(self, end_ms: int) -> None:
        """Game over at ``end_ms``: samples of seats still sitting get the sit length up to the end."""
        for lst in self._pending_sit.values():
            for s in lst:
                s.extra["sit_ms"] = float(max(0, end_ms - s.now_ms))
        self._pending_sit.clear()
        self._sitting.clear()

    def _decide_free(self, seat: str, obs: Observation) -> Decision:
        board = BOARD_OF[seat]
        legal = list(obs.legal_moves)
        if self.pace:
            self._observe_pace(obs)
        ahead = team_time_edge(obs.clocks_ms, self.team) > 0
        key = (obs.fens["A"], obs.fens["B"], ahead)
        cached = self._cache
        # Reuse only a move: after a pass the controller searches again (the clocks have moved on).
        if cached and cached[0] == key and _legalize(cached[1].move_for(board), legal):
            res, ready = cached[1], cached[2]
            self._moved(seat, obs)
            return Decision(kind="move", uci=_legalize(res.move_for(board), legal),
                            think_ms=max(0, ready - obs.now_ms), nodes=0, eval=res.q)
        pre_ms = 0
        sit = self._sitting.get(seat)
        if self.sit_recheck and sit is not None:
            if obs.now_ms < sit["next_ms"] and obs.fens[board] == sit["own_fen"]:
                self.rechecks_skipped += 1
                return Decision(kind="wait", wait_ms=sit["next_ms"] - obs.now_ms, stall_reason="free_pass")
            res, think, nodes = self._free_search(seat, obs, recheck=True)
            self.rechecks += 1
            if not res.no_root and res.move_for(board) is None:
                return self._free_wait(seat, obs, res, think, nodes)
            pre_ms = think  # the re-look wants a move: a full search follows, after it
        res, think, nodes = self._free_search(seat, obs, pre_ms=pre_ms)
        move = _legalize(res.move_for(board), legal)
        if move is None and res.move_for(board) is None and not res.no_root:
            return self._free_wait(seat, obs, res, think, nodes)
        if move is None:
            b = parse_board(obs.fens[board])
            move = evaluate_moves(b, legal, wanted=None, rng=random.Random(obs.ply))[0].uci
        self._moved(seat, obs)
        return Decision(kind="move", uci=move, think_ms=think, nodes=nodes, eval=res.q)


class CentralizedHivemindSeat:
    kind = "hivemind"

    def __init__(self, seat: str, controller: HivemindTeamController) -> None:
        self.seat = seat
        self.controller = controller
        self.name = f"Hivemind-{seat}-central-" + (f"free-k{controller.clock_k:.0f}" if controller.free
                                                   else f"{controller.nodes}n")
        self._closed = False

    def decide(self, obs: Observation) -> Decision:
        return self.controller.decide(self.seat, obs)

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self.controller.process.release()
