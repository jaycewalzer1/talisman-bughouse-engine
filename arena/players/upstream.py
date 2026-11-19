"""Upstream aminwoo/hivemind v3 (twin-s-clock, clock model) as an arena opponent: ``upstream-v3``.

Build: ``sh tools/build_upstream.sh`` (clones the v3.0.x tag into ``third_party/upstream-hivemind``, builds the ONNX
Runtime engine, fetches twin-s-clock it13 and converts it to FP32). Versions and checks: ``docs/upstream/v3_arm.md``.

Upstream's dialect differs from our fork's (theory 06 section 2.2, upstream ``engine/src/interface/uci.cc``):
* ``position fen <A> | <B>`` only: no ``clocks``/``elapsed``/``pace`` suffix.
* Time is ``setoption`` state, set before every search:
  ``TimeModel clock``; ``TimeLeadMs`` = our team's lead over the diagonal opponents (``lead_ms`` below);
  ``TeamMoveTimeMs`` / ``OpponentMoveTimeMs`` = what one action costs each team (EMAs of observed move times,
  ``MoveTimeTracker``); ``RootMoveTimeMs`` = the budget of the move being searched (v3.0.1 "charges the searched
  move its clock budget"). Sitting is the engine's own rule (``can_sit = lead > 0`` at every node); we never send
  ``RequireMoveOn`` or ``TimeAdvantage``.
* ``go nodes N`` -> ``bestmove (<a>,<b>)``, ``pass`` for no move; then a permanent-brain search runs until the next
  command, so we ``stop`` straight after bestmove, as for our fork. ``BackgroundSearch false``, ``Ponder false``.
* ``policy`` prints the raw network priors (softmax over the legal actions plus pass) for the current position and
  time options; ``UpstreamHivemindProcess.raw_policy`` parses it (tools/upstream_check.py compares it with an
  independent ONNX Runtime evaluation).

Sign of the lead (upstream ``environment/time_state.h``): without increment one clock runs on each board, so the white
team's (our T1: A_white + B_black) lead is ``Aw - Bw = Bb - Ab``, and T2's is its negation. We send the mean of the two
expressions, which is exact while the board sums are equal and splits any difference (move-landing lag) evenly.
Equivalently ``lead_ms = team_time_edge / 2``.

Arena cost model (spec v3 D2, "equal wall-clock as nodes-as-time"): each search's think time is ``nodes_searched x
ms_per_node`` (the nodes the engine reports, not the nodes requested; ``charge="requested"`` restores the old rule),
with ``ms_per_node`` this arm's own (measured for this binary+network, see ``docs/upstream/v3_arm.md``).
Budget: fixed ``nodes`` per search, or ``clock_k``: remaining clock / k converted to nodes (clamped). Upstream itself
has no two-board time manager (``wtime``/``btime`` are ignored for bughouse; budget = nodes else movetime), so a
front end has to choose; these are the same two rules our centralized controller offers.

Wave 2 (``kind="wave2"``, ``Wave2TeamController``/``wave2_team``): the spec v4 durative tree built into upstream v3
(``third_party/wave2-hivemind``, branch talisman/wave2). Same dialect and network; it also gets the four real
clocks, per-seat paces and the on-turn seats' elapsed times per search, and the engine flags (``WAVE2_FLAGS``:
DurativeTree, SitGate, FlagLoss, InstantTransfer, MateRaceByLanding, PassCapK). Contract and test round:
``docs/wave2_tests.md``.
"""

from __future__ import annotations

import logging
import os
import random
import subprocess
import threading
import time
from dataclasses import dataclass
from dataclasses import replace as _dc_replace

from arena.interfaces import BOARD_OF, SEATS, TEAM_OF, Decision, Observation
from arena.players.heuristic import evaluate_moves, parse_board, team_seats
from arena.players.hivemind import HivemindProcess, SearchResult, _legalize

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
UPSTREAM_DIR = os.environ.get("UPSTREAM_HIVEMIND_DIR", os.path.join(REPO_ROOT, "third_party", "upstream-hivemind"))
UPSTREAM_ENGINE_DIR = os.path.join(UPSTREAM_DIR, "engine")
UPSTREAM_BINARY = os.environ.get("UPSTREAM_BINARY", os.path.join(UPSTREAM_ENGINE_DIR, "build-ninja", "hivemind.bin"))
UPSTREAM_MODEL = os.environ.get("UPSTREAM_MODEL",
                                os.path.join(UPSTREAM_ENGINE_DIR, "models", "twin-s-clock-fp32.onnx"))
UPSTREAM_NETWORK = "twin-s-clock it13 (HF aminwoo/bughouse-twin-s-clock@edaa246, sha256 f4ac25e3...)"
# Simulated ms per searched node for this arm: measure_upstream_ms_per_node on the M4 Max, ONNX Runtime CPU provider
# (HIVEMIND_COREML=off, the provider our stock MS_PER_NODE = 5.5 was measured on): 0.78 / 0.69 / 0.62 at
# 100 / 400 / 1600 nodes. The engine's default Core ML provider measures 0.26 / 0.21 / 0.20. docs/upstream/v3_arm.md.
# Spec v3 D2 wants the gate GPU's numbers for both arms; override per run with ms_per_node.
UPSTREAM_MS_PER_NODE = 0.65
DEFAULT_MOVE_MS = 100  # upstream TimeState::DEFAULT_MOVE_TIME_MS ("the fastest possible move")

# Wave 2 (spec v4, durative tree): branch talisman/wave2 of the hivemind fork, based on upstream v3.0.3, built by
# WAVE2=1 sh tools/build_upstream.sh into third_party/wave2-hivemind. Same twin-s-clock network and inputs; its own
# engine dir (cwd) and its own measured ms_per_node (the durative tree may cost differently per node).
WAVE2_DIR = os.environ.get("WAVE2_HIVEMIND_DIR", os.path.join(REPO_ROOT, "third_party", "wave2-hivemind"))
WAVE2_ENGINE_DIR = os.path.join(WAVE2_DIR, "engine")
WAVE2_BINARY = os.environ.get("WAVE2_BINARY", os.path.join(WAVE2_ENGINE_DIR, "build-ninja", "hivemind.bin"))
_WAVE2_OWN_MODEL = os.path.join(WAVE2_ENGINE_DIR, "models", "twin-s-clock-fp32.onnx")
# The wave 2 build fetches its own copy of it13; before that, stock upstream's file (same sha256) stands in.
WAVE2_MODEL = os.environ.get("WAVE2_MODEL", _WAVE2_OWN_MODEL if os.path.isfile(_WAVE2_OWN_MODEL) else UPSTREAM_MODEL)
WAVE2_MS_PER_NODE = UPSTREAM_MS_PER_NODE  # placeholder until measured (MEASURE=1 measures it per row)
# The UCI contract of wave 2 (docs/wave2_tests.md): player option -> (engine option, wave 2 value). The engine's
# defaults are stock v3.0.3 (DurativeTree false, SitGate true, FlagLoss false, InstantTransfer true,
# MateRaceByLanding false, PassCapK 8); the wave 2 arm sends every flag explicitly, so an ablation row switches one
# off with its player option (e.g. {"sit_gate": true}). MateRaceByLanding stays off in wave 2: it switches off the
# root mate provers, and with the gate off the tree then passed instead of mating in won positions (drop mates 68/80
# at 800 nodes against 80/80 with it off or the gate on; docs/wave2_local_check.md). Row racewin switches it on.
WAVE2_FLAGS = {"durative_tree": ("DurativeTree", True), "sit_gate": ("SitGate", False),
               "flag_loss": ("FlagLoss", True), "instant_transfer": ("InstantTransfer", False),
               "mate_race_by_landing": ("MateRaceByLanding", False), "pass_cap_k": ("PassCapK", 8)}
TEAM_OPTION = {"T1": "white", "T2": "black"}
MAX_LEAD_MS = 1 << 30

_LOG = logging.getLogger(__name__)


def upstream_available(binary: str = UPSTREAM_BINARY, model: str = UPSTREAM_MODEL) -> bool:
    return os.path.isfile(binary) and os.access(binary, os.X_OK) and os.path.isfile(model)


def lead_ms(clocks_ms: dict, team: str) -> int:
    """``team``'s lead over its diagonal opponents in ms (positive: ahead), as upstream's ``TimeLeadMs`` wants it.
    T1 = A_white + B_black: ``((Aw - Bw) + (Bb - Ab)) / 2``; T2 is the negation."""
    aw, ab, bw, bb = (int(clocks_ms[s]) for s in ("A_white", "A_black", "B_white", "B_black"))
    t1 = ((aw - bw) + (bb - ab)) / 2.0
    lead = int(round(t1 if team == "T1" else -t1))
    return max(-MAX_LEAD_MS, min(MAX_LEAD_MS, lead))


def _moves_made(fen: str, seat: str, start_fullmove: int, start_white_to_move: bool) -> int:
    """Moves ``seat`` has made on its board since the game's start FEN, from the FEN's move counter and side."""
    parts = fen.split()
    white_to_move = parts[1] == "w"
    full = int(parts[5]) if len(parts) > 5 else 1
    plies = 2 * (full - start_fullmove) + (0 if white_to_move else 1) - (0 if start_white_to_move else 1)
    if seat.endswith("white"):
        return (plies + 1) // 2 if start_white_to_move else plies // 2
    return plies // 2 if start_white_to_move else (plies + 1) // 2


class MoveTimeTracker:
    """Per-team and per-seat EMAs of the time one action costs, read off the clocks.

    A seat's clock only runs while it is on turn, so between two observations where the seat is NOT on turn its clock
    drop divided by the moves it made in between is its mean time per move, sitting included (which is what upstream's
    move time means: what an action costs the acting team). Each such sample updates its team's EMA and its seat's
    EMA once per move (``seat_ms``: wave 2's ``SeatPaceMs``). ``default_ms`` stands in until a team has a sample
    (None: use the caller's fallback)."""

    def __init__(self, alpha: float = 0.3, default_ms: int | None = None) -> None:
        if not 0.0 < alpha <= 1.0:
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        self.alpha = float(alpha)
        self.default_ms = default_ms
        self.ema: dict[str, float] = {}
        self.samples: dict[str, int] = {"T1": 0, "T2": 0}
        self.seat_ema: dict[str, float] = {}
        self.seat_samples: dict[str, int] = {s: 0 for s in SEATS}
        self._start: dict[str, tuple[int, bool]] = {}
        self._last: dict[str, tuple[int, int]] = {}  # seat -> (moves made, clock) at its last off-turn observation

    def observe(self, fens: dict, clocks_ms: dict, to_move: dict) -> None:
        for board in ("A", "B"):
            parts = fens[board].split()
            if board not in self._start:
                self._start[board] = (int(parts[5]) if len(parts) > 5 else 1, parts[1] == "w")
        for seat in SEATS:
            board = BOARD_OF[seat]
            if to_move.get(board) == seat:
                continue  # its clock is running: no clean reading
            start_full, start_white = self._start[board]
            made = _moves_made(fens[board], seat, start_full, start_white)
            clock = int(clocks_ms[seat])
            prev = self._last.get(seat)
            self._last[seat] = (made, clock)
            if prev is None or made <= prev[0]:
                continue
            n = made - prev[0]
            per_move = max(1.0, (prev[1] - clock) / n)
            team = TEAM_OF[seat]
            ema, sema = self.ema.get(team), self.seat_ema.get(seat)
            for _ in range(n):
                ema = per_move if ema is None else (1 - self.alpha) * ema + self.alpha * per_move
                sema = per_move if sema is None else (1 - self.alpha) * sema + self.alpha * per_move
            self.ema[team], self.seat_ema[seat] = ema, sema
            self.samples[team] += n
            self.seat_samples[seat] += n

    def move_ms(self, team: str, fallback: int) -> int:
        v = self.ema.get(team)
        if v is None:
            v = self.default_ms if self.default_ms is not None else fallback
        return max(1, int(round(v)))

    def seat_ms(self, seat: str, fallback: int) -> int:
        """The seat's own EMA; before its first sample its team's (``move_ms``), else ``fallback``."""
        v = self.seat_ema.get(seat)
        return max(1, int(round(v))) if v is not None else self.move_ms(TEAM_OF[seat], fallback)


@dataclass
class RawPolicy:
    value: float
    wdl: tuple[float, float, float]  # win, draw, loss
    board_a: dict[str, float]  # move (engine UCI, "pass") -> prior; empty when not our turn there
    board_b: dict[str, float]
    lines: list[str]


def parse_policy(lines: list[str]) -> RawPolicy:
    value, wdl = 0.0, (0.0, 0.0, 0.0)
    boards: dict[str, dict[str, float]] = {"A": {}, "B": {}}
    current = None
    for line in lines:
        s = line.strip()
        if s.startswith("Value:"):
            value = float(s.split()[1])
        elif s.startswith("WDL:"):
            w, d, l = (float(x) for x in s.split()[1:4])
            wdl = (w, d, l)
        elif s.startswith("Board A"):
            current = "A"
        elif s.startswith("Board B"):
            current = "B"
        elif current and ":" in s and not s.startswith("("):
            mv, p = s.rsplit(":", 1)
            try:
                boards[current][mv.strip()] = float(p)
            except ValueError:
                pass
    return RawPolicy(value, wdl, boards["A"], boards["B"], lines)


class UpstreamHivemindProcess(HivemindProcess):
    """One upstream engine subprocess (``hivemind.bin --model <onnx>``, cwd = upstream engine dir).

    ``coreml``: value of ``HIVEMIND_COREML`` for the engine (None: the engine's default, the Neural Engine on macOS;
    ``"off"`` forces the CPU provider, which matches Python ONNX Runtime CPU to ~1e-6). ``cwd``: the engine dir the
    binary runs in (default upstream's; the wave 2 engine runs in ``WAVE2_ENGINE_DIR``)."""

    def __init__(self, binary: str = UPSTREAM_BINARY, model: str = UPSTREAM_MODEL, *, options: dict | None = None,
                 reset_tree: bool = True, timeout_s: float = 120.0, refs: int = 1,
                 coreml: str | None = None, cwd: str | None = None) -> None:
        super().__init__(binary, model, options=options, reset_tree=reset_tree, timeout_s=timeout_s, refs=refs)
        self.coreml = coreml
        self.cwd = cwd or UPSTREAM_ENGINE_DIR

    def start(self) -> None:
        if self._proc is not None:
            return
        if self.closed:
            raise RuntimeError("UpstreamHivemindProcess is closed")
        if not upstream_available(self.binary, self.model):
            raise FileNotFoundError(f"upstream binary/model missing: {self.binary} / {self.model} "
                                    "(run sh tools/build_upstream.sh)")
        env = dict(os.environ)
        if self.coreml is not None:
            env["HIVEMIND_COREML"] = self.coreml
        self._proc = subprocess.Popen([self.binary, "--model", self.model], stdin=subprocess.PIPE,
                                      stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1,
                                      cwd=self.cwd, env=env)
        threading.Thread(target=self._reader, daemon=True).start()
        self._send("uci")
        self._wait_for(lambda l: l == "uciok")
        self._opts = {}
        for k, v in {"Ponder": "false", "BackgroundSearch": "false", "TimeModel": "clock",
                     **self.extra_options}.items():
            self.set_option(k, v)
        self._sync()

    def _set_time(self, team: str, lead: int, team_move_ms: int, opp_move_ms: int, root_move_ms: int) -> None:
        self.set_option("Team", TEAM_OPTION[team])
        self.set_option("TimeLeadMs", int(lead))
        self.set_option("TeamMoveTimeMs", max(1, int(team_move_ms)))
        self.set_option("OpponentMoveTimeMs", max(1, int(opp_move_ms)))
        self.set_option("RootMoveTimeMs", max(0, int(root_move_ms)))

    def search_clock(self, fen_a: str, fen_b: str, *, team: str, nodes: int, lead: int, team_move_ms: int,
                     opp_move_ms: int, root_move_ms: int = 0, multipv: int = 1,
                     options: dict | None = None) -> SearchResult:
        """Joint search for ``team`` under the clock model. ``lead`` is ``TimeLeadMs`` (see ``lead_ms``)."""
        with self._lock:
            self.start()
            self._set_time(team, lead, team_move_ms, opp_move_ms, root_move_ms)
            self.set_option("MultiPV", int(multipv))
            for k, v in (options or {}).items():
                self.set_option(k, v)
            if self.reset_tree:
                self._send("ucinewgame")
            self._send(f"position fen {fen_a} | {fen_b}")
            self._sync()
            lines: list[str] = []
            t0 = time.perf_counter()
            self._send(f"go nodes {max(1, int(nodes))}")
            best = self._wait_for(lambda l: l.startswith("bestmove"), collect=lines)
            wall_ms = int((time.perf_counter() - t0) * 1000)
            self._send("stop")  # end the permanent brain
            self._sync()
        return self._parse(lines, best, wall_ms)

    def raw_policy(self, fen_a: str, fen_b: str, *, team: str, lead: int, team_move_ms: int, opp_move_ms: int,
                   root_move_ms: int = 0, options: dict | None = None) -> RawPolicy:
        """The engine's own ``policy`` command under exactly the options a search would get."""
        with self._lock:
            self.start()
            self._set_time(team, lead, team_move_ms, opp_move_ms, root_move_ms)
            for k, v in (options or {}).items():
                self.set_option(k, v)
            self._send(f"position fen {fen_a} | {fen_b}")
            self._sync()
            lines: list[str] = []
            self._send("policy")
            self._send("isready")  # policy prints no terminator: readyok marks its end
            self._wait_for(lambda l: l == "readyok", collect=lines)
        return parse_policy(lines)

    def search(self, *a, **k):  # our fork's dialect (TimeAdvantage, clocks suffix) does not apply
        raise NotImplementedError("UpstreamHivemindProcess speaks upstream's UCI: use search_clock()")


def measure_upstream_ms_per_node(nodes: int = 800, repeats: int = 3, **kw) -> float:
    """Wall ms per *searched* node from startpos (team T1, level lead, 100 ms move times)."""
    start = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR[] w KQkq - 0 1"
    p = UpstreamHivemindProcess(**kw)
    try:
        args = dict(team="T1", lead=0, team_move_ms=DEFAULT_MOVE_MS, opp_move_ms=DEFAULT_MOVE_MS)
        p.search_clock(start, start, nodes=64, **args)  # warm-up
        rs = [p.search_clock(start, start, nodes=nodes, **args) for _ in range(repeats)]
    finally:
        p.close()
    return sum(r.wall_ms for r in rs) / max(1, sum(r.nodes for r in rs))


class UpstreamTeamController:
    """One upstream engine process per team, searching joint actions for both seats (like our centralized
    ``HivemindTeamController``): one search serves both seats when they are asked at the same position.

    Every search sends ``TimeLeadMs`` from the four clocks, ``TeamMoveTimeMs``/``OpponentMoveTimeMs`` from the
    clock-read EMAs (``MoveTimeTracker``; before a team has a sample, this search's own budget stands in, i.e. equal
    speeds), and ``RootMoveTimeMs`` = this search's charged budget in ms. A pass on the seat's board becomes a wait
    that ends when the next move lands (any board) or after ``sit_ms`` (default: the team's move time, at least
    100 ms), then the controller searches again. ``serial`` (default True): the team has one machine.

    ``reuse_tree=True`` keeps the tree between searches (no ``ucinewgame``; upstream reuses ~13% of visits);
    off by default so every decision gets exactly its budget.
    """

    kind = "upstream"
    label = "Upstream-v3"

    def __init__(self, team: str, process: UpstreamHivemindProcess, nodes: int = 400,
                 ms_per_node: float = UPSTREAM_MS_PER_NODE, *, clock_k: float | None = None, min_nodes: int = 16,
                 max_nodes: int = 20000, serial: bool = True, sit_ms: int | None = None,
                 charge: str = "searched", ema_alpha: float = 0.3, seed: int = 0) -> None:
        if clock_k is not None and clock_k <= 0:
            raise ValueError(f"clock_k must be > 0, got {clock_k}")
        if charge not in ("searched", "requested"):
            raise ValueError(f"charge must be 'searched' or 'requested', got {charge!r}")
        self.team = team
        self.opp_team = "T2" if team == "T1" else "T1"
        self.process = process
        self.nodes = int(nodes)
        self.ms_per_node = float(ms_per_node)
        self.clock_k = clock_k
        self.min_nodes, self.max_nodes = int(min_nodes), int(max_nodes)
        self.serial = bool(serial)
        self.sit_ms = sit_ms
        self.charge = charge
        self.tracker = MoveTimeTracker(alpha=ema_alpha)
        self._rng = random.Random(f"{seed}:{team}")
        self.searches = 0
        self.passes = 0
        self.no_root_results = 0
        self.fallback_moves = 0
        self.nodes_requested = 0
        self.nodes_searched = 0
        self.last_options: dict = {}
        self._busy_until = 0
        self._cache: tuple[tuple[str, str], SearchResult, int, int] | None = None
        self._seats: dict[str, "UpstreamSeat"] = {}
        self._capped = False  # budget() clamped to max_nodes during the current decision

    def budget(self, clock_ms: int) -> int:
        if self.clock_k is None:
            return self.nodes
        n = int(round(max(0, clock_ms) / self.clock_k / self.ms_per_node))
        self._capped = self._capped or n > self.max_nodes
        return min(self.max_nodes, max(self.min_nodes, n))

    def player(self, seat: str) -> "UpstreamSeat":
        if seat not in self._seats:
            if self._seats:
                self.process.acquire()
            self._seats[seat] = UpstreamSeat(seat, self)
        return self._seats[seat]

    def time_options(self, obs: Observation, budget_ms: int) -> dict:
        self.tracker.observe(obs.fens, obs.clocks_ms, obs.to_move)
        return {"lead": lead_ms(obs.clocks_ms, self.team),
                "team_move_ms": self.tracker.move_ms(self.team, budget_ms),
                "opp_move_ms": self.tracker.move_ms(self.opp_team, budget_ms),
                "root_move_ms": budget_ms}

    def default_sit_ms(self, opts: dict) -> int:
        """How long a pass waits (unless an event wakes it) when ``sit_ms`` is not set: the team's move time."""
        return max(DEFAULT_MOVE_MS, int(opts["team_move_ms"]))

    def decide(self, seat: str, obs: Observation) -> Decision:
        if not obs.legal_moves:
            return Decision(kind="wait", wait_ms=200, stall_reason="no_moves")
        board = BOARD_OF[seat]
        key = (obs.fens["A"], obs.fens["B"])
        cached = self._cache
        if cached and cached[0] == key and _legalize(cached[1].move_for(board), list(obs.legal_moves)):
            res, ready, sit_default = cached[1], cached[2], cached[3]
            think, nodes = max(0, ready - obs.now_ms), 0
            self.tracker.observe(obs.fens, obs.clocks_ms, obs.to_move)
        else:
            budget = self.budget(obs.clocks_ms[seat])
            budget_ms = int(round(budget * self.ms_per_node))
            opts = self.time_options(obs, budget_ms)
            self.last_options = dict(opts)
            res = self.process.search_clock(obs.fens["A"], obs.fens["B"], team=self.team, nodes=budget, **opts)
            self.searches += 1
            searched = res.nodes if res.nodes > 0 else budget
            self.nodes_requested += budget
            self.nodes_searched += searched
            charged = searched if self.charge == "searched" else budget
            own = int(round(charged * self.ms_per_node))
            queued = max(0, self._busy_until - obs.now_ms) if self.serial else 0
            think, nodes = queued + own, charged
            self._busy_until = obs.now_ms + think
            sit_default = self.default_sit_ms(opts)
            self._cache = (key, res, obs.now_ms + think, sit_default)
            if res.no_root:
                self.no_root_results += 1
                _LOG.error("upstream returned bestmove (none) for %s (%s) with %d legal moves; fenA=%s fenB=%s "
                           "clocks=%s", seat, self.team, len(obs.legal_moves), obs.fens["A"], obs.fens["B"],
                           obs.clocks_ms)
        move = _legalize(res.move_for(board), list(obs.legal_moves))
        if move is None and res.move_for(board) is None and not res.no_root:
            self.passes += 1
            sit = self.sit_ms if self.sit_ms is not None else sit_default
            return Decision(kind="wait", think_ms=think, wait_ms=think + int(sit), nodes=nodes, eval=res.q,
                            stall_reason="upstream_pass")
        if move is None:
            self.fallback_moves += 1
            b = parse_board(obs.fens[board])
            move = evaluate_moves(b, list(obs.legal_moves), wanted=None, rng=random.Random(obs.ply))[0].uci
        return Decision(kind="move", uci=move, think_ms=think, nodes=nodes, eval=res.q)


class UpstreamSeat:
    kind = "upstream"

    def __init__(self, seat: str, controller: UpstreamTeamController) -> None:
        self.seat = seat
        self.controller = controller
        self.kind = controller.kind
        c = controller
        self.name = f"{c.label}-{seat}-" + (f"k{c.clock_k:g}" if c.clock_k is not None else f"{c.nodes}n")
        self._closed = False

    def decide(self, obs: Observation) -> Decision:
        self.controller._capped = False
        dec = self.controller.decide(self.seat, obs)
        if self.controller._capped and dec.nodes > 0:
            dec = _dc_replace(dec, capped=True)
        return dec

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self.controller.process.release()


CONTROLLER_KW = ("clock_k", "min_nodes", "max_nodes", "serial", "sit_ms", "charge", "ema_alpha")


def upstream_team(team: str, *, seed: int = 0, nodes: int = 400, **kw) -> dict:
    """{seat: Player} for one team played by upstream v3 (``kind="upstream"``; the setup is always centralized:
    upstream's engine searches joint actions for a team). Options: ``binary``, ``model``, ``ms_per_node``,
    ``coreml``, ``reuse_tree``, ``engine_options`` and the controller's ``CONTROLLER_KW``."""
    proc = UpstreamHivemindProcess(kw.get("binary") or UPSTREAM_BINARY, kw.get("model") or UPSTREAM_MODEL,
                                   options=kw.get("engine_options"), reset_tree=not kw.get("reuse_tree", False),
                                   coreml=kw.get("coreml"))
    ckw = {k: kw[k] for k in CONTROLLER_KW if kw.get(k) is not None}
    ctl = UpstreamTeamController(team, proc, nodes=nodes,
                                 ms_per_node=float(kw.get("ms_per_node") or UPSTREAM_MS_PER_NODE), seed=seed, **ckw)
    return {s: ctl.player(s) for s in team_seats(team)}


# ---------------------------------------------------------------------------------------------- wave 2


def seat_values(values: dict) -> str:
    """``"<Aw> <Ab> <Bw> <Bb>"`` (whole ms, >= 0), the seat order of ``SeatClocksMs``/``SeatPaceMs``/``SeatElapsedMs``."""
    return " ".join(str(max(0, int(round(values[s])))) for s in SEATS)


def wave2_flag_options(**overrides) -> dict:
    """The wave 2 engine flags as UCI options; ``overrides`` (player option names of ``WAVE2_FLAGS``, None = keep)
    switch single flags for ablation rows."""
    bad = set(overrides) - set(WAVE2_FLAGS)
    if bad:
        raise ValueError(f"unknown wave 2 flag(s) {sorted(bad)}; expected some of {sorted(WAVE2_FLAGS)}")
    out = {}
    for key, (uci, value) in WAVE2_FLAGS.items():
        v = overrides.get(key)
        out[uci] = value if v is None else v
    k = out["PassCapK"]
    if isinstance(k, bool) or not 1 <= int(k) <= 64:
        raise ValueError(f"pass_cap_k must be in 1..64, got {k}")
    out["PassCapK"] = int(k)
    for uci in out:
        if uci != "PassCapK" and not isinstance(out[uci], bool):
            raise ValueError(f"{uci} must be true or false, got {out[uci]!r}")
    return out


def seat_elapsed_ms(obs: Observation) -> dict:
    """{seat: ms} the seat on turn on each board has been thinking (``now - last landing`` there), 0 for the seats
    not on turn and for every seat when the observation has no landing times."""
    out = {s: 0 for s in SEATS}
    if obs.last_move_ms is not None:
        for board, seat in obs.to_move.items():
            if seat in out:
                out[seat] = max(0, int(obs.now_ms) - int(obs.last_move_ms[board]))
    return out


# An opponent seat on turn that has gone SIT_INFER_FACTOR x its expected thinking time without moving is taken to be
# sitting. The arena, like a human, sees only that seat's clock running, not whether it thinks or waits; without
# this the tree prices that seat as landing at max(1, pace - elapsed) ms, so waiting it out looks free and the engine
# sits into sit-offs it loses on the lower clock (spec v4 sit-off rule: the lower clock must move). The expected time
# is min(seat pace, that seat's own budget): a pace EMA counts that seat's earlier sits too and runs high, while both
# arms think about clock / clock_k, so the budget is the seat's own clock at its decision start (now + elapsed) /
# clock_k (this search's budget only without clock_k). When the other team is on turn on both boards it is doubled:
# a serial brain searches its second seat after the first (queued + own). Our own budget is not the opponent's: with
# it, an opponent with more clock than us read as sitting exactly when we were behind (review finding).
# A wave 2 pass also re-checks after SIT_INFER_FACTOR x budget (default_sit_ms):
# a sitting opponent lands nothing, so no event would wake us before RULES v3 adjudicates the sit-off (5 s), and
# the seat's clock runs whether it sits or searches.
SIT_INFER_FACTOR = 1.5


def seat_sitting(obs: Observation, team: str, paces: dict, budget_ms: int,
                 factor: float = SIT_INFER_FACTOR, clock_k: float | None = None) -> dict:
    """{seat: 0/1} for ``SeatSitting``: 1 for a seat of the other team on turn whose thinking so far
    (``seat_elapsed_ms``) exceeds ``factor x min(its pace, its budget)``, its budget being (its clock + elapsed) /
    ``clock_k`` (``budget_ms`` without ``clock_k``), doubled while the other team is on turn on both boards; 0 for
    every other seat (the engine ignores our own)."""
    elapsed = seat_elapsed_ms(obs)
    out = {s: 0 for s in SEATS}
    on_turn = [s for s in obs.to_move.values() if s in out and TEAM_OF[s] != team]
    for seat in on_turn:
        if clock_k:
            own = (max(0, int(obs.clocks_ms[seat])) + elapsed[seat]) / float(clock_k)
        else:
            own = float(budget_ms)
        if len(on_turn) == 2:
            own *= 2
        expected = max(1.0, min(float(paces[seat]), own))
        if elapsed[seat] > factor * expected:
            out[seat] = 1
    return out


# Wave 2b: the UCI names of the seats' pockets (SeatArrivalsMs) and how long a landing counts as recent (the engine's
# RECENT_LANDING_WINDOW_MS; older ones are dropped by the engine too).
UCI_SEAT = {"A_white": "Aw", "A_black": "Ab", "B_white": "Bw", "B_black": "Bb"}
ARRIVAL_WINDOW_MS = 5000
POCKET_PIECES = "PNBRQ"


def fen_pockets(fen: str) -> dict:
    """{"white": {piece: n}, "black": {piece: n}} of a crazyhouse FEN's ``[...]`` pocket (P N B R Q; kings and
    anything else ignored). A FEN without brackets has empty pockets."""
    board = fen.split()[0] if fen else ""
    inside = board[board.find("[") + 1:board.find("]")] if "[" in board and "]" in board else ""
    out = {"white": {p: 0 for p in POCKET_PIECES}, "black": {p: 0 for p in POCKET_PIECES}}
    for ch in inside:
        if ch.upper() in POCKET_PIECES:
            out["white" if ch.isupper() else "black"][ch.upper()] += 1
    return out


class PocketLandingTracker:
    """Recent pocket landings for ``SeatArrivalsMs``, inferred from the pockets the controller sees: a piece more in a
    seat's pocket than at the previous observation landed when the last move landed on the other board (the capture
    that sent it), else at ``now``. Only the arena's observations at our own decisions are seen, so two captures on
    the other board between them both get the later landing time, and a capture undone by a drop of the same piece
    in between is missed; the landings themselves are exact otherwise. The first observation (and a clock going
    back: a new game) only records the pockets."""

    def __init__(self, window_ms: int = ARRIVAL_WINDOW_MS) -> None:
        self.window_ms = int(window_ms)
        self._pockets: dict | None = None
        self._now = None
        self.landings: list[tuple[int, str, str]] = []  # (landed at ms, seat, piece)

    def observe(self, fens: dict, now_ms: int, last_move_ms: dict | None = None) -> None:
        now = int(now_ms)
        if self._now is not None and now < self._now:
            self._pockets, self.landings = None, []
        pockets = {}
        for board in ("A", "B"):
            parsed = fen_pockets(fens[board])
            for color in ("white", "black"):
                pockets[f"{board}_{color}"] = parsed[color]
        if self._pockets is not None:
            for seat in SEATS:
                other = "B" if BOARD_OF[seat] == "A" else "A"
                landed = now if last_move_ms is None else min(now, int(last_move_ms[other]))
                for piece in POCKET_PIECES:
                    for _ in range(pockets[seat][piece] - self._pockets[seat][piece]):
                        self.landings.append((landed, seat, piece))
        self._pockets, self._now = pockets, now
        self.landings = [x for x in self.landings if now - x[0] < self.window_ms]

    def value(self, now_ms: int) -> str:
        """``SeatArrivalsMs``: ``"<pocket>:<piece>:<ms_ago>,..."``, newest first; "" when nothing landed lately."""
        now = int(now_ms)
        recent = sorted((x for x in self.landings if 0 <= now - x[0] < self.window_ms), key=lambda x: -x[0])
        return ",".join(f"{UCI_SEAT[seat]}:{piece}:{now - at}" for at, seat, piece in recent)


class Wave2TeamController(UpstreamTeamController):
    """The wave 2 engine (spec v4 durative tree in upstream v3) as a team brain: everything ``UpstreamTeamController``
    sends, plus per search ``SeatClocksMs`` (the arena's four clocks, current at ``now``: the on-turn seats are
    already charged their thinking so far), ``SeatPaceMs`` (per-seat EMA of observed action times; a seat without a
    sample uses its team's EMA, then this search's budget), ``SeatElapsedMs`` (``seat_elapsed_ms``) and
    ``SeatSitting`` (``seat_sitting``: opponent seats on turn well past their pace, so the sit-off rule applies) and
    ``SeatArrivalsMs`` (``PocketLandingTracker``: pieces that landed in each pocket in the last 5 s; the wide planes
    read them only with a wave 2b wide network; without InstantTransfer any network's tree holds an opponent seat's
    landings younger than its elapsed time as not yet droppable). The flags
    (``WAVE2_FLAGS``) are set once when the engine starts (``engine_options``).

    A pass waits ``min(PassCapK, SIT_INFER_FACTOR) x`` this search's budget unless an event (a landing on either
    board) wakes it first: the tree's "a pass lasts until the next event on the other board, capped at k x
    RootMoveTimeMs", re-checked sooner because a sitting opponent lands nothing; ``sit_ms`` overrides it.

    ``RootMoveTimeMs`` (what each of our actions costs in the tree) is ``max(budget, root_min_ms) + root_lag_ms``:
    under arena rules v3 a landing costs at least ``min_move_ms`` (100) plus a lag of 0..150 ms (mean 75), so the
    wave 2 pool sets 100 and 75; the opponents' paces, read off the clocks, already include both. Both 0 (the
    default) send the bare budget. ``SeatElapsedMs`` is ``now - last landing`` on the board: the arena cannot see
    when a sitting opponent re-decided, while self-play sends the time since the seat's current decision began,
    so for a sitting opponent the tree here lands it sooner (``max(1, pace - elapsed)``) than self-play would."""

    kind = "wave2"
    label = "Wave2"

    def __init__(self, team: str, process: UpstreamHivemindProcess, nodes: int = 400,
                 ms_per_node: float = WAVE2_MS_PER_NODE, *, pass_cap_k: int = 8, root_min_ms: int = 0,
                 root_lag_ms: int = 0, **kw) -> None:
        super().__init__(team, process, nodes, ms_per_node, **kw)
        self.pass_cap_k = int(pass_cap_k)
        if root_min_ms < 0 or root_lag_ms < 0:
            raise ValueError(f"root_min_ms and root_lag_ms must be >= 0, got {root_min_ms}, {root_lag_ms}")
        self.root_min_ms, self.root_lag_ms = int(root_min_ms), int(root_lag_ms)
        self.landings = PocketLandingTracker()

    def time_options(self, obs: Observation, budget_ms: int) -> dict:
        opts = super().time_options(obs, budget_ms)  # observes the clocks first
        paces = {s: self.tracker.seat_ms(s, budget_ms) for s in SEATS}
        self.landings.observe(obs.fens, obs.now_ms, obs.last_move_ms)
        opts["options"] = {
            "SeatClocksMs": seat_values(obs.clocks_ms),
            "SeatPaceMs": seat_values(paces),
            "SeatElapsedMs": seat_values(seat_elapsed_ms(obs)),
            "SeatSitting": seat_values(seat_sitting(obs, self.team, paces, budget_ms, clock_k=self.clock_k)),
            "SeatArrivalsMs": self.landings.value(obs.now_ms),
        }
        opts["root_move_ms"] = max(int(budget_ms), self.root_min_ms) + self.root_lag_ms
        return opts

    def default_sit_ms(self, opts: dict) -> int:
        budget = int(opts["root_move_ms"])
        return max(DEFAULT_MOVE_MS, min(self.pass_cap_k * budget, int(SIT_INFER_FACTOR * budget) + 1))


# Wave 2 controller options beyond CONTROLLER_KW (Wave2TeamController: our own action cost in the tree).
WAVE2_CONTROLLER_KW = ("root_min_ms", "root_lag_ms")


def wave2_team(team: str, *, seed: int = 0, nodes: int = 400, **kw) -> dict:
    """{seat: Player} for one team played by the wave 2 engine (``kind="wave2"``, always one team brain). Options as
    ``upstream_team`` (``binary``/``model`` default to ``WAVE2_BINARY``/``WAVE2_MODEL``) plus the flags of
    ``WAVE2_FLAGS`` (``durative_tree``, ``sit_gate``, ``flag_loss``, ``instant_transfer``, ``mate_race_by_landing``,
    ``pass_cap_k``); ``engine_options`` are sent after the flags, so they win."""
    flags = wave2_flag_options(**{k: kw.get(k) for k in WAVE2_FLAGS})
    proc = UpstreamHivemindProcess(kw.get("binary") or WAVE2_BINARY, kw.get("model") or WAVE2_MODEL,
                                   options={**flags, **(kw.get("engine_options") or {})},
                                   reset_tree=not kw.get("reuse_tree", False), coreml=kw.get("coreml"),
                                   cwd=WAVE2_ENGINE_DIR)
    ckw = {k: kw[k] for k in CONTROLLER_KW if kw.get(k) is not None}
    ckw.update({k: int(kw[k]) for k in WAVE2_CONTROLLER_KW if kw.get(k) is not None})
    ctl = Wave2TeamController(team, proc, nodes=nodes, ms_per_node=float(kw.get("ms_per_node") or WAVE2_MS_PER_NODE),
                              seed=seed, pass_cap_k=flags["PassCapK"], **ckw)
    return {s: ctl.player(s) for s in team_seats(team)}


def measure_wave2_ms_per_node(nodes: int = 800, repeats: int = 3, *, flags: dict | None = None, **kw) -> float:
    """``measure_upstream_ms_per_node`` for the wave 2 binary with the wave 2 flags set (``flags``: overrides as for
    ``wave2_flag_options``), at level clocks of 60 s and 100 ms paces."""
    kw.setdefault("binary", WAVE2_BINARY)
    kw.setdefault("model", WAVE2_MODEL)
    kw.setdefault("cwd", WAVE2_ENGINE_DIR)
    kw["options"] = {**wave2_flag_options(**(flags or {})), "SeatClocksMs": "60000 60000 60000 60000",
                     "SeatPaceMs": "100 100 100 100", "SeatElapsedMs": "0 0 0 0", **(kw.get("options") or {})}
    return measure_upstream_ms_per_node(nodes=nodes, repeats=repeats, **kw)


__all__ = ["UpstreamHivemindProcess", "UpstreamTeamController", "UpstreamSeat", "MoveTimeTracker", "RawPolicy",
           "lead_ms", "parse_policy", "upstream_team", "upstream_available", "measure_upstream_ms_per_node",
           "UPSTREAM_BINARY", "UPSTREAM_MODEL", "UPSTREAM_MS_PER_NODE", "UPSTREAM_NETWORK",
           "Wave2TeamController", "wave2_team", "wave2_flag_options", "seat_values", "seat_elapsed_ms", "seat_sitting",
           "measure_wave2_ms_per_node", "PocketLandingTracker", "fen_pockets", "WAVE2_BINARY", "WAVE2_MODEL", "WAVE2_MS_PER_NODE", "WAVE2_FLAGS",
           "WAVE2_ENGINE_DIR"]
