"""The trained scheduler as an arena player (kind "agent", setup "sharing"). See docs/phase1_scheduler.md.

Each decision runs the quick look, samples gate, talk and think, then either waits or runs the long search and
plays its move. ``features`` turns on optional inputs and decisions (FEATURE_GROUPS); every extra engine probe
they need is paid for out of the same node budget as the move. Sampling does not keep gradients: with ``trajectory_prefix`` the player records every decision
to ``<prefix>_<seat>.json`` on close, and arena/train_scheduler.py recomputes the log probabilities from those.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, replace

import torch

from arena.interfaces import BOARD_OF, OPPONENT, PARTNER, SEATS, TEAM_OF, Decision, Message, Observation
from arena.players.context import add_to_pocket, capturable, captures_of, checking_drops
from arena.players.heuristic import LETTER, evaluate_moves, parse_board, seat_color, team_seats, team_time_edge
from arena.players.hivemind import (
    DEFAULT_BINARY,
    DEFAULT_MODEL,
    MS_PER_NODE,
    HivemindProcess,
    SearchResult,
    _legalize,
    _pv_needs,
    pv_offers,
)
from arena.players.scheduler import (
    END_WAIT_OPTIONS,
    GATE_OPTIONS,
    MOVE_INDEX,
    PIECES,
    SERVE_OPTIONS,
    START_EXPLORATION,
    TALK_OPTIONS,
    CommState,
    Features,
    SchedulerPolicy,
    exploration_prob,
    plan_nodes,
    plausible_wait,
    sample_action,
    sample_end_wait,
    sample_serve,
)

WAIT_RECHECK_MS = 5000
QUICK_NODES = 16
EXPECTED_DECISIONS_PER_GAME = 60  # rough; only scales the nodes_used_fraction input
REQUEST_WINDOW_MS = 3000

# Optional feature groups. Each adds inputs, and "serve" a decision; all are off by default.
#   tempo   how forced the move is: legal move count, in check, gap to the engine's second choice
#   mate    what one more piece in our pocket is worth (forced mate or eval gain), and whether the partner can
#           capture one
#   danger  what one more piece in our opponent's pocket would cost us; sends the partner "hold_X"
#   serve   when the partner asked for a piece: may play a capture of it that is close to the engine's choice
FEATURE_GROUPS = ("tempo", "mate", "danger", "serve")
MAX_GAIN_PROBES = 2  # extra searches per decision for "mate"
HOLD_THRESHOLD = 0.25  # eval we would lose before telling the partner to hold a piece
HOLD_RESEND_MS = 5000
SERVE_LINES = 4  # candidate moves the long search reports when a capture the partner wants is legal
SERVE_MAX_COST = 0.4  # a capture further than this from the engine's choice is never offered to the policy


def load_policy(path: str | None) -> SchedulerPolicy:
    """A checkpoint's policy, or a fresh one (which plays like stock) when path is None."""
    policy = SchedulerPolicy()
    if path:
        policy.load_state_dict(torch.load(path, weights_only=True))
    return policy.eval()


class SchedulerPlayer:
    kind = "agent"

    def __init__(self, seat: str, process: HivemindProcess, policy: SchedulerPolicy, nodes: int = 100,
                 ms_per_node: float = MS_PER_NODE, *, quick_nodes: int = QUICK_NODES, max_consecutive_sits: int = 2,
                 lend: bool = False, train_step: int = 0, total_steps: int = 0, seed: int = 0,
                 trajectory_prefix: str | None = None, features: tuple[str, ...] = (),
                 start_exploration: float = START_EXPLORATION, probe_nodes: int | None = None) -> None:
        self.seat = seat
        self.process = process
        self.policy = policy
        self.nodes = int(nodes)
        self.ms_per_node = float(ms_per_node)
        self.quick_nodes = int(quick_nodes)
        self.max_consecutive_sits = max_consecutive_sits
        self.lend = lend
        self.train_step = train_step
        self.total_steps = total_steps
        self.trajectory_prefix = trajectory_prefix
        unknown = set(features) - set(FEATURE_GROUPS)
        if unknown:
            raise ValueError(f"unknown scheduler features {sorted(unknown)}; choose from {FEATURE_GROUPS}")
        self.features = frozenset(features)
        self.start_exploration = float(start_exploration)
        self.probe_nodes = int(probe_nodes if probe_nodes is not None else quick_nodes)
        self._last_hold: tuple[str | None, int] = (None, -HOLD_RESEND_MS)
        self.name = f"Scheduler-{seat}-{self.nodes}n"
        self.comm = CommState(seat)
        self.rng = torch.Generator().manual_seed(seed)
        self.steps: list[dict] = []
        self._team = TEAM_OF[seat]
        self._board = BOARD_OF[seat]
        self._sits = 0
        self._nodes_used = 0
        self._last_eval: float | None = None
        self._quick: dict = {}  # the last quick look's Features fields, reused while waiting
        self._waiting = False
        self._waiting_for: str | None = None
        self._stall_reason = "time_advantage"
        self._last_sent_ms = -1
        self._closed = False

    # ------------------------------------------------------------------ inputs

    def _holding(self, obs: Observation) -> tuple[str, ...]:
        pocket = parse_board(obs.fens[self._board]).pockets[seat_color(self.seat)]
        return tuple(LETTER[pt] for pt in LETTER for _ in range(pocket.count(pt)))

    def _features(self, obs: Observation, holding: tuple[str, ...]) -> Features:
        clocks, partner = obs.clocks_ms, PARTNER[self.seat]
        expected = self.nodes * EXPECTED_DECISIONS_PER_GAME
        return Features(
            **self._quick,
            own_clock_ms=clocks[self.seat], partner_clock_ms=clocks[partner],
            own_opponent_clock_ms=clocks[OPPONENT[self.seat]], partner_opponent_clock_ms=clocks[OPPONENT[partner]],
            partner_waiting=self.comm.partner_waiting, offered_piece=self.comm.offered_piece,
            offered_plies=self.comm.offered_plies, requested_piece=self.comm.requested_piece,
            hold_piece=self.comm.hold_piece, holding=holding, sits_in_row=self._sits, nodes_used_fraction=self._nodes_used / expected,
        )

    def _msg(self, token: str, text: str, **payload) -> Message:
        return Message(from_seat=self.seat, to_seat=PARTNER[self.seat], token=token, sent_ms=0, text=text,
                       payload=payload)

    def _exploration(self, f: Features) -> float:
        return exploration_prob(self.train_step, self.total_steps, plausible_wait(f), self.start_exploration)

    def _probes(self, obs: Observation, holding: tuple[str, ...], q: float, ahead: bool,
                wants: tuple[str, ...]) -> tuple[dict, int]:
        """The "mate" and "danger" inputs and the nodes they cost. Each probe is a short search of the position
        with one more piece in a pocket. Only pieces that could be dropped with check and that could really
        arrive are probed, so most decisions run none."""
        if not self.features & {"mate", "danger"}:
            return {}, 0
        fen = obs.fens[self._board]
        board, color = parse_board(fen), seat_color(self.seat)
        partner = PARTNER[self.seat]
        partner_board = parse_board(obs.fens[BOARD_OF[partner]])
        out: dict = {}
        spent = 0

        def probe(to_color: bool, piece: str) -> SearchResult:
            fens = dict(obs.fens)
            fens[self._board] = add_to_pocket(fen, to_color, piece)
            self._nodes_used += self.probe_nodes
            return self.process.search(fens["A"], fens["B"], team=self._team, nodes=self.probe_nodes,
                                       time_advantage=ahead, require=self._board)

        if "mate" in self.features:
            incoming = capturable(partner_board, seat_color(partner))  # the partner's captures land in our pocket
            supply = [incoming.get(p, 0.0) for p in PIECES]
            checks = checking_drops(board, color)
            reachable = [p for p in PIECES if p in checks and p not in holding
                         and (incoming.get(p) or p == self.comm.offered_piece or p in wants)]
            reachable.sort(key=lambda p: -incoming.get(p, 0.0))
            gain, mates = [0.0] * len(PIECES), []
            for piece in reachable[:MAX_GAIN_PROBES]:
                res = probe(color, piece)
                spent += self.probe_nodes
                if res.q is not None:
                    gain[PIECES.index(piece)] = max(0.0, res.q - q)
                if res.mate is not None and res.mate > 0:
                    mates.append(piece)
            out.update(gain=tuple(gain), mate_with=tuple(mates), supply=tuple(supply))

        if "danger" in self.features:
            their_pocket = board.pockets[not color]
            arriving = capturable(partner_board, not seat_color(partner))  # those land in our opponent's pocket
            checks = checking_drops(board, not color)
            risky = [p for p in PIECES if p in checks and arriving.get(p)
                     and their_pocket.count(next(pt for pt, l in LETTER.items() if l == p)) == 0]
            risky.sort(key=lambda p: -arriving[p])
            danger = [0.0] * len(PIECES)
            for piece in risky[:1]:
                res = probe(not color, piece)
                spent += self.probe_nodes
                if res.q is not None:
                    danger[PIECES.index(piece)] = max(0.0, q - res.q)
            out.update(danger=tuple(danger))
        return out, spent

    def _hold_message(self, obs: Observation, danger: tuple[float, ...]) -> list[Message]:
        """Ask the partner not to lose the piece that would hurt us most, at most once every HOLD_RESEND_MS."""
        if len(danger) != len(PIECES) or max(danger) < HOLD_THRESHOLD:
            return []
        piece = PIECES[max(range(len(PIECES)), key=lambda i: danger[i])]
        last_piece, last_ms = self._last_hold
        if piece == last_piece and obs.now_ms - last_ms < HOLD_RESEND_MS:
            return []
        self._last_hold = (piece, obs.now_ms)
        return [self._msg(f"hold_{piece}", f"hold your {piece}", piece=piece, value=round(max(danger), 3))]

    # ------------------------------------------------------------------ decisions

    def decide(self, obs: Observation) -> Decision:
        self.comm.update(obs)
        if not obs.legal_moves:
            return Decision(kind="wait", wait_ms=200, stall_reason="no_moves")
        holding = self._holding(obs)
        if self.comm.offered_piece in holding:
            self.comm.clear_offer()
        if self._waiting and self._waiting_for in holding:  # the piece arrived, which ends the wait
            self._waiting = False
        if self._waiting and self._keep_waiting(obs, holding):
            return Decision(kind="wait", wait_ms=WAIT_RECHECK_MS, stall_reason=self._stall_reason)
        self._waiting = False
        return self._decide(obs, holding)

    def _keep_waiting(self, obs: Observation, holding: tuple[str, ...]) -> bool:
        f = self._features(obs, holding)
        with torch.no_grad():
            choice, _ = sample_end_wait(self.policy, f, self.rng)
        self.steps.append({"type": "end_wait", "features": asdict(f), "choice": choice})
        return END_WAIT_OPTIONS[choice] == "keep_waiting"

    def _decide(self, obs: Observation, holding: tuple[str, ...]) -> Decision:
        edge = team_time_edge(obs.clocks_ms, self._team)
        ahead = edge > 0
        may_sit = ahead and self._sits < self.max_consecutive_sits
        borrow = self.lend and obs.partner_machine_free
        base = self.nodes * 2 if borrow else self.nodes
        rate = self.ms_per_node / (2 if borrow else 1)

        tempo = "tempo" in self.features
        quick = self._search(obs, self.quick_nodes, ahead, None if may_sit else self._board, 2 if tempo else 1)
        fen = obs.fens[self._board]
        q = quick.q if quick.q is not None else 0.0
        offers = pv_offers(quick.pv, self._board, fen)
        self._quick = dict(
            eval=q, eval_change=0.0 if self._last_eval is None else q - self._last_eval,
            search_passed=quick.move_for(self._board) is None,
            wants=tuple(_pv_needs(quick.pv, self._board, fen)), can_offer=bool(offers),
        )
        self._last_eval = q
        if tempo:
            others = [line_q for _, _, line_q in quick.multi[1:]]
            self._quick.update(n_legal=len(obs.legal_moves), in_check=parse_board(fen).is_check(),
                               top_gap=q - max(others) if others else 1.0)
        extra, probe_spent = self._probes(obs, holding, q, ahead, self._quick["wants"])
        self._quick.update(extra)
        looked = self.quick_nodes + probe_spent  # nodes spent before the long search

        f = self._features(obs, holding)
        epsilon = self._exploration(f)
        with torch.no_grad():
            action = sample_action(self.policy, f, epsilon, self.rng)
        self.steps.append({"type": "decide", "features": asdict(f), "gate": action.gate, "talk": action.talk,
                           "think": action.think, "epsilon": epsilon})
        msgs = self._messages(obs, action.gate, action.talk, offers, self._hold_message(obs, f.danger))

        if action.gate != MOVE_INDEX:
            gate = GATE_OPTIONS[action.gate]
            self._sits += 1
            self._waiting = True
            self._waiting_for = gate[5:] if gate.startswith("wait_") else None
            self._stall_reason = "need_piece" if self._waiting_for else "time_advantage"
            think = int(round(looked * rate))
            return Decision(kind="wait", think_ms=think, wait_ms=think + WAIT_RECHECK_MS, nodes=looked,
                            eval=quick.q, messages=msgs, stall_reason=self._stall_reason, borrow=borrow)

        long_nodes = plan_nodes(base, action.think, looked)
        wanted = self.comm.requested_piece if "serve" in self.features else None
        wanted_captures = captures_of(parse_board(fen), list(obs.legal_moves), wanted) if wanted else []
        res = self._search(obs, long_nodes, ahead, self._board, SERVE_LINES if wanted_captures else 1)
        move = _legalize(res.move_for(self._board), list(obs.legal_moves))
        if move is None:  # engine move not legal in the arena (e.g. rules mismatch): 1-ply fallback
            b = parse_board(fen)
            move = evaluate_moves(b, list(obs.legal_moves), wanted=None, rng=random.Random(obs.ply))[0].uci
        if wanted_captures:
            move = self._serve(obs, f, res, move, wanted_captures)
            if move in wanted_captures:  # tell the partner it is coming, as stock does
                self.comm.requested_piece = None
                if msgs or obs.now_ms != self._last_sent_ms:
                    msgs += (self._msg(f"have_{wanted}", f"{wanted} coming", piece=wanted, plies=0),)
                    self._last_sent_ms = obs.now_ms
        self._sits = 0
        spent = looked + long_nodes
        return Decision(kind="move", uci=move, think_ms=int(round(spent * rate)), nodes=spent, eval=res.q,
                        messages=msgs, borrow=borrow)

    def _serve(self, obs: Observation, f: Features, res: SearchResult, move: str, wanted_captures: list[str]) -> str:
        """The move to play when a capture the partner asked for is legal: the engine's choice, or the best such
        capture among the engine's other candidate lines if the policy chooses to serve."""
        if move in wanted_captures or res.q is None:
            return move
        legal = list(obs.legal_moves)
        options = [(line_q, _legalize(a if self._board == "A" else b, legal)) for a, b, line_q in res.multi[1:]]
        options = [(line_q, uci) for line_q, uci in options if uci in wanted_captures]
        if not options:
            return move
        best_q, capture = max(options)
        cost = max(0.0, res.q - best_q)
        if cost > SERVE_MAX_COST:
            return move
        f = replace(f, serve_cost=cost)
        with torch.no_grad():
            choice, _ = sample_serve(self.policy, f, self.rng)
        self.steps.append({"type": "serve", "features": asdict(f), "choice": choice})
        return capture if SERVE_OPTIONS[choice] == "serve" else move

    def _search(self, obs: Observation, nodes: int, ahead: bool, require: str | None, multipv: int = 1) -> SearchResult:
        self._nodes_used += nodes
        more = {"multipv": multipv} if multipv > 1 else {}
        return self.process.search(obs.fens["A"], obs.fens["B"], team=self._team, nodes=nodes,
                                   time_advantage=ahead, require=require, **more)

    def _messages(self, obs: Observation, gate: int, talk: int, offers: list[tuple[str, int]],
                  extra: list[Message] | None = None) -> tuple[Message, ...]:
        # A message wakes a waiting partner at the same instant, so two waiting partners could wake each other
        # forever when delivery is instant. One send per seat per instant stops that.
        if obs.now_ms == self._last_sent_ms:
            return ()
        gate_name, talk_name = GATE_OPTIONS[gate], TALK_OPTIONS[talk]
        msgs: list[Message] = []
        asks = []
        if gate_name.startswith("wait_"):
            asks.append(gate_name[5:])
        if talk_name.startswith("need_") and talk_name[5:] not in asks:
            asks.append(talk_name[5:])
        for piece in asks:
            msgs.append(self._msg(f"need_{piece}", f"need {piece}", piece=piece,
                                  deadlineTs=obs.now_ms + REQUEST_WINDOW_MS))
        if talk_name == "offer" and offers:
            piece, plies = offers[0]
            msgs.append(self._msg(f"have_{piece}", f"{piece} in {plies} plies", piece=piece, plies=plies))
        msgs.extend(extra or [])
        if gate != MOVE_INDEX:
            msgs.append(self._msg("stall", "waiting", deadlineTs=obs.now_ms + WAIT_RECHECK_MS))
        if msgs:
            self._last_sent_ms = obs.now_ms
        return tuple(msgs)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self.trajectory_prefix:
            with open(f"{self.trajectory_prefix}_{self.seat}.json", "w") as f:
                json.dump({"seat": self.seat, "steps": self.steps}, f, separators=(",", ":"))
        self.process.release()


def agent_team(team: str, setup: str, *, seed: int, nodes: int, **kw) -> dict:
    """{seat: SchedulerPlayer} for one team, with the same compute split as the hivemind sharing setup."""
    if setup != "sharing":
        raise ValueError(f"kind 'agent' needs setup 'sharing', got {setup!r}")
    torch.set_num_threads(1)  # many games run side by side and this network is tiny
    policy = load_policy(kw.get("path"))
    pkw = dict(options=kw.get("engine_options"), reset_tree=kw.get("reset_tree", True))
    binary, model = kw.get("binary", DEFAULT_BINARY), kw.get("model", DEFAULT_MODEL)
    ms = float(kw.get("ms_per_node", MS_PER_NODE))
    compute = kw.get("compute") or "throughput"
    if compute == "throughput":
        ms *= 2
    lend = bool(kw.get("lend", False))
    if lend and compute != "throughput":
        raise ValueError("lend needs compute='throughput' (each player must be half a machine)")
    skw = {k: kw[k] for k in ("max_consecutive_sits", "train_step", "total_steps", "trajectory_prefix", "quick_nodes",
                              "start_exploration", "probe_nodes") if kw.get(k) is not None}
    skw["features"] = tuple(kw.get("features") or ())
    return {s: SchedulerPlayer(s, HivemindProcess(binary, model, **pkw), policy, nodes=max(1, nodes // 2),
                               ms_per_node=ms, lend=lend, seed=seed * 7 + SEATS.index(s), **skw)
            for s in team_seats(team)}
