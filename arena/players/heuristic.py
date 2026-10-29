"""Fast, engine-free bughouse player that communicates.

Move choice is a 1-ply greedy search over the seat's own board (python-chess ``CrazyhouseBoard``):

* immediate mate (arena rules, see ``arena/RULES.md``: a check that a drop could block is *not* mate)
  scores +1000; a check that leaves the opponent stuck (no legal move, blockable) scores +30;
* material: value of the captured piece (a promoted piece counts as a pawn, as it transfers as one);
* checks (+0.3, drop checks +0.5), drops near the enemy king, castling, central squares;
* a crude static-exchange penalty if the moved/dropped piece lands en prise;
* for the best few candidates, a heavy penalty if the opponent then has a mate in one;
* a bonus for capturing a piece type the partner needs (from a ``need_X`` message, or, for the
  centralized team brain, computed directly from the partner's board).

Communication (``comm``):

* ``"none"``    silent: sends nothing, ignores its inbox, only looks at its own board.
* ``"request"`` the request protocol: if a drop of a piece type it lacks would mate, send ``need_X``
  (payload ``piece``, ``deadlineTs``, ``value``) and, if its team is not behind on time, sit
  (``stall_reason="need_piece"``) and send ``stall``. A partner ``no`` or ``go`` ends the sit. If the
  partner asked for X and it captures X it replies ``have_X``, otherwise ``no``. It sends ``go`` when its
  partner sits while their team is behind on time. A partner ``sit`` is honoured while the team is
  ahead on time.
* ``"share"``   ``"request"`` plus a status message (token ``ok``, payload ``kind="status"``, best move,
  eval, needed pieces) on every decision.

Think time depends on the number of legal moves, the number of tactical (capturing/checking) moves and
the seat's remaining clock, roughly 200-2500 ms, and shrinks under time pressure. All randomness comes
from a ``random.Random`` seeded with (seed, seat), so play is deterministic given a seed.

``team_time_edge`` (used by every player here) defines "ahead/behind on time" for a team as the sum over
its two seats of (own clock - clock of the opponent on the same board).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import chess
import chess.variant

from arena.interfaces import BOARD_OF, COLOR_OF, OPPONENT, PARTNER, TEAM_OF, Decision, Message, Observation

PIECE_VALUE = {chess.PAWN: 1.0, chess.KNIGHT: 3.0, chess.BISHOP: 3.0, chess.ROOK: 5.0, chess.QUEEN: 9.0,
               chess.KING: 0.0}
LETTER = {chess.PAWN: "P", chess.KNIGHT: "N", chess.BISHOP: "B", chess.ROOK: "R", chess.QUEEN: "Q"}
PTYPE = {v: k for k, v in LETTER.items()}
DROP_TYPES = (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN)  # cheapest first
CENTER = chess.SquareSet([chess.D4, chess.E4, chess.D5, chess.E5])

MATE_SCORE = 1000.0
STUCK_SCORE = 30.0
MATED_PENALTY = 500.0


# ---------------------------------------------------------------------------------------------- helpers

def team_seats(team: str) -> tuple[str, str]:
    return tuple(s for s in TEAM_OF if TEAM_OF[s] == team)  # type: ignore[return-value]


def team_time_edge(clocks_ms: dict[str, int], team: str) -> int:
    """Sum over the team's seats of (own clock - same-board opponent clock). >0: team ahead on time."""
    return sum(int(clocks_ms[s]) - int(clocks_ms[OPPONENT[s]]) for s in team_seats(team))


def seat_color(seat: str) -> bool:
    return chess.WHITE if COLOR_OF[seat] == "white" else chess.BLACK


def parse_board(fen: str) -> chess.variant.CrazyhouseBoard:
    return chess.variant.CrazyhouseBoard(fen)


def as_side(board: chess.variant.CrazyhouseBoard, color: bool) -> chess.variant.CrazyhouseBoard:
    """Copy of board with ``color`` to move (hypothetical analysis when it is not that side's turn)."""
    b = board.copy(stack=False)
    if b.turn != color:
        b.turn = color
        b.ep_square = None
    return b


def is_bughouse_mate(b: chess.variant.CrazyhouseBoard) -> bool:
    """Arena rule: in check, no legal move, and the check cannot be blocked by a (future) drop."""
    if not b.is_check():
        return False
    if any(True for _ in b.generate_legal_moves()):
        return False
    checkers = b.checkers()
    if len(checkers) > 1:
        return True
    sq = next(iter(checkers))
    king = b.king(b.turn)
    return b.piece_type_at(sq) in (chess.KNIGHT, chess.PAWN) or chess.square_distance(sq, king) <= 1


def is_stuck(b: chess.variant.CrazyhouseBoard) -> bool:
    return b.is_check() and not any(True for _ in b.generate_legal_moves()) and not is_bughouse_mate(b)


def captured_type(b: chess.variant.CrazyhouseBoard, move: chess.Move) -> int | None:
    """Piece type that would transfer to the partner if move is played (promoted pieces -> pawn)."""
    if move.drop:
        return None
    if b.is_en_passant(move):
        return chess.PAWN
    pt = b.piece_type_at(move.to_square)
    if pt is None:
        return None
    if b.promoted & chess.BB_SQUARES[move.to_square]:
        return chess.PAWN
    return pt


def opponent_has_mate_in_one(b: chess.variant.CrazyhouseBoard) -> bool:
    """b: position with the opponent to move."""
    for m in list(b.generate_legal_moves()):
        if b.gives_check(m):
            b.push(m)
            mate = is_bughouse_mate(b)
            b.pop()
            if mate:
                return True
    return False


def mate_drops_needed(board: chess.variant.CrazyhouseBoard, color: bool) -> list[tuple[int, str]]:
    """Drops of piece types ``color`` does NOT hold that would mate immediately: [(ptype, uci), ...],
    cheapest piece first."""
    b0 = as_side(board, color)
    out: list[tuple[int, str]] = []
    for pt in DROP_TYPES:
        if b0.pockets[color].count(pt) > 0:
            continue
        b = b0.copy(stack=False)
        b.pockets[color].add(pt)
        for m in list(b.generate_legal_drops()):
            if m.drop != pt or not b.gives_check(m):
                continue
            b.push(m)
            mate = is_bughouse_mate(b)
            b.pop()
            if mate:
                out.append((pt, m.uci()))
                break
    return out


def can_capture_type(board: chess.variant.CrazyhouseBoard, color: bool, pt: int) -> bool:
    """Does ``color`` attack an enemy piece of type pt on this board (pseudo-legal, ignoring turn)?"""
    for sq in board.pieces(pt, not color):
        if board.attackers(color, sq):
            return True
    return False


@dataclass
class MoveEval:
    uci: str
    score: float
    mate: bool = False
    captures: int | None = None
    check: bool = False


def evaluate_moves(board: chess.variant.CrazyhouseBoard, legal: list[str], *, wanted: dict[int, float] | None,
                   rng: random.Random, safety_top_k: int = 6) -> list[MoveEval]:
    """Score every legal move (1-ply), sorted best first. ``wanted``: piece type -> capture bonus."""
    wanted = wanted or {}
    us = board.turn
    enemy_king = board.king(not us)
    evals: list[MoveEval] = []
    for uci in legal:
        try:
            move = chess.Move.from_uci(uci)
        except ValueError:
            continue
        if not board.is_legal(move):
            evals.append(MoveEval(uci, -1.0 + rng.random() * 0.01))
            continue
        cap = captured_type(board, move)
        moved_pt = move.drop if move.drop else board.piece_type_at(move.from_square)
        castle = board.is_castling(move)
        check = board.gives_check(move)
        score = PIECE_VALUE[cap] if cap else 0.0
        if cap and cap in wanted:
            score += wanted[cap]
        b = board.copy(stack=False)
        b.push(move)
        mate = False
        if check:
            if is_bughouse_mate(b):
                mate = True
                score += MATE_SCORE
            elif is_stuck(b):
                score += STUCK_SCORE
            else:
                score += 0.5 if move.drop else 0.3
        if move.promotion:
            score += PIECE_VALUE[move.promotion] - 1.0
        # crude static exchange: is the piece we just placed en prise?
        if not mate and moved_pt != chess.KING:
            attackers = b.attackers(not us, move.to_square)
            if attackers:
                v = PIECE_VALUE[move.promotion or moved_pt]
                defenders = b.attackers(us, move.to_square)
                min_att = min(PIECE_VALUE[b.piece_type_at(s)] or 0.5 for s in attackers)
                if not defenders:
                    score -= 0.9 * v
                elif min_att < v:
                    score -= 0.9 * (v - min_att)
        if castle:
            score += 0.4
        elif moved_pt == chess.KING:
            score -= 0.3
        if move.to_square in CENTER:
            score += 0.1
        if move.drop and enemy_king is not None and chess.square_distance(move.to_square, enemy_king) <= 2:
            score += 0.15
        if not move.drop and moved_pt in (chess.KNIGHT, chess.BISHOP) and \
                chess.square_rank(move.from_square) == (0 if us == chess.WHITE else 7):
            score += 0.15
        score += rng.random() * 0.05
        evals.append(MoveEval(uci, score, mate, cap, check))
    evals.sort(key=lambda e: -e.score)
    # safety pass: does the opponent get a mate in one after our best few candidates?
    for e in evals[:safety_top_k]:
        if e.mate:
            continue
        b = board.copy(stack=False)
        b.push(chess.Move.from_uci(e.uci))
        if opponent_has_mate_in_one(b):
            e.score -= MATED_PENALTY
    evals.sort(key=lambda e: -e.score)
    return evals


def score_to_eval(score: float) -> float:
    if score >= MATE_SCORE / 2:
        return 1.0
    if score <= -MATED_PENALTY / 2:
        return -1.0
    return round(math.tanh(score / 6.0), 3)


def think_time(n_legal: int, n_tactical: int, my_clock_ms: int, rng: random.Random, lo: int, hi: int,
               quick: bool = False) -> int:
    if quick:
        t = lo + rng.random() * 100
    else:
        t = min(hi, lo + 15 * n_legal + 120 * n_tactical)
        if my_clock_ms < 60_000:  # time pressure: speed up
            t *= max(0.2, my_clock_ms / 60_000)
        t *= rng.uniform(0.85, 1.15)
    t = min(t, hi, max(50, my_clock_ms / 8))
    return int(max(50, t))


# ---------------------------------------------------------------------------------------------- players

class HeuristicTeam:
    """Centralized heuristic controller: one brain for both seats of a team. It sees both seats'
    situations directly, so it needs no messages: a seat captures what its partner needs for a mate, and
    a seat only sits for a piece when its partner can actually capture that piece now and the team is not
    behind on time. ``player(seat)`` returns the per-seat ``HeuristicPlayer`` view."""

    def __init__(self, team: str, seed: int = 0, **kw) -> None:
        self.team = team
        self._players = {s: HeuristicPlayer(s, seed=seed, comm="none", team_brain=self, **kw)
                         for s in team_seats(team)}

    def player(self, seat: str) -> "HeuristicPlayer":
        return self._players[seat]

    def partner_needs(self, seat: str, obs: Observation) -> list[tuple[int, str]]:
        p = PARTNER[seat]
        return mate_drops_needed(parse_board(obs.fens[BOARD_OF[p]]), seat_color(p))

    def partner_can_capture(self, seat: str, obs: Observation, pt: int) -> bool:
        p = PARTNER[seat]
        return can_capture_type(parse_board(obs.fens[BOARD_OF[p]]), seat_color(p), pt)

    def partner_stalling(self, seat: str) -> bool:
        return self._players[PARTNER[seat]]._stall is not None


class HeuristicPlayer:
    kind = "heuristic"

    def __init__(self, seat: str, seed: int | None = 0, *, comm: str = "request", min_think_ms: int = 200,
                 max_think_ms: int = 2500, stall_max_ms: int = 6000, request_window_ms: int = 5000,
                 min_stall_clock_ms: int = 4000, team_brain: HeuristicTeam | None = None) -> None:
        if comm not in ("none", "request", "share"):
            raise ValueError(f"unknown comm mode {comm!r}")
        self.seat = seat
        self.comm = comm
        self.team_brain = team_brain
        self.name = f"Heuristic-{seat}" + ("-central" if team_brain else f"-{comm}")
        self.min_think_ms = min_think_ms
        self.max_think_ms = max_think_ms
        self.stall_max_ms = stall_max_ms
        self.request_window_ms = request_window_ms
        self.min_stall_clock_ms = min_stall_clock_ms
        self._rng = random.Random(f"heuristic-{seed}-{seat}")
        self._stall: dict | None = None  # {"piece": pt, "until": ms}
        self._no_stall_until: dict[int, int] = {}  # piece type -> ms (after a refusal / expired stall)
        self._sent_need: dict[int, int] = {}  # piece type -> deadline of the request we sent
        self._partner_requests: dict[int, tuple[int, float]] = {}  # piece type -> (deadline, value)
        self._partner_sitting = False
        self._go_sent = False
        self._sit_until = 0

    # ------------------------------------------------------------------ messages

    def _msg(self, token: str, text: str | None = None, **payload) -> Message:
        return Message(from_seat=self.seat, to_seat=PARTNER[self.seat], token=token, sent_ms=0, text=text,
                       payload=payload)

    def _read_inbox(self, obs: Observation) -> tuple[list[int], bool]:
        """Returns (newly requested piece types, partner said go)."""
        new_requests: list[int] = []
        go = False
        if self.comm == "none":
            return new_requests, go
        for m in obs.inbox:
            tok = m.token
            if tok.startswith("need_") and tok[5:] in PTYPE:
                pt = PTYPE[tok[5:]]
                deadline = int(m.payload.get("deadlineTs", obs.now_ms + self.request_window_ms))
                self._partner_requests[pt] = (deadline, float(m.payload.get("value", 0.5)))
                new_requests.append(pt)
            elif tok == "stall":
                self._partner_sitting = True
                self._go_sent = False
            elif tok == "go":
                go = True
            elif tok == "sit":
                self._sit_until = int(m.payload.get("deadlineTs", obs.now_ms + 2000))
            elif tok == "no":
                if self._stall is not None:
                    self._no_stall_until[self._stall["piece"]] = obs.now_ms + self.stall_max_ms
                    self._stall = None
        return new_requests, go

    # ------------------------------------------------------------------ decide

    def decide(self, obs: Observation) -> Decision:
        seat = self.seat
        now = obs.now_ms
        msgs: list[Message] = []
        new_requests, go = self._read_inbox(obs)
        if go:
            self._stall = None
            self._sit_until = 0
        partner = PARTNER[seat]
        if obs.to_move.get(BOARD_OF[partner]) != partner:
            self._partner_sitting = False
        self._partner_requests = {pt: v for pt, v in self._partner_requests.items() if v[0] >= now}

        if not obs.legal_moves:
            return Decision(kind="wait", wait_ms=200, stall_reason="no_moves")

        board = parse_board(obs.fens[BOARD_OF[seat]])
        team = TEAM_OF[seat]
        edge = team_time_edge(obs.clocks_ms, team)
        my_clock = int(obs.clocks_ms[seat])

        # what the partner wants
        wanted: dict[int, float] = {}
        if self.team_brain is not None:
            for pt, _ in self.team_brain.partner_needs(seat, obs):
                wanted[pt] = 2.0 + PIECE_VALUE[pt]
        else:
            for pt, (_, value) in self._partner_requests.items():
                wanted[pt] = 2.0 + PIECE_VALUE[pt] * (0.5 + value)

        evals = evaluate_moves(board, list(obs.legal_moves), wanted=wanted, rng=self._rng)
        best = evals[0]
        n_tactical = sum(1 for e in evals if e.captures or e.check)
        ev = score_to_eval(best.score)

        # do we need a piece we lack for a mating drop?
        needs: list[tuple[int, str]] = []
        if (self.comm != "none" or self.team_brain is not None) and not best.mate:
            needs = mate_drops_needed(board, seat_color(seat))
        need_pt = needs[0][0] if needs else None
        if need_pt is not None and self.comm != "none":
            if self._sent_need.get(need_pt, -1) < now:
                deadline = now + self.request_window_ms
                self._sent_need[need_pt] = deadline
                letter = LETTER[need_pt]
                msgs.append(self._msg(f"need_{letter}", f"need {letter} to mate ({needs[0][1]})",
                                      piece=letter, deadlineTs=deadline, value=0.9))

        # sit for it?
        can_stall = (
            need_pt is not None and not go and not board.is_check() and best.score < 3.0
            and edge >= 0 and my_clock > self.min_stall_clock_ms
            and self._no_stall_until.get(need_pt, -1) < now
        )
        if can_stall and self.team_brain is not None:
            can_stall = self.team_brain.partner_can_capture(seat, obs, need_pt)
        if can_stall:
            first = self._stall is None or self._stall["piece"] != need_pt
            if first:
                budget = min(self.stall_max_ms, max(1500, my_clock // 6))
                self._stall = {"piece": need_pt, "until": now + budget}
            if now < self._stall["until"]:
                if first and self.comm != "none":
                    msgs.append(self._msg("stall", f"sitting for {LETTER[need_pt]}",
                                          piece=LETTER[need_pt], deadlineTs=self._stall["until"]))
                if self.comm == "share":
                    msgs.append(self._status(best, ev, needs))
                return Decision(kind="wait", wait_ms=max(1, self._stall["until"] - now), eval=ev,
                                messages=tuple(msgs), stall_reason="need_piece")
            self._no_stall_until[need_pt] = now + self.stall_max_ms
        self._stall = None

        # partner asked us to sit
        if self._sit_until > now and edge > 0 and not best.mate and best.score < 3.0 \
                and my_clock > self.min_stall_clock_ms:
            if self.comm == "share":
                msgs.append(self._status(best, ev, needs))
            return Decision(kind="wait", wait_ms=self._sit_until - now, eval=ev, messages=tuple(msgs),
                            stall_reason="partner_request")

        # partner sits while we are behind on time: tell them to move
        if self.comm != "none" and self._partner_sitting and edge < 0 and not self._go_sent:
            msgs.append(self._msg("go", "we're behind on time, move"))
            self._go_sent = True

        # answer requests
        if self.comm != "none":
            if best.captures is not None and best.captures in self._partner_requests:
                letter = LETTER[best.captures]
                msgs.append(self._msg(f"have_{letter}", f"{letter} coming", piece=letter))
                self._partner_requests.pop(best.captures, None)
            for pt in new_requests:
                if pt in self._partner_requests and pt != best.captures:
                    msgs.append(self._msg("no", f"can't get {LETTER[pt]} now", piece=LETTER[pt]))
            if self.comm == "share":
                msgs.append(self._status(best, ev, needs))

        think = think_time(len(obs.legal_moves), n_tactical, my_clock, self._rng, self.min_think_ms,
                           self.max_think_ms, quick=best.mate)
        return Decision(kind="move", uci=best.uci, think_ms=think, eval=ev, messages=tuple(msgs))

    def _status(self, best: MoveEval, ev: float, needs: list[tuple[int, str]]) -> Message:
        return self._msg("ok", f"plan {best.uci} ({ev:+.2f})", kind="status", best=best.uci, eval=ev,
                         needs=[LETTER[pt] for pt, _ in needs])

    def close(self) -> None:
        pass
