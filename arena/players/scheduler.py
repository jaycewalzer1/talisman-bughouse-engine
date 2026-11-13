"""Phase 1 scheduler policy. The design is in docs/phase1_scheduler.md and the tests are in arena/tests_phase1/.

    .venv/bin/python -m pytest arena/tests_phase1 -q

Hivemind stays frozen and picks the moves. This policy decides around it:

    quick look -> gate (move, sit, or wait for a piece) -> talk -> think budget -> long search -> play

arena/players/scheduler_player.py plays it in the arena and arena/train_scheduler.py trains it.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from arena.interfaces import BOARD_OF, PARTNER, Observation

PIECES = ("P", "N", "B", "R", "Q")
GATE_OPTIONS = ("move",) + tuple(f"wait_{p}" for p in PIECES) + ("sit",)  # 7 options; sit waits silently
TALK_OPTIONS = ("quiet",) + tuple(f"need_{p}" for p in PIECES) + ("offer",)  # 7 options
THINK_MULTIPLIERS = (0.25, 0.5, 1.0, 2.0, 4.0)  # 5 options, times the player's base node budget
END_WAIT_OPTIONS = ("keep_waiting", "stop")  # 2 options
SERVE_OPTIONS = ("own_move", "serve")  # play the engine's move, or the capture the partner asked for
BASE_THINK_INDEX = THINK_MULTIPLIERS.index(1.0)

START_EXPLORATION = 0.05
MESSAGE_TTL_MS = 15_000  # an offer or request this old no longer says anything about the position

MOVE_INDEX = GATE_OPTIONS.index("move")
SIT_INDEX = GATE_OPTIONS.index("sit")
QUIET_INDEX = TALK_OPTIONS.index("quiet")
STOP_INDEX = END_WAIT_OPTIONS.index("stop")

# Added to the option stock would pick. e^6 makes that option about 98.5% likely even against six others, so an
# untrained seat rarely waits by accident. The tests only need about 4.8, which would mean 1 random wait in 20.
STOCK_LOGIT = 6.0
HIDDEN_SIZE = 64


@dataclass(frozen=True)
class Features:
    """Everything the policy sees at one decision, from this seat's point of view."""
    # from the quick look
    eval: float  # our team's side, [-1, 1]
    eval_change: float  # this eval minus the eval at our last decision
    search_passed: bool  # the quick look wants to pass on our board, which is how stock decides to wait
    wants: tuple[str, ...]  # pieces the plan wants to drop but we do not hold (_pv_needs)
    can_offer: bool  # the plan captures something the partner could use (pv_offers)
    # clocks, in ms
    own_clock_ms: int
    partner_clock_ms: int
    own_opponent_clock_ms: int
    partner_opponent_clock_ms: int
    # from the partner, tracked by CommState
    partner_waiting: bool
    offered_piece: str | None  # the piece the partner last offered, if any
    offered_plies: int  # how many plies away the partner said it was, 0 if no offer
    requested_piece: str | None  # the piece the partner last asked for, if any
    # our own state
    holding: tuple[str, ...]  # pieces in our pocket
    sits_in_row: int
    nodes_used_fraction: float  # our nodes so far this game / our expected nodes per game
    # Optional inputs, filled only when the player's feature group is on (arena/players/scheduler_player.py).
    # "tempo": how forced the move is
    n_legal: int = 0
    in_check: bool = False
    top_gap: float = 0.0  # quick-look eval of the engine's choice minus its best other line; 1.0 when only one
    # "mate": what one more piece in our pocket would be worth, and whether the partner can get it
    gain: tuple[float, ...] = ()  # per PIECES: eval with the piece minus eval without, 0 if not probed
    mate_with: tuple[str, ...] = ()  # pieces that give a forced mate at once
    supply: tuple[float, ...] = ()  # per PIECES: 1 partner can capture one now, 0.5 after its opponent moves
    # "danger": what one more piece in our opponent's pocket would cost us
    danger: tuple[float, ...] = ()  # per PIECES: eval without minus eval with, 0 if not probed
    hold_piece: str | None = None  # the piece the partner last asked us not to lose, if any
    # "serve": only set on a serve decision
    serve_cost: float = 0.0  # eval given up by playing the capture the partner asked for


class CommState:
    """Remembers what the partner has told this seat. Update it with each Observation's inbox.

    Tokens: "stall" means the partner is waiting. "have_X" with payload {"piece": "X", "plies": n} is an offer.
    "need_X" is a request. "hold_X" asks us not to lose an X. The partner stops waiting once obs.to_move shows it is not its turn. Offers and
    requests expire after MESSAGE_TTL_MS."""

    def __init__(self, seat: str) -> None:
        self.partner = PARTNER[seat]
        self.partner_waiting = False
        self.offered_piece = None
        self.offered_plies = 0
        self.requested_piece = None
        self.hold_piece = None
        self._offer_ms = 0
        self._request_ms = 0
        self._hold_ms = 0

    def update(self, obs: Observation) -> None:
        for m in obs.inbox:
            if m.token == "stall":
                self.partner_waiting = True
            elif m.token.startswith("have_"):
                self.offered_piece = m.payload.get("piece", m.token[5:])
                self.offered_plies = int(m.payload.get("plies", 0))
                self._offer_ms = obs.now_ms
            elif m.token.startswith("need_"):
                self.requested_piece = m.payload.get("piece", m.token[5:])
                self._request_ms = obs.now_ms
            elif m.token.startswith("hold_"):
                self.hold_piece = m.payload.get("piece", m.token[5:])
                self._hold_ms = obs.now_ms
        if obs.to_move[BOARD_OF[self.partner]] != self.partner:
            self.partner_waiting = False
        if self.offered_piece is not None and obs.now_ms - self._offer_ms > MESSAGE_TTL_MS:
            self.clear_offer()
        if self.requested_piece is not None and obs.now_ms - self._request_ms > MESSAGE_TTL_MS:
            self.requested_piece = None
        if self.hold_piece is not None and obs.now_ms - self._hold_ms > MESSAGE_TTL_MS:
            self.hold_piece = None

    def clear_offer(self) -> None:
        self.offered_piece, self.offered_plies = None, 0

    partner_waiting: bool
    offered_piece: str | None
    offered_plies: int
    requested_piece: str | None
    hold_piece: str | None


# to_tensor layout. The stock starting point reads SEARCH_PASSED and WANTS back out of the tensor.
SEARCH_PASSED = 2
WANTS = slice(4, 4 + len(PIECES))
FEATURE_SIZE: int = 61  # 32 base inputs, then 29 optional ones that stay 0 while their feature group is off

CLOCK_SCALE_MS = 60_000


def _one_hot(piece: str | None) -> list[float]:
    return [float(p == piece) for p in PIECES]


def _clock(ms: int) -> float:
    return min(ms / CLOCK_SCALE_MS, 3.0)


def _per_piece(values: tuple[float, ...], top: float) -> list[float]:
    return [max(0.0, min(top, v)) for v in values] if len(values) == len(PIECES) else [0.0] * len(PIECES)


def to_tensor(f: Features) -> torch.Tensor:
    """Features -> a float tensor of shape (FEATURE_SIZE,), with values roughly in [-1, 1] and never past 3."""
    values = [
        f.eval, max(-2.0, min(2.0, f.eval_change)), float(f.search_passed), float(f.can_offer),
        *(float(p in f.wants) for p in PIECES),
        *(min(f.holding.count(p) / 4, 1.0) for p in PIECES),
        _clock(f.own_clock_ms), _clock(f.partner_clock_ms), _clock(f.own_opponent_clock_ms),
        _clock(f.partner_opponent_clock_ms),
        float(f.partner_waiting), *_one_hot(f.offered_piece), min(f.offered_plies / 10, 1.0),
        *_one_hot(f.requested_piece),
        min(f.sits_in_row / 4, 1.0), min(f.nodes_used_fraction, 2.0),
        min(f.n_legal / 40, 1.0), float(f.in_check), max(-1.0, min(1.0, f.top_gap)),
        *_per_piece(f.gain, 2.0), *(float(p in f.mate_with) for p in PIECES), *_per_piece(f.supply, 1.0),
        *_per_piece(f.danger, 2.0), *_one_hot(f.hold_piece), min(f.serve_cost, 1.0),
    ]
    return torch.tensor(values, dtype=torch.float32)


def gate_mask(f: Features) -> torch.Tensor:
    """Bool tensor over GATE_OPTIONS: which are allowed. Waiting for a piece we already hold makes no sense."""
    return torch.tensor([not (g.startswith("wait_") and g[5:] in f.holding) for g in GATE_OPTIONS])


def talk_mask(f: Features) -> torch.Tensor:
    """Bool tensor over TALK_OPTIONS. "offer" is only allowed when f.can_offer."""
    return torch.tensor([t != "offer" or f.can_offer for t in TALK_OPTIONS])


@dataclass
class Heads:
    """Unnormalized logits, one tensor per decision."""
    gate: torch.Tensor  # (len(GATE_OPTIONS),)
    talk: torch.Tensor  # (len(TALK_OPTIONS),)
    think: torch.Tensor  # (len(THINK_MULTIPLIERS),)
    end_wait: torch.Tensor  # (len(END_WAIT_OPTIONS),)
    serve: torch.Tensor | None = None  # (len(SERVE_OPTIONS),)


def _stock_gate(x: torch.Tensor) -> torch.Tensor:
    """What stock would do, as gate logits: move; or if the search passed, wait for the piece the plan wants,
    or sit silently when it wants none."""
    logits = torch.zeros(len(GATE_OPTIONS))
    if x[SEARCH_PASSED] < 0.5:
        logits[MOVE_INDEX] = STOCK_LOGIT
        return logits
    wanted = [i for i, p in enumerate(PIECES) if x[WANTS][i] > 0.5]
    for i in wanted:
        logits[GATE_OPTIONS.index(f"wait_{PIECES[i]}")] = STOCK_LOGIT
    if not wanted:
        logits[SIT_INDEX] = STOCK_LOGIT
    return logits


def _favor(size: int, index: int) -> torch.Tensor:
    logits = torch.zeros(size)
    logits[index] = STOCK_LOGIT
    return logits


class SchedulerPolicy(nn.Module):
    """One shared body and one small output layer per head. Each head's logits are a fixed stock starting point
    plus a learned correction, and the output layers start at zero, so an untrained policy behaves like stock.

    forward takes one feature tensor. The talk head also sees the gate choice, one-hot: gate_choice is the index
    into GATE_OPTIONS, or None (all zeros) when the gate has not been chosen yet.

    think_logit is how strongly an untrained policy sticks to the normal think budget. It is saved with the
    weights. The serve head has no starting preference: it is only asked when a capture the partner wants is
    close to the engine's own choice."""

    def __init__(self, think_logit: float = STOCK_LOGIT) -> None:
        super().__init__()
        self.register_buffer("think_logit", torch.tensor(float(think_logit)))
        self.body = nn.Sequential(nn.Linear(FEATURE_SIZE, HIDDEN_SIZE), nn.ReLU(),
                                  nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE), nn.ReLU())
        self.gate = nn.Linear(HIDDEN_SIZE, len(GATE_OPTIONS))
        self.talk = nn.Linear(HIDDEN_SIZE + len(GATE_OPTIONS), len(TALK_OPTIONS))
        self.think = nn.Linear(HIDDEN_SIZE, len(THINK_MULTIPLIERS))
        self.end_wait = nn.Linear(HIDDEN_SIZE, len(END_WAIT_OPTIONS))
        self.serve = nn.Linear(HIDDEN_SIZE, len(SERVE_OPTIONS))
        for head in (self.gate, self.talk, self.think, self.end_wait, self.serve):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self, x: torch.Tensor, gate_choice: int | None = None) -> Heads:
        summary = self.body(x)
        chosen = torch.zeros(len(GATE_OPTIONS))
        if gate_choice is not None:
            chosen[gate_choice] = 1.0
        think_start = torch.zeros(len(THINK_MULTIPLIERS))
        think_start[BASE_THINK_INDEX] = self.think_logit
        return Heads(
            gate=_stock_gate(x) + self.gate(summary),
            talk=_favor(len(TALK_OPTIONS), QUIET_INDEX) + self.talk(torch.cat([summary, chosen])),
            think=think_start + self.think(summary),
            end_wait=_favor(len(END_WAIT_OPTIONS), STOP_INDEX) + self.end_wait(summary),
            serve=self.serve(summary),
        )


def plausible_wait(f: Features) -> bool:
    """A forced wait is only worth trying when the partner offered a piece or the plan wants one."""
    return f.offered_piece is not None or bool(f.wants)


def exploration_prob(train_step: int, total_steps: int, plausible: bool, start: float = START_EXPLORATION) -> float:
    """Chance of forcing a wait at this decision. 0 when not plausible. start at step 0, falling in a straight
    line to 0 at total_steps."""
    if not plausible or total_steps <= 0:
        return 0.0
    return start * max(0.0, 1.0 - train_step / total_steps)


def mixture_probs(policy_probs: torch.Tensor, epsilon: float, explore_mask: torch.Tensor) -> torch.Tensor:
    """The probabilities actions are really sampled from: with chance epsilon pick uniformly among
    explore_mask's options, otherwise follow policy_probs. REINFORCE must use these, not policy_probs."""
    count = explore_mask.sum()
    if count == 0:
        return policy_probs
    return (1.0 - epsilon) * policy_probs + epsilon * explore_mask.float() / count


@dataclass
class Action:
    gate: int  # index into GATE_OPTIONS
    talk: int  # index into TALK_OPTIONS
    think: int  # index into THINK_MULTIPLIERS
    log_prob: torch.Tensor  # log probability of all three together, with gradients attached


def _gate_probs(heads: Heads, f: Features, epsilon: float) -> torch.Tensor:
    mask = gate_mask(f)
    probs = torch.softmax(heads.gate.masked_fill(~mask, float("-inf")), -1)
    explore = mask.clone()
    explore[MOVE_INDEX] = False
    return mixture_probs(probs, epsilon, explore)


def _talk_log_probs(policy: SchedulerPolicy, x: torch.Tensor, f: Features, gate: int) -> torch.Tensor:
    logits = policy(x, gate_choice=gate).talk.masked_fill(~talk_mask(f), float("-inf"))
    return torch.log_softmax(logits, -1)


def _draw(probs: torch.Tensor, rng: torch.Generator) -> int:
    return int(torch.multinomial(probs.detach(), 1, generator=rng))


def action_log_prob(policy: SchedulerPolicy, f: Features, gate: int, talk: int, think: int,
                    epsilon: float) -> torch.Tensor:
    """Log probability of an action as it was sampled. Training replays recorded decisions through this."""
    x = to_tensor(f)
    heads = policy(x)
    return (torch.log(_gate_probs(heads, f, epsilon)[gate])
            + _talk_log_probs(policy, x, f, gate)[talk]
            + torch.log_softmax(heads.think, -1)[think])


def sample_action(policy: SchedulerPolicy, f: Features, epsilon: float, rng: torch.Generator) -> Action:
    """Sample gate, then talk given the gate, then think. Think reads only the body's summary, because the
    budget is only used when the gate says move. All three are always sampled, even when think won't be used."""
    x = to_tensor(f)
    heads = policy(x)
    gate = _draw(_gate_probs(heads, f, epsilon), rng)
    talk = _draw(_talk_log_probs(policy, x, f, gate).exp(), rng)
    think = _draw(torch.softmax(heads.think, -1), rng)
    return Action(gate, talk, think, action_log_prob(policy, f, gate, talk, think, epsilon))


def end_wait_log_prob(policy: SchedulerPolicy, f: Features, choice: int) -> torch.Tensor:
    return torch.log_softmax(policy(to_tensor(f)).end_wait, -1)[choice]


def sample_end_wait(policy: SchedulerPolicy, f: Features, rng: torch.Generator) -> tuple[int, torch.Tensor]:
    """While waiting: index into END_WAIT_OPTIONS and its log probability. This is a decision of its own, so its
    log probability joins the game's list alongside the Action ones."""
    choice = _draw(torch.softmax(policy(to_tensor(f)).end_wait, -1), rng)
    return choice, end_wait_log_prob(policy, f, choice)


def serve_log_prob(policy: SchedulerPolicy, f: Features, choice: int) -> torch.Tensor:
    return torch.log_softmax(policy(to_tensor(f)).serve, -1)[choice]


def sample_serve(policy: SchedulerPolicy, f: Features, rng: torch.Generator) -> tuple[int, torch.Tensor]:
    """With a capture the partner asked for on offer: index into SERVE_OPTIONS and its log probability. A decision
    of its own, like end_wait."""
    choice = _draw(torch.softmax(policy(to_tensor(f)).serve, -1), rng)
    return choice, serve_log_prob(policy, f, choice)


def plan_nodes(base_nodes: int, think_index: int, quick_nodes: int) -> int:
    """Nodes left for the long search. The quick look comes out of the same budget, so a decision never
    costs more than base_nodes * multiplier in total. At least 1."""
    return max(1, int(base_nodes * THINK_MULTIPLIERS[think_index]) - quick_nodes)


def team_reward(result: str, team: str) -> float:
    """+1 if team won, -1 if it lost, 0 for a draw. result is GameResult.result: "T1", "T2" or "draw"."""
    if result == "draw":
        return 0.0
    return 1.0 if result == team else -1.0


def reinforce_loss(log_probs: list[torch.Tensor], reward: float, baseline: float = 0.0) -> torch.Tensor:
    """The loss to minimize for one player's decisions in one game. Minimizing it makes the taken actions more
    likely after a reward above baseline and less likely after one below it."""
    return -(reward - baseline) * torch.stack(log_probs).sum()
