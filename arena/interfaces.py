"""Shared contracts for the arena. Every arena module codes against these.

Event JSON format: docs/events.md. Module ownership:
- arena/state.py    BughouseState (two boards, pockets, four clocks, rules)
- arena/sim.py      Simulator (event-driven timeline, message channel, emits events)
- arena/players/    Player implementations and team setups (incl. Hivemind wrapper)
- arena/run.py, arena/serve.py, arena/stats.py   match runner, live WebSocket server, statistics
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol

Seat = Literal["A_white", "A_black", "B_white", "B_black"]
Board = Literal["A", "B"]
Team = Literal["T1", "T2"]

SEATS: tuple[Seat, ...] = ("A_white", "A_black", "B_white", "B_black")
BOARD_OF: dict[str, str] = {"A_white": "A", "A_black": "A", "B_white": "B", "B_black": "B"}
COLOR_OF: dict[str, str] = {"A_white": "white", "A_black": "black", "B_white": "white", "B_black": "black"}
TEAM_OF: dict[str, str] = {"A_white": "T1", "B_black": "T1", "A_black": "T2", "B_white": "T2"}
PARTNER: dict[str, str] = {"A_white": "B_black", "B_black": "A_white", "A_black": "B_white", "B_white": "A_black"}
OPPONENT: dict[str, str] = {"A_white": "A_black", "A_black": "A_white", "B_white": "B_black", "B_black": "B_white"}


def seat_of(board: str, color: str) -> str:
    return f"{board}_{color}"


@dataclass(frozen=True)
class Message:
    from_seat: str
    to_seat: str
    token: str  # see docs/events.md
    sent_ms: int
    text: str | None = None
    payload: dict = field(default_factory=dict)


@dataclass(frozen=True)
class MoveResult:
    """Outcome of BughouseState.apply_move."""
    san: str
    capture: str | None  # piece letter as it enters the partner's pocket (FEN case), else None
    fen: str  # bracket-form FEN of the moved board after the move


@dataclass(frozen=True)
class GameResult:
    result: str  # "T1" | "T2" | "draw"
    reason: str  # "checkmate" | "flag" | "stalemate" | "max_plies" | "aborted"
    board: str | None = None
    loser_seat: str | None = None


class GameState(Protocol):
    """Implemented by arena.state.BughouseState(base_ms, increment_ms=0)."""

    clocks_ms: dict[str, int]  # remaining time per seat as of the last advance_clocks call

    def fen(self, board: str) -> str: ...  # bracket-form FEN incl. pocket
    def to_move(self, board: str) -> str: ...  # seat whose turn it is on that board
    def legal_moves(self, seat: str) -> list[str]: ...  # UCI incl. drops "N@f3"; [] if not seat's turn
    def apply_move(self, seat: str, uci: str, now_ms: int) -> MoveResult: ...  # raises ValueError if illegal
    def advance_clocks(self, now_ms: int) -> GameResult | None: ...  # run clocks to now; flag result if any
    def result(self) -> GameResult | None: ...  # checkmate/stalemate/flag once the game is over
    def copy(self) -> "GameState": ...


@dataclass(frozen=True)
class Observation:
    """What a player sees when asked to decide. Bughouse is fully observable: both boards, all pockets,
    all four clocks. inbox holds messages delivered to this seat since its last decision."""
    now_ms: int
    seat: str
    fens: dict[str, str]  # {"A": fen, "B": fen}
    clocks_ms: dict[str, int]
    to_move: dict[str, str]  # {"A": seat, "B": seat}
    legal_moves: list[str]  # this seat's legal moves; [] when it is not this seat's turn
    inbox: tuple[Message, ...] = ()
    ply: int = 0
    # Split setups with lending: True when the partner's machine is idle (partner not searching and not
    # lending it already), so this seat may borrow it for one search (Decision.borrow).
    partner_machine_free: bool = False


@dataclass(frozen=True)
class Decision:
    """A player's response. The simulator charges think_ms on the player's clock and lands the action at
    now_ms + think_ms. If the position changed meanwhile and the move became illegal, the simulator asks
    again (the spent time is lost).

    kind="move": play uci.  kind="wait": sit for up to wait_ms (woken early by any event on either board
    or a delivered message)."""
    kind: Literal["move", "wait"]
    uci: str | None = None
    think_ms: int = 0
    wait_ms: int = 0
    nodes: int = 0
    eval: float | None = None  # thinker's perspective, [-1, 1]
    messages: tuple[Message, ...] = ()  # sent when the decision lands (sent_ms is overwritten by the sim)
    stall_reason: str | None = None  # for kind="wait": "need_piece" | "time_advantage" | ...
    # The search behind this decision also ran on the partner's machine (only when
    # Observation.partner_machine_free). The partner's machine stays busy for think_ms; a partner that
    # must decide meanwhile is deferred until it is returned.
    borrow: bool = False


class Player(Protocol):
    seat: str
    name: str
    kind: str  # e.g. "hivemind", "random", "heuristic"

    def decide(self, obs: Observation) -> Decision: ...
    def close(self) -> None: ...
