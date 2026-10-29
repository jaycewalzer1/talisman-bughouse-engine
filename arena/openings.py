"""Opening pairs for paired matches.

Each opening pair gives a start FEN for board A and one for board B (standard chess openings, a few
plies deep, no captures, empty pockets, White to move). `paired_schedule` plays every pair twice with
the teams swapped, so neither config gets a lucky opening; stats.py clusters on the pair id.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterator

import chess

# (name, SAN moves). Even ply counts (White to move on both boards) and no captures, so the
# empty-pocket start is consistent with bughouse rules.
OPENINGS: tuple[tuple[str, str], ...] = (
    ("Ruy Lopez", "e4 e5 Nf3 Nc6 Bb5 a6"),
    ("Italian Game", "e4 e5 Nf3 Nc6 Bc4 Bc5"),
    ("Four Knights", "e4 e5 Nf3 Nc6 Nc3 Nf6"),
    ("Philidor Defence", "e4 e5 Nf3 d6 d4 Nf6"),
    ("Vienna Game", "e4 e5 Nc3 Nf6 Bc4 Nc6"),
    ("Sicilian Moscow", "e4 c5 Nf3 d6 Bb5+ Bd7"),
    ("Closed Sicilian", "e4 c5 Nc3 Nc6 g3 g6"),
    ("Sicilian Alapin", "e4 c5 c3 Nf6 e5 Nd5"),
    ("French Advance", "e4 e6 d4 d5 e5 c5"),
    ("French Winawer", "e4 e6 d4 d5 Nc3 Bb4"),
    ("Caro-Kann Advance", "e4 c6 d4 d5 e5 Bf5"),
    ("Pirc Defence", "e4 d6 d4 Nf6 Nc3 g6"),
    ("Alekhine Defence", "e4 Nf6 e5 Nd5 d4 d6"),
    ("Modern Defence", "e4 g6 d4 Bg7 Nc3 d6"),
    ("Queen's Gambit Declined", "d4 d5 c4 e6 Nc3 Nf6"),
    ("Slav Defence", "d4 d5 c4 c6 Nf3 Nf6"),
    ("King's Indian", "d4 Nf6 c4 g6 Nc3 Bg7"),
    ("Nimzo-Indian", "d4 Nf6 c4 e6 Nc3 Bb4"),
    ("Queen's Indian", "d4 Nf6 c4 e6 Nf3 b6"),
    ("Grunfeld", "d4 Nf6 c4 g6 Nc3 d5"),
    ("London System", "d4 d5 Bf4 Nf6 e3 e6"),
    ("Dutch Defence", "d4 f5 g3 Nf6 Bg2 e6"),
    ("English Opening", "c4 e5 Nc3 Nf6 g3 d5"),
    ("Reti Opening", "Nf3 d5 g3 Nf6 Bg2 e6"),
)


def opening_fen(moves: str) -> str:
    """Bracket-form FEN (empty pocket) after playing the SAN moves from the start position."""
    board = chess.Board()
    for san in moves.split():
        board.push_san(san)
    placement, rest = board.fen().split(" ", 1)
    return f"{placement}[] {rest}"


@dataclass(frozen=True)
class OpeningPair:
    id: int
    name: str  # "<A opening> / <B opening>"
    fen_a: str
    fen_b: str

    @property
    def fens(self) -> dict[str, str]:
        return {"A": self.fen_a, "B": self.fen_b}


def _build_pairs() -> tuple[OpeningPair, ...]:
    n = len(OPENINGS)
    pairs = []
    for i in range(n):
        j = (7 * i + 5) % n  # 7 is coprime with 24: a permutation with no fixed point
        (na, ma), (nb, mb) = OPENINGS[i], OPENINGS[j]
        pairs.append(OpeningPair(i, f"{na} / {nb}", opening_fen(ma), opening_fen(mb)))
    return tuple(pairs)


OPENING_PAIRS: tuple[OpeningPair, ...] = _build_pairs()


def paired_schedule(n_games: int, seed: int = 0) -> Iterator[tuple[OpeningPair, bool]]:
    """Yield (opening_pair, swap_sides) for n_games games.

    Games come in twos: game 2k plays pair p with the contender as T1 (swap_sides=False), game 2k+1
    plays the same pair with the contender as T2 (swap_sides=True). Pair order is a seeded shuffle;
    after all pairs are used the order is reshuffled. An odd n_games leaves the last pair unmatched.
    """
    rng = random.Random(seed)
    order: list[OpeningPair] = []
    for g in range(n_games):
        if g % 2 == 0:
            if not order:
                order = list(OPENING_PAIRS)
                rng.shuffle(order)
            pair = order.pop()
        yield pair, bool(g % 2)
