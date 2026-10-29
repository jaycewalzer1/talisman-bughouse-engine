"""Uniform random legal-move player. Fast and deterministic given a seed; used in tests and smoke runs."""

from __future__ import annotations

import random

from arena.interfaces import Decision, Observation


class RandomPlayer:
    kind = "random"

    def __init__(self, seat: str, seed: int | None = None, think_ms: int = 300) -> None:
        self.seat = seat
        self.name = f"Random-{seat}"
        self.think_ms = int(think_ms)
        self._rng = random.Random(f"random-{seed}-{seat}") if seed is not None else random.Random()

    def decide(self, obs: Observation) -> Decision:
        if not obs.legal_moves:
            return Decision(kind="wait", wait_ms=200, stall_reason="no_moves")
        return Decision(kind="move", uci=self._rng.choice(list(obs.legal_moves)), think_ms=self.think_ms)

    def close(self) -> None:
        pass
