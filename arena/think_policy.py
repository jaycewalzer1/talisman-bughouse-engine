"""Learned time policy, arena side (spec v7 sections 4.2 and 6.5).

A wave2 network with a think_gain head predicts, per board, the chance p that
searching a decision 4x longer changes the move. The budget scale is
s = clamp(0.5 + 1.5 p, 0.5, 2.0) (spec v2 B): a fresh head (p = 1/3) gives
s = 1, so turning the policy on with an untrained head changes nothing but the
quick look's cost.

The wave2 controller (arena/players/upstream.py, keyword ``think_head``) runs a
QUICK_LOOK_NODES search to read p and then searches ``budget(...)`` nodes, so
the quick look is paid out of the scaled budget.
"""

from __future__ import annotations

QUICK_LOOK_NODES = 16
MIN_SCALE = 0.5
MAX_SCALE = 2.0


def scale(p: float) -> float:
    """Budget scale for a think_gain probability: clamp(0.5 + 1.5 p, 0.5, 2.0)."""
    return min(MAX_SCALE, max(MIN_SCALE, 0.5 + 1.5 * float(p)))


def budget(clock_budget_nodes: int, p: float | None, min_nodes: int, max_nodes: int) -> int:
    """Main-search nodes after a quick look that read ``p``.

    ``p`` None (the network has no head, or the line was missing): the clock
    budget unchanged. Otherwise round(clock_budget_nodes x scale(p)) minus the
    quick look, clamped to [min_nodes, max_nodes].
    """
    if p is None:
        return clock_budget_nodes
    nodes = round(clock_budget_nodes * scale(p)) - QUICK_LOOK_NODES
    return int(min(max_nodes, max(min_nodes, nodes)))
