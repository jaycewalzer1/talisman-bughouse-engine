"""The three experimental team setups, for both teams.

``make_players(setup, kind="heuristic", *, seed=0, nodes=200, opponent_setup=None, **kw)`` returns a
``{seat: Player}`` dict for all four seats. T1 (A_white + B_black) uses ``setup``; T2 uses
``opponent_setup`` (default: the same setup).

Setups
------
* ``"centralized"``: one controller per team decides for both teammates.
  - hivemind: ONE engine process per team (``HivemindTeamController``) searching joint actions over
    both boards with the team's whole budget of ``nodes`` per search. When both seats are asked at the
    same position, one search serves both (the second seat's move lands when that search finishes).
  - heuristic: ``HeuristicTeam`` sees both seats' situations (no messages needed).
* ``"silent"``: two independent players per team; nothing sent, inbox ignored.
  - hivemind: one engine process per seat, ``nodes // 2`` nodes per search.
  - heuristic: own-board-only greedy players (``comm="none"``).
* ``"sharing"``: two independent players that send their search result (best move, eval, needed pieces)
  to the partner on every decision and use partner messages (requests, sitting, go/sit) when deciding.
  - hivemind: as silent (``nodes // 2`` per seat) with ``comm="share"``.
  - heuristic: ``comm="share"`` (request protocol + status message every decision).
  Zero message delay is the simulator's ``channel_delay_ms=0`` (set by the runner, not here).

Equal team compute
------------------
Compute is matched per *team decision*: a team has a budget of ``nodes`` network-evaluated MCTS nodes
for each decision it makes. Centralized spends all ``nodes`` on one joint search; each independent
player spends ``nodes // 2`` on its own search. Simulated think time is the fixed cost model
``think_ms = nodes_requested * ms_per_node`` (``hivemind.MS_PER_NODE``, measured), so a centralized
search costs twice the clock time of an independent one, and with the default ``compute="nodes"`` the
two independent players together have the same node budget as the controller.
Caveat: two independent players can search *simultaneously* (both clocks run), so per unit of
simulated time the team may use up to twice the controller's throughput. ``compute="throughput"``
instead models each independent player as half a machine (``ms_per_node`` doubled), which makes the
team's nodes-per-second equal across setups at the price of slower independent moves.
For the heuristic kind there are no nodes; think time follows the same complexity/clock formula in all
setups, so the setups differ only in information and coordination.

``kind="random"`` ignores the setup (all seats random; useful for tests).
"""

from __future__ import annotations

from arena.interfaces import SEATS, TEAM_OF
from arena.players.heuristic import HeuristicPlayer, HeuristicTeam, team_seats
from arena.players.hivemind import (
    DEFAULT_BINARY,
    DEFAULT_MODEL,
    MS_PER_NODE,
    HivemindPlayer,
    HivemindProcess,
    HivemindTeamController,
)
from arena.players.random_player import RandomPlayer

SETUPS = ("centralized", "silent", "sharing")
KINDS = ("random", "heuristic", "hivemind")


def _team_players(team: str, setup: str, kind: str, *, seed: int, nodes: int, **kw) -> dict:
    seats = team_seats(team)
    if kind == "random":
        return {s: RandomPlayer(s, seed=seed, think_ms=kw.get("think_ms", 300)) for s in seats}

    if kind == "heuristic":
        hkw = {k: v for k, v in kw.items() if k in ("min_think_ms", "max_think_ms", "stall_max_ms",
                                                    "request_window_ms", "min_stall_clock_ms")}
        if setup == "centralized":
            brain = HeuristicTeam(team, seed=seed, **hkw)
            return {s: brain.player(s) for s in seats}
        comm = "none" if setup == "silent" else "share"
        return {s: HeuristicPlayer(s, seed=seed, comm=comm, **hkw) for s in seats}

    if kind == "hivemind":
        binary = kw.get("binary", DEFAULT_BINARY)
        model = kw.get("model", DEFAULT_MODEL)
        ms = float(kw.get("ms_per_node", MS_PER_NODE))
        pkw = dict(options=kw.get("engine_options"), reset_tree=kw.get("reset_tree", True))
        skw = {k: kw[k] for k in ("max_consecutive_sits", "sit_ms") if k in kw}
        if setup == "centralized":
            ctl = HivemindTeamController(team, HivemindProcess(binary, model, **pkw), nodes=nodes,
                                         ms_per_node=ms, **skw)
            return {s: ctl.player(s) for s in seats}
        half = max(1, nodes // 2)
        if kw.get("compute", "nodes") == "throughput":
            ms *= 2
        comm = "none" if setup == "silent" else "share"
        return {s: HivemindPlayer(s, HivemindProcess(binary, model, **pkw), nodes=half, ms_per_node=ms,
                                  comm=comm, **skw) for s in seats}

    raise ValueError(f"unknown kind {kind!r}; expected one of {KINDS}")


def make_players(setup: str, kind: str = "heuristic", *, seed: int = 0, nodes: int = 200,
                 opponent_setup: str | None = None, **kw) -> dict:
    """{seat: Player} for all four seats; T1 uses ``setup``, T2 ``opponent_setup or setup``.
    Hivemind engine processes start lazily on the first search; ``Player.close()`` releases them."""
    t2_setup = opponent_setup or setup
    for s in (setup, t2_setup):
        if s not in SETUPS:
            raise ValueError(f"unknown setup {s!r}; expected one of {SETUPS}")
    players: dict = {}
    players.update(_team_players("T1", setup, kind, seed=seed, nodes=nodes, **kw))
    players.update(_team_players("T2", t2_setup, kind, seed=seed + 1, nodes=nodes, **kw))
    assert set(players) == set(SEATS) and all(TEAM_OF[s] in ("T1", "T2") for s in players)
    return players
