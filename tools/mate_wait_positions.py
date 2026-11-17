"""Hand-built "wait for the mating piece" positions, played out with the real engine.

    PYTHONPATH=. .venv/bin/python tools/mate_wait_positions.py --rollouts 3 --out runs/mate_wait_local

Board A (A_white, T1, to move): a drop of piece X mates at once, but A_white's pocket is empty and nothing
else mates. Board B (B_black, T1's partner, to move): a white X hangs on d5 to ...exd5, which sends X to
A_white. If A_white moves first, A_black gets a move to defend (...Nxf6, ...h6, ...Bxa3).

Branches, all with T1 = sharing and T2 = centralized (stock) at --nodes:
* ``stock``: equal clocks, the engine decides. RequireMoveOn does not stop it passing when it sees a forced
  mate across both boards (see arena/players/hivemind.py), so this is whatever the stock engine chooses.
* ``move``: A_white's first move is forced to be the quiet ``a2a3`` (``b2b3`` in the rook position); the engine plays everything after.
* ``wait``: equal clocks, A_white is forced to sit until a piece lands (at most --wait-ms).
* ``wait_short``: as ``wait``, but A_white has only --short-ms on its clock: less than the partner's think
  time, so waiting should lose on time. The control.

One game at a time, in-process (no worker pool). Writes <out>/rollouts.jsonl and prints a table.
"""

from __future__ import annotations

import argparse
import json
import os
import time

from arena.interfaces import SEATS, Decision, Observation
from arena.probe_wait import ForcedWait, own_pocket
from arena.run import build_players, setup_label
from arena.sim import Simulator
from arena.state import BughouseState

A_MINOR = "r5nk/pp5p/5P2/8/8/8/PP3PPP/6K1[] w - - 0 1"  # Q@g7, B@g7, P@g7 and N@f7 all mate
A_ROOK = "1n4k1/pp3ppp/8/8/8/B7/PP3PPP/6K1[] w - - 0 1"  # R@f8 mates (Ba3 guards f8)
A_PHILIDOR = "5rk1/ppp2ppp/2nb4/8/8/8/PPP5/1K4R1[Q] w - - 0 1"  # Rg1 pins g7; White holds a queen, needs a knight
B_HANGING = "6k1/5ppp/4p3/3{X}4/8/8/5PPP/6K1[] b - - 0 1"  # ...exd5 wins X for A_white

POSITIONS = {
    "Q": (A_MINOR, "Q@g7"),
    "R": (A_ROOK, "R@f8"),
    "B": (A_MINOR, "B@g7"),
    "N": (A_MINOR, "N@f7"),
    "P": (A_MINOR, "P@g7"),
    # Philidor-style, after Vasquez's sit for a knight (2024 Chess.com Bughouse Championship, qualifier 4):
    # with the knight, N@h6+ Kh8 Q@g8+ Rxg8 Nf7# is mate in 3; without it there is no forced mate in 2.
    "N3": (A_PHILIDOR, "N@h6"),
}
HANGING = {"N3": "N"}  # the piece on d5 when the key isn't a piece letter
BRANCHES = ("stock", "move", "wait", "wait_short")
QUIET = {"R": "b2b3"}  # quiet first move for the move branch; a2a3 elsewhere (a3 is taken by the bishop here)
SEAT = "A_white"


class ForcedFirstMove:
    """Wraps a player: its first decision is the given move, every later one is the wrapped player's."""

    def __init__(self, inner, uci: str):
        self.inner, self.uci, self.done = inner, uci, False
        self.seat, self.name, self.kind = inner.seat, inner.name + "-forced", inner.kind

    def decide(self, obs: Observation) -> Decision:
        if not self.done:
            if self.uci not in obs.legal_moves:
                raise ValueError(f"forced move {self.uci} is not legal here")
            self.done = True
            return Decision(kind="move", uci=self.uci, think_ms=200)
        return self.inner.decide(obs)

    def close(self) -> None:
        self.inner.close()


def clocks_for(branch: str, base_ms: int, short_ms: int) -> dict[str, int]:
    clocks = {s: base_ms for s in SEATS}
    if branch == "wait_short":
        clocks[SEAT] = short_ms
    return clocks


def play(piece: str, branch: str, k: int, args) -> dict:
    a_fen, mate_drop = POSITIONS[piece]
    state = BughouseState(args.base_ms, fens={"A": a_fen, "B": B_HANGING.format(X=HANGING.get(piece, piece))})
    state.clocks_ms = clocks_for(branch, args.base_ms, args.short_ms)
    players, created = build_players("sharing", "centralized", "hivemind", seed=1000 * k + 7, nodes=args.nodes)
    if branch == "move":
        players[SEAT] = ForcedFirstMove(players[SEAT], QUIET.get(piece, "a2a3"))
    if branch.startswith("wait"):
        players[SEAT] = ForcedWait(players[SEAT], args.wait_ms, own_pocket(a_fen, SEAT))
    t0 = time.perf_counter()
    try:
        sim = Simulator(state, players, game_id=f"matewait-{piece}-{branch}-{k}", setup=setup_label("sharing", "centralized"),
                        max_plies=args.max_plies)
        res = sim.run()
    finally:
        for p in created:
            try:
                p.close()
            except Exception:  # noqa: BLE001 - best effort
                pass
    moves = [e for e in sim.events if e["type"] == "move"]
    first_a = next((e for e in moves if e["seat"] == SEAT), None)
    sat = any(e["type"] == "stall" and e["seat"] == SEAT and e["active"]
              and e.get("reason") != "probe_wait" for e in sim.events if first_a is None or e["seq"] < first_a["seq"])
    score = 0.5 if res.result == "draw" else float(res.result == "T1")
    return {"piece": piece, "branch": branch, "rollout": k, "score": score, "result": res.result, "reason": res.reason,
            "board": res.board, "plies": len(moves), "end_ms": sim.now,
            "a_first": first_a["uci"] if first_a else None, "a_first_ms": first_a["ts"] if first_a else None,
            "mate_drop_first": bool(first_a and first_a["uci"] == mate_drop),
            "engine_sat": sat, "wall_s": round(time.perf_counter() - t0, 1)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pieces", nargs="+", default=list(POSITIONS), choices=list(POSITIONS))
    ap.add_argument("--branches", nargs="+", default=list(BRANCHES), choices=list(BRANCHES))
    ap.add_argument("--rollouts", type=int, default=3)
    ap.add_argument("--nodes", type=int, default=200)
    ap.add_argument("--base-ms", type=int, default=60_000)
    ap.add_argument("--wait-ms", type=int, default=5_000)
    ap.add_argument("--short-ms", type=int, default=500)
    ap.add_argument("--max-plies", type=int, default=80)
    ap.add_argument("--out", default="runs/mate_wait_local")
    args = ap.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "rollouts.jsonl")
    rows = []
    with open(path, "a") as f:
        for piece in args.pieces:
            for branch in args.branches:
                for k in range(args.rollouts):
                    row = play(piece, branch, k, args)
                    rows.append(row)
                    f.write(json.dumps(row) + "\n")
                    f.flush()
                    print(f"{piece} {branch:<10} #{k}: {row['result']:<4} {row['reason']:<9} board {row['board']} "
                          f"plies {row['plies']:>3}  A_white first {row['a_first']} at {row['a_first_ms']} ms  "
                          f"({row['wall_s']} s)", flush=True)

    print(f"\nT1 score by piece and branch ({args.rollouts} rollouts each, {args.nodes} nodes)")
    print("piece  " + "  ".join(f"{b:>10}" for b in args.branches))
    for piece in args.pieces:
        cells = []
        for b in args.branches:
            r = [x["score"] for x in rows if x["piece"] == piece and x["branch"] == b]
            cells.append(f"{sum(r) / len(r):>10.2f}" if r else f"{'-':>10}")
        print(f"{piece:<6} " + "  ".join(cells))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
