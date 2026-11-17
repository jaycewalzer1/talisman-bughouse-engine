"""Bughouse opening sets (audit WP-G; theory 05 section 2.6 and item 11-12).

The chess lines in ``arena/openings.py`` are mostly bughouse-unsound (early ...c5, c4, fianchettos, ...f5, ...b6:
theory 05 section 2.6), so gates and self-play on them start from positions a strong bughouse team never reaches.
This module adds:

* ``CANON``: the named bughouse lines from theory 05 section 2.6, each ending with White to move.
* ``CONTROLS``: three chess openings kept as controls (gate on both sets and report both: a gain that shows on the
  bughouse set and not on the controls is a bughouse gain, not an artefact).
* A bare start (the initial position on both boards) for a share of games, so the net can choose its own pairing.

Lines with captures are made bughouse-consistent: a piece captured on one board goes into the pocket of the
capturer's partner on the other board (white captures on A -> Black's pocket on B, and so on), so the pair's start
position is one a real game could reach. A line is therefore rendered per pair, not per board.

Sets (all registered in ``arena.openings.OPENING_SETS``; pair ids start at 3000):
* ``bughouse``: every ordered pair of two different canon lines + every ordered pair of two different controls +
  the bare start, repeated so it is ``BARE_SHARE`` (25%) of a full cycle of ``paired_schedule``. Distinct pair ids
  (the clusters for the bootstrap): ``len(CANON) * (len(CANON) - 1) + 6 + 1``.
* ``bughouse_canon``: the canon pairs only. ``chess_controls``: the 6 control pairs. ``bare``: the bare start once.

``--openings all`` (24 chess lines) stays available as the immediate stopgap the audit names.
Scoring (audit: "scored with it04 at high nodes and kept at small |Q|"): ``python -m arena.openings_bughouse
--score`` searches every canon/control pair with stock it04 and writes ``arena/openings_bughouse_scores.json``;
``KEEP_ABS_BALANCE`` and ``_kept`` drop pairs whose it04 balance (T1's edge with the no-advantage bias cancelled,
see ``score_pairs``) is large once that file exists.
"""

from __future__ import annotations

import argparse
import json
import os

import chess
import chess.variant

from arena.openings import OpeningPair

# (name, SAN moves, source note). Every line has an even ply count (White to move on its board).
CANON: tuple[tuple[str, str, str], ...] = (
    ("Fortress", "e4 e6 d4 d6", "theory 05 [2][15]: by far the most common bughouse opening"),
    ("Federkevic", "e4 e6 d4 b6", "theory 05 [2]"),
    ("Barbeau", "e4 e6 d4 Qh4", "theory 05 [2]"),
    ("e6-Nf6 ...d5", "e4 e6 d4 Nf6 Nc3 d5", "theory 05 MiniGreat/Sorsi; 3.Nc3 is our filler move"),
    ("e6-Nf6 ...Be7", "e4 e6 d4 Nf6 Nc3 Be7", "theory 05 MiniGreat/Sorsi; 3.Nc3 is our filler move"),
    ("e6-Nc6", "e4 e6 d4 Nc6", "theory 05 Sorsi"),
    ("Qe7 anti-f7", "e4 e6 d4 Qe7", "theory 05 [13]"),
    ("Scandinavian retreat", "e4 d5 exd5 Qxd5 Nc3 Qd8", "theory 05 [13]"),
    ("Leaf Gambit", "e4 d5 exd5 e6", "theory 05 Sorsi"),
    ("Exchange French", "e4 e6 d4 d5 exd5 exd5", "theory 05 Sorsi; 3...exd5 is our completion"),
    ("French no ...c5", "e4 e6 d4 d5 e5 Nc6", "theory 05 JarlCarlander; 3...Nc6 is our choice of non-c5 move"),
    ("Alekhine sacrifice", "e4 Nf6 e5 d5", "theory 05 [15]"),
    ("Two Knights", "e4 e5 Nf3 Nc6 Bc4 Nf6", "theory 05 '1.e4 e5 2.Nf3 Nc6 3.Bc4 lines'"),
    ("Hungarian", "e4 e5 Nf3 Nc6 Bc4 Be7", "theory 05 '1.e4 e5 2.Nf3 Nc6 3.Bc4 lines'"),
    ("d4 d5 Nf3", "d4 d5 Nf3 Nf6", "theory 05 '1.d4 d5 2.Nf3/Nc3 systems without c4'"),
    ("d4 d5 Nc3", "d4 d5 Nc3 Nf6", "theory 05 '1.d4 d5 2.Nf3/Nc3 systems without c4'"),
)

CONTROLS: tuple[tuple[str, str, str], ...] = (
    ("Sicilian Alapin", "e4 c5 c3 Nf6 e5 Nd5", "chess control (early ...c5)"),
    ("Queen's Gambit Declined", "d4 d5 c4 e6 Nc3 Nf6", "chess control (early c4)"),
    ("King's Indian", "d4 Nf6 c4 g6 Nc3 Bg7", "chess control (fianchetto)"),
)

BARE = ("Bare start", "", "initial position (no forced opening)")
BARE_SHARE = 0.25
PAIR_ID0 = 3000
BARE_ID = 3999
CONTROL_ID0 = 3900
SCORES_PATH = os.path.join(os.path.dirname(__file__), "openings_bughouse_scores.json")
KEEP_ABS_BALANCE = 0.35  # it04 |balance| above this at the scoring budget: dropped from ``bughouse`` (with scores)

_PIECE_LETTER = {chess.PAWN: "p", chess.KNIGHT: "n", chess.BISHOP: "b", chess.ROOK: "r", chess.QUEEN: "q"}


def play_line(moves: str) -> tuple[chess.variant.CrazyhouseBoard, dict[bool, list[str]]]:
    """Board after the SAN moves (pockets as crazyhouse would fill them, cleared below) and, per colour, the
    letters of the pieces that colour captured (promoted pieces count as pawns)."""
    b = chess.variant.CrazyhouseBoard()
    caps: dict[bool, list[str]] = {chess.WHITE: [], chess.BLACK: []}
    for san in moves.split():
        mv = b.parse_san(san)
        if b.is_capture(mv):
            sq = mv.to_square
            if b.is_en_passant(mv):
                caps[b.turn].append("p")
            else:
                pt = b.piece_type_at(sq)
                caps[b.turn].append("p" if b.promoted & chess.BB_SQUARES[sq] else _PIECE_LETTER[pt])
        b.push(mv)
    if b.turn != chess.WHITE:
        raise ValueError(f"line must end with White to move: {moves!r}")
    return b, caps


def pair_fens(moves_a: str, moves_b: str) -> dict[str, str]:
    """Bracket FENs for a pair: each board's line, pockets filled from the OTHER board's captures (bughouse)."""
    (ba, ca), (bb, cb) = play_line(moves_a), play_line(moves_b)
    out = {}
    for board, other_caps in (("A", cb), ("B", ca)):
        b = ba if board == "A" else bb
        b = b.copy()
        b.pockets[chess.WHITE].reset()
        b.pockets[chess.BLACK].reset()
        # On the other board, White's captures feed the partner (Black here); Black's feed White here.
        for letter in other_caps[chess.WHITE]:
            b.pockets[chess.BLACK].add(chess.PIECE_SYMBOLS.index(letter))
        for letter in other_caps[chess.BLACK]:
            b.pockets[chess.WHITE].add(chess.PIECE_SYMBOLS.index(letter))
        out[board] = b.fen()
    return out


def _pair(pid: int, la: tuple, lb: tuple) -> OpeningPair:
    f = pair_fens(la[1], lb[1])
    return OpeningPair(pid, f"{la[0]} / {lb[0]}", f["A"], f["B"])


def _ordered(lines: tuple, id0: int) -> tuple[OpeningPair, ...]:
    out = []
    for la in lines:
        for lb in lines:
            if la[0] != lb[0]:
                out.append(_pair(id0 + len(out), la, lb))
    return tuple(out)


def load_scores(path: str = SCORES_PATH) -> dict[str, dict] | None:
    try:
        with open(path) as f:
            return json.load(f)["pairs"]
    except (OSError, ValueError, KeyError):
        return None


def _kept(pairs: tuple[OpeningPair, ...], scores: dict | None) -> tuple[OpeningPair, ...]:
    if not scores:
        return pairs
    return tuple(p for p in pairs if abs(scores.get(p.name, {}).get("balance", 0.0)) <= KEEP_ABS_BALANCE)


CANON_PAIRS: tuple[OpeningPair, ...] = _ordered(CANON, PAIR_ID0)
CONTROL_PAIRS: tuple[OpeningPair, ...] = _ordered(CONTROLS, CONTROL_ID0)
BARE_PAIR = _pair(BARE_ID, BARE, BARE)


def bughouse_set(scores: dict | None = None, bare_share: float = BARE_SHARE) -> tuple[OpeningPair, ...]:
    """Canon + controls (kept by |Q| when scores exist) + the bare start repeated to ``bare_share`` of a cycle.
    Repeats share one pair id, so the bootstrap still counts the bare start as one cluster."""
    base = _kept(CANON_PAIRS, scores) + _kept(CONTROL_PAIRS, scores)
    if not 0.0 <= bare_share < 1.0:
        raise ValueError(f"bare_share must be in [0, 1), got {bare_share}")
    copies = max(1, round(bare_share * len(base) / (1.0 - bare_share))) if bare_share > 0 else 0
    return base + (BARE_PAIR,) * copies


BUGHOUSE_SETS: dict[str, tuple[OpeningPair, ...]] = {
    "bughouse": bughouse_set(load_scores()),
    "bughouse_canon": CANON_PAIRS,
    "chess_controls": CONTROL_PAIRS,
    "bare": (BARE_PAIR,),
}


def clusters(pairs: tuple[OpeningPair, ...]) -> int:
    return len({p.id for p in pairs})


# ------------------------------------------------------------------------------------------------ scoring

def score_pairs(nodes: int = 800, out: str = SCORES_PATH, limit: int | None = None) -> dict:
    """Search every canon/control pair and the bare start with stock it04 for BOTH teams (both are on turn: White
    moves on A for T1 and on B for T2), equal clocks so ``TimeAdvantage false`` as the arena sends it. it04's
    no-advantage bias (about -270 cp each side at startpos-like positions, see hivemind.py) cancels in
    ``balance = (q_t1 - q_t2) / 2`` (T1's expected edge, [-1, 1]); the keep rule filters on |balance|.
    Writes ``{"pairs": {name: {"q_t1", "q_t2", "balance", "nodes"}}}``. Single process."""
    from arena.players.hivemind import HivemindProcess

    pairs = CANON_PAIRS + CONTROL_PAIRS + (BARE_PAIR,)
    if limit:
        pairs = pairs[:limit]
    proc = HivemindProcess()
    res: dict[str, dict] = {}
    try:
        for i, p in enumerate(pairs):
            r1 = proc.search(p.fen_a, p.fen_b, team="T1", nodes=nodes, time_advantage=False)
            r2 = proc.search(p.fen_a, p.fen_b, team="T2", nodes=nodes, time_advantage=False)
            q1, q2 = r1.q or 0.0, r2.q or 0.0
            res[p.name] = {"q_t1": round(q1, 4), "q_t2": round(q2, 4), "balance": round((q1 - q2) / 2, 4),
                           "nodes": [r1.nodes, r2.nodes]}
            print(f"{i + 1}/{len(pairs)} {p.name}: Q_T1 {q1:+.3f} Q_T2 {q2:+.3f} balance {(q1 - q2) / 2:+.3f}",
                  flush=True)
    finally:
        proc.close()
    doc = {"engine": "stock it04 (build-ninja/hivemind, hivemind-fp32.onnx)", "nodes": nodes,
           "time_advantage": False, "keep_abs_balance": KEEP_ABS_BALANCE, "pairs": res}
    with open(out, "w") as f:
        json.dump(doc, f, indent=1, sort_keys=True)
    return doc


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Bughouse opening sets: list or score with it04")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--nodes", type=int, default=800)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=SCORES_PATH)
    args = ap.parse_args(argv)
    if args.score:
        score_pairs(args.nodes, args.out, args.limit)
        return 0
    for name, s in BUGHOUSE_SETS.items():
        print(f"{name}: {len(s)} entries, {clusters(s)} clusters")
    for name, moves, src in CANON + CONTROLS:
        print(f"  {name:24s} {moves:28s} {src}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
