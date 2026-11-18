"""Regression sets for the clock engine (audit 2026-11-17 WP-H, eval 10; theory 02 section 2.8, theory 04 table 3).

Two position files under ``tools/positions/``:

* ``sit_move.jsonl``: sit-versus-move hand positions for the theory 02 section 2.7/2.8 cases, each with clocks set
  both ways (diagonal +-10 s). The verdict is the human one: ``sit`` (pass on the attack board), ``move``, or
  ``any`` (not scored).
* ``drop_mates.jsonl``: drop-mate puzzles (smothered, yo-yo, f7 families, back rank); tracked per network as
  policy top-1 (is it a first move of a forced mate?) and value sign (positive for the attacker).

Every base position is written in one frame: the attack board has the attacker as White, to move; the supply
board has the attacker's partner as Black. ``placements`` expands it into up to four real seatings by mirroring
colours and/or swapping boards, so each case is asked of both policy heads and both teams:

    A_white  attack on A, as written            (T1)      A_black  attack on A, colours mirrored  (T2)
    B_white  attack on B, boards swapped        (T2)      B_black  boards swapped and mirrored    (T1)

Clock variants give the four clocks by role: ``us_att`` (attacker), ``opp_att`` (his opponent), ``us_sup``
(partner), ``opp_sup`` (partner's opponent). The attacker's diagonal is ``us_att - opp_sup``.

``claims`` are checked with python-chess under bughouse rules (captures go to the partner, a check that a dropped
piece could block is not mate) by ``verify``; a position whose claims fail is reported and not scored.

Modes (all single process, CPU ONNX Runtime; the engine only with ``--engine-nodes``):

    .venv/bin/python tools/regression.py verify
    .venv/bin/python tools/regression.py run --model third_party/hivemind/engine/models/hivemind-fp32.onnx \\
        [--engine-nodes 200 --binary hivemind] [--out runs/regression/it04.json] [--baseline old.json]

``run`` scores the raw network (policy top-1 over the legal moves plus the pass slot on the tested board, value
sign) and, with ``--engine-nodes N``, the engine's joint choice after an N-node search (clock binary: FreePass,
MateProbeAnyTime, TreeMoveMs; stock binary: TimeAdvantage only, as the arena sends them). ``--baseline`` prints
the entries whose verdict changed against an earlier ``--out`` file: the per-iteration regression gate.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass

import chess
import chess.variant
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from arena.interfaces import BOARD_OF, OPPONENT, PARTNER, SEATS, TEAM_OF  # noqa: E402
from arena.players.heuristic import team_time_edge  # noqa: E402
from arena.state import is_bughouse_mate  # noqa: E402

POSITIONS_DIR = os.path.join(ROOT, "tools", "positions")
SIT_MOVE_FILE = os.path.join(POSITIONS_DIR, "sit_move.jsonl")
DROP_MATE_FILE = os.path.join(POSITIONS_DIR, "drop_mates.jsonl")
MODELS_DIR = os.path.join(ROOT, "third_party", "hivemind", "engine", "models")
PASS_INDEX = 0  # marginal policy slot of a pass (engine convention, NOTES.md "Policy-index parity")
TA_CHANNEL = 31  # time-advantage plane in each board block
PIECE_OF = {"P": chess.PAWN, "N": chess.KNIGHT, "B": chess.BISHOP, "R": chess.ROOK, "Q": chess.QUEEN}
ROLES = ("us_att", "opp_att", "us_sup", "opp_sup")
OTHER = {"A": "B", "B": "A"}


# ---------------------------------------------------------------------------------------------- bughouse helpers

def push_bughouse(board: chess.variant.CrazyhouseBoard, move: chess.Move) -> chess.Piece | None:
    """Push ``move``; a captured piece goes to the partner (removed from the mover's pocket), as arena/state.py
    does. Returns the captured piece (None if none)."""
    mover = board.turn
    pocket = board.pockets[mover]
    before = {pt: pocket.count(pt) for pt in chess.PIECE_TYPES}
    board.push(move)
    for pt in chess.PIECE_TYPES:
        if pocket.count(pt) > before[pt]:
            pocket.remove(pt)
            return chess.Piece(pt, not mover)
    return None


def after(board: chess.variant.CrazyhouseBoard, move: chess.Move) -> chess.variant.CrazyhouseBoard:
    b = board.copy(stack=False)
    push_bughouse(b, move)
    return b


def mates_now(board: chess.variant.CrazyhouseBoard, move: chess.Move) -> bool:
    return is_bughouse_mate(after(board, move))


def forced_mate_moves(board: chess.variant.CrazyhouseBoard, depth: int, checks_only_from: int = 2) -> list[str]:
    """First moves of the side to move that force a bughouse mate within ``depth`` of its own moves, with both
    pockets as held (no supply from the partner) and the defender always replying (no sitting). From depth
    ``checks_only_from`` up the attacker's moves are restricted to checks (the puzzle families are forcing)."""

    def wins(b: chess.variant.CrazyhouseBoard, d: int) -> bool:
        for m in _attacker_moves(b, d, checks_only_from):
            if _forces(b, m, d):
                return True
        return False

    def _forces(b, m, d) -> bool:
        nb = after(b, m)
        if is_bughouse_mate(nb):
            return True
        if d <= 1:
            return False
        replies = list(nb.legal_moves)
        if not replies:
            return False  # stuck but not mate (blockable check): in bughouse the defender waits for a piece
        return all(wins(after(nb, r), d - 1) for r in replies)

    return sorted(m.uci() for m in _attacker_moves(board, depth, checks_only_from) if _forces(board, m, depth))


def _attacker_moves(b, d, checks_only_from):
    moves = list(b.legal_moves)
    if d >= checks_only_from:
        moves = [m for m in moves if b.gives_check(m)]
    return moves


def mirror_uci(uci: str) -> str:
    """The same move with colours mirrored (rank r -> 9 - r)."""
    def sq(s: str) -> str:
        return s[0] + str(9 - int(s[1]))
    if "@" in uci:
        return uci[:2] + sq(uci[2:4])
    return sq(uci[:2]) + sq(uci[2:4]) + uci[4:]


def mirror_fen(fen: str) -> str:
    return chess.variant.CrazyhouseBoard(fen).mirror().fen()


# ---------------------------------------------------------------------------------------------- placements

@dataclass
class Case:
    """One scored entry: a base position in one placement under one clock variant."""
    id: str
    set: str
    family: str
    placement: str  # attacker seat
    tag: str
    fen_a: str
    fen_b: str
    team: str
    board: str  # the tested (attack) board
    clocks: dict
    elapsed: dict
    expect: str | None = None  # sit / move / any (sit_move set)
    best: tuple[str, ...] = ()  # drop_mates set: first moves of a forced mate, in this placement's frame
    value_sign: int | None = None
    verified: bool = True
    note: str = ""

    @property
    def diagonal_ms(self) -> int:
        seat = self.placement
        return int(self.clocks[seat] - self.clocks[OPPONENT[PARTNER[seat]]])

    @property
    def time_advantage(self) -> bool:
        return team_time_edge(self.clocks, self.team) > 0


def role_seats(placement: str) -> dict[str, str]:
    us = placement
    return {"us_att": us, "opp_att": OPPONENT[us], "us_sup": PARTNER[us], "opp_sup": OPPONENT[PARTNER[us]]}


def place(attack: str, supply: str, placement: str) -> tuple[str, str]:
    """(fenA, fenB) for the base position in ``placement`` (the attacker's seat)."""
    mirrored = placement.endswith("black")
    att, sup = (mirror_fen(attack), mirror_fen(supply)) if mirrored else (attack, supply)
    return (att, sup) if BOARD_OF[placement] == "A" else (sup, att)


def expand(entry: dict) -> list[Case]:
    out = []
    for placement in entry.get("placements", list(SEATS)):
        fen_a, fen_b = place(entry["attack"], entry["supply"], placement)
        seats = role_seats(placement)
        mirrored = placement.endswith("black")
        best = tuple(mirror_uci(m) if mirrored else m for m in entry.get("best", ()))
        for var in entry["clocks"]:
            clocks = {seats[r]: int(var["clocks"][r]) for r in ROLES}
            out.append(Case(
                id=f"{entry['id']}@{placement}/{var['tag']}", set=entry["set"], family=entry["family"],
                placement=placement, tag=var["tag"], fen_a=fen_a, fen_b=fen_b, team=TEAM_OF[placement],
                board=BOARD_OF[placement], clocks=clocks, elapsed=dict(entry.get("elapsed", {"A": 0, "B": 0})),
                expect=var.get("expect"), best=best, value_sign=entry.get("value_sign"), note=entry.get("note", "")))
    return out


def load_entries(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip() and not line.startswith("#")]


# ---------------------------------------------------------------------------------------------- verification

def verify_entry(entry: dict) -> list[str]:
    """Claims of a base entry, checked in its written frame. Returns the failed claims (empty = all hold)."""
    errors = []
    att = chess.variant.CrazyhouseBoard(entry["attack"])
    sup = chess.variant.CrazyhouseBoard(entry["supply"])
    claims = entry.get("claims", {})
    if att.turn != chess.WHITE:
        errors.append("attack board must have White (the attacker) to move")
    if entry["set"] == "drop_mates" and sup.turn != chess.WHITE:
        errors.append("drop_mates: the supply board must have the partner's opponent (White) to move")
    for drop in claims.get("mating_drops", []):
        b = att.copy(stack=False)
        b.pockets[chess.WHITE].add(PIECE_OF[drop[0]])
        m = chess.Move.from_uci(drop)
        if m not in b.legal_moves or not mates_now(b, m):
            errors.append(f"{drop} does not mate with the piece in hand")
    if claims.get("no_mate_now") and any(mates_now(att, m) for m in att.legal_moves):
        errors.append("a move on the attack board mates now")
    if "supply_capture" in claims:
        m = chess.Move.from_uci(claims["supply_capture"])
        want = PIECE_OF[claims.get("piece", "N")]
        if sup.turn != chess.BLACK or m not in sup.legal_moves:
            errors.append(f"supply capture {m} is not legal for the partner")
        else:
            got = after_capture_piece(sup, m)
            if got is None or got.piece_type != want:
                errors.append(f"supply capture {m} does not win a {claims.get('piece', 'N')}")
    if claims.get("pinned_supply"):
        want = PIECE_OF[claims.get("piece", "N")]
        if sup.turn != chess.WHITE:
            errors.append("pinned_supply: White (partner's opponent) must be to move on the supply board")
        for m in sup.legal_moves:
            nb = after(sup, m)
            if not any(after_capture_piece(nb, r) and after_capture_piece(nb, r).piece_type == want
                       for r in nb.legal_moves):
                errors.append(f"pinned_supply: after {m} the partner cannot win a {claims.get('piece', 'N')}")
                break
    if claims.get("no_supply"):
        want = PIECE_OF[claims.get("piece", "N")]
        if sup.pieces(want, chess.WHITE) or sup.pockets[chess.WHITE].count(want):
            errors.append("no_supply: the partner's opponent holds the needed piece type")
    if claims.get("every_move_allows_mate"):
        for m in att.legal_moves:
            nb = after(att, m)
            if not any(mates_now(nb, r) for r in nb.legal_moves):
                errors.append(f"every_move_allows_mate: {m} holds")
                break
    if "forced_mate" in claims:
        depth = int(claims["forced_mate"])
        found = forced_mate_moves(att, depth)
        if not found:
            errors.append(f"no forced mate within {depth}")
        elif sorted(entry.get("best", [])) != found:
            errors.append(f"best {sorted(entry.get('best', []))} != solver {found}")
    return errors


def after_capture_piece(board, move) -> chess.Piece | None:
    if board.is_capture(move) and not board.is_en_passant(move):
        return board.piece_at(move.to_square)
    if board.is_en_passant(move):
        return chess.Piece(chess.PAWN, not board.turn)
    return None


# ---------------------------------------------------------------------------------------------- network

class NetEval:
    """Raw network outputs (no search) for bughouse positions given as FENs and clocks, via ONNX Runtime on CPU.

    Inputs: stock 74 channels (it04: the time-advantage plane only), or the clock layouts 82 / 86 (clocks and,
    for 86, elapsed times as the engine fills them). The time-advantage plane is set from ``team_time_edge`` > 0,
    as arena/players/hivemind.py sends ``TimeAdvantage``, unless a position gives ``time_advantage``."""

    def __init__(self, model: str, threads: int = 4):
        import onnxruntime as ort
        from hivemind.domain.move2planes import make_label_index
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        opts.inter_op_num_threads = 1
        self.model = model
        self.session = ort.InferenceSession(model, opts, providers=["CPUExecutionProvider"])
        self.names = [o.name for o in self.session.get_outputs()]
        self.channels = int(self.session.get_inputs()[0].shape[1])
        self.per_board = self.channels // 2
        self.label_index = make_label_index()

    def planes(self, fen_a: str, fen_b: str, team: str, clocks: dict | None, elapsed: dict | None,
               time_advantage: bool | None = None) -> np.ndarray:
        from hivemind.domain.board import BughouseBoard
        from hivemind.domain.board2planes import board2planes
        board = BughouseBoard()
        board.boards[0].set_fen(fen_a)
        board.boards[1].set_fen(fen_b)
        side = chess.WHITE if team == "T1" else chess.BLACK
        if self.per_board == 37:
            x = board2planes(board, side)
        else:
            c = tuple(int(clocks[s]) for s in SEATS) if clocks else None
            if self.per_board >= 43:
                e = (int(elapsed["A"]), int(elapsed["B"])) if (elapsed and c) else None
                x = board2planes(board, side, clocks=c, elapsed=e, elapsed_planes=True,
                                 diagonal_plane=self.per_board >= 44)
            else:
                x = board2planes(board, side, clocks=c, clock_planes=True)
        if time_advantage is None:
            time_advantage = bool(clocks) and team_time_edge(clocks, team) > 0
        x[TA_CHANNEL] = float(time_advantage)
        x[self.per_board + TA_CHANNEL] = float(time_advantage)
        if x.shape[0] != self.channels:
            raise ValueError(f"planes have {x.shape[0]} channels, network wants {self.channels}")
        return x

    def run(self, planes: np.ndarray, batch: int = 64) -> dict[str, np.ndarray]:
        outs: dict[str, list] = {}
        for s in range(0, len(planes), batch):
            res = self.session.run(None, {"data": planes[s:s + batch].astype(np.float32)})
            for name, arr in zip(self.names, res):
                outs.setdefault(name, []).append(arr)
        return {k: np.concatenate(v) for k, v in outs.items()}

    def board_policy(self, logits: np.ndarray, fen: str) -> dict:
        """Softmax over the legal moves plus the pass slot of ``fen``'s side to move: top-1, p(pass), ranks."""
        board = chess.variant.CrazyhouseBoard(fen)
        from hivemind.domain.move2planes import policy_label
        moves, idx = [], []
        for m in board.legal_moves:
            label = policy_label(board, m)
            if label in self.label_index:
                moves.append(m.uci())
                idx.append(self.label_index[label])
        moves.append("pass")
        idx.append(PASS_INDEX)
        z = logits[idx].astype(np.float64)
        p = np.exp(z - z.max())
        p /= p.sum()
        order = np.argsort(-p)
        return {"top": moves[order[0]], "top_p": float(p[order[0]]), "p_pass": float(p[-1]),
                "pass_rank": int(np.nonzero(order == len(moves) - 1)[0][0]), "n_legal": len(moves) - 1,
                "probs": dict(zip(moves, p.tolist()))}

    def on_turn_boards(self, fen_a: str, fen_b: str, team: str) -> list[str]:
        out = []
        for b, fen in (("A", fen_a), ("B", fen_b)):
            color = "white" if fen.split()[1] == "w" else "black"
            if TEAM_OF[f"{b}_{color}"] == team:
                out.append(b)
        return out


def evaluate_cases(net: NetEval, cases: list[Case]) -> list[dict]:
    x = np.stack([net.planes(c.fen_a, c.fen_b, c.team, c.clocks, c.elapsed) for c in cases])
    out = net.run(x)
    rows = []
    for i, c in enumerate(cases):
        logits = out["pi_a"][i] if c.board == "A" else out["pi_b"][i]
        pol = net.board_policy(logits, c.fen_a if c.board == "A" else c.fen_b)
        rows.append({"value": float(out["value"][i, 0]), "policy": pol})
    return rows


# ---------------------------------------------------------------------------------------------- scoring

def score_case(case: Case, net_row: dict, engine_move: str | None | object = ...) -> dict:
    pol = net_row["policy"]
    row = {"id": case.id, "set": case.set, "family": case.family, "placement": case.placement, "tag": case.tag,
           "diag_ms": case.diagonal_ms, "ta": case.time_advantage, "verified": case.verified,
           "value": round(net_row["value"], 4), "top": pol["top"], "top_p": round(pol["top_p"], 4),
           "p_pass": round(pol["p_pass"], 4), "pass_rank": pol["pass_rank"]}
    if case.set == "sit_move":
        row["expect"] = case.expect
        row["net_action"] = "sit" if pol["top"] == "pass" else "move"
        row["net_ok"] = None if case.expect in (None, "any") else row["net_action"] == case.expect
    else:
        row["best"] = list(case.best)
        row["net_ok"] = pol["top"] in case.best
        row["value_ok"] = None if case.value_sign is None else bool(np.sign(net_row["value"]) == case.value_sign)
        row["p_best"] = round(sum(pol["probs"].get(m, 0.0) for m in case.best), 4)
    if engine_move is not ...:
        if case.set == "sit_move":
            row["engine_action"] = "none" if engine_move == "no_root" else ("sit" if engine_move is None else "move")
            row["engine_ok"] = None if case.expect in (None, "any") else row["engine_action"] == case.expect
        else:
            row["engine_move"] = engine_move
            row["engine_ok"] = engine_move in case.best
    return row


def engine_moves(cases: list[Case], binary: str, model: str, nodes: int, tree_ms: int, pass_exploration: int = 0) -> list:
    from arena.players.hivemind import HivemindProcess, resolve_binary
    path = resolve_binary(binary)
    clock = os.path.basename(path).startswith("hivemind-clock") or binary == "hivemind-clock"
    proc = HivemindProcess(path, os.path.abspath(model))
    out = []
    try:
        for c in cases:
            if clock:
                res = proc.search(c.fen_a, c.fen_b, team=c.team, nodes=nodes, time_advantage=c.time_advantage,
                                  clocks=c.clocks, elapsed=c.elapsed, require_option=False,
                                  options={"FreePass": True, "MateProbeAnyTime": True, "TreeMoveMs": tree_ms,
                                           **({"PassExploration": pass_exploration} if pass_exploration else {})})
            else:
                res = proc.search(c.fen_a, c.fen_b, team=c.team, nodes=nodes, time_advantage=c.time_advantage)
            out.append("no_root" if res.no_root else res.move_for(c.board))
    finally:
        proc.close()
    return out


def summarize(rows: list[dict]) -> dict:
    """Per set and family: scored count and hits, for the network and (when present) the engine."""
    out: dict = {}
    for r in rows:
        if not r["verified"]:
            continue
        for key in (r["set"], f"{r['set']}/{r['family']}"):
            s = out.setdefault(key, {"n": 0, "net_ok": 0, "net_scored": 0, "engine_ok": 0, "engine_scored": 0,
                                     "value_ok": 0, "value_scored": 0})
            s["n"] += 1
            for k in ("net", "engine", "value"):
                v = r.get(f"{k}_ok")
                if v is not None:
                    s[f"{k}_scored"] += 1
                    s[f"{k}_ok"] += int(bool(v))
    return out


def format_summary(summary: dict) -> str:
    lines = [f"{'set/family':<34} {'n':>4} {'net':>9} {'value+':>9} {'engine':>9}"]
    for key in sorted(summary):
        s = summary[key]

        def frac(k):
            return f"{s[k + '_ok']}/{s[k + '_scored']}" if s[k + "_scored"] else "-"
        lines.append(f"{key:<34} {s['n']:>4} {frac('net'):>9} {frac('value'):>9} {frac('engine'):>9}")
    return "\n".join(lines)


def load_cases(paths: list[str], *, check: bool = True) -> tuple[list[Case], dict[str, list[str]]]:
    cases, failures = [], {}
    for path in paths:
        for entry in load_entries(path):
            errs = verify_entry(entry) if check else []
            if errs:
                failures[entry["id"]] = errs
            for c in expand(entry):
                c.verified = not errs
                cases.append(c)
    return cases, failures


def diff_baseline(rows: list[dict], baseline_path: str) -> list[str]:
    with open(baseline_path) as f:
        old = {r["id"]: r for r in json.load(f)["rows"]}
    changes = []
    for r in rows:
        o = old.get(r["id"])
        if o is None:
            continue
        for k in ("net_ok", "engine_ok", "value_ok"):
            if k in r and k in o and r[k] != o[k]:
                changes.append(f"{r['id']}: {k} {o[k]} -> {r[k]} (top {o.get('top')} -> {r.get('top')})")
    return changes


# ---------------------------------------------------------------------------------------------- CLI

def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("verify", help="check every claim of the position files with python-chess")
    v.add_argument("--file", action="append", default=None)
    r = sub.add_parser("run", help="score a network (and optionally the engine) on the sets")
    r.add_argument("--file", action="append", default=None, help="position file(s) (default: both sets)")
    r.add_argument("--model", default=os.path.join(MODELS_DIR, "hivemind-clock-v2-fp32.onnx"))
    r.add_argument("--engine-nodes", type=int, default=0, help="also run the engine at N nodes (0 = net only)")
    r.add_argument("--binary", default=None,
                   help="engine binary: hivemind (stock) or hivemind-clock; default by the network's channels")
    r.add_argument("--tree-ms", type=int, default=500, help="TreeMoveMs for the clock binary (default 500)")
    r.add_argument("--pass-exploration", type=int, default=0,
                   help="clock binary: send PassExploration N (spec v3 E3; 0 = off, the engine default)")
    r.add_argument("--out", default=None, help="write rows and summary as JSON")
    r.add_argument("--baseline", default=None, help="an earlier --out file: print the changed verdicts")
    r.add_argument("--threads", type=int, default=4)
    r.add_argument("--rows", action="store_true", help="print every row")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    files = args.file or [SIT_MOVE_FILE, DROP_MATE_FILE]
    if args.cmd == "verify":
        n_bad = 0
        for path in files:
            entries = load_entries(path)
            for e in entries:
                errs = verify_entry(e)
                n_bad += bool(errs)
                print(f"{'FAIL' if errs else 'ok  '} {e['id']:<24} {'; '.join(errs)}")
            print(f"{path}: {len(entries)} base positions, {len(expand_all(entries))} cases")
        return 1 if n_bad else 0
    t0 = time.perf_counter()
    cases, failures = load_cases(files)
    for k, errs in failures.items():
        print(f"unverified (not scored): {k}: {'; '.join(errs)}")
    net = NetEval(args.model, threads=args.threads)
    net_rows = evaluate_cases(net, cases)
    moves = [...] * len(cases)
    binary = args.binary
    if args.engine_nodes:
        if binary is None:
            binary = "hivemind" if net.per_board == 37 else "hivemind-clock"
        moves = engine_moves(cases, binary, args.model, args.engine_nodes, args.tree_ms, args.pass_exploration)
    rows = [score_case(c, nr, m) for c, nr, m in zip(cases, net_rows, moves)]
    summary = summarize(rows)
    print(f"model {args.model} ({net.channels} channels)"
          + (f", engine {binary} at {args.engine_nodes} nodes" if args.engine_nodes else ""))
    if args.rows:
        for r in rows:
            print(json.dumps(r))
    print(format_summary(summary))
    if args.baseline:
        changes = diff_baseline(rows, args.baseline)
        print(f"\n{len(changes)} verdicts changed against {args.baseline}")
        for line in changes:
            print("  " + line)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"model": args.model, "channels": net.channels, "engine_nodes": args.engine_nodes,
                       "binary": binary, "summary": summary, "rows": rows,
                       "wall_s": round(time.perf_counter() - t0, 1)}, f, indent=1)
    return 0


def expand_all(entries: list[dict]) -> list[Case]:
    return [c for e in entries for c in expand(e)]


if __name__ == "__main__":
    sys.exit(main())
