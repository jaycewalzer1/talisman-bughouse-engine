"""Compare training-side (Python) and engine-side (C++) policy indices.

Training labels come from hivemind.domain.move2planes.policy_label and
make_label_index, which hivemind.data.generate_planes uses. The engine index
comes from get_fast_policy_index via tools/cpp/policy_dump.

Usage: .venv/bin/python tools/policy_parity.py [--random N] [--seed S]
"""

import argparse
import collections
import random
import subprocess
import sys
from pathlib import Path

import chess
from chess.variant import CrazyhouseBoard

from hivemind.domain.move2planes import make_label_index, policy_label

ROOT = Path(__file__).resolve().parents[1]
DUMP = ROOT / "tools" / "cpp" / "policy_dump"

# Targeted positions (crazyhouse FEN, pockets in brackets). Each exercises a
# move type the random sample may under-represent.
TARGETED = [
    # White and Black can castle both ways.
    "r3k2r/pppqbppp/2n1bn2/3pp3/3PP3/2N1BN2/PPPQBPPP/R3K2R[] w KQkq - 0 10",
    "r3k2r/pppqbppp/2n1bn2/3pp3/3PP3/2N1BN2/PPPQBPPP/R3K2R[] b KQkq - 0 10",
    # Promotions (incl. capture-promotions) for both colors.
    "1n2k3/P1P5/8/8/8/8/1p1p4/4K1N1[] w - - 0 40",
    "1n2k3/P1P5/8/8/8/8/1p1p4/4K1N1[] b - - 0 40",
    # En passant available.
    "4k3/8/8/3pP3/8/8/8/4K3[] w - d6 0 30",
    "4k3/8/8/8/3Pp3/8/8/4K3[] b - d3 0 30",
    # Full pockets: every drop type.
    "r1bqk2r/pppp1ppp/2n2n2/4p3/1b2P3/2NP1N2/PPP2PPP/R1BQKB1R[PNBRQpnbrq] w KQkq - 0 6",
    "r1bqk2r/pppp1ppp/2n2n2/4p3/1b2P3/2NP1N2/PPP2PPP/R1BQKB1R[PNBRQpnbrq] b KQkq - 0 6",
]

POCKET_PIECES = "PNBRQ"


def to_fairy_fen(board: CrazyhouseBoard) -> str:
    """python-chess crazyhouse FEN -> Fairy-Stockfish bughouse FEN."""
    placement, rest = board.fen().split(" ", 1)
    placement = placement.replace("[", "/").replace("]", "")
    return f"{placement} {rest}"


def random_position(rng: random.Random) -> CrazyhouseBoard:
    board = CrazyhouseBoard()
    for _ in range(rng.randint(0, 60)):
        moves = list(board.legal_moves)
        if not moves:
            break
        board.push(rng.choice(moves))
        # Bughouse partners feed pockets; emulate with random gifts.
        if rng.random() < 0.15:
            color = rng.choice([chess.WHITE, chess.BLACK])
            piece = chess.Piece.from_symbol(rng.choice(POCKET_PIECES)).piece_type
            board.pockets[color].add(piece)
    if board.is_game_over():
        return random_position(rng)
    return board


def run_dump(lines):
    out = subprocess.run(
        [str(DUMP)], input="\n".join(lines) + "\n",
        capture_output=True, text=True, check=True).stdout
    rows = collections.defaultdict(dict)
    for line in out.splitlines():
        pos, board_num, _stm, uci, kind, _frm, _to, idx = line.split("\t")
        rows[(int(pos), int(board_num))][uci] = (kind, int(idx))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--random", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--examples", type=int, default=3)
    args = parser.parse_args()

    label_to_index = make_label_index()
    rng = random.Random(args.seed)
    boards = [CrazyhouseBoard(fen) for fen in TARGETED]
    boards += [random_position(rng) for _ in range(args.random)]
    if len(boards) % 2:
        boards.append(CrazyhouseBoard())
    # The dump takes board pairs; pair consecutive positions.
    pairs = [(boards[i], boards[i + 1]) for i in range(0, len(boards), 2)]
    rows = run_dump([f"{to_fairy_fen(a)}|{to_fairy_fen(b)}" for a, b in pairs])

    stats = collections.Counter()
    examples = collections.defaultdict(list)

    def record(key, detail):
        stats[key] += 1
        if len(examples[key]) < args.examples:
            examples[key].append(detail)

    for pos, pair in enumerate(pairs):
        for board_num, board in enumerate(pair):
            engine = rows[(pos, board_num)]
            python_moves = {m.uci(): m for m in board.legal_moves}
            for uci, move in python_moves.items():
                kind = ("castling" if board.is_castling(move)
                        else "drop" if move.drop
                        else "promotion" if move.promotion
                        else "en_passant" if board.is_en_passant(move)
                        else "normal")
                color = "w" if board.turn == chess.WHITE else "b"
                label = policy_label(board, move)
                py_idx = label_to_index.get(label, -1)
                if uci not in engine:
                    # The engine filters R/B underpromotions (no policy planes);
                    # training folds them into the queen-promotion slot.
                    if move.promotion in (chess.ROOK, chess.BISHOP):
                        stats[("unrepresented_underpromotion", kind, color)] += 1
                    else:
                        record(("missing_in_engine", kind, color), f"{uci} | {board.fen()}")
                    continue
                cpp_idx = engine[uci][1]
                if cpp_idx == py_idx:
                    stats[("match", kind, color)] += 1
                else:
                    record(("index_mismatch", kind, color),
                           f"{uci} label={label} py={py_idx} cpp={cpp_idx} | {board.fen()}")
            for uci, (kind, _idx) in engine.items():
                if uci != "pass" and uci not in python_moves:
                    record(("missing_in_python", kind, "-"), f"{uci} | {board.fen()}")
            cpp_pass = engine["pass"][1]
            py_pass = label_to_index["pass"]
            stats[("match" if cpp_pass == py_pass else "index_mismatch", "pass", "-")] += 1
            if cpp_pass != py_pass and not examples[("index_mismatch", "pass", "-")]:
                examples[("index_mismatch", "pass", "-")].append(
                    f"py={py_pass} cpp={cpp_pass}")

    print(f"positions: {len(boards)} ({len(TARGETED)} targeted, seed {args.seed})")
    width = max(len(" ".join(k)) for k in stats)
    for key in sorted(stats):
        print(f"  {' '.join(key):<{width}}  {stats[key]}")
    bad = {k: v for k, v in examples.items() if v}
    for key, items in sorted(bad.items()):
        print(f"\n{' '.join(key)}:")
        for item in items:
            print(f"  {item}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
