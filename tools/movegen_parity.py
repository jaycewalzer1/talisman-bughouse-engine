"""Compare legal moves: upstream Fairy-Stockfish vs Hivemind's vendored copy.

Positions come from random bughouse games (tools/planes_parity.random_game).
Upstream moves come from `go perft 1` (UCI_Variant bughouse); Hivemind's from
tools/cpp/policy_dump. Hivemind deliberately omits R/B underpromotions (the
policy has no planes for them); those are reported separately.

Usage: .venv/bin/python tools/movegen_parity.py [--games N] [--seed S]
"""

import argparse
import collections
import random
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from planes_parity import random_game  # noqa: E402
from policy_parity import run_dump, to_fairy_fen  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "third_party" / "Fairy-Stockfish" / "src" / "stockfish"


def upstream_moves(fens):
    # `go perft` runs on a search thread: readyok can arrive before its
    # output, and quit can cut it short. Delimit on "Nodes searched" and only
    # quit once every result is in.
    proc = subprocess.Popen([str(UPSTREAM)], stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, text=True, bufsize=1)
    proc.stdin.write("uci\nsetoption name UCI_Variant value bughouse\n")
    results = []
    for fen in fens:
        proc.stdin.write(f"position fen {fen}\ngo perft 1\n")
        proc.stdin.flush()
        moves = set()
        for line in proc.stdout:
            line = line.strip()
            if line.startswith("Nodes searched"):
                break
            if ": " in line and not line.startswith(("option", "info", "id ")):
                move, count = line.rsplit(": ", 1)
                if count.isdigit():
                    moves.add(move)
        results.append(moves)
    proc.stdin.write("quit\n")
    proc.stdin.close()
    proc.wait()
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--examples", type=int, default=3)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    games = [random_game(rng) for _ in range(args.games)]
    boards = [(g.board.boards[0], g.board.boards[1]) for g in games]

    hivemind = run_dump([f"{to_fairy_fen(a)}|{to_fairy_fen(b)}" for a, b in boards])
    upstream_fens = [b.fen() for pair in boards for b in pair]
    upstream = upstream_moves(upstream_fens)
    assert len(upstream) == len(upstream_fens), (len(upstream), len(upstream_fens))

    stats = collections.Counter()
    examples = collections.defaultdict(list)
    for pos, pair in enumerate(boards):
        for board_num, board in enumerate(pair):
            if board.is_game_over():
                continue
            ours = {u for u in hivemind[(pos, board_num)] if u != "pass"}
            theirs = upstream[2 * pos + board_num]
            stats["positions"] += 1
            stats["moves_agree"] += len(ours & theirs)
            for move in sorted(theirs - ours):
                kind = ("underpromotion" if len(move) == 5 and move[-1] in "rb"
                        else "only_upstream")
                stats[kind] += 1
                if kind != "underpromotion" and len(examples[kind]) < args.examples:
                    examples[kind].append(f"{move} | {board.fen()}")
            for move in sorted(ours - theirs):
                stats["only_hivemind"] += 1
                if len(examples["only_hivemind"]) < args.examples:
                    examples["only_hivemind"].append(f"{move} | {board.fen()}")

    for key in ("positions", "moves_agree", "underpromotion", "only_upstream", "only_hivemind"):
        print(f"  {key:<16} {stats[key]}")
    for key, items in examples.items():
        print(f"\n{key}:")
        for item in items:
            print(f"  {item}")
    sys.exit(1 if stats["only_upstream"] or stats["only_hivemind"] else 0)


if __name__ == "__main__":
    main()
