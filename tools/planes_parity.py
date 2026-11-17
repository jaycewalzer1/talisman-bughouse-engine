"""Compare Python (training) and C++ (engine) input planes and network outputs.

Replays random bughouse games through hivemind.domain.board.BughouseBoard
(the training encoder's board) and the engine's Board via
tools/cpp/planes_dump, then compares board2planes vs board_to_planes channel
by channel, and the network's value/policy on each side.

Usage: .venv/bin/python tools/planes_parity.py [--games N] [--seed S] [--clocks]

--clocks: the 43-channel clock layout (spec v2). Uses tools/cpp/planes_dump_clock
(built against engine/build-clock) and hivemind-clock-v2-fp32.onnx, sends random
clocks (some near 0, some negative, some above 300 s) and, for most records, random
elapsed times (some 0, some negative, some above 60 s) to both sides, and compares
channels 37..42 too. It also checks the engine's outputs against it04 (the stock
network fed the same planes without channels 37..42): value, wdl, moves_left, pi_a
and pi_b must match whatever the new planes hold, and the engine's think_gain
logits must match Python ORT on the same planes. The shipped stock ONNX
(hivemind-fp32.onnx) carries FP16-rounded weights, so against it the outputs differ
by up to ~2e-2 (as v1 did); for an exact check pass --it04-onnx with it04 exported
from its checkpoint in FP32 (make_clock_net.export_onnx), where the tolerance is 1e-3.
"""

import argparse
import collections
import random
import subprocess
import sys
import tempfile
from pathlib import Path

import chess
import numpy as np
import onnxruntime as ort

from hivemind.domain.board import BughouseBoard
from hivemind.domain.board2planes import board2planes
from hivemind.domain.move2planes import make_label_index, policy_label

ROOT = Path(__file__).resolve().parents[1]
DUMP = ROOT / "tools" / "cpp" / "planes_dump"
MODEL = ROOT / "third_party/hivemind/engine/models/hivemind-fp32.onnx"
START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR/ w KQkq - 0 1"

PER_BOARD = 37
NB_INPUT = 2 * PER_BOARD * 64
NB_POLICY = 73 * 64
RECORD = NB_INPUT + 5 + 2 * NB_POLICY
EXTRA = 0  # floats after pi_b in each record (2 think_gain logits in the clock build)
STOCK_MODEL = ROOT / "third_party/hivemind/engine/models/hivemind-fp32.onnx"
STOCK_PER_BOARD = 37


def use_clock_layout():
    """Switch the module to the 43-channel clock build (--clocks, spec v2)."""
    global DUMP, MODEL, PER_BOARD, NB_INPUT, RECORD, EXTRA, CHANNEL_NAMES
    DUMP = ROOT / "tools" / "cpp" / "planes_dump_clock"
    MODEL = ROOT / "third_party/hivemind/engine/models/hivemind-clock-v2-fp32.onnx"
    PER_BOARD = 43
    NB_INPUT = 2 * PER_BOARD * 64
    EXTRA = 2
    RECORD = NB_INPUT + 5 + 2 * NB_POLICY + EXTRA
    CHANNEL_NAMES = CHANNEL_NAMES + ["clock_us_lin", "clock_us_exp",
                                     "clock_them_lin", "clock_them_exp",
                                     "elapsed_lin", "elapsed_exp"]


def random_elapsed(rng):
    """(ms on A, ms on B) since the last move landed, or None (no elapsed field)."""
    if rng.random() < 0.1:
        return None
    def one():
        r = rng.random()
        if r < 0.1:
            return 0
        if r < 0.15:
            return rng.randint(-3000, -1)
        if r < 0.8:
            return rng.randint(1, 12000)
        return rng.randint(12000, 120000)
    return [one(), one()]


def random_clocks(rng):
    """Four clocks in ms (A_white, A_black, B_white, B_black), edge cases included."""
    def one():
        r = rng.random()
        if r < 0.1:
            return rng.randint(-2000, 0)
        if r < 0.3:
            return rng.randint(1, 15000)
        if r < 0.9:
            return rng.randint(15000, 300000)
        return rng.randint(300000, 900000)
    return [one() for _ in range(4)]

_PIECES = "PNBRQK"
CHANNEL_NAMES = (
    [f"us_{p}" for p in _PIECES] + [f"them_{p}" for p in _PIECES]
    + [f"pocket_us_{p}" for p in _PIECES[:5]] + [f"pocket_them_{p}" for p in _PIECES[:5]]
    + ["promoted_us", "promoted_them", "ep_square", "on_turn", "constant",
       "castle_us_OO", "castle_us_OOO", "castle_them_OO", "castle_them_OOO",
       "time_advantage", "last_from", "last_to", "halfmove_clock",
       "repetition_2", "repetition_3"]
)
assert len(CHANNEL_NAMES) == PER_BOARD


def channel_name(channel):
    board = "A" if channel < PER_BOARD else "B"
    return f"{board}.{CHANNEL_NAMES[channel % PER_BOARD]}"


class Game:
    """A replayable bughouse game: the move list plus the Python board."""

    def __init__(self):
        self.board = BughouseBoard()
        self.moves = []  # (board_num, uci)

    def push(self, board_num, move):
        self.moves.append((board_num, move.uci()))
        self.board.push(board_num, move)

    def push_uci(self, board_num, uci):
        self.push(board_num, self.board.parse_uci(board_num, uci))


def random_game(rng):
    game = Game()
    for _ in range(rng.randint(0, 90)):
        live = [b for b in (0, 1) if not game.board.boards[b].is_game_over()]
        if not live:
            break
        board_num = rng.choice(live)
        moves = list(game.board.boards[board_num].legal_moves)
        # Prefer drops/captures sometimes so pockets and promotions show up.
        tactical = [m for m in moves if m.drop or m.promotion
                    or game.board.boards[board_num].is_capture(m)]
        pool = tactical if tactical and rng.random() < 0.4 else moves
        game.push(board_num, rng.choice(pool))
    return game


def targeted_games():
    games = []
    # Two- and three-fold repetition on board A, plain play on B.
    for shuffles in (1, 2):
        game = Game()
        game.push_uci(1, "e2e4")
        for _ in range(shuffles):
            for uci in ("g1f3", "g8f6", "f3g1", "f6g8"):
                game.push_uci(0, uci)
        games.append(game)
    # En passant: capturable, and a double push with no capturing pawn.
    game = Game()
    for uci in ("e2e4", "a7a6", "e4e5", "d7d5"):
        game.push_uci(0, uci)
    games.append(game)
    game = Game()
    game.push_uci(0, "e2e4")
    games.append(game)
    # Captures feed the partner's pocket.
    game = Game()
    for uci in ("e2e4", "d7d5", "e4d5", "d8d5"):
        game.push_uci(0, uci)
    games.append(game)
    return games


def run_engine(records):
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "planes.bin"
        subprocess.run([str(DUMP), str(MODEL), str(out)],
                       input="\n".join(records) + "\n", text=True, check=True,
                       stderr=subprocess.DEVNULL)
        data = np.fromfile(out, dtype=np.float32)
    data = data.reshape(-1, RECORD)
    planes = data[:, :NB_INPUT].reshape(-1, 2 * PER_BOARD, 8, 8)
    value = data[:, NB_INPUT]
    pi_a = data[:, NB_INPUT + 5:NB_INPUT + 5 + NB_POLICY]
    pi_b = data[:, NB_INPUT + 5 + NB_POLICY:NB_INPUT + 5 + 2 * NB_POLICY]
    extra = {"wdl": data[:, NB_INPUT + 1:NB_INPUT + 4], "moves_left": data[:, NB_INPUT + 4],
             "think_gain": data[:, NB_INPUT + 5 + 2 * NB_POLICY:]}
    return planes, value, pi_a, pi_b, extra


def legal_indices(board, label_index):
    return sorted({label_index[policy_label(board, m)]
                   for m in board.legal_moves
                   if policy_label(board, m) in label_index})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--examples", type=int, default=3)
    parser.add_argument("--clocks", action="store_true",
                        help="43-channel clock build with random clocks and elapsed times")
    parser.add_argument("--it04-onnx", type=Path, default=None,
                        help="--clocks: it04 reference exported in FP32 from its checkpoint "
                             "(default: the stock hivemind-fp32.onnx, FP16-rounded weights, "
                             "reported but not enforced)")
    args = parser.parse_args()
    if args.clocks:
        use_clock_layout()

    rng = random.Random(args.seed)
    games = targeted_games() + [random_game(rng) for _ in range(args.games)]

    records, py_planes, meta = [], [], []
    for game in games:
        team = rng.choice([chess.WHITE, chess.BLACK])
        # Random clocks; the plane is "team's board-A clock beats board-B's".
        game.board.times = [[rng.randint(100, 1800) for _ in range(2)] for _ in range(2)]
        time_advantage = game.board.time_advantage(team) > 0
        moves = " ".join(f"{b}:{u}" for b, u in game.moves)
        fields = ["w" if team else "b", str(int(time_advantage)),
                  f"{START_FEN}|{START_FEN}", moves]
        if args.clocks:
            clocks = random_clocks(rng)
            elapsed = random_elapsed(rng)
            fields.append(" ".join(map(str, clocks)))
            if elapsed is not None:
                fields.append(" ".join(map(str, elapsed)))
            py_planes.append(board2planes(game.board, team, clocks=clocks, elapsed=elapsed,
                                          elapsed_planes=True))
        else:
            py_planes.append(board2planes(game.board, team))
        records.append("\t".join(fields))
        meta.append((game, team))
    py_planes = np.stack(py_planes)

    cpp_planes, cpp_value, cpp_pi_a, cpp_pi_b, cpp_extra = run_engine(records)
    assert cpp_planes.shape == py_planes.shape, (cpp_planes.shape, py_planes.shape)

    # --- planes ---------------------------------------------------------
    print(f"positions: {len(games)} ({len(games) - args.games} targeted, seed {args.seed})")
    diff = np.abs(cpp_planes - py_planes) > 1e-6
    bad_channels = np.nonzero(diff.any(axis=(0, 2, 3)))[0]
    positions_with_diff = diff.any(axis=(1, 2, 3))
    print(f"plane mismatch: {positions_with_diff.sum()}/{len(games)} positions, "
          f"{len(bad_channels)}/{2 * PER_BOARD} channels")
    for channel in bad_channels:
        hits = np.nonzero(diff[:, channel].any(axis=(1, 2)))[0]
        print(f"  {channel:2d} {channel_name(channel):<22} {len(hits):4d} positions")
        for pos in hits[:args.examples]:
            game, team = meta[pos]
            board_num = 0 if channel < PER_BOARD else 1
            py = py_planes[pos, channel]
            cpp = cpp_planes[pos, channel]
            if np.all(py == py.flat[0]) and np.all(cpp == cpp.flat[0]):
                detail = f"py={py.flat[0]:.4f} cpp={cpp.flat[0]:.4f}"
            else:
                detail = (f"py squares={np.flatnonzero(py).tolist()} "
                          f"cpp squares={np.flatnonzero(cpp).tolist()}")
            last = [u for b, u in game.moves if b == board_num][-3:]
            print(f"       {detail} | team={'w' if team else 'b'} "
                  f"last={last} fen={game.board.boards[board_num].fen()}")

    # --- network outputs ------------------------------------------------
    session = ort.InferenceSession(str(MODEL), providers=["CPUExecutionProvider"])
    names = [o.name for o in session.get_outputs()]

    def infer(planes):
        values, pa, pb = [], [], []
        for start in range(0, len(planes), 64):
            out = dict(zip(names, session.run(None, {"data": planes[start:start + 64]})))
            values.append(out["value"][:, 0]); pa.append(out["pi_a"]); pb.append(out["pi_b"])
        return np.concatenate(values), np.concatenate(pa), np.concatenate(pb)

    py_on_cpp_value, py_on_cpp_a, py_on_cpp_b = infer(cpp_planes)
    py_value, py_a, py_b = infer(py_planes)

    def summarize(label, a, b):
        d = np.abs(a - b)
        print(f"  {label:<44} max {d.max():.2e}  mean {d.mean():.2e}")

    print("\nnetwork outputs:")
    summarize("value: C++ engine vs Python ORT, same planes", cpp_value, py_on_cpp_value)
    summarize("pi_a:  C++ engine vs Python ORT, same planes", cpp_pi_a, py_on_cpp_a)
    summarize("pi_b:  C++ engine vs Python ORT, same planes", cpp_pi_b, py_on_cpp_b)
    summarize("value: end to end (C++ planes vs Python planes)", cpp_value, py_value)

    # Policy over legal moves, end to end: does the move ranking survive?
    label_index = make_label_index()
    top1_agree, total = 0, 0
    for pos, (game, team) in enumerate(meta):
        for board_num, py_logits, cpp_logits in ((0, py_a[pos], cpp_pi_a[pos]),
                                                 (1, py_b[pos], cpp_pi_b[pos])):
            board = game.board.boards[board_num]
            idx = legal_indices(board, label_index)
            if not idx or board.is_game_over():
                continue
            total += 1
            top1_agree += int(idx[int(np.argmax(py_logits[idx]))]
                              == idx[int(np.argmax(cpp_logits[idx]))])
    print(f"  top-1 legal move agrees end to end: {top1_agree}/{total}")

    it04_bad = False
    if args.clocks:
        # The widened network must equal it04 whatever channels 37..42 hold: feed it04 the
        # same C++ planes without them and compare every output it04 has.
        reference = args.it04_onnx or STOCK_MODEL
        enforce = args.it04_onnx is not None
        stock = ort.InferenceSession(str(reference), providers=["CPUExecutionProvider"])
        stock_names = [o.name for o in stock.get_outputs()]
        keep = [c for c in range(2 * PER_BOARD) if c % PER_BOARD < STOCK_PER_BOARD]
        stock_planes = np.ascontiguousarray(cpp_planes[:, keep])
        ref = collections.defaultdict(list)
        for start in range(0, len(stock_planes), 64):
            for name, arr in zip(stock_names, stock.run(None, {"data": stock_planes[start:start + 64]})):
                ref[name].append(arr)
        ref = {k: np.concatenate(v) for k, v in ref.items()}
        wdl_name = "wdl" if "wdl" in ref else "wdl_out"
        ml_name = "moves_left" if "moves_left" in ref else "plys_to_end"
        print(f"\nclock engine (v2 network, C++) vs it04 {reference.name} (Python ORT, planes 37..42 "
              f"dropped){'' if enforce else ' [FP16-rounded reference: not enforced]'}:")
        checks = [("value", cpp_value, ref["value"][:, 0]), ("pi_a", cpp_pi_a, ref["pi_a"]),
                  ("pi_b", cpp_pi_b, ref["pi_b"]),
                  ("wdl", cpp_extra["wdl"], ref[wdl_name].reshape(len(cpp_value), -1)),
                  ("moves_left", cpp_extra["moves_left"], ref[ml_name].reshape(-1))]
        for label, a, b in checks:
            d = np.abs(a - b)
            it04_bad |= enforce and bool(d.max() > 1e-3)
            print(f"  {label:<11} max {d.max():.2e}  mean {d.mean():.2e}")
        v2_tg = None
        if "think_gain" in names:
            v2_tg = np.concatenate([
                dict(zip(names, session.run(None, {"data": cpp_planes[s:s + 64]})))["think_gain"]
                for s in range(0, len(cpp_planes), 64)])
            d = np.abs(cpp_extra["think_gain"] - v2_tg)
            it04_bad |= bool(d.max() > 1e-3)
            print(f"  think_gain  C++ vs Python ORT, same planes: max {d.max():.2e}; "
                  f"C++ logits range [{cpp_extra['think_gain'].min():.4f}, "
                  f"{cpp_extra['think_gain'].max():.4f}]")
        else:
            print("  think_gain  (network has no think_gain output)")

    if positions_with_diff.any():
        worst = np.argsort(-np.abs(cpp_value - py_value))[:args.examples]
        print("\nlargest end-to-end value gaps:")
        for pos in worst:
            chans = [channel_name(c) for c in np.nonzero(diff[pos].any(axis=(1, 2)))[0]]
            print(f"  pos {pos}: cpp {cpp_value[pos]:+.3f} py {py_value[pos]:+.3f} "
                  f"differing channels {chans}")
    sys.exit(1 if positions_with_diff.any() or it04_bad else 0)


if __name__ == "__main__":
    main()
