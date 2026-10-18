"""Which policy index does the network use for castling in real openings?

Plays well-known lines where O-O-O (or O-O) is a standard plan, then ranks
the logits at the python-chess square (e1c1/e1g1) and the engine's
king-to-rook square (e1a1/e1h1) among all legal moves.
Run: .venv/bin/python tools/probe_castling_openings.py
"""

import chess
import numpy as np
import onnxruntime as ort

from hivemind.domain.board import BughouseBoard
from hivemind.domain.board2planes import board2planes
from hivemind.domain.move2planes import make_label_index, policy_label

MODEL = "third_party/hivemind/engine/models/hivemind-fp32.onnx"

LINES = {
    "Sicilian Yugoslav, W O-O-O": "e2e4 c7c5 g1f3 d7d6 d2d4 c5d4 f3d4 g8f6 b1c3 g7g6 c1e3 f8g7 f2f3 e8g8 d1d2 b8c6 f1c4 c8d7",
    "English Attack, W O-O-O": "e2e4 c7c5 g1f3 d7d6 d2d4 c5d4 f3d4 g8f6 b1c3 a7a6 c1e3 e7e5 d4b3 c8e6 f2f3 f8e7 d1d2 e8g8",
    "Caro-Kann classical, W O-O-O": "e2e4 c7c6 d2d4 d7d5 b1c3 d5e4 c3e4 c8f5 e4g3 f5g6 h2h4 h7h6 g1f3 b8d7 h4h5 g6h7 f1d3 h7d3 d1d3 e7e6 c1d2 g8f6",
    "QGD exchange, B O-O-O": "d2d4 d7d5 c2c4 e7e6 b1c3 g8f6 c1g5 b8d7 c4d5 e6d5 e2e3 c7c6 f1d3 f8e7 d1c2 h7h6 g5h4 d8a5 g1f3 b7b6 e1g1 c8b7",
    "Italian, W O-O": "e2e4 e7e5 g1f3 b8c6 f1c4 f8c5 c2c3 g8f6 d2d3 d7d6",
    "Ruy Lopez, W O-O": "e2e4 e7e5 g1f3 b8c6 f1b5 a7a6 b5a4 g8f6",
    "Petrov, B O-O": "e2e4 e7e5 g1f3 g8f6 f3e5 d7d6 e5f3 f6e4 d2d4 d6d5 f1d3 b8c6 e1g1 f8e7 c2c4 c6b4",
}

session = ort.InferenceSession(MODEL, providers=["CPUExecutionProvider"])
label_index = make_label_index()
for name, line in LINES.items():
    bb = BughouseBoard()
    for uci in line.split():
        bb.push_uci(0, uci)
    a = bb.boards[0]
    team = a.turn
    bb.boards[1].turn = not team  # partner on turn too
    out = dict(zip([o.name for o in session.get_outputs()],
                   session.run(None, {"data": board2planes(bb, team)[None]})))
    logits = out["pi_a"][0]
    legal = {}
    for m in a.legal_moves:
        legal.setdefault(label_index[policy_label(a, m)], a.san(m))
    order = sorted(legal, key=lambda i: -logits[i])
    castles = [m for m in a.legal_moves if a.is_castling(m)]
    for m in castles:
        engine_idx = label_index[policy_label(a, m)]  # king -> rook square
        mirrored = m.uci() if a.turn == chess.WHITE else chess.Move(
            chess.square_mirror(m.from_square), chess.square_mirror(m.to_square)).uci()
        python_idx = label_index[mirrored]  # king landing square (old labels)
        def rank(i):
            return sorted(set(legal) | {i}, key=lambda j: -logits[j]).index(i) + 1
        print(f"{name:32s} {a.san(m):6s} engine-index rank {rank(engine_idx):2d}/{len(legal)}"
              f"  python-index rank {rank(python_idx):2d}  (top: {legal[order[0]]})")
