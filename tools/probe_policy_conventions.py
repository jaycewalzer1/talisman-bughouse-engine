"""Ask the network which policy index it learned for castling and pass.

Ranks the logit at each candidate index among legal-move logits over random
positions with castling available. Run: .venv/bin/python tools/probe_policy_conventions.py
"""
import sys, random, numpy as np, chess
sys.path.insert(0, "tools")
from policy_parity import random_position
from hivemind.domain.board import BughouseBoard
from hivemind.domain.board2planes import board2planes
from hivemind.domain.move2planes import make_map, mirrorMoveUCI
import onnxruntime as ort
M = "third_party/hivemind/engine/models/hivemind-fp32.onnx"
s = ort.InferenceSession(M, providers=["CPUExecutionProvider"])
lab = {l: i for i, l in enumerate(make_map())}
def lbl(b, m):
    u = m.uci(); u = mirrorMoveUCI(u) if b.turn == chess.BLACK else u
    return u[:-1] if len(u) == 5 and u[-1] != "n" else u
rng = random.Random(1)
ranks = {"py_OO": [], "cpp_OO": [], "py_OOO": [], "cpp_OOO": [], "pass63": [], "pass0": []}
gap = []
n = 0
while n < 300:
    a = random_position(rng)
    castles = [m for m in a.legal_moves if a.is_castling(m)]
    if not castles: continue
    bb = BughouseBoard(); bb.boards[0] = a.copy()
    team = a.turn  # team on turn on board A
    bb.boards[1].turn = not team   # partner's board: partner (opposite color) on turn
    x = board2planes(bb, team)[None].astype(np.float32)
    out = dict(zip([o.name for o in s.get_outputs()], s.run(None, {"data": x})))
    pa = out["pi_a"][0]
    legal = sorted({lab[lbl(a, m)] for m in a.legal_moves if lbl(a, m) in lab})
    def rank(i):  # rank of logit i among legal-move logits + itself (0 = best)
        pool = set(legal) | {i}
        return sorted(pool, key=lambda j: -pa[j]).index(i) / max(1, len(pool) - 1)
    for m in castles:
        k = "OO" if chess.square_file(m.to_square) == 6 else "OOO"
        u = lbl(a, m)
        cpp = {"e1g1": "e1h1", "e1c1": "e1a1"}[u]
        ranks["py_" + k].append(rank(lab[u])); ranks["cpp_" + k].append(rank(lab[cpp]))
    ranks["pass63"].append(rank(63)); ranks["pass0"].append(rank(0))
    gap.append(pa[63] - pa[0])
    n += 1
for k, v in ranks.items():
    print(f"{k:8s} n={len(v):4d} mean percentile rank among legal moves: {np.mean(v):.3f} (0=top, 1=bottom)")
print(f"logit[63]-logit[0]: mean {np.mean(gap):.2f}, min {np.min(gap):.2f}, max {np.max(gap):.2f}")
