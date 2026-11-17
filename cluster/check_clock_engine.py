"""Checks the hivemind-clock build's think_gain output end to end, for the GPU (TensorRT) path.

Spec v2 adds the think_gain binding and Engine::evaluateThinkGain to engine.cc (TensorRT). The Mac only builds the
ONNX Runtime backend, so this is the check to run once on a GPU node after cluster/build.sh, before any large run.

What it does:
  1. Copies the v2 network with random weights in the last think_gain layer (a freshly widened network has that
     layer at exactly 0, so every position would print 0.5 and a wrong binding could still pass).
  2. Plays random bughouse positions (random clocks and elapsed times), and for each one runs the engine through
     UCI: ``position ... clocks ... elapsed ...`` then ``go nodes 1 sample``.
  3. Requires an ``info string think_gain <pa> <pb>`` line for every position (no line, or "failed", is an error),
     and compares it with Python ONNX Runtime on the planes the engine itself encoded (the sample record). Those
     planes are quantized to bytes and TensorRT runs in FP16, so the tolerance is 0.03 in probability.

Planes parity itself (C++ vs Python encoder, channels 37..42) is CPU code shared by both backends: it is checked on
the Mac with ``tools/planes_parity.py --clocks``.

    .venv/bin/python cluster/check_clock_engine.py [--positions 64] [--binary hivemind-clock] [--model <v2 onnx>]

Exit 0 when every position printed think_gain and all agree within the tolerance.
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from arena.players.hivemind import ENGINE_DIR, HivemindProcess, resolve_binary  # noqa: E402
from arena.recorder import view_sample  # noqa: E402

HEAD_WEIGHT = "think_gain_head.body.2.weight"
HEAD_BIAS = "think_gain_head.body.2.bias"


def randomized_model(src: str, dst: str, seed: int) -> None:
    """``src`` with random weights in the last think_gain layer, saved to ``dst``."""
    import onnx
    from onnx import numpy_helper

    model = onnx.load(src)
    rng = np.random.default_rng(seed)
    found = set()
    for init in model.graph.initializer:
        if init.name in (HEAD_WEIGHT, HEAD_BIAS):
            arr = numpy_helper.to_array(init)
            scale = 0.5 if init.name == HEAD_WEIGHT else 0.3
            new = (rng.standard_normal(arr.shape) * scale).astype(arr.dtype)
            init.CopyFrom(numpy_helper.from_array(new, init.name))
            found.add(init.name)
    if found != {HEAD_WEIGHT, HEAD_BIAS}:
        raise SystemExit(f"{src}: no think_gain head ({HEAD_WEIGHT} / {HEAD_BIAS}); needs a spec v2 network")
    onnx.save(model, dst)


def random_positions(count: int, seed: int):
    """(fen_a, fen_b, team, time_advantage, clocks, elapsed) from random play, both boards still live."""
    from hivemind.domain.board import BughouseBoard

    rng = random.Random(seed)
    out = []
    while len(out) < count:
        board = BughouseBoard()
        for _ in range(rng.randint(0, 60)):
            live = [b for b in (0, 1) if not board.boards[b].is_game_over()]
            if not live:
                break
            b = rng.choice(live)
            board.push(b, rng.choice(list(board.boards[b].legal_moves)))
        if any(x.is_game_over() for x in board.boards):
            continue
        fen_a, fen_b = board.fen()
        clocks = {s: rng.choice([rng.randint(0, 15000), rng.randint(15000, 300000), rng.randint(300000, 600000)])
                  for s in ("A_white", "A_black", "B_white", "B_black")}
        elapsed = {"A": rng.randint(0, 20000), "B": rng.randint(0, 20000)}
        team = rng.choice(["T1", "T2"])
        out.append((fen_a, fen_b, team, rng.random() < 0.5, clocks, elapsed))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--binary", default="hivemind-clock")
    ap.add_argument("--model", default=os.path.join(ENGINE_DIR, "models", "hivemind-clock-v2-fp32.onnx"))
    ap.add_argument("--positions", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tolerance", type=float, default=0.03, help="max |p_engine - p_ort| (default 0.03)")
    ap.add_argument("--timeout", type=float, default=1800.0, help="engine start wait, s (TensorRT engine build)")
    args = ap.parse_args(argv)

    import onnxruntime as ort

    binary = resolve_binary(args.binary)
    with tempfile.TemporaryDirectory() as tmp:
        model = os.path.join(tmp, "think-check-fp32.onnx")
        randomized_model(args.model, model, args.seed)
        session = ort.InferenceSession(model, providers=["CPUExecutionProvider"])
        names = [o.name for o in session.get_outputs()]
        input_width = session.get_inputs()[0].shape[1]
        print(f"binary {binary}\nmodel  {args.model} (last think_gain layer randomized, seed {args.seed})")

        process = HivemindProcess(binary, model, timeout_s=args.timeout)
        missing = failed = no_root = 0
        worst = 0.0
        diffs = []
        try:
            process.start()
            # Positions where the team is not on turn on either board give 'bestmove (none)' and no evaluation:
            # skipped and replaced, so --positions comparisons are made.
            for fen_a, fen_b, team, adv, clocks, elapsed in random_positions(4 * args.positions, args.seed):
                if len(diffs) + missing + failed >= args.positions:
                    break
                res = process.search(fen_a, fen_b, team=team, nodes=1, time_advantage=adv, clocks=clocks,
                                     elapsed=elapsed, sample=True)
                if res.no_root:
                    no_root += 1
                    continue
                if any(l.startswith("info string think_gain failed") for l in res.lines):
                    failed += 1
                    continue
                if res.think_gain is None or res.sample is None:
                    missing += 1
                    continue
                view = view_sample(res.sample)
                if view.channels != input_width:
                    raise SystemExit(f"engine planes have {view.channels} channels, network wants {input_width}")
                planes = (np.frombuffer(view.planes, dtype=np.uint8).astype(np.float32) / 255.0)
                planes = planes.reshape(1, view.channels, 8, 8)
                out = dict(zip(names, session.run(None, {"data": planes})))
                ref = [1.0 / (1.0 + math.exp(-float(x))) for x in out["think_gain"][0]]
                d = max(abs(a - b) for a, b in zip(res.think_gain, ref))
                diffs.append(d)
                worst = max(worst, d)
        finally:
            process.close()

    n = len(diffs)
    spread = f"mean {np.mean(diffs):.4f}" if n else "no comparisons"
    ok = n == args.positions and missing == 0 and failed == 0 and worst <= args.tolerance
    print(f"positions {args.positions}: compared {n}, no think_gain line {missing}, 'think_gain failed' {failed} "
          f"(skipped {no_root} with no root); "
          f"max |p_engine - p_ort| {worst:.4f} ({spread}), tolerance {args.tolerance}: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
