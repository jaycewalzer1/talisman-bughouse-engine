"""Checks the hivemind-clock build's network outputs end to end, for the GPU (TensorRT, FP16) path.

The Mac only builds the ONNX Runtime backend, so this is the check to run once on a GPU node after cluster/build.sh,
before any large run. It compares the engine (TensorRT runs the network in FP16) with Python ONNX Runtime in FP32
on the same positions:

  - value, WDL and the move priors of both boards (spec v3 / audit inputs 8): the engine's ``policy`` command
    prints the raw value, the WDL probabilities and the softmax over the legal moves plus pass of each board the
    team is on turn at; the reference is the FP32 network on Python's planes (``board2planes``, the encoder
    ``tools/planes_parity.py`` proves identical to the engine's), normalised over the same moves. Tolerances: max
    |diff| 0.02 on value, on each WDL probability and on each move prior, and the top-1 move agreeing on at least 99%
    of boards (``--value-tolerance``, ``--prior-tolerance``, ``--min-top1``).
  - think_gain (spec v2): a copy of the network with random weights in the last think_gain layer (a freshly
    widened network has that layer at exactly 0, so every position would print 0.5 and a wrong binding could still
    pass); every position must print ``info string think_gain <pa> <pb>`` after ``go nodes 1 sample``, within 0.03
    in probability of Python's.
  - planes: the engine's sample record (planes quantised to bytes, clipped to [0, 1]) must agree with Python's
    planes to one quantisation step, so the reference really is run on the engine's input.

Positions come from random play with random clocks and elapsed times (a third with our board-A clock within 30 s of
the diagonal seat's, so the spec v3 diagonal plane is not saturated) and a random TimeAdvantage.

    .venv/bin/python cluster/check_clock_engine.py [--positions 64] [--binary hivemind-clock] [--model <onnx>]
        [--diagonal-scale 15]

--model is the FP32 network (default hivemind-clock-v3-fp32.onnx, 44 planes per board; the build decides the
width). The engine loads it too (TensorRT converts it to FP16 itself) unless --engine-model names another file of the
same network, e.g. the -fp16.onnx for an ONNX Runtime build with -DHIVEMIND_ORT_FP16=ON (it gets the same think_gain
randomization). --diagonal-scale must match the build's HIVEMIND_DIAGONAL_SCALE_S. Exit 0 when every check passes.
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

BOARD_CHANNELS_STOCK = 37
TIME_ADVANTAGE_CHANNEL = 31
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
    """(fen_a, fen_b, team, time_advantage, clocks, elapsed) from random play, both boards still live.

    A third of the clocks put the diagonal seat (the team's colour on board B) within 30 s of the team's board-A
    clock, where the diagonal plane tanh((t_ourA - t_oppB) / s) is not saturated."""
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
        if rng.random() < 0.33:
            colour = "white" if team == "T1" else "black"
            clocks[f"B_{colour}"] = max(0, clocks[f"A_{colour}"] + rng.randint(-30000, 30000))
        out.append((fen_a, fen_b, team, rng.random() < 0.5, clocks, elapsed))
    return out


def python_planes(fen_a: str, fen_b: str, team: str, time_advantage: bool, clocks: dict, elapsed: dict,
                  channels: int, diagonal_scale: float) -> np.ndarray:
    """[1, channels, 8, 8] float planes the engine encodes for this position (board2planes, plane 31 set from the
    TimeAdvantage option as the engine does)."""
    import chess
    from hivemind.domain.board import BughouseBoard
    from hivemind.domain.board2planes import board2planes

    per_board = channels // 2
    board = BughouseBoard()
    board.set_fen(f"{fen_a} | {fen_b}")
    side = chess.WHITE if team == "T1" else chess.BLACK
    seats = tuple(clocks[s] for s in ("A_white", "A_black", "B_white", "B_black"))
    kwargs = {}
    if per_board > BOARD_CHANNELS_STOCK:
        kwargs = dict(clocks=seats, clock_planes=True)
        if per_board >= 43:
            kwargs.update(elapsed=(elapsed["A"], elapsed["B"]), elapsed_planes=True)
        if per_board >= 44:
            kwargs.update(diagonal_plane=True, diagonal_scale_s=diagonal_scale)
    planes = board2planes(board, side, **kwargs)
    if planes.shape[0] != channels:
        raise SystemExit(f"Python planes have {planes.shape[0]} channels, the network wants {channels}")
    planes[TIME_ADVANTAGE_CHANNEL] = planes[per_board + TIME_ADVANTAGE_CHANNEL] = float(time_advantage)
    return planes[None].astype(np.float32)


def parse_policy_output(lines: list[str]) -> dict:
    """The engine's ``policy`` printout -> {"value": v, "wdl": (w, d, l), "A": {uci: p} or None, "B": ...}."""
    out: dict = {"value": None, "wdl": None, "A": None, "B": None}
    section = None
    for line in lines:
        text = line.strip()
        if text.startswith("Value:"):
            out["value"] = float(text.split()[1])
        elif text.startswith("WDL:"):
            out["wdl"] = tuple(float(x) for x in text.split()[1:4])
        elif text.startswith("Board A (") or text.startswith("Board B ("):
            section = text[6]
            out[section] = {}
        elif text == "(not our turn)" and section:
            out[section] = None
        elif section and out[section] is not None and ": " in text and line.startswith("  "):
            move, prob = text.rsplit(": ", 1)
            out[section][move] = float(prob)
    return out


def engine_policy(process, fen_a: str, fen_b: str, clocks: dict, elapsed: dict) -> dict:
    """Run the engine's ``policy`` command on this position (Team and TimeAdvantage as last set)."""
    from arena.players.hivemind import position_command

    with process._lock:
        process._send(position_command(fen_a, fen_b, clocks, elapsed))
        process._send("policy")
        lines: list[str] = []
        process._send("isready")
        process._wait_for(lambda l: l == "readyok", collect=lines)
    return parse_policy_output(lines)


def reference_priors(fen: str, board_idx: int, logits: np.ndarray, moves: list[str]) -> dict[str, float] | None:
    """FP32 priors over the engine's printed moves (pass included): softmax of the network's logits at each move's
    policy index, as get_normalized_probability does. None when a move cannot be mapped."""
    import chess
    from hivemind.domain.board import BughouseBoard
    from hivemind.domain.move2planes import make_label_index, policy_label

    global _LABEL_INDEX
    if _LABEL_INDEX is None:
        _LABEL_INDEX = make_label_index()
    single = BughouseBoard()
    single.set_fen(fen)
    board = single.boards[board_idx]
    indices = []
    for uci in moves:
        if uci == "pass":
            indices.append(_LABEL_INDEX["pass"])
            continue
        text = uci[0].upper() + uci[1:] if len(uci) > 1 and uci[1] == "@" else uci
        try:
            move = board.parse_uci(text)
        except ValueError:
            return None
        label = policy_label(board, move)
        if label not in _LABEL_INDEX:
            return None
        indices.append(_LABEL_INDEX[label])
    chosen = logits[indices].astype(np.float64)
    chosen = np.exp(chosen - chosen.max())
    chosen /= chosen.sum()
    return dict(zip(moves, chosen.tolist()))


_LABEL_INDEX = None


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x.astype(np.float64) - np.max(x))
    return e / e.sum()


def compare_outputs(engine: dict, ref: dict, fen_pair: tuple[str, str]) -> dict:
    """Max |diff| of value, WDL and priors, top-1 agreement per board, and unmapped boards."""
    result = {"value": abs(engine["value"] - float(ref["value"][0][0])), "prior": 0.0, "top1": [], "unmapped": 0}
    loss, draw, win = _softmax(ref["wdl_out"][0])
    result["wdl"] = max(abs(a - b) for a, b in zip(engine["wdl"], (win, draw, loss)))
    for board_idx, key, logits in ((0, "A", ref["pi_a"][0]), (1, "B", ref["pi_b"][0])):
        priors = engine[key]
        if not priors:
            continue
        fen = f"{fen_pair[0]} | {fen_pair[1]}"
        expected = reference_priors(fen, board_idx, logits, list(priors))
        if expected is None:
            result["unmapped"] += 1
            continue
        result["prior"] = max(result["prior"], max(abs(priors[m] - expected[m]) for m in priors))
        result["top1"].append(max(priors, key=priors.get) == max(expected, key=expected.get))
    return result


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--binary", default="hivemind-clock")
    ap.add_argument("--model", default=os.path.join(ENGINE_DIR, "models", "hivemind-clock-v3-fp32.onnx"))
    ap.add_argument("--engine-model", default=None,
                    help="network file the engine loads (default: --model); same network, e.g. its FP16 ONNX")
    ap.add_argument("--positions", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tolerance", type=float, default=0.03,
                    help="think_gain: max |p_engine - p_ort| (default 0.03)")
    ap.add_argument("--value-tolerance", type=float, default=0.02,
                    help="max |diff| of value and of each WDL probability (default 0.02)")
    ap.add_argument("--prior-tolerance", type=float, default=0.02,
                    help="max |diff| of any move prior (default 0.02)")
    ap.add_argument("--min-top1", type=float, default=0.99,
                    help="minimum share of boards whose top-1 move agrees (default 0.99)")
    ap.add_argument("--diagonal-scale", type=float, default=15.0,
                    help="s of the diagonal plane, as the build's HIVEMIND_DIAGONAL_SCALE_S (default 15)")
    ap.add_argument("--timeout", type=float, default=1800.0, help="engine start wait, s (TensorRT engine build)")
    args = ap.parse_args(argv)

    import onnxruntime as ort

    binary = resolve_binary(args.binary)
    with tempfile.TemporaryDirectory() as tmp:
        model = os.path.join(tmp, "think-check-fp32.onnx")
        randomized_model(args.model, model, args.seed)
        engine_model = model
        if args.engine_model:
            engine_model = os.path.join(tmp, "think-check-engine.onnx")
            randomized_model(args.engine_model, engine_model, args.seed)
        session = ort.InferenceSession(model, providers=["CPUExecutionProvider"])
        names = [o.name for o in session.get_outputs()]
        input_width = session.get_inputs()[0].shape[1]
        reference = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
        print(f"binary {binary}\nmodel  {args.model} (FP32 reference; the engine copy has its last think_gain "
              f"layer randomized, seed {args.seed}), {input_width} input channels")

        if args.engine_model:
            print(f"engine loads {args.engine_model} (same think_gain randomization)")
        process = HivemindProcess(binary, engine_model, timeout_s=args.timeout)
        missing = failed = no_root = 0
        worst = 0.0
        diffs = []
        worst_plane = 0.0
        out_diffs = {"value": [], "wdl": [], "prior": []}
        top1: list[bool] = []
        unmapped = 0
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
                view = view_sample(res.sample, channels=input_width)
                if view.channels != input_width:
                    raise SystemExit(f"engine planes have {view.channels} channels, network wants {input_width}")
                sample_planes = (np.frombuffer(view.planes, dtype=np.uint8).astype(np.float32) / 255.0)
                sample_planes = sample_planes.reshape(1, view.channels, 8, 8)
                planes = python_planes(fen_a, fen_b, team, adv, clocks, elapsed, input_width, args.diagonal_scale)
                worst_plane = max(worst_plane, float(np.max(np.abs(np.clip(planes, 0.0, 1.0) - sample_planes))))
                out = dict(zip(names, session.run(None, {"data": planes})))
                ref = [1.0 / (1.0 + math.exp(-float(x))) for x in out["think_gain"][0]]
                d = max(abs(a - b) for a, b in zip(res.think_gain, ref))
                diffs.append(d)
                worst = max(worst, d)

                # value / WDL / priors: the engine's policy printout vs FP32 ORT on the same planes.
                printed = engine_policy(process, fen_a, fen_b, clocks, elapsed)
                if printed["value"] is None or printed["wdl"] is None:
                    raise SystemExit("the engine's 'policy' command printed no Value/WDL line")
                fp32 = dict(zip([o.name for o in reference.get_outputs()], reference.run(None, {"data": planes})))
                cmp = compare_outputs(printed, fp32, (fen_a, fen_b))
                for key in out_diffs:
                    out_diffs[key].append(cmp[key])
                top1.extend(cmp["top1"])
                unmapped += cmp["unmapped"]
        finally:
            process.close()

    n = len(diffs)
    spread = f"mean {np.mean(diffs):.4f}" if n else "no comparisons"
    think_ok = n == args.positions and missing == 0 and failed == 0 and worst <= args.tolerance
    print(f"positions {args.positions}: compared {n}, no think_gain line {missing}, 'think_gain failed' {failed} "
          f"(skipped {no_root} with no root)")
    print(f"  think_gain  max |p_engine - p_ort| {worst:.4f} ({spread}), tolerance {args.tolerance}: "
          f"{'PASS' if think_ok else 'FAIL'}")
    plane_ok = worst_plane <= 0.5 / 255.0 + 1e-6
    print(f"  planes      engine sample vs Python (clipped to [0, 1]) max |diff| {worst_plane:.5f}, "
          f"tolerance {0.5 / 255.0:.5f} (half a byte step): {'PASS' if plane_ok else 'FAIL'}")
    outputs_ok = bool(n)
    for key, tol in (("value", args.value_tolerance), ("wdl", args.value_tolerance),
                     ("prior", args.prior_tolerance)):
        values = out_diffs[key]
        worst_key = max(values) if values else float("nan")
        ok_key = bool(values) and worst_key <= tol
        outputs_ok &= ok_key
        mean = f"mean {np.mean(values):.4f}" if values else "no comparisons"
        print(f"  {key:<11} max |engine - fp32 ORT| {worst_key:.4f} ({mean}), tolerance {tol}: "
              f"{'PASS' if ok_key else 'FAIL'}")
    share = sum(top1) / len(top1) if top1 else 0.0
    top1_ok = bool(top1) and share >= args.min_top1 and unmapped == 0
    outputs_ok &= top1_ok
    print(f"  top-1       {sum(top1)}/{len(top1)} boards agree ({share:.3f}), minimum {args.min_top1}; "
          f"boards with a move the reference could not map: {unmapped}: {'PASS' if top1_ok else 'FAIL'}")
    ok = think_ok and plane_ok and outputs_ok
    print(f"overall: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
