#!/usr/bin/env python3
"""Writes one tuning round of candidates for cluster/tune_match.sh: each candidate is our engine (the space file's
base options plus parameter values) against upstream-it13 at 120 s. Standard library only. Read docs/tuning.md.

    # one-at-a-time: the baseline plus each parameter one step up and one step down (bools flipped, choices swapped)
    python3 cluster/tune_round.py oat --name r1 --params clock_k,CPUCTInitPermille --mult 2
    # chosen values, one candidate per value, plus the baseline
    python3 cluster/tune_round.py grid --name g1 --set clock_k=55,220,440,1000 --set hybrid_margin_permille=0,40
    # SPSA: K antithetic pairs (theta +- c*step*Delta); with --prev, first a step from that SPSA round's results
    python3 cluster/tune_round.py spsa --name s1 --k 8
    python3 cluster/tune_round.py spsa --name s2 --prev runs/tune/s1
    # after an OAT round: one candidate with every side that beat the baseline, plus the baseline, longer
    python3 cluster/tune_round.py combine --name r2 --prev runs/tune/r1 --parts 8

Writes <runs>/<name>/round.json (mode, baseline, each candidate's parameters and full options, the schedule),
<runs>/<name>/candidates/<cand>.json (what tune_match.sh plays), <runs>/<name>/space.json (the space this round was
written from, with "value" updated by combine/spsa) and <runs>/<name>/launch.sh (one tune_match.sh line per candidate),
and prints the launch lines and the cost: games x GPU_MIN_PER_GAME (1.6 GPU-minutes a game at 120 s, the long test's
rate: one 8 h job plays 250-300 games). Nothing is submitted.

Candidate options: the space's "base" (the hybrid's switches), then each parameter: where "player" -> the player
option of the same name, where "engine" -> engine_options[name] (only when it differs from the engine's default, so
the baseline sends exactly the hybrid's options). Charge (the space's "charge", or --charge): "fixed" sends both arms'
ms_per_node from the space file (no per-job measurement noise between candidates); "measure" leaves them to MEASURE=1
as the final match does. Parameters marked "speed" are left out of oat and spsa unless --include-speed, and need
--charge measure (docs/tuning.md).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import shlex
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tune_status  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GPU_MIN_PER_GAME = 1.6
GAMES_PER_8H_JOB = 300  # 120 s, WORKERS=3, RTX 6000 (runs/wave2c/r4/long: 238-296 a part)
GPU_CAP = 32  # qos inferno: gres/gpu=32 per user, shared with training jobs


# ----------------------------------------------------------------------------------------------- the space
def load_space(path: str) -> dict:
    with open(path) as f:
        space = json.load(f)
    names = [p["name"] for p in space["params"]]
    if len(set(names)) != len(names):
        raise ValueError(f"{path}: duplicate parameter names")
    for p in space["params"]:
        check_param(p)
    return space


def check_param(p: dict) -> None:
    t = p.get("type")
    if p.get("where") not in ("player", "engine"):
        raise ValueError(f"{p['name']}: where must be player or engine")
    if t in ("int", "float"):
        if not p["min"] <= p["value"] <= p["max"] or p["step"] <= 0:
            raise ValueError(f"{p['name']}: need min <= value <= max and step > 0")
        if p.get("scale", "linear") not in ("linear", "log2") or (p.get("scale") == "log2" and p["min"] <= 0):
            raise ValueError(f"{p['name']}: scale must be linear or log2 (log2 needs min > 0)")
    elif t == "bool":
        if not isinstance(p["value"], bool):
            raise ValueError(f"{p['name']}: bool value must be true or false")
    elif t == "choice":
        if p["value"] not in p["choices"]:
            raise ValueError(f"{p['name']}: value not among choices")
    else:
        raise ValueError(f"{p['name']}: unknown type {t!r}")


def numeric(p: dict) -> bool:
    return p["type"] in ("int", "float")


def shift(p: dict, v: float, steps: float) -> float:
    """v moved by this many steps: + steps x step, or for "scale": "log2" x 2^(steps x step). Not clamped."""
    if p.get("scale") == "log2":
        return v * 2.0 ** (steps * p["step"])
    return v + steps * p["step"]


def in_steps(p: dict, a: float, b: float) -> float:
    """How many steps a is above b (the inverse of shift)."""
    if p.get("scale") == "log2":
        return math.log2(a / b) / p["step"]
    return (a - b) / p["step"]


def clamp(p: dict, v: float) -> float:
    return min(max(v, p["min"]), p["max"])


def snap(p: dict, v: float):
    """v clamped to the range, rounded to an int for int parameters and to 1e-6 for floats."""
    v = clamp(p, v)
    return int(round(v)) if p["type"] == "int" else round(v, 6)


def candidate_kw(space: dict, values: dict, charge: str) -> dict:
    """{"contender_kw", "opponent_kw", "changed"}: the arena options of one candidate."""
    ckw = dict(space.get("base") or {})
    eo = {}
    changed = {}
    for p in space["params"]:
        v = values.get(p["name"], p["value"])
        if v != p["value"]:
            changed[p["name"]] = v
        if p["where"] == "player":
            ckw[p["name"]] = v
        elif v != p.get("default", p["value"]):
            # An engine option is sent only when it differs from the engine's own default ("default", else the
            # space's value is that default), so the baseline sends exactly the hybrid's options.
            eo[p["name"]] = v
    if eo:
        ckw["engine_options"] = eo
    okw = {}
    if charge == "fixed":
        ch = space["charge"]
        ckw["ms_per_node"] = float(ch["contender_ms_per_node"])
        okw["ms_per_node"] = float(ch["opponent_ms_per_node"])
    elif charge != "measure":
        raise ValueError(f"charge must be fixed or measure, got {charge!r}")
    return {"contender_kw": ckw, "opponent_kw": okw, "changed": changed}


def label(name: str, v) -> str:
    s = str(v).lower() if isinstance(v, bool) else format(v, "g") if isinstance(v, (int, float)) else str(v)
    return re.sub(r"[^\w.-]", "_", f"{name}={s}").replace("=", "_")


# ----------------------------------------------------------------------------------------------- round kinds
def oat_values(space: dict, names: list[str], mult: float) -> list[tuple[str, dict]]:
    out = [("base", {})]
    for p in space["params"]:
        if p["name"] not in names:
            continue
        if numeric(p):
            for sign in (+1, -1):
                v = snap(p, shift(p, p["value"], sign * mult))
                if v != p["value"]:
                    out.append((label(p["name"], v), {p["name"]: v}))
        elif p["type"] == "bool":
            out.append((label(p["name"], not p["value"]), {p["name"]: not p["value"]}))
        else:
            for c in p["choices"]:
                if c != p["value"]:
                    out.append((label(p["name"], c), {p["name"]: c}))
    return out


def spsa_values(space: dict, names: list[str], theta: dict, k: int, c: float, rng: random.Random):
    """K antithetic pairs around theta: plus = theta + c*step*Delta, minus = theta - c*step*Delta, Delta in {-1,+1}.
    Returns ([(cand, values)], [{"pair", "delta"}]). Values are snapped to the range (int parameters rounded); the
    gradient uses the values actually played, so rounding and clamping do not bias it."""
    cands, pairs = [], []
    params = [p for p in space["params"] if p["name"] in names]
    for j in range(k):
        delta = {p["name"]: rng.choice((-1, 1)) for p in params}
        plus = {p["name"]: snap(p, shift(p, theta[p["name"]], c * delta[p["name"]])) for p in params}
        minus = {p["name"]: snap(p, shift(p, theta[p["name"]], -c * delta[p["name"]])) for p in params}
        cands += [(f"s{j + 1:02d}p", plus), (f"s{j + 1:02d}m", minus)]
        pairs.append({"pair": j + 1, "plus": f"s{j + 1:02d}p", "minus": f"s{j + 1:02d}m", "delta": delta})
    return cands, pairs


def spsa_gradient(prev_dir: str, prev: dict, space: dict) -> tuple[dict, list[dict]]:
    """Mean over the previous round's pairs of (score(plus) - score(minus)) / (v_plus - v_minus in steps), per
    parameter, with the scores paired on the game pairs both finished. Returns ({name: score per step}, per pair)."""
    names = prev["spsa"]["params"]
    params = {p["name"]: p for p in space["params"]}
    by_name = {c["name"]: c for c in prev["candidates"]}
    grads = {n: [] for n in names}
    detail = []
    for pr in prev["spsa"]["pairs"]:
        a = tune_status.load_candidate(os.path.join(prev_dir, pr["plus"]))
        b = tune_status.load_candidate(os.path.join(prev_dir, pr["minus"]))
        shared = sorted(set(a["pairs"]) & set(b["pairs"]))
        if not shared:
            detail.append({"pair": pr["pair"], "shared_pairs": 0})
            continue
        ds = sum(a["pairs"][k] - b["pairs"][k] for k in shared) / (2.0 * len(shared))
        detail.append({"pair": pr["pair"], "shared_pairs": len(shared), "dscore": ds})
        vp, vm = by_name[pr["plus"]]["values"], by_name[pr["minus"]]["values"]
        for n in names:
            dv = in_steps(params[n], vp[n], vm[n])
            if dv != 0:
                grads[n].append(ds / dv)
    return {n: (sum(g) / len(g) if g else 0.0) for n, g in grads.items()}, detail


# ----------------------------------------------------------------------------------------------- writing
def write_round(out: str, name: str, mode: str, space: dict, cands: list[tuple[str, dict]], args, extra: dict) -> dict:
    if os.path.exists(os.path.join(out, "round.json")) and not args.force:
        raise SystemExit(f"{out}/round.json exists: pick another --name (or --force to rewrite a round never launched)")
    os.makedirs(os.path.join(out, "candidates"), exist_ok=True)
    charge = args.charge or space.get("charge", {}).get("mode", "measure")
    rows, lines = [], []
    for cand, values in cands:
        kw = candidate_kw(space, values, charge)
        full = {p["name"]: values.get(p["name"], p["value"]) for p in space["params"]}
        row = {"name": cand, "round": name, "values": full, "changed": kw["changed"],
               "contender_kw": kw["contender_kw"], "opponent_kw": kw["opponent_kw"], "charge": charge}
        path = os.path.join(out, "candidates", f"{cand}.json")
        with open(path, "w") as f:
            json.dump(row, f, indent=1)
        rows.append(row)
        env = (f"NAME={name} CAND={shlex.quote(os.path.relpath(path, REPO))} PARTS={args.parts} "
               f"PAIRS_PER_PART={args.pairs_per_part} SEED_BASE={args.seed_base} TIME={args.time}")
        if os.path.abspath(args.runs) != os.path.join(REPO, "runs", "tune"):
            env += f" TUNE_RUNS={shlex.quote(os.path.relpath(args.runs, REPO))}"
        lines.append(f"{env} bash cluster/tune_match.sh")
    rnd = {"name": name, "mode": mode, "baseline": "base" if any(c == "base" for c, _ in cands) else None,
           "charge": charge, "schedule": {"parts": args.parts, "pairs_per_part": args.pairs_per_part,
                                          "seed_base": args.seed_base, "base_ms": 120000, "time": args.time},
           "space_file": os.path.relpath(args.space, REPO) if os.path.isabs(args.space) else args.space,
           "candidates": [{"name": r["name"], "params": r["changed"], "changed": r["changed"],
                           "values": r["values"]} for r in rows], **extra}
    with open(os.path.join(out, "round.json"), "w") as f:
        json.dump(rnd, f, indent=1)
    with open(os.path.join(out, "space.json"), "w") as f:
        json.dump(space, f, indent=1)
    with open(os.path.join(out, "launch.sh"), "w") as f:
        f.write("#!/bin/bash\n# Round %s (%s): run on PACE from the repo root.\nset -euo pipefail\n" % (name, mode))
        f.write("\n".join(lines) + "\n")
    games = 2 * args.pairs_per_part * args.parts
    jobs = len(cands) * args.parts
    hours = time_hours(args.time)
    plays = min(games, int(args.parts * GAMES_PER_8H_JOB * hours / 8))
    print(f"round {name} ({mode}): {len(cands)} candidates in {out}, charge {charge}")
    for line in lines:
        print(line)
    print(f"# cost: {jobs} jobs of {args.parts} parts x {args.pairs_per_part} pairs; about {plays} games a candidate "
          f"in {args.time} ({games} scheduled), {len(cands) * plays} games, "
          f"~{len(cands) * plays * GPU_MIN_PER_GAME / 60:.0f} GPU-hours "
          f"(at most {jobs * hours:.0f} by the time limits)")
    print(f"# {jobs} GPUs at once against the per-user cap of {GPU_CAP} (shared with training): "
          + ("fits in one wave" if jobs <= GPU_CAP - 4 else
             f"about {math.ceil(jobs / (GPU_CAP - 4))} waves of {args.time}"))
    print(f"# status: python3 cluster/tune_status.py {os.path.relpath(out, REPO)}")
    return rnd


def time_hours(t: str) -> float:
    parts = [int(x) for x in t.split(":")]
    h, m, s = ([0] * (3 - len(parts)) + parts)[-3:]
    return h + m / 60 + s / 3600


def parse_value(p: dict, raw: str):
    """A --values entry in the parameter's type, checked against its range or choices."""
    t = p["type"]
    if t == "bool":
        if raw.lower() not in ("true", "false"):
            raise SystemExit(f"{p['name']}: {raw!r} is not true/false")
        return raw.lower() == "true"
    if t == "choice":
        if raw not in p["choices"]:
            raise SystemExit(f"{p['name']}: {raw!r} not in {p['choices']}")
        return raw
    v = float(raw)
    if not p["min"] <= v <= p["max"]:
        raise SystemExit(f"{p['name']}: {raw} outside {p['min']}..{p['max']}")
    return int(round(v)) if t == "int" else v


def pick_names(space: dict, args, want) -> list[str]:
    if args.params:
        names = [n.strip() for n in args.params.split(",") if n.strip()]
        known = {p["name"] for p in space["params"]}
        bad = [n for n in names if n not in known]
        if bad:
            raise SystemExit(f"unknown parameter(s) {bad}")
        return names
    return [p["name"] for p in space["params"] if want(p) and (args.include_speed or not p.get("speed"))]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)
    for mode in ("oat", "grid", "spsa", "combine"):
        p = sub.add_parser(mode)
        p.add_argument("--name", required=True, help="round name: folder <runs>/<name>, job names tune_<name>_*")
        p.add_argument("--space", default=None, help="space file (default cluster/tune_space.json, or --prev's)")
        p.add_argument("--runs", default=os.path.join(REPO, "runs", "tune"))
        p.add_argument("--params", default=None, help="comma-separated parameter names (default: all non-speed)")
        p.add_argument("--include-speed", action="store_true")
        p.add_argument("--charge", choices=("fixed", "measure"), default=None)
        p.add_argument("--parts", type=int, default={"oat": 2, "grid": 2, "spsa": 1, "combine": 6}[mode])
        p.add_argument("--pairs-per-part", type=int, default=150)
        p.add_argument("--seed-base", type=int, default=5000)
        p.add_argument("--time", default="08:00:00")
        p.add_argument("--force", action="store_true")
        if mode == "oat":
            p.add_argument("--mult", type=float, default=1.0, help="perturbation in steps (default 1)")
        if mode == "grid":
            p.add_argument("--set", action="append", required=True, metavar="NAME=V1,V2,...",
                           help="a parameter and the values to play, one candidate each (repeatable)")
        if mode == "spsa":
            p.add_argument("--prev", default=None, help="previous SPSA round folder: step from its results first")
            p.add_argument("--k", type=int, default=6, help="antithetic pairs (2k candidates)")
            p.add_argument("--c", type=float, default=2.0, help="perturbation in steps")
            p.add_argument("--gain", type=float, default=30.0,
                           help="step taken = gain x gradient (score per step), in steps")
            p.add_argument("--max-move", type=float, default=2.0, help="largest move per round, in steps")
            p.add_argument("--seed", type=int, default=None, help="Delta draws (default: from the round name)")
            p.add_argument("--with-center", action="store_true", help="also play theta itself as 'base'")
        if mode == "combine":
            p.add_argument("--prev", required=True, help="the OAT round folder")
            p.add_argument("--rule", choices=("positive", "ci"), default="positive",
                           help="take a side when its dElo > 0 and LLR > 0 (positive) or its 95%% range is above 0")
    args = ap.parse_args(argv)
    if args.space is None:
        prev_space = os.path.join(args.prev, "space.json") if getattr(args, "prev", None) else None
        args.space = prev_space if prev_space and os.path.exists(prev_space) else os.path.join(REPO, "cluster",
                                                                                               "tune_space.json")
    space = load_space(args.space)
    out = os.path.join(args.runs, args.name)
    if args.pairs_per_part < 1 or args.parts < 1:
        raise SystemExit("--parts and --pairs-per-part must be >= 1")

    if args.mode == "oat":
        names = pick_names(space, args, lambda p: True)
        write_round(out, args.name, "oat", space, oat_values(space, names, args.mult), args, {"mult": args.mult})
        return 0

    if args.mode == "grid":
        cands = [("base", {})]
        for item in args.set:
            pname, _, raw_values = item.partition("=")
            p = next((q for q in space["params"] if q["name"] == pname), None)
            if p is None or not raw_values:
                raise SystemExit(f"--set {item!r}: want NAME=V1,V2,... with NAME in the space")
            for raw in (x.strip() for x in raw_values.split(",") if x.strip()):
                v = parse_value(p, raw)
                if v != p["value"] and (label(p["name"], v), {p["name"]: v}) not in cands:
                    cands.append((label(p["name"], v), {p["name"]: v}))
        write_round(out, args.name, "grid", space, cands, args, {"set": args.set})
        return 0

    if args.mode == "combine":
        with open(os.path.join(args.prev, "round.json")) as f:
            prev = json.load(f)
        st = tune_status.status(args.prev)
        best: dict = {}
        for row in st["candidates"]:
            vb = row.get("vs_baseline")
            if not vb or len(row["params"]) != 1:
                continue
            (pname, v), = row["params"].items()
            ok = vb["delo_ci95"][0] > 0 if args.rule == "ci" else (vb["delo"] > 0 and vb["llr"] > 0)
            if ok and (pname not in best or vb["delo"] > best[pname][1]):
                best[pname] = (v, vb["delo"])
        if not best:
            print(f"no single-parameter candidate of {args.prev} beat the baseline by rule {args.rule}: nothing to "
                  "combine")
            return 1
        values = {n: v for n, (v, _) in best.items()}
        new_space = json.loads(json.dumps(space))
        for p in new_space["params"]:
            if p["name"] in values:
                p.setdefault("default", p["value"]) if p["where"] == "engine" else None
                p["value"] = values[p["name"]]
        # The combined candidate is played against the old baseline ("base"); space_next.json holds its values as
        # the next round's baseline.
        cands = [("base", {}), ("combined", values)]
        write_round(out, args.name, "combine", space, cands, args,
                    {"prev": os.path.relpath(args.prev, REPO), "taken": {n: {"value": v, "delo": d}
                                                                         for n, (v, d) in best.items()}})
        with open(os.path.join(out, "space_next.json"), "w") as f:
            json.dump(new_space, f, indent=1)
        print(f"# taken: {values}; the next round's space (values updated): {os.path.relpath(out, REPO)}/space_next.json")
        return 0

    # spsa
    names = pick_names(space, args, numeric)
    bad = [n for n in names if not numeric(next(p for p in space["params"] if p["name"] == n))]
    if bad:
        raise SystemExit(f"SPSA takes numeric parameters only, not {bad}")
    theta = {p["name"]: p["value"] for p in space["params"]}
    extra: dict = {}
    if args.prev:
        with open(os.path.join(args.prev, "round.json")) as f:
            prev = json.load(f)
        if prev.get("mode") != "spsa":
            raise SystemExit(f"{args.prev} is not an SPSA round")
        theta.update(prev["spsa"]["theta"])
        grad, detail = spsa_gradient(args.prev, prev, space)
        moves = {}
        for p in space["params"]:
            n = p["name"]
            if n not in grad:
                continue
            move = max(-args.max_move, min(args.max_move, args.gain * grad[n]))
            theta[n] = clamp(p, shift(p, theta[n], move))
            moves[n] = round(move, 3)
        names = prev["spsa"]["params"] if not args.params else names
        extra["step_from"] = {"prev": os.path.relpath(args.prev, REPO), "gradient_score_per_step": grad,
                              "moves_in_steps": moves, "pairs": detail}
        print("# step from " + args.prev + ": " + ", ".join(f"{n} {moves[n]:+.2f} steps -> {theta[n]:g}"
                                                            for n in moves))
    space = json.loads(json.dumps(space))
    for p in space["params"]:  # the space this round plays around: theta snapped as its values
        if p["name"] in names:
            p.setdefault("default", p["value"]) if p["where"] == "engine" else None
            p["value"] = snap(p, theta[p["name"]])
    seed = args.seed if args.seed is not None else int.from_bytes(args.name.encode()[:8].ljust(8, b"\0"), "big")
    rng = random.Random(seed)
    cands, pairs = spsa_values(space, names, theta, args.k, args.c, rng)
    if args.with_center:
        cands = [("base", {})] + cands
    extra["spsa"] = {"params": names, "theta": {n: theta[n] for n in names}, "k": args.k, "c": args.c,
                     "gain": args.gain, "max_move": args.max_move, "seed": seed, "pairs": pairs}
    write_round(out, args.name, "spsa", space, cands, args, extra)
    return 0


if __name__ == "__main__":
    sys.exit(main())
