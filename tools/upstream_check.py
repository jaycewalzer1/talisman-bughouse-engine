"""Checks for the upstream-v3 arena arm (arena/players/upstream.py). Build first: sh tools/build_upstream.sh

  .venv/bin/python tools/upstream_check.py parity [--positions 20] [--coreml off|default]
      The arm's raw policy (the engine's own ``policy`` command, under exactly the time options a search sends)
      against an independent evaluation: upstream's Python plane encoder (third_party/upstream-hivemind/src, run in a
      subprocess so it cannot clash with our fork's ``hivemind`` package) + the two clock planes from the model
      card's formulas + ONNX Runtime CPU on the same FP32 network. Pass: same action set on every on-turn board and
      max |p_engine - p_ref| below --tol (1e-4; measured ~2e-6 on both the CPU and the default Core ML provider).
      ``--engine wave2`` checks the wave 2 engine (third_party/wave2-hivemind) the same way, with the wave 2 flags
      set and the time sent as the four seat clocks and per-seat paces (``SeatClocksMs``/``SeatPaceMs``, chosen so
      that the lead and speeds equal the reference's): this checks the engine's seat -> plane mapping and signs.
      Prints PASS or FAIL on its last line.

  .venv/bin/python tools/upstream_check.py smoke [--games 6] [--base-ms 60000] [--out runs/upstream/smoke]
      it04 (stock binary, centralized, fixed nodes) vs upstream-v3 in single in-process games, paired with swapped
      teams on the bughouse opening set, nodes-as-time with each arm's own ms_per_node.

  .venv/bin/python tools/upstream_check.py fairy [--games 10] [--move-s 1] [--base-ms 180000]
      it04 vs Fairy-Stockfish (arena/players/fairy.py) at matched time per move: Fairy searches a fixed --move-s
      wall seconds (CECP st) and is charged exactly that; it04 gets --move-s / 5.5 ms nodes (its MS_PER_NODE), so
      both spend the same arena clock per move on this Mac's CPU. Paired, swapped teams, single process.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import subprocess
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

UPSTREAM_SRC = os.path.join(ROOT, "third_party", "upstream-hivemind", "src")
CASTLE_ENGINE = {"e1g1": "e1h1", "e1c1": "e1a1", "e8g8": "e8h8", "e8c8": "e8a8"}


# ------------------------------------------------------------------------------------------------ reference side

def _reference_main() -> int:
    """Subprocess: read jobs (JSON lines) on stdin, print priors per board (JSON lines). Upstream src first on path."""
    sys.path.insert(0, UPSTREAM_SRC)
    import chess
    import numpy as np
    import onnxruntime as ort

    from hivemind.domain.board import BughouseBoard
    from hivemind.domain.board2planes import board2planes
    from hivemind.domain.move2planes import make_map, mirrorMove

    assert "upstream-hivemind" in sys.modules["hivemind"].__file__, sys.modules["hivemind"].__file__
    labels = make_map()
    index: dict[str, int] = {}
    for i, lab in enumerate(labels):
        index.setdefault(lab, i)
    pass_index = index["pass"]
    jobs = [json.loads(l) for l in sys.stdin if l.strip()]
    sess = ort.InferenceSession(jobs[0]["model"], providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    for job in jobs:
        team = chess.WHITE if job["team"] == "T1" else chess.BLACK
        bb = BughouseBoard()
        bb.set_fen(f"{job['fen_a']} | {job['fen_b']}")
        p74 = board2planes(bb, team, flip=False)
        lead, tm, om = job["lead"], job["team_ms"], job["opp_ms"]
        if lead == 0:
            lead_v = 0.0
        else:
            ahead_ms = tm if lead > 0 else om
            lead_v = min(1.0, math.log2(1.0 + abs(lead) / ahead_ms) / 10.0) * (1 if lead > 0 else -1)
        speed_v = max(-1.0, min(1.0, math.log2(tm / om) / 5.0))
        sit = 1.0 if lead > 0 else 0.0
        p74[31] = sit
        p74[37 + 31] = sit
        extra = np.zeros((2, 8, 8), np.float32)
        extra[0], extra[1] = lead_v, speed_v
        planes = np.concatenate([p74[:37], extra, p74[37:], extra])[None].astype(np.float32)
        out = sess.run(None, {name: planes})
        res = {"value": float(np.asarray(out[0]).reshape(-1)[0])}
        for k, board_idx in (("A", 0), ("B", 1)):
            cb = bb.boards[board_idx]
            ours = team if board_idx == 0 else (not team)
            if cb.turn != ours:
                res[k] = {}
                continue
            logits = np.asarray(out[1 + board_idx][0], np.float64)
            names, idx = ["pass"], [pass_index]
            for mv in cb.legal_moves:
                u = mv.uci()
                if cb.is_castling(mv):
                    u = CASTLE_ENGINE.get(u, u)  # engine encodes castling king -> rook square
                m = chess.Move.from_uci(u)
                if cb.turn == chess.BLACK:
                    m = mirrorMove(m)
                lab = m.uci()
                if len(lab) == 5 and lab[-1] != "n":
                    lab = lab[:-1]
                names.append(mv.uci())
                idx.append(index[lab])
            x = logits[idx]
            e = np.exp(x - x.max())
            res[k] = dict(zip(names, (e / e.sum()).tolist()))
        print(json.dumps(res), flush=True)
    return 0


def reference_policies(jobs: list[dict], model: str) -> list[dict]:
    payload = "\n".join(json.dumps({**j, "model": model}) for j in jobs) + "\n"
    out = subprocess.run([sys.executable, "-I", os.path.abspath(__file__), "_reference"], input=payload,
                         capture_output=True, text=True, check=True, cwd=ROOT)
    return [json.loads(l) for l in out.stdout.splitlines() if l.strip()]


# ------------------------------------------------------------------------------------------------ positions

def random_positions(n: int, seed: int) -> list[dict]:
    """Bughouse-like positions: random walks with random pocket gifts, plus a few fixed ones (castling both ways,
    promotions, drops); random clocks and move times. Our team is on turn on at least one board."""
    import chess
    from chess.variant import CrazyhouseBoard

    rng = random.Random(seed)
    fixed = [
        "r3k2r/pppqbppp/2n1bn2/3pp3/3PP3/2N1BN2/PPPQBPPP/R3K2R[] w KQkq - 0 10",
        "r3k2r/pppqbppp/2n1bn2/3pp3/3PP3/2N1BN2/PPPQBPPP/R3K2R[] b KQkq - 0 10",
        "1n2k3/P1P5/8/8/8/8/1p1p4/4K1N1[] w - - 0 40",
        "r1bqk2r/pppp1ppp/2n2n2/4p3/1b2P3/2NP1N2/PPP2PPP/R1BQKB1R[PNBRQpnbrq] w KQkq - 0 6",
        "r1bqk2r/pppp1ppp/2n2n2/4p3/1b2P3/2NP1N2/PPP2PPP/R1BQKB1R[PNBRQpnbrq] b KQkq - 0 6",
    ]

    def walk() -> str:
        b = CrazyhouseBoard()
        for _ in range(rng.randint(0, 40)):
            moves = list(b.legal_moves)
            if not moves:
                break
            b.push(rng.choice(moves))
            if rng.random() < 0.2:
                b.pockets[rng.choice([chess.WHITE, chess.BLACK])].add(rng.choice([1, 2, 3, 4, 5]))
        if b.is_game_over():
            return walk()
        return b.fen()

    out = []
    while len(out) < n:
        i = len(out)
        fa = fixed[i] if i < len(fixed) else walk()
        fb = walk()
        team = rng.choice(["T1", "T2"])
        ta = CrazyhouseBoard(fa).turn == (team == "T1")
        tb = CrazyhouseBoard(fb).turn == (team != "T1")
        if not (ta or tb):
            team = "T2" if team == "T1" else "T1"
        lead = rng.choice([0, 1, -1]) * int(math.exp(rng.uniform(math.log(50), math.log(20000))))
        out.append({"fen_a": fa, "fen_b": fb, "team": team, "lead": lead,
                    "team_ms": int(math.exp(rng.uniform(math.log(80), math.log(3000)))),
                    "opp_ms": int(math.exp(rng.uniform(math.log(80), math.log(3000))))})
    return out


def _engine_to_ref(board_fen: str, priors: dict[str, float]) -> dict[str, float]:
    """Engine move strings -> python-chess UCI (castling king->rook to king->target)."""
    from chess.variant import CrazyhouseBoard

    b = CrazyhouseBoard(board_fen)
    legal = {m.uci() for m in b.legal_moves}
    back = {v: k for k, v in CASTLE_ENGINE.items()}
    out = {}
    for mv, p in priors.items():
        if mv != "pass" and mv not in legal and back.get(mv) in legal:
            mv = back[mv]
        out[mv] = p
    return out


def seat_time_options(team: str, lead: int, team_ms: int, opp_ms: int, base: int = 60_000) -> dict:
    """Wave 2 seat options that encode ``lead`` (``team``'s, upstream's TimeLeadMs) and the two teams' move times:
    T1's own lead t1 = ((Aw - Bw) + (Bb - Ab)) / 2, so Aw = Bb = base + t1, Ab = Bw = base; paces per team."""
    from arena.interfaces import SEATS, TEAM_OF
    from arena.players.upstream import seat_values

    t1 = lead if team == "T1" else -lead
    clocks = {"A_white": base + t1, "A_black": base, "B_white": base, "B_black": base + t1}
    if min(clocks.values()) < 0:
        clocks = {s: v - min(clocks.values()) for s, v in clocks.items()}
    pace = {s: team_ms if TEAM_OF[s] == team else opp_ms for s in SEATS}
    return {"SeatClocksMs": seat_values(clocks), "SeatPaceMs": seat_values(pace), "SeatElapsedMs": "0 0 0 0"}


def cmd_parity(args) -> int:
    from arena.players.upstream import (
        UPSTREAM_MODEL,
        UPSTREAM_NETWORK,
        WAVE2_BINARY,
        WAVE2_ENGINE_DIR,
        WAVE2_MODEL,
        UpstreamHivemindProcess,
        wave2_flag_options,
    )

    wave2 = args.engine == "wave2"
    model = WAVE2_MODEL if wave2 else UPSTREAM_MODEL
    jobs = random_positions(args.positions, args.seed)
    ref = reference_policies(jobs, model)
    coreml = None if args.coreml == "default" else args.coreml
    if wave2:
        proc = UpstreamHivemindProcess(WAVE2_BINARY, model, options=wave2_flag_options(), coreml=coreml,
                                       cwd=WAVE2_ENGINE_DIR, timeout_s=1800)
    else:
        proc = UpstreamHivemindProcess(coreml=coreml, timeout_s=1800)
    worst, fails, boards = 0.0, 0, 0
    top1_same = 0
    try:
        for i, (job, r) in enumerate(zip(jobs, ref)):
            extra = seat_time_options(job["team"], job["lead"], job["team_ms"], job["opp_ms"]) if wave2 else None
            pol = proc.raw_policy(job["fen_a"], job["fen_b"], team=job["team"], lead=job["lead"],
                                  team_move_ms=job["team_ms"], opp_move_ms=job["opp_ms"], options=extra)
            for k, eng, fen in (("A", pol.board_a, job["fen_a"]), ("B", pol.board_b, job["fen_b"])):
                rk = r[k]
                ek = _engine_to_ref(fen, eng)
                if not rk and not ek:
                    continue
                boards += 1
                if set(rk) != set(ek):
                    fails += 1
                    print(f"pos {i} board {k}: action sets differ: engine-only {sorted(set(ek) - set(rk))[:5]} "
                          f"ref-only {sorted(set(rk) - set(ek))[:5]}")
                    continue
                d = max(abs(ek[m] - rk[m]) for m in rk)
                worst = max(worst, d)
                same = max(ek, key=ek.get) == max(rk, key=rk.get)
                top1_same += same
                if d > args.tol:
                    fails += 1
                print(f"pos {i} board {k} team {job['team']} lead {job['lead']:+d} tm {job['team_ms']} "
                      f"om {job['opp_ms']}: {len(rk)} actions, max|dp| {d:.2e}, top1 {'same' if same else 'DIFF'}"
                      f" ({max(ek, key=ek.get)})")
    finally:
        proc.close()
    print(f"engine: {args.engine} ({proc.binary}); network: {UPSTREAM_NETWORK} ({model}); provider: {args.coreml}")
    print(f"{len(jobs)} positions, {boards} on-turn boards, top-1 agree {top1_same}/{boards}, "
          f"max |dp| {worst:.2e}, failures {fails} (tol {args.tol:g})")
    ok = fails == 0 and boards > 0
    print(f"{args.engine} parity: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


# ------------------------------------------------------------------------------------------------ smoke

def cmd_smoke(args) -> int:
    from arena.openings_bughouse import BUGHOUSE_SETS
    from arena.openings import paired_schedule
    from arena.players import team_players
    from arena.players.upstream import UPSTREAM_NETWORK, upstream_team
    from arena.run import _close, event_to_dict
    from arena.sim import Simulator
    from arena.state import BughouseState
    from arena.stats import summarize_game

    os.makedirs(args.out, exist_ok=True)
    pairs = BUGHOUSE_SETS[args.openings]
    rows = []
    t_start = time.perf_counter()
    for i, (pair, swap) in enumerate(paired_schedule(args.games, args.seed, pairs)):
        up_team = "T1" if swap else "T2"
        it_team = "T2" if swap else "T1"
        players = {}
        players.update(team_players(it_team, "centralized", "hivemind", seed=i, nodes=args.it04_nodes))
        players.update(upstream_team(up_team, seed=i, nodes=args.upstream_nodes, ms_per_node=args.upstream_ms,
                                     coreml=None if args.coreml == "default" else args.coreml))
        ctl = players["A_white" if up_team == "T1" else "A_black"].controller
        t0 = time.perf_counter()
        try:
            state = BughouseState(args.base_ms, 0, fens=pair.fens)
            sim = Simulator(state, players, game_id=f"s{i:03d}", setup="centralized-vs-centralized",
                            max_plies=args.max_plies)
            result = sim.run()
            events = [event_to_dict(e) for e in sim.events]
        finally:
            for p in {id(p): p for p in players.values()}.values():
                _close(p)
        with open(os.path.join(args.out, f"game_{i:03d}.jsonl"), "w") as f:
            for e in events:
                f.write(json.dumps(e, separators=(",", ":")) + "\n")
        s = summarize_game(events)
        score = 0.5 if result.result == "draw" else (1.0 if result.result == up_team else 0.0)
        row = {"game": i, "opening": pair.name, "upstream_team": up_team, "result": result.result,
               "reason": result.reason, "upstream_score": score, "plies": s.get("plies"),
               "upstream_searches": ctl.searches, "upstream_passes": ctl.passes,
               "upstream_no_root": ctl.no_root_results, "upstream_fallback_moves": ctl.fallback_moves,
               "upstream_nodes_requested": ctl.nodes_requested, "upstream_nodes_searched": ctl.nodes_searched,
               "last_time_options": ctl.last_options, "wall_s": round(time.perf_counter() - t0, 1)}
        rows.append(row)
        print(json.dumps(row), flush=True)
    total = sum(r["upstream_score"] for r in rows)
    summary = {"games": len(rows), "upstream_score": total, "upstream_mean": total / max(1, len(rows)),
               "errors": sum(r["upstream_no_root"] + r["upstream_fallback_moves"] for r in rows),
               "aborted": sum(r["reason"] == "aborted" for r in rows),
               "it04": f"stock build-ninja/hivemind + hivemind-fp32.onnx, centralized, {args.it04_nodes} nodes "
                       f"at 5.5 ms/node (MS_PER_NODE)",
               "upstream": f"{UPSTREAM_NETWORK}, {args.upstream_nodes} nodes at {args.upstream_ms} ms/searched node,"
                           f" provider {args.coreml}",
               "base_ms": args.base_ms, "openings": args.openings, "wall_s": round(time.perf_counter() - t_start, 1)}
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump({"summary": summary, "games": rows}, f, indent=1)
    print(json.dumps(summary))
    return 0 if summary["aborted"] == 0 else 1


def cmd_fairy(args) -> int:
    from arena.openings import paired_schedule
    from arena.openings_bughouse import BUGHOUSE_SETS
    from arena.players import team_players
    from arena.players.fairy import fairy_team
    from arena.players.hivemind import MS_PER_NODE
    from arena.run import _close, event_to_dict
    from arena.sim import Simulator
    from arena.state import BughouseState
    from arena.stats import summarize_game

    os.makedirs(args.out, exist_ok=True)
    it04_nodes = args.it04_nodes or max(1, round(args.move_s * 1000 / MS_PER_NODE))
    rows = []
    t_start = time.perf_counter()
    for i, (pair, swap) in enumerate(paired_schedule(args.games, args.seed, BUGHOUSE_SETS[args.openings])):
        fy_team = "T1" if swap else "T2"
        it_team = "T2" if swap else "T1"
        players = {}
        players.update(team_players(it_team, "centralized", "hivemind", seed=i, nodes=it04_nodes))
        fplayers = fairy_team(fy_team, base_ms=args.base_ms, move_s=args.move_s, charge="move")
        players.update(fplayers)
        t0 = time.perf_counter()
        try:
            state = BughouseState(args.base_ms, 0, fens=pair.fens)
            sim = Simulator(state, players, game_id=f"f{i:03d}", setup="centralized-vs-fairy",
                            max_plies=args.max_plies)
            result = sim.run()
            events = [event_to_dict(e) for e in sim.events]
        finally:
            for pl in {id(pl): pl for pl in players.values()}.values():
                _close(pl)
        with open(os.path.join(args.out, f"game_{i:03d}.jsonl"), "w") as f:
            for e in events:
                f.write(json.dumps(e, separators=(",", ":")) + "\n")
        s = summarize_game(events)
        score = 0.5 if result.result == "draw" else (1.0 if result.result == it_team else 0.0)
        fs = list(fplayers.values())
        nodes, wall, searches = (sum(p.nodes_total for p in fs), sum(p.wall_ms_total for p in fs),
                                 sum(p.searches for p in fs))
        row = {"game": i, "opening": pair.name, "fairy_team": fy_team, "result": result.result,
               "reason": result.reason, "it04_score": score, "plies": s.get("plies"),
               "fairy_searches": searches, "fairy_sits": sum(p.sits for p in fs),
               "fairy_fallback_moves": sum(p.fallback_moves for p in fs),
               "fairy_us_per_node": round(1000 * wall / nodes, 3) if nodes else None,
               "wall_s": round(time.perf_counter() - t0, 1)}
        rows.append(row)
        print(json.dumps(row), flush=True)
    total = sum(r["it04_score"] for r in rows)
    summary = {"games": len(rows), "it04_score": total, "it04_mean": total / max(1, len(rows)),
               "reasons": {k: sum(r["reason"] == k for r in rows) for k in sorted({r["reason"] for r in rows})},
               "fallback_moves": sum(r["fairy_fallback_moves"] for r in rows),
               "it04": f"stock build-ninja/hivemind + hivemind-fp32.onnx, centralized, {it04_nodes} nodes "
                       f"at {MS_PER_NODE} ms/node",
               "fairy": f"Fairy-Stockfish CECP, 1 thread, st {args.move_s} s/move charged {args.move_s * 1000} ms",
               "base_ms": args.base_ms, "openings": args.openings, "wall_s": round(time.perf_counter() - t_start, 1)}
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump({"summary": summary, "games": rows}, f, indent=1)
    print(json.dumps(summary))
    return 0 if summary["reasons"].get("aborted", 0) == 0 else 1


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "_reference":
        return _reference_main()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("parity")
    p.add_argument("--positions", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--coreml", default="off", help="HIVEMIND_COREML for the engine: off (CPU), default, gpu, all")
    p.add_argument("--tol", type=float, default=1e-4)
    p.add_argument("--engine", choices=("upstream", "wave2"), default="upstream",
                   help="stock upstream v3 (default) or the wave 2 engine (flags on, seat clocks and paces)")
    s = sub.add_parser("smoke")
    s.add_argument("--games", type=int, default=6)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--base-ms", type=int, default=60000)
    s.add_argument("--max-plies", type=int, default=400)
    s.add_argument("--openings", default="bughouse")
    s.add_argument("--it04-nodes", type=int, default=100)
    s.add_argument("--upstream-nodes", type=int, default=800)
    s.add_argument("--upstream-ms", type=float, default=0.65)
    s.add_argument("--coreml", default="default")
    s.add_argument("--out", default=os.path.join(ROOT, "runs", "upstream", "smoke"))
    fa = sub.add_parser("fairy")
    fa.add_argument("--games", type=int, default=10)
    fa.add_argument("--seed", type=int, default=0)
    fa.add_argument("--move-s", type=int, default=1, help="Fairy's fixed whole seconds per move (CECP st)")
    fa.add_argument("--it04-nodes", type=int, default=0, help="0: move_s / MS_PER_NODE (matched time)")
    fa.add_argument("--base-ms", type=int, default=180000)
    fa.add_argument("--max-plies", type=int, default=400)
    fa.add_argument("--openings", default="bughouse")
    fa.add_argument("--out", default=os.path.join(ROOT, "runs", "upstream", "fairy"))
    args = ap.parse_args(argv)
    return {"parity": cmd_parity, "smoke": cmd_smoke, "fairy": cmd_fairy}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
