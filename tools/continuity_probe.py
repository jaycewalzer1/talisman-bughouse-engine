"""Continuity baseline for wave 2 (docs/continuity_baseline.md): tree-reuse hit rate and node scaling.

Measures what the "continuity" fixes (team tree reuse, joint team actions in the durative tree, plan carry) will be
judged against, on the existing builds, without engine changes.

1. ``reuse``: plays short local arena games, wave 2 (durative on = the headline flags, or off = the ``nodur`` row:
   DurativeTree off, SitGate on) against stock upstream it13, with ``reuse_tree`` on for both arms (the pool rows run
   with it off: the arena sends ``ucinewgame`` before every search). Every search of both engines is logged with the
   options it was sent and its output; a search starts from a reused subtree when the engine prints
   ``info string Tree reuse: N visits recovered`` (verbose UCI output, no engine change needed).
2. ``positions``: picks middlegame wave 2 decisions out of those logs (fens, team, lead, move times and the seat
   options exactly as the arena sent them).
3. ``scaling``: re-searches every picked position with ``go nodes N`` (fresh tree, one search worker, ``build-sp1``)
   under durative on and off, and ``summary`` prints bestmove agreement with deep references.

Exact commands: docs/continuity_baseline.md. Runs in the main repo's ``.venv``.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from arena.players import upstream as up  # noqa: E402

REUSE_RE = re.compile(r"info string Tree reuse: (\d+) visits recovered")
# Printed only by a probe build (a temporary print in Agent::try_reuse_tree, not committed; the patch is in
# docs/continuity_baseline.md): how many retained candidates the lookup saw, how many hold exactly this board (team
# and board signature, clocks ignored) and the most visits among those, i.e. what a clock-tolerant reuse could adopt.
PROBE_RE = re.compile(r"info string ReuseProbe candidates (\d+) sigmatch (\d+) sigvisits (\d+) hashhit (\d+)")
SP1_BINARY = os.path.join(up.WAVE2_ENGINE_DIR, "build-sp1", "hivemind.bin")
# "cont": the headline flags plus the wave 2c continuity preset (spec v5: TeamTreeReuse, JointPlans, PlanCarry). In
# ``reuse`` a hit is then the engine's ``Team tree reuse: N visits recovered`` line; in ``scaling`` every search
# starts from ucinewgame, so only JointPlans acts there (fixes 1 and 3 need a previous search of the same game).
MODES = {"on": {}, "off": {"durative_tree": False, "sit_gate": True}, "cont": {"continuity": True}}


# ---------------------------------------------------------------------------------------------- 1. reuse

def _install_search_log(path: str) -> None:
    """Wrap UpstreamHivemindProcess.search_clock so every search (both arms) is appended to ``path``."""
    orig = up.UpstreamHivemindProcess.search_clock
    out = open(path, "a")

    def logged(self, fen_a, fen_b, **kw):
        res = orig(self, fen_a, fen_b, **kw)
        reuse = [int(m.group(1)) for l in res.lines if (m := REUSE_RE.search(l))]
        probe = [tuple(map(int, m.groups())) for l in res.lines if (m := PROBE_RE.search(l))]
        # Wave 2c fix 1 prints its own line instead of "Tree reuse:" (parse_continuity fills these; None = no line).
        team = getattr(res, "reused_visits", None)
        visits = reuse[0] if reuse else (team or 0)
        row = {"binary": self.binary, "arm": "wave2" if self.binary == up.WAVE2_BINARY else "upstream",
               "engine_flags": {k: self.extra_options.get(k) for k in ("DurativeTree", "SitGate", "TeamTreeReuse",
                                                                      "JointPlans", "PlanCarry")},
               "fen_a": fen_a, "fen_b": fen_b, **{k: v for k, v in kw.items() if k != "options"},
               "seat_options": kw.get("options") or {}, "move_a": res.move_a, "move_b": res.move_b,
               "q": res.q, "nodes_searched": res.nodes, "reused_visits": visits,
               "reused": visits > 0, "team_tree_line": team is not None,
               "reuse_refusal": getattr(res, "reuse_refusal", None),
               "plan_carried": getattr(res, "plan_carried", 0),
               "probe": list(probe[0]) if probe else None, "wall_ms": res.wall_ms, "t": time.time()}
        out.write(json.dumps(row) + "\n")
        out.flush()
        return res

    up.UpstreamHivemindProcess.search_clock = logged


def cmd_reuse(a) -> int:
    from arena import pool as pool_mod

    os.makedirs(a.out, exist_ok=True)
    kw = {"reuse_tree": True, "ms_per_node": a.ms_per_node, **MODES[a.mode]}
    okw = {"reuse_tree": True, "ms_per_node": a.ms_per_node}
    _install_search_log(os.path.join(a.out, "searches.jsonl"))
    return pool_mod.main(["run", "--pool", "arena/pools/wave2.json", "--opponent", "upstream-it13",
                          "--openings", "bughouse", "--rules", "v3", "--base-ms", str(a.base_ms),
                          "--pairs", str(a.pairs), "--seed", str(a.seed), "--out", a.out,
                          "--contender-kw", json.dumps(kw), "--opponent-kw", json.dumps(okw)])


def _read(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def reuse_table(dirs: list[str]) -> list[str]:
    rows = ["| Run | Arm | Searches | Started from a reused subtree | Mean reused visits (all searches) | "
            "Mean reused visits (hits only) | Mean nodes searched |", "|---|---|---|---|---|---|---|"]
    for d in dirs:
        s = _read(os.path.join(d, "searches.jsonl"))
        for arm in ("wave2", "upstream"):
            xs = [r for r in s if r["arm"] == arm]
            if not xs:
                continue
            hits = [r for r in xs if r["reused"]]
            rows.append(f"| {os.path.basename(d)} | {arm} | {len(xs)} | {len(hits)} ({len(hits) / len(xs):.1%}) | "
                        f"{sum(r['reused_visits'] for r in xs) / len(xs):.1f} | "
                        f"{(sum(r['reused_visits'] for r in hits) / len(hits)) if hits else 0:.1f} | "
                        f"{sum(r['nodes_searched'] for r in xs) / len(xs):.0f} |")
    # Wave 2c runs (TeamTreeReuse / PlanCarry on): why a search adopted nothing, and how many carried halves applied.
    cont_rows = ["", "Wave 2c telemetry (wave 2 arm, searches that printed a `Team tree reuse` line):", "",
                 "| Run | Searches with the line | Adopted | Refused, by reason | Boards with a carried plan applied |",
                 "|---|---|---|---|---|"]
    any_cont = False
    for d in dirs:
        xs = [r for r in _read(os.path.join(d, "searches.jsonl")) if r["arm"] == "wave2" and r.get("team_tree_line")]
        if not xs:
            continue
        any_cont = True
        reasons: dict[str, int] = {}
        for r in xs:
            if r.get("reuse_refusal"):
                reasons[r["reuse_refusal"]] = reasons.get(r["reuse_refusal"], 0) + 1
        hits = sum(1 for r in xs if r["reused"])
        cont_rows.append(f"| {os.path.basename(d)} | {len(xs)} | {hits} ({hits / len(xs):.1%}) | "
                         + ", ".join(f"{k} {v}" for k, v in sorted(reasons.items(), key=lambda kv: -kv[1]))
                         + f" | {sum(r.get('plan_carried', 0) for r in xs)} |")
    rows += cont_rows if any_cont else []
    probe_rows =["", "Probe build only: the same board (clocks ignored) among the retained candidates, i.e. the "
                  "ceiling of a clock-tolerant reuse.", "",
                  "| Run | Arm | Searches with probe line | Mean candidates indexed | Board found (clocks ignored) | "
                  "Mean visits of that subtree (all searches) | (when found) | Mean nodes searched |",
                  "|---|---|---|---|---|---|---|---|"]
    any_probe = False
    for d in dirs:
        s = _read(os.path.join(d, "searches.jsonl"))
        for arm in ("wave2", "upstream"):
            xs = [r for r in s if r["arm"] == arm and r.get("probe")]
            if not xs:
                continue
            any_probe = True
            found = [r for r in xs if r["probe"][1] > 0]
            rows_n = len(xs)
            probe_rows.append(
                f"| {os.path.basename(d)} | {arm} | {rows_n} | {sum(r['probe'][0] for r in xs) / rows_n:.0f} | "
                f"{len(found)} ({len(found) / rows_n:.1%}) | {sum(r['probe'][2] for r in xs) / rows_n:.1f} | "
                f"{(sum(r['probe'][2] for r in found) / len(found)) if found else 0:.1f} | "
                f"{sum(r['nodes_searched'] for r in xs) / rows_n:.0f} |")
    return rows + (probe_rows if any_probe else [])


# ---------------------------------------------------------------------------------------------- 2. positions

def _fullmove(fen: str) -> int:
    return int(fen.split()[-1])


def _men(fen: str) -> int:
    return sum(c.isalpha() for c in fen.split("[")[0].split()[0])


def cmd_positions(a) -> int:
    pool = []
    for d in a.runs:
        for r in _read(os.path.join(d, "searches.jsonl")):
            if r["arm"] != "wave2":
                continue
            fm = min(_fullmove(r["fen_a"]), _fullmove(r["fen_b"]))
            if fm < a.min_fullmove or fm > a.max_fullmove:
                continue
            if min(_men(r["fen_a"]), _men(r["fen_b"])) < 16:
                continue
            if r["q"] is None or abs(r["q"]) > a.max_abs_q:
                continue
            pool.append({k: r[k] for k in ("fen_a", "fen_b", "team", "lead", "team_move_ms", "opp_move_ms",
                                           "root_move_ms", "seat_options", "q", "nodes")} | {"run": d})
    rng = random.Random(a.seed)
    # De-duplicate (both seats can be asked at one position) and spread over the runs.
    seen, uniq = set(), []
    for p in pool:
        k = (p["fen_a"], p["fen_b"], p["team"])
        if k not in seen:
            seen.add(k)
            uniq.append(p)
    pick = rng.sample(uniq, min(a.n, len(uniq)))
    with open(a.out, "w") as f:
        for i, p in enumerate(pick):
            f.write(json.dumps({"id": i, **p}) + "\n")
    print(f"{len(uniq)} candidate middlegame decisions, wrote {len(pick)} to {a.out}")
    return 0


# ---------------------------------------------------------------------------------------------- 3. scaling

def cmd_scaling(a) -> int:
    positions = _read(a.positions)
    if a.ids:
        lo, hi = a.ids
        positions = [p for p in positions if lo <= p["id"] < hi]
    done = set()
    if os.path.exists(a.out):
        done = {(r["id"], r["mode"], r["nodes"], r.get("rep", 0)) for r in _read(a.out)}
    out = open(a.out, "a")
    for mode in a.modes:
        if mode == "it13":
            # Deep reference only: stock upstream it13 (its own build-ninja, 4 workers), no wave 2 flags and no seat
            # timing options (upstream does not know them), same team / lead / move times.
            proc = up.UpstreamHivemindProcess(up.UPSTREAM_BINARY, up.UPSTREAM_MODEL, reset_tree=True, timeout_s=1800)
        else:
            proc = up.UpstreamHivemindProcess(a.binary, up.WAVE2_MODEL,
                                              options=up.wave2_flag_options(**MODES[mode]),
                                              reset_tree=True, timeout_s=1800, cwd=up.WAVE2_ENGINE_DIR)
        try:
            for nodes in a.nodes:
                for p in positions:
                    key = (p["id"], mode, nodes, a.rep)
                    if key in done:
                        continue
                    res = proc.search_clock(p["fen_a"], p["fen_b"], team=p["team"], nodes=nodes, lead=p["lead"],
                                            team_move_ms=p["team_move_ms"], opp_move_ms=p["opp_move_ms"],
                                            root_move_ms=p["root_move_ms"],
                                            options=None if mode == "it13" else p["seat_options"])
                    row = {"id": p["id"], "mode": mode, "nodes": nodes, "rep": a.rep, "move_a": res.move_a,
                           "move_b": res.move_b, "q": res.q, "nodes_searched": res.nodes, "wall_ms": res.wall_ms,
                           "binary": proc.binary}
                    out.write(json.dumps(row) + "\n")
                    out.flush()
                print(f"{mode} {nodes} done", flush=True)
        finally:
            proc.close()
    return 0


def _on_turn(fen: str) -> str:
    return fen.split()[1]


def scaling_tables(positions_path: str, results_path: str) -> list[str]:
    pos = {p["id"]: p for p in _read(positions_path)}
    res: dict = {}
    for r in _read(results_path):
        res[(r["id"], r["mode"], r["nodes"], r.get("rep", 0))] = r
    budgets = sorted({k[2] for k in res if k[1] != "it13"})

    def joint(r):
        return (r["move_a"], r["move_b"])

    def team_boards(p):
        # The boards our team is on turn on: T1 = A white + B black.
        w = "w" if p["team"] == "T1" else "b"
        bw = "b" if p["team"] == "T1" else "w"
        return [b for b, fen, side in (("a", p["fen_a"], w), ("b", p["fen_b"], bw)) if _on_turn(fen) == side]

    def agree(ra, rb, p):
        both = [b for b in team_boards(p)]
        per = [ra[f"move_{b}"] == rb[f"move_{b}"] for b in both]
        return joint(ra) == joint(rb), (sum(per) / len(per)) if per else 1.0

    # Wave 2 modes present in the results, in table order (cont: the wave 2c continuity preset).
    modes = [m for m in ("on", "off", "cont") if any(k[1] == m for k in res)]
    LABEL = {"on": "durative on", "off": "durative off", "cont": "continuity"}
    lines = []
    for ref_mode in ("off", "on", "cont", "it13"):
        ref_budgets = sorted({k[2] for k in res if k[1] == ref_mode})
        if not ref_budgets:
            continue
        deep = max(ref_budgets)
        if not all((i, ref_mode, deep, 0) in res for i in pos):
            continue
        lines += ["", f"Reference: {'upstream it13' if ref_mode == 'it13' else 'durative ' + ref_mode} at {deep} nodes (joint action / mean per on-turn board).", "",
                  "| Nodes | " + " | ".join(f"{LABEL[m]}: joint | {m}: per board" for m in modes) + " | n |",
                  "|---|" + "---|---|" * len(modes) + "---|"]
        for n in budgets:
            cells = []
            cnt = 0
            for mode in modes:
                js, bs = [], []
                for i, p in pos.items():
                    r, ref = res.get((i, mode, n, 0)), res[(i, ref_mode, deep, 0)]
                    if r is None:
                        continue
                    j, b = agree(r, ref, p)
                    js.append(j)
                    bs.append(b)
                cnt = max(cnt, len(js))
                cells += [f"{sum(js)}/{len(js)} ({sum(js) / len(js):.0%})" if js else "-",
                          f"{sum(bs) / len(bs):.0%}" if bs else "-"]
            lines.append(f"| {n} | " + " | ".join(cells) + f" | {cnt} |")
    # on vs off at the same budget
    lines += ["", "On vs off at the same budget (joint action equal), and passes chosen:", "",
              "| Nodes | on = off | " + " | ".join(f"{m}: root passes on an on-turn board" for m in modes) + " |",
              "|---|---|" + "---|" * len(modes)]
    for n in budgets:
        same = [joint(res[(i, "on", n, 0)]) == joint(res[(i, "off", n, 0)]) for i in pos
                if (i, "on", n, 0) in res and (i, "off", n, 0) in res]
        passes = {}
        for mode in modes:
            c = 0
            for i, p in pos.items():
                r = res.get((i, mode, n, 0))
                if r and any(r[f"move_{b}"] is None for b in team_boards(p)):
                    c += 1
            passes[mode] = c
        lines.append(f"| {n} | {sum(same)}/{len(same)} | " + " | ".join(str(passes[m]) for m in modes) + " |")
    # repeat noise
    reps = [(k, r) for k, r in res.items() if k[3] == 1]
    if reps:
        lines += ["", "Repeat (same search run again, noise floor):", "", "| Mode | Nodes | same joint action |",
                  "|---|---|---|"]
        for mode in modes:
            for n in budgets:
                xs = [joint(r) == joint(res[(k[0], k[1], k[2], 0)]) for k, r in reps
                      if k[1] == mode and k[2] == n and (k[0], k[1], k[2], 0) in res]
                if xs:
                    lines.append(f"| {mode} | {n} | {sum(xs)}/{len(xs)} |")
    # speed
    lines += ["", "Wall time per searched node (ms), build-sp1:", "", "| Nodes | " + " | ".join(modes) + " |",
              "|---|" + "---|" * len(modes)]
    for n in budgets:
        cells = []
        for mode in modes:
            xs = [r for k, r in res.items() if k[1] == mode and k[2] == n and k[3] == 0]
            cells.append(f"{sum(r['wall_ms'] for r in xs) / max(1, sum(r['nodes_searched'] for r in xs)):.3f}"
                         if xs else "-")
        lines.append(f"| {n} | " + " | ".join(cells) + " |")
    return lines


def cmd_summary(a) -> int:
    if a.reuse:
        print("\n".join(reuse_table(a.reuse)))
    if a.positions and a.results:
        print("\n".join(scaling_tables(a.positions, a.results)))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="tools/continuity_probe.py", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("reuse", help="arena games with reuse_tree on, every search logged")
    p.add_argument("--mode", choices=tuple(MODES), required=True)
    p.add_argument("--pairs", type=int, default=2)
    p.add_argument("--base-ms", type=int, default=60000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ms-per-node", type=float, default=0.22)
    p.add_argument("--out", required=True)
    p = sub.add_parser("positions", help="pick middlegame wave 2 decisions from reuse runs")
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--n", type=int, default=40)
    p.add_argument("--min-fullmove", type=int, default=8)
    p.add_argument("--max-fullmove", type=int, default=30)
    p.add_argument("--max-abs-q", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    p = sub.add_parser("scaling", help="bestmove per position, mode and node budget")
    p.add_argument("--positions", required=True)
    p.add_argument("--nodes", type=int, nargs="+", default=[1000, 4000, 16000])
    p.add_argument("--modes", nargs="+", choices=tuple(MODES) + ("it13",), default=["on", "off"])
    p.add_argument("--binary", default=SP1_BINARY)
    p.add_argument("--rep", type=int, default=0, help="repeat index (1 = a second run, for the noise floor)")
    p.add_argument("--ids", type=int, nargs=2, metavar=("LO", "HI"),
                   help="only positions with LO <= id < HI (splits a long budget into short calls; reruns resume)")
    p.add_argument("--out", required=True)
    p = sub.add_parser("summary", help="markdown tables")
    p.add_argument("--reuse", nargs="*", default=[])
    p.add_argument("--positions")
    p.add_argument("--results")
    a = ap.parse_args(argv)
    return {"reuse": cmd_reuse, "positions": cmd_positions, "scaling": cmd_scaling, "summary": cmd_summary}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
