"""Pass and sit statistics from recorded self-play shards and arena event logs (audit 2026-11-17 WP-H: search 4 /
inputs 9 measurement, training 3(b) stats, self-play 2 stat, theory 02 section 3.5).

Shards (``.hvm`` chunks plus their ``.pos.jsonl`` sidecars, as arena.selfplay writes them):

* For every sample and every board where the sample's team is on turn: does the marginal visit policy carry the
  pass slot (index 0) with nonzero visits, its visit share, whether the pass is the most-visited entry, and the
  joint-policy mass on joint actions that pass on that board; split by the sign and size of that seat's diagonal
  (own clock minus the partner's opponent's clock) and by the node count. With ``--model``, also the raw network's
  pass prior on the same positions (softmax over the legal moves plus the pass slot).
* Value labels by game end reason (from the run's ``summary.json``): samples, mean |root_q - outcome|, and the
  root_q -> outcome calibration by material bucket (men on both boards, kings and pockets excluded).

Event logs (``game_*.jsonl``, e.g. the gate's): decisions, counted as every move and every start of a sit spell
(a ``stall`` event that turns a seat from moving to sitting; re-searches inside a spell are not decisions), with
the seat's diagonal at that time (clocks replayed from the last move's snapshot), as a sit rate by diagonal; sit-length distribution; and whether each flagged seat was sitting or moving when it flagged.
Seats are grouped by their player name (``free`` in the name = the free-mode contender, else the opponent).

    .venv/bin/python tools/pass_stats.py --shards runs/custom/local1h/it01/selfplay/training_data \\
        --events runs/custom/local1h/it01/gate [--model <onnx>] [--json out.json]

Single process, no engine.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
from collections import defaultdict

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from arena.interfaces import OPPONENT, PARTNER, TEAM_OF  # noqa: E402
from arena.recorder import PASS_POLICY_INDEX, TEAM_OF_BYTE, read_chunk, read_sidecar, view_sample  # noqa: E402

PASS_INDEX = 0  # marginal slot of a pass; PASS_POLICY_INDEX (4672) marks a pass inside a joint entry
DIAG_BINS_S = (-math.inf, -30, -10, -2, 2, 10, 30, math.inf)
NODE_BINS = (0, 32, 64, 128, 256, 512, math.inf)
MATERIAL_BINS = (0, 30, 40, 50, 60, math.inf)  # men on both boards, kings and pockets excluded (start: 60)


def diag_sign(ms: float) -> str:
    return "ahead" if ms > 0 else ("behind" if ms < 0 else "equal")


def bin_label(x: float, edges, unit: str = "") -> str:
    for lo, hi in zip(edges, edges[1:]):
        if lo <= x < hi:
            lo_s = "-inf" if lo == -math.inf else f"{lo:g}"
            hi_s = "inf" if hi == math.inf else f"{hi:g}"
            return f"[{lo_s},{hi_s}){unit}"
    return "?"


def _order(key: str):
    """Sort key: bins by their lower edge, then other labels alphabetically."""
    k = key.removeprefix("bin ")
    if k.startswith("["):
        lo = k[1:].split(",")[0]
        return (1, -math.inf if lo == "-inf" else float(lo), key)
    return (0, 0.0, key)


def on_turn_seats(fen_a: str, fen_b: str, team: str) -> dict[str, str]:
    """{board: seat} for the boards where ``team`` is to move."""
    out = {}
    for b, fen in (("A", fen_a), ("B", fen_b)):
        seat = f"{b}_{'white' if fen.split()[1] == 'w' else 'black'}"
        if TEAM_OF[seat] == team:
            out[b] = seat
    return out


def seat_diagonal(clocks: dict, seat: str) -> int:
    """Own clock minus the partner's opponent's clock (ms): the race a sit on this seat's board runs."""
    return int(clocks[seat]) - int(clocks[OPPONENT[PARTNER[seat]]])


def material(fen_a: str, fen_b: str) -> int:
    """Men on both boards, kings and pockets excluded (60 at the start). Bughouse conserves material (a capture
    moves a piece to a pocket), so density on the boards, not the total, is what changes over a game."""
    n = 0
    for fen in (fen_a, fen_b):
        board = fen.split(" ")[0].split("[")[0]
        n += sum(1 for ch in board if ch.isalpha() and ch.lower() != "k")
    return n


# --------------------------------------------------------------------------------------------- shards

def shard_rows(chunk_dirs: list[str]) -> list[dict]:
    """One row per (sample, on-turn board). Samples without a sidecar line are skipped (no clocks)."""
    rows = []
    for d in chunk_dirs:
        for path in sorted(glob.glob(os.path.join(d, "*.hvm"))):
            channels, samples, version = read_chunk(path, with_version=True)
            metas = read_sidecar(path)
            if metas is None or len(metas) != len(samples):
                print(f"skip {path}: no matching sidecar", file=sys.stderr)
                continue
            for i, (buf, meta) in enumerate(zip(samples, metas)):
                v = view_sample(buf, channels, version)
                team = TEAM_OF_BYTE[v.team]
                if meta.get("team") and meta["team"] != team:
                    raise ValueError(f"{path}#{i}: sidecar team {meta['team']} != sample team {team}")
                for board, seat in on_turn_seats(meta["fenA"], meta["fenB"], team).items():
                    pol = v.policy_a if board == "A" else v.policy_b
                    p_pass = sum(p for idx, p in pol if idx == PASS_INDEX)
                    top = max(pol, key=lambda e: e[1])[0] if pol else None
                    k = 0 if board == "A" else 1
                    joint_pass = sum(e[2] for e in v.joint if e[k] == PASS_POLICY_INDEX)
                    rows.append({
                        "chunk": os.path.basename(path), "i": i, "game": v.game_id or meta.get("gameId"),
                        "board": board, "seat": seat, "team": team, "diag_ms": seat_diagonal(meta["clocks"], seat),
                        "nodes": v.nodes, "nodes_requested": meta.get("nodes"), "p_pass": p_pass,
                        "pass_visited": p_pass > 0, "pass_top": top == PASS_INDEX, "joint_pass": joint_pass,
                        "n_entries": len(pol), "root_q": v.root_q, "outcome": v.outcome,
                        "material": material(meta["fenA"], meta["fenB"]), "meta": meta})
    return rows


def add_pass_prior(rows: list[dict], model: str) -> None:
    import numpy as np
    from regression import NetEval
    net = NetEval(model)
    x = np.stack([net.planes(r["meta"]["fenA"], r["meta"]["fenB"], r["team"], r["meta"]["clocks"],
                             r["meta"].get("elapsed"), bool(r["meta"].get("timeAdvantage", False))) for r in rows])
    out = net.run(x)
    for i, r in enumerate(rows):
        logits = out["pi_a"][i] if r["board"] == "A" else out["pi_b"][i]
        pol = net.board_policy(logits, r["meta"]["fenA"] if r["board"] == "A" else r["meta"]["fenB"])
        r["prior_pass"] = pol["p_pass"]
        r["prior_pass_top"] = pol["top"] == "pass"


def group(rows: list[dict], key) -> dict[str, dict]:
    acc: dict[str, dict] = defaultdict(lambda: defaultdict(float))
    for r in rows:
        g = acc[key(r)]
        g["n"] += 1
        g["pass_visited"] += r["pass_visited"]
        g["pass_top"] += r["pass_top"]
        g["p_pass"] += r["p_pass"]
        g["joint_pass"] += r["joint_pass"]
        if "prior_pass" in r:
            g["prior_n"] += 1
            g["prior_pass"] += r["prior_pass"]
            g["prior_pass_top"] += r["prior_pass_top"]
    out = {}
    for k, g in acc.items():
        n = g["n"]
        row = {"n": int(n), "share_pass_visited": round(g["pass_visited"] / n, 4),
               "share_pass_top": round(g["pass_top"] / n, 4), "mean_pass_visits": round(g["p_pass"] / n, 4),
               "mean_joint_pass": round(g["joint_pass"] / n, 4)}
        if g["prior_n"]:
            row["mean_prior_pass"] = round(g["prior_pass"] / g["prior_n"], 4)
            row["share_prior_pass_top"] = round(g["prior_pass_top"] / g["prior_n"], 4)
        out[k] = row
    return dict(sorted(out.items(), key=lambda kv: _order(kv[0])))


def game_reasons(chunk_dirs: list[str]) -> dict[int, str]:
    """game number -> end reason, from each chunk dir's run ``summary.json`` (one level up)."""
    out = {}
    for d in chunk_dirs:
        path = os.path.join(os.path.dirname(os.path.abspath(d.rstrip("/"))), "summary.json")
        if os.path.isfile(path):
            with open(path) as f:
                for g in json.load(f).get("games", []):
                    if "game_num" in g:
                        out[int(g["game_num"])] = g.get("reason") or "?"
    return out


def value_label_stats(rows: list[dict], reasons: dict[int, str]) -> dict:
    """Per sample (first on-turn board row only): |root_q - outcome| by end reason; calibration by material."""
    seen, by_reason, by_mat = set(), defaultdict(list), defaultdict(list)
    for r in rows:
        key = (r["chunk"], r["i"])
        if key in seen:
            continue
        seen.add(key)
        reason = r["meta"].get("reason") or reasons.get(int(r["game"]), "?")
        by_reason[reason].append(abs(r["root_q"] - r["outcome"]))
        by_mat[bin_label(r["material"], MATERIAL_BINS)].append((r["root_q"], r["outcome"]))
    reason_out = {k: {"samples": len(v), "mean_abs_q_minus_outcome": round(sum(v) / len(v), 4)}
                  for k, v in sorted(by_reason.items())}
    mat_out = {}
    for k, v in sorted(by_mat.items(), key=lambda kv: _order(kv[0])):
        q = [a for a, _ in v]
        z = [b for _, b in v]
        mat_out[k] = {"samples": len(v), "mean_root_q": round(sum(q) / len(v), 4),
                      "mean_outcome": round(sum(z) / len(v), 4),
                      "mse": round(sum((a - b) ** 2 for a, b in v) / len(v), 4),
                      "sign_agree": round(sum((a > 0) == (b > 0) for a, b in v if b != 0) / max(1, sum(b != 0 for b in z)), 4)}
    return {"by_reason": reason_out, "calibration_by_material": mat_out}


# --------------------------------------------------------------------------------------------- event logs

def _replay_clocks(snap: dict, snap_ts: int, fens: dict, now: int) -> dict:
    """Clocks at ``now``: the last snapshot, minus the time since then for the seat on turn on each board."""
    out = dict(snap)
    for b in ("A", "B"):
        seat = f"{b}_{'white' if fens[b].split()[1] == 'w' else 'black'}"
        out[seat] = out[seat] - max(0, now - snap_ts)
    return out


def event_stats(paths: list[str]) -> dict:
    """Decisions (moved vs sat) by diagonal, sit lengths and flag state, from arena event logs."""
    decisions: dict[str, list[dict]] = defaultdict(list)
    sits: dict[str, list[dict]] = defaultdict(list)
    flags = defaultdict(lambda: defaultdict(int))
    games = 0
    for path in paths:
        with open(path) as f:
            events = [json.loads(line) for line in f if line.strip()]
        start = next((e for e in events if e["type"] == "game_start"), None)
        if start is None:
            continue
        games += 1
        kind = {s: ("contender" if "free" in v.get("name", "") else "opponent") for s, v in start["seats"].items()}
        fens = dict(start["fens"])
        snap, snap_ts = dict(start["clocks"]), 0
        sitting: dict[str, tuple[int, int]] = {}  # seat -> (start ts, diagonal at start)
        for e in events:
            t = e.get("ts", 0)
            typ = e["type"]
            if typ == "move":
                seat = e["seat"]
                d = seat_diagonal(_replay_clocks(snap, snap_ts, fens, t), seat)
                decisions[kind[seat]].append({"diag_ms": d, "sat": False})
                if seat in sitting:
                    st, d0 = sitting.pop(seat)
                    sits[kind[seat]].append({"ms": t - st, "diag_ms": d0, "ended": "move"})
                fens[e["board"]] = e["fen"]
                snap, snap_ts = dict(e["clocks"]), t
            elif typ == "stall" and e.get("active") and e["seat"] not in sitting:
                # a new sit spell; "active": false (the seat thinks again) keeps the spell open until it moves
                seat = e["seat"]
                d = seat_diagonal(_replay_clocks(snap, snap_ts, fens, t), seat)
                decisions[kind[seat]].append({"diag_ms": d, "sat": True, "reason": e.get("reason")})
                sitting[seat] = (t, d)
            elif typ == "game_end":
                for seat, (st, d) in sitting.items():
                    sits[kind[seat]].append({"ms": t - st, "diag_ms": d, "ended": "game_end"})
                if e.get("reason") == "flag" and e.get("loserSeat"):
                    loser = e["loserSeat"]
                    flags[kind[loser]]["sitting" if loser in sitting else "moving"] += 1
                sitting = {}
    out = {"games": games, "decisions": {}, "sit_lengths": {}, "flags": {k: dict(v) for k, v in flags.items()}}
    for k, rows in sorted(decisions.items()):
        by = defaultdict(lambda: [0, 0])
        for r in rows:
            for key in (diag_sign(r["diag_ms"]), "bin " + bin_label(r["diag_ms"] / 1000, DIAG_BINS_S, "s")):
                by[key][0] += 1
                by[key][1] += r["sat"]
        out["decisions"][k] = {"n": len(rows), "sit_rate": round(sum(r["sat"] for r in rows) / max(1, len(rows)), 4),
                               "by_diagonal": {key: {"n": n, "sit_rate": round(s / n, 4)}
                                               for key, (n, s) in sorted(by.items(), key=lambda kv: _order(kv[0]))}}
    for k, rows in sorted(sits.items()):
        per = {}
        for sign in ("ahead", "equal", "behind", "all"):
            ms = sorted(r["ms"] for r in rows if sign == "all" or diag_sign(r["diag_ms"]) == sign)
            if ms:
                per[sign] = {"n": len(ms), "median_ms": ms[len(ms) // 2], "p90_ms": ms[int(0.9 * (len(ms) - 1))],
                             "mean_ms": round(sum(ms) / len(ms)), "max_ms": ms[-1]}
        out["sit_lengths"][k] = per
    return out


# --------------------------------------------------------------------------------------------- CLI

def run(args) -> dict:
    report: dict = {"config": {k: v for k, v in vars(args).items()}}
    if args.shards:
        rows = shard_rows(args.shards)
        if args.model and rows:
            add_pass_prior(rows, args.model)
        report["shard_rows"] = len(rows)
        report["samples"] = len({(r["chunk"], r["i"]) for r in rows})
        report["pass_by_diag_sign"] = group(rows, lambda r: diag_sign(r["diag_ms"]))
        report["pass_by_diag_bin"] = group(rows, lambda r: bin_label(r["diag_ms"] / 1000, DIAG_BINS_S, "s"))
        report["pass_by_nodes"] = group(rows, lambda r: bin_label(r["nodes"], NODE_BINS))
        report["pass_all"] = group(rows, lambda r: "all")
        report["value_labels"] = value_label_stats(rows, game_reasons(args.shards))
    if args.events:
        paths = []
        for p in args.events:
            paths += sorted(glob.glob(os.path.join(p, "game_*.jsonl"))) if os.path.isdir(p) else [p]
        report["events"] = event_stats(paths)
    return report


def print_report(rep: dict) -> None:
    def table(title, groups):
        print(f"\n{title}")
        cols = ["n", "share_pass_visited", "share_pass_top", "mean_pass_visits", "mean_joint_pass",
                "mean_prior_pass", "share_prior_pass_top"]
        print(f"  {'group':<16}" + "".join(f"{c[:18]:>20}" for c in cols))
        for k, g in groups.items():
            print(f"  {k:<16}" + "".join(f"{g.get(c, '-')!s:>20}" for c in cols))
    if "pass_all" in rep:
        print(f"{rep['samples']} samples, {rep['shard_rows']} (sample, on-turn board) rows")
        table("pass by diagonal sign of the on-turn seat", rep["pass_by_diag_sign"])
        table("pass by diagonal (s)", rep["pass_by_diag_bin"])
        table("pass by searched nodes", rep["pass_by_nodes"])
        table("all", rep["pass_all"])
        print("\nvalue labels by end reason:", json.dumps(rep["value_labels"]["by_reason"]))
        print("root_q calibration by material:")
        for k, v in rep["value_labels"]["calibration_by_material"].items():
            print(f"  {k:<12} {json.dumps(v)}")
    if "events" in rep:
        ev = rep["events"]
        print(f"\nevent logs: {ev['games']} games")
        for k, d in ev["decisions"].items():
            print(f"  {k}: {d['n']} decisions, sit rate {d['sit_rate']}")
            for key, v in d["by_diagonal"].items():
                print(f"    {key:<18} n={v['n']:<5} sit_rate={v['sit_rate']}")
        for k, d in ev["sit_lengths"].items():
            print(f"  sit lengths {k}: {json.dumps(d)}")
        print(f"  flags (flagged seat sitting vs moving): {json.dumps(ev['flags'])}")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shards", action="append", default=[], help="directory of .hvm chunks with sidecars")
    ap.add_argument("--events", action="append", default=[], help="event-log directory (game_*.jsonl) or file")
    ap.add_argument("--model", default=None, help="ONNX network: also report its raw pass prior on the samples")
    ap.add_argument("--json", default=None, help="write the report as JSON")
    args = ap.parse_args(argv)
    if not args.shards and not args.events:
        ap.error("give --shards and/or --events")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    rep = run(args)
    print_report(rep)
    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w") as f:
            json.dump(rep, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
