#!/usr/bin/env python3
"""Hoarding diagnosis (spec v7 4.3): does a contender hold pieces in its pockets longer than its opponent, and does
its search think holding is better than the results say?

Reads arena match folders (``part*/results.jsonl`` + ``game_*.jsonl``) and compares the contender's seats ("us",
the results row's ``contender_team``) with the opponent's ("them"), each split by that side's own result (won,
lost). Every number comes with a 95% range from a bootstrap over games (500 resamples, seeded). Pockets are
tracked from the events (the start FENs, ``pass`` events in, drops out) and checked against every move's FEN.

  1. drop rate: drops / own moves made with a non-empty pocket, overall and by game third (ply / game plies)
  2. holding: per piece that lands in a seat's pocket, ms and own moves until that seat drops it (FIFO per piece
     type; censored at game end): median, mean, p90 over dropped pieces and the share never dropped
  3. pocket value at own decisions (P1 N3 B3 R5 Q9, the deciding seat's pocket at each think start) and the
     material balance (team on-board + pockets minus the other team's) by fifth of the game
  4. calibration: mean search eval at own thinks (thinker's view, [-1, 1]) against the game result (+1/0/-1) by
     the thinking seat's pocket value (0, 1-4, 5-9, 10+)
  5. endings: mated / flag / sit-off / draw, the mated seat's and its partner's pocket at the mate, drop-mates
  6. (``--model``) network outputs on sampled decision positions from the engine's own ``policy`` command (the
     planes the engine builds, checked against an independent encoder by ``tools/upstream_check.py parity
     --engine wave2``): value for the deciding team and the prior mass on drops for the deciding seat's board,
     against the result and whether that seat's next move was a drop

Usage:
  .venv/bin/python tools/hoarding_diagnosis.py MATCH_DIR [--ref NAME=DIR ...] [--out tables.md] [--json out.json]
      [--model ONNX --positions 600 [--binary BIN] [--max-games N]]
MATCH_DIR holds part*/ folders (e.g. runs/study/full/hybrid1/anchors/it2/hybrid-it13). Writes markdown tables.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import sys
from collections import deque

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tools.study_games import SEATS, WHITE  # noqa: E402

VALUE = {"P": 1, "N": 3, "B": 3, "R": 5, "Q": 9}
PIECES = "PNBRQ"
TEAM_OF = {"A_white": "T1", "B_black": "T1", "A_black": "T2", "B_white": "T2"}
PARTNER = {"A_white": "B_black", "B_black": "A_white", "A_black": "B_white", "B_white": "A_black"}
BOARD_OF = {s: s[0] for s in SEATS}
POCKET_BINS = ((0, 0, "0"), (1, 4, "1-4"), (5, 9, "5-9"), (10, 10 ** 6, "10+"))
SIDES = ("us", "them")
OUTCOMES = ("all", "won", "lost")
BOOT = 500


def pocket_bin(v: int) -> str:
    return next(name for lo, hi, name in POCKET_BINS if lo <= v <= hi)


def fen_pockets(fen: str) -> tuple[dict, dict]:
    """({piece: n} white, {piece: n} black) of a bracket FEN's pocket."""
    board = fen.split()[0]
    m = re.search(r"\[([^\]]*)\]", board)
    w, b = dict.fromkeys(PIECES, 0), dict.fromkeys(PIECES, 0)
    for ch in (m.group(1) if m else ""):
        if ch.upper() in VALUE:
            (w if ch.isupper() else b)[ch.upper()] += 1
    return w, b


def on_board(fen: str, white: bool) -> int:
    placement = fen.split()[0].split("[")[0]
    return sum(VALUE.get(c.upper(), 0) for c in placement if c.isalpha() and c.isupper() == white)


def pocket_value(p: dict) -> int:
    return sum(VALUE[k] * n for k, n in p.items())


# ---------------------------------------------------------------------------------------------- one game

def game_records(events: list, contender_team: str) -> dict | None:
    """Everything the tables need from one game, per seat event, sides as "us"/"them" (us = contender_team)."""
    start = next((e for e in events if e["type"] == "game_start"), None)
    end = next((e for e in events if e["type"] == "game_end"), None)
    if start is None or end is None:
        return None
    side_of = {s: ("us" if TEAM_OF[s] == contender_team else "them") for s in SEATS}
    res = end.get("result")
    outcome = {}
    for side in SIDES:
        team = contender_team if side == "us" else ("T2" if contender_team == "T1" else "T1")
        outcome[side] = 1 if res == team else (-1 if res in ("T1", "T2") else 0)
    fens = dict(start["fens"])
    pockets = {}
    for b in ("A", "B"):
        w, bl = fen_pockets(fens[b])
        pockets[f"{b}_white"], pockets[f"{b}_black"] = w, bl
    clocks = dict(start["clocks"])
    last_land = {"A": 0, "B": 0}
    held: dict[str, dict[str, deque]] = {s: {p: deque() for p in PIECES} for s in SEATS}
    own_moves = dict.fromkeys(SEATS, 0)
    for s in SEATS:  # pieces in the start pockets landed at 0
        for p in PIECES:
            held[s][p].extend((0, 0) for _ in range(pockets[s][p]))
    total_plies = sum(1 for e in events if e["type"] == "move")
    out = {"outcome": outcome, "reason": end.get("reason"), "moves": [], "holds": [], "thinks": [], "balance": [],
           "end": None, "mismatch": 0, "checked": 0, "dur": int(end["ts"]), "plies": total_plies}
    think_start: dict[str, dict] = {}
    team_think = {"T1": [], "T2": []}
    ply = 0
    for e in events:
        t = e["type"]
        if t == "pass":
            s = e["toSeat"]
            p = e["piece"].upper()
            if p in VALUE:
                pockets[s][p] += 1
                held[s][p].append((int(e["ts"]), own_moves[s]))
        elif t == "thinking":
            s = e["seat"]
            if e["phase"] == "start":
                think_start[s] = {"ts": int(e["ts"]), "pocket": pocket_value(pockets[s]),
                                  "team_pocket": pocket_value(pockets[s]) + pocket_value(pockets[PARTNER[s]]),
                                  "fens": dict(fens), "clocks": _clocks_at(clocks, fens, last_land, int(e["ts"])),
                                  "think_ms": int(e.get("think_ms") or 0),
                                  "team_ms": [float(np.mean(team_think[x][-20:])) if team_think[x] else None
                                              for x in ("T1", "T2")]}
            else:
                st = think_start.pop(s, None)
                team_think[TEAM_OF[s]].append(float(e.get("think_ms") or 0))
                if st is not None:
                    out["thinks"].append({"seat": s, "side": side_of[s], "eval": e.get("eval"),
                                          "next_drop": None, **st})
        elif t == "move":
            s, b = e["seat"], e["board"]
            before = pocket_value(pockets[s])
            drop = bool(e.get("isDrop"))
            piece = None
            if drop:
                piece = e["uci"].split("@")[0].upper()
                if piece in VALUE and pockets[s][piece] > 0:
                    pockets[s][piece] -= 1
                    landed, at_moves = held[s][piece].popleft()
                    out["holds"].append({"seat": s, "side": side_of[s], "piece": piece,
                                         "ms": int(e["ts"]) - landed, "moves": own_moves[s] + 1 - at_moves,
                                         "dropped": True})
            own_moves[s] += 1
            out["moves"].append({"seat": s, "side": side_of[s], "drop": drop, "pocket_before": before,
                                 "frac": ply / max(1, total_plies)})
            for th in reversed(out["thinks"]):  # the decision this move came from
                if th["seat"] == s and th["next_drop"] is None:
                    th["next_drop"] = drop
                    break
            ply += 1
            fens[b] = e["fen"]
            clocks = dict(e["clocks"])
            last_land[b] = int(e["ts"])
            w, bl = fen_pockets(e["fen"])  # the moved board's FEN carries both its pockets: check the tracking
            out["checked"] += 1
            if w != pockets[f"{b}_white"] or bl != pockets[f"{b}_black"]:
                out["mismatch"] += 1
                pockets[f"{b}_white"], pockets[f"{b}_black"] = w, bl  # trust the FEN from here
                for seat, pk in ((f"{b}_white", w), (f"{b}_black", bl)):
                    for p in PIECES:
                        while len(held[seat][p]) > pk[p]:
                            held[seat][p].pop()
                        while len(held[seat][p]) < pk[p]:
                            held[seat][p].append((int(e["ts"]), own_moves[seat]))
            bal = {}
            for side in SIDES:
                ours = [x for x in SEATS if side_of[x] == side]
                theirs = [x for x in SEATS if side_of[x] != side]
                ob = sum(on_board(fens[BOARD_OF[x]], WHITE[x]) for x in ours) - \
                    sum(on_board(fens[BOARD_OF[x]], WHITE[x]) for x in theirs)
                pk = sum(pocket_value(pockets[x]) for x in ours) - sum(pocket_value(pockets[x]) for x in theirs)
                bal[side] = (ob, pk)
            out["balance"].append({"frac": int(e["ts"]) / max(1, int(end["ts"])), **bal})
            last_move = e
    for s in SEATS:  # censored: never dropped
        for p in PIECES:
            for landed, at_moves in held[s][p]:
                out["holds"].append({"seat": s, "side": side_of[s], "piece": p, "ms": int(end["ts"]) - landed,
                                     "moves": own_moves[s] - at_moves, "dropped": False})
    loser = end.get("loserSeat")
    lm = last_move if total_plies else None
    out["end"] = {"reason": end.get("reason"), "loser": loser, "loser_side": side_of.get(loser),
                  "drop_mate": bool(end.get("reason") == "checkmate" and lm is not None and lm.get("isDrop")),
                  "mated_pocket": pocket_value(pockets[loser]) if loser else None,
                  "partner_pocket": pocket_value(pockets[PARTNER[loser]]) if loser else None}
    return out


def _clocks_at(clocks: dict, fens: dict, last_land: dict, ts: int) -> dict:
    """The four clocks at ``ts``: the last landing's clocks minus the running time of the seat on turn per board."""
    out = dict(clocks)
    for b in ("A", "B"):
        seat = f"{b}_white" if fens[b].split()[1] == "w" else f"{b}_black"
        out[seat] = max(0, int(out[seat]) - max(0, ts - last_land[b]))
    return out


def load_games(match_dir: str, max_games: int | None = None) -> list:
    """[(results row, events)] for every finished game under ``match_dir`` (part*/ folders, or the folder itself)."""
    parts = sorted(glob.glob(os.path.join(match_dir, "part*/"))) or [match_dir]
    out = []
    for part in parts:
        rf = os.path.join(part, "results.jsonl")
        if not os.path.exists(rf):
            continue
        for line in open(rf):
            if not line.strip():
                continue
            row = json.loads(line)
            gp = os.path.join(part, row.get("file") or "")
            if not row.get("file") or not os.path.exists(gp):
                continue
            out.append((row, [json.loads(x) for x in open(gp) if x.strip()]))
            if max_games and len(out) >= max_games:
                return out
    return out


# ---------------------------------------------------------------------------------------------- statistics

def boot(per_game: list, stat, n: int = BOOT, seed: int = 7) -> tuple:
    """(estimate, lo, hi): ``stat`` (a list of per-game items -> float or None) and its 95% bootstrap range over
    games."""
    est = stat(per_game)
    if est is None or len(per_game) < 2:
        return est, None, None
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(per_game), len(per_game))
        v = stat([per_game[i] for i in idx])
        if v is not None and not (isinstance(v, float) and math.isnan(v)):
            vals.append(v)
    if not vals:
        return est, None, None
    return est, float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def ratio(num_key, den_key):
    def f(items):
        d = sum(x[den_key] for x in items)
        return sum(x[num_key] for x in items) / d if d else None
    return f


def flat_stat(fn):
    """A per-game list of lists -> fn over the concatenation (None when empty)."""
    def f(items):
        xs = [v for it in items for v in it]
        return fn(xs) if xs else None
    return f


def mean(xs):
    return float(np.mean(xs))


def median(xs):
    return float(np.median(xs))


def p90(xs):
    return float(np.percentile(xs, 90))


def select(games: list, side: str, outcome: str) -> list:
    """Games for ``side`` restricted to that side's own result."""
    if outcome == "all":
        return games
    want = 1 if outcome == "won" else -1
    return [g for g in games if g["outcome"][side] == want]


# ---------------------------------------------------------------------------------------------- tables

def drop_table(games):
    rows = []
    for side in SIDES:
        for oc in OUTCOMES:
            gs = select(games, side, oc)
            per = []
            for g in gs:
                mv = [m for m in g["moves"] if m["side"] == side and m["pocket_before"] > 0]
                d = {"n": len(mv), "d": sum(m["drop"] for m in mv)}
                for i in range(3):
                    sub = [m for m in mv if i / 3 <= m["frac"] < (i + 1) / 3 or (i == 2 and m["frac"] >= 1)]
                    d[f"n{i}"], d[f"d{i}"] = len(sub), sum(m["drop"] for m in sub)
                per.append(d)
            row = {"side": side, "outcome": oc, "games": len(gs), "opportunities": sum(x["n"] for x in per),
                   "drops": sum(x["d"] for x in per), "rate": boot(per, ratio("d", "n"))}
            for i in range(3):
                row[f"third{i + 1}"] = boot(per, ratio(f"d{i}", f"n{i}"))
            rows.append(row)
    return rows


def holding_table(games):
    rows = []
    for side in SIDES:
        for piece in ("all", *PIECES):
            per_ms, per_mv, per_never = [], [], []
            for g in games:
                hs = [h for h in g["holds"] if h["side"] == side and (piece == "all" or h["piece"] == piece)]
                per_ms.append([h["ms"] for h in hs if h["dropped"]])
                per_mv.append([h["moves"] for h in hs if h["dropped"]])
                per_never.append({"n": len(hs), "never": sum(not h["dropped"] for h in hs)})
            n = sum(x["n"] for x in per_never)
            if not n:
                continue
            rows.append({"side": side, "piece": piece, "landed": n,
                         "never": boot(per_never, ratio("never", "n")),
                         "ms_median": boot(per_ms, flat_stat(median)), "ms_mean": boot(per_ms, flat_stat(mean)),
                         "ms_p90": boot(per_ms, flat_stat(p90)),
                         "moves_median": boot(per_mv, flat_stat(median)),
                         "moves_mean": boot(per_mv, flat_stat(mean))})
    return rows


def pocket_table(games):
    rows = []
    for side in SIDES:
        for oc in OUTCOMES:
            gs = select(games, side, oc)
            per = [[t["pocket"] for t in g["thinks"] if t["side"] == side] for g in gs]
            per10 = [{"n": len(x), "k": sum(v >= 10 for v in x)} for x in per]
            team = [[t["team_pocket"] for t in g["thinks"] if t["side"] == side] for g in gs]
            rows.append({"side": side, "outcome": oc, "games": len(gs), "decisions": sum(len(x) for x in per),
                         "seat_pocket": boot(per, flat_stat(mean)), "share_10plus": boot(per10, ratio("k", "n")),
                         "team_pocket": boot(team, flat_stat(mean))})
    return rows


def balance_table(games, fifths=5):
    rows = []
    for side in SIDES:
        for oc in OUTCOMES:
            gs = select(games, side, oc)
            row = {"side": side, "outcome": oc, "games": len(gs)}
            for i in range(fifths):
                inb = [[b[side] for b in g["balance"] if i / fifths <= b["frac"] < (i + 1) / fifths] for g in gs]
                row[f"total{i}"] = boot(inb, flat_stat(lambda xs: mean([a + p for a, p in xs])))[0]
                row[f"pocket{i}"] = boot(inb, flat_stat(lambda xs: mean([p for _, p in xs])))[0]
            per = [[b[side] for b in g["balance"]] for g in gs]
            row["total"] = boot(per, flat_stat(lambda xs: mean([a + p for a, p in xs])))
            row["board"] = boot(per, flat_stat(lambda xs: mean([a for a, _ in xs])))
            row["pocket"] = boot(per, flat_stat(lambda xs: mean([p for _, p in xs])))
            rows.append(row)
    return rows


def calibration_table(games):
    rows = []
    for side in SIDES:
        for _, _, name in POCKET_BINS:
            per = []
            for g in games:
                xs = [(t["eval"], g["outcome"][side]) for t in g["thinks"]
                      if t["side"] == side and t["eval"] is not None and pocket_bin(t["pocket"]) == name]
                per.append(xs)
            n = sum(len(x) for x in per)
            if not n:
                continue
            rows.append({"side": side, "bin": name, "thinks": n,
                         "games": sum(1 for x in per if x),
                         "eval": boot(per, flat_stat(lambda xs: mean([e for e, _ in xs]))),
                         "result": boot(per, flat_stat(lambda xs: mean([r for _, r in xs]))),
                         "gap": boot(per, flat_stat(lambda xs: mean([e - r for e, r in xs])))})
    return rows


def endings_table(games):
    rows = []
    n = len(games)
    for side in SIDES:
        lost = [g for g in games if g["outcome"][side] == -1]
        mates = [g for g in lost if g["end"]["reason"] == "checkmate" and g["end"]["loser_side"] == side]
        per = [{"g": 1, "mate": int(g in mates),
                "flag": int(g["outcome"][side] == -1 and g["end"]["reason"] in ("flag",)),
                "sitoff": int(g["outcome"][side] == -1 and g["end"]["reason"] == "sitoff_adjudicated"),
                "draw": int(g["outcome"][side] == 0),
                "dm_given": int(g["end"]["drop_mate"] and g["end"]["loser_side"] not in (side, None)),
                "dm_recv": int(g["end"]["drop_mate"] and g["end"]["loser_side"] == side)} for g in games]
        mp = [[g["end"]["mated_pocket"]] for g in mates]
        pp = [[g["end"]["partner_pocket"]] for g in mates]
        held5 = [{"n": 1, "k": int(g["end"]["mated_pocket"] >= 5)} for g in mates]
        rows.append({"side": side, "games": n, "lost": len(lost),
                     "mated": boot(per, ratio("mate", "g")), "flagged": boot(per, ratio("flag", "g")),
                     "sitoff": boot(per, ratio("sitoff", "g")), "draws": boot(per, ratio("draw", "g")),
                     "drop_mates_given": boot(per, ratio("dm_given", "g")),
                     "drop_mates_received": boot(per, ratio("dm_recv", "g")),
                     "mated_seat_pocket": boot(mp, flat_stat(mean)) if mates else (None, None, None),
                     "partner_pocket": boot(pp, flat_stat(mean)) if mates else (None, None, None),
                     "mated_holding_5plus": boot(held5, ratio("k", "n")) if mates else (None, None, None)})
    return rows


def score_line(games):
    per = [{"s": (g["outcome"]["us"] + 1) / 2, "n": 1} for g in games]
    return boot(per, ratio("s", "n"))


# ---------------------------------------------------------------------------------------------- network outputs

def network_targets(rows_games: list, model: str, positions: int, binary: str | None = None,
                    engine_options: dict | None = None, seed: int = 11) -> list:
    """Network value and drop-prior mass at sampled decision positions (stratified by side and pocket bin), from the
    engine's ``policy`` command under time options rebuilt from the logged clocks and think times."""
    from arena.players.upstream import (WAVE2_BINARY, WAVE2_ENGINE_DIR, UpstreamHivemindProcess, lead_ms,
                                        seat_values)

    pool = {}
    for g in rows_games:
        for t in g["thinks"]:
            if t["next_drop"] is None:
                continue
            pool.setdefault((t["side"], pocket_bin(t["pocket"])), []).append((g, t))
    rng = np.random.default_rng(seed)
    per_cell = max(1, positions // max(1, len(pool)))
    picked = []
    for key in sorted(pool):
        items = pool[key]
        for i in rng.choice(len(items), size=min(per_cell, len(items)), replace=False):
            picked.append(items[int(i)])
    proc = UpstreamHivemindProcess(binary or WAVE2_BINARY, model, options=engine_options or {}, cwd=WAVE2_ENGINE_DIR,
                                   coreml="off")
    out = []
    try:
        for g, t in picked:
            team = TEAM_OF[t["seat"]]
            other = "T2" if team == "T1" else "T1"
            c = t["clocks"]
            idx = {"T1": 0, "T2": 1}
            tm = t["team_ms"][idx[team]] or max(100, t["think_ms"])
            om = t["team_ms"][idx[other]] or tm
            paces = {s: (tm if TEAM_OF[s] == team else om) for s in SEATS}
            opts = {"SeatClocksMs": seat_values(c), "SeatPaceMs": seat_values(paces),
                    "SeatElapsedMs": "0 0 0 0", "SeatSitting": "0 0 0 0"}
            pol = proc.raw_policy(t["fens"]["A"], t["fens"]["B"], team=team, lead=lead_ms(c, team),
                                  team_move_ms=int(tm), opp_move_ms=int(om), root_move_ms=max(1, t["think_ms"]),
                                  options=opts)
            priors = pol.board_a if BOARD_OF[t["seat"]] == "A" else pol.board_b
            out.append({"side": t["side"], "bin": pocket_bin(t["pocket"]), "value": pol.value,
                        "drop_mass": sum(v for k, v in priors.items() if "@" in k),
                        "pass": priors.get("pass", 0.0), "dropped": bool(t["next_drop"]),
                        "result": g["outcome"][t["side"]], "game": id(g)})
    finally:
        proc.close()
    return out


def network_table(samples: list) -> list:
    rows = []
    for side in SIDES:
        for _, _, name in POCKET_BINS:
            xs = [x for x in samples if x["side"] == side and x["bin"] == name]
            if not xs:
                continue
            by_game: dict[int, list] = {}
            for x in xs:
                by_game.setdefault(x["game"], []).append(x)
            per = list(by_game.values())
            rows.append({"side": side, "bin": name, "positions": len(xs),
                         "value": boot(per, flat_stat(lambda v: mean([x["value"] for x in v]))),
                         "result": boot(per, flat_stat(lambda v: mean([x["result"] for x in v]))),
                         "drop_mass": boot(per, flat_stat(lambda v: mean([x["drop_mass"] for x in v]))),
                         "dropped": boot(per, flat_stat(lambda v: mean([float(x["dropped"]) for x in v])))})
    return rows


# ---------------------------------------------------------------------------------------------- report

def analyse(match_dir: str, max_games: int | None = None) -> dict:
    games = []
    for row, events in load_games(match_dir, max_games):
        g = game_records(events, row["contender_team"])
        if g is not None:
            games.append(g)
    return {"dir": match_dir, "games": games, "n": len(games), "score": score_line(games),
            "mismatch": sum(g["mismatch"] for g in games), "checked": sum(g["checked"] for g in games),
            "drop": drop_table(games), "holding": holding_table(games), "pocket": pocket_table(games),
            "balance": balance_table(games), "calibration": calibration_table(games),
            "endings": endings_table(games)}


def fmt(x, digits=3, pct=False) -> str:
    """``est [lo, hi]`` (or ``est`` / ``-``)."""
    if isinstance(x, tuple):
        est, lo, hi = x
    else:
        est, lo, hi = x, None, None
    if est is None:
        return "-"
    f = (lambda v: f"{100 * v:.{max(0, digits - 2)}f}%") if pct else (lambda v: f"{v:.0f}"
                                                                         if abs(v) >= 100 else f"{v:.{digits}f}")
    if lo is None:
        return f(est)
    return f"{f(est)} [{f(lo)}, {f(hi)}]"


def table(header: list, rows: list) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def render(rep: dict, us: str = "us", them: str = "them") -> str:
    name = {"us": us, "them": them}
    s = [f"Match `{rep['dir']}`: {rep['n']} games, {us} score {fmt(rep['score'])}. Pocket tracking agreed with the "
         f"moved board's FEN at {rep['checked'] - rep['mismatch']} of {rep['checked']} moves."]
    s.append("\n**1. Drop rate** (drops / own moves made with a non-empty pocket; thirds of the game by ply)\n")
    s.append(table(["side", "result", "games", "moves w/ pocket", "drops", "rate", "1st third", "2nd third",
                    "3rd third"],
                   [[name[r["side"]], r["outcome"], r["games"], r["opportunities"], r["drops"], fmt(r["rate"]),
                     fmt(r["third1"]), fmt(r["third2"]), fmt(r["third3"])] for r in rep["drop"]]))
    s.append("\n**2. Holding** (per piece landed in a seat's pocket; time and own moves until that seat drops it; "
             "FIFO per piece type; dropped pieces only for ms/moves)\n")
    s.append(table(["side", "piece", "landed", "never dropped", "ms median", "ms mean", "ms p90", "moves median",
                    "moves mean"],
                   [[name[r["side"]], r["piece"], r["landed"], fmt(r["never"]), fmt(r["ms_median"], 0),
                     fmt(r["ms_mean"], 0), fmt(r["ms_p90"], 0), fmt(r["moves_median"], 1), fmt(r["moves_mean"], 2)]
                    for r in rep["holding"]]))
    s.append("\n**3a. Pocket value at own decisions** (P1 N3 B3 R5 Q9; the deciding seat's pocket, and its team's "
             "two pockets)\n")
    s.append(table(["side", "result", "games", "decisions", "seat pocket", "share >= 10", "team pockets"],
                   [[name[r["side"]], r["outcome"], r["games"], r["decisions"], fmt(r["seat_pocket"], 2),
                     fmt(r["share_10plus"]), fmt(r["team_pocket"], 2)] for r in rep["pocket"]]))
    s.append("\n**3b. Material balance** (team on-board + pockets minus the other team's, at every move; mean over "
             "the game, and the total by fifth of game time; pocket part in brackets)\n")
    s.append(table(["side", "result", "total", "on board", "pockets"] + [f"fifth {i + 1}" for i in range(5)],
                   [[name[r["side"]], r["outcome"], fmt(r["total"], 2), fmt(r["board"], 2), fmt(r["pocket"], 2)]
                    + [f"{r[f'total{i}']:+.1f} ({r[f'pocket{i}']:+.1f})" if r[f"total{i}"] is not None else "-"
                       for i in range(5)] for r in rep["balance"]]))
    s.append("\n**4. Calibration** (search eval at own think ends, thinker's view, vs the game result +1/0/-1, by "
             "the thinking seat's pocket value)\n")
    s.append(table(["side", "pocket", "thinks", "games", "mean eval", "mean result", "eval - result"],
                   [[name[r["side"]], r["bin"], r["thinks"], r["games"], fmt(r["eval"]), fmt(r["result"]),
                     fmt(r["gap"])] for r in rep["calibration"]]))
    s.append("\n**5. Endings** (shares of all games; the mated seat's and its partner's pocket at the mate)\n")
    s.append(table(["side", "lost", "mated", "flagged", "sit-off", "draws", "drop-mates given", "drop-mates received",
                    "mated seat pocket", "its partner's pocket", "mated holding >= 5"],
                   [[name[r["side"]], r["lost"], fmt(r["mated"]), fmt(r["flagged"]), fmt(r["sitoff"]),
                     fmt(r["draws"]), fmt(r["drop_mates_given"]), fmt(r["drop_mates_received"]),
                     fmt(r["mated_seat_pocket"], 2), fmt(r["partner_pocket"], 2), fmt(r["mated_holding_5plus"])]
                    for r in rep["endings"]]))
    if rep.get("network"):
        s.append(f"\n**6. Network outputs** (`{rep['network']['model']}`, {len(rep['network']['samples'])} decision "
                 "positions stratified by side and pocket bin: value for the deciding team, prior mass on drops on "
                 "the deciding seat's board; against the result and whether that decision dropped)\n")
        s.append(table(["side", "pocket", "positions", "value", "result", "drop prior mass", "dropped"],
                       [[name[r["side"]], r["bin"], r["positions"], fmt(r["value"]), fmt(r["result"]),
                         fmt(r["drop_mass"]), fmt(r["dropped"])] for r in rep["network"]["table"]]))
    return "\n".join(s) + "\n"


def summary_line(rep: dict) -> dict:
    """The few headline numbers of a match (for reference rows)."""
    d = {r["side"]: r for r in rep["drop"] if r["outcome"] == "all"}
    p = {r["side"]: r for r in rep["pocket"] if r["outcome"] == "all"}
    h = {r["side"]: r for r in rep["holding"] if r["piece"] == "all"}
    b = {r["side"]: r for r in rep["balance"] if r["outcome"] == "all"}
    return {side: {"drop_rate": d[side]["rate"], "seat_pocket": p[side]["seat_pocket"],
                   "hold_ms_median": h[side]["ms_median"], "never": h[side]["never"], "balance": b[side]["total"]}
            for side in SIDES}


def render_refs(refs: list) -> str:
    rows = []
    for label, rep in refs:
        sl = summary_line(rep)
        for side in SIDES:
            rows.append([label, side, rep["n"], fmt(rep["score"]) if side == "us" else "", fmt(sl[side]["drop_rate"]),
                         fmt(sl[side]["seat_pocket"], 2), fmt(sl[side]["hold_ms_median"], 0),
                         fmt(sl[side]["never"]), fmt(sl[side]["balance"], 2)])
    return table(["match", "side", "games", "contender score", "drop rate", "seat pocket", "hold ms median",
                  "never dropped", "material balance"], rows) + "\n"


def _jsonable(rep: dict) -> dict:
    return {k: v for k, v in rep.items() if k != "games"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("match", help="match folder with part*/ (contender = results row's contender_team)")
    ap.add_argument("--us", default="contender", help="label of the contender's side")
    ap.add_argument("--them", default="opponent", help="label of the opponent's side")
    ap.add_argument("--ref", action="append", default=[], help="NAME=DIR reference match (headline numbers)")
    ap.add_argument("--out", help="write the markdown tables here (default: stdout)")
    ap.add_argument("--json", help="write the numbers as JSON here")
    ap.add_argument("--max-games", type=int, help="read at most this many games per match (quick checks)")
    ap.add_argument("--model", help="ONNX: also read the network's value and drop priors (item 6)")
    ap.add_argument("--binary", help="wave 2 engine binary (default: arena WAVE2_BINARY)")
    ap.add_argument("--positions", type=int, default=600, help="decision positions for --model")
    args = ap.parse_args(argv)

    rep = analyse(args.match, args.max_games)
    if not rep["n"]:
        print(f"no finished games under {args.match}", file=sys.stderr)
        return 1
    if args.model:
        from arena.players.upstream import WAVE2_FLAGS, WAVE2_PRESETS, wave2_flag_options

        kw = {}
        spec = _first_spec(args.match)
        if spec:  # the contender's wave 2 flags, as the match ran them
            pk = spec["teams"][spec["contender_team"]]["player_kw"]
            kw = {k: v for k, v in pk.items() if k in WAVE2_FLAGS or k in WAVE2_PRESETS}
        samples = network_targets(rep["games"], args.model, args.positions, args.binary, wave2_flag_options(**kw))
        rep["network"] = {"model": args.model, "samples": samples, "table": network_table(samples)}
    text = render(rep, args.us, args.them)
    refs = []
    for r in args.ref:
        label, _, d = r.partition("=")
        rr = analyse(d, args.max_games)
        if rr["n"]:
            refs.append((label, rr))
    if refs:
        text += "\n**References** (contender = us, opponent = them)\n\n" + render_refs([(args.us, rep)] + refs)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text)
    else:
        print(text)
    if args.json:
        out = {"match": _jsonable(rep), "refs": {label: _jsonable(r) for label, r in refs}}
        if rep.get("network"):
            out["match"]["network"] = {"model": args.model, "table": rep["network"]["table"]}
        with open(args.json, "w") as f:
            json.dump(out, f, indent=1, default=str)
    return 0


def _first_spec(match_dir: str) -> dict | None:
    for part in sorted(glob.glob(os.path.join(match_dir, "part*/"))) or [match_dir]:
        sp = os.path.join(part, "specs.jsonl")
        if os.path.exists(sp):
            for line in open(sp):
                if line.strip():
                    return json.loads(line)
    return None


if __name__ == "__main__":
    sys.exit(main())
