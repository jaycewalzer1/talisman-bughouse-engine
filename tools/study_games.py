#!/usr/bin/env python3
"""Game study: per-game features and key moments for the study page (viewer/study.html).

Reads arena match folders (results.jsonl + game_*.jsonl per part) under runs/study/ (synced from PACE) and writes
  runs/study/study.json            groups (one per match row) with aggregate features, wins and losses split
  runs/study/games_<group>.json    one row per game: features + key moments, from the contender's side ("us")

Everything is from the contender's side: our team = the results row's contender_team. Evals are each engine's own
search eval at the end of a think, turned to our side ([-1, 1], +1 = we win). Key moments per game:
  turn      the last time the mean of both engines' latest evals crosses 0 and stays on the final side
  swing     the biggest moves of our engine's eval between two of its searches (drops in losses, rises in wins),
            with the moves, captures and piece passes that happened in between
  mate      the final move: board, piece, drop or not, whose seat was mated, clocks and the clock lead then
  sit       every sit (stall) with its seat, start and length
Plus per-game counts that bear on fairness: stale searches (the thinking seat's or its opponent's pocket changed
between think start and end), turns ahead on the clock (from results.jsonl), think time per decision.

Usage: python3 tools/study_games.py [runs/study]
"""
import glob
import json
import math
import os
import re
import sys
from collections import defaultdict

VALUE = {"p": 1, "n": 3, "b": 3, "r": 5, "q": 9, "k": 0}
SEATS = ("A_white", "A_black", "B_white", "B_black")
BOARD_OF = {s: s[0] for s in SEATS}
WHITE = {"A_white": True, "A_black": False, "B_white": True, "B_black": False}
OPP_ON_BOARD = {"A_white": "A_black", "A_black": "A_white", "B_white": "B_black", "B_black": "B_white"}


def material(fen, white):
    """(on-board material, pocket material) of one side from a bughouse FEN with [pocket]."""
    board, _, rest = fen.partition(" ")
    m = re.match(r"([^\[]*)(?:\[([^\]]*)\])?", board)
    placement, pocket = m.group(1), m.group(2) or ""
    on = sum(VALUE[c.lower()] for c in placement if c.isalpha() and c.isupper() == white)
    poc = sum(VALUE[c.lower()] for c in pocket if c.isalpha() and c.isupper() == white)
    return on, poc


def wilson(score, n):
    if n == 0:
        return [0, 1]
    z = 1.96
    c = (score + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(score * (1 - score) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(c - h, 3), round(c + h, 3)]


def study_game(path, row):
    ev = [json.loads(line) for line in open(path) if line.strip()]
    start = next(e for e in ev if e["type"] == "game_start")
    seats = start["seats"]
    us = row["contender_team"]
    ours = {s for s in SEATS if seats[s]["team"] == us}
    def k_of(team_seats):
        name = seats[sorted(team_seats)[0]]["name"]
        m = re.search(r"-k([\d.]+)$", name)
        return float(m.group(1)) if m else None
    clocks = dict(start["clocks"])
    fens = dict(start["fens"])
    last_eval = {}                       # seat -> eval on our side
    our_series, their_series, mean_series = [], [], []
    between = []                         # events since our engine's last search end
    swings = []
    think_start = {}
    stale = {"us": 0, "them": 0}
    searches = {"us": 0, "them": 0}
    pocket_changed_at = {s: -1 for s in SEATS}   # last ts a seat's pocket changed
    sits, open_sit = [], {}
    mat_samples = []                     # (ts, our total, their total, our pocket, their pocket)
    lead_samples = []                    # (ts, our clocks - their clocks)
    moves = {"us": 0, "them": 0}
    drops = {"us": 0, "them": 0}
    end = None
    last_move = None
    for e in ev:
        t = e["type"]
        if t == "thinking":
            s = e["seat"]
            side = "us" if s in ours else "them"
            if e["phase"] == "start":
                think_start[s] = e["ts"]
            else:
                searches[side] += 1
                ts0 = think_start.get(s, e["ts"])
                if pocket_changed_at[s] > ts0 or pocket_changed_at[OPP_ON_BOARD[s]] > ts0:
                    stale[side] += 1
                if e.get("eval") is not None:
                    v = e["eval"] if s in ours else -e["eval"]
                    last_eval[s] = v
                    if side == "us":
                        if our_series:
                            swings.append({"ts0": our_series[-1][0], "ts": e["ts"], "from": round(our_series[-1][1], 3),
                                           "to": round(v, 3), "delta": round(v - our_series[-1][1], 3),
                                           "seat": s, "events": between[-12:]})
                        between = []
                        our_series.append((e["ts"], v))
                    else:
                        their_series.append((e["ts"], v))
                    o = [last_eval[x] for x in last_eval if x in ours]
                    th = [last_eval[x] for x in last_eval if x not in ours]
                    if o and th:
                        mean_series.append((e["ts"], (o[-1] + th[-1]) / 2))
        elif t == "move":
            s = e["seat"]
            side = "us" if s in ours else "them"
            moves[side] += 1
            drops[side] += bool(e.get("isDrop"))
            if e.get("isDrop"):
                pocket_changed_at[s] = e["ts"]
            fens[e["board"]] = e["fen"]
            clocks = e["clocks"]
            between.append({"ts": e["ts"], "k": "move", "us": s in ours, "board": e["board"], "san": e["san"],
                            "cap": e.get("capture")})
            last_move = e
            ot = ou = tt = tu = 0
            for seat in SEATS:
                on, poc = material(fens[BOARD_OF[seat]], WHITE[seat])
                if seat in ours:
                    ot += on + poc; ou += poc
                else:
                    tt += on + poc; tu += poc
            mat_samples.append((e["ts"], ot, tt, ou, tu))
            lead_samples.append((e["ts"], sum(clocks[x] for x in ours) - sum(clocks[x] for x in SEATS if x not in ours)))
        elif t == "pass":
            pocket_changed_at[e["toSeat"]] = e["ts"]
            between.append({"ts": e["ts"], "k": "pass", "us": e["toSeat"] in ours, "piece": e["piece"],
                            "to": e["toSeat"]})
        elif t == "stall":
            s = e["seat"]
            if e.get("active"):
                open_sit[s] = (e["ts"], e.get("reason"))
            elif s in open_sit:
                t0, why = open_sit.pop(s)
                sits.append({"seat": s, "us": s in ours, "ts": t0, "ms": e["ts"] - t0, "why": why})
                between.append({"ts": t0, "k": "sit", "us": s in ours, "seat": s, "ms": e["ts"] - t0})
        elif t == "game_end":
            end = e
    if end is None:
        return None
    for s, (t0, why) in open_sit.items():
        sits.append({"seat": s, "us": s in ours, "ts": t0, "ms": end["ts"] - t0, "why": why, "open": True})

    score = 1.0 if end["result"] == us else (0.5 if end["result"] in ("draw", None, "") else 0.0)
    if end["result"] not in ("T1", "T2"):
        score = 0.5
    final_sign = 1 if score > 0.5 else (-1 if score < 0.5 else 0)

    turn = None
    if final_sign and mean_series:
        for i in range(len(mean_series) - 1, -1, -1):
            if mean_series[i][1] * final_sign <= 0:
                nxt = mean_series[i + 1] if i + 1 < len(mean_series) else mean_series[i]
                turn = {"ts": nxt[0], "frac": round(nxt[0] / max(1, end["ts"]), 3)}
                break
        else:
            turn = {"ts": mean_series[0][0], "frac": 0.0}
    key_swings = sorted(swings, key=lambda w: w["delta"] * (final_sign or -1))[:3] if swings else []
    key_swings = [w for w in key_swings if abs(w["delta"]) >= 0.15]

    mate = None
    if last_move is not None:
        loser = end.get("loserSeat")
        mate = {"ts": end["ts"], "board": end.get("board"), "san": last_move["san"], "drop": bool(last_move.get("isDrop")),
                "by_us": last_move["seat"] in ours, "loser": loser, "loser_us": loser in ours if loser else None,
                "reason": end["reason"], "clocks": last_move["clocks"],
                "lead_ms": lead_samples[-1][1] if lead_samples else 0}

    team_rows = row.get("teams", {})
    themt = "T1" if us == "T2" else "T2"
    ct = row.get("clock_teams", {})
    def frac_ahead(team):
        c = ct.get(team) or {}
        n = (c.get("turns_ahead") or 0) + (c.get("turns_behind") or 0)
        return round((c.get("turns_ahead") or 0) / n, 3) if n else None
    def avg(xs):
        return round(sum(xs) / len(xs), 3) if xs else None
    dur = end["ts"]
    return {
        "file": os.path.relpath(path),
        "game": row["game_id"], "pair": row.get("pair_index"), "opening": row.get("opening"), "swap": row.get("swap"),
        "us_team": us, "k_us": k_of(ours), "k_them": k_of(set(SEATS) - ours),
        "score": score, "reason": end["reason"], "board": end.get("board"), "plies": row.get("plies"),
        "dur_s": round(dur / 1000, 1),
        "think_per_dec_us": round(team_rows.get(us, {}).get("think_ms", 0) / max(1, team_rows.get(us, {}).get("decisions", 1))),
        "think_per_dec_them": round(team_rows.get(themt, {}).get("think_ms", 0) / max(1, team_rows.get(themt, {}).get("decisions", 1))),
        "nodes_us": team_rows.get(us, {}).get("nodes"), "nodes_them": team_rows.get(themt, {}).get("nodes"),
        "ahead_us": frac_ahead(us), "ahead_them": frac_ahead(themt),
        "sits_us": sum(1 for x in sits if x["us"]), "sits_them": sum(1 for x in sits if not x["us"]),
        "sit_s_us": round(sum(x["ms"] for x in sits if x["us"]) / 1000, 1),
        "sit_s_them": round(sum(x["ms"] for x in sits if not x["us"]) / 1000, 1),
        "moves_us": moves["us"], "moves_them": moves["them"], "drops_us": drops["us"], "drops_them": drops["them"],
        "stale_us": round(stale["us"] / max(1, searches["us"]), 3), "stale_them": round(stale["them"] / max(1, searches["them"]), 3),
        "mat_diff": avg([a - b for _, a, b, _, _ in mat_samples]),
        "pocket_us": avg([c for *_, c, _ in mat_samples]), "pocket_them": avg([d for *_, d in mat_samples]),
        "lead_mean_s": round((avg([x for _, x in lead_samples]) or 0) / 1000, 2),
        "lead_end_s": round((lead_samples[-1][1] if lead_samples else 0) / 1000, 2),
        "clock_left_us_s": round(sum(mate["clocks"][x] for x in ours) / 2000, 1) if mate else None,
        "clock_left_them_s": round(sum(mate["clocks"][x] for x in SEATS if x not in ours) / 2000, 1) if mate else None,
        "turn": turn, "swings": key_swings, "mate": mate, "sits": sits[:40],
        "curve_us": [(round(a / 1000, 2), round(b, 3)) for a, b in our_series[::max(1, len(our_series) // 120)]],
        "curve_them": [(round(a / 1000, 2), round(b, 3)) for a, b in their_series[::max(1, len(their_series) // 120)]],
        "lead_curve": [(round(a / 1000, 2), round(b / 1000, 2)) for a, b in lead_samples[::max(1, len(lead_samples) // 120)]],
        "mat_curve": [(round(a / 1000, 2), b - c) for a, b, c, _, _ in mat_samples[::max(1, len(mat_samples) // 120)]],
    }


NUMERIC = ("dur_s", "plies", "think_per_dec_us", "think_per_dec_them", "ahead_us", "ahead_them", "sits_us", "sits_them",
           "sit_s_us", "sit_s_them", "moves_us", "moves_them", "drops_us", "drops_them", "stale_us", "stale_them",
           "mat_diff", "pocket_us", "pocket_them", "lead_mean_s", "lead_end_s", "clock_left_us_s", "clock_left_them_s")


def aggregate(games):
    out = {"n": len(games)}
    if not games:
        return out
    s = sum(g["score"] for g in games) / len(games)
    out["score"] = round(s, 3)
    out["ci"] = wilson(s, len(games))
    for k in NUMERIC:
        xs = [g[k] for g in games if g.get(k) is not None]
        out[k] = round(sum(xs) / len(xs), 3) if xs else None
    out["mate_on_our_seat"] = round(sum(1 for g in games if g["mate"] and g["mate"]["loser_us"]) / len(games), 3)
    out["mate_by_drop"] = round(sum(1 for g in games if g["mate"] and g["mate"]["drop"]) / len(games), 3)
    out["turn_frac"] = round(sum(g["turn"]["frac"] for g in games if g["turn"]) / max(1, sum(1 for g in games if g["turn"])), 3)
    reasons = defaultdict(int)
    for g in games:
        reasons[g["reason"]] += 1
    out["reasons"] = dict(reasons)
    return out


def groups(root):
    """(group id, label, list of part folders) for every match row found."""
    found = []
    for cand in sorted(glob.glob(os.path.join(root, "tune/*/*/"))):
        rnd, name = cand.rstrip("/").split("/")[-2:]
        if rnd == "chk1" or glob.glob(os.path.join(cand, "part*/results.jsonl")):
            found.append((f"{rnd}/{name}", f"tune {rnd}: {name}", sorted(glob.glob(os.path.join(cand, "part*/")))))
    for row in sorted(glob.glob(os.path.join(root, "wave2c/r4/long/*/"))):
        name = row.rstrip("/").split("/")[-1]
        found.append((f"long/{name}", f"r4 long test: {name}", sorted(glob.glob(os.path.join(row, "part*/")))))
    for a in sorted(glob.glob(os.path.join(root, "full/*/anchors/it*/*/"))):
        parts = a.rstrip("/").split("/")
        found.append((f"anchor/{parts[-4]}-{parts[-2]}-{parts[-1]}", f"{parts[-4]} {parts[-2]} vs {parts[-1]} (anchor)",
                      sorted(glob.glob(os.path.join(a, "part*/")))))
    return found


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "runs/study"
    summary = []
    for gid, label, parts in groups(root):
        games = []
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
                try:
                    g = study_game(gp, row)
                except Exception as err:  # a malformed game is reported, not fatal
                    print(f"skip {gp}: {err!r}", file=sys.stderr)
                    continue
                if g:
                    games.append(g)
        if not games:
            continue
        safe = gid.replace("/", "__")
        with open(os.path.join(root, f"games_{safe}.json"), "w") as f:
            json.dump(games, f, separators=(",", ":"))
        ks = sorted({(g["k_us"], g["k_them"]) for g in games})
        summary.append({"id": gid, "file": f"games_{safe}.json", "label": label, "k": ks,
                        "all": aggregate(games), "wins": aggregate([g for g in games if g["score"] == 1]),
                        "losses": aggregate([g for g in games if g["score"] == 0])})
        a = summary[-1]["all"]
        print(f"{gid:40s} n={a['n']:5d} score={a['score']:.3f} ahead_us={a['ahead_us']} sits_us={a['sits_us']} "
              f"sits_them={a['sits_them']} stale_us={a['stale_us']} stale_them={a['stale_them']}")
    with open(os.path.join(root, "study.json"), "w") as f:
        json.dump({"groups": summary}, f, indent=1)


if __name__ == "__main__":
    main()
