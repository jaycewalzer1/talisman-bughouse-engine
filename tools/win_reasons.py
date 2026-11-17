"""How the contender's games ended, per eval run: wins and losses by checkmate and by flag, and how much clock
each side spent.

    .venv/bin/python tools/win_reasons.py runs/beat/c50_v100 runs/beat/clock110_v100 ...

Reads <run>/summary.json. "think/move" is average simulated think time per completed search; "sits/100" is
stalls per 100 decisions. A win by flag means the opponent's clock ran out, so a faster contender that wins
mostly by checkmate is winning on the board, not by flagging.
"""

from __future__ import annotations

import json
import os
import sys


def row(run: str) -> str:
    path = os.path.join(run, "summary.json")
    name = os.path.basename(run.rstrip("/"))
    if not os.path.exists(path):
        return f"{name:<22} no result yet"
    games = [g for g in json.load(open(path))["games"] if g.get("result")]
    win = {"checkmate": 0, "flag": 0, "other": 0}
    loss = {"checkmate": 0, "flag": 0, "other": 0}
    draws = 0
    think = {"us": [0, 0], "them": [0, 0]}  # think ms, decisions
    sits = {"us": 0, "them": 0}
    for g in games:
        us = g["contender_team"]
        them = "T2" if us == "T1" else "T1"
        reason = g["reason"] if g["reason"] in ("checkmate", "flag") else "other"
        if g["result"] == "draw":
            draws += 1
        elif g["result"] == us:
            win[reason] += 1
        else:
            loss[reason] += 1
        for side, team in (("us", us), ("them", them)):
            t = g["teams"][team]
            think[side][0] += t.get("think_ms", 0)
            think[side][1] += t.get("decisions", 0)
            sits[side] += t.get("stalls", 0)
    per = {s: think[s][0] / max(1, think[s][1]) for s in think}
    sit = {s: 100 * sits[s] / max(1, think[s][1]) for s in sits}
    w, l = sum(win.values()), sum(loss.values())
    return (f"{name:<22} {len(games):>5} {w:>5} {win['checkmate']:>6} {win['flag']:>5} {l:>5} {loss['checkmate']:>6} "
            f"{loss['flag']:>5} {draws:>5}   {per['us']:>6.0f} {per['them']:>6.0f}   {sit['us']:>5.1f} {sit['them']:>5.1f}")


def main(argv: list[str]) -> int:
    print(f"{'run':<22} {'games':>5} {'wins':>5} {'mate':>6} {'flag':>5} {'loss':>5} {'mate':>6} {'flag':>5} "
          f"{'draws':>5}   {'think/move ms':>13}   {'sits/100':>11}")
    print(f"{'':<22} {'':>5} {'':>5} {'':>6} {'':>5} {'':>5} {'':>6} {'':>5} {'':>5}   {'us':>6} {'them':>6}   "
          f"{'us':>5} {'them':>5}")
    for run in argv:
        print(row(run))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
