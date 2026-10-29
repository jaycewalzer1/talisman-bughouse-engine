"""Wait-regret probe: at decision points from recorded games, would sitting have beaten moving?

    python -m arena.probe_wait runs/sharing_vs_silent --kind hivemind --nodes 200 \
        --waits 2000 5000 --positions 4 --rollouts 4 --workers 3 --out runs/probe_wait

For every sampled decision point (a seat starting to think about a move) the recorded game is replayed up
to that moment and forked. From the fork the game is played out several times per branch with fresh
players of the same setups:

* ``play``: everyone plays normally, as in the recording.
* ``wait_<W>``: the probed seat first sits for up to W ms (it stops early when a piece lands in its own
  pocket), then plays normally.

The probed team's mean score per branch, and the difference wait - play, says how much a forced wait was
worth there. By default only decision points where the seat had asked its partner for a piece
(a ``need_X`` message within ``--need-window-ms``) are probed: the moments when waiting is plausible.
``--any`` samples any decision point instead.

Approximations: the fork restarts every seat's decision (searches in flight are dropped), and players
start fresh, so messages in flight and remembered partner requests are lost. Both branches share these
approximations. Hivemind searches are multi-threaded, which is where rollout-to-rollout variation comes
from; heuristic and random players vary with the rollout seed.

Writes ``<out>/rollouts.jsonl`` (one line per rollout) and ``<out>/summary.json`` and prints a table.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

from arena.interfaces import BOARD_OF, COLOR_OF, TEAM_OF, Decision, GameResult, Observation

MAX_PLIES = 400


# ---------------------------------------------------------------------------------------------- recordings

def load_events(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def team_setups(game_start: dict) -> tuple[str, str]:
    """(T1 setup, T2 setup) from game_start.setup ("silent" or "T1=sharing,T2=silent")."""
    setup = game_start["setup"]
    if "=" not in setup:
        return setup, setup
    parts = dict(p.split("=", 1) for p in setup.split(","))
    return parts["T1"], parts["T2"]


def replay(events: list[dict], until_seq: int):
    """BughouseState with every move before event ``until_seq`` applied, clocks run to that event's ts."""
    from arena.state import BughouseState

    start = events[0]
    state = BughouseState(start["base_ms"], start.get("increment_ms", 0) or 0, fens=start["fens"])
    for e in events[:until_seq]:
        if e["type"] == "move":
            state.apply_move(e["seat"], e["uci"], e["ts"])
            if state.clocks_ms != e["clocks"]:
                raise ValueError(f"replay diverged at seq {e['seq']}: {state.clocks_ms} != {e['clocks']}")
    state.advance_clocks(events[until_seq]["ts"])
    return state


def candidate_points(events: list[dict], *, need_window_ms: int, any_point: bool, min_ply: int) -> list[int]:
    """seqs of thinking-start events worth probing."""
    out = []
    plies = 0
    last_need: dict[str, int] = {}
    for e in events:
        if e["type"] == "move":
            plies += 1
        elif e["type"] == "message" and e["token"].startswith("need_"):
            last_need[e["fromSeat"]] = e["ts"]
        elif e["type"] == "thinking" and e["phase"] == "start" and plies >= min_ply:
            asked = e["seat"] in last_need and e["ts"] - last_need[e["seat"]] <= need_window_ms
            if any_point or asked:
                out.append(e["seq"])
    return out


def own_pocket(fen: str, seat: str) -> str:
    pocket = fen.split("[", 1)[1].split("]", 1)[0] if "[" in fen else ""
    white = COLOR_OF[seat] == "white"
    return "".join(sorted(c for c in pocket if c.isupper() == white))


# ---------------------------------------------------------------------------------------------- forced wait

class ForcedWait:
    """Wraps a player: sit until ``until_ms`` or until a new piece lands in the seat's own pocket, then
    hand every decision to the wrapped player."""

    def __init__(self, inner, until_ms: int, pocket: str):
        self.inner = inner
        self.seat = inner.seat
        self.name = inner.name + "-probe"
        self.kind = inner.kind
        self.until_ms = until_ms
        self.pocket = pocket
        self.done = False

    def decide(self, obs: Observation) -> Decision:
        if not self.done:
            got_piece = own_pocket(obs.fens[BOARD_OF[self.seat]], self.seat) != self.pocket
            if obs.now_ms < self.until_ms and not got_piece and obs.legal_moves:
                return Decision(kind="wait", wait_ms=self.until_ms - obs.now_ms, stall_reason="probe_wait")
            self.done = True
        return self.inner.decide(obs)

    def close(self) -> None:
        self.inner.close()


# ---------------------------------------------------------------------------------------------- rollouts

def rollout(task: dict) -> dict:
    """Play one rollout from a fork. Returns the task plus the probed team's score."""
    from arena.run import build_players, setup_label
    from arena.sim import Simulator

    events = load_events(task["file"])
    fork = events[task["seq"]]
    state = replay(events, task["seq"])
    plies = sum(1 for e in events[:task["seq"]] if e["type"] == "move")
    t1, t2 = team_setups(events[0])
    players, created = build_players(t1, t2, task["kind"], seed=task["seed"], nodes=task["nodes"],
                                     player_kw=task["player_kw"])
    seat = fork["seat"]
    if task["wait_ms"] > 0:
        players[seat] = ForcedWait(players[seat], fork["ts"] + task["wait_ms"],
                                   own_pocket(state.fen(BOARD_OF[seat]), seat))
    t0 = time.perf_counter()
    try:
        sim = Simulator(state, players, game_id=f"probe-{task['id']}", setup=setup_label(t1, t2),
                        channel_delay_ms=events[0].get("channel_delay_ms", 0) or 0,
                        max_plies=max(1, MAX_PLIES - plies))
        res: GameResult = sim.run()
    finally:
        for p in created:
            try:
                p.close()
            except Exception:  # noqa: BLE001 - best effort
                pass
    team = TEAM_OF[seat]
    score = 0.5 if res.result == "draw" else float(res.result == team)
    return {**{k: task[k] for k in ("id", "file", "seq", "branch", "wait_ms", "rollout")},
            "seat": seat, "fork_ts": fork["ts"], "score": score, "result": res.result, "reason": res.reason,
            "end_ts": sim.now, "wall_s": round(time.perf_counter() - t0, 2)}


def make_tasks(args) -> list[dict]:
    files = []
    for src in args.games:
        files += sorted(glob.glob(os.path.join(src, "game_*.jsonl"))) if os.path.isdir(src) else [src]
    rng = random.Random(args.seed)
    tasks = []
    branches = [("play", 0)] + [(f"wait_{w}", w) for w in args.waits]
    player_kw = {"compute": args.compute, "lend": args.lend}
    for path in files:
        events = load_events(path)
        if events[-1].get("reason") == "aborted":
            continue
        points = candidate_points(events, need_window_ms=args.need_window_ms, any_point=args.any,
                                  min_ply=args.min_ply)
        for seq in sorted(rng.sample(points, min(args.positions, len(points)))):
            pid = f"{os.path.basename(path)}@{seq}"
            for branch, wait_ms in branches:
                for k in range(args.rollouts):
                    tasks.append({"id": f"{pid}/{branch}/{k}", "position": pid, "file": path, "seq": seq,
                                  "branch": branch, "wait_ms": wait_ms, "rollout": k, "kind": args.kind,
                                  "nodes": args.nodes, "player_kw": player_kw,
                                  "seed": args.seed * 100003 + len(tasks)})
    return tasks


# ---------------------------------------------------------------------------------------------- summary

def summarize(rows: list[dict], waits: list[int], n_boot: int = 2000, seed: int = 0) -> dict:
    """Per position: mean score per branch. Overall: mean (wait - play) per W with a bootstrap 95% CI over
    positions, and how often the wait branch scored higher."""
    by_pos: dict[str, dict[str, list[float]]] = {}
    for r in rows:
        pid = f"{os.path.basename(r['file'])}@{r['seq']}"
        by_pos.setdefault(pid, {}).setdefault(r["branch"], []).append(r["score"])
    positions = {pid: {b: sum(v) / len(v) for b, v in br.items()} for pid, br in by_pos.items()}
    rng = random.Random(seed)
    overall = {}
    for w in waits:
        b = f"wait_{w}"
        diffs = [p[b] - p["play"] for p in positions.values() if b in p and "play" in p]
        if not diffs:
            continue
        boots = sorted(sum(rng.choice(diffs) for _ in diffs) / len(diffs) for _ in range(n_boot))
        overall[b] = {"positions": len(diffs), "mean_diff": round(sum(diffs) / len(diffs), 4),
                      "ci95": [round(boots[int(0.025 * n_boot)], 4), round(boots[int(0.975 * n_boot) - 1], 4)],
                      "wait_better": sum(d > 0 for d in diffs), "play_better": sum(d < 0 for d in diffs)}
    return {"positions": positions, "overall": overall}


def format_summary(summary: dict) -> str:
    lines = [f"{'branch':<12} {'positions':>9} {'wait-play':>10} {'95% CI':>18} {'wait>play':>10} {'play>wait':>10}"]
    for b, o in summary["overall"].items():
        ci = f"[{o['ci95'][0]:+.3f}, {o['ci95'][1]:+.3f}]"
        lines.append(f"{b:<12} {o['positions']:>9} {o['mean_diff']:>+10.3f} {ci:>18} "
                     f"{o['wait_better']:>10} {o['play_better']:>10}")
    return "\n".join(lines)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(prog="python -m arena.probe_wait", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("games", nargs="+", help="run directories (game_*.jsonl) or game files")
    ap.add_argument("--kind", default="hivemind", choices=("random", "heuristic", "hivemind"))
    ap.add_argument("--nodes", type=int, default=200, help="team node budget, as in the recorded run")
    ap.add_argument("--compute", default="throughput", choices=("throughput", "nodes"))
    ap.add_argument("--lend", action="store_true")
    ap.add_argument("--waits", type=int, nargs="+", default=[2000, 5000], help="forced wait lengths (ms)")
    ap.add_argument("--positions", type=int, default=4, help="decision points per game")
    ap.add_argument("--rollouts", type=int, default=4, help="rollouts per branch and position")
    ap.add_argument("--need-window-ms", type=int, default=5000)
    ap.add_argument("--any", action="store_true", help="probe any decision point, not only after a need_X")
    ap.add_argument("--min-ply", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--out", default="runs/probe_wait")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    tasks = make_tasks(args)
    if not tasks:
        print("no decision points to probe (try --any, or games from a sharing setup)", file=sys.stderr)
        return 1
    print(f"{len(tasks)} rollouts over {len({t['position'] for t in tasks})} positions", flush=True)
    rows: list[dict] = []
    t0 = time.perf_counter()
    with open(os.path.join(args.out, "rollouts.jsonl"), "w") as f:
        def keep(row: dict) -> None:
            rows.append(row)
            f.write(json.dumps(row) + "\n")
            f.flush()
            print(f"[{len(rows)}/{len(tasks)}] {row['id']} -> {row['score']} ({row['reason']}, "
                  f"{row['wall_s']:.0f}s)", flush=True)

        if args.workers <= 1:
            for t in tasks:
                keep(rollout(t))
        else:
            with ProcessPoolExecutor(max_workers=args.workers) as ex:
                for fut in as_completed([ex.submit(rollout, t) for t in tasks]):
                    keep(fut.result())
    summary = summarize(rows, args.waits, n_boot=args.boot, seed=args.seed)
    summary["config"] = vars(args)
    summary["wall_s"] = round(time.perf_counter() - t0, 1)
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print()
    print(format_summary(summary))
    print(f"\nwrote {args.out}/summary.json ({summary['wall_s']:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
