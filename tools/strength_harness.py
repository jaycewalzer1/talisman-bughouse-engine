"""Local strength harness for wave 2 (docs/strength_harness.md): the yardstick every later engine change is judged by.

Subcommands (run from the main repo, in its ``.venv``; exact commands and the recorded baselines in the doc):

1. ``positions``: decision points from REAL arena games (the PACE wave 2 rows copied to ``runs/pace_wave2/`` and any
   local arena run with ``specs.jsonl`` + ``game_NNNN.jsonl``). Each game is replayed through ``BughouseState`` and
   the arena's own team controller code (``Wave2TeamController``: ``MoveTimeTracker``, ``PocketLandingTracker``,
   ``seat_decisions``) with no engine attached, so every decision carries the fen pair, the four clocks, the
   pockets, ``TimeLeadMs``, the move times and the seat options that the CURRENT arena sends at that state, the
   action the seat played (a move, or ``pass`` when it sat) and the game result for the deciding team. Sampled in
   strata: move choices (middlegame / endgame) and sit-or-act moments (the seat passed, or waiting was plausible:
   small lead and an opponent sitting, a partner capture available or coming, or a piece just landed).
2. ``labels``: the reference: stock upstream it13 (``third_party/upstream-hivemind`` build-ninja) at a deep node
   budget (default 32k) on every position, fresh tree per search. The outcome labels are in the positions file.
3. ``score``: one engine config (``CONFIGS`` or ``--kw``) at 1k/4k/16k nodes on every position, fresh tree per
   search; ``report`` prints agreement with the reference, mean root q with Brier / log-loss against the game
   outcome, and on the sit-or-act subset the sit rate ahead / behind on the clock and act-vs-sit agreement with the
   reference, with game-cluster bootstrap 95% CIs.
4. ``match``: a local short-clock match through the arena pool (``arena/pools/wave2.json`` rows: rules v3,
   bughouse openings, centralized teams, clock_k 110, nodes-as-time), colour-swapped pairs, resumable in slices
   (``--max-games`` / ``--max-minutes``), memory-checked before every game, with score and 95% CI.
   ``validate`` replays a match's own games and checks the reconstructed options against what the engines were
   actually sent (``--log-searches``).

Model and weights everywhere: twin-s-clock it13 (HF ``aminwoo/bughouse-twin-s-clock@edaa246``) on ONNX Runtime.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
import re
import subprocess
import sys
import threading
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from arena.interfaces import BOARD_OF, PARTNER, SEATS, TEAM_OF, Observation  # noqa: E402
from arena.players import upstream as up  # noqa: E402

OUT_DIR = os.path.join("runs", "strength")
PACE_DIRS = [os.path.join("runs", "pace_wave2", r) for r in ("nodur", "ref_120", "same_120")]
SP1_BINARY = os.path.join(up.WAVE2_ENGINE_DIR, "build-sp1", "hivemind.bin")
NINJA_BINARY = up.WAVE2_BINARY
POOL = "arena/pools/wave2.json"
# Simulated ms per searched node for matches (both arms on build-ninja, 4 workers, ONNX Runtime + Core ML, it13):
# measured on the 36 GB M4 Max on 2026-11-21 at 1200 nodes over 20 harness positions with nothing else running
# (it13 0.2025 / 0.2067, nodur 0.2091 / 0.2034, durative_on 0.2460). Pinned so every match charges the same
# budgets whatever else runs on the Mac; a config without an entry is measured once per match (cached in OUT).
# PACE's GPU rows charged wave 2 nodur 0.0530 and it13 0.0424 (wave 2 26% slower per node); locally they are equal.
MS_PER_NODE_M4MAX = {"it13": 0.205, "nodur": 0.206, "durative_on": 0.246}
# The wave 2 pool row's own-action cost (Wave2TeamController.root_min_ms / root_lag_ms): used for the options of
# every decision, also those an it13 team took (what a wave 2 team would have been sent in its place).
WAVE2_ROOT_MIN_MS, WAVE2_ROOT_LAG_MS = 100, 75

# Engine configs: "wave2" = the wave 2 binary with these player flags (arena.players.upstream.wave2_flag_options,
# so every unset flag keeps its wave 2 value); "upstream" = stock upstream it13, no seat options.
CONFIGS = {
    "nodur": {"kind": "wave2", "flags": {"durative_tree": False, "sit_gate": True}},
    "durative_on": {"kind": "wave2", "flags": {}},
    "continuity": {"kind": "wave2", "flags": {"continuity": True}},
    "it13": {"kind": "upstream"},
}


def _read(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def _config(name: str, kw: str | None = None) -> dict:
    """A named config, or a custom wave 2 one: ``--kw '{"sit_gate": true}'`` with any new name."""
    if kw:
        flags = json.loads(kw)
        up.wave2_flag_options(**flags)  # validates the flag names now, not after an hour
        return {"kind": "wave2", "flags": flags}
    if name not in CONFIGS:
        raise SystemExit(f"unknown config {name!r}: one of {sorted(CONFIGS)} or a new name with --kw JSON")
    return CONFIGS[name]


# ---------------------------------------------------------------------------------------------- 1. positions

def _fullmove(fen: str) -> int:
    return int(fen.split()[-1])


def _men(fen: str) -> int:
    return sum(c.isalpha() for c in fen.split("[")[0].split()[0])


def _team_on_turn_boards(fens: dict, team: str) -> list[str]:
    w = "w" if team == "T1" else "b"
    out = []
    for b in ("A", "B"):
        side = fens[b].split()[1]
        if (side == w) if b == "A" else (side != w):
            out.append(b)
    return out


def _controller(team: str, spec: dict) -> up.Wave2TeamController:
    """The team brain the arena would build for ``team`` in this game, with no engine (only its trackers run)."""
    t = spec["teams"][team]
    kw = t.get("player_kw") or {}
    ckw = {k: kw[k] for k in ("clock_k", "min_nodes", "max_nodes", "serial", "charge", "ema_alpha")
           if kw.get(k) is not None}
    return up.Wave2TeamController(team, None, nodes=t["nodes"],
                                  ms_per_node=float(kw.get("ms_per_node") or up.WAVE2_MS_PER_NODE),
                                  root_min_ms=int(kw.get("root_min_ms", WAVE2_ROOT_MIN_MS)),
                                  root_lag_ms=int(kw.get("root_lag_ms", WAVE2_ROOT_LAG_MS)), **ckw)


def _arrivals(value: str) -> list[tuple[str, str, int]]:
    out = []
    for item in filter(None, (value or "").split(",")):
        seat, piece, ago = item.split(":")
        out.append((seat, piece, int(ago)))
    return out


class ReplayError(RuntimeError):
    pass


def replay_game(path: str, spec: dict) -> list[dict]:
    """Every engine search of either team in one game file, with the options the current arena sends at that state.

    A search shows in the event log as a ``thinking`` start with nodes > 0 (a move) or an active ``stall`` with
    reason ``upstream_pass`` (a pass: a pass is never served from the team's cached search). A ``thinking`` start
    with nodes 0 is the team's other seat played from the same search (recorded as ``played_other``). Re-searches
    of a seat that is already sitting and passes again leave no event, so the trackers miss those observations
    (they change the pace EMAs only through the moves seen, so the effect is small; ``validate`` measures it)."""
    from arena.state import BughouseState, state_rules

    events = _read(path)
    if not events or events[0]["type"] != "game_start" or events[-1]["type"] != "game_end":
        raise ReplayError("incomplete game")
    if events[-1].get("error"):
        raise ReplayError("aborted game")
    start, end = events[0], events[-1]
    state = BughouseState(start["base_ms"], start.get("increment_ms", 0), fens=start["fens"],
                          **state_rules(spec.get("rules")))
    last_land = {"A": 0, "B": 0}
    ctls = {t: _controller(t, spec) for t in ("T1", "T2")}
    sitting = {s: False for s in SEATS}
    pending: dict[str, dict | None] = {}  # seat -> its decision awaiting its landing (None: cached half)
    last_search: dict[str, dict] = {}  # team -> its latest search record
    decisions: list[dict] = []
    result = end.get("result")

    def observe(seat: str, ts: int) -> Observation:
        state.advance_clocks(ts)
        return Observation(now_ms=ts, seat=seat, fens={b: state.fen(b) for b in ("A", "B")},
                           clocks_ms={s: int(state.clocks_ms[s]) for s in SEATS},
                           to_move={b: state.to_move(b) for b in ("A", "B")}, legal_moves=[],
                           last_move_ms=dict(last_land))

    def search(seat: str, ts: int, ev_i: int) -> dict:
        team = TEAM_OF[seat]
        ctl = ctls[team]
        obs = observe(seat, ts)
        budget = ctl.budget(obs.clocks_ms[seat])
        budget_ms = int(round(budget * ctl.ms_per_node))
        opts = ctl.time_options(obs, budget_ms)
        opts.pop("new_game", None)
        board = BOARD_OF[seat]
        partner = PARTNER[seat]
        pboard = BOARD_OF[partner]
        pb = state.boards[pboard]
        partner_on_turn = state.to_move(pboard) == partner
        partner_can_capture = partner_on_turn and any(
            pb.is_capture(m) for m in pb.legal_moves if m.drop is None)
        arrivals = _arrivals(opts["options"]["SeatArrivalsMs"])
        ours = {up.UCI_SEAT[s] for s in SEATS if TEAM_OF[s] == team}
        rec = {
            "seat": seat, "board": board, "team": team, "ts": ts, "ev": ev_i,
            "played_by": spec["teams"][team]["kind"],
            "played_by_flags": {k: v for k, v in (spec["teams"][team].get("player_kw") or {}).items()
                                if k in up.WAVE2_FLAGS or k in up.WAVE2_PRESETS},
            "fen_a": obs.fens["A"], "fen_b": obs.fens["B"], "clocks": obs.clocks_ms,
            "pockets": {b: up.fen_pockets(obs.fens[b]) for b in ("A", "B")},
            "lead": opts["lead"], "team_move_ms": opts["team_move_ms"], "opp_move_ms": opts["opp_move_ms"],
            "root_move_ms": opts["root_move_ms"], "budget_ms": budget_ms, "budget_nodes": budget,
            "seat_options": opts["options"], "on_turn": _team_on_turn_boards(obs.fens, team),
            "legal_count": sum(1 for _ in state.boards[board].legal_moves),
            "flags": {"after_sit": sitting[seat],
                      "opp_sitting": any(sitting[s] for s in SEATS if TEAM_OF[s] != team
                                         and state.to_move(BOARD_OF[s]) == s),
                      "partner_can_capture": bool(partner_can_capture),
                      "recent_arrival": any(s in ours and ago <= 1500 for s, _, ago in arrivals)},
            "result": 0.5 if result not in ("T1", "T2") else (1.0 if result == team else 0.0),
            "game_reason": end.get("reason"),
        }
        last_search[team] = rec
        return rec

    # Searches that leave no event: a sitting seat is asked again at every landing (either board) and when its
    # wait runs out (Decision.wait_ms = think + the controller's default sit); when it passes again nothing is
    # logged. Re-create them: a waiting seat without a "thinking" start at that instant passed again.
    starts_at = {(e["seat"], int(e["ts"])) for e in events if e["type"] == "thinking" and e.get("phase") == "start"}
    thinking = {s: False for s in SEATS}
    wake_at: dict[str, int | None] = {s: None for s in SEATS}

    def schedule_wake(seat: str, rec: dict) -> None:
        if rec["played_by"] == "wave2":
            root = rec["root_move_ms"]
            sit = max(up.DEFAULT_MOVE_MS, min(8 * root, int(up.SIT_INFER_FACTOR * root) + 1))
        else:
            sit = max(up.DEFAULT_MOVE_MS, int(rec["team_move_ms"]))
        wake_at[seat] = rec["ts"] + rec["budget_ms"] + sit

    def waiting(seat: str) -> bool:
        return sitting[seat] and not thinking[seat] and state.to_move(BOARD_OF[seat]) == seat

    def hidden_pass(seat: str, ts: int, ev_i: int) -> None:
        rec = search(seat, ts, ev_i)
        rec.update(played="pass", hidden=True, game_eval=None)
        decisions.append(rec)
        schedule_wake(seat, rec)

    for i, ev in enumerate(events):
        typ, ts = ev["type"], int(ev["ts"])
        while True:  # timer wakes due before this event
            due = sorted((w, s) for s, w in wake_at.items() if w is not None and w < ts and waiting(s))
            if not due:
                break
            w, s = due[0]
            wake_at[s] = None
            if (s, w) not in starts_at:
                hidden_pass(s, w, i)
        if typ == "thinking" and ev.get("phase") == "start":
            seat = ev["seat"]
            thinking[seat] = True
            wake_at[seat] = None
            if ev.get("nodes", 0) > 0:
                pending[seat] = search(seat, ts, i)
            else:  # served from the team's last search: the controller only observes the clocks
                team = TEAM_OF[seat]
                obs = observe(seat, ts)
                ctls[team].tracker.observe(obs.fens, obs.clocks_ms, obs.to_move)
                prev = last_search.get(team)
                same = prev is not None and (prev["fen_a"], prev["fen_b"]) == (obs.fens["A"], obs.fens["B"])
                pending[seat] = {"cached_of": prev} if same else None
        elif typ == "stall":
            seat = ev["seat"]
            if ev.get("active"):
                sitting[seat] = True
                if ev.get("reason") == "upstream_pass":
                    rec = search(seat, ts, i)
                    rec["played"] = "pass"
                    rec["game_eval"] = None
                    decisions.append(rec)
                    schedule_wake(seat, rec)
            else:
                sitting[seat] = False
        elif typ == "move":
            seat, board = ev["seat"], ev["board"]
            state.advance_clocks(ts)
            try:
                state.apply_move(seat, ev["uci"], ts)
            except ValueError as exc:
                raise ReplayError(f"event {i}: {exc}") from None
            if state.fen(board) != ev["fen"]:
                raise ReplayError(f"event {i}: fen mismatch {state.fen(board)} != {ev['fen']}")
            if {s: int(state.clocks_ms[s]) for s in SEATS} != {s: int(v) for s, v in ev["clocks"].items()}:
                raise ReplayError(f"event {i}: clock mismatch")
            last_land[board] = ts
            sitting[seat] = False
            rec = pending.pop(seat, None)
            if rec is not None and "cached_of" in rec:
                if rec["cached_of"] is not None:
                    rec["cached_of"]["played_other"] = ev["uci"]
            elif rec is not None:
                rec["played"] = ev["uci"]
                rec["captures"] = ev.get("capture")
                decisions.append(rec)
            for s in SEATS:  # the landing wakes every waiting seat
                if waiting(s):
                    wake_at[s] = None
                    if (s, ts) not in starts_at:
                        hidden_pass(s, ts, i)
        elif typ == "thinking" and ev.get("phase") == "end":
            # The landing follows directly when the move was legal; otherwise the decision is stale (re-asked).
            thinking[ev["seat"]] = False
            nxt = events[i + 1] if i + 1 < len(events) else None
            if not (nxt and nxt["type"] == "move" and nxt["seat"] == ev["seat"]):
                pending.pop(ev["seat"], None)
            elif ev["seat"] in pending and pending[ev["seat"]] is not None and "cached_of" not in pending[ev["seat"]]:
                pending[ev["seat"]]["game_eval"] = ev.get("eval")
    # Hindsight flag: the partner's next landing (within 1 s) was a capture, i.e. waiting would have brought a piece.
    moves = [(int(e["ts"]), e["seat"], e.get("capture")) for e in events if e["type"] == "move"]
    for d in decisions:
        partner = PARTNER[d["seat"]]
        nxt = next(((t, c) for t, s, c in moves if s == partner and t > d["ts"]), None)
        d["flags"]["partner_captures_next"] = bool(nxt and nxt[1] and nxt[0] - d["ts"] <= 1000)
    return decisions


def _game_specs(run_dir: str) -> dict[int, dict]:
    return {s["index"]: s for s in _read(os.path.join(run_dir, "specs.jsonl"))}


def iter_run_decisions(run_dir: str):
    specs = _game_specs(run_dir)
    for path in sorted(glob.glob(os.path.join(run_dir, "game_*.jsonl"))):
        idx = int(re.search(r"game_(\d+)\.jsonl", path).group(1))
        spec = specs.get(idx)
        if spec is None:
            continue
        try:
            decs = replay_game(path, spec)
        except ReplayError as exc:
            print(f"skip {path}: {exc}", file=sys.stderr)
            continue
        for d in decs:
            d["run"] = run_dir
            d["game"] = f"{os.path.basename(run_dir)}/{os.path.basename(path)}"
            d["opening"] = spec.get("opening")
            yield d


def phase_of(d: dict) -> str | None:
    """None (opening, not sampled), "middlegame" or "endgame" (late: fullmove 30+ on a board, or few men)."""
    fms = [_fullmove(d["fen_a"]), _fullmove(d["fen_b"])]
    men = min(_men(d["fen_a"]), _men(d["fen_b"]))
    if min(fms) < 8:
        return None
    return "endgame" if max(fms) >= 30 or men <= 16 else "middlegame"


def sit_or_act(d: dict) -> bool:
    """A moment where waiting is a real option: the seat passed (``pass``), or it acts after sitting on this turn
    (``after_sit``), or an opponent on turn is sitting (``opp_sitting``), or the lead is small (|lead| <= 1 s) and the
    partner's next landing, within 1 s, is a capture (``partner_captures_next``: a piece was about to arrive).
    ``partner_can_capture`` and ``recent_arrival`` are stored but too common at 120 s clocks to select on (about
    40% and 60% of all decisions)."""
    if d["played"] == "pass":
        return True
    f = d["flags"]
    return bool(f["after_sit"] or f["opp_sitting"] or (abs(d["lead"]) <= 1000 and f["partner_captures_next"]))


def cmd_positions(a) -> int:
    pool = []
    for run in a.runs:
        n0 = len(pool)
        for d in iter_run_decisions(run):
            ph = phase_of(d)
            if ph is None or d["legal_count"] < 2:
                continue
            d["phase"] = ph
            d["sit_or_act"] = sit_or_act(d)
            pool.append(d)
        print(f"{run}: {len(pool) - n0} candidate decisions", flush=True)
    rng = random.Random(a.seed)
    rng.shuffle(pool)
    n_sit = round(a.n * a.sit_share)
    n_move = a.n - n_sit
    strata = {
        "sit_pass": ([d for d in pool if d["played"] == "pass"], n_sit // 2),
        "sit_act": ([d for d in pool if d["sit_or_act"] and d["played"] != "pass"], n_sit - n_sit // 2),
        "move_middlegame": ([d for d in pool if not d["sit_or_act"] and d["phase"] == "middlegame"], n_move // 2),
        "move_endgame": ([d for d in pool if not d["sit_or_act"] and d["phase"] == "endgame"], n_move - n_move // 2),
    }
    picked, used_games, seen = [], {}, set()
    for name, (cands, want) in strata.items():
        got = 0
        # Spread over games: at most one position per game per pass, then allow more if the stratum is short.
        for cap in (1, 2, 4, 1 << 30):
            for d in cands:
                if got >= want:
                    break
                key = (d["fen_a"], d["fen_b"], d["team"])
                if key in seen or used_games.get((name, d["game"]), 0) >= cap:
                    continue
                seen.add(key)
                used_games[(name, d["game"])] = used_games.get((name, d["game"]), 0) + 1
                picked.append({**d, "stratum": name})
                got += 1
        print(f"{name}: {got} of {want} wanted ({len(cands)} candidates)")
    # Shuffled ids: any id slice (--ids LO HI) is a random subset of every stratum.
    rng.shuffle(picked)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        for i, d in enumerate(picked):
            f.write(json.dumps({"id": i, **d}) + "\n")
    print(f"wrote {len(picked)} positions to {a.out}")
    return 0


# ---------------------------------------------------------------------------------------------- 2./3. searches

def _process(cfg: dict, binary: str | None) -> up.UpstreamHivemindProcess:
    if cfg["kind"] == "upstream":
        return up.UpstreamHivemindProcess(binary or up.UPSTREAM_BINARY, up.UPSTREAM_MODEL, reset_tree=True,
                                          timeout_s=1800)
    return up.UpstreamHivemindProcess(binary or NINJA_BINARY, up.WAVE2_MODEL,
                                      options=up.wave2_flag_options(**cfg["flags"]), reset_tree=True,
                                      timeout_s=1800, cwd=up.WAVE2_ENGINE_DIR)


def _search(proc, cfg: dict, p: dict, nodes: int):
    """One fresh-tree search with exactly the arena's inputs for this config kind: a wave 2 team gets the seat
    options and its own-action cost (root_move_ms); stock it13 gets lead and move times with RootMoveTimeMs = the
    bare budget, as UpstreamTeamController sends."""
    if cfg["kind"] == "upstream":
        return proc.search_clock(p["fen_a"], p["fen_b"], team=p["team"], nodes=nodes, lead=p["lead"],
                                 team_move_ms=p["team_move_ms"], opp_move_ms=p["opp_move_ms"],
                                 root_move_ms=p["budget_ms"])
    return proc.search_clock(p["fen_a"], p["fen_b"], team=p["team"], nodes=nodes, lead=p["lead"],
                             team_move_ms=p["team_move_ms"], opp_move_ms=p["opp_move_ms"],
                             root_move_ms=p["root_move_ms"], options=p["seat_options"])


def _run_searches(positions: list[dict], cfg: dict, name: str, nodes_list: list[int], out_path: str,
                  binary: str | None, jobs: int, rep: int = 0) -> None:
    done = set()
    if os.path.exists(out_path):
        done = {(r["id"], r["config"], r["nodes"], r.get("rep", 0)) for r in _read(out_path)}
    todo = [(p, n) for n in nodes_list for p in positions if (p["id"], name, n, rep) not in done]
    print(f"{name}: {len(todo)} searches to run ({len(done)} rows already in {out_path})", flush=True)
    if not todo:
        return
    lock = threading.Lock()
    out = open(out_path, "a")
    t0 = time.time()
    count = [0]

    def worker(chunk):
        proc = _process(cfg, binary)
        try:
            for p, n in chunk:
                res = _search(proc, cfg, p, n)
                row = {"id": p["id"], "config": name, "nodes": n, "rep": rep, "move_a": res.move_a,
                       "move_b": res.move_b, "q": res.q, "nodes_searched": res.nodes, "wall_ms": res.wall_ms,
                       "binary": proc.binary, "kind": cfg["kind"], "flags": cfg.get("flags")}
                with lock:
                    out.write(json.dumps(row) + "\n")
                    out.flush()
                    count[0] += 1
                    if count[0] % 25 == 0:
                        print(f"  {name}: {count[0]}/{len(todo)} ({time.time() - t0:.0f} s)", flush=True)
        finally:
            proc.close()

    chunks = [todo[k::jobs] for k in range(jobs)]
    threads = [threading.Thread(target=worker, args=(c,)) for c in chunks if c]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    out.close()
    print(f"{name}: done in {time.time() - t0:.0f} s", flush=True)


def _slice(positions: list[dict], ids) -> list[dict]:
    if not ids:
        return positions
    lo, hi = ids
    return [p for p in positions if lo <= p["id"] < hi]


def cmd_labels(a) -> int:
    positions = _slice(_read(a.positions), a.ids)
    _run_searches(positions, CONFIGS["it13"], f"ref_it13", [a.nodes], a.out, a.binary, a.jobs)
    return 0


def cmd_score(a) -> int:
    cfg = _config(a.config, a.kw)
    positions = _slice(_read(a.positions), a.ids)
    _run_searches(positions, cfg, a.config, a.nodes, a.out, a.binary, a.jobs, a.rep)
    return 0


# ---------------------------------------------------------------------------------------------- report

def _boot(rows: list[dict], stat, n_boot: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """stat(rows) with a 95% percentile CI from resampling whole games (positions of one game are correlated)."""
    est = stat(rows)
    if est is None:
        return float("nan"), float("nan"), float("nan")
    by: dict[str, list[dict]] = {}
    for r in rows:
        by.setdefault(r["game"], []).append(r)
    clusters = list(by.values())
    rng = random.Random(seed)
    vals = []
    for _ in range(n_boot):
        sample = [r for _ in clusters for r in rng.choice(clusters)]
        v = stat(sample)
        if v is not None:
            vals.append(v)
    vals.sort()
    if not vals:
        return est, float("nan"), float("nan")
    return est, vals[int(0.025 * len(vals))], vals[min(len(vals) - 1, int(0.975 * len(vals)))]


def _mean(key):
    def f(rows):
        xs = [r[key] for r in rows if r.get(key) is not None]
        return sum(xs) / len(xs) if xs else None
    return f


def _p(q: float) -> float:
    return min(0.99, max(0.01, (q + 1.0) / 2.0))


def enrich(p: dict, r: dict, ref: dict | None) -> dict:
    """One scored row: agreement with the reference and the move played, root q and its outcome losses, sits."""
    b = p["board"].lower()
    mv = r[f"move_{b}"]
    row = {"game": p["game"], "id": p["id"], "sit_or_act": p["sit_or_act"], "lead": p["lead"],
           "played_pass": p["played"] == "pass", "q": r["q"], "sits": 1.0 if mv is None else 0.0,
           "agree_played": 1.0 if (mv or "pass") == p["played"] else 0.0}
    if r["q"] is not None:
        y = p["result"]
        pr = _p(r["q"])
        row["brier"] = (pr - y) ** 2
        row["logloss"] = -(y * math.log(pr) + (1 - y) * math.log(1 - pr))
    if ref is not None:
        rm = ref[f"move_{b}"]
        row["agree_board"] = 1.0 if mv == rm else 0.0
        row["agree_joint"] = 1.0 if all(r[f"move_{x.lower()}"] == ref[f"move_{x.lower()}"]
                                        for x in p["on_turn"]) else 0.0
        row["agree_actsit"] = 1.0 if (mv is None) == (rm is None) else 0.0
    return row


def _fmt(t, pct=False) -> str:
    est, lo, hi = t
    if est != est:
        return "-"
    if pct:
        return f"{est:.0%} [{lo:.0%}, {hi:.0%}]"
    return f"{est:.3f} [{lo:.3f}, {hi:.3f}]"


def report_lines(positions: list[dict], labels: list[dict], scores: list[dict], configs: list[str] | None,
                 n_boot: int) -> list[str]:
    pos = {p["id"]: p for p in positions}
    ref = {r["id"]: r for r in labels}
    ref_nodes = sorted({r["nodes"] for r in labels})
    keyset: dict[tuple, list] = {}
    seen = set()
    for r in scores:
        if r.get("rep", 0) != 0 or r["id"] not in pos or (r["id"], r["config"], r["nodes"]) in seen:
            continue  # the first row of a (position, config, budget) counts, if two runs overlapped
        seen.add((r["id"], r["config"], r["nodes"]))
        keyset.setdefault((r["config"], r["nodes"]), []).append(r)
    names = configs or list(dict.fromkeys(k[0] for k in sorted(keyset, key=lambda k: (list(CONFIGS).index(k[0])
                                                                                       if k[0] in CONFIGS else 99,
                                                                                       k[1]))))
    n_sit = sum(p["sit_or_act"] for p in positions)
    n_games = len({p["game"] for p in positions})
    lines = [f"Positions: {len(positions)} from {n_games} games ({n_sit} sit-or-act). Reference: stock upstream "
             f"it13 at {'/'.join(str(n) for n in ref_nodes)} nodes, {len(ref)} labelled. 95% CIs: game-cluster "
             f"bootstrap, {n_boot} resamples.", "",
             "| Config | Nodes | n | Joint agree | Board agree | Played-move agree | Mean root q | Brier | Log-loss |",
             "|---|---|---|---|---|---|---|---|---|"]
    rows_by = {}
    for name in names:
        for nodes in sorted(n for c, n in keyset if c == name):
            rows = [enrich(pos[r["id"]], r, ref.get(r["id"])) for r in keyset[(name, nodes)]]
            rows_by[(name, nodes)] = rows
            lines.append(f"| {name} | {nodes} | {len(rows)} | {_fmt(_boot(rows, _mean('agree_joint'), n_boot), True)} | "
                         f"{_fmt(_boot(rows, _mean('agree_board'), n_boot), True)} | "
                         f"{_fmt(_boot(rows, _mean('agree_played'), n_boot), True)} | "
                         f"{_fmt(_boot(rows, _mean('q'), n_boot))} | {_fmt(_boot(rows, _mean('brier'), n_boot))} | "
                         f"{_fmt(_boot(rows, _mean('logloss'), n_boot))} |")
    if labels:
        rrows = [enrich(pos[r["id"]], r, r) for r in labels if r["id"] in pos]
        rows_by[("ref", ref_nodes[-1])] = rrows
        lines.append(f"| reference it13 | {ref_nodes[-1]} | {len(rrows)} | 100% | 100% | "
                     f"{_fmt(_boot(rrows, _mean('agree_played'), n_boot), True)} | "
                     f"{_fmt(_boot(rrows, _mean('q'), n_boot))} | {_fmt(_boot(rrows, _mean('brier'), n_boot))} | "
                     f"{_fmt(_boot(rrows, _mean('logloss'), n_boot))} |")
    base_rate = sum(p["result"] for p in positions) / max(1, len(positions))
    lines += ["", f"Outcome base rate (deciding team's score): {base_rate:.3f}; a constant forecast of it has Brier "
              f"{sum((base_rate - p['result']) ** 2 for p in positions) / max(1, len(positions)):.3f}.", "",
              "Sit-or-act subset (the deciding seat's board; sit = the search passes there):", "",
              "| Config | Nodes | n | Sit rate, ahead on clock | Sit rate, behind | Sit rate, all | Act-vs-sit agrees with reference | Agrees with what was played (sit/act) |",
              "|---|---|---|---|---|---|---|---|"]
    for (name, nodes), rows in rows_by.items():
        sub = [r for r in rows if r["sit_or_act"]]
        ahead = [r for r in sub if r["lead"] > 0]
        behind = [r for r in sub if r["lead"] < 0]
        for r in sub:
            r["actsit_played"] = 1.0 if (r["sits"] == 1.0) == r["played_pass"] else 0.0
        label = "reference it13" if name == "ref" else name
        lines.append(f"| {label} | {nodes} | {len(sub)} | {_fmt(_boot(ahead, _mean('sits'), n_boot), True)} (n {len(ahead)}) | "
                     f"{_fmt(_boot(behind, _mean('sits'), n_boot), True)} (n {len(behind)}) | "
                     f"{_fmt(_boot(sub, _mean('sits'), n_boot), True)} | "
                     + ("-" if name == "ref" else _fmt(_boot(sub, _mean('agree_actsit'), n_boot), True))
                     + f" | {_fmt(_boot(sub, _mean('actsit_played'), n_boot), True)} |")
    played = [p for p in positions if p["sit_or_act"]]
    lines += ["", f"In the games themselves, the seat sat at {sum(p['played'] == 'pass' for p in played)} of "
              f"{len(played)} sit-or-act positions (by design of the sample: about half are passes)."]
    return lines


def cmd_report(a) -> int:
    positions = _read(a.positions)
    labels = _read(a.labels) if a.labels and os.path.exists(a.labels) else []
    scores = _read(a.scores) if os.path.exists(a.scores) else []
    print("\n".join(report_lines(positions, labels, scores, a.configs, a.boot)))
    return 0


# ---------------------------------------------------------------------------------------------- 4. match

def free_memory_pct() -> int | None:
    try:
        out = subprocess.run(["memory_pressure"], capture_output=True, text=True, timeout=30).stdout
    except (OSError, subprocess.TimeoutExpired):
        return None
    m = re.search(r"free percentage:\s*(\d+)%", out)
    return int(m.group(1)) if m else None


def _player_kw(cfg: dict) -> dict:
    return dict(cfg.get("flags") or {})


_SEARCH_LOG = None


def _install_search_log(path: str) -> None:
    """Log every engine search of this process (both arms) with the options it was sent (for ``validate``)."""
    global _SEARCH_LOG
    if _SEARCH_LOG is not None:
        return
    _SEARCH_LOG = open(path, "a")
    orig = up.UpstreamHivemindProcess.search_clock

    def logged(self, fen_a, fen_b, **kw):
        res = orig(self, fen_a, fen_b, **kw)
        row = {"binary": self.binary, "fen_a": fen_a, "fen_b": fen_b,
               **{k: v for k, v in kw.items() if k not in ("options", "new_game")},
               "seat_options": kw.get("options") or {}, "move_a": res.move_a, "move_b": res.move_b, "q": res.q,
               "nodes_searched": res.nodes}
        _SEARCH_LOG.write(json.dumps(row) + "\n")
        _SEARCH_LOG.flush()
        return res

    up.UpstreamHivemindProcess.search_clock = logged


def _play(spec: dict, log_dir: str | None) -> dict:
    if log_dir:
        _install_search_log(os.path.join(log_dir, f"searches_{os.getpid()}.jsonl"))
    from arena.run import run_one
    return run_one(spec)


def match_specs(a) -> list[dict]:
    from arena import pool as pool_mod

    ca, cb = _config(a.a, a.a_kw), _config(a.b, a.b_kw)
    if ca["kind"] != "wave2":
        raise SystemExit("match: A is the contender and must be a wave 2 config (swap A and B; the score is A's)")
    pool = pool_mod.Pool.load(a.pool)
    # Stock it13 plays as the pool's upstream-it13 row; a wave 2 config as the wave2-self row with its flags.
    opp = "upstream-it13" if cb["kind"] == "upstream" else "wave2-self"
    okw = _player_kw(cb)
    ms = {}
    if a.ms_per_node == "measure":
        pass
    elif a.ms_per_node:
        ma, mb = (float(x) for x in a.ms_per_node.split(","))
        ms = {"a": ma, "b": mb}
    elif a.a in MS_PER_NODE_M4MAX and a.b in MS_PER_NODE_M4MAX and not a.a_kw and not a.b_kw:
        ms = {"a": MS_PER_NODE_M4MAX[a.a], "b": MS_PER_NODE_M4MAX[a.b]}
    pool = pool_mod.for_evaluation(pool, opponent=opp, contender_kw={**_player_kw(ca), **({"ms_per_node": ms["a"]} if ms else {})},
                                   opponent_kw={**okw, **({"ms_per_node": ms["b"]} if ms else {})})
    if not ms:
        pool, used = pool_mod.apply_ms_per_node(pool, os.path.join(a.out, "ms_per_node.json"), nodes=800)
    specs = pool_mod.make_schedule(pool, a.pairs, seed=a.seed, openings="bughouse", base_ms=a.base_ms, rules="v3")
    for s in specs:
        s["out"] = a.out
    return specs


def cmd_match(a) -> int:
    from arena import pool as pool_mod

    os.makedirs(a.out, exist_ok=True)
    meta_path = os.path.join(a.out, "match.json")
    meta = {"a": a.a, "b": a.b, "a_kw": a.a_kw, "b_kw": a.b_kw, "pairs": a.pairs, "base_ms": a.base_ms,
            "seed": a.seed, "pool": a.pool, "ms_per_node": a.ms_per_node}
    specs_path = os.path.join(a.out, "specs.jsonl")
    if os.path.exists(specs_path):
        with open(meta_path) as f:
            old = json.load(f)
        if any(old.get(k) != meta[k] for k in meta):
            raise SystemExit(f"{a.out} holds a different match ({old}); use another --out")
        specs = _read(specs_path)
        for s in specs:
            s["out"] = a.out
    else:
        specs = match_specs(a)
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        with open(specs_path, "w") as f:
            for s in specs:
                f.write(json.dumps({k: v for k, v in s.items() if k != "out"}) + "\n")
    done = {s["index"] for s in specs
            if pool_mod.summary_from_game_file(s, os.path.join(a.out, pool_mod.game_file(s))) is not None}
    todo = [s for s in specs if s["index"] not in done][: a.max_games if a.max_games else None]
    print(f"{a.out}: {len(done)} of {len(specs)} games played; this call plays up to {len(todo)}", flush=True)
    t0 = time.time()
    log_dir = a.out if a.log_searches else None
    played = 0
    stop = None
    if a.workers <= 1:
        for s in todo:
            if a.max_minutes and time.time() - t0 > 60 * a.max_minutes:
                stop = "time"
                break
            free = free_memory_pct()
            if free is not None and free < a.min_free_pct:
                stop = f"memory ({free}% free < {a.min_free_pct}%)"
                break
            g0 = time.time()
            res = _play(s, log_dir)
            played += 1
            tag = res.get("error") or f"{res.get('result')} by {res.get('reason')}"
            print(f"game {s['index']} ({s['opening']}) A={s['contender_team']}: {tag}, {time.time() - g0:.0f} s wall",
                  flush=True)
    else:
        from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            queue, running = list(todo), {}
            while queue or running:
                while queue and len(running) < a.workers and stop is None:
                    if a.max_minutes and time.time() - t0 > 60 * a.max_minutes:
                        stop = "time"
                        break
                    free = free_memory_pct()
                    if free is not None and free < a.min_free_pct:
                        stop = f"memory ({free}% free < {a.min_free_pct}%)"
                        break
                    s = queue.pop(0)
                    running[ex.submit(_play, s, log_dir)] = (s, time.time())
                if stop is not None and not running:
                    break
                if not running:
                    break
                fin, _ = wait(list(running), return_when=FIRST_COMPLETED)
                for fut in fin:
                    s, g0 = running.pop(fut)
                    res = fut.result()
                    played += 1
                    tag = res.get("error") or f"{res.get('result')} by {res.get('reason')}"
                    print(f"game {s['index']} ({s['opening']}) A={s['contender_team']}: {tag}, "
                          f"{time.time() - g0:.0f} s wall", flush=True)
    wall = time.time() - t0
    if stop:
        print(f"stopped early: {stop}", flush=True)
    if played:
        print(f"played {played} games in {wall:.0f} s: {3600 * played / wall:.1f} games per hour with "
              f"{a.workers} worker(s)", flush=True)
        with open(os.path.join(a.out, "throughput.jsonl"), "a") as f:
            f.write(json.dumps({"when": time.strftime("%Y-%m-%dT%H:%M:%S"), "games": played, "wall_s": round(wall, 1),
                                "workers": a.workers, "load": os.getloadavg()[0]}) + "\n")
    print("\n".join(match_summary(a.out, specs)))
    return 0


def match_summary(out: str, specs: list[dict] | None = None) -> list[str]:
    from arena import pool as pool_mod

    specs = specs or _read(os.path.join(out, "specs.jsonl"))
    sums = []
    for s in specs:
        sm = pool_mod.summary_from_game_file(s, os.path.join(out, pool_mod.game_file(s)))
        if sm is not None:
            sums.append(sm)
    if not sums:
        return ["no games finished yet"]
    meta = json.load(open(os.path.join(out, "match.json")))
    res = pool_mod.results_by_opponent(sums)
    lines = [f"Match {meta['a']} (A, contender) vs {meta['b']}: {len(sums)} of {len(specs)} games, "
             f"{meta['base_ms'] / 1000:g} s base"]
    for name, r in res["by_opponent"].items():
        lines.append(f"A's score {r['score']:.3f}, 95% CI [{r['ci95'][0]:.3f}, {r['ci95'][1]:.3f}] "
                     f"(W {r['wins']} / D {r['draws']} / L {r['losses']}; pentanomial {r['pentanomial']}; "
                     f"nodes/game A {r['contender_nodes_per_game']:.0f}, B {r['baseline_nodes_per_game']:.0f}; "
                     f"reasons {r['reasons']})")
    return lines


def cmd_match_summary(a) -> int:
    print("\n".join(match_summary(a.out)))
    return 0


def cmd_validate(a) -> int:
    """Replay a match's games and compare every reconstructed search with the logged one (same fens, team and
    clocks): the share of searches found, and how often each option agrees exactly."""
    logged = []
    for path in glob.glob(os.path.join(a.out, "searches_*.jsonl")):
        logged += _read(path)
    index = {}
    for r in logged:
        index.setdefault((r["fen_a"], r["fen_b"], r["team"]), []).append(r)
    found = total = 0
    agree: dict[str, int] = {}
    asked: dict[str, int] = {}
    for d in iter_run_decisions(a.out):
        total += 1
        cands = index.get((d["fen_a"], d["fen_b"], d["team"]), [])
        if d["played_by"] == "wave2":
            cands = [c for c in cands if c["seat_options"].get("SeatClocksMs") == d["seat_options"]["SeatClocksMs"]]
        if not cands:
            continue
        found += 1
        c = cands[0]
        checks = {"lead": c["lead"] == d["lead"], "team_move_ms": c["team_move_ms"] == d["team_move_ms"],
                  "opp_move_ms": c["opp_move_ms"] == d["opp_move_ms"],
                  "root_move_ms": c["root_move_ms"] == (d["root_move_ms"] if d["played_by"] == "wave2"
                                                        else d["budget_ms"]),
                  "nodes": c["nodes"] == d["budget_nodes"]}
        if d["played_by"] == "wave2":
            for k in ("SeatClocksMs", "SeatPaceMs", "SeatElapsedMs", "SeatSitting", "SeatArrivalsMs"):
                checks[k] = c["seat_options"].get(k) == d["seat_options"][k]
        for k, ok in checks.items():
            agree[k] = agree.get(k, 0) + int(ok)
            asked[k] = asked.get(k, 0) + 1
    print(f"{found} of {total} reconstructed searches found in the log ({len(logged)} logged searches)")
    for k, v in agree.items():
        print(f"  {k}: {v}/{asked[k]} exact")
    return 0


# ---------------------------------------------------------------------------------------------- CLI

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="tools/strength_harness.py", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("positions", help="sample decision points from real arena games")
    p.add_argument("--runs", nargs="+", default=PACE_DIRS)
    p.add_argument("--n", type=int, default=300)
    p.add_argument("--sit-share", type=float, default=0.4, help="share of sit-or-act positions (half of them passes)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=os.path.join(OUT_DIR, "positions.jsonl"))
    for name, helptext in (("labels", "deep it13 reference moves"), ("score", "one config at several budgets")):
        p = sub.add_parser(name, help=helptext)
        if name == "score":
            p.add_argument("config")
            p.add_argument("--kw", default=None, help="JSON wave 2 player flags: defines a new config named CONFIG")
            p.add_argument("--nodes", type=int, nargs="+", default=[1000, 4000, 16000])
            p.add_argument("--rep", type=int, default=0, help="repeat index (1 = a second run, for the noise floor)")
            p.add_argument("--out", default=os.path.join(OUT_DIR, "scores.jsonl"))
        else:
            p.add_argument("--nodes", type=int, default=32000)
            p.add_argument("--out", default=os.path.join(OUT_DIR, "labels.jsonl"))
        p.add_argument("--positions", default=os.path.join(OUT_DIR, "positions.jsonl"))
        p.add_argument("--binary", default=None, help="engine binary (default: wave 2 build-ninja / stock upstream)")
        p.add_argument("--jobs", type=int, default=1, help="engine processes in parallel")
        p.add_argument("--ids", type=int, nargs=2, metavar=("LO", "HI"), help="only ids LO <= id < HI")
    p = sub.add_parser("report", help="markdown tables")
    p.add_argument("--positions", default=os.path.join(OUT_DIR, "positions.jsonl"))
    p.add_argument("--labels", default=os.path.join(OUT_DIR, "labels.jsonl"))
    p.add_argument("--scores", default=os.path.join(OUT_DIR, "scores.jsonl"))
    p.add_argument("--configs", nargs="*", default=None)
    p.add_argument("--boot", type=int, default=2000)
    p = sub.add_parser("match", help="local short-clock match A (wave 2 config) vs B, resumable")
    p.add_argument("a")
    p.add_argument("b")
    p.add_argument("--a-kw", default=None)
    p.add_argument("--b-kw", default=None)
    p.add_argument("--pairs", type=int, default=100, help="opening pairs in the schedule (2 games each)")
    p.add_argument("--base-ms", type=int, default=30000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--pool", default=POOL)
    p.add_argument("--ms-per-node", default=None, metavar="A,B",
                   help="both arms' simulated ms per node, or 'measure' (once, cached in OUT); default: "
                        "MS_PER_NODE_M4MAX when both configs have an entry, else measure")
    p.add_argument("--max-games", type=int, default=0, help="play at most this many games in this call")
    p.add_argument("--max-minutes", type=float, default=0, help="start no new game after this many minutes")
    p.add_argument("--min-free-pct", type=int, default=25, help="stop when memory_pressure free %% drops below")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--log-searches", action="store_true", help="log every search (for validate)")
    p.add_argument("--out", required=True)
    p = sub.add_parser("match-summary", help="score of a match so far")
    p.add_argument("--out", required=True)
    p = sub.add_parser("validate", help="reconstructed options vs the options a logged match sent")
    p.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    return {"positions": cmd_positions, "labels": cmd_labels, "score": cmd_score, "report": cmd_report,
            "match": cmd_match, "match-summary": cmd_match_summary, "validate": cmd_validate}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
