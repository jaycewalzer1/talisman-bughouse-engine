"""Match statistics: per-game summaries from event lists, and contender-vs-baseline metrics with
95% confidence intervals from a cluster bootstrap over opening pairs.

Games that share an opening pair are not independent (each pair is played twice, teams swapped), so
the bootstrap resamples whole opening pairs, not games. All functions are pure; the bootstrap uses a
fixed seed so results are reproducible.
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Callable, Iterable, Sequence

import chess
import chess.variant

from arena.interfaces import BOARD_OF, OPPONENT, PARTNER, SEATS, TEAM_OF

TEAMS = ("T1", "T2")
SIGNS = ("ahead", "behind")  # diagonal clock (own minus partner's opponent) > 0, or <= 0


def _side_to_move(fen: str | None) -> str:
    parts = (fen or "").split()
    return "black" if len(parts) > 1 and parts[1] == "b" else "white"


def _in_check(fen: str | None) -> bool:
    try:
        return chess.variant.CrazyhouseBoard(fen).is_check()
    except Exception:  # not a parsable FEN (test fakes): unknown, count as no check
        return False


def _clock_metrics(events: list[dict], start: dict, end: dict | None) -> dict:
    """Replay the clocks from game_start and the move events (each carries the clocks at landing; between
    events only the seat on turn on each board loses time) and derive, per team:
      turns_<sign>, sits_<sign>, sit_ms_<sign>: turns, turns with a sit spell, and total sit length, bucketed by
          the sign of the seat's diagonal clock when the turn began;
    and per game: flag_while ("sitting" | "moving" | None) for the flagged seat, flag_other_check (the board
    that did not flag had its side to move in check when the flag fell)."""
    clocks = {s: int((start.get("clocks") or {}).get(s, 0)) for s in SEATS}
    fens = dict(start.get("fens") or {})
    on_turn = {b: f"{b}_{_side_to_move(fens.get(b))}" for b in ("A", "B")}
    t_ref = int(start.get("ts", 0))

    def now_clocks(ts: int) -> dict:
        out = dict(clocks)
        for seat in on_turn.values():
            out[seat] = max(0, out[seat] - (ts - t_ref))
        return out

    agg = {t: {f"{k}_{sg}": 0 for k in ("turns", "sits", "sit_ms") for sg in SIGNS} for t in TEAMS}
    turn_sign: dict[str, str] = {}
    sit_start: dict[str, int] = {}
    thinking: set[str] = set()

    def begin_turn(seat: str, ts: int) -> None:
        c = now_clocks(ts)
        sign = "ahead" if c[seat] - c[OPPONENT[PARTNER[seat]]] > 0 else "behind"
        turn_sign[seat] = sign
        agg[TEAM_OF[seat]][f"turns_{sign}"] += 1

    def end_sit(seat: str, ts: int) -> None:
        if seat in sit_start and seat in turn_sign:
            agg[TEAM_OF[seat]][f"sit_ms_{turn_sign[seat]}"] += ts - sit_start.pop(seat)
        sit_start.pop(seat, None)

    for seat in on_turn.values():
        begin_turn(seat, t_ref)
    for e in events:
        typ, ts = e.get("type"), int(e.get("ts", 0))
        if typ == "thinking":
            (thinking.add if e.get("phase") == "start" else thinking.discard)(e.get("seat"))
        elif typ == "stall":
            seat = e.get("seat")
            if e.get("active"):
                if seat not in sit_start:
                    sit_start[seat] = ts
                    if seat in turn_sign:
                        agg[TEAM_OF[seat]][f"sits_{turn_sign[seat]}"] += 1
            else:
                end_sit(seat, ts)
        elif typ == "move":
            seat, board = e.get("seat"), e.get("board") or BOARD_OF.get(e.get("seat"), "A")
            end_sit(seat, ts)
            thinking.discard(seat)
            turn_sign.pop(seat, None)
            clocks.update({s: int(v) for s, v in (e.get("clocks") or {}).items()})
            t_ref = ts
            fens[board] = e.get("fen")
            on_turn[board] = OPPONENT[seat]
            begin_turn(on_turn[board], ts)

    flag_while = flag_other_check = None
    if end is not None and end.get("reason") == "flag" and end.get("loserSeat"):
        loser = end["loserSeat"]
        flag_while = "sitting" if loser in sit_start and loser not in thinking else "moving"
        other = "B" if end.get("board") == "A" else "A"
        flag_other_check = _in_check(fens.get(other))
    end_ts = int(end.get("ts", t_ref)) if end else t_ref
    for seat in list(sit_start):
        end_sit(seat, end_ts)
    return {"teams": agg, "flag_while": flag_while, "flag_other_check": flag_other_check}


def summarize_game(events: Iterable[dict]) -> dict:
    """Derive a per-game summary from one game's event list (docs/events.md).

    Per-team fields (under "teams": {"T1": {...}, "T2": {...}}):
      decisions      number of completed searches ("thinking" end events; start events if no ends)
      think_ms       total think time of those decisions
      nodes          total nodes searched
      moves, messages, stalls, captures
    Per-team clock fields (under "clock_teams", kept apart so "teams" stays as older summaries have it):
      turns_ahead/behind, sits_ahead/behind, sit_ms_ahead/behind   turns, turns with a sit, total sit length,
                     by the sign of the seat's diagonal clock (own minus partner's opponent) at turn start
    Top-level: game_id, setup, result, reason, board, loser_seat, loser_team, plies, duration_ms,
    flag (bool), seats (from game_start), flag_while ("sitting"/"moving"/None: what the flagged seat was doing),
    flag_other_check (bool/None: the other board's side to move was in check when the flag fell).
    """
    events = list(events)
    start = next((e for e in events if e.get("type") == "game_start"), {})
    end = next((e for e in reversed(events) if e.get("type") == "game_end"), None)

    teams = {t: {"decisions": 0, "think_ms": 0, "nodes": 0, "moves": 0, "messages": 0, "stalls": 0,
                 "captures": 0} for t in TEAMS}
    think_end = [e for e in events if e.get("type") == "thinking" and e.get("phase") == "end"]
    think = think_end or [e for e in events if e.get("type") == "thinking" and e.get("phase") == "start"]
    for e in think:
        t = teams[TEAM_OF[e["seat"]]]
        t["decisions"] += 1
        t["think_ms"] += int(e.get("think_ms") or 0)
        t["nodes"] += int(e.get("nodes") or 0)
    for e in events:
        typ = e.get("type")
        if typ == "move":
            teams[TEAM_OF[e["seat"]]]["moves"] += 1
            if e.get("capture"):
                teams[TEAM_OF[e["seat"]]]["captures"] += 1
        elif typ == "message":
            teams[TEAM_OF[e["fromSeat"]]]["messages"] += 1
        elif typ == "stall" and e.get("active"):
            teams[TEAM_OF[e["seat"]]]["stalls"] += 1

    capped = {t: 0 for t in TEAMS}  # searches whose budget hit the controller's max_nodes (thinking start "capped")
    for e in events:
        if e.get("type") == "thinking" and e.get("phase") == "start" and e.get("capped"):
            capped[TEAM_OF[e["seat"]]] += 1
    # Wave 2c (spec v5): per team, searches (thinking starts with nodes), searches that adopted team-tree visits,
    # the visits adopted, and boards whose carried plan biased the root. All 0 for players that report none.
    continuity = {t: {"searches": 0, "reuse_hits": 0, "reused_visits": 0, "plan_carries": 0} for t in TEAMS}
    for e in events:
        if e.get("type") == "thinking" and e.get("phase") == "start" and int(e.get("nodes") or 0) > 0:
            c = continuity[TEAM_OF[e["seat"]]]
            c["searches"] += 1
            if e.get("reused_visits"):
                c["reuse_hits"] += 1
                c["reused_visits"] += int(e["reused_visits"])
            c["plan_carries"] += int(e.get("plan_carried") or 0)
    cm = _clock_metrics(events, start, end)
    loser_seat = end.get("loserSeat") if end else None
    reason = end.get("reason") if end else "aborted"
    return {
        "game_id": start.get("game_id"),
        "setup": start.get("setup"),
        "seats": start.get("seats"),
        "result": end.get("result") if end else "draw",
        "reason": reason,
        "board": end.get("board") if end else None,
        "loser_seat": loser_seat,
        "loser_team": TEAM_OF.get(loser_seat) if loser_seat else None,
        "flag": reason == "flag",
        "plies": teams["T1"]["moves"] + teams["T2"]["moves"],
        "duration_ms": max((int(e.get("ts", 0)) for e in events), default=0),
        "teams": teams,
        "clock_teams": cm["teams"],
        "flag_while": cm["flag_while"],
        "flag_other_check": cm["flag_other_check"],
        "capped_decisions": capped,
        "continuity": continuity,
    }


# ---------------------------------------------------------------- contender-perspective per-game rows

def contender_row(summary: dict) -> dict:
    """Flatten a summary annotated with "contender_team" (and "opening_id") into contender/baseline
    numbers. Keys prefixed c_ are the contender's, b_ the baseline's."""
    ct = summary["contender_team"]
    bt = "T2" if ct == "T1" else "T1"
    res = summary["result"]
    score = 1.0 if res == ct else 0.5 if res == "draw" else 0.0
    flag = bool(summary.get("flag"))
    row = {
        "cluster": summary.get("opening_id", summary.get("game_id")),
        "score": score,
        "flag": 1.0 if flag else 0.0,
        "draw": 1.0 if res == "draw" else 0.0,
        "sitoff": 1.0 if summary.get("reason") == "sitoff_adjudicated" else 0.0,
    }
    for prefix, team, other in (("c", ct, bt), ("b", bt, ct)):
        t = summary["teams"][team]
        lost_on_flag = flag and summary.get("loser_team") == team
        row[f"{prefix}_flagged"] = 1.0 if lost_on_flag else 0.0
        row[f"{prefix}_flagged_sitting"] = 1.0 if lost_on_flag and summary.get("flag_while") == "sitting" else 0.0
        row[f"{prefix}_flagged_moving"] = 1.0 if lost_on_flag and summary.get("flag_while") == "moving" else 0.0
        won_on_flag = flag and summary.get("loser_team") == other
        row[f"{prefix}_flag_win_check"] = 1.0 if won_on_flag and summary.get("flag_other_check") else 0.0
        row[f"{prefix}_decisions"] = t["decisions"]
        row[f"{prefix}_capped"] = (summary.get("capped_decisions") or {}).get(team, 0)
        co = (summary.get("continuity") or {}).get(team, {})  # absent in older summaries: reads as 0
        for k in ("searches", "reuse_hits", "reused_visits", "plan_carries"):
            row[f"{prefix}_{k}"] = co.get(k, 0)
        row[f"{prefix}_think_ms"] = t["think_ms"]
        row[f"{prefix}_nodes"] = t["nodes"]
        row[f"{prefix}_messages"] = t["messages"]
        ct_ = (summary.get("clock_teams") or {}).get(team, {})  # absent in older summaries: reads as 0
        for k in ("turns", "sits", "sit_ms"):
            for sg in SIGNS:
                row[f"{prefix}_{k}_{sg}"] = ct_.get(f"{k}_{sg}", 0)
    return row


# ---------------------------------------------------------------- statistics

Statistic = Callable[[Sequence[dict]], float]


def mean_of(key: str) -> Statistic:
    def stat(rows: Sequence[dict]) -> float:
        return sum(r[key] for r in rows) / len(rows) if rows else float("nan")
    return stat


def ratio_of(num: str, den: str) -> Statistic:
    """Pooled ratio sum(num)/sum(den), e.g. mean think time per decision over all decisions."""
    def stat(rows: Sequence[dict]) -> float:
        d = sum(r[den] for r in rows)
        return sum(r[num] for r in rows) / d if d else float("nan")
    return stat


def cluster_bootstrap_ci(rows: Sequence[dict], statistic: Statistic, *, cluster_key: str = "cluster",
                         n_boot: int = 2000, seed: int = 0, alpha: float = 0.05) -> dict:
    """Point estimate plus percentile CI from resampling whole clusters with replacement.

    Returns {"est", "lo", "hi", "n", "clusters"}. With a single cluster the CI is degenerate (lo=hi=est).
    """
    rows = list(rows)
    est = statistic(rows)
    groups: dict = defaultdict(list)
    for r in rows:
        groups[r[cluster_key]].append(r)
    keys = sorted(groups, key=str)
    out = {"est": est, "lo": est, "hi": est, "n": len(rows), "clusters": len(keys)}
    if len(keys) < 2:
        return out
    rng = random.Random(seed)
    boots = []
    for _ in range(n_boot):
        sample: list[dict] = []
        for _k in keys:
            sample.extend(groups[keys[rng.randrange(len(keys))]])
        v = statistic(sample)
        if v == v:  # skip NaN
            boots.append(v)
    if not boots:
        return out
    boots.sort()
    lo_i = int((alpha / 2) * (len(boots) - 1))
    hi_i = int(round((1 - alpha / 2) * (len(boots) - 1)))
    out["lo"], out["hi"] = boots[lo_i], boots[hi_i]
    return out


METRICS: dict[str, Statistic] = {
    "score": mean_of("score"),  # contender team score: win 1, draw 0.5, loss 0
    "flag_rate": mean_of("flag"),  # fraction of games decided on time
    "contender_flagged": mean_of("c_flagged"),
    "baseline_flagged": mean_of("b_flagged"),
    "contender_latency_ms": ratio_of("c_think_ms", "c_decisions"),  # mean think time per decision
    "baseline_latency_ms": ratio_of("b_think_ms", "b_decisions"),
    "contender_nodes_per_game": mean_of("c_nodes"),
    "baseline_nodes_per_game": mean_of("b_nodes"),
    "contender_nodes_per_decision": ratio_of("c_nodes", "c_decisions"),
    # share of searches whose budget was cut to max_nodes (an equal-clock row wants ~0 on both sides)
    "contender_capped_share": ratio_of("c_capped", "c_decisions"),
    "baseline_capped_share": ratio_of("b_capped", "b_decisions"),
    "baseline_nodes_per_decision": ratio_of("b_nodes", "b_decisions"),
    # wave 2c (spec v5): share of searches that adopted a node of the team's tree, visits adopted per hit, and
    # carried plans applied per search (n/a for players that report no searches; 0 for ones that never fire)
    "contender_reuse_per_search": ratio_of("c_reuse_hits", "c_searches"),
    "baseline_reuse_per_search": ratio_of("b_reuse_hits", "b_searches"),
    "contender_reused_visits_per_hit": ratio_of("c_reused_visits", "c_reuse_hits"),
    "contender_plan_carries_per_search": ratio_of("c_plan_carries", "c_searches"),
    "baseline_plan_carries_per_search": ratio_of("b_plan_carries", "b_searches"),
    "contender_messages_per_game": mean_of("c_messages"),
    "baseline_messages_per_game": mean_of("b_messages"),
    "draw_rate": mean_of("draw"),
    "sitoff_rate": mean_of("sitoff"),  # games ended by sit-off adjudication
    # flags split by what the flagged seat was doing: sitting (waiting) or moving (thinking or about to)
    "contender_flagged_sitting": mean_of("c_flagged_sitting"),
    "contender_flagged_moving": mean_of("c_flagged_moving"),
    "baseline_flagged_sitting": mean_of("b_flagged_sitting"),
    "baseline_flagged_moving": mean_of("b_flagged_moving"),
    # games won on time while the other board's side to move was in check
    "contender_flag_win_other_check": mean_of("c_flag_win_check"),
    "baseline_flag_win_other_check": mean_of("b_flag_win_check"),
}
for _p, _who in (("c", "contender"), ("b", "baseline")):
    for _sg in SIGNS:
        # share of turns with a sit, and mean sit length, by the sign of the diagonal clock at turn start
        METRICS[f"{_who}_pass_rate_{_sg}"] = ratio_of(f"{_p}_sits_{_sg}", f"{_p}_turns_{_sg}")
        METRICS[f"{_who}_sit_ms_{_sg}"] = ratio_of(f"{_p}_sit_ms_{_sg}", f"{_p}_sits_{_sg}")


def compute_stats(summaries: Sequence[dict], *, n_boot: int = 2000, seed: int = 0) -> dict:
    """Contender-vs-baseline metrics with 95% cluster-bootstrap CIs.

    Each summary must carry "contender_team" ("T1"/"T2") and "opening_id" (cluster); summaries with an
    "error" key are skipped. Returns {"games": n, "wins"/"draws"/"losses": ..., "metrics": {name: ci}}.
    """
    rows = [contender_row(s) for s in summaries if "error" not in s]
    wins = sum(1 for r in rows if r["score"] == 1.0)
    draws = sum(1 for r in rows if r["score"] == 0.5)
    return {
        "games": len(rows),
        "wins": wins,
        "draws": draws,
        "losses": len(rows) - wins - draws,
        "metrics": {name: cluster_bootstrap_ci(rows, f, n_boot=n_boot, seed=seed) for name, f in METRICS.items()},
    }


def format_table(stats: dict, contender: str = "contender", baseline: str = "baseline") -> str:
    """Readable results table."""
    m = stats["metrics"]

    def ci(name: str, fmt: str = "{:.3f}") -> str:
        c = m[name]
        if c["est"] != c["est"]:
            return "n/a"
        return f"{fmt.format(c['est'])}  [{fmt.format(c['lo'])}, {fmt.format(c['hi'])}]"

    clusters = m["score"]["clusters"]
    lines = [
        f"{contender} (contender) vs {baseline} (baseline): {stats['games']} games, {clusters} opening pairs",
        f"W/D/L (contender): {stats['wins']}/{stats['draws']}/{stats['losses']}",
        "",
        f"{'metric':<28}{'estimate  [95% CI, cluster bootstrap]'}",
        f"{'contender score':<28}{ci('score')}",
        f"{'flag rate (any side)':<28}{ci('flag_rate')}",
        "",
        f"{'':<22}{contender + ' (contender)':<36}{baseline + ' (baseline)'}",
        f"{'lost on time':<22}{ci('contender_flagged'):<36}{ci('baseline_flagged')}",
        f"{'latency ms/decision':<22}{ci('contender_latency_ms', '{:.0f}'):<36}{ci('baseline_latency_ms', '{:.0f}')}",
        f"{'nodes/decision':<22}{ci('contender_nodes_per_decision', '{:.0f}'):<36}"
        f"{ci('baseline_nodes_per_decision', '{:.0f}')}",
        f"{'nodes/game':<22}{ci('contender_nodes_per_game', '{:.0f}'):<36}{ci('baseline_nodes_per_game', '{:.0f}')}",
        f"{'messages/game':<22}{ci('contender_messages_per_game', '{:.1f}'):<36}"
        f"{ci('baseline_messages_per_game', '{:.1f}')}",
        f"{'flagged while sitting':<22}{ci('contender_flagged_sitting'):<36}{ci('baseline_flagged_sitting')}",
        f"{'flagged while moving':<22}{ci('contender_flagged_moving'):<36}{ci('baseline_flagged_moving')}",
        f"{'flag win, other chk':<22}{ci('contender_flag_win_other_check'):<36}"
        f"{ci('baseline_flag_win_other_check')}",
        f"{'pass rate, ahead':<22}{ci('contender_pass_rate_ahead'):<36}{ci('baseline_pass_rate_ahead')}",
        f"{'pass rate, behind':<22}{ci('contender_pass_rate_behind'):<36}{ci('baseline_pass_rate_behind')}",
        f"{'sit ms, ahead':<22}{ci('contender_sit_ms_ahead', '{:.0f}'):<36}{ci('baseline_sit_ms_ahead', '{:.0f}')}",
        f"{'sit ms, behind':<22}{ci('contender_sit_ms_behind', '{:.0f}'):<36}"
        f"{ci('baseline_sit_ms_behind', '{:.0f}')}",
        "",
        f"{'draw rate':<28}{ci('draw_rate')}",
        f"{'sit-off adjudicated':<28}{ci('sitoff_rate')}",
    ]
    return "\n".join(lines)
