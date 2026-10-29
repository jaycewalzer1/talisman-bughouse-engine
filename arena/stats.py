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

from arena.interfaces import TEAM_OF

TEAMS = ("T1", "T2")


def summarize_game(events: Iterable[dict]) -> dict:
    """Derive a per-game summary from one game's event list (docs/events.md).

    Per-team fields (under "teams": {"T1": {...}, "T2": {...}}):
      decisions      number of completed searches ("thinking" end events; start events if no ends)
      think_ms       total think time of those decisions
      nodes          total nodes searched
      moves, messages, stalls, captures
    Top-level: game_id, setup, result, reason, board, loser_seat, loser_team, plies, duration_ms,
    flag (bool), seats (from game_start).
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
    }


# ---------------------------------------------------------------- contender-perspective per-game rows

def contender_row(summary: dict) -> dict:
    """Flatten a summary annotated with "contender_team" (and "opening_id") into contender/baseline
    numbers. Keys prefixed c_ are the contender's, b_ the baseline's."""
    ct = summary["contender_team"]
    bt = "T2" if ct == "T1" else "T1"
    res = summary["result"]
    score = 1.0 if res == ct else 0.5 if res == "draw" else 0.0
    row = {
        "cluster": summary.get("opening_id", summary.get("game_id")),
        "score": score,
        "flag": 1.0 if summary.get("flag") else 0.0,
        "c_flagged": 1.0 if summary.get("flag") and summary.get("loser_team") == ct else 0.0,
        "b_flagged": 1.0 if summary.get("flag") and summary.get("loser_team") == bt else 0.0,
    }
    for prefix, team in (("c", ct), ("b", bt)):
        t = summary["teams"][team]
        row[f"{prefix}_decisions"] = t["decisions"]
        row[f"{prefix}_think_ms"] = t["think_ms"]
        row[f"{prefix}_nodes"] = t["nodes"]
        row[f"{prefix}_messages"] = t["messages"]
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
    "baseline_nodes_per_decision": ratio_of("b_nodes", "b_decisions"),
    "contender_messages_per_game": mean_of("c_messages"),
    "baseline_messages_per_game": mean_of("b_messages"),
}


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
    ]
    return "\n".join(lines)
