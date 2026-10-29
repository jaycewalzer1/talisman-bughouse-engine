"""Opponent pool and rotation schedule for training against stock Hivemind without overfitting to it.

    python -m arena.pool show     --pool arena/pools/phase1.json
    python -m arena.pool schedule --pool arena/pools/phase1.json --pairs 20
    python -m arena.pool run      --pool arena/pools/phase1.json --pairs 20 --workers 1 --out runs/pool_smoke
    python -m arena.pool add-snapshot --pool arena/pools/phase1.json --name it01 --path runs/phase1/it01.pt

The pool (a JSON file) names one **contender** (the team being trained) and its **opponents**, each in a
group:

* ``target``: the opponent we want to beat (stock Hivemind, centralized, equal compute).
* ``snapshot``: frozen earlier versions of the contender, so it keeps general strength. Only the latest
  ``snapshot_window`` count; the list grows as Phase 1 saves checkpoints (``add-snapshot``).
* ``variant``: stock Hivemind with other settings (node budget, sit rules, split setups) and the
  heuristic team, so the contender cannot fit one exact opponent.

``shares`` gives each group's fraction of training games (default 0.5 / 0.3 / 0.2). An empty group's share
goes to ``variant`` (so before the first snapshot exists the target keeps its 0.5), or, if ``variant`` is
empty too, is spread over the remaining groups in proportion. The rotation is deterministic: each opening pair goes to the group
furthest below its share so far, then to the entry of that group furthest below its weight (smooth
weighted round-robin), so any stretch of the schedule is close to the target mix. Each pair is played twice
with the teams swapped, as in arena.run. Training schedules draw only from the training openings; the
evaluation openings (arena.openings.EVAL_PAIRS) are held out for the final match.

A schedule game is a normal arena.run spec with per-team players under ``teams``, so arena.run.play_game
plays it. ``equal_compute`` marks games where both teams had the same kind and node budget.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field

from arena.openings import OPENING_SETS, paired_schedule

GROUPS = ("target", "snapshot", "variant")
DEFAULT_SHARES = {"target": 0.5, "snapshot": 0.3, "variant": 0.2}


@dataclass(frozen=True)
class TeamSpec:
    name: str
    setup: str  # centralized | silent | sharing
    kind: str = "hivemind"
    nodes: int = 200
    player_kw: dict = field(default_factory=dict)  # compute, lend, max_consecutive_sits, sit_ms, ...
    group: str = "target"  # opponents only: target | snapshot | variant
    weight: float = 1.0
    path: str | None = None  # snapshot checkpoint, for agents trained in Phase 1

    def team_dict(self) -> dict:
        d = {"setup": self.setup, "kind": self.kind, "nodes": self.nodes, "player_kw": dict(self.player_kw)}
        if self.path:
            d["path"] = self.path
        return d


@dataclass
class Pool:
    contender: TeamSpec
    opponents: list[TeamSpec]
    shares: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_SHARES))
    snapshot_window: int = 5

    # ------------------------------------------------------------------ persistence

    @classmethod
    def from_dict(cls, d: dict) -> "Pool":
        pool = cls(contender=TeamSpec(**d["contender"]), opponents=[TeamSpec(**o) for o in d["opponents"]],
                   shares=dict(d.get("shares", DEFAULT_SHARES)), snapshot_window=int(d.get("snapshot_window", 5)))
        pool.validate()
        return pool

    @classmethod
    def load(cls, path: str) -> "Pool":
        with open(path) as f:
            return cls.from_dict(json.load(f))

    def to_dict(self) -> dict:
        return {"contender": asdict(self.contender), "shares": self.shares, "snapshot_window": self.snapshot_window,
                "opponents": [asdict(o) for o in self.opponents]}

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
            f.write("\n")

    def validate(self) -> None:
        names = [o.name for o in self.opponents]
        if len(set(names)) != len(names):
            raise ValueError(f"opponent names must be unique: {names}")
        for o in self.opponents:
            if o.group not in GROUPS:
                raise ValueError(f"{o.name}: group {o.group!r} not in {GROUPS}")
            if o.weight <= 0:
                raise ValueError(f"{o.name}: weight must be > 0")
        if set(self.shares) - set(GROUPS) or any(v < 0 for v in self.shares.values()):
            raise ValueError(f"shares must be non-negative and keyed by {GROUPS}: {self.shares}")
        if not any(self.shares.get(g, 0) > 0 and self.members(g) for g in GROUPS):
            raise ValueError("no group has both a positive share and at least one opponent")

    # ------------------------------------------------------------------ membership

    def members(self, group: str) -> list[TeamSpec]:
        ms = [o for o in self.opponents if o.group == group]
        if group == "snapshot":
            ms = ms[-self.snapshot_window:] if self.snapshot_window > 0 else []
        return ms

    def effective_shares(self) -> dict[str, float]:
        live = {g: self.shares.get(g, 0.0) for g in GROUPS if self.members(g) and self.shares.get(g, 0.0) > 0}
        if "variant" in live:
            live["variant"] += sum(self.shares.get(g, 0.0) for g in GROUPS if g not in live)
        total = sum(live.values())
        return {g: v / total for g, v in live.items()}

    def add_snapshot(self, name: str, path: str, *, kind: str = "agent", setup: str | None = None,
                     nodes: int | None = None, player_kw: dict | None = None) -> TeamSpec:
        """Register a frozen contender checkpoint as a snapshot opponent (defaults copied from the
        contender, kind "agent")."""
        snap = TeamSpec(name=name, setup=setup or self.contender.setup, kind=kind,
                        nodes=nodes if nodes is not None else self.contender.nodes,
                        player_kw=dict(player_kw if player_kw is not None else self.contender.player_kw),
                        group="snapshot", path=path)
        self.opponents.append(snap)
        self.validate()
        return snap


# ---------------------------------------------------------------------- rotation

class _SmoothRR:
    """Smooth weighted round-robin: each pick goes to the key furthest below its share so far."""

    def __init__(self) -> None:
        self.counts: dict[str, int] = {}

    def pick(self, weights: dict[str, float]) -> str:
        total = sum(weights.values())
        n = sum(self.counts.get(k, 0) for k in weights) + 1
        best = max(weights, key=lambda k: (weights[k] / total * n - self.counts.get(k, 0), weights[k]))
        self.counts[best] = self.counts.get(best, 0) + 1
        return best


def make_schedule(pool: Pool, n_pairs: int, *, seed: int = 0, openings: str = "train", base_ms: int = 60000,
                  increment_ms: int = 0, channel_delay_ms: int = 0, max_plies: int = 400) -> list[dict]:
    """2 * n_pairs game specs (each opening pair twice, teams swapped), opponents rotated by group share."""
    shares = pool.effective_shares()
    group_rr = _SmoothRR()
    member_rr = {g: _SmoothRR() for g in shares}
    pairs = paired_schedule(2 * n_pairs, seed, OPENING_SETS[openings])
    c = pool.contender
    specs = []
    for k in range(n_pairs):
        group = group_rr.pick(shares)
        members = {o.name: o for o in pool.members(group)}
        weights = {n: (1.0 if group == "snapshot" else o.weight) for n, o in members.items()}
        opp = members[member_rr[group].pick(weights)]
        for _ in range(2):
            pair, swap = next(pairs)
            contender_team = "T2" if swap else "T1"
            opp_team = "T1" if swap else "T2"
            i = len(specs)
            specs.append({
                "index": i + 1, "game_id": f"p{i + 1:05d}", "pair_index": k,
                "contender_team": contender_team, "contender": c.name, "opponent": opp.name, "group": group,
                "equal_compute": opp.kind == c.kind and opp.nodes == c.nodes,
                "opening_id": pair.id, "opening": pair.name, "fens": pair.fens, "openings": openings, "swap": swap,
                "teams": {contender_team: c.team_dict(), opp_team: opp.team_dict()},
                "t1_setup": (opp if swap else c).setup, "t2_setup": (c if swap else opp).setup,
                "kind": c.kind, "nodes": c.nodes, "seed": seed * 100003 + i,
                "base_ms": base_ms, "increment_ms": increment_ms, "channel_delay_ms": channel_delay_ms,
                "max_plies": max_plies, "charge_wall_time": False,
            })
    return specs


def schedule_mix(specs: list[dict]) -> dict:
    """Games per group and per opponent, and each group's fraction of the schedule."""
    n = len(specs) or 1
    by_group: dict[str, int] = {}
    by_opp: dict[str, int] = {}
    for s in specs:
        by_group[s["group"]] = by_group.get(s["group"], 0) + 1
        by_opp[s["opponent"]] = by_opp.get(s["opponent"], 0) + 1
    return {"games": len(specs), "groups": {g: {"games": v, "share": round(v / n, 3)} for g, v in by_group.items()},
            "opponents": by_opp}


# ---------------------------------------------------------------------- running

def unbuildable(specs: list[dict]) -> list[str]:
    from arena.players import KINDS

    bad = {f"{s['opponent']} (kind {t['kind']})" for s in specs for t in s["teams"].values() if t["kind"] not in KINDS}
    return sorted(bad)


def results_by_opponent(summaries: list[dict], n_boot: int = 2000, seed: int = 0) -> dict:
    """Contender score per opponent and per group, with cluster-bootstrap CIs (arena.stats)."""
    from arena.stats import compute_stats

    ok = [s for s in summaries if "error" not in s]
    out: dict = {"by_group": {}, "by_opponent": {}}
    for key, bucket in (("group", "by_group"), ("opponent", "by_opponent")):
        for name in sorted({s[key] for s in ok}):
            sub = [s for s in ok if s[key] == name]
            st = compute_stats(sub, n_boot=n_boot, seed=seed)
            sc = st["metrics"]["score"]
            out[bucket][name] = {"games": st["games"], "wins": st["wins"], "losses": st["losses"],
                                 "score": sc["est"], "ci95": [sc["lo"], sc["hi"]]}
    return out


def format_results(res: dict) -> str:
    lines = [f"{'':<26} {'games':>6} {'contender score':>16} {'95% CI':>18}"]
    for bucket, title in (("by_group", "group"), ("by_opponent", "opponent")):
        lines.append(f"-- by {title}")
        for name, r in res[bucket].items():
            ci = f"[{r['ci95'][0]:.3f}, {r['ci95'][1]:.3f}]"
            lines.append(f"{name:<26} {r['games']:>6} {r['score']:>16.3f} {ci:>18}")
    return "\n".join(lines)


def run_schedule(specs: list[dict], out: str, workers: int = 1, n_boot: int = 2000, seed: int = 0) -> dict:
    from arena.run import run_one

    os.makedirs(out, exist_ok=True)
    for s in specs:
        s["out"] = out
    keep = ("opponent", "group", "equal_compute", "contender")
    summaries: list[dict] = []

    def done(spec: dict, summary: dict) -> None:
        summary.update({k: spec[k] for k in keep})
        summaries.append(summary)
        tag = summary.get("error") or f"{summary['result']} by {summary['reason']}"
        print(f"[{len(summaries)}/{len(specs)}] {spec['game_id']} vs {spec['opponent']:<22} "
              f"contender={spec['contender_team']} -> {tag}", flush=True)

    t0 = time.perf_counter()
    if workers <= 1:
        for s in specs:
            done(s, run_one(s))
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(run_one, s): s for s in specs}
            for fut in as_completed(futs):
                done(futs[fut], fut.result())
    summaries.sort(key=lambda s: s["game_id"])
    res = results_by_opponent(summaries, n_boot=n_boot, seed=seed)
    report = {"wall_s": round(time.perf_counter() - t0, 1), "mix": schedule_mix(specs), "results": res,
              "errors": sum("error" in s for s in summaries), "games": summaries}
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(report, f, indent=2)
    return report


# ---------------------------------------------------------------------- CLI

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m arena.pool", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("show", "schedule", "run", "add-snapshot"):
        p = sub.add_parser(name)
        p.add_argument("--pool", default="arena/pools/phase1.json")
        if name in ("schedule", "run"):
            p.add_argument("--pairs", type=int, default=20, help="opening pairs (2 games each)")
            p.add_argument("--openings", default="train", choices=("train", "eval", "all"))
            p.add_argument("--seed", type=int, default=0)
            p.add_argument("--base-ms", type=int, default=60000)
            p.add_argument("--channel-delay-ms", type=int, default=0)
            p.add_argument("--max-plies", type=int, default=400)
        if name == "schedule":
            p.add_argument("--out", default=None, help="write the schedule as JSON lines")
        if name == "run":
            p.add_argument("--workers", type=int, default=1)
            p.add_argument("--boot", type=int, default=2000)
            p.add_argument("--out", default="runs/pool")
        if name == "add-snapshot":
            p.add_argument("--name", required=True)
            p.add_argument("--path", required=True)
            p.add_argument("--kind", default="agent")
    args = ap.parse_args(argv)
    pool = Pool.load(args.pool)

    if args.cmd == "show":
        print(f"contender: {pool.contender.name} ({pool.contender.setup}, {pool.contender.kind}, "
              f"{pool.contender.nodes} nodes)")
        eff = pool.effective_shares()
        for g in GROUPS:
            ms = pool.members(g)
            print(f"{g:<9} share {pool.shares.get(g, 0):.2f} -> {eff.get(g, 0):.2f} effective, "
                  f"{len(ms)} active: {', '.join(m.name for m in ms) or '-'}")
        return 0

    if args.cmd == "add-snapshot":
        snap = pool.add_snapshot(args.name, args.path, kind=args.kind)
        pool.save(args.pool)
        print(f"added snapshot {snap.name} ({snap.path}); {len(pool.members('snapshot'))} active")
        return 0

    specs = make_schedule(pool, args.pairs, seed=args.seed, openings=args.openings, base_ms=args.base_ms,
                          channel_delay_ms=args.channel_delay_ms, max_plies=args.max_plies)
    if args.cmd == "schedule":
        if args.out:
            with open(args.out, "w") as f:
                for s in specs:
                    f.write(json.dumps(s) + "\n")
        for s in specs[::2]:
            print(f"pair {s['pair_index'] + 1:>4}  {s['group']:<8} {s['opponent']:<24} {s['opening']}")
        print(json.dumps(schedule_mix(specs), indent=2))
        return 0

    bad = unbuildable(specs)
    if bad:
        print(f"cannot play these opponents yet (no player kind for them): {', '.join(bad)}", file=sys.stderr)
        return 1
    report = run_schedule(specs, args.out, workers=args.workers, n_boot=args.boot, seed=args.seed)
    print()
    print(format_results(report["results"]))
    print(f"\nwrote {args.out}/summary.json ({report['wall_s']:.0f}s, {report['errors']} errors)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
