"""Opponent pool and rotation schedule for training against stock Hivemind without overfitting to it.

    python -m arena.pool show     --pool arena/pools/phase1.json
    python -m arena.pool schedule --pool arena/pools/phase1.json --pairs 20
    python -m arena.pool run      --pool arena/pools/phase1.json --pairs 20 --workers 1 --out runs/pool_smoke
    python -m arena.pool add-snapshot --pool arena/pools/phase1.json --name it01 --path runs/phase1/it01.pt
    python -m arena.pool run      --agent --features mate,danger --policy runs/x/latest.pt \
        --opponent stock-centralized --openings eval --pairs 800 --workers 3 --out runs/eval/x

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

``run --agent`` scores one fixed scheduler policy: the contender plays as the scheduler agent with no forced
exploration (untrained without ``--policy``), and ``--opponent`` keeps a single opponent. That is the evaluation
that tells two policies apart; scores logged during training include exploration and a changing opponent mix.
``run --setup centralized --nodes 100`` does the same for stock Hivemind in a chosen team setup and node budget,
which is how the team setups and think budgets are compared with no scheduler involved.

A schedule game is a normal arena.run spec with per-team players under ``teams``, so arena.run.play_game
plays it. ``equal_compute`` marks games where both teams had the same kind and node budget.

Evaluation harness (docs/custom_engine_spec_v3.md, audit WP-A), all off by default:

* ``run --skip-done`` resumes a row: games whose ``game_NNNN.jsonl`` already ends in a clean ``game_end`` are
  rescored from that file instead of played again, so a row cut off by a job's time limit is finished by
  submitting it again with the same arguments. ``specs.jsonl`` (the schedule) is written at the start of every
  run and a resume refuses a different schedule. ``results.jsonl`` gets one game summary per line as each game
  lands (tools/sprt.py and tools/rescore.py read it).
* ``run --sprt ELO0,ELO1[,ALPHA,BETA]`` runs the GSPRT of tools/sprt.py over normalised Elo after every finished
  game pair and stops the row when it is decided (the decision goes into summary.json either way).
* Per-arm ``ms_per_node``: ``measure --name X`` measures wall ms per *searched* node for X's binary+network on
  this machine and stores it in X's ``player_kw`` in the pool file; ``run --measure-ms-per-node`` does the same at
  run time for every engine team whose ``player_kw`` has no ``ms_per_node`` (or ``"ms_per_node": "measure"``),
  one measurement per distinct binary+network, kept in ``<out>/ms_per_node.json`` so a resumed row charges the
  same values. Without either, the player's default (``hivemind.MS_PER_NODE``) applies as before.
* Results per opponent also carry ``contender_nodes_per_game`` / ``baseline_nodes_per_game``, the ``reasons``
  count (``GameResult.reason``), the pentanomial pair counts and Elo / normalised Elo.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import CancelledError, ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field, replace

from arena.openings import OPENING_SETS, paired_schedule

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO not in sys.path:  # tools.sprt is imported from the repo root (a namespace package)
    sys.path.append(_REPO)

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


def equalize_ms_per_node(pool: Pool) -> Pool:
    """Every engine team charged the first engine opponent's ms_per_node (its own, or the player default when it
    has none): the equal-nodes diagnostic, which takes network speed out of an equal-clock row."""
    from arena.players.hivemind import MS_PER_NODE

    ref = next((o for o in pool.opponents if o.kind in ENGINE_KINDS), None)
    if ref is None:
        raise ValueError("--equal-ms-per-node needs an engine opponent")
    ms = ref.player_kw.get("ms_per_node")
    ms = float(MS_PER_NODE if ms in (None, "measure") else ms)
    fix = lambda t: replace(t, player_kw={**t.player_kw, "ms_per_node": ms}) if t.kind in ENGINE_KINDS else t
    return replace(pool, contender=fix(pool.contender), opponents=[fix(o) for o in pool.opponents])


# ---------------------------------------------------------------------- rotation

def for_evaluation(pool: Pool, *, agent: bool = False, features: list[str] | None = None, policy: str | None = None,
                   opponent: str | None = None, setup: str | None = None, lend: bool = False,
                   nodes: int | None = None, opponent_nodes: int | None = None, rule: str | None = None,
                   lend_share: float | None = None, contender_kw: dict | None = None,
                   opponent_kw: dict | None = None, jitter: tuple[float, float] | None = None) -> Pool:
    """The pool narrowed for scoring one fixed team against one opponent.

    With agent (or features or policy) the contender plays as the scheduler agent, untrained when policy is None.
    rule adds one of the agent's fixed hand-written rules (arena.players.scheduler_player.RULES).
    With setup the contender is stock Hivemind in that team setup (lend only means something for a split team;
    lend_share is the part of the partner's idle machine it borrows, all of it when None).
    nodes and opponent_nodes replace the node budgets. With opponent, only that opponent is played.
    contender_kw and opponent_kw are merged into the player options (e.g. clock_k, serial, max_consecutive_sits).
    jitter (lo, hi) gives every centralized Hivemind team on both sides a think-time factor per search drawn from
    that range, so the two teams do not move in lock step (custom-engine gate); other teams are left as they are."""
    contender, opponents = pool.contender, pool.opponents
    if lend_share is not None and (setup in (None, "centralized") or not lend or not 0.0 < lend_share <= 1.0):
        raise ValueError("lend_share needs a split setup with lend, and a value in (0, 1]")
    if setup is not None:
        split = {} if setup == "centralized" else {"compute": "throughput", "lend": lend}
        if lend_share is not None:
            split["lend_share"] = lend_share
        contender = replace(contender, setup=setup, kind="hivemind", player_kw=split)
    if agent or features or policy or rule:
        contender = replace(contender, kind="agent", path=os.path.abspath(policy) if policy else None,
                            player_kw={**contender.player_kw, "features": list(features or []),
                                       **({"rule": rule} if rule else {})})
    if nodes is not None:
        contender = replace(contender, nodes=nodes)
    if opponent is not None:
        opponents = [replace(o, group="target") for o in opponents if o.name == opponent]
        if not opponents:
            names = ", ".join(o.name for o in pool.opponents)
            raise ValueError(f"no opponent named {opponent!r}; the pool has {names}")
    if opponent_nodes is not None:
        opponents = [replace(o, nodes=opponent_nodes) for o in opponents]
    if contender_kw:
        contender = replace(contender, player_kw={**contender.player_kw, **contender_kw})
    if opponent_kw:
        opponents = [replace(o, player_kw={**o.player_kw, **opponent_kw}) for o in opponents]
    if jitter is not None:
        def jittered(t: TeamSpec) -> TeamSpec:
            if t.kind != "hivemind" or t.setup != "centralized":
                return t
            return replace(t, player_kw={**t.player_kw, "jitter": [float(jitter[0]), float(jitter[1])]})
        contender = jittered(contender)
        opponents = [jittered(o) for o in opponents]
    return replace(pool, contender=contender, opponents=opponents)


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


def _compute_kind(kind: str) -> str:
    return "hivemind" if kind == "agent" else kind  # an agent spends the same engine nodes as stock Hivemind


def make_schedule(pool: Pool, n_pairs: int, *, seed: int = 0, openings: str = "train", base_ms: int = 60000,
                  increment_ms: int = 0, channel_delay_ms: int = 0, max_plies: int = 400,
                  skip_pairs: int = 0, rules: str | None = None) -> list[dict]:
    """2 * n_pairs game specs (each opening pair twice, teams swapped), opponents rotated by group share.

    skip_pairs moves the rotation that many picks along first. Training passes the pairs already played, so a short
    iteration continues the rotation and does not meet the same first few opponents every time."""
    shares = pool.effective_shares()
    group_rr = _SmoothRR()
    member_rr = {g: _SmoothRR() for g in shares}
    pairs = paired_schedule(2 * n_pairs, seed, OPENING_SETS[openings])
    c = pool.contender
    specs = []
    for k in range(-skip_pairs, n_pairs):
        group = group_rr.pick(shares)
        members = {o.name: o for o in pool.members(group)}
        weights = {n: (1.0 if group == "snapshot" else o.weight) for n, o in members.items()}
        opp = members[member_rr[group].pick(weights)]
        if k < 0:
            continue
        for _ in range(2):
            pair, swap = next(pairs)
            contender_team = "T2" if swap else "T1"
            opp_team = "T1" if swap else "T2"
            i = len(specs)
            specs.append({
                "index": i + 1, "game_id": f"p{i + 1:05d}", "pair_index": k,
                "contender_team": contender_team, "contender": c.name, "opponent": opp.name, "group": group,
                "equal_compute": _compute_kind(opp.kind) == _compute_kind(c.kind) and opp.nodes == c.nodes,
                "opening_id": pair.id, "opening": pair.name, "fens": pair.fens, "openings": openings, "swap": swap,
                "teams": {contender_team: c.team_dict(), opp_team: opp.team_dict()},
                "t1_setup": (opp if swap else c).setup, "t2_setup": (c if swap else opp).setup,
                "kind": c.kind, "nodes": c.nodes, "seed": seed * 100003 + i,
                "base_ms": base_ms, "increment_ms": increment_ms, "channel_delay_ms": channel_delay_ms,
                "max_plies": max_plies, "charge_wall_time": False,
            })
            if rules and rules != "v2":
                specs[-1]["rules"] = rules  # arena.run.play_game: "v3" = arena.sim.RULES_V3
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
    """Contender score per opponent and per group, with cluster-bootstrap CIs (arena.stats), the nodes each side
    searched per game, game-end reasons, and the pentanomial pair counts with Elo / normalised Elo
    (tools/sprt.py)."""
    from arena.stats import compute_stats
    from tools.sprt import elo, nelo, pentanomial

    ok = [s for s in summaries if "error" not in s]
    out: dict = {"by_group": {}, "by_opponent": {}}
    for key, bucket in (("group", "by_group"), ("opponent", "by_opponent")):
        for name in sorted({s[key] for s in ok}):
            sub = [s for s in ok if s[key] == name]
            st = compute_stats(sub, n_boot=n_boot, seed=seed)
            sc = st["metrics"]["score"]
            reasons: dict[str, int] = {}
            for s in sub:
                reasons[str(s.get("reason"))] = reasons.get(str(s.get("reason")), 0) + 1
            row = {"games": st["games"], "wins": st["wins"], "losses": st["losses"],
                   "score": sc["est"], "ci95": [sc["lo"], sc["hi"]], "draws": st["draws"],
                   "clusters": sc["clusters"], "reasons": dict(sorted(reasons.items()))}
            for m in ("contender_nodes_per_game", "baseline_nodes_per_game", "flag_rate"):
                c = st["metrics"][m]
                row[m] = c["est"]
                row[m + "_ci95"] = [c["lo"], c["hi"]]
            try:
                penta = pentanomial(sub)
            except ValueError:
                penta = [0] * 5
            row.update({"pentanomial": penta, "elo": elo(penta), "nelo": nelo(penta)})
            out[bucket][name] = row
    return out


def format_results(res: dict) -> str:
    lines = [f"{'':<26} {'games':>6} {'contender score':>16} {'95% CI':>18} {'nodes/game c':>13} "
             f"{'nodes/game b':>13}  reasons"]
    for bucket, title in (("by_group", "group"), ("by_opponent", "opponent")):
        lines.append(f"-- by {title}")
        for name, r in res[bucket].items():
            ci = f"[{r['ci95'][0]:.3f}, {r['ci95'][1]:.3f}]"
            cn, bn = r.get("contender_nodes_per_game", float("nan")), r.get("baseline_nodes_per_game", float("nan"))
            reasons = " ".join(f"{k}={v}" for k, v in r.get("reasons", {}).items())
            lines.append(f"{name:<26} {r['games']:>6} {r['score']:>16.3f} {ci:>18} {cn:>13.0f} {bn:>13.0f}  {reasons}")
    return "\n".join(lines)


# ---------------------------------------------------------------------- resume and rescoring

SUMMARY_KEEP = ("opponent", "group", "equal_compute", "contender", "pair_index")
_SPEC_IDENTITY = ("game_id", "index", "opening_id", "contender_team", "opponent", "teams", "base_ms",
                  "increment_ms", "max_plies", "seed", "rules")


def game_file(spec: dict) -> str:
    """The per-game event file arena.run.run_one writes for this spec."""
    return f"game_{spec['index']:04d}.jsonl"


def summary_from_game_file(spec: dict, path: str) -> dict | None:
    """The summary run_one would have returned for this game, rebuilt from its event file; None when the file is
    missing, cut short (no final ``game_end``) or records a crashed game (those are played again)."""
    from arena.stats import summarize_game

    if not os.path.exists(path):
        return None
    events = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                return None
    if not events or events[-1].get("type") != "game_end" or events[-1].get("error"):
        return None
    summary = summarize_game(events)
    summary.update({k: spec[k] for k in ("game_id", "contender_team", "opening_id", "opening", "swap",
                                         "t1_setup", "t2_setup") if k in spec})
    summary["file"] = os.path.basename(path)
    summary.update({k: spec[k] for k in SUMMARY_KEEP if k in spec})
    return summary


def _spec_identity(spec: dict) -> dict:
    return {k: spec.get(k) for k in _SPEC_IDENTITY}


def _write_jsonl(path: str, rows: list[dict]) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        for r in rows:
            f.write(json.dumps(r, separators=(",", ":")) + "\n")
    os.replace(tmp, path)


def _check_resume_schedule(out: str, specs: list[dict]) -> None:
    path = os.path.join(out, "specs.jsonl")
    if not os.path.exists(path):
        return
    with open(path) as f:
        old = {d["game_id"]: _spec_identity(d) for d in (json.loads(l) for l in f if l.strip())}
    for s in specs:
        if s["game_id"] in old and old[s["game_id"]] != json.loads(json.dumps(_spec_identity(s))):
            raise ValueError(f"{out} was started with a different schedule ({s['game_id']} differs): resume it "
                             "with the same arguments, or use another --out")


# ---------------------------------------------------------------------- per-arm ms_per_node

ENGINE_KINDS = ("hivemind", "agent")


def measure_searched_ms_per_node(binary: str | None = None, model: str | None = None, nodes: int = 400,
                                 repeats: int = 3) -> dict:
    """Wall ms per *searched* node for one binary+network on this machine: startpos, ``go nodes N`` ``repeats``
    times after a warm-up, total wall / total nodes the engine reports (contract 5 charges searched nodes, so the
    rate is per searched node; hivemind.measure_ms_per_node divides by the requested count instead). Also returns
    the per-requested-node rate for comparison with the old fixed MS_PER_NODE."""
    from arena.players.hivemind import DEFAULT_MODEL, HivemindProcess, resolve_binary

    binary, model = resolve_binary(binary), os.path.abspath(model or DEFAULT_MODEL)
    start = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR[] w KQkq - 0 1"
    p = HivemindProcess(binary, model)
    try:
        p.search(start, start, team="T1", nodes=8)  # warm-up (a TensorRT engine file builds here on first use)
        res = [p.search(start, start, team="T1", nodes=nodes) for _ in range(repeats)]
    finally:
        p.close()
    wall, searched = sum(r.wall_ms for r in res), sum(max(1, r.nodes) for r in res)
    return {"binary": binary, "model": model, "nodes_requested": nodes, "repeats": repeats,
            "nodes_searched_mean": searched / repeats, "wall_ms_mean": wall / repeats,
            "ms_per_node": wall / searched, "ms_per_requested_node": wall / (nodes * repeats),
            "host": os.uname().nodename, "when": time.strftime("%Y-%m-%dT%H:%M:%S")}


def arm_key(team: TeamSpec) -> str:
    """binary|model of an engine team, resolved to absolute paths (one ms_per_node per key)."""
    from arena.players.hivemind import DEFAULT_MODEL, resolve_binary

    kw = team.player_kw
    return f"{resolve_binary(kw.get('binary'))}|{os.path.abspath(kw.get('model') or DEFAULT_MODEL)}"


def _wants_measure(team: TeamSpec) -> bool:
    return team.kind in ENGINE_KINDS and team.player_kw.get("ms_per_node") in (None, "measure")


def apply_ms_per_node(pool: Pool, cache_path: str | None = None, *, nodes: int = 400, repeats: int = 3,
                      measure=measure_searched_ms_per_node, override: dict | None = None) -> tuple[Pool, dict]:
    """Every engine team without an ``ms_per_node`` gets the measured one for its binary+network. One measurement
    per key; ``cache_path`` (JSON {key: measurement}) is read first and written after, so a resumed row reuses
    the values it started with. ``override`` {key: ms} skips the measurement for those keys. Returns the new pool
    and {key: measurement} for the keys used."""
    cache: dict = {}
    if cache_path and os.path.exists(cache_path):
        with open(cache_path) as f:
            cache = json.load(f)
    for key, ms in (override or {}).items():
        cache[key] = {"ms_per_node": float(ms), "source": "override"}
    used: dict = {}

    def fix(t: TeamSpec) -> TeamSpec:
        if not _wants_measure(t):
            return t
        key = arm_key(t)
        if key not in cache:
            binary, model = key.split("|", 1)
            print(f"measuring ms_per_node for {binary} with {model} ...", flush=True)
            cache[key] = measure(binary, model, nodes=nodes, repeats=repeats)
            print(f"  {cache[key]['ms_per_node']:.4f} ms per searched node", flush=True)
        used[key] = cache[key]
        return replace(t, player_kw={**t.player_kw, "ms_per_node": float(cache[key]["ms_per_node"])})

    new = replace(pool, contender=fix(pool.contender), opponents=[fix(o) for o in pool.opponents])
    if cache_path and used:
        os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(cache, f, indent=2)
    return new, used


def run_schedule(specs: list[dict], out: str, workers: int = 1, n_boot: int = 2000, seed: int = 0, *,
                 skip_done: bool = False, sprt: dict | None = None, extra: dict | None = None,
                 run_one=None) -> dict:
    """Play the specs into ``out`` and write summary.json. ``skip_done`` rescores finished games from their files
    instead of playing them; ``sprt`` ({"elo0", "elo1", "alpha", "beta"}) stops the row once the GSPRT is decided;
    ``extra`` is merged into summary.json (e.g. the measured ms_per_node)."""
    if run_one is None:
        from arena.run import run_one
    from tools.sprt import pentanomial, status

    os.makedirs(out, exist_ok=True)
    for s in specs:
        s["out"] = out
    if skip_done:
        _check_resume_schedule(out, specs)
    _write_jsonl(os.path.join(out, "specs.jsonl"), [{k: v for k, v in s.items() if k != "out"} for s in specs])
    summaries: list[dict] = []
    todo = specs
    if skip_done:
        todo = []
        for s in specs:
            prev = summary_from_game_file(s, os.path.join(out, game_file(s)))
            if prev is None:
                todo.append(s)
            else:
                summaries.append(prev)
        if summaries:
            print(f"resuming {out}: {len(summaries)} of {len(specs)} games already played", flush=True)
    _write_jsonl(os.path.join(out, "results.jsonl"), summaries)
    results_f = open(os.path.join(out, "results.jsonl"), "a")
    stopped: dict = {}

    def decided() -> bool:
        if not sprt:
            return False
        st = status(pentanomial(summaries), sprt["elo0"], sprt["elo1"], sprt["alpha"], sprt["beta"])
        if st["decision"] != "continue" and not stopped:
            stopped.update(st)
            print(f"SPRT decided {st['decision']} after {st['games']} games (LLR {st['llr']:+.2f}, "
                  f"nElo {st['nelo']:+.1f}): stopping the row", flush=True)
        return bool(stopped)

    def done(spec: dict, summary: dict) -> None:
        summary.update({k: spec[k] for k in SUMMARY_KEEP if k in spec})
        summaries.append(summary)
        results_f.write(json.dumps(summary, separators=(",", ":")) + "\n")
        results_f.flush()
        tag = summary.get("error") or f"{summary['result']} by {summary['reason']}"
        print(f"[{len(summaries)}/{len(specs)}] {spec['game_id']} vs {spec['opponent']:<22} "
              f"contender={spec['contender_team']} -> {tag}", flush=True)

    t0 = time.perf_counter()
    try:
        if not decided():
            if workers <= 1:
                for s in todo:
                    done(s, run_one(s))
                    if decided():
                        break
            else:
                with ProcessPoolExecutor(max_workers=workers) as ex:
                    futs = {ex.submit(run_one, s): s for s in todo}
                    for fut in as_completed(futs):
                        try:
                            res = fut.result()
                        except CancelledError:
                            continue
                        done(futs[fut], res)
                        if decided():
                            for f in futs:
                                f.cancel()
    finally:
        results_f.close()
    summaries.sort(key=lambda s: s["game_id"])
    res = results_by_opponent(summaries, n_boot=n_boot, seed=seed)
    played = {s["game_id"] for s in summaries}
    report = {"wall_s": round(time.perf_counter() - t0, 1), "mix": schedule_mix(specs), "results": res,
              "errors": sum("error" in s for s in summaries), "games": summaries,
              "unplayed": sum(s["game_id"] not in played for s in specs)}
    if sprt:
        report["sprt"] = stopped or status(pentanomial(summaries), sprt["elo0"], sprt["elo1"], sprt["alpha"],
                                           sprt["beta"])
    report.update(extra or {})
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(report, f, indent=2)
    return report


# ---------------------------------------------------------------------- CLI

def _jitter_arg(text: str | None) -> tuple[float, float] | None:
    if not text:
        return None
    lo, hi = (float(v) for v in text.split(","))
    if not 0 < lo <= hi:
        raise ValueError(f"--jitter must be LO,HI with 0 < LO <= HI, got {text!r}")
    return lo, hi


def _sprt_arg(text: str | None) -> dict | None:
    if not text:
        return None
    vals = [float(v) for v in text.split(",")]
    if len(vals) not in (2, 4) or vals[0] >= vals[1]:
        raise ValueError(f"--sprt must be ELO0,ELO1 or ELO0,ELO1,ALPHA,BETA with ELO0 < ELO1, got {text!r}")
    alpha, beta = (vals[2], vals[3]) if len(vals) == 4 else (0.05, 0.05)
    return {"elo0": vals[0], "elo1": vals[1], "alpha": alpha, "beta": beta}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m arena.pool", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("show", "schedule", "run", "add-snapshot", "measure"):
        p = sub.add_parser(name)
        p.add_argument("--pool", default="arena/pools/phase1.json")
        if name in ("schedule", "run"):
            p.add_argument("--pairs", type=int, default=20, help="opening pairs (2 games each)")
            p.add_argument("--openings", default="train", choices=("train", "eval", "all"))
            p.add_argument("--seed", type=int, default=0)
            p.add_argument("--base-ms", type=int, default=60000)
            p.add_argument("--channel-delay-ms", type=int, default=0)
            p.add_argument("--max-plies", type=int, default=400)
            p.add_argument("--rules", choices=("v2", "v3"), default="v2",
                           help="arena rules profile: v2 (default, as before) or v3 (arena.sim.RULES_V3, spec v3 D4: "
                                "min move time and lag, scaled ply cap with scored aborts, sit-off adjudication)")
            p.add_argument("--agent", action="store_true", help="the contender plays as the scheduler agent")
            p.add_argument("--features", default="", help="comma-separated scheduler feature groups (implies --agent)")
            p.add_argument("--policy", default=None, help="scheduler checkpoint (implies --agent); untrained if left out")
            p.add_argument("--rule", default=None, help="a fixed scheduler rule: mate_wait or mate_ask (implies --agent)")
            p.add_argument("--opponent", default=None, help="play only this opponent")
            p.add_argument("--setup", default=None, choices=("centralized", "silent", "sharing"),
                           help="the contender is stock Hivemind in this team setup")
            p.add_argument("--lend", action="store_true", help="with --setup: a split team lends idle compute")
            p.add_argument("--lend-share", type=float, default=None,
                           help="with --setup and --lend: the part of the idle machine borrowed (default 1, all)")
            p.add_argument("--nodes", type=int, default=None, help="the contender's node budget")
            p.add_argument("--opponent-nodes", type=int, default=None, help="the opponents' node budget")
            p.add_argument("--contender-kw", type=json.loads, default=None,
                           help='JSON player options for the contender, e.g. \'{"clock_k": 110, "serial": true}\'')
            p.add_argument("--opponent-kw", type=json.loads, default=None, help="JSON player options for the opponents")
            p.add_argument("--jitter", default=None,
                           help="LO,HI: think-time factor per search for every centralized Hivemind team on both "
                                "sides (out-of-step games; off by default)")
        if name == "schedule":
            p.add_argument("--out", default=None, help="write the schedule as JSON lines")
        if name == "run":
            p.add_argument("--workers", type=int, default=1)
            p.add_argument("--boot", type=int, default=2000)
            p.add_argument("--out", default="runs/pool")
            p.add_argument("--skip-done", action="store_true",
                           help="resume: rescore games already played into --out instead of playing them again")
            p.add_argument("--sprt", default=None, metavar="ELO0,ELO1[,ALPHA,BETA]",
                           help="GSPRT over normalised Elo after every game pair; stop the row when decided")
            p.add_argument("--measure-ms-per-node", action="store_true",
                           help="measure ms per searched node per binary+network for engine teams without one")
            p.add_argument("--measure-nodes", type=int, default=400)
            p.add_argument("--ms-per-node-cache", default=None,
                           help="JSON file of measured ms_per_node per binary+network (default <out>/ms_per_node.json);"
                                " point several rows at one file so they charge the same values")
            p.add_argument("--equal-ms-per-node", action="store_true",
                           help="diagnostic: every engine team is charged the (first) opponent's ms_per_node, so "
                                "both sides get the same nodes per second of clock (the equal-nodes row)")
        if name == "measure":
            p.add_argument("--name", required=True, help="the contender's or an opponent's name")
            p.add_argument("--nodes", type=int, default=400)
            p.add_argument("--repeats", type=int, default=3)
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

    if args.cmd == "measure":
        team = next((t for t in [pool.contender, *pool.opponents] if t.name == args.name), None)
        if team is None:
            ap.error(f"no team named {args.name!r} in {args.pool}")
        m = measure_searched_ms_per_node(team.player_kw.get("binary"), team.player_kw.get("model"),
                                         nodes=args.nodes, repeats=args.repeats)
        new = replace(team, player_kw={**team.player_kw, "ms_per_node": round(m["ms_per_node"], 5)})
        if team is pool.contender:
            pool.contender = new
        else:
            pool.opponents = [new if o is team else o for o in pool.opponents]
        pool.save(args.pool)
        print(json.dumps(m, indent=2))
        print(f"stored ms_per_node {new.player_kw['ms_per_node']} for {args.name} in {args.pool}")
        return 0

    features = [f for f in args.features.split(",") if f]
    if features:
        from arena.players.scheduler_player import FEATURE_GROUPS

        if set(features) - set(FEATURE_GROUPS):
            ap.error(f"--features must come from {', '.join(FEATURE_GROUPS)}")
    if args.rule and "mate" not in features:
        ap.error("--rule needs --features to include mate")
    try:
        pool = for_evaluation(pool, agent=args.agent, features=features, policy=args.policy, opponent=args.opponent,
                              setup=args.setup, lend=args.lend, nodes=args.nodes, opponent_nodes=args.opponent_nodes,
                              rule=args.rule, lend_share=args.lend_share, contender_kw=args.contender_kw,
                              opponent_kw=args.opponent_kw, jitter=_jitter_arg(args.jitter))
    except ValueError as exc:
        ap.error(str(exc))
    specs = make_schedule(pool, args.pairs, seed=args.seed, openings=args.openings, base_ms=args.base_ms,
                          channel_delay_ms=args.channel_delay_ms, max_plies=args.max_plies,
                          rules=args.rules)
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
    try:
        sprt = _sprt_arg(args.sprt)
    except ValueError as exc:
        ap.error(str(exc))
    extra: dict = {}
    if args.measure_ms_per_node or any(t.player_kw.get("ms_per_node") == "measure"
                                       for t in [pool.contender, *pool.opponents]):
        cache = args.ms_per_node_cache or os.path.join(args.out, "ms_per_node.json")
        pool, used = apply_ms_per_node(pool, cache, nodes=args.measure_nodes)
        extra["ms_per_node"] = used
    if args.equal_ms_per_node:
        pool = equalize_ms_per_node(pool)
        extra["equal_ms_per_node"] = pool.contender.player_kw.get("ms_per_node")
    if extra:
        specs = make_schedule(pool, args.pairs, seed=args.seed, openings=args.openings, base_ms=args.base_ms,
                              channel_delay_ms=args.channel_delay_ms, max_plies=args.max_plies,
                          rules=args.rules)
    report = run_schedule(specs, args.out, workers=args.workers, n_boot=args.boot, seed=args.seed,
                          skip_done=args.skip_done, sprt=sprt, extra=extra)
    print()
    print(format_results(report["results"]))
    if "sprt" in report:
        from tools.sprt import format_status

        print(format_status(report["sprt"], "SPRT"))
    print(f"\nwrote {args.out}/summary.json ({report['wall_s']:.0f}s, {report['errors']} errors, "
          f"{report['unplayed']} unplayed)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
