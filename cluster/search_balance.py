#!/usr/bin/env python3
"""Search balance controller: deep search targets vs more self-play games (docs/search_balance.md, design B).

Compass search over theta = (f, k, b, lambda, src) with a paired, compute-matched mini-train Z-test, a mini-gate veto,
a forced revert test after every promotion, an overhead ledger capped at 10% of the run's GPU-h, and a simulator.
Pure Python (stdlib only); runs with the repo or the fork python.

    python cluster/search_balance.py plan RUN_ROOT N [--prior P] [--segments 5] [--segment-games 2000] [--generator M]
    python cluster/search_balance.py decide RUN_ROOT N [--dry]
    python cluster/search_balance.py realise RUN_ROOT N [--accounting FILE]
    python cluster/search_balance.py status RUN_ROOT [--json]
    python cluster/search_balance.py simulate --iterations 40 --seeds 200 [--sigma 4] [--markdown]

RUN_ROOT is runs/full/<RUN> (iterations in RUN_ROOT/runs/rl-itN). Files this script reads (all optional except the
plan for decide; missing ones degrade to "no evidence" or nominal numbers):
    balance/prior.json            {"sigma_hat", "tau_hat", "tau_df", "gamma": {"3200": 1.04}, "fits_bench": {"8": true},
                                   "reanalyse_fidelity": false, "eta_batch": 1.0, "calibration": [[dx, delo], ...],
                                   "score_stage": false, "probe_bias_measured": false, "search_q_blend": false,
                                   "policy_positions": false}
                                  The four flags say what exists/was measured (all false = not yet): score_stage =
                                  the minitrain + score stages are built and wired (else no challenger is planned);
                                  probe_bias_measured = the probe's bias toward deep targets was measured and found
                                  small (else f > 0 arms are illegal); search_q_blend = the trainer has
                                  --search-q-blend (else lambda > 0 arms are illegal); policy_positions = the trainer
                                  has train.policy_positions (else steps_rule is "positions").
                                  tau_hat/tau_df: between-data noise of d from the replicate round (mini-trains on
                                  different games); without it no promotion until the run's own repeats give 3 df.
    balance/itN/score.json        {"scores": {"inc_s0", "chal_s0", "inc_s1", "chal_s1"}, "probe_version": V}
                                  (lower is better; d_s = inc_s - chal_s)
    balance/itN/minigate.json     {"games", "wins", "losses", "draws"} (challenger mini's view) or {"games", "score"}
    balance/itN/realised.json     {"segNN": {"gpu_s", "games", "records", "visits" | "deep_visits"+"fast_visits",
                                   "deep", "masked", "sit_rate", "flag_rate"}}   (written by `realise`; gpu_s only
                                   from --accounting / balance/itN/accounting.json {"segNN": gpu_s})
    balance/itN/overhead.json     {"gpu_s": {"minitrain", "score", "probe", "minigate"}}
    runs/rl-itN/search/train/segNN/cap_counts.json   engine sidecar {records, deep, masked, deep_visits, fast_visits}
    runs/rl-itN/stages/match_generator.done          main gate {"elo", "score", "sprt": {"decision"}}
    metrics/itN.json              run_metrics.py record: alerts[].code, cost.gpu_hours(_cum), network.heldout.loss,
                                  outcome.gate_generator.elo
Files it writes (atomic tmp + rename, never rewritten): balance/itN.plan.json (plan), balance/itN.json (decide).
"""

import argparse
import copy
import hashlib
import json
import math
import os
import random
import re
import statistics
import sys

SCHEMA = 1
F_GRID = (0.0, 0.0625, 0.125, 0.25, 0.5)
K_GRID = (2, 4, 8)
B_GRID = (800, 400, 200)                       # index + = smaller budget ("b-")
L_GRID = (0.0, 0.25, 0.5)
SRC_GRID = ("cap", "reanalyse")
DIRS = ("f+", "k+", "f-", "k-", "b-", "b+", "l+", "l-", "src")
THETA0 = {"f": 0.0, "k": None, "b": 800, "lambda": 0.0, "src": "cap"}
DEFAULT_K = 4                                  # k given to f when f leaves 0
NODES = 800
RHO = 0.01                                     # raw-policy + per-game overhead, in units of one stock game
GAMES_PER_GPU_H = 770.0
V_GPU_S_PER_VISIT = 4.24e-5
SEGMENT_GAMES, SEGMENTS = 2000, 5
GAMES = SEGMENT_GAMES * SEGMENTS
FIXED_STAGES_GPU_H = 0.66 + 1.72 + 1.22       # val + train + gate
ITER_GPU_H = GAMES / GAMES_PER_GPU_H + FIXED_STAGES_GPU_H   # ~16.6 nominal
MINI_STEPS, STEP_S = 200, 2.9
COST_ITEMS = {"minitrain_seed": 2 * MINI_STEPS * STEP_S / 3600.0,   # one seed pair = 2 mini-trains
              "score": 0.05, "probe": 0.55, "minigate": 0.3}
OVERHEAD_CAP = 0.10
# Z_PROMOTE 3.0 and no single-observation promotion: re-tuned in the simulator with a shared per-iteration noise
# term tau (left theta0 on a_flat/b_theta0 <= ~5% for tau up to 2 sigma); 2.0/3.0 gave 12-38% at tau = sigma.
Z_PROMOTE, Z_SINGLE, Z_REJECT, Z_REVERT = 3.0, float("inf"), -1.0, 1.5
MAX_OBS = 3
TAU_DF_MIN = 3                                 # df of the between-data noise estimate needed before any promotion
TABU_ITERS, FREEZE_ITERS = 3, 2
DECAY_AGE, DECAY_WEIGHT = 3, 0.5
ELO_WORTH = 5.0                                # delta_min = probe gain worth ~+5 Elo per iteration
MINIGATE_P = 0.10
REVIEW_SHIFT = 0.20
COST_OVERRUN = 1.3
PROBE_EVERY = 2
PROBE_REGRESSION_CODES = ("heldout_value_loss_up", "probe_regression")
SPRT_REGRESSED = ("H0",)


class InputError(Exception):
    """A required input is missing or corrupt."""


# ------------------------------------------------------------------------------------------------ numbers + json


def sig(x, digits=6):
    if isinstance(x, float):
        if not math.isfinite(x):
            return None
        return float(f"{x:.{digits}g}")
    if isinstance(x, dict):
        return {str(k): sig(v, digits) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [sig(v, digits) for v in x]
    return x


def canon(value):
    """Round floats and round-trip through JSON, so in-memory state equals what a file reload gives."""
    return json.loads(json.dumps(sig(value), sort_keys=True))


def dumps(value):
    return json.dumps(sig(value), sort_keys=True, indent=1) + "\n"


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_file(path):
    try:
        with open(path, "rb") as f:
            return sha256_bytes(f.read())
    except OSError:
        return None


def digest(value):
    return sha256_bytes(json.dumps(sig(value), sort_keys=True).encode())


def load_json(path, required=False):
    """None when missing; InputError when corrupt (or missing and required)."""
    if not os.path.isfile(path):
        if required:
            raise InputError(f"missing {path}")
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError) as e:
        raise InputError(f"corrupt {path}: {e}")


def load_soft(path):
    try:
        return load_json(path)
    except InputError:
        return None


def write_atomic(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w") as f:
        f.write(text)
    os.replace(tmp, path)


def det_hash(*parts):
    return int(hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()[:12], 16)


def get(d, *keys):
    for k in keys:
        if not isinstance(d, dict):
            return None
        d = d.get(k)
    return d


def num(x):
    return float(x) if isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) else None


def phi(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


# ------------------------------------------------------------------------------------------------ theta


def theta(f=0.0, k=None, b=800, lam=0.0, src="cap"):
    f = float(f)
    return {"f": f, "k": int(k) if f > 0 and k is not None else (DEFAULT_K if f > 0 else None), "b": int(b),
            "lambda": float(lam), "src": src}


def norm(t):
    return theta(t.get("f", 0.0), t.get("k"), t.get("b", 800), t.get("lambda", 0.0), t.get("src", "cap"))


def key(t):
    t = norm(t)
    return f"f={t['f']:g},k={t['k'] if t['k'] else '-'},b={t['b']},l={t['lambda']:g},src={t['src']}"


def pair_key(inc, chal):
    return f"{key(inc)}>{key(chal)}"


def same_selfplay(a, b):
    a, b = norm(a), norm(b)
    return all(a[x] == b[x] for x in ("f", "k", "b", "src"))


def on_grid(t):
    t = norm(t)
    if t["f"] not in F_GRID or t["b"] not in B_GRID or t["lambda"] not in L_GRID or t["src"] not in SRC_GRID:
        return False
    if t["f"] > 0 and t["k"] not in K_GRID:
        return False
    if t["f"] == 0 and (t["b"] != NODES or t["src"] != "cap"):
        return False            # b < 800 with no deep decisions would mask every policy target
    return True


def fits_bench(t, prior):
    t = norm(t)
    if t["k"] == 8:
        return bool(get(prior or {}, "fits_bench", "8"))
    return True


def src_available(t, prior):
    return norm(t)["src"] == "cap" or bool((prior or {}).get("reanalyse_fidelity"))


def deep_available(t, prior):
    """f > 0 arms (deep targets) only after the probe's bias toward deep targets was measured (finding: the probe
    deep targets come from the same net's deeper search on the same positions, so the probe favours deep arms
    whether or not they buy Elo; the simulator's f_proxy surface leaves theta0 in over half the runs)."""
    return norm(t)["f"] == 0 or bool((prior or {}).get("probe_bias_measured"))


def lambda_available(t, prior):
    """lambda > 0 arms only once the trainer has --search-q-blend."""
    return norm(t)["lambda"] == 0 or bool((prior or {}).get("search_q_blend"))


def legal(t, prior):
    return on_grid(t) and fits_bench(t, prior) and src_available(t, prior) and deep_available(t, prior) \
        and lambda_available(t, prior)


def exploration_blockers(prior):
    """Why no challenger can be planned at all (empty = exploration possible)."""
    out = []
    if not (prior or {}).get("score_stage"):
        out.append("no minitrain/score stage (prior.score_stage false): a challenger could never be judged")
    return out


def move(t, d, step):
    t = dict(norm(t))
    idx = lambda grid, v: grid.index(v) if v in grid else None  # noqa: E731
    try:
        if d in ("f+", "f-"):
            i = idx(F_GRID, t["f"]) + (step if d == "f+" else -step)
            if not 0 <= i < len(F_GRID):
                return None
            t["f"] = F_GRID[i]
            t["k"] = (t["k"] or DEFAULT_K) if t["f"] > 0 else None
        elif d in ("k+", "k-"):
            if t["f"] == 0:
                return None
            i = idx(K_GRID, t["k"]) + (step if d == "k+" else -step)
            if not 0 <= i < len(K_GRID):
                return None
            t["k"] = K_GRID[i]
        elif d in ("b-", "b+"):
            i = idx(B_GRID, t["b"]) + (step if d == "b-" else -step)
            if not 0 <= i < len(B_GRID):
                return None
            t["b"] = B_GRID[i]
        elif d in ("l+", "l-"):
            i = idx(L_GRID, t["lambda"]) + (step if d == "l+" else -step)
            if not 0 <= i < len(L_GRID):
                return None
            t["lambda"] = L_GRID[i]
        elif d == "src":
            t["src"] = "reanalyse" if t["src"] == "cap" else "cap"
    except TypeError:
        return None
    t = norm(t)
    return t if on_grid(t) else None


def all_arms(prior=None, legal_only=True):
    out = [theta(0.0, None, NODES, lam) for lam in L_GRID]
    for f in F_GRID[1:]:
        for k in K_GRID:
            for b in B_GRID:
                for lam in L_GRID:
                    for src in SRC_GRID:
                        out.append(theta(f, k, b, lam, src))
    return [t for t in out if not legal_only or legal(t, prior)]


def engine_args(t):
    t = norm(t)
    args = []
    if t["f"] > 0 and t["src"] == "cap":
        args += ["--deep-fraction", f"{t['f']:g}", "--deep-multiplier", str(t["k"])]
    if t["b"] < NODES:
        args += ["--fast-nodes", str(t["b"])]
    if t["src"] == "reanalyse":
        args += ["--record-state-output", "{segment_dir}/states.jsonl"]
    return args


def trainer_args(t):
    t = norm(t)
    return ["--search-q-blend", f"{t['lambda']:g}"] if t["lambda"] > 0 else []


# ------------------------------------------------------------------------------------------------ cost


def gamma_of(n, table):
    if n is None or int(n) == NODES:
        return 1.0
    samples = (table or {}).get(str(int(n))) or []
    return float(statistics.median(samples)) if samples else 1.0


def cost(t, table=None, eta_batch=1.0):
    """Self-play GPU per game relative to a stock 800-node game (theta0 -> exactly 1.0)."""
    t = norm(t)
    f, k, b = t["f"], t["k"] or 1, t["b"]
    if t["src"] == "reanalyse":   # games play at b; deep targets are bought offline at batch efficiency eta
        raw = (b / NODES) * gamma_of(b, table) + f * k * gamma_of(k * NODES, table) / max(eta_batch, 1e-6) + RHO
    else:
        raw = (1 - f) * (b / NODES) * gamma_of(b, table) + f * k * gamma_of(k * NODES, table) + RHO
    return raw / (1.0 + RHO)


def games_for(t, table, segment_games, eta_batch=1.0):
    return int(round(segment_games / cost(t, table, eta_batch)))


def gamma_sample(plan, realised, table):
    """(nodes, gamma_hat, realised_c) from the challenger segment vs its twin, or None."""
    chal = plan.get("challenger")
    x, t = plan.get("challenger_segment"), plan.get("twin_segment")
    if not chal or not x or not realised:
        return None
    rx, rt = realised.get(x) or {}, realised.get(t) or {}
    gx, gt = num(rx.get("gpu_s")), num(rt.get("gpu_s"))
    vt = num(rt.get("visits")) or ((num(rt.get("deep_visits")) or 0) + (num(rt.get("fast_visits")) or 0)) or None
    if not gx or not gt or not vt:
        return None
    v_inc = gt / vt
    chal = norm(chal)
    deep_v, fast_v = num(rx.get("deep_visits")), num(rx.get("fast_visits"))
    n_deep = chal["k"] * NODES if chal["f"] > 0 else chal["b"]
    if deep_v and fast_v is not None and chal["f"] > 0:
        g = (gx / v_inc - fast_v * gamma_of(chal["b"], table)) / deep_v
    else:
        vx = num(rx.get("visits")) or ((deep_v or 0) + (fast_v or 0)) or None
        if not vx:
            return None
        g = (gx / vx) / v_inc
    g = min(max(g, 0.3), 5.0)
    c_real = None
    if num(rx.get("games")) and num(rt.get("games")):
        c_real = (gx / rx["games"]) / (gt / rt["games"])
    return n_deep, g, c_real


# ------------------------------------------------------------------------------------------------ state


def init_state(prior=None, run_gpu_h=0.0):
    prior = prior or {}
    gamma = {str(int(n)): [float(v)] for n, v in (prior.get("gamma") or {}).items() if num(v)}
    return canon({"theta_star": THETA0, "step": 2, "dir_idx": 0, "tabu": {}, "rejected": [], "evidence": {},
                  "gamma": gamma, "overhead": {"cum_gpu_h": 0.0, "run_gpu_h": float(run_gpu_h), "frac": 0.0},
                  "mode": "explore", "last_promotion": None, "revert_test": False, "frozen_until": None,
                  "post_gates": [], "probe": {"version": 0, "iter": None}, "sigma_samples": [], "history": []})


def tabu(S, t, n):
    until = S["tabu"].get(key(t))
    return until is not None and n <= until


def propose(S, n, prior, depth=0):
    star = S["theta_star"]
    for i in range(len(DIRS)):
        j = (S["dir_idx"] + i) % len(DIRS)
        cand = move(star, DIRS[j], S["step"])
        if cand and legal(cand, prior) and not tabu(S, cand, n) and pair_key(star, cand) not in S["rejected"]:
            S["dir_idx"] = j
            return cand
    if S["step"] > 1 and depth < 4:
        S["step"] //= 2
        S["dir_idx"] = 0
        S["rejected"] = []
        return propose(S, n, prior, depth + 1)
    S["mode"] = "maintenance"
    return None


def maintenance_pick(S, n, run_seed, prior):
    star = S["theta_star"]
    cands = []
    for d in DIRS:
        c = move(star, d, 1)
        if c and legal(c, prior) and not tabu(S, c, n) and key(c) not in [key(x) for x in cands]:
            cands.append(c)
    if not cands:
        return None
    return cands[det_hash(run_seed, n, S["dir_idx"]) % len(cands)]


def overhead_items(refresh, seeds, minigate):
    items = {"minitrain": seeds * COST_ITEMS["minitrain_seed"], "score": COST_ITEMS["score"]}
    if refresh:
        items["probe"] = COST_ITEMS["probe"]
    if minigate:
        items["minigate"] = COST_ITEMS["minigate"]
    return items


def projection(S, items, iter_gpu_h):
    extra = sum(items.values())
    oh = S["overhead"]
    return (oh["cum_gpu_h"] + extra) / max(oh["run_gpu_h"] + iter_gpu_h + extra, 1e-9)


def plan_core(n, state, ctx, failsafe_reason=None):
    """Pure: the plan for iteration n from the state after iteration n-1."""
    S = copy.deepcopy(state)
    prior = ctx.get("prior") or {}
    seed = ctx.get("run_seed", 0)
    seg_games, nseg = ctx.get("segment_games", SEGMENT_GAMES), ctx.get("segments", SEGMENTS)
    iter_gpu_h = ctx.get("iter_gpu_h", ITER_GPU_H)
    eta = float(prior.get("eta_batch", 1.0) or 1.0)
    mode, reason, chal, revert_test = S.get("mode", "explore"), "", None, False
    blockers = exploration_blockers(prior)
    if failsafe_reason:
        mode, reason = "failsafe", failsafe_reason
    elif blockers:
        mode, reason = "frozen", "; ".join(blockers)
    elif S.get("frozen_until") is not None and n <= S["frozen_until"]:
        mode, reason = "frozen", f"promotions frozen until it{S['frozen_until']} after a revert"
    elif S.get("revert_test") and S.get("last_promotion"):
        chal, revert_test, reason = norm(S["last_promotion"]["from"]), True, "forced revert test after a promotion"
        mode = "explore" if mode not in ("maintenance",) else mode
    elif mode == "maintenance":
        if n % 3 == 0:
            chal = maintenance_pick(S, n, seed, prior)
            reason = "maintenance re-test of a +-1 neighbour"
        else:
            reason = "maintenance: challenger every 3rd iteration"
    else:
        mode = "explore"
        chal = propose(S, n, prior)
        if chal is None:
            mode = S["mode"]
            if n % 3 == 0:
                chal = maintenance_pick(S, n, seed, prior)
            reason = "converged: every neighbour at step 1 rejected"
        else:
            reason = f"compass step {S['step']} direction {DIRS[S['dir_idx']]}"
    if chal is not None and key(chal) == key(S["theta_star"]):
        chal = None
    # probe + overhead budget, dropping in the spec's order
    probe = S.get("probe") or {"version": 0, "iter": None}
    refresh = bool(chal) and (probe["version"] == 0 or probe["iter"] is None or n - probe["iter"] >= PROBE_EVERY)
    seeds, minigate, dropped = 2, True, []
    if chal is not None:
        for item in ("minigate", "probe", "second_seed"):
            if projection(S, overhead_items(refresh, seeds, minigate), iter_gpu_h) <= OVERHEAD_CAP:
                break
            if item == "minigate":
                minigate = False
            elif item == "probe" and refresh:
                if probe["version"] == 0:
                    continue         # nothing stale to reuse: keep the refresh
                refresh = False
            elif item == "second_seed":
                seeds = 1
            dropped.append(item)
        if projection(S, overhead_items(refresh, seeds, minigate), iter_gpu_h) > OVERHEAD_CAP:
            chal, mode, refresh = None, "frozen", False
            dropped.append("exploration")
            reason = "overhead cap: exploration dropped this iteration"
    star = norm(S["theta_star"])
    table = S.get("gamma") or {}
    names = [f"seg{k:02d}" for k in range(nseg)]
    kind = None if chal is None else ("lambda" if same_selfplay(star, chal) else "selfplay")
    segs = {}
    for k, name in enumerate(names):
        segs[name] = {"theta": "inc", "seed": 1000 * n + k, "games": games_for(star, table, seg_games, eta),
                      "engine_args": engine_args(star)}
    x, t = (names[-1], names[0]) if kind == "selfplay" else (None, names[0] if kind else None)
    if x:
        segs[x] = {"theta": "chal", "seed": 1000 * n + 0, "games": games_for(chal, table, seg_games, eta),
                   "twin": t, "engine_args": engine_args(chal)}
    mini = None
    if chal is not None:
        mini = {"steps": ctx.get("mini_steps", MINI_STEPS), "seeds": [1000 * n + 10 * s + 7 for s in range(seeds)],
                "init": "generator checkpoint at iteration start", "replay_fraction": 0.5,
                # seed s trains on the games with index % 2 == s of each arm's segment (X and T share seeds, so the
                # halves stay paired): the seed pair's spread then includes which games were played, not only
                # the trainer seed
                "data_half": {"by": "game_index % 2", "seed_half": list(range(seeds)) if seeds > 1 else [None]},
                "common_args": ["--search-value-weight", "0", "--search-wdl-weight", "1",
                                "--search-moves-left-weight", "0.1", "--export", "last"],
                "arms": {"inc": {"data": [t], "trainer_args": trainer_args(star)},
                         "chal": {"data": [x or t], "trainer_args": trainer_args(chal)}}}
    version = probe["version"] + (1 if refresh else 0)
    plan = {"schema": SCHEMA, "run_seed": seed, "iteration": n, "mode": mode, "reason": reason,
            "theta_star": star, "challenger": chal, "challenger_kind": kind, "revert_test": revert_test,
            "challenger_segment": x, "twin_segment": t, "segments": segs,
            "val": {"engine_args": [], "assert_stock": True}, "trainer_args": trainer_args(star),
            "steps_rule": "policy_positions" if prior.get("policy_positions") else "positions",
            "minitrain": mini, "minigate_allowed": bool(chal) and minigate,
            "dropped": dropped,
            "probe": {"version": version, "refresh": refresh, "path": f"balance/probe/deep_v{version}.dst"
                      if version else None},
            "cost": {"v_gpu_s_per_visit": V_GPU_S_PER_VISIT, "gamma": {k: gamma_of(k, table) for k in table},
                     "c_inc": cost(star, table, eta), "c_chal": cost(chal, table, eta) if chal else None},
            "overhead_planned": overhead_items(refresh, seeds, minigate) if chal else {},
            "state_in": S}
    return canon(plan)


# ------------------------------------------------------------------------------------------------ decide


def sigma_hat(S, prior):
    vals = [v for v in S.get("sigma_samples", []) if num(v) is not None]
    p = num((prior or {}).get("sigma_hat"))
    if p:
        vals = vals + [p]
    if not vals:
        return None, 0
    return math.sqrt(sum(v * v for v in vals) / len(vals)), len(vals)


def tau_hat(S, prior, sh=None):
    """(tau, df): the between-data noise of one observation's d, i.e. what changes with which games were played
    (and with the X/T pairing breaking at the first time-budgeted root scan). Pools the prior's replicate-round
    estimate (tau_hat with tau_df degrees of freedom) with the spread of each pair's d across iterations beyond its
    seed noise. (None, 0) when neither exists."""
    ss, df, seed_var = 0.0, 0, 0.0
    pools = [E for E in (S.get("evidence") or {}).values() if len(E) >= 2] + list(S.get("tau_pool") or [])
    for E in pools:
        ds = [o["d"] for o in E]
        m = sum(ds) / len(ds)
        ss += sum((x - m) ** 2 for x in ds)
        df += len(ds) - 1
        for o in E:
            s1 = max(num(o.get("sigma_n")) or 0.0, sh or 0.0)
            seed_var += s1 * s1 / max(o.get("seeds", 1), 1) * (len(ds) - 1) / len(ds)
    own = max(ss - seed_var, 0.0) / df if df else None
    p, pdf = num((prior or {}).get("tau_hat")), int(num((prior or {}).get("tau_df")) or 0)
    if p is not None and pdf <= 0:
        pdf = 1
    if p is None and own is None:
        return None, 0
    var = ((p * p * pdf if p is not None else 0.0) + (own * df if own is not None else 0.0)) / \
        ((pdf if p is not None else 0) + df)
    return math.sqrt(var), (pdf if p is not None else 0) + df


def delta_min(sh, calibration):
    """Probe gain worth ~ELO_WORTH Elo per iteration: regression through the origin of main-gate Elo gains on main-net
    probe gains once there are 3 points; clamped to [0.5, 3] sigma_hat because mini-net d and main-net probe gains are
    not the same scale (see the simulation section); 1.5 sigma_hat before that."""
    pts = [(x, y) for x, y in (calibration or []) if num(x) is not None and num(y) is not None]
    if sh is None:
        return None, "none"
    if len(pts) >= 3:
        sxx = sum(x * x for x, _ in pts)
        beta = sum(x * y for x, y in pts) / sxx if sxx > 0 else 0.0
        if beta > 0:
            return min(max(ELO_WORTH / beta, 0.5 * sh), 3.0 * sh), f"calibrated beta={beta:.4g} on {len(pts)} points"
    return 1.5 * sh, "1.5 sigma_hat"


def evaluate(S, pk, n, prior, calibration):
    E = S["evidence"].get(pk) or []
    sh, nsig = sigma_hat(S, prior)
    tau, tau_df = tau_hat(S, prior, sh)
    dm, dm_src = delta_min(sh, calibration)
    if not E or sh is None:
        return {"n": len(E), "mean": None, "se": None, "Z": None, "delta_min": dm, "delta_min_source": dm_src,
                "sigma_hat": sh, "tau_hat": tau, "tau_df": tau_df, "eligible": False}
    ws, means, ses = [], [], []
    for o in E:
        w = DECAY_WEIGHT if n - o["iter"] > DECAY_AGE else 1.0
        s1 = max(num(o.get("sigma_n")) or 0.0, sh)
        ws.append(w)
        means.append(o["d"])
        ses.append(math.sqrt(s1 * s1 / max(o.get("seeds", 1), 1) + (tau or 0.0) ** 2))
    sw = sum(ws)
    mean = sum(w * m for w, m in zip(ws, means)) / sw
    se = math.sqrt(sum((w * s) ** 2 for w, s in zip(ws, ses))) / sw
    z = mean / se if se > 0 else 0.0
    # no promotion on seed-pair noise alone: the between-data noise must have been measured (replicate round on
    # different games, or the run's own repeated observations)
    noise_known = tau is not None and tau_df >= TAU_DF_MIN
    single_ok = num((prior or {}).get("sigma_hat")) is not None or nsig >= 3
    eligible = noise_known and ((z >= Z_PROMOTE and mean >= dm and len(E) >= 2) or
                                (len(E) == 1 and single_ok and z >= Z_SINGLE and mean >= dm))
    return {"n": len(E), "mean": mean, "se": se, "Z": z, "delta_min": dm, "delta_min_source": dm_src,
            "sigma_hat": sh, "tau_hat": tau, "tau_df": tau_df, "noise_known": noise_known,
            "eligible": bool(eligible)}


def score_diffs(scores):
    ds = []
    for s in range(4):
        i, c = num((scores or {}).get(f"inc_s{s}")), num((scores or {}).get(f"chal_s{s}"))
        if i is not None and c is not None:
            ds.append(i - c)
    if not ds:
        return None
    d = sum(ds) / len(ds)
    sigma_n = abs(ds[0] - ds[1]) / math.sqrt(2) if len(ds) >= 2 else None
    return d, sigma_n, len(ds)


def minigate_p_worse(mg):
    """One-sided p-value of a score this low if the challenger mini were equal; small = challenger worse."""
    if not mg:
        return None, None
    g = num(mg.get("games"))
    w, l, dr = num(mg.get("wins")), num(mg.get("losses")), num(mg.get("draws"))
    if w is not None and l is not None:
        dr = dr or 0.0
        g = w + l + dr
        if g <= 0:
            return None, None
        score = (w + 0.5 * dr) / g
        var = max((w + 0.25 * dr) / g - score ** 2, 0.25 * 0.5)
    else:
        score = num(mg.get("score"))
        if score is None or not g:
            return None, None
        var = 0.25
    return score, phi((score - 0.5) / math.sqrt(var / g))


def decide_core(plan, obs, ctx):
    """Pure: the result + next state for iteration plan['iteration'] given its observations.
    obs: scores, minigate, realised, overhead_gpu_h ({item: gpu_h} or None), iter_gpu_h, gate {elo, score, sprt},
         alerts [codes], calibration [[dx, delo]]."""
    n = plan["iteration"]
    S = copy.deepcopy(plan["state_in"])
    prior = ctx.get("prior") or {}
    chal = plan.get("challenger")
    star = norm(plan["theta_star"])
    results, decision = {}, {"action": "none", "reason": plan.get("reason") or "no challenger"}
    # overhead ledger
    oh = obs.get("overhead_gpu_h")
    if oh is None:
        oh = {}
        if chal and obs.get("scores"):
            oh = {k: v for k, v in plan.get("overhead_planned", {}).items() if k != "minigate"}
        if chal and obs.get("minigate"):
            oh["minigate"] = COST_ITEMS["minigate"]
    this_oh = sum(v for v in oh.values() if num(v))
    iter_gpu_h = num(obs.get("iter_gpu_h")) or ctx.get("iter_gpu_h", ITER_GPU_H)
    S["overhead"]["cum_gpu_h"] += this_oh
    S["overhead"]["run_gpu_h"] += iter_gpu_h + this_oh
    S["overhead"]["frac"] = S["overhead"]["cum_gpu_h"] / max(S["overhead"]["run_gpu_h"], 1e-9)
    S["overhead"]["this_iter_gpu_h"] = this_oh
    # probe version (kept when the probe was built even if the mini-trains or scores failed)
    if plan.get("probe", {}).get("refresh") and (obs.get("scores") or obs.get("probe_built")):
        S["probe"] = {"version": plan["probe"]["version"], "iter": n}
    # realised cost -> gamma
    realised = obs.get("realised") or {}
    gs = gamma_sample(plan, realised, S.get("gamma"))
    if gs:
        nodes, g, c_real = gs
        S["gamma"].setdefault(str(int(nodes)), []).append(g)
        S["gamma"][str(int(nodes))] = S["gamma"][str(int(nodes))][-9:]
        results["gamma_sample"] = {"nodes": nodes, "gamma": g}
        if c_real and plan["cost"].get("c_chal"):
            results["c_realised"] = c_real
            results["cost_overrun"] = c_real > COST_OVERRUN * plan["cost"]["c_chal"]
    if realised:
        results["realised"] = realised
        x, t = plan.get("challenger_segment"), plan.get("twin_segment")
        if x and t and realised.get(x) and realised.get(t):
            review = False
            for m in ("sit_rate", "flag_rate"):
                a, b = num(realised[x].get(m)), num(realised[t].get(m))
                if a is not None and b is not None and abs(a - b) > REVIEW_SHIFT * max(b, 0.01):
                    review = True
            results["review"] = review
    # main mix
    x = plan.get("challenger_segment")
    if x:
        rec = {s: num((realised.get(s) or {}).get("records")) for s in plan["segments"]}
        if all(v for v in rec.values()):
            mix = rec[x] / sum(rec.values())
        else:
            mix = 1.0 / len(plan["segments"])
        main_mix = {"inc": 1 - mix, "chal": mix}
    else:
        main_mix = {"inc": 1.0, "chal": 0.0}
    # evidence
    sd = score_diffs(obs.get("scores")) if chal else None
    ev = None
    if chal and sd is None:
        decision = {"action": "none", "reason": "no mini-train scores: no evidence, no move"}
    elif chal:
        d, sigma_n, seeds = sd
        pk = pair_key(star, chal)
        S["evidence"].setdefault(pk, []).append({"iter": n, "d": d, "sigma_n": sigma_n, "seeds": seeds,
                                                 "probe_v": plan.get("probe", {}).get("version")})
        if sigma_n is not None:
            S["sigma_samples"].append(sigma_n)
        results.update({"d": d, "sigma_n": sigma_n, "scores": obs.get("scores")})
        ev = evaluate(S, pk, n, prior, obs.get("calibration"))
        gate = obs.get("gate") or {}
        if plan.get("revert_test"):
            o = S["evidence"][pk][-1]
            sh = ev["sigma_hat"] or abs(d) or 1.0
            z_old = d / (max(num(o.get("sigma_n")) or 0.0, sh) / math.sqrt(o["seeds"]))
            if z_old >= Z_REVERT:
                decision = revert(S, n, f"revert test: old recipe better, Z={z_old:.3g}")
            else:
                S["revert_test"] = False
                S["tabu"][key(chal)] = n + TABU_ITERS
                decision = {"action": "keep", "reason": f"revert test passed (Z_old={z_old:.3g})"}
        else:
            eligible = ev["eligible"]
            score, p_worse = minigate_p_worse(obs.get("minigate"))
            if score is not None:
                results["minigate"] = {"games": num((obs.get("minigate") or {}).get("games")), "score": score,
                                       "p_worse": p_worse}
            veto = p_worse is not None and p_worse < MINIGATE_P
            sprt = gate.get("sprt")
            regressed = sprt in SPRT_REGRESSED and (num(gate.get("score")) or 0.5) < 0.5
            zs = f"Z={ev['Z']:.3g} n={ev['n']} mean={ev['mean']:.4g} delta_min={ev['delta_min']:.4g}" \
                if ev["Z"] is not None else f"n={ev['n']} (no sigma estimate yet)"
            if eligible and not plan.get("minigate_allowed"):
                decision = {"action": "continue", "reason": f"{zs}; eligible but the mini-gate was dropped"}
            elif eligible and score is None:
                decision = {"action": "continue", "reason": f"{zs}; eligible, waiting for the mini-gate"}
            elif eligible and not veto and not regressed:
                S["last_promotion"] = {"from": star, "to": norm(chal), "iter": n}
                S["theta_star"] = norm(chal)
                S["revert_test"], S["post_gates"], S["rejected"] = True, [], []
                if S["mode"] == "maintenance":
                    S["mode"], S["step"] = "explore", 1
                decision = {"action": "promote", "reason": zs}
            elif (ev["Z"] is not None and ev["Z"] <= Z_REJECT) or ev["n"] >= MAX_OBS:
                S["tabu"][key(chal)] = n + TABU_ITERS
                S["rejected"].append(pk)
                gone = S["evidence"].pop(pk, None)    # a later re-test starts a fresh evidence episode
                if gone and len(gone) >= 2:            # ...but its spread still measures the between-data noise
                    S["tau_pool"] = (S.get("tau_pool") or []) + [[{k: o[k] for k in ("d", "sigma_n", "seeds")}
                                                                  for o in gone]]
                    S["tau_pool"] = S["tau_pool"][-20:]
                S["dir_idx"] = (S["dir_idx"] + 1) % len(DIRS)
                decision = {"action": "reject", "reason": zs + ("; mini-gate veto" if eligible and veto else "")}
            else:
                decision = {"action": "continue", "reason": zs + ("; mini-gate veto" if eligible and veto else "")
                            + ("; SPRT regressed" if regressed else "")}
        decision.update({k: ev[k] for k in ("Z", "mean", "se", "n", "delta_min", "delta_min_source", "sigma_hat",
                                            "tau_hat", "tau_df")})
        if not ev.get("noise_known") and ev["Z"] is not None and decision["action"] == "continue":
            decision["reason"] += "; between-data noise not measured yet (prior.tau_hat or 3 df of repeats)"
    # revert rules on the main net (after the promotion iteration)
    lp = S.get("last_promotion")
    if lp and not lp.get("reverted") and n > lp["iter"] and decision["action"] != "revert":
        elo = num(get(obs, "gate", "elo"))
        if elo is not None and len(S["post_gates"]) < 2:
            S["post_gates"].append(elo)
        why = None
        if len(S["post_gates"]) == 2 and all(e < 0 for e in S["post_gates"]) and n - lp["iter"] <= 2:
            why = f"two negative main gates after the promotion: {S['post_gates']}"
        elif n - lp["iter"] <= 3 and any(c in PROBE_REGRESSION_CODES for c in obs.get("alerts") or []):
            why = "probe regression alert after the promotion"
        if why:
            prev = decision
            decision = revert(S, n, why)
            if prev.get("Z") is not None:
                decision["superseded"] = prev
    S["history"] = (S.get("history") or []) + [{"iter": n, "action": decision["action"],
                                                "challenger": key(chal) if chal else None}]
    S["history"] = S["history"][-50:]
    out = {"schema": SCHEMA, "iteration": n, "plan_sha256": digest(plan), "mode": plan["mode"],
           "theta_star_planned": star, "challenger": chal, "results": results, "decision": decision,
           "state": S, "main_mix": main_mix}
    return canon(out)


def revert(S, n, why):
    lp = S["last_promotion"]
    S["theta_star"] = norm(lp["from"])
    S["tabu"][key(lp["to"])] = n + TABU_ITERS
    S["frozen_until"] = n + FREEZE_ITERS
    S["revert_test"], S["rejected"] = False, []
    S["last_promotion"] = dict(lp, reverted=n)
    return {"action": "revert", "reason": why}


# ------------------------------------------------------------------------------------------------ files


def bpath(run, *parts):
    return os.path.join(run, "balance", *parts)


def it_dir(run, n):
    return os.path.join(run, "runs", f"rl-it{n}")


def metrics_fields(run, n):
    m = load_soft(os.path.join(run, "metrics", f"it{n}.json")) or {}
    alerts = [a.get("code") for a in m.get("alerts") or [] if isinstance(a, dict)]
    return {"gpu_hours": num(get(m, "cost", "gpu_hours")), "gpu_hours_cum": num(get(m, "cost", "gpu_hours_cum")),
            "done": m.get("done") is True,
            "heldout_loss": num(get(m, "network", "heldout", "loss")),
            "gate_elo": num(get(m, "outcome", "gate_generator", "elo")), "alerts": sorted(a for a in alerts if a)}


def calibration_points(run, n, prior):
    pts = [list(p) for p in (prior or {}).get("calibration") or [] if isinstance(p, (list, tuple)) and len(p) == 2]
    prev = None
    for i in range(1, n + 1):
        m = metrics_fields(run, i)
        if prev is not None and m["heldout_loss"] is not None and m["gate_elo"] is not None:
            pts.append([prev - m["heldout_loss"], m["gate_elo"]])
        prev = m["heldout_loss"] if m["heldout_loss"] is not None else prev
    return pts


def load_prior(run, path=None):
    return load_soft(path or bpath(run, "prior.json")) or {}


def previous_state(run, n):
    """(state, failsafe_reason) for planning iteration n."""
    prev = bpath(run, f"it{n - 1}.json")
    earlier = [f for f in os.listdir(bpath(run)) if f.startswith("it") and f.endswith(".json")] \
        if os.path.isdir(bpath(run)) else []
    earlier = [f for f in earlier if f[2:].split(".")[0].isdigit() and int(f[2:].split(".")[0]) < n]
    try:
        p = load_json(prev)
        if p is not None and isinstance(p.get("state"), dict):
            return p["state"], None
        if p is not None:
            raise InputError(f"{prev} has no state")
    except InputError as e:
        return fallback_state(run, n), str(e)
    if not earlier:
        return None, None            # first controlled iteration
    return fallback_state(run, n), f"missing balance/it{n - 1}.json (decide did not run)"


def fallback_state(run, n):
    for i in range(n - 1, 0, -1):
        r = load_soft(bpath(run, f"it{i}.json"))
        if r and isinstance(r.get("state"), dict):
            return r["state"]
        p = load_soft(bpath(run, f"it{i}.plan.json"))
        if p and isinstance(p.get("state_in"), dict):
            return p["state_in"]
    return None


def make_ctx(run, args, prior):
    return {"prior": prior, "run_seed": getattr(args, "run_seed", 0) or 0,
            "segment_games": getattr(args, "segment_games", SEGMENT_GAMES) or SEGMENT_GAMES,
            "segments": getattr(args, "segments", SEGMENTS) or SEGMENTS,
            "mini_steps": getattr(args, "mini_steps", MINI_STEPS) or MINI_STEPS}


def cmd_plan(run, n, args):
    out = bpath(run, f"it{n}.plan.json")
    prior_path = args.prior or bpath(run, "prior.json")
    prev_path = bpath(run, f"it{n - 1}.json")
    inputs = {"balance/prior.json": sha256_file(prior_path), f"balance/it{n - 1}.json": sha256_file(prev_path)}
    if os.path.exists(out):
        old = load_json(out, required=True)
        bad = {k: v for k, v in inputs.items() if old.get("inputs", {}).get(k) != v}
        if bad:
            raise SystemExit(f"error: {out} exists and its inputs changed: {sorted(bad)}")
        print(out)
        return old
    prior = load_prior(run, args.prior)
    ctx = make_ctx(run, args, prior)
    state, fs = previous_state(run, n)
    m = metrics_fields(run, n - 1) if n > 1 else {}
    if state is None:
        base = m.get("gpu_hours_cum") or (n - 1) * ITER_GPU_H
        state = init_state(prior, run_gpu_h=base)
        if fs:
            state["mode"] = "explore"
    elif m.get("gpu_hours_cum"):
        state = copy.deepcopy(state)
        state["overhead"]["run_gpu_h"] = max(state["overhead"]["run_gpu_h"], m["gpu_hours_cum"])
    plan = plan_core(n, state, ctx, failsafe_reason=fs)
    if args.generator:
        plan["model"] = {"generator": args.generator, "sha256": sha256_file(args.generator)}
    plan["inputs"] = dict(inputs, **{f"metrics/it{n - 1}.json#fields": digest(m) if m else None})
    if args.prior:
        plan["prior_path"] = os.path.abspath(args.prior)  # decide reads the same prior
    write_atomic(out, dumps(plan))
    print(out)
    return plan


def gather_obs(run, n, plan, prior_path=None):
    d = bpath(run, f"it{n}")
    score = load_json(os.path.join(d, "score.json"))
    minigate = load_json(os.path.join(d, "minigate.json"))
    realised = copy.deepcopy(load_soft(os.path.join(d, "realised.json")) or {})
    for seg in plan.get("segments", {}):
        cc = load_soft(os.path.join(it_dir(run, n), "search", "train", seg, "cap_counts.json"))
        if cc:
            r = realised.setdefault(seg, {})
            for k in ("records", "deep", "masked", "deep_visits", "fast_visits"):
                if k not in r and num(cc.get(k)) is not None:
                    r[k] = cc[k]
    ohf = load_soft(os.path.join(d, "overhead.json"))
    oh = None
    if ohf and isinstance(ohf.get("gpu_s"), dict):
        oh = {k: v / 3600.0 for k, v in ohf["gpu_s"].items() if num(v) is not None}
    marker = load_soft(os.path.join(it_dir(run, n), "stages", "match_generator.done")) or {}
    m = metrics_fields(run, n)
    gate = {"elo": num(marker.get("elo")) if num(marker.get("elo")) is not None else m["gate_elo"],
            "score": num(marker.get("score")), "sprt": get(marker, "sprt", "decision")}
    prior = load_prior(run, prior_path)
    # The monitor writes metrics/itN.json while the iteration runs (partial GPU-hours, balance stages included): use it
    # only once the record is done, net of this iteration's balance overhead; else decide_core takes the nominal cost.
    iter_gpu_h = None
    if m["done"] and m["gpu_hours"] is not None:
        iter_gpu_h = max(m["gpu_hours"] - sum(v for v in (oh or {}).values() if num(v)), 0.0) or None
    pr = plan.get("probe") or {}
    probe_marker = load_soft(os.path.join(it_dir(run, n), "stages", "balance_probe.done")) or {}
    probe_built = bool(pr.get("refresh") and not probe_marker.get("skipped")
                       and probe_marker.get("version") == pr.get("version") and pr.get("version"))
    obs = {"scores": (score or {}).get("scores") if score else None, "minigate": minigate, "realised": realised,
           "overhead_gpu_h": oh, "iter_gpu_h": iter_gpu_h, "gate": gate, "alerts": m["alerts"],
           "calibration": calibration_points(run, n, prior)}
    if probe_built:
        obs["probe_built"] = True
    owned = {"plan": sha256_file(bpath(run, f"it{n}.plan.json")),
             "score": sha256_file(os.path.join(d, "score.json")),
             "minigate": sha256_file(os.path.join(d, "minigate.json"))}
    return obs, owned, prior


def cmd_decide(run, n, args):
    out = bpath(run, f"it{n}.json")
    plan = load_json(bpath(run, f"it{n}.plan.json"), required=True)
    obs, owned, prior = gather_obs(run, n, plan, getattr(args, "prior", None) or plan.get("prior_path"))
    if args.dry:
        S = copy.deepcopy(plan["state_in"])
        sd = score_diffs(obs["scores"]) if plan.get("challenger") else None
        ev = {"eligible": False}
        if sd:
            pk = pair_key(plan["theta_star"], plan["challenger"])
            S["evidence"].setdefault(pk, []).append({"iter": n, "d": sd[0], "sigma_n": sd[1], "seeds": sd[2]})
            if sd[1] is not None:
                S["sigma_samples"].append(sd[1])
            ev = evaluate(S, pk, n, prior, obs["calibration"])
        ev["run_minigate"] = bool(ev["eligible"] and plan.get("minigate_allowed") and not plan.get("revert_test"))
        print(json.dumps(sig(ev), sort_keys=True))
        return ev
    if os.path.exists(out):
        old = load_json(out, required=True)
        bad = {k: v for k, v in owned.items() if old.get("inputs", {}).get(k) != v}
        if bad:
            raise SystemExit(f"error: {out} exists and its inputs changed: {sorted(bad)}")
        print(out)
        return old
    if plan.get("challenger") and obs["scores"] is None and not getattr(args, "no_score", False):
        raise SystemExit(f"error: balance/it{n}/score.json missing; rerun balance_score (decide writes nothing), "
                         f"or --no-score to record 'no evidence, no move' and the overhead spent")
    result = decide_core(plan, obs, {"prior": prior})
    result["inputs"] = dict(owned, other=digest({k: obs[k] for k in ("realised", "overhead_gpu_h", "iter_gpu_h",
                                                                      "gate", "alerts", "calibration")}))
    write_atomic(out, dumps(result))
    print(out)
    return result


def segment_realised(run, n, seg, gpu_s=None):
    """What segment `seg` of iteration n actually cost and did, from files self-play leaves: the rl-loop marker
    (games, records, flag games), seat_clocks.tsv (sum of the per-record node budgets = visits), cap_counts.json
    (deep/masked records and visits, when the cap was on), the self-play log's hybrid line (sits per search) and
    the job accounting's GPU seconds (only from `gpu_s`; never guessed, so gamma is not sampled without it)."""
    it = it_dir(run, n)
    marker = load_soft(os.path.join(it, "stages", f"{seg}.done"))
    if not marker:
        return None
    out = {"games": marker.get("games"), "records": marker.get("positions")}
    labels = marker.get("labels") or {}
    if num(labels.get("flag_games")) is not None and num(marker.get("games")):
        out["flag_rate"] = labels["flag_games"] / marker["games"]
    clocks = os.path.join(it, "search", "train_run", seg, "seat_clocks.tsv")
    if os.path.isfile(clocks):
        with open(clocks) as f:
            header = f.readline().rstrip("\n").split("\t")
            if "nodes" in header:
                col, visits, rows = header.index("nodes"), 0, 0
                for line in f:
                    cells = line.rstrip("\n").split("\t")
                    if len(cells) > col and cells[col].isdigit():
                        visits += int(cells[col])
                        rows += 1
                out["visits"], out["visit_rows"] = visits, rows
    cc = load_soft(os.path.join(it, "search", "train", seg, "cap_counts.json"))
    for k in ("records", "deep", "masked", "deep_visits", "fast_visits"):
        if cc and num(cc.get(k)) is not None:
            out[k] = cc[k]
    log = os.path.join(it, "search", "logs", f"{seg}.log")
    if os.path.isfile(log):
        with open(log, errors="replace") as f:
            for line in f:
                m = re.match(r"selfplay hybrid: (\d+) sits in \d+ triggered of (\d+) searches", line)
                if m and int(m.group(2)):
                    out["sit_rate"] = int(m.group(1)) / int(m.group(2))
    if num(gpu_s) is not None:
        out["gpu_s"] = float(gpu_s)
    return {k: v for k, v in out.items() if v is not None}


def cmd_realise(run, n, args):
    """balance/itN/realised.json for the plan's segments (decide reads it for gamma, cost_overrun and review)."""
    plan = load_json(bpath(run, f"it{n}.plan.json"), required=True)
    acc_path = args.accounting or bpath(run, f"it{n}", "accounting.json")
    acc = load_soft(acc_path) or {}
    out = {}
    for seg in plan.get("segments", {}):
        r = segment_realised(run, n, seg, acc.get(seg))
        if r:
            out[seg] = r
    path = bpath(run, f"it{n}", "realised.json")
    old = load_soft(path)
    if old is not None and canon(old) != canon(out):
        if not args.force:
            raise SystemExit(f"error: {path} exists with other numbers (pass --force after a segment was redone)")
    write_atomic(path, dumps(out))
    print(path)
    return out


def cmd_status(run, args):
    rows = []
    its = sorted({int(f[2:].split(".")[0]) for f in os.listdir(bpath(run))
                  if f.startswith("it") and f[2:].split(".")[0].isdigit()}) if os.path.isdir(bpath(run)) else []
    for n in its:
        p = load_soft(bpath(run, f"it{n}.plan.json")) or {}
        r = load_soft(bpath(run, f"it{n}.json")) or {}
        dec = r.get("decision") or {}
        rows.append({"iteration": n, "mode": p.get("mode"), "theta_star": key(p["theta_star"]) if p.get(
            "theta_star") else None, "challenger": key(p["challenger"]) if p.get("challenger") else None,
            "action": dec.get("action"), "Z": dec.get("Z"), "mean": dec.get("mean"), "reason": dec.get("reason"),
            "overhead_frac": get(r, "state", "overhead", "frac"), "c_chal": get(p, "cost", "c_chal")})
    if getattr(args, "json", False):
        print(json.dumps(sig(rows), sort_keys=True, indent=1))
        return rows
    if not rows:
        print(f"{run}: no balance files")
        return rows
    print(f"{'it':>3} {'mode':<11} {'theta*':<34} {'challenger':<34} {'action':<9} {'Z':>6} {'oh%':>5}")
    for r in rows:
        z = f"{r['Z']:.2f}" if r["Z"] is not None else "-"
        oh = f"{100 * r['overhead_frac']:.1f}" if r["overhead_frac"] is not None else "-"
        print(f"{r['iteration']:>3} {str(r['mode']):<11} {str(r['theta_star']):<34} {str(r['challenger']):<34} "
              f"{str(r['action']):<9} {z:>6} {oh:>5}")
    return rows


# ------------------------------------------------------------------------------------------------ simulator
#
# A synthetic world with a true Elo-per-iteration surface over the knobs. Every arm gets the same self-play GPU time
# (games are budgeted in GPU-h), so the Elo per GPU-h ranking is the ranking of g. Units: d is in "Elo-equivalent"
# probe units (kappa = 1), so delta_min ~ 5.


G0 = 30.0          # Elo per iteration at theta0
A_VALUE, A_POLICY = 6.0, 9.0   # elasticity of gain to log(value positions) and log(policy positions)
GATE_SD = 35.0
MINI_ELO_SCALE = 0.5            # mini nets differ by this share of the arms' true per-iteration gain gap
JITTER = 0.06
SURFACES = ("a_flat", "b_theta0", "c_interior", "d_edge_k8", "e_drift", "f_proxy", "g_lambda")
FIXED = {"fixed_theta0": THETA0, "fixed_f1/8_k4": theta(0.125, 4), "fixed_f1/4_k2": theta(0.25, 2),
         "fixed_f1/4_k4_b200": theta(0.25, 4, 200)}


def true_gamma(n, world):
    if n is None or n == NODES:
        return 1.0
    return {200: 1.3, 400: 1.15, 1600: 1.0, 3200: 1.03, 6400: world["g6400"]}.get(int(n), 1.0)


def true_cost(t, world):
    t = norm(t)
    f, k, b = t["f"], t["k"] or 1, t["b"]
    raw = (1 - f) * (b / NODES) * true_gamma(b, world) + f * k * true_gamma(k * NODES, world) + RHO
    return raw / (1 + RHO)


def g_true(surface, t, n, world):
    t = norm(t)
    if surface == "a_flat":
        return G0
    f, k, b, lam = t["f"], t["k"], t["b"], t["lambda"]
    games_rel = 1.0 / true_cost(t, world)            # the arm's games at equal GPU, vs theta0
    pol_rel = games_rel * (f + (1 - f) * (1.0 if b == NODES else 0.0))
    base = A_VALUE * math.log(games_rel) + A_POLICY * math.log(max(pol_rel, 1e-3))
    q, sf, B = {2: 0.5, 4: 1.0, 8: 0.9}, 0.10, 3.0       # (b): deep targets help, not enough to pay for them
    lam_term = {0.0: 0.0, 0.25: -1.0, 0.5: -2.5}
    if surface == "c_interior":
        q, sf, B = {2: 0.45, 4: 1.0, 8: 0.85}, 0.12, 22.0
    elif surface == "d_edge_k8":
        q, sf, B = {2: 0.3, 4: 0.6, 8: 1.0}, 0.12, 30.0
    elif surface == "e_drift":                            # theta0 best early, interior f=1/4 k=4 by it20
        q, sf, B = {2: 0.45, 4: 1.0, 8: 0.85}, 0.12, 4.0 + 18.0 * min(1.0, n / 20.0)
    elif surface == "g_lambda":
        lam_term = {0.0: 0.0, 0.25: 7.0, 0.5: 4.0}
    benefit = B * (1.0 - math.exp(-f / sf)) * q[k] if f > 0 else 0.0
    return G0 + base + benefit + lam_term[lam]


def bias(surface, t):
    """Proxy-adversarial surface: the probe score rewards deep targets that do not buy Elo."""
    if surface != "f_proxy":
        return 0.0
    t = norm(t)
    return 12.0 * (1.0 - math.exp(-t["f"] / 0.12)) * {2: 0.5, 4: 1.0, 8: 1.2}[t["k"]] if t["f"] > 0 else 0.0


def oracle(surface, n, world, prior):
    arms = [a for a in all_arms(prior) if a["src"] == "cap"]
    return max(arms, key=lambda a: (round(g_true(surface, a, n, world), 9), -cost(a), key(a) == key(THETA0)))


def realised_segment(t, games, world, rng):
    """Synthetic realised numbers for one segment (for the gamma estimator)."""
    t = norm(t)
    per_game = 145.3 * 760.8 / NODES
    deep = per_game * games * t["f"] * (t["k"] or 1) * NODES
    fast = per_game * games * (1 - t["f"]) * t["b"]
    gpu = V_GPU_S_PER_VISIT * (deep * true_gamma((t["k"] or 1) * NODES, world) + fast * true_gamma(t["b"], world))
    gpu *= 1 + rng.uniform(-JITTER, JITTER)
    return {"gpu_s": gpu, "games": games, "deep_visits": deep, "fast_visits": fast, "visits": deep + fast}


def sim_iteration_gpu(plan_segments, thetas, world, rng):
    gpu = 0.0
    for name, s in plan_segments.items():
        gpu += s["games"] * true_cost(thetas[s["theta"]], world) / GAMES_PER_GPU_H
    return gpu * (1 + rng.uniform(-JITTER, JITTER)) + FIXED_STAGES_GPU_H


def mini_gate(rng, delta_elo, games=100, draw=0.1):
    p = 1.0 / (1.0 + 10 ** (-delta_elo / 400.0))
    w = l = dr = 0
    for _ in range(games):
        u = rng.random()
        if u < draw:
            dr += 1
        elif rng.random() < p:
            w += 1
        else:
            l += 1
    return {"games": games, "wins": w, "losses": l, "draws": dr}


REPLICATES = 10       # replicate round before the run: theta0 vs theta0 mini-trains on R different game sets


def replicate_prior(rng, sigma, tau, replicates=REPLICATES):
    """The prior's noise numbers as the replicate round would MEASURE them (not the true values): R iterations'
    worth of different games, two seeds each, both arms theta0, so d_rs = tau_r + e_rs."""
    pairs = []
    for _ in range(replicates):
        shared = rng.gauss(0, tau) if tau else 0.0
        pairs.append([shared + rng.gauss(0, sigma) for _ in range(2)])
    sig2 = sum((a - b) ** 2 / 2 for a, b in pairs) / replicates
    means = [(a + b) / 2 for a, b in pairs]
    m = sum(means) / replicates
    between = sum((x - m) ** 2 for x in means) / (replicates - 1)
    return {"sigma_hat": math.sqrt(sig2), "tau_hat": math.sqrt(max(between - sig2 / 2, 0.0)),
            "tau_df": replicates - 1}


SIM_FLAGS = {"score_stage": True, "probe_bias_measured": True, "search_q_blend": True}


def run_controller(surface, seed, iters, sigma, prior_extra=None, tau=0.0):
    rng = random.Random(det_hash("sim", surface, seed))
    world = {"g6400": rng.choice((1.0, 1.5)), "k8_ok": surface == "d_edge_k8" or rng.random() < 0.5}
    prior = dict(SIM_FLAGS, fits_bench={"8": world["k8_ok"]})
    prior.update(replicate_prior(random.Random(det_hash("replicate", surface, seed)), sigma, tau))
    prior.update(prior_extra or {})
    ctx = {"prior": prior, "run_seed": seed}
    S = init_state(prior)
    rows = []
    for n in range(1, iters + 1):
        plan = plan_core(n, S, ctx)
        star, chal = norm(plan["theta_star"]), plan["challenger"]
        thetas = {"inc": star, "chal": chal or star}
        gpu = sim_iteration_gpu(plan["segments"], thetas, world, rng)
        g_star = g_true(surface, star, n, world)
        if chal and plan["challenger_kind"] == "selfplay":
            share = 1.0 / len(plan["segments"])
            g_main = (1 - share) * g_star + share * g_true(surface, dict(norm(chal), **{"lambda": star["lambda"]}),
                                                           n, world)
        else:
            g_main = g_star
        obs = {"iter_gpu_h": gpu, "gate": {"elo": g_main + rng.gauss(0, GATE_SD)}, "alerts": [], "calibration": []}
        if chal:
            gc = g_true(surface, chal, n, world)
            nseed = len(plan["minitrain"]["seeds"])
            scores = {}
            shared = rng.gauss(0, tau) if tau else 0.0      # which games were played: common to both seeds
            for s in range(nseed):
                d = (gc - g_star) + bias(surface, chal) - bias(surface, star) + shared + rng.gauss(0, sigma)
                base = 2.1 + rng.gauss(0, 0.05)
                scores[f"inc_s{s}"], scores[f"chal_s{s}"] = base, base - d
            obs["scores"] = scores
            if plan["challenger_kind"] == "selfplay":
                x, t = plan["challenger_segment"], plan["twin_segment"]
                obs["realised"] = {x: realised_segment(chal, plan["segments"][x]["games"], world, rng),
                                   t: realised_segment(star, plan["segments"][t]["games"], world, rng)}
            tmp = copy.deepcopy(plan["state_in"])
            pk = pair_key(star, chal)
            sd = score_diffs(scores)
            tmp["evidence"].setdefault(pk, []).append({"iter": n, "d": sd[0], "sigma_n": sd[1], "seeds": sd[2]})
            if sd[1] is not None:
                tmp["sigma_samples"].append(sd[1])
            ev = evaluate(tmp, pk, n, prior, S.get("_calib", []))
            if ev["eligible"] and plan["minigate_allowed"] and not plan["revert_test"]:
                obs["minigate"] = mini_gate(rng, MINI_ELO_SCALE * (gc - g_star))
        obs["calibration"] = S.get("_calib", [])
        res = decide_core(plan, obs, ctx)
        calib = S.get("_calib", []) + [[g_main + rng.gauss(0, sigma), obs["gate"]["elo"]]]
        S = res["state"]
        S["_calib"] = calib
        oh = res["state"]["overhead"]["this_iter_gpu_h"]
        rows.append({"g": g_main, "gpu": gpu + oh, "oh": oh, "star": key(S["theta_star"]),
                     "action": res["decision"]["action"], "chal": key(chal) if chal else None})
    return rows, world, prior


def run_fixed(surface, t, seed, iters):
    rng = random.Random(det_hash("sim", surface, seed))
    world = {"g6400": rng.choice((1.0, 1.5)), "k8_ok": surface == "d_edge_k8" or rng.random() < 0.5}
    rows = []
    for n in range(1, iters + 1):
        games = games_for(t, {}, SEGMENT_GAMES) * SEGMENTS
        gpu = games * true_cost(t, world) / GAMES_PER_GPU_H * (1 + rng.uniform(-JITTER, JITTER)) + FIXED_STAGES_GPU_H
        rows.append({"g": g_true(surface, t, n, world), "gpu": gpu, "oh": 0.0})
    return rows, world


def run_thompson(surface, seed, iters, sigma):
    """Baseline from design A, simplified: independent Gaussian prior per arm (no GP kernel, no beta regression),
    Thompson-sampled challenger, promote at P(better) >= 0.95. Same overhead items, no mini-gate."""
    rng = random.Random(det_hash("sim", surface, seed))
    world = {"g6400": rng.choice((1.0, 1.5)), "k8_ok": surface == "d_edge_k8" or rng.random() < 0.5}
    prior = dict(SIM_FLAGS, fits_bench={"8": world["k8_ok"]})
    arms = [a for a in all_arms(prior) if a["src"] == "cap"]
    keys = [key(a) for a in arms]
    i0 = keys.index(key(THETA0))
    m = len(arms)
    mu = [0.0] * m
    cov = [[(0.0 if i == i0 or j == i0 else (64.0 if i == j else 0.0)) for j in range(m)] for i in range(m)]
    star = i0
    rows, cum_oh, run_gpu, probe_iter = [], 0.0, 0.0, None
    for n in range(1, iters + 1):
        sample = [mu[i] + math.sqrt(max(cov[i][i], 0.0)) * rng.gauss(0, 1) for i in range(m)]  # marginal TS
        cands = [i for i in range(m) if i != star]
        c = max(cands, key=lambda i: sample[i])
        refresh = probe_iter is None or n - probe_iter >= PROBE_EVERY
        items = overhead_items(refresh, 2, False)
        if (cum_oh + sum(items.values())) / (run_gpu + ITER_GPU_H + sum(items.values())) > OVERHEAD_CAP:
            c = None
        segs = {f"seg{k:02d}": {"theta": "inc", "games": games_for(arms[star], {}, SEGMENT_GAMES)}
                for k in range(SEGMENTS)}
        kind = None
        if c is not None:
            kind = "lambda" if same_selfplay(arms[star], arms[c]) else "selfplay"
            if kind == "selfplay":
                segs["seg04"] = {"theta": "chal", "games": games_for(arms[c], {}, SEGMENT_GAMES)}
        gpu = sim_iteration_gpu(segs, {"inc": arms[star], "chal": arms[c] if c is not None else arms[star]},
                                world, rng)
        g_star = g_true(surface, arms[star], n, world)
        g_main = g_star
        oh = 0.0
        if c is not None:
            if kind == "selfplay":
                g_main = 0.8 * g_star + 0.2 * g_true(surface, dict(arms[c], **{"lambda": arms[star]["lambda"]}),
                                                     n, world)
            oh = sum(items.values())
            if refresh:
                probe_iter = n
            gc = g_true(surface, arms[c], n, world)
            d = sum((gc - g_star) + bias(surface, arms[c]) - bias(surface, arms[star]) + rng.gauss(0, sigma)
                    for _ in range(2)) / 2
            # posterior update for x = e_c - e_star, noise sigma^2/2 (Sherman-Morrison)
            x = [0.0] * m
            x[c] += 1.0
            x[star] -= 1.0
            cx = [sum(cov[i][j] * x[j] for j in (c, star)) for i in range(m)]
            s = sigma * sigma / 2 + cx[c] - cx[star]
            resid = d - (mu[c] - mu[star])
            for i in range(m):
                mu[i] += cx[i] * resid / s
            for i in range(m):
                if cx[i]:
                    for j in range(m):
                        if cx[j]:
                            cov[i][j] -= cx[i] * cx[j] / s
            var = cov[c][c] + cov[star][star] - 2 * cov[c][star]
            if var > 0 and phi((mu[c] - mu[star]) / math.sqrt(var)) >= 0.95:
                star = c
        cum_oh += oh
        run_gpu += gpu + oh
        rows.append({"g": g_main, "gpu": gpu + oh, "oh": oh, "star": keys[star]})
    return rows


def summarize(rows, T):
    r = rows[:T]
    return sum(x["g"] for x in r) / sum(x["gpu"] for x in r)


def simulate(iterations=40, seeds=200, sigma=4.0, surfaces=SURFACES, lengths=(4, 8, 10, 20, 40), thompson=True,
             tau=0.0):
    lengths = tuple(sorted({L for L in lengths if L <= iterations} | {iterations}))
    out = {"iterations": iterations, "seeds": seeds, "sigma": sigma, "tau": tau, "surfaces": {}}
    for surface in surfaces:
        acc = {"controller": {L: [] for L in lengths}, "oracle": {L: [] for L in lengths},
               "thompson": {L: [] for L in lengths}}
        for name in FIXED:
            acc[name] = {L: [] for L in lengths}
        depart = {L: 0 for L in lengths}
        found_at, found_end, oh_fracs, oh_max = [], {L: 0 for L in lengths}, [], 0.0
        promotions, reverts = 0, 0
        for s in range(seeds):
            rows, world, prior = run_controller(surface, s, iterations, sigma, tau=tau)
            orc = [oracle(surface, n, world, prior) for n in range(1, iterations + 1)]
            # the oracle knows the true surface and the true gamma: its games fill the nominal GPU budget exactly
            orows = [{"g": g_true(surface, t, n, world), "gpu": ITER_GPU_H} for n, t in enumerate(orc, 1)]
            fixed = {name: run_fixed(surface, t, s, iterations)[0] for name, t in FIXED.items()
                     if legal(t, prior)}
            th = run_thompson(surface, s, iterations, sigma) if thompson else None
            first = None
            for n, row in enumerate(rows, 1):
                if first is None and row["star"] == key(orc[n - 1]) and key(orc[n - 1]) != key(THETA0):
                    first = n
            for L in lengths:
                acc["controller"][L].append(summarize(rows, L))
                acc["oracle"][L].append(summarize(orows, L))
                for name, fr in fixed.items():
                    acc[name][L].append(summarize(fr, L))
                if th:
                    acc["thompson"][L].append(summarize(th, L))
                if any(r["star"] != key(THETA0) for r in rows[:L]):
                    depart[L] += 1
                if rows[L - 1]["star"] == key(orc[L - 1]):
                    found_end[L] += 1
            found_at.append(first)
            tot_oh, tot = sum(r["oh"] for r in rows), sum(r["gpu"] for r in rows)
            oh_fracs.append(tot_oh / tot)
            oh_max = max(oh_max, max(sum(r["oh"] for r in rows[:L]) / sum(r["gpu"] for r in rows[:L])
                                     for L in range(1, iterations + 1)))
            promotions += sum(r["action"] == "promote" for r in rows)
            reverts += sum(r["action"] == "revert" for r in rows)
        mean = lambda xs: sum(xs) / len(xs) if xs else None  # noqa: E731
        res = {"elo_per_gpu_h": {arm: {L: mean(v[L]) for L in lengths} for arm, v in acc.items() if v[lengths[0]]},
               "departure_rate": {L: depart[L] / seeds for L in lengths},
               "found_optimum_rate": {L: found_end[L] / seeds for L in lengths},
               "first_found_median": statistics.median([f for f in found_at if f]) if any(found_at) else None,
               "first_found_rate": sum(1 for f in found_at if f) / seeds,
               "overhead_mean": mean(oh_fracs), "overhead_max": oh_max,
               "promotions_per_run": promotions / seeds, "reverts_per_run": reverts / seeds}
        o = res["elo_per_gpu_h"]["oracle"]
        res["regret_vs_oracle"] = {L: 1 - res["elo_per_gpu_h"]["controller"][L] / o[L] for L in lengths}
        z = res["elo_per_gpu_h"].get("fixed_theta0", {})
        res["captured_vs_theta0"] = {L: ((res["elo_per_gpu_h"]["controller"][L] - z[L]) / (o[L] - z[L])
                                         if z and o[L] - z[L] > 0.02 * o[L] else None) for L in lengths}
        out["surfaces"][surface] = res
    return out


def markdown(result):
    lengths = sorted(next(iter(result["surfaces"].values()))["departure_rate"])
    lines = [f"Sim: {result['seeds']} seeds x {result['iterations']} iterations per surface, "
             f"per-seed mini-train noise sigma = {result['sigma']:g}, shared per-iteration noise tau = "
             f"{result.get('tau', 0):g} Elo-equivalent probe units; prior noise from a simulated {REPLICATES}-"
             f"replicate round; prior flags {sorted(k for k, v in SIM_FLAGS.items() if v)}.", "",
             "Elo per GPU-h as % of the oracle (oracle = best legal fixed recipe, per iteration on e_drift), at "
             "iteration " + "/".join(str(L) for L in lengths) + ":", "",
             "| surface | controller (B) | TS baseline (A-lite) | fixed theta0 | fixed f1/8 k4 | fixed f1/4 k2 | "
             "fixed f1/4 k4 b200 |", "|---|---|---|---|---|---|---|"]
    for s, r in result["surfaces"].items():
        e, o = r["elo_per_gpu_h"], r["elo_per_gpu_h"]["oracle"]
        cell = lambda arm: "/".join(f"{100 * e[arm][L] / o[L]:.1f}" for L in lengths) if arm in e else "-"  # noqa
        lines.append(f"| {s} | {cell('controller')} | {cell('thompson')} | {cell('fixed_theta0')} | "
                     f"{cell('fixed_f1/8_k4')} | {cell('fixed_f1/4_k2')} | {cell('fixed_f1/4_k4_b200')} |")
    lines += ["", "| surface | left theta0 by it4/8/10/20/40 | theta* = optimum at end of it8/20/40 | "
                  "share of the oracle's gain over theta0 captured, it8/20/40 | "
                  "median first iteration at optimum (share of runs) | promotions/run | reverts/run | "
                  "overhead mean / max |", "|---|---|---|---|---|---|---|---|"]
    for s, r in result["surfaces"].items():
        dep = "/".join(f"{100 * r['departure_rate'][L]:.0f}%" for L in lengths if L in (4, 8, 10, 20, 40))
        fnd = "/".join(f"{100 * r['found_optimum_rate'][L]:.0f}%" for L in lengths if L in (8, 20, 40))
        ff = f"{r['first_found_median']} ({100 * r['first_found_rate']:.0f}%)" if r["first_found_median"] else \
            f"- ({100 * r['first_found_rate']:.0f}%)"
        cap = "/".join("-" if r["captured_vs_theta0"][L] is None else f"{100 * r['captured_vs_theta0'][L]:.0f}%"
                       for L in lengths if L in (8, 20, 40))
        lines.append(f"| {s} | {dep} | {fnd} | {cap} | {ff} | {r['promotions_per_run']:.2f} | {r['reverts_per_run']:.2f} | "
                     f"{100 * r['overhead_mean']:.1f}% / {100 * r['overhead_max']:.1f}% |")
    return "\n".join(lines)


def cmd_simulate(args):
    surfaces = tuple(args.surfaces.split(",")) if args.surfaces else SURFACES
    res = simulate(args.iterations, args.seeds, args.sigma, surfaces, thompson=not args.no_thompson, tau=args.tau)
    if args.json:
        write_atomic(args.json, dumps(res))
    if args.markdown:
        print(markdown(res))
    else:
        print(json.dumps(sig(res), sort_keys=True, indent=1))
    return res


# ------------------------------------------------------------------------------------------------ cli


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan", help="write balance/itN.plan.json (before iteration N's self-play)")
    p.add_argument("run")
    p.add_argument("n", type=int)
    p.add_argument("--prior")
    p.add_argument("--generator")
    p.add_argument("--segments", type=int, default=SEGMENTS)
    p.add_argument("--segment-games", type=int, default=SEGMENT_GAMES)
    p.add_argument("--mini-steps", type=int, default=MINI_STEPS)
    p.add_argument("--run-seed", type=int, default=0)
    d = sub.add_parser("decide", help="write balance/itN.json (after match/promote)")
    d.add_argument("run")
    d.add_argument("n", type=int)
    d.add_argument("--dry", action="store_true", help="print eligibility (run the mini-gate?) and write nothing")
    d.add_argument("--prior", help="the prior plan read (default: the plan's prior_path, else balance/prior.json)")
    d.add_argument("--no-score", action="store_true",
                   help="the score stage failed: decide without scores (action none) and keep the ledger")
    r = sub.add_parser("realise", help="write balance/itN/realised.json from the segments' files (before decide)")
    r.add_argument("run")
    r.add_argument("n", type=int)
    r.add_argument("--accounting", help="JSON {segNN: gpu_s} from the job accounting (default "
                                        "balance/itN/accounting.json); without it no gamma sample")
    r.add_argument("--force", action="store_true")
    s = sub.add_parser("status", help="print the balance history of a run")
    s.add_argument("run")
    s.add_argument("--json", action="store_true")
    m = sub.add_parser("simulate", help="synthetic-world simulation of the controller")
    m.add_argument("--iterations", type=int, default=40)
    m.add_argument("--seeds", type=int, default=200)
    m.add_argument("--sigma", type=float, default=4.0)
    m.add_argument("--tau", type=float, default=0.0, help="shared per-iteration noise in d (which games were played)")
    m.add_argument("--surfaces")
    m.add_argument("--json")
    m.add_argument("--markdown", action="store_true")
    m.add_argument("--no-thompson", action="store_true")
    a = ap.parse_args(argv)
    try:
        if a.cmd == "plan":
            cmd_plan(a.run, a.n, a)
        elif a.cmd == "decide":
            cmd_decide(a.run, a.n, a)
        elif a.cmd == "realise":
            cmd_realise(a.run, a.n, a)
        elif a.cmd == "status":
            cmd_status(a.run, a)
        else:
            cmd_simulate(a)
    except InputError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
