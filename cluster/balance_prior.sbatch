#!/bin/bash
#SBATCH -J balance_prior -A paceship-bughouse -q inferno
#SBATCH -N1 -n6 --mem=48G --gres=gpu:rtx_6000:1 -t 8:00:00
#SBATCH -o balance_prior_%A_%a.log
# The measured prior the search-balance controller needs before it may explore (docs/search_balance.md sections 8 and
# 10; cluster/README.md "Search balance"). One GPU job array on an existing run's finished iteration(s), never touching
# the run's own files except RUN_ROOT/balance/prior.json (atomic, written once all tasks are in) and RUN_ROOT/probe/
# (probe_net.py build, the same frozen set the monitor jobs build). Work goes to RUN_ROOT/balance/prior_work/.
#   task 0      bench (budget sweep): BENCH_GAMES stock-setting games at each of BENCH_NODES (800 1600 3200 6400) plus
#               one f=1/8,k=8 cap mix, PARALLEL_GAMES at once -> gamma(n) = (GPU-s per visit at n) / (at 800), and
#               fits_bench["8"] = the 6400 and the k=8 runs finished (no OOM)
#   task 1      probe bias: a challenger segment X at f=1/8,k=4 with the twin seg00's seed and GPU time (games =
#               seg00's / c), probe-mode self-play (PROBE_GAMES at f=1/8,k=16, policy mask, no deep Dirichlet, seed
#               1000N+998, current best net) into a deep probe, and mini-trains inc(seg00) vs chal(X), 2 seeds
#   task 2..    replicate r: theta0 vs theta0 mini-trains on two disjoint finished train segments (2 seeds each, same
#               seed for both arms), for sigma_hat (seed-pair spread) and tau_hat (spread of d across replicates)
#   merge       run by whichever task finishes last (mkdir lock): every mini net is scored with cluster/probe_net.py
#               eval on the deep probe and the run's frozen probe; prior.json is written
# Score of a mini net (lower is better): the live stage's own function, hivemind.cli.balance.evaluate_net (deep probe
# policy CE + alpha WDL + 0.1 moves-left, + 0.5 alpha WDL on RUN_ROOT/probe), so sigma/tau carry over to live d. A fork
# without it falls back to cluster/probe_net.py: deep heldout.loss + 0.5 x frozen heldout.value_loss (prior.json
# "score_definition" says which; a fallback prior should be remeasured once the live scorer exists).
# Flags written: fits_bench (bench), probe_bias_measured (true only when the deep-arm bias is small: |d_deep/sigma_deep
# - d_frozen/sigma_frozen| <= BIAS_Z_MAX, 1.0, with sigmas from >= 3 replicates: ITERS="1 2" or more), and capability checks of the fork checkout: score_stage (`hivemind
# balance score --help` works), search_q_blend (distill-train lists --search-q-blend), policy_positions
# (hivemind.cli.train.policy_positions exists). APPLY_FLAGS=0 writes every measurement with all four flags false.
#
#   bash cluster/balance_prior.sbatch plan                  # RUN_ROOT=runs/full/hybrid1 ITERS="1": pairs, cost, sbatch line
#   RUN_ROOT=runs/full/hybrid1 ITERS="1" sbatch --array=0-3 cluster/balance_prior.sbatch     # as `plan` prints it
#   RUN_ROOT=... TASK=2 bash cluster/balance_prior.sbatch   # one task by hand (Mac: tiny settings, DEVICE=cpu)
# Settings: RUN_ROOT (required), ITERS (finished iterations to take segments from, "1"), REPLICATES (max 10),
# MINI_STEPS (200), LR (1e-4), TRUNK_LR (5e-5), BATCH_SIZE (1024), WINDOW (3 replayed iterations, 50%), BENCH_NODES,
# BENCH_GAMES (36), PROBE_GAMES (150), PARALLEL_GAMES (6), ENGINE (W2/engine/build-balance-sp1/hivemind: never
# build-sp1, which live runs use; BUILD=1 builds it here from the checkout under a lock), MODEL (current best ONNX,
# default RUN_ROOT/models/<lineage>.onnx), DEVICE (cuda under Slurm, else cpu), BIAS_Z_MAX (1.0), APPLY_FLAGS (1),
# FORCE (0; 1 rewrites prior.json from the finished work in balance/prior_work; move that folder away to measure anew).
# GPU cost (RTX 6000, the section 2 numbers): bench ~0.8 GPU-h, bias ~3.8 (X segment 2.6 at 2000-game segments, probe
# 0.55, 4 minis 0.65), each replicate ~0.75, merge ~0.1 per 8 nets: about 4.7 + 0.75 R GPU-h, i.e. 6.2 for hybrid1
# after it1 (R = 2), 7.7 after it2 (R = 4), 12.2 at R = 10. Each task is one GPU; wall time = the longest task (~3 h).
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(git rev-parse --show-toplevel)}" || exit 1
[ -n "${SLURM_JOB_ID:-}" ] && source cluster/env.sh

REPO="$PWD"
RUN_ROOT="${RUN_ROOT:?set RUN_ROOT, e.g. runs/full/hybrid1}"
case "$RUN_ROOT" in /*) ;; *) RUN_ROOT="$REPO/$RUN_ROOT" ;; esac
W2="${W2:-$REPO/third_party/wave2-hivemind}"
PY="$W2/.venv/bin/python"
ITERS="${ITERS:-1}"
WORK="$RUN_ROOT/balance/prior_work"
ENGINE="${ENGINE:-$W2/engine/build-balance-sp1/hivemind}"
DEVICE="${DEVICE:-$([ -n "${SLURM_JOB_ID:-}" ] && echo cuda || echo cpu)}"
export RUN_ROOT ITERS WORK ENGINE DEVICE REPO W2
export REPLICATES="${REPLICATES:-10}" MINI_STEPS="${MINI_STEPS:-200}" LR="${LR:-1e-4}" TRUNK_LR="${TRUNK_LR:-5e-5}"
export BATCH_SIZE="${BATCH_SIZE:-1024}" WINDOW="${WINDOW:-3}" BENCH_NODES="${BENCH_NODES:-800 1600 3200 6400}"
export BENCH_GAMES="${BENCH_GAMES:-36}" PROBE_GAMES="${PROBE_GAMES:-150}" PARALLEL_GAMES="${PARALLEL_GAMES:-6}"
export MODEL="${MODEL:-}" BIAS_Z_MAX="${BIAS_Z_MAX:-1.0}" APPLY_FLAGS="${APPLY_FLAGS:-1}" FORCE="${FORCE:-0}"
die() { echo "balance_prior: $*" >&2; exit 1; }
[ -d "$RUN_ROOT/runs" ] || die "$RUN_ROOT has no runs/"
[ -x "$PY" ] || die "$PY missing (the fork's env: wave2_iterate.sbatch env step)"

# The helper, run as `$PY -c "$LIB" <command> ...` (fork python: torch + hivemind).
read -r -d '' LIB <<'EOF'
import glob, hashlib, json, math, os, re, shutil, subprocess, sys, time
E = os.environ
ROOT, WORK, REPO = E["RUN_ROOT"], E["WORK"], E["REPO"]
sys.path.insert(0, os.path.join(REPO, "cluster"))
import search_balance as sbal  # noqa: E402


def load(p):
    try:
        with open(p) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def dump(p, v):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    t = f"{p}.tmp{os.getpid()}"
    with open(t, "w") as f:
        json.dump(v, f, indent=1, sort_keys=True)
        f.write("\n")
    os.replace(t, p)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def it_dir(n):
    return os.path.join(ROOT, "runs", f"rl-it{n}")


def iters():
    return [int(x) for x in E["ITERS"].split()]


def pairs():
    """Disjoint pairs of finished theta0 train segments, in order: (n, a, b)."""
    out = []
    for n in iters():
        st = os.path.join(it_dir(n), "stages")
        segs = sorted(f[:-5] for f in os.listdir(st) if re.fullmatch(r"seg\d\d\.done", f)) if os.path.isdir(st) else []
        plan = load(os.path.join(ROOT, "balance", f"it{n}.plan.json")) or {}
        segs = [s for s in segs if (plan.get("segments", {}).get(s) or {}).get("theta", "inc") == "inc"
                and os.path.isdir(os.path.join(it_dir(n), "search", "train", s))]
        out += [(n, segs[i], segs[i + 1]) for i in range(0, len(segs) - 1, 2)]
    return out[:int(E["REPLICATES"])]


def tasks():
    return 2 + len(pairs())


def generator_onnx(n):
    """The ONNX file that generated iteration n's games (stages/generator holds its sha256)."""
    want = open(os.path.join(it_dir(n), "stages", "generator")).read().strip()
    for p in sorted(glob.glob(os.path.join(ROOT, "**", "*.onnx"), recursive=True)):
        if not p.endswith("-sp1.onnx") and sha(p) == want:
            return p
    raise SystemExit(f"no ONNX with the it{n} generator's sha256 {want[:12]} under {ROOT}")


def best_onnx():
    if E.get("MODEL"):
        return E["MODEL"]
    c = [p for p in glob.glob(os.path.join(ROOT, "models", "*.onnx")) if not re.search(r"-(it\d+|sp1)\.onnx$", p)]
    if len(c) != 1:
        raise SystemExit(f"set MODEL: {len(c)} candidate generator ONNX files in {ROOT}/models")
    return c[0]


def init_ckpt(n):
    import probe_net
    path, _ = probe_net.previous_network(ROOT, n)
    if not path:
        raise SystemExit(f"no start checkpoint for it{n}")
    return path


def engine_extra(n):
    from hivemind import wave2
    from hivemind.cli import selfplay
    marker = load(os.path.join(it_dir(n), "stages", "seg00.done")) or {}
    w = dict(marker.get("wave2") or {})
    if w.get("base_mix"):
        w["base_mix"] = wave2.parse_base_mix(w["base_mix"])
    opts = wave2.Options(**w) if w else wave2.STOCK
    return [*selfplay.time_model_options("clock"), *opts.engine_args()]


def play(name, n, model, games, seed, extra):
    """Self-play into WORK/name (data/ and games/); returns {elapsed_s, visits, games, ok}. Resumes nothing: a
    finished run (summary.json) is kept, a partial one is replaced."""
    from hivemind.cli import selfplay
    from pathlib import Path
    out = os.path.join(WORK, name)
    s = load(os.path.join(out, "summary.json"))
    if s:
        return s
    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(out)
    m = Path(model).resolve()
    copy = selfplay.one_worker_copy(m)
    selfplay.refresh_copy(m, copy)
    data = os.path.join(out, "val" if name.startswith("probe") else "data")  # 'val' in the path: probe_net heldout
    cmd = selfplay.engine_command(E["ENGINE"], copy, games, seed, Path(data), Path(out) / "games",
                                  int(E["PARALLEL_GAMES"]), [*engine_extra(n), *extra])
    t0 = time.time()
    with open(os.path.join(out, "selfplay.log"), "w") as log:
        rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT).returncode
    el = time.time() - t0
    visits = 0
    clocks = os.path.join(out, "games", "seat_clocks.tsv")
    if os.path.isfile(clocks):
        with open(clocks) as f:
            head = f.readline().rstrip("\n").split("\t")
            col = head.index("nodes") if "nodes" in head else None
            for line in f:
                c = line.rstrip("\n").split("\t")
                if col is not None and len(c) > col and c[col].isdigit():
                    visits += int(c[col])
    text = open(os.path.join(out, "selfplay.log"), errors="replace").read()
    s = {"name": name, "games": games, "seed": seed, "extra": extra, "elapsed_s": round(el, 3), "visits": visits,
         "returncode": rc, "oom": bool(re.search(r"out of memory|OutOfMemory|bad_alloc", text, re.I)),
         "ok": rc == 0, "model": str(m), "data": data, "command": [str(x) for x in cmd]}
    s["ok"] = s["ok"] and not s["oom"]
    if s["ok"]:
        dump(os.path.join(out, "summary.json"), s)
    return s


def mini(name, n, data, seed):
    """One mini-train (the live stage's settings): init = iteration n's start network, 50% replay of the WINDOW
    iterations before n when they exist; WORK/minis/name/last.pt."""
    from hivemind.cli import train
    out = os.path.join(WORK, "minis", name)
    if os.path.isfile(os.path.join(out, "last.pt")):
        return out
    shutil.rmtree(out, ignore_errors=True)
    replay = [d for k in range(max(1, n - int(E["WINDOW"])), n)
              for d in sorted(glob.glob(os.path.join(it_dir(k), "search", "train", "seg*")))]
    val = [os.path.join(it_dir(n), "search", "val")]
    cmd = train.distill_command(init_ckpt(n), data, val, replay, int(E["MINI_STEPS"]), 0.5 if replay else 0.0,
                                int(E["BATCH_SIZE"]), seed, out, lr=E["LR"], trunk_lr=E["TRUNK_LR"])
    cmd[0] = sys.executable
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out + ".log", "w") as log:
        if subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=E["W2"]).returncode:
            raise SystemExit(f"mini-train {name} failed (see {out}.log)")
    return out


def task_bench():
    n = iters()[-1]
    model, rows = best_onnx(), {}
    for nodes in E["BENCH_NODES"].split():
        rows[nodes] = play(f"bench_{nodes}", n, model, int(E["BENCH_GAMES"]), 7000 + int(nodes) // 100,
                           ["--nodes", nodes])
    rows["mix_f8_k8"] = play("bench_mix_f8_k8", n, model, int(E["BENCH_GAMES"]), 7999,
                             ["--deep-fraction", "0.125", "--deep-multiplier", "8"])
    return {"rows": rows}


def task_bias():
    n = iters()[-1]
    t = os.path.join(it_dir(n), "search", "train", "seg00")
    tg = (load(os.path.join(it_dir(n), "stages", "seg00.done")) or {}).get("games")
    if not tg or not os.path.isdir(t):
        raise SystemExit(f"it{n} seg00 not finished")
    chal = sbal.theta(0.125, 4)
    xg = sbal.games_for(chal, {}, tg)
    x = play("bias_x", n, generator_onnx(n), xg, (load(os.path.join(it_dir(n), "stages", "seg00.done")) or {})
             .get("seed", 1000 * n), sbal.engine_args(chal))
    if not x["ok"]:
        raise SystemExit("bias: challenger segment failed")
    probe = play("probe_deep", n, best_onnx(), int(E["PROBE_GAMES"]), 1000 * n + 998,
                 ["--deep-fraction", "0.125", "--deep-multiplier", "16", "--fast-policy-mask", "true",
                  "--deep-dirichlet", "false"])
    if not probe["ok"]:
        raise SystemExit("bias: probe-mode self-play failed")
    nets = {}
    for s in (0, 1):
        seed = 1000 * n + 10 * s + 7
        nets[f"inc_s{s}"] = mini(f"bias_inc_s{s}", n, [t], seed)
        nets[f"chal_s{s}"] = mini(f"bias_chal_s{s}", n, [x["data"]], seed)
    return {"iteration": n, "twin_games": tg, "chal_games": xg, "theta": chal, "nets": nets, "x": x, "probe": probe}


def task_replicate(r):
    n, a, b = pairs()[r]
    nets = {}
    for s in (0, 1):
        seed = 1000 * n + 10 * s + 7
        nets[f"inc_s{s}"] = mini(f"rep{r}_{a}_s{s}", n, [os.path.join(it_dir(n), "search", "train", a)], seed)
        nets[f"chal_s{s}"] = mini(f"rep{r}_{b}_s{s}", n, [os.path.join(it_dir(n), "search", "train", b)], seed)
    return {"iteration": n, "segments": [a, b], "nets": nets}


def run_task(k):
    k = int(k)
    t0 = time.time()
    out = task_bench() if k == 0 else task_bias() if k == 1 else task_replicate(k - 2)
    out["task"], out["elapsed_s"] = k, round(time.time() - t0, 1)
    dump(os.path.join(WORK, "done", f"task{k}.json"), out)


# -------------------------------------------------------------------- merge

def score_nets(probe_root, nets, n, tag):
    """{net path: heldout dict} via probe_net.py eval (cached per net and probe)."""
    out = {}
    for name, d in nets.items():
        path = os.path.join(d, "last.pt")
        res = os.path.join(WORK, "scores", tag, os.path.basename(d) + ".json")  # the mini's own (unique) name
        if not load(res):
            subprocess.run([sys.executable, os.path.join(REPO, "cluster", "probe_net.py"), "eval", probe_root, str(n),
                            "--net", path, "--prev", init_ckpt(n), "--out", res, "--device", E["DEVICE"],
                            "--ablation-positions", "64", "--top", "1"], check=True, stdout=subprocess.DEVNULL)
        out[name] = (load(res) or {}).get("heldout") or {}
    return out


def build_probe(root, sources=None):
    if os.path.isfile(os.path.join(root, "probe", "manifest.json")):
        return
    cmd = [sys.executable, os.path.join(REPO, "cluster", "probe_net.py"), "build", root]
    if sources:
        cmd += ["--from", *sources, "--positions", "100000", "--max-chunks", "100000"]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)


def live_scorer():
    """The live `hivemind balance score` formula on one net, or None when the fork does not have it yet."""
    try:
        from hivemind.cli.balance import ALPHA, evaluate_net
        from hivemind.distill.data import chunk_paths
    except Exception:  # noqa: BLE001
        return None
    paths = chunk_paths([os.path.join(WORK, "probe_deep", "val")])
    swarm = ROOT + "/probe" if os.path.isfile(os.path.join(ROOT, "probe", "manifest.json")) else None
    from pathlib import Path
    import torch
    device = torch.device(E["DEVICE"])
    return lambda net: evaluate_net(net, paths, Path(swarm) if swarm else None, ALPHA, device, 256)


def pair_stats(ds_by_rep):
    """ds_by_rep: [[d_s0, d_s1], ...] -> sigma_hat (RMS seed-pair spread), tau_hat, tau_df."""
    sig = [abs(d[0] - d[1]) / math.sqrt(2) for d in ds_by_rep if len(d) == 2]
    sh = math.sqrt(sum(x * x for x in sig) / len(sig)) if sig else None
    means = [sum(d) / len(d) for d in ds_by_rep if d]
    if len(means) >= 2 and sh is not None:
        m = sum(means) / len(means)
        var = sum((x - m) ** 2 for x in means) / (len(means) - 1)
        tau = math.sqrt(max(var - sh * sh / 2, 0.0))
        return sh, tau, len(means) - 1
    return sh, None, 0


def capability(cmd, needle=None):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=E["W2"], timeout=120)
    except (OSError, subprocess.SubprocessError):
        return False
    return r.returncode == 0 and (needle is None or needle in r.stdout + r.stderr)


def merge():
    done = {int(f[4:-5]): load(os.path.join(WORK, "done", f)) for f in os.listdir(os.path.join(WORK, "done"))}
    bench, bias = done.get(0), done.get(1)
    reps = [done[k] for k in sorted(done) if k >= 2]
    n_last = iters()[-1]
    build_probe(ROOT)   # the run's frozen probe (same as the monitor jobs'; a no-op when it exists)
    deep_root = os.path.join(WORK, "deep_root")
    deep_ok = bool(bias and bias.get("probe"))
    if deep_ok:
        os.makedirs(os.path.join(deep_root, "runs"), exist_ok=True)
        build_probe(deep_root, [bias["probe"]["data"]])

    live = live_scorer() if deep_ok else None

    def score(entry):
        """(primary score, frozen-probe score, deep-including score) per net. Primary = the live stage's own formula
        (hivemind.cli.balance.evaluate_net) when the fork has it, else the probe_net.py stand-in in the header."""
        n = entry["iteration"]
        frozen = score_nets(ROOT, entry["nets"], n, "frozen")
        s = {}
        if live:
            for k, d in entry["nets"].items():
                res = os.path.join(WORK, "scores", "live", os.path.basename(d) + ".json")
                if not load(res):
                    dump(res, live(os.path.join(d, "last.pt")))
                s[k] = load(res)["score"]
            return s, {k: frozen[k]["loss"] for k in entry["nets"]}, dict(s)
        deep = score_nets(deep_root, entry["nets"], n, "deep") if deep_ok else {}
        for k in entry["nets"]:
            s[k] = deep[k]["loss"] + 0.5 * frozen[k]["value_loss"] if deep_ok else frozen[k]["loss"]
        return s, {k: frozen[k]["loss"] for k in entry["nets"]}, {k: deep[k]["loss"] for k in deep}

    def ds(scores):
        return [scores[f"inc_s{s}"] - scores[f"chal_s{s}"] for s in (0, 1)]

    rep_rows, rep_d, rep_df, rep_dd = [], [], [], []
    for e in reps:
        s, fz, dp = score(e)
        rep_d.append(ds(s)), rep_df.append(ds(fz))
        if dp:
            rep_dd.append(ds(dp))
        rep_rows.append({"iteration": e["iteration"], "segments": e["segments"], "scores": s, "d": ds(s)})
    sh, tau, tau_df = pair_stats(rep_d)
    prior = {"schema": 1, "measured_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "run": os.path.basename(ROOT),
             "iterations": iters(), "sigma_hat": sh, "tau_hat": tau, "tau_df": tau_df, "eta_batch": 1.0,
             "reanalyse_fidelity": False, "calibration": [],
             "score_definition": ("hivemind.cli.balance.evaluate_net (the live balance score)" if live else
                                  "probe_net deep heldout.loss + 0.5 * frozen heldout.value_loss" if deep_ok
                                  else "probe_net frozen heldout.loss"),
             "replicates": rep_rows, "mini_steps": int(E["MINI_STEPS"])}
    # bench -> gamma, fits_bench
    gamma, fits8 = {}, False
    if bench:
        rows = bench["rows"]
        base = rows.get("800")
        if base and base["ok"] and base["visits"]:
            v0 = base["elapsed_s"] / base["visits"]
            for nodes, r in rows.items():
                if nodes.isdigit() and nodes != "800" and r["ok"] and r["visits"]:
                    gamma[nodes] = round((r["elapsed_s"] / r["visits"]) / v0, 4)
        fits8 = bool(rows.get("6400", {}).get("ok") and rows.get("mix_f8_k8", {}).get("ok"))
        prior["bench"] = {k: {x: r.get(x) for x in ("games", "elapsed_s", "visits", "ok", "oom")} for k, r in rows.items()}
    prior["gamma"], prior["fits_bench"] = gamma, {"8": fits8}
    # probe bias
    bias_small = False
    if bias:
        s, fz, dp = score(bias)
        d_main, d_frozen = ds(s), ds(fz)
        sd_frozen = pair_stats(rep_df)[0]
        sd_deep = pair_stats(rep_dd)[0] if rep_dd else None
        z = None
        if dp and sd_deep and sd_frozen:
            d_deep = ds(dp)
            z = (sum(d_deep) / 2) / sd_deep - (sum(d_frozen) / 2) / sd_frozen
            # sigma from fewer than 3 replicates is too rough to call the bias small
            bias_small = abs(z) <= float(E["BIAS_Z_MAX"]) and len(reps) >= 3
        prior["probe_bias"] = {"d": d_main, "d_frozen": d_frozen, "d_deep": ds(dp) if dp else None,
                               "sigma_deep": sd_deep, "sigma_frozen": sd_frozen, "bias_z": z,
                               "bias_z_max": float(E["BIAS_Z_MAX"]), "small": bias_small, "replicates": len(reps),
                               "chal_games": bias["chal_games"], "twin_games": bias["twin_games"]}
    apply = E["APPLY_FLAGS"] == "1"
    caps = {"score_stage": capability([sys.executable, "-m", "hivemind", "balance", "score", "--help"]),
            "search_q_blend": capability([sys.executable, "-m", "hivemind", "distill-train", "--help"],
                                         "--search-q-blend"),
            "policy_positions": capability([sys.executable, "-c", "from hivemind.cli.train import policy_positions"])}
    prior.update({k: bool(apply and v) for k, v in caps.items()})
    prior["probe_bias_measured"] = bool(apply and bias_small)
    prior["capabilities"] = caps
    # GPU-hours of the whole job (each task held one GPU for its elapsed time)
    prior["cost_gpu_h"] = round(sum((d or {}).get("elapsed_s", 0) for d in done.values()) / 3600, 4)
    dump(os.path.join(ROOT, "balance", "prior.json"), prior)
    print(json.dumps({k: prior[k] for k in ("sigma_hat", "tau_hat", "tau_df", "gamma", "fits_bench",
                                            "probe_bias_measured", "score_stage", "cost_gpu_h")}))


def plan():
    ps = pairs()
    print(f"run {ROOT}  iterations {iters()}  replicates {len(ps)}")
    for r, (n, a, b) in enumerate(ps):
        print(f"  task {r + 2}: it{n} {a} vs {b}")
    if not ps:
        print("  no pair of finished theta0 train segments: sigma/tau cannot be measured yet")
    gpu = 0.8 + 3.8 + 0.75 * len(ps) + 0.1 * (1 + len(ps))
    print(f"estimated {gpu:.1f} GPU-hours over {tasks()} tasks (one GPU each)")
    print(f"RUN_ROOT={os.path.relpath(ROOT, REPO)} ITERS=\"{E['ITERS']}\" sbatch --array=0-{tasks() - 1} "
          "cluster/balance_prior.sbatch")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "plan":
        plan()
    elif cmd == "tasks":
        print(tasks())
    elif cmd == "task":
        run_task(sys.argv[2])
    elif cmd == "merge":
        merge()
EOF

if [ "${1:-}" = plan ]; then (cd "$W2" && "$PY" -c "$LIB" plan); exit $?; fi
if [ -f "$RUN_ROOT/balance/prior.json" ] && [ "$FORCE" != 1 ]; then
  die "$RUN_ROOT/balance/prior.json exists (FORCE=1 rewrites it from balance/prior_work; move prior_work away to measure anew)"
fi
TASK="${TASK:-${SLURM_ARRAY_TASK_ID:?set TASK or submit with --array (bash cluster/balance_prior.sbatch plan)}}"
NTASKS=$(cd "$W2" && "$PY" -c "$LIB" tasks) || die "cannot list the tasks"
mkdir -p "$WORK/done"
echo "commit $(git rev-parse HEAD)  wave2 $(git -C "$W2" rev-parse HEAD)  engine $ENGINE  task $TASK of $NTASKS  $(hostname)"

if [ ! -x "$ENGINE" ] && [ "${BUILD:-0}" = 1 ]; then
  bdir=$(dirname "$ENGINE")
  exec 9>"$W2/engine/.build-balance-sp1.lock"
  command -v flock >/dev/null && flock 9
  if [ ! -x "$ENGINE" ]; then
    if [ -n "${SLURM_JOB_ID:-}" ]; then
      BACKEND=(-DHIVEMIND_BACKEND=tensorrt -DTensorRT_DIR="$TensorRT_DIR" -DCUDA_TOOLKIT_ROOT_DIR="$CUDA_TOOLKIT_ROOT_DIR"
               -DHIVEMIND_NATIVE_ARCH=OFF)
    else
      BACKEND=(-DHIVEMIND_BACKEND=onnxruntime -DONNXRuntime_ROOT="${ONNXRuntime_ROOT:-/opt/homebrew/opt/onnxruntime}")
    fi
    (cd "$W2/engine" && cmake --preset ninja-release -B "$bdir" -DHIVEMIND_SEARCH_WORKERS=1 -DHIVEMIND_USE_CCACHE=OFF \
       "${BACKEND[@]}" && cmake --build "$bdir" --target hivemind -j "${CMAKE_BUILD_PARALLEL_LEVEL:-2}") \
      || die "engine build into $bdir failed"
  fi
  exec 9>&-
fi
[ -x "$ENGINE" ] || die "engine $ENGINE missing (BUILD=1 builds it; never point ENGINE at build-sp1)"
grep -q -- "--deep-fraction" <<<"$("$ENGINE" --help 2>&1)" || die "$ENGINE has no playout-cap flags (--deep-fraction)"

if [ ! -f "$WORK/done/task$TASK.json" ]; then
  (cd "$W2" && "$PY" -c "$LIB" task "$TASK") || die "task $TASK failed"
fi
# The last task in merges (scoring needs every mini net). A merge cut off leaves merge.lock: remove it and rerun any task.
have=$(find "$WORK/done" -name 'task*.json' | wc -l | tr -d ' ')
if [ "$have" -ge "$NTASKS" ]; then
  if mkdir "$WORK/merge.lock" 2>/dev/null; then
    trap 'rmdir "$WORK/merge.lock" 2>/dev/null' EXIT
    [ -f "$RUN_ROOT/balance/prior.json" ] && [ "$FORCE" != 1 ] && { echo "prior.json already written"; exit 0; }
    (cd "$W2" && "$PY" -c "$LIB" merge) || die "merge failed"
    echo "wrote $RUN_ROOT/balance/prior.json"
  else
    echo "another task is merging ($WORK/merge.lock)"
  fi
else
  echo "task $TASK done; $have of $NTASKS tasks in, the last one merges"
fi
