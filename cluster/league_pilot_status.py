#!/usr/bin/env python3
"""Status of a league pilot run (cluster/launch_league_pilot.sh, docs/custom_engine_spec_v7_league.md 7.2). Standard
library only; every file it reads may be missing (a run that has not got there yet), and then says so.

    python3 cluster/league_pilot_status.py runs/full/league1
    python3 cluster/league_pilot_status.py runs/full/league1 --json

Per iteration (RUN_ROOT/runs/rl-itN/stages):
  stages     the stage markers present (generator, league_plan, seg*, val, think_labels, train, match_*, gate2, promote)
  league     stages/league_plan.done: segments by opponent kind (self, past:itM, fixed:kK), PFSP weights when recorded
  self-play  finished segments per kind: games, learner W-D-L, the learner's mean pace k and think scale (each segment marker's
             "league" summary, spec v7 5.3)
  labels     stages/think_labels.done: records, labelled, label rates per board, noise rate (or the failure)
  train      steps; the 100 ms gate (match_generator, the old curve) and the 100 ms score vs it13
  gate2      per network (candidate rl-itN/models/C.onnx, incumbent = the iteration's generator) and arm:
             RUN_ROOT/gate2/nets/<sha16>/<arm>/summary.json (score with its range, pairs, turns ahead); the decision
             from stages/gate2.done, else a provisional one from `python -m hivemind.gate2 decide` (the wave2-hivemind
             env) once both networks have pairs
GPU-hours: sacct over full_<RUN>, sp_<RUN>, g2_<RUN>_*, tg_<RUN>, lb_<RUN>, mon_<RUN>_*, am_<RUN>_* since the run's
first watchdog line (the watchdog's cap, cluster/full_run_watchdog.sh), against launch_settings.txt's gpu_hours; "n/a
off-cluster" without sacct.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
W2 = os.path.join(REPO, "third_party", "wave2-hivemind")
ARMS = ("equal", "free")


def read_json(path: str):
    """The JSON in `path`, or None (missing or not JSON, e.g. a marker being written)."""
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def sha256(path: str) -> str | None:
    if not os.path.isfile(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def settings(root: str) -> dict:
    """launch_settings.txt as {key: value} (fields separated by two spaces, "key value")."""
    try:
        line = open(os.path.join(root, "launch_settings.txt")).read().strip()
    except OSError:
        return {}
    out = {}
    for field in line.split("  "):
        key, _, value = field.partition(" ")
        if key:
            out[key] = value
    return out


def kind(theta: str) -> str:
    """self, past or fixed from a league theta ("self", "past:it2", "fixed:k110"); other thetas as they are."""
    return str(theta).split(":", 1)[0] if theta else "plain"


def first(d: dict, *keys):
    for k in keys:
        if isinstance(d, dict) and d.get(k) is not None:
            return d[k]
    return None


# ----------------------------------------------------------------------------------------------- one iteration
def league_plan(stages: str) -> dict | None:
    plan = read_json(os.path.join(stages, "league_plan.done"))
    if plan is None:
        return None
    segs = plan.get("segments") or {}
    by = {}
    pfsp = {}
    for name, spec in sorted(segs.items()):
        theta = spec.get("theta", "self")
        by.setdefault(theta, []).append(name)
        weight = first(spec.get("league") or {}, "pfsp_weight", "pfsp", "weight")
        if weight is not None:
            pfsp[theta] = weight
    out = {"segments": len(segs), "by_theta": {t: len(v) for t, v in by.items()},
           "by_kind": {}, "pfsp": pfsp or plan.get("pfsp"), "summary": plan.get("summary")}
    for t, v in by.items():
        out["by_kind"][kind(t)] = out["by_kind"].get(kind(t), 0) + len(v)
    if plan.get("failsafe") or plan.get("failed"):
        out["failsafe"] = plan.get("failsafe") or plan.get("failed")
    return out


def selfplay(stages: str, plan_segments: dict) -> dict:
    """Finished segments per kind: games, learner W-D-L, mean k and think scale (game-weighted means)."""
    kinds: dict = {}
    for path in sorted(glob.glob(os.path.join(stages, "seg*.done"))):
        m = read_json(path)
        if not isinstance(m, dict):
            continue
        name = os.path.basename(path)[:-5]
        lg = m.get("league") or {}
        theta = first(lg, "theta") or (plan_segments.get(name) or {}).get("theta") or m.get("theta") or "plain"
        k = kinds.setdefault(kind(theta), {"segments": 0, "games": 0, "wins": 0, "draws": 0, "losses": 0,
                                           "_k": [0.0, 0], "_s": [0.0, 0]})
        games = int(m.get("games") or 0)
        k["segments"] += 1
        k["games"] += games
        for key, alt in (("wins", "w"), ("draws", "d"), ("losses", "l")):
            k[key] += int(first(lg, key, alt, "learner_" + key) or 0)
        mean_k = first(lg, "mean_k_learner", "mean_k", "k_mean", "k")
        if mean_k is not None:
            k["_k"][0] += float(mean_k) * max(games, 1)
            k["_k"][1] += max(games, 1)
        scale = first(lg, "mean_s", "think_scale", "mean_think_scale", "scale")
        if scale is not None:
            k["_s"][0] += float(scale) * max(games, 1)
            k["_s"][1] += max(games, 1)
    for k in kinds.values():
        sk, ss = k.pop("_k"), k.pop("_s")
        k["mean_k"] = sk[0] / sk[1] if sk[1] else None
        k["think_scale"] = ss[0] / ss[1] if ss[1] else None
        n = k["wins"] + k["draws"] + k["losses"]
        k["learner_score"] = (k["wins"] + 0.5 * k["draws"]) / n if n else None
    return kinds


def net_status(root: str, sha16: str | None) -> dict | None:
    if not sha16:
        return None
    d = os.path.join(root, "gate2", "nets", sha16)
    if not os.path.isdir(d):
        return {"sha16": sha16, "dir": None}
    cfg = read_json(os.path.join(d, "config.json")) or {}
    out = {"sha16": sha16, "dir": d, "label": cfg.get("label"), "done": os.path.exists(os.path.join(d, "done")),
           "arms": {}}
    for arm in ARMS:
        s = read_json(os.path.join(d, arm, "summary.json"))
        parts = sorted(glob.glob(os.path.join(d, arm, "part*")))
        if s is None:
            out["arms"][arm] = {"parts": len(parts), "merged": False}
        else:
            out["arms"][arm] = {"merged": True, **{k: s.get(k) for k in (
                "games", "pairs", "score", "score_ci", "elo", "turns_ahead", "think_head_missing", "gave_up")}}
    return out


def provisional_decision(cand: str, inc: str) -> dict | None:
    """hivemind.gate2 decide on the two net dirs (the wave2-hivemind env), or None when it cannot run."""
    py = os.path.join(W2, ".venv", "bin", "python")
    if not os.path.exists(py):
        return None
    try:
        r = subprocess.run([py, "-m", "hivemind.gate2", "decide", cand, inc], cwd=W2, capture_output=True,
                           text=True, timeout=60)
        return json.loads(r.stdout) if r.returncode == 0 else None
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def iteration(root: str, it_dir: str, decide: bool = True) -> dict:
    stages = os.path.join(it_dir, "stages")
    n = int(re.sub(r"\D", "", os.path.basename(it_dir)) or 0)
    present = sorted(os.path.basename(p)[:-5] for p in glob.glob(os.path.join(stages, "*.done")))
    plain = [s for s in present if not re.fullmatch(r"seg\d+", s)]
    out: dict = {"iteration": n, "stages": plain, "segments_done": len(present) - len(plain)}
    plan = league_plan(stages)
    out["league_plan"] = plan
    segs = (read_json(os.path.join(stages, "league_plan.done")) or {}).get("segments") or {}
    out["selfplay"] = selfplay(stages, segs)
    labels = read_json(os.path.join(stages, "think_labels.done"))
    out["think_labels"] = labels
    train = read_json(os.path.join(stages, "train.done")) or {}
    out["train_steps"] = train.get("steps")
    out["gate_100ms"] = (read_json(os.path.join(stages, "match_generator.done")) or {}).get("score")
    fixed = [p for p in glob.glob(os.path.join(stages, "match_*.done")) if not p.endswith("match_generator.done")]
    out["vs_it13_100ms"] = (read_json(fixed[0]) or {}).get("score") if fixed else None
    promote = read_json(os.path.join(stages, "promote.done"))
    out["promoted"] = promote.get("promoted") if isinstance(promote, dict) else None
    # gate2: candidate and incumbent nets, the decision
    try:
        gen = open(os.path.join(stages, "generator")).read().strip()
    except OSError:
        gen = ""
    inc16 = gen[:16] if re.fullmatch(r"[0-9a-f]{64}", gen) else None
    cand_sha = sha256(os.path.join(it_dir, "models", "C.onnx"))
    g2 = read_json(os.path.join(stages, "gate2.done"))
    cand16 = cand_sha[:16] if cand_sha else ((g2 or {}).get("candidate") or {}).get("sha16")
    out["gate2"] = {"candidate": net_status(root, cand16), "incumbent": net_status(root, inc16)}
    if isinstance(g2, dict):
        out["gate2"]["decision"] = {"final": True, "promoted": g2.get("promoted"), "reason": g2.get("reason"),
                                    "arms": g2.get("arms")}
    elif decide and cand16 and inc16:
        cd, idir = (os.path.join(root, "gate2", "nets", s) for s in (cand16, inc16))
        if glob.glob(os.path.join(cd, "*", "pairs.tsv")) and glob.glob(os.path.join(idir, "*", "pairs.tsv")):
            d = provisional_decision(cd, idir)
            if d:
                out["gate2"]["decision"] = {"final": False, "promoted": d.get("promoted"), "reason": d.get("reason"),
                                            "arms": d.get("arms")}
    return out


# ----------------------------------------------------------------------------------------------- GPU-hours
def cap_since(root: str) -> str:
    """The run's first watchdog line (the watchdog counts its cap from there), else launch_settings.txt's mtime."""
    try:
        for line in open(os.path.join(root, "watchdog.log")):
            if " watching full_" in line:
                return line[:19].replace(" ", "T")
    except OSError:
        pass
    import time
    path = os.path.join(root, "launch_settings.txt")
    t = os.path.getmtime(path) if os.path.exists(path) else time.time() - 30 * 86400
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(t))


def counted(name: str, run: str) -> str | None:
    """The cap group of a job name of this run, or None (spec v7 7.3, cluster/full_run_watchdog.sh)."""
    if name in (f"full_{run}", f"sp_{run}"):
        return name.split("_", 1)[0]
    if name in (f"tg_{run}", f"lb_{run}"):
        return name.split("_", 1)[0]
    for prefix in ("g2", "mon", "am"):
        tail = "_" if prefix == "g2" else "_it"
        if name.startswith(f"{prefix}_{run}{tail}"):
            return prefix
    return None


def gpu_hours(root: str, run: str) -> dict:
    if shutil.which("sacct") is None:
        return {"used": None, "note": "n/a off-cluster"}
    since = cap_since(root)
    try:
        r = subprocess.run(["sacct", "-n", "-X", "-P", "-u", os.environ.get("USER", ""), "-S", since,
                            "-o", "JobName,ElapsedRaw,AllocTRES"], capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError) as err:
        return {"used": None, "note": f"sacct failed: {err}"}
    if r.returncode != 0:
        return {"used": None, "note": f"sacct failed: {r.stderr.strip()[:200]}"}
    by: dict = {}
    for line in r.stdout.splitlines():
        cols = line.split("|")
        if len(cols) < 3:
            continue
        group = counted(cols[0], run)
        m = re.search(r"gres/gpu=(\d+)", cols[2])
        if group and m and cols[1].isdigit():
            by[group] = by.get(group, 0.0) + int(cols[1]) * int(m.group(1)) / 3600
    return {"used": round(sum(by.values()), 1), "by_kind": {k: round(v, 1) for k, v in sorted(by.items())},
            "since": since}


def status(root: str, decide: bool = True) -> dict:
    root = root.rstrip("/")
    run = os.path.basename(root)
    st = settings(root)
    its = sorted(glob.glob(os.path.join(root, "runs", "rl-it*")),
                 key=lambda p: int(re.sub(r"\D", "", os.path.basename(p)) or 0))
    gh = gpu_hours(root, run)
    cap = st.get("gpu_hours")
    gh["cap"] = float(cap) if cap else None
    jobs = {}
    try:
        for line in open(os.path.join(root, "pilot_jobs.tsv")):
            k, _, v = line.strip().partition("\t")
            jobs[k] = v
    except OSError:
        pass
    return {"run": run, "root": root, "exists": os.path.isdir(root), "settings": st, "pilot_jobs": jobs,
            "stopped": os.path.exists(os.path.join(root, "STOP_WORKERS")),
            "iterations": [iteration(root, d, decide) for d in its if os.path.isdir(d)], "gpu_hours": gh}


# ----------------------------------------------------------------------------------------------- text
def f3(x) -> str:
    return "-" if x is None else f"{x:.3f}" if isinstance(x, float) else str(x)


def format_text(st: dict) -> str:
    out = [f"league pilot {st['root']}"]
    if not st["exists"]:
        return out[0] + ": no such run folder"
    s = st["settings"]
    if s:
        out.append(f"  settings: iters {s.get('iters', '-')}, games {s.get('games_per_iter', '-')}, workers "
                   f"{s.get('workers', '-')}, engine {s.get('engine', '-')[:12]}, extra {s.get('wave2_extra', '-')}")
    if st["pilot_jobs"]:
        out.append("  pilot jobs: " + ", ".join(f"{k} {v}" for k, v in st["pilot_jobs"].items()))
    gh = st["gpu_hours"]
    used = gh.get("used")
    cap = gh.get("cap")
    line = f"  GPU-hours: {gh['note']}" if used is None else f"  GPU-hours: {used} of {cap if cap else 'no cap'}"
    if gh.get("by_kind"):
        line += " (" + ", ".join(f"{k} {v}" for k, v in gh["by_kind"].items()) + f"; since {gh['since']})"
    out.append(line + ("   STOPPED at the cap (STOP_WORKERS)" if st["stopped"] else ""))
    if not st["iterations"]:
        out.append("  no iterations yet (runs/rl-it*)")
    for it in st["iterations"]:
        out.append(f"it{it['iteration']}: stages {' '.join(it['stages']) or '-'}; {it['segments_done']} segments done")
        p = it["league_plan"]
        if p:
            out.append(f"  league plan: {p['segments']} segments, by kind "
                       + ", ".join(f"{k} {v}" for k, v in p["by_kind"].items())
                       + "; by theta " + ", ".join(f"{k} {v}" for k, v in sorted(p["by_theta"].items()))
                       + (f"; PFSP {json.dumps(p['pfsp'])}" if p.get("pfsp") else "")
                       + (f"; FAILSAFE {p['failsafe']}" if p.get("failsafe") else ""))
        else:
            out.append("  league plan: none")
        for k, v in sorted(it["selfplay"].items()):
            out.append(f"  self-play {k}: {v['segments']} segments, {v['games']} games, learner "
                       f"{v['wins']}-{v['draws']}-{v['losses']} (score {f3(v['learner_score'])}), mean k "
                       f"{f3(v['mean_k'])}, think scale {f3(v['think_scale'])}")
        lab = it["think_labels"]
        if isinstance(lab, dict) and lab.get("failed"):
            out.append(f"  think labels: FAILED ({lab['failed']}); trained without labels")
        elif isinstance(lab, dict):
            out.append(f"  think labels: {lab.get('labelled', '-')} of {lab.get('records', '-')} records, rates "
                       f"{f3(lab.get('rate_a'))}/{f3(lab.get('rate_b'))}, noise {f3(lab.get('noise_rate'))}, "
                       f"{f3(lab.get('gpu_s'))} GPU-s")
        out.append(f"  train steps {f3(it['train_steps'])}; 100 ms gate {f3(it['gate_100ms'])}, vs it13 "
                   f"{f3(it['vs_it13_100ms'])}; promoted {f3(it['promoted'])}")
        g = it["gate2"]
        for role in ("candidate", "incumbent"):
            net = g.get(role)
            if not net:
                out.append(f"  gate2 {role}: not known yet")
                continue
            if not net.get("dir"):
                out.append(f"  gate2 {role} {net['sha16']}: no net dir yet")
                continue
            arms = []
            for arm, a in net["arms"].items():
                if a.get("merged"):
                    ci = a.get("score_ci") or [None, None]
                    arms.append(f"{arm} {f3(a.get('score'))} [{f3(ci[0])}, {f3(ci[1])}] {a.get('pairs')} pairs, "
                                f"ahead {f3(a.get('turns_ahead'))}")
                else:
                    arms.append(f"{arm} running ({a['parts']} parts)")
            out.append(f"  gate2 {role} {net['sha16']}{' done' if net['done'] else ''}: " + "; ".join(arms))
        dec = g.get("decision")
        if dec:
            arms = dec.get("arms") or {}
            each = "; ".join(f"{arm} d {f3(a.get('d'))} SE {f3(a.get('se'))} pairs {a.get('pairs')}"
                             f"{' DROP' if a.get('drop') else ''}" for arm, a in arms.items())
            out.append(f"  gate2 decision ({'final' if dec['final'] else 'provisional'}): promoted "
                       f"{dec.get('promoted')}: {dec.get('reason')}" + (f" [{each}]" if each else ""))
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_root")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--no-decide", action="store_true", help="skip the provisional hivemind.gate2 decide")
    args = ap.parse_args(argv)
    st = status(args.run_root, decide=not args.no_decide)
    print(json.dumps(st, indent=1) if args.json else format_text(st))
    return 0


if __name__ == "__main__":
    sys.exit(main())
