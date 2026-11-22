#!/usr/bin/env python3
"""Per-iteration KPI records for an rl-loop run (cluster/README.md, "Concurrent self-play, budgets and the final match").
Every iteration runs/rl-itN of a run gets metrics/itN.json: the gate and anchor results (Elo vs the fixed it13 anchor
is the target), GPU cost, self-play statistics, the training curve, search speed, the network probe
(metrics/itN.probe.json from cluster/probe_net.py, merged when present), what changed since the previous iteration,
alerts and, on a run with the search-balance controller (cluster/search_balance.py, rl-loop --search-balance), its
balance/itN.json merged as "balance". A KPI that cannot be measured from what is on disk is null and listed under "unavailable" with the reason.
    .venv/bin/python cluster/run_metrics.py runs/full/hybrid1               # write metrics/itN.json
    .venv/bin/python cluster/run_metrics.py runs/full/hybrid1 --summary     # + metrics/summary.json, JSON on stdout
    .venv/bin/python cluster/run_metrics.py runs/full/hybrid1 --alerts      # + append new alerts to alerts.tsv
Idempotent: records are rewritten, finished iterations are cached by their files' mtimes (metrics/.cache.json), so
--summary (the dev panel calls it over ssh) only re-reads the running iteration. Stdlib only (the repo .venv).
"""

import argparse
import contextlib
import datetime
import fcntl
import glob
import json
import math
import os
import random
import re
import shutil
import struct
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from training_curve import blocks, games as game_scores, load  # noqa: E402  (same gate parsing as the panel's curve)
try:  # the search-balance controller (stdlib only); its absence only drops the balance section's derived fields
    import search_balance as sb  # noqa: E402
except Exception:  # noqa: BLE001
    sb = None

SCHEMA = 1
BLOCK = 100                # gate games per block, as training_curve.py
SAMPLE_POSITIONS = 20000   # self-play positions read per iteration (chunk stats)
SAMPLE_PGN_GAMES = 2000    # self-play games read per iteration (drop rate)
# Alert thresholds. Rates are fractions: 0.03 is 3 points.
FLAG_JUMP = 0.03           # flag_rate_jump: flag rate up more than this vs the previous iteration
ENTROPY_DROP = 0.25        # entropy_collapse: policy target entropy below (1 - this) x it1's
GATE_LOW = 0.45            # gate_low: score vs it13 below this ...
GATE_LOW_GAMES = 200       # ... over at least this many games
FLAT_ELO = 10.0            # elo_per_gpu_flat: Elo vs it13 gained less than this ...
FLAT_WINDOW = 3            # ... over this many finished iterations (level "stop": a recommendation, never acted on)
ANCHOR_DROP = 0.05         # anchor_drop: an anchor score down more than this vs its previous reading
CACHE_VERSION = 3          # 3: anchors/ read and in the signature

TS = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) (.*)$")
IT = re.compile(r"\bit(\d+)\b")
LABELS = re.compile(r"^it(\d+) (seg\d+|val) labels: (\d+) games, (\d+) records, (\d+) flag games, "
                    r"(\d+) repeated positions, terminations (\{.*\})")
TRAINING = re.compile(r"^it(\d+) training: (\d+) steps from (\S+), (\d+) new positions x([\d.]+), (\d+) replay "
                      r"positions x([\d.]+) \(replay share ([\d.]+), (\d+) dirs, batch (\d+)\)")
# selfplay.cc: "selfplay game N/M raw R samples S events E termination T [lead ..] [clocks ..] [hybrid P/T/S ..]"
GAME = re.compile(r"^selfplay game (\d+)/\d+ .*?\bsamples (\d+) events (\d+) termination (.+?)"
                  r"(?= (?:lead|resign-check|flag-seat|clocks|hybrid) |$)")
HYBRID = re.compile(r"\bhybrid (\d+)/(\d+)/(\d+)\b")
# tournament.cc: "tournament hybrid game N/M contender: P sits in T triggered of S searches, ..."
MATCH_HYBRID = re.compile(r"^tournament hybrid game (\d+)/\d+ contender: (\d+) sits in (\d+) triggered of (\d+) searches")
PGN_MOVE = re.compile(r"\d+[ABab]\. (\S+)")
LR = re.compile(r"^learning rates: trunk (\S+), heads ([^,\s]+)")


def now():
    return datetime.datetime.now().replace(microsecond=0)


def stamp(text):
    try:
        return datetime.datetime.strptime(text, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None


def read(path):
    try:
        with open(path, errors="replace") as f:
            return f.read()
    except OSError:
        return None


def write_json(path, value):
    """Atomically, so the panel never reads half a file. A unique temporary name: two collectors (monitor jobs, the
    watchdog, the panel's ssh call) may write the same file at once, and a shared '.tmp' is moved away under one."""
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=directory, prefix=os.path.basename(path) + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(value, f, indent=1)
            f.write("\n")
        os.chmod(temporary, 0o644)  # mkstemp makes it 0600; the records are read by the panel and the group
        os.replace(temporary, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.remove(temporary)
        raise


@contextlib.contextmanager
def locked(metrics_dir):
    """Serialize collectors of one run (metrics/.lock), so the alerts.tsv dedup check and append stay atomic. A
    filesystem without flock runs unlocked: the unique temporary files above still keep every write whole."""
    os.makedirs(metrics_dir, exist_ok=True)
    with open(os.path.join(metrics_dir, ".lock"), "a") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX)
        except OSError:
            pass
        yield


def rnd(x, digits=6):
    return None if x is None else round(x, digits) + 0.0  # + 0.0: no "-0.0" in the JSON


def quantile(values, q):
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def elo(score):
    score = min(max(score, 0.001), 0.999)
    return -400 * math.log10(1 / score - 1)


def wilson_elo(score, n, z=1.96):
    """Elo CI from the score's Wilson interval over games (the engine's 'game-level Wilson approximation')."""
    if not n:
        return None
    centre = (score + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(score * (1 - score) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(elo(centre - half), 6), round(elo(centre + half), 6)]


# ---------------------------------------------------------------------------------------------------- rl-loop.log


def parse_log(root):
    """Per iteration: first/last/finished timestamps, label lines per segment, the training line; and the loop
    settings lines in order (a relaunch prints a new one)."""
    out = {"first": {}, "last": {}, "finished": {}, "labels": {}, "training": {}, "loops": [], "start": None}
    text = read(os.path.join(root, "runs", "rl-loop.log")) or ""
    for line in text.splitlines():
        m = TS.match(line)
        if not m:
            continue
        ts, body = stamp(m[1]), m[2]
        if ts is None:
            continue
        out["start"] = out["start"] or ts
        if body.startswith("loop: iterations "):
            out["loops"].append((ts, body[len("loop: "):].strip()))
            continue
        i = IT.search(body)
        if not i:
            continue
        n = int(i[1])
        out["first"].setdefault(n, ts)
        out["last"][n] = ts
        if body.startswith(f"it{n} finished"):
            out["finished"][n] = ts
        lab = LABELS.match(body)
        if lab:
            try:
                terms = json.loads(lab[7])
            except ValueError:
                terms = {}
            out["labels"][(n, lab[2])] = {"games": int(lab[3]), "records": int(lab[4]), "flag_games": int(lab[5]),
                                          "repeat_records": int(lab[6]), "terminations": terms}
        tr = TRAINING.match(body)
        if tr:
            out["training"][n] = {"steps": int(tr[2]), "init": tr[3], "new_positions": int(tr[4]),
                                  "new_passes": float(tr[5]), "replay_positions": int(tr[6]),
                                  "replay_passes": float(tr[7]), "replay_fraction": float(tr[8]),
                                  "replay_dirs": int(tr[9]), "batch": int(tr[10])}
    return out


def windows(log, iterations):
    """[start, end) of each iteration for apportioning GPU time: from the previous iteration's end (the run's
    first log line for the first) to this one's 'finished' line, or now while it runs. Gaps between iterations
    (a requeue) go to the iteration that follows them."""
    out, previous = {}, log["start"]
    for n in iterations:
        start = previous or log["first"].get(n)
        end = log["finished"].get(n)
        if start is None:
            continue
        out[n] = (start, end or now())
        previous = end or out[n][1]
    return out


# ---------------------------------------------------------------------------------------------------- matches


def match_record(it_dir, name):
    """M for a gate or anchor match: the stage marker when the match finished, else the games so far."""
    stages, arms = os.path.join(it_dir, "stages"), os.path.join(it_dir, "arms")
    marker = load(os.path.join(stages, f"match_{name}.done"))
    source = "generator" if marker and marker.get("same_as") == "generator" else name
    scores = game_scores(os.path.join(arms, f"match_{source}", "games.tsv"))
    summary = load(os.path.join(arms, f"match_{source}", "summary.json")) or {}
    if not scores and not marker:
        return None
    if marker:
        wins, losses, draws = marker.get("wins"), marker.get("losses"), marker.get("draws")
        score, rating, ci = marker.get("score"), marker.get("elo"), marker.get("ci")
    else:
        wins, losses = sum(s == 1 for s in scores), sum(s == 0 for s in scores)
        draws = len(scores) - wins - losses
        score = sum(scores) / len(scores)
        rating, ci = round(elo(score), 6), wilson_elo(score, len(scores))
    n = sum(v for v in (wins, losses, draws) if isinstance(v, int)) or len(scores)
    time_model = (marker or {}).get("time_model") or summary.get("time_model")
    return {"games": n, "score": score, "elo": rating, "ci": ci, "wins": wins, "losses": losses, "draws": draws,
            "movetime": (marker or {}).get("movetime", summary.get("move_time_ms")),
            "clock": (summary.get("clock") or {"time_model": time_model}) if time_model == "clock" else None,
            "blocks": blocks(scores, BLOCK)}


def match_names(it_dir):
    names = set()
    for directory, prefix, suffix in (("stages", "match_", ".done"), ("arms", "match_", "")):
        try:
            entries = os.listdir(os.path.join(it_dir, directory))
        except OSError:
            continue
        for e in entries:
            if e.startswith(prefix) and e.endswith(suffix) and not e.endswith(".log") and not e.endswith(".tmp"):
                if directory == "arms" and not os.path.isdir(os.path.join(it_dir, directory, e)):
                    continue
                names.add(e[len(prefix):len(e) - len(suffix)] if suffix else e[len(prefix):])
    return sorted(names)


def match_search(it_dir, unavailable):
    """Search speed and time use of the contender in the generator gate (the iteration's own network)."""
    stages, arms = os.path.join(it_dir, "stages"), os.path.join(it_dir, "arms")
    marker = load(os.path.join(stages, "match_generator.done")) or {}
    summary = load(os.path.join(arms, "match_generator", "summary.json")) or {}
    nps = marker.get("nps")
    if nps is None and summary.get("performance"):
        perf = summary["performance"]
        nps = [perf.get("contender", {}).get("nps"), perf.get("baseline", {}).get("nps")]
    if nps is None:
        unavailable["search.nps"] = "the generator gate has not finished (no summary.json)"
    unavailable["search.depth"] = "the engine logs no search depth in matches or self-play"
    # One log for a fixed-length gate; with --gate-sprt each chunk of games logs to match_generator/chunkNN.log and
    # numbers its games from 1 again, so games are keyed per log.
    logs = [os.path.join(arms, "match_generator.log")]
    if not os.path.isfile(logs[0]):
        logs = sorted(glob.glob(os.path.join(arms, "match_generator", "chunk*.log")))
    sits = {}
    for i, path in enumerate(logs):
        for line in (read(path) or "").splitlines():
            m = MATCH_HYBRID.match(line)
            if m:
                sits[i, int(m[1])] = (int(m[2]), int(m[4]))  # a resumed match prints a game again: the last counts
    searches = sum(s for _, s in sits.values())
    sit_gate = rnd(sum(p for p, _ in sits.values()) / searches) if searches else None
    if sit_gate is None:
        unavailable["search.sit_gate_rate"] = ("no 'tournament hybrid game' lines in arms/match_generator.log or "
                                              "arms/match_generator/chunk*.log (hybrid off, or the gate has not "
                                              "started)")
    # games.tsv: index, outcome, contender_white, lead, termination, then searches/nodes/ns for each side.
    c_ns = c_s = b_ns = b_s = 0
    try:
        with open(os.path.join(arms, "match_generator", "games.tsv")) as f:
            next(f, None)
            for line in f:
                cols = line.rstrip("\n").split("\t")
                if len(cols) >= 11 and cols[5].isdigit():
                    c_s, c_ns = c_s + int(cols[5]), c_ns + int(cols[7])
                    b_s, b_ns = b_s + int(cols[8]), b_ns + int(cols[10])
    except (OSError, ValueError):
        pass
    time_use = ({"contender_ms_per_search": round(c_ns / c_s / 1e6, 3) if c_s else None,
                 "baseline_ms_per_search": round(b_ns / b_s / 1e6, 3) if b_s else None,
                 "movetime": marker.get("movetime", summary.get("move_time_ms"))} if c_s or b_s else None)
    if time_use is None:
        unavailable["search.time_use"] = "no games in arms/match_generator/games.tsv yet"
    return {"nps": nps, "depth": None, "sit_gate_rate": sit_gate, "time_use": time_use}


# ---------------------------------------------------------------------------------------------------- self-play data


def read_dst(path):
    """The fields the KPIs need from an HDST chunk (hivemind/distill/data.py read_chunk, stdlib struct instead of
    numpy): per position the pocket plane values, the WDL target, and each board's policy target."""
    with open(path, "rb") as f:
        raw = f.read()
    if raw[:4] != b"HDST":
        raise ValueError("not an HDST chunk")
    version, planes = struct.unpack_from("<II", raw, 4)
    n, entries = struct.unpack_from("<QQ", raw, 12)
    if version not in (2, 3) or planes not in (74, 78, 130):
        raise ValueError(f"version {version}, {planes} planes")
    offset = 28 + 8 * n * planes                                # plane bits
    values = struct.unpack_from(f"<{n * planes}e", raw, offset)
    offset += 2 * n * planes + 4 * n                            # plane values, value
    wdl = struct.unpack_from(f"<{3 * n}f", raw, offset)
    offset += 12 * n + 2 * n + 2 * n + n                        # wdl, moves_left, ply, outcome
    counts = struct.unpack_from(f"<{2 * n}H", raw, offset)
    offset += 4 * n
    index = struct.unpack_from(f"<{entries}H", raw, offset)
    offset += 2 * entries
    prob = struct.unpack_from(f"<{entries}e", raw, offset)
    if sum(counts) != entries:
        raise ValueError("inconsistent policy counts")
    return {"n": n, "planes": planes, "values": values, "wdl": wdl, "counts": counts, "index": index, "prob": prob}


def chunk_header_positions(path):
    try:
        with open(path, "rb") as f:
            head = f.read(28)
        return struct.unpack_from("<Q", head, 12)[0] if head[:4] == b"HDST" else 0
    except (OSError, struct.error):
        return 0


def chunk_stats(paths, seed):
    """Value-target balance, policy-target entropy, pass share and pocket sizes over at most SAMPLE_POSITIONS
    positions, whole chunks picked at random (seeded, so a rerun reads the same ones)."""
    paths = sorted(paths)
    random.Random(seed).shuffle(paths)
    wdl_counts = [0, 0, 0]
    unlabelled = positions = 0
    entropies, pockets = [], []
    passes = targets = 0
    for path in paths:
        if positions >= SAMPLE_POSITIONS:
            break
        try:
            c = read_dst(path)
        except (OSError, ValueError, struct.error):
            continue
        n, planes, block = c["n"], c["planes"], c["planes"] // 2
        positions += n
        values, wdl = c["values"], c["wdl"]
        for r in range(n):
            w = wdl[3 * r:3 * r + 3]
            if sum(w) > 0:
                wdl_counts[max(range(3), key=w.__getitem__)] += 1
            else:
                unlabelled += 1
            # Pocket planes 12..21 of each board block hold count / 16 (board2planes.py, MAX_NUM_DROPS).
            base = r * planes
            pockets.append(16 * sum(values[base + b * block + p] for b in (0, 1) for p in range(12, 22)))
        start = 0
        for count in c["counts"]:
            if count:
                ps = [p for p in c["prob"][start:start + count] if p > 0]
                total = sum(ps)
                if total > 0:
                    entropies.append(-sum(p / total * math.log(p / total) for p in ps))
                # distill_data.cc policy_actions: legal moves, then the pass (policy index 0) last.
                if c["index"][start + count - 1] == 0:
                    targets += 1
                    segment = c["prob"][start:start + count]
                    passes += segment[-1] >= max(segment) and segment[-1] > 0
            start += count
    labelled = sum(wdl_counts)
    return {"positions_sampled": positions,
            "value_target": {"win": rnd(wdl_counts[2] / labelled), "draw": rnd(wdl_counts[1] / labelled),
                             "loss": rnd(wdl_counts[0] / labelled)} if labelled else None,
            "unlabelled": unlabelled,
            "policy_target_entropy": {"mean": rnd(sum(entropies) / len(entropies)),
                                      "p10": rnd(quantile(entropies, 0.1))} if entropies else None,
            "pass_rate": rnd(passes / targets) if targets else None,
            "pocket_mean": rnd(sum(pockets) / len(pockets)) if pockets else None}


def log_stats(logs):
    """Game lengths (plies over both boards: the PGN's events) and the hybrid's sit / trigger counts per search,
    from the self-play engine logs."""
    lengths, sits, triggered, searches = [], 0, 0, 0
    for path in logs:
        games = {}
        for line in (read(path) or "").splitlines():
            m = GAME.match(line.strip())
            if m:
                h = HYBRID.search(line)
                games[int(m[1])] = (int(m[3]), tuple(int(v) for v in h.groups()) if h else None)
        for events, hybrid in games.values():  # a resumed segment prints a game again: the last line counts
            lengths.append(events)
            if hybrid:
                sits, triggered, searches = sits + hybrid[0], triggered + hybrid[1], searches + hybrid[2]
    return {"game_length": {"mean": rnd(sum(lengths) / len(lengths), 2), "p50": quantile(lengths, 0.5),
                            "p90": quantile(lengths, 0.9)} if lengths else None,
            "sit_rate": rnd(sits / searches) if searches else None,
            "hybrid_wait_rate": rnd(triggered / searches) if searches else None}


def pgn_drop_rate(pgns):
    """Share of played moves that are drops, over at most SAMPLE_PGN_GAMES games split across the segments."""
    if not pgns:
        return None
    quota = max(1, SAMPLE_PGN_GAMES // len(pgns))
    moves = drops = 0
    for path in pgns:
        seen = 0
        try:
            with open(path, errors="replace") as f:
                for line in f:
                    if line.startswith("[Event "):
                        seen += 1
                        if seen > quota:
                            break
                    elif line and line[0].isdigit():
                        for move in PGN_MOVE.findall(line):
                            moves += 1
                            drops += "@" in move
        except OSError:
            continue
    return rnd(drops / moves) if moves else None


CHUNK_KEYS = ("pocket_mean", "value_target", "policy_target_entropy", "pass_rate")
DATA_KEYS = ("game_length", "sit_rate", "hybrid_wait_rate", "drop_rate", "pocket_mean", "value_target",
             "policy_target_entropy", "pass_rate")


def selfplay(it_dir, n, log, cache, unavailable):
    stages = os.path.join(it_dir, "stages")
    try:
        segs = sorted(f[:-5] for f in os.listdir(stages) if f.startswith("seg") and f.endswith(".done"))
    except OSError:
        segs = []
    markers = {s: load(os.path.join(stages, s + ".done")) or {} for s in segs}
    games = positions = flag = repeat = records = 0
    terminations = {}
    for s in segs:
        m = markers[s]
        lab = m.get("labels") or log["labels"].get((n, s)) or {}
        games += m.get("games", lab.get("games", 0))
        positions += m.get("positions", lab.get("records", 0))
        flag += lab.get("flag_games", 0)
        repeat += lab.get("repeat_records", 0)
        records += lab.get("records", 0)
        for k, v in (lab.get("terminations") or {}).items():
            terminations[k] = terminations.get(k, 0) + v
    out = {"games": games, "positions": positions, "segments": len(segs), "terminations": terminations,
           "flag_rate": rnd(flag / games) if games else None, "repeat_rate": rnd(repeat / records) if records else None}
    if not games:
        unavailable["selfplay.flag_rate"] = "no finished self-play segment"
        unavailable["selfplay.repeat_rate"] = "no finished self-play segment"
    # Chunk stats read only finished segments (a running one's chunks are still in its checkpoint); the engine logs
    # and PGNs of running segments count too, so a live iteration shows its game lengths and hybrid rates. Cached
    # by the finished set and the logs' sizes, and kept once measured: rl-loop may archive the self-play later.
    search = os.path.join(it_dir, "search")
    try:
        played = sorted(f[:-4] for f in os.listdir(os.path.join(search, "logs"))
                        if f.startswith("seg") and f.endswith(".log"))
    except OSError:
        played = []
    sizes = []
    for s in played:
        try:
            sizes.append([s, os.path.getsize(os.path.join(search, "logs", s + ".log"))])
        except OSError:
            pass
    key = [segs, sizes]
    cached = cache.get(str(n))
    if cached and (cached.get("key") == key or not os.path.isdir(os.path.join(it_dir, "search", "train"))):
        stats = cached["stats"]
    else:
        chunks = []
        for s in segs:
            d = os.path.join(search, "train", s)
            try:
                chunks += [os.path.join(d, f) for f in os.listdir(d) if f.endswith(".dst")]
            except OSError:
                pass
        stats = {}
        stats.update(chunk_stats(chunks, seed=n) if chunks else {})
        stats.update(log_stats([os.path.join(search, "logs", s + ".log") for s in played]))
        stats["drop_rate"] = pgn_drop_rate([p for p in (os.path.join(search, "train_run", s, "games.pgn")
                                                        for s in played) if os.path.isfile(p)])
        stats["chunks"] = len(chunks)
        previous = (cached or {}).get("stats") or {}
        for k in DATA_KEYS:  # never lose a measurement to an archived folder
            if stats.get(k) is None and previous.get(k) is not None:
                stats[k] = previous[k]
        if played:
            cache[str(n)] = {"key": key, "stats": stats}
    reasons = {
        "game_length": "no 'selfplay game' lines in search/logs/segNN.log",
        "sit_rate": "no hybrid counts in the self-play logs (hybrid off, or logs archived)",
        "hybrid_wait_rate": "no hybrid counts in the self-play logs (hybrid off, or logs archived)",
        "drop_rate": "no games in search/train_run/segNN/games.pgn",
        "pocket_mean": "no readable search/train/segNN/*.dst chunks for the finished segments",
        "value_target": "no labelled positions in the sampled chunks",
        "policy_target_entropy": "no policy targets in the sampled chunks",
        "pass_rate": "no policy targets ending in the pass in the sampled chunks",
    }
    for k in DATA_KEYS:
        out[k] = stats.get(k)
        if out[k] is None:
            unavailable[f"selfplay.{k}"] = ("no finished self-play segment" if not segs and k in CHUNK_KEYS
                                            else "no self-play played yet" if not played else reasons[k])
    return out


# ---------------------------------------------------------------------------------------------------- training


def training(it_dir, n, log, unavailable):
    text = read(os.path.join(it_dir, "arms", "C.log")) or ""
    curve, lr, trunk_lr = [], None, None
    for line in text.splitlines():
        if line.startswith("{"):
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if isinstance(row, dict):
                curve.append(row)
        m = LR.match(line)
        if m:
            trunk_lr, lr = float(m[1]), float(m[2])
    done = load(os.path.join(it_dir, "stages", "train.done")) or {}
    line = log["training"].get(n, {})
    get = lambda k: done.get(k, line.get(k))  # noqa: E731
    grad = curve[-1].get("grad_norm") if curve else None
    out = {"steps": get("steps"), "lr": lr, "trunk_lr": trunk_lr, "replay_fraction": get("replay_fraction"),
           "new_positions": get("new_positions"), "replay_positions": get("replay_positions"),
           "curve": curve, "final": curve[-1] if curve else None, "grad_norm": grad}
    if lr is None:
        unavailable["training.lr"] = "no 'learning rates:' line in arms/C.log (training not started, or no --trunk-lr)"
        unavailable["training.trunk_lr"] = unavailable["training.lr"]
    if not curve:
        unavailable["training.curve"] = "no JSON metric lines in arms/C.log yet"
    if grad is None:
        unavailable["training.grad_norm"] = "the trainer clips gradients at 5.0 but does not log their norm"
    for k in ("steps", "replay_fraction", "new_positions", "replay_positions"):
        if out[k] is None:
            unavailable[f"training.{k}"] = "no train.done and no 'training:' line in rl-loop.log yet"
    return out


# ---------------------------------------------------------------------------------------------------- cost


def job_ids(root):
    ids = []
    for line in (read(os.path.join(root, "jobs.txt")) or "").split():
        if line.strip().isdigit():
            ids.append(line.strip())
    for line in (read(os.path.join(root, "workers.tsv")) or "").splitlines():
        cols = line.split("\t")
        if len(cols) >= 3 and cols[2].strip().isdigit():
            ids.append(cols[2].strip())
    return sorted(set(ids), key=ids.index)


def sacct_jobs(ids):
    """[(start, end, gpus)] of the jobs that ran (running ones until now), or (None, reason)."""
    if not ids:
        return None, "no job ids in jobs.txt or workers.tsv"
    if not shutil.which("sacct"):
        return None, "sacct not on PATH (not on the cluster)"
    try:
        result = subprocess.run(["sacct", "-n", "-X", "-P", "-j", ",".join(ids), "-o", "JobID,Start,End,AllocTRES"],
                                capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError) as error:
        return None, f"sacct failed: {error}"
    if result.returncode:
        return None, f"sacct failed: {result.stderr.strip()[:200]}"
    jobs = []
    for line in result.stdout.splitlines():
        cols = line.split("|")
        if len(cols) < 4:
            continue
        start = stamp(cols[1].replace("T", " "))
        if start is None:  # pending: never ran
            continue
        end = stamp(cols[2].replace("T", " ")) or now()
        m = re.search(r"gres/gpu=(\d+)", cols[3])
        jobs.append((start, end, int(m[1]) if m else 1))  # the run's jobs hold 1 GPU each
    return jobs, None


def gpu_hours(jobs, window):
    start, end = window
    return sum(max(0.0, (min(e, end) - max(s, start)).total_seconds()) * g for s, e, g in jobs) / 3600


# ---------------------------------------------------------------------------------------------------- settings


def settings(it_dir, n, root, log, record):
    """Everything that defines how iteration N was run, flattened, for the diff against N-1."""
    out = {"launch": (read(os.path.join(root, "launch_settings.txt")) or "").strip() or None}
    first = log["first"].get(n)
    loops = [s for ts, s in log["loops"] if first is None or ts <= first]
    out["loop"] = loops[-1] if loops else None
    generator = (read(os.path.join(it_dir, "stages", "generator")) or "").strip()
    out["generator"] = generator[:12] or None
    train = load(os.path.join(it_dir, "stages", "train.done")) or {}
    out["train.init"] = (train.get("init") or "")[:12] or None
    t = record["training"]
    for k in ("steps", "lr", "trunk_lr", "replay_fraction"):
        out[f"train.{k}"] = t.get(k)
    out["train.batch"] = log["training"].get(n, {}).get("batch")
    out["promoted"] = record["outcome"]["promoted"]
    marker = load(os.path.join(it_dir, "stages", "match_generator.done")) or {}
    if not marker:
        segs = sorted(f for f in os.listdir(os.path.join(it_dir, "stages")) if f.startswith("seg")
                      and f.endswith(".done")) if os.path.isdir(os.path.join(it_dir, "stages")) else []
        marker = load(os.path.join(it_dir, "stages", segs[0])) if segs else {}
        marker = marker or {}
    for k, v in (marker.get("wave2") or {}).items():
        out[f"wave2.{k}"] = v
    out["match.movetime"] = marker.get("movetime")
    out["match.time_model"] = marker.get("time_model")
    return out


# ---------------------------------------------------------------------------------------------------- search balance

BALANCE_FILES = ("it{n}.plan.json", "it{n}.json", "it{n}/realised.json", "it{n}/overhead.json", "it{n}/score.json",
                 "it{n}/minigate.json", "prior.json")
THETA0 = {"f": 0.0, "k": None, "b": 800, "lambda": 0.0, "src": "cap"}


def theta_key(t):
    if not isinstance(t, dict):
        return None
    if sb:
        return sb.key(t)
    return f"f={t.get('f', 0):g},k={t.get('k') or '-'},b={t.get('b', 800)},l={t.get('lambda', 0):g},src={t.get('src', 'cap')}"


def prior_status(prior):
    """What the controller may do given balance/prior.json: 'none' (no file: it holds theta0), 'blocked' (a prior, but
    the score stage is not built: still theta0) or 'ready' (exploration allowed; some arms may still be masked)."""
    if prior is None:
        return {"status": "none", "text": "holding theta0: no measured prior", "flags": {}, "blockers": [],
                "masked": []}
    flags = {k: bool(prior.get(k)) for k in ("score_stage", "probe_bias_measured", "search_q_blend",
                                              "policy_positions", "reanalyse_fidelity")}
    flags["fits_bench_k8"] = bool((prior.get("fits_bench") or {}).get("8"))
    blockers = sb.exploration_blockers(prior) if sb else ([] if prior.get("score_stage") else ["no score stage"])
    masked = [m for m, ok in (("f>0 (probe bias not measured)", flags["probe_bias_measured"]),
                              ("lambda>0 (no --search-q-blend)", flags["search_q_blend"]),
                              ("k=8 (bench)", flags["fits_bench_k8"]),
                              ("src=reanalyse (fidelity)", flags["reanalyse_fidelity"])) if not ok]
    text = ("holding theta0: prior present but " + "; ".join(blockers)) if blockers else "exploring allowed"
    return {"status": "blocked" if blockers else "ready", "text": text, "flags": flags, "blockers": blockers,
            "masked": masked, "sigma_hat": prior.get("sigma_hat"), "tau_hat": prior.get("tau_hat"),
            "tau_df": prior.get("tau_df"), "gpu_hours": prior.get("cost_gpu_h"),
            "measured_at": prior.get("measured_at")}


def balance_section(root, n):
    """The search-balance record of iteration n (docs/search_balance.md section 9), or None when the run has no
    balance/itN.plan.json or itN.json (controller off): then the metrics record has no "balance" key at all."""
    bdir = os.path.join(root, "balance")
    plan = load(os.path.join(bdir, f"it{n}.plan.json"))
    res = load(os.path.join(bdir, f"it{n}.json"))
    if plan is None and res is None:
        return None
    plan, res = plan or {}, res or {}
    prior = load(os.path.join(bdir, "prior.json"))
    realised = (res.get("results") or {}).get("realised") or load(os.path.join(bdir, f"it{n}", "realised.json")) or {}
    dec = res.get("decision") or {}
    results = res.get("results") or {}
    state = res.get("state") or plan.get("state_in") or {}
    star = plan.get("theta_star") or res.get("theta_star_planned") or state.get("theta_star")
    chal = plan.get("challenger") if plan else res.get("challenger")
    mean, se = dec.get("mean"), dec.get("se")
    interval = [rnd(mean - 1.96 * se), rnd(mean + 1.96 * se)] if mean is not None and se is not None else None
    segs = plan.get("segments") or {}
    arms = {}
    for seg, r in realised.items():
        arm = (segs.get(seg) or {}).get("theta", "inc")
        a = arms.setdefault(arm, {"segments": 0, "games": 0, "gpu_s": 0.0, "visits": 0, "records": 0, "deep": 0,
                                  "masked": 0, "_sit": [], "_flag": []})
        a["segments"] += 1
        for k in ("games", "records", "deep", "masked"):
            a[k] += r.get(k) or 0
        a["gpu_s"] += r.get("gpu_s") or 0.0
        a["visits"] += r.get("visits") or ((r.get("deep_visits") or 0) + (r.get("fast_visits") or 0))
        for k, dst in (("sit_rate", "_sit"), ("flag_rate", "_flag")):
            if r.get(k) is not None:
                a[dst].append(r[k])
    per_arm = {}
    for arm, a in arms.items():
        g = a["gpu_s"]
        per_arm[arm] = {"segments": a["segments"], "games": a["games"], "records": a["records"],
                        "gpu_hours": rnd(g / 3600) if g else None,
                        "games_per_gpu_h": rnd(a["games"] / (g / 3600)) if g else None,
                        "visits_per_s": rnd(a["visits"] / g) if g and a["visits"] else None,
                        "deep_share": rnd(a["deep"] / a["records"]) if a["records"] else None,
                        "masked_share": rnd(a["masked"] / a["records"]) if a["records"] else None,
                        "sit_rate": rnd(sum(a["_sit"]) / len(a["_sit"])) if a["_sit"] else None,
                        "flag_rate": rnd(sum(a["_flag"]) / len(a["_flag"])) if a["_flag"] else None}
    evidence = []
    for pk, obs in sorted((state.get("evidence") or {}).items()):
        ds = [o.get("d") for o in obs if o.get("d") is not None]
        evidence.append({"pair": pk, "n": len(ds), "mean": rnd(sum(ds) / len(ds)) if ds else None,
                         "d": [rnd(x) for x in ds], "iters": [o.get("iter") for o in obs]})
    overhead = dict(state.get("overhead") or {})
    status = prior_status(prior)
    star_key = theta_key(star)
    if status["status"] != "ready" and star_key and star_key != theta_key(THETA0):
        status["text"] = status["text"].replace("holding theta0", f"holding {star_key}")
    return {"mode": plan.get("mode") or res.get("mode"), "reason": plan.get("reason"),
            "theta_star": star, "theta_star_key": star_key,
            "theta_star_next": state.get("theta_star") if res else None,
            "challenger": chal, "challenger_key": theta_key(chal), "challenger_kind": plan.get("challenger_kind"),
            "challenger_segment": plan.get("challenger_segment"), "twin_segment": plan.get("twin_segment"),
            "revert_test": plan.get("revert_test"),
            "decided": bool(res),
            "decision": {k: dec.get(k) for k in ("action", "reason", "Z", "mean", "se", "n", "delta_min", "sigma_hat",
                                                 "tau_hat", "tau_df")} if dec else None,
            "d": results.get("d"), "sigma_n": results.get("sigma_n"), "d_interval95": interval,
            "minigate": results.get("minigate"), "review": results.get("review"),
            "cost_overrun": results.get("cost_overrun"), "c_realised": results.get("c_realised"),
            "cost": {"c_inc": (plan.get("cost") or {}).get("c_inc"), "c_chal": (plan.get("cost") or {}).get("c_chal")},
            "games": {s: v.get("games") for s, v in segs.items()},
            "overhead": {"this_iter_gpu_h": overhead.get("this_iter_gpu_h"), "cum_gpu_h": overhead.get("cum_gpu_h"),
                         "run_gpu_h": overhead.get("run_gpu_h"), "frac": overhead.get("frac"),
                         "planned_gpu_h": rnd(sum((plan.get("overhead_planned") or {}).values())) if plan else None,
                         "cap": 0.10},
            "dropped": plan.get("dropped"), "per_arm": per_arm, "main_mix": res.get("main_mix"),
            "probe": {"version": (plan.get("probe") or {}).get("version"),
                      "refresh": (plan.get("probe") or {}).get("refresh")},
            "evidence": evidence, "tabu": state.get("tabu") or {}, "history": state.get("history") or [],
            "prior": status}


# ---------------------------------------------------------------------------------------------------- one iteration


def anchor_records(root, n, unavailable):
    """Fixed-anchor results of iteration n (cluster/launch_anchor_match.sh): RUN_ROOT/anchors/itN/<anchor>/summary.json,
    written once every part of that anchor has finished. Anchors usually finish after iteration.done."""
    out_dir = os.path.join(root, "anchors", f"it{n}")
    try:
        names = sorted(e.name for e in os.scandir(out_dir) if e.is_dir())
    except OSError:
        return {}
    anchors = {}
    for name in names:
        s = load(os.path.join(out_dir, name, "summary.json"))
        if s:
            anchors[name] = {"games": s.get("games"), "score": s.get("score"), "elo": s.get("elo"), "ci": s.get("ci"),
                             "wins": s.get("wins"), "losses": s.get("losses"), "draws": s.get("draws"),
                             "movetime": s.get("movetime"), "clock": s.get("clock"), "blocks": s.get("blocks") or []}
        else:
            parts = glob.glob(os.path.join(out_dir, name, "part*"))
            done = len(glob.glob(os.path.join(out_dir, name, "part*", "summary.json")))
            gave_up = len(glob.glob(os.path.join(out_dir, name, "part*", "GAVE_UP")))
            unavailable[f"outcome.anchors.{name}"] = (
                f"anchors/it{n}/{name}: {done} of {len(parts)} parts finished"
                + (f", {gave_up} given up after their job limit" if gave_up else "") + "; no summary.json yet")
    return anchors


def signature(it_dir, metrics_dir, n):
    """mtimes and sizes of everything a finished iteration's record is read from."""
    sig = []
    root = os.path.dirname(os.path.dirname(it_dir))  # RUN_ROOT/runs/rl-itN
    for p in sorted(glob.glob(os.path.join(root, "anchors", f"it{n}", "*", "summary.json"))
                    + glob.glob(os.path.join(root, "anchors", f"it{n}", "*", "part*", "summary.json"))
                    + glob.glob(os.path.join(root, "anchors", f"it{n}", "*", "part*", "GAVE_UP"))):
        st = os.stat(p)
        sig.append([os.path.relpath(p, root), st.st_mtime_ns, st.st_size])
    for sub in ("stages", "arms", "search/logs", "arms/C", "arms/match_generator"):
        d = os.path.join(it_dir, sub)
        try:
            for e in sorted(os.scandir(d), key=lambda e: e.name):
                if e.is_file():
                    st = e.stat()
                    sig.append([f"{sub}/{e.name}", st.st_mtime_ns, st.st_size])
        except OSError:
            pass
    for name in match_names(it_dir):
        for f in ("summary.json", "games.tsv"):
            p = os.path.join(it_dir, "arms", f"match_{name}", f)
            if os.path.isfile(p):
                st = os.stat(p)
                sig.append([f"match_{name}/{f}", st.st_mtime_ns, st.st_size])
    probe = os.path.join(metrics_dir, f"it{n}.probe.json")
    if os.path.isfile(probe):
        st = os.stat(probe)
        sig.append(["probe", st.st_mtime_ns, st.st_size])
    # Search balance: only files that exist, so a run without the controller keeps its old signature (and cache).
    for rel in BALANCE_FILES:
        p = os.path.join(root, "balance", rel.format(n=n))
        if os.path.isfile(p):
            st = os.stat(p)
            sig.append([f"balance/{rel.format(n=n)}", st.st_mtime_ns, st.st_size])
    return sig


def iteration(root, n, log, metrics_dir, data_cache):
    it_dir = os.path.join(root, "runs", f"rl-it{n}")
    stages = os.path.join(it_dir, "stages")
    unavailable = {}
    done = os.path.isfile(os.path.join(stages, "iteration.done"))
    gate_generator = match_record(it_dir, "generator")
    gate_it13 = match_record(it_dir, "it0")  # rl-loop's fixed baseline it0 is the starting it13 network
    # Fixed anchors at the real clock, plus any extra gate match rl-loop played; its it0 match is gate_it13 only.
    anchors = anchor_records(root, n, unavailable)
    for name in match_names(it_dir):
        if name not in ("generator", "it0") and name not in anchors:
            m = match_record(it_dir, name)
            if m:
                anchors[name] = m
    promote = load(os.path.join(stages, "promote.done"))
    if gate_generator is None:
        unavailable["outcome.gate_generator"] = "the generator gate has not started"
    if gate_it13 is None:
        unavailable["outcome.gate_it13"] = "no match vs it0 (=it13) yet"
    outcome = {"gate_generator": gate_generator, "gate_it13": gate_it13,
               "promoted": None if promote is None else promote.get("promoted"),
               "anchors": anchors, "elo_vs_it13": gate_it13["elo"] if gate_it13 else None}
    if promote is None:
        unavailable["outcome.promoted"] = "no promote.done yet"
    if outcome["elo_vs_it13"] is None:
        unavailable["outcome.elo_vs_it13"] = "no match vs it0 (=it13) yet"
    probe = load(os.path.join(metrics_dir, f"it{n}.probe.json"))
    if probe is None:
        unavailable["network"] = f"no metrics/it{n}.probe.json (cluster/probe_net.py has not run on this iteration)"
    record = {"schema": SCHEMA, "run": os.path.basename(os.path.normpath(root)), "iteration": n, "done": done,
              "generated_at": now().isoformat(), "outcome": outcome,
              "cost": {"gpu_hours": None, "gpu_hours_cum": None, "elo_per_gpu_hour": None, "wall_hours": None},
              "selfplay": selfplay(it_dir, n, log, data_cache, unavailable),
              "training": training(it_dir, n, log, unavailable),
              "search": match_search(it_dir, unavailable),
              "network": probe, "changes": {}, "alerts": [], "unavailable": unavailable}
    bal = balance_section(root, n)
    if bal is not None:
        record["balance"] = bal
        arms = bal["per_arm"]
        record["cost"]["selfplay_gpu_h_by_arm"] = {a: v["gpu_hours"] for a, v in arms.items()} or None
        record["cost"]["balance_overhead_gpu_hours"] = bal["overhead"]["this_iter_gpu_h"]
    record["_settings"] = settings(it_dir, n, root, log, record)
    return record


# ---------------------------------------------------------------------------------------------------- the run


def alerts_for(history):
    """Alerts for the last record of `history` (records it1..itN in order), from finished data only, so an alert
    written to alerts.tsv never rests on a half-played match."""
    rec, n = history[-1], history[-1]["iteration"]
    prev = history[-2] if len(history) > 1 else None
    out = []
    sp_complete = rec["training"]["steps"] is not None or rec["done"]  # train started: all self-play is in

    def value_loss(r):
        heldout = ((r.get("network") or {}).get("heldout") or {}).get("value_loss")
        if heldout is not None:
            return heldout, "held-out value loss"
        final = r["training"]["final"] or {}
        for k in ("search_wdl", "wdl", "search_value", "value"):
            if final.get(k) is not None:
                return final[k], f"final {k}"
        return None, None

    if rec["done"] and len(history) >= 3 and all(r["done"] for r in history[-3:]):
        losses = [value_loss(r) for r in history[-3:]]
        if all(v is not None for v, _ in losses) and len({s for _, s in losses}) == 1 \
                and losses[0][0] < losses[1][0] < losses[2][0]:
            out.append({"level": "warn", "code": "heldout_value_loss_up",
                        "message": f"{losses[2][1]} rose two iterations in a row: "
                                   + " -> ".join(f"{v:.4f}" for v, _ in losses)})
    if sp_complete and prev and rec["selfplay"]["flag_rate"] is not None and prev["selfplay"]["flag_rate"] is not None:
        jump = rec["selfplay"]["flag_rate"] - prev["selfplay"]["flag_rate"]
        if jump > FLAG_JUMP:
            out.append({"level": "warn", "code": "flag_rate_jump",
                        "message": f"flag rate {prev['selfplay']['flag_rate']:.3f} -> {rec['selfplay']['flag_rate']:.3f}"
                                   f" (+{100 * jump:.1f} points)"})
    first = (history[0]["selfplay"].get("policy_target_entropy") or {}).get("mean")
    ent = (rec["selfplay"].get("policy_target_entropy") or {}).get("mean")
    if sp_complete and first and ent is not None and ent < (1 - ENTROPY_DROP) * first:
        out.append({"level": "warn", "code": "entropy_collapse",
                    "message": f"policy target entropy {ent:.3f} vs it{history[0]['iteration']} {first:.3f} "
                               f"({100 * (ent / first - 1):.0f}%)"})
    gate = rec["outcome"]["gate_it13"]
    if rec["done"] and gate and gate["games"] >= GATE_LOW_GAMES and gate["score"] is not None \
            and gate["score"] < GATE_LOW:
        out.append({"level": "warn", "code": "gate_low",
                    "message": f"score vs it13 {gate['score']:.3f} over {gate['games']} games (< {GATE_LOW})"})
    finished = [r for r in history if r["done"]]
    if rec["done"] and len(finished) >= FLAT_WINDOW and finished[-1] is rec:
        window = finished[-FLAT_WINDOW:]
        before = finished[-FLAT_WINDOW - 1]["outcome"]["elo_vs_it13"] if len(finished) > FLAT_WINDOW else 0.0
        now_elo = rec["outcome"]["elo_vs_it13"]
        if now_elo is not None and before is not None:
            gain = now_elo - before
            if gain < FLAT_ELO:
                gpu = [r["cost"]["gpu_hours"] for r in window]
                spent = f" for {sum(gpu):.0f} GPU-hours" if all(g is not None for g in gpu) else ""
                out.append({"level": "stop", "code": "elo_per_gpu_flat",
                            "message": f"Elo vs it13 gained {gain:+.1f} over the last {FLAT_WINDOW} iterations"
                                       f"{spent} (< {FLAT_ELO:g}): recommend stopping the run"})
    if rec["done"]:
        for name, m in rec["outcome"]["anchors"].items():
            earlier = [r["outcome"]["anchors"][name] for r in history[:-1] if r["done"] and name in r["outcome"]["anchors"]]
            if earlier and m["score"] is not None and earlier[-1]["score"] is not None \
                    and m["score"] < earlier[-1]["score"] - ANCHOR_DROP:
                out.append({"level": "warn", "code": "anchor_drop",
                            "message": f"{name}: score {earlier[-1]['score']:.3f} -> {m['score']:.3f}"})
    return out


def iterations_of(root):
    runs = os.path.join(root, "runs")
    try:
        names = os.listdir(runs)
    except OSError:
        return []
    return sorted(int(d[5:]) for d in names if d.startswith("rl-it") and d[5:].isdigit())


def collect(root, out_dir=None):
    """Write metrics/itN.json for every iteration of the run at `root`; returns the records in order."""
    root = os.path.abspath(root)
    metrics_dir = out_dir or os.path.join(root, "metrics")
    cache_path = os.path.join(metrics_dir, ".cache.json")
    cache = load(cache_path) or {}
    if cache.get("version") != CACHE_VERSION:
        cache = {"version": CACHE_VERSION, "records": {}, "data": {}}
    log = parse_log(root)
    numbers = iterations_of(root)
    records, fresh = [], set()
    for n in numbers:
        it_dir = os.path.join(root, "runs", f"rl-it{n}")
        sig = signature(it_dir, metrics_dir, n)
        cached = cache["records"].get(str(n))
        if cached and cached["record"]["done"] and cached["sig"] == sig:
            records.append(json.loads(json.dumps(cached["record"])))
            continue
        records.append(iteration(root, n, log, metrics_dir, cache["data"]))
        fresh.add(n)
        cache["records"][str(n)] = {"sig": sig, "record": None}
    # Cost: sacct only when some record is new (a cached finished iteration keeps its GPU-hours).
    span = windows(log, numbers)
    if fresh:
        jobs, reason = sacct_jobs(job_ids(root))
        for r in records:
            n = r["iteration"]
            if n not in fresh:
                continue
            w = span.get(n)
            if w:
                end = log["finished"].get(n) or (log["last"].get(n) if r["done"] else now()) or w[1]
                r["cost"]["wall_hours"] = round((end - log["first"].get(n, w[0])).total_seconds() / 3600, 4)
            else:
                r["unavailable"]["cost.wall_hours"] = "the iteration has no lines in runs/rl-loop.log"
            if jobs is not None and w:
                r["cost"]["gpu_hours"] = round(gpu_hours(jobs, w), 4)
            else:
                r["unavailable"]["cost.gpu_hours"] = reason or "the iteration has no lines in runs/rl-loop.log"
    # Cross-iteration fields, cheap, always recomputed.
    cum, prev_elo = 0.0, 0.0
    # Search balance: its stages run inside the main job's iteration window, so their GPU-hours are already part of
    # gpu_hours (balance_overhead_gpu_hours says how much). The prior job (cluster/balance_prior.sbatch) runs outside
    # every window: its GPU-hours are added once, to the first iteration with a balance record (gpu_hours_total).
    first_bal = next((r["iteration"] for r in records if "balance" in r), None)
    for i, r in enumerate(records):
        g = r["cost"]["gpu_hours"]
        if "balance" in r:
            extra = (r["balance"]["prior"].get("gpu_hours") or 0.0) if r["iteration"] == first_bal else 0.0
            r["cost"]["balance_prior_gpu_hours"] = extra or None
            g = None if g is None else round(g + extra, 4)
            r["cost"]["gpu_hours_total"] = g
        cum = None if cum is None or g is None else cum + g
        r["cost"]["gpu_hours_cum"] = None if cum is None else round(cum, 4)
        el = r["outcome"]["elo_vs_it13"]
        if g and el is not None and prev_elo is not None:
            r["cost"]["elo_per_gpu_hour"] = round((el - prev_elo) / g, 4)
        r["unavailable"].pop("cost.gpu_hours_cum", None)
        r["unavailable"].pop("cost.elo_per_gpu_hour", None)
        if r["cost"]["gpu_hours_cum"] is None:
            r["unavailable"]["cost.gpu_hours_cum"] = "GPU-hours missing for this or an earlier iteration"
        if r["cost"]["elo_per_gpu_hour"] is None:
            r["unavailable"]["cost.elo_per_gpu_hour"] = ("needs this iteration's GPU-hours and Elo vs it13 "
                                                         "(and the previous iteration's Elo)")
        prev_elo = el  # it0 is it13 itself: 0 before it1
        old = records[i - 1]["_settings"] if i else None
        r["changes"] = {k: [old.get(k), v] for k, v in r["_settings"].items() if old.get(k) != v} if old else {}
        r["alerts"] = alerts_for(records[:i + 1])
    for r in records:
        cache["records"][str(r["iteration"])]["record"] = r
        out = {k: v for k, v in r.items() if not k.startswith("_")}
        write_json(os.path.join(metrics_dir, f"it{r['iteration']}.json"), out)
    write_json(cache_path, cache)
    return records


def append_alerts(root, records, out_dir=None):
    """Append alerts not yet in alerts.tsv (one per iteration, code and anchor); returns the new lines."""
    path = os.path.join(out_dir or os.path.abspath(root), "alerts.tsv")

    def key(iteration, code, message):
        return (str(iteration), code, message.split(":")[0] if code == "anchor_drop" else "")

    seen = set()
    for line in (read(path) or "").splitlines():
        cols = line.split("\t")
        if len(cols) >= 5:
            seen.add(key(cols[1], cols[3], cols[4]))
    new = []
    for r in records:
        for a in r["alerts"]:
            k = key(r["iteration"], a["code"], a["message"])
            if k not in seen:
                seen.add(k)
                new.append("\t".join((now().isoformat(), str(r["iteration"]), a["level"], a["code"], a["message"])))
    if new:
        with open(path, "a") as f:
            f.write("".join(line + "\n" for line in new))
    return new


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("roots", nargs="+", help="run folders (each holds runs/rl-itN)")
    ap.add_argument("--summary", action="store_true", help="write metrics/summary.json per run, print all runs as JSON")
    ap.add_argument("--alerts", action="store_true", help="append new alerts to alerts.tsv and print them")
    ap.add_argument("--out", help="write under OUT/<run>/ instead of the run folder (a read-only copy of a run)")
    args = ap.parse_args()
    summary = {"schema": SCHEMA, "generated_at": now().isoformat(), "runs": []}
    for root in args.roots:
        if not os.path.isdir(os.path.join(root, "runs")):
            print(f"{root}: no runs/ folder, skipped", file=sys.stderr)
            continue
        base = os.path.join(args.out, os.path.basename(os.path.normpath(root))) if args.out else None
        metrics_dir = os.path.join(base, "metrics") if base else os.path.join(os.path.abspath(root), "metrics")
        with locked(metrics_dir):
            records = collect(root, metrics_dir)
            new_alerts = append_alerts(root, records, base) if args.alerts else []
        public = [{k: v for k, v in r.items() if not k.startswith("_")} for r in records]
        entry = {"run": os.path.basename(os.path.normpath(root)), "root": os.path.abspath(root),
                 "settings": (read(os.path.join(root, "launch_settings.txt")) or "").strip() or None,
                 "iterations": public,
                 "alerts": [dict(a, iteration=r["iteration"]) for r in public for a in r["alerts"]]}
        summary["runs"].append(entry)
        if args.summary:
            write_json(os.path.join(base or os.path.abspath(root), "metrics", "summary.json"),
                       {"schema": SCHEMA, "generated_at": summary["generated_at"], "runs": [entry]})
        if args.alerts:
            for line in new_alerts:
                print(line, file=sys.stderr if args.summary else sys.stdout)  # stdout stays JSON with --summary
        elif not args.summary:
            for r in public:
                o = r["outcome"]
                gate = o["gate_generator"]["score"] if o["gate_generator"] else None
                print(f"{entry['run']} it{r['iteration']}: done {r['done']}, games {r['selfplay']['games']}, "
                      f"gate {gate}, elo vs it13 {o['elo_vs_it13']}, GPU-h {r['cost']['gpu_hours']}, "
                      f"alerts {len(r['alerts'])}, unavailable {len(r['unavailable'])}")
    if args.summary:
        json.dump(summary, sys.stdout)
        print()


if __name__ == "__main__":
    main()
