"""cluster/run_metrics.py on a copy of the tiny local rl-loop run runs/wave2/int-shared (one real iteration), and on a
four-iteration run made from it with edited stage markers, a synthetic rl-loop.log and a fake sacct on PATH (cost
apportioning, changes, every alert rule, alerts.tsv dedup, the mtime cache, self-play stats surviving an archive).
    .venv/bin/python -m pytest tests/cluster -q
"""

import contextlib
import json
import os
import shutil
import subprocess
import sys

import pytest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "cluster"))

import run_metrics as rm  # noqa: E402

FIXTURE = os.path.join(REPO, "runs", "wave2", "int-shared")
SCRIPT = os.path.join(REPO, "cluster", "run_metrics.py")
pytestmark = pytest.mark.skipif(not os.path.isdir(os.path.join(FIXTURE, "runs", "rl-it1")),
                                reason="runs/wave2/int-shared is not on this machine")


def copy_fixture(dest):
    shutil.copytree(FIXTURE, dest, ignore=shutil.ignore_patterns("*.pt", "*.onnx", "artifacts", "models", "metrics",
                                                                 "alerts.tsv"))
    return dest


def no_sacct(monkeypatch, tmp_path):
    empty = tmp_path / "empty_bin"
    empty.mkdir(exist_ok=True)
    monkeypatch.setenv("PATH", str(empty))


def read_json(path):
    with open(path) as f:
        return json.load(f)


def write_json(path, value):
    with open(path, "w") as f:
        json.dump(value, f)


# ---------------------------------------------------------------------------------------------------- one real iteration


def test_int_shared_record(tmp_path, monkeypatch):
    no_sacct(monkeypatch, tmp_path)
    root = copy_fixture(str(tmp_path / "int-shared"))
    rm.collect(root)
    r = read_json(os.path.join(root, "metrics", "it1.json"))
    assert r["schema"] == 1 and r["run"] == "int-shared" and r["iteration"] == 1 and r["done"] is True
    o = r["outcome"]
    assert o["gate_generator"]["games"] == 2 and o["gate_generator"]["score"] == 0.5
    assert o["gate_generator"]["blocks"] == [{"games": 2, "score": 0.5}]
    assert o["gate_generator"]["ci"] == [-392.519316, 392.519316] and o["gate_generator"]["movetime"] == 100
    assert o["gate_it13"]["score"] == 0.5 and o["elo_vs_it13"] == 0.0 and o["promoted"] is False
    assert o["anchors"] == {}  # the it0 match is gate_it13, not an anchor; no anchors/ in this run
    sp = r["selfplay"]
    # The four segments' markers: 2 games each, 424 + 233 + 226 + 424 positions, 73 + 0 + 23 + 45 repeats.
    assert (sp["games"], sp["positions"], sp["segments"]) == (8, 1307, 4)
    assert sp["terminations"] == {"checkmate": 8} and sp["flag_rate"] == 0.0
    assert sp["repeat_rate"] == pytest.approx(141 / 1307, abs=1e-6)
    # Checked against hivemind.distill.data.read_chunk with numpy on these chunks.
    assert sp["value_target"] == {"win": pytest.approx(0.457536, abs=1e-6), "draw": 0.0,
                                  "loss": pytest.approx(0.542464, abs=1e-6)}
    assert sp["pocket_mean"] == pytest.approx(11.150727, abs=1e-5)
    assert sp["policy_target_entropy"]["mean"] == pytest.approx(1.148129, abs=1e-5)
    assert sp["pass_rate"] == pytest.approx(0.295617, abs=1e-5)
    assert sp["game_length"]["p50"] > 0 and 0 < sp["drop_rate"] < 1
    assert 0 <= sp["sit_rate"] <= sp["hybrid_wait_rate"] <= 1
    t = r["training"]
    assert (t["steps"], t["lr"], t["trunk_lr"], t["new_positions"], t["replay_fraction"]) == (8, 1e-4, 5e-5, 1307, 0.0)
    assert len(t["curve"]) == 4 and t["final"] == t["curve"][-1] and t["final"]["search_wdl"] == 0.62932
    s = r["search"]
    assert s["nps"] == [19190.690229, 18969.255498] and s["sit_gate_rate"] == 0.0
    assert s["time_use"]["movetime"] == 100 and s["time_use"]["contender_ms_per_search"] > 0
    assert r["network"] is None and "network" in r["unavailable"]
    assert r["cost"]["gpu_hours"] is None and "sacct" in r["unavailable"]["cost.gpu_hours"]
    assert r["cost"]["wall_hours"] == pytest.approx((3 * 60 + 25) / 3600, abs=1e-3)  # 12:55:56 -> 12:59:21
    assert r["changes"] == {} and r["alerts"] == []
    # No search balance on this run: no "balance" key and the cost section as before.
    assert "balance" not in r and set(r["cost"]) == {"gpu_hours", "gpu_hours_cum", "elo_per_gpu_hour", "wall_hours"}
    # Every null KPI says why.
    assert r["search"]["depth"] is None and "search.depth" in r["unavailable"]
    assert r["training"]["grad_norm"] is None and "training.grad_norm" in r["unavailable"]


def test_probe_is_merged(tmp_path, monkeypatch):
    no_sacct(monkeypatch, tmp_path)
    root = copy_fixture(str(tmp_path / "int-shared"))
    os.makedirs(os.path.join(root, "metrics"))
    probe = {"schema": 1, "iteration": 1, "heldout": {"value_loss": 0.5}}
    write_json(os.path.join(root, "metrics", "it1.probe.json"), probe)
    rm.collect(root)
    r = read_json(os.path.join(root, "metrics", "it1.json"))
    assert r["network"] == probe and "network" not in r["unavailable"]


def test_unfinished_iteration(tmp_path, monkeypatch):
    """A running iteration: one segment done, one still playing, nothing trained."""
    no_sacct(monkeypatch, tmp_path)
    root = copy_fixture(str(tmp_path / "run"))
    it = os.path.join(root, "runs", "rl-it1")
    for f in os.listdir(os.path.join(it, "stages")):
        if f not in ("generator", "seg00.done"):
            os.remove(os.path.join(it, "stages", f))
    shutil.rmtree(os.path.join(it, "arms"))
    with open(os.path.join(root, "runs", "rl-loop.log"), "w") as f:
        f.write("2026-11-22 12:55:56 loop: iterations 1..4, 8 games (2 per segment), clock time model\n"
                "2026-11-22 12:55:56 it1 self-play seg00: 2 games, seed 1000\n")
    rm.collect(root)
    r = read_json(os.path.join(root, "metrics", "it1.json"))
    assert r["done"] is False and r["outcome"]["gate_generator"] is None and r["outcome"]["promoted"] is None
    assert r["selfplay"]["games"] == 2 and r["selfplay"]["segments"] == 1
    # Chunk stats from the finished segment only; game lengths from every segment log on disk.
    assert r["selfplay"]["value_target"] is not None and r["selfplay"]["game_length"] is not None
    assert r["training"]["steps"] is None and "training.steps" in r["unavailable"]
    assert "outcome.gate_it13" in r["unavailable"] and r["cost"]["wall_hours"] > 0  # running: until now
    assert r["alerts"] == []


# ---------------------------------------------------------------------------------------------------- four iterations

SCORES = {1: 0.55, 2: 0.56, 3: 0.565, 4: 0.44}  # vs it13, 400 games each
ANCHOR_SCORES = {2: 0.6, 4: 0.475}                 # hybrid-it13 at 120 s (launch_anchor_match.sh, every 2nd)


def anchor_summary(root, n, name, score, games=400):
    d = os.path.join(root, "anchors", f"it{n}", name)
    os.makedirs(d, exist_ok=True)
    wins = round(score * games)
    write_json(os.path.join(d, "summary.json"),
               {"anchor": name, "iteration": n, "games": games, "score": score, "elo": round(rm.elo(score), 1),
                "ci": [-40.0, 40.0], "score_ci": [score - 0.05, score + 0.05], "wins": wins, "losses": games - wins,
                "draws": 0, "errors": 0, "movetime": None, "clock": 120000,
                "blocks": [{"games": 100, "score": score}] * (games // 100), "parts": 2})


def match_marker(score, wave2):
    wins = round(score * 400)
    return {"wins": wins, "losses": 400 - wins, "draws": 0, "score": score, "elo": round(rm.elo(score), 6),
            "ci": [-30.0, 30.0], "nps": [1000.0, 900.0], "movetime": 100, "time_model": "clock", "wave2": wave2}


def make_run(tmp_path):
    """it1..it4 from int-shared's it1: scores above, it2 one flag game in seg00, it3 trained with 16 steps,
    one fake 1-GPU job running 09:59..14:00 and a log that gives it1 50 minutes and it2..it4 an hour each."""
    root = copy_fixture(str(tmp_path / "multi"))
    runs = os.path.join(root, "runs")
    for n in (2, 3, 4):
        shutil.copytree(os.path.join(runs, "rl-it1"), os.path.join(runs, f"rl-it{n}"),
                        ignore=shutil.ignore_patterns("*.pt", "*.onnx"))
    lines = ["2026-11-22 10:00:00 loop: iterations 1..4, 8 games (2 per segment), clock time model"]
    for n in (1, 2, 3, 4):
        stages = os.path.join(runs, f"rl-it{n}", "stages")
        wave2 = read_json(os.path.join(stages, "match_generator.done"))["wave2"]
        marker = match_marker(SCORES[n], wave2)
        write_json(os.path.join(stages, "match_it0.done"), marker)  # no same_as: a match of its own
        write_json(os.path.join(stages, "match_generator.done"), dict(marker, score=0.5))
        if n == 2:
            seg = read_json(os.path.join(stages, "seg00.done"))
            seg["labels"]["flag_games"] = 1
            write_json(os.path.join(stages, "seg00.done"), seg)
        if n == 3:
            train = read_json(os.path.join(stages, "train.done"))
            write_json(os.path.join(stages, "train.done"), dict(train, steps=16))
        start = f"2026-11-22 {9 + n:02d}:{10 if n == 1 else 0:02d}:00"
        lines += [f"{start} it{n} self-play seg00: 2 games, seed {1000 * n}",
                  f"2026-11-22 {9 + n:02d}:50:00 it{n} finished: vs generator {{}}"]
    with open(os.path.join(runs, "rl-loop.log"), "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(root, "jobs.txt"), "w") as f:
        f.write("101\n")
    os.remove(os.path.join(root, "workers.tsv"))
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "sacct").write_text("#!/bin/sh\necho '101|2026-11-22T09:59:00|2026-11-22T14:00:00|"
                                   "billing=1,cpu=6,gres/gpu=1,mem=48G'\n")
    (bin_dir / "sacct").chmod(0o755)
    for n, score in ANCHOR_SCORES.items():
        anchor_summary(root, n, "hybrid-it13", score)
    metrics = os.path.join(root, "metrics")
    os.makedirs(metrics)
    for n, loss in ((1, 0.50), (2, 0.55), (3, 0.60)):  # held-out value loss up twice by it3
        write_json(os.path.join(metrics, f"it{n}.probe.json"), {"schema": 1, "iteration": n,
                                                                 "heldout": {"value_loss": loss}})
    return root, str(bin_dir)


def codes(record):
    return sorted(a["code"] for a in record["alerts"])


def test_cost_changes_and_alerts(tmp_path, monkeypatch):
    root, bin_dir = make_run(tmp_path)
    monkeypatch.setenv("PATH", bin_dir + os.pathsep + os.environ["PATH"])
    records = rm.collect(root)
    r = {x["iteration"]: read_json(os.path.join(root, "metrics", f"it{x['iteration']}.json")) for x in records}
    # GPU-hours: it1 10:00 (the loop's first line) .. 10:50, then each iteration from the previous one's end.
    assert [r[n]["cost"]["gpu_hours"] for n in (1, 2, 3, 4)] == pytest.approx([50 / 60, 1, 1, 1], abs=1e-3)
    assert r[4]["cost"]["gpu_hours_cum"] == pytest.approx(50 / 60 + 3, abs=1e-3)
    assert r[1]["cost"]["wall_hours"] == pytest.approx(40 / 60, abs=1e-3)  # its own first line 10:10 .. 10:50
    elo = {n: rm.elo(SCORES[n]) for n in SCORES}
    assert r[1]["cost"]["elo_per_gpu_hour"] == pytest.approx(elo[1] / (50 / 60), abs=1e-2)
    assert r[2]["cost"]["elo_per_gpu_hour"] == pytest.approx(elo[2] - elo[1], abs=1e-2)
    # Changes: it3 trained 16 steps (8 before and after), it1 has no predecessor.
    assert r[1]["changes"] == {}
    assert r[3]["changes"]["train.steps"] == [8, 16] and r[4]["changes"]["train.steps"] == [16, 8]
    assert "train.steps" not in r[2]["changes"]
    # Alerts.
    assert codes(r[1]) == []
    assert codes(r[2]) == ["flag_rate_jump"]               # 0 -> 1 flag game in 8
    assert codes(r[3]) == ["heldout_value_loss_up"]       # 0.50 -> 0.55 -> 0.60 (flags back to 0: no jump)
    # Anchors come from anchors/itN/<anchor>/summary.json, the gate's it0 match is not one.
    assert set(r[2]["outcome"]["anchors"]) == {"hybrid-it13"} and r[1]["outcome"]["anchors"] == {}
    a = r[2]["outcome"]["anchors"]["hybrid-it13"]
    assert (a["games"], a["score"], a["clock"], a["movetime"], a["wins"]) == (400, 0.6, 120000, None, 240)
    assert a["ci"] == [-40.0, 40.0] and len(a["blocks"]) == 4
    # it4: score 0.44 over 400 games; Elo vs it13 fell since it1 (3 iterations); the hybrid-it13 anchor fell 0.125.
    assert codes(r[4]) == ["anchor_drop", "elo_per_gpu_flat", "gate_low"]
    flat = [a for a in r[4]["alerts"] if a["code"] == "elo_per_gpu_flat"][0]
    assert flat["level"] == "stop" and "3 GPU-hours" in flat["message"]
    assert all(a["level"] == "warn" for a in r[4]["alerts"] if a["code"] != "elo_per_gpu_flat")


def test_alerts_cli_appends_once(tmp_path, monkeypatch):
    root, bin_dir = make_run(tmp_path)
    env = dict(os.environ, PATH=bin_dir + os.pathsep + os.environ["PATH"])
    first = subprocess.run([sys.executable, SCRIPT, root, "--alerts"], capture_output=True, text=True, env=env)
    assert first.returncode == 0, first.stderr
    printed = [line.split("\t") for line in first.stdout.splitlines()]
    assert sorted((p[1], p[3]) for p in printed) == [("2", "flag_rate_jump"), ("3", "heldout_value_loss_up"),
                                                     ("4", "anchor_drop"), ("4", "elo_per_gpu_flat"),
                                                     ("4", "gate_low")]
    with open(os.path.join(root, "alerts.tsv")) as f:
        assert f.read().splitlines() == first.stdout.splitlines()
    again = subprocess.run([sys.executable, SCRIPT, root, "--alerts"], capture_output=True, text=True, env=env)
    assert again.returncode == 0 and again.stdout == ""
    with open(os.path.join(root, "alerts.tsv")) as f:
        assert len(f.read().splitlines()) == 5


def test_summary_cli(tmp_path, monkeypatch):
    root, bin_dir = make_run(tmp_path)
    env = dict(os.environ, PATH=bin_dir + os.pathsep + os.environ["PATH"])
    out = subprocess.run([sys.executable, SCRIPT, root, "--summary", "--alerts"], capture_output=True, text=True,
                         env=env)
    assert out.returncode == 0, out.stderr
    summary = json.loads(out.stdout)  # alert lines go to stderr with --summary
    assert summary["schema"] == 1 and len(summary["runs"]) == 1
    run = summary["runs"][0]
    assert run["run"] == "multi" and [i["iteration"] for i in run["iterations"]] == [1, 2, 3, 4]
    assert all(not any(k.startswith("_") for k in i) for i in run["iterations"])
    assert {a["code"] for a in run["alerts"]} >= {"gate_low", "elo_per_gpu_flat"}
    assert read_json(os.path.join(root, "metrics", "summary.json"))["runs"][0]["run"] == "multi"
    assert "flag_rate_jump" in out.stderr


def test_out_dir_leaves_the_run_alone(tmp_path, monkeypatch):
    no_sacct(monkeypatch, tmp_path)
    root = copy_fixture(str(tmp_path / "ro"))
    env = dict(os.environ, PATH=str(tmp_path / "empty_bin"))
    out = subprocess.run([sys.executable, SCRIPT, root, "--out", str(tmp_path / "out"), "--summary", "--alerts"],
                         capture_output=True, text=True, env=env)
    assert out.returncode == 0, out.stderr
    assert not os.path.exists(os.path.join(root, "metrics")) and not os.path.exists(os.path.join(root, "alerts.tsv"))
    assert os.path.isfile(tmp_path / "out" / "ro" / "metrics" / "it1.json")
    assert os.path.isfile(tmp_path / "out" / "ro" / "metrics" / "summary.json")


def test_cache_and_archived_selfplay(tmp_path, monkeypatch):
    root, bin_dir = make_run(tmp_path)
    monkeypatch.setenv("PATH", bin_dir + os.pathsep + os.environ["PATH"])
    first = {r["iteration"]: r for r in rm.collect(root)}
    # Unchanged finished iterations come from the cache: their generated_at stays.
    stamp = first[2]["generated_at"]
    monkeypatch.setattr(rm, "now", lambda: rm.datetime.datetime(2030, 1, 1))
    second = {r["iteration"]: r for r in rm.collect(root)}
    assert second[2]["generated_at"] == stamp
    # A changed marker re-reads that iteration; its self-play, archived meanwhile, keeps its measured stats.
    it2 = os.path.join(root, "runs", "rl-it2")
    shutil.rmtree(os.path.join(it2, "search"))
    marker = os.path.join(it2, "stages", "match_it0.done")
    write_json(marker, dict(read_json(marker), score=0.6))
    third = {r["iteration"]: r for r in rm.collect(root)}
    assert third[2]["generated_at"] != stamp and third[2]["outcome"]["gate_it13"]["score"] == 0.6
    for k in ("value_target", "policy_target_entropy", "pocket_mean", "game_length", "drop_rate"):
        assert third[2]["selfplay"][k] == first[2]["selfplay"][k], k


def test_partial_match_from_games_tsv(tmp_path):
    """A gate still being played: score, Elo and a Wilson CI from games.tsv alone."""
    it = tmp_path / "rl-it1"
    (it / "stages").mkdir(parents=True)
    (it / "arms" / "match_generator").mkdir(parents=True)
    rows = ["config\t--contender x"] + [f"{i}\t{1 if i % 4 else -1}\t1\t0\tcheckmate\t10\t1000\t500000000\t"
                                         f"10\t900\t600000000\t100" for i in range(200)]
    (it / "arms" / "match_generator" / "games.tsv").write_text("\n".join(rows) + "\n")
    m = rm.match_record(str(it), "generator")
    assert m["games"] == 200 and m["score"] == 0.75 and (m["wins"], m["losses"], m["draws"]) == (150, 50, 0)
    assert m["elo"] == pytest.approx(190.85, abs=0.01) and m["ci"][0] < m["elo"] < m["ci"][1]
    assert [b["games"] for b in m["blocks"]] == [100, 100]
    unavailable = {}
    s = rm.match_search(str(it), unavailable)
    assert s["time_use"]["contender_ms_per_search"] == 50.0 and s["time_use"]["baseline_ms_per_search"] == 60.0
    assert s["nps"] is None and "search.nps" in unavailable


def test_anchor_lands_after_iteration_done(tmp_path, monkeypatch):
    """Anchors finish after iteration.done: the cached finished record is refreshed when summary.json appears, and
    an anchor still playing is listed under unavailable."""
    root, bin_dir = make_run(tmp_path)
    monkeypatch.setenv("PATH", bin_dir + os.pathsep + os.environ["PATH"])
    shutil.rmtree(os.path.join(root, "anchors", "it4"))
    os.makedirs(os.path.join(root, "anchors", "it4", "hybrid-it13", "part1"))
    os.makedirs(os.path.join(root, "anchors", "it4", "hybrid-it13", "part2"))
    first = {r["iteration"]: r for r in rm.collect(root)}
    assert first[4]["outcome"]["anchors"] == {}
    assert "0 of 2 parts" in first[4]["unavailable"]["outcome.anchors.hybrid-it13"]
    assert "anchor_drop" not in codes(first[4])
    anchor_summary(root, 4, "hybrid-it13", ANCHOR_SCORES[4])
    second = {r["iteration"]: r for r in rm.collect(root)}
    assert second[4]["outcome"]["anchors"]["hybrid-it13"]["score"] == ANCHOR_SCORES[4]
    assert "outcome.anchors.hybrid-it13" not in second[4]["unavailable"] and "anchor_drop" in codes(second[4])
    assert read_json(os.path.join(root, "metrics", "it4.json"))["outcome"]["anchors"]["hybrid-it13"]["games"] == 400


def test_sit_gate_rate_from_sprt_chunks(tmp_path, monkeypatch):
    """rl-loop --gate-sprt logs the gate per chunk (arms/match_generator/chunkNN.log, games numbered from 1 again)."""
    no_sacct(monkeypatch, tmp_path)
    root = copy_fixture(str(tmp_path / "sprt"))
    arms = os.path.join(root, "runs", "rl-it1", "arms")
    os.remove(os.path.join(arms, "match_generator.log"))
    for k, (sits, searches) in enumerate(((3, 100), (5, 100))):
        with open(os.path.join(arms, "match_generator", f"chunk{k:02d}.log"), "w") as f:
            for g in (1, 2):
                f.write(f"tournament hybrid game {g}/2 contender: {sits if g == 1 else 0} sits in 4 triggered of "
                        f"{searches // 2} searches, x\n")
    rm.collect(root)
    r = read_json(os.path.join(root, "metrics", "it1.json"))
    assert r["search"]["sit_gate_rate"] == pytest.approx(8 / 200) and "search.sit_gate_rate" not in r["unavailable"]


def test_concurrent_collectors(tmp_path, monkeypatch):
    """Monitor jobs, the watchdog and the panel can collect one run at once: none fails, alerts.tsv has no dupes."""
    root, bin_dir = make_run(tmp_path)
    env = dict(os.environ, PATH=bin_dir + os.pathsep + os.environ["PATH"])
    for _ in range(3):
        with contextlib.suppress(FileNotFoundError):
            os.remove(os.path.join(root, "metrics", ".cache.json"))
        procs = [subprocess.Popen([sys.executable, SCRIPT, root, "--alerts"], stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE, text=True, env=env) for _ in range(4)]
        for p in procs:
            _, err = p.communicate()
            assert p.returncode == 0, err
    with open(os.path.join(root, "alerts.tsv")) as f:
        assert len(f.read().splitlines()) == 5
    assert not [f for f in os.listdir(os.path.join(root, "metrics")) if f.endswith(".tmp")]


# ---------------------------------------------------------------------------------------------------- search balance

BALANCE = os.path.join(REPO, "cluster", "search_balance.py")
THETA0 = {"f": 0.0, "k": None, "b": 800, "lambda": 0.0, "src": "cap"}
F8K4 = {"f": 0.125, "k": 4, "b": 800, "lambda": 0.0, "src": "cap"}


def controller(*args):
    out = subprocess.run([sys.executable, BALANCE, *map(str, args)], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    return out.stdout


def test_balance_without_prior_holds_theta0(tmp_path, monkeypatch):
    """The controller's real files on the fixture copy: no prior.json, so the plan is frozen at theta0 and the record
    says so; realised.json's per-segment numbers are summed per arm."""
    no_sacct(monkeypatch, tmp_path)
    root = copy_fixture(str(tmp_path / "bal"))
    controller("plan", root, 1, "--segments", 4, "--segment-games", 2)
    controller("realise", root, 1)
    controller("decide", root, 1)
    rm.collect(root)
    r = read_json(os.path.join(root, "metrics", "it1.json"))
    b = r["balance"]
    assert b["mode"] == "frozen" and b["theta_star"] == THETA0 and b["challenger"] is None and b["decided"] is True
    assert b["decision"]["action"] == "none" and b["d"] is None and b["d_interval95"] is None
    assert b["prior"]["status"] == "none" and b["prior"]["text"] == "holding theta0: no measured prior"
    assert b["games"] == {"seg00": 2, "seg01": 2, "seg02": 2, "seg03": 2}
    inc = b["per_arm"]["inc"]
    assert (inc["segments"], inc["games"], inc["records"]) == (4, 8, 1307) and inc["gpu_hours"] is None and b["challenger_segment"] is None
    assert b["overhead"]["frac"] == 0.0 and b["overhead"]["cap"] == 0.1
    assert r["cost"]["balance_overhead_gpu_hours"] == 0 and r["cost"]["selfplay_gpu_h_by_arm"] == {"inc": None}
    # The other sections are untouched by the merge.
    assert r["selfplay"]["games"] == 8 and r["outcome"]["gate_generator"]["games"] == 2


def balance_files(root, n, prior=True):
    """A hand-written iteration with a challenger (f=1/8, k=4 in seg03 twinned with seg00), its decision and realised
    costs, and a prior that cost 6 GPU-hours."""
    bdir = os.path.join(root, "balance")
    os.makedirs(os.path.join(bdir, f"it{n}"), exist_ok=True)
    segs = {f"seg{k:02d}": {"theta": "inc", "seed": 1000 * n + k, "games": 2, "engine_args": []} for k in range(3)}
    segs["seg03"] = {"theta": "chal", "seed": 1000 * n, "games": 1, "twin": "seg00",
                     "engine_args": ["--deep-fraction", "0.125", "--deep-multiplier", "4"]}
    state = {"theta_star": THETA0, "evidence": {"x>y": [{"iter": n, "d": 0.004, "sigma_n": 0.002, "seeds": 2}]},
             "overhead": {"cum_gpu_h": 0.9, "run_gpu_h": 18.0, "frac": 0.05, "this_iter_gpu_h": 0.9},
             "tabu": {}, "history": [{"iter": n, "action": "continue", "challenger": "f=0.125"}]}
    write_json(os.path.join(bdir, f"it{n}.plan.json"),
               {"schema": 1, "iteration": n, "mode": "explore", "reason": "compass step 2 direction f+",
                "theta_star": THETA0, "challenger": F8K4, "challenger_kind": "selfplay", "revert_test": False,
                "segments": segs, "cost": {"c_inc": 1.0, "c_chal": 1.3713}, "probe": {"version": 1, "refresh": True},
                "overhead_planned": {"minitrain": 0.65, "score": 0.05, "probe": 0.55}, "dropped": [],
                "state_in": dict(state, evidence={}, overhead=dict(state["overhead"], cum_gpu_h=0.0))})
    realised = {s: {"games": g["games"], "records": 300, "gpu_s": 1800.0, "visits": 240000, "sit_rate": 0.1,
                    "flag_rate": 0.05} for s, g in segs.items()}
    realised["seg03"].update(deep=40, masked=0, visits=330000)
    write_json(os.path.join(bdir, f"it{n}.json"),
               {"schema": 1, "iteration": n, "mode": "explore", "theta_star_planned": THETA0, "challenger": F8K4,
                "results": {"d": 0.004, "sigma_n": 0.002, "realised": realised, "review": False},
                "decision": {"action": "continue", "reason": "Z=1.4 n=1", "Z": 1.4, "mean": 0.004, "se": 0.002,
                             "n": 1, "delta_min": 0.003},
                "state": state, "main_mix": {"inc": 0.75, "chal": 0.25}})
    if prior:
        write_json(os.path.join(bdir, "prior.json"), {"sigma_hat": 0.002, "tau_hat": 0.001, "tau_df": 3,
                                                      "score_stage": True, "cost_gpu_h": 6.0})


def test_balance_challenger_cost_and_cache(tmp_path, monkeypatch):
    root, bin_dir = make_run(tmp_path)
    monkeypatch.setenv("PATH", bin_dir + os.pathsep + os.environ["PATH"])
    first = {r["iteration"]: r for r in rm.collect(root)}
    assert all("balance" not in r for r in first.values())
    gpu = [first[n]["cost"]["gpu_hours"] for n in (1, 2, 3, 4)]
    # The controller is switched on from it3 (the it1/it2 files stay as they were).
    balance_files(root, 3)
    recs = {r["iteration"]: r for r in rm.collect(root)}
    assert "balance" not in recs[2] and "balance" not in recs[1]
    b = recs[3]["balance"]
    assert b["challenger_key"] == "f=0.125,k=4,b=800,l=0,src=cap" and b["decision"]["action"] == "continue"
    assert b["d_interval95"] == pytest.approx([0.004 - 1.96 * 0.002, 0.004 + 1.96 * 0.002])
    assert b["per_arm"]["chal"]["games"] == 1 and b["per_arm"]["inc"]["segments"] == 3
    assert b["per_arm"]["chal"]["games_per_gpu_h"] == 2.0 and b["per_arm"]["inc"]["games_per_gpu_h"] == 4.0
    assert b["per_arm"]["chal"]["deep_share"] == pytest.approx(40 / 300, abs=1e-6)
    assert b["per_arm"]["chal"]["visits_per_s"] == pytest.approx(330000 / 1800)
    assert b["overhead"]["frac"] == 0.05 and b["overhead"]["planned_gpu_h"] == 1.25
    assert b["evidence"] == [{"pair": "x>y", "n": 1, "mean": 0.004, "d": [0.004], "iters": [3]}]
    assert b["prior"]["status"] == "ready" and b["prior"]["masked"]  # f>0 still masked: bias not measured
    assert b["main_mix"] == {"inc": 0.75, "chal": 0.25}
    # Cost: in-window overhead is already in gpu_hours; the prior job's 6 GPU-h land once, on the first balance it.
    c3, c4 = recs[3]["cost"], recs[4]["cost"]
    assert c3["gpu_hours"] == pytest.approx(gpu[2]) and c3["balance_overhead_gpu_hours"] == 0.9
    assert c3["balance_prior_gpu_hours"] == 6.0 and c3["gpu_hours_total"] == pytest.approx(gpu[2] + 6.0)
    assert c3["selfplay_gpu_h_by_arm"] == {"inc": 1.5, "chal": 0.5}
    assert c3["gpu_hours_cum"] == pytest.approx(sum(gpu[:3]) + 6.0, abs=1e-3)
    assert c3["elo_per_gpu_hour"] == pytest.approx((rm.elo(SCORES[3]) - rm.elo(SCORES[2])) / (gpu[2] + 6.0), abs=1e-2)
    assert "balance" not in recs[4] and c4["gpu_hours_cum"] == pytest.approx(sum(gpu) + 6.0, abs=1e-3)
    assert read_json(os.path.join(root, "metrics", "it3.json"))["balance"]["mode"] == "explore"
    # A balance file written later refreshes the cached finished iteration.
    path = os.path.join(root, "balance", "it3.json")
    rec = read_json(path)
    rec["decision"]["action"] = "promote"
    write_json(path, rec)
    again = {r["iteration"]: r for r in rm.collect(root)}
    assert again[3]["balance"]["decision"]["action"] == "promote"


def test_balance_prior_blocked_and_other_incumbent(tmp_path, monkeypatch):
    no_sacct(monkeypatch, tmp_path)
    root = copy_fixture(str(tmp_path / "blk"))
    balance_files(root, 1, prior=False)
    write_json(os.path.join(root, "balance", "prior.json"), {"sigma_hat": 0.002})  # measured, no score stage
    plan = read_json(os.path.join(root, "balance", "it1.plan.json"))
    write_json(os.path.join(root, "balance", "it1.plan.json"), dict(plan, theta_star=F8K4, challenger=None))
    rm.collect(root)
    p = read_json(os.path.join(root, "metrics", "it1.json"))["balance"]["prior"]
    assert p["status"] == "blocked" and p["text"].startswith("holding f=0.125,k=4,b=800,l=0,src=cap: prior present")
