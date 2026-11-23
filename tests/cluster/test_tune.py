"""cluster/tune_status.py, cluster/tune_round.py, cluster/tune_match.sh (dry run) and cluster/tune_space.json: pairing
and scoring of a round, the GSPRTs, candidate writing (oat, grid, spsa step, combine), that every candidate's options
are accepted by the arena, and that the dry run submits and creates nothing.
    .venv/bin/python -m pytest tests/cluster/test_tune.py -q
"""

import json
import math
import os
import random
import subprocess
import sys

import pytest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "cluster"))
sys.path.insert(0, REPO)

import tune_round as tr  # noqa: E402
import tune_status as ts  # noqa: E402

SPACE = os.path.join(REPO, "cluster", "tune_space.json")


def write_row(part_dir, pair_points, finished=True, errors=()):
    """An arena row with one colour-swapped pair per entry of pair_points (0, 0.5, 1, 1.5 or 2 contender points)."""
    os.makedirs(part_dir, exist_ok=True)
    lines = []
    # results of (game 1: contender T1, game 2: contender T2) giving the pair's contender points
    results = {2: ("T1", "T2"), 1.5: ("T1", "draw"), 1: ("T1", "T1"), 0.5: ("draw", "T1"), 0: ("T2", "T1")}
    for k, pts in enumerate(pair_points):
        r = results[pts]
        for j, (ct, res) in enumerate((("T1", r[0]), ("T2", r[1]))):
            g = {"game_id": f"p{2 * k + j + 1:05d}", "pair_index": k, "contender_team": ct, "result": res,
                 "opponent": "upstream-it13"}
            if (k, j) in errors:
                g["error"] = "boom"
            lines.append(json.dumps(g))
    with open(os.path.join(part_dir, "results.jsonl"), "w") as f:
        f.write("\n".join(lines) + "\n")
    if finished:
        with open(os.path.join(part_dir, "summary.json"), "w") as f:
            json.dump({"games": []}, f)


def make_round(tmp_path, rows, mode="oat"):
    rnd = tmp_path / "r1"
    for name, parts in rows.items():
        for k, pts in enumerate(parts, 1):
            write_row(str(rnd / name / f"part{k}"), pts)
    (rnd / "round.json").write_text(json.dumps({"name": "r1", "mode": mode, "baseline": "base",
                                                "candidates": [{"name": n, "params": {}} for n in rows]}))
    return str(rnd)


# ----------------------------------------------------------------------------------------------- statistics
def test_llr_functions_match_the_fishtest_formulas():
    # rl_loop.py sprt_llr on 60-30-10 for [0, 20]: n (s1 - s0)(2 mean - s0 - s1) / (2 var)
    w, d, l = 60, 10, 30
    n = 100
    mean = (w + d / 2) / n
    var = w / n + d / n / 4 - mean ** 2
    s0, s1 = 0.5, 1 / (1 + 10 ** (-20 / 400))
    want = n * (s1 - s0) * (2 * mean - s0 - s1) / (2 * var)
    assert ts.llr_trinomial(w, d, l, 0, 20) == pytest.approx(want, rel=1e-4)
    assert ts.llr_trinomial(0, 0, 0, 0, 20) == 0.0
    lo, hi = ts.sprt_bounds(0.05, 0.05)
    assert lo == pytest.approx(-2.944, abs=1e-3) and hi == pytest.approx(2.944, abs=1e-3)
    # pentanomial: a sample at exactly elo1's score has positive LLR, at elo0's negative, symmetric
    assert ts.llr_pentanomial([10, 0, 80, 0, 10], 0, 20) < 0
    assert ts.llr_pentanomial([5, 0, 80, 0, 15], 0, 20) > 0
    assert ts.score_elo(0.5) == 0 and ts.score_elo(0.64) == pytest.approx(100, abs=1)


def test_status_pairs_candidates_on_shared_pairs_and_scores(tmp_path):
    base = [[1] * 40 + [2] * 10 + [0] * 10]          # 0.5833
    good = [[1] * 40 + [2] * 20]                     # same openings, the 10 losses turned into wins
    short = [[1] * 30]                               # a part still running: 30 pairs only
    rnd = make_round(tmp_path, {"base": base, "good": good, "short": short})
    st = ts.status(rnd)
    rows = {r["name"]: r for r in st["candidates"]}
    assert rows["base"]["vs_opponent"]["pentanomial"] == [10, 0, 40, 0, 10]
    assert rows["base"]["vs_opponent"]["games"] == 120
    assert rows["good"]["vs_opponent"]["score"] == pytest.approx(80 / 120)
    vb = rows["good"]["vs_baseline"]
    assert vb["shared_pairs"] == 60 and vb["dscore"] == pytest.approx(10 / 60)
    assert vb["delo"] > 0 and vb["delo_ci95"][0] > 0 and vb["llr"] > 0
    assert rows["short"]["vs_baseline"]["shared_pairs"] == 30
    assert rows["short"]["vs_baseline"]["dscore"] == pytest.approx(0.0)
    assert "vs_baseline" not in rows["base"]
    table = ts.format_table(st)
    assert table.splitlines()[2].startswith("good")  # ranked by dElo
    assert "(baseline)" in table


def test_errors_and_half_pairs_are_left_out(tmp_path):
    d = tmp_path / "c" / "part1"
    write_row(str(d), [2, 2, 0], errors={(1, 0)})
    c = ts.load_candidate(str(tmp_path / "c"))
    assert c["errors"] == 1 and c["games"] == 5 and len(c["pairs"]) == 2
    # a line being written is skipped
    with open(d / "results.jsonl", "a") as f:
        f.write('{"game_id": "p00007", "pair_ind')
    assert ts.load_candidate(str(tmp_path / "c"))["games"] == 5


def test_decisions_and_cancel_lines(tmp_path):
    base = [[1] * 200]
    bad = [[1] * 100 + [0] * 100]   # loses half the openings the baseline split: clearly worse
    rnd = make_round(tmp_path, {"base": base, "bad": bad})
    for k in (1,):
        os.remove(os.path.join(rnd, "bad", f"part{k}", "summary.json"))  # still running
    st = ts.status(rnd)
    bad_row = next(r for r in st["candidates"] if r["name"] == "bad")
    assert bad_row["vs_baseline"]["decision"] == "H0" and bad_row["vs_opponent"]["decision"] == "H0"
    assert ts.stoppable(st) == ["bad"]
    st["mode"] = "spsa"
    assert ts.stoppable(st) == []
    out = subprocess.run([sys.executable, os.path.join(REPO, "cluster", "tune_status.py"), rnd, "--cancel"],
                         capture_output=True, text=True, check=True).stdout
    assert 'scancel -u "$USER" --name=tune_r1_bad_p1' in out
    # no decision before MIN_GAMES
    assert ts.decide(10.0, ts.MIN_GAMES - 1, 0.05, 0.05) == "continue"


def test_round_without_round_json(tmp_path):
    rnd = tmp_path / "x"
    write_row(str(rnd / "base" / "part1"), [1, 2])
    write_row(str(rnd / "other" / "part1"), [0, 2])
    st = ts.status(str(rnd))
    assert st["baseline"] == "base" and {r["name"] for r in st["candidates"]} == {"base", "other"}


# ----------------------------------------------------------------------------------------------- the space
def test_space_file_is_valid_and_its_values_are_the_engine_defaults():
    space = tr.load_space(SPACE)
    by = {p["name"]: p for p in space["params"]}
    assert space["base"] == {"durative_tree": False, "sit_gate": True, "hybrid": True}
    # player options are known to the arena's wave2 team
    from arena.players.upstream import CONTROLLER_KW, WAVE2_CONTROLLER_KW, WAVE2_FLAGS
    for p in space["params"]:
        if p["where"] == "player":
            assert p["name"] in (*CONTROLLER_KW, *WAVE2_CONTROLLER_KW, *WAVE2_FLAGS), p["name"]
    # engine options exist in the fork's UCI handler with these defaults
    uci = open(os.path.join(REPO, "third_party", "wave2-hivemind", "engine", "src", "interface", "uci.cc")).read()
    for p in space["params"]:
        if p["where"] == "engine":
            assert f'option name {p["name"]} type' in uci, p["name"]
            assert f'name == "{p["name"]}"' in uci, p["name"]
    pool = json.load(open(os.path.join(REPO, "arena", "pools", "wave2.json")))["contender"]["player_kw"]
    assert by["clock_k"]["value"] == pool["clock_k"]
    assert by["root_min_ms"]["value"] == pool["root_min_ms"] and by["root_lag_ms"]["value"] == pool["root_lag_ms"]


def test_baseline_sends_the_hybrid_options_only():
    space = tr.load_space(SPACE)
    kw = tr.candidate_kw(space, {}, "fixed")
    assert "engine_options" not in kw["contender_kw"] and kw["changed"] == {}
    assert kw["contender_kw"]["ms_per_node"] == space["charge"]["contender_ms_per_node"]
    assert kw["opponent_kw"] == {"ms_per_node": space["charge"]["opponent_ms_per_node"]}
    from arena.players.upstream import wave2_flag_options
    flags = {k: v for k, v in kw["contender_kw"].items()
             if k in ("durative_tree", "sit_gate", "hybrid", "pass_cap_k") or k.startswith("hybrid_")}
    assert wave2_flag_options(**flags) == wave2_flag_options(durative_tree=False, sit_gate=True, hybrid=True)
    m = tr.candidate_kw(space, {"CPUCTInitPermille": 2500}, "measure")
    assert m["contender_kw"]["engine_options"] == {"CPUCTInitPermille": 2500}
    assert "ms_per_node" not in m["contender_kw"] and m["opponent_kw"] == {}


def test_every_oat_candidate_is_accepted_by_the_arena(tmp_path):
    from arena.pool import Pool, for_evaluation, make_schedule, unbuildable
    from arena.players.upstream import wave2_flag_options, WAVE2_FLAGS, WAVE2_PRESETS

    space = tr.load_space(SPACE)
    names = [p["name"] for p in space["params"]]
    cands = tr.oat_values(space, names, 1.0)
    assert len(cands) > 1 + len(names)
    pool = Pool.load(os.path.join(REPO, "arena", "pools", "wave2.json"))
    for name, values in cands:
        kw = tr.candidate_kw(space, values, "fixed")
        ev = for_evaluation(pool, opponent="upstream-it13", contender_kw=kw["contender_kw"],
                            opponent_kw=kw["opponent_kw"])
        assert unbuildable(make_schedule(ev, 1, openings="bughouse", rules="v3")) == [], name
        ckw = ev.contender.player_kw
        wave2_flag_options(**{k: ckw.get(k) for k in (*WAVE2_FLAGS, *WAVE2_PRESETS)})  # raises on a bad value
        assert ev.opponents[0].player_kw["ms_per_node"] == space["charge"]["opponent_ms_per_node"]


def test_oat_steps_clamps_and_log_scale():
    space = tr.load_space(SPACE)
    by = dict(tr.oat_values(space, ["clock_k", "min_nodes", "PUCTWidening", "hybrid_event_lands"], 1.0))
    assert by["clock_k_155.563"]["clock_k"] == pytest.approx(110 * 2 ** 0.5, abs=1e-3)
    assert by["clock_k_77.7817"]["clock_k"] == pytest.approx(110 / 2 ** 0.5, abs=1e-3)
    assert "min_nodes_116" in by and not any(k.startswith("min_nodes_-") for k in by)  # 16 - 100 clamps to 16
    assert by["PUCTWidening_false"] == {"PUCTWidening": False}
    assert set(by) >= {"base", "hybrid_event_lands_sit", "hybrid_event_lands_none"}
    p = {"name": "x", "type": "float", "value": 100, "min": 10, "max": 1000, "step": 1, "scale": "log2"}
    assert tr.in_steps(p, tr.shift(p, 100, 2.5), 100) == pytest.approx(2.5)


def run_round(tmp_path, *argv):
    return tr.main([*argv, "--runs", str(tmp_path / "tune")])


def test_oat_and_grid_rounds_write_candidates_and_launch_lines(tmp_path, capsys):
    assert run_round(tmp_path, "oat", "--name", "r1", "--params", "clock_k,CPUCTInitPermille") == 0
    out = capsys.readouterr().out
    rnd = json.load(open(tmp_path / "tune" / "r1" / "round.json"))
    assert rnd["baseline"] == "base" and len(rnd["candidates"]) == 5
    assert rnd["schedule"] == {"parts": 2, "pairs_per_part": 150, "seed_base": 5000, "base_ms": 120000,
                               "time": "08:00:00"}
    assert out.count("bash cluster/tune_match.sh") == 5 and "GPU-hours" in out
    launch = (tmp_path / "tune" / "r1" / "launch.sh").read_text()
    assert launch.count("tune_match.sh") == 5 and "SEED_BASE=5000" in launch
    c = json.load(open(tmp_path / "tune" / "r1" / "candidates" / "CPUCTInitPermille_3250.json"))
    assert c["contender_kw"]["engine_options"] == {"CPUCTInitPermille": 3250}
    e = ts.final_match_engine(str(tmp_path / "tune" / "r1"), "CPUCTInitPermille_3250")
    assert e["kind"] == "wave2" and e["model"] is None and "ms_per_node" not in e["options"]
    assert e["options"]["engine_options"] == {"CPUCTInitPermille": 3250} and e["options"]["hybrid"] is True
    with pytest.raises(SystemExit):  # a round is written once
        run_round(tmp_path, "oat", "--name", "r1")
    assert run_round(tmp_path, "grid", "--name", "g1", "--set", "clock_k=55,110,220,1000",
                     "--set", "PUCTWidening=false", "--set", "hybrid_event_lands=none") == 0
    g = json.load(open(tmp_path / "tune" / "g1" / "round.json"))
    assert [c["name"] for c in g["candidates"]] == ["base", "clock_k_55", "clock_k_220", "clock_k_1000",
                                                    "PUCTWidening_false", "hybrid_event_lands_none"]
    for bad in ("clock_k=5000", "nope=1", "clock_k"):
        with pytest.raises(SystemExit):
            run_round(tmp_path, "grid", "--name", "g2", "--set", bad)


def test_spsa_step_moves_theta_up_the_measured_gradient(tmp_path, capsys):
    assert run_round(tmp_path, "spsa", "--name", "s1", "--params", "CPUCTInitPermille,PolicyTemperaturePermille",
                     "--k", "4", "--seed", "1") == 0
    s1 = tmp_path / "tune" / "s1"
    rnd = json.load(open(s1 / "round.json"))
    assert len(rnd["candidates"]) == 8 and rnd["baseline"] is None
    vals = {c["name"]: c["values"] for c in rnd["candidates"]}
    # synthetic truth: higher CPUCT wins, temperature does nothing. plus/minus play the same openings.
    rng = random.Random(0)
    for pr in rnd["spsa"]["pairs"]:
        for side in ("plus", "minus"):
            v = vals[pr[side]]["CPUCTInitPermille"]
            p_win = 0.5 + (v - 3000) / 4000
            pts = [sum(rng.random() < p_win for _ in range(2)) for _ in range(300)]
            write_row(str(s1 / pr[side] / "part1"), pts)
    capsys.readouterr()
    assert run_round(tmp_path, "spsa", "--name", "s2", "--prev", str(s1), "--k", "4") == 0
    s2 = json.load(open(tmp_path / "tune" / "s2" / "round.json"))
    step = s2["step_from"]
    assert step["moves_in_steps"]["CPUCTInitPermille"] > 0.5
    assert abs(step["moves_in_steps"]["PolicyTemperaturePermille"]) < step["moves_in_steps"]["CPUCTInitPermille"]
    assert s2["spsa"]["theta"]["CPUCTInitPermille"] > 3000
    space2 = json.load(open(tmp_path / "tune" / "s2" / "space.json"))
    cp = next(p for p in space2["params"] if p["name"] == "CPUCTInitPermille")
    assert cp["value"] > 3000 and cp["default"] == 3000  # the engine default stays known: options sent vs it


def test_combine_takes_the_sides_that_beat_the_baseline(tmp_path, capsys):
    assert run_round(tmp_path, "oat", "--name", "r1", "--params", "CPUCTInitPermille,PolicyTemperaturePermille") == 0
    r1 = tmp_path / "tune" / "r1"
    write_row(str(r1 / "base" / "part1"), [1] * 200)
    write_row(str(r1 / "CPUCTInitPermille_3250" / "part1"), [1] * 150 + [2] * 50)
    write_row(str(r1 / "CPUCTInitPermille_2750" / "part1"), [1] * 150 + [0] * 50)
    write_row(str(r1 / "PolicyTemperaturePermille_1100" / "part1"), [1] * 200)
    write_row(str(r1 / "PolicyTemperaturePermille_900" / "part1"), [1] * 190 + [0] * 10)
    assert run_round(tmp_path, "combine", "--name", "r2", "--prev", str(r1)) == 0
    r2 = json.load(open(tmp_path / "tune" / "r2" / "round.json"))
    assert r2["taken"].keys() == {"CPUCTInitPermille"}
    combined = json.load(open(tmp_path / "tune" / "r2" / "candidates" / "combined.json"))
    assert combined["contender_kw"]["engine_options"] == {"CPUCTInitPermille": 3250}
    nxt = json.load(open(tmp_path / "tune" / "r2" / "space_next.json"))
    cp = next(p for p in nxt["params"] if p["name"] == "CPUCTInitPermille")
    assert cp["value"] == 3250 and cp["default"] == 3000
    # the next round's baseline (from space_next) sends the combined value
    assert tr.candidate_kw(nxt, {}, "fixed")["contender_kw"]["engine_options"] == {"CPUCTInitPermille": 3250}


def test_tune_match_dry_run_prints_the_final_match_settings_and_creates_nothing(tmp_path):
    cand = tmp_path / "cand.json"
    space = tr.load_space(SPACE)
    kw = tr.candidate_kw(space, {"clock_k": 220.0}, "fixed")
    cand.write_text(json.dumps({"name": "clock_k_220", "round": "t", **kw}))
    env = {**os.environ, "DRY_RUN": "1", "NAME": "t_dry", "CAND": str(cand), "PARTS": "3", "TUNE_RUNS": "runs/tune"}
    r = subprocess.run(["bash", os.path.join(REPO, "cluster", "tune_match.sh")], cwd=REPO, env=env,
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    lines = [l for l in r.stdout.splitlines() if "sbatch" in l]
    assert len(lines) == 3
    pin = open(os.path.join(REPO, "cluster", "wave2_engine.ref")).read().strip()
    for k, line in enumerate(lines, 1):
        assert f"-J tune_t_dry_clock_k_220_p{k} cluster/eval.sbatch" in line
        assert f"SEED={5000 + k} " in line and "BASE_MS=120000" in line and f"ENGINE_REF={pin}" in line
        for s in ("POOL=arena/pools/wave2.json", "SETUP=pool", "OPENINGS=bughouse", "RULES=v3", "MEASURE=1",
                  "OPPONENT=upstream-it13", "PAIRS=150", "-t 08:00:00"):
            assert s in line
    assert not os.path.exists(os.path.join(REPO, "runs", "tune", "t_dry"))
    bad = subprocess.run(["bash", os.path.join(REPO, "cluster", "tune_match.sh")], cwd=REPO,
                         env={**env, "TUNE_RUNS": "/tmp/x"}, capture_output=True, text=True)
    assert bad.returncode != 0 and "under runs/" in bad.stderr
