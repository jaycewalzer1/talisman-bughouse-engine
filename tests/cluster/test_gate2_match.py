"""cluster/gate2_match.sh: the real-clock two-score gate's matches for one network (docs/custom_engine_spec_v7_league.md
3.3), against a fake Slurm (tests/cluster/fake_slurm). Nothing runs: jobs are only queued, and finished parts are
written by hand. The script runs from a scratch copy of the repo (its cluster/ script, symlinks to .venv, arena and the
wave2-hivemind checkout) so its runs/ is the test's own.
    .venv/bin/python -m pytest tests/cluster/test_gate2_match.py -q
"""

import json
import os
import shutil
import subprocess

import pytest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FAKE = os.path.join(REPO, "tests", "cluster", "fake_slurm")
W2 = os.path.join(REPO, "third_party", "wave2-hivemind")
HYBRID = {"durative_tree": False, "sit_gate": True, "flag_loss": True, "instant_transfer": False,
          "mate_race_by_landing": False, "pass_cap_k": 8, "base_mix": "60000:0.25,120000:0.25,180000:0.35,300000:0.15",
          "seat_move_times": True, "team_tree_reuse": False, "team_tree_reuse_tolerance_ms": 250, "joint_plans": False,
          "joint_plan_free_permille": 500, "plan_carry": False, "plan_carry_permille": 300,
          "own_pace_below_root": False, "wait_node_acting_eval": False, "blockable_check_not_mate": False,
          "hybrid_mode": "gate", "hybrid_event_lands": "all", "hybrid_max_wait_ms": 1000, "hybrid_min_clock_ms": 3000,
          "hybrid_margin_permille": 20, "hybrid_pass_prior_permille": 150}

pytestmark = pytest.mark.skipif(not os.path.exists(os.path.join(REPO, ".venv", "bin", "python")),
                                reason="needs the repo's .venv")


class Gate:
    def __init__(self, tmp):
        self.repo = tmp / "repo"
        (self.repo / "cluster").mkdir(parents=True)
        (self.repo / "third_party").mkdir()
        shutil.copy(os.path.join(REPO, "cluster", "gate2_match.sh"), self.repo / "cluster")
        shutil.copy(os.path.join(REPO, "cluster", "wave2_engine.ref"), self.repo / "cluster")
        for name in (".venv", "arena"):
            os.symlink(os.path.join(REPO, name), self.repo / name)
        os.symlink(W2, self.repo / "third_party" / "wave2-hivemind")
        self.root = self.repo / "runs" / "full" / "g1"
        (self.root / "gate2").mkdir(parents=True)
        (self.root / "gate2" / "wave2.json").write_text(json.dumps(HYBRID))
        head = subprocess.run(["git", "-C", W2, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        self.head = head
        (self.root / "launch_settings.txt").write_text(f"run g1  engine {head}  config hybrid\n")
        self.onnx = tmp / "C.onnx"
        self.onnx.write_bytes(b"candidate network")
        import hashlib
        self.sha = hashlib.sha256(b"candidate network").hexdigest()
        self.net = self.root / "gate2" / "nets" / self.sha[:16]
        self.q = tmp / "q"
        self.q.mkdir()
        self.env = {**os.environ, "FAKE_SLURM_Q": str(self.q), "PATH": f"{FAKE}:{os.environ['PATH']}",
                    "USER": os.environ.get("USER", "me")}
        for key in list(self.env):
            if key.startswith("GATE2_") or key in ("DRY_RUN", "NO_SUBMIT", "MAX_TRIES"):
                del self.env[key]

    def call(self, **settings):
        env = {**self.env, **{k: str(v) for k, v in settings.items()}}
        return subprocess.run(["bash", str(self.repo / "cluster" / "gate2_match.sh"), str(self.root), "it1-cand",
                               str(self.onnx)], env=env, capture_output=True, text=True, cwd="/")

    def jobs(self):
        return sorted((self.q / name).read_text().strip() for name in os.listdir(self.q) if name.endswith(".name"))

    def env_of(self, name):
        n = next(f[:-5] for f in os.listdir(self.q) if f.endswith(".name") and (self.q / f).read_text().strip() == name)
        env = {}
        for line in (self.q / f"{n}.env").read_text().splitlines():
            if line.startswith("declare -x ") and "=" in line:
                key, _, value = line[11:].partition("=")
                env[key] = value.strip('"').replace('\\"', '"')
        return env

    def finish(self, arm, part, pairs=50, contender_wins=lambda i, g: True, summary=True):
        """A finished eval row: `pairs` colour-swapped pairs, pair i's games p(2i+1), p(2i+2)."""
        d = self.net / arm / f"part{part}"
        d.mkdir(parents=True, exist_ok=True)
        rows = []
        for i in range(pairs):
            for g in range(2):
                ct = "T1" if g == 0 else "T2"
                win = contender_wins(i, g)
                rows.append({"game_id": f"p{2 * i + g + 1:05d}", "pair_index": i, "opening_id": 100 + i,
                             "contender_team": ct, "result": ct if win else ("T2" if ct == "T1" else "T1"),
                             "clock_teams": {ct: {"turns_ahead": 3, "turns_behind": 1},
                                             ("T2" if ct == "T1" else "T1"): {"turns_ahead": 1, "turns_behind": 3}},
                             "think": {ct: {"think_head_missing": 1}}})
        (d / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        if summary:
            (d / "summary.json").write_text(json.dumps({"games": rows, "ms_per_node": {
                "wave2|x": {"ms_per_node": 0.05}, "upstream|y": {"ms_per_node": 0.04}}}))


@pytest.fixture
def gate(tmp_path):
    return Gate(tmp_path)


def test_dry_run_prints_the_sbatch_lines(gate):
    r = gate.call(DRY_RUN=1)
    assert r.returncode == 0, r.stderr
    lines = [line for line in r.stdout.splitlines() if line.startswith("DRY_RUN")]
    assert len(lines) == 8
    sha8 = gate.sha[:8]
    for arm in ("equal", "free"):
        for k in range(1, 5):
            (line,) = [line for line in lines if f"-J g2_g1_{sha8}_{arm}_p{k} " in line]
            assert f"SEED={7000 + k} " in line and " PAIRS=50 " in line and " BASE_MS=60000 " in line
            assert "OPPONENT=upstream-it13" in line and f"ENGINE_REF={gate.head}" in line
            assert "POOL=arena/pools/wave2.json SETUP=pool OPENINGS=bughouse RULES=v3 MEASURE=1 MEASURE_NODES=800" in line
            assert "WARM_KINDS=upstream\\ wave2 MEASURE_WARM_MS=3000 MEASURE_REPEATS=10" in line
            assert f"NAME=full/g1/gate2/nets/{gate.sha[:16]}/{arm}/part{k} " in line
            assert "-t 05:00:00" in line and line.endswith("cluster/eval.sbatch")
    config = json.loads((gate.net / "config.json").read_text())
    assert config["settings"] == {"games": 400, "parts": 4, "base_ms": 60000, "free_k": 160, "equal_k": 110,
                                  "seed_base": 7000, "opp_sit": "observed", "arms": ["equal", "free"]}
    model = str((gate.net / "model.onnx").resolve()).replace(str(gate.repo.resolve()), str(gate.repo))
    kw = config["contender_kw"]
    assert kw["equal"] == {"durative_tree": False, "sit_gate": True, "hybrid_mode": "gate", "model": kw["equal"]["model"],
                           "opp_sit_pace": "observed", "clock_k": 110}
    assert kw["free"] == {**kw["equal"], "clock_k": 160, "think_head": True}
    assert kw["equal"]["model"].endswith(f"nets/{gate.sha[:16]}/model.onnx") and model
    assert (gate.net / "model.onnx").read_bytes() == b"candidate network"
    assert config["label"] == "it1-cand" and config["sha256"] == gate.sha and config["opponent"] == "upstream-it13"
    assert gate.jobs() == [] and (gate.net / "jobs.tsv").read_text() == ""


def test_settings_from_the_environment(gate):
    r = gate.call(DRY_RUN=1, GATE2_GAMES=16, GATE2_PARTS=2, GATE2_BASE_MS=120000, GATE2_FREE_K=200,
                  GATE2_SEED_BASE=100, GATE2_ARMS="equal", GATE2_TIME="01:00:00", GATE2_OPP_SIT="own_k")
    assert r.returncode == 0, r.stderr
    lines = [line for line in r.stdout.splitlines() if line.startswith("DRY_RUN")]
    assert len(lines) == 2 and all("_equal_p" in line for line in lines)
    assert "SEED=101 " in lines[0] and "SEED=102 " in lines[1] and " PAIRS=4 " in lines[0]
    assert "BASE_MS=120000" in lines[0] and "-t 01:00:00" in lines[0] and "own_k" in lines[0]
    assert gate.call(GATE2_GAMES=15, DRY_RUN=1).returncode == 2
    assert gate.call(GATE2_OPP_SIT="fixed", DRY_RUN=1).returncode == 2


def test_submits_once_and_merges_finished_parts(gate):
    r = gate.call()
    assert r.returncode == 0, r.stderr
    names = gate.jobs()
    assert len(names) == 8 and f"g2_g1_{gate.sha[:8]}_free_p3" in names
    env = gate.env_of(f"g2_g1_{gate.sha[:8]}_free_p3")
    assert env["SEED"] == "7003" and env["PAIRS"] == "50" and env["WARM_KINDS"] == "upstream wave2"
    assert json.loads(env["CONTENDER_KW"])["think_head"] is True
    assert r.stdout.count("submitted") == 8
    assert len((gate.net / "jobs.tsv").read_text().splitlines()) == 8
    # Idempotent: every part is queued, so nothing to do and nothing printed.
    again = gate.call()
    assert again.returncode == 0 and again.stdout == "" and len(gate.jobs()) == 8
    # All parts finish: equal wins every game, free loses the second game of every 4th pair.
    for k in range(1, 5):
        gate.finish("equal", k)
        gate.finish("free", k, contender_wins=lambda i, g: not (i % 4 == 0 and g == 1))
    for name in os.listdir(gate.q):
        if name.endswith(".state"):
            (gate.q / name).write_text("COMPLETED\n")
    r = gate.call()
    assert r.returncode == 0, r.stderr
    assert "gate2 it1-cand equal: 400 games, 200 pairs, score 1.000" in r.stdout
    assert (gate.net / "done").is_file()
    s = json.loads((gate.net / "free" / "summary.json").read_text())
    assert s["games"] == 400 and s["pairs"] == 200 and s["losses"] == 52 and s["score"] == pytest.approx(348 / 400)
    assert s["turns_ahead"] == pytest.approx(0.75) and s["think_head_missing"] == 400
    assert s["ms_per_node"]["part1"] == {"wave2|x": 0.05, "upstream|y": 0.04}
    assert s["contender_kw"]["clock_k"] == 160 and s["sha16"] == gate.sha[:16] and s["gave_up"] == []
    lo, hi = s["score_ci"]
    assert lo < 0.87 < hi
    rows = (gate.net / "free" / "pairs.tsv").read_text().splitlines()
    assert rows[0] == "part\tpair_index\topening_id\tcontender_team\tscore_pair\tgame_ids"
    assert rows[1] == "1\t0\t100\tT1,T2\t0.5\tp00001,p00002" and len(rows) == 201
    # The fork's reader takes the file (loaded by path: another `hivemind`, the old stack's, may be imported).
    import importlib.util
    spec = importlib.util.spec_from_file_location("w2_gate2", os.path.join(W2, "src", "hivemind", "gate2.py"))
    gate2 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate2)
    assert len(gate2.read_pairs(gate.net, "free")) == 200
    assert gate.call().stdout == ""  # done: nothing more


def test_settings_mismatch_is_refused(gate):
    assert gate.call(DRY_RUN=1).returncode == 0
    r = gate.call(DRY_RUN=1, GATE2_GAMES=200)
    assert r.returncode == 1 and "other settings" in r.stderr
    (gate.root / "gate2" / "settings.json").write_text(json.dumps({"games": 400, "parts": 2}))
    r = gate.call(DRY_RUN=1)
    assert r.returncode == 1 and "differ from" in r.stderr


def test_stop_workers_submits_nothing(gate):
    (gate.root / "STOP_WORKERS").touch()
    r = gate.call()
    assert r.returncode == 0 and gate.jobs() == []
    assert r.stdout.count("not submitted") == 8 and "STOP_WORKERS" in r.stdout


def test_a_failed_squeue_merges_only(gate, tmp_path):
    bad = tmp_path / "bad"
    bad.mkdir()
    (bad / "squeue").write_text("#!/bin/bash\nexit 1\n")
    (bad / "squeue").chmod(0o755)
    gate.env["PATH"] = f"{bad}:{gate.env['PATH']}"
    r = gate.call()
    assert r.returncode == 0 and gate.jobs() == [] and "squeue failed" in r.stdout


def test_gives_up_after_max_tries_and_merges_what_it_has(gate):
    assert gate.call(GATE2_ARMS="equal", GATE2_GAMES=8, GATE2_PARTS=2).returncode == 0
    for name in os.listdir(gate.q):
        if name.endswith(".state"):
            (gate.q / name).write_text("FAILED\n")
    gate.finish("equal", 1, pairs=2)
    gate.finish("equal", 2, pairs=1, summary=False)  # cut off: one pair played
    with open(gate.net / "jobs.tsv", "a") as f:
        f.write("equal\tpart2\t9\tx\ty\nequal\tpart2\t10\tx\ty\n")
    r = gate.call(GATE2_ARMS="equal", GATE2_GAMES=8, GATE2_PARTS=2)
    assert "equal part2: 3 jobs and no result; not resubmitted" in r.stdout
    assert (gate.net / "equal" / "part2" / "GAVE_UP").is_file() and len(gate.jobs()) == 2
    r = gate.call(GATE2_ARMS="equal", GATE2_GAMES=8, GATE2_PARTS=2)
    s = json.loads((gate.net / "equal" / "summary.json").read_text())
    assert s["pairs"] == 3 and s["gave_up"] == [2] and (gate.net / "done").is_file()
