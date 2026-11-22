"""cluster/search_balance.py: cost model, compass proposals, plan/decide files, promotion + forced revert test,
rejection, overhead drop order, gamma estimation, failsafe, immutability and delete-and-rerun byte identity, the
fixture's real gate marker, and the simulator's determinism.
    .venv/bin/python -m pytest tests/cluster/test_search_balance.py -q
"""

import json
import os
import shutil
import subprocess
import sys

import pytest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "cluster"))

import search_balance as sb  # noqa: E402

SCRIPT = os.path.join(REPO, "cluster", "search_balance.py")
FIXTURE = os.path.join(REPO, "runs", "wave2", "int-shared")
# What a prior holds once the score stage exists, the probe bias and the between-data noise were measured and the
# trainer has --search-q-blend; without these the controller holds theta0 (see test_default_prior_holds_theta0).
OPEN = {"score_stage": True, "probe_bias_measured": True, "search_q_blend": True, "tau_hat": 0.0, "tau_df": 9}


class Args:
    def __init__(self, **kw):
        self.prior = None
        self.generator = None
        self.segments = 5
        self.segment_games = 2000
        self.mini_steps = 200
        self.run_seed = 0
        self.dry = False
        self.json = False
        self.__dict__.update(kw)


def write(path, value):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(value, f)


def read(path):
    with open(path) as f:
        return json.load(f)


def scores(d, base=2.1, spread=0.0):
    return {"scores": {"inc_s0": base, "chal_s0": base - d - spread, "inc_s1": base, "chal_s1": base - d + spread}}


def plan(run, n, **kw):
    return sb.cmd_plan(str(run), n, Args(**kw))


def decide(run, n, **kw):
    return sb.cmd_decide(str(run), n, Args(**kw))


# ------------------------------------------------------------------------------------------------ pure parts


def test_cost_model():
    assert sb.cost(sb.THETA0) == 1.0
    assert sb.games_for(sb.THETA0, {}, 2000) == 2000
    c = sb.cost(sb.theta(0.125, 4))
    assert abs(c - 1.385 / 1.01) < 1e-9
    assert sb.games_for(sb.theta(0.125, 4), {}, 2000) == round(2000 / c)
    # gamma table raises the deep cost
    assert sb.cost(sb.theta(0.125, 4), {"3200": [1.5]}) > c
    assert sb.gamma_of(800, {"800": [9.0]}) == 1.0


def test_grid_rules():
    assert sb.legal(sb.THETA0, {})
    assert not sb.on_grid(sb.theta(0.0, None, 400))           # all records masked: illegal
    assert not sb.legal(sb.theta(0.125, 8), OPEN)             # k=8 needs the bench
    assert sb.legal(sb.theta(0.125, 8), dict(OPEN, fits_bench={"8": True}))
    assert not sb.legal(sb.theta(0.125, 4, src="reanalyse"), OPEN)
    # deep targets wait for the probe-bias measurement, lambda for the trainer's --search-q-blend
    assert not sb.legal(sb.theta(0.125, 4), {}) and sb.legal(sb.theta(0.125, 4), {"probe_bias_measured": True})
    assert not sb.legal(sb.theta(0, None, 800, 0.25), {})
    assert sb.legal(sb.theta(0, None, 800, 0.25), {"search_q_blend": True})
    assert sb.key(sb.theta(0.125, 4)) == "f=0.125,k=4,b=800,l=0,src=cap"


def test_first_challenger_and_engine_args():
    S = sb.init_state(OPEN)
    c = sb.propose(S, 1, OPEN)
    assert sb.key(c) == sb.key(sb.theta(0.125, 4))
    assert sb.engine_args(c) == ["--deep-fraction", "0.125", "--deep-multiplier", "4"]
    assert sb.engine_args(sb.theta(0.25, 4, 200)) == ["--deep-fraction", "0.25", "--deep-multiplier", "4",
                                                       "--fast-nodes", "200"]
    assert sb.engine_args(sb.THETA0) == [] and sb.trainer_args(sb.THETA0) == []
    assert sb.trainer_args(sb.theta(0, None, 800, 0.25)) == ["--search-q-blend", "0.25"]


def test_step_halving_and_maintenance():
    S = sb.init_state(OPEN)
    seen = set()
    for _ in range(40):
        c = sb.propose(S, 1, OPEN)
        if c is None:
            break
        seen.add(sb.key(c))
        S["rejected"].append(sb.pair_key(S["theta_star"], c))
    assert S["mode"] == "maintenance" and S["step"] == 1
    assert sb.key(sb.theta(0.0625, 4)) in seen                # step 1 reached f = 1/16


def test_minigate_p():
    score, p = sb.minigate_p_worse({"games": 100, "wins": 35, "losses": 55, "draws": 10})
    assert score == 0.4 and p < 0.10
    score, p = sb.minigate_p_worse({"games": 100, "wins": 50, "losses": 40, "draws": 10})
    assert p > 0.5


def test_delta_min_calibration_clamped():
    assert sb.delta_min(4.0, [])[0] == 6.0
    dm, src = sb.delta_min(4.0, [[1, 5], [2, 10], [3, 15]])   # beta = 5 Elo per unit -> 1.0, clamped to 0.5 sigma
    assert dm == 2.0 and "calibrated" in src


# ------------------------------------------------------------------------------------------------ files


@pytest.fixture
def run(tmp_path):
    root = tmp_path / "RUN"
    write(root / "balance" / "prior.json", dict(OPEN, sigma_hat=0.002))
    return root


def test_plan_file(run):
    p = plan(run, 1)
    assert os.path.isfile(run / "balance" / "it1.plan.json")
    assert p["mode"] == "explore" and p["challenger_kind"] == "selfplay"
    segs = p["segments"]
    assert sorted(segs) == [f"seg{k:02d}" for k in range(5)]
    assert segs["seg04"]["theta"] == "chal" and segs["seg04"]["twin"] == "seg00"
    assert segs["seg04"]["seed"] == segs["seg00"]["seed"] == 1000
    assert segs["seg01"]["seed"] == 1001 and segs["seg01"]["games"] == 2000 and segs["seg01"]["engine_args"] == []
    assert segs["seg04"]["games"] == sb.games_for(sb.theta(0.125, 4), {}, 2000)
    assert p["val"] == {"engine_args": [], "assert_stock": True}
    assert p["minitrain"]["seeds"] == [1007, 1017]
    assert p["probe"]["refresh"] is True and p["probe"]["version"] == 1
    assert p["minigate_allowed"] is True


def test_promote_then_revert_test_then_keep(run):
    plan(run, 1)
    write(run / "balance" / "it1" / "score.json", scores(0.01, spread=0.0005))
    r1 = decide(run, 1)
    # n = 1, Z = 0.01 / (0.002/sqrt2) = 7 >= 3, but no mini-gate yet -> continue
    assert r1["decision"]["action"] == "continue"
    plan(run, 2)
    write(run / "balance" / "it2" / "score.json", scores(0.01, spread=0.0005))
    dry = decide(run, 2, dry=True)
    assert dry["eligible"] and dry["run_minigate"]
    assert not os.path.exists(run / "balance" / "it2.json")
    write(run / "balance" / "it2" / "minigate.json", {"games": 100, "wins": 48, "losses": 42, "draws": 10})
    r2 = decide(run, 2)
    assert r2["decision"]["action"] == "promote"
    assert sb.key(r2["state"]["theta_star"]) == sb.key(sb.theta(0.125, 4))
    p3 = plan(run, 3)
    assert p3["revert_test"] and sb.key(p3["challenger"]) == sb.key(sb.THETA0)
    assert p3["segments"]["seg01"]["engine_args"] == ["--deep-fraction", "0.125", "--deep-multiplier", "4"]
    assert p3["segments"]["seg04"]["engine_args"] == [] and p3["segments"]["seg04"]["games"] == 2000
    write(run / "balance" / "it3" / "score.json", scores(-0.01))      # old recipe clearly worse
    r3 = decide(run, 3)
    assert r3["decision"]["action"] == "keep" and r3["state"]["revert_test"] is False


def test_revert_test_reverts(run):
    plan(run, 1)
    write(run / "balance" / "it1" / "score.json", scores(0.01))
    decide(run, 1)
    plan(run, 2)
    write(run / "balance" / "it2" / "score.json", scores(0.01))
    write(run / "balance" / "it2" / "minigate.json", {"games": 100, "wins": 50, "losses": 40, "draws": 10})
    assert decide(run, 2)["decision"]["action"] == "promote"
    plan(run, 3)
    write(run / "balance" / "it3" / "score.json", scores(0.01))       # old recipe better now
    r3 = decide(run, 3)
    assert r3["decision"]["action"] == "revert"
    assert sb.key(r3["state"]["theta_star"]) == sb.key(sb.THETA0)
    p4 = plan(run, 4)
    assert p4["mode"] == "frozen" and p4["challenger"] is None


def test_two_negative_gates_revert(run):
    plan(run, 1)
    write(run / "balance" / "it1" / "score.json", scores(0.01))
    decide(run, 1)
    plan(run, 2)
    write(run / "balance" / "it2" / "score.json", scores(0.01))
    write(run / "balance" / "it2" / "minigate.json", {"games": 100, "wins": 50, "losses": 40, "draws": 10})
    decide(run, 2)
    for n in (3, 4):
        plan(run, n)
        if read(run / "balance" / f"it{n}.plan.json")["challenger"]:
            write(run / "balance" / f"it{n}" / "score.json", scores(-0.001))
        write(run / "runs" / f"rl-it{n}" / "stages" / "match_generator.done", {"elo": -20.0, "score": 0.47})
        r = decide(run, n)
    assert r["decision"]["action"] == "revert" and "negative main gates" in r["decision"]["reason"]


def test_reject_and_tabu(run):
    plan(run, 1)
    write(run / "balance" / "it1" / "score.json", scores(-0.01))
    r = decide(run, 1)
    assert r["decision"]["action"] == "reject"
    k = sb.key(sb.theta(0.125, 4))
    assert r["state"]["tabu"][k] == 4
    p2 = plan(run, 2)
    assert sb.key(p2["challenger"]) != k
    assert p2["challenger_kind"] in ("selfplay", "lambda")


def test_lambda_challenger_has_no_segment(run):
    S = sb.init_state(dict(OPEN, sigma_hat=0.002))
    S["dir_idx"] = sb.DIRS.index("l+")
    p = sb.plan_core(1, S, {"prior": OPEN})
    assert p["challenger_kind"] == "lambda" and p["challenger_segment"] is None
    assert all(s["theta"] == "inc" for s in p["segments"].values())
    assert p["minitrain"]["arms"]["chal"] == {"data": ["seg00"], "trainer_args": ["--search-q-blend", "0.5"]}


def test_overhead_drop_order():
    S = sb.init_state(OPEN)
    S["overhead"] = {"cum_gpu_h": 2.0, "run_gpu_h": 14.0, "frac": 0.14}
    p = sb.plan_core(5, S, {"prior": OPEN})
    assert p["dropped"] == ["minigate", "second_seed"] and not p["minigate_allowed"]
    assert len(p["minitrain"]["seeds"]) == 1 and p["probe"]["refresh"]     # no probe yet: the refresh is kept
    S["overhead"] = {"cum_gpu_h": 5.0, "run_gpu_h": 20.0, "frac": 0.25}
    p = sb.plan_core(5, S, {"prior": OPEN})
    assert p["challenger"] is None and p["mode"] == "frozen" and "exploration" in p["dropped"]


def test_gamma_and_review_from_realised(run):
    plan(run, 1)
    p = read(run / "balance" / "it1.plan.json")
    games_x = p["segments"]["seg04"]["games"]
    v = sb.V_GPU_S_PER_VISIT
    write(run / "balance" / "it1" / "score.json", scores(0.0))
    write(run / "balance" / "it1" / "realised.json", {
        "seg00": {"gpu_s": v * 1e8, "visits": 1e8, "games": 2000, "records": 290000, "sit_rate": 0.10,
                  "flag_rate": 0.05},
        "seg04": {"gpu_s": v * (5e7 * 1.0 + 4e7 * 1.2), "games": games_x, "records": 209000, "sit_rate": 0.15,
                  "flag_rate": 0.05}})
    write(run / "runs" / "rl-it1" / "search" / "train" / "seg04" / "cap_counts.json",
          {"records": 209000, "deep": 26100, "masked": 0, "deep_visits": 4e7, "fast_visits": 5e7})
    r = decide(run, 1)
    assert r["results"]["gamma_sample"]["nodes"] == 3200
    assert abs(r["results"]["gamma_sample"]["gamma"] - 1.2) < 1e-4
    assert r["results"]["review"] is True
    assert r["state"]["gamma"]["3200"] == [1.2]
    assert r["main_mix"]["chal"] == 0.2     # records are not known for every segment: equal-GPU share
    p2 = plan(run, 2)
    assert p2["cost"]["gamma"]["3200"] == 1.2


def test_failsafe_on_corrupt_state(run):
    plan(run, 1)
    write(run / "balance" / "it1" / "score.json", scores(0.0))
    decide(run, 1)
    with open(run / "balance" / "it1.json", "w") as f:
        f.write("{not json")
    p = plan(run, 2)
    assert p["mode"] == "failsafe" and p["challenger"] is None
    assert all(s["engine_args"] == [] for s in p["segments"].values())


def test_missing_decide_is_failsafe(run):
    plan(run, 1)
    p = plan(run, 2)                   # it1.json never written
    assert p["mode"] == "failsafe" and p["challenger"] is None


def test_decide_requires_scores(run):
    plan(run, 1)
    with pytest.raises(SystemExit):
        decide(run, 1)
    assert not os.path.exists(run / "balance" / "it1.json")


def test_decide_without_scores_keeps_the_ledger(run):
    """Review finding: a failed minitrain/score stage made decide refuse, so the iteration's overhead, gamma sample
    and probe version were lost and the next plan was failsafe. --no-score records 'none' and keeps them."""
    p = plan(run, 1)
    assert p["challenger"] and p["probe"]["refresh"]
    write(run / "balance" / "it1" / "overhead.json", {"gpu_s": {"probe": 1800.0, "minitrain": 1800.0}})
    write(run / "runs" / "rl-it1" / "stages" / "balance_probe.done", {"version": 1, "gpu_s": 1800.0})
    r = decide(run, 1, no_score=True)
    assert r["decision"] == {"action": "none", "reason": "no mini-train scores: no evidence, no move"}
    assert r["state"]["overhead"]["cum_gpu_h"] == pytest.approx(1.0)
    assert r["state"]["probe"] == {"version": 1, "iter": 1}
    assert plan(run, 2)["mode"] != "failsafe"


def test_decide_reads_the_plans_prior(tmp_path, monkeypatch):
    """Review finding: plan honoured --prior but decide always read balance/prior.json."""
    root, custom = tmp_path / "RUN", tmp_path / "elsewhere" / "prior.json"
    write(custom, dict(OPEN, sigma_hat=0.002))
    p = plan(root, 1, prior=str(custom))
    assert p["challenger"] and p["prior_path"] == str(custom)
    seen = []
    real = sb.load_prior
    monkeypatch.setattr(sb, "load_prior", lambda run, path=None: seen.append(path) or real(run, path))
    write(root / "balance" / "it1" / "score.json", scores(0.003))
    decide(root, 1, dry=True)
    decide(root, 1)
    assert seen == [str(custom), str(custom)]
    other = tmp_path / "other.json"
    write(other, {})
    seen.clear()
    decide(root, 1, dry=True, prior=str(other))
    assert seen == [str(other)]


def test_iteration_gpu_hours_only_from_a_done_metrics_record(run):
    """Review finding: metrics/itN.json is partial while the iteration runs and already holds the balance stages."""
    def run_gpu(record):
        shutil.rmtree(run / "balance" / "it1", ignore_errors=True)
        for f in ("it1.json",):
            if os.path.exists(run / "balance" / f):
                os.remove(run / "balance" / f)
        write(run / "metrics" / "it1.json", record)
        write(run / "balance" / "it1" / "score.json", scores(0.003))
        write(run / "balance" / "it1" / "overhead.json", {"gpu_s": {"minitrain": 3600.0}})
        r = decide(run, 1)
        return r["state"]["overhead"]["run_gpu_h"] - p["state_in"]["overhead"]["run_gpu_h"]
    p = plan(run, 1)
    assert run_gpu({"done": False, "cost": {"gpu_hours": 3.0}}) == pytest.approx(sb.ITER_GPU_H + 1.0)
    assert run_gpu({"done": True, "cost": {"gpu_hours": 15.0}}) == pytest.approx(15.0)  # 14 + 1 overhead, once


def test_delete_and_rerun_byte_identical(run):
    for n in (1, 2, 3):
        plan(run, n)
        if read(run / "balance" / f"it{n}.plan.json")["challenger"]:
            write(run / "balance" / f"it{n}" / "score.json", scores(0.004 * n, spread=0.001))
        write(run / "metrics" / f"it{n}.json", {"cost": {"gpu_hours": 17.0}, "alerts": [],
                                                 "generated_at": f"t{n}"})
        decide(run, n)
    files = {f: open(run / "balance" / f, "rb").read() for f in os.listdir(run / "balance") if f.endswith(".json")}
    for f in ("it2.json", "it3.plan.json", "it3.json"):
        os.remove(run / "balance" / f)
    write(run / "metrics" / "it2.json", {"cost": {"gpu_hours": 17.0}, "alerts": [], "generated_at": "later"})
    decide(run, 2)
    plan(run, 3)
    decide(run, 3)
    for f, data in files.items():
        assert open(run / "balance" / f, "rb").read() == data, f


def test_immutable_inputs_checked(run):
    plan(run, 1)
    write(run / "balance" / "it1" / "score.json", scores(0.003))
    decide(run, 1)
    decide(run, 1)                                     # same inputs: fine, not rewritten
    write(run / "balance" / "it1" / "score.json", scores(0.009))
    with pytest.raises(SystemExit):
        decide(run, 1)


def test_metrics_degrade_gracefully(run):
    write(run / "metrics" / "it1.json", {"outcome": None, "cost": "weird", "alerts": [1, {"x": 2}]})
    plan(run, 1)
    write(run / "balance" / "it1" / "score.json", {"scores": {"inc_s0": 2.0, "chal_s0": 1.99}})  # one seed only
    r = decide(run, 1)
    assert r["results"]["sigma_n"] is None and r["decision"]["n"] == 1


def test_cli_and_status(run):
    out = subprocess.run([sys.executable, SCRIPT, "plan", str(run), "1"], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    write(run / "balance" / "it1" / "score.json", scores(0.003))
    out = subprocess.run([sys.executable, SCRIPT, "decide", str(run), "1", "--dry"], capture_output=True, text=True)
    assert json.loads(out.stdout)["run_minigate"] in (True, False)
    out = subprocess.run([sys.executable, SCRIPT, "decide", str(run), "1"], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    out = subprocess.run([sys.executable, SCRIPT, "status", str(run), "--json"], capture_output=True, text=True)
    rows = json.loads(out.stdout)
    assert rows[0]["iteration"] == 1 and rows[0]["challenger"] == "f=0.125,k=4,b=800,l=0,src=cap"


@pytest.mark.skipif(not os.path.isdir(os.path.join(FIXTURE, "runs", "rl-it1")),
                    reason="runs/wave2/int-shared is not on this machine")
def test_fixture_gate_marker(tmp_path):
    root = tmp_path / "int-shared"
    shutil.copytree(os.path.join(FIXTURE, "runs", "rl-it1", "stages"), root / "runs" / "rl-it1" / "stages")
    plan(root, 1)
    write(root / "balance" / "it1" / "score.json", scores(0.0))
    r = decide(root, 1)
    assert r["decision"]["action"] in ("continue", "reject", "none")
    obs, _, _ = sb.gather_obs(str(root), 1, read(root / "balance" / "it1.plan.json"))
    assert obs["gate"]["elo"] == 0.0 and obs["gate"]["score"] == 0.5


# ------------------------------------------------------------------------------------------------ simulator


def test_simulator_deterministic_and_capped():
    a = sb.simulate(iterations=8, seeds=3, sigma=4.0, surfaces=("a_flat", "c_interior"), thompson=True)
    b = sb.simulate(iterations=8, seeds=3, sigma=4.0, surfaces=("a_flat", "c_interior"), thompson=True)
    assert json.dumps(sb.sig(a), sort_keys=True) == json.dumps(sb.sig(b), sort_keys=True)
    for r in a["surfaces"].values():
        assert r["overhead_max"] <= sb.OVERHEAD_CAP + 1e-9
    assert "| a_flat |" in sb.markdown(a)


def test_sim_resume_midway_matches():
    """plan/decide are pure: re-running the controller from a saved state reproduces the same trajectory."""
    rows, _, _ = sb.run_controller("c_interior", 5, 12, 4.0)
    rows2, _, _ = sb.run_controller("c_interior", 5, 12, 4.0)
    assert rows == rows2


# ------------------------------------------------------------------------------------------------ review fixes


def test_default_prior_holds_theta0(tmp_path):
    """Nothing builds score.json yet, deep/lambda arms are unmeasured: with no prior flags the controller plans no
    challenger and decide needs no score.json (it records 'none')."""
    root = tmp_path / "RUN"
    p = plan(root, 1)
    assert p["challenger"] is None and p["mode"] == "frozen" and "score_stage" in p["reason"]
    assert p["steps_rule"] == "positions"                      # train.policy_positions does not exist yet
    assert all(s["engine_args"] == [] for s in p["segments"].values())
    assert decide(root, 1)["decision"]["action"] == "none"
    assert sb.plan_core(1, sb.init_state({}), {"prior": {"score_stage": True}})["challenger"] is None  # all masked


def test_between_data_noise_blocks_promotion(tmp_path):
    """Review finding: two iterations with d = 0.105 then 0.02 (a 5x swing) but tight seed pairs gave Z = 20 and
    eligible. The swing itself now measures the between-data noise, and without a measured tau no promotion."""
    root = tmp_path / "RUN"
    write(root / "balance" / "prior.json", {"sigma_hat": 0.005, "score_stage": True, "probe_bias_measured": True})
    plan(root, 1)
    write(root / "balance" / "it1" / "score.json", scores(0.105, spread=0.001))
    assert decide(root, 1)["decision"]["action"] == "continue"
    plan(root, 2)
    write(root / "balance" / "it2" / "score.json", scores(0.02, spread=0.001))
    ev = decide(root, 2, dry=True)
    assert not ev["eligible"] and not ev["run_minigate"]
    assert ev["tau_hat"] > 0.05 and ev["tau_df"] == 1 and ev["Z"] < 3   # se now carries the 0.06 spread
    # with a measured prior tau of the same size (replicate round) it is still far from Z_PROMOTE
    S = sb.init_state({})
    S["evidence"]["p"] = [{"iter": 1, "d": 0.105, "sigma_n": 0.0014, "seeds": 2},
                          {"iter": 2, "d": 0.02, "sigma_n": 0.0014, "seeds": 2}]
    ev = sb.evaluate(S, "p", 2, {"sigma_hat": 0.005, "tau_hat": 0.06, "tau_df": 9}, [])
    assert ev["noise_known"] and not ev["eligible"] and ev["Z"] < 2


@pytest.mark.skipif(not os.path.isdir(os.path.join(FIXTURE, "runs", "rl-it1")),
                    reason="runs/wave2/int-shared is not on this machine")
def test_realise_from_fixture_files(tmp_path):
    root = tmp_path / "int-shared"
    shutil.copytree(os.path.join(FIXTURE, "runs", "rl-it1"), root / "runs" / "rl-it1",
                    ignore=shutil.ignore_patterns("*.dst", "*.onnx", "*.pt"))
    plan(root, 1)
    write(root / "balance" / "it1" / "accounting.json", {"seg00": 120.0})
    out = sb.cmd_realise(str(root), 1, Args(accounting=None, force=False))
    assert out["seg00"]["games"] == 2 and out["seg00"]["records"] == 424 and out["seg00"]["gpu_s"] == 120.0
    assert out["seg00"]["visit_rows"] == 424 and out["seg00"]["visits"] > 424 * 700
    assert abs(out["seg00"]["sit_rate"] - 1 / 424) < 1e-6 and out["seg00"]["flag_rate"] == 0.0
    assert "gpu_s" not in out["seg01"]                         # no accounting: no gamma sample, nothing guessed
    assert read(root / "balance" / "it1" / "realised.json") == sb.canon(out)
    write(root / "balance" / "it1" / "accounting.json", {"seg00": 999.0})
    with pytest.raises(SystemExit):
        sb.cmd_realise(str(root), 1, Args(accounting=None, force=False))


def test_simulator_shared_noise_keeps_flat_surface_at_theta0():
    """tau = sigma of shared per-iteration noise in d (which games were played); the prior's numbers are a
    simulated replicate round's estimates, not the true values."""
    left = 0
    for s in range(40):
        rows, _, prior = sb.run_controller("a_flat", s, 20, 4.0, tau=4.0)
        assert prior["tau_df"] == sb.REPLICATES - 1 and prior["tau_hat"] != 4.0
        left += any(r["star"] != sb.key(sb.THETA0) for r in rows)
    assert left <= 4
