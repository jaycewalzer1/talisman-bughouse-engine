"""Tests for the evaluation harness (audit 2026-11-17 WP-A): resumable rows, rescoring, GSPRT, per-arm ms_per_node."""

from __future__ import annotations

import json
import math
import os
import random
import shutil

import pytest

from arena.pool import (Pool, apply_ms_per_node, arm_key, for_evaluation, main, make_schedule, run_schedule,
                        summary_from_game_file)
from tools import rescore as rescore_mod
from tools import sprt

REAL_ROW = "runs/custom/local1h/it01/gate"


def pool_dict():
    return {
        "contender": {"name": "me", "setup": "centralized", "kind": "heuristic", "nodes": 200},
        "opponents": [{"name": "stock", "setup": "centralized", "kind": "heuristic", "group": "target"}],
    }


def _row(tmp_path, pairs=3, extra=()):
    path, out = tmp_path / "pool.json", tmp_path / "row"
    path.write_text(json.dumps(pool_dict()))
    assert main(["run", "--pool", str(path), "--pairs", str(pairs), "--boot", "200", "--base-ms", "20000",
                 "--out", str(out), *extra]) == 0
    return path, out


def _comparable(results):
    return json.loads(json.dumps(results))


# ---------------------------------------------------------------- rescore

def test_rescore_equals_summary_of_a_fresh_row(tmp_path):
    _, out = _row(tmp_path)
    report = json.loads((out / "summary.json").read_text())
    rep = rescore_mod.rescore(str(out), n_boot=200, seed=0)
    assert _comparable(rep["results"]) == report["results"]
    by_id = {g["game_id"]: g for g in report["games"]}
    for g in rep["games"]:
        old = {k: v for k, v in by_id[g["game_id"]].items() if k != "wall_s"}
        assert {k: v for k, v in g.items()} == old
    # results.jsonl has every game as it landed
    lines = [json.loads(l) for l in (out / "results.jsonl").read_text().splitlines()]
    assert sorted(l["game_id"] for l in lines) == sorted(by_id)
    assert report["results"]["by_opponent"]["stock"]["reasons"]
    assert "contender_nodes_per_game" in report["results"]["by_opponent"]["stock"]


@pytest.mark.skipif(not os.path.exists(os.path.join(REAL_ROW, "summary.json")), reason="local1h gate row not present")
def test_rescore_of_an_existing_run_equals_its_summary(tmp_path):
    row = tmp_path / "gate"
    shutil.copytree(REAL_ROW, row)
    old = json.loads((row / "summary.json").read_text())
    rep = rescore_mod.rescore(str(row), n_boot=2000, seed=1)  # the it01 gate ran with --seed 1, --boot 2000
    for name, r in old["results"]["by_opponent"].items():
        new = rep["results"]["by_opponent"][name]
        assert {k: new[k] for k in r} == r
    assert len(rep["games"]) == len(old["games"])
    for a, b in zip(rep["games"], sorted(old["games"], key=lambda g: g["game_id"])):
        assert (a["result"], a["reason"], a["teams"]) == (b["result"], b["reason"], b["teams"])


# ---------------------------------------------------------------- resume

def test_skip_done_plays_only_missing_cut_and_crashed_games(tmp_path):
    from arena.run import run_one

    _, out = _row(tmp_path, pairs=4)
    full = json.loads((out / "summary.json").read_text())
    (out / "summary.json").unlink()
    (out / "game_0002.jsonl").unlink()  # never written (job cut off)
    lines = (out / "game_0005.jsonl").read_text().splitlines()
    (out / "game_0005.jsonl").write_text("\n".join(lines[:-1]) + "\n")  # cut mid-write: no game_end
    lines = (out / "game_0007.jsonl").read_text().splitlines()
    end = json.loads(lines[-1])
    end["error"] = "RuntimeError: engine died"
    (out / "game_0007.jsonl").write_text("\n".join(lines[:-1] + [json.dumps(end)]) + "\n")

    played = []

    def counting(spec):
        played.append(spec["index"])
        return run_one(spec)

    pool = Pool.from_dict(pool_dict())
    specs = make_schedule(pool, 4, seed=0, base_ms=20000)
    report = run_schedule(specs, str(out), n_boot=200, skip_done=True, run_one=counting)
    assert sorted(played) == [2, 5, 7]
    assert report["results"]["by_opponent"]["stock"]["games"] == full["results"]["by_opponent"]["stock"]["games"]
    assert len([json.loads(l) for l in (out / "results.jsonl").read_text().splitlines()]) == 8
    # a different schedule into the same folder is refused
    with pytest.raises(ValueError, match="different schedule"):
        run_schedule(make_schedule(pool, 4, seed=1, base_ms=20000), str(out), skip_done=True, run_one=counting)


def test_summary_from_game_file_rejects_missing_and_partial(tmp_path):
    spec = {"index": 1, "game_id": "p00001", "contender_team": "T1"}
    assert summary_from_game_file(spec, str(tmp_path / "nope.jsonl")) is None
    (tmp_path / "g.jsonl").write_text('{"type":"game_start"}\n{"type":"move"')
    assert summary_from_game_file(spec, str(tmp_path / "g.jsonl")) is None


# ---------------------------------------------------------------- SPRT

def _game_probs(target_nelo: float, draw: float = 0.2) -> tuple[float, float, float]:
    """(win, draw, loss) of independent games whose normalised Elo is target_nelo."""
    def nelo_of(d):
        w, l = (1 - draw) / 2 + d, (1 - draw) / 2 - d
        mu = w + draw / 2
        var = w + draw / 4 - mu * mu
        return (mu - 0.5) / math.sqrt(var) * sprt.NELO_PER_NT
    lo, hi = -0.39, 0.39
    for _ in range(80):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if nelo_of(mid) < target_nelo else (lo, mid)
    d = (lo + hi) / 2
    return (1 - draw) / 2 + d, draw, (1 - draw) / 2 - d


def _simulate(target: float, runs: int, seed: int, max_pairs: int = 20000) -> dict:
    rng = random.Random(seed)
    w, d, _ = _game_probs(target)
    out = {"H0": 0, "H1": 0, "continue": 0}
    lo, hi = sprt.bounds()
    for _ in range(runs):
        counts = [0] * 5
        decision = "continue"
        for n in range(max_pairs):
            pts = 0.0
            for _g in range(2):
                u = rng.random()
                pts += 1.0 if u < w else 0.5 if u < w + d else 0.0
            counts[int(pts * 2)] += 1
            if n % 10 == 9:  # look every 10 pairs (cheaper; slightly conservative)
                v = sprt.llr(counts, 0.0, 20.0)
                if v >= hi or v <= lo:
                    decision = "H1" if v >= hi else "H0"
                    break
        out[decision] += 1
    return out


def test_sprt_on_simulated_pairs_hits_nominal_alpha_and_beta():
    runs = 200
    at0 = _simulate(0.0, runs, seed=1)
    at1 = _simulate(20.0, runs, seed=2)
    # nominal 5% each; 200 runs -> standard error ~1.5%, so allow up to 10%
    assert at0["H1"] / runs <= 0.10, at0
    assert at1["H0"] / runs <= 0.10, at1
    assert at0["continue"] == 0 and at1["continue"] == 0
    # and it is not trivially conservative: a clearly better side is accepted
    assert _simulate(40.0, 50, seed=3)["H1"] >= 48
    # the LLR under each hypothesis' maximum-likelihood distribution honours the constraint
    phat = [0.05, 0.2, 0.4, 0.25, 0.1]
    for t in (0.0, 0.08):
        p = sprt.mle_t(phat, t)
        xs = [b / 2 for b in sprt.BUCKETS]
        mu = sum(x * q for x, q in zip(xs, p))
        sd = math.sqrt(sum(q * (x - mu) ** 2 for x, q in zip(xs, p)))
        assert (mu - 0.5) / sd == pytest.approx(t, abs=1e-6) and sum(p) == pytest.approx(1.0)


def test_pentanomial_counts_whole_pairs_only():
    g = lambda i, res, ct="T1", **kw: {"game_id": f"p{i:05d}", "pair_index": (i - 1) // 2, "result": res,
                                       "contender_team": ct, "opponent": "x", **kw}
    games = [g(1, "T1"), g(2, "T1", "T2"), g(3, "draw"), g(4, "T2"), g(5, "T1"), g(7, "T1"), g(8, "T2", error="x")]
    assert sprt.pentanomial(games) == [0, 1, 1, 0, 0]  # pairs 0 (1+0) and 1 (0.5+0); 2 and 3 incomplete/errored
    st = sprt.status([0, 0, 0, 0, 0])
    assert st["decision"] == "continue" and st["llr"] == 0.0


def test_run_schedule_stops_when_sprt_is_decided(tmp_path):
    pool = Pool.from_dict(pool_dict())
    specs = make_schedule(pool, 400, seed=0)

    def always_contender(spec):
        return {"game_id": spec["game_id"], "contender_team": spec["contender_team"], "opening_id": spec["opening_id"],
                "result": spec["contender_team"], "reason": "checkmate", "flag": False, "loser_team": None,
                "teams": {t: {"decisions": 1, "think_ms": 1, "nodes": 1, "moves": 1, "messages": 0, "stalls": 0,
                              "captures": 0} for t in ("T1", "T2")}}

    report = run_schedule(specs, str(tmp_path / "r"), n_boot=50, run_one=always_contender,
                          sprt={"elo0": 0.0, "elo1": 20.0, "alpha": 0.05, "beta": 0.05})
    assert report["sprt"]["decision"] == "H1"
    assert report["unplayed"] > 0 and len(report["games"]) < len(specs)


# ---------------------------------------------------------------- per-arm ms_per_node

def _engine_pool(contender_kw=None, opponent_kw=None):
    return Pool.from_dict({
        "contender": {"name": "me", "setup": "centralized", "kind": "hivemind", "nodes": 100,
                      "player_kw": contender_kw or {}},
        "opponents": [{"name": "stock", "setup": "centralized", "kind": "hivemind", "nodes": 100,
                       "player_kw": opponent_kw or {}, "group": "target"}],
    })


def test_ms_per_node_measured_once_per_binary_and_network(tmp_path):
    calls = []

    def fake(binary, model, nodes, repeats):
        calls.append(model)
        return {"ms_per_node": 0.25 if model.endswith("clock.onnx") else 0.5}

    # same binary+network on both arms (a same_* control): one measurement, both charged the same
    same, used = apply_ms_per_node(_engine_pool(), str(tmp_path / "ms.json"), measure=fake)
    assert len(calls) == 1 and len(used) == 1
    assert same.contender.player_kw["ms_per_node"] == same.opponents[0].player_kw["ms_per_node"] == 0.5

    # different networks: each arm its own value; the cache file is reused on resume (no new measurement)
    pool = _engine_pool({"binary": "hivemind-clock", "model": str(tmp_path / "clock.onnx")})
    mixed, used = apply_ms_per_node(pool, str(tmp_path / "ms.json"), measure=fake)
    assert len(calls) == 2 and len(used) == 2
    assert mixed.contender.player_kw["ms_per_node"] == 0.25 and mixed.opponents[0].player_kw["ms_per_node"] == 0.5
    again, _ = apply_ms_per_node(pool, str(tmp_path / "ms.json"), measure=fake)
    assert len(calls) == 2 and again == mixed

    # an explicit value is kept; "measure" asks for one
    fixed, used = apply_ms_per_node(_engine_pool({"ms_per_node": 5.5}, {"ms_per_node": "measure"}),
                                    str(tmp_path / "ms2.json"), measure=fake)
    assert fixed.contender.player_kw["ms_per_node"] == 5.5 and fixed.opponents[0].player_kw["ms_per_node"] == 0.5
    assert list(used) == [arm_key(fixed.opponents[0])]

    # the value reaches each team's player options in the schedule
    spec = make_schedule(mixed, 1)[0]
    ct, ot = spec["contender_team"], "T2" if spec["contender_team"] == "T1" else "T1"
    assert spec["teams"][ct]["player_kw"]["ms_per_node"] == 0.25
    assert spec["teams"][ot]["player_kw"]["ms_per_node"] == 0.5


def test_equal_ms_per_node_charges_both_arms_the_opponents_rate(tmp_path):
    from arena.pool import equalize_ms_per_node

    pool, _ = apply_ms_per_node(_engine_pool({"model": str(tmp_path / "clock.onnx")}), None,
                                measure=lambda b, m, nodes, repeats: {"ms_per_node": 0.25 if "clock" in m else 0.5})
    eq = equalize_ms_per_node(pool)
    assert eq.contender.player_kw["ms_per_node"] == eq.opponents[0].player_kw["ms_per_node"] == 0.5
    assert eq.contender.player_kw["model"] == str(tmp_path / "clock.onnx")


def test_ms_per_node_reaches_the_controller():
    from arena.players import setups

    seen = {}

    class FakeController:
        def __init__(self, team, process, nodes, ms_per_node, **kw):
            seen[team] = ms_per_node

        def player(self, seat):
            return seat

    real_ctl, real_proc = setups.HivemindTeamController, setups.HivemindProcess
    setups.HivemindTeamController, setups.HivemindProcess = FakeController, lambda *a, **k: None
    try:
        setups.team_players("T1", "centralized", "hivemind", seed=0, nodes=100, ms_per_node=0.125)
    finally:
        setups.HivemindTeamController, setups.HivemindProcess = real_ctl, real_proc
    assert seen == {"T1": 0.125}


def test_for_evaluation_keeps_an_unjittered_contender():
    pool = _engine_pool()
    ev = for_evaluation(pool, contender_kw={"jitter": [1.0, 1.0]}, opponent_kw={"serial": True})
    assert ev.contender.player_kw["jitter"] == [1.0, 1.0] and ev.opponents[0].player_kw == {"serial": True}


# ---------------------------------------------------------------- compare_evals

def test_compare_evals_pairs_rows_with_the_same_schedule(tmp_path, capsys):
    from tools import compare_evals

    _, out = _row(tmp_path, pairs=4)
    games, _, finished = compare_evals.load(str(out))
    assert finished and len(games) == 8
    d, lo, hi, paired = compare_evals.diff_ci(games, games)
    assert paired and d == lo == hi == 0.0
    flipped = [{**g, "result": g["contender_team"]} for g in games]  # the same games, all won
    d, lo, hi, paired = compare_evals.diff_ci(games, flipped)
    assert paired and lo <= d <= hi and d == pytest.approx(1 - sum(
        1.0 if g["result"] == g["contender_team"] else 0.5 if g["result"] == "draw" else 0.0 for g in games) / 8)
    other = [{**g, "opening_id": -1 - i} for i, g in enumerate(games)]
    assert compare_evals.diff_ci(games, other)[3] is False
    compare_evals.main([str(out), str(out)])
    text = capsys.readouterr().out
    assert "nodes/g c" in text and "+0.000" in text


def _tiny_onnx(path, weight: float, width: int = 3):
    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    w = numpy_helper.from_array(np.full((width, width), weight, dtype=np.float32), "w")
    g = helper.make_graph([helper.make_node("MatMul", ["x", "w"], ["y"])], "g",
                          [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, width])],
                          [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, width])], [w])
    onnx.save(helper.make_model(g), str(path))
    return str(path)


def test_one_measurement_per_architecture_across_iterations(tmp_path):
    """Audit 2026-11-17: it00 and it01 (same binary, same architecture) were measured apart and charged 2.46 vs
    2.58 ms/node. Networks of one architecture now share one cache entry; another shape is measured anew."""
    pytest.importorskip("onnx")
    from arena.pool import network_shape_key

    it00, it01 = _tiny_onnx(tmp_path / "it00.onnx", 0.1), _tiny_onnx(tmp_path / "it01.onnx", 0.7)
    wide = _tiny_onnx(tmp_path / "wide.onnx", 0.1, width=4)
    assert network_shape_key(it00) == network_shape_key(it01) != network_shape_key(wide)
    assert network_shape_key(str(tmp_path / "missing.onnx")) is None
    calls = []

    def fake(binary, model, nodes, repeats):
        calls.append(model)
        return {"ms_per_node": 2.0 + len(calls)}

    cache = str(tmp_path / "ms.json")
    a, _ = apply_ms_per_node(_engine_pool({"binary": "hivemind-clock", "model": it00}), cache, measure=fake)
    b, _ = apply_ms_per_node(_engine_pool({"binary": "hivemind-clock", "model": it01}), cache, measure=fake)
    assert a.contender.player_kw["ms_per_node"] == b.contender.player_kw["ms_per_node"]
    c, _ = apply_ms_per_node(_engine_pool({"binary": "hivemind-clock", "model": wide}), cache, measure=fake)
    assert c.contender.player_kw["ms_per_node"] != a.contender.player_kw["ms_per_node"]
    assert sum(1 for m in calls if m in (it00, it01)) == 1


def _upstream_pool(upstream_kw=None):
    return Pool.from_dict({
        "contender": {"name": "me", "setup": "centralized", "kind": "hivemind", "nodes": 100,
                      "player_kw": {"binary": "hivemind-clock", "model": "/nonexistent/clock.onnx"}},
        "opponents": [{"name": "upstream-v3", "setup": "centralized", "kind": "upstream", "nodes": 800,
                       "player_kw": {"ms_per_node": 0.65, **(upstream_kw or {})}, "group": "target"}],
    })


def test_gate_against_upstream_measures_and_equalizes_both_arms(tmp_path):
    """Audit 2026-11-17: the eqn row against upstream-v3 raised 'needs an engine opponent', and upstream's 0.65 was
    a Mac CPU figure. "measure" now measures upstream on this machine with its own engine, and the equal-nodes
    row charges both arms upstream's rate."""
    from arena.pool import equalize_ms_per_node

    pool = for_evaluation(_upstream_pool({"ms_per_node": "measure", "max_nodes": 100000}),
                          opponent="upstream-v3")
    ups = []
    measured, used = apply_ms_per_node(
        pool, str(tmp_path / "ms.json"), measure=lambda b, m, nodes, repeats: {"ms_per_node": 0.3},
        measure_upstream=lambda b, m, nodes, repeats, coreml=None: ups.append((b, m)) or {"ms_per_node": 0.2})
    assert len(ups) == 1 and measured.opponents[0].player_kw["ms_per_node"] == 0.2
    assert measured.contender.player_kw["ms_per_node"] == 0.3 and len(used) == 2
    assert any(k.startswith("upstream|") for k in used)
    eq = equalize_ms_per_node(measured)
    assert eq.contender.player_kw["ms_per_node"] == eq.opponents[0].player_kw["ms_per_node"] == 0.2
    # the pool's fixed figure still works without measuring
    eq = equalize_ms_per_node(for_evaluation(_upstream_pool(), opponent="upstream-v3"))
    assert eq.contender.player_kw["ms_per_node"] == 0.65


def test_gate_sitoff_override_reaches_the_simulator(tmp_path):
    specs = make_schedule(_engine_pool(), 1, rules="v3", sitoff_ms=0)
    assert all(s["sitoff_ms"] == 0 and s["rules"] == "v3" for s in specs)
    assert "sitoff_ms" not in make_schedule(_engine_pool(), 1, rules="v2", sitoff_ms=0)[0]
    assert "sitoff_ms" not in make_schedule(_engine_pool(), 1, rules="v3")[0]


def test_node_cap_hits_are_counted_per_team():
    """Audit 2026-11-17: at max_nodes 1600 the cap, not the clock, set both gate arms' budgets. A decision whose
    budget was cut is marked, and the game summary and the per-row stats count it."""
    from arena.interfaces import SEATS
    from arena.players import HivemindTeamController, SearchResult
    from arena.stats import compute_stats, summarize_game
    from arena.tests.test_players import START, StubProcess, make_obs

    stub = StubProcess(SearchResult("e2e4", None, 0.1, 20, None, 50, 3))
    ctl = HivemindTeamController("T1", stub, ms_per_node=1.0, clock_k=10, max_nodes=50)
    d = ctl.player("A_white").decide(make_obs("A_white", {"A": START, "B": START}))
    assert d.capped and d.nodes == 50
    ctl = HivemindTeamController("T1", StubProcess(SearchResult("e2e4", None, 0.1, 20, None, 50, 3)),
                                 ms_per_node=1.0, clock_k=10, max_nodes=100000)
    assert not ctl.player("A_white").decide(make_obs("A_white", {"A": START, "B": START})).capped
    events = [{"type": "game_start", "seq": 0, "ts": 0, "game_id": "g", "setup": "s", "seats": {s: s for s in SEATS}},
              {"type": "thinking", "seq": 1, "ts": 0, "seat": "A_white", "phase": "start", "think_ms": 5,
               "nodes": 50, "capped": True},
              {"type": "thinking", "seq": 2, "ts": 0, "seat": "A_black", "phase": "start", "think_ms": 5,
               "nodes": 50},
              {"type": "game_end", "seq": 3, "ts": 5, "result": "T1", "reason": "flag", "board": "A",
               "loserSeat": "A_black"}]
    s = summarize_game(events)
    assert s["capped_decisions"] == {"T1": 1, "T2": 0}
    st = compute_stats([{**s, "contender_team": "T1", "opening_id": "o"}], n_boot=10)
    assert st["metrics"]["contender_capped_share"]["est"] == 1.0
    assert st["metrics"]["baseline_capped_share"]["est"] == 0.0
