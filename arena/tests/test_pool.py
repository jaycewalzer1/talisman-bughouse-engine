"""Tests for arena.pool: opponent pool, group shares, rotation schedule, held-out openings."""

from __future__ import annotations

import json

import pytest

from arena.openings import EVAL_OPENINGS, EVAL_PAIRS, TRAIN_PAIRS
from arena.pool import Pool, TeamSpec, main, make_schedule, schedule_mix, unbuildable


def pool_dict(**over):
    d = {
        "contender": {"name": "me", "setup": "sharing", "kind": "heuristic", "nodes": 200},
        "shares": {"target": 0.5, "snapshot": 0.3, "variant": 0.2},
        "snapshot_window": 5,
        "opponents": [
            {"name": "stock", "setup": "centralized", "kind": "heuristic", "group": "target"},
            {"name": "silent", "setup": "silent", "kind": "heuristic", "group": "variant", "weight": 1.0},
            {"name": "rand", "setup": "silent", "kind": "random", "group": "variant", "weight": 3.0},
        ],
    }
    d.update(over)
    return d


def test_openings_split_is_disjoint():
    train = {n for p in TRAIN_PAIRS for n in p.name.split(" / ")}
    evals = {n for p in EVAL_PAIRS for n in p.name.split(" / ")}
    assert evals == set(EVAL_OPENINGS) and not (train & evals)
    assert len(TRAIN_PAIRS) == 16 * 15 and len(EVAL_PAIRS) == 8 * 7
    assert len({p.id for p in TRAIN_PAIRS + EVAL_PAIRS}) == len(TRAIN_PAIRS) + len(EVAL_PAIRS)


def test_empty_snapshot_share_goes_to_variants():
    pool = Pool.from_dict(pool_dict())
    assert pool.effective_shares() == pytest.approx({"target": 0.5, "variant": 0.5})


def test_schedule_hits_shares_and_rotates_snapshots():
    pool = Pool.from_dict(pool_dict())
    for i in range(7):
        pool.add_snapshot(f"it{i:02d}", f"ckpt/it{i:02d}.pt", kind="heuristic")
    assert [s.name for s in pool.members("snapshot")] == [f"it{i:02d}" for i in range(2, 7)]
    specs = make_schedule(pool, 20, seed=1)
    mix = schedule_mix(specs)
    assert mix["games"] == 40
    assert {g: v["games"] for g, v in mix["groups"].items()} == {"target": 20, "snapshot": 12, "variant": 8}
    # any 10 consecutive pairs are on the 5/3/2 mix
    groups = [s["group"] for s in specs[::2]]
    for k in range(0, 11):
        w = groups[k:k + 10]
        assert abs(w.count("target") - 5) <= 1 and abs(w.count("snapshot") - 3) <= 1
    # weighted variants: rand (weight 3) three times as often as silent
    assert mix["opponents"]["rand"] == 6 and mix["opponents"]["silent"] == 2
    # old snapshots outside the window never play
    assert "it00" not in mix["opponents"] and "it01" not in mix["opponents"]


def test_schedule_pairs_swap_sides_and_hold_out_eval_openings():
    pool = Pool.from_dict(pool_dict())
    specs = make_schedule(pool, 30, seed=2)
    eval_ids = {p.id for p in EVAL_PAIRS}
    for a, b in zip(specs[::2], specs[1::2]):
        assert a["opening_id"] == b["opening_id"] and a["opponent"] == b["opponent"]
        assert (a["contender_team"], b["contender_team"]) == ("T1", "T2")
        assert a["teams"]["T1"]["setup"] == "sharing" and b["teams"]["T2"]["setup"] == "sharing"
        assert a["opening_id"] not in eval_ids
    assert all(s["equal_compute"] for s in specs if s["opponent"] != "rand")
    assert not any(s["equal_compute"] for s in specs if s["opponent"] == "rand")
    assert all(s["opening_id"] in eval_ids for s in make_schedule(pool, 3, openings="eval"))


def test_unbuildable_kinds_reported_and_validation():
    pool = Pool.from_dict(pool_dict())
    pool.add_snapshot("it01", "ckpt/it01.pt")  # kind "agent" has no player yet
    assert unbuildable(make_schedule(pool, 10)) == ["it01 (kind agent)"]
    with pytest.raises(ValueError, match="unique"):
        Pool.from_dict(pool_dict(opponents=[{"name": "a", "setup": "silent", "group": "target"},
                                            {"name": "a", "setup": "silent", "group": "variant"}]))
    with pytest.raises(ValueError, match="group"):
        Pool.from_dict(pool_dict(opponents=[{"name": "a", "setup": "silent", "group": "boss"}]))


def test_cli_add_snapshot_and_run_end_to_end(tmp_path):
    path = tmp_path / "pool.json"
    path.write_text(json.dumps(pool_dict()))
    assert main(["add-snapshot", "--pool", str(path), "--name", "it01", "--path", "x.pt", "--kind", "heuristic"]) == 0
    saved = json.loads(path.read_text())
    assert saved["opponents"][-1]["group"] == "snapshot" and saved["opponents"][-1]["path"] == "x.pt"
    out = tmp_path / "run"
    assert main(["run", "--pool", str(path), "--pairs", "4", "--boot", "50", "--out", str(out)]) == 0
    report = json.loads((out / "summary.json").read_text())
    assert report["errors"] == 0 and report["mix"]["games"] == 8
    assert set(report["results"]["by_group"]) <= {"target", "snapshot", "variant"}
    g = report["games"][0]
    assert {"opponent", "group", "equal_compute"} <= set(g)
    first = [json.loads(l) for l in (out / g["file"]).read_text().splitlines()][0]
    assert first["type"] == "game_start"


def test_shipped_phase1_pool_is_valid():
    pool = Pool.load("arena/pools/phase1.json")
    assert pool.members("target")[0].name == "stock-centralized"
    assert pool.effective_shares()["target"] == pytest.approx(0.5)
    assert not unbuildable(make_schedule(pool, 10))
