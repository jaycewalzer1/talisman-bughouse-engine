"""cluster/phase0_chk1.sh (docs/custom_engine_spec_v7_league.md 7.4) on a tiny fake chk1 round (SRC) into a temp study
root (STUDY): only the small files are copied, status.txt and the study data are written, round.json is optional.
    .venv/bin/python -m pytest tests/cluster/test_phase0_chk1.py -q
"""

import json
import os
import shutil
import subprocess

import pytest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SCRIPT = os.path.join(REPO, "cluster", "phase0_chk1.sh")

pytestmark = pytest.mark.skipif(shutil.which("rsync") is None, reason="no rsync")


def game(i, contender_won):
    """One results row and its event log: the contender (T1) wins or loses by checkmate."""
    seats = {"A_white": {"name": "Wave2-A_white-k160", "team": "T1", "kind": "wave2"},
             "A_black": {"name": "Upstream-v3-A_black-k160", "team": "T2", "kind": "upstream"},
             "B_white": {"name": "Upstream-v3-B_white-k160", "team": "T2", "kind": "upstream"},
             "B_black": {"name": "Wave2-B_black-k160", "team": "T1", "kind": "wave2"}}
    start = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR[] w KQkq - 0 1"
    clocks = {s: 60000 for s in seats}
    events = [{"type": "game_start", "seq": 0, "ts": 0, "game_id": f"p{i:05d}", "seats": seats,
               "fens": {"A": start, "B": start}, "clocks": clocks, "base_ms": 60000},
              {"type": "game_end", "seq": 1, "ts": 1000, "result": "T1" if contender_won else "T2",
               "reason": "checkmate", "fens": {"A": start, "B": start}, "clocks": clocks}]
    row = {"game_id": f"p{i:05d}", "pair_index": (i - 1) // 2, "seats": seats, "contender_team": "T1",
           "result": "T1" if contender_won else "T2", "reason": "checkmate", "file": f"game_{i:04d}.jsonl"}
    return row, events


def fake_round(src, with_round_json=False):
    """sym160 and rev_opp160 rows, one part each, plus files that must not be copied."""
    for row, wins in (("sym160", (1, 1, 1, 0)), ("rev_opp160", (0, 0, 0, 1))):
        part = src / row / "part1"
        part.mkdir(parents=True)
        lines = []
        for i, w in enumerate(wins, 1):
            r, events = game(i, w)
            lines.append(json.dumps(r))
            (part / r["file"]).write_text("".join(json.dumps(e) + "\n" for e in events))
        (part / "results.jsonl").write_text("\n".join(lines) + "\n")
        (part / "specs.jsonl").write_text("{}\n")
        (part / "eval_123.log").write_text("big log\n")
        (part / "engine.onnx").write_bytes(b"x" * 100)
        (src / row / "jobs.tsv").write_text(f"{row}\tpart1\tseed 5001\t16153523\n")
        (src / row / "candidate.json").write_text("{}")
    (src / "candidates").mkdir()
    (src / "candidates" / "sym160.json").write_text("{}")
    if with_round_json:
        (src / "round.json").write_text(json.dumps({"name": "chk1", "mode": "check", "baseline": None, "candidates": [
            {"name": "sym160", "params": {}}, {"name": "rev_opp160", "params": {}}]}))


def run(tmp_path, **env):
    full = {k: v for k, v in os.environ.items() if k not in ("DRY_RUN", "REMOTE")}
    full.update({"SRC": str(tmp_path / "src"), "STUDY": str(tmp_path / "study"), **env})
    return subprocess.run(["bash", SCRIPT], env=full, capture_output=True, text=True, cwd=REPO, timeout=120)


@pytest.mark.parametrize("with_round_json", [False, True])
def test_a_fake_round_gives_status_and_study_data(tmp_path, with_round_json):
    fake_round(tmp_path / "src", with_round_json)
    r = run(tmp_path)
    assert r.returncode == 0, r.stderr
    chk1 = tmp_path / "study" / "tune" / "chk1"
    assert (chk1 / "sym160" / "part1" / "game_0001.jsonl").exists()
    assert (chk1 / "sym160" / "part1" / "specs.jsonl").exists() and (chk1 / "candidates" / "sym160.json").exists()
    assert (chk1 / "sym160" / "jobs.tsv").exists() and (chk1 / "sym160" / "candidate.json").exists()
    assert not (chk1 / "sym160" / "part1" / "eval_123.log").exists()
    assert not (chk1 / "sym160" / "part1" / "engine.onnx").exists()
    assert (chk1 / "round.json").exists() == with_round_json  # never written by the script
    status = (chk1 / "status.txt").read_text()
    assert "sym160" in status and "rev_opp160" in status
    assert status.strip() in r.stdout
    study = json.loads((tmp_path / "study" / "study.json").read_text())
    groups = {g["id"]: g for g in study["groups"]}
    assert groups["chk1/sym160"]["all"]["n"] == 4 and groups["chk1/sym160"]["all"]["score"] == 0.75
    assert groups["chk1/rev_opp160"]["all"]["score"] == 0.25
    assert (tmp_path / "study" / "games_chk1__sym160.json").exists()


def test_dry_run_prints_the_rsync_from_pace(tmp_path):
    r = subprocess.run(["bash", SCRIPT], env={**os.environ, "DRY_RUN": "1", "STUDY": str(tmp_path / "study")},
                       capture_output=True, text=True, cwd=REPO)
    assert r.returncode == 0, r.stderr
    assert r.stdout.startswith("DRY_RUN rsync -a --prune-empty-dirs")
    assert "pace:/storage/project/ps-bughouse-0/jwalzer3/talisman-bughouse-engine/runs/tune/chk1/" in r.stdout
    assert "--exclude=\\*" in r.stdout and not (tmp_path / "study").exists()


def test_a_failed_rsync_stops(tmp_path):
    r = run(tmp_path, SRC=str(tmp_path / "missing"))
    assert r.returncode == 1 and "rsync from" in r.stderr
    assert not (tmp_path / "study" / "study.json").exists()
