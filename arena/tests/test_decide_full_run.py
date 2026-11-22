"""cluster/decide_full_run.py on synthetic rounds (every blocker, each config and width choice, the hybrid, the long
test's merged rows), and dry runs of cluster/launch_wave2c.sh, cluster/launch_long_test.sh and
cluster/launch_full_run.sh with a fake sbatch on PATH (every submission's dependency)."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys

import pytest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "cluster"))

import decide_full_run as dfr  # noqa: E402
from wave2c_configs import BENCHED, CONFIGS, LONG_ROWS, ROWS  # noqa: E402

# ---------------------------------------------------------------------- synthetic round


def write_row(path, games: int, score: float, errors: int = 0, opening_base: int = 0):
    """results.jsonl of `games` games (pairs on their own openings, from opening_base) whose contender scores `score`."""
    os.makedirs(path, exist_ok=True)
    wins = round(score * games)
    team = {"decisions": 60, "think_ms": 30000, "nodes": 50000, "messages": 0, "stalls": 2}
    with open(os.path.join(path, "results.jsonl"), "w") as f:
        for i in range(games):
            ct = "T1" if i % 2 == 0 else "T2"
            other = "T2" if ct == "T1" else "T1"
            g = {"game_id": f"p{i + 1:05d}", "contender_team": ct, "result": ct if i < wins else other,
                 "reason": "checkmate", "flag": False, "opening_id": opening_base + i // 2, "pair_index": i // 2,
                 "teams": {"T1": team, "T2": team}, "opponent": "upstream-it13", "group": "target"}
            f.write(json.dumps(g) + "\n")
        for i in range(errors):
            f.write(json.dumps({"game_id": f"e{i}", "error": "engine died", "opponent": "upstream-it13"}) + "\n")


def shuffle_rows(path, seed: int = 0):
    """Spread wins and losses over the pairs at random (write_row puts the wins first), so the pair bootstrap sees
    the spread of real games: about +-0.04 at 600 games."""
    import random

    games = [json.loads(line) for line in open(os.path.join(path, "results.jsonl")).read().splitlines()]
    played = [g for g in games if "error" not in g]
    outcome = [g["result"] == g["contender_team"] for g in played]
    random.Random(seed).shuffle(outcome)
    for g, won in zip(played, outcome):
        other = "T2" if g["contender_team"] == "T1" else "T1"
        g["result"] = g["contender_team"] if won else other
    with open(os.path.join(path, "results.jsonl"), "w") as f:
        for g in games:
            f.write(json.dumps(g) + "\n")


def make_round(tmp, scores: dict | None = None, games: int = 600, build=True, gpu="PASS",
               smokes=dfr.SMOKES, rate=300.0, bench_widths=("narrow", "wide")):
    rd = os.path.join(tmp, "runs", "wave2c", "r1")
    os.makedirs(rd)
    if build:
        open(os.path.join(rd, "build.ok"), "w").write("Sat Nov 21\n")
    if gpu:
        checks = [{"check": "pin", "status": "PASS"},
                  {"check": "wide_parity", "status": "PASS" if gpu == "PASS" else "FAIL"}]
        json.dump({"overall": gpu, "checks": checks}, open(os.path.join(rd, "gpu_check.json"), "w"))
    base = {row: 0.44 for row in ROWS}
    base.update({"c_same_120": 0.50, "c_ref_120": 0.23, "c_all_120": 0.45, "c_all_nod1_120": 0.46})
    base.update(scores or {})
    for row, score in base.items():
        if score is None:
            continue
        n = games[row] if isinstance(games, dict) else games
        write_row(os.path.join(rd, "rows", row), n, score)
        shuffle_rows(os.path.join(rd, "rows", row))
    for config, width in smokes:
        arm = os.path.join(rd, f"smoke_{config}", width)
        os.makedirs(os.path.join(arm, "runs", "rl-it1", "stages"))
        open(os.path.join(arm, "runs", "rl-it1", "stages", "iteration.done"), "w").write("{}\n")
        open(os.path.join(arm, "results.tsv"), "w").write(
            "iteration\tgames\tvs_generator\tvs_it13\tpromoted\nrl-it1\t200\t0.48\t0.47\tFalse\n")
    if rate:
        for config in BENCHED:
            for width in bench_widths:
                d = os.path.join(rd, "bench", f"{config}_{width}")
                os.makedirs(d)
                json.dump({"games": 150, "games_asked": 150, "exit": 0, "games_per_gpu_hour": rate * 0.9,
                           "games_per_gpu_hour_steady": rate}, open(os.path.join(d, "bench.json"), "w"))
    return rd


def write_long(rd, lrow, parts: list[tuple[int, float]], errors: int = 0):
    """The long test's row lrow as parts (games, score), each on its own openings (as distinct seeds give)."""
    for k, (games, score) in enumerate(parts, 1):
        path = os.path.join(rd, "long", lrow, f"part{k}")
        write_row(path, games, score, errors=errors if k == 1 else 0, opening_base=1000 * k)
        shuffle_rows(path, seed=k)


def write_ab(tmp, pairs):
    ab = os.path.join(tmp, "ab2")
    os.makedirs(ab)
    lines = ["iteration\tnarrow_vs_it13\tnarrow_elo\tnarrow_promoted\twide_vs_it13\twide_elo\twide_promoted\t"
             "wide_arrival_labels"]
    for i, (n, w) in enumerate(pairs, 1):
        lines.append(f"rl-it{i}\t{n}\t0\tTrue\t{w}\t0\tTrue\t100")
    open(os.path.join(ab, "results_ab.tsv"), "w").write("\n".join(lines) + "\n")
    return ab


def run_decide(tmp, rd, ab=None, queue=()):
    return dfr.decide(rd, ab or os.path.join(tmp, "no_ab"), run="full1", queue=list(queue),
                      runs_full=os.path.join(tmp, "runs", "full"))


# ---------------------------------------------------------------------- GO and the choices


def test_go_picks_nodur_narrow_and_prints_the_command(tmp_path):
    rd = make_round(str(tmp_path))
    d = run_decide(str(tmp_path), rd)
    assert d.go, d.blockers
    assert (d.config, d.wide) == ("nodur", 0)
    assert d.rate == 300.0
    assert d.gpu_hours == pytest.approx(10 * 50_500 / 300 + 13)
    assert d.chain == -(-1.1 * d.gpu_hours // 16) + 10
    assert d.command == (f"RUN=full1 CONFIG=nodur WIDE=0 CHAIN={d.chain} GAMES_PER_ITER=50000 "
                         "bash cluster/launch_full_run.sh")
    text = dfr.report(d)
    assert text.startswith("GO\n") and 'WAVE2_EXTRA="--durative-tree false --sit-gate true"' in text


def test_continuity_chosen_when_its_range_clears_nodur(tmp_path):
    rd = make_round(str(tmp_path), {"c_all_120": 0.56, "c_nodur_120": 0.43})
    d = run_decide(str(tmp_path), rd)
    assert d.go and d.config == "cont"
    assert "CONFIG=cont" in d.command
    assert any("c_all_120" in n and "ranges disjoint" in n for n in d.notes)


def test_margin_rule_needs_500_games_each(tmp_path):
    # +0.06 with overlapping ranges: qualifies at 600 games a row, not at 300
    rd = make_round(str(tmp_path / "a"), {"c_all_120": 0.50, "c_nodur_120": 0.44}, games=600)
    stats = {r: dfr.row_stats(os.path.join(rd, "rows", r)) for r in ("c_all_120", "c_nodur_120")}
    assert stats["c_all_120"]["ci95"][0] <= stats["c_nodur_120"]["ci95"][1]   # ranges overlap
    assert run_decide(str(tmp_path / "a"), rd).config == "cont"
    rd = make_round(str(tmp_path / "b"), {"c_all_120": 0.50, "c_nodur_120": 0.44}, games=300)
    d = run_decide(str(tmp_path / "b"), rd)
    assert d.config == "nodur" and d.go


def test_no_preset_nod1_and_best_single_switch_compete(tmp_path):
    rd = make_round(str(tmp_path), {"c_all_nod1_120": 0.58, "c_carry_120": 0.62, "c_joint_120": 0.50,
                                    "c_nodur_120": 0.43})
    d = run_decide(str(tmp_path), rd)
    assert d.config == "carry"                       # both qualify, the higher score wins
    assert any("borrows" in n or "its rate and smoke are cont's" in n for n in d.notes)
    rd2 = make_round(str(tmp_path / "x"), {"c_all_nod1_120": 0.58, "c_nodur_120": 0.43})
    assert run_decide(str(tmp_path / "x"), rd2).config == "cont_nod1"


def test_hybrid_is_a_candidate_with_its_own_bench_and_smoke(tmp_path):
    rd = make_round(str(tmp_path), {"h_best_120": 0.57, "c_nodur_120": 0.43})
    d = run_decide(str(tmp_path), rd)
    assert d.go and d.config == "hybrid"
    assert "CONFIG=hybrid" in d.command and not any("borrows" in n or "smoke are" in n for n in d.notes)
    assert 'WAVE2_EXTRA="--durative-tree false --sit-gate true --hybrid"' in dfr.report(d)
    # a hybrid tie (the local result) stays nodur
    rd = make_round(str(tmp_path / "tie"), {"h_best_120": 0.44, "c_nodur_120": 0.44})
    assert run_decide(str(tmp_path / "tie"), rd).config == "nodur"
    # the hybrid needs its own smoke: without smoke_hybrid the round is NO-GO
    smokes = tuple(s for s in dfr.SMOKES if s[0] != "hybrid")
    d = run_decide(str(tmp_path / "s"), make_round(str(tmp_path / "s"), smokes=smokes))
    assert not d.go and len(blockers(d, "B3")) == 2 and all("smoke_hybrid" in b for b in blockers(d, "B3"))
    d = run_decide(str(tmp_path / "m"), make_round(str(tmp_path / "m"), {"h_best_120": None}))
    assert not d.go and "h_best_120 no games" in blockers(d, "B5")[0]


def test_merged_row_stats_sums_parts_and_clusters_openings(tmp_path):
    rd = make_round(str(tmp_path))
    write_long(rd, "l_hybrid_120", [(300, 0.5), (280, 0.6), (40, 0.5)], errors=3)
    m = dfr.merged_row_stats(os.path.join(rd, "long", "l_hybrid_120"))
    assert m["games"] == 620 and m["parts"] == 3 and m["errors"] == 3 and not m["finished"]
    assert m["score"] == pytest.approx((150 + 168 + 20) / 620)
    assert m["ci95"][0] < m["score"] < m["ci95"][1]
    one = dfr.row_stats(os.path.join(rd, "long", "l_hybrid_120", "part1"))
    assert m["ci95"][1] - m["ci95"][0] < one["ci95"][1] - one["ci95"][0]   # tighter than one part
    assert dfr.merged_row_stats(os.path.join(rd, "long", "nothing")) is None


def test_long_rows_are_preferred_over_the_round_rows(tmp_path):
    # the round rows tie (0.44 / 0.44); the long test separates them: hybrid chosen from the merged rows
    rd = make_round(str(tmp_path), {"h_best_120": 0.44, "c_nodur_120": 0.44})
    write_long(rd, "l_hybrid_120", [(300, 0.52)] * 5)
    write_long(rd, "l_nodur_120", [(300, 0.43)] * 5)
    write_long(rd, "l_same_120", [(300, 0.50)] * 5)
    d = run_decide(str(tmp_path), rd)
    assert d.go and d.config == "hybrid", (d.blockers, d.notes)
    assert any("l_hybrid_120 (long test, 5 parts)" in n and "QUALIFIES" in n and "l_nodur_120 (long test" in n
               for n in d.notes)
    assert any(n.startswith("control ok: l_same_120") for n in d.notes)
    # and the other way: the round row says hybrid, the long test says no
    rd = make_round(str(tmp_path / "b"), {"h_best_120": 0.60, "c_nodur_120": 0.43})
    write_long(rd, "l_hybrid_120", [(300, 0.45)] * 5)
    write_long(rd, "l_nodur_120", [(300, 0.44)] * 5)
    write_long(rd, "l_same_120", [(300, 0.50)] * 5)
    d = run_decide(str(tmp_path / "b"), rd)
    assert d.go and d.config == "nodur", d.blockers
    # a long row under 250 games is not read: the round row decides
    rd = make_round(str(tmp_path / "c"), {"h_best_120": 0.60, "c_nodur_120": 0.43})
    write_long(rd, "l_hybrid_120", [(100, 0.40)])
    d = run_decide(str(tmp_path / "c"), rd)
    assert d.config == "hybrid" and any("is read from the round row" in n for n in d.notes)


def test_long_test_blockers_control_errors_and_queue(tmp_path):
    rd = make_round(str(tmp_path / "c"))
    write_long(rd, "l_same_120", [(300, 0.42)] * 5)
    d = run_decide(str(tmp_path / "c"), rd)
    assert not d.go and "l_same_120" in blockers(d, "B4")[0]
    rd = make_round(str(tmp_path / "e"))
    write_long(rd, "l_nodur_120", [(300, 0.44)] * 2, errors=60)
    d = run_decide(str(tmp_path / "e"), rd)
    assert not d.go and "long l_nodur_120 errors 60" in blockers(d, "B5")[0]
    d = run_decide(str(tmp_path / "q"), make_round(str(tmp_path / "q")),
                   queue=["w2c_l_hybrid_120_p3", "w2c_l_same_120_p1"])
    assert not d.go and "2 of them long-test parts" in blockers(d, "B5")[0]


def test_long_control_is_checked_at_any_game_count(tmp_path):
    # long rows read in place of the round's, their control short (200 games) and off half: B4, not a note
    rd = make_round(str(tmp_path / "a"), {"h_best_120": 0.44, "c_nodur_120": 0.44})
    write_long(rd, "l_hybrid_120", [(300, 0.50)] * 5)
    write_long(rd, "l_nodur_120", [(300, 0.43)] * 5)
    write_long(rd, "l_same_120", [(100, 0.38), (100, 0.38)])
    d = run_decide(str(tmp_path / "a"), rd)
    assert not d.go and blockers(d, "B4") and "l_same_120 (long test, 2 parts)" in blockers(d, "B4")[0], d.blockers
    # short but at half: no blocker, the round's control still checked too
    rd = make_round(str(tmp_path / "b"), {"h_best_120": 0.44, "c_nodur_120": 0.44})
    write_long(rd, "l_hybrid_120", [(300, 0.50)] * 5)
    write_long(rd, "l_same_120", [(100, 0.50), (100, 0.50)])
    d = run_decide(str(tmp_path / "b"), rd)
    assert not blockers(d, "B4"), d.blockers
    assert any(n.startswith("control ok: l_same_120") for n in d.notes)
    assert any(n.startswith("control ok: c_same_120") for n in d.notes)
    # long rows read with no long control at all: B4
    rd = make_round(str(tmp_path / "c"), {"h_best_120": 0.44, "c_nodur_120": 0.44})
    write_long(rd, "l_hybrid_120", [(300, 0.52)] * 5)
    d = run_decide(str(tmp_path / "c"), rd)
    assert not d.go and "no l_same_120 games" in blockers(d, "B4")[0]


def test_wide_chosen_from_the_ab(tmp_path):
    rd = make_round(str(tmp_path))
    ab = write_ab(str(tmp_path), [(0.50, 0.53), (0.49, 0.54), (0.51, 0.55)])
    d = run_decide(str(tmp_path), rd, ab)
    assert d.go and d.wide == 1 and "WIDE=1" in d.command


def test_narrow_when_wide_is_not_clearly_ahead_or_too_short(tmp_path):
    rd = make_round(str(tmp_path))
    close = write_ab(str(tmp_path / "c"), [(0.50, 0.52), (0.50, 0.52)])        # +0.02 < 0.03
    assert run_decide(str(tmp_path), rd, close).wide == 0
    last_behind = write_ab(str(tmp_path / "l"), [(0.45, 0.55), (0.52, 0.51)])  # mean ahead, last behind
    assert run_decide(str(tmp_path), rd, last_behind).wide == 0
    one = write_ab(str(tmp_path / "o"), [(0.40, 0.60)])                       # one iteration only
    assert run_decide(str(tmp_path), rd, one).wide == 0


# ---------------------------------------------------------------------- blockers


def blockers(d, code):
    return [b for b in d.blockers if b.startswith(code)]


def test_blocker_build(tmp_path):
    d = run_decide(str(tmp_path), make_round(str(tmp_path), build=False))
    assert not d.go and blockers(d, "B1")


def test_blocker_gpu_missing_and_failed(tmp_path):
    d = run_decide(str(tmp_path / "m"), make_round(str(tmp_path / "m"), gpu=None))
    assert not d.go and blockers(d, "B2")
    d = run_decide(str(tmp_path / "f"), make_round(str(tmp_path / "f"), gpu="FAIL"))
    assert not d.go and "wide_parity" in blockers(d, "B2")[0]


def test_blocker_smoke(tmp_path):
    smokes = tuple(s for s in dfr.SMOKES if s != ("cont", "wide"))
    d = run_decide(str(tmp_path), make_round(str(tmp_path), smokes=smokes))
    assert not d.go and len(blockers(d, "B3")) == 1 and "smoke_cont/wide" in blockers(d, "B3")[0]


def test_blocker_control_off_half(tmp_path):
    d = run_decide(str(tmp_path), make_round(str(tmp_path), {"c_same_120": 0.42}))
    assert not d.go and blockers(d, "B4")
    d = run_decide(str(tmp_path / "ok"), make_round(str(tmp_path / "ok"), {"c_same_120": 0.47}))
    assert d.go


def test_blocker_rows_missing_short_errors_or_queued(tmp_path):
    d = run_decide(str(tmp_path / "m"), make_round(str(tmp_path / "m"), {"c_waiteval_120": None}))
    assert not d.go and "c_waiteval_120 no games" in blockers(d, "B5")[0]
    d = run_decide(str(tmp_path / "s"), make_round(str(tmp_path / "s"), games={**{r: 600 for r in ROWS},
                                                                                "c_ref_120": 200}))
    assert not d.go and "c_ref_120 200 games" in blockers(d, "B5")[0]
    rd = make_round(str(tmp_path / "e"))
    write_row(os.path.join(rd, "rows", "c_joint_120"), 300, 0.4, errors=40)
    d = run_decide(str(tmp_path / "e"), rd)
    assert not d.go and "c_joint_120 errors 40" in blockers(d, "B5")[0]
    d = run_decide(str(tmp_path / "q"), make_round(str(tmp_path / "q")), queue=["w2c_c_all_120"])
    assert not d.go and "still running" in blockers(d, "B5")[0]


def test_blocker_rate_and_budget(tmp_path):
    d = run_decide(str(tmp_path / "r"), make_round(str(tmp_path / "r"), rate=None))
    assert not d.go and blockers(d, "B6") and d.command is None
    # the wide bench missing blocks only a wide choice
    rd = make_round(str(tmp_path / "w"), bench_widths=("narrow",))
    ab = write_ab(str(tmp_path / "w"), [(0.50, 0.55), (0.50, 0.56)])
    assert blockers(run_decide(str(tmp_path / "w"), rd, ab), "B6")
    d = run_decide(str(tmp_path / "b"), make_round(str(tmp_path / "b"), rate=40.0))
    assert not d.go and "B7 budget" in blockers(d, "B7")[0] and d.gpu_hours > 10_000


def test_main_without_a_round_is_no_go(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(dfr, "REPO", str(tmp_path))
    assert dfr.main([]) == 1
    assert capsys.readouterr().out.startswith("NO-GO")


# ---------------------------------------------------------------------- configs agree with the rows


def test_rows_in_launch_checks_match_the_configs():
    text = open(os.path.join(REPO, "cluster", "launch_checks.sh")).read()
    block = text[text.index('if [ "$WHAT" = wave2c ]'):]
    block = block[:block.index("\nfi\n")]
    rows = re.findall(r"^\s*w2c\s+(\S+)", block, re.M)
    assert sorted(rows) == sorted(ROWS) and len(rows) == 12
    for name, (row, extra) in CONFIGS.items():
        line = next(l for l in block.splitlines() if re.match(rf"\s*w2c\s+{row}\s", l))
        if "team-tree-reuse" in extra or "continuity" in extra:
            assert "$TOL" in line, row          # the 1000 ms tolerance rides on every row with fix 1 on
        if "own-pace-below-root false" in extra:
            assert '\\"own_pace_below_root\\": false' in line
        if "--hybrid" in extra:
            assert '"durative_tree": false, "sit_gate": true, "hybrid": true' in line, row


# ---------------------------------------------------------------------- dry runs with a fake sbatch

FAKE_SBATCH = r"""#!/bin/bash
n=$(( $(cat "$FAKE_DIR/n" 2>/dev/null || echo 1000) + 1 )); echo $n > "$FAKE_DIR/n"
printf '%s\t%s\tNAME=%s\tRUN=%s\tRUNS_DIR=%s\tOUT=%s\tCONFIG=%s\tWAVE2_EXTRA=%s\tMAX_ITER=%s\tWIDE=%s\tAB=%s\tGAMES=%s\n' \
  "$n" "$*" "${NAME:-}" "${RUN:-}" "${RUNS_DIR:-}" "${OUT:-}" "${CONFIG:-}" "${WAVE2_EXTRA:-}" "${MAX_ITER:-}" \
  "${WIDE:-}" "${AB:-}" "${GAMES_PER_ITER:-}" >> "$FAKE_DIR/calls.tsv"
case " $* " in *" --parsable "*) echo "$n" ;; *) echo "Submitted batch job $n" ;; esac
"""


@pytest.fixture
def fake_slurm(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "sbatch").write_text(FAKE_SBATCH)
    (bin_dir / "squeue").write_text("#!/bin/bash\ncat \"$FAKE_DIR/queue\" 2>/dev/null\nexit 0\n")
    for f in ("sbatch", "squeue"):
        os.chmod(bin_dir / f, 0o755)
    env = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}", "FAKE_DIR": str(tmp_path), "NO_PULL": "1",
           "USER": os.environ.get("USER", "me")}
    for k in ("SBATCH_DEPENDENCY", "ROUND", "RUN", "CONFIG", "WAVE2_EXTRA", "OUT", "NAME"):
        env.pop(k, None)
    return tmp_path, env


def calls(tmp):
    out = []
    for line in open(tmp / "calls.tsv").read().splitlines():
        parts = line.split("\t")
        rec = {"id": parts[0], "args": parts[1]}
        rec.update(p.split("=", 1) for p in parts[2:])
        jm = re.search(r"-J (\S+)", rec["args"])
        rec["job"] = jm.group(1) if jm else None
        dm = re.search(r"--dependency[= ](\S+)", rec["args"])
        rec["dep"] = dm.group(1) if dm else None
        out.append(rec)
    return out


needs_w2 = pytest.mark.skipif(not os.path.isdir(os.path.join(REPO, "third_party", "wave2-hivemind", ".git"))
                              or shutil.which("git") is None, reason="needs the wave2-hivemind checkout")


@needs_w2
def test_launch_wave2c_dry_run_dependencies(fake_slurm):
    tmp, env = fake_slurm
    env["W2C_RUNS"] = str(tmp / "wave2c")
    p = subprocess.run(["bash", "cluster/launch_wave2c.sh"], cwd=REPO, env=env, capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    c = calls(tmp)
    build = next(r for r in c if "w2c_build" in r["args"])
    assert build["dep"] is None and "build.ok" in build["args"]
    gpu = next(r for r in c if r["args"].endswith("cluster/wave2c_gpu_check.sbatch"))
    assert gpu["dep"] == f"afterok:{build['id']}" and gpu["OUT"].endswith("wave2c/r1")
    rows = [r for r in c if r["args"].endswith("cluster/eval.sbatch")]
    assert sorted(r["job"] for r in rows) == sorted(f"w2c_{row}" for row in ROWS)
    assert all(r["dep"] == f"afterok:{build['id']}" for r in rows)
    assert all(r["NAME"].endswith("/rows/" + r["job"][4:]) for r in rows)
    bench = [r for r in c if r["args"].endswith("cluster/wave2c_bench.sbatch")]
    smoke = [r for r in c if r["args"].endswith("cluster/wave2_iterate.sbatch")]
    assert sorted(r["CONFIG"] for r in bench) == sorted(BENCHED)
    assert all(r["dep"] == f"afterok:{gpu['id']}" for r in bench + smoke)
    assert {r["RUN"]: r["WAVE2_EXTRA"] for r in smoke} == {f"smoke_{c}": CONFIGS[c][1] for c in BENCHED}
    assert all(r["AB"] == "1" and r["RUNS_DIR"].endswith("wave2c/r1") for r in smoke)
    watchdog = next(r for r in c if r["args"].endswith("cluster/wave2c_watchdog.sh"))
    assert watchdog["dep"] == f"afterany:{build['id']}"
    assert len(c) == 1 + 1 + 12 + 3 + 3 + 1
    assert open(tmp / "wave2c" / "LATEST").read().strip() == "r1"
    jobs = open(tmp / "wave2c" / "r1" / "jobs.tsv").read()
    assert f"build\t{build['id']}" in jobs and f"gpu\t{gpu['id']}" in jobs
    # a second launch takes the next round folder, never the old one
    p = subprocess.run(["bash", "cluster/launch_wave2c.sh"], cwd=REPO, env=env, capture_output=True, text=True)
    assert p.returncode == 0 and "wave2c/r2" in p.stdout


@needs_w2
def test_launch_wave2c_refuses_while_old_jobs_run(fake_slurm):
    tmp, env = fake_slurm
    env["W2C_RUNS"] = str(tmp / "wave2c")
    (tmp / "queue").write_text("ab2_w1\n")
    p = subprocess.run(["bash", "cluster/launch_wave2c.sh"], cwd=REPO, env=env, capture_output=True, text=True)
    assert p.returncode == 1 and "ab2_w1" in p.stderr and not (tmp / "calls.tsv").exists()


@needs_w2
def test_launch_full_run_dry_run_chain(fake_slurm):
    tmp, env = fake_slurm
    env.update({"FULL_RUNS": str(tmp / "full"), "RUN": "full1", "CONFIG": "cont", "WIDE": "1", "CHAIN": "3",
                "GAMES_PER_ITER": "50000"})
    p = subprocess.run(["bash", "cluster/launch_full_run.sh"], cwd=REPO, env=env, capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    c = calls(tmp)
    chain = [r for r in c if r["args"].endswith("cluster/wave2_iterate.sbatch")]
    assert len(chain) == 3
    for r in chain:
        assert r["job"] == "full_full1" and r["dep"] == "singleton"
        assert (r["WAVE2_EXTRA"], r["WIDE"], r["MAX_ITER"], r["GAMES"]) == (CONFIGS["cont"][1], "1", "10", "50000")
        assert r["RUNS_DIR"] == str(tmp / "full")
    assert any(r["args"].endswith("cluster/full_run_watchdog.sh") for r in c)
    # the same RUN again is refused, RESUME=1 with the same settings adds to it, with others is refused
    p = subprocess.run(["bash", "cluster/launch_full_run.sh"], cwd=REPO, env=env, capture_output=True, text=True)
    assert p.returncode == 1 and "exists" in p.stderr
    p = subprocess.run(["bash", "cluster/launch_full_run.sh"], cwd=REPO, env={**env, "RESUME": "1"},
                       capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    p = subprocess.run(["bash", "cluster/launch_full_run.sh"], cwd=REPO, env={**env, "RESUME": "1", "WIDE": "0"},
                       capture_output=True, text=True)
    assert p.returncode == 1 and "other settings" in p.stderr


def _git_repo(path):
    """A git repo with one commit; its HEAD."""
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    subprocess.run(["git", "-C", str(path), "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q",
                    "--allow-empty", "-m", "x"], check=True)
    return subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True,
                          check=True).stdout.strip()


@pytest.mark.skipif(not os.path.isdir(os.path.join(REPO, "runs")), reason="needs the repo's runs/ folder")
def test_launch_long_test_dry_run(fake_slurm):
    tmp, env = fake_slurm
    runs = tmp / "wave2c"
    (runs / "r1").mkdir(parents=True)
    (runs / "LATEST").write_text("r1\n")
    # W2C_RUNS under runs/ as on PACE (runs/wave2c), so the parts' NAME is the folder with runs/ stripped: eval.sbatch
    # writes runs/<NAME>, which must be the folder decide_full_run reads (<round>/long/<row>/part<k>)
    env["W2C_RUNS"] = "runs/" + os.path.relpath(runs, os.path.join(REPO, "runs"))
    w2 = tmp / "w2"
    pin = _git_repo(w2)
    env["W2_DIR"] = str(w2)
    (runs / "r1" / "engine.ref").write_text(pin + "\n")
    run = lambda **kw: subprocess.run(["bash", "cluster/launch_long_test.sh"], cwd=REPO, env={**env, **kw},
                                      capture_output=True, text=True)
    evals = lambda c: [r for r in c if r["args"].endswith("cluster/eval.sbatch")]
    # no build queued, no BUILD and no build.ok: refused, nothing submitted
    p = run()
    assert p.returncode == 1 and "build.ok" in p.stderr and not (tmp / "calls.tsv").exists()
    # the round's build (its jobs.tsv) in the queue: every part waits for it
    (runs / "r1" / "jobs.tsv").write_text("build\t4242\ngpu\t4243\n")
    (tmp / "queue").write_text("4242 w2c_build\n4250 w2c_c_same_120\n")
    p = run()
    assert p.returncode == 0, p.stderr
    c = calls(tmp)
    assert len(evals(c)) == 3 * 5
    assert all(r["dep"] == "afterok:4242" and "--parsable" in r["args"] for r in evals(c))
    assert sorted({r["NAME"].rsplit("/", 2)[-2] for r in evals(c)}) == sorted(LONG_ROWS)
    for lrow in LONG_ROWS:
        mine = [r for r in evals(c) if f"/long/{lrow}/" in r["NAME"]]
        assert [r["NAME"].rsplit("/", 1)[1] for r in mine] == [f"part{k}" for k in range(1, 6)]
        assert [r["job"] for r in mine] == [f"w2c_{lrow}_p{k}" for k in range(1, 6)]
        for k, r in enumerate(mine, 1):
            assert not r["NAME"].startswith("runs/"), r["NAME"]
            written = os.path.realpath(os.path.join(REPO, "runs", r["NAME"]))
            assert written == os.path.realpath(runs / "r1" / "long" / lrow / f"part{k}")
            assert os.path.isdir(written)
    # the long test's own watchdog: afterany on the build, long parts only
    wd = [r for r in c if r["args"].endswith("cluster/wave2c_watchdog.sh")]
    assert len(wd) == 1 and wd[0]["job"] == "w2c_watchdog_long" and wd[0]["dep"] == "afterany:4242"
    assert "GPU-hours: at most 120" in p.stdout
    tsv = [l for l in (runs / "r1" / "long" / "jobs.tsv").read_text().splitlines() if not l.startswith("watchdog")]
    assert len(tsv) == 15 and "seed 1001" in tsv[0] and "seed 1005" in tsv[4]
    # a build of another round in the queue moves the checkout: refused, nothing submitted
    n = len(calls(tmp))
    (tmp / "queue").write_text("4242 w2c_build\n5555 w2c_build\n")
    p = run()
    assert p.returncode == 1 and "5555" in p.stderr and len(calls(tmp)) == n
    (tmp / "queue").write_text("5555 w2c_build\n")
    (runs / "r1" / "build.ok").write_text("Sun Nov 22\n")
    p = run()
    assert p.returncode == 1 and "5555" in p.stderr and len(calls(tmp)) == n
    (runs / "r1" / "build.ok").unlink()
    # a second launch never reuses a folder: parts 6-10 with new seeds; BUILD= works when the queue has none
    (tmp / "queue").write_text("")
    p = run(PARTS="5", BUILD="77")
    assert p.returncode == 0, p.stderr
    c2 = evals(calls(tmp)[n:])
    assert {r["NAME"].rsplit("/", 1)[1] for r in c2} == {f"part{k}" for k in range(6, 11)}
    assert all(r["dep"] == "afterok:77" for r in c2)
    assert "seed 1010" in (runs / "r1" / "long" / "jobs.tsv").read_text()
    # after the build passed (build.ok), no dependency at all, while the checkout is at the round's pin
    (runs / "r1" / "build.ok").write_text("Sun Nov 22\n")
    n = len(calls(tmp))
    p = run(PARTS="1")
    assert p.returncode == 0, p.stderr
    c3 = calls(tmp)[n:]
    assert len(evals(c3)) == 3 and all(r["dep"] is None for r in c3)
    assert any(r["job"] == "w2c_watchdog_long" for r in c3)
    # ... and refused once the checkout moved off it (a later round's build, a manual rebuild)
    subprocess.run(["git", "-C", str(w2), "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q",
                    "--allow-empty", "-m", "y"], check=True)
    n = len(calls(tmp))
    p = run(PARTS="1")
    assert p.returncode == 1 and pin in p.stderr and len(calls(tmp)) == n


@pytest.mark.parametrize("long_only", [False, True])
def test_watchdog_cancels_a_long_row_with_errors(fake_slurm, long_only):
    """wave2c_watchdog.sh reads each long-test row merged over its parts and cancels all of a failing row's parts, in
    the round's copy and in the long test's own (LONG=1, which ignores rows/)."""
    tmp, env = fake_slurm
    out = tmp / "wave2c" / "r1"
    # l_hybrid_120: 12 errors in 160 games over two parts (7.5%, neither part alone has 100); l_nodur_120 clean
    write_row(str(out / "long" / "l_hybrid_120" / "part1"), 80, 0.5, errors=6)
    write_row(str(out / "long" / "l_hybrid_120" / "part2"), 80, 0.5, errors=6, opening_base=1000)
    write_row(str(out / "long" / "l_nodur_120" / "part1"), 160, 0.5)
    write_row(str(out / "rows" / "c_reuse_120"), 120, 0.5, errors=20)
    (tmp / "queue").write_text("w2c_l_hybrid_120_p1\nw2c_l_hybrid_120_p2\nw2c_l_nodur_120_p1\nw2c_c_reuse_120\n"
                               "w2c_watchdog\n")
    (tmp / "bin" / "scancel").write_text("#!/bin/bash\necho \"$*\" >> \"$FAKE_DIR/scancel.log\"\n")
    os.chmod(tmp / "bin" / "scancel", 0o755)
    p = subprocess.run(["bash", "cluster/wave2c_watchdog.sh"], cwd=REPO, capture_output=True, text=True,
                       env={**env, "OUT": str(out), "BUILD": "none", "GPU": "none", "LONG": "1" if long_only else "0",
                            "POLL_S": "0", "LIFE_S": "0"})
    assert p.returncode == 0, p.stderr
    cancelled = sorted(l.split("-n ")[1] for l in (tmp / "scancel.log").read_text().splitlines())
    expected = ["w2c_l_hybrid_120_p1", "w2c_l_hybrid_120_p2"] + ([] if long_only else ["w2c_c_reuse_120"])
    assert cancelled == sorted(expected)
    assert "long row l_hybrid_120 has errors over its parts: games errors = 172 12" in (out / "watchdog.log").read_text()
    handover = calls(tmp)
    assert len(handover) == 1 and ("w2c_watchdog_long" in handover[0]["args"]) == long_only


def test_eval_refuses_a_checkout_off_engine_ref():
    """eval.sbatch's ENGINE_REF guard (the long test's parts carry their round's pin)."""
    text = open(os.path.join(REPO, "cluster", "eval.sbatch")).read()
    assert 'if [ "$have" != "$ENGINE_REF" ]' in text and "refused" in text
    assert "ENGINE_REF=\"$PIN\"" in open(os.path.join(REPO, "cluster", "launch_long_test.sh")).read()


def test_launch_long_test_row_options():
    """The long rows run h_best_120's and c_nodur_120's arena options, and the control is the hybrid on both sides."""
    text = open(os.path.join(REPO, "cluster", "launch_long_test.sh")).read()
    checks = open(os.path.join(REPO, "cluster", "launch_checks.sh")).read()
    hyb = re.search(r"^HYB='(.*)'$", text, re.M).group(1)
    nodur = re.search(r"^NODUR='(.*)'$", text, re.M).group(1)
    assert re.search(rf"w2c\s+h_best_120\s+upstream-it13\s+'{re.escape(hyb)}'", checks)
    assert re.search(rf"w2c\s+c_nodur_120\s+upstream-it13\s+'{re.escape(nodur)}'", checks)
    rows = re.findall(r"^row\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)$", text, re.M)
    assert [r[0] for r in rows] == list(LONG_ROWS)
    assert rows == [("l_hybrid_120", "upstream-it13", '"$HYB"', '""'), ("l_nodur_120", "upstream-it13", '"$NODUR"', '""'),
                    ("l_same_120", "wave2-self", '"$HYB"', '"$HYB"')]


# ---------------------------------------------------------------------- wave2_iterate MAX_ITER and build_sp1 stamp


@pytest.fixture
def fake_w2(tmp_path):
    """A wave2-hivemind stand-in: a git repo, engines that list --durative-tree, a .venv/bin/python."""
    w2 = tmp_path / "w2"
    (w2 / "engine" / "build-ninja").mkdir(parents=True)
    (w2 / ".venv" / "bin").mkdir(parents=True)
    os.symlink(sys.executable, w2 / ".venv" / "bin" / "python")
    eng = w2 / "engine" / "build-ninja" / "hivemind"
    eng.write_text("#!/bin/bash\necho '  --durative-tree'\n")
    os.chmod(eng, 0o755)
    subprocess.run(["git", "init", "-q", str(w2)], check=True)
    subprocess.run(["git", "-C", str(w2), "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q",
                    "--allow-empty", "-m", "x"], check=True)
    # cmake stand-in: "configure" does nothing, "--build" writes build-sp1/hivemind
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "cmake").write_text("#!/bin/bash\necho cmake \"$@\" >> \"$FAKE_DIR/cmake.log\"\n"
                                   "if [ \"$1\" = --build ]; then mkdir -p build-sp1; cp build-ninja/hivemind "
                                   "build-sp1/hivemind; fi\n")
    os.chmod(bin_dir / "cmake", 0o755)
    env = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}", "FAKE_DIR": str(tmp_path)}
    env.pop("SLURM_JOB_ID", None)
    return w2, env


def test_build_sp1_rebuilds_only_when_stale(fake_w2, tmp_path):
    w2, env = fake_w2
    run = lambda: subprocess.run(["bash", "cluster/build_sp1.sh", str(w2)], cwd=REPO, env=env,
                                 capture_output=True, text=True)
    p = run()
    assert p.returncode == 0 and "building" in p.stdout
    head = subprocess.run(["git", "-C", str(w2), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    assert (w2 / "engine" / "build-sp1" / ".talisman_commit").read_text().strip() == head
    assert "up to date" in run().stdout
    (w2 / "engine" / "build-sp1" / ".talisman_commit").write_text("0" * 40 + "\n")   # built from an older pin
    assert "building" in run().stdout


def test_wave2_iterate_stops_at_max_iter(fake_w2, tmp_path):
    w2, env = fake_w2
    root = tmp_path / "runs_full" / "full1"
    (root / "runs" / "rl-it10" / "stages").mkdir(parents=True)
    for i in range(1, 11):
        (root / "runs" / f"rl-it{i}" / "stages").mkdir(parents=True, exist_ok=True)
        (root / "runs" / f"rl-it{i}" / "stages" / "iteration.done").write_text("{}\n")
    for step in ("engines", "env", "init"):
        (root / f".done_{step}").write_text("\n")
    env.update({"RUN": "full1", "RUNS_DIR": os.path.relpath(tmp_path / "runs_full", REPO), "W2": str(w2),
                "ITERS": "1", "MAX_ITER": "10"})
    p = subprocess.run(["bash", "cluster/wave2_iterate.sbatch"], cwd=REPO, env=env, capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    assert "nothing left to run" in p.stdout and "rl-loop it" not in p.stdout
    assert (root / "results.tsv").read_text().count("rl-it") == 10
