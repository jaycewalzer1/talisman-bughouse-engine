import json

from arena.bench import main


def run(tmp_path, *extra):
    out = tmp_path / "bench.json"
    code = main(["--kind", "random", "--setups", "sharing", "--games", "2", "--out", str(out), *extra])
    return code, json.loads(out.read_text())


def test_report_has_games_per_hour_and_no_errors(tmp_path):
    code, saved = run(tmp_path)
    assert code == 0
    assert len(saved["games"]) == 2
    assert saved["report"]["projection"]["games_per_hour"] > 0
    assert saved["report"]["aborted"] == 0 and saved["report"]["invalid_moves"] == 0


def test_no_game_starts_after_the_time_limit(tmp_path):
    code, saved = run(tmp_path, "--max-seconds", "0")
    assert code == 0
    assert saved["games"] == [] and saved["report"]["projection"] is None
