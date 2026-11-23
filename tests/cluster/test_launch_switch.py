"""cluster/launch_full_run.sh RESUME=1 REPLACE_PENDING=1: switching a running run to concurrent self-play, more
iterations and a GPU-hour cap, with the run's watchdog replaced (NEW_WATCHDOG=1), against a fake Slurm
(tests/cluster/fake_slurm). Nothing runs: the jobs are only queued.
    .venv/bin/python -m pytest tests/cluster/test_launch_switch.py -q
"""

import os
import subprocess

import pytest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FAKE = os.path.join(REPO, "tests", "cluster", "fake_slurm")
LAUNCH = os.path.join(REPO, "cluster", "launch_full_run.sh")


def pinned():
    head = subprocess.run(["git", "-C", os.path.join(REPO, "third_party", "wave2-hivemind"), "rev-parse", "HEAD"],
                          capture_output=True, text=True).stdout.strip()
    return head and head == open(os.path.join(REPO, "cluster", "wave2_engine.ref")).read().strip()


pytestmark = pytest.mark.skipif(not pinned(), reason="the wave2-hivemind checkout is not at the pin")


class Slurm:
    def __init__(self, tmp):
        self.q = tmp / "q"
        self.q.mkdir()
        self.runs = tmp / "full"
        self.env = {**os.environ, "FAKE_SLURM_Q": str(self.q), "PATH": f"{FAKE}:{os.environ['PATH']}",
                    "FULL_RUNS": str(self.runs), "NO_PULL": "1", "USER": os.environ.get("USER", "me")}
        for key in ("WORKERS", "SEGMENT_GAMES", "PER_JOB", "ITERS", "GPU_HOURS", "BALANCE", "CHAIN", "MONITOR",
                    "ANCHORS", "CAP_SINCE", "OLD_WATCHDOG", "NEW_WATCHDOG", "DRY_RUN", "BALANCE_ANY_SEGMENT"):
            self.env.pop(key, None)

    def launch(self, **settings):
        env = {**self.env, **{k: str(v) for k, v in settings.items()}}
        return subprocess.run(["bash", LAUNCH], env=env, capture_output=True, text=True, cwd=REPO)

    def jobs(self):
        out = {}
        for name in os.listdir(self.q):
            if name.endswith(".state"):
                n = name[:-6]
                out[n] = {"name": self.read(n, "name"), "state": self.read(n, "state"), "dep": self.read(n, "dep")}
        return out

    def read(self, n, what):
        return (self.q / f"{n}.{what}").read_text().strip()

    def env_of(self, n):
        env = {}
        for line in (self.q / f"{n}.env").read_text().splitlines():
            if line.startswith("declare -x ") and "=" in line:
                key, _, value = line[11:].partition("=")
                env[key] = value.strip('"')
        return env

    def set(self, n, state):
        (self.q / f"{n}.state").write_text(state + "\n")

    def log(self):
        return (self.q / "sbatch.log").read_text().splitlines()


@pytest.fixture
def hybrid(tmp_path):
    """A hybrid1-like run: launched single-GPU with search balance, its first job running, its watchdog running."""
    slurm = Slurm(tmp_path)
    first = slurm.launch(RUN="h1", CONFIG="hybrid", WIDE=0, CHAIN=3, GAMES_PER_ITER=10000, ITERS=4, BALANCE=1)
    assert first.returncode == 0, first.stderr
    jobs = slurm.jobs()
    full = sorted(n for n, j in jobs.items() if j["name"] == "full_h1")
    (watchdog,) = [n for n, j in jobs.items() if j["name"] == "full_watchdog"]
    slurm.set(full[0], "RUNNING")
    slurm.set(watchdog, "RUNNING")
    (slurm.q / f"{watchdog}.out").write_text("2026-11-22 15:13:54 watching full_h1 (monitor 1, anchors hybrid-it13)\n")
    (slurm.runs / "h1" / "watchdog.log").write_text(
        "2026-11-22 12:28:44 watching full_h1\n2026-11-22 15:13:54 watching full_h1 (monitor 1, anchors hybrid-it13)\n")
    # Another run's watchdog, which must be left alone.
    other = slurm.launch(RUN="other", CONFIG="hybrid", WIDE=0, CHAIN=1, GAMES_PER_ITER=10000)
    assert other.returncode == 0, other.stderr
    (other_wd,) = [n for n, j in slurm.jobs().items() if j["name"] == "full_watchdog" and n != watchdog]
    slurm.set(other_wd, "RUNNING")
    (slurm.q / f"{other_wd}.out").write_text("2026-11-22 16:00:00 watching full_other (monitor 1, anchors off)\n")
    return slurm, full, watchdog, other_wd


SWITCH = dict(RESUME=1, REPLACE_PENDING=1, NEW_WATCHDOG=1, RUN="h1", WORKERS=5, ITERS=20, GPU_HOURS=400, CHAIN=10)


def test_switch_to_workers_more_iterations_and_a_cap(hybrid):
    slurm, full, watchdog, other_wd = hybrid
    before = len(slurm.jobs())
    dry = slurm.launch(DRY_RUN=1, **SWITCH)
    assert dry.returncode == 0, dry.stderr
    assert len(slurm.jobs()) == before  # a dry run submits and cancels nothing
    assert f"DRY_RUN sbatch --parsable --dependency=afterany:{watchdog} cluster/full_run_watchdog.sh" in dry.stderr

    result = slurm.launch(**SWITCH)
    assert result.returncode == 0, result.stderr
    jobs = slurm.jobs()
    new = sorted((n for n, j in jobs.items() if int(n) > int(other_wd)), key=int)
    new_wd, chain = new[0], new[1:]
    # The new watchdog first, waiting on the old one; then the chain; then the cancels.
    assert jobs[new_wd] == {"name": "full_watchdog", "state": "PENDING", "dep": f"afterany:{watchdog}"}
    assert len(chain) == 10 and all(jobs[n]["name"] == "full_h1" and jobs[n]["dep"] == "singleton" for n in chain)
    wd_env = slurm.env_of(new_wd)
    assert (wd_env["GPU_HOURS"], wd_env["CAP_SINCE"], wd_env["MONITOR"], wd_env["ANCHORS"]) == (
        "400", "2026-11-22T12:28:44", "1", "hybrid-it13")
    env = slurm.env_of(chain[0])
    assert (env["WORKERS"], env["SEGMENT_GAMES"], env["ITERS"], env["MAX_ITER"], env["BALANCE"], env["GAMES_PER_ITER"]) \
        == ("5", "2000", "2", "20", "1", "10000")  # BALANCE=1: the segment the balance prior is calibrated on
    assert jobs[full[0]]["state"] == "RUNNING"  # the running job is kept
    assert [jobs[n]["state"] for n in full[1:]] == ["CANCELLED", "CANCELLED"]
    assert jobs[watchdog]["state"] == "CANCELLED" and jobs[other_wd]["state"] == "RUNNING"
    cancels = [line.split(" ", 1)[1] for line in slurm.log() if " scancel " in line]
    assert cancels == [f"scancel state=PENDING name= {full[1]} {full[2]}", f"scancel state= name= {watchdog}"]
    root = slurm.runs / "h1"
    settings = (root / "launch_settings.txt").read_text()
    assert "iters 20" in settings and "workers 5  segment_games 2000  per_job 2  gpu_hours 400  balance 1" in settings
    history = (root / "launch_settings.history").read_text().splitlines()
    assert "iters 4" in history[0] and "workers" not in history[0]
    assert history[1].split("\t")[1] == f"watchdog {watchdog} replaced by {new_wd}"


def test_a_cap_change_needs_the_new_watchdog(hybrid):
    slurm, *_ = hybrid
    before = slurm.jobs()
    result = slurm.launch(**{**SWITCH, "NEW_WATCHDOG": 0})
    assert result.returncode == 1 and "needs NEW_WATCHDOG=1" in result.stderr
    assert slurm.jobs() == before


@pytest.mark.parametrize("change, message", [
    ({"GAMES_PER_ITER": 20000}, "may change engine, balance, iters"),
    ({"WIDE": 1}, "may change engine, balance, iters"),
    ({"SEGMENT_GAMES": 300, "BALANCE_ANY_SEGMENT": 1}, "multiple of SEGMENT_GAMES"),
    ({"SEGMENT_GAMES": 500}, "calibrated on 2000-game segments"),
])
def test_what_the_lineage_is_stays(hybrid, change, message):
    slurm, *_ = hybrid
    before = slurm.jobs()
    result = slurm.launch(**{**SWITCH, **change})
    assert result.returncode == 1 and message in result.stderr
    assert slurm.jobs() == before


def test_balance_is_kept_unless_given(hybrid):
    slurm, full, *_ = hybrid
    result = slurm.launch(RESUME=1, REPLACE_PENDING=1, RUN="h1", WORKERS=5)
    assert result.returncode == 0, result.stderr
    newest = max(slurm.jobs(), key=int)
    assert slurm.env_of(newest)["BALANCE"] == "1"
    assert "balance 1" in (slurm.runs / "h1" / "launch_settings.txt").read_text()


def test_a_smaller_balance_segment_needs_the_override(hybrid):
    slurm, *_ = hybrid
    result = slurm.launch(**{**SWITCH, "SEGMENT_GAMES": 500, "BALANCE_ANY_SEGMENT": 1})
    assert result.returncode == 0, result.stderr
    assert slurm.env_of(max(slurm.jobs(), key=int))["SEGMENT_GAMES"] == "500"


def test_workers_without_balance_keep_the_500_segment(hybrid):
    slurm, *_ = hybrid
    result = slurm.launch(**{**SWITCH, "BALANCE": 0})
    assert result.returncode == 0, result.stderr
    assert slurm.env_of(max(slurm.jobs(), key=int))["SEGMENT_GAMES"] == "500"


def test_a_watchdog_without_a_log_must_be_named(hybrid):
    slurm, full, watchdog, other_wd = hybrid
    (slurm.q / f"{other_wd}.out").unlink()  # e.g. a pending hand-over copy
    before = slurm.jobs()
    result = slurm.launch(**SWITCH)
    assert result.returncode == 1 and "OLD_WATCHDOG" in result.stderr
    assert slurm.jobs() == before
    named = slurm.launch(OLD_WATCHDOG=watchdog, **SWITCH)
    assert named.returncode == 0, named.stderr
    assert slurm.jobs()[watchdog]["state"] == "CANCELLED"


def test_a_failed_cancel_of_the_old_watchdog_cancels_the_new_one(hybrid):
    slurm, full, watchdog, other_wd = hybrid
    slurm.env["FAKE_SCANCEL_FAIL"] = watchdog
    result = slurm.launch(**SWITCH)
    assert result.returncode == 0
    jobs = slurm.jobs()
    new_wd = min((n for n, j in jobs.items() if int(n) > int(other_wd)), key=int)
    assert jobs[watchdog]["state"] == "RUNNING" and jobs[new_wd]["state"] == "CANCELLED"
    assert "WITHOUT the new GPU-hour cap" in result.stderr
