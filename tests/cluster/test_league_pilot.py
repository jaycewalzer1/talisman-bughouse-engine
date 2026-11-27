"""The league pilot's cluster side (docs/custom_engine_spec_v7_league.md 7): cluster/launch_league_pilot.sh against the
fake Slurm (tests/cluster/fake_slurm) with fake engines, a fake wave2-hivemind python and a fake gate2 script; the
watchdog's GPU-hour cap with gate2 jobs; cluster/league_pilot_status.py on a hand-built run folder. Nothing runs.
    .venv/bin/python -m pytest tests/cluster/test_league_pilot.py -q

The launcher's pin check reads PILOT_REF (a temp file here, cluster/wave2_engine.ref by default) and the fork's git
objects only; the fork's checkout is never moved (the pin given is its own HEAD). The dry runs go through the real
cluster/launch_full_run.sh, which only warns in DRY_RUN when the fork is off cluster/wave2_engine.ref, so unlike
test_launch_switch.py these tests do not skip when the fork has moved past the pin.
"""

import json
import os
import subprocess

import pytest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FAKE = os.path.join(REPO, "tests", "cluster", "fake_slurm")
W2 = os.path.join(REPO, "third_party", "wave2-hivemind")
PILOT = os.path.join(REPO, "cluster", "launch_league_pilot.sh")
WATCHDOG = os.path.join(REPO, "cluster", "full_run_watchdog.sh")
STATUS = os.path.join(REPO, "cluster", "league_pilot_status.py")
D5802EE = "d5802ee8f962f33bd211be499b1b9b9e5010ec57"
OLD_PIN = "0839774"  # the PACE checkout's commit: before the crash fix

ALL_OPTIONS = "--durative-tree --pace-k-min --opponent-model --think-policy"
LOOP_OPTIONS = "--gate2 --league --pace --think-labels"
CONFIG = "--durative-tree false --sit-gate true --hybrid"
PILOT_FLAGS = f"{CONFIG} --pace --league --think-labels --think-policy --think-weight 0.1 --gate2"

pytestmark = pytest.mark.skipif(not os.path.isdir(os.path.join(W2, ".git")), reason="no wave2-hivemind checkout")


def script(path, text):
    path.write_text("#!/bin/bash\n" + text)
    path.chmod(0o755)
    return path


class Pilot:
    def __init__(self, tmp):
        self.tmp = tmp
        self.q = tmp / "q"
        self.q.mkdir()
        self.runs = tmp / "full"
        self.calls = tmp / "calls"
        head = subprocess.run(["git", "-C", W2, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        (tmp / "ref").write_text(head + "\n")
        self.engine(ALL_OPTIONS)
        self.loop(LOOP_OPTIONS)
        script(tmp / "gate2.sh", f'echo "$* DRY_RUN=$DRY_RUN GATE2_FREE_K=$GATE2_FREE_K '
                                 f'dep=${{SBATCH_DEPENDENCY:-}}" >> {self.calls}/gate2\necho "fake gate2 $*"\n')
        (tmp / "think.sbatch").write_text("#!/bin/bash\n")
        (tmp / "it13.onnx").write_bytes(b"onnx")
        self.calls.mkdir()
        self.env = {**os.environ, "FAKE_SLURM_Q": str(self.q), "PATH": f"{FAKE}:{os.environ['PATH']}",
                    "FULL_RUNS": str(self.runs), "NO_PULL": "1", "USER": os.environ.get("USER", "me"), "RUN": "league1",
                    "PILOT_REF": str(tmp / "ref"), "PILOT_SP_ENGINE": str(tmp / "engine"),
                    "PILOT_MATCH_ENGINE": str(tmp / "engine"), "PILOT_PY": str(tmp / "py"),
                    "PILOT_GATE2_SCRIPT": str(tmp / "gate2.sh"), "PILOT_THINK_CHECK": str(tmp / "think.sbatch"),
                    "IT13_ONNX": str(tmp / "it13.onnx")}
        for key in ("WORKERS", "SEGMENT_GAMES", "PER_JOB", "ITERS", "GPU_HOURS", "BALANCE", "CHAIN", "CONFIG",
                    "WAVE2_EXTRA", "DRY_RUN", "BUILD", "PACE", "LEAGUE", "THINK", "GATE2", "EXTRA", "RESUME",
                    "GAMES_PER_ITER", "WIDE", "ALLOW_BUSY", "PILOT_LAUNCH", "PILOT_W2", "BASE_MIX"):
            self.env.pop(key, None)
        for key in list(self.env):
            if key.startswith("GATE2_"):
                self.env.pop(key)

    def engine(self, options):
        script(self.tmp / "engine", f'echo "usage: hivemind {options}"\n')

    def loop(self, options):
        script(self.tmp / "py", 'if [ "$2" = hivemind.gate2 ]; then shift 3; '
                                'printf \'{"durative_tree": false, "argv": "%s"}\\n\' "$*"; exit 0; fi\n'
                                f'echo "usage: rl-loop {options}"\n')

    def launch(self, **settings):
        env = {**self.env, **{k: str(v) for k, v in settings.items()}}
        return subprocess.run(["bash", PILOT], env=env, capture_output=True, text=True, cwd=REPO)

    def jobs(self):
        out = {}
        for name in os.listdir(self.q):
            if name.endswith(".state"):
                n = name[:-6]
                out[n] = {k: (self.q / f"{n}.{k}").read_text().strip() for k in ("name", "state", "dep")}
        return out

    def gate2_calls(self):
        path = self.calls / "gate2"
        return path.read_text().splitlines() if path.exists() else []


@pytest.fixture
def pilot(tmp_path):
    return Pilot(tmp_path)


def chain_lines(err):
    return [line for line in err.splitlines() if line.startswith("DRY_RUN env ") and "-J full_league1" in line]


def test_dry_run_prints_the_whole_launch(pilot):
    r = pilot.launch(DRY_RUN=1)
    assert r.returncode == 0, r.stderr
    chain = chain_lines(r.stderr)
    assert len(chain) == 2  # CHAIN=2
    assert chain[0].startswith(f"DRY_RUN env MAX_ITER=2 RUN=league1 RUNS_DIR={pilot.runs} WIDE=0 ITERS=2 "
                               f"GAMES_PER_ITER=10000 WAVE2_EXTRA={PILOT_FLAGS} WORKERS=3 SEGMENT_GAMES=500 sbatch")
    assert "--dependency=singleton cluster/wave2_iterate.sbatch" in chain[0]
    assert "gpu_hours 100" in r.stdout and "workers 3  segment_games 500" in r.stdout
    assert "about 41 GPU-hours for iteration 1" in r.stdout and "about 30 for iteration 2" in r.stdout
    assert f"DRY_RUN sbatch --parsable -J tg_league1 -o tg_league1_%j.log {pilot.tmp / 'think.sbatch'}" in r.stderr
    tg = [line for line in r.stdout.splitlines() if line.startswith("think check: ")][0].split()[2]
    assert f"the chain's sbatch gets --dependency=afterok:{tg},singleton" in r.stderr
    assert "DRY_RUN sbatch --parsable cluster/full_run_watchdog.sh" in r.stderr
    # wave2.json from the config's wave 2 flags, the incumbent's arms through the gate2 script (dry, scratch root)
    assert f"wave2-json --wave2 {PILOT_FLAGS}" in r.stdout
    (call,) = pilot.gate2_calls()
    root, label, onnx = call.split()[:3]
    assert (label, onnx) == ("incumbent", str(pilot.tmp / "it13.onnx"))
    assert root.endswith("/league1") and "/runs/.league_pilot_dry." in root
    assert "DRY_RUN=1 GATE2_FREE_K=160 dep=" in call
    # a dry run submits nothing, leaves nothing behind
    assert pilot.jobs() == {} and not pilot.runs.exists()
    assert not [d for d in os.listdir(os.path.join(REPO, "runs")) if d.startswith(".league_pilot_dry")]


def test_build_think_check_and_chain_dependencies(pilot, tmp_path):
    """Not dry, against the fake Slurm, with a stand-in for launch_full_run.sh that submits its chain as the real one
    does (sbatch ... --dependency=singleton): the shim makes the chain wait for the build and the think check."""
    fake_launch = script(tmp_path / "launch_full_run.sh", f"""
env | grep -E '^(RUN|WAVE2_EXTRA|ITERS|GAMES_PER_ITER|WORKERS|SEGMENT_GAMES|GPU_HOURS|CHAIN|WIDE|NO_PULL|CONFIG|PACE|BUILD)=' \\
  | sort > {pilot.calls}/launch_env
mkdir -p "$FULL_RUNS/$RUN" && echo "config custom  engine x" > "$FULL_RUNS/$RUN/launch_settings.txt"
for _ in $(seq "$CHAIN"); do sbatch --parsable -J "full_$RUN" --dependency=singleton {tmp_path}/iterate.sbatch; done
""")
    pilot.engine("--durative-tree")  # old engines: BUILD=1 builds them
    r = pilot.launch(BUILD=1, PILOT_LAUNCH=fake_launch)
    assert r.returncode == 0, r.stderr
    assert "build for: " in r.stdout and "lists no --pace-k-min" in r.stdout
    jobs = pilot.jobs()
    by_name = {}
    for n, j in sorted(jobs.items(), key=lambda kv: int(kv[0])):
        by_name.setdefault(j["name"], []).append((n, j))
    ((b, build),) = by_name["lb_league1"]
    ((t, think),) = by_name["tg_league1"]
    assert build["dep"] == "" and think["dep"] == f"afterok:{b}"
    chain = by_name["full_league1"]
    assert len(chain) == 2 and all(j["dep"] == f"afterok:{b}:{t},singleton" for _, j in chain)
    assert "cluster/build.sh" in (pilot.q / f"{b}.args").read_text()
    assert "build_sp1.sh" in (pilot.q / f"{b}.args").read_text()
    env = dict(line.split("=", 1) for line in (pilot.calls / "launch_env").read_text().splitlines())
    assert env == {"RUN": "league1", "WAVE2_EXTRA": PILOT_FLAGS, "ITERS": "2", "GAMES_PER_ITER": "10000",
                   "WORKERS": "3", "SEGMENT_GAMES": "500", "GPU_HOURS": "100", "CHAIN": "2", "WIDE": "0",
                   "NO_PULL": "1"}  # no CONFIG (it would replace WAVE2_EXTRA), no switch variables
    root = pilot.runs / "league1"
    assert json.loads((root / "gate2" / "wave2.json").read_text())["argv"] == "--wave2 " + PILOT_FLAGS
    (call,) = pilot.gate2_calls()
    assert call.startswith(f"{root} incumbent {pilot.tmp / 'it13.onnx'} DRY_RUN=0") and call.endswith(f"dep=afterok:{b}")
    assert (root / "pilot_jobs.tsv").read_text() == f"build\t{b}\nthink_check\t{t}\n"


def test_a_pin_without_the_crash_fix_is_refused(pilot):
    full = subprocess.run(["git", "-C", W2, "rev-parse", OLD_PIN], capture_output=True, text=True).stdout.strip()
    (pilot.tmp / "ref").write_text(full + "\n")
    r = pilot.launch(DRY_RUN=1)
    assert r.returncode == 1 and "does not contain d5802ee" in r.stderr
    assert pilot.jobs() == {} and chain_lines(r.stderr) == []


def test_the_crash_fix_itself_is_accepted(pilot):
    (pilot.tmp / "ref").write_text(D5802EE + "\n")
    r = pilot.launch(DRY_RUN=1, PACE=0, LEAGUE=0, THINK=0, GATE2=0)
    # the fork is past d5802ee here, so its checkout is "off the pin": that needs the build, not a refusal
    assert "does not contain" not in r.stderr
    assert r.returncode == 0 or "not the pin" in r.stderr


def test_switches_off(pilot):
    pilot.engine("--durative-tree")  # none of the new options needed
    pilot.loop("--wave2")
    r = pilot.launch(DRY_RUN=1, PACE=0, LEAGUE=0, THINK=0, GATE2=0)
    assert r.returncode == 0, r.stderr
    chain = chain_lines(r.stderr)
    assert len(chain) == 2 and f"WAVE2_EXTRA={CONFIG} WORKERS=3" in chain[0]
    assert "tg_league1" not in r.stderr and "--dependency=afterok" not in r.stderr
    assert pilot.gate2_calls() == [] and "wave2-json" not in r.stdout


def test_one_switch_off_and_extra(pilot):
    r = pilot.launch(DRY_RUN=1, THINK=0, EXTRA="--league-past 2")
    assert r.returncode == 0, r.stderr
    assert f"WAVE2_EXTRA={CONFIG} --pace --league --gate2 --league-past 2 WORKERS=3" in chain_lines(r.stderr)[0]
    assert "tg_league1" not in r.stderr


def test_off_default_gate_settings_reach_rl_loop_too(pilot):
    r = pilot.launch(DRY_RUN=1, GATE2_FREE_K=110)
    assert r.returncode == 0, r.stderr
    assert "--gate2 --gate2-free-k 110 WORKERS=3" in chain_lines(r.stderr)[0]
    assert "GATE2_FREE_K=110" in pilot.gate2_calls()[0]


@pytest.mark.parametrize("engine, loop, missing", [
    ("--durative-tree", LOOP_OPTIONS, "lists no --pace-k-min"),
    ("--durative-tree --pace-k-min --think-policy", LOOP_OPTIONS, "lists no --opponent-model"),
    (ALL_OPTIONS, "--pace --league", "rl-loop lists no --think-labels"),
])
def test_missing_options_print_the_build(pilot, engine, loop, missing):
    pilot.engine(engine)
    pilot.loop(loop)
    for dry in (1, 0):
        r = pilot.launch(DRY_RUN=dry)
        assert r.returncode == 1 and missing in r.stderr
        assert "BUILD=1 RUN=league1 bash cluster/launch_league_pilot.sh" in r.stderr
        assert "cluster/build.sh" in r.stderr and "cluster/build_sp1.sh" in r.stderr
        assert pilot.jobs() == {} and chain_lines(r.stderr) == [] and not pilot.runs.exists()


def test_an_existing_run_is_refused_before_anything_is_submitted(pilot):
    (pilot.runs / "league1").mkdir(parents=True)
    r = pilot.launch(BUILD=1)
    assert r.returncode == 1 and "exists" in r.stderr
    assert pilot.jobs() == {}


# ----------------------------------------------------------------------------------------------- watchdog cap
def fake_sacct(tmp, rows):
    d = tmp / "sacct_bin"
    d.mkdir(exist_ok=True)
    body = "".join(f"echo '{r}'\n" for r in rows)
    script(d / "sacct", f'case " $* " in *" -P "*) {body};; esac\nexit 0\n')
    return d


def run_watchdog(tmp, rows, cap):
    q = tmp / "q"
    q.mkdir(exist_ok=True)
    env = {**os.environ, "FAKE_SLURM_Q": str(q), "USER": os.environ.get("USER", "me"),
           "PATH": f"{fake_sacct(tmp, rows)}:{FAKE}:{os.environ['PATH']}"}
    dummy = tmp / "job.sbatch"
    dummy.write_text("#!/bin/bash\n")
    ids = {}
    for name in ("g2_league1_0123abcd_equal_p1", "g2_other_0123abcd_equal_p1", "full_other"):
        ids[name] = subprocess.run(["sbatch", "--parsable", "-J", name, str(dummy)], env=env, capture_output=True,
                                   text=True, check=True).stdout.strip()
    wd = {**env, "RUN": "league1", "RUNS_DIR": str(tmp / "full"), "GPU_HOURS": str(cap), "MONITOR": "0",
          "POLL_S": "0"}
    r = subprocess.run(["bash", WATCHDOG], env=wd, capture_output=True, text=True, cwd=REPO, timeout=60)
    states = {name: (q / f"{n}.state").read_text().strip() for name, n in ids.items()}
    return r, (tmp / "full" / "league1" / "watchdog.log").read_text(), states


ROWS = ["full_league1|3600|billing=6,cpu=6,gres/gpu=1", "g2_league1_0123abcd_free_p2|32400|cpu=6,gres/gpu=1",
        "tg_league1|1800|cpu=6,gres/gpu=1", "g2_other_0123abcd_equal_p1|360000|cpu=6,gres/gpu=1",
        "g2_league10_0123abcd_equal_p1|360000|cpu=6,gres/gpu=1", "full_watchdog|86400|cpu=1"]


def test_gate2_jobs_count_toward_the_cap(tmp_path):
    r, log, states = run_watchdog(tmp_path, ROWS, 10)
    assert r.returncode == 0, r.stderr
    assert "GPU-hours 10.5 of 10" in log and "STOP at the cap" in log  # 1 + 9 + 0.5; other runs not counted
    assert (tmp_path / "full" / "league1" / "STOP_WORKERS").exists()
    assert states == {"g2_league1_0123abcd_equal_p1": "CANCELLED", "g2_other_0123abcd_equal_p1": "PENDING",
                      "full_other": "PENDING"}


def test_without_pilot_jobs_the_cap_is_counted_as_before(tmp_path):
    rows = [r for r in ROWS if not r.startswith(("g2_league1", "tg_"))]
    r, log, states = run_watchdog(tmp_path, rows, 10)
    assert r.returncode == 0, r.stderr
    assert "GPU-hours 1.0 of 10" in log and "STOP" not in log
    assert set(states.values()) == {"PENDING"}


def run_watchdog_cancel(tmp, failed):
    """The watchdog with the chain pending and the pilot's jobs queued; sacct reports `failed` ("<name> <state>") for
    the --name query that lists that job, nothing else."""
    q = tmp / "q"
    q.mkdir(exist_ok=True)
    d = tmp / "sacct_bin"
    d.mkdir(exist_ok=True)
    name, state = failed.split()
    script(d / "sacct", f'case " $* " in *" {name}"*|*",{name} "*) echo "999 {name} {state}" ;; esac\nexit 0\n')
    env = {**os.environ, "FAKE_SLURM_Q": str(q), "USER": os.environ.get("USER", "me"),
           "PATH": f"{d}:{FAKE}:{os.environ['PATH']}"}
    dummy = tmp / "job.sbatch"
    dummy.write_text("#!/bin/bash\n")
    ids = {}
    for job in ("full_league1", "g2_league1_0123abcd_equal_p1", "g2_league1_0123abcd_free_p2", "tg_league1",
                "g2_other_0123abcd_equal_p1", "full_other"):
        ids[job] = subprocess.run(["sbatch", "--parsable", "-J", job, str(dummy)], env=env, capture_output=True,
                                  text=True, check=True).stdout.strip()
    (q / f"{ids['g2_league1_0123abcd_free_p2']}.state").write_text("RUNNING\n")
    wd = {**env, "RUN": "league1", "RUNS_DIR": str(tmp / "full"), "MONITOR": "0", "POLL_S": "0"}
    r = subprocess.run(["bash", WATCHDOG], env=wd, capture_output=True, text=True, cwd=REPO, timeout=60)
    states = {job: (q / f"{n}.state").read_text().strip() for job, n in ids.items()}
    return r, (tmp / "full" / "league1" / "watchdog.log").read_text(), states


@pytest.mark.parametrize("failed", ["lb_league1 FAILED", "tg_league1 TIMEOUT"])
def test_a_failed_build_or_think_check_cancels_the_run(tmp_path, failed):
    """The chain waits on lb_/tg_ (afterok) and would stay pending forever: the watchdog cancels it, and with it the
    pilot's gate2 arms, running or pending, of this run only."""
    r, log, states = run_watchdog_cancel(tmp_path, failed)
    assert r.returncode == 0, r.stderr
    assert "the pilot's build or think check failed" in log and "CANCEL league pilot jobs" in log
    assert states == {"full_league1": "CANCELLED", "g2_league1_0123abcd_equal_p1": "CANCELLED",
                      "g2_league1_0123abcd_free_p2": "CANCELLED", "tg_league1": "CANCELLED",
                      "g2_other_0123abcd_equal_p1": "PENDING", "full_other": "PENDING"}


def test_a_crash_cancel_takes_the_gate2_arms_with_it(tmp_path):
    r, log, states = run_watchdog_cancel(tmp_path, "full_league1 FAILED")
    assert r.returncode == 0, r.stderr
    assert "a job failed" in log and "pilot's build" not in log
    assert states["g2_league1_0123abcd_free_p2"] == "CANCELLED" and states["g2_league1_0123abcd_equal_p1"] == "CANCELLED"
    assert states["g2_other_0123abcd_equal_p1"] == "PENDING" and states["full_other"] == "PENDING"


# ----------------------------------------------------------------------------------------------- status
def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(data if isinstance(data, str) else json.dumps(data))


def test_status_of_a_pilot_iteration(tmp_path):
    root = tmp_path / "league1"
    write(root / "launch_settings.txt", "config custom  wave2_extra --gate2  iters 2  engine abc  gpu_hours 100")
    it = root / "runs" / "rl-it1"
    st = it / "stages"
    gen = "f4ac25e32a993776" + "0" * 48
    write(st / "generator", gen + "\n")
    write(st / "league_plan.done", {"segments": {
        "seg00": {"theta": "self"}, "seg01": {"theta": "self"}, "seg02": {"theta": "past:it0"},
        "seg03": {"theta": "fixed:k110", "league": {"pfsp_weight": 0.5}}}, "summary": "4 segments"})
    write(st / "seg00.done", {"games": 500, "league": {"theta": "self", "mean_k": 150.0, "think_scale": 1.0}})
    write(st / "seg03.done", {"games": 500, "league": {"theta": "fixed:k110", "wins": 300, "draws": 0,
                                                       "losses": 200, "mean_k": 110.0, "think_scale": 1.2}})
    write(st / "think_labels.done", {"records": 1000, "labelled": 20, "rate_a": 0.3, "rate_b": 0.25,
                                     "noise_rate": 0.1, "gpu_s": 5000})
    write(st / "train.done", {"steps": 1234})
    write(st / "match_generator.done", {"score": 0.52})
    write(st / "match_twin-s-clock-w2-it0.done", {"score": 0.51})
    write(it / "models" / "C.onnx", "candidate")
    import hashlib
    cand16 = hashlib.sha256(b"candidate").hexdigest()[:16]
    for sha, score in ((cand16, 0.55), (gen[:16], 0.5)):
        for arm in ("equal", "free"):
            write(root / "gate2" / "nets" / sha / arm / "summary.json",
                  {"games": 400, "pairs": 200, "score": score, "score_ci": [score - 0.05, score + 0.05],
                   "turns_ahead": 0.5})
        write(root / "gate2" / "nets" / sha / "done", "")
    write(st / "gate2.done", {"promoted": True, "reason": "promoted: d sum +0.100",
                              "arms": {"equal": {"pairs": 200, "d": 0.05, "se": 0.03, "drop": False},
                                       "free": {"pairs": 200, "d": 0.05, "se": 0.02, "drop": False}}})
    write(st / "promote.done", {"promoted": True, "score": 0.52})
    sacct = fake_sacct(tmp_path, ROWS)
    env = {**os.environ, "PATH": f"{sacct}:{os.environ['PATH']}"}
    r = subprocess.run(["python3", STATUS, str(root), "--json"], env=env, capture_output=True, text=True, check=True)
    s = json.loads(r.stdout)
    assert s["gpu_hours"]["used"] == 10.5 and s["gpu_hours"]["cap"] == 100.0
    (i,) = s["iterations"]
    assert i["league_plan"]["by_kind"] == {"self": 2, "past": 1, "fixed": 1}
    assert i["league_plan"]["pfsp"] == {"fixed:k110": 0.5}
    assert i["selfplay"]["fixed"]["learner_score"] == 0.6 and i["selfplay"]["self"]["mean_k"] == 150.0
    assert i["think_labels"]["noise_rate"] == 0.1 and i["train_steps"] == 1234
    assert (i["gate_100ms"], i["vs_it13_100ms"]) == (0.52, 0.51)
    assert i["gate2"]["candidate"]["arms"]["equal"]["score"] == 0.55
    assert i["gate2"]["incumbent"]["sha16"] == "f4ac25e32a993776"
    assert i["gate2"]["decision"]["final"] and i["gate2"]["decision"]["promoted"]
    text = subprocess.run(["python3", STATUS, str(root)], env=env, capture_output=True, text=True, check=True).stdout
    assert "GPU-hours: 10.5 of 100.0" in text and "gate2 decision (final): promoted True" in text


def test_status_is_robust_to_missing_files(tmp_path):
    env = {**os.environ, "PATH": "/usr/bin:/bin"}  # no sacct
    root = tmp_path / "league1"
    write(root / "runs" / "rl-it1" / "stages" / "seg00.done", "{not json")
    r = subprocess.run(["python3", STATUS, str(root)], env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "GPU-hours: n/a off-cluster" in r.stdout and "league plan: none" in r.stdout
    r = subprocess.run(["python3", STATUS, str(tmp_path / "nothing")], env=env, capture_output=True, text=True)
    assert r.returncode == 0 and "no such run folder" in r.stdout
