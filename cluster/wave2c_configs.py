"""The wave 2c candidate configurations, in one place: each arena row of the readiness round and the rl-loop flags
(WAVE2_EXTRA of cluster/wave2_iterate.sbatch) that train the same engine settings.

    .venv/bin/python cluster/wave2c_configs.py nodur       # prints that config's WAVE2_EXTRA
    .venv/bin/python cluster/wave2c_configs.py --list      # name, row, WAVE2_EXTRA

Used by cluster/launch_wave2c.sh (bench and smoke jobs), cluster/launch_full_run.sh (CONFIG=<name>),
cluster/wave2_selfplay_bench.py and cluster/decide_full_run.py. The arena side of each row is in
cluster/launch_checks.sh (wave2c block); tests check the two agree. Fix 1 (team tree reuse) runs at a 1000 ms
tolerance wherever it is on (docs/wave2_local_check.md 10.3), set here and in the rows, not as the engine default.
No imports beyond the standard library: the wave2-hivemind env runs it too.
"""

from __future__ import annotations

import sys

TOL = "--team-tree-reuse-tolerance-ms 1000"

# name -> (arena row of the readiness round, rl-loop WAVE2_EXTRA on top of the --wave2 preset)
CONFIGS: dict[str, tuple[str, str]] = {
    "nodur": ("c_nodur_120", "--durative-tree false --sit-gate true"),
    "cont": ("c_all_120", f"--continuity {TOL}"),
    "cont_nod1": ("c_all_nod1_120", f"--continuity --own-pace-below-root false {TOL}"),
    "reuse": ("c_reuse_120", f"--team-tree-reuse true {TOL}"),
    "joint": ("c_joint_120", "--joint-plans true"),
    "carry": ("c_carry_120", "--plan-carry true"),
    "ownpace": ("c_ownpace_120", "--own-pace-below-root true"),
    "waiteval": ("c_waiteval_120", "--wait-node-acting-eval true"),
    "mateblock": ("c_mateblock_120", "--blockable-check-not-mate true"),
}
# The two configs the round benches and smoke-trains (the full run's candidates); a chosen config outside these
# borrows the measurements of BENCHED_AS (its nearest benched config) and says so.
BENCHED = ("nodur", "cont")
BENCHED_AS = {"nodur": "nodur", "cont": "cont", "cont_nod1": "cont", "reuse": "cont", "joint": "cont",
              "carry": "cont", "ownpace": "cont", "waiteval": "cont", "mateblock": "cont"}
SINGLE_SWITCH = ("reuse", "joint", "carry", "ownpace", "waiteval", "mateblock")
# Every row of `launch_checks.sh wave2c`: the configs' rows plus the control and the durative-on reference.
ROWS = tuple(row for row, _ in CONFIGS.values()) + ("c_same_120", "c_ref_120")


def extra(name: str) -> str:
    if name not in CONFIGS:
        raise SystemExit(f"unknown config {name!r}: one of {', '.join(CONFIGS)}")
    return CONFIGS[name][1]


if __name__ == "__main__":
    if sys.argv[1:] == ["--list"]:
        for name, (row, flags) in CONFIGS.items():
            print(f"{name:<10} {row:<16} {flags}")
    elif len(sys.argv) == 2:
        print(extra(sys.argv[1]))
    else:
        raise SystemExit(__doc__)
