"""The wave 2c candidate configurations, in one place: each arena row of the readiness round and the rl-loop flags
(WAVE2_EXTRA of cluster/wave2_iterate.sbatch) that train the same engine settings.

    .venv/bin/python cluster/wave2c_configs.py nodur       # prints that config's WAVE2_EXTRA
    .venv/bin/python cluster/wave2c_configs.py --list      # name, row, WAVE2_EXTRA

Used by cluster/launch_wave2c.sh (bench and smoke jobs), cluster/launch_full_run.sh (CONFIG=<name>),
cluster/wave2_selfplay_bench.py and cluster/decide_full_run.py. The arena side of each row is in
cluster/launch_checks.sh (wave2c block); tests check the two agree. The hybrid (spec v6,
docs/custom_engine_spec_v6_hybrid.md) is a candidate too: row h_best_120, and the long test's rows
(cluster/launch_long_test.sh, LONG_ROWS below). Fix 1 (team tree reuse) runs at a 1000 ms tolerance wherever it is
on (docs/wave2_local_check.md 10.3), set here and in the rows, not as the engine default.
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
    # spec v6 H3 at its defaults (HybridEventLands all, margin 20, prior floor 150): nodur's v3 tree and gate, plus a
    # root sit when behind that waits for the other board's thinking seat. Local: 0.502 [0.438, 0.562] vs nodur
    # over 273 games (docs/custom_engine_spec_v6_hybrid.md, Measurements)
    "hybrid": ("h_best_120", "--durative-tree false --sit-gate true --hybrid"),
}
# The three configs the round benches and smoke-trains (the full run's candidates); a chosen config outside these
# borrows the measurements of BENCHED_AS (its nearest benched config) and says so.
BENCHED = ("nodur", "cont", "hybrid")
BENCHED_AS = {"nodur": "nodur", "cont": "cont", "hybrid": "hybrid", "cont_nod1": "cont", "reuse": "cont",
              "joint": "cont", "carry": "cont", "ownpace": "cont", "waiteval": "cont", "mateblock": "cont"}
SINGLE_SWITCH = ("reuse", "joint", "carry", "ownpace", "waiteval", "mateblock")
# The hybrid configs, each a candidate of its own against nodur (no runner-up was picked: the permissive p300m0 scored
# 0.456 against nodur locally, and HybridEventLands sit was only scored on positions).
HYBRID = ("hybrid",)
# Every row of `launch_checks.sh wave2c`: the configs' rows plus the control and the durative-on reference.
ROWS = tuple(row for row, _ in CONFIGS.values()) + ("c_same_120", "c_ref_120")
# The long test (cluster/launch_long_test.sh): runs/wave2c/<ROUND>/long/<row>/part<k>, merged over its parts by
# cluster/decide_full_run.py. long row -> the round row it stands in for when it has games (c_same_120: the control).
LONG_ROWS = {"l_hybrid_120": "h_best_120", "l_nodur_120": "c_nodur_120", "l_same_120": "c_same_120"}


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
