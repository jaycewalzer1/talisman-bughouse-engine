#!/usr/bin/env python3
"""Deep search targets for a run's frozen probe set (runs/full/<RUN>/probe/, from cluster/probe_net.py build):
each probe position whose self-play segment has a state dump (--record-state-output) is searched again at 16 x 800
nodes with the current best network, into runs/full/<RUN>/balance/probe/frozen_deep_vV/ and frozen_deep_vV.json
(docs/search_balance.md sections 3 and 6). A thin wrapper over `python -m hivemind.cli.reanalyse probe`.

    third_party/wave2-hivemind/.venv/bin/python cluster/probe_deep_targets.py \
        --run runs/full/hybrid1 --model runs/full/hybrid1/models/twin-s-clock-w2.onnx \
        --engine third_party/wave2-hivemind/engine/build-balance-sp1/hivemind [--nodes 12800] [--dry-run]

Runs whose games predate --record-state-output have no state dumps: the command then exits 2 without writing,
and the probe targets come from probe-mode self-play instead (spec section 3).
"""

from pathlib import Path
import sys

FORK_SRC = Path(__file__).resolve().parents[1] / "third_party" / "wave2-hivemind" / "src"
sys.path.insert(0, str(FORK_SRC))

from hivemind.cli import reanalyse  # noqa: E402


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv in (["-h"], ["--help"]):
        print(__doc__)
    return reanalyse.main(["probe", *argv])


if __name__ == "__main__":
    sys.exit(main())
