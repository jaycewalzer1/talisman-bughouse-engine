"""Starts the engine once with a long wait, so the TensorRT engine file exists before anything that only waits 120 s.

The first start on a node can build that file (200 s in job 15498121). Run it at the top of every cluster job.
"""

import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from arena.players.hivemind import HivemindProcess  # noqa: E402

start = time.monotonic()
process = HivemindProcess(timeout_s=1800)
process.start()
process.close()
print(f"engine first start took {time.monotonic() - start:.0f} s")
