"""Starts the engine once with a long wait, so the TensorRT engine file exists before anything that only waits 120 s.

The first start on a node can build that file (200 s in job 15498121). Run it at the top of every cluster job.
With no arguments it warms the stock binary and network; ``--binary hivemind-clock --model <v2 onnx>`` warms the
clock build (one engine file per network and GPU type). ``--kind upstream`` / ``--kind wave2`` warms stock upstream v3
or the wave 2 engine (their own binaries and twin-s-clock network; ``--binary``/``--model`` override).
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from arena.players.hivemind import DEFAULT_MODEL, HivemindProcess, resolve_binary  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
ap.add_argument("--binary", default=None, help="hivemind (default), hivemind-clock, or a path")
ap.add_argument("--model", default=None)
ap.add_argument("--kind", choices=("hivemind", "upstream", "wave2"), default="hivemind")
args = ap.parse_args()

start = time.monotonic()
if args.kind == "hivemind":
    process = HivemindProcess(resolve_binary(args.binary), args.model or DEFAULT_MODEL, timeout_s=1800)
else:
    from arena.players import upstream as up

    w2 = args.kind == "wave2"
    process = up.UpstreamHivemindProcess(args.binary or (up.WAVE2_BINARY if w2 else up.UPSTREAM_BINARY),
                                         args.model or (up.WAVE2_MODEL if w2 else up.UPSTREAM_MODEL),
                                         cwd=up.WAVE2_ENGINE_DIR if w2 else None, timeout_s=1800)
process.start()
process.close()
print(f"{args.kind} engine first start took {time.monotonic() - start:.0f} s")
