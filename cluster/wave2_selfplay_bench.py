"""Wave 2 self-play throughput R (games per GPU-hour) for one candidate config and network width, measured on the
exact command rl-loop runs for a self-play segment (hivemind.cli.selfplay.engine_command with the --wave2 preset,
the config's WAVE2_EXTRA, the clock time model, 800 nodes +-5%, PARALLEL games at once on the one-worker engine).

    third_party/wave2-hivemind/.venv/bin/python cluster/wave2_selfplay_bench.py --config nodur --width narrow \\
        --games 150 --out runs/wave2c/<ROUND>/bench/nodur_narrow

Run with the wave2-hivemind env's Python (it imports that `hivemind` package). The network is copied into --out first
(narrow: it13, engine/models/twin-s-clock.onnx; wide: it13 widened, engine/models/twin-s-clock-wide.onnx, made by
`python -m hivemind.wave2 widen`), so jobs running at once never share a -sp1 copy; its TensorRT engine file is then
built on the first start, which the steady rate leaves out. Writes <out>/bench.json:
  games              finished games ("selfplay game " lines; rl-loop counts the same)
  wall_s             command start to exit
  games_per_gpu_hour end to end: games x 3600 / wall_s (includes the TensorRT engine build and the tail)
  games_per_gpu_hour_steady
                     games finished after the first PARALLEL x 3600 / the time between that game and the last one:
                     the rate with the pipeline full, which a long segment approaches. null under 3 x PARALLEL games
  exit, engine (path + wave2-hivemind commit), model, config, wave2_extra, width, engine_args
Exit 0 only when the engine exited 0 and played every game.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wave2c_configs import extra as config_extra  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
W2 = Path(os.environ.get("W2", REPO / "third_party" / "wave2-hivemind"))
MODELS = {"narrow": "twin-s-clock.onnx", "wide": "twin-s-clock-wide.onnx"}


def rates(times: list[float], start: float, end: float, parallel: int) -> dict:
    """End-to-end and steady games per hour from the finish times of the games (seconds, same clock as start/end)."""
    n = len(times)
    out = {"games": n, "wall_s": round(end - start, 1),
           "games_per_gpu_hour": round(n * 3600 / (end - start), 1) if end > start and n else None,
           "games_per_gpu_hour_steady": None}
    if n >= 3 * parallel:
        t0, t1 = times[parallel - 1], times[-1]
        if t1 > t0:
            out["games_per_gpu_hour_steady"] = round((n - parallel) * 3600 / (t1 - t0), 1)
    return out


def engine_args(wave2_extra: str) -> list[str]:
    """The engine options rl-loop --wave2 <wave2_extra> gives a self-play segment."""
    from hivemind import wave2
    from hivemind.cli import selfplay

    parser = argparse.ArgumentParser()
    wave2.add_arguments(parser)
    options = wave2.from_args(parser.parse_args(["--wave2", *shlex.split(wave2_extra)]))
    return [*selfplay.time_model_options("clock"), *options.engine_args()]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="a name in cluster/wave2c_configs.py")
    ap.add_argument("--width", choices=tuple(MODELS), default="narrow")
    ap.add_argument("--games", type=int, default=150)
    ap.add_argument("--parallel-games", type=int, default=6)
    ap.add_argument("--seed", type=int, default=7001)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--engine", type=Path, default=W2 / "engine" / "build-sp1" / "hivemind")
    ap.add_argument("--model", type=Path, default=None, help="default: engine/models/<it13 or wide>.onnx")
    args = ap.parse_args(argv)

    from hivemind.cli import selfplay

    if args.out.exists() and any(args.out.iterdir()):
        sys.exit(f"{args.out} is not empty: every bench gets a fresh folder")
    args.out.mkdir(parents=True, exist_ok=True)
    source = args.model or W2 / "engine" / "models" / MODELS[args.width]
    if not source.is_file():
        sys.exit(f"model {source} missing (narrow: hivemind fetch-network; wide: python -m hivemind.wave2 widen)")
    model = args.out / f"bench-{args.width}.onnx"
    shutil.copyfile(source, model)
    sp1 = selfplay.one_worker_copy(model)
    selfplay.refresh_copy(model, sp1)
    flags = config_extra(args.config)
    extra = engine_args(flags)
    command = selfplay.engine_command(args.engine, sp1, args.games, args.seed, (args.out / "data").resolve(),
                                      (args.out / "games").resolve(), args.parallel_games, extra)
    print(" ".join(command), flush=True)
    finished: list[float] = []
    start = time.monotonic()
    with open(args.out / "selfplay.log", "w") as log:
        proc = subprocess.Popen(command, cwd=W2 / "engine", stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, errors="replace")
        for line in proc.stdout:
            log.write(line)
            if line.startswith("selfplay game "):
                finished.append(time.monotonic())
        code = proc.wait()
    end = time.monotonic()
    commit = subprocess.run(["git", "-C", str(W2), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    report = {**rates([t - start for t in finished], 0.0, end - start, args.parallel_games),
              "games_asked": args.games, "parallel_games": args.parallel_games, "exit": code,
              "config": args.config, "wave2_extra": flags, "width": args.width, "engine": str(args.engine),
              "engine_commit": commit, "model": str(source), "engine_args": extra}
    (args.out / "bench.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report))
    return 0 if code == 0 and report["games"] == args.games else 1


if __name__ == "__main__":
    sys.exit(main())
