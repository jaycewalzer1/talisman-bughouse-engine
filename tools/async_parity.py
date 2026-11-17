"""Parity check: C++ async_selfplay (third_party/hivemind/engine/src/tools/async_selfplay.cc) vs the Python arena.

Runs the same openings, clocks, seeds and deterministic stub player through both:
- C++: `async_selfplay stub` (clock build), one game per tab-separated input line;
- Python: arena.sim.Simulator on arena.state.BughouseState with StubPlayer below (same algorithm as
  async_selfplay::StubPlayer: SplitMix64 seeded seed*4+seat, r % 100 < wait_pct -> wait, else
  sorted(legal)[next() % n] with a fixed think time per seat),
and diffs every emitted event (moves, SAN, FENs, timestamps, clocks, stalls, results) plus a per-decision
trace of the legal-move set (count and FNV-1a hash of the sorted UCI list).

    .venv/bin/python tools/async_parity.py --games 200
    .venv/bin/python tools/async_parity.py --games 20 --seed 7 --show 3

Exit status 0 only when every game matches exactly.
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from arena.interfaces import SEATS, Decision, Observation  # noqa: E402
from arena.openings import OPENING_PAIRS  # noqa: E402
from arena.sim import Simulator  # noqa: E402
from arena.state import BughouseState  # noqa: E402

DEFAULT_BINARY = ROOT / "third_party/hivemind/engine/build-clock/async_selfplay"
M64 = (1 << 64) - 1
START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR[] w KQkq - 0 1"


class SplitMix64:
    def __init__(self, seed: int):
        self.state = seed & M64

    def next(self) -> int:
        self.state = (self.state + 0x9E3779B97F4A7C15) & M64
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & M64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & M64
        return z ^ (z >> 31)


def fnv1a64(s: str) -> int:
    h = 0xCBF29CE484222325
    for c in s.encode():
        h = ((h ^ c) * 0x100000001B3) & M64
    return h


class StubPlayer:
    name = "stub"
    kind = "stub"

    def __init__(self, seat: str, seed: int, think_ms: int, wait_ms: int, wait_pct: int, trace: list):
        self.seat = seat
        self.rng = SplitMix64(seed * 4 + SEATS.index(seat))
        self.think_ms, self.wait_ms, self.wait_pct = think_ms, wait_ms, wait_pct
        self.trace = trace

    def decide(self, obs: Observation) -> Decision:
        moves = sorted(obs.legal_moves)
        self.trace.append({"ts": obs.now_ms, "seat": self.seat, "n": len(moves),
                           "h": format(fnv1a64(" ".join(moves)), "x")})
        r = self.rng.next()
        if not moves or r % 100 < self.wait_pct:
            return Decision(kind="wait", wait_ms=self.wait_ms,
                            stall_reason="no_moves" if not moves else "stub_wait")
        return Decision(kind="move", uci=moves[self.rng.next() % len(moves)], think_ms=self.think_ms)

    def close(self) -> None:
        pass


def make_configs(n: int, seed: int) -> list[dict]:
    """Varied games: opening pairs plus the start position, short clocks (flags), increments, stub waits
    (stalls and wakes), short ply caps, think times that keep the boards out of step."""
    rng = random.Random(seed)
    fens = [(p.fen_a, p.fen_b) for p in OPENING_PAIRS] + [(START_FEN, START_FEN)]
    out = []
    for i in range(n):
        fa, fb = fens[i % len(fens)]
        out.append({
            "game_id": f"parity-{seed}-{i}",
            "fenA": fa, "fenB": fb,
            "base_ms": rng.choice([2000, 5000, 15000, 60000, 180000]),
            "inc_ms": rng.choice([0, 0, 0, 50, 200]),
            "max_plies": rng.choice([60, 200, 400, 400, 600]),
            "seed": rng.randrange(1 << 40),
            "think": [rng.choice([1, 7, 30, 50, 120, 333, 500]) for _ in SEATS],
            "wait": [rng.choice([1, 40, 150, 200, 450]) for _ in SEATS],
            "wait_pct": rng.choice([0, 0, 3, 10, 25]),
        })
    return out


def to_tsv(c: dict) -> str:
    return "\t".join([c["game_id"], c["fenA"], c["fenB"], str(c["base_ms"]), str(c["inc_ms"]),
                      str(c["max_plies"]), str(c["seed"]), ",".join(map(str, c["think"])),
                      ",".join(map(str, c["wait"])), str(c["wait_pct"])])


def run_python(c: dict) -> dict:
    trace: list = []
    state = BughouseState(c["base_ms"], c["inc_ms"], fens={"A": c["fenA"], "B": c["fenB"]})
    players = {s: StubPlayer(s, c["seed"], c["think"][k], c["wait"][k], c["wait_pct"], trace)
               for k, s in enumerate(SEATS)}
    sim = Simulator(state, players, game_id=c["game_id"], setup="stub", max_plies=c["max_plies"])
    sim.run()
    # round-trip through JSON so both sides compare as plain JSON values
    return json.loads(json.dumps({"game_id": c["game_id"], "events": sim.events, "trace": trace}))


def run_cpp(binary: Path, configs: list[dict]) -> list[dict]:
    proc = subprocess.run([str(binary), "stub"], input="".join(to_tsv(c) + "\n" for c in configs),
                          capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise SystemExit(f"async_selfplay failed ({proc.returncode}): {proc.stderr.strip()}")
    return [json.loads(line) for line in proc.stdout.splitlines() if line.strip()]


def diff_game(py: dict, cc: dict) -> list[str]:
    diffs = []
    for key in ("events", "trace"):
        a, b = py[key], cc[key]
        for i, (x, y) in enumerate(zip(a, b)):
            if x != y:
                fields = sorted(k for k in set(x) | set(y) if x.get(k, "<missing>") != y.get(k, "<missing>"))
                detail = ", ".join(f"{k}: py={x.get(k, '<missing>')!r} cpp={y.get(k, '<missing>')!r}"
                                   for k in fields)
                diffs.append(f"{key}[{i}] ({x.get('type', x.get('seat'))}) {detail}")
                break
        if len(a) != len(b):
            diffs.append(f"{key}: py has {len(a)}, cpp has {len(b)}")
    return diffs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    ap.add_argument("--show", type=int, default=5, help="print the first difference of this many games")
    args = ap.parse_args()

    configs = make_configs(args.games, args.seed)
    t0 = time.perf_counter()
    cpp = run_cpp(args.binary, configs)
    t_cpp = time.perf_counter() - t0
    if len(cpp) != len(configs):
        print(f"C++ returned {len(cpp)} games for {len(configs)} configs")
        return 1
    t0 = time.perf_counter()
    py = [run_python(c) for c in configs]
    t_py = time.perf_counter() - t0

    bad = 0
    events = plies = 0
    reasons: dict[str, int] = {}
    for c, p, q in zip(configs, py, cpp):
        events += len(p["events"])
        plies += sum(1 for e in p["events"] if e["type"] == "move")
        end = p["events"][-1]
        reasons[end.get("reason", "?")] = reasons.get(end.get("reason", "?"), 0) + 1
        d = diff_game(p, q)
        if d:
            bad += 1
            if bad <= args.show:
                print(f"{c['game_id']}: " + " | ".join(d))
    print(f"games {len(configs)}  differing {bad}  events {events}  moves {plies}  "
          f"endings {dict(sorted(reasons.items()))}  cpp {t_cpp:.1f}s  python {t_py:.1f}s")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
