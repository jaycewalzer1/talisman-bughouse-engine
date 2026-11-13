"""One row per training run, to compare runs side by side from their train_log.jsonl.

    python tools/compare_runs.py runs/waves/*

"early" and "late" are the first and last third of the iterations done so far. target is the score against the
main opponent only (--target in training), the one number whose opponent never changes.
"""

import json
import os
import sys


def mean(rows: list[dict], key: str) -> float | None:
    values = [r[key] for r in rows if r.get(key) is not None]
    return sum(values) / len(values) if values else None


def target(rows: list[dict]) -> tuple[float | None, int]:
    games = sum(r.get("target_games", 0) for r in rows)
    points = sum(r["target_score"] * r["target_games"] for r in rows if r.get("target_score") is not None)
    return (points / games if games else None), games


def cell(value, width: int, digits: int = 3) -> str:
    return f"{'-':>{width}}" if value is None else f"{value:>{width}.{digits}f}"


def main() -> None:
    print(f"{'run':<10} {'iters':>5} {'target early':>12} {'target late':>11} {'games':>6} {'± late':>7} "
          f"{'wait early':>10} {'wait late':>9} {'think':>6} {'mates seen':>10} {'waited':>6} "
          f"{'serve offers':>12} {'served':>6}")
    for run in sys.argv[1:]:
        path = os.path.join(run, "train_log.jsonl")
        if not os.path.exists(path):
            print(f"{os.path.basename(run):<10} no log yet")
            continue
        with open(path) as f:
            rows = [json.loads(line) for line in f]
        third = max(1, len(rows) // 3)
        early, late = rows[:third], rows[-third:]
        late_score, late_games = target(late)
        # half-width of a 95% range for a win share from that many games
        spread = 1.96 * (0.25 / late_games) ** 0.5 if late_games else None
        print(f"{os.path.basename(run):<10} {len(rows):>5} {cell(target(early)[0], 12)} {cell(late_score, 11)} "
              f"{late_games:>6} {cell(spread, 7)} {cell(mean(early, 'wait_rate'), 10)} "
              f"{cell(mean(late, 'wait_rate'), 9)} {cell(mean(late, 'think_mean'), 6, 2)} "
              f"{sum(r.get('mate_seen', 0) for r in rows):>10} {sum(r.get('mate_waited', 0) for r in rows):>6} "
              f"{sum(r.get('serve_offers', 0) for r in rows):>12} {sum(r.get('served', 0) for r in rows):>6}")


if __name__ == "__main__":
    main()
