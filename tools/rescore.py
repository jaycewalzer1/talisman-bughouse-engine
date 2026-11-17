"""Rebuild an arena row's results from its per-game event files.

    python tools/rescore.py runs/beat/long_clock110_v300            # print the row's results from game_*.jsonl
    python tools/rescore.py runs/beat/long_clock110_v300 --write    # also write summary.json (and results.jsonl)

A row (arena.pool run) leaves one ``game_NNNN.jsonl`` per finished game. This reads them back, attaches each
game's schedule entry (``specs.jsonl``, written by arena.pool at the start of a run; for rows older than that, the
entries of an existing ``summary.json``) and recomputes the per-opponent results exactly as arena.pool does at the
end of a row: score with the cluster-bootstrap 95% range, nodes per game each side, end reasons, pentanomial pair
counts and Elo / normalised Elo. Use it on a row that was cut off before it wrote ``summary.json``, or to read a
row while it is still running. Games that are cut short or crashed are left out and counted.

``--write`` refuses to overwrite a summary.json whose game count is larger than the rescore's (a partial rescore
never replaces a full result) unless ``--force``.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO not in sys.path:
    sys.path.insert(0, REPO)


def _specs_of(row: str) -> dict[str, dict]:
    """file name -> schedule entry (enough of one for summary_from_game_file and the per-opponent split)."""
    path = os.path.join(row, "specs.jsonl")
    if os.path.exists(path):
        from arena.pool import game_file

        with open(path) as f:
            specs = [json.loads(l) for l in f if l.strip()]
        return {game_file(s): s for s in specs}
    path = os.path.join(row, "summary.json")
    if os.path.exists(path):
        with open(path) as f:
            games = json.load(f).get("games", [])
        out = {}
        for g in games:
            if not g.get("file"):
                continue
            spec = dict(g)
            if "pair_index" not in spec:  # pool schedules each pair as two consecutive games
                digits = "".join(ch for ch in g["file"] if ch.isdigit())
                spec["pair_index"] = (int(digits) - 1) // 2
            out[g["file"]] = spec
        return out
    raise FileNotFoundError(f"{row} has neither specs.jsonl nor summary.json: cannot tell which team is which")


def rescore_games(row: str) -> list[dict]:
    """Summaries of every cleanly finished game in the row, sorted by game_id."""
    from arena.pool import summary_from_game_file

    specs = _specs_of(row)
    out = []
    for path in sorted(glob.glob(os.path.join(row, "game_*.jsonl"))):
        spec = specs.get(os.path.basename(path))
        if spec is None:
            continue
        s = summary_from_game_file(spec, path)
        if s is not None:
            out.append(s)
    out.sort(key=lambda s: s["game_id"])
    return out


def rescore(row: str, n_boot: int = 2000, seed: int = 0) -> dict:
    """{"results", "games", "errors", "unfinished", "scheduled"} in summary.json's shape."""
    from arena.pool import results_by_opponent

    specs = _specs_of(row)
    games = rescore_games(row)
    files = glob.glob(os.path.join(row, "game_*.jsonl"))
    return {"results": results_by_opponent(games, n_boot=n_boot, seed=seed), "errors": 0, "games": games,
            "unfinished": len(files) - len(games), "scheduled": len(specs),
            "unplayed": len(specs) - len(games)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("rows", nargs="+")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0, help="bootstrap seed (arena.pool uses the row's --seed)")
    ap.add_argument("--write", action="store_true", help="write summary.json and results.jsonl")
    ap.add_argument("--force", action="store_true", help="with --write: overwrite a summary.json with more games")
    args = ap.parse_args(argv)
    from arena.pool import format_results

    status = 0
    for row in args.rows:
        rep = rescore(row, n_boot=args.boot, seed=args.seed)
        print(f"== {row}: {len(rep['games'])} finished games of {rep['scheduled']} scheduled, "
              f"{rep['unfinished']} cut short or crashed")
        print(format_results(rep["results"]))
        if not args.write:
            continue
        path = os.path.join(row, "summary.json")
        if os.path.exists(path) and not args.force:
            with open(path) as f:
                old = json.load(f)
            if len([g for g in old.get("games", []) if "error" not in g]) > len(rep["games"]):
                print(f"not writing {path}: it has more games than the rescore (use --force)", file=sys.stderr)
                status = 1
                continue
            rep = {**{k: v for k, v in old.items() if k not in rep}, **rep}
        with open(path, "w") as f:
            json.dump(rep, f, indent=2)
        with open(os.path.join(row, "results.jsonl"), "w") as f:
            for g in rep["games"]:
                f.write(json.dumps(g, separators=(",", ":")) + "\n")
        print(f"wrote {path}")
    return status


if __name__ == "__main__":
    sys.exit(main())
