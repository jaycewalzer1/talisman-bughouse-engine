"""GSPRT over normalised Elo for a paired arena row, reading games as they land.

    python tools/sprt.py runs/beat/same_100_v100                      # one look at a row (finished or not)
    python tools/sprt.py runs/custom/main/it03/gate_180 --follow 120  # look again every 2 minutes until decided
    python tools/sprt.py ROW --elo0 0 --elo1 20 --alpha 0.05 --beta 0.05 --json

Each opening pair is played twice with the teams swapped (arena.pool), so the unit is a game pair: its contender
points (0, 0.5, 1, 1.5, 2) fall into one of five pentanomial buckets. The test is the generalised SPRT of
H0: nElo = elo0 against H1: nElo = elo1 (default [0, 20]) in normalised Elo, the per-pair-variance scale fishtest
uses (Van den Bergh): nt = (mu - 0.5) / (sqrt(2) * sigma_pair), mu and sigma_pair the mean and per-pair standard
deviation of the per-game score, nElo = nt * 800 / ln 10. The LLR is pairs x sum_i phat_i ln(p1_i / p0_i), with
p0, p1 the maximum-likelihood pentanomial distributions under each hypothesis (fishtest's MLE_t_value fixed
point), so it stays exact far from the bounds (the closed-form approximation goes to 0 on a one-sided sample). Stop for H1 when LLR >= ln((1 - beta) / alpha), for H0 when
LLR <= ln(beta / (1 - alpha)). Normalised Elo does not depend on the draw rate, so the same bounds mean the same
thing at 60 s and at 180 s.

Input is the row's ``results.jsonl`` (one game summary per line, written by arena.pool as each game finishes) or,
for rows run before that existed, the per-game ``game_*.jsonl`` files rescored with tools/rescore.py. Only pairs
with both games finished and neither in error count. Exit status: 0 = H1 accepted, 1 = H0 accepted, 2 = continue.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

NELO_PER_NT = 800.0 / math.log(10.0)
BUCKETS = (0.0, 0.5, 1.0, 1.5, 2.0)


def bounds(alpha: float = 0.05, beta: float = 0.05) -> tuple[float, float]:
    """(lower, upper) LLR stopping bounds."""
    return math.log(beta / (1.0 - alpha)), math.log((1.0 - beta) / alpha)


def pentanomial(summaries: list[dict]) -> list[int]:
    """Counts of finished game pairs by contender points 0, 0.5, 1, 1.5, 2.

    A summary needs ``contender_team`` and ``result``; the pair key is ``pair_index`` when present, else the game
    index from ``game_id``/``file`` (games 1+2, 3+4, ... are a pair, as arena.pool schedules them). Pairs with a
    missing game or an errored game are left out."""
    pairs: dict = {}
    for s in summaries:
        key = _pair_key(s)
        pairs.setdefault(key, []).append(s)
    counts = [0] * 5
    for games in pairs.values():
        if len(games) != 2 or any("error" in g for g in games):
            continue
        pts = sum(_points(g) for g in games)
        counts[BUCKETS.index(pts)] += 1
    return counts


def _points(s: dict) -> float:
    res, ct = s.get("result"), s["contender_team"]
    return 1.0 if res == ct else 0.5 if res == "draw" else 0.0


def _pair_key(s: dict):
    if s.get("pair_index") is not None:
        return ("p", s.get("opponent"), int(s["pair_index"]))
    for field in ("file", "game_id"):
        digits = "".join(ch for ch in str(s.get(field) or "") if ch.isdigit())
        if digits:
            return ("i", s.get("opponent"), (int(digits) - 1) // 2)
    raise ValueError(f"cannot tell which pair game {s.get('game_id')!r} belongs to")


def _moments(counts: list[int], eps: float = 1e-3) -> tuple[float, float, float]:
    """(pairs, mean per-game score, per-pair variance of the per-game score). eps per bucket keeps the variance
    positive on a tiny or one-sided sample (as fishtest regularises its counts)."""
    n = sum(counts)
    reg = [c + eps for c in counts]
    tot = sum(reg)
    xs = [b / 2.0 for b in BUCKETS]
    mu = sum(x * c for x, c in zip(xs, reg)) / tot
    var = sum(c * (x - mu) ** 2 for x, c in zip(xs, reg)) / tot
    return n, mu, var


def nelo(counts: list[int]) -> float:
    """Normalised Elo estimate from pentanomial counts (0 with no pairs)."""
    n, mu, var = _moments(counts)
    if n == 0:
        return 0.0
    return (mu - 0.5) / math.sqrt(2.0 * var) * NELO_PER_NT


def elo(counts: list[int]) -> float:
    """Logistic Elo from the mean score (for reading only; the test runs on nElo)."""
    n, mu, _ = _moments(counts)
    if n == 0:
        return 0.0
    mu = min(max(mu, 1e-6), 1 - 1e-6)
    return -400.0 * math.log10(1.0 / mu - 1.0)


def _secular(v: list[float], p: list[float]) -> float:
    """The x with sum p_i v_i / (1 + x v_i) = 0 and every 1 + x v_i > 0 (safeguarded Newton; the sum falls
    with x)."""
    lo = -1.0 / max(v) if max(v) > 0 else -1e12
    hi = -1.0 / min(v) if min(v) < 0 else 1e12
    span = hi - lo
    lo, hi = lo + 1e-12 * span, hi - 1e-12 * span
    x = min(max(0.0, lo), hi)
    for _ in range(100):
        f = df = 0.0
        for vi, pi in zip(v, p):
            d = 1.0 + x * vi
            f += pi * vi / d
            df -= pi * vi * vi / (d * d)
        if abs(f) < 1e-15:
            break
        if f > 0:
            lo = x
        else:
            hi = x
        step = x - f / df if df < 0 else (lo + hi) / 2
        x = step if lo < step < hi else (lo + hi) / 2
        if hi - lo < 1e-15 * max(1.0, abs(x)):
            break
    return x


def mle_t(phat: list[float], t: float, ref: float = 0.5) -> list[float]:
    """Maximum-likelihood pentanomial pdf under the constraint (mu - ref) / sigma = t (sigma: per-pair standard
    deviation of the per-game score); Van den Bergh's fixed point, as fishtest's MLE_t_value."""
    xs = [b / 2.0 for b in BUCKETS]
    p = [1.0 / len(xs)] * len(xs)
    for _ in range(100):
        mu = sum(x * q for x, q in zip(xs, p))
        sigma = math.sqrt(max(1e-300, sum(q * (x - mu) ** 2 for x, q in zip(xs, p))))
        v = [x - ref - t * sigma * (1.0 + ((mu - x) / sigma) ** 2) / 2.0 for x in xs]
        lam = _secular(v, phat)
        new = [ph / (1.0 + lam * vi) for ph, vi in zip(phat, v)]
        tot = sum(new)
        new = [q / tot for q in new]
        if max(abs(a - b) for a, b in zip(new, p)) < 1e-12:
            p = new
            break
        p = new
    return p


def llr(counts: list[int], elo0: float = 0.0, elo1: float = 20.0) -> float:
    """Generalised log-likelihood ratio of nElo = elo1 against nElo = elo0 for the pair counts: the multinomial
    likelihood at the constrained maximum under each hypothesis (exact, also far from the bounds)."""
    n = sum(counts)
    if n == 0:
        return 0.0
    reg = [c + 1e-3 for c in counts]
    phat = [c / sum(reg) for c in reg]
    t0, t1 = (e / NELO_PER_NT * math.sqrt(2.0) for e in (elo0, elo1))
    p0, p1 = mle_t(phat, t0), mle_t(phat, t1)
    return n * sum(ph * (math.log(a) - math.log(b)) for ph, a, b in zip(phat, p1, p0))


def status(counts: list[int], elo0: float = 0.0, elo1: float = 20.0, alpha: float = 0.05,
           beta: float = 0.05) -> dict:
    """One look: {"pairs", "games", "pentanomial", "score", "elo", "nelo", "llr", "lower", "upper", "decision"}
    with decision "H1", "H0" or "continue"."""
    lo, hi = bounds(alpha, beta)
    value = llr(counts, elo0, elo1)
    decision = "H1" if value >= hi else "H0" if value <= lo else "continue"
    n = sum(counts)
    score = sum(b * c for b, c in zip(BUCKETS, counts)) / (2 * n) if n else float("nan")
    return {"pairs": n, "games": 2 * n, "pentanomial": list(counts), "score": score, "elo": elo(counts),
            "nelo": nelo(counts), "llr": value, "lower": lo, "upper": hi, "elo0": elo0, "elo1": elo1,
            "alpha": alpha, "beta": beta, "decision": decision}


def load_row(row: str) -> list[dict]:
    """Game summaries of one arena row: results.jsonl if it exists, else a rescore of the game files."""
    path = os.path.join(row, "results.jsonl")
    if os.path.exists(path):
        out = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        out.append(json.loads(line))
                    except json.JSONDecodeError:  # a line being written right now
                        pass
        return out
    try:
        from tools.rescore import rescore_games
    except ImportError:  # run as a script from elsewhere
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from rescore import rescore_games  # noqa: E402

    return rescore_games(row)


def format_status(st: dict, name: str = "") -> str:
    return (f"{name + ': ' if name else ''}{st['games']} games in {st['pairs']} pairs  pentanomial {st['pentanomial']}"
            f"  score {st['score']:.3f}  Elo {st['elo']:+.1f}  nElo {st['nelo']:+.1f}  LLR {st['llr']:+.2f}"
            f" [{st['lower']:.2f}, {st['upper']:.2f}] for nElo [{st['elo0']:g}, {st['elo1']:g}]  -> {st['decision']}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("rows", nargs="+", help="arena row folders (each with results.jsonl or game_*.jsonl)")
    ap.add_argument("--elo0", type=float, default=0.0)
    ap.add_argument("--elo1", type=float, default=20.0)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--beta", type=float, default=0.05)
    ap.add_argument("--follow", type=float, default=0.0, metavar="SECONDS",
                    help="look again every SECONDS until every row is decided")
    ap.add_argument("--json", action="store_true", help="print one JSON object per row")
    args = ap.parse_args(argv)
    while True:
        decisions = []
        for row in args.rows:
            st = status(pentanomial(load_row(row)), args.elo0, args.elo1, args.alpha, args.beta)
            decisions.append(st["decision"])
            print(json.dumps({"row": row, **st}) if args.json else format_status(st, row), flush=True)
        if not args.follow or all(d != "continue" for d in decisions):
            break
        time.sleep(args.follow)
    if len(decisions) == 1:
        return {"H1": 0, "H0": 1}.get(decisions[0], 2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
