"""Elo hunt (docs/elo_hunt_analysis.md): eval trajectories, clock lead, node ratios per row.
Usage: python tools/elo_hunt_report_eval.py games.pkl"""
import collections, math, pickle, statistics as st, sys

games = pickle.load(open(sys.argv[1], "rb"))
ROWS = ["l_hybrid_120", "l_nodur_120", "l_same_120"]


def elo(s):
    s = min(max(s, 1e-3), 1 - 1e-3)
    return -400 * math.log10(1 / s - 1)


def ours(t):
    """eval of a thinking event in our team's perspective (evals are team-perspective of the searching team)."""
    return t[5] if t[2] else -t[5]


for row in ROWS:
    gs = [g for g in games if g["row"] == row]
    print(f"\n================ {row}")
    # 1. eval drift by game time (consensus of both sides' searches in a window)
    print("-- mean eval in OUR perspective by game time window (s): from our searches / their searches; and n games alive")
    for t0 in (0, 5, 10, 20, 30, 40, 50, 60, 75, 90, 110):
        w = (t0 * 1000, t0 * 1000 + 5000)
        u, th = [], []
        for g in gs:
            a = [ours(t) for t in g["thinks"] if t[5] is not None and w[0] <= t[0] < w[1] and t[2]]
            b = [ours(t) for t in g["thinks"] if t[5] is not None and w[0] <= t[0] < w[1] and not t[2]]
            if a: u.append(st.mean(a))
            if b: th.append(st.mean(b))
        if u and th:
            print(f"  t={t0:3d}-{t0+5:3d}s games={len(u):4d} ours={st.mean(u):+.3f} theirs={st.mean(th):+.3f} consensus={(st.mean(u)+st.mean(th))/2:+.3f}")
    # 2. score conditional on consensus eval at 20 s and 40 s
    for T in (10, 20, 40):
        print(f"-- score by consensus eval bucket at t={T}s (mean of each side's last search eval before T)")
        buckets = collections.defaultdict(list)
        for g in gs:
            a = [ours(t) for t in g["thinks"] if t[5] is not None and t[0] < T * 1000 and t[2]]
            b = [ours(t) for t in g["thinks"] if t[5] is not None and t[0] < T * 1000 and not t[2]]
            if not a or not b or g["end_ts"] < T * 1000:
                continue
            e = (a[-1] + b[-1]) / 2
            k = max(-3, min(3, round(e / 0.2)))
            buckets[k].append(g["score"])
        for k in sorted(buckets):
            v = buckets[k]
            print(f"   eval~{k*0.2:+.1f}: n={len(v):4d} score={st.mean(v):.3f}")
    # 3. who "drops" value: per side, mean change of own team-perspective eval between consecutive searches of that
    #    side (all boards) -> negative = side's position deteriorating in its own eyes
    print("-- per side: mean change of the side's own eval between its consecutive searches, by game phase (search index)")
    for us in (True, False):
        ph = collections.defaultdict(list)
        for g in gs:
            ev = [t[5] for t in g["thinks"] if t[2] == us and t[5] is not None]
            for i in range(1, len(ev)):
                ph[min(i // 20, 4)].append(ev[i] - ev[i - 1])
        print(f"   {'us  ' if us else 'them'} " + " ".join(f"[{k*20}-{k*20+19}] {st.mean(v):+.4f}" for k, v in sorted(ph.items())))
    # 4. clock lead
    print("-- score by our share of turns spent ahead on the team clock lead (clock_teams)")
    bk = collections.defaultdict(list)
    for g in gs:
        ct = g["clock_teams"][g["us"]]
        tot = ct["turns_ahead"] + ct["turns_behind"]
        if tot:
            bk[min(4, int(ct["turns_ahead"] / tot * 5))].append(g["score"])
    for k in sorted(bk):
        print(f"   ahead {k*20}-{k*20+20}%: n={len(bk[k]):4d} score={st.mean(bk[k]):.3f}")
    ah = [g["clock_teams"][g["us"]]["turns_ahead"] / max(1, g["clock_teams"][g["us"]]["turns_ahead"] + g["clock_teams"][g["us"]]["turns_behind"]) for g in gs]
    print(f"   mean share ahead (us) = {st.mean(ah):.3f}")
    # sits ahead/behind
    for side in ("us", "them"):
        f = (lambda g: g["us"]) if side == "us" else (lambda g: "T2" if g["us"] == "T1" else "T1")
        c = [g["clock_teams"][f(g)] for g in gs]
        print(f"   {side}: sits_ahead/game={st.mean(x['sits_ahead'] for x in c):.2f} sits_behind/game={st.mean(x['sits_behind'] for x in c):.2f}"
              f" sit_ms_ahead/game={st.mean(x['sit_ms_ahead'] for x in c):.0f}")
    # 5. time of the decisive moment: game end ts by outcome
    w = [g["end_ts"] / 1000 for g in gs if g["score"] == 1]
    l = [g["end_ts"] / 1000 for g in gs if g["score"] == 0]
    print(f"-- game duration (s): won mean {st.mean(w):.1f} median {st.median(w):.1f}; lost mean {st.mean(l):.1f} median {st.median(l):.1f}")
    # 6. drops / pieces: captures & drops per side
    for us in (True, False):
        dr = st.mean(sum(1 for m in g["moves"] if m[2] == us and m[4]) for g in gs)
        cp = st.mean(sum(1 for m in g["moves"] if m[2] == us and m[5]) for g in gs)
        print(f"   {'us  ' if us else 'them'} drops/game={dr:.1f} captures/game={cp:.1f}")
    # 7. score by search nodes ratio within game (our mean nodes / their mean nodes)
    print("-- score by within-game node ratio (our mean nodes per search / theirs)")
    bk = collections.defaultdict(list)
    for g in gs:
        a = [t[4] for t in g["thinks"] if t[2]]; b = [t[4] for t in g["thinks"] if not t[2]]
        if a and b:
            r = st.mean(a) / st.mean(b)
            bk[round(r, 1)].append(g["score"])
    for k in sorted(bk):
        if len(bk[k]) >= 15:
            print(f"   ratio~{k:.1f}: n={len(bk[k]):4d} score={st.mean(bk[k]):.3f}")
    # 8. mean think ms of search per side by sit/no: early (first 10 searches) vs later
    for us in (True, False):
        e = [t[3] for g in gs for t in g["thinks"][:20] if t[2] == us]
        print(f"   {'us  ' if us else 'them'} early think_ms {st.mean(e):.0f}")
