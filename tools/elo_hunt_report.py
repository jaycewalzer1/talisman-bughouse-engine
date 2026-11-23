"""Elo hunt (docs/elo_hunt_analysis.md): results, time, sits, openings per row from the pickle of tools/elo_hunt_parse.py.
Usage: python tools/elo_hunt_report.py games.pkl"""
import collections, math, pickle, statistics as st, sys

games = pickle.load(open(sys.argv[1], "rb"))
ROWS = ["l_hybrid_120", "l_nodur_120", "l_same_120"]


def elo(s):
    s = min(max(s, 1e-3), 1 - 1e-3)
    return -400 * math.log10(1 / s - 1)


def summ(gs):
    n = len(gs)
    if not n:
        return "n=0"
    s = sum(g["score"] for g in gs) / n
    se = (st.pstdev([g["score"] for g in gs]) / math.sqrt(n)) if n > 1 else 0
    return f"n={n:5d} score={s:.3f} +-{se:.3f} elo={elo(s):+5.0f}"


def by(gs, key, title, minn=1, sort=None):
    print(f"\n-- {title}")
    groups = collections.defaultdict(list)
    for g in gs:
        groups[key(g)].append(g)
    for k in sorted(groups, key=sort or (lambda k: str(k))):
        if len(groups[k]) >= minn:
            print(f"  {str(k):40s} {summ(groups[k])}")
    return groups


for row in ROWS:
    gs = [g for g in games if g["row"] == row]
    print(f"\n================ {row}: {summ(gs)}")
    by(gs, lambda g: (g["part"], round(g["ms_them"] / g["ms_us"], 3)), "by part (ms_them/ms_us: >1 means we are charged less per node)")
    by(gs, lambda g: (g["reason"], g["flag"]), "by termination")
    by(gs, lambda g: "us=T1 (A white, B black)" if g["us"] == "T1" else "us=T2 (A black, B white)", "by our colours")
    print("\n-- decisive games: which board ended it, whose seat was mated")
    c = collections.Counter()
    for g in gs:
        if g["result"] == "draw":
            continue
        we_lost = g["score"] == 0
        c[(g["board"], "we lost" if we_lost else "we won", g["loser_seat"])] += 1
    for k, v in sorted(c.items()):
        print("  ", k, v)
    by(gs, lambda g: min(g["plies"] // 40 * 40, 240), "by game length (plies, bucket start)", sort=lambda k: k)
    # pairs
    pairs = collections.defaultdict(list)
    for g in gs:
        pairs[(g["part"], g["pair_index"])].append(g["score"])
    pc = collections.Counter(sum(v) for v in pairs.values() if len(v) == 2)
    print("\n-- colour-swapped pair totals (0..2):", dict(sorted(pc.items())))
    pt = [sum(v) / 2 for v in pairs.values() if len(v) == 2]
    print(f"   pair-based score {st.mean(pt):.3f} +- {st.pstdev(pt)/math.sqrt(len(pt)):.3f} over {len(pt)} pairs")
    # time
    print("\n-- per-decision search (thinking end events)")
    for us in (True, False):
        th = [t for g in gs for t in g["thinks"] if t[2] == us]
        tm = [t[3] for t in th]; nd = [t[4] for t in th]
        print(f"  {'us  ' if us else 'them'} decisions={len(th):6d} think_ms mean={st.mean(tm):6.0f} median={st.median(tm):6.0f}"
              f" nodes mean={st.mean(nd):7.0f} median={st.median(nd):7.0f} ms/node={sum(tm)/sum(nd):.4f}")
    print("\n-- decisions per game and total think per game")
    for us in (True, False):
        dec = [sum(1 for t in g["thinks"] if t[2] == us) for g in gs]
        tot = [sum(t[3] for t in g["thinks"] if t[2] == us) for g in gs]
        mv = [sum(1 for m in g["moves"] if m[2] == us) for g in gs]
        print(f"  {'us  ' if us else 'them'} decisions/game={st.mean(dec):.1f} moves/game={st.mean(mv):.1f} think_s/game={st.mean(tot)/1000:.1f}")
    # remaining clock trajectory by our move number
    print("\n-- mean remaining clock (s) per seat-group, sampled at the k-th move of the game (all seats' moves)")
    T = {"T1": ("A_white", "B_black"), "T2": ("A_black", "B_white")}
    for k in (10, 30, 60, 100, 150, 200):
        u, t, n = [], [], 0
        for g in gs:
            if len(g["moves"]) > k:
                cl = g["moves"][k][3]
                us, them = T[g["us"]], T["T2" if g["us"] == "T1" else "T1"]
                u.append(sum(cl[s] for s in us) / 2); t.append(sum(cl[s] for s in them) / 2)
        if u:
            print(f"  move {k:3d}: n={len(u):4d} us={st.mean(u)/1000:6.1f} them={st.mean(t)/1000:6.1f}")
    print("\n-- clock at game end (s, mean of team's two seats): by outcome")
    for lab, f in (("we won", lambda g: g["score"] == 1), ("we lost", lambda g: g["score"] == 0)):
        sel = [g for g in gs if f(g)]
        u = [sum(g["last_clocks"][s] for s in T[g["us"]]) / 2000 for g in sel]
        t = [sum(g["last_clocks"][s] for s in T["T2" if g["us"] == "T1" else "T1"]) / 2000 for g in sel]
        mins = [min(g["last_clocks"][s] for s in T[g["us"]]) / 1000 for g in sel]
        print(f"  {lab}: n={len(sel)} us={st.mean(u):.1f} them={st.mean(t):.1f} our min seat={st.mean(mins):.1f}"
              f" games where our min seat < 10 s: {sum(m < 10 for m in mins)}")
    # sits
    print("\n-- sits (stall active->inactive): per game, count and total ms")
    for us in (True, False):
        by_r = collections.defaultdict(lambda: [0, 0])
        for g in gs:
            for s in g["stalls"]:
                if s[1] == us:
                    by_r[s[2]][0] += 1; by_r[s[2]][1] += s[4]
        n = len(gs)
        print(f"  {'us  ' if us else 'them'} " + "  ".join(f"{r}: {c/n:.2f}/game {ms/max(c,1)/1000:.2f}s each" for r, (c, ms) in by_r.items()))
    print("\n-- score by number of our sits in the game (minus theirs)")
    by(gs, lambda g: max(-6, min(6, sum(1 for s in g["stalls"] if s[1]) - sum(1 for s in g["stalls"] if not s[1]))), "  diff sits us-them", sort=lambda k: k, minn=20)
    print("\n-- hybrid result field totals")
    h = collections.Counter()
    for g in gs:
        for tm, v in (g.get("hybrid") or {}).items():
            for kk, vv in v.items():
                h[("us" if tm == g["us"] else "them", kk)] += vv
    print("  ", dict(h))

    if row == "l_same_120":
        continue
    # openings
    op = by(gs, lambda g: g["opening"], "by opening (min 30 games)", minn=30, sort=None)
    sc = sorted(((sum(x["score"] for x in v) / len(v), len(v), k) for k, v in op.items()), reverse=True)
    print("  openings:", len(op), "  sd of opening score:", round(st.pstdev([s for s, n, k in sc if n >= 10]), 3))
