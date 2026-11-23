"""elohunt_speed GPU bench: wall ms per searched node (go nodes N, go->bestmove as the arena measures it) for
binary x TensorRT engine file x flags, interleaved, plus raw inference. Prints one JSON line per config run."""
import subprocess, sys, time, json, os, threading, queue, statistics, argparse
ap = argparse.ArgumentParser()
ap.add_argument("--binary"); ap.add_argument("--model"); ap.add_argument("--flags", default="stock")
ap.add_argument("--nodes", type=int, default=800); ap.add_argument("--repeats", type=int, default=10)
ap.add_argument("--positions", default=None); ap.add_argument("--pos-repeats", type=int, default=2)
ap.add_argument("--label", default="")
a = ap.parse_args()
p = subprocess.Popen([a.binary, "--model", a.model], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                     stderr=subprocess.DEVNULL, text=True, bufsize=1, cwd=os.path.dirname(os.path.dirname(a.binary)))
q = queue.Queue()
threading.Thread(target=lambda: [q.put(l.rstrip("\n")) for l in p.stdout], daemon=True).start()
def send(s): p.stdin.write(s + "\n"); p.stdin.flush()
def wait(pred, col=None):
    while True:
        l = q.get(timeout=900)
        if col is not None: col.append(l)
        if pred(l): return l
t_start = time.perf_counter()
send("uci"); wait(lambda l: l == "uciok")
opts = {"Ponder": "false", "BackgroundSearch": "false", "TimeModel": "clock"}
if a.flags in ("nodur", "hybrid"):
    opts.update({"DurativeTree": "false", "SitGate": "true", "FlagLoss": "true", "InstantTransfer": "false",
                 "MateRaceByLanding": "false", "PassCapK": "8", "SeatClocksMs": "60000 60000 60000 60000",
                 "SeatPaceMs": "100 100 100 100", "SeatElapsedMs": "0 0 0 0"})
    if a.flags == "hybrid": opts["HybridMode"] = "gate"
for k, v in opts.items(): send(f"setoption name {k} value {v}")
send("isready"); wait(lambda l: l == "readyok")
load_s = time.perf_counter() - t_start
start = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR[] w KQkq - 0 1"
def search(fen, nodes, team="white", lead=0, own=100, opp=100, root=0):
    for k, v in (("Team", team), ("TimeLeadMs", lead), ("TeamMoveTimeMs", own), ("OpponentMoveTimeMs", opp),
                 ("RootMoveTimeMs", root)):
        send(f"setoption name {k} value {v}")
    send("ucinewgame"); send(f"position fen {fen}"); send("isready"); wait(lambda l: l == "readyok")
    lines = []
    t0 = time.perf_counter(); send(f"go nodes {nodes}")
    best = wait(lambda l: l.startswith("bestmove"), lines)
    wall = (time.perf_counter() - t0) * 1000
    send("stop"); send("isready"); wait(lambda l: l == "readyok")
    info = [l for l in lines if l.startswith("info depth")]
    toks = info[-1].split() if info else []
    n = int(toks[toks.index("nodes") + 1]) if "nodes" in toks else 0
    return wall, n
sp = f"{start} | {start}"
search(sp, 64); search(sp, a.nodes); search(sp, a.nodes)  # warm-up
runs = [search(sp, a.nodes) for _ in range(a.repeats)]
pos_runs = []
if a.positions:
    for line in open(a.positions):
        if not line.strip() or line.startswith("#"): continue
        team, lead, own, opp, root, fen = line.rstrip("\n").split("\t")
        fa, fb = fen.split("|")
        for _ in range(a.pos_repeats):
            pos_runs.append(search(f"{fa.strip()} | {fb.strip()}", a.nodes, team, lead, own, opp, root))
send("quit")
def summ(rs):
    if not rs: return None
    return dict(n=len(rs), ms_per_node=sum(w for w, _ in rs) / max(1, sum(n for _, n in rs)),
                wall_med=statistics.median(w for w, _ in rs), nodes_mean=statistics.mean(n for _, n in rs))
print(json.dumps(dict(label=a.label, binary=a.binary, model=a.model, flags=a.flags, nodes=a.nodes, load_s=round(load_s, 2),
                      startpos=summ(runs), positions=summ(pos_runs), walls=[round(w, 1) for w, _ in runs],
                      host=os.uname().nodename, when=time.strftime("%H:%M:%S"))), flush=True)
