"""Parse arena long-test games into one record per game (pickled) for the Elo-hunt analysis."""
import glob, json, os, pickle, sys

TEAM = {"A_white": "T1", "B_black": "T1", "A_black": "T2", "B_white": "T2"}
BOARD = {"A_white": "A", "A_black": "A", "B_white": "B", "B_black": "B"}


def parse_game(path, res):
    ct = res["contender_team"]
    g = dict(res)
    g["us"] = ct
    g["score"] = 1.0 if res["result"] == ct else (0.5 if res["result"] == "draw" else 0.0)
    g["part"] = path.split("/")[-2]
    thinks, moves, stalls, open_stall = [], [], [], {}
    last_clocks = None
    for line in open(path):
        d = json.loads(line)
        t = d["type"]
        if t == "thinking" and d["phase"] == "end":
            s = d["seat"]
            thinks.append((d["ts"], s, TEAM[s] == ct, d["think_ms"], d["nodes"], d["eval"]))
        elif t == "move":
            s = d["seat"]
            moves.append((d["ts"], s, TEAM[s] == ct, d["clocks"], d["isDrop"], d["capture"]))
            last_clocks = d["clocks"]
        elif t == "stall":
            s = d["seat"]
            if d["active"]:
                open_stall[s] = (d["ts"], d["reason"])
            elif s in open_stall:
                t0, r = open_stall.pop(s)
                stalls.append((s, TEAM[s] == ct, r, t0, d["ts"] - t0))
        elif t == "game_end":
            g["end_ts"] = d["ts"]
    for s, (t0, r) in open_stall.items():
        stalls.append((s, TEAM[s] == ct, r, t0, g.get("end_ts", t0) - t0))
    g["thinks"], g["moves"], g["stalls"], g["last_clocks"] = thinks, moves, stalls, last_clocks
    return g


def main(root, out):
    games = []
    for d in sorted(glob.glob(os.path.join(root, "l_*"))):
        row = os.path.basename(d)
        for part in sorted(glob.glob(os.path.join(d, "part*"))):
            msn = json.load(open(os.path.join(part, "ms_per_node.json")))
            ms = {v["kind"] if v["kind"] == "upstream" else "wave2": v["ms_per_node"] for v in msn.values()}
            for line in open(os.path.join(part, "results.jsonl")):
                res = json.loads(line)
                g = parse_game(os.path.join(part, res["file"]), res)
                g["row"] = row
                g["ms_us"] = ms.get("wave2")
                g["ms_them"] = ms.get("upstream", ms.get("wave2"))
                games.append(g)
    pickle.dump(games, open(out, "wb"))
    print(len(games), "games")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
