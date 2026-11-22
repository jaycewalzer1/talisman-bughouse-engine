#!/usr/bin/env python3
"""Network diagnostics for one RL iteration: how well the network predicts a fixed probe set, how its use of each
input group changes, and how far each layer moved since the generator before it.

    python cluster/probe_net.py build runs/full/hybrid1 [--positions 2000]      # once per run: runs/full/<RUN>/probe/
    python cluster/probe_net.py eval runs/full/hybrid1 3 [--device cuda]       # -> runs/full/<RUN>/metrics/it3.probe.json

Runs with the fork's python (torch + hivemind): third_party/wave2-hivemind/.venv/bin/python.

build freezes ~--positions records from the first iteration's label chunks (val first, train to fill), stratified into
categories (CATEGORIES, first match wins) so rare situations are not drowned by middlegames. The set is never rebuilt
unless --force: every iteration of a run must be scored on the same positions. The first iteration's val games are
never trained on (rl-loop passes them as --search-val only), so the heldout numbers use only val-sourced positions.

eval loads iteration N's arms/C/last.pt (what promotion installs; --checkpoint best for best.pt) and the network that
generated iteration N's games: the last promoted iteration before N, else the checkpoint iteration 1 trained from
(rl-it1/stages/train.done "init" sha256, searched in artifacts/). Value semantics: the twin-board value output is
P(win) - P(loss) of the WDL head (index order loss, draw, win), the head the rl-loop trains (--search-value-weight 0).

Input ablation fills a plane with 0 where 0 is a common in-distribution value (empty square, pocket or landing
plane) and with its probe-set mean where 0 would be a statement (turn, castling, sit right, clock lead and speed,
halfmove, repetition, wide clock planes); each group says which ("fill": zero, mean or mixed).
"""

import argparse
import datetime
import hashlib
import json
import os
import sys
import time

import numpy as np

SCHEMA = 1
PROBE_FILE, MANIFEST = "probe.npz", "manifest.json"
MASK64 = (1 << 64) - 1
# First match wins, so a mate race in the opening counts as a mate race.
CATEGORIES = ("near_end", "check_or_swing", "time_trouble", "crowded_pockets", "opening", "middlegame")
NEAR_END_PLIES = 12        # moves_left is remaining plies / 100
OPENING_PLY = 20
CROWDED_POCKET = 8         # pieces in the four pockets together
SWING_MATERIAL = 8         # |ours - theirs| on one board, pockets included (P1 N3 B3 R5 Q9)
LOW_CLOCK_MS = 10_000      # a v3 chunk's own seat clock on either board
LEAD_QUANTILE = 0.1        # v2 chunks have no seat clocks: the bottom decile of the team-lead plane instead
PIECE_VALUES = (1, 3, 3, 5, 9, 0)

# Planes within one board block (hivemind.domain.board2planes, constants.py, wide_planes.py).
OUR_PIECES, THEIR_PIECES, OUR_POCKET, THEIR_POCKET = range(0, 6), range(6, 12), range(12, 17), range(17, 22)
PROMOTED, EP, TURN, CONSTANT, CASTLING, SIT = (22, 23), 24, 25, 26, range(27, 31), 31
LAST_MOVE, HALFMOVE, REPETITION, LEAD, SPEED = (32, 33), 34, (35, 36), 37, 38
WIDE = 39  # first wide extra plane
WIDE_CLOCKS, OUR_PRESENCE, THEIR_PRESENCE, OUR_RECENT, THEIR_RECENT = (
    range(WIDE, WIDE + 6), range(WIDE + 6, WIDE + 11), range(WIDE + 11, WIDE + 16), range(WIDE + 16, WIDE + 21),
    range(WIDE + 21, WIDE + 26))
MEAN_FILLED = {TURN, *CASTLING, SIT, HALFMOVE, *REPETITION, LEAD, SPEED, *WIDE_CLOCKS}

PER_ROW = ("plane_bits", "plane_values", "value", "wdl", "moves_left", "ply", "outcome", "policy_count", "arrivals",
           "has_arrivals", "seat_clocks_ms", "elapsed_ms", "recent_landed")


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def rel(path, root):
    path, root = os.path.abspath(path), os.path.abspath(root)
    return os.path.relpath(path, root) if path.startswith(root + os.sep) else path


def write_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=1)
    os.replace(tmp, path)


def num(x, digits=6):
    x = float(x)
    return round(x, digits) if np.isfinite(x) else None


# ---------------------------------------------------------------- chunks and positions

def subset(chunk, rows):
    """Rows of a chunk dict (read_chunk's arrays), policy entries included."""
    rows = np.asarray(rows, np.int64)
    totals = chunk["policy_count"].astype(np.int64).sum(1)
    starts = np.cumsum(totals) - totals
    take = totals[rows]
    within = np.arange(int(take.sum())) - np.repeat(np.cumsum(take) - take, take)
    entries = np.repeat(starts[rows], take) + within
    out = {k: v[rows] for k, v in chunk.items() if k in PER_ROW}
    out["policy_index"], out["policy_probability"] = chunk["policy_index"][entries], chunk["policy_probability"][entries]
    return out


def with_timing(chunk):
    """Every chunk gets the v3 timing fields (unknown = -1, no landings) so chunks of both versions concatenate."""
    n = len(chunk["value"])
    chunk.setdefault("seat_clocks_ms", np.full((n, 4), -1.0, np.float32))
    chunk.setdefault("elapsed_ms", np.full((n, 2), -1.0, np.float32))
    chunk.setdefault("recent_landed", np.zeros((n, 20), np.uint8))
    return chunk


def concat(chunks):
    return {k: np.concatenate([c[k] for c in chunks]) for k in chunks[0]}


def popcount(bits):
    """Set bits per int64, vectorized."""
    return np.unpackbits(np.ascontiguousarray(bits).view(np.uint8).reshape(*bits.shape, 8), axis=-1).sum(-1)


def material(bits, values, block, planes):
    """(ours, theirs) material on one board, pockets included."""
    o = block * planes
    counts = popcount(bits[:, o:o + 12]).astype(np.float64)
    pockets = np.rint(values[:, o + 12:o + 22].astype(np.float64) * 16)
    weights = np.array(PIECE_VALUES, np.float64)
    ours = counts[:, :6] @ weights + pockets[:, :5] @ weights[:5]
    theirs = counts[:, 6:12] @ weights + pockets[:, 5:] @ weights[:5]
    return ours, theirs


def in_check(bits, values, block, planes):
    """Is the side on turn in check on this board. Each block is drawn from its own team's side (that side's pieces
    first, at the bottom), so treating it as White on a vertically mirrored board keeps every attack."""
    import chess
    o = block * planes
    board = chess.Board(None)
    piece = [int(b) & MASK64 for b in bits[o:o + 12]]
    board.pawns, board.knights, board.bishops, board.rooks, board.queens, board.kings = (
        piece[i] | piece[i + 6] for i in range(6))
    board.occupied_co[chess.WHITE] = piece[0] | piece[1] | piece[2] | piece[3] | piece[4] | piece[5]
    board.occupied_co[chess.BLACK] = piece[6] | piece[7] | piece[8] | piece[9] | piece[10] | piece[11]
    board.occupied = board.occupied_co[chess.WHITE] | board.occupied_co[chess.BLACK]
    board.turn = chess.WHITE if values[o + TURN] > 0.5 else chess.BLACK
    return board.is_check()


def fen_text(bits, values, planes):
    """Display text from the planes: per board, the team-relative position (uppercase = the team's side there,
    at the bottom; board B's is the partner's), pockets [ours|theirs], who is on turn, then the team's clock lead."""
    parts = []
    for block, name in ((0, "A"), (1, "B")):
        o = block * planes
        grid = [["1"] * 8 for _ in range(8)]
        for p in range(12):
            bb = int(bits[o + p]) & MASK64
            letter = "PNBRQK"[p % 6] if p < 6 else "pnbrqk"[p % 6]
            while bb:
                square = (bb & -bb).bit_length() - 1
                grid[square // 8][square % 8] = letter
                bb &= bb - 1
        ranks = []
        for rank in range(7, -1, -1):
            text, empty = "", 0
            for c in grid[rank]:
                if c == "1":
                    empty += 1
                else:
                    text += (str(empty) if empty else "") + c
                    empty = 0
            ranks.append(text + (str(empty) if empty else ""))
        counts = np.rint(values[o + 12:o + 22].astype(np.float64) * 16).astype(int)
        ours = "".join(l * n for l, n in zip("PNBRQ", counts[:5]))
        theirs = "".join(l * n for l, n in zip("pnbrq", counts[5:]))
        parts.append(f"{name} {'/'.join(ranks)}[{ours}|{theirs}] {'us' if values[o + TURN] > 0.5 else 'them'}")
    return " ; ".join(parts) + f" ; lead {float(values[LEAD]):+.2f}"


def categorize(chunk, planes, lead_cut):
    """Category index (into CATEGORIES) of every row."""
    bits, values = chunk["plane_bits"], chunk["plane_values"]
    n = len(bits)
    moves_left = chunk["moves_left"].astype(np.float64)
    decided = chunk["wdl"].sum(1) > 0
    near_end = decided & (moves_left >= 0) & (moves_left * 100 <= NEAR_END_PLIES)
    swing = np.zeros(n, bool)
    for block in range(2):
        ours, theirs = material(bits, values, block, planes)
        swing |= np.abs(ours - theirs) >= SWING_MATERIAL
    clocks = chunk["seat_clocks_ms"]
    own = clocks[:, [0, 2]]
    timed = (own >= 0).all(1)
    low = np.where(timed, own.min(1) < LOW_CLOCK_MS, values[:, LEAD].astype(np.float64) <= lead_cut)
    pockets = np.rint(values[:, 12:22].astype(np.float64) * 16).sum(1) \
        + np.rint(values[:, planes + 12:planes + 22].astype(np.float64) * 16).sum(1)
    crowded = pockets >= CROWDED_POCKET
    opening = chunk["ply"].astype(np.int64) <= OPENING_PLY
    out = np.full(n, len(CATEGORIES) - 1, np.int64)
    for i in range(n - 1, -1, -1):
        if near_end[i]:
            out[i] = 0
        elif swing[i] or any(in_check(bits[i], values[i], b, planes) for b in range(2)):
            out[i] = 1
        elif low[i]:
            out[i] = 2
        elif crowded[i]:
            out[i] = 3
        elif opening[i]:
            out[i] = 4
    return out


def first_iteration(root):
    runs = os.path.join(root, "runs")
    its = sorted(int(d[5:]) for d in os.listdir(runs) if d.startswith("rl-it") and d[5:].isdigit()) \
        if os.path.isdir(runs) else []
    for n in its:
        if os.path.isdir(os.path.join(runs, f"rl-it{n}", "search", "val")):
            return n
    return None


def chunk_files(sources):
    out = []
    for source in sources:
        if os.path.isdir(source):
            for dirpath, _, files in sorted(os.walk(source)):
                out.extend(os.path.join(dirpath, f) for f in sorted(files) if f.endswith(".dst"))
        elif source.endswith(".dst"):
            out.append(source)
    return out


def kind(path):
    return "val" if "val" in os.path.normpath(path).split(os.sep) else "train"


def build(args):
    from hivemind.distill.data import HEADER_BYTES, WIDE_PLANES, _header, read_chunk

    root = os.path.abspath(args.root)
    probe_dir = os.path.join(root, "probe")
    if os.path.isfile(os.path.join(probe_dir, MANIFEST)) and not args.force:
        print(f"{probe_dir} exists; kept (the probe set stays fixed for the run; --force rebuilds)")
        return 0
    if args.sources:
        sources = args.sources
    else:
        n = first_iteration(root)
        if n is None:
            raise SystemExit(f"{root}: no runs/rl-itN/search/val yet; nothing to sample")
        search = os.path.join(root, "runs", f"rl-it{n}", "search")
        sources = [os.path.join(search, "val"), os.path.join(search, "train")]
    files = chunk_files(sources)
    rng = np.random.default_rng(args.seed)
    val = [f for f in files if kind(f) == "val"]
    train = [f for f in files if kind(f) == "train"]
    # Live chunks run to hundreds per iteration; a random few hold plenty of every category.
    pick = lambda paths: [paths[i] for i in sorted(rng.permutation(len(paths))[:args.max_chunks])]
    files = pick(val) + pick(train)
    if not files:
        raise SystemExit(f"no .dst chunks under {sources}")
    wide = False
    for f in files:
        with open(f, "rb") as stream:
            wide |= _header(stream.read(HEADER_BYTES), f)[2] == WIDE_PLANES
    layout = "wide" if wide else "narrow"
    planes = (130 if wide else 78) // 2
    chunks, origin = [], []
    for f in files:
        chunk = with_timing(read_chunk(f, layout=layout))
        chunks.append(subset(chunk, np.arange(len(chunk["value"]))))
        origin += [(rel(f, root), i) for i in range(len(chunk["value"]))]
    pool = concat(chunks)
    timed = (pool["seat_clocks_ms"][:, [0, 2]] >= 0).all(1)
    lead = pool["plane_values"][:, LEAD].astype(np.float64)
    lead_cut = float(min(np.quantile(lead[~timed], LEAD_QUANTILE), 0.0)) if (~timed).any() else -np.inf
    category = categorize(pool, planes, lead_cut)
    is_val = np.array([kind(path) == "val" for path, _ in origin])

    # Equal quotas per category, val rows first; what a category lacks is filled from the rest, val first.
    quota = args.positions // len(CATEGORIES)
    order = rng.permutation(len(category))
    order = np.concatenate((order[is_val[order]], order[~is_val[order]]))
    chosen = []
    for c in range(len(CATEGORIES)):
        chosen.extend(order[category[order] == c][:quota].tolist())
    taken = set(chosen)
    chosen.extend([i for i in order.tolist() if i not in taken][:max(0, args.positions - len(chosen))])
    chosen = np.array(sorted(chosen, key=lambda i: (category[i], origin[i])), np.int64)

    probe = subset(pool, chosen)
    np_str = lambda xs: np.array(xs, dtype=str)
    probe["category"] = np_str([CATEGORIES[category[i]] for i in chosen])
    probe["source"] = np_str([f"{origin[i][0]}#{origin[i][1]}" for i in chosen])
    probe["source_kind"] = np_str(["val" if is_val[i] else "train" for i in chosen])
    probe["fen"] = np_str([fen_text(pool["plane_bits"][i], pool["plane_values"][i], planes) for i in chosen])
    os.makedirs(probe_dir, exist_ok=True)
    path = os.path.join(probe_dir, PROBE_FILE)
    tmp = path + f".tmp{os.getpid()}.npz"
    np.savez_compressed(tmp, **probe)
    os.replace(tmp, path)
    counts = {c: int((probe["category"] == c).sum()) for c in CATEGORIES}
    manifest = {"schema": SCHEMA, "run": os.path.basename(root), "built_at": now(), "path": rel(path, root),
                "sha256": sha256(path), "positions": int(len(chosen)), "layout": layout, "planes": 2 * planes,
                "seed": args.seed, "categories": counts,
                "by_source": {k: int((probe["source_kind"] == k).sum()) for k in ("val", "train")},
                "candidates": {"positions": int(len(category)),
                               "categories": {c: int((category == i).sum()) for i, c in enumerate(CATEGORIES)}},
                "thresholds": {"near_end_plies": NEAR_END_PLIES, "opening_ply": OPENING_PLY,
                               "crowded_pocket": CROWDED_POCKET, "swing_material": SWING_MATERIAL,
                               "low_clock_ms": LOW_CLOCK_MS, "lead_cut_untimed": num(lead_cut)},
                "chunks": [rel(f, root) for f in files]}
    write_json(os.path.join(probe_dir, MANIFEST), manifest)
    print(f"probe set: {len(chosen)} positions ({layout}) from {len(files)} chunks, {counts} -> {path}")
    return 0


# ---------------------------------------------------------------- networks

def network_path(root, n, which):
    arms = os.path.join(root, "runs", f"rl-it{n}", "arms", "C")
    for name in ((f"{which}.pt", "best.pt" if which == "last" else "last.pt")):
        if os.path.isfile(os.path.join(arms, name)):
            return os.path.join(arms, name)
    return None


def previous_network(root, n):
    """(path, label) of the network that generated iteration n's games."""
    for k in range(n - 1, 0, -1):
        marker = os.path.join(root, "runs", f"rl-it{k}", "stages", "promote.done")
        try:
            with open(marker) as f:
                promoted = json.load(f).get("promoted")
        except (OSError, ValueError):
            continue
        last = os.path.join(root, "runs", f"rl-it{k}", "arms", "C", "last.pt")
        if promoted and os.path.isfile(last):
            return last, f"it{k}"
    # No promotion yet: the checkpoint iteration 1 trained from (train.done records its sha256).
    candidates = []
    artifacts = os.path.join(root, "artifacts")
    if os.path.isdir(artifacts):
        for dirpath, _, files in sorted(os.walk(artifacts)):
            candidates.extend(os.path.join(dirpath, f) for f in sorted(files) if f.endswith(".pt"))
    try:
        with open(os.path.join(root, "runs", "rl-it1", "stages", "train.done")) as f:
            init = json.load(f).get("init")
    except (OSError, ValueError):
        init = None
    if init:
        for path in candidates:
            if sha256(path) == init:
                return path, "start"
    # train.done not written yet: the start network by name (it13), never the generator checkpoint, which promotion
    # overwrites.
    for path in candidates:
        if "it13" in os.path.basename(path):
            return path, "start"
    return None, None


def load_model(path, device, wide=None):
    import torch
    from hivemind.architectures.twin_board import load_twin_board_checkpoint
    model, _, _ = load_twin_board_checkpoint(torch.load(path, map_location="cpu"), wide=wide)
    # channels_last: about 20% faster on CPU for the depthwise blocks; weights and outputs are unchanged.
    return model.to(device).to(memory_format=torch.channels_last).eval()


def model_chunk(probe, model_layout):
    """The probe set's arrays in the planes a network reads."""
    from hivemind.distill.data import narrow_planes, widen_planes
    chunk = {k: probe[k] for k in (*PER_ROW, "policy_index", "policy_probability")}
    stored = "wide" if probe["plane_bits"].shape[1] == 130 else "narrow"
    if stored == "narrow" and model_layout == "wide":
        chunk["plane_bits"], chunk["plane_values"] = widen_planes(chunk)
    elif stored == "wide" and model_layout == "narrow":
        chunk["plane_bits"], chunk["plane_values"] = narrow_planes(chunk["plane_bits"]), \
            narrow_planes(chunk["plane_values"])
    return chunk


def input_groups(planes):
    """{name: plane indices} over the network's input (two blocks of `planes`)."""
    wide = planes > WIDE
    extras = lambda *ranges: [p for r in ranges for p in r] if wide else []
    groups = {}
    for block, name in ((0, "a"), (1, "b")):
        o = block * planes
        groups[f"{name}_pieces"] = [o + p for p in (*OUR_PIECES, *THEIR_PIECES, *PROMOTED)]
        groups[f"{name}_pocket_ours"] = [o + p for p in (*OUR_POCKET, *extras(OUR_PRESENCE, OUR_RECENT))]
        groups[f"{name}_pocket_theirs"] = [o + p for p in (*THEIR_POCKET, *extras(THEIR_PRESENCE, THEIR_RECENT))]
        groups[f"{name}_all"] = [o + p for p in range(planes) if p != CONSTANT]
    both = lambda *ps: [b * planes + p for b in range(2) for p in ps]
    groups["clocks"] = both(SIT, LEAD, SPEED, *extras(WIDE_CLOCKS))
    groups["turn"] = both(TURN)
    groups["castling_ep"] = both(EP, *CASTLING)
    groups["history"] = both(*LAST_MOVE, HALFMOVE, *REPETITION)
    return groups


def group_of(name):
    parts = name.split(".")
    return ".".join(parts[:2]) if parts[0] in ("blocks", "exchanges") and len(parts) > 1 else parts[0]


def run_model(model, planes_all, batch, hooks=None):
    """Value (W - L), WDL probabilities and both boards' raw policy logits for every row."""
    import torch
    values, wdls, logits = [], [], ([], [])
    with torch.no_grad():
        for start in range(0, len(planes_all), batch):
            out = model(planes_all[start:start + batch].contiguous(memory_format=torch.channels_last))
            values.append(out[0].float().squeeze(1))
            wdls.append(torch.softmax(out[3].float(), 1))
            for board in range(2):
                logits[board].append(out[1][board].float())
    return torch.cat(values), torch.cat(wdls), [torch.cat(x) for x in logits]


def log_policy(logits, legal):
    import torch
    return torch.log_softmax(logits.masked_fill(~legal, float("-inf")), 1)


def kl_rows(p_log, q_log, legal):
    """KL(p || q) over the legal moves, per row."""
    import torch
    return torch.where(legal, p_log.exp() * (p_log - q_log), torch.zeros_like(p_log)).sum(1)


def evaluate(args):
    import torch
    from hivemind.distill import train
    from hivemind.distill.data import DistillDataset

    started = time.time()
    root = os.path.abspath(args.root)
    n = args.iteration
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    manifest_path = os.path.join(root, "probe", MANIFEST)
    if not os.path.isfile(manifest_path):
        raise SystemExit(f"no probe set at {manifest_path}: run `probe_net.py build {args.root}` first")
    with open(manifest_path) as f:
        manifest = json.load(f)
    probe_path = os.path.join(root, manifest["path"])
    probe = dict(np.load(probe_path, allow_pickle=False))
    net = args.net or network_path(root, n, args.checkpoint)
    if not net:
        raise SystemExit(f"no arms/C/{args.checkpoint}.pt for iteration {n} under {root}")
    prev, prev_label = (args.prev, "given") if args.prev else previous_network(root, n)
    model = load_model(net, device)
    layout = train.layout(model)
    previous = load_model(prev, device, wide=layout == "wide") if prev else None
    planes = model.planes_per_board

    chunk = model_chunk(probe, layout)
    rows = len(chunk["value"])
    data = DistillDataset([chunk], device, source=1)
    index = torch.arange(rows, device=device)
    planes_all = data.planes(index)
    targets = data.policy_targets(index, 0), data.policy_targets(index, 1)
    result_wdl = data.wdl  # [loss, draw, win]
    decided = result_wdl.sum(1) > 0
    actual = (result_wdl[:, 2] - result_wdl[:, 0])

    # The rl-loop's loss weights for search rows (cli/train.py distill_command), so "loss" matches C.log's search_loss.
    search = {"policy_a": 1.0, "policy_b": 1.0, "wdl": 1.0, "value": 0.0, "moves_left": 0.1, "arrivals": 0.05}
    weights = {"raw": search, "search": search}
    val_rows = np.flatnonzero(probe["source_kind"] == "val")
    held_rows = val_rows if len(val_rows) else np.arange(rows)
    held = train.evaluate(model, DistillDataset([subset(chunk, held_rows)], device, source=1), weights,
                          batch_size=args.batch)
    model.eval()

    # Dead channels: the max of every ReLU's output over all probe positions, per channel.
    peaks = {}

    def hook(name):
        def record(_, __, output):
            dims = [d for d in range(output.dim()) if d != 1]
            peak = output.detach().float().amax(dim=dims)
            peaks[name] = peak if name not in peaks else torch.maximum(peaks[name], peak)
        return record

    handles = [m.register_forward_hook(hook(name)) for name, m in model.named_modules()
               if isinstance(m, torch.nn.ReLU)]
    value, wdl, logits = run_model(model, planes_all, args.batch)
    for handle in handles:
        handle.remove()
    legal = [targets[b][1] for b in range(2)]
    masks = [targets[b][2] for b in range(2)]
    logp = [log_policy(logits[b], legal[b]) for b in range(2)]

    def pooled(per_board):
        """Mean over the rows where each board is on turn, both boards together."""
        total = sum(float(per_board[b][masks[b]].sum()) for b in range(2))
        count = sum(int(masks[b].sum()) for b in range(2))
        return total / max(1, count)

    held_t = torch.as_tensor(held_rows, device=device)
    held_mask = torch.zeros(rows, dtype=torch.bool, device=device)
    held_mask[held_t] = True
    ce = [-torch.where(legal[b], targets[b][0] * logp[b], torch.zeros_like(logp[b])).sum(1) for b in range(2)]
    best = [logits[b].masked_fill(~legal[b], float("-inf")).argmax(1) for b in range(2)]
    agree_search = [(best[b] == targets[b][0].argmax(1)).float() for b in range(2)]
    masks_held = [masks[b] & held_mask for b in range(2)]

    def pooled_on(per_board, ms):
        total = sum(float(per_board[b][ms[b]].sum()) for b in range(2))
        return total / max(1, sum(int(ms[b].sum()) for b in range(2)))

    heldout = {"loss": num(held["loss"]),
               "policy_loss": num(pooled_on(ce, masks_held)),
               "value_loss": num(held["wdl"]),
               "top1": num(pooled_on(agree_search, masks_held)),
               "value_mae": num(held["value_mae"]),
               "value_mse": num(held["value"]), "moves_left": num(held.get("moves_left", float("nan"))),
               "positions": int(len(held_rows)), "subset": "val" if len(val_rows) else "all"}

    # Calibration of P(win) against the game result, rows with a result.
    p_win = wdl[:, 2]
    won = (result_wdl[:, 2] > 0.5).float()
    buckets, ece, total = [], 0.0, int(decided.sum())
    for i in range(10):
        lo, hi = i / 10, (i + 1) / 10
        sel = decided & (p_win >= lo) & ((p_win < hi) if i < 9 else (p_win <= hi))
        count = int(sel.sum())
        pred = float(p_win[sel].mean()) if count else None
        act = float(won[sel].mean()) if count else None
        if count:
            ece += count / max(1, total) * abs(pred - act)
        buckets.append({"lo": lo, "hi": hi, "n": count, "predicted": num(pred) if count else None,
                        "actual": num(act) if count else None})
    calibration = {"buckets": buckets, "ece": num(ece) if total else None, "positions": total}

    # Input groups, on every step-th row (--ablation-positions): one forward pass per group is most of the CPU time.
    step = max(1, -(-rows // args.ablation_positions)) if args.ablation_positions else 1
    sub = torch.arange(0, rows, step, device=device)
    sub_planes, sub_legal = planes_all[sub], [legal[b][sub] for b in range(2)]
    sub_masks, sub_logp = [masks[b][sub] for b in range(2)], [logp[b][sub] for b in range(2)]
    means = planes_all.mean(0, keepdim=True)
    groups = {}
    for name, idx in input_groups(planes).items():
        ablated = sub_planes.clone()
        zeroed = [p for p in idx if p % planes not in MEAN_FILLED]
        meaned = [p for p in idx if p % planes in MEAN_FILLED]
        if zeroed:
            ablated[:, zeroed] = 0
        if meaned:
            ablated[:, meaned] = means[:, meaned]
        v, _, lg = run_model(model, ablated, args.batch)
        kl = [kl_rows(sub_logp[b], log_policy(lg[b], sub_legal[b]), sub_legal[b]) for b in range(2)]
        groups[name] = {"policy_kl": num(pooled_on(kl, sub_masks)), "value_delta": num((v - value[sub]).abs().mean()),
                        "planes": idx, "fill": "mixed" if zeroed and meaned else ("mean" if meaned else "zero")}

    # Agreement with the previous network, and the layers that moved.
    agreement = {"prev_top1": None, "prev_policy_kl": None, "search_top1": num(pooled(agree_search))}
    swings, prev_value = [], None
    if previous is not None:
        prev_value, _, prev_logits = run_model(previous, planes_all, args.batch)
        prev_logp = [log_policy(prev_logits[b], legal[b]) for b in range(2)]
        prev_best = [prev_logits[b].masked_fill(~legal[b], float("-inf")).argmax(1) for b in range(2)]
        agreement["prev_top1"] = num(pooled([(best[b] == prev_best[b]).float() for b in range(2)]))
        agreement["prev_policy_kl"] = num(pooled([kl_rows(prev_logp[b], logp[b], legal[b]) for b in range(2)]))

    layers, current = [], {}
    for name, p in model.named_parameters():
        current.setdefault(group_of(name), []).append((name, p))
    old = dict(previous.named_parameters()) if previous is not None else {}
    dead = {}
    for name, peak in peaks.items():
        g = group_of(name)
        d = dead.setdefault(g, [0, 0])
        d[0] += int((peak <= 1e-6).sum())
        d[1] += peak.numel()
    for g, params in current.items():
        diff = norm = 0.0
        comparable = bool(old)
        for name, p in params:
            q = old.get(name)
            if q is None or q.shape != p.shape:
                comparable = False
                break
            diff += float((p.detach().float() - q.detach().float()).pow(2).sum())
            norm += float(q.detach().float().pow(2).sum())
        layers.append({"name": g, "delta_rel": num(diff ** 0.5 / norm ** 0.5) if comparable and norm > 0 else None,
                       "dead_frac": num(dead[g][0] / dead[g][1]) if g in dead and dead[g][1] else None,
                       "relu_channels": dead[g][1] if g in dead else 0})

    category = probe["category"]
    fen, source = probe["fen"], probe["source"]
    err = (value - actual).abs()
    err = torch.where(decided, err, torch.full_like(err, -1.0))
    worst = [{"fen": str(fen[i]), "predicted": num(value[i]), "actual": num(actual[i]), "source": str(source[i]),
              "category": str(category[i])}
             for i in torch.argsort(err, descending=True)[:args.top].tolist() if float(err[i]) >= 0]
    if prev_value is not None:
        change = (value - prev_value).abs()
        swings = [{"fen": str(fen[i]), "prev": num(prev_value[i]), "now": num(value[i]),
                   "actual": num(actual[i]) if bool(decided[i]) else None, "source": str(source[i]),
                   "category": str(category[i])}
                  for i in torch.argsort(change, descending=True)[:args.top].tolist()]

    # Per category: where the network is good or bad, and what moved.
    by_category = {}
    cat_t = np.asarray(category)
    for c in CATEGORIES:
        rows_c = torch.as_tensor(cat_t == c, device=device)
        if not bool(rows_c.any()):
            continue
        mc = [masks[b] & rows_c for b in range(2)]
        entry = {"n": int(rows_c.sum()), "policy_loss": num(pooled_on(ce, mc)),
                 "top1": num(pooled_on(agree_search, mc)),
                 "value_mae_result": num(err[rows_c & decided].mean()) if bool((rows_c & decided).any()) else None}
        if prev_value is not None:
            entry["prev_top1"] = num(pooled_on([(best[b] == prev_best[b]).float() for b in range(2)], mc))
            entry["value_shift"] = num((value - prev_value)[rows_c].mean())
        by_category[c] = entry

    out = {"schema": SCHEMA, "iteration": n, "run": os.path.basename(root), "generated_at": now(),
           "network": rel(net, root), "sha256": sha256(net),
           "previous": {"network": rel(prev, root), "sha256": sha256(prev), "label": prev_label} if prev else None,
           "probe_set": {"path": manifest["path"], "positions": manifest["positions"], "sha256": manifest["sha256"]},
           "heldout": heldout, "calibration": calibration, "input_groups": groups,
           "input_fill": "zero for piece, square, pocket, presence and landing planes; probe-set mean for turn, "
                         "castling, sit right, clock lead/speed, halfmove, repetition and wide clock planes",
           "ablation_positions": int(len(sub)), "agreement": agreement, "layers": layers, "worst": worst, "swings": swings, "categories": by_category,
           "device": str(device), "seconds": round(time.time() - started, 1)}
    if manifest["sha256"] != sha256(probe_path):
        out["warning"] = "probe.npz does not match its manifest sha256"
    path = args.out or os.path.join(root, "metrics", f"it{n}.probe.json")
    write_json(path, out)
    print(f"it{n}: heldout loss {heldout['loss']}, top1 {heldout['top1']}, ece {calibration['ece']}, "
          f"prev top1 {agreement['prev_top1']} ({out['seconds']} s) -> {path}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build", help="freeze RUN_ROOT/probe/ (once)")
    b.add_argument("root")
    b.add_argument("--from", dest="sources", nargs="+", default=None,
                   help="chunk dirs or .dst files (default: the first iteration's search/val and search/train); "
                        "a path with a 'val' component counts as held out")
    b.add_argument("--positions", type=int, default=2000)
    b.add_argument("--max-chunks", type=int, default=24, help="chunks read per kind (val, train), chosen at random")
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--force", action="store_true", help="rebuild an existing probe set")
    e = sub.add_parser("eval", help="write RUN_ROOT/metrics/itN.probe.json")
    e.add_argument("root")
    e.add_argument("iteration", type=int)
    e.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    e.add_argument("--checkpoint", default="last", choices=("last", "best"))
    e.add_argument("--net", help="network checkpoint (default: runs/rl-itN/arms/C/<checkpoint>.pt)")
    e.add_argument("--prev", help="previous network checkpoint (default: the generator of iteration N)")
    e.add_argument("--out", help="output path (default: RUN_ROOT/metrics/itN.probe.json)")
    e.add_argument("--batch", type=int, default=256)
    e.add_argument("--threads", type=int, default=4, help="CPU threads (this Mac is shared)")
    e.add_argument("--ablation-positions", type=int, default=1000,
                   help="probe rows (evenly spaced) the input-group ablation runs on; 0 = all")
    e.add_argument("--top", type=int, default=20, help="rows in worst and swings")
    args = ap.parse_args(argv)
    if args.command == "eval" and args.iteration < 1:
        ap.error("iteration must be >= 1")
    return build(args) if args.command == "build" else evaluate(args)


if __name__ == "__main__":
    sys.exit(main())
