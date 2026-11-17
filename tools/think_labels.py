"""think_gain labels for recorded self-play shards (docs/custom_engine_spec_v2.md section B).

For each chosen sample of each chunk, the position it came from (the chunk's ``.pos.jsonl`` sidecar written by
arena.selfplay: both FENs, clocks, elapsed times, the sample's team and TreeMoveMs) is searched twice with the
clock engine, at N and at FACTOR x N nodes (default 4), with the free-mode options of the original search
(FreePass, MateProbeAnyTime, the same TreeMoveMs, the same TimeAdvantage input). On each board where the sample's
team is on turn, the label is 1.0 when the two searches choose a different entry there (a pass counts as a choice),
else 0.0. Boards where the team is not on turn are masked, and so are both boards when either search returns no
root. Samples not chosen keep whatever labels they had (masked in fresh arena chunks).

Output: one HVM6 chunk per input chunk, same file name, under ``--out`` (plus a copy of its sidecar), which
``convert_selfplay_data.py`` reads; the label columns are ``think_gain_a`` / ``think_gain_b`` with their masks
(arena/recorder.py documents the bytes). HVM5 input becomes HVM6 with NaN arrival counts (no arrival label).

    .venv/bin/python tools/think_labels.py --in runs/custom/main/it01/selfplay/training_data \\
        --out runs/custom/main/it01/selfplay/training_data_think --nodes 200 --fraction 0.25 \\
        --model third_party/hivemind/engine/models/hivemind-clock-v2-fp32.onnx

Notes: the engine's searches are multi-threaded and not bit-for-bit deterministic, so part of the label is search
noise (two runs at the same N can disagree too). That floor is measured on a sample of the labelled positions
(``--repeat-fraction``, default 0.05; ``--repeat-check`` = all of them): a third search at N, and
``think_labels.json`` logs ``noise_rate`` next to ``rate_a`` / ``rate_b``. Read a label rate against it: measured
equal searches disagree 33% of the time at N = 32 and 4-6% at N = 200, so N defaults to 200.
Cost: (1 + FACTOR) x N nodes per labelled position (+ N on the repeat sample), single engine process (no worker
pools on the shared Mac).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from arena.interfaces import TEAM_OF  # noqa: E402
from arena.recorder import (  # noqa: E402
    HEADER,
    NB_POLICY_VALUES,
    TRAILER,
    VERSION_V6,
    FORMATS,
    base_sample,
    decode_trailer,
    encode_trailer,
    read_chunk,
    read_sidecar,
    sidecar_path,
)


def teams_on_turn(fen_a: str, fen_b: str) -> dict[str, str]:
    """{board: team on turn there}, from the side-to-move field of each bracket-form FEN."""
    out = {}
    for board, fen in (("A", fen_a), ("B", fen_b)):
        color = "white" if fen.split()[1] == "w" else "black"
        out[board] = TEAM_OF[f"{board}_{color}"]
    return out


def search_options(meta: dict) -> dict:
    return {"FreePass": True, "MateProbeAnyTime": True, "TreeMoveMs": int(meta.get("treeMs") or 0)}


def _search(proc, meta: dict, nodes: int):
    return proc.search(meta["fenA"], meta["fenB"], team=meta["team"], nodes=int(nodes),
                       time_advantage=bool(meta.get("timeAdvantage", False)), clocks=meta.get("clocks"),
                       elapsed=meta.get("elapsed"), options=search_options(meta), require_option=False)


def label_position(proc, meta: dict, nodes: int, factor: int = 4,
                   noise: dict | None = None) -> tuple[float | None, float | None]:
    """(label A, label B) for one sidecar position; None = masked. With ``noise`` (a dict), a second N-node search
    is run too and ``noise['n']`` / ``noise['changed']`` count how often two equal searches already disagree."""
    on_turn = teams_on_turn(meta["fenA"], meta["fenB"])
    ours = [b for b in ("A", "B") if on_turn[b] == meta["team"]]
    if not ours:
        return None, None
    short = _search(proc, meta, nodes)
    long = _search(proc, meta, nodes * factor)
    if short.no_root or long.no_root:
        return None, None
    labels = {b: (1.0 if short.move_for(b) != long.move_for(b) else 0.0) for b in ours}
    if noise is not None:
        again = _search(proc, meta, nodes)
        if not again.no_root:
            for b in ours:
                noise["n"] += 1
                noise["changed"] += int(again.move_for(b) != short.move_for(b))
    return labels.get("A"), labels.get("B")


def relabel_chunk(path: str, out_dir: str, proc, *, nodes: int, factor: int, choose, stats: dict,
                  noise: dict | None = None, repeat=None) -> str:
    """Write ``out_dir/<name>`` as HVM6 with think_gain labels on the samples ``choose(i)`` picks. With ``noise``,
    the labelled positions ``repeat()`` picks (all of them when ``repeat`` is None) also get the repeat search."""
    channels, samples, version = read_chunk(path, with_version=True)
    metas = read_sidecar(path)
    if metas is not None and len(metas) != len(samples):
        raise ValueError(f"{path}: sidecar has {len(metas)} lines for {len(samples)} samples")
    if metas is None:
        stats["chunks_without_positions"] += 1
    out = []
    for i, buf in enumerate(samples):
        base = base_sample(buf, channels, version)
        if version >= VERSION_V6:
            tg, mask, arrivals = decode_trailer(buf[-TRAILER.size:])
            old = tuple(tg[k] if mask[k] else None for k in range(2))
        else:
            old, arrivals = (None, None), None
            stats["samples_without_arrivals"] += 1
        labels = old
        if metas is not None and choose(i):
            check = noise is not None and (repeat is None or repeat())
            labels = label_position(proc, metas[i], nodes, factor, noise if check else None)
            stats["labelled"] += 1
            for k, b in enumerate("ab"):
                if labels[k] is not None:
                    stats[f"n_{b}"] += 1
                    stats[f"pos_{b}"] += int(labels[k] > 0.5)
        out.append(base + encode_trailer(labels, arrivals))
    os.makedirs(out_dir, exist_ok=True)
    dst = os.path.join(out_dir, os.path.basename(path))
    with open(dst + ".tmp", "wb") as f:
        f.write(HEADER.pack(FORMATS[VERSION_V6], VERSION_V6, channels, NB_POLICY_VALUES, len(out)))
        for b in out:
            f.write(b)
    os.replace(dst + ".tmp", dst)
    if metas is not None:
        shutil.copyfile(sidecar_path(path), sidecar_path(dst))
    stats["samples"] += len(out)
    return dst


def run(args, process=None) -> dict:
    chunks = sorted(os.path.join(args.input, f) for f in os.listdir(args.input) if f.endswith(".hvm"))
    if not chunks:
        raise SystemExit(f"no .hvm chunks in {args.input}")
    if os.path.abspath(args.out) == os.path.abspath(args.input):
        raise SystemExit("--out must differ from --in")
    rng = random.Random(args.seed)
    budget = [args.max_positions if args.max_positions else None]

    def choose(_i: int) -> bool:
        if budget[0] is not None and budget[0] <= 0:
            return False
        pick = rng.random() < args.fraction
        if pick and budget[0] is not None:
            budget[0] -= 1
        return pick

    own = process is None
    if own:
        from arena.players.hivemind import HivemindProcess, resolve_binary
        opts = dict(kv.split("=", 1) for kv in args.engine_option)
        process = HivemindProcess(resolve_binary(args.binary), os.path.abspath(args.model), options=opts)
    stats = {"chunks": 0, "samples": 0, "labelled": 0, "n_a": 0, "n_b": 0, "pos_a": 0, "pos_b": 0,
             "chunks_without_positions": 0, "samples_without_arrivals": 0}
    repeat_fraction = 1.0 if args.repeat_check else args.repeat_fraction
    noise = {"n": 0, "changed": 0} if repeat_fraction > 0 else None
    repeat_rng = random.Random(args.seed + 1)  # its own stream: the labelled samples do not depend on it

    def repeat() -> bool:
        return repeat_rng.random() < repeat_fraction
    t0 = time.perf_counter()
    try:
        for path in chunks:
            relabel_chunk(path, args.out, process, nodes=args.nodes, factor=args.factor, choose=choose, stats=stats,
                          noise=noise, repeat=repeat)
            stats["chunks"] += 1
            print(f"{os.path.basename(path)}: {stats['labelled']} labelled so far", flush=True)
    finally:
        if own:
            process.close()
    stats["rate_a"] = round(stats["pos_a"] / stats["n_a"], 4) if stats["n_a"] else None
    stats["rate_b"] = round(stats["pos_b"] / stats["n_b"], 4) if stats["n_b"] else None
    if noise is not None:
        stats["noise_n"], stats["noise_changed"] = noise["n"], noise["changed"]
        stats["noise_rate"] = round(noise["changed"] / noise["n"], 4) if noise["n"] else None
    stats["wall_s"] = round(time.perf_counter() - t0, 1)
    report = {"config": {k: v for k, v in vars(args).items()}, "stats": stats}
    with open(os.path.join(args.out, "think_labels.json"), "w") as f:
        json.dump(report, f, indent=2)
    return report


def parse_args(argv=None):
    from arena.players.hivemind import ENGINE_DIR

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="input", required=True, help="directory of .hvm chunks (+ .pos.jsonl sidecars)")
    ap.add_argument("--out", required=True, help="output directory for the labelled HVM6 chunks")
    ap.add_argument("--nodes", type=int, default=200,
                    help="N: the short search (default 200; below ~200 two equal searches disagree too often)")
    ap.add_argument("--factor", type=int, default=4, help="the long search is FACTOR x N nodes (default 4)")
    ap.add_argument("--fraction", type=float, default=1.0, help="share of samples to label (default all)")
    ap.add_argument("--max-positions", type=int, default=0, help="stop labelling after this many (0 = no cap)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--engine-option", action="append", default=[], metavar="NAME=VALUE",
                    help="extra UCI option for the engine (repeatable), e.g. BatchSize=1")
    ap.add_argument("--repeat-check", action="store_true",
                    help="search every labelled position a second time at N (same as --repeat-fraction 1)")
    ap.add_argument("--repeat-fraction", type=float, default=0.05,
                    help="share of labelled positions searched a second time at N, to log noise_rate "
                         "(default 0.05; 0 = off)")
    ap.add_argument("--binary", default="hivemind-clock")
    ap.add_argument("--model", default=os.path.join(ENGINE_DIR, "models", "hivemind-clock-v2-fp32.onnx"))
    args = ap.parse_args(argv)
    if any("=" not in kv for kv in args.engine_option):
        ap.error("--engine-option takes NAME=VALUE")
    if not 0.0 <= args.fraction <= 1.0 or not 0.0 <= args.repeat_fraction <= 1.0 or args.nodes < 1 \
            or args.factor < 2:
        ap.error("need 0 <= --fraction <= 1, 0 <= --repeat-fraction <= 1, --nodes >= 1, --factor >= 2")
    return args


def main(argv=None) -> int:
    report = run(parse_args(argv))
    s = report["stats"]
    print(f"\n{s['samples']} samples in {s['chunks']} chunks, {s['labelled']} positions searched; "
          f"board A {s['n_a']} labels ({s['rate_a']} changed), board B {s['n_b']} labels ({s['rate_b']} changed); "
          f"noise floor {s.get('noise_rate')} ({s.get('noise_n', 0)} repeat labels); "
          f"{s['chunks_without_positions']} chunks had no positions file; {s['wall_s']} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
