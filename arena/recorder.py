"""Training samples from arena games, written as native HVM5 / HVM6 self-play chunks.

The ``hivemind-clock`` engine prints one serialized ``TrainingSample`` per ``go nodes N sample`` search, byte for
byte as ``engine/src/tools/selfplay.cc`` writes a sample into a chunk, with ``gameId``, ``outcome`` and
``movesLeft`` left at 0 and ``wdl`` at 1. At game end the arena fills those in (``finish_game``) and appends the
samples to chunks (``ShardWriter``) that ``hivemind.data.convert_selfplay_data.read_binary_shard`` reads.

Chunk layout (little endian, no padding)::

    header  '<4sIHHQ'   magic b"HVM5", version 5, input channels, policy values (73*64), sample count
    sample  '<QIHHBBbBf' gameId, nodes, macroPly, movesLeft, team (0 = engine "white" = our T1), hasTimeAdvantage,
                         outcome (+1/0/-1, team's view), wdl (= outcome + 1), rootQ
            planes       channels * 64 bytes (uint8)
            policyA      u16 count, then count * '<Hf' (index, probability)
            policyB      the same
            jointPolicy  u16 count, then count * '<HHf' (indexA, indexB, probability); index 4672 is a pass

HVM6 (docs/custom_engine_spec_v2.md, "Shard format"): the header is the same struct with magic b"HVM6" and version 6
(the version marker), and every sample is the HVM5 sample above followed by one 90-byte trailing block::

    trailer '<ffBB20f'  think_gain_a, think_gain_b   (f32 label: 1.0 = a 4x longer search changed the move, 0.0 not)
                        think_mask_a, think_mask_b   (u8: 1 = label present, 0 = masked; masked labels are 0.0)
                        arrivals[20]                 (f32 counts, index = pocket * 5 + piece, pockets in the order
                                                      ours on A, theirs on A, ours on B, theirs on B (the sample's
                                                      team's view), pieces P N B R Q; NaN = no arrival label)

The arena recorder writes arrival counts for every sample and masks both think_gain labels; tools/think_labels.py
fills the think_gain labels. Next to each chunk the writer can put ``<chunk>.pos.jsonl``: one JSON line per sample,
in chunk order, with the position it came from (both FENs, clocks, elapsed times, seat, time, TreeMoveMs), which
tools/think_labels.py needs to search the position again (the planes alone do not give exact clocks).

Sidecar keys (spec v3, shared contract 2; ``RecordedSample.meta`` + ``finish_game``): ``gameId t seat team ply fenA
fenB clocks elapsed treeMs nodes timeAdvantage`` (v2), then, when the controller filled them, ``nodes_requested
nodes_searched sit_ms start_offset_ms pace clocks_ms elapsed_ms`` (WP-C; ``clocks_ms`` = the four exact clocks in
A_white A_black B_white B_black order, ``elapsed_ms`` = (A, B), floats), and per game ``reason`` (the
``GameResult.reason``), ``opening`` (the opening-pair key: the opening id when given, else the two start FENs),
``gameEndMs`` and ``arrivalWindowMs``. The converter (``hivemind.data.convert_selfplay_data``) turns these into
Parquet columns: exact clock/elapsed scalars that the loader writes into planes 37-42 (the uint8 planes put clocks
on a 1.18 s grid), the end-reason report, the opening-pair validation split and the arrivals end-of-game mask.

movesLeft (spec v3, one definition): the number of moves landed on either board after the sample's search started
(all moves in the game minus ``ply``). Only bare ``(now_ms, seq, bytes)`` tuples without positions fall back to the
number of samples from this one to the end (the stock selfplay.cc count); the arena always records positions.
"""

from __future__ import annotations

import json
import math
import os
import struct
from dataclasses import dataclass
from typing import NamedTuple

HEADER = struct.Struct("<4sIHHQ")
META = struct.Struct("<QIHHBBbBf")
POLICY_ENTRY = struct.Struct("<Hf")
JOINT_ENTRY = struct.Struct("<HHf")
U16 = struct.Struct("<H")
MAGIC = b"HVM5"
VERSION = 5
MAGIC_V6 = b"HVM6"
VERSION_V6 = 6
FORMATS = {5: MAGIC, 6: MAGIC_V6}  # chunk version -> magic
TRAILER = struct.Struct("<ffBB20f")  # HVM6 per-sample trailing block, 90 bytes
ARRIVAL_PIECES = "PNBRQ"
ARRIVAL_POCKETS = ("ours_A", "theirs_A", "ours_B", "theirs_B")
NB_ARRIVALS = len(ARRIVAL_POCKETS) * len(ARRIVAL_PIECES)  # 20
ARRIVAL_WINDOW_MS = 5000
# Spec v3 (WP-D): optional pace-scaled arrival window = ARRIVAL_PACE_MOVES x the mean pace of the four seats,
# clamped to [ARRIVAL_WINDOW_MIN_MS, ARRIVAL_WINDOW_MAX_MS]. Off by default (fixed 5 s window).
ARRIVAL_PACE_MOVES = 10
ARRIVAL_WINDOW_MIN_MS = 2000
ARRIVAL_WINDOW_MAX_MS = 15000
SIDECAR_V3_KEYS = ("nodes_requested", "nodes_searched", "sit_ms", "start_offset_ms", "pace", "clocks_ms",
                   "elapsed_ms")
# GameResult.reason values whose games are kept out of training (spec v3 D4: aborts are scored, not trained on).
NON_TRAINING_REASONS = ("aborted", "max_plies")
NO_ARRIVALS = (float("nan"),) * NB_ARRIVALS
BOARD_VALUES = 64
NB_POLICY_VALUES = 73 * BOARD_VALUES
PASS_POLICY_INDEX = NB_POLICY_VALUES
CHANNEL_CHOICES = (88, 86, 82, 74)  # v3 clock build (2 * 44, diagonal plane), v2 (2 * 43), v1 (2 * 41), stock (2 * 37)
TEAM_OF_BYTE = {0: "T1", 1: "T2"}
DEFAULT_SAMPLES_PER_CHUNK = 4096


class RecordedSample(NamedTuple):
    """One recorded search: the engine's sample bytes plus where and when it was taken (filled by the free-mode
    HivemindTeamController). ``ply`` = moves landed in the game before the search started."""
    now_ms: int
    seq: int
    buf: bytes
    seat: str = ""
    team: str = ""
    ply: int = 0
    fen_a: str = ""
    fen_b: str = ""
    clocks: dict | None = None
    elapsed: dict | None = None
    tree_ms: int = 0
    nodes: int = 0
    time_advantage: bool = False
    # spec v3 shared contract 2 (filled by the free-mode controller, WP-C); None = not recorded
    nodes_requested: int | None = None
    nodes_searched: int | None = None
    sit_ms: float | None = None          # how long the seat then sat (0 if it moved)
    start_offset_ms: float | None = None  # time-odds offset of the sample's team
    pace: tuple | None = None            # the four pace values sent (A_white A_black B_white B_black), ms
    clocks_ms: tuple | None = None       # exact clocks (A_white A_black B_white B_black), ms, float
    elapsed_ms: tuple | None = None      # exact elapsed times (A, B), ms, float

    def meta(self, game_id: int) -> dict:
        """The sidecar line (``<chunk>.pos.jsonl``) for this sample."""
        out = {"gameId": int(game_id), "t": self.now_ms, "seat": self.seat, "team": self.team, "ply": self.ply,
               "fenA": self.fen_a, "fenB": self.fen_b, "clocks": self.clocks, "elapsed": self.elapsed,
               "treeMs": self.tree_ms, "nodes": self.nodes, "timeAdvantage": bool(self.time_advantage)}
        for key in SIDECAR_V3_KEYS:
            value = getattr(self, key)
            if value is not None:
                out[key] = list(map(float, value)) if key in ("pace", "clocks_ms", "elapsed_ms") else value
        return out


@dataclass(frozen=True)
class SampleView:
    game_id: int
    nodes: int
    macro_ply: int
    moves_left: int
    team: int
    has_time_advantage: int
    outcome: int
    wdl: int
    root_q: float
    channels: int
    planes: bytes
    policy_a: list[tuple[int, float]]
    policy_b: list[tuple[int, float]]
    joint: list[tuple[int, int, float]]
    # HVM6 trailing block (None for an HVM5 sample)
    think_gain: tuple[float, float] | None = None
    think_mask: tuple[int, int] | None = None
    arrivals: tuple[float, ...] | None = None


def encode_sample(*, planes: bytes, policy_a, policy_b, joint=(), game_id: int = 0, nodes: int = 0,
                  macro_ply: int = 0, moves_left: int = 0, team: int = 0, has_time_advantage: int = 0,
                  outcome: int = 0, wdl: int | None = None, root_q: float = 0.0) -> bytes:
    """One sample in chunk layout (what the engine prints after ``info string sample``, base64-decoded)."""
    if len(planes) % BOARD_VALUES:
        raise ValueError(f"planes must be channels * 64 bytes, got {len(planes)}")
    out = [META.pack(game_id, nodes, macro_ply, moves_left, team, has_time_advantage, outcome,
                     outcome + 1 if wdl is None else wdl, root_q), bytes(planes)]
    for pol in (policy_a, policy_b):
        out.append(U16.pack(len(pol)))
        out += [POLICY_ENTRY.pack(i, p) for i, p in pol]
    out.append(U16.pack(len(joint)))
    out += [JOINT_ENTRY.pack(a, b, p) for a, b, p in joint]
    return b"".join(out)


def _sparse_end(buf: bytes, pos: int, entry: struct.Struct) -> int:
    if pos + 2 > len(buf):
        raise ValueError("truncated sample")
    n = U16.unpack_from(buf, pos)[0]
    end = pos + 2 + n * entry.size
    if end > len(buf):
        raise ValueError("truncated sample")
    return end


def sample_size(buf: bytes, channels: int, offset: int = 0, version: int = VERSION) -> int:
    """Bytes taken by the sample that starts at ``offset`` when planes have ``channels`` channels (``version`` 6:
    including the trailing block)."""
    pos = offset + META.size + channels * BOARD_VALUES
    for entry in (POLICY_ENTRY, POLICY_ENTRY, JOINT_ENTRY):
        pos = _sparse_end(buf, pos, entry)
    if version >= VERSION_V6:
        pos += TRAILER.size
        if pos > len(buf):
            raise ValueError("truncated sample")
    return pos - offset


def infer_channels(buf: bytes, version: int = VERSION) -> int:
    """The plane channel count for which ``buf`` is exactly one sample (88, 86, 82 or 74)."""
    for c in CHANNEL_CHOICES:
        try:
            if sample_size(buf, c, version=version) == len(buf):
                return c
        except ValueError:
            continue
    raise ValueError(f"cannot parse a {len(buf)}-byte v{version} sample with {CHANNEL_CHOICES} channels")


def infer_format(buf: bytes) -> tuple[int, int]:
    """(channels, version) for which ``buf`` is exactly one sample; HVM5 is tried first."""
    for version in (VERSION, VERSION_V6):
        try:
            return infer_channels(buf, version), version
        except ValueError:
            continue
    raise ValueError(f"cannot parse a {len(buf)}-byte sample as HVM5 or HVM6")


def base_sample(buf: bytes, channels: int, version: int) -> bytes:
    """The HVM5 part of a sample (the whole sample for version 5, all but the trailing block for version 6)."""
    return buf[:-TRAILER.size] if version >= VERSION_V6 else buf


def encode_trailer(think_gain=None, arrivals=None) -> bytes:
    """The HVM6 trailing block. ``think_gain``: None (both masked) or (a, b) with None for a masked board;
    ``arrivals``: 20 counts, or None (all NaN: no arrival label)."""
    tg = tuple(think_gain) if think_gain is not None else (None, None)
    if len(tg) != 2:
        raise ValueError("think_gain must have two entries")
    arr = tuple(float(x) for x in (NO_ARRIVALS if arrivals is None else arrivals))
    if len(arr) != NB_ARRIVALS:
        raise ValueError(f"arrivals must have {NB_ARRIVALS} values, got {len(arr)}")
    return TRAILER.pack(float(tg[0] or 0.0), float(tg[1] or 0.0), int(tg[0] is not None), int(tg[1] is not None),
                        *arr)


def decode_trailer(block: bytes) -> tuple[tuple[float, float], tuple[int, int], tuple[float, ...]]:
    vals = TRAILER.unpack(block)
    return (vals[0], vals[1]), (vals[2], vals[3]), tuple(vals[4:])


def view_sample(buf: bytes, channels: int | None = None, version: int | None = None) -> SampleView:
    if version is None and channels:
        c = channels
        try:
            version = VERSION if sample_size(buf, c) == len(buf) else VERSION_V6
        except ValueError:
            version = VERSION_V6
    elif version is None:
        c, version = infer_format(buf)
    else:
        c = channels or infer_channels(buf, version)
    if sample_size(buf, c, version=version) != len(buf):
        raise ValueError("sample has trailing or missing bytes")
    meta = META.unpack_from(buf, 0)
    pos = META.size
    planes = buf[pos:pos + c * BOARD_VALUES]
    pos += c * BOARD_VALUES
    pols = []
    for entry in (POLICY_ENTRY, POLICY_ENTRY, JOINT_ENTRY):
        n = U16.unpack_from(buf, pos)[0]
        pos += 2
        pols.append([entry.unpack_from(buf, pos + i * entry.size) for i in range(n)])
        pos += n * entry.size
    extra = {}
    if version >= VERSION_V6:
        tg, mask, arr = decode_trailer(buf[pos:pos + TRAILER.size])
        extra = {"think_gain": tg, "think_mask": mask, "arrivals": arr}
    return SampleView(*meta, channels=c, planes=planes, policy_a=pols[0], policy_b=pols[1], joint=pols[2], **extra)


def patch_sample(buf: bytes, *, game_id: int, outcome: int, moves_left: int) -> bytes:
    """``buf`` with gameId, outcome, wdl (= outcome + 1) and movesLeft replaced; everything else unchanged."""
    if outcome not in (-1, 0, 1):
        raise ValueError(f"outcome must be -1, 0 or 1, got {outcome}")
    gid, nodes, ply, _, team, adv, _, _, q = META.unpack_from(buf, 0)
    head = META.pack(int(game_id), nodes, ply, min(int(moves_left), 0xFFFF), team, adv, outcome, outcome + 1, q)
    return head + buf[META.size:]


def outcome_for(team_byte: int, result: str) -> int:
    """+1 / 0 / -1 for the sample's team given the game result ("T1", "T2" or "draw"). Flags count as wins."""
    if result not in ("T1", "T2"):
        return 0
    return 1 if TEAM_OF_BYTE[team_byte] == result else -1


def move_log(events) -> list[tuple[int, str, str | None]]:
    """(ts, board, captured piece letter or None) per landed move, in landing order, from arena event dicts."""
    return [(int(e["ts"]), e["board"], e.get("capture")) for e in events if e.get("type") == "move"]


def pocket_index(team: str, receiver_seat: str) -> int:
    """Index in ARRIVAL_POCKETS of ``receiver_seat``'s pocket from ``team``'s view."""
    from arena.interfaces import BOARD_OF, TEAM_OF
    return (0 if BOARD_OF[receiver_seat] == "A" else 2) + (0 if TEAM_OF[receiver_seat] == team else 1)


def captures(events) -> list[tuple[int, int, str, str]]:
    """(move number, ts, receiving seat, piece) for every capture in a game's arena events, from the ``move`` +
    ``pass`` event pairs (move numbers count from 0 in landing order)."""
    out, n = [], -1
    for e in events:
        if e.get("type") == "move":
            n += 1
        elif e.get("type") == "pass" and n >= 0:
            out.append((n, int(e["ts"]), e["toSeat"], str(e["piece"])))
    return out


def arrival_counts(caps, team: str, ply: int, start_ms: int, window_ms: int = ARRIVAL_WINDOW_MS) -> tuple[float, ...]:
    """Spec v2 C: pieces arriving in each pocket (team-relative, ARRIVAL_POCKETS) x piece type (P N B R Q), counting
    captures (``captures(events)``) whose move landed after the sample's search started (move number >= ``ply``,
    the moves landed when it started) and at most ``window_ms`` after ``start_ms``. A captured promoted piece
    arrives as a pawn (the arena's ``pass`` event already says so)."""
    counts = [0.0] * NB_ARRIVALS
    for n, ts, receiver, piece in caps:
        if n < ply or ts > start_ms + window_ms:
            continue
        p = piece.upper()
        if p in ARRIVAL_PIECES:
            counts[pocket_index(team, receiver) * len(ARRIVAL_PIECES) + ARRIVAL_PIECES.index(p)] += 1.0
    return tuple(counts)


def _as_record(s) -> RecordedSample:
    if isinstance(s, RecordedSample):
        return s
    t, seq, buf = s
    return RecordedSample(t, seq, buf)


def game_end_info(events) -> tuple[str | None, int | None]:
    """(reason, ts) of a game's ``game_end`` event, or (None, None) when the events have none."""
    for e in reversed(list(events or ())):
        if e.get("type") == "game_end":
            return e.get("reason"), int(e["ts"])
    return None, None


def opening_key(events) -> str | None:
    """The opening-pair key of a game: its two start FENs from ``game_start`` (``"<fenA>|<fenB>"``)."""
    for e in events or ():
        if e.get("type") == "game_start" and e.get("fens"):
            return f"{e['fens'].get('A', '')}|{e['fens'].get('B', '')}"
    return None


def arrival_window_ms(rec: "RecordedSample", mode: str = "fixed") -> int:
    """The arrivals window of one sample: ARRIVAL_WINDOW_MS (``mode`` "fixed"), or with ``mode`` "pace"
    ARRIVAL_PACE_MOVES x the mean of the four pace values the sample was searched with, clamped to
    [ARRIVAL_WINDOW_MIN_MS, ARRIVAL_WINDOW_MAX_MS] (fixed when the sample has no pace)."""
    if mode == "fixed":
        return ARRIVAL_WINDOW_MS
    if mode != "pace":
        raise ValueError(f"unknown arrival window mode {mode!r}")
    pace = getattr(rec, "pace", None)
    if not pace:
        return ARRIVAL_WINDOW_MS
    w = ARRIVAL_PACE_MOVES * sum(float(p) for p in pace) / len(pace)
    return int(round(min(max(w, ARRIVAL_WINDOW_MIN_MS), ARRIVAL_WINDOW_MAX_MS)))


def finish_game(samples, result: str, game_id: int, events=None, *, version: int = VERSION,
                with_meta: bool = False, mask_arrivals_past_end: bool = False, arrival_window: str = "fixed",
                opening_id=None, drop_non_training: bool = False):
    """Order a game's samples by search time (``RecordedSample``s, or (now_ms, seq, bytes) tuples, from both
    teams' controllers) and fill gameId, outcome and movesLeft.

    movesLeft (spec v3, the one definition): the number of moves landed on either board after the sample's search
    started (all moves in the game, from ``events``, minus ``ply``). Only without events, or for bare tuples, the
    v1 fallback is used: the number of samples from this one to the end of the game.
    ``version`` 6 appends the HVM6 trailing block: think_gain masked, arrival counts from ``events`` (NaN without
    events). ``with_meta`` also returns the sidecar dicts, in the same order; each carries the game's ``reason``,
    ``opening`` (``opening_id`` or the start FENs), ``gameEndMs`` and the sample's ``arrivalWindowMs``.

    Spec v3 (WP-D), both off by default so older callers get the same bytes:
    ``mask_arrivals_past_end``: an arrivals label whose window runs past the game end (``game_end`` ts) is written
    as NaN (no label): pieces that never arrived because the game ended are not "no arrivals". The converter applies
    the same mask from the sidecar, so data from callers that leave this off is masked there.
    ``arrival_window``: "fixed" (5 s) or "pace" (``arrival_window_ms``).
    ``drop_non_training``: return no samples when the game ended by a reason in NON_TRAINING_REASONS."""
    if version not in FORMATS:
        raise ValueError(f"unknown chunk version {version}")
    reason, end_ms = game_end_info(events)
    if drop_non_training and reason in NON_TRAINING_REASONS:
        return ([], []) if with_meta else []
    opening = str(opening_id) if opening_id is not None else opening_key(events)
    recs = sorted((_as_record(s) for s in samples), key=lambda r: (r.now_ms, r.seq))
    n = len(recs)
    total_moves = None if events is None else len(move_log(events))
    caps = None if events is None else captures(events)
    out, metas = [], []
    for i, r in enumerate(recs):
        moves_left = n - i if total_moves is None or not r.team else max(0, total_moves - r.ply)
        b = patch_sample(r.buf, game_id=game_id, outcome=outcome_for(META.unpack_from(r.buf, 0)[4], result),
                         moves_left=moves_left)
        window = arrival_window_ms(r, arrival_window)
        if version >= VERSION_V6:
            arr = None
            past_end = mask_arrivals_past_end and end_ms is not None and r.now_ms + window > end_ms
            if caps is not None and r.team and not past_end:
                arr = arrival_counts(caps, r.team, r.ply, r.now_ms, window)
            b += encode_trailer(None, arr)
        out.append(b)
        m = r.meta(game_id)
        m.update({"reason": reason, "opening": opening, "gameEndMs": end_ms, "arrivalWindowMs": window})
        metas.append(m)
    return (out, metas) if with_meta else out


class ShardWriter:
    """Appends finished samples to ``<directory>/chunk_<run_id>_<NNNNNN>.hvm`` chunks of ``samples_per_chunk``
    (the naming selfplay.cc uses; convert_selfplay_data splits train/validation by run id and game id).
    ``channels`` is taken from the first sample when not given; every sample must match it."""

    def __init__(self, directory: str, run_id: str, *, channels: int | None = None,
                 samples_per_chunk: int = DEFAULT_SAMPLES_PER_CHUNK, version: int = VERSION) -> None:
        if not run_id or "/" in run_id:
            raise ValueError(f"bad run id {run_id!r}")
        if version not in FORMATS:
            raise ValueError(f"unknown chunk version {version}")
        self.version = version
        self.directory = directory
        self.run_id = run_id
        self.channels = channels
        self.samples_per_chunk = max(1, int(samples_per_chunk))
        self.pending: list[bytes] = []
        self.pending_meta: list[dict | None] = []
        self.paths: list[str] = []
        self.written = 0
        self._index = 0
        os.makedirs(directory, exist_ok=True)
        while os.path.exists(self._path(self._index)):  # never overwrite chunks from an earlier run
            self._index += 1

    def _path(self, index: int) -> str:
        return os.path.join(self.directory, f"chunk_{self.run_id}_{index:06d}.hvm")

    def append(self, samples, metas=None) -> None:
        """Queue finished samples (``metas``: optional sidecar dicts, one per sample, written to
        ``<chunk>.pos.jsonl`` when every sample of the chunk has one)."""
        samples = list(samples)
        metas = list(metas) if metas is not None else [None] * len(samples)
        if len(metas) != len(samples):
            raise ValueError("one meta per sample")
        for b, m in zip(samples, metas):
            if self.channels is None:
                self.channels = infer_channels(b, self.version)
            if sample_size(b, self.channels, version=self.version) != len(b):
                raise ValueError("sample does not match the writer's channel count / chunk version")
            self.pending.append(b)
            self.pending_meta.append(m)
            if len(self.pending) >= self.samples_per_chunk:
                self.flush()

    def flush(self) -> str | None:
        if not self.pending:
            return None
        path = self._path(self._index)
        self._index += 1
        tmp = path + ".tmp"
        if all(m is not None for m in self.pending_meta):
            side = sidecar_path(path)
            with open(side + ".tmp", "w") as f:
                for m in self.pending_meta:
                    f.write(json.dumps(m, separators=(",", ":")) + "\n")
            os.replace(side + ".tmp", side)
        with open(tmp, "wb") as f:
            f.write(HEADER.pack(FORMATS[self.version], self.version, self.channels, NB_POLICY_VALUES,
                                len(self.pending)))
            for b in self.pending:
                f.write(b)
        os.replace(tmp, path)
        self.written += len(self.pending)
        self.pending = []
        self.pending_meta = []
        self.paths.append(path)
        return path

    def close(self) -> None:
        self.flush()


def sidecar_path(chunk_path: str) -> str:
    return chunk_path + ".pos.jsonl"


def read_sidecar(chunk_path: str) -> list[dict] | None:
    """The sidecar lines of a chunk, or None when it has none."""
    side = sidecar_path(chunk_path)
    if not os.path.exists(side):
        return None
    with open(side) as f:
        return [json.loads(line) for line in f if line.strip()]


def chunk_version(path: str) -> int:
    with open(path, "rb") as f:
        magic, version = HEADER.unpack(f.read(HEADER.size))[:2]
    return version


def read_chunk(path: str, with_version: bool = False):
    """(channels, raw samples) of one HVM5 or HVM6 chunk (and the version with ``with_version``); a light reader
    for checks that do not need hivemind installed."""
    with open(path, "rb") as f:
        data = f.read()
    magic, version, channels, policy_values, count = HEADER.unpack_from(data, 0)
    if FORMATS.get(version) != magic or policy_values != NB_POLICY_VALUES:
        raise ValueError(f"{path}: not an HVM5/HVM6 chunk ({magic!r} v{version}, {policy_values} policy values)")
    pos, out = HEADER.size, []
    for _ in range(count):
        n = sample_size(data, channels, pos, version=version)
        out.append(data[pos:pos + n])
        pos += n
    if pos != len(data):
        raise ValueError(f"{path}: trailing bytes")
    return (channels, out, version) if with_version else (channels, out)
