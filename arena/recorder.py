"""Training samples from arena games, written as native HVM5 self-play chunks.

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
"""

from __future__ import annotations

import os
import struct
from dataclasses import dataclass

HEADER = struct.Struct("<4sIHHQ")
META = struct.Struct("<QIHHBBbBf")
POLICY_ENTRY = struct.Struct("<Hf")
JOINT_ENTRY = struct.Struct("<HHf")
U16 = struct.Struct("<H")
MAGIC = b"HVM5"
VERSION = 5
BOARD_VALUES = 64
NB_POLICY_VALUES = 73 * BOARD_VALUES
PASS_POLICY_INDEX = NB_POLICY_VALUES
CHANNEL_CHOICES = (82, 74)  # clock-plane build (2 * 41), stock (2 * 37)
TEAM_OF_BYTE = {0: "T1", 1: "T2"}
DEFAULT_SAMPLES_PER_CHUNK = 4096


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


def sample_size(buf: bytes, channels: int, offset: int = 0) -> int:
    """Bytes taken by the sample that starts at ``offset`` when planes have ``channels`` channels."""
    pos = offset + META.size + channels * BOARD_VALUES
    for entry in (POLICY_ENTRY, POLICY_ENTRY, JOINT_ENTRY):
        pos = _sparse_end(buf, pos, entry)
    return pos - offset


def infer_channels(buf: bytes) -> int:
    """The plane channel count for which ``buf`` is exactly one sample (82 or 74)."""
    for c in CHANNEL_CHOICES:
        try:
            if sample_size(buf, c) == len(buf):
                return c
        except ValueError:
            continue
    raise ValueError(f"cannot parse a {len(buf)}-byte sample with {CHANNEL_CHOICES} channels")


def view_sample(buf: bytes, channels: int | None = None) -> SampleView:
    c = channels or infer_channels(buf)
    if sample_size(buf, c) != len(buf):
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
    return SampleView(*meta, channels=c, planes=planes, policy_a=pols[0], policy_b=pols[1], joint=pols[2])


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


def finish_game(samples, result: str, game_id: int) -> list[bytes]:
    """Order a game's samples by search time ((now_ms, seq, bytes) tuples from both teams' controllers) and fill
    gameId, outcome and movesLeft (samples from this one to the end of the game, as selfplay.cc counts)."""
    ordered = [b for _, _, b in sorted(samples, key=lambda t: (t[0], t[1]))]
    n = len(ordered)
    return [patch_sample(b, game_id=game_id, outcome=outcome_for(META.unpack_from(b, 0)[4], result),
                         moves_left=n - i) for i, b in enumerate(ordered)]


class ShardWriter:
    """Appends finished samples to ``<directory>/chunk_<run_id>_<NNNNNN>.hvm`` chunks of ``samples_per_chunk``
    (the naming selfplay.cc uses; convert_selfplay_data splits train/validation by run id and game id).
    ``channels`` is taken from the first sample when not given; every sample must match it."""

    def __init__(self, directory: str, run_id: str, *, channels: int | None = None,
                 samples_per_chunk: int = DEFAULT_SAMPLES_PER_CHUNK) -> None:
        if not run_id or "/" in run_id:
            raise ValueError(f"bad run id {run_id!r}")
        self.directory = directory
        self.run_id = run_id
        self.channels = channels
        self.samples_per_chunk = max(1, int(samples_per_chunk))
        self.pending: list[bytes] = []
        self.paths: list[str] = []
        self.written = 0
        self._index = 0
        os.makedirs(directory, exist_ok=True)
        while os.path.exists(self._path(self._index)):  # never overwrite chunks from an earlier run
            self._index += 1

    def _path(self, index: int) -> str:
        return os.path.join(self.directory, f"chunk_{self.run_id}_{index:06d}.hvm")

    def append(self, samples) -> None:
        for b in samples:
            if self.channels is None:
                self.channels = infer_channels(b)
            if sample_size(b, self.channels) != len(b):
                raise ValueError("sample does not match the writer's channel count")
            self.pending.append(b)
            if len(self.pending) >= self.samples_per_chunk:
                self.flush()

    def flush(self) -> str | None:
        if not self.pending:
            return None
        path = self._path(self._index)
        self._index += 1
        tmp = path + ".tmp"
        with open(tmp, "wb") as f:
            f.write(HEADER.pack(MAGIC, VERSION, self.channels, NB_POLICY_VALUES, len(self.pending)))
            for b in self.pending:
                f.write(b)
        os.replace(tmp, path)
        self.written += len(self.pending)
        self.pending = []
        self.paths.append(path)
        return path

    def close(self) -> None:
        self.flush()


def read_chunk(path: str) -> tuple[int, list[bytes]]:
    """(channels, raw samples) of one chunk; a light reader for checks that do not need hivemind installed."""
    with open(path, "rb") as f:
        data = f.read()
    magic, version, channels, policy_values, count = HEADER.unpack_from(data, 0)
    if magic != MAGIC or version != VERSION or policy_values != NB_POLICY_VALUES:
        raise ValueError(f"{path}: not an HVM5 chunk ({magic!r} v{version}, {policy_values} policy values)")
    pos, out = HEADER.size, []
    for _ in range(count):
        n = sample_size(data, channels, pos)
        out.append(data[pos:pos + n])
        pos += n
    if pos != len(data):
        raise ValueError(f"{path}: trailing bytes")
    return channels, out
