"""cluster/probe_net.py on a copy of the local one-iteration run runs/wave2/int-shared.

    third_party/wave2-hivemind/.venv/bin/python -m pytest tests/cluster/test_probe_net.py -q
Needs torch and hivemind (the fork's python); skipped elsewhere. CPU only, small probe sets.
"""

import importlib.util
import json
import os
import shutil

import numpy as np
import pytest

pytest.importorskip("torch")
# The repo .venv has torch and the old upstream hivemind, without distill: the fork is needed.
pytest.importorskip("hivemind.distill.data")
pytest.importorskip("hivemind.architectures.twin_board")
pytest.importorskip("chess")

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIXTURE = os.path.join(REPO, "runs", "wave2", "int-shared")
if not os.path.isdir(os.path.join(FIXTURE, "runs", "rl-it1")):
    pytest.skip("runs/wave2/int-shared is not on this machine", allow_module_level=True)

spec = importlib.util.spec_from_file_location("probe_net", os.path.join(REPO, "cluster", "probe_net.py"))
probe_net = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe_net)

EVAL = ["--threads", "4", "--ablation-positions", "100"]


def copy_run(tmp_path_factory, name):
    # Never mutate the fixture itself: build and eval write probe/ and metrics/ into the run.
    root = str(tmp_path_factory.mktemp(name) / "int-shared")
    shutil.copytree(FIXTURE, root, ignore=shutil.ignore_patterns("probe", "metrics"))
    return root


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    root = copy_run(tmp_path_factory, "probe")
    assert probe_net.main(["build", root, "--positions", "200"]) == 0
    assert probe_net.main(["eval", root, "1", *EVAL]) == 0
    return root


def load(path):
    with open(path) as f:
        return json.load(f)


def test_build_manifest(run):
    manifest = load(os.path.join(run, "probe", "manifest.json"))
    assert manifest["positions"] == 200
    assert sum(manifest["categories"].values()) == 200
    assert set(manifest["categories"]) == set(probe_net.CATEGORIES)
    assert manifest["by_source"]["val"] > 0
    assert manifest["sha256"] == probe_net.sha256(os.path.join(run, manifest["path"]))
    probe = np.load(os.path.join(run, manifest["path"]))
    assert probe["plane_bits"].shape == (200, 78)
    assert len(probe["fen"]) == 200 and all(" ; B " in f for f in probe["fen"])
    assert probe["policy_index"].shape[0] == int(probe["policy_count"].astype(np.int64).sum())


def test_build_is_frozen(run):
    manifest = load(os.path.join(run, "probe", "manifest.json"))
    probe_net.main(["build", run, "--positions", "50"])
    assert load(os.path.join(run, "probe", "manifest.json")) == manifest


def test_force_rebuild_is_deterministic(tmp_path_factory, run):
    other = copy_run(tmp_path_factory, "rebuild")
    probe_net.main(["build", other, "--positions", "200"])
    a = np.load(os.path.join(run, "probe", "probe.npz"))
    b = np.load(os.path.join(other, "probe", "probe.npz"))
    assert list(a["source"]) == list(b["source"])
    probe_net.main(["build", other, "--positions", "120", "--force"])
    assert load(os.path.join(other, "probe", "manifest.json"))["positions"] == 120


def test_subset_keeps_policy_rows():
    chunk = {"value": np.arange(3, dtype=np.float32), "policy_count": np.array([[2, 0], [1, 1], [0, 3]], np.uint16),
             "policy_index": np.arange(7, dtype=np.uint16), "policy_probability": np.ones(7, np.float16)}
    out = probe_net.subset(chunk, [2, 0])
    assert out["value"].tolist() == [2, 0]
    assert out["policy_index"].tolist() == [4, 5, 6, 0, 1]


def start_position(planes=39):
    """Both blocks at the start position, the team on turn on board A (bitboards as board2planes writes them)."""
    bits = np.zeros(2 * planes, np.int64)
    values = np.zeros(2 * planes, np.float32)
    ours = (0xFF00, 0x42, 0x24, 0x81, 0x08, 0x10)
    theirs = (0x00FF000000000000, 0x4200000000000000, 0x2400000000000000, 0x8100000000000000,
              0x0800000000000000, 0x1000000000000000)
    for block in range(2):
        o = block * planes
        for p, bb in enumerate(ours + theirs):
            bits[o + p] = np.array(bb, np.uint64).astype(np.int64)
            values[o + p] = 1
        values[o + probe_net.CONSTANT] = 1
    values[probe_net.TURN] = 1
    return bits, values


def test_fen_text_start_position():
    bits, values = start_position()
    text = probe_net.fen_text(bits, values, 39)
    assert text.startswith("A rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR[|] us ; B rnbqkbnr/")
    assert " them ; lead" in text
    assert not probe_net.in_check(bits, values, 0, 39)


def test_in_check():
    bits, values = start_position()
    # Their queen to e2 (square 12), our king on e1: the side on turn (us) is in check.
    bits[10] = np.array(1 << 12, np.uint64).astype(np.int64)
    bits[0] = np.array(0xEF00, np.uint64).astype(np.int64)
    assert probe_net.in_check(bits, values, 0, 39)


@pytest.mark.parametrize("planes", (39, 65))
def test_input_groups_cover_every_plane(planes):
    groups = probe_net.input_groups(planes)
    semantic = [p for name, idx in groups.items() if not name.endswith("_all") for p in idx]
    assert len(semantic) == len(set(semantic)), "groups overlap"
    every = {b * planes + p for b in range(2) for p in range(planes) if p != probe_net.CONSTANT}
    assert set(semantic) == every
    assert set(groups["a_all"]) | set(groups["b_all"]) == every


def test_previous_network(tmp_path_factory, run):
    it13 = os.path.join(run, "artifacts", "twin-s-clock-it13", "twin-s-clock-it13.pt")
    path, label = probe_net.previous_network(run, 1)
    assert probe_net.sha256(path) == probe_net.sha256(it13) and label == "start"
    # A promotion at it1 makes it1's last.pt the generator of it2.
    other = copy_run(tmp_path_factory, "promoted")
    with open(os.path.join(other, "runs", "rl-it1", "stages", "promote.done"), "w") as f:
        json.dump({"promoted": True, "score": 0.6}, f)
    assert probe_net.previous_network(other, 2) == (os.path.join(other, "runs", "rl-it1", "arms", "C", "last.pt"),
                                                     "it1")
    assert probe_net.previous_network(other, 1)[1] == "start"


def finite(x):
    return x is not None and np.isfinite(x)


def test_eval_contract(run):
    out = load(os.path.join(run, "metrics", "it1.probe.json"))
    assert out["schema"] == 1 and out["iteration"] == 1
    assert out["network"] == os.path.join("runs", "rl-it1", "arms", "C", "last.pt")
    assert out["sha256"] == probe_net.sha256(os.path.join(run, out["network"]))
    assert out["previous"]["label"] == "start"
    manifest = load(os.path.join(run, "probe", "manifest.json"))
    assert out["probe_set"] == {"path": manifest["path"], "positions": 200, "sha256": manifest["sha256"]}
    for key in ("loss", "policy_loss", "value_loss", "top1", "value_mae"):
        assert finite(out["heldout"][key]), key
    assert out["heldout"]["subset"] == "val"
    buckets = out["calibration"]["buckets"]
    assert len(buckets) == 10 and set(buckets[0]) == {"lo", "hi", "n", "predicted", "actual"}
    assert sum(b["n"] for b in buckets) == out["calibration"]["positions"] > 0
    assert 0 <= out["calibration"]["ece"] <= 1
    assert set(out["input_groups"]) == set(probe_net.input_groups(39))
    for name, group in out["input_groups"].items():
        assert finite(group["policy_kl"]) and group["policy_kl"] >= -1e-6, name
        assert finite(group["value_delta"]) and group["planes"], name
    # The network reads its pieces: blanking a board moves the policy far more than castling rights do.
    assert out["input_groups"]["a_pieces"]["policy_kl"] > out["input_groups"]["castling_ep"]["policy_kl"]
    agreement = out["agreement"]
    assert 0 <= agreement["prev_top1"] <= 1 and 0 <= agreement["search_top1"] <= 1
    assert agreement["prev_policy_kl"] >= 0
    names = [layer["name"] for layer in out["layers"]]
    assert names[0] == "stem" and "policy_head" in names and "value_head" in names
    assert all(finite(layer["delta_rel"]) and layer["delta_rel"] > 0 for layer in out["layers"])
    assert len(out["worst"]) == 20 and len(out["swings"]) == 20
    assert set(out["worst"][0]) >= {"fen", "predicted", "actual", "source"}
    assert set(out["swings"][0]) >= {"fen", "prev", "now", "actual"}
    errors = [abs(w["predicted"] - w["actual"]) for w in out["worst"]]
    assert errors == sorted(errors, reverse=True)


def test_eval_same_network_has_no_change(tmp_path_factory, run):
    path = str(tmp_path_factory.mktemp("same") / "same.json")
    net = os.path.join(run, "runs", "rl-it1", "arms", "C", "last.pt")
    probe_net.main(["eval", run, "1", "--prev", net, "--out", path, *EVAL])
    out = load(path)
    assert out["agreement"]["prev_top1"] == 1.0
    assert abs(out["agreement"]["prev_policy_kl"]) < 1e-5
    assert all(layer["delta_rel"] == 0 for layer in out["layers"])
    assert all(abs(s["now"] - s["prev"]) < 1e-5 for s in out["swings"])


def test_heldout_matches_trainer(tmp_path_factory):
    """With every val position in the probe set, heldout loss is the trainer's search_loss on the same val set
    (the last JSON line of arms/C.log is last.pt's evaluation)."""
    root = copy_run(tmp_path_factory, "full")
    probe_net.main(["build", root, "--positions", "5000"])
    probe_net.main(["eval", root, "1", "--threads", "4", "--ablation-positions", "50"])
    out = load(os.path.join(root, "metrics", "it1.probe.json"))
    with open(os.path.join(root, "runs", "rl-it1", "arms", "C.log"), errors="replace") as f:
        last = [json.loads(line) for line in f if line.startswith("{")][-1]
    assert out["heldout"]["positions"] == 434
    assert out["heldout"]["loss"] == pytest.approx(last["search_loss"], abs=1e-4)
    assert out["heldout"]["value_loss"] == pytest.approx(last["search_wdl"], abs=1e-4)
    assert out["heldout"]["value_mae"] == pytest.approx(last["search_value_mae"], abs=1e-4)
