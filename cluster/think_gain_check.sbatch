#!/bin/bash
#SBATCH -J tg_check -A paceship-bughouse -q inferno
#SBATCH -N1 -n6 --mem=32G --gres=gpu:rtx_6000:1 -t 01:00:00
#SBATCH -o tg_check_%j.log
# Spec v7 6.3 (docs/custom_engine_spec_v7_league.md): the think_gain head on the TensorRT engine, before a league
# pilot trusts it (cluster/launch_league_pilot.sh THINK=1 submits this as tg_<RUN>, after the build). The TensorRT
# binding mirrors the arrivals head and cannot run on the Mac; this is its check. Each check is one line of
# OUT/think_check.tsv (name, PASS / FAIL / SKIP, seconds, detail) and the whole goes to OUT/think_check.json
# ("overall": PASS only when no check FAILED). Exit 1 on any FAIL, so a chain after this job (afterok) never starts.
#   build        cluster/build_sp1.sh (TensorRT build-sp1 at the checkout's commit; skipped when ENGINE is given)
#   headed       python -m hivemind.think add-head on it13's checkpoint (fetched, SHA-256 checked): headed .pt + ONNX,
#                PyTorch outputs identical and ONNX Runtime within 1e-4 of it13 exported the same way
#   probe        the headed network with a random think_gain last layer (so the head's output depends on the
#                position and a mis-bound row or board would show), exported the same way
#   compare      on POSITIONS (50) random positions, the engine's `policy` command with it13, the headed and the
#                probe network: it13 prints no think gain; the headed network prints 1/3 on both boards and its
#                value, WDL and move priors equal it13's within TOL; the probe's think gain and value match Python
#                ONNX Runtime on the engine's own input planes (`hivemind dumpplanes`) within TOL (TensorRT is FP16)
#   uci_search   a 64-node search prints `info string think_gain` with the headed network and nothing with it13
#   sbatch cluster/think_gain_check.sbatch                    (OUT defaults to runs/think_gain_check/<job id>)
#   DRY_RUN=1 bash cluster/think_gain_check.sbatch            prints the plan, runs nothing
#   OUT=/tmp/tg ENGINE=<ONNX Runtime build>/hivemind bash cluster/think_gain_check.sbatch   (a local run on the Mac)
# Variables: OUT, W2 (wave2-hivemind checkout), ENGINE, IT13_ONNX, IT13_PT, POSITIONS, SEED, TOL.
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(git rev-parse --show-toplevel)}"
[ -n "${SLURM_JOB_ID:-}" ] && [ "${DRY_RUN:-0}" != 1 ] && source cluster/env.sh

OUT="${OUT:-runs/think_gain_check/${SLURM_JOB_ID:-local_$$}}"
W2="${W2:-$PWD/third_party/wave2-hivemind}"
W2PY="$W2/.venv/bin/python"
PY="${PY:-.venv/bin/python}"
ENGINE_GIVEN="${ENGINE:-}"
ENGINE="${ENGINE:-$W2/engine/build-sp1/hivemind}"
IT13_ONNX="${IT13_ONNX:-$W2/engine/models/twin-s-clock-it13.onnx}"
IT13_PT="${IT13_PT:-}"
POSITIONS="${POSITIONS:-50}"
SEED="${SEED:-0}"
TOL="${TOL:-2e-2}"
HEADED="$OUT/headed"
NET="twin-s-clock-it13-think"

if [ "${DRY_RUN:-0}" = 1 ]; then
  echo "DRY_RUN think_gain_check: OUT=$OUT"
  [ -z "$ENGINE_GIVEN" ] && echo "  build: bash cluster/build_sp1.sh $W2"
  echo "  headed: $W2PY -m hivemind.think add-head --checkpoint ${IT13_PT:-<fetched clock-it13-checkpoint>} --out $HEADED"
  echo "  probe: $HEADED/$NET-probe.onnx (random think_gain last layer)"
  echo "  compare: $PY $OUT/think_gain_check.py compare --engine $ENGINE --it13 $IT13_ONNX --positions $POSITIONS --tol $TOL"
  echo "  uci_search: $PY $OUT/think_gain_check.py search --engine $ENGINE --it13 $IT13_ONNX"
  exit 0
fi

mkdir -p "$OUT/logs"
TSV="$OUT/think_check.tsv"
printf 'check\tstatus\tseconds\tdetail\n' > "$TSV"
echo "commit $(git rev-parse HEAD)  wave2 $(git -C "$W2" rev-parse HEAD)  engine $ENGINE"
command -v nvidia-smi >/dev/null && nvidia-smi -L

check() {  # name, detail, command...: output to OUT/logs/<name>.log, one row in the tsv
  local name="$1" detail="$2" start=$SECONDS status
  shift 2
  echo; echo "== $name: $*"
  if "$@" > "$OUT/logs/$name.log" 2>&1; then status=PASS; else status=FAIL; fi
  tail -n 5 "$OUT/logs/$name.log"
  printf '%s\t%s\t%s\t%s\n' "$name" "$status" "$((SECONDS - start))" "$detail" >> "$TSV"
  echo "$name: $status"
}
skip() { printf '%s\tSKIP\t0\t%s\n' "$1" "$2" >> "$TSV"; echo "$1: SKIP ($2)"; }

if [ -z "$ENGINE_GIVEN" ]; then
  check build "build-sp1 at $(git -C "$W2" rev-parse --short HEAD)" bash cluster/build_sp1.sh "$W2"
else
  skip build "ENGINE given: $ENGINE"
fi

headed() {
  local pt="$IT13_PT"
  if [ -z "$pt" ]; then
    pt=$(cd "$W2" && "$W2PY" -c 'from pathlib import Path; from hivemind.cli.fetch_network import fetch_network
print(fetch_network("clock-it13-checkpoint", Path("artifacts/distill/twin-s-clock-it13").resolve()))' | tail -n 1)
  fi
  [ -f "$pt" ] || { echo "no it13 checkpoint at '$pt'"; return 1; }
  (cd "$W2" && "$W2PY" -m hivemind.think add-head --checkpoint "$pt" --out "$HEADED") || return 1
  # add-head names its files after the checkpoint; the checks below use one name.
  local stem; stem=$(basename "$pt" .pt)
  if [ "$stem-think" != "$NET" ]; then
    cp "$HEADED/$stem-think.pt" "$HEADED/$NET.pt" && cp "$HEADED/$stem-think.onnx" "$HEADED/$NET.onnx"
  fi
}
check headed "it13 + fresh think_gain head (hivemind.think add-head), parity checked" headed

probe() {
  (cd "$W2" && "$W2PY" - "$HEADED/$NET.pt" "$HEADED" <<'EOF'
import sys
from pathlib import Path
import torch
from hivemind.distill.train import export
source, out = Path(sys.argv[1]), Path(sys.argv[2])
checkpoint = torch.load(source, map_location="cpu")
torch.manual_seed(7)
last = "think_gain_head.body.2"
checkpoint["model"][f"{last}.weight"] = torch.randn_like(checkpoint["model"][f"{last}.weight"]) * 0.5
checkpoint["model"][f"{last}.bias"] = torch.tensor([0.3])
torch.save(checkpoint, out / f"{source.stem}-probe.pt")
export(out, checkpoint["size"], checkpoint_name=f"{source.stem}-probe.pt", path=out / f"{source.stem}-probe.onnx")
EOF
  )
}
check probe "headed it13 with a random think_gain last layer" probe

cat > "$OUT/think_gain_check.py" <<'EOF'
"""Helper of cluster/think_gain_check.sbatch: drives the engine's UCI `policy` command and `dumpplanes`, compares
with Python ONNX Runtime. Run with the main repo's .venv (python-chess, onnxruntime)."""
import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path.cwd()))
from tools.upstream_check import random_positions  # noqa: E402


class Uci:
    def __init__(self, engine, model):
        self.proc = subprocess.Popen([engine, "--model", model], cwd=Path(engine).parent, stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        self.send("uci")
        self.until("uciok")
        self.send("setoption name Ponder value false")
        # The flag time model: the planes are TimeState::legacy(team, false), as dumpplanes writes them.
        self.send("setoption name TimeModel value flag")
        self.send("setoption name TimeAdvantage value false")

    def send(self, line):
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()

    def until(self, last):
        lines = []
        for line in self.proc.stdout:
            line = line.rstrip("\n")
            lines.append(line)
            if line.startswith(last):
                return lines
        raise RuntimeError(f"engine ended before {last!r}: {lines[-5:]}")

    def policy(self, job):
        self.send(f"setoption name Team value {'white' if job['team'] == 'T1' else 'black'}")
        self.send(f"position fen {job['fen_a']} | {job['fen_b']}")
        self.send("policy")
        self.send("isready")
        return parse_policy(self.until("readyok"))

    def search(self, nodes):
        self.send("position startpos")
        self.send(f"go nodes {nodes}")
        return self.until("bestmove")

    def close(self):
        self.send("quit")
        self.proc.wait(timeout=60)


def parse_policy(lines):
    out = {"think": None, "A": {}, "B": {}}
    board = None
    for line in lines:
        s = line.strip()
        if s.startswith("Value:"):
            out["value"] = float(s.split()[1])
        elif s.startswith("WDL:"):
            out["wdl"] = [float(x) for x in s.split()[1:4]]
        elif s.startswith("Think gain:"):
            out["think"] = [float(x) for x in s.split()[2:4]]
        elif s.startswith("Board A"):
            board = "A"
        elif s.startswith("Board B"):
            board = "B"
        elif board and ":" in s and not s.startswith("("):
            move, p = s.rsplit(":", 1)
            try:
                out[board][move.strip()] = float(p)
            except ValueError:
                pass
    return out


def engine_planes(engine, jobs, work):
    fens = work / "positions.txt"
    fens.write_text("".join(f"{j['fen_a']};{j['fen_b']};{'w' if j['team'] == 'T1' else 'b'};0\n" for j in jobs))
    out = work / "planes.f32"
    subprocess.run([engine, "dumpplanes", "--fens", str(fens), "--output", str(out)], check=True,
                   cwd=Path(engine).parent)
    planes = np.fromfile(out, np.float32)
    return planes.reshape(len(jobs), -1, 8, 8)


def onnx_outputs(model, planes):
    import onnxruntime as ort
    session = ort.InferenceSession(model, providers=["CPUExecutionProvider"])
    names = [o.name for o in session.get_outputs()]
    return dict(zip(names, session.run(None, {session.get_inputs()[0].name: planes})))


def compare(args):
    jobs = random_positions(args.positions, args.seed)
    results = {}
    for name, model in (("it13", args.it13), ("headed", args.headed), ("probe", args.probe)):
        uci = Uci(args.engine, model)
        try:
            results[name] = [uci.policy(job) for job in jobs]
        finally:
            uci.close()
    failures = []
    if any(r["think"] is not None for r in results["it13"]):
        failures.append("it13 printed a think gain")
    worst = {"headed_think": 0.0, "headed_vs_it13": 0.0, "probe_think": 0.0, "probe_value": 0.0}
    for i, (plain, headed) in enumerate(zip(results["it13"], results["headed"])):
        if headed["think"] is None:
            failures.append(f"pos {i}: the headed network printed no think gain")
            continue
        worst["headed_think"] = max(worst["headed_think"], *(abs(p - 1 / 3) for p in headed["think"]))
        diffs = [abs(plain["value"] - headed["value"])] + [abs(a - b) for a, b in zip(plain["wdl"], headed["wdl"])]
        for board in "AB":
            if set(plain[board]) != set(headed[board]):
                failures.append(f"pos {i} board {board}: move sets differ")
            diffs += [abs(plain[board][m] - headed[board].get(m, math.inf)) for m in plain[board]]
        worst["headed_vs_it13"] = max(worst["headed_vs_it13"], max(diffs))
    planes = engine_planes(args.engine, jobs, Path(args.out))
    reference = onnx_outputs(args.probe, planes)
    if "think_gain" not in reference:
        failures.append("the probe ONNX has no think_gain output")
    else:
        expected = 1 / (1 + np.exp(-reference["think_gain"].astype(np.float64)))
        for i, probe in enumerate(results["probe"]):
            if probe["think"] is None:
                failures.append(f"pos {i}: the probe printed no think gain")
                continue
            worst["probe_think"] = max(worst["probe_think"], float(np.abs(expected[i] - probe["think"]).max()))
            worst["probe_value"] = max(worst["probe_value"], abs(float(reference["value"][i, 0]) - probe["value"]))
        spread = float(expected.std(axis=0).min())
        if spread < 1e-3:
            failures.append(f"the probe's think gain barely varies (std {spread:.2e}): the check would prove nothing")
    failures += [f"{k} {v:.3e} > tol {args.tol}" for k, v in worst.items() if not v <= args.tol]
    report = {"positions": len(jobs), "tol": args.tol, "worst": worst, "failures": failures[:20]}
    (Path(args.out) / "compare.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report))
    return 1 if failures else 0


def search(args):
    failures = []
    for name, model, want in (("headed", args.headed, True), ("it13", args.it13, False)):
        uci = Uci(args.engine, model)
        try:
            lines = uci.search(64)
        finally:
            uci.close()
        think = [line for line in lines if line.startswith("info string think_gain ")]
        print(name, think or "no think_gain line")
        if want and (len(think) != 1 or len(think[0].split()) != 5):
            failures.append(f"{name}: expected one think_gain line, got {think}")
        if not want and think:
            failures.append(f"{name}: printed {think}")
    print("uci search:", "FAIL " + "; ".join(failures) if failures else "PASS")
    return 1 if failures else 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("compare", "search"))
    parser.add_argument("--engine", required=True)
    parser.add_argument("--it13", required=True)
    parser.add_argument("--headed", required=True)
    parser.add_argument("--probe", required=True)
    parser.add_argument("--positions", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tol", type=float, default=2e-2)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    return {"compare": compare, "search": search}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
EOF

NETS=(--engine "$ENGINE" --it13 "$IT13_ONNX" --headed "$HEADED/$NET.onnx" --probe "$HEADED/$NET-probe.onnx")
check compare "policy: it13 / headed / probe on $POSITIONS positions vs ONNX Runtime, tol $TOL" \
  "$PY" "$OUT/think_gain_check.py" compare "${NETS[@]}" --positions "$POSITIONS" --seed "$SEED" --tol "$TOL" --out "$OUT"
check uci_search "info string think_gain after a search (headed), none (it13)" \
  "$PY" "$OUT/think_gain_check.py" search "${NETS[@]}" --out "$OUT"

"$PY" - "$TSV" "$OUT/think_check.json" "$(git -C "$W2" rev-parse HEAD)" <<'EOF'
import csv, json, sys
rows = list(csv.DictReader(open(sys.argv[1]), delimiter="\t"))
overall = "FAIL" if any(r["status"] == "FAIL" for r in rows) or not rows else "PASS"
json.dump({"overall": overall, "engine_commit": sys.argv[3], "checks": rows}, open(sys.argv[2], "w"), indent=1)
print()
for r in rows:
    print(f"{r['check']:<12} {r['status']:<5} {r['seconds']:>6}s  {r['detail']}")
print(f"think gain check: {overall}")
sys.exit(0 if overall == "PASS" else 1)
EOF
