#!/bin/bash
# The final match: a round robin of three engines, GAMES (200) games per pairing at 120 s and equal clocks, on the
# wave 2 arena setup of the long test (pool arena/pools/wave2.json, bughouse openings, v3 rules, MEASURE=1).
# Run it on PACE from the repo root:
#   ENGINES=cluster/final_match_engines.json NAME=final1 bash cluster/launch_final_match.sh
#   .venv/bin/python cluster/final_match_status.py runs/final_match/final1
# ENGINES is a JSON list of three engines, each {"name", "kind": "wave2" | "upstream", "options": {...}, "model":
# path or null} (cluster/final_match_engines.example.json): kind wave2 is our engine with those player options (e.g.
# the hybrid's), on "model" (an ONNX network, e.g. a run's promoted generator) or on it13; kind upstream is stock
# upstream it13. Each model is copied to runs/final_match/<NAME>/models/ at launch, so a later promotion does not change
# what is measured. A pairing's contender must be a wave2 engine (the pool's contender); two upstream engines cannot
# meet. Each pairing is split into PARTS (2) parallel 8 h eval jobs (-J fm_<NAME>_<pair>_p<k>) with seeds SEED_BASE+k;
# the same part number plays the same openings in every pairing. Cost: about 1.6 GPU-minutes a game at 120 s, so
# 3 x GAMES games is about 16 GPU-hours (the time limits cap it at 3 x PARTS x 8).
# Settings: ENGINES (required), NAME (required), GAMES (200, even), PARTS (2), SEED_BASE (3000), BASE_MS (120000).
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
ENGINES="${ENGINES:?set ENGINES, a JSON list of three engines (cluster/final_match_engines.example.json)}"
NAME="${NAME:?set NAME, the folder under runs/final_match}"
MATCH="$NAME"   # eval.sbatch reads its own NAME (the row folder), set per part below
GAMES="${GAMES:-200}"
PARTS="${PARTS:-2}"
SEED_BASE="${SEED_BASE:-3000}"
BASE_MS="${BASE_MS:-120000}"
OUT="runs/final_match/$NAME"
[ -e "$OUT" ] && { echo "$OUT exists: pick another NAME" >&2; exit 1; }
[ $((GAMES % (2 * PARTS))) -eq 0 ] || { echo "GAMES must split into colour-swapped pairs over $PARTS parts" >&2; exit 1; }
W2_DIR=third_party/wave2-hivemind
PIN=$(cat cluster/wave2_engine.ref)
have=$(git -C "$W2_DIR" rev-parse HEAD)
if [ "$have" != "$PIN" ] && ! git -C "$W2_DIR" diff --quiet "$have" "$PIN" -- engine 2>/dev/null; then
  echo "$W2_DIR is at $have with other engine code than the pin $PIN: build at the pin first" >&2; exit 1
fi
# eval.sbatch refuses unless the fork HEAD equals ENGINE_REF exactly, so pin the commit actually checked out (same engine/
# code as the pin, checked above), not the pin itself.
RUN_REF="$have"
mkdir -p "$OUT/models"
cp "$ENGINES" "$OUT/engines.json"

# One line per pairing: label, opponent pool name, contender options JSON, opponent options JSON (models copied).
.venv/bin/python - "$OUT" > "$OUT/pairings.tsv" <<'EOF'
import itertools, json, os, re, shutil, sys
out = sys.argv[1]
engines = json.load(open(os.path.join(out, "engines.json")))
if len(engines) != 3 or len({e["name"] for e in engines}) != 3:
    sys.exit("ENGINES must list three engines with distinct names")
def kw(e):
    k = dict(e.get("options") or {})
    if e.get("model"):
        src = e["model"]
        if not os.path.isfile(src):
            sys.exit(f"{e['name']}: model {src} not found")
        dst = os.path.abspath(os.path.join(out, "models", re.sub(r"[^\w.-]", "_", e["name"]) + ".onnx"))
        shutil.copyfile(src, dst)
        k["model"] = dst
    return k
for e in engines:
    if e["kind"] not in ("wave2", "upstream"):
        sys.exit(f"{e['name']}: kind must be wave2 or upstream")
    e["kw"] = kw(e)
for a, b in itertools.combinations(engines, 2):
    if a["kind"] == "upstream":
        a, b = b, a
    if a["kind"] == "upstream":
        sys.exit(f"{a['name']} and {b['name']} are both upstream: one must be a wave2 engine")
    opponent = "upstream-it13" if b["kind"] == "upstream" else "wave2-self"
    label = re.sub(r"[^\w.-]", "_", f"{a['name']}_vs_{b['name']}")
    print("\t".join([label, opponent, json.dumps(a["kw"]), json.dumps(b["kw"])]))
EOF

# Warm measurement: a cold first-measured arm reads ~20% slow and gets ~20% fewer nodes per clock second
# (docs/speed_gap.md; hybrid1 it2 anchors charged ours 0.050-0.052 vs the opponent 0.040-0.042).
W2=(POOL=arena/pools/wave2.json SETUP=pool OPENINGS=bughouse RULES=v3 MEASURE=1 MEASURE_NODES=800
    "WARM_KINDS=upstream wave2" "MEASURE_WARM_MS=${MEASURE_WARM_MS:-3000}" "MEASURE_REPEATS=${MEASURE_REPEATS:-10}")
PAIRS_PER_PART=$((GAMES / 2 / PARTS))
ids=()
while IFS=$'\t' read -r label opponent ckw okw; do
  for k in $(seq "$PARTS"); do
    dir="$OUT/$label/part$k"
    mkdir -p "$dir"
    id=$(export NAME="${dir#runs/}" ENGINE_REF="$RUN_REF" OPPONENT="$opponent" CONTENDER_KW="$ckw" BASE_MS="$BASE_MS" \
           PAIRS="$PAIRS_PER_PART" SEED=$((SEED_BASE + k)) "${W2[@]}"
         [ "$okw" = "{}" ] || export OPPONENT_KW="$okw"
         sbatch --parsable -J "fm_${MATCH}_${label}_p$k" cluster/eval.sbatch)
    ids+=("$id")
    printf '%s\tpart%s\t%s\n' "$label" "$k" "$id" >> "$OUT/jobs.tsv"
  done
  echo "$label: vs $opponent, $GAMES games in $PARTS parts"
done < "$OUT/pairings.tsv"
echo "final match $OUT: ${#ids[@]} jobs (${ids[*]}), engine $RUN_REF (pin $PIN)"
echo "status: .venv/bin/python cluster/final_match_status.py $OUT"
