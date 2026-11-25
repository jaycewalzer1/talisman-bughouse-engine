#!/bin/bash
# Fixed-anchor Elo for one finished iteration of a run: the iteration's network (rl-itN/models/C.onnx) at the real
# 120 s clock against fixed anchors, so iterations are measured on one scale (the gate's 100 ms games against a
# moving generator are not). Same arena setup as the final match and the long test (pool arena/pools/wave2.json,
# bughouse openings, v3 rules, MEASURE=1, equal clocks), same engine options as the iteration's gate (the "wave2"
# dict of stages/match_generator.done, mapped onto the arena's player options), one eval.sbatch job per part.
#   bash cluster/launch_anchor_match.sh runs/full/hybrid1 4        # from the repo root on PACE; the watchdog calls it
# Anchors (ANCHORS, space separated):
#   hybrid-it13    it13 under the same (hybrid) engine options: what training added to the network alone
#   upstream-it13  it13 under stock upstream v3 options (as in final_match_engines.example.json): the headline
# Out: RUN_ROOT/anchors/itN/{config.json,C.onnx,jobs.tsv,done} and per anchor <anchor>/part<k>/ (eval rows, resumable)
# merged into <anchor>/summary.json once every part has finished:
#   {"anchor","iteration","games","score","elo","ci":[elo lo, elo hi],"score_ci":[lo,hi],"wins","losses","draws",
#    "errors","movetime":null,"clock":BASE_MS,"blocks":[{"games","score"}],"parts","engine_ref","contender_kw",
#    "opponent","opponent_kw"}
# and <anchor>/games.tsv (a config line, then one game a line: index, contender result +1/-1/0, game_id, opening_id,
# contender team, reason, part; the gate's games.tsv reading in training_curve.py works on it).
# Idempotent: a call does only what is missing. A part with its summary.json is done, a part whose job (-J
# am_<RUN>_it<N>_<anchor>_p<k>) is queued is left alone, any other part is submitted again (eval resumes its games),
# at most MAX_TRIES (3) times. Prints nothing when there is nothing to do.
# Cost: about 1.6 GPU-minutes a game at 120 s, so ANCHOR_GAMES x anchors is about 21 GPU-hours with the defaults.
# Settings: ANCHORS ("hybrid-it13 upstream-it13"), ANCHOR_GAMES (400 per anchor, even), ANCHOR_PARTS (2: an 8 h job
# plays about 270-300 games at 120 s), ANCHOR_EVERY (2: only iterations N with N % ANCHOR_EVERY == 0; FORCE=1 any),
# SEED_BASE (5000: part k plays seed SEED_BASE + k in every iteration, so iterations meet the same openings),
# BASE_MS (120000), MAX_TRIES (3), NO_SUBMIT (1: merge only), DRY_RUN (1: print the sbatch commands, submit nothing).
set -euo pipefail
RUN_ROOT="${1:-${RUN_ROOT:-}}"
ITER="${2:-${ITER:-}}"
[ -n "$RUN_ROOT" ] && [ -n "$ITER" ] || { echo "usage: launch_anchor_match.sh RUN_ROOT ITER" >&2; exit 2; }
case "$ITER" in ''|*[!0-9]*) echo "ITER is a number" >&2; exit 2 ;; esac
RUN_ROOT=$(cd "$RUN_ROOT" && pwd -P)
cd "$(dirname "$0")/.." || exit 1
REPO=$(pwd -P)
ANCHORS="${ANCHORS:-hybrid-it13 upstream-it13}"
GAMES="${ANCHOR_GAMES:-400}"
PARTS="${ANCHOR_PARTS:-2}"
EVERY="${ANCHOR_EVERY:-2}"
SEED_BASE="${SEED_BASE:-5000}"
BASE_MS="${BASE_MS:-120000}"
MAX_TRIES="${MAX_TRIES:-3}"
PY=.venv/bin/python
RUN=$(basename "$RUN_ROOT")
IT="$RUN_ROOT/runs/rl-it$ITER"
OUT="$RUN_ROOT/anchors/it$ITER"

[ "${FORCE:-0}" = 1 ] || [ $((ITER % EVERY)) -eq 0 ] || exit 0
[ -e "$OUT/done" ] && exit 0
[ -f "$IT/stages/iteration.done" ] || { echo "it$ITER: no stages/iteration.done yet" >&2; exit 1; }
[ -f "$IT/models/C.onnx" ] || { echo "it$ITER: no models/C.onnx" >&2; exit 1; }
[ $((GAMES % (2 * PARTS))) -eq 0 ] || { echo "ANCHOR_GAMES must split into colour-swapped pairs over $PARTS parts" >&2; exit 1; }
# eval.sbatch takes its folder relative to the repo's runs/ (it writes runs/<NAME>)
case "$OUT/" in "$REPO/runs/"*) ;; *) echo "$RUN_ROOT is not under $REPO/runs: eval.sbatch cannot write there" >&2; exit 1 ;; esac

# The engine the run plays: the commit in its launch_settings.txt ("engine <sha>"), else the pin. The checkout may be
# at another commit only when the engine code is the same (eval.sbatch then gets the checkout's HEAD as ENGINE_REF).
W2_DIR=third_party/wave2-hivemind
want=$(sed -n 's/.*  engine \([0-9a-f]\{7,40\}\).*/\1/p' "$RUN_ROOT/launch_settings.txt" 2>/dev/null | head -1)
want="${want:-$(cat cluster/wave2_engine.ref)}"
have=$(git -C "$W2_DIR" rev-parse HEAD 2>/dev/null || echo none)
if [ "${have#"$want"}" = "$have" ] && ! git -C "$W2_DIR" diff --quiet "$have" "$want" -- engine 2>/dev/null; then
  echo "it$ITER anchors: $W2_DIR is at $have with other engine code than the run's $want; not submitted" >&2; exit 1
fi

# config.json is written once per iteration (contender options, model copy, each anchor's opponent), so a later
# promotion or a changed default does not change what a resubmitted part plays.
mkdir -p "$OUT" && touch "$OUT/jobs.tsv"
if [ ! -f "$OUT/config.json" ]; then
  $PY - "$IT" "$OUT" "$ANCHORS" "$have" "$GAMES" "$PARTS" "$SEED_BASE" "$BASE_MS" > "$OUT/config.json.tmp" <<'EOF'
import hashlib, json, os, shutil, sys
sys.path.insert(0, os.getcwd())
from arena.players.upstream import WAVE2_FLAGS, wave2_flag_options
it, out, anchors, ref, games, parts, seed_base, base_ms = sys.argv[1:]
gate = None
for name in ("match_generator", "match_it0"):
    try:
        gate = json.load(open(os.path.join(it, "stages", name + ".done"))).get("wave2")
    except (OSError, ValueError):
        continue
    if gate:
        break
if not gate:
    sys.exit(f"{it}: no wave2 engine options in stages/match_*.done")
# The gate's engine flags as arena player options, only those off the arena's wave 2 values (base_mix and
# seat_move_times are match settings: the arena plays its own equal clocks)
kw = {k: v for k, v in gate.items() if k in WAVE2_FLAGS and v != WAVE2_FLAGS[k][1]}
full = {k: v for k, v in gate.items() if k in WAVE2_FLAGS}
if wave2_flag_options(**kw) != wave2_flag_options(**full):
    sys.exit(f"the reduced options {kw} send other engine options than the gate's {full}")
model = os.path.join(out, "C.onnx")
shutil.copyfile(os.path.join(it, "models", "C.onnx"), model)
sha = hashlib.sha256(open(model, "rb").read()).hexdigest()
known = {"hybrid-it13": ("wave2-self", dict(kw)), "upstream-it13": ("upstream-it13", {})}
bad = [a for a in anchors.split() if a not in known]
if bad:
    sys.exit(f"unknown anchor(s) {bad}; known: {sorted(known)}")
json.dump({"iteration": int(os.path.basename(it)[5:]), "engine_ref": ref, "model": model, "model_sha256": sha,
           "contender_kw": {**kw, "model": model}, "games": int(games), "parts": int(parts),
           "seed_base": int(seed_base), "base_ms": int(base_ms), "gate_wave2": gate,
           "anchors": {a: {"opponent": known[a][0], "opponent_kw": known[a][1]} for a in anchors.split()}},
          sys.stdout, indent=1)
EOF
  mv "$OUT/config.json.tmp" "$OUT/config.json"
fi

# One line per anchor: name, opponent, opponent options, contender options (all from config.json)
lines=$($PY -c '
import json, sys
c = json.load(open(sys.argv[1]))
for a, v in c["anchors"].items():
    print("\t".join([a, v["opponent"], json.dumps(v["opponent_kw"]), json.dumps(c["contender_kw"]),
                     str(c["games"]), str(c["parts"]), str(c["seed_base"]), str(c["base_ms"]), c["engine_ref"]]))' "$OUT/config.json")
# A failed squeue is not an empty queue: resubmitting then would put a second job on a part still being played.
if ! queued=$(squeue -h -u "$USER" -o "%j" 2>/dev/null); then
  [ "${NO_SUBMIT:-0}" = 1 ] || echo "it$ITER anchors: squeue failed; merging only, nothing submitted"
  NO_SUBMIT=1
fi

merge() {  # parts -> <anchor>/summary.json and games.tsv
  $PY - "$1" "$2" "$ITER" <<'EOF'
import glob, json, math, os, sys
sys.path.insert(0, os.getcwd())
from arena.stats import cluster_bootstrap_ci, contender_row, mean_of
from tools.compare_evals import load
adir, cfg_path, it = sys.argv[1], sys.argv[2], int(sys.argv[3])
cfg = json.load(open(cfg_path))
anchor = os.path.basename(adir)
def elo(s, n):
    s = min(max(s, 0.5 / n), 1 - 0.5 / n)  # a clean sweep reads as half a game short of it, not infinite
    return -400 * math.log10(1 / s - 1)
games, rows, errors = [], [], 0
parts = sorted(glob.glob(os.path.join(adir, "part*")), key=lambda d: int(d.rsplit("part", 1)[1]))
for part in parts:
    g, report, _ = load(part)
    errors += (report or {}).get("errors", 0)
    for x in g:
        games.append((os.path.basename(part), x))
if not games:
    sys.exit(f"{adir}: no games")
rows = [contender_row(x) for _, x in games]
ci = cluster_bootstrap_ci(rows, mean_of("score"), n_boot=1000)
scores = [r["score"] for r in rows]
blocks = [{"games": len(c), "score": sum(c) / len(c)} for c in (scores[i:i + 100] for i in range(0, len(scores), 100))]
summary = {"anchor": anchor, "iteration": it, "games": len(rows), "score": ci["est"], "elo": round(elo(ci["est"], len(rows)), 1),
           "ci": [round(elo(ci["lo"], len(rows)), 1), round(elo(ci["hi"], len(rows)), 1)], "score_ci": [ci["lo"], ci["hi"]],
           "wins": scores.count(1.0), "losses": scores.count(0.0), "draws": scores.count(0.5), "errors": errors,
           "movetime": None, "clock": cfg["base_ms"], "blocks": blocks, "parts": len(parts),
           "engine_ref": cfg["engine_ref"], "model_sha256": cfg["model_sha256"], "contender_kw": cfg["contender_kw"],
           **cfg["anchors"][anchor]}
with open(os.path.join(adir, "games.tsv"), "w") as f:
    f.write(f"config\tit{it} vs {anchor} at {cfg['base_ms']} ms, {len(parts)} parts, seeds {cfg['seed_base']}+k\n")
    for i, ((part, x), s) in enumerate(zip(games, scores)):
        f.write("\t".join(map(str, [i, int(2 * s - 1), x.get("game_id"), x.get("opening_id"), x.get("contender_team"),
                                    x.get("reason"), part])) + "\n")
tmp = os.path.join(adir, "summary.json.tmp")
json.dump(summary, open(tmp, "w"), indent=1)
os.replace(tmp, os.path.join(adir, "summary.json"))
print(f"{anchor}: {summary['games']} games, score {summary['score']:.3f}, Elo {summary['elo']:+.0f} "
      f"[{summary['ci'][0]:+.0f}, {summary['ci'][1]:+.0f}]")
EOF
}

# Warm measurement: a cold first-measured arm reads ~20% slow and gets ~20% fewer nodes per clock second
# (docs/speed_gap.md; hybrid1 it2 anchors charged ours 0.050-0.052 vs the opponent 0.040-0.042).
W2=(POOL=arena/pools/wave2.json SETUP=pool OPENINGS=bughouse RULES=v3 MEASURE=1 MEASURE_NODES=800
    "WARM_KINDS=upstream wave2" "MEASURE_WARM_MS=${MEASURE_WARM_MS:-3000}" "MEASURE_REPEATS=${MEASURE_REPEATS:-10}")
all_done=1
while IFS=$'\t' read -r anchor opponent okw ckw games parts seed_base base_ms ref; do
  adir="$OUT/$anchor"
  # eval.sbatch plays only ENGINE_REF exactly, so a part gets the checkout's HEAD, not the commit frozen in
  # config.json at the first launch: after the fork moves on (engine/ unchanged), a resubmitted part would be refused.
  if [ "$have" != "$ref" ] && ! git -C "$W2_DIR" diff --quiet "$ref" "$have" -- engine 2>/dev/null; then
    [ -f "$adir/summary.json" ] || echo "it$ITER $anchor: $W2_DIR is at $have with other engine code than this iteration's first parts ($ref); not submitted" >&2
    NO_SUBMIT_ANCHOR=1
  else
    NO_SUBMIT_ANCHOR=0
  fi
  [ -f "$adir/summary.json" ] && continue
  parts_done=1
  for k in $(seq "$parts"); do
    dir="$adir/part$k"
    [ -f "$dir/summary.json" ] && continue
    parts_done=0
    name="am_${RUN}_it${ITER}_${anchor}_p$k"
    grep -qxF "$name" <<<"$queued" && continue
    tries=$(awk -F'\t' -v a="$anchor" -v p="part$k" '$1 == a && $2 == p' "$OUT/jobs.tsv" | wc -l | tr -d ' ')
    if [ "$tries" -ge "$MAX_TRIES" ]; then
      mkdir -p "$dir"; [ -e "$dir/GAVE_UP" ] || { echo "it$ITER $anchor part$k: $tries jobs and no result; not resubmitted (see $dir)"; touch "$dir/GAVE_UP"; }
      continue
    fi
    [ "${NO_SUBMIT:-0}" = 1 ] || [ "$NO_SUBMIT_ANCHOR" = 1 ] && continue
    mkdir -p "$dir"
    vars=(NAME="${dir#"$REPO"/runs/}" ENGINE_REF="$have" OPPONENT="$opponent" CONTENDER_KW="$ckw" BASE_MS="$base_ms"
          PAIRS=$((games / 2 / parts)) SEED=$((seed_base + k)) "${W2[@]}")
    [ "$okw" = "{}" ] || vars+=(OPPONENT_KW="$okw")
    if [ "${DRY_RUN:-0}" = 1 ]; then
      printf 'DRY_RUN'; printf ' %q' "${vars[@]}" sbatch --parsable -J "$name" -o "$dir/eval_%j.log" cluster/eval.sbatch; echo
      id="dry$k"
    else
      id=$(env "${vars[@]}" sbatch --parsable -J "$name" -o "$dir/eval_%j.log" cluster/eval.sbatch)
    fi
    [ "${DRY_RUN:-0}" = 1 ] || printf '%s\tpart%s\t%s\t%s\t%s\n' "$anchor" "$k" "$id" "$(date +%FT%T)" "$have" >> "$OUT/jobs.tsv"
    echo "it$ITER $anchor part$k: submitted $id ($((games / parts)) games at $base_ms ms, try $((tries + 1)))"
  done
  if [ "$parts_done" = 1 ]; then merge "$adir" "$OUT/config.json"; else all_done=0; fi
done <<<"$lines"
[ "$all_done" = 1 ] && date +%FT%T > "$OUT/done"
exit 0
