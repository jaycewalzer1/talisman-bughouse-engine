#!/bin/bash
# The real-clock two-score gate's matches for one network (docs/custom_engine_spec_v7_league.md 3.3): rl-loop --gate2
# calls this every few minutes for the candidate and the incumbent and decides once both net dirs are done
# (hivemind.gate2 decide). Same arena setup as the anchors (launch_anchor_match.sh): pool arena/pools/wave2.json,
# bughouse openings, v3 rules, MEASURE=1 with the warm measurement, equal clocks, one eval.sbatch job per arm and part.
#   bash cluster/gate2_match.sh RUN_ROOT LABEL ONNX     # from anywhere; RUN_ROOT is runs/full/<RUN>
# Arms, both against upstream-it13 (pool clock_k 110), our side with opp_sit_pace GATE2_OPP_SIT:
#   equal  our wave 2 contender at clock_k GATE2_EQUAL_K (110), no think head: strength at equal pace
#   free   clock_k GATE2_FREE_K (160) and think_head true: our time policy
# Contender options: RUN_ROOT/gate2/wave2.json (Options.as_dict(), written by rl-loop or `python -m hivemind.gate2
# wave2-json`), mapped onto arena options as launch_anchor_match.sh maps the gate's wave2 dict, plus model.
# Out: RUN_ROOT/gate2/nets/<first 16 hex of the ONNX sha256>/ {model.onnx, config.json (written once), jobs.tsv,
# <arm>/part<k>/ (eval rows, resumable), <arm>/summary.json, <arm>/pairs.tsv, done (both arms merged)}.
# summary.json: {"arm","label","sha16","games","pairs","score","score_ci","elo","wins","losses","draws","errors",
#   "turns_ahead" (the contender's share of clock-lead turns, clock_teams), "ms_per_node" {part: {key: ms}},
#   "think_head_missing" (when the rows count it), "parts","gave_up","contender_kw","opponent","base_ms"}
# pairs.tsv: part, pair_index, opening_id, contender_team, score_pair, game_ids (hivemind.gate2.read_pairs).
# Every network plays seed GATE2_SEED_BASE + k in part k, so candidate and incumbent meet the same openings in the
# same colours pair by pair. A net dir whose config.json settings differ from the current ones is refused.
# Idempotent: a call submits what is missing and merges what is finished, one line per action, nothing when there is
# nothing to do. A part with its summary.json is done, a part whose job (-J g2_<RUN>_<sha8>_<arm>_p<k>) is queued is
# left alone, any other part is submitted again (eval resumes its games) at most MAX_TRIES (3) times, then GAVE_UP
# (an arm whose parts are all done or given up is merged from the games they have). A failed squeue, NO_SUBMIT=1 or
# RUN_ROOT/STOP_WORKERS (the watchdog's GPU-hour cap) means merge only.
# Settings: GATE2_GAMES (400 per arm, 200 pairs), GATE2_PARTS (4), GATE2_BASE_MS (60000), GATE2_FREE_K (160),
# GATE2_EQUAL_K (110), GATE2_SEED_BASE (7000), GATE2_OPP_SIT (observed|own_k), GATE2_ARMS ("equal free"),
# GATE2_TIME (05:00:00), MAX_TRIES (3), DRY_RUN (1: print the sbatch commands, submit nothing), NO_SUBMIT,
# SBATCH_DEPENDENCY (e.g. afterok:<build job>; sbatch does not read it, so it is passed as --dependency).
set -euo pipefail
RUN_ROOT="${1:-}"; LABEL="${2:-}"; ONNX="${3:-}"
[ -n "$RUN_ROOT" ] && [ -n "$LABEL" ] && [ -n "$ONNX" ] || { echo "usage: gate2_match.sh RUN_ROOT LABEL ONNX" >&2; exit 2; }
[ -f "$ONNX" ] || { echo "gate2 $LABEL: no network at $ONNX" >&2; exit 1; }
ONNX=$(cd "$(dirname "$ONNX")" && pwd -P)/$(basename "$ONNX")
RUN_ROOT=$(cd "$RUN_ROOT" && pwd -P)
cd "$(dirname "$0")/.." || exit 1
REPO=$(pwd -P)
GAMES="${GATE2_GAMES:-400}"
PARTS="${GATE2_PARTS:-4}"
BASE_MS="${GATE2_BASE_MS:-60000}"
FREE_K="${GATE2_FREE_K:-160}"
EQUAL_K="${GATE2_EQUAL_K:-110}"
SEED_BASE="${GATE2_SEED_BASE:-7000}"
OPP_SIT="${GATE2_OPP_SIT:-observed}"
ARMS="${GATE2_ARMS:-equal free}"
MAX_TRIES="${MAX_TRIES:-3}"
PY=.venv/bin/python
RUN=$(basename "$RUN_ROOT")
for v in GAMES PARTS BASE_MS FREE_K EQUAL_K SEED_BASE MAX_TRIES; do
  case "${!v}" in ''|*[!0-9]*) echo "gate2: $v must be a whole number, not '${!v}'" >&2; exit 2 ;; esac
done
case "$OPP_SIT" in observed|own_k) ;; *) echo "gate2: GATE2_OPP_SIT is observed or own_k" >&2; exit 2 ;; esac
for a in $ARMS; do case "$a" in equal|free) ;; *) echo "gate2: unknown arm '$a' (equal, free)" >&2; exit 2 ;; esac; done
[ "$PARTS" -ge 1 ] && [ $((GAMES % (2 * PARTS))) -eq 0 ] || { echo "GATE2_GAMES must split into colour-swapped pairs over $PARTS parts" >&2; exit 2; }
# eval.sbatch takes its folder relative to the repo's runs/ (it writes runs/<NAME>)
case "$RUN_ROOT/" in "$REPO/runs/"*) ;; *) echo "$RUN_ROOT is not under $REPO/runs: eval.sbatch cannot write there" >&2; exit 1 ;; esac
[ -f "$RUN_ROOT/gate2/wave2.json" ] || { echo "gate2 $LABEL: no $RUN_ROOT/gate2/wave2.json (rl-loop --gate2 or python -m hivemind.gate2 wave2-json writes it)" >&2; exit 1; }

SHA=$( (sha256sum "$ONNX" 2>/dev/null || shasum -a 256 "$ONNX") | cut -c1-64)
OUT="$RUN_ROOT/gate2/nets/${SHA:0:16}"
SETTINGS=$($PY -c '
import json, sys
g, p, b, f, e, s, o, arms = sys.argv[1:]
print(json.dumps({"games": int(g), "parts": int(p), "base_ms": int(b), "free_k": int(f), "equal_k": int(e),
                  "seed_base": int(s), "opp_sit": o, "arms": arms.split()}, sort_keys=True))' \
  "$GAMES" "$PARTS" "$BASE_MS" "$FREE_K" "$EQUAL_K" "$SEED_BASE" "$OPP_SIT" "$ARMS")
same_settings() {  # $1: a JSON file holding the settings (or a config.json with "settings")
  $PY -c '
import json, sys
d = json.load(open(sys.argv[1]))
sys.exit(0 if d.get("settings", d) == json.loads(sys.argv[2]) else 1)' "$1" "$SETTINGS"
}
if [ -f "$RUN_ROOT/gate2/settings.json" ] && ! same_settings "$RUN_ROOT/gate2/settings.json"; then
  echo "gate2 $LABEL: settings $SETTINGS differ from $RUN_ROOT/gate2/settings.json; refused" >&2; exit 1
fi
if [ -f "$OUT/config.json" ] && ! same_settings "$OUT/config.json"; then
  echo "gate2 $LABEL: $OUT was played with other settings than $SETTINGS; refused (never mixed)" >&2; exit 1
fi
[ -e "$OUT/done" ] && exit 0

# The engine the run plays: the commit in its launch_settings.txt ("engine <sha>"), else the pin. The checkout may be
# at another commit only when the engine code is the same (eval.sbatch then gets the checkout's HEAD as ENGINE_REF).
W2_DIR=third_party/wave2-hivemind
want=$(sed -n 's/.*  engine \([0-9a-f]\{7,40\}\).*/\1/p' "$RUN_ROOT/launch_settings.txt" 2>/dev/null | head -1 || true)
want="${want:-$(cat cluster/wave2_engine.ref)}"
have=$(git -C "$W2_DIR" rev-parse HEAD 2>/dev/null || echo none)
if [ "${have#"$want"}" = "$have" ] && ! git -C "$W2_DIR" diff --quiet "$have" "$want" -- engine 2>/dev/null; then
  echo "gate2 $LABEL: $W2_DIR is at $have with other engine code than the run's $want; not submitted" >&2; exit 1
fi

# config.json is written once per network (model copy, contender options per arm, opponent), so a later change of
# wave2.json or a default does not change what a resubmitted part plays.
mkdir -p "$OUT" && touch "$OUT/jobs.tsv"
if [ ! -f "$OUT/config.json" ]; then
  $PY - "$RUN_ROOT/gate2/wave2.json" "$ONNX" "$OUT" "$LABEL" "$SHA" "$have" "$SETTINGS" > "$OUT/config.json.tmp" <<'EOF'
import json, os, shutil, sys
sys.path.insert(0, os.getcwd())
from arena.players.upstream import WAVE2_FLAGS, wave2_flag_options
wave2_json, onnx, out, label, sha, ref, settings = sys.argv[1:]
settings = json.loads(settings)
gate = json.load(open(wave2_json))
# The engine flags as arena player options, only those off the arena's wave 2 values (base_mix and seat_move_times
# are self-play settings: the arena plays its own equal clocks), exactly as launch_anchor_match.sh maps them.
kw = {k: v for k, v in gate.items() if k in WAVE2_FLAGS and v != WAVE2_FLAGS[k][1]}
full = {k: v for k, v in gate.items() if k in WAVE2_FLAGS}
if wave2_flag_options(**kw) != wave2_flag_options(**full):
    sys.exit(f"the reduced options {kw} send other engine options than {full}")
model = os.path.join(out, "model.onnx")
shutil.copyfile(onnx, model + ".tmp")
os.replace(model + ".tmp", model)
base = {**kw, "model": model, "opp_sit_pace": settings["opp_sit"]}
arms = {"equal": {**base, "clock_k": settings["equal_k"]},
        "free": {**base, "clock_k": settings["free_k"], "think_head": True}}
json.dump({"label": label, "sha256": sha, "source": onnx, "engine_ref": ref, "settings": settings,
           "wave2": gate, "opponent": "upstream-it13", "opponent_kw": {},
           "contender_kw": {a: arms[a] for a in settings["arms"]}}, sys.stdout, indent=1)
EOF
  mv "$OUT/config.json.tmp" "$OUT/config.json"
  echo "gate2 $LABEL: net dir $OUT"
fi

# One line per arm: name, contender options, engine ref (all from config.json)
lines=$($PY -c '
import json, sys
c = json.load(open(sys.argv[1]))
for a in c["settings"]["arms"]:
    print("\t".join([a, json.dumps(c["contender_kw"][a]), c["engine_ref"]]))' "$OUT/config.json")
# A failed squeue is not an empty queue: resubmitting then would put a second job on a part still being played.
if ! queued=$(squeue -h -u "$USER" -o "%j" 2>/dev/null); then
  [ "${NO_SUBMIT:-0}" = 1 ] || echo "gate2 $LABEL: squeue failed; merging only, nothing submitted"
  NO_SUBMIT=1
fi
STOPPED=0
[ -e "$RUN_ROOT/STOP_WORKERS" ] && STOPPED=1

merge() {  # parts -> <arm>/summary.json and pairs.tsv
  $PY - "$1" "$OUT/config.json" <<'EOF'
import glob, json, math, os, sys
adir, cfg_path = sys.argv[1:]
cfg = json.load(open(cfg_path))
arm = os.path.basename(adir)
def elo(s, n):
    s = min(max(s, 0.5 / n), 1 - 0.5 / n)  # a clean sweep reads as half a game short of it, not infinite
    return -400 * math.log10(1 / s - 1)
def points(g):
    return 1.0 if g.get("result") == g["contender_team"] else 0.5 if g.get("result") == "draw" else 0.0
def pair_index(g):  # as cluster/tune_status.py pairs games
    if g.get("pair_index") is not None:
        return int(g["pair_index"])
    digits = "".join(ch for ch in str(g.get("game_id") or g.get("file") or "") if ch.isdigit())
    return (int(digits) - 1) // 2
def counter(g, name):  # a per-team counter the controller reports (e.g. think_head_missing), for our team
    found = None
    for v in g.values():
        if isinstance(v, dict) and isinstance(v.get(g["contender_team"]), dict) and name in v[g["contender_team"]]:
            found = (found or 0) + v[g["contender_team"]][name]
    return found
games, errors, by_pair, ms, gave_up = [], 0, {}, {}, []
ahead = behind = 0
missing = None
parts = sorted((d for d in glob.glob(os.path.join(adir, "part*"))  # part<k> folders only, not a stray part1.log
                if os.path.isdir(d) and d.rsplit("part", 1)[1].isdigit()), key=lambda d: int(d.rsplit("part", 1)[1]))
for part in parts:
    k = int(part.rsplit("part", 1)[1])
    if os.path.exists(os.path.join(part, "GAVE_UP")) and not os.path.exists(os.path.join(part, "summary.json")):
        gave_up.append(k)
    rows = []
    if os.path.exists(os.path.join(part, "results.jsonl")):
        for line in open(os.path.join(part, "results.jsonl")):
            try:
                rows.append(json.loads(line))
            except ValueError:
                pass  # a line still being written
    elif os.path.exists(os.path.join(part, "summary.json")):
        rows = list(json.load(open(os.path.join(part, "summary.json"))).get("games") or [])
    try:
        ms[k] = {key: v.get("ms_per_node") for key, v in
                 (json.load(open(os.path.join(part, "summary.json"))).get("ms_per_node") or {}).items()}
    except (OSError, ValueError):
        pass
    for g in rows:
        if "error" in g:
            errors += 1
            continue
        games.append(g)
        by_pair.setdefault((k, pair_index(g)), []).append(g)
        ct = (g.get("clock_teams") or {}).get(g["contender_team"]) or {}
        ahead += ct.get("turns_ahead", 0)
        behind += ct.get("turns_behind", 0)
        m = counter(g, "think_head_missing")
        if m is not None:
            missing = (missing or 0) + m
pairs = sorted((key, gs) for key, gs in by_pair.items() if len(gs) == 2)
scores = [sum(points(g) for g in gs) / 2 for _, gs in pairs]
n = len(scores)
# Every part is done or given up here, so no pairs (all games errors, or every part GAVE_UP) is final: the arm is
# still merged (pairs 0, a header-only pairs.tsv) so the net dir reaches done and hivemind.gate2 decides "incomplete"
# at once, instead of rl-loop polling until --gate2-timeout-h.
score = sum(points(g) for g in games) / len(games) if games else None
pm = sum(scores) / n if n else None
half = 1.959963984540054 * math.sqrt(sum((s - pm) ** 2 for s in scores) / (n - 1) / n) if n > 1 else float("nan")
summary = {"arm": arm, "label": cfg["label"], "sha16": cfg["sha256"][:16], "games": len(games), "pairs": n,
           "score": score, "score_ci": [max(0.0, pm - half), min(1.0, pm + half)] if n > 1 else None,
           "elo": round(elo(score, len(games)), 1) if games else None, "wins": sum(points(g) == 1 for g in games),
           "losses": sum(points(g) == 0 for g in games), "draws": sum(points(g) == 0.5 for g in games),
           "errors": errors, "turns_ahead": ahead / (ahead + behind) if ahead + behind else None,
           "ms_per_node": {f"part{k}": v for k, v in sorted(ms.items())}, "parts": len(parts), "gave_up": gave_up,
           "contender_kw": cfg["contender_kw"][arm], "opponent": cfg["opponent"],
           "base_ms": cfg["settings"]["base_ms"], "engine_ref": cfg["engine_ref"]}
if missing is not None:
    summary["think_head_missing"] = missing
with open(os.path.join(adir, "pairs.tsv.tmp"), "w") as f:
    f.write("part\tpair_index\topening_id\tcontender_team\tscore_pair\tgame_ids\n")
    for ((k, i), gs), s in zip(pairs, scores):
        f.write("\t".join(map(str, [k, i, gs[0].get("opening_id"), ",".join(str(g["contender_team"]) for g in gs), s,
                                    ",".join(str(g.get("game_id")) for g in gs)])) + "\n")
os.replace(os.path.join(adir, "pairs.tsv.tmp"), os.path.join(adir, "pairs.tsv"))
json.dump(summary, open(os.path.join(adir, "summary.json.tmp"), "w"), indent=1)
os.replace(os.path.join(adir, "summary.json.tmp"), os.path.join(adir, "summary.json"))
if not n:
    print(f"gate2 {cfg['label']} {arm}: no finished pairs ({len(games)} games, {errors} errors, gave up {gave_up}); "
          "merged as 0 pairs, the gate cannot promote", file=sys.stderr)
    print(f"gate2 {cfg['label']} {arm}: 0 pairs ({len(games)} games, {errors} errors)")
else:
    print(f"gate2 {cfg['label']} {arm}: {summary['games']} games, {n} pairs, score {score:.3f}, Elo {summary['elo']:+.0f}")
EOF
}

# Warm measurement: a cold first-measured arm reads ~20% slow and gets ~20% fewer nodes per clock second
# (docs/speed_gap.md, 4ffe3f6, audit #4).
W2=(POOL=arena/pools/wave2.json SETUP=pool OPENINGS=bughouse RULES=v3 MEASURE=1 MEASURE_NODES=800
    "WARM_KINDS=upstream wave2" "MEASURE_WARM_MS=${MEASURE_WARM_MS:-3000}" "MEASURE_REPEATS=${MEASURE_REPEATS:-10}")
all_done=1
while IFS=$'\t' read -r arm ckw ref; do
  adir="$OUT/$arm"
  # eval.sbatch plays only ENGINE_REF exactly, so a part gets the checkout's HEAD, not the commit frozen in
  # config.json: after the fork moves on (engine/ unchanged), a resubmitted part would be refused.
  if [ "$have" != "$ref" ] && ! git -C "$W2_DIR" diff --quiet "$ref" "$have" -- engine 2>/dev/null; then
    [ -f "$adir/summary.json" ] || echo "gate2 $LABEL $arm: $W2_DIR is at $have with other engine code than this network's first parts ($ref); not submitted" >&2
    NO_SUBMIT_ARM=1
  else
    NO_SUBMIT_ARM=0
  fi
  [ -f "$adir/summary.json" ] && continue
  parts_done=1
  for k in $(seq "$PARTS"); do
    dir="$adir/part$k"
    [ -f "$dir/summary.json" ] && continue
    [ -e "$dir/GAVE_UP" ] && continue
    parts_done=0
    name="g2_${RUN}_${SHA:0:8}_${arm}_p$k"
    grep -qxF "$name" <<<"$queued" && continue
    tries=$(awk -F'\t' -v a="$arm" -v p="part$k" '$1 == a && $2 == p' "$OUT/jobs.tsv" | wc -l | tr -d ' ')
    if [ "$tries" -ge "$MAX_TRIES" ]; then
      mkdir -p "$dir"; echo "gate2 $LABEL $arm part$k: $tries jobs and no result; not resubmitted (see $dir)"; touch "$dir/GAVE_UP"
      continue
    fi
    [ "${NO_SUBMIT:-0}" = 1 ] || [ "$NO_SUBMIT_ARM" = 1 ] && continue
    if [ "$STOPPED" = 1 ]; then
      echo "gate2 $LABEL $arm part$k: not submitted ($RUN_ROOT/STOP_WORKERS)"; continue
    fi
    mkdir -p "$dir"
    vars=(NAME="${dir#"$REPO"/runs/}" ENGINE_REF="$have" OPPONENT=upstream-it13 CONTENDER_KW="$ckw" BASE_MS="$BASE_MS"
          PAIRS=$((GAMES / 2 / PARTS)) SEED=$((SEED_BASE + k)) "${W2[@]}")
    if [ "${DRY_RUN:-0}" = 1 ]; then
      printf 'DRY_RUN'; printf ' %q' "${vars[@]}" sbatch --parsable ${SBATCH_DEPENDENCY:+--dependency=$SBATCH_DEPENDENCY} -J "$name" -t "${GATE2_TIME:-05:00:00}" -o "$dir/eval_%j.log" cluster/eval.sbatch; echo
      id="dry$k"
    else
      id=$(env "${vars[@]}" sbatch --parsable ${SBATCH_DEPENDENCY:+--dependency=$SBATCH_DEPENDENCY} -J "$name" -t "${GATE2_TIME:-05:00:00}" -o "$dir/eval_%j.log" cluster/eval.sbatch)
    fi
    [ "${DRY_RUN:-0}" = 1 ] || printf '%s\tpart%s\t%s\t%s\t%s\n' "$arm" "$k" "$id" "$(date +%FT%T)" "$have" >> "$OUT/jobs.tsv"
    echo "gate2 $LABEL $arm part$k: submitted $id ($((GAMES / PARTS)) games at $BASE_MS ms, try $((tries + 1)))"
  done
  if [ "$parts_done" = 1 ]; then merge "$adir" || all_done=0; else all_done=0; fi
done <<<"$lines"
[ "$all_done" = 1 ] && date +%FT%T > "$OUT/done"
exit 0
