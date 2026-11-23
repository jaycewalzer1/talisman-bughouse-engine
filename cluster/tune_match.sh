#!/bin/bash
# One tuning candidate (a JSON of engine options written by cluster/tune_round.py) against stock upstream-it13 at
# 120 s, split into PARTS parallel 8 h eval jobs (cluster/eval.sbatch, -J tune_<NAME>_<cand>_p<k>). Run it on PACE
# from the repo root; tune_round.py writes one line per candidate into runs/tune/<NAME>/launch.sh:
#   NAME=r1 CAND=runs/tune/r1/candidates/base.json bash cluster/tune_match.sh
#   DRY_RUN=1 NAME=r1 CAND=... bash cluster/tune_match.sh        # print the sbatch lines, submit and create nothing
#   python3 cluster/tune_status.py runs/tune/r1                   # leaderboard, GSPRT, paired vs the baseline
# The arena settings are the final match's (cluster/launch_final_match.sh): pool arena/pools/wave2.json, the pool's
# contender (kind wave2) with the candidate's options, opponent upstream-it13, bughouse openings, rules v3, BASE_MS
# 120000 equal clocks, MEASURE=1 (measures only an arm the candidate does not give an ms_per_node: with the space's
# "fixed" charge both arms carry theirs), the engine pin as ENGINE_REF. Part k plays seed SEED_BASE + k with the same
# PAIRS_PER_PART for every candidate of a round, so candidates meet the same openings in the same colours, pair by
# pair (tune_status.py pairs them). SEED_BASE 5000 keeps tuning off the final match's (3000) and the long test's
# (1000) schedules. A folder is never reused: the candidate's folder must not exist.
# Settings: NAME (round), CAND (candidate JSON), PARTS (2), PAIRS_PER_PART (150: ~300 games fit 8 h, so the job ends
# about when its schedule does), SEED_BASE (5000), BASE_MS (120000), TIME (08:00:00), WORKERS (3), TUNE_RUNS
# (runs/tune), DRY_RUN (0), SBATCH_DEPENDENCY (e.g. afterok:<job>). Cost: about 1.6 GPU-minutes a game at 120 s.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
NAME="${NAME:?set NAME, the tuning round (folder under TUNE_RUNS)}"
CAND="${CAND:?set CAND, a candidate JSON from cluster/tune_round.py}"
ROUND="$NAME"   # eval.sbatch reads its own NAME (the part folder), set per part below
PARTS="${PARTS:-2}"
PAIRS_PER_PART="${PAIRS_PER_PART:-150}"
SEED_BASE="${SEED_BASE:-5000}"
BASE_MS="${BASE_MS:-120000}"
TIME="${TIME:-08:00:00}"
TUNE_RUNS="${TUNE_RUNS:-runs/tune}"
DRY_RUN="${DRY_RUN:-0}"
case "$PARTS$PAIRS_PER_PART" in *[!0-9]*) echo "PARTS and PAIRS_PER_PART must be numbers" >&2; exit 1 ;; esac
[ "$PARTS" -ge 1 ] && [ "$PAIRS_PER_PART" -ge 1 ] || { echo "PARTS and PAIRS_PER_PART must be >= 1" >&2; exit 1; }
case "$TUNE_RUNS" in runs/*) ;; *) echo "TUNE_RUNS must be under runs/ (eval.sbatch writes runs/<NAME>)" >&2; exit 1 ;; esac
[ -f "$CAND" ] || { echo "no candidate file $CAND" >&2; exit 1; }

# The candidate: its name, contender options and opponent options (one line each, JSON on one line).
C=()
while IFS= read -r line; do C+=("$line"); done < <(python3 - "$CAND" <<'EOF'
import json, re, sys
c = json.load(open(sys.argv[1]))
name = c["name"]
if not re.fullmatch(r"[\w.-]+", name):
    sys.exit(f"candidate name {name!r}: letters, digits, _ . - only")
print(name)
print(json.dumps(c["contender_kw"], separators=(",", ":")))
print(json.dumps(c.get("opponent_kw") or {}, separators=(",", ":")))
EOF
)
[ "${#C[@]}" -eq 3 ] || { echo "could not read $CAND" >&2; exit 1; }
CNAME="${C[0]}"; CKW="${C[1]}"; OKW="${C[2]}"
OUT="$TUNE_RUNS/$ROUND/$CNAME"

W2_DIR=third_party/wave2-hivemind
PIN=$(cat cluster/wave2_engine.ref)
have=$(git -C "$W2_DIR" rev-parse HEAD)
if [ "$have" != "$PIN" ] && ! git -C "$W2_DIR" diff --quiet "$have" "$PIN" -- engine 2>/dev/null; then
  echo "$W2_DIR is at $have with other engine code than the pin $PIN: build at the pin first" >&2; exit 1
fi
if [ "$DRY_RUN" != 1 ]; then
  [ -e "$OUT" ] && { echo "$OUT exists: a round's candidate is launched once (new NAME, or another candidate)" >&2; exit 1; }
  mkdir -p "$OUT"
  cp "$CAND" "$OUT/candidate.json"
fi

W2=(POOL=arena/pools/wave2.json SETUP=pool OPENINGS=bughouse RULES=v3 MEASURE=1 MEASURE_NODES=800
    "WARM_KINDS=upstream wave2")
ids=()
for k in $(seq "$PARTS"); do
  dir="$OUT/part$k"
  job="tune_${ROUND}_${CNAME}_p$k"
  if [ "$DRY_RUN" = 1 ]; then
    printf 'NAME=%q ENGINE_REF=%s OPPONENT=upstream-it13 CONTENDER_KW=%q %sBASE_MS=%s PAIRS=%s SEED=%s WORKERS=%s %s sbatch --parsable %s-t %s -J %s cluster/eval.sbatch\n' \
      "${dir#runs/}" "$PIN" "$CKW" "$( [ "$OKW" = "{}" ] || printf 'OPPONENT_KW=%q ' "$OKW")" "$BASE_MS" \
      "$PAIRS_PER_PART" $((SEED_BASE + k)) "${WORKERS:-3}" "${W2[*]}" \
      "${SBATCH_DEPENDENCY:+--dependency=$SBATCH_DEPENDENCY }" "$TIME" "$job"
    continue
  fi
  mkdir -p "$dir"
  id=$(export NAME="${dir#runs/}" ENGINE_REF="$PIN" OPPONENT=upstream-it13 CONTENDER_KW="$CKW" BASE_MS="$BASE_MS" \
         PAIRS="$PAIRS_PER_PART" SEED=$((SEED_BASE + k)) WORKERS="${WORKERS:-3}" "${W2[@]}"
       [ "$OKW" = "{}" ] || export OPPONENT_KW="$OKW"
       sbatch --parsable ${SBATCH_DEPENDENCY:+--dependency=$SBATCH_DEPENDENCY} -t "$TIME" -J "$job" cluster/eval.sbatch)
  ids+=("$id")
  printf '%s\tpart%s\tseed %s\t%s\n' "$CNAME" "$k" $((SEED_BASE + k)) "$id" >> "$OUT/jobs.tsv"
done
if [ "$DRY_RUN" = 1 ]; then
  echo "# dry run: $CNAME of round $ROUND, $PARTS parts x $PAIRS_PER_PART pairs; nothing submitted"
else
  echo "$CNAME (round $ROUND): ${#ids[@]} jobs (${ids[*]}), engine pin $PIN, $OUT"
fi
