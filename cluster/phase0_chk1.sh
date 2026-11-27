#!/bin/bash
# Phase 0 of the league plan (docs/plan_league_2026-11-25.md, spec v7 7.4), run on the Mac: copy the chk1 round
# (sym160: both teams at clock_k 160; rev_opp160) from PACE into the study tree, print its leaderboard, rebuild the
# study page's data.
#   bash cluster/phase0_chk1.sh
# 1. rsync ${REMOTE:-pace}:<PACE repo>/runs/tune/chk1/ -> runs/study/tune/chk1/, only the small files: results.jsonl,
#    game_*.jsonl, summary.json, jobs.tsv, candidate.json and candidates/*.json, round.json, specs.jsonl,
#    ms_per_node.json (no engines, ONNX or logs). SRC=<dir> copies from a local folder instead (a copy already here,
#    or tests). DRY_RUN=1 prints the rsync and stops.
# 2. python3 cluster/tune_status.py runs/study/tune/chk1 > runs/study/tune/chk1/status.txt, printed. chk1's rows may
#    have no round.json: tune_status.py then lists every <row>/part* folder itself (mode "unknown", baseline "base"
#    only when there is one), so nothing is written for it.
# 3. python3 tools/study_games.py runs/study (study.json and games_chk1__<row>.json for viewer/study.html).
# STUDY=<dir> replaces runs/study (tests).
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
REMOTE="${REMOTE:-pace}"
PACE_REPO="${PACE_REPO:-/storage/project/ps-bughouse-0/jwalzer3/talisman-bughouse-engine}"
STUDY="${STUDY:-runs/study}"
DST="$STUDY/tune/chk1"
SRC="${SRC:-$REMOTE:$PACE_REPO/runs/tune/chk1}"
PY="${PYTHON3:-python3}"

filters=(--include='*/' --include='results.jsonl' --include='game_*.jsonl' --include='summary.json'
         --include='jobs.tsv' --include='candidate.json' --include='candidates/*.json' --include='round.json'
         --include='specs.jsonl' --include='ms_per_node.json' --exclude='*')
cmd=(rsync -a --prune-empty-dirs "${filters[@]}" "${SRC%/}/" "$DST/")
if [ "${DRY_RUN:-0}" = 1 ]; then
  printf 'DRY_RUN'; printf ' %q' "${cmd[@]}"; echo
  exit 0
fi
mkdir -p "$DST"
echo "sync: ${SRC%/}/ -> $DST/"
"${cmd[@]}" || { echo "rsync from $SRC failed (PACE reachable? REMOTE=$REMOTE; SRC=<dir> for a local copy)" >&2; exit 1; }
rows=$(find "$DST" -mindepth 2 -maxdepth 2 -type d -name 'part*' | sed 's|/part[^/]*$||' | sort -u | xargs -n1 basename 2>/dev/null | tr '\n' ' ')
echo "rows: ${rows:-none}$( [ -f "$DST/round.json" ] || echo ' (no round.json: tune_status.py lists the row folders)')"

"$PY" cluster/tune_status.py "$DST" > "$DST/status.txt"
cat "$DST/status.txt"
echo "status: $DST/status.txt"

"$PY" tools/study_games.py "$STUDY"
echo "study: $STUDY/study.json (viewer: cd viewer && npx vite --port 5180, then /study.html)"
