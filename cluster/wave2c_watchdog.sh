#!/bin/bash
#SBATCH -J w2c_watchdog -A paceship-bughouse -q inferno
#SBATCH -N1 -n1 --mem=2G -t 24:00:00
#SBATCH -o w2c_watchdog_%j.log
# Wave 2c readiness round, job (e): stops the round on failure, so GPU hours are not spent past a bad build. CPU only.
#   build    BUILD (cluster/build.sh) did not complete                      -> cancel every w2c_* job
#   gpu      GPU (cluster/wave2c_gpu_check.sbatch) ended other than COMPLETED -> cancel every w2c_* job
#   errors   a row with errors in more than 5% of its games (at least 100 games) -> cancel that row; a long-test row
#            (OUT/long/<row>/part<k>, cluster/launch_long_test.sh) is read merged over its parts -> cancel every part
# Every 15 minutes while any w2c_* job is left. The round's other readings (control, scores, smokes, R) are
# cluster/decide_full_run.py's, not the watchdog's. Every decision goes to OUT/watchdog.log. It resubmits itself
# before its 24 h limit while jobs remain.
#   OUT=runs/wave2c/<ROUND> BUILD=<job> GPU=<job> sbatch --dependency=afterany:<build> cluster/wave2c_watchdog.sh
# The long test's own copy (launch_long_test.sh submits it as w2c_watchdog_long): LONG=1 watches only the long-test
# parts (w2c_l_*) and lasts while they do; BUILD=none or GPU=none skips that check.
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(git rev-parse --show-toplevel)}"
OUT="${OUT:?set OUT, the round folder (runs/wave2c/<ROUND>)}"
BUILD="${BUILD:?set BUILD, the cluster/build.sh job id}"
GPU="${GPU:?set GPU, the cluster/wave2c_gpu_check.sbatch job id}"
LONG="${LONG:-0}"
WATCH='^w2c_'; [ "$LONG" = 1 ] && WATCH='^w2c_l_'
POLL_S="${POLL_S:-900}"
LIFE_S="${LIFE_S:-84600}"   # 23.5 h, then hand over to a fresh copy
PY=.venv/bin/python
LOG="$OUT/watchdog.log"
mkdir -p "$OUT"

say() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }
jobs_named() { squeue -h -u "$USER" -o "%j" | grep -E "$1" | grep -v '^w2c_watchdog' | sort -u; }
cancel() {  # regex of job names, reason
  local names; names=$(jobs_named "$1")
  [ -z "$names" ] && return 0
  say "CANCEL $(echo $names) : $2"
  for n in $names; do scancel -u "$USER" -n "$n"; done
}
state() { sacct -n -X -j "$1" -o State | head -1 | tr -d ' '; }

if [ "$BUILD" != none ]; then
  s=$(state "$BUILD")
  if [ "$s" != COMPLETED ]; then
    cancel "$WATCH" "build $BUILD ended $s"
    exit 0
  fi
fi
say "watching (build $BUILD, gpu check $GPU, round $OUT$([ "$LONG" = 1 ] && echo ', long-test parts only'))"

# Errors in more than 5% of at least 100 games, over every results.jsonl given: prints "games errors".
too_many_errors() {
  $PY - "$@" <<'EOF'
import json, sys
rows = []
for path in sys.argv[1:]:
    for line in open(path):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass
errors = sum("error" in r for r in rows)
print(len(rows), errors)
sys.exit(0 if len(rows) >= 100 and errors > 0.05 * len(rows) else 1)
EOF
}

start=$SECONDS
gpu_ok=0
[ "$GPU" = none ] && gpu_ok=1
while [ -n "$(jobs_named "$WATCH")" ]; do
  if [ $gpu_ok = 0 ]; then
    s=$(state "$GPU")
    case "$s" in
      COMPLETED) gpu_ok=1; say "gpu check $GPU COMPLETED: $(grep -o '"overall": "[A-Z]*"' "$OUT/gpu_check.json" 2>/dev/null)" ;;
      PENDING|RUNNING|REQUEUED|CONFIGURING|COMPLETING|"") ;;
      *) cancel '^w2c_' "gpu check $GPU ended $s (see $OUT/gpu_check.tsv)"; exit 0 ;;
    esac
  fi
  if [ "$LONG" != 1 ]; then
    for d in "$OUT"/rows/*/; do
      [ -f "$d/results.jsonl" ] || continue
      n=$(basename "$d")
      if r=$(too_many_errors "$d/results.jsonl"); then
        cancel "^w2c_$n\$" "row $n has errors: games errors = $r"
      fi
    done
  fi
  # the long test's rows, merged over their parts (each part is job w2c_<row>_p<k>)
  for d in "$OUT"/long/*/; do
    [ -d "$d" ] || continue
    n=$(basename "$d")
    files=("$d"part*/results.jsonl)
    [ -f "${files[0]}" ] || continue
    if r=$(too_many_errors "${files[@]}"); then
      cancel "^w2c_${n}_p[0-9]+\$" "long row $n has errors over its parts: games errors = $r"
    fi
  done
  if [ $((SECONDS - start)) -ge "$LIFE_S" ]; then
    say "handing over to a fresh watchdog"
    OUT=$OUT BUILD=$BUILD GPU=$GPU LONG=$LONG sbatch $([ "$LONG" = 1 ] && echo -J w2c_watchdog_long) \
      cluster/wave2c_watchdog.sh >/dev/null
    exit 0
  fi
  sleep "$POLL_S"
done
say "no $([ "$LONG" = 1 ] && echo long-test || echo 'wave 2c') jobs left; done. Next: .venv/bin/python cluster/decide_full_run.py"
