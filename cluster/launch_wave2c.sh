#!/bin/bash
# Launches the wave 2c readiness round on PACE from the repo root, after pulling custom-engine. After it,
# `.venv/bin/python cluster/decide_full_run.py` says GO or NO-GO for the full run and prints its launch command.
#   w2c_build (cluster/build.sh)  -> (a) w2c_gpu: GPU engine checks (cluster/wave2c_gpu_check.sbatch)
#                                       -> (c) w2c_bench_nodur, w2c_bench_cont: self-play rate R, narrow and wide
#                                       -> (d) w2c_smoke_nodur, w2c_smoke_cont: one wave2_iterate iteration per config,
#                                              narrow and wide (AB=1), self-play -> train -> gate
#                                 -> (b) the 11 wave2c rows (cluster/launch_checks.sh wave2c)
#                                 -> (e) w2c_watchdog (cluster/wave2c_watchdog.sh): cancels the round if the build or
#                                        the GPU check fails
# Everything goes to runs/wave2c/<ROUND> (ROUND: the first free r1, r2, ...; runs/wave2c/LATEST names it). A round
# folder is never reused: eval resumes whatever partial games it finds, so an old round's rows would mix in.
#   bash cluster/launch_wave2c.sh
# Settings: ROUND (first free rN), SMOKE_GAMES (200 self-play games per smoke arm), BENCH_GAMES (150 per width),
# ALLOW_BUSY (0; 1 launches while wave 2 jobs of an earlier round (w2_*, ab*_w*) are queued: they run from the same
# wave2-hivemind checkout, which this round moves to the pin and rebuilds), NO_PULL (0; 1 skips the git pull, for dry
# runs), W2C_RUNS (runs/wave2c, for dry runs). Read docs/wave2_tests.md "Wave 2c readiness round" before the numbers.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
W2C_RUNS="${W2C_RUNS:-runs/wave2c}"
SMOKE_GAMES="${SMOKE_GAMES:-200}"
BENCH_GAMES="${BENCH_GAMES:-150}"
W2=third_party/wave2-hivemind
W2_REPO="${WAVE2_REPO:-https://github.com/jaycewalzer1/hivemind}"

if [ "${NO_PULL:-0}" != 1 ]; then
  git fetch -q origin
  git checkout -q custom-engine
  git pull -q --ff-only
  git submodule update -q --init third_party/hivemind
fi
echo "code: $(git log --oneline -1)"

# The wave2-hivemind checkout must be at the pin. The build would move it there too, but the jobs of an earlier round
# run from this checkout, so it is moved here, before anything is submitted, and only when it is clean.
pin=$(cat cluster/wave2_engine.ref)
busy=$(squeue -h -u "$USER" -o "%j" | grep -E '^(w2_|ab[0-9]*_w[01]$)' | sort -u | tr '\n' ' ' || true)
if [ -n "$busy" ] && [ "${ALLOW_BUSY:-0}" != 1 ]; then
  echo "wave 2 jobs still queued: $busy" >&2
  echo "They run from $W2, which this round moves to $pin and rebuilds. Wait for them (bash cluster/status_wave2c.sh)," >&2
  echo "or ALLOW_BUSY=1 bash cluster/launch_wave2c.sh (with every wave 2c option off the new engine plays as the old:" >&2
  echo "spec v5 parity 0.4/0.6)." >&2
  exit 1
fi
if [ -d "$W2/.git" ] && [ "$(git -C "$W2" rev-parse HEAD)" != "$pin" ]; then
  if [ -n "$(git -C "$W2" status --porcelain --untracked-files=no)" ]; then
    echo "$W2 has uncommitted changes and is not at the pin $pin: commit or move them first" >&2
    exit 1
  fi
  git -C "$W2" cat-file -e "$pin^{commit}" 2>/dev/null || git -C "$W2" fetch -q "$W2_REPO" talisman/wave2
  git -C "$W2" -c advice.detachedHead=false checkout -q --detach "$pin"
fi
if [ -d "$W2/.git" ]; then
  have=$(git -C "$W2" rev-parse HEAD)
  [ "$have" = "$pin" ] || { echo "$W2 is at $have, not the pin $pin" >&2; exit 1; }
  echo "wave2 engine: $have = cluster/wave2_engine.ref"
else
  echo "wave2 engine: no checkout yet; the build clones it at the pin $pin"
fi

if [ -z "${ROUND:-}" ]; then
  n=1; while [ -e "$W2C_RUNS/r$n" ]; do n=$((n + 1)); done; ROUND="r$n"
fi
OUT="$W2C_RUNS/$ROUND"
if [ -e "$OUT" ]; then echo "$OUT exists: rounds are never reused, pick another ROUND" >&2; exit 1; fi
mkdir -p "$OUT"
echo "$ROUND" > "$W2C_RUNS/LATEST"
echo "round: $OUT"

B=$(sbatch --parsable -A paceship-bughouse -q inferno -N1 -n6 --mem=64G --gres=gpu:rtx_6000:1 -t 4:00:00 \
  -J w2c_build -o build_w2c_%j.log \
  --wrap "cd $PWD && CMAKE_BUILD_PARALLEL_LEVEL=2 bash cluster/build.sh && date > $OUT/build.ok")

# (a) GPU engine checks
G=$(OUT="$OUT" sbatch --parsable --dependency="afterok:$B" cluster/wave2c_gpu_check.sbatch)

# (b) the 11 rows, each one 8 h job, into runs/wave2c/<ROUND>/rows (launch_checks passes SBATCH_DEPENDENCY on)
R=$(SBATCH_DEPENDENCY="afterok:$B" W2C_PREFIX="${OUT#runs/}/rows" bash cluster/launch_checks.sh wave2c \
  | sed -n 's/^Submitted batch job \([0-9]*\).*/\1/p' | tr '\n' ' ')

# (c) self-play rate R of the two candidate configs, (d) one training iteration of each (both widths)
C=(); S=()
for config in nodur cont; do
  C+=("$(OUT="$OUT" CONFIG="$config" BENCH_GAMES="$BENCH_GAMES" \
    sbatch --parsable -J "w2c_bench_$config" --dependency="afterok:$G" cluster/wave2c_bench.sbatch)")
  S+=("$(RUNS_DIR="$OUT" RUN="smoke_$config" AB=1 ITERS=1 GAMES_PER_ITER="$SMOKE_GAMES" \
    SEGMENT_GAMES=$((SMOKE_GAMES / 2)) VAL_GAMES=24 MATCH_GAMES=60 \
    WAVE2_EXTRA="$(.venv/bin/python cluster/wave2c_configs.py "$config")" \
    sbatch --parsable -J "w2c_smoke_$config" -t 8:00:00 --dependency="afterok:$G" cluster/wave2_iterate.sbatch)")
done

# (e) the watchdog starts with the build's end, whatever its outcome
W=$(OUT="$OUT" BUILD="$B" GPU="$G" sbatch --parsable --dependency="afterany:$B" cluster/wave2c_watchdog.sh)

printf 'build\t%s\ngpu\t%s\nrows\t%s\nbench\t%s\nsmoke\t%s\nwatchdog\t%s\n' \
  "$B" "$G" "$R" "${C[*]}" "${S[*]}" "$W" > "$OUT/jobs.tsv"
echo "build $B   gpu check $G   rows $R"
echo "bench ${C[*]}   smoke ${S[*]}   watchdog $W"
echo "GPU-hours: at most 118 by the time limits (build 4, gpu check 4, bench 2 x 3, smoke 2 x 8, rows 11 x 8);"
echo "about 100 expected (the rows run to their 8 h limit, the build, checks, bench and smokes end sooner)"
echo "status: bash cluster/status_wave2c.sh    decide: .venv/bin/python cluster/decide_full_run.py"
