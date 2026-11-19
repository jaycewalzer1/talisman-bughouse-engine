#!/bin/bash
# Launches the whole wave 2 round on PACE from the repo root, after pulling custom-engine:
#   build_w2 (cluster/build.sh) -> the wave 2 test round (launch_checks.sh wave2, 11 rows)
#                               -> the medium A/B training run (RUN, 2 arms x CHAIN chained 16 h jobs)
#                               -> wave2_watchdog.sh, which cancels the round on failure
# Every job waits for the build; the watchdog starts whatever the build's outcome and cancels everything if it failed.
#   bash cluster/launch_wave2.sh
#   RUN=ab2 CHAIN=2 GAMES_PER_ITER=5000 bash cluster/launch_wave2.sh
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
RUN="${RUN:-ab1}"
CHAIN="${CHAIN:-3}"
GAMES_PER_ITER="${GAMES_PER_ITER:-10000}"

git fetch -q origin
git checkout -q custom-engine
git pull -q --ff-only
git submodule update -q --init third_party/hivemind
echo "code: $(git log --oneline -1)"

B=$(sbatch --parsable -A paceship-bughouse -q inferno -N1 -n6 --mem=64G --gres=gpu:rtx_6000:1 -t 4:00:00 \
  -J build_w2 -o build_w2_%j.log \
  --wrap "cd $PWD && CMAKE_BUILD_PARALLEL_LEVEL=2 bash cluster/build.sh")
echo "build: $B"

SBATCH_DEPENDENCY="afterok:$B" bash cluster/launch_checks.sh wave2 >/dev/null

for arm in 0 1; do
  for _ in $(seq "$CHAIN"); do
    RUN="$RUN" AB=1 WIDE="$arm" ITERS=1 GAMES_PER_ITER="$GAMES_PER_ITER" \
      sbatch --parsable -J "${RUN}_w$arm" --dependency="afterok:$B,singleton" cluster/wave2_iterate.sbatch >/dev/null
  done
done

BUILD="$B" RUN="$RUN" sbatch --parsable --dependency="afterany:$B" cluster/wave2_watchdog.sh >/dev/null

squeue -u "$USER" -o "%.9i %.22j %.3t %.24E %R"
echo "build log: build_w2_$B.log    watchdog log: runs/wave2/watchdog.log"
