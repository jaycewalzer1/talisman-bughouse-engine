#!/bin/bash
# Finishes the ab2 A/B (narrow it13 vs wide, both nodur: --durative-tree false --sit-gate true) to two gated iterations
# per arm, so cluster/decide_full_run.py can choose the width. Run it after cluster/launch_wave2c.sh, from the repo root:
#   bash cluster/resume_ab2.sh
# State on 2026-11-21: narrow it1 trained but its gate was skipped (rl-loop took a staged rl-it1 with arms/C/last.pt for
# the finished pilot; fixed in wave2-hivemind 72e79eb), wide it1 gated 0.519 vs it13, wide it2 has its self-play and
# validation and needs training and the gate. So: narrow resumes it1's matches, then it2 (two chained 16 h jobs); wide
# resumes it2 (one job). Every job waits for the wave 2c round's build (w2c_build), which builds the pinned engine in the
# shared wave2-hivemind checkout; build_sp1.sh then rebuilds build-sp1 at the pin. About 25-48 GPU-hours.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

B=$(squeue -h -u "$USER" -n w2c_build -o "%i" | head -1)
if [ -z "$B" ]; then
  echo "no w2c_build job in the queue: run bash cluster/launch_wave2c.sh first (or BUILD=<job id> if it already finished)" >&2
  [ -n "${BUILD:-}" ] || exit 1
  B="$BUILD"
fi
DEP="afterok:$B,singleton"

# The skipped narrow job also wrote wave2_iterate's .done_loop_1 without an iteration.done behind it, which would skip
# it1 again: set any such marker aside (renamed, not deleted).
for arm in narrow wide; do
  d="runs/wave2/ab2/$arm"
  for m in "$d"/.done_loop_*; do
    [ -e "$m" ] || continue
    n=${m##*_}
    if [ ! -f "$d/runs/rl-it$n/stages/iteration.done" ]; then
      mv "$m" "$m.skipped-$(date +%Y%m%d%H%M)"; echo "set aside $m (rl-it$n has no iteration.done)"
    fi
  done
done
SETTINGS=(RUN=ab2 AB=1 GAMES_PER_ITER=10000 WAVE2_EXTRA="--durative-tree false --sit-gate true")

for _ in 1 2; do
  env "${SETTINGS[@]}" WIDE=0 ITERS=1 MAX_ITER=2 \
    sbatch --parsable -J ab2_w0 --dependency="$DEP" cluster/wave2_iterate.sbatch
done
env "${SETTINGS[@]}" WIDE=1 ITERS=1 MAX_ITER=2 \
  sbatch --parsable -J ab2_w1 --dependency="$DEP" cluster/wave2_iterate.sbatch

squeue -u "$USER" -n ab2_w0,ab2_w1 -o "%.9i %.10j %.3t %.28E"
echo "read: cat runs/wave2/ab2/results_ab.tsv   (or bash cluster/status_wave2c.sh)"
