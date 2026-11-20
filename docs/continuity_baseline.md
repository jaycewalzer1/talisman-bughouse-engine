# Continuity baseline (Step 0, measurement only)

This is the baseline that the three continuity fixes will be judged against: team tree reuse, joint team actions in
the durative tree, and plan carry. The engine code was not changed. Measured on 2026-11-20 on the 36 GB M4 Max. The
wave 2 checkout is `third_party/wave2-hivemind` at `261081b` (branch `talisman/wave2`, clean working tree). Network:
twin-s-clock it13 (`engine/models/twin-s-clock-fp32.onnx`, HF `aminwoo/bughouse-twin-s-clock@edaa246`) on ONNX
Runtime, for both wave 2 and upstream. Script: `tools/continuity_probe.py`. Raw output: `runs/continuity_baseline/`.

Modes:
- **on**: the headline wave 2 flags. DurativeTree on, SitGate off. This is the `ref_120` row.
- **off**: DurativeTree off, SitGate on. This is the `nodur` row.

## Summary

1. **The engine never reuses a tree under the seat clock, in either mode, and stock upstream it13 doesn't either.**
   Across 522 wave 2 searches and 522 upstream searches (with `reuse_tree` on, so no `ucinewgame`), zero searches
   started from a reused subtree. The retained-candidate lookup is keyed by `search_hash_key(team, searchTime_)`,
   which includes the clock state. The real clocks never match a node of the old tree exactly, so the lookup always
   misses. Turning DurativeTree off does not help.
2. **There is something to reuse when the clocks are ignored.** A temporary probe print (below) looked for the same
   board (same team and the same board/pocket signature, any clocks) among the retained candidates:

   | Mode | Same board found | Its visits (when found) | Its visits / searched nodes (when found) |
   |---|---|---|---|
   | on | 104 of 316 (32.9%) | 583 | about 28% |
   | off | 44 of 206 (21.4%) | 286 | about 14% |

   A clock-tolerant reuse (fix 1) could therefore start about a third of durative decisions with roughly 580 visits
   already in place. Each search is about 2050 nodes. This is a ceiling. The probe matches only the exact successor
   boards that the old tree retained, and it does not check that the re-timed subtree is still valid.
3. **Node scaling: the gap between on and off does not close with more nodes.** The table below uses 40 middlegame
   positions and compares each search's joint action with the deepest reference that is independent of both modes,
   upstream it13 at 64k nodes:

   | Nodes | on agrees with it13@64k | off agrees with it13@64k |
   |---|---|---|
   | 1k | 23/40 (57%) | 23/40 (57%) |
   | 4k | 20/40 (50%) | 25/40 (62%) |
   | 16k | 22/40 (55%) | 27/40 (68%) |

   With more nodes, off moves toward the it13 reference (57% to 68%) and on stays flat (57%, 50%, 55%). On and off
   also drift apart: they choose the same joint action 29/40 times at 1k, 22/40 at 4k and 21/40 at 16k. At 16k the
   durative root passes on an on-turn board in 7/40 positions, against 4/40 for off. Repeating a search gives the same
   action 39/40 (on) and 40/40 (off) times at 4k, so these differences are not search noise. **This rules out "the
   durative tree just needs more nodes".** At the budgets we can afford, the durative tree converges to different
   moves. That is consistent with diagnosis (b): our two seats act at separate nodes, so a coordinated plan sits much
   deeper in the tree.

## 1. Reuse hit rate

Setup: local arena games with `arena/pools/wave2.json`, wave 2 contender vs `upstream-it13`, centralized teams,
clock_k 110, serial, rules v3, bughouse openings, 60 s base clocks (shorter than the PACE round's 120 s, to keep each
call short), and `reuse_tree: true` for both arms. Every search of both engines is logged with the options it was
sent and the engine output (`searches.jsonl`). A reused start is detected from the engine's verbose line
`info string Tree reuse: N visits recovered`, which UCI play prints for every non-background search.

| Run | Arm | Searches | Started from a reused subtree | Mean reused visits | Mean nodes searched |
|---|---|---|---|---|---|
| reuse_on (seed 0, 3 games) | wave2 | 200 | 0 (0.0%) | 0.0 | 2047 |
| reuse_on (seed 0) | upstream it13 | 200 | 0 (0.0%) | 0.0 | 1966 |
| reuse_on_s7 (seed 7, 2 games) | wave2 | 116 | 0 (0.0%) | 0.0 | 2061 |
| reuse_on_s7 | upstream it13 | 116 | 0 (0.0%) | 0.0 | 2087 |
| reuse_off (seed 0, 4 games) | wave2 | 206 | 0 (0.0%) | 0.0 | 2076 |
| reuse_off | upstream it13 | 206 | 0 (0.0%) | 0.0 | 2095 |

The probe build only covers the wave 2 arm. It prints the same board among the retained candidates with the clocks
ignored, which is the ceiling of a clock-tolerant reuse:

| Run | Searches | Mean candidates indexed | Board found (clocks ignored) | Mean visits of that subtree (all searches) | (when found) | Exact-hash hits |
|---|---|---|---|---|---|---|
| reuse_on | 200 | 48 | 71 (35.5%) | 220.4 | 620.9 | 0 |
| reuse_on_s7 | 116 | 50 | 33 (28.4%) | 141.8 | 498.5 | 0 |
| reuse_off | 206 | 120 | 44 (21.4%) | 61.0 | 285.6 | 0 |

Notes:
- 10 of the 522 wave 2 searches were the first search of a fresh process and had no candidates. They are counted
  in the denominators.
- `arena/players/upstream.py` says that "upstream reuses ~13% of visits". That figure does not hold under the seat
  clock and arena timing. Here upstream reused nothing either.
- Game results (5 games on, 4 games off) are far too few to say anything about strength. Wave 2 won 1/5 on and 2/4
  off. All 9 games ended in checkmate.
- History: the first attempt, `reuse --mode on --pairs 4` and `--mode off` in one shell call, was killed (SIGKILL,
  exit 137, cause unknown) after 3 games. Those 3 games are kept as `reuse_on`. The other runs were each their own
  shorter call, with free memory checked first (about 80% free).

### The probe print (temporary, not committed)

This patch adds the print to `Agent::try_reuse_tree` in `engine/src/search/agent.cc`. To rebuild the probe binary:
apply the patch, configure a scratch build dir the same way as `build-ninja` (`ninja-release` preset, ONNX Runtime,
4 search workers), build with `-j 2`, then revert the source (`git checkout engine/src/search/agent.cc`). The binary
used here was `/private/tmp/claude-501/-Users-jaycewalzer/87135253-5596-42e6-a8c3-73fada075e52/scratchpad/build-probe/hivemind.bin`
(scratch, release build of `261081b` plus this patch). Without the patch, the script still measures the real hit
rate, and the probe table is simply left out.

```diff
@@ Agent::try_reuse_tree, after the exact-hash lookup, before discard(rootNode, ...)
+    // TEMPORARY continuity probe (not committed).
+    if (!background) {
+        int sigMatches = 0;
+        int sigBestVisits = 0;
+        for (const auto& [h, c] : nextRootCandidates_) {
+            if (c.node && c.node->get_team_to_play() == teamSide
+                && c.signature == signature) {
+                ++sigMatches;
+                sigBestVisits = std::max(sigBestVisits, c.node->get_visits());
+            }
+        }
+        std::cout << "info string ReuseProbe candidates " << nextRootCandidates_.size()
+                  << " sigmatch " << sigMatches << " sigvisits " << sigBestVisits
+                  << " hashhit " << (reused ? 1 : 0)
+                  << " hashvisits " << (reused ? reused->get_visits() : 0) << std::endl;
+    }
```

## 2. Node scaling

Positions: 40 wave 2 decisions sampled (seed 0) from the arena logs above, out of 303 candidates. They are
middlegame positions: the lower of the two boards' fullmove numbers is 8 to 30 (median 17), each board has at least
16 men, and the decision's |q| is 0.9 or less. 13 come from `reuse_on`, 7 from `reuse_on_s7` and 20 from
`reuse_off`. 16 are for T1 and 24 for T2. Each position carries the exact options that the arena sent with it: team,
TimeLeadMs, the team/opponent/root move times, and SeatClocksMs, SeatPaceMs, SeatElapsedMs, SeatSitting and
SeatArrivalsMs. The positions are in `runs/continuity_baseline/positions.jsonl`.

Each search is `go nodes N` on a fresh tree (`ucinewgame`), using `build-sp1`, which has 1 search worker and gives
deterministic results up to the 39/40 noise floor. The it13 reference is stock upstream (`build-ninja`, 4 workers,
stock options, no seat timing because upstream does not have those options), with the same team, lead and move
times. "Joint" means that both boards' moves are equal. "Per board" is the mean agreement over the boards on which
our team is on turn. A pass counts as a move.

Reference: upstream it13 at 64k nodes. This is the main table.

| Nodes | durative on: joint | on: per board | durative off: joint | off: per board | n |
|---|---|---|---|---|---|
| 1000 | 23/40 (57%) | 64% | 23/40 (57%) | 65% | 40 |
| 4000 | 20/40 (50%) | 59% | 25/40 (62%) | 69% | 40 |
| 16000 | 22/40 (55%) | 61% | 27/40 (68%) | 70% | 40 |

For scale, it13 at 16k agrees with it13 at 64k on 28/40 positions (70%). So off at 16k (27/40) is about as close to
the reference as it13's own 16k search, and on at 16k (22/40) is clearly further away.

Reference: durative off at 16k. Off at 16k scores 100% here trivially, because it is the reference.

| Nodes | durative on: joint | on: per board | durative off: joint | off: per board |
|---|---|---|---|---|
| 1000 | 24/40 (60%) | 64% | 28/40 (70%) | 75% |
| 4000 | 21/40 (52%) | 57% | 30/40 (75%) | 80% |
| 16000 | 21/40 (52%) | 57% | 40/40 (100%) | 100% |

Reference: durative on at 16k. On at 16k scores 100% here trivially.

| Nodes | durative on: joint | on: per board | durative off: joint | off: per board |
|---|---|---|---|---|
| 1000 | 29/40 (72%) | 78% | 24/40 (60%) | 64% |
| 4000 | 31/40 (78%) | 85% | 22/40 (55%) | 57% |
| 16000 | 40/40 (100%) | 100% | 21/40 (52%) | 57% |

On vs off at the same budget, and root passes on a board where our team is on turn:

| Nodes | on = off (joint) | on: passes | off: passes |
|---|---|---|---|
| 1000 | 29/40 | 4 | 3 |
| 4000 | 22/40 | 4 | 4 |
| 16000 | 21/40 | 7 | 4 |

Noise floor. The same search was rerun with `--rep 1`:

| Mode | Nodes | same joint action |
|---|---|---|
| on | 4000 | 39/40 |
| off | 4000 | 40/40 |

Mean root q over the 40 positions:

| Nodes | on | off | it13 at 64k |
|---|---|---|---|
| 1k | -0.083 | -0.127 | -0.128 |
| 4k | -0.091 | -0.131 | |
| 16k | -0.116 | -0.137 | |

The durative root starts more optimistic and drifts down as it searches. Off stays level and close to it13.

Wall time per searched node on `build-sp1`: on 0.51 / 0.51 / 0.60 ms at 1k / 4k / 16k, off 0.52 at every budget.
The durative tree slows down per node at 16k.

What was skipped: wave 2 at 64k nodes (on and off). On `build-sp1` this costs about 0.5 to 0.6 ms per node, which is
about 22 min per mode for 40 positions and would have exceeded the run's time budget. The deep reference is
therefore it13 at 64k plus each wave 2 mode at 16k. Running `scaling --nodes 64000` in four `--ids` slices (10
positions each, about 6 min per slice) would add it.

## Exact commands

Run from the main repo, in its `.venv`. Run one command at a time, with no other heavy jobs running.

```sh
PROBE=/private/tmp/claude-501/-Users-jaycewalzer/87135253-5596-42e6-a8c3-73fada075e52/scratchpad/build-probe/hivemind.bin
# 1. reuse (about 50 s per game; the probe binary is optional, without it the probe table is omitted)
WAVE2_BINARY=$PROBE .venv/bin/python tools/continuity_probe.py reuse --mode on  --pairs 4 --base-ms 60000 --out runs/continuity_baseline/reuse_on   # killed after 3 games, kept
WAVE2_BINARY=$PROBE .venv/bin/python tools/continuity_probe.py reuse --mode on  --pairs 1 --seed 7 --base-ms 60000 --out runs/continuity_baseline/reuse_on_s7   # 105 s
WAVE2_BINARY=$PROBE .venv/bin/python tools/continuity_probe.py reuse --mode off --pairs 2 --base-ms 60000 --out runs/continuity_baseline/reuse_off  # 199 s
# 2. positions
.venv/bin/python tools/continuity_probe.py positions --runs runs/continuity_baseline/reuse_on runs/continuity_baseline/reuse_on_s7 runs/continuity_baseline/reuse_off --n 40 --out runs/continuity_baseline/positions.jsonl
# 3. scaling (reruns resume: finished (id, mode, nodes, rep) rows are skipped)
S="--positions runs/continuity_baseline/positions.jsonl --out runs/continuity_baseline/scaling.jsonl"
.venv/bin/python tools/continuity_probe.py scaling $S --modes on off --nodes 1000 4000      # 3.5 min
.venv/bin/python tools/continuity_probe.py scaling $S --modes on  --nodes 16000           # 6.5 min
.venv/bin/python tools/continuity_probe.py scaling $S --modes off --nodes 16000           # 5.5 min
.venv/bin/python tools/continuity_probe.py scaling $S --modes it13 --nodes 16000 64000    # about 11 min (run it as two calls to stay under 10 min)
.venv/bin/python tools/continuity_probe.py scaling $S --modes on off --nodes 4000 --rep 1 # 3 min, noise floor
# tables
.venv/bin/python tools/continuity_probe.py summary --reuse runs/continuity_baseline/reuse_on runs/continuity_baseline/reuse_on_s7 runs/continuity_baseline/reuse_off \
    --positions runs/continuity_baseline/positions.jsonl --results runs/continuity_baseline/scaling.jsonl
```

To judge a fix with this baseline: rerun the reuse step with the fix's UCI option set (pass it through
`--contender-kw` in `cmd_reuse`, or add a mode to `MODES`) and compare the hit rate and the reused visits against
the probe ceiling above. Then rerun the scaling step on the same `positions.jsonl` with the fix's mode. A fix
should bring durative agreement with it13 at 64k up toward the off column, and it should make the on column rise
with nodes.
