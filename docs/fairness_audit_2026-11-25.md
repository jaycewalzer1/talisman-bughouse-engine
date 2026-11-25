# Fairness audit of the match harness, 2026-11-25

Scope: arena/sim.py (event loop), arena/players/upstream.py (both the `upstream` and `wave2` kinds), arena/pool.py
(spec building, swap, ms_per_node measurement), cluster/tune_match.sh, cluster/tune_space.json, cluster/eval.sbatch,
cluster/launch_anchor_match.sh, cluster/launch_final_match.sh, and the two engines' node accounting
(third_party/upstream-hivemind at v3.0.3, third_party/wave2-hivemind at d5802ee = cluster/wave2_engine.ref).
I changed no code and committed nothing.

Data read: every `results.jsonl` under runs/study/tune/r1 (base 923 games, clock_k_160/220/440/1000 1200 each,
clock_k_55 843), runs/study/wave2c/r4/long (l_hybrid_120, l_nodur_120, l_same_120) and
runs/study/full/hybrid1/anchors/it2. I also read the per-event game logs of the first 200 games of base,
clock_k_160 and clock_k_55. The scripts were throwaway and are not in the repo.

"Confirmed" means I saw it in the code or the logs. "Suspected" means the code allows it but I did not measure it.

## Summary, ranked by how much each could move a match score

| # | Finding | Status | Favours | Size |
|---|---|---|---|---|
| 1 | Whoever is ahead on the clock gets to sit (in both engines' trees and under the arena's sit-off rule). A clock_k gap fixes who is ahead for the whole game. | Confirmed | Higher-k side, whichever engine it is | Explains the whole 0.5 to 0.99 jump (and 0.004 at k 55) |
| 2 | One joint search moves both seats; the second seat's move is charged 0 nodes. The sitting side gets this far more often. | Confirmed, symmetric | The side ahead on clock | Part of #1; 27% of the ahead side's moves |
| 3 | The wave2 controller guesses whether an opponent seat is sitting using wave2's own clock_k, not the opponent's. | Confirmed (code) | Unclear; distorts wave2 only when the k values differ | Unknown; a confound inside the k sweep |
| 4 | Measured ms/node: the first-measured arm reads ~20% slow. This changes nodes per move, not time per move. | Confirmed in anchors it2 and r4 long | The second-measured arm (the opponent) | Small: about 20% more nodes for the opponent |
| 5 | The mate provers' CPU wall time is never charged: fixed slices per search, outside the counted nodes. | Suspected | The side that runs more, smaller searches (higher k) | Unknown, probably small next to #1 |
| 6 | Moves are applied stale with respect to the other board. | Confirmed, symmetric | The faster thinker | Part of #1; this is how real bughouse works |

Checked and found fair: clock charging per kind, the node definition, overshoot, the lag and minimum-move rules,
ordering at equal timestamps, seat/colour/opening balance, swaps, seeds, the fixed charge in tune r1, and what each
side can see (no private information leaks). Details are in the last section.

**Bottom line:** the r1 clock_k sweep does not measure playing strength. It measures which team holds the clock lead,
and that is set almost entirely by the k ratio, because the opponent's time manager is fixed at 110. The k 55 row is
the mirror test: there the stock upstream engine scores 0.996 against wave2 using the same mechanism. Nothing in the
harness treats the two kinds differently in a way that explains the numbers.

## 1. The clock lead decides who may sit, and clock_k sets the lead for the whole game (confirmed)

Mechanism:
- Time per move does not depend on ms_per_node. The budget is `clock / clock_k / ms_per_node` nodes
  (arena/players/upstream.py:511-516), and the charge is `searched_nodes x ms_per_node` (upstream.py:553-559). So each
  search costs about `clock / clock_k` of simulated time, plus lag and queueing. The side with the higher k always
  spends less clock per move, and so it leads on the clock from move 1 to the end.
- Both engines let a team sit only when it holds a time advantage. Upstream v3.0.3 uses `time.can_sit(teamToPlay)`
  throughout the tree (third_party/upstream-hivemind/engine/src/search/searchthread.cc:60,172,378,755,911), keyed
  off `TimeLeadMs`, which the harness computes from the four clocks (upstream.py:156-162, 527). Wave2 runs with
  SitGate true (tune_space.json "base"), which is the same gate.
- The arena's sit-off rule ends a mutual sit in favour of the higher clock (sim.py:221-245, 334-340; RULES_V3
  sitoff_ms 5000). So the side behind must move, and the side ahead can wait for its partner's captures and then
  drop-mate.
- With 120 s clocks and ~90 s left unused, the side ahead never pays for its sits.

Evidence (results.jsonl, `clock_teams`; c = wave2 contender, o = upstream opponent):

| row | score | c share of turns ahead | c sits/game (s) | o sits/game | c nodes/move | o nodes/move |
|---|---|---|---|---|---|---|
| base (110 v 110) | 0.516 | 0.60 | 3.9 (6.9 s) | 3.7 | 19 896 | 19 914 |
| k 160 | 0.994 | 0.98 | 8.4 (19.9 s) | 1.3 (all behind) | 11 273 | 20 612 |
| k 220 | 0.988 | 0.98 | 9.1 (25.5 s) | 1.3 | 7 015 | 20 112 |
| k 440 | 0.979 | 0.98 | 9.7 (29.3 s) | 1.5 | 3 028 | 19 506 |
| k 1000 | 0.907 | 0.98 | 9.3 (32.2 s) | 1.4 | 1 192 | 19 209 |
| k 55 | 0.004 | 0.002 | 1.5 (behind) | 9.1 (40 s) | 35 164 | 13 113 |

At k 1000, wave2 searches 1 192 nodes a move against upstream's 19 209 and still scores 0.907. Search size barely
matters once the lead is fixed. At k 55 the roles reverse, and stock upstream wins 0.996 the same way. In base,
where the lead changes hands during the game, being ahead most of the game is worth only about 0.60.

Is this the harness treating the teams differently? No. The rule applies the same way to both kinds, as the k 55
mirror shows. But a score against a fixed-110 opponent then measures time-odds exploitation, not strength. The
opponent cannot adapt its k, and the binary sit privilege turns a small clock edge into a near-certain win.

Fixes:
1. Control run: stock upstream at k 160 vs stock upstream at 110 (and wave2 at 110 vs wave2 at 160). If the curve
   comes back for upstream against itself, finding #1 is settled.
2. Do not tune clock_k against a fixed-k opponent. Score it only in matches where both sides use the same k (a k
   change then changes nothing structural), or tune both teams' time managers together.
3. Report the clock-lead share (`clock_teams.turns_ahead`) next to every score. Treat any row with a lead share over
   0.9 as a time-odds result, not a strength result.
4. If strength is the target, compare at equal time per move: give both sides the same k and vary only nodes
   through ms_per_node or a fixed nodes budget. Or use a rule set where sitting is not a free privilege of the lead
   (for example, cap the total sit time).

## 2. A joint search moves both seats; the second seat's move is charged 0 nodes (confirmed, symmetric)

`UpstreamTeamController.decide` (upstream.py:540-545) caches the last joint search by `(fenA, fenB)`. When the
team's other seat decides at the same position, it takes its move from that search with `nodes = 0` and
`think = ready - now`. This happens when a landing wakes a sitting seat and hands the other board to the same team
at the same timestamp (sim.py:511-514): the first seat searches, and the second gets the move free. The rule is the
same for both kinds. Physically it is right, because one search really did produce both moves. But the team that
sits gets far more of these free moves:

| row (first 200 games) | wave2 moves with nodes 0 | upstream moves with nodes 0 |
|---|---|---|
| base | 570 of 11 116 (5%) | 591 of 11 112 (5%) |
| k 160 | 2 478 of 9 180 (27%) | 37 of 9 177 (0.4%) |
| k 55 | 16 of 8 783 | 3 421 of 8 787 (39%) |

This adds to #1 and is not a separate bias. Fix: none needed for fairness. If a cached move should be charged,
charge the second seat for the remaining search time (this is already done: it lands when the search ends).

## 3. The wave2 controller guesses opponent sitting from its own clock_k (confirmed in code, wave2 only)

`seat_decisions(..., clock_k=self.clock_k)` (upstream.py:760-767, called at 910) estimates an opponent seat's
expected think time as `clock / clock_k`, using wave2's own k. When the opponent runs k 110 and wave2 runs k 160,
the threshold `1.5 x clock / 160 ≈ clock / 107` is shorter than upstream's real `clock / 110` plus lag (0-150 ms)
plus queueing. So wave2 marks upstream seats as `SeatSitting 1, SeatElapsedMs 0` before they land. Wave2's tree then
applies the sit-off logic (the lower clock must move), which makes waiting look free to the team that is ahead.

Only wave2's own decisions are affected; the harness gives upstream nothing wrong. But the effect changes with k, so
inside the k sweep, part of the "k effect" is a change in what wave2's tree believes. The direction is unclear:
waiting more might help or hurt wave2. Fix: estimate the opponent's think time from the opponent's observed pace
(the seat EMA already tracked) or from a configured opponent k, never from our own.

## 4. Measured ms/node: cold first arm (confirmed in old runs; affects nodes, not time)

Measurement runs per part, contender first (arena/pool.py:581-628). In runs/study/full/hybrid1/anchors/it2, part1
read 0.0523 for the contender (it2 network) against 0.0403 for hybrid-it13, and part2 read 0.0503 against 0.0423.
Both used `repeats 3` with no `warm_ms`. So this run predates today's launcher default
(`MEASURE_WARM_MS=3000 MEASURE_REPEATS=10`, cluster/launch_anchor_match.sh:128-129, changed in 4ffe3f6). The two
arms also have different architecture keys, so part of the gap may be real. In r4 long the contender got ~12-20%
fewer nodes per move (16.5k vs 18.7-20.4k).

Because the budget divides by ms_per_node and the charge multiplies by it, both sides still spend the same time
per move (1 016 vs 1 015 ms in the anchors; 1 023 vs 1 022 in r4). So this bias costs the first-measured arm about
20% of its nodes and does not touch the clock lead. Given #1 (1.2k nodes still wins), that is worth a few Elo, not
tenths of a point. Tune r1 is not affected: both arms used the fixed 0.0432 (seen in every r1 specs.jsonl).

Fixes: keep the warm defaults. Measure the arms in alternating order and average, or measure once per round, cache
the value, and pass the same number to all parts. Record warm_ms and repeats in every ms_per_node.json (already done
when warm is set).

## 5. Mate provers' wall time is not charged (suspected)

The printed `nodes` (the value the arena charges) is `searchInfo.get_nodes_searched()`, which counts tree visits only
(wave2 agent.cc:8014, upstream agent.cc:7738; searchinfo.h is identical in both). The root mate scans,
FrozenRace, selected-move certification and the preflight run in CPU wall-time slices next to the tree and are not
counted. Their time constants are fixed per search (search_params.h: preflight 5 ms, MATE_PROBE_FIRST_SLICE_MS 50,
FROZEN_RACE_MAX_MS 250, SELECTED_MOVE_CERT_MAX_MS 100). A proven mate prints `nodes 1` (wave2 agent.cc:7744-7746;
upstream 7469), so the move is charged min_move_ms plus lag. About 100 such moves per 200 games occur on each side.

The code and constants are identical in both engines, so this is not an advantage of one kind. But a fixed free
allowance per search is worth relatively more to the side that searches more often with smaller budgets (high k).
The ms/node measurement from startpos does not include it. Size not measured. Fix: charge measured wall time for the
mate work (or count prover nodes into the printed nodes at a calibrated rate). Or check its size by running a k row
with MateProbe, FrozenRace and CertifySelectedMove off on both sides.

## 6. Stale application across boards (confirmed, symmetric)

A move is searched at decision time and applied at `now + cost` if it is still legal (sim.py:416-438, 458-471). The
seat's own board cannot change while it thinks, because its opponent there is not on turn. Pockets only grow. So a
searched move is never made illegal, and it is never re-searched or discarded. It is played against the other
board's newer state. The same applies to both kinds. The faster thinker acts on fresher information, which is part of
why time matters so much in #1. This is real-time bughouse working as intended, not a harness bias.

## Checked and found fair

- **Clock charging per kind.** Both kinds go through the same `UpstreamTeamController.decide`: the charge is the
  reported searched nodes times that team's ms_per_node (upstream.py:553-559), plus serial queueing (558), then the
  simulator's min_move_ms 100 and SplitMix lag 0-150 (sim.py:417-420), with one lag stream for both. In the logs,
  `think_ms - nodes x 0.0432` has the same distribution for the leading side in either direction (p50 80-89 ms,
  which is lag) and a larger one for the trailing side (queueing), mirrored between k 160 and k 55. Overshoot of
  `go nodes` is ~0.2% (25 253 requested, 25 290 charged) and is charged.
- **Node definition.** searchinfo.h is identical in both engines, `increment_nodes` sits in shared searchthread
  code, and the `go nodes` stop condition and printed nodes match (agent.cc 5983/8486 vs 5979/8195). Tree reuse is
  off in all rows (no continuity flags; `reset_tree` True; `ucinewgame` each search). The permanent brain is stopped
  right after bestmove (upstream.py:347). The search behind a pass is charged like any search. Hybrid sits are
  ordinary waits that the clock runs through.
- **nodes=0 think events.** All are cache hits (#2), not missing counts.
- **Sit, pass, repetition and flag rules.** The arena enforces them per board and seat with no reference to
  `kind` (sim.py). The pass wait lengths differ per controller (upstream: team pace EMA; wave2: min(8, 1.5) x
  budget), but both are woken by any landing, and this is engine design, not a harness rule.
- **Information.** `Observation` (sim.py:350-366) is the same for every seat and holds only public state. Wave2 also
  receives SeatClocks, paces, elapsed, sitting and arrivals, but all of them are derived from that public
  observation (upstream.py:906-930). Stock upstream's dialect cannot take them. No seat sees another's pending
  decision, nodes or think time.
- **Equal timestamps.** The heap is FIFO on `(t, counter)`. At game start board A decides first, so T1 (A_white)
  goes first. T1 scores 0.502 (base), 0.512 (l_same), 0.505 (k 160): no seat bias.
- **Pairing.** Each opening is played twice with the teams swapped (pool.py:250-278): r1 rows have exactly 600 of
  1200 games with the contender as T1. Parts use seed SEED_BASE + k, and every candidate in a round gets the same
  seeds. l_same_120 (wave2 vs itself) scored 0.503.
- **Fixed charge in tune r1.** tune_round puts `ms_per_node 0.0432` into both contender_kw and opponent_kw, so
  MEASURE=1 measures nothing (`_wants_measure`, pool.py:553). Confirmed in specs.jsonl.
- **Final-match launcher.** It uses the pool's k 110 for every engine (final_match_engines.example.json sets no
  clock_k), so #1 does not apply unless someone carries a tuned k into a final row. It has the same warm
  measurement defaults as the anchor launcher. #4 (order) and #5 still apply.

## Not checked

- The CUDA/TensorRT batch behaviour on PACE, in case batch fill differs per engine under shared-GPU load. I only
  read the measured values.
- The wave2 hybrid sit branch in detail (hybrid.h). I only confirmed that its visits go through the same tree
  counters.
- No code was run against the engines. The size of #5 is unmeasured.
