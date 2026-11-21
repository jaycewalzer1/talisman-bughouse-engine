# Durative tree diagnosis (wave 2 at 261081b)

> Note: paths under `/private/tmp/claude-501/-Users-jaycewalzer/b46049ad-27b8-4fc1-93ab-85cae96e771f/scratchpad/diag/` are scratch and may be gone.

Date: 2026-11-20. Diagnosis only. Nothing in the main repo or in `third_party/wave2-hivemind` was edited, built, committed
or checked out. All work is in this scratch directory:
`/private/tmp/claude-501/-Users-jaycewalzer/b46049ad-27b8-4fc1-93ab-85cae96e771f/scratchpad/diag/`.

**Model and weights in every run:** twin-s-clock it13, `engine/models/twin-s-clock-fp32.onnx` (HF
`aminwoo/bughouse-twin-s-clock@edaa246`), ONNX Runtime (Core ML EP), 1 search worker. Reference moves (it13 at 64k)
come from the baseline's `runs/continuity_baseline/scaling.jsonl`. Nothing here was re-run for it13.

## Setup

- **Binaries.** `hivemind_sp1.bin` is a copy of `engine/build-sp1/hivemind.bin` (built 2026-11-19 13:09, 1 worker,
  right after 261081b). `build-base/hivemind.bin` is my own build of `git archive 261081b engine` (Release,
  FAST_BUILD, 1 worker, no ccache). It gives the same output as the baseline: on at 1k and 4k it matches the
  baseline's moves on 40/40 positions, with the same mean q (-0.083 / -0.091).
  `build-p1/hivemind.bin` is the same build plus the diagnostic patch `patch_p1_waitnode.diff` (D3 below).
- **Driver.** `drive.py` sends the same UCI setup as `tools/continuity_probe.py` / `arena/players/upstream.py`
  (`Ponder false`, `BackgroundSearch false`, `TimeModel clock`, Team, TimeLeadMs, Team/Opponent/RootMoveTimeMs,
  the seat options of each position, `ucinewgame`, `position fen A | B`, `go nodes N`).
  The modes are the arena's:
  `on` = DurativeTree true, SitGate false, FlagLoss true, InstantTransfer false, MateRaceByLanding false, PassCapK 8;
  `off` = the same with DurativeTree false and SitGate true.
- **Experiments.** `exp_scale.py OUT TAG BINARY MODE NODES... [--extra K=V,...]` runs the 40 positions in
  `runs/continuity_baseline/positions.jsonl`. `summ.py OUT` prints agreement with it13@64k. Raw rows:
  `res.jsonl`. Special `--extra` values: `{root}` = the position's `root_move_ms`; `RootMoveTimeMs=ownpace` =
  the mean `SeatPaceMs` of our own two seats; `SeatPaceMs=relpace` (confounded, see D2).
- **Cost.** About 0.5 to 1 ms per node, so one 4k sweep over the 40 positions takes 2 to 3 minutes.

## Summary

The durative machinery is **not** semantically broken when time is synchronous. With all four seats at the same pace
(`SeatPaceMs` = RootMoveTimeMs), elapsed 0, the gate on and InstantTransfer on, the durative tree picks the same joint
action as v3 at the same move times on **14 of 15** positions where our team is on turn on both boards (D0 below).
Its sign handling, backup, terminal verdicts and the root charge therefore agree with v3 whenever the order of
events matches v3's alternation. The gap comes from the parts that only show up when time is asynchronous:

1. **D1 (confirmed, the optimistic q level): our team is modelled about 15% faster than the opponent.** Below the root, every
   action of the root team costs `RootMoveTimeMs`, but every opponent action costs its observed pace. In the arena
   both teams really spend about 625 ms per move, while RootMoveTimeMs averages 546 ms. Charging our actions at our
   own observed pace removes the optimism: mean q goes from -0.091 to -0.145 at 4k (off -0.131, it13 -0.128). It
   does **not** improve move agreement: 23/19/23 of 40 at 1k/4k/16k against the baseline's 23/20/22, and 25/40
   the same as off at 16k against 21/40. It also does not remove the drift with nodes: q -0.125, -0.145, -0.168 at
   1k/4k/16k, so the level moves and the slope stays. The experiment charges the root action at our pace too, so it
   slightly over-corrects.
2. **D2 (confirmed, structural): asynchronous single-seat sequencing.** This is the spec v5 fix 2 diagnosis, and the
   experiment localises it. On the 25 positions where we are on turn on only one board, even the synchronised
   setting matches v3 only 13/25 times, because the opponent's seat on the other board acts alone first. Not a sign
   or value bug. The in-progress v5 fix 2 targets it.
3. **D3 (confirmed as a modelling error, impact not shown): forced-wait nodes are evaluated by the network from the
   waiting team's side.** The network gives the side to play a tempo bonus: V(our team to play) + V(their team to
   play) on the same boards averages **+0.26**. A wait node's leaf value is therefore biased toward the team that is
   *not* about to act. A patched build (`build-p1`) evaluates wait nodes from the acting team and negates the value.
   It is correct by construction, but it did not improve agreement (18/40 at 1k, 19/40 at 4k, against 23 and 20)
   and moved mean q by only 0.01. Real, but not shown to cost strength.
4. **D4 (confirmed): SeatElapsedMs means different things in the arena and in self-play.** It changes the root move
   on 2 of the 3 positions with a "sitting" opponent (including a pass into a sit-off on position 38).
5. **Python test (confirmed real, not environment-only): 1 of the 8 "bsdtar" failures is a real wave 2 test
   regression.** With a GNU-tar-compatible shim, 7 of the 8 pass, but
   `test_training_downloads_its_games_and_deletes_them_again` still fails, because its mock does not accept the
   `lr` / `trunk_lr` / `extra` keyword arguments that wave 2 added. Pristine v3.0.3 passes all 32 tests in those two
   files with the same shim.

**Not found:** a single semantic bug that explains "on does not converge with nodes". The control (D0) says the
remaining move-choice gap lives in the asynchronous single-seat structure (D2), the subject of v5 fix 2. D1 and D3 are
real value errors, but fixing them in a patched build or by options did not raise agreement on these 40 positions.
The q drift from 1k to 16k (about -0.04) persists in the only variant run at 16k (ownpace). The in-game arena eval
slide was not tested here. Treat the "non-convergence is a semantic value bug" hypothesis as **not reproduced**.

---

## D0. Control: the durative tree agrees with v3 when time is synchronous

Experiment (4000 nodes, `build-base`). In `on_syncgate`, every seat lands at `RootMoveTimeMs`, nobody has elapsed
time, and the gate and InstantTransfer are on. `off_root` is v3 with Team/OpponentMoveTimeMs = RootMoveTimeMs, so
both trees charge the same times.

```
python3 exp_scale.py res.jsonl on_syncgate build-base/hivemind.bin on 4000 \
  --extra "SeatPaceMs={root} {root} {root} {root},SeatElapsedMs=0 0 0 0,SeatSitting=0 0 0 0,SitGate=true,InstantTransfer=true"
python3 exp_scale.py res.jsonl off_root build-base/hivemind.bin off 4000 --extra "TeamMoveTimeMs={root},OpponentMoveTimeMs={root}"
```

| Positions | on_syncgate = off_root (joint action) | on_base = off_root |
|---|---|---|
| we are on turn on both boards (15) | **14/15** | 9/15 |
| we are on turn on one board (25) | 13/25 | 12/25 |

Mean q: on_syncgate -0.116, off_root -0.121. This refutes a backup-sign error, a team-alternation error, a wrong
terminal verdict, a wrong pass/sit legality and a transposition merge as the main cause: each of those would also
break the synchronous case. (`build-base` with the plain headline flags and synchronous paces, `on_sync`, gives 26/40
the same as off and q -0.121.)

## D1. Our team runs 15% faster than the opponent in the tree (root-team action cost)

**Where.** `engine/src/environment/time_state.h`:
- `TimeState::action_time_ms`, lines 698-701: `team_of_seat(seat) == rootTeam ? rootMoveMs : paceMs[seat]`;
- used for the ready time of every root-team seat that comes on turn below the root (`advance_action`, line 535) and
  for our pass cap (line 566);
- `TimeState::durative`, lines 189-193 and 207-214: `SeatPaceMs` is read for all four seats, but our own two are used
  only in the speed plane.

**What is wrong.** v3 (`TimeState::clocked`, `teamNextMoveTimeMs`) charges this search's budget only for our *next*
action, and the team's usual move time (EMA) after that. The durative tree charges the root budget for every one of
our actions in the line. The arena's opponent paces are clock drop per move made (`MoveTimeTracker`, which includes
sits and the serial brain's queueing), so the two teams are timed with different yardsticks. Over all 522 wave 2
searches of the baseline arena runs: our own seats' observed pace is **629 ms**, the opponent's **623 ms**, and
RootMoveTimeMs **546 ms**. In the tree we move 1.15x as often as the opponent, while the network's speed plane
(built from the paces) says the teams are equally fast. Over a 3-5 s horizon that gives our team free actions. This
matches the observed "root q starts optimistic and drifts down", and the arena eval sliding over the game is
consistent with it. Self-play is affected much less, because there the root budget is the drawn move cost and
nearly equals the realised pace.

**Repro (q).**
```
python3 exp_scale.py res.jsonl on_base    build-base/hivemind.bin on 1000 4000
python3 exp_scale.py res.jsonl on_ownpace build-base/hivemind.bin on 1000 4000 16000 --extra "RootMoveTimeMs=ownpace"
python3 summ.py res.jsonl
```

| Run | 1k agree / q | 4k agree / q | 16k agree / q |
|---|---|---|---|
| on (baseline) | 23/40, -0.083 | 20/40, -0.091 | 22/40, -0.116 (baseline) |
| on, RootMoveTimeMs = our own pace | 23/40, -0.125 | 19/40, -0.145 | 23/40, -0.168 |
| on, ownpace + wait-node patch (D3) | 22/40, -0.146 | 18/40, -0.157 | not run |
| off (baseline) | 23/40, -0.127 | 25/40, -0.131 | 27/40, -0.137 |
| it13 64k | | | -0.128 |

Split by position type at 4k, mean q: when we are on turn on both boards, base -0.065, ownpace -0.133, off -0.148,
it13 -0.142; on one board, base -0.107, ownpace -0.153, off -0.121, it13 -0.119.

**Caveat.** `ownpace` also charges the root action itself at our pace instead of the budget, so it is an
approximation of the fix.

**Proposed fix.** Below the root, give the root team's seats `paceMs[seat]` (the `SeatPaceMs` the arena already
sends), as v3 does with `TeamMoveTimeMs`, and keep `rootMoveMs` for the root action only. That is,
`action_time_ms` uses `rootMoveMs` only for the action charged at the root. The ready times of our seats coming on
turn and our pass cap use the pace. Alternatively, compute both teams' paces with the same estimator (thinking time
only, excluding sits) in `MoveTimeTracker`.

**v5.** Not addressed. v5 keeps "the root keeps its v4 meaning", and `action_time_ms` is only mentioned as a reuse
key (spec v5 line 167).

**Impact.** It removes the optimistic *level* of q (about +0.05), but not the downward drift with nodes. Agreement
with it13@64k barely moves (16k: 23/40 against 22/40). Rank: the largest value error found, but it is not shown to
explain the move-choice gap. Its effect on strength is unmeasured (it needs an arena or PACE row).

## D2. Asynchronous single-seat sequencing (where the move-choice gap lives)

**Where.** `TimeState::turn_at_node`, `time_state.h:849-876`: below the root only the earliest-ready seat acts. The
root's "early opponent landing" is at lines 456-469 and 482-541. `collect_batch` restricts the actions to the acting
boards at `searchthread.cc:914-920`.

**Evidence.**
- D0: synchronous timing reproduces v3 on 14/15 positions where both boards are ours. On the 25 single-board-root
  positions it reproduces v3 only 13/25. In those positions the opponent's seat on the other board lands "early"
  (its ready time is at most our budget). It acts alone at ready 0, and its other seat acts a pace later, as two
  nodes with a forced wait in between, where v3 has one joint reply.
- q is sensitive to small timing changes. One variant (`relpace`: opponent paces scaled by root/own) gave q -0.22,
  but it also changed the speed plane, because our own seats' paces were left unscaled. It is confounded and is not
  used as evidence.

**What is wrong.** This is not a code bug: it is the v4 design. Each seat's action is chosen at its own node, with
the opponent's and our other seat's actions interleaved by ready time. Coordinated two-board plans sit two to four
plies deeper (two action nodes plus forced waits per joint move of v3).

**Fix.** Spec v5 fix 2 (`JointPlans`) targets exactly this. Also skip or collapse wait nodes (v5 lists this as
open).

**v5.** Yes (fix 2). This report does not test the v5 code.

## D3. Wait-node leaf value comes from the wrong team (tempo bias)

**Where.** `engine/src/search/searchthread.cc`, `SearchThread::collect_batch`:
- lines 914-920 set `ctx.forcedWait`;
- line 970 calls `board_to_planes(board, ..., ctx.teamToPlay, ctx.time)`, the waiting team;
- in `SearchThread::process_batch`, lines 1122-1176 back up that network value for the wait node.

**What is wrong.** The network's value means "the team to play acts next on its on-turn boards" (v3 semantics). At a
forced-wait node the *other* team acts next, by construction. On the 40 baseline root positions, the network value
for our team plus the value for theirs, same boards and clocks, averages **+0.26**:
- +0.264 on the 25 positions where each team is on turn on one board, which is exactly the wait-node shape;
- +0.227 on the other 15;
- up to +0.95 on single positions.

So a wait node's first evaluation, often its only one at the frontier, overstates the waiting team by about a quarter
of the value range.

**Repro.**
```
python3 exp_tempo.py        # policy command with Team swapped, same boards and seat options; output in exp_tempo.json
```
For example, position 9: V(T1) 0.825, V(T2) -0.089, sum 0.736. Position 28: V(T2) 0.356, V(T1) 0.589, sum 0.946.

**Diagnostic patch.** `patch_p1_waitnode.diff`, built as `build-p1`, evaluates wait nodes from `~teamToPlay` and
negates the value and the WDL. Result: agreement 18/40 at 1k (base 23) and 19/40 at 4k (base 20); mean q -0.093 and
-0.103 (base -0.083, -0.091). Combined with D1's ownpace: 22/40 and 18/40, q -0.146 and -0.157.

**Status.** The modelling error is confirmed and the patch is the principled fix, but **no strength or agreement
gain was shown**, so do not rank it high. Cheaper and cleaner: skip the evaluation of wait nodes altogether and back
up the child's value (v5's open item "skip the network evaluation at wait nodes"). This also returns the node budget
lost to wait nodes.

**v5.** Not addressed. It is listed as an open optimisation and framed as speed, not value bias. Fix 2's
**forced** halves add more wait-like single-action nodes with the same bias.

## D4. SeatElapsedMs: arena and self-play differ (real)

**Where.**
- `arena/players/upstream.py` `seat_elapsed_ms` (line 591) sends `now - last landing on the board`.
- Self-play `SeatClockGame::elapsed_ms` (`engine/src/tools/seat_clock_game.h:468`) sends the time since the seat's
  *current decision* began. That is 0 while it sits (`decisionStartMs_` = the future wake time, line 342), and it is
  reset on wake (line 378).
- The tree reads it at `time_state.h:217-220`: `readyMs = max(1, pace - elapsed)`, and for a sitting seat
  `capReadyMs` = the same.

**What is wrong.** For an opponent that is sitting, or has just come back from a pass, the arena value includes the
whole wait. The tree then lands that seat at 1 ms, while self-play, and so training, sees it land a full pace later.
The `Wave2TeamController` docstring already says so. Combined with the arena's `seat_sitting` guess, the sit-off rule
can then make a root pass "force" the opponent to move.

**Repro.**
```
python3 exp_scale.py res.jsonl on_el0 build-base/hivemind.bin on 4000 --extra "SeatElapsedMs=0 0 0 0,SeatSitting=0 0 0 0"
```
The root move changes on positions 29 (B@d7 to e1g1; off plays e1g1) and 38 ((pass,pass) to (pass,c1e3)), the two
of the three positions the arena marked as sitting. q moves by more than 0.05 on 7, 13, 24 and 30. Overall agreement
19/40 against 20/40 and q -0.085, so the effect is small on this sample.

**Fix.** Send the self-play meaning from the arena: for an opponent seat, the time since its board was handed to it,
or since the last landing that could have woken its pass, capped at its pace. With `SeatSitting=1`, send elapsed 0
and let the pass model decide.

**v5.** Not addressed.

## Known open items

| Item | Verdict | Evidence |
|---|---|---|
| Arena `SeatElapsedMs` meaning differs from self-play | **Real** (D4) | Code at the lines above; the root move changes on 2/3 "sitting" positions |
| `racewin` ablation also disables the root mate provers | **Real confound, already documented** | `agent.cc:6133-6145` (`run_search` drops `enableRootMateSearch`, `enableMateProbe`, `certifySelectedMove` when `mate_race_by_landing()`); `docs/wave2_tests.md:108` and `docs/wave2_local_check.md:135` already say so. The row cannot isolate the landing rule until landing-aware provers exist. A cleaner ablation: a separate UCI switch for the provers, so `racewin` can keep them. |
| `arena/players/upstream.py:380` "upstream reuses ~13% of visits" | **Wrong under the seat clock** | Baseline: 0 of 522 upstream searches reused anything (the key includes the clock state). The comment should say the figure is from lead-only, flag-mode play, or be dropped. Comment only, no behaviour. |
| 8 wave2-hivemind Python test failures "are macOS bsdtar" | **7 are environment-only, 1 is a real regression** | See below |

### Python tests: details

I ran `tests/data/test_archive.py` and `tests/test_cli_rl_loop.py` from `git archive 261081b` (`pyexport/`), with
the repo's `.venv` Python and `PYTHONPATH=pyexport/src`. Nothing was built.
- Plain bsdtar: 8 failed, 24 passed. Every `test_archive` failure is `tar: Option --compare is not supported`.
- With `tarshim/tar` on PATH (emulates GNU `--compare` and drops `--format` / `-I` in extract mode, both of which
  bsdtar rejects; the restore path also fails on bsdtar with "Option --format is not permitted in mode -x"):
  **1 failed, 31 passed**. The remaining failure is
  `tests/test_cli_rl_loop.py::test_training_downloads_its_games_and_deletes_them_again`:
  `TypeError: <lambda>() got an unexpected keyword argument 'lr'`. The test's mock at
  `tests/test_cli_rl_loop.py:544` is `lambda init, new, val, old, *args:`, while `src/hivemind/cli/rl_loop.py:516-520`
  (wave 2, commit `1495b78`, plus `4d30b3c`) passes `lr=`, `trunk_lr=` and `extra=`.
- The same two files from v3.0.3 (`7a4dbdb`) with the shim: **32 passed**.

Fix: give the mock `**kwargs`. The archive code also needs GNU tar (`--compare`, `--format` on extract). On the Mac,
`gtar` is not installed, so locally they stay red.

## Refuted suspects

| Suspect | Verdict | Evidence |
|---|---|---|
| Backup sign or perspective when one team acts twice in a row | Refuted | `backup` (`searchthread.cc:602-645`) negates per ply. The tree keeps strict team alternation through forced waits (`collect_batch:914-967`), and the child team is always `~team`. D0: synchronous timing reproduces v3 (14/15). |
| Transposition / `canonicalize_child` merging different time states | Refuted | The node key is `search_hash_key(team, time)` = board hash mixed with `exact_key()`, which includes the whole durative block (clocks, ready, sitting, cap, side, tieNext, atRoot, pending, earlyMs, lastCapture) (`board.h:113-119`, `time_state.h:704-723`, `798-838`). Team mismatches are refused (`searchthread.cc:1366`). |
| Terminal, flag or sit-off verdicts disagree with the arena (main cause) | Refuted as main cause | FlagLoss off: 20/40, q -0.091, same moves as base on 40/40 (`on_noFL`). Gate on: 22/40 (`on_gate`). See the minor rule difference below. |
| Delayed transfer (InstantTransfer false) | Refuted as main cause | `on_IT`: 20/40, q -0.097. |
| Pass valuation and wake-ups | Not the main cause | `on_gate` (no free passes): 22/40; the PACE `sitgate` row also loses (0.206). Root passes on 2 to 7 of 40 in every variant. |
| NN value at nodes with future ready times / the lead plane from node clocks | Not separately confirmed | The planes use `sync_lead` from the node clocks (`planes.cc` `clock_lead_plane_value`), as v3 does. D0's agreement shows no plane error when synchronous. The speed-plane mismatch is part of D1. |
| FPU and prior normalisation with one-board expansion | No defect found | Priors per board are normalised over legal moves plus pass (`process_batch`); single-action nodes have prior 1. D0 control. |
| Root charging of early opponent landings | No defect found | `advance_action:456-541` charges the opponent only up to its landing and dates it back (`charge_late`). Equivalent ordering in D0. Its effect is part of D2. |
| Opponent pace too fast or slow | Folded into D1 | The opponent pace (623) equals our real pace (629). It is *our* cost (546) that is off. |

**Minor rule difference (not durative-specific, both modes).** The engine's mate test counts a drop-blockable check as
mate when the partner has no capture available (`Board::can_partner_provide_blocking_piece`, `board.cc:479-560`; with
SitGate off `can_sit` is almost always true). Arena RULES.md: a blockable check is never mate. Unmeasured and rare.

## Files

All under `/private/tmp/claude-501/-Users-jaycewalzer/b46049ad-27b8-4fc1-93ab-85cae96e771f/scratchpad/diag/`:
- `drive.py`, `exp_scale.py`, `summ.py`, `exp_tempo.py`: the drivers;
- `res.jsonl`, `exp_tempo.json`: the raw results;
- `patch_p1_waitnode.diff`: the D3 patch;
- `w2src/`, `w2p1/`: the source exports;
- `build-base/`, `build-p1/`: the builds;
- `tarshim/tar`, `pyexport/`, `pyv303/`: the Python test check.
