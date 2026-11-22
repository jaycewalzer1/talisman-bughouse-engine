# Local strength harness (wave 2)

The yardstick every later wave 2 change (the hybrid "v3 tree picks the move, the durative timing model decides when
to act" first) is judged by on this Mac before it goes to PACE. Tool: `tools/strength_harness.py`. Tests:
`arena/tests/test_strength_harness.py`. Raw output: `runs/strength/` (git-ignored).

**Model and weights in every measurement here:** twin-s-clock it13 (HF `aminwoo/bughouse-twin-s-clock@edaa246`,
`engine/models/twin-s-clock-fp32.onnx`) on ONNX Runtime with the Core ML EP, for both wave 2 and stock upstream.
Wave 2 engine: `third_party/wave2-hivemind` at `72e79eb` (the pin in `cluster/wave2_engine.ref`), `build-ninja`
(4 search workers). Stock it13: `third_party/upstream-hivemind/engine/build-ninja` (v3.0.x, 4 workers). Machine: the
36 GB M4 Max, 2026-11-21.

## What it measures

1. **positions** (300 decision points from real games). The PACE game logs of `runs/wave2/nodur`, `ref_120` and
   `same_120` (120 s clocks, wave 2 vs it13 and wave 2 vs itself; copied read-only to `runs/pace_wave2/`, 66 MB) are
   replayed event by event through `BughouseState` and the arena's own team controller (`Wave2TeamController`:
   `MoveTimeTracker`, `PocketLandingTracker`, `seat_decisions`), with no engine attached. Every search either team
   ran becomes a decision with: the fen pair, the four clocks, the pockets, `TimeLeadMs`, team / opponent move
   times, the wave 2 `RootMoveTimeMs` (budget, min 100, + 75 lag) and the bare budget that stock it13 gets, the seat
   options (`SeatClocksMs`, `SeatPaceMs`, `SeatElapsedMs`, `SeatSitting`, `SeatArrivalsMs`), the action the seat
   played (a move, or `pass`), the partner half when the same search served it, and the game result for the
   deciding team (1 / 0.5 / 0).
   - A search shows in the log as a `thinking` start with nodes > 0 (a move) or an active `stall` with reason
     `upstream_pass` (a pass). A seat that is already sitting is asked again at every landing and when its wait
     runs out, and a repeated pass leaves no event; the replay re-creates those searches (`hidden: true`).
   - **Check of the reconstruction** (`validate`): a 2-game local match with every search logged
     (`runs/strength/smoke_validate`). 225 of the 227 reconstructed searches are found in the log (224 logged); lead
     and team/opponent move times 225/225 exact, `RootMoveTimeMs` 221/225, budget nodes 223/225, and all five seat
     options 115/115 exact on the wave 2 searches. The misses are timer wake-ups of a sitting seat, whose exact time
     the log does not record, and two identical opening positions in two games (a validate artefact).
   - **Caveat:** the options are what the *current* arena code sends at that state. The PACE games were played on
     2026-11-19 with the arena and engine of that day (before the D4/D7 arena fixes changed `SeatElapsedMs` and
     `SeatSitting`), so for those two options the stored values are not byte-for-byte what PACE sent then. For
     judging engines that the current arena will drive, the current values are the right ones.
   - **Strata** (ids shuffled, so any `--ids` slice samples every stratum; at most one position per game per
     stratum while candidates last): 60 `sit_pass` (the seat passed), 60 `sit_act` (a sit-or-act moment where the
     seat acted), 90 `move_middlegame`, 90 `move_endgame` (both boards at fullmove 8+; endgame = a board at fullmove
     30+ or 16 or fewer men). Openings (a board below fullmove 8) and forced moves are left out.
   - **Sit-or-act** = the seat passed, or it acts after sitting on this turn (`after_sit`), or an opponent on turn
     is sitting (`opp_sitting`), or |lead| <= 1 s and the partner's next landing, within 1 s, is a capture
     (`partner_captures_next`, a piece about to arrive). `partner_can_capture` and `recent_arrival` are stored too
     but hold at about 40% / 60% of all decisions, too common to select on. At 120 s clocks |lead| <= 2 s held at
     99% of decisions, so "small lead" alone selects nothing.
2. **labels**: stock it13 at **32k nodes** (fresh tree, `go nodes`) on all 300 positions, the deepest budget the
   Mac affords in this round (about 0.2 ms/node alone, 0.38 under the round's load: 30 to 60 min). For scale, the
   continuity baseline found it13 at 16k agreeing with it13 at 64k on 70% of positions. The outcome labels are the
   game results in the positions file.
3. **score** `<config>` at 1k / 4k / 16k nodes, fresh tree per search, the arena's inputs for that config's kind
   (wave 2: seat options + wave 2 root; stock it13: no seat options, root = bare budget). `report` gives
   - joint agreement (every board our team is on turn on) and board agreement (the deciding seat's board, a pass
     counts as a move) with it13 @ 32k, and agreement with the move actually played;
   - mean root q (deciding team's view), Brier and log-loss of p = (q + 1) / 2 against the game result;
   - on the sit-or-act subset: the sit rate (search passes on the deciding board) when ahead (lead > 0) and behind
     (lead < 0), and act-vs-sit agreement with the reference;
   - 95% CIs from a game-cluster bootstrap (2000 resamples; positions of one game are resampled together).
   Configs: `nodur` (DurativeTree false, SitGate true), `durative_on` (the headline wave 2 flags), `continuity`
   (the preset: TeamTreeReuse, JointPlans, PlanCarry, OwnPaceBelowRoot, BlockableCheckNotMate; on isolated
   positions with a fresh tree only JointPlans, D1 and D5 act), `it13` (stock upstream), or any new name with
   `--kw '<wave 2 player flags JSON>'`.
4. **match** A B: the arena pool (`arena/pools/wave2.json` rows: centralized teams, clock_k 110, serial,
   charge searched, wave 2 root_min 100 / root_lag 75), rules v3, bughouse openings, colour-swapped pairs, 30 s base
   by default. A is the contender (a wave 2 config); B is `it13` (pool row `upstream-it13`) or a wave 2 config (row
   `wave2-self` with its flags). The schedule is written once (`specs.jsonl`, `match.json`); every call plays the
   games not yet finished (`--max-games`, `--max-minutes`) and refuses a different match in the same `--out`. Free
   memory (`memory_pressure`) is checked before every game; below `--min-free-pct` (25) the call stops cleanly.
   Simulated ms per node is pinned (`MS_PER_NODE_M4MAX`: it13 0.205, nodur 0.206, durative_on 0.246, measured
   alone at 1200 nodes over 20 harness positions), so a match charges the same budgets whatever else runs; other
   configs are measured once per match. **Note:** on PACE's GPUs nodur was charged 0.0530 against it13's 0.0424
   (26% fewer nodes per second of clock); on this Mac the two are equally fast, so the local match does not carry
   that handicap.

## Commands

From the main repo, in its `.venv`. Each step resumes (finished rows are skipped), so a long step can be cut into
short calls with `--ids LO HI` or `--max-games` / `--max-minutes`. Check `memory_pressure` first.

```sh
# 0. the PACE game logs (read-only copy, 66 MB)
mkdir -p runs/pace_wave2
rsync -a pace:/storage/project/ps-bughouse-0/jwalzer3/talisman-bughouse-engine/runs/wave2/{nodur,ref_120,same_120} runs/pace_wave2/
# 1. positions (25 s)
.venv/bin/python tools/strength_harness.py positions                       # -> runs/strength/positions.jsonl
# 2. reference (about 30 min alone; split with --ids 0 100 etc.)
.venv/bin/python tools/strength_harness.py labels --nodes 32000            # -> runs/strength/labels.jsonl
# 3. a config at three budgets (1k+4k: about 6 min alone; 16k: about 16 min alone for 300 positions)
.venv/bin/python tools/strength_harness.py score nodur --nodes 1000 4000 16000
.venv/bin/python tools/strength_harness.py score hybrid --kw '{"durative_tree": false, "sit_gate": true}' --nodes 1000 4000
.venv/bin/python tools/strength_harness.py report                          # tables below
# 4. a match (resumable; stop and start freely)
.venv/bin/python tools/strength_harness.py match nodur it13 --pairs 150 --workers 2 --max-minutes 80 --out runs/strength/match_nodur_vs_it13
.venv/bin/python tools/strength_harness.py match-summary --out runs/strength/match_nodur_vs_it13
# reconstruction check: a short match with every search logged, then compare
.venv/bin/python tools/strength_harness.py match nodur it13 --pairs 1 --log-searches --out runs/strength/smoke_validate
.venv/bin/python tools/strength_harness.py validate --out runs/strength/smoke_validate
```

## Baselines (2026-11-21)

All numbers: it13 network as above; wave 2 `72e79eb` `build-ninja`, stock it13 `upstream-hivemind` `build-ninja`.
Positions `runs/strength/positions.jsonl` (seed 0), reference `labels.jsonl`, scores `scores.jsonl`, this table
`runs/strength/report.md` (`report`). The 16k rows of durative_on and continuity ran in two `--ids` halves; every
config has all 300 positions at every budget.

### Position scores

Positions: 300 from 260 games (120 sit-or-act). Reference: stock upstream it13 at 32000 nodes, 300 labelled. 95% CIs: game-cluster bootstrap, 2000 resamples.

| Config | Nodes | n | Joint agree | Board agree | Played-move agree | Mean root q | Brier | Log-loss |
|---|---|---|---|---|---|---|---|---|
| nodur | 1000 | 300 | 61% [56%, 67%] | 71% [66%, 76%] | 67% [62%, 72%] | 0.059 [-0.016, 0.132] | 0.122 [0.103, 0.141] | 0.387 [0.336, 0.439] |
| nodur | 4000 | 300 | 76% [71%, 81%] | 82% [77%, 86%] | 73% [68%, 78%] | 0.060 [-0.019, 0.135] | 0.118 [0.099, 0.138] | 0.375 [0.324, 0.428] |
| nodur | 16000 | 300 | 87% [83%, 91%] | 91% [87%, 94%] | 81% [76%, 85%] | 0.054 [-0.026, 0.130] | 0.121 [0.101, 0.142] | 0.380 [0.327, 0.437] |
| durative_on | 1000 | 300 | 50% [45%, 56%] | 60% [55%, 66%] | 64% [59%, 70%] | 0.097 [0.024, 0.165] | 0.137 [0.117, 0.159] | 0.446 [0.384, 0.521] |
| durative_on | 4000 | 300 | 56% [51%, 62%] | 63% [58%, 69%] | 71% [66%, 75%] | 0.102 [0.027, 0.172] | 0.132 [0.111, 0.153] | 0.433 [0.369, 0.506] |
| durative_on | 16000 | 300 | 58% [52%, 63%] | 65% [60%, 70%] | 80% [75%, 84%] | 0.104 [0.032, 0.175] | 0.128 [0.108, 0.150] | 0.423 [0.361, 0.499] |
| continuity | 1000 | 300 | 49% [43%, 55%] | 60% [54%, 65%] | 63% [58%, 68%] | 0.062 [-0.012, 0.132] | 0.133 [0.114, 0.153] | 0.430 [0.373, 0.498] |
| continuity | 4000 | 300 | 57% [51%, 62%] | 64% [59%, 70%] | 70% [65%, 74%] | 0.062 [-0.013, 0.134] | 0.132 [0.112, 0.153] | 0.433 [0.368, 0.505] |
| continuity | 16000 | 300 | 58% [52%, 63%] | 65% [59%, 70%] | 73% [68%, 78%] | 0.048 [-0.028, 0.119] | 0.129 [0.109, 0.151] | 0.425 [0.361, 0.500] |
| it13 | 1000 | 300 | 61% [55%, 66%] | 70% [65%, 75%] | 68% [62%, 73%] | 0.068 [-0.009, 0.142] | 0.125 [0.106, 0.145] | 0.395 [0.344, 0.450] |
| it13 | 4000 | 300 | 76% [72%, 81%] | 82% [78%, 87%] | 73% [68%, 78%] | 0.069 [-0.012, 0.144] | 0.121 [0.101, 0.141] | 0.383 [0.330, 0.441] |
| it13 | 16000 | 300 | 89% [85%, 92%] | 91% [87%, 94%] | 80% [76%, 85%] | 0.063 [-0.017, 0.141] | 0.124 [0.103, 0.146] | 0.391 [0.334, 0.455] |
| reference it13 | 32000 | 300 | 100% | 100% | 77% [72%, 81%] | 0.058 [-0.023, 0.136] | 0.125 [0.104, 0.148] | 0.395 [0.336, 0.461] |

Outcome base rate (deciding team's score): 0.533; a constant forecast of it has Brier 0.246.

Sit-or-act subset (the deciding seat's board; sit = the search passes there):

| Config | Nodes | n | Sit rate, ahead on clock | Sit rate, behind | Sit rate, all | Act-vs-sit agrees with reference | Agrees with what was played (sit/act) |
|---|---|---|---|---|---|---|---|
| nodur | 1000 | 120 | 44% [34%, 55%] (n 84) | 3% [0%, 9%] (n 35) | 32% [24%, 39%] | 91% [85%, 96%] | 75% [67%, 83%] |
| nodur | 4000 | 120 | 45% [35%, 56%] (n 84) | 0% [0%, 0%] (n 35) | 32% [24%, 39%] | 96% [92%, 99%] | 80% [73%, 87%] |
| nodur | 16000 | 120 | 48% [37%, 58%] (n 84) | 0% [0%, 0%] (n 35) | 33% [26%, 42%] | 94% [90%, 98%] | 80% [73%, 87%] |
| durative_on | 1000 | 120 | 56% [45%, 66%] (n 84) | 14% [3%, 29%] (n 35) | 43% [35%, 52%] | 79% [71%, 87%] | 78% [70%, 85%] |
| durative_on | 4000 | 120 | 54% [43%, 65%] (n 84) | 17% [6%, 31%] (n 35) | 42% [34%, 51%] | 82% [74%, 88%] | 86% [79%, 92%] |
| durative_on | 16000 | 120 | 64% [54%, 74%] (n 84) | 23% [9%, 37%] (n 35) | 52% [43%, 60%] | 76% [68%, 84%] | 90% [85%, 95%] |
| continuity | 1000 | 120 | 49% [39%, 60%] (n 84) | 11% [3%, 23%] (n 35) | 38% [29%, 46%] | 80% [72%, 87%] | 78% [70%, 85%] |
| continuity | 4000 | 120 | 54% [43%, 65%] (n 84) | 14% [3%, 29%] (n 35) | 42% [33%, 50%] | 82% [76%, 89%] | 82% [75%, 88%] |
| continuity | 16000 | 120 | 61% [51%, 71%] (n 84) | 17% [6%, 31%] (n 35) | 48% [39%, 56%] | 78% [71%, 86%] | 82% [76%, 89%] |
| it13 | 1000 | 120 | 44% [34%, 55%] (n 84) | 3% [0%, 9%] (n 35) | 32% [24%, 39%] | 91% [85%, 96%] | 75% [67%, 83%] |
| it13 | 4000 | 120 | 44% [33%, 55%] (n 84) | 3% [0%, 9%] (n 35) | 32% [24%, 40%] | 94% [90%, 98%] | 78% [71%, 85%] |
| it13 | 16000 | 120 | 46% [36%, 57%] (n 84) | 0% [0%, 0%] (n 35) | 32% [24%, 40%] | 98% [96%, 100%] | 81% [74%, 88%] |
| reference it13 | 32000 | 120 | 46% [36%, 57%] (n 84) | 0% [0%, 0%] (n 35) | 32% [25%, 40%] | - | 81% [73%, 88%] |

In the games themselves, the seat sat at 60 of 120 sit-or-act positions (by design of the sample: about half are passes).

**Noise floor.** 4 search workers are not deterministic: nodur at 4k rerun on ids 0-99 (`--rep 1`) chose the same
joint action on 97/100.

**Reading.**
- **nodur and stock it13 are indistinguishable on positions** at every budget (joint agreement 61/76/87% against
  61/76/89%, the same sit rates, the same calibration). With DurativeTree off and SitGate on, wave 2 is the v3 tree, so
  this is expected; it also means the position score alone cannot explain PACE's 0.436 for nodur. Whatever costs
  nodur there is in game play (time use, the per-node speed charge), not in the move a search picks at a fixed node
  count.
- **The durative tree converges to other moves**: durative_on and continuity flatten at 57-58% joint agreement
  (60/56/58%, 49/57/58%) while nodur and it13 climb to 87-89%, the same picture as the 40-position continuity
  baseline, now on 300 positions with CIs that do not overlap from 4k on.
- **Calibration**: durative_on's root q is the most optimistic (+0.10 against +0.05 to +0.07 for the others and the
  reference) and its Brier and log-loss are the worst (0.128-0.137 against 0.118-0.125). The continuity preset
  (with D1) removes the optimism of the level (q 0.05-0.06) but not the loss. All configs are far better than the
  base-rate forecast (Brier 0.246).
- **Sitting**: the v3 rule (SitGate: sit only when lead > 0) shows as 0-3% sits when behind for nodur, it13 and the
  reference; the durative tree sits more overall (42-52% of sit-or-act positions against 32-33%), including 14-23%
  when behind, and its act-vs-sit choice agrees with it13 @ 32k on 76-82% against 91-98%. "Agrees with what was
  played" rises for durative_on (to 90%) partly because a third of the positions were played by the durative
  engine itself (`ref_120`, `same_120`): it is a consistency number, not a strength number.
- What the hybrid has to beat here: nodur's 87% / 91% (joint / board) at 16k and 94% act-vs-sit, with a sit-when-
  behind rate above 0 only where it helps. The positions test can show that a hybrid keeps v3's move choice and how
  its sitting differs; whether the different sitting wins games is only measurable in a match.

### Local match: nodur vs it13, 30 s base

`match nodur it13 --pairs 150 --workers 2 --max-minutes 80` (`runs/strength/match_nodur_vs_it13`), then 6 min with
3 workers alone for the throughput (same schedule, resumed). ms/node pinned: nodur 0.206, it13 0.205.

| Games | nodur score | 95% CI (game-cluster bootstrap) | W / D / L | Pentanomial (pairs) | End reasons | Nodes per game (nodur / it13) |
|---|---|---|---|---|---|---|
| 170 (85 colour-swapped pairs) | **0.532** | [0.459, 0.614] | 90 / 1 / 79 | [17, 0, 45, 1, 22] | 169 checkmate, 1 repetition | 55.8k / 56.1k |

Sits per game: nodur 4.5, it13 4.6 (PACE nodur row: 3.7 / 3.8). Mean game 108 plies, 18.6 s of simulated play.

**The local 30 s match does not reproduce PACE's direction.** PACE (120 s, 266 games) has nodur at 0.436
[0.361, 0.491]; locally nodur is level or slightly ahead, and the local CI excludes 0.436. Three differences, any of
which could account for it, none isolated here: (1) the per-node charge: PACE charged nodur 26% more ms per node than
it13 (0.0530 against 0.0424 on the GPU), locally both cost the same; (2) the budgets: about 1.3k nodes per search here
against about 20k on PACE (120 s / 110 / 0.053 ms); (3) the code: PACE ran the 2026-11-19 engine and arena (before the
D4/D7 arena fixes), this ran `72e79eb` and the current arena. The first is the cheapest to test: rerun this match with
`--ms-per-node 0.256,0.205` (PACE's ratio) and see whether the score drops toward 0.44.

### Throughput on this Mac (30 s base, nodur vs it13)

| Setting | Games per hour |
|---|---|
| 2 workers, while the labels (it13 32k) and the scoring chain ran beside it | 107 (145 games in 81 min) |
| 3 workers, nothing else running | 227 (25 games in 6.6 min) |
| 1 worker, smoke (budgets at a mis-measured 1.16 ms/node, so shorter searches) | about 120 |

So a 60-90 min slot alone fits roughly 230-340 games with 3 workers (memory stayed at 79-88% free throughout); a
0.05 difference in score needs about 400 games for a CI of +-0.05.

### Cost of this round (wall, shared machine)

positions 25 s; labels 300 x 32k: 69 min (about 0.38 ms/node under load); scoring 3600 searches: about 105 min across
parallel calls; match: 81 + 7 min. Everything ran in about 2 h 15 min of wall time.
