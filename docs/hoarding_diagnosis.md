# Hoarding diagnosis: hybrid1 it2 vs it0 (it13)

2026-11-27, spec v7 4.3 (WP2). Tool: `tools/hoarding_diagnosis.py` (tests `arena/tests/test_hoarding_diagnosis.py`).
Data: `runs/study/full/hybrid1/anchors/it2/hybrid-it13/part{1,2}`, 400 games at 120 s, rules v3. Contender = hybrid1
it2 (`runs/full/hybrid1/anchors/it2/C.onnx`, PACE only), opponent = `wave2-self` with twin-s-clock it13 (= it0,
`twin-s-clock-fp32.onnx`), both clock_k 110, hybrid options (durative_tree false, sit_gate true, hybrid_mode gate).
References: `runs/study/wave2c/r4/long/l_hybrid_120` (1345 games, it13 hybrid vs upstream-it13) and
`runs/study/tune/r1/base` (923 games, wave2 it13 vs upstream-it13), both at 120 s with it13 on both sides.

## Method

- Sides: "it2" = the contender's two seats, "it0" = the opponent's, each split by that side's own result.
- Pockets are tracked from the events (start FENs, `pass` events in, drops out) and checked against the pocket in
  every move's FEN: 46 862 of 46 862 agree.
- Holding: each landed piece is matched to the same seat's later drop of that piece type, first in first out;
  pieces still in hand at the end are "never dropped" (censored; ms/moves are over dropped pieces only).
- Pocket value P1 N3 B3 R5 Q9, read at every think start of the deciding seat.
- Calibration: each engine's own search eval at its think ends (thinker's view, [-1, 1]) against the result
  (+1/0/-1), binned by the thinking seat's pocket value.
- Item 6: the network's value and prior mass on drops for the deciding seat's board, from the wave 2 engine's own
  `policy` command (the planes the engine builds; `tools/upstream_check.py parity --engine wave2` matched an
  independent Python encoder + ONNX Runtime on this Mac today: max |dp| 8.5e-07, PASS). Time options are rebuilt
  from the logged clocks and think times (seat clocks at the think start, paces = the team's mean think time, elapsed
  and sitting 0). 800 positions stratified by side and pocket bin, it13 only (it2's ONNX is on PACE).
- 95% ranges: bootstrap over games (500 resamples). Run time about 4 minutes on the Mac for all three matches.

```
.venv/bin/python tools/hoarding_diagnosis.py runs/study/full/hybrid1/anchors/it2/hybrid-it13 --us it2 --them it0 \
  --ref "it13 hybrid vs upstream-it13 (r4 long 120 s)=runs/study/wave2c/r4/long/l_hybrid_120" \
  --ref "tune r1 base (wave2 vs upstream-it13)=runs/study/tune/r1/base"
M=third_party/wave2-hivemind/engine/models/twin-s-clock-it13.onnx
.venv/bin/python tools/hoarding_diagnosis.py runs/study/full/hybrid1/anchors/it2/hybrid-it13 --us it2 --them it0 \
  --model $M --positions 800
```

## The plan's numbers, checked

| plan (2026-11-25) | here | comment |
|---|---|---|
| score 0.365 vs it13 at 120 s | 0.365 [0.318, 0.411] (part1 0.345, part2 0.385) | reproduced |
| "equal think time" | equal time per move, but it2 searched 22% / 16% fewer nodes per game (945k vs 1218k, 1038k vs 1229k) | ms/node measured cold, contender first: 0.0523 vs 0.0403 and 0.0503 vs 0.0423 (audit #4) |
| +3.1 material | +3.24 [2.25, 4.25] mean over all moves, of which on board -0.51 [-1.19, 0.23] and pockets +3.75 [3.02, 4.48] | the whole "material lead" is pieces in hand; on the board it2 is level or behind |
| pocket 12.1 vs 8.6 | team pockets at own decisions 12.99 vs 9.44 | same finding; the plan's numbers sample at every move with the stale FEN pocket of the other board (study_games) |
| mated on its own seat 63% | 0.630 [0.585, 0.676] | true but not a separate signal: it is it2's loss rate. All 252 losses are mates of an it2 seat (no flags, no sit-offs, 4 draws) |

## Verdict

**Hoarding: yes, moderate and consistent.** Against it0 in the same games, and against the it13-vs-it13 reference
levels:

- it2 drops less when it can: 0.301 [0.295, 0.308] of its moves with a non-empty pocket are drops, against 0.362
  for it0 and 0.33 for both sides in both references (-17% relative to it0, -9% relative to the references). The
  gap is there in all three thirds of the game and in wins and losses alike (it2 won 0.334 vs it0 won 0.378; it2
  lost 0.283 vs it0 lost 0.334).
- it2 holds pieces longer: median 7.8 s vs 5.7 s (references 6.9 s), 4 vs 3 own moves; 35.5% of the pieces landing in
  an it2 pocket are never dropped against 23.3% for it0 (references 29-30%). Strongest for rooks (47% never dropped
  vs 32%) and pawns (40% vs 26%).
- it2 sits on more: seat pocket at its decisions 6.50 vs 4.75 (references 5.4); 27% of its decisions with 10+ points
  in hand vs 17%.
- It dies holding: an it2 seat that gets mated has 16.8 points in hand at the mate (its partner 10.9), against 11.2
  for a mated it0 seat; 88.5% of mated it2 seats hold 5+ points. In lost games the last fifth reads +0.7 material
  overall with +7.6 in pockets, i.e. about 7 points down on the boards with the difference in hand.

**Its search does not think holding is winning.** it2's search eval is too optimistic by 0.17-0.25 in every pocket
bin (it believes it is about even while scoring 0.365), and the gap shrinks as its pocket grows (0.252 at 0 points,
0.175 at 10+); its mean eval falls as its pocket grows. it0 is calibrated or slightly pessimistic and its eval rises
with its pocket. So the miscalibration is global, not specific to held pieces: there is no sign in the search evals
that it2 values pieces in hand above what they win.

**it13's network on the same positions** (item 6, local): it13 puts less prior mass on drops in it2's held-piece
positions (0.32 at 5-9 points, 0.32 at 10+) than in it0's (0.42, 0.37), and it2 actually dropped at about it13's prior
rate there (0.32, 0.40). So part of the drop-rate gap is the kind of position it2 reaches with pieces in hand (it13
would not drop much there either), not only it2's own preferences. it13's value on it2's positions is also too
optimistic (-0.08 vs a result of -0.27 at 10+). With 100 positions per cell these ranges are wide.

**What it implies.**
- Plain self-play cannot punish this: both sides of a self-play game share the trait, so holding looks neutral in
  the value targets, and the policy targets distil a search that drops less. The league (frozen it13 as a fixed
  opponent, spec v7 section 5) plays against a side that drops at 0.36 and should expose it directly; the gate's
  equal arm (it13 at equal pace) measures it.
- No special value-target change is indicated yet: the search evals show global over-optimism, not overrating of
  held pieces. Whether it2's network overrates held pieces (value) or under-weights drops (policy prior) is what the
  PACE runs below decide: if it2's value on held-piece positions is above it13's on the same positions, the value
  targets are the problem; if its drop prior mass is below it13's, the policy targets are.
- Caveat for the score, not for the hoarding: it2 got 16-22% fewer nodes per game than it0 through the cold ms/node
  measurement (audit #4). The gate (spec v7 3.1) measures warm.

## Pending on PACE

- Item 6 for it2: the same table with `runs/full/hybrid1/anchors/it2/C.onnx`, and it13 again under the same PACE
  binary for a like-for-like pair. Commands (repo root on PACE, after this commit is there; GPU node because the
  engine runs the network):

```
srun -A paceship-bughouse -q inferno -N1 -n4 --mem=16G --gres=gpu:rtx_6000:1 -t 00:30:00 --pty bash
M=runs/full/hybrid1/anchors/it2
T=tools/hoarding_diagnosis.py
.venv/bin/python $T $M/hybrid-it13 --us it2 --them it0 --model $M/C.onnx --out $M/hoarding_it2.md
N=third_party/wave2-hivemind/engine/models/twin-s-clock-it13.onnx
.venv/bin/python $T $M/hybrid-it13 --us it2 --them it0 --model $N --out $M/hoarding_it13.md
```

  Compare section 6 of the two files. Numbers: **pending (PACE)**.
- hybrid1's self-play value targets on held-piece positions (from the HDST chunks): **pending (PACE), and not built**:
  the tool reads arena match logs only; reading chunks needs the fork's HDST reader and is left for when the chunks
  are reachable.

## Tables (it2 vs it0)

**1. Drop rate** (drops / own moves made with a non-empty pocket; thirds of the game by ply)

| side | result | games | moves w/ pocket | drops | rate | 1st third | 2nd third | 3rd third |
|---|---|---|---|---|---|---|---|---|
| it2 | all | 400 | 19504 | 5872 | 0.301 [0.295, 0.308] | 0.219 [0.207, 0.231] | 0.327 [0.317, 0.336] | 0.323 [0.312, 0.333] |
| it2 | won | 144 | 6903 | 2303 | 0.334 [0.325, 0.342] | 0.225 [0.205, 0.246] | 0.332 [0.316, 0.348] | 0.395 [0.380, 0.411] |
| it2 | lost | 252 | 12472 | 3535 | 0.283 [0.276, 0.292] | 0.216 [0.204, 0.230] | 0.324 [0.313, 0.336] | 0.283 [0.271, 0.298] |
| it0 | all | 400 | 18857 | 6819 | 0.362 [0.356, 0.367] | 0.289 [0.277, 0.301] | 0.368 [0.359, 0.376] | 0.397 [0.385, 0.408] |
| it0 | won | 252 | 12110 | 4581 | 0.378 [0.373, 0.385] | 0.291 [0.278, 0.307] | 0.375 [0.364, 0.387] | 0.432 [0.419, 0.442] |
| it0 | lost | 144 | 6602 | 2206 | 0.334 [0.323, 0.346] | 0.288 [0.267, 0.308] | 0.355 [0.339, 0.372] | 0.339 [0.318, 0.359] |

**2. Holding** (per piece landed in a seat's pocket; time and own moves until that seat drops it; FIFO per piece type; dropped pieces only for ms/moves)

| side | piece | landed | never dropped | ms median | ms mean | ms p90 | moves median | moves mean |
|---|---|---|---|---|---|---|---|---|
| it2 | all | 9097 | 0.355 [0.340, 0.369] | 7798 [7539, 8235] | 10786 [10402, 11152] | 24084 [23340, 25204] | 4.0 [4.0, 4.0] | 5.55 [5.36, 5.73] |
| it2 | P | 3288 | 0.398 [0.378, 0.421] | 9205 [8275, 9821] | 12020 [11343, 12697] | 26312 [24587, 27856] | 5.0 [4.0, 5.0] | 6.07 [5.76, 6.40] |
| it2 | N | 2276 | 0.308 [0.288, 0.329] | 8740 [7803, 9526] | 11478 [10855, 12120] | 26131 [24044, 27363] | 4.0 [4.0, 5.0] | 5.82 [5.52, 6.13] |
| it2 | B | 1974 | 0.279 [0.257, 0.301] | 6578 [5790, 7175] | 8848 [8395, 9374] | 19707 [18112, 21636] | 3.0 [3.0, 4.0] | 4.65 [4.42, 4.90] |
| it2 | R | 638 | 0.473 [0.439, 0.507] | 6733 [5555, 7496] | 9468 [8482, 10497] | 20798 [19080, 23528] | 4.0 [3.0, 4.0] | 5.12 [4.66, 5.64] |
| it2 | Q | 921 | 0.394 [0.357, 0.427] | 7180 [6259, 7792] | 10186 [9434, 11135] | 23782 [20686, 26989] | 4.0 [3.2, 4.0] | 5.44 [5.04, 5.92] |
| it0 | all | 8892 | 0.233 [0.219, 0.246] | 5706 [5546, 6052] | 8397 [8112, 8689] | 19187 [18456, 19916] | 3.0 [3.0, 3.0] | 4.43 [4.30, 4.57] |
| it0 | P | 3339 | 0.262 [0.244, 0.280] | 7190 [6786, 7387] | 9259 [8810, 9705] | 20648 [19268, 22159] | 4.0 [4.0, 4.0] | 4.79 [4.58, 5.00] |
| it0 | N | 2224 | 0.195 [0.177, 0.214] | 5674 [5412, 6726] | 8806 [8330, 9405] | 19886 [18828, 21239] | 3.0 [3.0, 4.0] | 4.62 [4.38, 4.90] |
| it0 | B | 1822 | 0.195 [0.175, 0.215] | 5055 [4691, 5271] | 6797 [6423, 7148] | 15478 [14604, 16426] | 3.0 [3.0, 3.0] | 3.69 [3.51, 3.86] |
| it0 | R | 650 | 0.317 [0.277, 0.354] | 5018 [4632, 5471] | 7652 [6926, 8410] | 18434 [16047, 21799] | 3.0 [3.0, 3.0] | 4.18 [3.83, 4.54] |
| it0 | Q | 857 | 0.237 [0.202, 0.268] | 5401 [4963, 6129] | 8126 [7546, 8781] | 19802 [17718, 21775] | 3.0 [3.0, 3.0] | 4.39 [4.11, 4.68] |

**3a. Pocket value at own decisions** (P1 N3 B3 R5 Q9; the deciding seat's pocket, and its team's two pockets)

| side | result | games | decisions | seat pocket | share >= 10 | team pockets |
|---|---|---|---|---|---|---|
| it2 | all | 400 | 23418 | 6.50 [6.24, 6.77] | 0.270 [0.255, 0.285] | 12.99 [12.47, 13.54] |
| it2 | won | 144 | 8357 | 6.60 [6.06, 7.05] | 0.275 [0.248, 0.302] | 13.23 [12.16, 14.17] |
| it2 | lost | 252 | 14897 | 6.48 [6.16, 6.77] | 0.268 [0.248, 0.286] | 12.92 [12.29, 13.49] |
| it0 | all | 400 | 23444 | 4.75 [4.57, 4.95] | 0.172 [0.159, 0.185] | 9.44 [9.06, 9.84] |
| it0 | won | 252 | 14948 | 4.79 [4.53, 5.02] | 0.172 [0.154, 0.187] | 9.54 [9.00, 10.01] |
| it0 | lost | 144 | 8333 | 4.71 [4.40, 5.03] | 0.174 [0.153, 0.194] | 9.33 [8.72, 9.95] |

**3b. Material balance** (team on-board + pockets minus the other team's, at every move; mean over the game, and the total by fifth of game time; pocket part in brackets)

| side | result | total | on board | pockets | fifth 1 | fifth 2 | fifth 3 | fifth 4 | fifth 5 |
|---|---|---|---|---|---|---|---|---|---|
| it2 | all | 3.24 [2.25, 4.25] | -0.51 [-1.19, 0.23] | 3.75 [3.02, 4.48] | +0.4 (+0.2) | +2.1 (+1.9) | +3.1 (+3.1) | +4.3 (+4.5) | +5.5 (+7.7) |
| it2 | won | 6.37 [4.97, 7.65] | 2.48 [1.56, 3.49] | 3.89 [2.69, 4.97] | +0.4 (+0.2) | +2.3 (+1.9) | +4.6 (+3.6) | +8.3 (+4.9) | +14.0 (+8.0) |
| it2 | lost | 1.50 [0.42, 2.64] | -2.20 [-3.03, -1.45] | 3.71 [2.90, 4.49] | +0.4 (+0.3) | +2.0 (+2.0) | +2.4 (+2.8) | +2.1 (+4.4) | +0.7 (+7.6) |
| it0 | all | -3.24 [-4.25, -2.25] | 0.51 [-0.23, 1.19] | -3.75 [-4.48, -3.02] | -0.4 (-0.2) | -2.1 (-1.9) | -3.1 (-3.1) | -4.3 (-4.5) | -5.5 (-7.7) |
| it0 | won | -1.50 [-2.64, -0.42] | 2.20 [1.45, 3.03] | -3.71 [-4.49, -2.90] | -0.4 (-0.3) | -2.0 (-2.0) | -2.4 (-2.8) | -2.1 (-4.4) | -0.7 (-7.6) |
| it0 | lost | -6.37 [-7.65, -4.97] | -2.48 [-3.49, -1.56] | -3.89 [-4.97, -2.69] | -0.4 (-0.2) | -2.3 (-1.9) | -4.6 (-3.6) | -8.3 (-4.9) | -14.0 (-8.0) |

**4. Calibration** (search eval at own think ends, thinker's view, vs the game result +1/0/-1, by the thinking seat's pocket value)

| side | pocket | thinks | games | mean eval | mean result | eval - result |
|---|---|---|---|---|---|---|
| it2 | 0 | 4580 | 396 | 0.005 [-0.014, 0.023] | -0.247 [-0.347, -0.139] | 0.252 [0.153, 0.346] |
| it2 | 1-4 | 7279 | 400 | -0.046 [-0.080, -0.011] | -0.293 [-0.387, -0.195] | 0.247 [0.169, 0.324] |
| it2 | 5-9 | 5241 | 394 | -0.086 [-0.130, -0.039] | -0.301 [-0.406, -0.198] | 0.215 [0.135, 0.296] |
| it2 | 10+ | 6313 | 385 | -0.094 [-0.157, -0.033] | -0.269 [-0.386, -0.138] | 0.175 [0.093, 0.253] |
| it0 | 0 | 5500 | 400 | 0.075 [0.052, 0.100] | 0.240 [0.130, 0.341] | -0.165 [-0.258, -0.068] |
| it0 | 1-4 | 9105 | 400 | 0.175 [0.138, 0.211] | 0.295 [0.193, 0.395] | -0.120 [-0.195, -0.038] |
| it0 | 5-9 | 4812 | 393 | 0.263 [0.203, 0.317] | 0.309 [0.197, 0.408] | -0.046 [-0.116, 0.026] |
| it0 | 10+ | 4027 | 362 | 0.293 [0.227, 0.365] | 0.278 [0.158, 0.413] | 0.015 [-0.063, 0.094] |

**5. Endings** (shares of all games; the mated seat's and its partner's pocket at the mate)

| side | lost | mated | flagged | sit-off | draws | drop-mates given | drop-mates received | mated seat pocket | its partner's pocket | mated holding >= 5 |
|---|---|---|---|---|---|---|---|---|---|---|
| it2 | 252 | 0.630 [0.585, 0.676] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.010 [0.003, 0.020] | 0.247 [0.203, 0.292] | 0.407 [0.357, 0.454] | 16.79 [15.66, 17.87] | 10.87 [10.02, 11.74] | 0.885 [0.847, 0.925] |
| it0 | 144 | 0.360 [0.314, 0.405] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.010 [0.003, 0.020] | 0.407 [0.357, 0.454] | 0.247 [0.203, 0.292] | 11.15 [9.99, 12.57] | 8.87 [7.91, 9.82] | 0.764 [0.691, 0.833] |


**6. Network outputs** (`third_party/wave2-hivemind/engine/models/twin-s-clock-it13.onnx`, 800 decision positions stratified by side and pocket bin: value for the deciding team, prior mass on drops on the deciding seat's board; against the result and whether that decision dropped)

| side | pocket | positions | value | result | drop prior mass | dropped |
|---|---|---|---|---|---|---|
| it2 | 0 | 100 | -0.091 [-0.158, -0.031] | -0.360 [-0.544, -0.179] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| it2 | 1-4 | 100 | -0.053 [-0.151, 0.028] | -0.090 [-0.305, 0.112] | 0.237 [0.187, 0.287] | 0.280 [0.189, 0.375] |
| it2 | 5-9 | 100 | -0.121 [-0.232, 0.004] | -0.200 [-0.420, 0.015] | 0.318 [0.253, 0.385] | 0.320 [0.240, 0.416] |
| it2 | 10+ | 100 | -0.075 [-0.225, 0.059] | -0.270 [-0.492, -0.040] | 0.324 [0.257, 0.395] | 0.400 [0.307, 0.495] |
| it0 | 0 | 100 | -0.043 [-0.117, 0.020] | 0.270 [0.053, 0.472] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| it0 | 1-4 | 100 | 0.096 [-0.006, 0.200] | 0.240 [0.004, 0.465] | 0.285 [0.230, 0.360] | 0.270 [0.181, 0.369] |
| it0 | 5-9 | 100 | 0.221 [0.089, 0.359] | 0.340 [0.138, 0.540] | 0.420 [0.358, 0.481] | 0.410 [0.313, 0.503] |
| it0 | 10+ | 100 | 0.309 [0.166, 0.443] | 0.260 [0.020, 0.462] | 0.368 [0.306, 0.432] | 0.360 [0.268, 0.455] |

**References** (contender = us, opponent = them)

| match | side | games | contender score | drop rate | seat pocket | hold ms median | never dropped | material balance |
|---|---|---|---|---|---|---|---|---|
| it2 | us | 400 | 0.365 [0.318, 0.411] | 0.301 [0.295, 0.308] | 6.50 [6.24, 6.77] | 7798 [7539, 8235] | 0.355 [0.340, 0.369] | 3.24 [2.25, 4.25] |
| it2 | them | 400 |  | 0.362 [0.356, 0.367] | 4.75 [4.57, 4.95] | 5706 [5546, 6052] | 0.233 [0.219, 0.246] | -3.24 [-4.25, -2.25] |
| it13 hybrid vs upstream-it13 (r4 long 120 s) | us | 1345 | 0.514 [0.486, 0.538] | 0.332 [0.329, 0.336] | 5.40 [5.27, 5.54] | 6948 [6751, 7118] | 0.295 [0.287, 0.302] | 0.16 [-0.38, 0.68] |
| it13 hybrid vs upstream-it13 (r4 long 120 s) | them | 1345 |  | 0.331 [0.328, 0.335] | 5.33 [5.20, 5.46] | 6847 [6620, 7043] | 0.297 [0.288, 0.305] | -0.16 [-0.68, 0.38] |
| tune r1 base (wave2 vs upstream-it13) | us | 923 | 0.516 [0.487, 0.549] | 0.336 [0.332, 0.340] | 5.43 [5.28, 5.58] | 6886 [6629, 7112] | 0.287 [0.275, 0.296] | -0.04 [-0.71, 0.56] |
| tune r1 base (wave2 vs upstream-it13) | them | 923 |  | 0.330 [0.326, 0.334] | 5.51 [5.37, 5.69] | 7012 [6812, 7171] | 0.302 [0.292, 0.311] | 0.04 [-0.56, 0.71] |
