# Measurements, 2026-11-17 (audit WP-H)

First run of the WP-H instruments (audit `docs/audit_2026-11-17.md` section 3, WP-H) on what exists today: the
`runs/custom/local1h` shards and gate logs, and four networks. Everything here was run on the Mac, one process at a
time, CPU ONNX Runtime; the engine runs used 200 nodes. Raw outputs, commands and per-case rows:
`runs/wph/` (`run_all.sh` reproduces every number below; `runs/` is not in git).

## Networks and data

| tag | file | what it is |
|---|---|---|
| it04 | `third_party/hivemind/engine/models/hivemind-fp32.onnx` | stock `hivemind-it04-crossboard-risev33-loss1.556-p82.0`, 74 channels (FP32 export of FP16-rounded weights) |
| v2 | `.../models/hivemind-clock-v2-fp32.onnx` | it04 widened to 86 channels, clock/elapsed columns and new heads zero-initialised |
| it01 | `runs/custom/local1h/it01/model-fp32.onnx` | v2 after one local1h training round (12 games, 1,379 samples) |
| it02 | `runs/custom/local1h/it02/model-fp32.onnx` | it01 after the second round (12 games, 1,724 samples) |

Binaries: stock `engine/build-ninja/hivemind` for it04, clock build `engine/build-clock/hivemind-clock` for v2 and
it02 (free mode: `FreePass`, `MateProbeAnyTime`, `TreeMoveMs 500`). Shards: `runs/custom/local1h/it0{1,2}/selfplay/
training_data` (3,103 samples, 4,496 on-turn (sample, board) rows), recorded by v2 (it01 round) and it01 (it02 round).
Gate logs: `runs/custom/local1h/it0{1,2}/gate` (12 games, free-mode contender vs stock it04 at 50 nodes).

Caveat for every it01/it02 row: those nets were trained on these same shards, so shard-based numbers for them are
in-sample. The sets in section 1 are not.

## 1. Regression sets (`tools/regression.py`)

Sets: `tools/positions/sit_move.jsonl` (21 base positions from theory 02 2.7/2.8, 208 cases after the four seatings
and clock variants) and `drop_mates.jsonl` (10 puzzles, 80 cases: equal clocks, and a +2 s team lead that turns
`TimeAdvantage` on). Every claim is machine-checked under bughouse rules (`regression.py verify`: all 31 hold).
"net" = raw policy top-1 over legal moves plus pass on the tested board; "engine" = the joint search's choice there.

| family (cases) | it04 net | it04 engine | v2 net | v2 engine | it01 net | it02 net | it02 engine |
|---|---|---|---|---|---|---|---|
| partner supplies, expect sit (64) | 44 | 64 | 44 | 64 | 50 | 46 | 64 |
| sit-off for a pinned piece, sit ahead / move behind (80) | 76 | 44 | 76 | 42 | 72 | 48 | 40 |
| every move loses, expect sit (32) | 32 | 16 | 32 | 32 | 32 | 32 | 32 |
| no source, expect move (32) | 18 | 18 | 18 | 20 | 26 | 32 | 31 |
| **sit_move total (208)** | 170 | 142 | 170 | 158 | 180 | 158 | 167 |
| drop-mate top-1 (80) | 48 | 68 | 48 | 77 | 58 | 66 | 72 |
| drop-mate value > 0 (80) | 60 | - | 60 | - | 66 | 80 | - |

Sit-off family by clock (sits out of 20 per tag; expected 20 / 20 / 0 / 0):

| tag | it04 net | it04 engine | v2 engine | it01 net | it02 net | it02 engine |
|---|---|---|---|---|---|---|
| diag +10 s | 18 | 2 | 1 | 16 | 4 | 0 |
| diag +3 s | 18 | 2 | 1 | 16 | 4 | 0 |
| diag -3 s | 0 | 0 | 0 | 0 | 0 | 0 |
| diag -10 s | 0 | 0 | 0 | 0 | 0 | 0 |

What this says:

- **The it04 prior already gets the sit-off sign right (18/20 ahead, 0/20 behind), and the 200-node search throws it
  away**: the stock engine sits 2/20 and the clock engine 1/20 when ahead. This is the audit's search finding 8 (pass
  invisible at play budgets) showing up on hand positions, and it is the one place where the search makes the
  verdict worse than the raw net. The +3 and +10 rows are identical for every net: only the sign reaches it04 (the
  TimeAdvantage plane), and nothing in v2..it02 has learned the size.
- **Two local1h rounds unlearned the sit when ahead**: it02's raw net sits in 4/20 sit-off cases ahead (it04: 18/20)
  and gets the no-source control right everywhere (it04 sits there 14/16 times ahead). It moved toward "never sit"
  on the board, not toward reading the diagonal.
- **v2 = it04 on every case** (same tops, values within 1e-3): the widening is exact, as intended.
- **Stock rule visible**: the stock engine moves in all 16 "every move loses" cases when behind (TimeAdvantage false
  forbids the pass); the clock engine sits in all 32.
- **Drop mates**: the clock engine finds 77/80 (misses the 4-move yo-yo hook in 3 seatings); stock it04 misses the
  guarded contact mate `R@f8` in all 8 cases and plays `R@c8`, a back-rank "mate" a dropped piece could block, which
  is mate in crazyhouse and not in bughouse; it02's engine misses the yo-yo in all 8 (`P@e7`).
- **TimeAdvantage dominates the it04 head**: at equal clocks (plane off) it04's value is negative in 14/16 smothered
  cases with a mate in one on the board (mean value over the puzzles 0.00); with a 2 s lead (plane on) the mean is
  +0.90, but the raw top-1 then becomes the pass in 26/40 puzzle cases, including mate-in-one positions. it02's
  puzzle values are positive in all 80.

## 2. Pass statistics from shards (`tools/pass_stats.py`)

Per on-turn (sample, board) row; diagonal = the on-turn seat's clock minus its partner's opponent's.

| diagonal | rows | pass visited | pass top (most visits) | mean pass visit share |
|---|---|---|---|---|
| ahead | 2,052 | 0.899 | 0.267 | 0.216 |
| behind | 2,348 | 0.692 | 0.317 | 0.269 |
| equal | 96 | 0.302 | 0.000 | 0.010 |
| all | 4,496 | 0.778 | 0.287 | 0.239 |

Per round: it01 shards (searched by v2) pass-top 0.316 ahead / 0.238 behind; it02 shards (searched by it01) 0.212
ahead / **0.366 behind**. By node count the pass-top share falls from 0.65 (< 32 nodes) to 0.36 (32-63), 0.21
(64-127) and 0.10 (128-255): most recorded sits are tiny-budget searches.

Raw pass prior on the same rows (mean p(pass) / share with pass as top-1):

| net | ahead | behind |
|---|---|---|
| it04 | 0.243 / 0.357 | 0.065 / 0.048 |
| v2 | 0.243 / 0.357 | 0.065 / 0.048 |
| it01 | 0.203 / 0.255 | 0.173 / 0.184 |
| it02 | 0.180 / 0.192 | **0.265 / 0.296** |

So the search sits behind the diagonal far more than it04's prior would (0.32 vs 0.05), training copies that into
the prior, and by it02 the network prefers to sit **more when behind than when ahead**: the opposite of the human
rule (theory 02 2.7: "don't pause to wait for material when you're down"). This is the audit's 1.3 problem measured
(over-sitting data, the dedupe and the clock regime), and the reason it01's validation value accuracy fell.

Value labels (training 3(b)): mean |root_q - outcome| is 0.523 on checkmate-game samples (1,159) and 0.609 on
flag-game samples (1,944); per round 0.591 / 0.744 (it01) and 0.455 / 0.515 (it02). root_q -> outcome calibration by
men on the boards: sign agreement 0.94 at 30-39 men, 0.92 at 40-49, 0.79 at 50-59, 0.51 at the full 60 (it02 round).

## 3. Event logs: sits and flags (`tools/pass_stats.py --events`)

Gate games (12): decisions are moves plus starts of sit spells.

| seat group | decisions | sit rate ahead | sit rate behind | sit spells: median / p90 / max length |
|---|---|---|---|---|
| contender (free mode, clock net) | 587 | 0.172 (93) | 0.057 (491) | ahead 0.9 / 1.6 / 51 s; behind 2.2 / 39 / 56 s |
| stock it04 at 50 nodes | 670 | 0.230 (539) | 0.000 (122) | ahead 1.8 / 3.1 / 3.5 s |

Stock it04's pass rate vs diagonal, engine level: 0.12 at -2..2 s, 0.26 at 2..10 s, 0.28 (it01 gate 0.22, it02 gate
0.50) at 10..30 s, 0 whenever the seat is behind. Flags: all 4 gate flags were contender seats, and **all 4 flagged
while sitting** (theory 02 3.5: a network that only flags while sitting has learned to over-sit). Self-play flags (14
of 24 games) cannot be split: self-play ran with `no_events` (see "Not done").

## 4. Implied piece values (`tools/implied_values.py`)

dV from adding one piece to a pocket, 300 sampled sidecar positions, in pawns of the same pocket; first number on
all positions, the second on unsaturated ones (|V| < 0.5: it04/v2 10% of positions, it01 38%, it02 17%).

| net | pocket | N | B | R | Q | pawn dV |
|---|---|---|---|---|---|---|
| it04 (= v2) | ours_A | 5.5 / 4.7 | 4.8 / 4.1 | 7.3 / 6.0 | 16.3 / 10.8 | +0.012 |
| it04 | theirs_A | 2.2 / 2.1 | 1.8 / 1.8 | 3.4 / 3.1 | 6.0 / 4.7 | -0.024 |
| it04 | ours_B | 5.2 / 4.3 | 4.2 / 3.9 | 6.2 / 6.5 | 12.7 / 10.8 | +0.014 |
| it04 | theirs_B | 2.5 / 2.7 | 2.1 / 2.3 | 3.6 / 3.9 | 6.3 / 6.3 | -0.021 |
| it02 | ours_A | 0.75 / 0.58 | 0.74 / 0.74 | 3.2 / 2.6 | 2.4 / 2.3 | +0.038 |
| it02 | theirs_A | 1.8 / 1.8 | 1.5 / 1.4 | 2.6 / 2.5 | 4.5 / 4.5 | -0.015 |
| it02 | ours_B | 0.75 / 0.73 | 1.4 / 1.4 | 3.8 / 2.9 | 3.1 / 2.7 | +0.027 |
| it02 | theirs_B | 2.0 / 2.0 | 1.7 / 1.7 | 2.4 / 2.4 | 4.2 / 4.1 | -0.014 |

Folklore is P 1, N = B = R 2, Q 4. it04's opponent pockets sit close to it (N 2.1-2.7, R 3.1-3.9, Q 4.7-6.3); its own
pockets value a pawn very little, so the ratios there are large. After two rounds our own knight is worth less than
a pawn and our rook more than our queen on board A; one 1,700-sample round moved these more than any input the audit
proposes would, which is the drift problem of audit 1.4 in numbers (it01 is in between: own N 0.8, Q 2.9).

## 5. Clock response (`tools/clock_response.py`)

dV for +10 s on one of our clocks (mean |dV|, all 300 positions; the opponent clocks mirror it), and the same with
the TimeAdvantage plane held at its recorded value (`--hold-ta`), which isolates the clock planes:

| net | dV ours +10 s | by men on boards 30-39 / 40-49 / 50-59 / 60 | with the plane held |
|---|---|---|---|
| it04 | 0.382 | 0.041 / 0.413 / 0.461 / 0.563 | 0.000 (no clock planes) |
| v2 | 0.382 | same as it04 | 0.000 (zero-initialised) |
| it01 | 0.134 | 0.002 / 0.168 / 0.142 / 0.226 | < 0.001 |
| it02 | 0.114 | 0.004 / 0.176 / 0.102 / 0.090 | < 0.001 |

Every clock response of every net goes through the one-bit TimeAdvantage plane; the 43-channel clock and elapsed
planes still move the value by less than 0.001 per 10 s after two rounds (audit 1.4 and inputs 1). The response is
largest with full boards and vanishes at 30-39 men, the reverse of the human claim that time matters most on sparse
boards (theory 02 2.4). Training cut the TimeAdvantage response to a third without putting anything in its place.

Pass vs the on-turn seat's diagonal (raw p(pass), plane following the clocks):

| diag (s) | -30 | -10 | -2 | -1 | +1 | +2 | +10 | +30 |
|---|---|---|---|---|---|---|---|---|
| it04 (= v2) | 0.060 | 0.082 | 0.097 | 0.117 | 0.171 | 0.195 | 0.263 | 0.326 |
| it01 | 0.136 | 0.154 | 0.164 | 0.176 | 0.210 | 0.224 | 0.261 | 0.295 |
| it02 | 0.189 | 0.200 | 0.206 | 0.216 | 0.246 | 0.254 | 0.277 | 0.292 |

With the plane held, it02's p(pass) is 0.234 at -30 s and 0.232 at +30 s: flat, slightly the wrong way. The
diagonal dependence that remains is it04's (TimeAdvantage), and training has flattened it (ratio +30/-30: 5.4 for
it04, 2.2 for it01, 1.5 for it02).

## 6. Value and WDL heads against the outcome (`tools/pass_stats.py --model`)

The networks' own heads (not the search's root_q) on every shard sample, one row per sample, the sample team's view,
outcome in {-1, 0, +1}. The WDL head is ordered (loss, draw, win): hivemind trains it on `y_wdl = value + 1`
(`training/data_loaders.py:259`); read as (win, draw, loss) every WDL number comes out inverted. For all four nets
the value output equals W - L to 4 decimals (WDL score MSE = value MSE), so the two heads say the same thing and
only the value numbers are given. MSE / sign agreement on decisive samples:

| shards | it04 (= v2) | it01 | it02 | in-sample for |
|---|---|---|---|---|
| it01 round (1,379) | 0.930 / 0.712 | 0.659 / 0.765 | 0.247 / 0.913 | it01 (853 train, 526 val), it02 (replay) |
| it02 round (1,724) | 0.548 / 0.807 | **0.549 / 0.786** | 0.212 / 0.937 | it02 |
| both, by men on boards 60 / 50-59 / 40-49 / 30-39 (sign) | 0.51 / 0.70 / 0.80 / 0.99 | 0.56 / 0.71 / 0.87 / 0.85 | 0.64 / 0.90 / 0.98 / 1.00 | |

The only clean out-of-sample row is it01 on the it02-round shards: MSE 0.549 against it04's 0.548, sign agreement
0.786 against 0.807. One local1h round bought nothing on unseen games, which matches it01's validation value
accuracy of 0.54 in the log; it02's 0.21-0.25 is fit to games it trained on (it01 replay plus its own round), not
evidence of a better head. it01 also turned the sign wrong in the sparse 30-39 bucket (mean value +0.09 where the
outcome mean is -0.32; it04 0.99 sign agreement there, it01 0.85), a 435-sample bucket the it01 round had few of.
Draw mass is 0.4-2% for every net, and none of the 24 local1h games was drawn (14 flags, 10 mates), so nothing in
these shards can test the draw column.

Per-run files: `runs/wph/pass_prior_<net>.txt` (both rounds, all buckets) and `runs/wph/calib_<round>shards_<net>.txt`.

## Not done or limited

- Flags while sitting vs moving in **self-play**: self-play ran with `no_events`, so only the 12 gate games give it.
  Recording events in self-play (WP-C/WP-B) or a `sit_ms` sidecar field (spec v3 contract 2) would cover it.
- Sit-length per iteration counts sit spells in the gate only; the shards cannot show sit length because the dedupe
  drops repeat searches (audit 1.3 item 12).
- The net-head calibration (section 6) has one clean out-of-sample cell (it01 on the it02 round, 1,724 samples from
  12 games, so neighbouring samples are strongly correlated); the material buckets under 30 men have 19 samples.
- Theory 02 2.8 case 1 (partop in check cannot sit back) and case 4 (flag race plan) have no positions yet; the set
  has cases 2 and 3, the 2.7 sit-off for a piece, and a no-source control.
- The upstream twin-s-clock arm was not measured (no build here); `regression.py run --model` takes any ONNX with
  74/82/86 channels, the upstream layout needs its own planes.
