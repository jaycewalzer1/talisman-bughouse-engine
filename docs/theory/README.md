# Bughouse theory library: index and what it means for the engine

Written 2026-11-17. Seven topic files written from web sources plus the code in `third_party/hivemind` and `arena/`,
then a critic's pass (`99_gaps.md`) that checked the engine claims against the code and re-fetched a sample of the
pages. Every correction inside a file is marked `Corrected:` or `Note (critic):`; read those before quoting a file.

Source quality, in one line: FICS/ICC help files, the WCCC rulebooks, Wikipedia, the engine source trees (Hivemind,
Sjeng, Sunsetter, Fairy-Stockfish, multi-variant Stockfish, CrazyAra, Lc0, KataGo) and the cited papers were read in
full. The largest body of written bughouse strategy (chess.com blogs by Sorsi, JarlCarlander, chriswsh and others,
roughly 45 citations across files 02-05) was reachable only as search snippets and is marked `[snippet]` wherever
used. The HiveMind CoG 2023 paper is closed access (abstract only). Treat every `[snippet]` threshold ("2 s decisive",
"double sit 10 s = resign", "sit until 10") as a hypothesis to measure, not a fact.

## Index

| File | What it holds |
|---|---|
| [`01_rules_and_protocols.md`](01_rules_and_protocols.md) | Rulesets compared (FICS/ICC, chess.com, USCF 2004, WCCC 2004/2007, Hawaii): the blockable-check mate rule, flag mechanics (2 s claim delay, double flag = draw, timeseal, `minmovetime`), draws and repetition, castling with dropped rooks, ptell vocabulary and Sjeng's numeric mapping, BPGN format, why lichess has no bughouse. 42 web sources. |
| [`02_time_and_sitting.md`](02_time_and_sitting.md) | The diagonal clock comparison (one number, not two), graded value of time (10/15/20/60 s thresholds and who said them), sitzkrieg and sit-offs, when sitting while behind is right, move-fast tempo, Sjeng's sit rules, what stock's `TimeAdvantage` flag is and what it misses. 35 sources. |
| [`03_piece_values_and_material.md`](03_piece_values_and_material.md) | Human and engine value tables (Nilsson 1/2/2/2/4, Sunsetter, Sjeng, Droste & Fürnkranz learned values, Stockfish multi-variant in-hand tables and king-danger weights), in hand vs on board, marginal value of the first piece, flow, time vs material, `implied_values` probe proposal. 34 sources. |
| [`04_tactics_mating_patterns.md`](04_tactics_mating_patterns.md) | Why only contact, knight, double and occupied-line checks mate; the canonical mates (f7 sac, smothered, g7/h7 clamp, yo-yo, back rank); defensive drops and pawn walls; king safety as holes; the stock mate probe and `supply_search` read against the real rule; drop-mate puzzle gate proposal. 42 sources. |
| [`05_openings_and_team_strategy.md`](05_openings_and_team_strategy.md) | Colour roles (White attacks, Black holds), trades on tempo and flow control, the bughouse opening canon (Fortress, e6 systems, Qd8, Leaf Gambit, Alekhine sac) against the arena's chess-theory `openings.py`, communication as three facts, piece-danger head proposal, what one controller does differently from two humans. 32 sources + 7 repo files. |
| [`06_engines_and_ai.md`](06_engines_and_ai.md) | HiveMind paper and the stock it04 engine; upstream aminwoo/hivemind v3 clock model (`time_state.h`, twin-s-clock, +156 Elo then flat); CrazyAra, Sjeng `partner.c`, Fairy-Stockfish virtual drops, Sunsetter time allocation; no engine has a credible human rating; licences and protocols. 72 sources. |
| [`07_research_time_and_async.md`](07_research_time_and_async.md) | Production time managers (Stockfish, Lc0, KataGo), learned planning budgets (Muppidi et al. 2026), durative-move and simultaneous-move MCTS (ABCD, DUCT), KataGo/Gumbel self-play improvements for low-node fine-tuning, pentanomial and normalised-Elo SPRT arithmetic. 39 sources + 13 code files. |
| [`99_gaps.md`](99_gaps.md) | Critic's pass: 11 corrections with code evidence, claims resting on one weak source, topics missing from the library, engine questions nobody answered, disagreements left between files. |

Local references used throughout: `docs/custom_engine_spec.md` (v1: clock planes 37-40, `FreePass`, `MateProbeAnyTime`,
`TreeMoveMs`, out-of-step arena, jitter 0.5-2.0), `docs/custom_engine_spec_v2.md` (elapsed planes 41-42, `think_gain`
head, `arrivals` head, sit-off terminal rule D1), `docs/beat_stock.md` (gate design: 1,600-game rows, replicate rule,
fixed-node contender, 60 s and 180 s).

---

## What this means for the engine

Ranked by expected value per unit cost, with the evidence strength weighed in. Cost scale: **S** under a day of work and
under 50 GPU-hours; **M** a few days and 100-500 GPU-hours; **L** a week or more, or 500-2,000 GPU-hours. "Covered"
says whether spec v1, v2 or `beat_stock.md` already has it. Ideas the critic refuted are in the last section, not here.

### Tier 1: do before the first training round (they change what the round measures)

**1. One signed diagonal-clock plane, written into both board blocks.** `tanh((t_ourA − t_oppB) / s)`, same value in
both blocks, plus the diagonal opponent's absolute clock `min(t/300, 1)` per block. Every human source names this one
quantity (02 §2.2, 01 row 3, 04 row 9, 05 row 1); stock's flag is its sign (`board.py::time_advantage`), and the
lockstep tests showed what having only the sign costs. v1's planes 37-40 carry our seat and the same-board opponent, so
the diagonal has to be assembled across blocks by the cross-attention; the ablation null ("v1 planes suffice", 01 row 4)
is cheap to keep.
Files: 01, 02, 04, 05, 06 (upstream's lead plane is the same idea), 07 row 1. Covered: **no** (v1 has same-board planes
only). Cost: **S** for the channel; one ablation round **M**. Open: the scale `s` (files use 10, 15, 20, 30; pick after
the pass-rate measurement in item 6).

**2. A bughouse-sound opening set for training and gating.** `arena/openings.py` is chess theory: three Sicilians,
five fianchetto systems, the Dutch, five c4/...b6 systems. Every strong source calls these pawn-structure holes a top
team never allows (05 §2.6). Replace or augment with the canon (Fortress 1.e4 e6 2.d4 d6, e6-Nf6, e6-Nc6, 1.e4 d5
2.exd5 Qxd5 3.Nc3 Qd8, Leaf Gambit, Alekhine 1.e4 Nf6 2.e5 d5, 1.e4 e5 lines, 1.d4 d5 without c4), keep two or three
chess openings as controls, give a share of self-play the bare start position, and report the gate on both sets.
Files: 05 row 11-12. Covered: **no**. Cost: **S**. Evidence: strong agreement across sources, but the move orders of the
e6 systems and Leaf Gambit come from `[snippet]`s.

**3. Self-play clock regime: time odds, longer bases, realistic latency.** (a) 25% of games start with asymmetric
clocks so the diagonal difference at move 1 covers ±5…±60 s, and some games start with both teams low (02 §3.2);
(b) sample base time over 60-300 s, increment 0, since 10-30 s sits only exist at 180 s and up (01 row 16, 05 row 3);
(c) a 100 ms floor on every landing (FICS `minmovetime`), a minimum per-move latency so a seat with 3 s cannot make
30 moves (01 row 17, 02 §3.6b), and an additive lag term beside the multiplicative jitter, drawn from BPGN `Lag`
values or the chess.com shards (01 row 6); (d) a bimodal jitter mixture (premove-fast, normal, long sits) instead of the
unimodal 0.5-2.0× (05 row 6; the human distribution is a guess until item 10 measures it). Without (a) and (b) the
value head never sees the steep region of time value in undecided positions.
Files: 01, 02, 05. Covered: **partly** (v1 has k ∈ 60..300 and jitter 0.5-2.0; nothing else). Cost: **S** to write;
the GPU cost is that 180-300 s games are 3-5× longer than 60 s games, so budget **M-L** per round for the long share.

**4. Instrument the net before changing it.** Six measurements, all inference only, that tell the first round what to
look for: (i) `tools/implied_values.py`: ΔV per added/removed piece per type per pocket, fitted values per round
(03 §3.1; expect Q/P in 3.5-6, knight ≥ bishop in hand); (ii) ΔV per ±10 s on each clock, bucketed by board material
density (03 §3.10: the first seconds-to-pawns exchange rate anyone will have measured); (iii) pass rate and ΔV of the
stock policy as a function of the flag and of the diagonal difference (02 §3.1, 99 §3.3: what it04 actually learned
about time); (iv) sit-length distribution and win rate of games with sits > 5 s per iteration (06 item 4, 04 row 10);
(v) capture timing relative to the partner board's turn (05 row 5: "trades on tempo"); (vi) WDL calibration by
material-difference bucket (03 §3.8). Nobody has measured any of these for any bughouse engine.
Files: 02, 03, 04, 05, 06, 99. Covered: **no**. Cost: **S-M** (CPU/GPU inference over recorded shards).

**5. Fine-tuning protocol for the widened net.** Trunk learning rate ≤ 1e-4 for the first iteration, new (zero-init)
weights at full rate, held-out it04 parity checked before raising; precedents put transfer rates 10-35× below from-scratch
(07 row 18, CrazyAra 0.35 → 0.01, KataGo 6e-5 → 6e-6). Stock's RL config is `0.005 / div_factor`.
Files: 07. Covered: **no** (v1 says only that `train_loop.py --mode rl` must accept 41 channels). Cost: **S**.

### Tier 2: the search and self-play changes most likely to make low-node RL actually improve it04

**6. Gumbel root search and policy target for self-play.** Play budgets are 50-200 nodes (`beat_stock.md`), where the
AlphaZero visit-count target stops being an improvement operator; Gumbel AlphaZero (sample without replacement,
Sequential Halving, completed Q, `softmax(logits + (50 + maxN)·q)`) learns from 2-16 simulations with a 24× training-step
speedup at n=4 in 9×9 Go. Hivemind uses soft visit-count targets, no forced playouts. Start with the per-board heads
(the joint head is off: `joint_policy_loss_factor 0`, spec v2 D5).
Files: 07 row 14. Covered: **no**. Cost: **L** (C++ search path plus converter), one A/B round **M**. Evidence: strong
in Go, untested in bughouse; the largest single bet in this list.

**7. Durative tree: per-seat ready times, fast-forward passes, landing delay.** Three related fixes to the same
modelling error. (a) v1 `TreeMoveMs` charges every on-turn seat the same ms per macro-ply, so the modelled opponent
replies exactly as fast as we do and a sit can never pay off in the tree; replace with per-seat durations (an EMA of
observed think time for opponents, the chosen budget for us), advance to the earliest ready time, let that seat act,
alternate at exact ties (07 row 6, ABCD search; 06 item 5). (b) A one-sided pass (we sit on A while B plays on) is
reached only through many B macro-plies; make a pass last until the next event on the other board so "sit for the
knight" resolves at depth 1-2 (02 §3.4). (c) `board.cc` line 111 moves a captured piece into the partner pocket in the
same node; in the arena it lands with latency and only when the partner moves. Make the piece droppable only after the
next macro-ply on the receiving board, and ablate (03 §3.4b).
Files: 02, 03, 06, 07. Covered: **partly** (v1 `TreeMoveMs` constant; v2 D1 resolves the symmetric sitzkrieg, which
the critic confirmed is the only sitzkrieg shape). Cost: **M-L** C++, with hand-built position tests; the symmetric
sit-off regression test from 02 §3.3 (draw at equal clocks, sign flips with 1 s) belongs here.

**8. KataGo's cheap self-play multipliers.** Playout cap randomisation (p = 0.25 full searches recorded with noise,
else ~N/5 unrecorded; charge the realised node count to the clock so fast searches also make the team faster),
forced playouts `sqrt(2·P·ΣN)` with policy target pruning, policy-surprise sample weighting. Stock has ±5% uniform
node jitter and none of the rest. Measured 1.37×, 1.25× and a further gain in Go; drops and the pass action are
exactly the low-prior moves that never get a visit at 800 nodes (99 §3.8 raises pass exploration as an unaddressed
question; forced playouts on the pass action are the obvious first try).
Files: 07 rows 12, 13, 16; 99 §3.8. Covered: **no**. Cost: **M**.

**9. Think-time control: cheap baselines now, a return-trained gate later.** Now: clamp the arena's `clock/k` by absolute
remaining time like Sunsetter (≤ 40 ms under 4 s), add CrazyAra's two heuristics (stop at half time when top prior
> 0.9 and stable; extend by half on a 0.1 Q drop) and KataGo's `obviousMoves` as baselines the v2 `think_gain` head must
beat; when both boards are on turn, search the higher-gain board first and give the other a quick look (02 §3.6a,
06 item 6, 07 rows 5, 8, 10, 21). Later: upgrade `think_gain` from the supervised proxy ("move changes at 4N") to a
PPO gate over budgets {0 nodes (raw policy), 16, 50, 200, 800} with inputs = frozen trunk features + root value + clock
and elapsed planes, trained on return with variable-duration GAE in the jittered arena (Muppidi et al.: fixed large
budgets fall below 0.50 at long clocks; random budgets are worse than any fixed one, so the gate must be trained with
a jittered opponent but without jitter on its own budgets, 07 row 20).
Files: 02, 06, 07. Covered: **partly** (v2 B is the proxy head and the `(0.5 + 1.5p)` rule). Cost: baselines **S**;
gate **L** (well inside 10k GPU-hours per 07 row 4) and only after the clock net trains.

**10. Human data with its clocks.** The chess.com SL shards already carry per-move timestamps (`scrape_cc.py`
`moveTimestamps`), used by `generate_planes.py` to set plane 31 and to derive `pass` labels while dropping every sit
by the team behind on time. Three uses: (a) measure human move-time distributions by rating, move number and on-turn
state, and calibrate the arena jitter (item 3d) from them instead of guessing (99 §3.2); (b) a supervised warm-up of
planes 37-42 and the pass action from the stored clocks, so the first RL iteration does not start from zero clock
weights (06 item 12, 04 row 13); (c) decide whether to keep the censoring of behind-on-time sits, which re-encodes the
hand rule the project removes (99 §4). Do not treat the filtered pass labels as ground truth for when sitting is right.
Files: 01 row 7, 04 row 13, 05 row 15, 06 item 12, 99 C3. Covered: **no**. Cost: (a) **S**; (b) **M**.

### Tier 3: evaluation and rules (cheap, and the gate is wrong without some of them)

**11. Gate methodology.** (a) Decide equal-nodes vs equal-wall-clock now: upstream's 0.97M distilled net beat the 13.6M
it04 at fixed ms through 3× nps, so at equal time a small clock net may beat widening it04 (06 item 9); `beat_stock.md`
gates at fixed nodes and should say so or change. (b) Run gates as GSPRT with normalised-Elo bounds (≈ logistic Elo with
no draws): worst case ≈ 1.05e6/Δ² games, so [0, 20] Elo ≈ 2.6k games, [0, 10] ≈ 10.5k; 1,600-game rows (±24 Elo) only
see large effects and pass-Elo is biased upward, so the replicate rule stays (07 row 19). The pair correlation is already
handled by `arena/stats.py`'s cluster bootstrap over opening pairs (critic C7). (c) Report at 120 s and 180 s, not only
60 s; the human thresholds are tied to ~1 move/s and do not transfer to 60 s games with 200 ms moves (01 row 16,
02 §3.8). (d) Nodes-as-time: the arena already charges `nodes × ms_per_node` by default (`arena/sim.py`
`charge_wall_time=False`), so cluster GPU contention does not leak into results; keep it that way and use wall-clock only
for the final match (07 row 11). (e) Add a Fairy-Stockfish XBoard arm as a sanity floor and reproduce the paper's ">95%
at 2 s" with the stock binary before claiming anything (06 item 8). (f) A time-odds ladder against stock (+5/+15/+30/+60 s)
to measure Elo per second (02 §3.2).
Files: 01, 02, 06, 07. Covered: **partly** (`beat_stock.md`: rows, controls, replicate rule, 60 s and 180 s arms).
Cost: **S-M**.

**12. Regression sets independent of Elo.** (a) Sit-vs-move: 20-50 hand positions for each case in 02 §2.8 (partop in
check, partner on turn, every move loses, favourable flag race, sit right on exactly one board), asserting the chosen
joint action flips with a few seconds of diagonal difference (02 §3.5, §3.9). (b) Drop-mate puzzles: smothered
`N@h6+ Kh8 R@g8+ Rxg8 Nxf7#`, yo-yo `N@h3+ … Q@g2#`, f7-sac families, crazyhouse back-rank studies, plus a defensive-drop
subset from recorded losses; track policy top-1 and value sign per iteration (04 rows 3, 5). (c) The arena freeze test:
a defender in a blockable check with an empty hand and behind on the diagonal must be flaggable without the attacker
moving; `free_sit_ms` may or may not allow this (04 row 2, 99 §4). (d) Gate stats: pass frequency and length by diagonal
sign, flags while sitting vs while moving, wins by flag while the opponent's board is in check, sacrifice-then-pass
rate, castling rate and king file, per-colour capture rate (02 §3.5, 04 rows 7, 10, 05 rows 8, 14).
Files: 02, 04, 05. Covered: **no**. Cost: **S-M** (positions must be collected; lichess studies were unreachable).

**13. Draws, repetition and outcome scoring in the arena.** The arena produces no draw except `max_plies`, which starves
the WDL draw class. Implement FICS-style repetition (board only, count resets on any drop), score `max_plies` and double
flags as `outcome 0`, make the flag rule a profile (immediate vs FICS 0-2 s claim delay, double flag = draw), and log how
often self-play hits drop-capture loops (01 rows 2, 5, 11). Low direct Elo value, but the WDL head and the FICS-faithful
profile depend on it.
Files: 01. Covered: **no**. Cost: **S**.

### Tier 4: further inputs and heads (each cheap alone; bundle into one widening and ablate)

**14. Candidate input planes beyond items 1 and v1/v2.** In rough order of expected value: (a) `checkers` and
`empty interposition squares` bitboards per board, the exact unblockability predicate attackers compute (04 row 1);
(b) upstream's two derived planes, lead in move-time units and speed ratio, plus per-seat pace EMAs of the last ~5 move
times (06 items 1-2, 07 row 3; the attribution of upstream's gain to its planes is weak, critic C11); (c) "has at least
one" pocket presence planes, since `count/16` puts the decisive 0 → 1 knight step at 0.0625 (03 §3.2); (d) "landed in
this pocket in the last 5 s" planes, the only way to see flow rather than static pockets (03 §3.6). All zero-initialised,
all parity-safe. Each needs its own ablation, which is the real cost.
Files: 03, 04, 06, 07. Covered: **no**. Cost: **S** each to add, **M** each to ablate; do (a) and (b) first.

**15. Auxiliary heads beyond `arrivals`.** (a) Piece-danger head: 10 logits (5 piece types × 2 boards), "if the opponent
on this board received one more X now, is the side to move lost within 3 plies", labelled by short searches like
`think_labels.py`; Sjeng ran this probe every move (`CheckBadFlow`) and a 100-node joint search rarely reaches it
(05 row 4). (b) Refine `arrivals` to time bins (0-2 s, 2-5 s) or time-to-first-arrival, and weight its loss per piece
type roughly P1 N2 B2 R2 Q4 so pawns are not drowned by rare queens (02 §3.7, 05 row 13). (c) KataGo-style
opponent-reply policy (w ≈ 0.15), short-term value targets and an uncertainty head, which later enables
uncertainty-weighted playouts (07 row 15). (d) Later, feed `arrivals` back as root input planes (04 row 4a).
Files: 02, 04, 05, 07. Covered: **partly** (v2 C is the 5 s count head). Cost: **M** each.

**16. Search constants and the mate probe.** Retune `CPUCT_INIT` (Lc0-2019 values 3.0/19652; KataGo got ~25 Elo from
retuning alone), then variance-scaled cPUCT; each a separate SPRT (07 row 17). Keep the mate probe under `FreePass`
(every classical engine's sit logic is mate-threat logic, 06 item 10); report probe hit rate and depth, and raise
`maxMateMoves` if most found mates are depth ≤ 3 (04 row 12). Optional probe mode allowing virtual drops of exactly the
piece types the partner can capture in one move, used only to order root moves (04 row 4b). Check `pressure()` and probe
ordering do not use standard-chess piece values (04 row 6).
Files: 04, 06, 07. Covered: **partly** (v1 `MateProbeAnyTime`). Cost: **S-M** each.

**17. Distil a small clock net if the gate is at equal wall-clock.** Depends entirely on the item 11a decision; upstream's
twin-s result (06 item 9) is the only evidence and is at 100 ms/move on one GPU type.
Files: 06. Covered: **no**. Cost: **L**.

### Dropped or closed (critic refuted or already true; see `99_gaps.md` §1)

- Two diagonal planes, "opposite signs on the two diagonals" in time-odds sampling, "alternate on the favourable
  diagonal": with one clock running per board the boards' clock sums stay equal and there is exactly one diagonal
  difference (C1). Item 1 is one plane.
- "D1 misses asymmetric sit-offs": a sitzkrieg always has exactly the two diagonal seats on turn, so D1 covers it; the
  real gap is the one-sided sit, item 7b (C1).
- "Audit whether the opponent team can pass in the tree": done, it can (`searchthread.cc`, `joint_action.h` under
  `g_freePass`); keep only the regression test (C5).
- "Pentanomial scoring is missing": `arena/stats.py` already bootstraps over opening pairs; GSPRT and the gate-metric
  decision remain (C7).
- "Human SL data is FICS / has no timestamps / is unexplained noise": it is chess.com with timestamps used (C3); item 10.
- "The mate probe runs only with `TimeAdvantage`": reversed; it runs only without it (C4).
- FICS piece values Q10 N7 R4 B2 P1: the 1995 mirror; the live file says Q4 N2 R2 B2 P1 (C6). Values stay learned either
  way (03, 04, 05 agree).
- "Also charge the opponents' clocks in the tree" (06 item 3): spec v1 already charges every on-turn seat; read the
  `board.cc` tree-clock code before touching it (99 §4).
- Communication, lending, requests, a human-partner mode: out of scope by spec; Sjeng's `partner.c` mapping is the
  ready spec if that ever changes (01 row 13, 05 row 16).

### Decisions this library leaves open

1. Scale constant for the diagonal plane (10, 15, 20 or 30 s): measure the pass-rate curve first (item 4iii).
2. Gate metric, equal nodes or equal wall-clock (item 11a); it decides between items 1-16 on the widened it04 and item 17.
3. Which time controls to train and gate at: report 120 s and 180 s; sample 60-300 s in self-play.
4. Whether `free_sit_ms` lets an attacker freeze a stuck defender to flag (item 12c): one arena test.
5. Whether to keep `generate_planes.py`'s censoring of behind-on-time sits before reusing human data (item 10c).
6. Whether the per-board stem can combine clocks across blocks before the cross-attention (04 row 9 assumed not, from
   the architecture description, not the model code); after C1 this matters less, since item 1 writes the one number
   into both blocks.

### What was not found anywhere (99 §3)

A written sit inequality (arrival probability × gain vs lead spent; the v2 `arrivals` head makes it computable), human
move-time distributions (computable from the SL shards), what it04 learned about time, defensive technique as a system,
converting a time lead on a sparse board, a concrete repetition rule for drop loops, the HiveMind paper's "clock times
choose the move order" mechanism, bughouse-specific exploration of the pass action, chess.com's lag and premove mechanics,
and any collected tactical puzzle set. The Sorsi and JarlCarlander chess.com series should be read end to end by someone
with a login; they are the largest written body of bughouse theory and reached this library only as snippets.
