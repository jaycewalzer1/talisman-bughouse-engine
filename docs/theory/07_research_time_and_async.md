# 07 — Research on time and asynchrony: engine time managers, learned planning budgets, simultaneous-move MCTS, fine-tuning tricks, fair evaluation

Written 2026-11-17 for the custom engine (`docs/custom_engine_spec.md`, `docs/custom_engine_spec_v2.md`, `docs/beat_stock.md`): one agent on both boards, it04 weights, real clocks as inputs, no hand-written stalling rules, out-of-step self-play, goal = beat stock Hivemind. Everything below was read from the primary sources listed in section 4; claims resting on one weak source are marked. "Hivemind does X" statements come from reading the checked-out code under `third_party/hivemind` (paths given).

---

## 1. Summary

- **The right amount of thinking is state-dependent, and a learned gate beats every fixed budget.** Muppidi et al. (arXiv 2606.26463) train a small PPO gate on top of a *frozen* AlphaZero-style planner to pick a per-decision budget from a small set; it beats the best fixed budget in all five real-time games, and in the clock games (Speed Hex/Go) fixed large budgets fall *below 0.50* at long clocks because "stronger search often just increases the chance of burning clock" [2]. Random budgets are worse than any fixed budget [2]. This is the direct precedent for our think-time head, but it argues for training the gate on *return*, not on a "does the move change" label.
- **Production time managers are all variants of three ideas:** (a) a per-move share of remaining time shaped by a game-length model (Stockfish `optScale`×`timeLeft` with a 50-move horizon [4]; Lc0 log-logistic moves-to-go curve, midpoint 51.5, steepness 7 [8]; KataGo turns-left from stones on board [16]); (b) *in-search* adjustment from search statistics (Stockfish: best-move stability, falling eval, effort share of best move [5]; Lc0 smart pruning 1.33 = stop when no move can overtake [6][10]; KataGo futile-visits stop + "obvious move" discount from policy entropy/surprise [17][18]); (c) a hard maximum per move (Stockfish ≈81% of clock [4]; Lc0 smooth `max-move-budget` 0.3 [7]). Hivemind has **none** of (a)–(c) inside the engine: it only accepts `go nodes N` / `go movetime ms` and does not parse `wtime`/`btime` (`engine/src/interface/uci.cc`), so all time management lives in the arena controller (`clock/k`, k∈[60,300]).
- **Asynchrony has a clean game-tree model: the side whose next decision point comes first moves next.** Churchill, Saffidine and Buro's ABCD search handles *durative* moves by advancing the state to `min(τ(MAX), τ(MIN))` and letting whoever is ready act; simultaneous nodes are sequentialised with alternating move order (`Alt`), which they show empirically beats randomised ordering (0.54 vs 0.46) [27]. Our `TreeMoveMs` (every macro-ply costs a fixed `ms` on both boards) is the degenerate case where all durations are equal; the fix is per-seat durations.
- **Sequential models of simultaneous moves favour one player; decoupled selection fixes the bias but is not Nash-convergent.** Decoupled UCT with UCB1-Tuned won 62.3% in Tron vs 51.4–54.3% for sequential models [26]; DUCT provably fails to converge even in biased rock-paper-scissors, while Exp3/regret matching converge to approximate equilibria (Lisý et al.) [24][25]; in head-to-head play UCT/RM still beat the convergent methods [25]. For bughouse, the simultaneity is mostly *timing* (who commits first), so the Churchill durative model matters more than mixed strategies.
- **Real-time RL formalisms tell us which inputs are necessary:** Ramstedt & Pal's RTMDP augments the state with the action in flight [3]; Xiao et al. show the minimum extra inputs are "previous action" and "action-selection time", with the remaining-execution vector (VTG) the best representation [30]. Translated: the network needs to know *how long each on-turn side has already been thinking* (spec v2 planes 41–42 do this) and ideally the *expected remaining think time* of each opponent seat.
- **With 50–200 nodes per search (our regime), the AlphaZero visit-count policy target is a weak improvement operator.** Gumbel AlphaZero/MuZero (sample without replacement + Sequential Halving + completed Q-values, `π' = softmax(logits + σ(q))`, `σ(q) = (50 + max N)·q`) learns reliably with 2–16 simulations where MuZero fails, with 24.3× fewer training steps at n=4 on 9×9 Go [23].
- **KataGo's self-play recipe is the best-documented set of fine-tuning multipliers:** playout cap randomisation (p=0.25 full searches at N=600, else n=100 without noise, only full searches recorded) ×1.37; forced playouts `n_forced = sqrt(2·P·ΣN)` + policy target pruning ×1.25; auxiliary targets (ownership/score ×1.65, opponent-reply policy w=0.15 ×1.30); global pooling ×1.60 [14]. Later additions: policy-surprise sample weighting, auxiliary soft policy target (T=4, 8× weight), uncertainty-weighted playouts + variance-scaled cPUCT (+75 Elo), optimistic policy (+40–90 Elo), subtree value bias (+30–60 Elo) [15]. Hivemind's self-play has ±5% uniform node jitter (`nodeRandomFactor 0.05`), not PCR, and no forced playouts or target pruning.
- **Fair evaluation:** paired reversed-colour openings must be scored as *pairs* (pentanomial); the trinomial model over-estimates variance and costs ~20% more games [32]. Normalised Elo `en = el·sqrt(1−d)` (no draws in bughouse ⇒ normalised = logistic Elo). Worst-case SPRT length at α=β=0.05 is `T ≈ 1.05e6/(Δen)²` games: 262k games for bounds 2 Elo apart, 10.5k for 10 Elo, 2.6k for 20 Elo [33]. Our 1,600-game rows (±0.035 ≈ ±24 Elo) can only see large effects, and SPRT pass-Elo is biased upward, so replication is mandatory [35].

---

## 2. The knowledge

### 2.1 Time managers in production engines

#### 2.1.1 Stockfish (`timeman.cpp`, `search.cpp`, master Nov 2026) [4][5]

Two budgets, **optimum** and **maximum**, computed once per `go`:

```
mtg      = movestogo ? min(movestogo, 50) : 50;   if (scaledTime < 1000 && !movestogo) mtg = scaledTime*0.05
timeLeft = max(1, time + inc*(mtg-1) - moveOverhead*(2+mtg))
// sudden death:
originalTimeAdjust = 0.3272*log10(timeLeft) - 0.4141        (computed once per game)
logTimeInSec = log10(scaledTime/1000)
optConstant  = min(0.0029869 + 0.00033554*logTimeInSec, 0.004905)
maxConstant  = max(3.3744 + 3.0608*logTimeInSec, 3.1441)
optScale = min(0.012112 + (ply+3.22713)^0.46866 * optConstant, 0.19404*time/timeLeft) * originalTimeAdjust
maxScale = min(6.873, maxConstant + ply/12.352)
// behind on the clock: spend less
timeAdvantage = (time_us - time_them) / (1 + time_us + time_them);  optScale *= 1 + 0.9*min(timeAdvantage, 0)
optimumTime = max(1, optScale*timeLeft)
maximumTime = max(optimum, min(0.8097*time - moveOverhead, maxScale*optimum));  if Ponder: optimum += optimum/4
```

Numbers: at ply 0 with a long clock `optScale ≈ 1.7%` of `timeLeft`; the maximum is 3.1–6.9× the optimum, capped at ~81% of the clock. The opponent's clock enters **only** to cut time when behind (never to spend more when ahead) [4].

Inside iterative deepening the optimum is multiplied by four search-derived factors and the search stops when `elapsed > min(totalTime, maximum)` [5]:

```
fallingEval        = clamp((11.48 + 2.30*(prevAvgScore - bestValue) + 1.1*(iterValue - bestValue))/100, 0.576, 1.728)
timeReduction      = clamp(interp(rootDepth - lastBestMoveDepth, 4.96..18.79 -> 0.639..1.712), 0.629, 1.544)
reduction          = (1.468 + previousTimeReduction) / (2.284*timeReduction)        // stable best move -> less time
bestMoveInstability= 1.077 + 2.229 * totBestMoveChanges/threads                      // changes decay by /2 per iteration
highBestMoveEffort = clamp(interp(nodesEffort, 75800..104510 -> 0.969..0.714), 0.693, 0.838)  // best move ate >~76% of nodes -> less time
totalTime = optimum * fallingEval * reduction * bestMoveInstability * highBestMoveEffort
one legal move: totalTime = min(500 ms, totalTime);   stop immediately on mate_in(3) found / mated_in(2)
next depth only if elapsed <= 0.5*totalTime
```

Lessons: (1) time is spent where the search is *unstable* or the eval is *falling*; (2) a move the search is already sure about is played fast; (3) there is a hard per-move cap; (4) `nodestime` mode runs the same formulas on node counts, which is how fishtest removes hardware speed from results [36].

#### 2.1.2 Leela Chess Zero [6][7][8][9][10][11][12][13]

- **Legacy manager (still the default, `TimeManager=legacy`)** [8][9]: `movestogo = midpoint·(1 + 2·(move/midpoint)^steepness)^(1/steepness) − move` with `midpoint 51.5`, `steepness 7.0` (the comment says it approximates the median residual length of a log-logistic game-length distribution); `total = max(0, time + inc·(movestogo−1) − moveOverhead)`; `this_move = total/movestogo`; first move of a game ×`(1 + 1.8 + 0.25·min(12, ply))` because there is no tree to reuse; unused time goes into a piggy bank spent at rate `immediate-use` (default 1.0); deadline `min(this_move, time − overhead)`. Move overhead default 200 ms [9].
- **Smart pruning / futile search aversion** (`smart-pruning-factor` 1.33) [6][10][12]: near the end of the budget, estimate remaining visits from remaining time × nps; a move that cannot overtake the leader stops receiving visits; if everything but the leader is pruned, stop. Factor f>1 prunes earlier; with f the search can use up to (1+f)× less time than budgeted, which is why `slowmover` existed (2.4 in the 2018 post, 1.0 now). Disabled while pondering because the remaining time is unknown [6].
- **Smooth manager (PR1195)** [7]: target a **consistent total node count per move including reused tree nodes**. `avg_total_nodes = (game_time·nps / moves_left) / (1 − tree_reuse)`, `target_new = avg_total − reused_now`, `budget = (target_new/nps) / timeuse_factor`, capped at `max-move-budget 0.3` × base time. Estimators are EMAs: `init-tree-reuse 0.5`, `max-tree-reuse 0.7`, `init-nps 20000`, `init-timeuse 0.7`, `min-timeuse 0.3`; EMA steps are measured in "average move times" so a run of instant forced moves does not wreck them. The page notes "targeting the bestmove's visit count might work better".
- **KLD-gain stop** (`minimum-kldgain-per-node`, default 0; interval 100 nodes) [10][12][13]: abort when the KL divergence of the root visit distribution between checks falls below a threshold — introduced in self-play run T50 "to spend more time when the Policy prediction was bad, and less time on obvious moves" [13]. One secondary source (wiki glossary); the mechanism is in the options docs.
- **Moves-left head in search** [11]: `MovesLeftMaxEffect 0.0345`, `MovesLeftThreshold 0.8` (|Q| must exceed it), `MovesLeftSlope 0.0027` per predicted ply, scaled by `1.6521·|Q| − 0.6521·Q²`: a bonus for shorter wins / longer losses, only when the position is already decided. Current search constants: `CPuct 1.745`, `CPuctBase 38739`, `CPuctFactor 3.894`, `FpuValue 0.330` (reduction), `StickyEndgames true`, `DrawScore 0` [11]; the 2019 wiki still shows `cpuct 3.0`, `fpu-reduction 1.2`, `policy-softmax-temp 2.2` [10][12] — Hivemind's `CPUCT_INIT 3.0`, `CPUCT_BASE 19652`, `FPU_REDUCTION 1.0` (`search_params.h`) match that older generation.

#### 2.1.3 KataGo [14]–[22]

- **Base allocation** (`timecontrols.cpp`) [16]: estimate turns left from board area A and stones S (Fischer: typical `0.75A+15`, min `0.10A+20`, halved for our moves); Fischer rule: `rec = increment + lagBuffered(mainTimeLeft − increment)/turnsLeft`, `max = min(mainTimeLeft, increment + excess/5)`; when `mainTimeLeft ≤ increment` spend the remaining time. `applyLagBuffer` halves times below `2·lag` and subtracts `lag` otherwise.
- **Shaping** (`searchtimehelpers.cpp`, defaults from `setup.cpp`) [17][18]:
  - `rec *= overallocateTimeFactor` (1.0).
  - Midgame weight rises linearly to 1 at turn `midgameTurnPeakTime·area/361` (130) and decays as `exp(−(turn−peak)/(endgameTurnTimeDecay·area/361))` (100); `rec *= 1 + w·(midgameTimeFactor − 1)` (factor default 1.0, i.e. off unless configured).
  - **Obvious moves**: `obv = min(exp(−policyEntropy/0.30), exp(−surprise/0.15))`; `rec *= 1 + obv·(obviousMovesTimeFactor − 1)` (default 1.0 = off; range 0.01–1). "Surprise" is the KL from the raw policy to the search result.
  - Tree carry-over: `rec *= min(1, softplus(6·(rec − carried)/rec)/6)` so a reused tree reduces time but never to zero.
  - `rec = max(rec, tcMin); rec *= searchFactor; rec = min(rec, tcMax)`.
- **Futile-visits stop** [17][18]: `upperBoundVisitsLeft = ceil(proportionOfTimeLeft·rootVisits + threads−1)`; stop when it is `< futileVisitsThreshold·rootVisits` (default 0 = off, range 0.01–1), the most-visited move is also the move that would be played, and no other move can reach `futileVisitsThreshold·maxVisits` (at temperature, `maxVisits·0.01^T`).
- **Play faster when winning** (`playutils.cpp`) [21]: with the last three root win-loss values, `λ = (excess over searchFactorWhenWinningThreshold)/(1 − threshold)`, `searchFactor = 1 + λ·(searchFactorWhenWinning − 1)`; gtp example sets 0.40 at threshold 0.95 "for human-friendliness", with the comment to disable it for controlled matches [20].
- **Deliberate delay** [22]: `delayMoveScale` adds a random wait with mean `0.5·scale/(policyProb + 0.10)` — the engine *waits longer on moves the policy found less obvious*, purely cosmetic for humans, but it is a worked example of time as an output.
- **Dynamic playout-doubling advantage** [22]: when behind by handicap, KataGo searches *as if* it had `PDA` doublings fewer playouts, capped at 2.75, adjusted ±0.125 per turn by recent win-loss. This is an explicit "asymmetric compute between the sides" knob inside evaluation, a cousin of our yes/no `TimeAdvantage` plane.

#### 2.1.4 CrazyAra (Hivemind's ancestor) [38]

Section 9.2.4 of the CrazyAra paper: constant per-move budget from expected game length, remaining time and increment, adding 70% of the increment per move; sudden death assumes 50 moves and switches at move 40 to spending 5% of remaining time per move assuming 20 more moves; stop at half the budget if the top prior > 90% ("easy" position) or if there is one legal move; "critical" positions get an extra half; move time randomised by ±10% for variety. None of this survived into Hivemind's engine (no time parsing in `uci.cc`).

#### 2.1.5 What the three engines agree on

| Principle | Stockfish | Lc0 | KataGo |
|---|---|---|---|
| Per-move share from a game-length model | 50-move horizon, `optScale` | log-logistic curve (51.5, 7) | turns-left from stones |
| Spend less when the search is already sure | stable best move, high effort share, 1 legal move → 500 ms | smart pruning 1.33, KLD-gain | futile visits, obvious-move factor (entropy, surprise) |
| Spend more when unsure or the eval falls | `fallingEval` up to 1.728×, `bestMoveInstability` | (KLD gain) | — |
| Hard per-move cap | min(81% clock, 3–7× optimum) | `max-move-budget 0.3` (smooth) | `maxTimePerMove`, `increment + excess/5` |
| Opponent clock | only cut when behind | — | — |
| Reuse of the previous tree | — | tree-reuse EMA (0.5→0.7) | softplus carry-over |

Disagreement: Stockfish *spends more* when the eval falls; KataGo *spends less* when winning (as a courtesy option). Nobody spends more when ahead on time, which bughouse rewards (sitting on a time lead is a weapon in a way it is not in chess).

### 2.2 Learned planning budgets and value of computation

#### 2.2.1 "Finding the Time to Think: Learning Planning Budgets in Real-Time RL" (Muppidi, Darwish, Cope, Henriques, Foerster, arXiv 2606.26463, June 2026) [1][2]

- **Setting.** The environment advances every frame regardless of the agent (fallback = no-op or a committed action). The agent is "a procedure that runs in time", generalising Ramstedt & Pal's fixed one-step delay to a chosen delay k per decision. Each budget k wraps an anytime computation (MCTS for k frames); meta-level is an SMDP with holding time k.
- **Gate.** Inputs: raw observation, the planner's frozen trunk features (128-d), the planner's scalar value, and in clock games the remaining clock fraction. Output: distribution over k ∈ K and a value baseline. Trained with PPO on the discounted environment return over the k frames (variable-duration GAE with γ^k); **no reward shaping — the cost of thinking is only world progress or clock depletion**. The planner is frozen. Appendix architecture: 1×1 conv to 64 channels, 3 residual blocks, global average pool; Speed Hex uses a GRU.
- **Budget sets.** Frame games k ∈ {1,2,3,4} at 32 simulations per frame (so 32–128 sims); Speed Go sims ∈ {16, 32, 64, 96} with episode clocks randomised over {100, 300, …, 5700} units; "each simulation advances the player's clock by one unit".
- **Results (100 seeds).** Pac-Man 2370 vs 2149 best fixed (+10.3%); Tetris 45.6 vs 27.6 (+65%); Snake 16.54 vs 14.91 (+10.9%); Speed Hex 0.58 vs 0.46 (greedy) / 0.43 (k=128); Speed Go 0.59 vs 0.51 (k=64) / 0.50 (midpeak heuristic). Random budget choice is below every fixed k. In the strict-timeout Hex control the gate averages 0.952 vs policy-only, 0.903 vs fixed-2, 0.893 vs fixed-128, 0.690 vs always-32; fixed-128 dips to 0.490 at T=5700, greedy to 0.487: "stronger search often just increases the chance of burning clock, so the gate wins by learning a conservative time-management rule".
- **State dependence.** Pac-Man: k grows with distance to the nearest ghost (k=1 close, k=4 far). Tetris: bimodal — k=1 on empty boards, k=2/k=4 on dense boards, k=3 never. Snake: k=3 use jumps 3.2%→13.5% right after eating. Clock games: with T=300 the gate mostly picks the cheapest option; with T=4100 it spreads toward deeper search.
- **Caveats they state.** The base checkpoint choice "substantially affects downstream performance"; the budget set is hand-calibrated; holding times are deterministic in simulation while real hardware is a stochastic continuous-time SMDP; the paralysis mechanism ("naively planning how long to plan can paralyze the agent") is asserted in the abstract and avoided by freezing the planner and making the gate orders of magnitude cheaper than one MCTS rollout — it is not analysed. No value-uncertainty or policy-entropy heuristic baselines were run, so the gate-vs-KataGo-style-heuristic comparison is open.

#### 2.2.2 Real-time RL (Ramstedt & Pal, NeurIPS 2019) [3]

- RTMDP: state (s, a) where a is the action in flight; the policy's output at t affects the world at t+1; formally identical to a 1-step constant-delay MDP but motivated by *action-selection time*. Theorem 2: a real-time framework with a turn variable can express turn-based games, so real-time algorithms handle both.
- Lemma 1: action-value learning needs two updates to see an action's effect (slows SAC-style off-policy methods). Lemma 2: **state-value functions can be learned off-policy regardless of the action in flight** — so a state-value head (AlphaZero style) is the right object in real-time settings. RTAC merges actor and critic on that basis and outperforms SAC in real-time and turn-based MuJoCo/Avenue (numbers not in the readable text).

#### 2.2.3 Concurrent control (Xiao et al., ICLR 2020) [30]

Q is conditioned on `(s_t, a_{t−1}, t_AS, VTG)`: previous action, action-selection latency, and vector-to-go (remaining part of the previous action). Continuous-time Bellman discretised with discount γ^{t_AS/H}; still a contraction. Ablation: VTG matters most and is least hyper-parameter sensitive; `t_AS` and the previous action are the "theoretical minimum". Simulated grasping: 93.49% success with 31.3% shorter episodes vs blocking; real robot: 68.6% vs 81.4% success but 49% faster (the paper calls this "comparable"; it is not).

#### 2.2.4 Metareasoning: value of computation and stopping [28][29]

- Hay, Russell, Tolpin, Shimony: selecting simulations is a *metalevel MDP*, not a bandit — simulation cost is independent of the simulated action's utility, so UCT's regret objective is the wrong one. Stop when `max_i VOI_i ≤ cost`. The **myopic rule stops too early** because "changing the chosen action often takes more than one computation"; the *blinkered* policy (many computations, one action at a time) fixes it. VOI-aware root sampling in Pachi 9×9 reached a 64% win rate vs plain UCT at 10k samples/ply; the benefit is largest at intermediate budgets.
- Sezener & Dayan 2020: *static* (max of expectations) and *dynamic* (expectation of max) computation values bound action values; VOC′-greedy with static values can stop with zero value for every single computation when two or more are needed (Prop. 4); dynamic values do not have the problem (Prop. 5) but are expensive (Monte Carlo or bounds). Takeaway for us: any learned "value of thinking longer" label must look *several* computations ahead — a 4× horizon as in spec v2 is a reasonable non-myopic proxy.

### 2.3 MCTS for simultaneous and asynchronous moves

#### 2.3.1 Two ways to put simultaneous moves in a tree [25][26][27]

- **Sequential model**: inside the tree one side moves first and the other sees it (Tron's "sequential UCT"; Kovarsky & Buro's Maxmin/Minmax transforms). `mini(G) ≤ Nash(G) ≤ maxi(G)` with strict inequality iff no pure optimum exists; the side that moves second has an informational advantage [27][26]. In Tron the sequential variants won 51.4–54.3% overall [26].
- **Stacked-matrix model**: each node holds a matrix of joint moves. *Decoupled UCT* runs UCB1 independently per player over its own actions and backs up each player's reward to its own action; the joint child is the cell. DUCT with UCB1-Tuned won 62.3% ± 0.6 overall in Tron, the best of eight variants; plain DUCT(mix) was worst but one (39.5%) and Exp3 last (35.5%) [26]. In a nine-game study the relative ranking "depends strongly on the game and the opponent" but DUCT is best overall (abstract only; the PDF was not retrievable [39]).
- **Convergence**: DUCT does not converge to Nash even in biased rock-paper-scissors (Shafiei et al.), nor does the sequential model; Exp3 and regret matching are ε-Hannan consistent and SM-MCTS with them converges to a `(2D²ε + δ)`-NE for depth D (Lisý et al., Thm 4.11) [24][25]. Randomised tie-breaking in DUCT moves it much closer to NE [25]. Online head-to-head: UCT is "clearly the strongest sampling algorithm" in Oshi-Zumo and random games, RM "the only algorithm that did not lose significantly against any other"; with a good evaluation function RM and OOS were best in Tron, and with 5× more time the differences shrink [25].
- **Alternating-order sequentialisation with durative moves (ABCD)** [27]: units have ready times `u.ta`, `u.tm`; the player to move next is the one with the unit of minimum ready time; between `s.t` and `min(τ(MAX), τ(MIN))` only passes are possible and are skipped; at a true simultaneous node the order alternates down the tree (`Alt`), with the first mover's effects delayed until the second has chosen. In 5 ms searches ABCD-Alt scored 0.54 vs randomised order 0.46 head-to-head, and playout-based evaluation (0.92 avg vs scripts) crushed static evaluations (0.59, 0.80).

#### 2.3.2 How this maps to bughouse

Bughouse is not a simultaneous-move game at the move level — moves land one at a time and are observed — but it is an **asynchronous timing game**: each on-turn seat is a "unit" whose ready time is its own think time, which the other side does not know. Two consequences:

1. The opponent's next decision point is a random variable, not a fixed macro-ply. A tree in which every macro-ply costs the same `ms` on all four clocks (spec v1 `TreeMoveMs`) makes the opponent reply exactly as fast as we do; sitting then only "works" if the opponent is modelled as sitting too. ABCD says: advance to the earliest ready time, let *that* seat act, keep the others' timers running.
2. Which seat acts first when both are on turn is the only true simultaneity. The Churchill result (alternate the order, hide the first mover's effect from the second's move generation) is the cheap fix; a stacked-matrix DUCT at those nodes is the principled one, and the literature says DUCT with a good evaluation is the practical winner even though it is not Nash-convergent.

### 2.4 AlphaZero-family training improvements that matter when fine-tuning an existing net

#### 2.4.1 KataGo paper (Wu 2019) [14]

| Technique | Mechanism | Measured factor |
|---|---|---|
| Playout cap randomisation | p=0.25 of turns: full search N=600 (later 1000), recorded; else fast search n=100 (200) with Dirichlet noise and other exploration off, not recorded. Resolves value (needs many games) vs policy (needs deep search) | 1.37× (fixed caps 100–600 all worse) |
| Forced playouts + policy target pruning | root child c gets infinite urgency until `N(c) ≥ sqrt(k·P(c)·ΣN)`, k=2; afterwards subtract up to n_forced from non-top children while their PUCT stays below the top child's; children left with one playout are pruned from the target | 1.25× |
| Global pooling | mean, mean×width, max per channel in 2–3 blocks and heads | 1.60× |
| Opponent-reply policy target | extra policy head, w=0.15, never used in play | 1.30× |
| Ownership + score targets | ownership w=1.5/b²; score pdf/cdf 0.02 each; weights "guesses sized to 10–40% of main gradients" | 1.65× |
| Other | Dirichlet α=0.03·361/legal, ε=0.25; root softmax temperature 1.03; FPU reduction 0.2 (0 at a noisy root); move-selection temperature 0.8→0.2, half-life = board width; first r raw-policy moves, r~Exp(mean 0.04·b²); 2.5% of positions branch to an alternative move; 5% of games branch early; value weight 1.5; no resignation — after 5 turns under 5% win rate, visits are reduced and samples kept with prob 0.1+0.9λ; batch 256, lr 6e-5 per sample, 6e-6 in the final phase; L2 3e-5 | product of ablations ≈ 9.1×; ~50× less compute than ELF |

#### 2.4.2 KataGo methods added since (KataGoMethods.md) [15]

- **Policy-surprise weighting**: full-search positions get weight 0.5 + 0.5·(KL(prior → target) share); written floor(w) times plus one with prob w−floor(w); "one of the larger improvements between g170 and earlier".
- **Auxiliary soft policy target**: head predicting `target^(1/4)` renormalised, weighted ~8× nominally; "greatly speeds up policy learning" (no number).
- **Short-term value/score targets** (EMA of future MCTS values at ~6/16/50-turn horizons) → **uncertainty head** (Huber loss on squared error) → **uncertainty-weighted playouts** (weight ∝ 1/uncertainty, floored) + **variance-scaled cPUCT** (`cpuctUtilityStdevScale 0.85`): about +75 Elo over the previous release, ~+50 over a retuned constant cPUCT.
- **Optimistic policy**: extra policy head trained on the same target weighted by `clamp(0,1, sigmoid(3(z_value−1.5)) + sigmoid(3(z_score−1.5)))`, used at play time for both sides: +40–90 Elo.
- **Subtree value bias correction**: `NodeUtility = NN − 0.35·ObsBias(bucket by last two moves and local pattern)`, exponent 0.8: +30–60 Elo.
- Root policy temperature 1.25→1.1 early-game schedule; shaped Dirichlet noise (half uniform, half on high-logit moves; "suggestive only"). Current defaults from `setup.cpp`: `cpuctExploration 1.0`, `cpuctExplorationLog 0.45`, `fpuReductionMax 0.2`, `chosenMoveTemperatureEarly 0.5 → 0.1`, half-life 19, `subtreeValueBiasFactor 0.45`, `uncertaintyCoeff 0.25` [18].

#### 2.4.3 Gumbel AlphaZero / MuZero (Danihelka, Guez, Schrittwieser, Silver, ICLR 2022) [23]

- Problem: training the policy toward root visit counts "does not guarantee a policy improvement, especially when using small numbers of simulations"; root Dirichlet noise "can produce a worse policy"; acting with the best of the top-n most probable actions can also be worse (counterexample q=(0,0,1), π=(0.5,0.3,0.2), n=2).
- Root: sample n actions **without replacement** via Gumbel-Top-k on `g(a) + logits(a)`; **Sequential Halving** over log2(m) phases; final action = argmax `g + logits + σ(q̂)`, `σ(q̂) = (c_visit + max_b N(b))·c_scale·q̂` with `c_visit 50`, `c_scale 1.0` (0.1 on Atari); **completed Q-values** fill unvisited actions with a mixed value estimate; **improved policy** `π' = softmax(logits + σ(completedQ))` distilled with `KL(π', π)` — "trains all actions, not only the action taken". Non-root selection is deterministic: argmax `π'(a) − N(a)/(1+ΣN)`.
- Results: MuZero "fails to learn from 16 or less simulations" on 9×9 Go; Gumbel MuZero learns from 2; 24.3× training-step speed-up at n=4 and 5.9× at n=32 (Table 1); matches or exceeds at n=400 on 19×19 Go and chess. The authors "validated that Gumbel MuZero and Gumbel AlphaZero keep improving, even when learning from two simulations" (the AlphaZero evidence in the text is chess at n=400).
- The "Finding the Time to Think" planners use mctx's Gumbel MuZero policy [2]; Go-Exploit reports PCR and forced playouts + pruning each raised AUC further on top of its own method (figures only) [37].

#### 2.4.4 Fine-tuning schedules and loss weights (what the numbers were) [14][38]

- CrazyAra supervised: value weight α=0.01 "to avoid overfitting to the training values" (small human dataset; value generalisation gap larger than policy's); one-cycle lr max 0.35 → transfer-learning stage max **0.01** (35× lower) with all parameters trainable; move accuracy 46.3%→56.6% [38]. KataGo RL: value weight 1.5, lr 6e-5/sample dropping to 6e-6 for the final net [14]. Hivemind's RL config (`src/hivemind/config/train_config.py::rl_train_config`): NAG, max lr `0.005 / div_factor` (0.0025 at the default div_factor 2), min `0.00005 / div_factor`, wd 1e-4, batch 512, 1 epoch, policy 0.7 / value 0.15 / wdl 0.15 / plys-to-end 0.01, soft (non-sparse) policy labels, `q_value_ratio 0`.
- Temperature/KL schedules in use: KataGo move temperature 0.8→0.2 (half-life = board width) and root policy temperature 1.25→1.1 [14][15]; Lc0 play defaults temperature 0 with optional linear decay and endgame cutoff [11]; Hivemind self-play (`engine/src/tools/selfplay.h`): `mctsTemperature 1.0 × 0.93^(macroPly/2)` until macro-ply 20, first ~8 (max 30) macro-plies from the raw policy with 5% chance of a high temperature, Dirichlet α 0.3 ε 0.25, node jitter ±5%, resign at −0.90 for 3 plies with resignation disabled in 10% of games.

### 2.5 Evaluating fairly [31]–[36]

- **Pentanomial model.** A reversed-colour pair is one trial with outcomes {0, ¼, ½, ¾, 1}. Game outcomes within a pair are correlated (shared opening bias), so the trinomial model "overestimates Elo confidence intervals" and needs more games: in Van den Bergh's example (elo1=10, true 10, draw_elo 327, biases [−90, 200]) pentanomial SPRT passes in 1809 expected games vs 2191 trinomial, ~20% longer [32]; the gap between the two variance estimates measures RMS opening bias [31]. Fishtest FAQ: "unbalanced books are definitely better than balanced books for engine tests" [35].
- **Normalised Elo** [33]: `t_n = (µ − ½)/σ_pg` (pair outcomes scored 0..1, σ multiplied by √2), `e_n = 347.43·t_n`; with a balanced book `e_n = e_l·√(1−d)` where d is the draw ratio — at d=0 normalised and logistic Elo coincide. Expected worst-case SPRT duration at α=β=0.05: `T = 1046535/(e_n1 − e_n0)²` games (Δ=1: 1.05M; 2: 262k; 3: 116k; 4: 65k; 5: 42k; 6: 29k); when the truth sits on a bound, the numerator is 639770. LLR: `LLR = (N/2)·log((1 + (t̂ − t0)²)/(1 + (t̂ − t1)²))`, valid for both models; clamp `LLR/N` to `±(t1−t0)/2` early on when σ̂ is tiny. Bounds `LA = log(β/(1−α))`, `LB = log((1−β)/α)` [34].
- **Elo estimates from passed SPRTs are biased upward**; only the population of all tests is unbiased [35]. `LOS = Φ((W−L)/√(W+L))` ignores draws [34].
- **Hardware fairness.** Fishtest standardises on `nodestime=600` so "results are independent of engine speed"; STC 10+0.1, LTC 60+0.6; example passed non-regression tests used 38k (STC) and 91k (LTC) games [36].
- Hay et al. also warn that with 1000-game experiments a 64% win rate was "almost always better" — i.e. the effect sizes that are cheap to measure are tens of Elo, not single digits [28].

---

## 3. Implications for the engine

For each idea: what original Hivemind (stock binary + it04) already captures, how I know, and the concrete change.

| # | Idea | Does stock Hivemind capture it? | Concrete change |
|---|---|---|---|
| 1 | Real clock values as inputs (not a yes/no flag) | **No.** `board2planes.py` has 37 planes per board; time is one boolean plane (31, `TimeAdvantage`). `uci.cc` ignores `wtime/btime`. | Spec v1 planes 37–40 (own/opp `min(t/300,1)`, `exp(−t/10)`) do this. Add a **relative** plane per board: Stockfish's only clock-aware term is the *difference* [4], and the flag the network currently has is a thresholded difference, so the widened net will otherwise have to rebuild it from two saturating planes. Note (critic): the difference that matters in bughouse is the *diagonal* one (our seat vs the partner's opponent), which is what stock's flag already thresholds (`board.py::time_advantage`), not the same-board `t_us − t_them` written here; see 02 §2.2 for why there is exactly one such number. |
| 2 | How long each on-turn side has been thinking (t_AS / action in flight) | **No.** No elapsed-time input; the tree has no notion of elapsed time. | Spec v2 planes 41–42 (`min(e/10,1)`, `1−exp(−e/2)`) are exactly Xiao et al.'s `t_AS` [30] and Ramstedt's action-in-flight [3]. Keep them. Also expose the **opponent seat's elapsed time** on the board where *they* are on turn (the plane is "whoever is on turn", so this is already covered once per board — confirm the encoder fills it for the opponent's turn, not only ours). |
| 3 | Expected remaining think time of the opponent seat (VTG analogue) | **No.** | Cheap: per-seat EMA of observed think times, sent in the `clocks` suffix as `expect <ea> <eb>` ms and encoded as `min(x/10,1)`; used in the tree (item 6). Learned later: a 2-output **opponent think-time head** trained on recorded self-play (log-normal mean/var of the opponent's next think time). Xiao et al. found the remaining-execution representation the least hyper-parameter-sensitive [30]. |
| 4 | Budget per decision is state-dependent; learn it on return, frozen planner | **No.** Budget is `clock/k` in the arena; the engine has no time manager. | Spec v2's `think_gain` head is a *supervised proxy* ("move changes at 4N"). Upgrade to the Muppidi recipe [2]: a gate over budgets `{raw policy (0 nodes), 16, 50, 200, 800}` with inputs = frozen trunk global-pool features + root value + 4 clock planes + elapsed planes, trained with PPO on game outcome (variable-duration GAE γ^k) in the jittered arena with the network frozen. Budget: their Speed Go gate took 500 PPO updates × 256 envs × 32k steps; at our ~10 games/s per GPU this is well inside 10k GPU-hours. Keep the hand rule `(0.5+1.5p)` as the baseline the gate must beat. |
| 5 | A policy-only "committed action" when time is critical | **Partly.** `go nodes N` with small N exists; the arena never plays the raw policy. | Add budget 0 (raw policy argmax, ~2 ms) to the budget set [2]; the gate learns when the position is "ghost-adjacent" (both low clock and forced). Also KataGo's `obviousMoves` rule — `obv = min(exp(−H/0.30), exp(−KL/0.15))` [17] — is a free heuristic baseline that needs only the policy head. |
| 6 | Durative, asynchronous tree: whoever is ready first acts next | **No.** Stock tree alternates our joint action / their joint action with frozen clocks; v1 `TreeMoveMs` charges every clock the same ms per macro-ply. | Replace the constant with per-seat durations: at a node, each on-turn seat has a "ready time" = its expected think time (item 3 for them; for us, the budget chosen); advance the state to the minimum, charge all running clocks that amount, let that seat (only) act; others wait [27]. A pass by us = "set our ready time to the next event". This makes sitting meaningful in the tree: it only pays when the modelled opponent is expected to move first. Alternate the order at true ties [27]. |
| 7 | Decoupled selection at simultaneous nodes | **No.** Our two boards are one joint action (fine: same agent); the opponent is sequentialised. | Low priority: given asynchrony is modelled (item 6), true simultaneous nodes are rare (exact ties). If they matter, DUCT with randomised tie-breaking at those nodes is the empirically best cheap option [25][26]; do not bother with Exp3 (weakest in every study). |
| 8 | Stop early when the search is already sure | **Partly.** Early exit only on a proven mate/solved root (`should_exit_early_winning`, `ENABLE_MATE_EARLY_EXIT`); no smart pruning, KLD, or futility. With `go nodes` nothing is saved anyway. | Once the engine runs on a time budget (`go movetime` with real clocks), add Lc0 smart pruning (factor 1.33) [6][10] and the KataGo futility stop [17]; both save clock exactly when the think-time head would say "no gain". Report `kldgain` per 100 nodes in `info` so the arena can use it as a cheap `think_gain` proxy and as a label-quality check for the head. |
| 9 | Spend more when the eval is falling / best move unstable | **No.** | Log `bestMoveChanges` and root-Q drift per search (Stockfish factors [5]); feed them to the gate (item 4) as inputs rather than hand-coding multipliers. |
| 10 | Hard per-move cap, lag buffer, overhead | **Arena only** (`clock_k` clamp). | Keep a cap of ≤30% of remaining clock per search (Lc0 smooth [7]) and a lag buffer (KataGo `lagBuffer`) in the arena controller even in free mode — the learned gate must not be able to flag by one bad choice. Measure the real UCI/IPC overhead per move and subtract it (Lc0 default 200 ms is for networked play; ours should be ~a few ms). |
| 11 | Nodes-as-time for reproducible training and eval | **Unknown/partial.** The arena charges think time per search (`serial`, jitter); whether the charged time is wall-clock or nodes×fixed-nps was not verified in this pass. | Charge simulated time = `nodes / nps_fixed` everywhere (self-play and gate), as fishtest's `nodestime` does [36]; keep a wall-clock mode only for the final match against stock. Otherwise GPU contention on the cluster leaks into results. |
| 12 | Playout cap randomisation | **No.** `nodeRandomFactor 0.05` is ±5% uniform jitter around 800 nodes; every position is recorded. | Implement PCR [14]: p=0.25 full searches (record, Dirichlet on), else fast searches at N/5–N/6 (no record, no noise). With our budgets a natural pair is (800, 150). Expect ~1.3× data efficiency. Note the interaction with out-of-step play: the *charged* think time should follow the realised node count so fast searches also make the team faster on the clock — this is free, realistic asynchrony. |
| 13 | Forced playouts + policy target pruning | **No** (no such code in `searchthread.cc`/`selfplay.cc`; `rootDirichletAlpha` in `RuntimeConfig` defaults 0 and is set only in self-play). | Add at the root in self-play: force `sqrt(2·P·ΣN)` visits per child, prune the forced visits from the target [14]. Cheap and 1.25× in KataGo; matters more for us because the per-board action space includes drops and most low-prior moves never get a visit at 800 nodes. |
| 14 | Gumbel policy target for small budgets | **No.** Soft target = root visit distribution (`sparse_policy_label False`); joint policy from visit counts. | Our play budgets are 50–200 nodes on a 60 s clock (`beat_stock.md`), exactly where visit-count targets stop being improvements [23]. Implement Gumbel root search for self-play: sample m per-board actions without replacement, Sequential Halving, completed-Q, target `softmax(logits + (50+maxN)·q)`. Start with the per-board heads (the joint head is off, `joint_policy_loss_factor 0`). This is the single change most likely to make low-node self-play actually improve the net. |
| 15 | Auxiliary targets | **Partly.** WDL and plys-to-end heads exist and are trained in RL (`use_wdl`, `use_plys_to_end`; search uses `MOVES_LEFT_DISCOUNT 0.05`). No ownership/score/opponent-reply/arrival targets. | Spec v2's `arrivals` head is the bughouse analogue of KataGo's opponent-reply/ownership targets [14]. Add (a) **opponent-reply policy** on each board (w=0.15, never used in play) — it also feeds item 3; (b) **short-term value targets** (EMA of root Q over the next ~6 and ~16 team decisions) and the **uncertainty head** [15], which then enables uncertainty-weighted playouts; (c) **soft policy target** `target^(1/4)` with high weight [15]. Weights sized to 10–40% of the main gradients, as KataGo did. |
| 16 | Policy-surprise sample weighting | **No.** | Weight recorded positions by `0.5 + 0.5·KL(prior → target)` share [15]; trivial in `convert_selfplay_data.py`. Bughouse tactics (drop mates) are exactly the "surprising" positions. |
| 17 | Search-side upgrades independent of training | **Partly.** Dynamic FPU (reduction 1.0), CPUCT 3.0/19652 (Lc0-2019 values [10]), draw contempt 0, Q-veto, MCGS transpositions. No variance-scaled cPUCT, uncertainty weighting, subtree bias, optimistic policy. | Tune `CPUCT_INIT` first (KataGo: ~25 Elo from retuning alone [15]); then variance-scaled cPUCT (needs per-node utility variance, no new head) [15]. Subtree value bias needs a bucketing key (last move + drop type is natural). Each is a separate SPRT (item 19). |
| 18 | Fine-tuning learning rate | **Partly.** RL config max lr `0.005 / div_factor` (NAG; 0.0025 at div_factor 2), one epoch per iteration. | Precedent says 10–35× below the from-scratch rate when transferring (CrazyAra 0.35→0.01 [38]; KataGo 6e-5→6e-6 for the final net [14]). With widened zero-initialised stems and new heads, run the first iteration with the trunk at ≤1e-4 and the new weights at the full rate; check it04 parity does not drift on held-out positions before raising. |
| 19 | Evaluation statistics | **Partly.** Paired openings with swapped sides (`arena/openings.paired_schedule`, `pool.py`), 1,600-game rows, ±0.035 quoted as a 95% range, replicate-with-new-seed rule (`beat_stock.md`). **Corrected:** the pair correlation *is* handled: `arena/stats.py` computes its 95% intervals with a cluster bootstrap that resamples whole opening pairs, not games ("Games that share an opening pair are not independent"), which is the non-parametric equivalent of pentanomial scoring. There is no `cluster/results.sh`; what is missing is a sequential test (GSPRT) and nodes-as-time, not pair-aware variance. | Score pairs pentanomially [31][32]; run gates as GSPRT with bounds in normalised Elo (≈ logistic here, d≈0) using LLR (4.14) [33]. Pick bounds by budget: [0, 20] Elo ≈ 2.6k games worst case, [0, 10] ≈ 10.5k, [0, 5] ≈ 42k. Every "replicate" is still required because pass-Elo is biased [35]. Nodes-as-time (item 11) for all gates. |
| 20 | The random think-time jitter (0.5–2.0) | **Training data only.** | Keep it for data diversity, but note random budget allocation is below every fixed budget in play [2]; the gate (item 4) must be trained *with* the jittered opponent but *without* jitter on its own chosen budgets, or it learns to compensate for noise that will not exist at test time. |
| 21 | Spend less when clearly winning / more when ahead on time | **No.** | Chess engines never spend more when ahead on time [4][16]; bughouse should. Rather than a rule, the gate (item 4) sees both clocks and can learn it; as a sanity baseline add KataGo's `searchFactorWhenWinning` (0.4 at |Q|≥0.95) [20][21] to the arena controller to see whether the gate at least matches it. |

Priority given 10k GPU-hours: **14 (Gumbel target) → 12/13 (PCR, forced playouts) → 6 (durative tree) → 4/5 (return-trained gate with a policy-only option) → 15/16 (aux targets, surprise weighting) → 19/11 (pentanomial SPRT, nodes-as-time) → 17/18.**

---

## 4. Sources

Web, read as primary text unless noted. Inline citations [n] refer to these.

1. https://arxiv.org/abs/2606.26463 — "Finding the Time to Think: Learning Planning Budgets in Real-Time RL" (Muppidi, Darwish, Cope, Henriques, Foerster, v2 27 Jun 2026): abstract, authors, framing (variable-delay real-time RL, gating policy over a frozen planner).
2. https://arxiv.org/html/2606.26463 — full text and appendices: gate inputs/architecture, PPO hyper-parameters, budget sets, per-game results, strict-timeout analysis, state-dependence findings, limitations.
3. https://ar5iv.arxiv.org/html/1911.04448 — Ramstedt & Pal, "Real-Time Reinforcement Learning" (NeurIPS 2019): RTMDP definition, Theorems 1–2, Lemmas 1–2 (state-value vs action-value under delay).
4. https://raw.githubusercontent.com/official-stockfish/Stockfish/master/src/timeman.cpp — Stockfish optimum/maximum time formulas and constants, opponent-clock term, ponder bonus.
5. https://raw.githubusercontent.com/official-stockfish/Stockfish/master/src/search.cpp — per-iteration time factors (fallingEval, best-move stability, instability, effort share) and stop conditions.
6. https://lczero.org/blog/2018/09/time-management — Lc0 2018 time-management post: time curve, slowmover 2.4, move overhead, smart pruning mechanism, ponder behaviour.
7. https://draft.lczero.org/dev/old/timemgr — PR1195 "smooth" time manager: node-target budgeting with tree-reuse and timeuse EMAs, parameter table.
8. https://raw.githubusercontent.com/LeelaChessZero/lc0/master/src/search/classic/stoppers/legacy.cc — legacy time manager: moves-to-go curve (51.5, 7.0), first-move bonus, piggy bank, slowmover, deadline.
9. https://raw.githubusercontent.com/LeelaChessZero/lc0/master/src/search/classic/stoppers/factory.cc — `TimeManager` default `legacy`, available managers, `MoveOverheadMs 200`.
10. https://github.com/LeelaChessZero/lc0/wiki/Lc0-options — smart-pruning-factor 1.33, kldgain options, 2019 cpuct/fpu/policy-temperature defaults.
11. https://raw.githubusercontent.com/LeelaChessZero/lc0/master/src/search/classic/params.cc — moves-left head parameters, current CPuct/FPU/temperature/StickyEndgames defaults with help strings.
12. https://draft.lczero.org/dev/wiki/lc0-options/ — same option set (Dec 2019 snapshot), fpu-reduction 1.2, policy-softmax-temp 2.2.
13. https://lczero.org/dev/wiki/technical-explanation-of-leela-chess-zero/ — glossary: KLD gain used in T50 self-play to spend more time when the policy was bad (secondary, brief).
14. https://ar5iv.arxiv.org/html/1902.10565 — Wu, "Accelerating Self-Play Learning in Go" (KataGo): PCR, forced playouts/target pruning, global pooling, auxiliary targets, self-play settings, ablation factors.
15. https://raw.githubusercontent.com/lightvector/KataGo/master/docs/KataGoMethods.md — later KataGo methods with mechanisms and Elo gains (surprise weighting, soft policy target, uncertainty, dynamic cPUCT, optimistic policy, subtree bias).
16. https://raw.githubusercontent.com/lightvector/KataGo/master/cpp/search/timecontrols.cpp — KataGo base time allocation (turns-left estimates, Fischer/byo-yomi rules, lag buffer).
17. https://raw.githubusercontent.com/lightvector/KataGo/master/cpp/search/searchtimehelpers.cpp — midgame shaping, obvious-move factor, tree carry-over softplus, futile-visits stop (formulas).
18. https://raw.githubusercontent.com/lightvector/KataGo/master/cpp/program/setup.cpp — default values and ranges for the above and for search/exploration parameters.
19. https://raw.githubusercontent.com/lightvector/KataGo/master/cpp/search/searchparams.h — parameter comments (what each time/search knob means).
20. https://raw.githubusercontent.com/lightvector/KataGo/master/cpp/configs/gtp_example.cfg — lagBuffer 1.0, searchFactorAfterOnePass 0.5, searchFactorWhenWinning 0.4 @ 0.95, resignation settings.
21. https://raw.githubusercontent.com/lightvector/KataGo/master/cpp/program/playutils.cpp — `getSearchFactor` (play faster when winning) formula.
22. https://raw.githubusercontent.com/lightvector/KataGo/master/cpp/command/gtp.cpp — `delayMoveScale` humanlike delay, dynamic playout-doubling advantage.
23. https://object.cloud.sdsc.edu/v1/AUTH_da4962d3368042ac8337e2dfdd3e7bf3/ml-papers-txt/ICLR/2022/policy_improvement_by_planning_with_gumbel__15d5fc26.txt — Danihelka et al., "Policy improvement by planning with Gumbel" (ICLR 2022), text mirror: Gumbel-Top-k, Sequential Halving, completed Q, σ constants, few-simulation results.
24. https://ar5iv.arxiv.org/html/1310.8613 — Lisý, Kovařík, Lanctot, Bošanský, "Convergence of MCTS in Simultaneous Move Games" (NIPS 2013): SM-MCTS template, Hannan consistency, convergence theorem, RM vs Exp3.
25. https://dke.maastrichtuniversity.nl/m.winands/documents/sm-journal.pdf — Bošanský, Lisý, Lanctot, Čermák, Winands, "Algorithms for computing strategies in two-player simultaneous move games" (AIJ 2016; PDF extracted locally): DUCT definition, non-convergence and tie-breaking, offline vs online results, conclusions.
26. https://dke.maastrichtuniversity.nl/m.winands/documents/sm-tron-bnaic2013.pdf — Lanctot, Wittlinger, Winands, Den Teuling, "MCTS for Simultaneous Move Games: A Case Study in Tron" (BNAIC 2013; PDF extracted locally): sequential vs stacked-matrix models, variant win rates (DUCB1T 62.3%).
27. https://cdn.aaai.org/ojs/12527/12527-52-16049-1-2-20201228.pdf — Churchill, Saffidine, Buro, "Fast Heuristic Search for RTS Game Combat Scenarios" (AIIDE 2012; PDF extracted locally): durative moves, ready times, Alt/RAB sequentialisation, ABCD, results.
28. https://ar5iv.arxiv.org/html/1207.5879 — Hay, Russell, Tolpin, Shimony, "Selecting Computations: Theory and Applications" (UAI 2012): metalevel MDP, VOI stopping, myopic-stops-too-early, blinkered policy, 64% vs UCT in 9×9 Go.
29. https://ar5iv.arxiv.org/html/2002.04335 — Sezener & Dayan, static/dynamic value of computation for MCTS: definitions, premature-stopping proposition, costs.
30. https://ar5iv.arxiv.org/html/2004.06089 — Xiao et al., "Thinking While Moving" (ICLR 2020): concurrent-control Q with previous action, t_AS, VTG; ablation and results.
31. https://official-stockfish.github.io/docs/fishtest-wiki/Fishtest-Mathematics.html — pentanomial model, GSPRT, normalised Elo rationale, Elo estimation approach.
32. https://github.com/vdbergh/pentanomial — pentanomial SPRT README: BayesElo inputs, example frequencies, 1809 vs 2191 games (~20%).
33. https://www.cantate.be/Fishtest/normalized_elo_practical.pdf — Van den Bergh, "Comments on normalized Elo" (PDF extracted locally): definitions, `e_n = e_l·sqrt(1−d)`, duration formula `T = 1046535/Δ²`, LLR formulas and clamp.
34. https://www.chessprogramming.org/Match_Statistics — Elo from score, LOS, SPRT LLR and bounds, note on paired games and 5-nomial variance.
35. https://github.com/official-stockfish/fishtest/wiki/Fishtest-faq — biased Elo of passing patches, unbalanced books better than balanced.
36. https://official-stockfish.github.io/docs/fishtest-wiki/Creating-my-first-test.html — STC/LTC definitions, `nodestime=600` convention, example game counts.
37. https://ar5iv.arxiv.org/html/2302.12359 — "Targeted Search Control in AlphaZero" (Go-Exploit): independent confirmation that PCR and forced playouts + pruning add on top of other methods (figures only).
38. https://ar5iv.arxiv.org/html/1908.06660 — Czech et al., CrazyAra paper: time management section, MCTS constants, value weight 0.01 rationale, transfer-learning lr 0.01 vs 0.35.
39. https://cris.maastrichtuniversity.nl/portal/en/publications/monte-carlo-tree-search-variants-for-simultaneous-move-games(5e4c8118-4db7-49bf-b12d-d16ee90fe167).html — Tak, Lanctot, Winands (CIG 2014), record/abstract only (PDF not retrievable): DUCT best overall across nine games, ranking game- and opponent-dependent. **Weak source: abstract via search snippets.**

Local code read for the "does Hivemind capture it" column (not web): `third_party/hivemind/engine/src/search/search_params.h`, `search/agent.cc`, `search/searchthread.cc`, `environment/joint_action.h`, `interface/uci.cc`, `tools/selfplay.h`, `tools/selfplay.cc`; `third_party/hivemind/src/hivemind/domain/board2planes.py`, `config/train_config.py`, `training/data_loaders.py`; this repo's `arena/pool.py`, `arena/bench.py`, `arena/interfaces.py`, `docs/beat_stock.md`.

Not found / not read: the Tak et al. 2014 PDF; the "planning paralysis" mechanism in [2] (asserted, not analysed); Lc0's current self-play temperature schedule (only play-time defaults were documented); whether `cluster/results.sh` scores pairs pentanomially (grep negative, treated as unknown).
