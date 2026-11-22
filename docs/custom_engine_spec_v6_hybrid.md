# Custom engine spec v6: the hybrid (v3 tree chooses the move, the clock decides when)

Draft, 2026-11-21. Wave 2d. Builds on spec v4 (`docs/custom_engine_spec_v4_durative.md`) and v5
(`docs/custom_engine_spec_v5_continuity.md`) as built in `third_party/wave2-hivemind` (branch `talisman/wave2`, pinned
`72e79eb` in `cluster/wave2_engine.ref`; `docs/wave2_engine.md` there). Line numbers below are at `72e79eb`. Network
for every number quoted: twin-s-clock it13 (HF `aminwoo/bughouse-twin-s-clock@edaa246`).

## Why

| Evidence | Source |
|---|---|
| `nodur` (DurativeTree false, SitGate true: the v3 joint-action tree, sitting only when lead > 0) 0.436 [0.361, 0.491] vs it13 on PACE, 120 s; durative tree on 0.21-0.25; same-engine control 0.516 | `docs/custom_engine_spec_v5_continuity.md` "Why", `docs/wave2_tests.md` |
| The durative tree's core logic is sound (synchronous control matches v3 14/15); the loss is per-seat sequential action and states it13 never saw. Agreement with it13 does not rise with nodes | `docs/durative_diagnosis.md` D0, D2 |
| On 300 real positions nodur and it13 cannot be told apart (joint agreement with it13@32k 61/76/87% vs 61/76/89% at 1k/4k/16k; act-vs-sit 91-98%). The durative tree levels off at 58% and sits 14-23% of the time when behind (nodur and it13: 0-3%) | `docs/strength_harness.md` baselines |
| Local 30 s match nodur vs it13: 0.532 [0.459, 0.614], 170 games: PACE's 0.436 is not reproduced (candidates: PACE's 26% per-node charge, 1.3k vs 20k nodes per search, older arena) | `docs/strength_harness.md` |

So the v3 tree picks moves as well as it13, and the durative tree does not. What the durative model has that v3 lacks
is **when things land**: each opponent seat's remaining think time (`SeatPaceMs - SeatElapsedMs`), which seats are
sitting (`SeatSitting`), the four real clocks. v3 replaces all of that with one hard rule, `can_sit = lead > 0`
(`TimeState::can_sit`, `time_state.h:341`). The hybrid keeps the v3 tree for the move and uses the durative clock only
for the decision to act now or to wait.

**What this round cannot fix.** If nodur's PACE deficit is the per-node charge (0.0530 vs 0.0424 ms/node), no change of
sitting rule will close it. The primary yardstick is therefore **hybrid vs nodur on the same binary** (same cost per
node), and only second hybrid vs it13.

## What the code says (reading notes)

1. **v3 legality of a pass** (`joint_action.h:20-90`). A team on turn on one board may pass there (a "double sit")
   only with `teamHasTimeAdvantage`. A team on turn on both boards may pass on one of them if it has the advantage or
   if its move on the other board is a capture (`is_single_pass_legal`). The flag is `TimeState::can_sit(team)`, which
   on a v3 clocked state is `lead > 0`, whatever `SitGate` says (`with_sit_gate`, `time_state.h:305`: v3's tree cycles
   on pass-pass lines without the gate). In the tree the flag is read per leaf (`process_batch`, `searchthread.cc:1353`)
   and handed to `expand_leaf_node` -> `Node::try_init_and_expand` -> `JointCandidateGenerator::initialize`. The pass is
   always in the action list (`prepare_actions` appends `MOVE_NONE`, `searchthread.cc:567`); illegal pairs get
   `jointPrior -1`.
2. **The v3 gate equals the sit-off rule when each team is on turn on one board.** Without increment
   `Aw + Ab = Bw + Bb`, so the white team's lead `((Aw - Bw) + (Bb - Ab)) / 2` equals `Aw - Bw`, the difference of the
   two clocks on turn when Aw and Bw are to move. "Sit only when lead > 0" is "only the higher on-turn clock may sit",
   which is what sit-off adjudication rewards (arena `RULES.md`: after 5 s of both on-turn seats sitting, the lower
   clock loses). The gate is right about **mutual** sitting. It is wrong about **waiting for a landing that is
   coming anyway**: an opponent seat that is thinking will land in `pace - elapsed` ms whether we move or not, and
   during that wait both on-turn clocks run, so the lead does not move.
3. **What a v3 pass costs.** `TimeState::advance(team, onBoth)` (`time_state.h:376`) moves the lead only when the acting
   team is on turn on both boards. A pass on our only on-turn board is free in lead terms; the price comes one ply
   later, when the opponent's move hands its board to our partner and we are on turn on both boards. This matches the
   seat clocks (point 2). So under v3 the value of a pass is already priced correctly by the tree's own lines: after
   our pass the opponent acts, or sits back (legal for it: it leads), and then we, still behind, must move (gate).
4. **A v3 pass node is in distribution for the network.** After our pass the boards are unchanged and the opponent team
   is to play, the position at every other node of a v3 tree. The durative tree's wait nodes are not (diagnosis D3:
   the tempo bias of +0.26).
5. **The durative clock already knows "when".** `TimeState::durative()` (`time_state.h:177`) builds, from the seat
   options, each on-turn seat's ready time: ours = `RootMoveTimeMs`, an opponent's = `max(1, pace - elapsed)`, with a
   sitting opponent marked (`seatSitting`) and the sit-off rule (`sit_off_forces_move`, `time_state.h:1104`). The arena
   sends all seat options to the wave 2 binary in every row, including `nodur` (`Wave2TeamController.time_options`,
   `arena/players/upstream.py:818`), and `UCI::current_clock()` (`uci.cc:851`) simply does not build the durative
   state while `DurativeTree` is false. Self-play builds it in `SeatClockGame::search_clock` (`seat_clock_game.h:491`),
   also only under `--durative-tree`.
6. **Root-only hooks have a precedent.** PlanCarry passes root-only data to the workers in
   `SearchParams::RuntimeConfig` (`rootPlanMove`, `rootPlanWeight`, `search_params.h:953`) and mixes it into the root
   priors in `Node::try_init_and_expand` (`node.h:745`). The hybrid's root data travels the same way.
7. **The final choice** is `Node::get_best_move_idx_with_q_weight` in `Agent::run_search` (`agent.cc:8440`), followed
   by the mate-race veto (`race_safe_alternative`) and selected-move certification
   (`certified_mate_free_alternative`). Proven mates and proven losses return earlier (`return_proven_mate`,
   `return_proven_loss`). The root provers run in v3 alternation, which is correct for a v3 tree.
8. **The arena turns a pass into a wait** of `min(PassCapK, 1.5) x RootMoveTimeMs` that any landing ends
   (`Wave2TeamController.default_sit_ms`), then searches again. Self-play's `SeatClockGame::commit_pass` ends a pass 1 ms
   after the next landing on the other board, or at `PassCapK x` the seat's move time.

## Common rules

* **Defaults are bit-for-bit.** With `HybridMode off` (the default) the engine is `72e79eb` byte for byte:
  `engine/tests/stock_parity.sh` 8/8, `engine/tests/wave2_parity.sh` 8/8, `engine/tests/wave2_game_parity.sh` 5/5. New
  state enters node keys, `operator==` and planes only when its option is on, as in v5.
* **v3 tree only.** Every hybrid mode needs `DurativeTree false` and `TimeModel clock`. With `DurativeTree true`
  `setoption` accepts the options and says once `info string Hybrid acts only with DurativeTree false`; self-play and
  tournament refuse the flag (`Wave2GameOptions::validate`). `SitGate` keeps its meaning: the hybrid adds root passes
  that the gate forbids, it never removes one the gate allows (H1's optional veto aside).
* **Root only.** The hybrid changes what our team may do at the search root and how the root's choice is made.
  Below the root the tree is the v3 tree, gate included.
* **No trigger, no change.** A search in which the hybrid's trigger (below) finds no board is the `nodur` search
  exactly. This is a stronger parity than the defaults and is tested (test 0.3).
* **Not in ponder or background searches**: `SearchOptions::isPonder` or `background` switches the hybrid off for that
  search (the permanent brain's clock is v3's `advance()` anyway).
* House style: as v5. Comment density and naming of `time_state.h` and `searchthread.cc`; one section in
  `docs/wave2_engine.md` ("Wave 2d: the hybrid"), filled in as each part lands; small commits; every behaviour an
  option plus a self-play/tournament flag; a knob without its switch is refused.

## The trigger: when waiting is on the table

Computed once per search in `Agent::run_search`, before the tree runs, from `SearchOptions::hybrid.clock` (the durative
state of this decision, built with `SitGate false`, below). New free function
`hybrid_sit_boards(Board&, Color team, const TimeState& v3, const TimeState& durative, const HybridKnobs&)` in
`search/hybrid.h` (new; header-only like `mate_candidate.h`), returning a two-bit board mask and, per board, the
expected wait `eventMs`. A board X gets its bit when all hold:

| # | Condition | Why |
|---|---|---|
| T1 | our team is on turn on X and has at least one legal move there | otherwise the pass is forced or meaningless |
| T2 | the v3 rules forbid a pass on X in some joint action: `!v3.can_sit(team)` (both shapes), or, on turn on both boards, every non-capturing partner move (the single-pass rule) | the hybrid only adds what the gate forbids; with lead > 0 nodur may already sit and the mask stays empty |
| T3 | an event is coming on the other board Y: the seat on turn on Y is ours (our partner, deciding in this search: `eventMs = RootMoveTimeMs`) or an opponent seat that is not sitting (`eventMs = durative.ready_ms(seat)` = `max(1, pace - elapsed)`); a committed opponent seat counts as not sitting | waiting is for a landing; with a sitting opponent nothing lands and the sit-off rule says the lower clock moves |
| T4 | `eventMs <= min(HybridMaxWaitMs, PassCapK x RootMoveTimeMs)` | a short wait only; also bounds the arena's wait (`default_sit_ms`) |
| T5 | `!durative.sit_off_forces_move(team)` on X | the durative rule for "behind against a sitter" (redundant with T3 for opponents; kept as the stated rule) |
| T6 | our seat on X has `clock >= max(HybridMinClockMs, 4 x (eventMs + RootMoveTimeMs))` | flag safety: v3 knows no absolute clock |
| T7 | after the pass on X (with any move on Y), the opponents have no mate in one on either board (`Agent::has_immediate_mate`, `agent.cc:832`, on the board after our pass) | cheap; a sit must never hand over a mate in one |

`TimeState` gains one accessor, `ready_ms(seat)` (durative only; the seat's ready time at the root), and
`sitting(seat)`; both read existing fields.

Telemetry (verbose): `info string Hybrid: sit boards <A|B|AB|-> event <msA>/<msB> mode <mode>`; when the mask is empty
the line says which condition failed first (`no move`, `gate allows`, `no event`, `opponent sitting`, `wait`, `sit-off`,
`clock`, `mate in one`). The arena parses it into the search result for counting (below).

Where the durative state comes from:
* **UCI.** New `UCI::current_hybrid_clock()` next to `current_clock()` (`uci.cc:851`): the same
  `TimeState::durative(v3.with_sit_gate(false), team, sideA, sideB, durativeConfig)` that `DurativeTree true` would
  build (seat clocks, paces, elapsed, sitting, arrivals, `RootMoveTimeMs`, `PassCapK`, D1 if set), returned only when
  `HybridMode != off`. Without seat options the clocks derive from the lead around 2^29 ms (T6 always holds) and the
  opponent's event is a full pace away (elapsed 0): the hybrid then still works, on paces alone.
* **Self-play and tournament.** `SeatClockGame::search_clock` is split: `v3_clock(...)` (the current non-durative return)
  and `durative_clock(..., sitGate)` (the current durative branch). Under `--hybrid-mode` the search gets
  `v3_clock` as `options.clock` and `durative_clock(..., false)` as `options.hybrid.clock`. Seat clocks are required
  (`--base-mix`); without them the hybrid is refused.

## H3 (default): event-gated root sit, priced by the v3 tree

`HybridMode gate`. The trigger makes the pass on the masked boards legal **at the root only**; the v3 tree values it
like any other action; the root takes it only if it beats the best action that moves by a margin.

**Exact behaviour.**
1. Mask from the trigger. Empty: the search is nodur's (no further change; test 0.3).
2. **Root legality.** `JointActionRules` gains `uint8_t hybridSitBoards = 0` (bit 0 = A, bit 1 = B; default 0 =
   today's rules; the struct grows from 5 to 6 bytes, check `JointCandidateGenerator`'s layout and `sizeof(Node)`). In `is_joint_action_legal` (`joint_action.h:56`), a pass on a
   board whose bit is set is legal as a double sit or a single pass whatever `teamHasTimeAdvantage` and the partner
   capture say. `RequireMoveOn` still wins (a seat that must move keeps its move). The mask reaches the generator
   through `RuntimeConfig::hybridSitBoards` (root only, like `rootPlanMove`), read in `Node::try_init_and_expand` when
   `m_depth == 0`. Nothing below the root sees it.
3. **Root prior floor.** The network's sit plane is upstream's (`lead_sit_right`), so when we are behind the pass
   prior is near 0 and the pass would get no visits. At the root, on each masked board, the pass entry is mixed as
   PlanCarry mixes a carried move: `p(pass) <- (1 - w) p(pass) + w`, others `(1 - w) p`,
   `w = HybridPassPriorPermille / 1000` (default 150), after the Dirichlet noise (`mix_root_plan`, reused).
4. **Children of a hybrid sit.** `select_and_expand`'s `time_after_action` (`searchthread.cc:1753`): at depth 0, an
   action whose pass on a masked board the gate would have refused gets `next.advance(...)` as today, then
   `next = next.with_hybrid_sit()`. `TimeState::hybridSit_` is a sticky bit: it survives `advance()`, enters
   `exact_key()` only when set (as `sitGate_` does), and changes nothing else. Its only job is to keep the subtree
   below a hybrid sit apart from the rest of the tree: after (our pass, their pass back) the boards, team and lead are
   the root's again, and without the bit `canonicalize_child` could link the grandchild to the root (a cycle in the
   graph). Below the root the gate applies as in v3: the opponent may sit back exactly when it leads, and then we,
   behind, must move. That line is the sit-off, priced by the network.
5. **The choice.** After the tree, in `run_search` where `bestIdx` is taken (`agent.cc:8440`): if the chosen action is a
   hybrid sit, keep it only if `Q(sit) >= Q(best moving action) + HybridMarginPermille / 1000` (default 20) and the sit
   has at least 10% of the root's visits; otherwise take the best child that is not a hybrid sit (same q-weighted
   rule, with an exclusion predicate added to `get_best_move_idx_with_q_weight`). "Best moving action" means the best
   child that is not a hybrid sit. Proven results bypass this: a proven mate or a proven loss return as today; a
   solved WIN child that is a hybrid sit is kept (the tree proved it with the opponent's sit-back in the lines).
   `race_safe_alternative` and certification then run on the result as today.
6. **Tree reuse.** If the mask is non-empty, an exact-hash retained root (`try_reuse_tree`) is not adopted (its
   expanded children were generated under the gate's rules); the search starts fresh. With `ucinewgame` per search
   (every arena row) this never matters.
7. **Report.** `info string Hybrid: chose <sit A|sit B|act> q_sit <q> q_act <q> visits <n>/<n> (<reason>)`, reason one
   of `margin`, `visits`, `proven`, `no sit chosen`. `Agent::hybrid_report()` returns the same for self-play.

**Why this is the default.**
* One tree, one network semantics. Every value compared at the root is a v3 value on v3 states, and the states after
  a pass are in the network's training distribution (reading note 4). The durative tree's known value errors (D1
  optimism, D3 tempo bias, the 58% ceiling) never enter.
* The opponent's best answer to our sit is in the tree: it may sit back when it leads, and the gate then makes us
  move. So the tree prices the sit-off itself (reading note 3), and the margin only covers noise.
* No node split: the budget is nodur's. No cost when nothing triggers.
* The durative model does exactly the "when" part: T3-T6 read the landing times, sitting flags and clocks that only
  the durative state has.

**Node and time budget.** Unchanged: the same `go nodes N` / `movetime` as nodur. The trigger costs one durative state
build and at most two `has_immediate_mate` calls (microseconds). The root may carry up to two more legal actions.

**What H3 cannot do.** It never sits when we lead (the mask is empty: the gate already decides there), and it never
vetoes a gated sit. It does not model *how long* the wait is beyond T4 (v3 has no clock), so a wait for a slow event
costs absolute clock that the tree does not see; T4 and T6 bound that.

## H1: sit decision by a durative probe

`HybridMode probe`. The v3 tree chooses the move; a small durative search compares acting with sitting.

**Exact behaviour.**
1. Mask from the trigger, with T2 dropped when `HybridVeto true` (then a board where the gate allows a sit is also
   probed, so the probe may turn a v3 sit into a move). Empty mask: nodur's search, all of the budget.
2. **Budget split.** Non-empty mask: the main v3 search gets `N_main = N - N_probe`, the probe
   `N_probe = max(HybridProbeMinNodes, round(N x HybridProbePermille / 1000))` (defaults 250 permille, 64 nodes).
   Under `movetime` the same split applies to the time. The main search runs first.
3. **Skip.** No probe (the remaining nodes are not spent: the search ends `N_probe` short and reports it) when the main
   root is solved or returned a proven mate or loss, or when the chosen action `a*` already passes on every masked
   board (and `HybridVeto` is off).
4. **Probe.** A second `run_search` with `options.clock = hybrid.clock` (the durative state, SitGate off, FlagLoss,
   InstantTransfer, PassCapK, D1 and D3 as the UCI options say; JointPlans allowed; TeamTreeReuse and PlanCarry ignored
   in the probe), `targetNodes = N_probe`, root provers off (`enableRootMateSearch`, `enableMateProbe`,
   `certifySelectedMove` false: they reason in v3 alternation), no noise, and a root action filter of at most four
   pairs: `a*` and, per masked board X, `a*` with X's half replaced by a pass (each only if legal under the durative
   rules; with `HybridVeto` and `a*` passing on X, the alternative is the best main-tree child that moves on X).
   The filter is `RuntimeConfig::rootActionFilter` (array of 4 move pairs, root only), applied in
   `JointCandidateGenerator::makeCandidate` at depth 0 (pairs outside it get `jointPrior -1`), with the priors of the
   allowed pairs set equal. The probe saves and restores the agent's continuity state (`rootNode`,
   `nextRootCandidates_`, `lastSearchHash_`, `teamTree_`, `plans_`, `mateContinuations_`) through a new
   `SearchOptions::probe` flag, so the main search's tree is what stays.
5. **Decision.** Sit on X iff `Q_dur(sit X) >= Q_dur(a*) + HybridMarginPermille / 1000`, each child has at least 10% of
   the probe's visits, and the sit child is not a solved LOSS (a solved WIN sit is taken). Both values come from the
   same durative tree, so its common biases (D1 level, the speed plane) cancel in the difference; D3 does not cancel
   fully (only one side has a wait node next), which is what the margin is for.
6. **Report.** `bestmove` is printed once, after the probe (`SearchOptions::printBestMove = false` for the main
   search). The last `info depth ... nodes` line reports `N_main + N_probe actually searched`, so any front end that
   charges searched nodes (the arena's `charge searched`) charges the probe. Plus `info string Hybrid probe: <n> nodes,
   q_act <q>, q_sit <q>, chose <...>`.

**Pros / cons against H3.** The durative tree prices the wait in real time (landings, flags, delayed transfer), which
H3 cannot. But its values are the ones the diagnosis could not trust, it sits more when behind (14-23%), and it costs
a quarter of the nodes whenever the trigger fires. Run it only if H3 shows no gain, or as the ablation that tells
whether the durative values carry information H3 lacks.

## H2: v3 tree with priced passes

`HybridMode priced`. One tree; the root's hybrid sits are children whose subtree runs on the durative clock.

**Exact behaviour.**
1. Mask and root legality and prior floor as H3 (steps 1-3). With `HybridPricedScope all` also the passes the gate
   allows are priced (default `forbidden`: only the gate-forbidden ones, so nodur's sits keep their v3 values).
2. At depth 0, `time_after_action` gives a priced pass child `durative_time_after(hybrid.clock, board, team, action)`
   (`searchthread.cc:636`) instead of `advance()`. The child's state is durative, so everything below follows the v4
   rules (ready times, wake-ups, sit-off, flags) through the existing branches of `collect_batch` and
   `process_batch`. Keys separate the two kinds of state already (`exact_key` mixes the durative block). The
   durative root state travels as `RuntimeConfig::hybridRootClock` (`std::optional<TimeState>`, root only).
3. The choice as H3 step 5, with `HybridMarginPermille` covering the model gap.
4. Retained candidates under priced children carry durative keys and never match a v3 root; nothing else changes.

**Risks.** The root compares Q from two time models in one PUCT selection: durative children are optimistic (D1,
+0.05 without OwnPaceBelowRoot) and spread thinner (two to four plies per joint move), so the comparison is biased in
a direction that depends on depth. Durative nodes cost about 20% more (0.246 vs 0.206 ms/node on the Mac). This is
the variant most likely to sit for the wrong reason; it is specified for completeness and built last, if at all.

## Options

UCI (all advertised in the `uci` handshake; parsed in `UCI::setoption` next to `Continuity`, `uci.cc:749`):

| UCI option | Type, default | Self-play / tournament flag | Modes | Meaning |
|---|---|---|---|---|
| `HybridMode` | combo `off` `gate` `probe` `priced`, off | `--hybrid-mode <mode>` | - | the variant: H3 `gate`, H1 `probe`, H2 `priced` |
| `Hybrid` | check, false (preset) | `--hybrid <bool>` | - | `true` sets `HybridMode gate`, `false` sets `off`; knobs keep their values; last write wins |
| `HybridMaxWaitMs` | spin 0..10000, 1000 | `--hybrid-max-wait-ms <ms>` | all | T4: longest expected wait for an event |
| `HybridMinClockMs` | spin 0..600000, 3000 | `--hybrid-min-clock-ms <ms>` | all | T6: our seat's clock floor |
| `HybridMarginPermille` | spin 0..1000, 20 | `--hybrid-margin-permille <n>` | all | how much better a sit must be (q units x 1000) |
| `HybridPassPriorPermille` | spin 0..1000, 150 | `--hybrid-pass-prior-permille <n>` | gate, priced | root prior floor of the pass on a masked board |
| `HybridProbePermille` | spin 0..900, 250 | `--hybrid-probe-permille <n>` | probe | probe share of the budget |
| `HybridProbeMinNodes` | spin 1..100000, 64 | `--hybrid-probe-min-nodes <n>` | probe | probe floor |
| `HybridVeto` | check, false | `--hybrid-veto <bool>` | probe | also probe boards where the gate allows a sit |
| `HybridPricedScope` | combo `forbidden` `all`, forbidden | `--hybrid-priced-scope <s>` | priced | which root passes get a durative subtree |

Knobs need their switch (self-play `validate` refuses a non-default knob with `--hybrid-mode off` or with a mode it does
not act in; UCI accepts and says `info string <knob> acts only with HybridMode <modes>`). `--hybrid-mode` needs
`--durative-tree false` and `--base-mix`. `changes_search()` counts `hybridMode != off`. `OwnPaceBelowRoot` and
`WaitNodeActingEval` are allowed beside a hybrid mode (they shape the probe's or the priced subtree's durative clock;
the self-play check that ties them to `--durative-tree` accepts `--hybrid-mode probe|priced` instead).

State: `UCI` gains `HybridKnobs hybrid` (mode and knobs); `SearchOptions` gains `struct Hybrid { HybridKnobs knobs;
std::optional<TimeState> clock; }` and, for H1, `bool probe`, `bool printBestMove = true`. `RuntimeConfig` gains
`hybridSitBoards` (H3, H2), `rootActionFilter` (H1), `hybridRootClock` (H2), all root-only and empty by default.

Python and arena:
* `hivemind.wave2.Options` gains the fields; `rl-loop` is not changed this round (the hybrid is a play-time rule; the
  training targets stay the v3 tree's visits, below).
* `arena/players/upstream.py`: `WAVE2_HYBRID_FLAGS` (`hybrid_mode` -> `HybridMode` "off", knobs at the engine
  defaults), merged into `WAVE2_FLAGS`, sent only when not default (as the continuity flags, so old rows send exactly
  what they sent); `WAVE2_INT_RANGES` gets the spins; a preset `hybrid` in `WAVE2_PRESETS` expanding to
  `hybrid_mode: "gate"`. A row with the preset must also set `durative_tree: false` and `sit_gate: true` (the wave 2
  arm defaults `durative_tree` to true): `wave2_flag_options` refuses `hybrid_mode != off` with `DurativeTree true`.
  `_parse` reads the `Hybrid:` info strings into `SearchResult.hybrid` (`sit`, `act`, `skipped`), and
  `Wave2TeamController` counts `hybrid_sits` and `hybrid_triggers` per game for the game log.
* `tools/strength_harness.py`: configs `hybrid` (`{"durative_tree": false, "sit_gate": true, "hybrid": true}`),
  `hybrid_probe`, `hybrid_priced`; the report adds "trigger rate" and "hybrid sit rate" on all positions and on the
  sit-or-act subset.

## Self-play and tournament

* `run_selfplay` and `run_tournament` (seat-clock games only) pass `options.hybrid` with `durative_clock(..., false)`.
* Self-play samples its action from the root visits (`select_action(edges, temperature)`). A sampled hybrid sit is
  played only if the margin rule of H3 step 5 holds; otherwise the sample is redrawn over the children that are not
  hybrid sits. At temperature 0 the action is `run_search`'s result. H1: the probe's decision replaces the sampled
  action's half on X (a sit) only if X's seat is deciding now.
* Training records are unchanged: planes and the policy target are the main v3 tree's (`marginal_policy`,
  `joint_policy` of its root). A hybrid sit is the game's action, not a target. This keeps HVM6 the same format and the
  network's sit plane unchanged; whether to teach the network the new sits is a later decision.
* Telemetry: `selfplay hybrid: <sits> sits in <triggered> triggered of <searches> searches, kept <k> / refused margin
  <m> visits <v>` and the same per tournament game; `seat_clocks.tsv` unchanged.

## Interaction with existing flags

| Flag | Under H3 (gate) | Under H1 (probe) | Under H2 (priced) |
|---|---|---|---|
| `DurativeTree` | must be false (refused or inert otherwise) | same | same |
| `SitGate` | stays; the hybrid only adds root passes the gate forbids | same; `HybridVeto` may also turn a gated sit into a move | same |
| `FlagLoss` | not in the v3 tree; T6 stands in | in the probe | in priced subtrees |
| `InstantTransfer` | not in the v3 tree (v3 transfers at once) | in the probe | in priced subtrees |
| `MateRaceByLanding` | inert (the v3 clock is not landing-aware; root provers stay on) | probe provers are off anyway | inert at the root; in subtrees as v4 |
| `PassCapK` | T4 bound | T4, probe passes | T4, subtree passes |
| `SeatClocksMs` / `SeatPaceMs` / `SeatElapsedMs` / `SeatSitting` | trigger only (T3-T6) | trigger and probe | trigger and subtrees |
| `SeatArrivalsMs` | ignored (narrow network) | probe (pending pieces under delayed transfer) | subtrees |
| `TeamTreeReuse`, `JointPlans`, `PlanCarry`, `Continuity` | need `DurativeTree true`: inert (UCI) or refused (self-play) | JointPlans acts inside the probe if set; the others are ignored by the probe | JointPlans acts in subtrees; the others inert |
| `OwnPaceBelowRoot` (D1) | no effect | probe | subtrees |
| `WaitNodeActingEval` (D3) | no effect | probe | subtrees |
| `BlockableCheckNotMate` (D5) | board rule, applies as today | same | same |
| `RequireMoveOn` | a required board is never masked | same | same |

## Risks

1. **Mutual sitting until flag or sit-off.** Two hybrids (self-play, `wave2-self`) can sit on the two boards at the same
   instant. Bounds: a hybrid sit needs an event (T3: the other seat not sitting); on its wake the other seat is
   sitting, T3 fails, so the seat that is behind moves at its next decision (the gate) and the one ahead may sit (the
   arena then adjudicates in its favour, as for nodur). The longest joint sit is therefore one wake period
   (`min(PassCapK, 1.5) x budget` in the arena, `PassCapK x` move time in self-play) plus one search, far under the
   5 s sit-off. Test 3.3 checks it in games.
2. **Sitting into a mate or a lost race.** T7 (mate in one), the tree's own lines, and the unchanged
   `race_safe_alternative` and certification. A deeper mate that only the durative landing order shows is not covered
   in H3 (v3 alternation).
3. **Over-sitting through the prior floor.** 15% prior on a pass the network thinks is illegal buys it visits; the
   margin and the 10% visit floor decide. Tuned on the harness, not in games (test 4.2 sweep).
4. **Time waste.** Each sit costs a full search before the wait and another after it. Lead-neutral while each team is
   on one board, but absolute clock goes; T4 and T6 bound it, and the match measures it.
5. **Prover blind spots.** The root loss scan and mate scan enumerate v3-legal actions only: they can call a root lost
   while a hybrid sit was not checked (conservative), and they never prove a hybrid sit wins (the tree may).
6. **Arena re-search budget.** A woken seat searches again from scratch (`ucinewgame` per search in every row): the
   wait is paid twice in nodes. Acceptable for a first measurement; tree reuse across the wait is v5 territory.
7. **H1 and H2 value errors**, as described in their sections.

## Acceptance tests

Numbering as v5. C++ in `engine/tests/test_hybrid.cc` (new); Python in the wave2-hivemind and main-repo suites.

**0. Parity and plumbing**
* 0.1 `stock_parity.sh` 8/8, `wave2_parity.sh` 8/8, `wave2_game_parity.sh` 5/5 with every hybrid option at default.
* 0.2 Option parsing: each option sets, clamps and reports; knobs without their mode are refused in self-play and
  reported in UCI; `--hybrid-mode` with `--durative-tree true` or without `--base-mix` is refused; the `Hybrid` preset
  sets and clears `HybridMode`; `HybridMode gate` with `DurativeTree true` prints the inert line.
* 0.3 **No trigger, no change**: on the 8 `wave2_parity` positions with seat options that fail the trigger (every
  opponent seat `SeatSitting 1`, or `HybridMaxWaitMs 0`), `HybridMode gate` gives the same output as
  `HybridMode off` (1 worker, same compare rules as `wave2_parity.sh`); a new `engine/tests/hybrid_parity.sh` runs it.
* 0.4 `sizeof(Node)` unchanged or its growth stated (`JointActionRules` gains one byte); `exact_key` and `operator==` of a state without
  `hybridSit_` are bit-identical to `72e79eb` (`test_time_state.cc` style).

**1. Trigger rules** (`hybrid_sit_boards`, table-driven, no network)
* 1.1 Each of T1-T7 failing alone empties that board's bit and names the reason.
* 1.2 Lead > 0 on one board: mask empty (gate allows). Lead <= 0, opponent on B thinking with `pace 600, elapsed 500`:
  A masked, event 100. Same with `elapsed 0` and `HybridMaxWaitMs 500`: empty (`wait`).
* 1.3 Both boards ours: A masked when every partner move on B is quiet and lead <= 0; event = `RootMoveTimeMs`.
* 1.4 Opponent on B sitting, our clock lower: empty (`opponent sitting`; T5 agrees).
* 1.5 Our clock 2 s with `HybridMinClockMs 3000`: empty (`clock`).

**2. Search behaviour (H3)** (real network `engine/models/twin-s-clock-it13.onnx`, skipped without it, as v4 test 2)
* 2.1 **Sits when a needed piece is about to land.** Fixture from `kSitForKnight` (`test_durative_part2.cc:423`) one ply
  on: B after Bb's `Qd4+`, Bw to move and in check, A with Aw to move needing a knight for `N@f7#`; equal clocks
  (lead 0: v3 forbids every sit; the opponent may not sit back either), opponent pace 300, elapsed 200 (event 100 ms).
  With `HybridMode gate` the root plays the pass on A (and the root q is above nodur's best move's q by at least the
  margin); with `HybridMode off` it plays a move on A.
* 2.2 **Does not sit when behind against a sit-back.** The same fixture with our lead -10 s: the opponent leads, may
  sit in check, and then we must move; the root does not choose the pass on A (assert `moveA != MOVE_NONE`, at 2000 and
  8000 nodes).
* 2.3 **Does not sit without an event.** 2.1 with Bw `SeatSitting 1`, and 2.1 with elapsed 0 and
  `HybridMaxWaitMs 100`: mask empty, the result equals `HybridMode off` (same nodes, same action).
* 2.4 Margin: with `HybridMarginPermille 1000` no hybrid sit is ever kept (2.1 plays a move); with 0 and the uniform
  test network the pass is kept whenever the tree prefers it (rules-level).
* 2.5 No cycle: a 20k-node search at 2.1 terminates; every node below a hybrid sit has `hybridSit_` set and none is the
  root node (walk the tree).
* 2.6 Certification and the mate-race veto still run on a hybrid sit (a fixture where the pass on A lets a mate in
  two through: the result is a move).

**3. Self-play and tournament**
* 3.1 `selfplay --hybrid-mode gate --durative-tree false --base-mix 30000:1 --flag-loss true`, it13 fp32, 200 nodes,
  4 games: no error, the hybrid telemetry line, HVM6 records byte-identical in format to `HybridMode off`.
* 3.2 `tournament` hybrid vs nodur, 4 games: no error, per-game hybrid line.
* 3.3 **No mutual-sit flags.** 40 self-play games, both teams hybrid (200 nodes, 30 s base, FlagLoss on, sit-off 5000):
  zero games end `sitoff_adjudicated`, and no `flag` game whose flagging seat passed in its last two decisions while
  the other board's seat was sitting; report sits per game and the share of searches with a non-empty mask. The same
  run with `HybridMode off` as the reference count.

**4. Positions (harness; measurement, not pass/fail on CI)**
* 4.1 `score hybrid` at 1k / 4k / 16k on the 300 positions: joint and board agreement with it13@32k within the CIs of
  nodur (87% / 91% at 16k); act-vs-sit agreement at least 90%; sit rate when behind at most 15%; trigger rate
  reported.
* 4.2 Margin sweep 0 / 20 / 50 and prior 50 / 150 / 300 at 4k: pick the defaults by act-vs-sit agreement with the
  reference while the sit-when-behind rate stays above nodur's (else the hybrid does nothing).

**5. Games (measurement)**
* 5.1 Local: `match hybrid nodur` (row `wave2-self` with nodur's flags) 150 pairs at 30 s; success = score above 0.5 with
  the CI's lower end above 0.47; then `match hybrid it13` 150 pairs.
* 5.2 Local, the PACE cost: the same with `--ms-per-node 0.256,0.205` (PACE's 26% charge) against it13.
* 5.3 PACE (the user pushes and submits; nothing here submits): a `wave2c`-style row `c_hybrid_120`
  (`{"durative_tree": false, "sit_gate": true, "hybrid": true}` vs `upstream-it13`, 120 s, 300 pairs) beside
  `c_nodur_120`, and `c_hybrid_vs_nodur_120`. Rows go into `cluster/launch_checks.sh` as a new `wave2d` group; they
  need the engine at the new pin and `cluster/build.sh`'s parity checks.

**H1 additions** (when built): node accounting (`go nodes 1000` with a probe reports `N_main + N_probe` within one
batch of 1000; skip conditions leave `N_probe` unspent and say so), the probe leaves the main tree and plans untouched
(`rootNode` identity and `nextRootCandidates_` before/after), the action filter (only the listed pairs get visits),
and 2.1-2.3 with `HybridMode probe`. **H2 additions**: priced children carry durative states (keys differ from every
v3 node), their subtrees follow v4 rules (flag and sit-off terminals reachable), and 2.1-2.3 with `HybridMode priced`.

## Build order

1. `hybrid.h` trigger + `TimeState::ready_ms` / `sitting` / `hybridSit_`, tests 0.4 and 1.x. Commit.
2. UCI and self-play options, `current_hybrid_clock`, `search_clock` split, tests 0.2. Commit.
3. H3: rules mask, prior floor, sticky child bit, choice with margin, report; tests 2.x, 0.1, 0.3. Commit.
4. Self-play and tournament plumbing, telemetry; tests 3.x. Commit.
5. Arena flags and preset, harness configs; main-repo Python tests. Commit (main repo).
6. Measurements 4.x, 5.1, 5.2; doc section in `docs/wave2_engine.md` and a results section here.
7. Only if H3 shows no gain over nodur: H1, then (unlikely) H2.

## Open decisions

* Whether a gain from H3 should go into training (teach the sit plane the hybrid's sits) or stay a play-time rule.
* `HybridMaxWaitMs` relative to the budget (a fixed 1000 ms or `k x RootMoveTimeMs`); 120 s games run about 600 ms per
  move, 30 s local games much less, so a fixed value may trigger very differently on PACE and on the Mac.
* Whether T6 should use the arena's sit-off time (5 s) explicitly.
