# Custom engine spec v6: the hybrid (v3 tree chooses the move, the clock decides when)

Draft, 2026-11-21. Wave 2d. Builds on spec v4 (`docs/custom_engine_spec_v4_durative.md`) and v5
(`docs/custom_engine_spec_v5_continuity.md`) as built in `third_party/wave2-hivemind` (branch `talisman/wave2`, pinned
`72e79eb` in `cluster/wave2_engine.ref`; `docs/wave2_engine.md` there). Line numbers below are at `72e79eb`. Network
for every number quoted: twin-s-clock it13 (HF `aminwoo/bughouse-twin-s-clock@edaa246`).

## Review changes (2026-11-21, before implementation)

An adversarial read of this spec against the code at `72e79eb` (`joint_action.h`, `time_state.h`, `searchthread.cc`
`process_batch` / `select_and_expand` / `canonicalize_child` / `classify_terminal_position`, `node.h`
`try_init_and_expand`, `agent.cc` `run_search` final choice, `seat_clock_game.h` `commit_pass`, arena
`seat_decisions` / `Wave2TeamController.default_sit_ms`). Nothing was built or run. The sections below are already
edited; this list says what changed and why.

1. **H3 as first written was nodur in practice (fixed: the event lands).** In the one-board shape the trigger fires only
   when `lead <= 0` (T2). Our lead is the opponent's deficit, so whenever `lead < 0` the opponent may sit back at depth 1
   (`can_sit(~team)` = its lead > 0). The sit-back child is the root's position again (same boards, same lead, our
   team to move, gate on), so its value is our best move now: `Q(sit) = min(opponent replies, sit-back) <= Q(act) +
   noise`, and the 20 permille margin then refuses every sit. Only `lead == 0` exactly escapes, which never happens
   with millisecond clocks. Fix: in a triggered search the opponent's seat on the event board Y must land a move at
   depth 1 (that is what T3 assumes: the seat is thinking, not sitting). New option `HybridEventLands`
   (`all` default, `sit`, `none`), step 4 of H3, test 2.2 rewritten, test 2.7 added. `none` is the first draft's
   behaviour, kept as the ablation that shows the inertness.
2. **The rule is applied to every root child, not only to the sit (`all`).** Forbidding the opponent's pass on Y only
   below our sit gives the opponent fewer options in the sit branch than in the move branches, which biases the root
   toward sitting whenever the opponent leads. Under `all` the same assumption ("the thinking seat on Y lands a
   move") holds in every branch, so the comparison is fair. Cost: in a triggered search the move choice can differ
   from nodur's even when no sit is kept; "no trigger, no change" still holds exactly.
3. **The rule is a pass restriction, never `can_sit`.** `classify_terminal_position` and `Board::is_checkmate` take
   `can_sit` as the right to wait out a drop-blockable check. Turning the opponent's `can_sit` off would make such
   positions mate and could prove a false win for our sit. The depth-1 restriction is a new `mustMoveBoards` mask in
   `JointActionRules` (like `RequireMoveOn`, a forced pass stays legal); `can_sit` and the terminal rules are untouched.
4. **The cycle rationale was wrong; the sticky bit is replaced by a one-shot bit.** `canonicalize_child` already refuses
   to link a child to any node on the current trajectory (`isAncestor`, `searchthread.cc:1680`), the root included, so
   the pass/sit-back line could never form a cycle. What a key bit is needed for is the new depth-1 legality: depth-1
   states of a triggered search carry `hybridEvent_` (the event board), which enters `exact_key()` while set and is
   cleared by the next `advance()`. Depth-2 states are plain v3 states again, so tree reuse after the real landing
   still works.
5. **Both-boards shape dropped from H3.** With our team on turn on both boards the "event" was our partner's move,
   which in a joint tree lands in the same action, so `eventMs = RootMoveTimeMs` measured nothing; what seat X would
   really wait for is the opponent's reply on Y, a full pace that has not started. Worse, "legal as a double sit"
   would have allowed passing on both on-turn boards at once (no event at all). v3's capture rule already covers the
   useful case. H3 now masks only the shape "on turn on X alone, an opponent seat thinking on Y"; T1 says so and
   test 1.3 asserts an empty mask. Double sits with both boards on turn stay illegal in every mode.
6. **T7 removed.** After our pass on X the opponents cannot move on X (it is still our turn there), and any mate in one
   they hold on Y exists whether we sit or move, so the check could never separate the two. The danger of a sit is
   deeper (the opponent's Y move feeds their partner on X, who mates after we move); the tree's lines,
   `race_safe_alternative` and certification cover it, as for any move.
7. **T4 matched the wrong wake.** The arena wakes a pass after `min(PassCapK, 1.5) x budget`
   (`SIT_INFER_FACTOR`), not `PassCapK x`; an event between the two never arrives inside the wait, and the seat pays a
   wait plus a second full search for nothing. T4 is now `eventMs <= min(HybridMaxWaitMs, 1.5 x RootMoveTimeMs)` (and
   `<= PassCapK x RootMoveTimeMs`).
8. **Mutual sitting could reach the 5 s sit-off (risk 1 was wrong for large budgets).** If the thinking opponent
   decides to sit while we hybrid-sit, our wake comes after `1.5 x budget`, but the arena infers that seat as sitting
   only after `1.5 x min(pace, budget)` since its decision began, so the next search can trigger again and sit again.
   With budgets of a few seconds (early 120 s games) the chain can pass 5 s, and the lower clock (ours: we are behind
   by construction) loses. Fix: the arena caps a hybrid sit's wait at `eventMs + 100` (from the `Hybrid:` line) and,
   when the event seat has not landed by the wake, sends `SeatSitting 1` for it until its board lands (T3 then fails and
   the gate makes us move). Self-play already ends our pass at the opponent's landing (`commit_pass`) and sends the
   true sitting flag, so it needs only the same cap. Test 3.4 (arena) added.
9. **Every root pick honours the margin.** The first draft applied the exclusion only at `agent.cc:8440`. The
   most-visited fallback right after it, the solver PV index (`agent.cc:7930`), Gumbel's `decisionIdx`,
   `race_safe_alternative` and `certified_mate_free_alternative` (both pick "the best of the alternatives") and
   self-play's sampling could still play an unqualified hybrid sit. One predicate, `Agent::hybrid_sit_allowed(idx)`,
   is applied in all of them.
10. **A proof through a hybrid sit is conditional.** The tree proves it under "the event lands". The sit is kept (that
    is the knight case) but `run_search` does not report a mate score for it and the arena does not treat it as a proven
    mate; the `Hybrid:` line says `proven (assumes event)`.
11. **The prior floor takes visits from the move search.** 150 permille plus the visits the pass then earns is about
    15-25% of the root budget on triggered searches. That is a real cost against nodur's move quality and is measured:
    test 4.1 reports agreement on the triggered subset separately (the all-positions numbers would hide it).
12. **Tests that would pass on broken code.** 0.3 (no trigger, no change) passes trivially if the trigger never fires;
    it now has a positive control (the same positions with a triggering seat setup must print a non-empty mask on at
    least 3 of 8). 3.3 asked for zero sit-off games, which nodur's own gate sits can produce; it is now relative to the
    `off` reference. Every measurement (4.x, 5.x) reports hybrid sits per game and calls a run with no kept sits
    "inert", not "equal to nodur".
13. **Small corrections.** `TimeState` already has `seat_ready_ms(seat)` and `seat_sitting(seat)` (`time_state.h:795`);
    no new accessors. `JointActionRules` gains two bytes (`hybridSitBoards`, `mustMoveBoards`). H1 checks its sit
    variant with `action_loses_mate_race` before playing it (the main search's race veto and certification saw `a*`,
    not the sit).

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
   seat clocks (point 2). After our pass the opponent acts, or sits back (legal for it: it leads), and then we, still
   behind, must move (gate). *Review: the sit-back line returns the root's own position, so the tree values the pass at
   most at our best move and the root never keeps it (Review changes 1). H3 therefore forbids that sit-back at depth 1.*
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
| T1 | our team is on turn on X **and not on Y** (an opponent seat is on turn on Y), and has at least one legal move on X | the one-board shape only (Review changes 5); with no legal move the pass is forced anyway |
| T2 | the v3 rules forbid the pass on X: `!v3.can_sit(team)` (in this shape: lead <= 0) | the hybrid only adds what the gate forbids; with lead > 0 nodur may already sit and the mask stays empty |
| T3 | the opponent seat on turn on Y is thinking, not sitting: `!durative.seat_sitting(seat)`; `eventMs = durative.seat_ready_ms(seat)` = `max(1, pace - elapsed)` | waiting is for a landing; with a sitting opponent nothing lands and the sit-off rule says the lower clock moves |
| T4 | `eventMs <= min(HybridMaxWaitMs, 1.5 x RootMoveTimeMs, PassCapK x RootMoveTimeMs)` | a short wait only, and one that ends inside the arena's wake (`default_sit_ms`, `min(PassCapK, 1.5) x budget`; Review changes 7) |
| T5 | `!durative.sit_off_forces_move(team)` on X | the durative rule for "behind against a sitter" (redundant with T3; kept as the stated rule) |
| T6 | our seat on X has `clock >= max(HybridMinClockMs, 4 x (eventMs + RootMoveTimeMs))` | flag safety: v3 knows no absolute clock |

(T7 of the first draft, "no mate in one after the pass", is removed: it could never tell a sit from a move; Review
changes 6.) The mask has at most one bit. `TimeState` needs no new accessor (`seat_ready_ms`, `seat_sitting`,
`side_on` exist, `time_state.h:795`).

Telemetry (verbose): `info string Hybrid: sit boards <A|B|AB|-> event <msA>/<msB> mode <mode>`; when the mask is empty
the line says which condition failed first (`no move`, `both boards`, `gate allows`, `opponent sitting`, `wait`,
`sit-off`, `clock`). The arena parses it into the search result for counting (below).

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
2. **Root legality.** `JointActionRules` gains `uint8_t hybridSitBoards = 0` and `uint8_t mustMoveBoards = 0` (bit 0 =
   A, bit 1 = B; default 0 = today's rules; the struct grows from 5 to 7 bytes, check `JointCandidateGenerator`'s layout
   and `sizeof(Node)`). In `is_joint_action_legal` (`joint_action.h:55`), the one-board double sit (pass on X, the
   forced `MOVE_NONE` on the off-turn board Y) is legal when X's bit is set, whatever `teamHasTimeAdvantage` says.
   `is_double_sit_legal`'s `boardAOnTurn != boardBOnTurn` condition stays: a double sit with both boards on turn is
   never legal. `RequireMoveOn` still wins. The mask reaches the generator through `RuntimeConfig::hybridSitBoards`
   (root only, like `rootPlanMove`), read in `Node::try_init_and_expand` when `m_depth == 0`. Nothing below the root
   sees it.
3. **Root prior floor.** The network's sit plane is upstream's (`lead_sit_right`), so when we are behind the pass
   prior is near 0 and the pass would get no visits. At the root, on each masked board, the pass entry is mixed as
   PlanCarry mixes a carried move: `p(pass) <- (1 - w) p(pass) + w`, others `(1 - w) p`,
   `w = HybridPassPriorPermille / 1000` (default 150), after the Dirichlet noise (`mix_root_plan`, reused).
4. **The event lands (depth 1).** T3 says the opponent seat on Y is thinking, so it will land a move in about
   `eventMs`. The tree must assume the same, or the opponent "sits back" at depth 1, the root's position returns, and
   `Q(sit)` can never beat `Q(act)` (Review changes 1). `select_and_expand`'s `time_after_action`
   (`searchthread.cc:1753`): at depth 0 of a triggered search, every child state gets `next.advance(...)` as today, then
   `next = next.with_hybrid_event(Y)` (under `HybridEventLands all`, the default; under `sit` only the hybrid-sit child;
   under `none` no child). `TimeState::hybridEvent_` (2 bits: none, A, B) is **one-shot**: the next `advance()` clears
   it; it enters `exact_key()` only while set (as `sitGate_` does), so depth-1 nodes of a triggered search never share
   a node with gate-rule nodes, and depth-2 states are plain v3 states (tree reuse after the real landing still
   works). In `process_batch` (`searchthread.cc:1353`) a node whose state has the bit gets
   `rules.mustMoveBoards = bit(Y)`: in `is_joint_action_legal`, a pass on Y is illegal while Y can move (a forced pass
   stays legal, as for `RequireMoveOn`). **`can_sit` is not touched**: `classify_terminal_position` and
   `Board::is_checkmate` read it as the right to wait out a drop-blockable check, and turning it off would prove false
   mates (Review changes 3). In the sit branch the opponent is on turn on Y alone, so it must move there; in a move
   branch it is on turn on both boards and must move on Y, while a pass on X (its reply to our move) follows v3's rules.
   From depth 2 on the tree is v3 with the gate: after the opponent's Y move we are on turn on both boards, behind, and
   may pass on X only with a partner capture. No cycle is possible: `canonicalize_child` already refuses to link a
   child to a node on the current path (`isAncestor`), and with the event landing the root position cannot recur.
   `HybridEventLands none` is the first draft (sit-back allowed); it is the ablation for test 2.7.
5. **The choice.** After the tree, in `run_search` where `bestIdx` is taken (`agent.cc:8440`): if the chosen action is a
   hybrid sit, keep it only if `Q(sit) >= Q(best moving action) + HybridMarginPermille / 1000` (default 20) and the sit
   has at least 10% of the root's visits; otherwise take the best child that is not a hybrid sit (same q-weighted
   rule, with an exclusion predicate added to `get_best_move_idx_with_q_weight`). "Best moving action" means the best
   child that is not a hybrid sit. The predicate is `Agent::hybrid_sit_allowed(idx)` and every root pick uses it: the
   q-weighted pick, the most-visited fallback after it, the solver PV index (`agent.cc:7930`), Gumbel's `decisionIdx`,
   `race_safe_alternative` and `certified_mate_free_alternative` (their "best alternative"), and self-play's sample
   (Review changes 9). Proven results: a proven mate or a proven loss from the root provers return as today (they
   enumerate v3 actions only). A solved WIN child that is a hybrid sit is kept without the margin test, but the proof
   assumes the event lands (step 4), so no mate score is printed for it and the report says `proven (assumes event)`.
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
* The opponent's best move on Y after our sit is in the tree, and so is everything after it under v3's rules. What
  the tree does not see is the opponent choosing to sit instead of landing: T3 bets that a thinking seat lands. When
  the bet is wrong the arena finds out at the wake (the event did not land), marks the seat sitting, and the gate
  makes us move (arena section). The cost of a lost bet is one short wait, lead-neutral.
* No node split: the budget is nodur's. No cost when nothing triggers.
* The durative model does exactly the "when" part: T3-T6 read the landing times, sitting flags and clocks that only
  the durative state has.

**Node and time budget.** The same `go nodes N` / `movetime` as nodur. The trigger costs one durative state build
(microseconds). The root carries one more legal action, and with the prior floor it takes about 15-25% of the root's
visits on a triggered search, visits the move search does not get (measured in 4.1 on the triggered subset). Each
kept sit costs real clock: the search before it, the wait (at most T4), and a full search after the wake.

**What H3 cannot do.** It never sits when we lead (the mask is empty: the gate already decides there), it never
vetoes a gated sit, and it never sits when our team is on turn on both boards (v3's capture rule decides there). It does not model *how long* the wait is beyond T4 (v3 has no clock), so a wait for a slow event
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
   fully (only one side has a wait node next), which is what the margin is for. A sit decided by the probe is then
   checked with `action_loses_mate_race` (v3 alternation, as for any root action); if it loses the race, `a*` is
   played. The main search's race veto and certification saw `a*`, not the sit variant.
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
| `HybridEventLands` | combo `all` `sit` `none`, all | `--hybrid-event-lands <s>` | gate, priced | depth-1 rule: the thinking opponent seat on Y must move, below every root child (`all`), below the hybrid sit only (`sit`), or never (`none`, first draft; ablation) |
| `HybridMaxWaitMs` | spin 0..10000, 1000 | `--hybrid-max-wait-ms <ms>` | all | T4: longest expected wait for an event (also capped at 1.5 x `RootMoveTimeMs`) |
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
`hybridSitBoards`, `hybridEventBoard` and `hybridEventLands` (H3, H2), `rootActionFilter` (H1), `hybridRootClock`
(H2), all root-only and empty by default.

Python and arena:
* `hivemind.wave2.Options` gains the fields; `rl-loop` is not changed this round (the hybrid is a play-time rule; the
  training targets stay the v3 tree's visits, below).
* `arena/players/upstream.py`: `WAVE2_HYBRID_FLAGS` (`hybrid_mode` -> `HybridMode` "off", knobs at the engine
  defaults), merged into `WAVE2_FLAGS`, sent only when not default (as the continuity flags, so old rows send exactly
  what they sent); `WAVE2_INT_RANGES` gets the spins; a preset `hybrid` in `WAVE2_PRESETS` expanding to
  `hybrid_mode: "gate"`. A row with the preset must also set `durative_tree: false` and `sit_gate: true` (the wave 2
  arm defaults `durative_tree` to true): `wave2_flag_options` refuses `hybrid_mode != off` with `DurativeTree true`.
  `_parse` reads the `Hybrid:` info strings into `SearchResult.hybrid` (`sit`, `act`, `skipped`, and the event board
  and `eventMs`), and `Wave2TeamController` counts `hybrid_sits` and `hybrid_triggers` per game for the game log.
* **Wake and missed events** (Review changes 8). After a kept hybrid sit, `Wave2TeamController` waits
  `min(default_sit_ms, eventMs + 100)` (a landing still wakes it sooner). If the event seat has not landed by the wake,
  the controller records it as a missed event and sends `SeatSitting 1` (elapsed 0) for that seat in every search until
  its board lands, overriding `seat_decisions`' inference. T3 then fails, and the gate makes us move. This bounds a
  joint sit (we sit, the opponent decided to sit) to one event window plus one search, far below the 5 s sit-off,
  whatever the budget. Counted per game as `hybrid_missed`.
* `tools/strength_harness.py`: configs `hybrid` (`{"durative_tree": false, "sit_gate": true, "hybrid": true}`),
  `hybrid_probe`, `hybrid_priced`; the report adds "trigger rate" and "hybrid sit rate" on all positions and on the
  sit-or-act subset.

## Self-play and tournament

* `run_selfplay` and `run_tournament` (seat-clock games only) pass `options.hybrid` with `durative_clock(..., false)`.
* Self-play samples its action from the root visits (`select_action(edges, temperature)`). A sampled hybrid sit is
  played only if the margin rule of H3 step 5 holds; otherwise the sample is redrawn over the children that are not
  hybrid sits. At temperature 0 the action is `run_search`'s result. H1: the probe's decision replaces the sampled
  action's half on X (a sit) only if X's seat is deciding now.
* A hybrid sit's pass in `SeatClockGame::commit_pass` ends at the opponent's landing on Y as any pass does (self-play
  knows the opponent's real decision, and T3 reads its true `sitting_`), and in any case after
  `min(PassCapK, 1.5) x` the seat's move time, the arena's wake, so self-play and arena games wait alike.
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

1. **Mutual sitting until flag or sit-off.** Only the team that is behind can hybrid-sit (T2), so two hybrids never
   hybrid-sit at once; the risk is our hybrid sit meeting an opponent that decides to sit while we wait. In self-play
   T3 reads the true sitting flag and our pass ends at the landing or the cap. In the arena the first draft's bound
   was wrong: the arena infers an opponent as sitting only after `1.5 x min(pace, budget)`, so with multi-second
   budgets a chain of hybrid sits could pass 5 s and lose the sit-off on our lower clock. The wake cap and the
   missed-event mark (arena section) bound the joint sit to one event window plus one search. Tests 3.3 and 3.4.
2. **Sitting into a mate or a lost race.** The tree's own lines (the opponent's Y move, then its partner on X after we
   move), and the unchanged `race_safe_alternative` and certification, which now also respect the margin predicate. A
   deeper mate that only the durative landing order shows is not covered in H3 (v3 alternation).
2b. **The event-lands bet.** The tree assumes a thinking opponent lands a move (step 4). An opponent that leads and is
   in check may prefer to sit; then our sit was chosen on a line that does not happen. Cost: one wait, lead-neutral,
   and the missed-event mark stops a repeat. The match measures it (`hybrid_missed` per game).
3. **Over-sitting through the prior floor.** 15% prior on a pass the network thinks is illegal buys it visits; the
   margin and the 10% visit floor decide. Tuned on the harness, not in games (test 4.2 sweep). It also takes those
   visits from the move search (Review changes 11).
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
  **Positive control** in the same script: the same positions with a triggering seat setup (lead -2 s, the opponent
  on the one-board shape thinking with `pace 300, elapsed 200`) must print a non-empty `Hybrid: sit boards` line on at
  least 3 of the 8 (positions without the one-board shape are listed as skipped); otherwise the script fails, since an
  always-empty trigger would pass the parity half trivially.
* 0.4 `sizeof(Node)` unchanged or its growth stated (`JointActionRules` gains two bytes); `exact_key` and `operator==`
  of a state without `hybridEvent_` are bit-identical to `72e79eb` (`test_time_state.cc` style); `advance()` clears
  `hybridEvent_` and the key of the cleared state equals the plain state's.

**1. Trigger rules** (`hybrid_sit_boards`, table-driven, no network)
* 1.1 Each of T1-T6 failing alone empties that board's bit and names the reason; with all holding, exactly one bit.
* 1.2 Lead > 0 on one board: mask empty (gate allows). Lead <= 0, opponent on B thinking with `pace 600, elapsed 500`:
  A masked, event 100. Same with `elapsed 0` and `HybridMaxWaitMs 500`: empty (`wait`). `RootMoveTimeMs 200`,
  event 400, `HybridMaxWaitMs 1000`: empty (`wait`: above 1.5 x 200).
* 1.3 Both boards ours, lead <= 0: mask empty (`both boards`), whatever the partner moves are.
* 1.4 Opponent on B sitting, our clock lower: empty (`opponent sitting`; T5 agrees).
* 1.5 Our clock 2 s with `HybridMinClockMs 3000`: empty (`clock`).
* 1.6 Rules (`is_joint_action_legal`, no network): with `hybridSitBoards` = A and A alone on turn, (pass, none) is
  legal at lead <= 0; with both boards on turn and both bits set, (pass, pass) stays illegal; `mustMoveBoards` = B
  makes a pass on B illegal while B can move and legal when B cannot (forced); `RequireMoveOn` still wins.

**2. Search behaviour (H3)** (real network `engine/models/twin-s-clock-it13.onnx`, skipped without it, as v4 test 2)
* 2.1 **Sits when a needed piece is about to land.** Fixture from `kSitForKnight` (`test_durative_part2.cc:423`) one ply
  on: B after Bb's `Qd4+`, Bw to move and in check, A with Aw to move needing a knight for `N@f7#` (it comes from
  Bb's `Qxh4` after Bw's reply, paired with a pass on A by v3's capture rule); our lead -2 s (v3 forbids our sit; the
  opponent leads), opponent pace 300, elapsed 200 (event 100 ms), `RootMoveTimeMs 200`. With `HybridMode gate` the
  root plays the pass on A (q above the best moving child's q by at least the margin, or a solved WIN reported
  `proven (assumes event)`); with `HybridMode off` it plays a move on A. Check the fixture first: if Bw has a reply
  that saves the knight, pick a fixture where every reply loses it, and say so in the test.
* 2.2 **Event-lands bet is visible.** Same fixture with our lead -10 s: under `HybridEventLands all` the result is the
  same as 2.1 (the tree assumes Bw replies); with Bw `SeatSitting 1` (it sat) the mask is empty and A moves. Assert
  both, at 2000 and 8000 nodes.
* 2.3 **Does not sit without an event.** 2.1 with Bw `SeatSitting 1`, and 2.1 with elapsed 0 and
  `HybridMaxWaitMs 100`: mask empty, the result equals `HybridMode off` (same nodes, same action).
* 2.7 **The first draft was inert** (ablation, documents Review changes 1): 2.1 with `HybridEventLands none` (sit-back
  allowed) at lead -2 s keeps no sit (`Q(sit) <= Q(best move) + margin`, reason `margin`); at 2.1 under `all`, every
  depth-1 node below the root has `expanded_rules().mustMoveBoards == bit(B)` and every depth-2 node has none.
* 2.4 Margin: with `HybridMarginPermille 1000` no hybrid sit is ever kept (2.1 plays a move); with 0 and the uniform
  test network the pass is kept whenever the tree prefers it (rules-level).
* 2.5 No cycle: a 20k-node search at 2.1, and the same under `HybridEventLands none`, terminates; no node below the
  root's hybrid-sit child is the root node (walk the tree).
* 2.6 Certification and the mate-race veto still run on a hybrid sit (a fixture where the pass on A lets a mate in
  two through: the result is a move), and an alternative they pick is never an unqualified hybrid sit (a fixture where
  the leader is vetoed and the most-visited other child is a hybrid sit below the margin: the result moves).

**3. Self-play and tournament**
* 3.1 `selfplay --hybrid-mode gate --durative-tree false --base-mix 30000:1 --flag-loss true`, it13 fp32, 200 nodes,
  4 games: no error, the hybrid telemetry line, HVM6 records byte-identical in format to `HybridMode off`.
* 3.2 `tournament` hybrid vs nodur, 4 games: no error, per-game hybrid line.
* 3.3 **No mutual-sit flags.** 40 self-play games, both teams hybrid (200 nodes, 30 s base, FlagLoss on, sit-off 5000),
  and the same 40 seeds with `HybridMode off` as the reference: `sitoff_adjudicated` games no more than the reference
  (nodur's own gate sits can produce some, so zero is not the bar), none whose loser hybrid-sat in its last two
  decisions, and no `flag` game whose flagging seat hybrid-sat in its last two decisions. Report kept hybrid sits per
  game and the share of searches with a non-empty mask; a run with no kept sits fails the test (inert).
* 3.4 **Arena wake and missed events** (main-repo pytest, fake engine): a fake wave 2 process prints
  `Hybrid: chose sit A ... event 200`; the controller's wait is `min(default_sit_ms, 300)`; if the B seat has not
  landed by the wake the next search sends `SeatSitting` 1 for it and keeps sending it until B lands; a landing on B
  clears it.

**4. Positions (harness; measurement, not pass/fail on CI)**
* 4.1 `score hybrid` at 1k / 4k / 16k on the 300 positions: joint and board agreement with it13@32k within the CIs of
  nodur (87% / 91% at 16k); act-vs-sit agreement at least 90%; sit rate when behind at most 15%; trigger rate
  reported. Agreement is also reported on the **triggered subset** alone, against nodur on the same subset: the prior
  floor's lost visits show there and would be diluted in the 300. The harness positions must carry the seat options
  (paces, elapsed, sitting) from their arena games, or the trigger reads defaults; if they do not, add them first.
* 4.2 Margin sweep 0 / 20 / 50 and prior 50 / 150 / 300 at 4k: pick the defaults by act-vs-sit agreement with the
  reference while the sit-when-behind rate stays above nodur's (else the hybrid does nothing).

**5. Games (measurement)**
* 5.1 Local: `match hybrid nodur` (row `wave2-self` with nodur's flags) 150 pairs at 30 s; success = score above 0.5 with
  the CI's lower end above 0.47; then `match hybrid it13` 150 pairs. Every match reports kept hybrid sits,
  triggers and missed events per game; a match with (almost) no kept sits is "inert", not "equal to nodur".
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

1. `hybrid.h` trigger + `TimeState::hybridEvent_` (one-shot) + `JointActionRules` masks, tests 0.4 and 1.x. Commit.
2. UCI and self-play options, `current_hybrid_clock`, `search_clock` split, tests 0.2. Commit.
3. H3: root mask, prior floor, depth-1 event rule (`HybridEventLands`), margin predicate in every root pick, report;
   tests 2.x, 0.1, 0.3. Commit.
4. Self-play and tournament plumbing (hybrid-sit cap in `commit_pass`), telemetry; test 3.3. Commit.
5. Arena flags and preset, wake cap and missed-event mark, harness configs; main-repo Python tests incl. 3.4.
   Commit (main repo).
6. Measurements 4.x, 5.1, 5.2; doc section in `docs/wave2_engine.md` and a results section here.
7. Only if H3 shows no gain over nodur: H1, then (unlikely) H2.

## Open decisions

* Whether a gain from H3 should go into training (teach the sit plane the hybrid's sits) or stay a play-time rule.
* `HybridMaxWaitMs` relative to the budget. T4 is now also capped at 1.5 x `RootMoveTimeMs` (the arena's wake), so
  the fixed 1000 ms binds only on long budgets; whether to keep it at all is open.
* `HybridEventLands all` vs `sit`: `all` keeps the root comparison fair but changes nodur's move choice on triggered
  searches; the harness (4.1 triggered subset) decides.
* Whether T6 should use the arena's sit-off time (5 s) explicitly.

## Measurements (2026-11-21/22, H3 as built)

**Model and weights in every number here:** twin-s-clock it13 (HF `aminwoo/bughouse-twin-s-clock@edaa246`,
`engine/models/twin-s-clock-fp32.onnx`, ONNX Runtime with the Core ML EP), for wave 2 and stock it13 alike. Wave 2:
`third_party/wave2-hivemind` `talisman/wave2` at `e5f3aae` (H3 as built; local, not pushed, not the pin),
`build-ninja`, 4 search workers. Reference and it13 rows: stock upstream it13 `build-ninja`. Machine: the 36 GB M4
Max, shared with other jobs (load 35-100 during most of the run). Harness: `tools/strength_harness.py`
(`docs/strength_harness.md`), the same 300 positions (seed 0, from the PACE `nodur`, `ref_120`, `same_120` games) and
the same it13 @ 32k labels as the baselines there. Raw data: `runs/strength/scores.jsonl`,
`runs/strength/report_hybrid.md`, `runs/strength/match_hybrid_vs_nodur/`, `runs/strength/match_p300m0_vs_nodur/`
(git-ignored). The harness now keeps the engine's `Hybrid:` lines per row and reports the trigger, kept sits and the
triggered subset (commits `5e4d7ba`, `8d1f790`).

Variants: `hybrid` = H3 defaults (`HybridEventLands all`, margin 20, prior floor 150); `hybrid_sit` = event lands
below the sit only; `hybrid_none` = the first draft (sit-back allowed); `hybrid_m0` / `hybrid_m50` = margin 0 / 50;
`hybrid_p50` / `hybrid_p300` = prior floor 50 / 300; `hybrid_p300m0` = prior 300 and margin 0 (the most permissive).
All with `DurativeTree false`, `SitGate true`. H1 and H2 are not built.

### 4.1 / 4.2 Positions

All positions (n 300; joint / board agreement with it13 @ 32k):

| Config | 1k joint | 4k joint | 16k joint | 16k board |
|---|---|---|---|---|
| nodur | 61% [56, 67] | 76% [71, 81] | 87% [83, 91] | 91% [87, 94] |
| hybrid | 60% [54, 66] | 75% [71, 80] | 88% [84, 91] | 90% [87, 94] |
| hybrid_sit | 60% [54, 65] | 76% [71, 81] | 86% [82, 90] | 89% [85, 93] |
| hybrid_none | - | 77% [72, 81] | - | - |
| hybrid_m0 | 60% [54, 66] | 75% [70, 80] | - | - |
| hybrid_m50 | - | 76% [71, 81] | - | - |
| hybrid_p50 | - | 75% [70, 80] | - | - |
| hybrid_p300 | 60% [54, 66] | 75% [71, 80] | - | - |
| hybrid_p300m0 | 59% [53, 65] | 75% [70, 80] | - | - |

Root q, Brier and log-loss are nodur's to the third decimal for every variant (0.054-0.062; 0.118-0.122; 0.374-0.387).

The trigger and the triggered subset (the same 30 positions fire for every variant: the trigger reads only the
clocks; agreement with it13 @ 32k beside nodur on the same 30 ids and budget):

| Config | Nodes | Triggered | Kept sits | Joint agree, triggered | nodur, same ids | Act-vs-sit agree, triggered | nodur | Differs from nodur: triggered / rest |
|---|---|---|---|---|---|---|---|---|
| hybrid | 1000 | 30 (10%) | 1 | 67% [50, 83] | 70% [53, 87] | 97% | 100% | 2/30 / 17/270 |
| hybrid | 4000 | 30 | 2 | 77% [60, 90] | 83% [70, 97] | 93% | 100% | 2/30 / 12/270 |
| hybrid | 16000 | 30 | 2 | 90% [80, 100] | 97% [90, 100] | 93% | 100% | 2/30 / 8/270 |
| hybrid_sit | 1000 / 4000 / 16000 | 30 | 2 / 2 / 2 | 67% / 77% / 83% | 70% / 83% / 97% | 93% | 100% | 3/30, 4/30, 4/30 |
| hybrid_none | 4000 | 30 | 0 | 80% | 83% | 100% | 100% | 1/30 / 12/270 |
| hybrid_m0 | 1000 / 4000 | 30 | 2 / 3 | 60% / 73% | 70% / 83% | 93% / 90% | 100% | 5/30, 3/30 |
| hybrid_m50 | 4000 | 30 | 1 | 80% | 83% | 97% | 100% | 1/30 |
| hybrid_p50 | 4000 | 30 | 1 | 80% | 83% | 97% | 100% | 1/30 |
| hybrid_p300 | 1000 / 4000 | 30 | 4 / 3 | 53% / 77% | 70% / 83% | 87% / 90% | 100% | 7/30, 4/30 |
| hybrid_p300m0 | 1000 / 4000 | 30 | 5 / 5 | 57% / 77% | 70% / 83% | 83% / 83% | 100% | 9/30, 7/30 |

Noise floor: nodur at 4k rerun (`--rep 1`) differs from itself on 1/30 triggered and 11/270 other positions.
Sit rate when behind on clock (sit-or-act subset, n 35): nodur 0-3%, hybrid 3-6%, the permissive variants 6-9%,
it13 @ 32k 0%.

Reading:
* **The trigger is not the bottleneck; the value is.** It fires on 10% of positions (30 of 300, the most common
  failure being `both boards` and `gate allows`). On the 30, the v3 tree keeps a sit 0-5 times: on most triggered
  searches the root's q-weighted pick never chooses the sit ("no sit chosen"), because after the event lands the
  opponent's Y move is a free tempo in the network's eyes. `q_sit < q_act` on 19 of the 22 triggered rows logged with
  q at 1k (hybrid_sit).
* **Kept sits cost agreement, not gain it.** Every variant agrees with it13 @ 32k less than nodur on the triggered
  subset (e.g. 16k: 90% / 83% against 97%), and act-vs-sit agreement drops from 100% to 83-97%. The reference cannot
  sit when behind (v3 rules), so a kept sit always counts as a disagreement; this measures distance from v3, not
  strength. The prior floor's lost visits show at 1k (p300: 53% against 70%) and fade by 4k.
* **One kept sit is a conditional proof that a better move existed** (position 167, every variant): the sit is
  solved WIN through the event (`proven (assumes event)`) at about 120 nodes, which solves the root and stops the
  search, while the act `P@f4` had q 0.987 against the sit's 0.968. A fix that refused a proven sit whenever
  `q_act > q_sit` was tried and reverted: a solved child's q is not its proof value (in test 2.2 the correct proven
  sit has q_sit -0.45 against q_act -0.42), so the rule broke tests 2.1 and 2.2. Not fixed; a correct fix compares
  against the act's proof status or lets the search continue past a conditionally solved root. (The binary with
  the tried fix was live from 22:46:01 to 22:48:43; a few match games started in that window used it.)
* By 4.2's own rule (defaults by act-vs-sit agreement while the sit-when-behind rate stays above nodur's), margin
  20 and prior 150 stay: margin 50 / prior 50 agree as well but keep only the one proven sit (inert); margin 0 /
  prior 300 keep more sits and agree less. `HybridEventLands all` and `sit` are indistinguishable on positions;
  `none` is inert as predicted (0 kept sits).

### 5.1 Local matches (30 s base, vs nodur)

`match <config> nodur` (row `wave2-self` with nodur's flags), rules v3, bughouse openings, colour-swapped pairs,
ms/node pinned 0.206 for both sides (same binary), 2-3 workers in 25-min slices.

| Contender | Games | Score | 95% CI | W / D / L | Pentanomial | Triggers / kept sits / missed per game |
|---|---|---|---|---|---|---|
| hybrid (defaults) | 273 | **0.502** | [0.438, 0.562] | 136 / 2 / 135 | [39, 0, 57, 2, 38] | 11.2 / 0.35 / 0.15 |
| hybrid_p300m0 | 149 | **0.456** | [0.339, 0.532] | 68 / 0 / 81 | [24, 0, 33, 0, 17] | 9.9 / 0.99 / 0.30 |

On the 149 schedule slots both played (same seed, same openings): hybrid 0.497, p300m0 0.456; 93 of 149 results the
same.

Kept sits in the games (from the event logs): defaults 95 kept, the event landed first 53 times (waits 106-360 ms),
missed (the event seat had not moved by the wake) 42, and in 14 of the 95 the event seat was already sitting when we
sat (the arena cannot see a sit that began a few ms earlier). p300m0: 147 kept, 103 landed, 44 missed, 19 already
sitting. When we are behind, the opponent is ahead and nodur's gate lets it sit: T3's bet that a thinking seat lands
fails about 30-45% of the time in these matches. Games in which the contender kept at least one hybrid sit scored
0.386 (83 games; p300m0 0.372, 78 games) against 0.553 (190; p300m0 0.549, 71) without; the hybrid only fires when
behind on clock, so this split is confounded and says nothing causal by itself.

Wall: 103 min for the defaults (4 slices), 52 min for p300m0 (2 slices, beside the defaults and the scoring), plus
about 80 min of harness scoring alongside; about 3 h in all.

### Verdict and the PACE rows

* **No variant beats nodur.** The defaults are a tie with nodur (0.502 [0.438, 0.562], 273 games) and close to
  inert (0.35 kept sits per game, ~45% of them missed events); the most permissive variant keeps three times as many
  sits and scores lower (0.456 [0.339, 0.532], 149 games; the CI still includes 0.5). On positions every variant moves
  away from it13 @ 32k exactly where it sits. Test 5.1's success bar (score above 0.5 with the CI's lower end above
  0.47) is not met.
* **Pick for the PACE rows (least bad): `hybrid` with the H3 defaults** (`{"durative_tree": false, "sit_gate": true,
  "hybrid": true}`, i.e. `HybridMode gate`, `HybridEventLands all`, margin 20, prior floor 150). It is the only
  variant not below nodur locally, and at 120 s the per-node charge PACE puts on wave 2 (spec section 5.2) applies to
  it exactly as to nodur, so `c_hybrid_120` against `c_nodur_120` isolates the hybrid. Expect it to read as nodur
  within noise; a PACE row is still the only test at 20k nodes per search.
* Not run: 5.1 against it13, 5.2 (`--ms-per-node 0.256,0.205`), test 3.3, `hybrid_sit` and the margin / prior
  variants in matches.
* What would make the hybrid act: a value signal that rates waiting above the free tempo (the network's sit plane
  is v3's, trained on lead > 0 sits only), or H1's durative probe. The 30-45% missed-event rate also caps any gain
  from the timing side until the arena can tell a thinking opponent from a sitting one sooner.
