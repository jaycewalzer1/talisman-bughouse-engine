# Custom engine spec v5: continuity (team tree reuse, joint team actions, plan carry)

Draft, 2026-11-20. Wave 2c. Builds on spec v4 (`docs/custom_engine_spec_v4_durative.md`) as built in
`third_party/wave2-hivemind` (branch `talisman/wave2`, `261081b`; `docs/wave2_engine.md` there). Baseline the fixes are
judged against: `docs/continuity_baseline.md` (`d5e04c7`). Line numbers below are at `261081b`.

## Why

PACE test round 2026-11-19 (about 270-300 games per row, 120 s, equal clock, against upstream it13):

| Row | Score [95%] | nodes/ms |
|---|---|---|
| `ref_120` (durative tree on, gate off) | 0.227 [0.180, 0.285] | 12.2 |
| `sitgate` (durative on, gate on) | 0.206 | 15.7 |
| `nodur` (durative off, gate on) | 0.436 [0.361, 0.491] | 16.0 |
| it13 itself | | 20.3 |

`sitgate` searches nearly as fast as `nodur` and still scores half as much, and the eval calibration (Brier) is the same,
so the loss is move choice, not speed or value scale. Wave 2's own eval slides from about its 10th decision (-0.17 at
decisions 20-30, -0.49 at 45+) while `nodur` stays level. The Step 0 baseline adds two facts:

1. **No memory between decisions.** Exact tree reuse never fires under the seat clock: 0 of 522 wave 2 searches and 0
   of 522 it13 searches (`try_reuse_tree`, `agent.cc:9014`, keys on `search_hash_key(team, searchTime_)`, which holds
   the clocks). Ignoring clocks, the same board is among the retained candidates in 33% of durative decisions, with
   about 580 visits (28% of a 2050-node search). Self-play refuses reuse outright (`selfplay.cc:903`, `main.cc:365`),
   and the retained candidates are charged with v3's `advance()` (`agent.cc:9152`, `:9234`; `retain_played_line` `:9074`), which drops the durative
   block (`time_state.h:344`), so even the keys are of the wrong kind.
2. **No joint planning below the root, and more nodes do not fix it.** On 40 middlegame positions, agreement with it13
   at 64k nodes is 57/50/55% at 1k/4k/16k for durative on and 57/62/68% for off. The classic tree expands one
   `JointActionCandidate` (both boards, joint prior; `joint_action.h:183`) per team node. `TimeState::turn_at_node()`
   (`time_state.h:849`) lets only the earliest-ready seat act below the root, so our two seats act at separate nodes
   with opponent events in between, and a two-board plan sits several plies deeper than in the classic tree.

The three fixes answer these. Each is a UCI option and a self-play flag, off by default; a preset turns all three on.

## Common rules

* **Defaults are bit-for-bit.** With every new option off, the engine's output is identical to `261081b` with the same
  wave 2 options, and with all wave 2 options at default identical to stock v3.0.3 (`stock_parity.sh`). New state
  enters node keys, `operator==` and planes only when its option is on.
* **Durative only.** All three act only in a search whose clock `is_durative()` (UCI `DurativeTree true` under
  `TimeModel clock`; self-play `--durative-tree true`), like every other wave 2 tree option. With `DurativeTree` off
  they are accepted and do nothing (an `info string` says so once per `setoption`).
* **The root keeps its v4 meaning.** At the search root the root team still decides on every board it is on turn,
  landing at `RootMoveTimeMs` (`turn_at_node`, `atRoot`). Fix 2 changes nodes below the root only. UCI `bestmove` and
  `info ... pv` keep their format.
* House style: the comment density and naming of `time_state.h` and `searchthread.cc`; one doc section per fix in
  `docs/wave2_engine.md` ("Wave 2c"), filled in as each lands.

### Options

| UCI option | Type, default | Self-play / tournament flag | Meaning |
|---|---|---|---|
| `TeamTreeReuse` | check, false | `--team-tree-reuse <bool>` | Fix 1: keep the team's tree across decisions and adopt the node the game reached. |
| `TeamTreeReuseToleranceMs` | spin 0..5000, 250 | `--team-tree-reuse-tolerance-ms <ms>` | Fix 1: largest per-seat clock (and opponent ready-time) difference a retained node may have from the real clocks. |
| `JointPlans` | check, false | `--joint-plans <bool>` | Fix 2: below the root, a team that acts plans its other on-turn seat's move in the same joint action. |
| `JointPlanFreePermille` | spin 0..1000, 500 | `--joint-plan-free-permille <n>` | Fix 2: prior share of "decide later" (no plan) for the planned seat. |
| `PlanCarry` | check, false | `--plan-carry <bool>` | Fix 3: bias the next search toward the remembered half of the last chosen pair. |
| `PlanCarryPermille` | spin 0..1000, 300 | `--plan-carry-permille <n>` | Fix 3: prior mixing weight of the remembered move. |
| `Continuity` | check, false | `--continuity <bool>` | Preset: `true` sets the three switches on, `false` sets them off. Knobs keep their values. Last write wins. |

Parsing: `UCI::setoption` (`uci.cc:677` block, same `true`/`false` handling and `info string <name> set to ...`
replies), state in `uci.h` next to `durativeTree` (`bool teamTreeReuse, jointPlans, planCarry`, the three knobs),
advertised in `send_uci_response` after `SeatArrivalsMs` (`uci.cc:890`). The tree-side switches travel in
`DurativeConfig` (`time_state.h:36`: `jointPlans`, `jointPlanFreePermille`) and in `SearchOptions` (`agent.h:30`:
`teamTreeReuse`, `teamTreeReuseToleranceMs`, `planCarry`, `planCarryPermille`), filled by `UCI::go` (`uci.cc:260`
onward) and by self-play. Self-play flags: `Wave2GameOptions` (`seat_clock_game.h:28`) gains the fields,
`parse_wave2_option` (`:107`) the flags, `changes_search()` counts them, `validate()` checks ranges and refuses
`--team-tree-reuse`, `--joint-plans` or `--plan-carry` without `--durative-tree`. `main.cc` lists them in `--help`
(after line 141) for `selfplay` and `tournament` (`hivemind.wave2.missing_engine_options` reads that list). The
existing `--tree-reuse` stays refused with `--base-mix` (`main.cc:365`); `--team-tree-reuse` is allowed there.

---

## FIX 1: team tree reuse

### Behaviour

After every non-background search the agent keeps its whole tree: the root node, the board and the durative clock it
was rooted at (`TeamTree`, below). At the next search of the same team it looks in that tree for the node the game has
actually reached and adopts it as the root, re-timed to the real clocks. "Reached" is followed from the board: the
moves that landed since the last search are exactly the board difference, so the descent follows them.

1. **Descent (`Agent::find_team_tree_node`).** Breadth-first from the old root over expanded edges. An edge is followed
   only if, after its applied moves (fix 2: planned halves are not applied), each board's ply count is at most the
   live board's (`Board::game_ply`) and a board whose ply count equals the live one has the live board's FEN
   (position, side to move, castling, en passant; pockets compared separately: each pocket a sub-multiset of the
   live one, equal at a candidate). Wait edges `(pass, pass)` and pass edges keep ply counts and are followed. Depth
   cap `TEAM_TREE_MAX_PLIES = 8` joint plies (wait nodes count), node cap `TEAM_TREE_MAX_VISITED = 20000`. The path
   time is carried down with the durative charge the search used (`TimeState::after_action` with the edge's applied
   moves and hand pieces, as `time_after_action` does in `searchthread.cc:1438`), not `advance()`.
2. **Candidates.** A node is a candidate when its board equals the live board (both FENs and pockets), its team to
   play is our team, it is expanded and `UNSOLVED`, and its path time is close to the live clock `T'` (the clock this
   search was given):
   * same sides to move, same sitting bits for the seats on turn, same pending (delayed transfer) counts, same
     `outcome` (NONE);
   * every seat clock within `TeamTreeReuseToleranceMs` of `T'`;
   * for the other team's seats on turn, ready time within the tolerance of `T'`'s;
   * the **shape check** at the candidate passes under `T'` (below). At the root this is the strict part: `T'` is
     `atRoot`, so our team acts on every board it is on turn on. A candidate where one of our on-turn seats was not
     ready (it acted on one board only) has the wrong shape and is not adopted.
   Among candidates: smallest largest-clock-difference, then most visits.
3. **Re-timing (`Agent::retime_team_tree`).** Nodes store no time: every path time is recomputed from the root time
   at each selection (`select_and_expand`, `pathTime`). What a node does store and depends on time is (a) its hash,
   (b) its expansion (which boards acted, forced waits, whether a sit was legal, which drops were held back), (c) a
   solver verdict, (d) its transposition-table entry. The re-timing walk visits the adopted subtree breadth first
   from `T'`, carrying the new path time, single-threaded, before workers start:
   * (a) `set_hash(board.search_hash_key(team, nodeTime))` on every visited node, then the node goes into the fresh
     transposition table (this replaces `reindex_reused_subtree`'s stored-hash insert, `agent.cc:8982`, for this path);
   * (b) **shape check**: each node stores at expansion a 32-bit `expansionKey_` (new `Node` field, written in
     `try_init_and_expand`): a hash of the inputs that fixed its action lists besides the board: acting boards after
     the durative filter (`searchthread.cc:914-920`), `forcedWait`, the leaf `can_sit` (`leafTeamHasTimeAdvantage`,
     `:1055`), the held-back drop counts of the acting seats (`pending_in_hand`), and under fix 2 the planned board
     and any committed move. The walk recomputes the key under the new time; a mismatch replaces that child with a
     fresh unexpanded node (`Node::replace_child`, `node.h:828`; the old subtree goes to `discard()`);
   * (c) a node holding a solver verdict (`WIN`, `LOSS`, `DRAW`) is replaced by a fresh node when any clock moved
     (largest difference > 0). Flags and sit-off draws hinge on milliseconds; a fresh node re-proves a board mate at
     its first visit. With exactly equal clocks verdicts are kept;
   * nodes beyond `TEAM_TREE_MAX_RETAINED = 100000` walked nodes (`SearchParams::TREE_REUSE_REINDEX_MAX_NODES`) are not
     validated, so their parents' edges to them are replaced by fresh nodes too. Nothing unvalidated stays reachable.
   * Edge statistics of the parent (visits, Q) are kept even where a child was replaced, as CrazyAra-style reuse keeps
     them. This is a stated approximation; the replaced child earns its value again from its first visit.
4. **Adoption** goes through the existing guards in `run_search` (`agent.cc:6235-6267`: must-play loss, unexpanded
   loss, claimed win that loses the mate race), then `set_hash`, `set_depth(0)`, root noise, `configure_root_search`.
   Verbose UCI prints `info string Team tree reuse: <visits> visits recovered, clock diff <ms> ms, <pruned> pruned`
   (or `... none (<reason>)` with reason `no node`, `clock`, `shape`, `solved`), so the arena and the probe can count
   hits. The old exact path (`try_reuse_tree`) is not called when fix 1 is on for a durative clock.
5. **Node budget.** UCI `go nodes N` still searches N new nodes on top of the adopted visits
   (`reusedVisitsCount = false`), and `info ... nodes` reports new nodes only (`agent.cc:7751`), so the arena's
   "searched" charge stays what the engine did this decision. Self-play with fix 1 tops the root up to its budget
   (`reusedVisitsCount = true`, `minNewNodes = treeReuseMinNewFraction x budget`, as `--tree-reuse` does,
   `selfplay.cc:1550`).
6. **After the search.** With fix 1 on and a durative clock, the end of `run_search` (`agent.cc:8455-8475`) stores the
   whole tree as the `TeamTree` instead of `store_next_root_candidates`, and leaves `nextRootCandidates_` empty.

### Data structures

```cpp
// agent.h, beside nextRootCandidates_
struct TeamTree {
    std::shared_ptr<Node> root;    // the last search's root; owns the tree
    std::string boardA, boardB;    // its FENs (pockets included)
    std::array<int, 2> plies{};    // game_ply per board
    TimeState rootTime;            // the durative clock it was rooted at
    Stockfish::Color team;         // our team; a tree is only reused by the team that built it
};
std::optional<TeamTree> teamTree_;
```

`Node` gains `uint32_t expansionKey_ = 0` (written once under the node lock at expansion, read by the walk with no
workers running). Measure `sizeof(Node)` before and after and record it in `docs/wave2_engine.md`; with defaults the
field is written and never read.

### Files and functions

| File | Change |
|---|---|
| `search/agent.h`, `agent.cc` | `TeamTree teamTree_`; `find_team_tree_node`, `retime_team_tree`; `run_search` (around `:6229`) chooses the team path when `options.teamTreeReuse && searchTime_.is_durative()`; end of `run_search` (`:8455`) stores `teamTree_`; `reset_search_state` (`:6050`) clears it |
| `search/node.h` | `expansionKey_` and its accessor; set in `try_init_and_expand` (`:697`) from a new argument |
| `search/searchthread.cc` | `process_batch` computes the expansion key (`:1055-1118`) and passes it to `expand_leaf_node` |
| `environment/time_state.h` | `clock_distance(const TimeState& live) const` and `same_turn_state(const TimeState&) const` for the candidate test |
| `interface/uci.cc` | options; with `TeamTreeReuse` on, `go` (`uci.cc:300`) does not start `run_permanent_brain` for a durative clock (below) |
| `tools/selfplay.cc` | `play_seat_clock_game` (`:904`): with `--team-tree-reuse`, no `agent.reset_search_state()` per decision (`:1045`; still once at game start, `:933`), `searchOptions.teamTreeReuse`, top-up budget; summary line `selfplay team tree reuse: <hit %> of searches, <mean visits>` like `:1745`; the comment at `:903` updated |
| `main.cc` | flags and help; the `:365` refusal stays for `--tree-reuse` only |

### Both seats of the team

A team has one agent in self-play (`agents[2]`) and one engine per team in the arena, so the tree belongs to the team:
a decision of Bb adopts a node of the tree Aw's last search built and the other way round. The descent does not care
which seat moved last.

### Interactions

* **Permanent brain.** `run_permanent_brain` (`agent.cc:8480`) charges our action with `advance()`, which drops the
  durative block, and `run_background_search` replaces `rootNode` (via `try_reuse_tree`). Under fix 1 with a durative
  clock it would discard the team tree, so it is not started. The arena stops it at once anyway (`stop` after
  `bestmove`). A durative-aware permanent brain (continue the team tree below our chosen action, `verdictTeam` us) is
  out of scope; listed under Open.
* **Dirichlet noise.** UCI keeps its rule (`uci.cc:283-291`): with root noise on (`OpeningNoise` inside its plies)
  `reset_search_state()` runs before the search, so the team tree is dropped as retained roots are now. Self-play
  keeps the adopted root and re-noises it (`Node::refresh_root_noise`, as `run_search` already does at `:6273`).
* **SitGate.** `can_sit` at each node is in the expansion key, so a shift that changes the gate or the sit-off rule
  re-expands that node.
* **FlagLoss.** Clock-dependent verdicts are dropped when any clock moved (3c). A flag in the subtree is found again
  under `T'`.
* **InstantTransfer.** Pending counts must match at the candidate and are in the expansion key below it.
* **MateRaceByLanding.** Root provers are off under it as before; `earlyMs` is part of the path time and is
  recomputed. `lastCapture` must match at the candidate.
* **PassCapK.** Pass lengths are recomputed from `T'`; a pass whose wake-up order flips changes acting boards, which
  the expansion key catches.
* **SeatClocksMs, SeatPaceMs, SeatElapsedMs, SeatSitting, SeatArrivalsMs.** These build `T'`. Paces may drift between
  decisions: a pace change moves the other team's ready times, which the tolerance and the shape check cover. Wide
  landings are recomputed from `T'` along every path and are not compared.
* **`ucinewgame`** clears the team tree (through `reset_search_state`). The arena must therefore not send it before
  every search in fix 1 rows (arena section).

### Thread safety

The descent, the walk and the adoption run inside `run_search` under `searchMutex_` (`agent.cc:6146`), before
`dispatch_workers`, with no worker running and the permanent brain stopped (UCI `go` and `setoption` call `stop()`
first; self-play is sequential per agent). Replaced subtrees go to the GC thread (`discard`). `teamTree_` is written at
the end of `run_search`, after workers have joined. Parallel self-play games have their own agents.

### Memory

At most one tree per agent lives between searches. Adoption keeps at most `TEAM_TREE_MAX_RETAINED` (100k) nodes and
releases the rest of the old tree. The descent visits at most 20k nodes. Peak between searches: the previous search's
tree (bounded by its node budget, at most `max_nodes` 100000 in the arena pool) until adoption. Acceptance test 1.9
checks the bound.

---

## FIX 2: joint team actions in the durative tree

### Behaviour

Below the root, at a node where team T acts, T chooses one joint action over **all** its on-turn seats, as the classic
tree does, but each half keeps its own landing time:

* A seat of T that is ready now (it is in `acting_boards`) gets an **immediate** half: legal moves and pass, exactly
  as today.
* A seat of T that is on turn but not ready, and has no committed move, gets a **planned** half: one of its legal moves
  (or a drop made legal by a capture of the immediate half, below), or **decide later** (`MOVE_NONE`, no plan; the
  seat then decides freely when it is ready, as today). A planned move is not made on the board now. It is written
  into the durative state as that seat's **committed move** and lands at that seat's ready time.
* A seat with a committed move that becomes ready acts with a **forced** half: its committed move, the only action on
  that board. If no other seat of T acts there, the node has that single action (like a wait node, it costs one
  network evaluation; possible later optimisation: skip it).
* Opponent events interleave by time exactly as in v4: the planned seat's landing is an ordinary landing at its ready
  time (it wakes a pass on the other board 1 ms later, `advance_action` step 3).

A planned move stays legal until it lands: the planned seat is on turn, so its board does not change, and its pocket
only grows (the same argument the self-play game uses for committed moves, `seat_clock_game.h:142`).

Pass is not offered as a planned half: "decide later" covers it, and the seat may still pass when it is ready.

**Candidate list of the planned half.** The seat's legal moves on the board before the action, plus drops of each piece
type that one of the immediate half's captures would send it (`Board::hand_piece_type`), each tagged with the type it
needs. A drop is offered only where the transfer rule allows it **at the planned landing**: with `InstantTransfer`
true always; with it false only when the planned seat is sitting (its decision starts at its wake-up, after the
capture lands; `pending` is cleared then) and never for a seat already deciding. Existing held-back drops are filtered
as in `prepare_actions` (`searchthread.cc:930`).

**Priors.** Immediate halves as today. Planned half: the node's network policy for that board over the candidate list
(drop indices that are illegal before the capture take their raw logits, softmaxed with the rest at the node's policy
temperature). Then "decide later" gets `max(pass prior, JointPlanFreePermille / 1000)` and the moves share the rest in
proportion. Then each planned move that gives mate on its landing board (checked once per candidate at expansion, on
the board with the needed piece in the pocket) is raised to at least `SearchParams::JOINT_PLAN_MATE_PRIOR` = 0.25 and
the half is renormalised. The mate raise is there because the planned half's priors are read before the immediate
half's capture reaches the pocket; set the constant to 0 to ablate it. Joint factors (`jointFactorsA/B`,
`searchthread.cc:1105`) apply as for any half.

**Pair legality.** The generator (`JointCandidateGenerator`, `joint_action.h:220`) checks immediate halves exactly as
today (`is_joint_action_legal` with only the acting boards on turn). A pair with a planned drop that needs a supplied
type is legal only if the immediate half supplies that type. `legalPriorMass` gains the supplied-type categories (at
most 6 x 6 products) so widening by prior mass stays exact.

### Data structures

```cpp
// joint_action.h: JointActionCandidate. The new byte sits in existing padding (static_assert on sizeof).
uint8_t plannedMask = 0;   // bit 0: moveA is planned (not applied now), bit 1: moveB
Stockfish::Move applied_a() const { return plannedMask & 1 ? Stockfish::MOVE_NONE : moveA; }
Stockfish::Move applied_b() const { return plannedMask & 2 ? Stockfish::MOVE_NONE : moveB; }

// time_state.h: DurativeConfig
bool jointPlans = false;
int32_t jointPlanFreePermille = 500;

// time_state.h: TimeState::Durative
bool jointPlans = false;
std::array<Stockfish::Move, 4> committed{MOVE_NONE, MOVE_NONE, MOVE_NONE, MOVE_NONE};
```

`Durative::key()` mixes `committed` only when `jointPlans` is set, so keys with fix 2 off are exactly v4's.

New `TimeState` API: `planned_boards(team)` (on-turn, not ready, not committed boards of `team`; empty outside a
durative state, at the root or with fix 2 off), `committed_move(seat)`, and `advance_action(team, movedA, movedB,
handA, handB, plannedA, plannedB)` with the two new arguments defaulting to `MOVE_NONE`: a planned move is stored as
the seat's committed move and changes no ready time; a seat that acts with its committed move clears it; a committed
seat counts as "deciding" for the pass wake-up rule (it lands at its ready time). A committed seat may not be planned
again. `sit_off_forces_move` is unchanged (a committed seat is not sitting). Flags run as today: a committed seat whose
clock runs out before its landing flags (FlagLoss).

### Files and functions

| File | Change |
|---|---|
| `environment/time_state.h` | `DurativeConfig::jointPlans`, `Durative::committed`, key, `planned_boards`, `committed_move`, `advance_action` planned arguments, `durative()` copies the switch |
| `environment/joint_action.h` | `plannedMask`, `applied_a/b`; generator takes per-board planned flag, per-action needed type (planned board) and supplied type (immediate board); pair legality and `legalPriorMass` |
| `search/node.h` | `try_init_and_expand` passes the new per-action vectors; candidates carry `plannedMask` |
| `search/searchthread.h` | `LeafContext` gains `plannedA/B`, `forcedA/B`, the per-action needed/supplied vectors; `TrajectoryEntry` unchanged (the candidate carries the mask) |
| `search/searchthread.cc` | `collect_batch` (`:914-965`): planned and forced boards, candidate lists, mate tags; `process_batch` (`:1055-1118`): planned-half priors; `select_and_expand` (`:1438-1590`): `time_after_action` passes planned halves, `board.make_moves(applied_a(), applied_b())`; every unmake loop (`collect_batch` `:721`, `:839`, `:897`, `:985`; `canonicalize_child` `:1387`; `select_and_expand` makes at `:1512`, `:1569` and unmakes at `:1527`, `:1584`) and the per-board ply count (`:810-817`) use the applied halves (one helper, `unmake_applied(board, action)`) |
| `search/searchthread.cc` `classify_terminal_position` / `has_unavoidable_waiting_board_mate` | a seat with a committed move is not credited with a waiting-board mate unless its committed move is that mate |
| `search/agent.cc` | PV printing and `extract_pv_from_child`: a planned half prints as `pass` at its step, the move prints at its landing step; `retain_played_line` (`:9046`) uses applied halves |
| `interface/uci.cc` | option; `current_clock()` (`uci.cc:764`) sets `config.jointPlans`, `jointPlanFreePermille` |
| `tools/seat_clock_game.h`, `selfplay.cc`, `tournament.cc` | flags into `durative_config()` |
| `search/search_params.h` | `JOINT_PLAN_MATE_PRIOR` |

### Interactions

* **Root.** Unchanged (Common rules). Root edges never carry `plannedMask`, so `bestmove`, `root_edge_stats`,
  self-play policy targets and HVM6/HDST records keep their meaning.
* **SitGate.** Immediate halves use `can_sit` as now; "decide later" is not a sit and is always allowed.
* **FlagLoss.** As above. Acceptance test 4 of v4 (a 3 s seat makes at most 29 landed moves) holds with planned moves
  counted when they land.
* **InstantTransfer.** As in the candidate list. A planned drop of a held piece is never offered.
* **MateRaceByLanding.** A committed mate lands at its ready time and races by landing like any action; the waiting-board
  mate rule as in the table.
* **PassCapK.** A pass beside a planned seat wakes 1 ms after the planned landing (that seat lands, v4 step 4).
* **Seat options.** Build the root as now. `SeatSitting` matters for planned drops (sitting seats may drop supplied
  pieces with delayed transfer).
* **Fix 1.** Committed moves are in the path time and in the expansion key, so re-timing is exact. A retained node
  with a committed move for one of our seats cannot be adopted as a root (at the root both our on-turn seats decide):
  the shape check refuses it. Fix 3 then still carries the plan (below).
* **Transpositions.** Different commitments never merge (key); equal ones do.

### Thread safety and memory

Planning is per expansion under the node lock, like any expansion; `TimeState` is a per-path value; candidates are
immutable once generated. Memory: `JointActionCandidate` size unchanged (static_assert); `TimeState` grows by 17
bytes, copied per path step, not stored per node. Planned nodes have wider action lists (planned moves x immediate
moves), generated lazily by prior as now, so stored candidates grow with expansions, not with list sizes. Forced
landing nodes add one evaluation per committed move that lands without a free half beside it.

---

## FIX 3: plan carry

### Behaviour

When a search chooses a pair and only one of its boards gets played, the other half is remembered. When that board
comes up for decision again, the root's prior for that board is mixed toward the remembered move, so the partner
continues the plan unless the new search finds better.

1. **Recording (end of a non-background `run_search`).** For each board where our team was on turn at the root and the
   chosen half is a move (not a pass): a `CarriedPlan{board, move, fen of that board (pockets included),
   team, edgeQ, edgeVisitShare}`. Each search replaces both slots (an empty half clears its slot). Under fix 2, when
   fix 1's adoption was refused because the candidate held a committed move for one of our seats, that committed move
   is recorded too (it is the tree's plan for that seat).
2. **Validity at the next search.** A slot applies when the team is the same, our team is on turn on that board, the
   board's position, side to move, castling and en passant equal the recorded FEN, our seat's pocket holds at least
   what it held, and the move is legal now. Anything else drops the slot. So the half that was played (its board
   changed) is never carried.
3. **Bias.** At the root, the carried board's prior is mixed: `p(move) <- (1 - w) p(move) + w`, every other action on
   that board `p <- (1 - w) p`, with `w = PlanCarryPermille / 1000`, after root noise. Then the best joint candidate
   containing the move (by joint prior) is promoted to be expanded first (`Node::promote_joint_action`, `node.h:578`),
   so it gets its first visit before any other new edge: that is the "initial visit". A fresh root gets the mix in
   `process_batch` at its expansion (`searchPly == 0`); an adopted, already expanded root (fix 1) gets it through a new
   `Node::apply_root_plan(board, move, w)` that reweights the generator's priors under the node lock, like
   `refresh_root_noise` (`node.cc:230`), and calls `reindex_edge_priors_locked`.
4. **Telemetry.** Verbose UCI: `info string Plan carry: <board> <move> weight <w>` or `... dropped (<reason>)`.

### Self-play

`play_seat_clock_game` already plays only the deciding seats' halves (`selfplay.cc:1141-1152`). The undecided seat on
turn (if any) gets the other half recorded as above; when that seat's own `DECIDE` event comes, its search carries the
plan. With `--team-tree-reuse` the agent is not reset between decisions; without it, `reset_search_state()` must not
clear the plan slots (they live outside the tree), only game start and `ucinewgame` do.

### Data structures

```cpp
// agent.h
struct CarriedPlan {
    int board = -1;
    Stockfish::Move move = Stockfish::MOVE_NONE;
    std::string fen;               // that board's FEN when the plan was made
    Stockfish::Color team = Stockfish::WHITE;
    float edgeQ = 0.0f;            // telemetry only
    float edgeVisitShare = 0.0f;   // telemetry only
};
std::array<CarriedPlan, 2> plans_;   // one slot per board; cleared by clear_plans()

// SearchParams::RuntimeConfig (per-search copy, immutable during the search)
int rootPlanBoard = -1;
Stockfish::Move rootPlanMove = Stockfish::MOVE_NONE;
float rootPlanWeight = 0.0f;
```

### Files and functions

| File | Change |
|---|---|
| `search/agent.h/.cc` | `plans_`, `clear_plans()`, recording at the end of `run_search`, validity and `RuntimeConfig` fill at its start; `reset_search_state()` leaves `plans_` alone |
| `search/node.h/.cc` | `apply_root_plan`; promotion through `promote_joint_action` |
| `environment/joint_action.h` | generator `applyRootPlan(board, move, w)` beside `applyRootNoise` |
| `search/searchthread.cc` | `process_batch`: root mix for a fresh root |
| `interface/uci.cc` | options; `new_game()` (`uci.cc:109`) calls `agent->clear_plans()` |
| `tools/selfplay.cc`, `tournament.cc` | flag; plans cleared at game start |

### Interactions

* **Root noise.** Plan mix after the noise, so noisy self-play still explores; the noise is not discarded.
* **SitGate / sit-off.** Pass plans are not carried, so no plan pushes a sit. The mix only reweights legal root
  actions; legality is unchanged.
* **FlagLoss, InstantTransfer, MateRaceByLanding, PassCapK, seat options.** Not touched: the plan changes priors, not
  rules. The root mate provers can still overrule the choice.
* **Fix 1.** Applied to the adopted root as in 3.
* **Fix 2.** Committed moves feed plans (1).

### Thread safety and memory

Plans are written after workers join and read before they start, under `searchMutex_`. `RuntimeConfig` is copied into
each worker per search. Two slots, constant memory.

---

## Acceptance tests

New file `engine/tests/test_continuity.cc` unless a row says otherwise. Model-dependent tests need
`engine/models/twin-s-clock-it13.onnx` and skip without it, as `test_durative_part2.cc` does. Searches in tests use one
worker (`HIVEMIND_SEARCH_WORKERS=1` builds, or one thread) so they are deterministic.

### 0. Defaults

0.1 `ContinuityDefaults.OptionsAdvertisedAndOff`: the `uci` handshake lists the seven options with the defaults above;
    `current_clock()` with defaults equals the `261081b` clock for the same wave 2 options (`operator==` and
    `exact_key()`), on the wave 2 preset and on stock options.
0.2 `ContinuityDefaults.CandidateLayoutUnchanged`: `static_assert(sizeof(JointActionCandidate) == 48)` (or whatever it
    is at `261081b`, measured first), and every generated candidate has `plannedMask == 0` with fix 2 off.
0.3 `ContinuityDefaults.DurativeKeysUnchanged`: random durative lines (2000, as v4 test 4) give the same keys with
    `jointPlans` false as a build of the state without the new fields would (compare against keys recorded from
    `261081b` in `tests/data/durative_keys_261081b.txt`).
0.4 `engine/tests/wave2_parity.sh` (new, modelled on `stock_parity.sh`): builds this engine and `261081b` (a `git
    worktree` under the build dir, `JOBS=2`, one search worker each), runs `policy` and `go nodes 300` on
    `tests/data/stock_parity_positions.txt` with the headline wave 2 options plus `SeatClocksMs`/`SeatPaceMs`/
    `SeatElapsedMs`/`SeatArrivalsMs` per line, and compares as `stock_parity.sh` does. **8/8 identical.**
0.5 `stock_parity.sh`: **8/8 identical** (stock defaults).
0.6 `SelfPlayWave2Test.ContinuityOffIsByteIdentical`: a 2-game seat-clock self-play run with the wave 2 preset and no
    continuity flag writes the same HVM6 chunks, PGN and `seat_clocks.tsv` as recorded from `261081b`
    (`tests/data/selfplay_wave2_261081b/`).
0.7 The whole existing suite (519 tests at `261081b`: 518 pass, 1 skipped) still passes unchanged after each fix.

### 1. Fix 1

1.1 `TeamTreeReuse.ExactPathStillMissesUnderSeatClock` (documents the baseline): option off, a second search after a
    landing with the clocks 5 ms off adopts nothing.
1.2 `TeamTreeReuse.AdoptsAfterLandingWithClocksAFewMsOff`: search 800 nodes, play the chosen action and one opponent
    landing on the board, give the next search clocks 3, 5, 7 and 2 ms off the tree's path time; the adopted root is
    the expected node of the old tree (pointer, through a test peer), holds more than 0 visits, prints `Team tree
    reuse: N visits recovered`, and the search completes with N + 300 root visits at `go nodes 300`.
1.3 `TeamTreeReuse.RefusesBeyondTolerance`: the same with one clock `TeamTreeReuseToleranceMs + 1` off: fresh root,
    reason `clock`.
1.4 `TeamTreeReuse.RootShapeMismatchNotAdopted`: the matching node had one of our on-turn seats not ready; the live
    root (both our seats on turn) is not adopted, reason `shape`.
1.5 `TeamTreeReuse.RetimeRecomputesHashesAndPrunesShapeChanges`: an offset that flips a near-tie of ready times two
    plies below the adopted node: that node is replaced, every other retained node's hash equals
    `search_hash_key(team, pathTime)` under the new root time, and the transposition table holds exactly the walked
    nodes.
1.6 `TeamTreeReuse.ClockVerdictsDroppedWhenClocksMove`: a retained subtree with a FlagLoss-proven LOSS: dropped at a
    1 ms offset, kept at 0.
1.7 `TeamTreeReuse.BothSeatsOfTheTeam`: a sequence Aw decides, landings, Bb decides: Bb's search adopts a node of the
    tree Aw's search built.
1.8 `SelfPlayWave2Test.TeamTreeReuseInSeatClockGames` (`test_selfplay_wave2.cc`): 3 games, uniform net, 4 s base,
    wave 2 preset plus `--team-tree-reuse true`: the summary reports reuse in more than 0 searches, both seats of each
    team among the reusing decisions, one `seat_clocks.tsv` row per record as before, no error.
1.9 `TeamTreeReuse.MemoryBounded`: an old tree of more than 100k nodes: after adoption at most
    `TEAM_TREE_MAX_RETAINED` nodes are reachable from the root, and `Node::s_liveNodes` returns to that after the GC
    thread drains.
1.10 `TeamTreeReuse.NoPermanentBrainOverTeamTree`: UCI `go` with fix 1 and a durative clock leaves the team tree in
    place after `bestmove` and `stop` (the next search adopts).
1.11 `TeamTreeReuse.OpeningNoiseDiscards` (UCI) and `TeamTreeReuse.SelfPlayNoiseRefreshes` (self-play): as Interactions.
1.12 `TeamTreeReuse.FourWorkersAfterAdoption`: 50 repeats of adopt + 4-worker search on the uniform net: no crash, root
    visits = adopted + searched.

### 2. Fix 2

2.1 `JointPlans.PlannedHalfLandsAtItsReadyTime` (rules): a node where Aw is ready at 100 and Bb on turn at 400:
    `after_action` with a planned Bb move keeps Bb's ready time, stores the committed move, leaves board B unmoved; the
    node at 400 gives Bb only its committed move; the landing hands B to Bw as a normal move would.
2.2 `JointPlans.CommittedMovesStayLegal`: 2000 random lines with fix 2: every committed move is legal when it lands.
2.3 `JointPlans.SuppliedDropOnlyWhenTransferAllows`: a capture of a knight on A offers the partner's planned `N@`
    drops with InstantTransfer true, and with InstantTransfer false only for a sitting partner.
2.4 `JointPlans.FindsCaptureFeedingAMatingDropAtLowNodes` (needs it13). Construction:
    * Our team white (Aw, Bb). Board B: Bw's king on h1 smothered (rook g1, pawns g2, h2, nothing guarding f2), black
      (Bb) to move with an empty pocket: `N@f2#` needs a knight. Board A: Aw to move, `Qxf6` wins a black knight on f6
      that a pawn recaptures (a queen for a knight, so the capture looks bad on its own), and no mate for black on A
      within two plies.
    * The search root is a below-root state (test-only construction): a black-team root where Ab moves on A, so that
      after it white acts with Aw ready at 100 ms and Bb on turn but ready at 400 ms (Bb pace 500, elapsed 100);
      Ab's pace 50 ms, so Ab's reply on A lands at 150, between Aw's capture and Bb's landing. InstantTransfer true,
      FlagLoss true, SitGate false. Search `run_search` from that state for white with `SearchOptions::clock` set to it.
    * Expected: with fix 2 at N nodes, the most visited root edge is `(Qxf6, N@f2 planned)` and its child's value is a
      win (Q > 0.9 or solved WIN); with fix 2 off at the same N, the most visited edge's A half is not `Qxf6`. N is the
      smallest of 64, 128, 256, 512 where fix 2 passes; the test records N and the old tree's choice. If the old tree
      also finds `Qxf6` at that N, make the position harder (more Ab replies, a second plausible quiet move on A)
      until it misses, and say so in `docs/wave2_engine.md`.
    * Mirror for the black team (colours swapped) gives the same result.
2.5 `JointPlans.DecideLaterKeepsOldBehaviourAvailable`: at a planning node the "decide later" pair exists for every
    immediate half, with prior share at least `JointPlanFreePermille / 1000`.
2.6 `JointPlans.RootUnchanged`: with fix 2 on, a search from a UCI root (atRoot) has no planned root edge and the same
    root action set as fix 2 off.
2.7 `JointPlans.PvPrintsPlannedHalvesAsPass`: an `info ... pv` line through a planned step parses with the arena's
    `_INFO_RE` and shows the move at its landing step.

### 3. Fix 3

3.1 `PlanCarry.BiasesPartnersNextSearch` (needs it13): a position with our team on turn on A and B, where B has two
    near-equal moves. Search 1 chooses `(mA, mB)`. Search 2 at the position after `mA` and an opponent reply on A
    (B unchanged): with plan carry the root prior of `mB` equals `(1 - w) p + w`, the pair containing `mB` is expanded
    first, and at 64 nodes the chosen B half is `mB`; with plan carry off at 64 nodes on the same position it is the
    other move (choose the position so that holds; record both).
3.2 `PlanCarry.DroppedWhenBoardChangedOrIllegal`: board B changed, our pocket lost a piece, the move illegal, other team:
    each drops the slot with its reason.
3.3 `PlanCarry.NoPassPlans`: a chosen pass half records nothing.
3.4 `PlanCarry.AdoptedRootReweighted`: with fix 1, an adopted expanded root gets the mix through `apply_root_plan`
    (edge priors checked, order reindexed).
3.5 `PlanCarry.ClearedByNewGame`: `ucinewgame` clears both slots; `reset_search_state` alone does not.
3.6 `SelfPlayWave2Test.PlanCarryInSeatClockGames`: a seat-clock game with `--plan-carry`: some decisions record a plan
    for an undecided seat, and that seat's next search applies it (counter in the summary line).

### 4. No change to flag and terminal semantics

4.1 `ContinuityRules.DurativeAcceptanceHoldsWithContinuity`: v4 acceptance tests 1, 3, 4, 5, 6 (`DurativeAcceptance.*`,
    `DurativeMateRace.*`, `DurativeRepetition.*`, the search halves in `DurativeSearchTest` and
    `DurativePart2SearchTest`) rerun as typed/parameterised tests with `Continuity` on: same verdicts (single loser
    at 100/600 ms, sit-off draw and its 1 s flip, at most 29 landed moves for a 3 s seat with planned moves counted
    on landing, mirror mate race WIN/LOSS, no repetition draw in the durative tree).
4.2 `ContinuityRules.OutcomeUnchangedByCommitment`: a committed seat that flags before its landing loses on the clock
    (FlagLoss) exactly as an uncommitted one; a sit-off draw state is reached the same way.
4.3 `tools/regression.py run --engine wave2 --engine-nodes 800` with the continuity flags: drop_mates and sit-off
    family at least the `261081b` numbers (`docs/wave2_local_check.md` section 7: 80/80, 20/20 ahead, 0/20 behind).

### 5. Python and arena

5.1 `arena/tests/test_upstream.py`: `test_wave2_continuity_flags` (new keys, `continuity` expands to the three, unknown
    keys refused), `test_wave2_continuity_implies_reuse_tree` (no `ucinewgame` between searches, one at the
    controller's first search), `test_wave2_parses_reuse_line` (the `Team tree reuse` and `Plan carry` info strings
    reach `SearchResult`).
5.2 wave2-hivemind `tests/`: `Options` fields, `engine_args` emits only non-default flags, `--continuity`,
    `missing_engine_options` empty against the new `--help`.

## Order of implementation

One commit per numbered step at least, in this order; every step ends with the full C++ suite, `stock_parity.sh` and
`wave2_parity.sh` (8/8 each), and nothing is pushed.

1. Fix 1: options and plumbing (no behaviour), tests 0.1, 0.4, 0.5. Then `expansionKey_`, descent, re-timing,
   adoption, self-play; tests 1.x, 0.6, 0.7.
2. Fix 2: `TimeState` and generator (rules tests 2.1-2.3, 2.5, 0.2, 0.3); then search integration and PV (2.4, 2.6,
   2.7); then 4.1, 4.2.
3. Fix 3: recording, validity, bias, self-play; tests 3.x.
4. Arena, Python and cluster plumbing (below); 5.x; regression 4.3; a local smoke (2 pairs per row at 60 s on the Mac,
   plumbing only) and a rerun of `tools/continuity_probe.py reuse` and `scaling` with the new modes (`cont`,
   `reuse`, `joint`, `carry` added to its `MODES`) against `docs/continuity_baseline.md`.
5. `docs/wave2_engine.md` "Wave 2c" section, `cluster/wave2_engine.ref` updated to the new engine commit (the user
   pushes `talisman/wave2` and `custom-engine`).

Builds: `CMAKE_BUILD_PARALLEL_LEVEL=2` / `JOBS=2`, one build at a time; check memory first (a GPU memory guard job may
run). Python through the main repo `.venv` and the wave2-hivemind `uv` env.

## Arena and cluster plumbing

### Arena (`arena/players/upstream.py`, main repo)

* `WAVE2_FLAGS` (`upstream.py:84`) gains `"team_tree_reuse": ("TeamTreeReuse", False)`, `"joint_plans":
  ("JointPlans", False)`, `"plan_carry": ("PlanCarry", False)`, `"team_tree_reuse_tolerance_ms":
  ("TeamTreeReuseToleranceMs", 250)`, `"joint_plan_free_permille": ("JointPlanFreePermille", 500)`,
  `"plan_carry_permille": ("PlanCarryPermille", 300)`. `wave2_flag_options` validates the integer ranges as it does
  `pass_cap_k`. A player option `continuity: true` expands in Python to the three switches (no `Continuity` UCI sent,
  so the order of options never matters).
* Any of the three switches on implies `reuse_tree=True` for the process (no `ucinewgame` before every search), and the
  controller sends one `ucinewgame` at its first search, so the tree and the plans start clean each game. Rows with
  the switches off keep `reuse_tree` false, as in the PACE round.
* `UpstreamHivemindProcess._parse` records `reused_visits` and `plan_carried` from the info strings into
  `SearchResult`; the controller counts them and the eval summary reports reuse hits per search.
* `measure_wave2_ms_per_node` takes the new flags through `flags` as now, so `MEASURE=1` measures each row's own flag
  set (fix 2 changes node cost).
* Pool `arena/pools/wave2.json`: new opponent `wave2c-self` (the contender's `player_kw` plus `"continuity": true`),
  for the control row.

### `cluster/launch_checks.sh wave2c` (new block, not part of `all`)

Same common settings as `wave2` (`W2` array: pool, `SETUP=pool`, bughouse openings, rules v3, `MEASURE=1` at 800
nodes, `WARM_KINDS=upstream wave2`), against `upstream-it13` at 120 s, 600 pairs, no SPRT:

| Row | Opponent | `CONTENDER_KW` | Question |
|---|---|---|---|
| `c_all_120` | upstream-it13 | `'{"continuity": true}'` | all three fixes on top of the headline settings |
| `c_reuse_120` | upstream-it13 | `'{"team_tree_reuse": true}'` | fix 1 alone |
| `c_joint_120` | upstream-it13 | `'{"joint_plans": true}'` | fix 2 alone |
| `c_carry_120` | upstream-it13 | `'{"plan_carry": true}'` | fix 3 alone |
| `c_nodur_120` | upstream-it13 | `'{"durative_tree": false, "sit_gate": true}'` | the durative-off, gate-on reference, rerun on the new binary in the same round |
| `c_same_120` | wave2c-self | `'{"continuity": true}'` | control: continuity against itself must read 0.50 (800 pairs) |
| `c_ref_120` (with `WAVE2C_REF=1`) | upstream-it13 | `'{}'` | the durative-on reference on the new binary. Optional: tests 0.4 and 0.6 make the binary bit-identical to the one `ref_120` ran, so the PACE `ref_120` (0.227) stands unless the round wants a same-week reference |

Reading: each fix-alone row against `ref_120` (0.227, or `c_ref_120`); `c_all_120` against `c_nodur_120`. With about
1200 games a row's 95% range is about +-0.03, so a difference needs about 0.04. Read `c_same_120` first. The success
mark for this round is `c_all_120` at or above `c_nodur_120`; the PACE gap between durative on and off was 0.21. Each
row also reports reuse hits per search and plan carries per search (from the arena summary), so a fix that never fires
is visible. Cost: 6 rows x about 8 GPU-hours = about 48 GPU-hours (plus 8 for `c_ref_120`), behind the build with
`SBATCH_DEPENDENCY=afterok:<build job>`.

The header comment of `launch_checks.sh` lists `wave2c`; the `WHAT` error message names it; `docs/wave2_tests.md` gets a
"Wave 2c rows" section with this table.

### Training (`cluster/wave2_iterate.sbatch`, wave2-hivemind `hivemind.wave2`)

`hivemind.wave2.Options` gains `team_tree_reuse`, `joint_plans`, `plan_carry` (bool, default False) and the three
knobs, `BOOLEAN_FIELDS` and `add_arguments` the switches, and a `--continuity` argument that sets the three; the
`WAVE2` preset does not change. Then:

```sh
# a continuity lineage: self-play and gate matches with all three fixes
RUN=w2c ITERS=1 WAVE2_EXTRA="--continuity" sbatch cluster/wave2_iterate.sbatch
# one fix alone, as an ablation lineage
RUN=w2c-reuse ITERS=1 WAVE2_EXTRA="--team-tree-reuse true" sbatch cluster/wave2_iterate.sbatch
# A/B arms (narrow and wide) with continuity
RUN=w2c-ab AB=1 WAVE2_EXTRA="--continuity" sbatch cluster/wave2_iterate.sbatch
```

`WAVE2_EXTRA` is split on spaces and passed to `rl-loop` (`wave2_iterate.sbatch:176`), which passes the flags to the
engine's `selfplay` and `tournament`. With `--team-tree-reuse` self-play tops each search up to its node budget, so
records carry reused visits and a run plays more games per hour; compare lineages per game, not per hour. The usage
comment at `wave2_iterate.sbatch:35` gets the continuity example.

## Open (not in this spec)

* A durative-aware permanent brain (continue the team tree below our chosen action between decisions).
* Skipping the network evaluation at wait nodes and forced landing nodes.
* Landing-aware root mate provers (still off under MateRaceByLanding).
* Root semantics for a seat that is mid-decision in self-play (the root still lets every on-turn seat of the root
  team act at the root budget; fix 3 carries the half that is not played).
* TensorRT build of all of this on PACE (`cluster/build.sh`), after the user pushes.
