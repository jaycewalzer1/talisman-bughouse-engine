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
| `TeamTreeReuse` | check, false | `--team-tree-reuse <bool>` | Fix 1: keep the team's tree across decisions and adopt the node the game reached. The flag needs `--durative-tree true` and `--base-mix` (seat-clock games; the build refuses it without seat clocks too, since only `play_seat_clock_game` and `play_seat_clock_match` keep agents between decisions). |
| `TeamTreeReuseToleranceMs` | spin 0..5000, 250 | `--team-tree-reuse-tolerance-ms <ms>` | Fix 1: largest per-seat clock (and opponent ready-time) difference a retained node may have from the real clocks. |
| `JointPlans` | check, false | `--joint-plans <bool>` | Fix 2: below the root, a team that acts plans its other on-turn seat's move in the same joint action. |
| `JointPlanFreePermille` | spin 0..1000, 500 | `--joint-plan-free-permille <n>` | Fix 2: prior share of "decide later" (no plan) for the planned seat. |
| `PlanCarry` | check, false | `--plan-carry <bool>` | Fix 3: bias the next search toward the remembered half of the last chosen pair. The flag needs `--durative-tree true` and `--base-mix` (as built: only the seat-clock loops play one seat's half and keep the agent until the partner decides). |
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

1. **Descent (`Agent::find_team_tree_node`).** From the old root over expanded edges (as built: depth-first with
   make/unmake; the board test below leaves only the played line and its wait nodes, so the caps rarely bind and
   the order does not change which nodes are found). An edge is followed
   only if, after its applied moves (fix 2: planned halves are not applied), each board's ply count is at most the
   live board's (`Board::game_ply`) and a board whose ply count equals the live one has the live board's FEN
   (position, side to move, castling, en passant; pockets compared separately: each pocket a sub-multiset of the
   live one, equal at a candidate). The arena sends a FEN only and no move list (`position fen <A> | <B>`,
   `upstream.py:266`), so the FEN comparison leaves out the halfmove and fullmove counters. Ply counts are compared
   as offsets from the old root (moves applied on that board along the path, against the live ply minus the stored
   root ply), both read from the same front end's counters. Wait edges `(pass, pass)` and pass edges keep ply counts and are followed. Depth
   cap `TEAM_TREE_MAX_PLIES = 8` joint plies (wait nodes count), node cap `TEAM_TREE_MAX_VISITED = 20000`. The path
   time is carried down with the durative charge the search used (`TimeState::after_action` with the edge's applied
   moves and hand pieces, as `time_after_action` does in `searchthread.cc:1438`), not `advance()`.
2. **Candidates.** A node is a candidate when its board equals the live board (both FENs and pockets), its team to
   play is our team, it is expanded and `UNSOLVED`, and its path time is close to the live clock `T'` (the clock this
   search was given):
   * same sides to move, same sitting bits for the seats on turn, same pending (delayed transfer) counts, same
     `outcome` (NONE);
   * **no committed move for any seat** (fix 2), ours or the other team's. A live root never has one: the game has
     not committed the other team to anything, and a node whose state holds an opponent commitment would force that
     opponent move throughout the adopted subtree;
   * every seat clock within `TeamTreeReuseToleranceMs` of `T'`;
   * for the other team's seats on turn, ready time within the tolerance of `T'`'s;
   * the **shape check** at the candidate passes under `T'` (below). At the root this is the strict part: `T'` is
     `atRoot`, so our team acts on every board it is on turn on. A candidate where one of our on-turn seats was not
     ready (it acted on one board only) has the wrong shape and is not adopted.
   Among candidates: smallest largest-clock-difference, then most visits.
   (As built, the expansion key the shape check compares also holds the time inputs of the terminal check, see
   3(b), and a further reason `no tree` reports a search with nothing retained.)

   **Expected hit rate (read before tuning).** The Step 0 ceiling (33% of decisions) ignored clocks and shape. Two
   things cut it, and neither is a bug in the design:
   * the tree lands each opponent action at that seat's pace (an EMA), so at the candidate the opponent's clock is off
     by |real think time - pace|, which is often far above 250 ms;
   * the live root acts on every board our team is on turn on (`atRoot`), while in the tree a partner who is
     mid-decision or sitting acts at its own node (a sitting partner wakes 1 ms after the landing,
     `advance_action` step "a pass on the other board ends 1 ms after this landing"). When both our seats are on turn
     the candidate almost never has the root's shape.
   * (found while building fix 1) a durative node has one timing, not a choice: the tree holds the other team's
     landings only in the order their paces predict. When the real order differs, no node has the live boards
     (`no node`). In self-play a seat of ours whose move is in flight also acted at the old root (the root acts on
     every board our team is on turn), so the tree's line on that board is not the game's.
     First self-play numbers (`docs/wave2_engine.md` in the wave 2 repo, "Wave 2c"): it13, 400 nodes, 60 s,
     2 games: 28-33% of searches adopt at 250-3000 ms, the refusals split about evenly between `no node` and
     `shape`; on the uniform test net 2-4%, almost all `no node`.
   So step 1 of the implementation order ends with a **gate**: `tools/continuity_probe.py reuse` with fix 1 reports
   the hit rate and the refusal reasons (`no node`, `clock`, `shape`, `committed`, `solved`) at tolerances 250, 1000
   and 3000 ms, and the distribution of the largest clock difference at board-matching nodes. If the hit rate at the
   chosen default is under 10% of searches, report it and choose the default from the data before fixes 2-3 and
   before any PACE row. A larger tolerance is safe for correctness (everything below the root is re-timed, re-shaped
   and verdict-checked); it only trades how stale the retained statistics are. Retained priors and values were also
   computed from the old node's clock planes (and, for a wide network, its timing planes); that is part of the same
   approximation.
3. **Re-timing (`Agent::retime_team_tree`).** Nodes store no time: every path time is recomputed from the root time
   at each selection (`select_and_expand`, `pathTime`). What a node does store and depends on time is (a) its hash,
   (b) its expansion (which boards acted, forced waits, whether a sit was legal, which drops were held back), (c) a
   solver verdict, (d) its transposition-table entry. The re-timing walk visits the adopted subtree from `T'`,
   carrying the new path time, single-threaded, before workers start. It is a depth-first walk with
   `make_moves`/`unmake_moves`, because the path time needs `hand_piece_type` and the expansion key needs the board's
   legal drops and pockets. A breadth-first count runs first, without a board, and marks the first
   `TEAM_TREE_MAX_RETAINED` nodes, so the cap keeps the shallow part of the tree. The walk only enters marked nodes.
   * **The tree is a graph (MCGS).** `canonicalize_child` merges children with equal `search_hash_key`, which
     includes the durative key, so one node can have several parents. Two paths that reached the same time under the
     old root can reach different times under `T'`, because flag clamps, pass caps and wake-ups are not
     shift-invariant. The walk keeps a `visited` map from node to the hash it recomputed. When a node is reached
     again with a different recomputed hash, the edge from the later parent is replaced by a fresh unexpanded node:
     the node is split, and the shared copy keeps the hash and state of its first path. With an equal hash the node
     stays shared and is not walked again;
   * (a) `set_hash(board.search_hash_key(team, nodeTime))` on every visited node, then the node goes into the fresh
     transposition table (this replaces `reindex_reused_subtree`'s stored-hash insert, `agent.cc:8982`, for this path).
     Before overwriting, the walk keeps the node's **old** hash for (c);
   * (b) **shape check**: each node stores at expansion a 32-bit `expansionKey_` (new `Node` field, written in
     `try_init_and_expand`): a hash of the inputs that fixed its action lists besides the board: acting boards after
     the durative filter (`searchthread.cc:914-920`), `forcedWait`, the leaf `can_sit` (`leafTeamHasTimeAdvantage`,
     `:1055`), the held-back drop counts of the acting seats (`pending_in_hand`), and under fix 2 the planned board
     and any committed move. **Correction (fix 1 build):** also the time inputs of the terminal check that a node
     passed once, before it was expanded: the other team's `can_sit`, a clock `outcome`, and with
     MateRaceByLanding the landing order of the seats on turn, the early landings and the last capturer. An
     expanded node is never classified again, so without them a node that is terminal under `T'` (a flag, a
     landing race) would stay expanded. The walk recomputes the key under the new time; a mismatch replaces that child with a
     fresh unexpanded node (`Node::replace_child`, `node.h:828`; the old subtree goes to `discard()`);
   * (c) a node holding a solver verdict (`WIN`, `LOSS`, `DRAW`) is kept only when its recomputed hash equals its
     old hash. The hash holds the whole durative key, so equal hashes mean the node's time state is unchanged. A
     clock comparison alone is not enough: `rootMoveMs` (each search's own budget, which the arena changes every
     decision) and the paces set every ready time of our team below the root (`action_time_ms`). Otherwise the node
     is replaced by a fresh node. Flags, sit-off draws and landing-time mate races hinge on milliseconds, and a fresh
     node re-proves a board mate at its first visit. The walk goes top-down, so the topmost solved node is replaced
     and its parent is always `UNSOLVED`. That is the case in which `Node::replace_child` (`node.h:828`) restores
     `childNodeTypes`, `unsolvedChildCount` and `provenWinningChildCount`;
   * nodes beyond `TEAM_TREE_MAX_RETAINED = 100000` walked nodes (`SearchParams::TREE_REUSE_REINDEX_MAX_NODES`) are not
     validated, so their parents' edges to them are replaced by fresh nodes too. Nothing unvalidated stays reachable.
   * **Edge statistics where a child was replaced (by shape, verdict, split or cap) are cleared.** The parent's edge
     goes back to a new edge's state: 0 visits, Q and value sum at the `Q_INIT` seed (**correction:** the draft said
     0; the first backup of an exact-backup edge adds to the seed, so the seed has to be `Q_INIT`), and the removed
     visits and value are taken off the parent node's own `m_visits` and `valueSum`, with the sign convention backup
     uses (`accumulate_edge_value`). An edge seeded with a transposition's pseudo-visit takes one visit more off
     than the parent counted for it; totals stay at least 0. The fresh node takes its path's recomputed hash and
     goes into the new transposition table, as `canonicalize_child` would at its first visit (so after a split the
     table holds both, test 1.15). If they were kept, a
     stale verdict would carry on as a soft one: an edge that proved a flag win holds Q near +1 over hundreds of
     visits, and PUCT and the final visit-count choice would follow it after the proof itself is gone. Ancestors
     above the parent keep their totals, which is a stated approximation bounded by the removed visits. Test 1.14
     covers it.
   * The adopted root's `parentEdges` are cleared, so that a proof found in the new search does not propagate into
     the discarded old tree.
4. **Adoption** goes through the existing guards in `run_search` (`agent.cc:6235-6267`: must-play loss, unexpanded
   loss, claimed win that loses the mate race), then `set_hash`, `set_depth(0)`, root noise, `configure_root_search`.
   Verbose UCI prints `info string Team tree reuse: <visits> visits recovered, clock diff <ms> ms, <pruned> pruned`
   (or `... none (<reason>)` with reason `no node`, `clock`, `shape`, `committed`, `solved`), so the arena and the probe can count
   hits. The old exact path (`try_reuse_tree`) is not called when fix 1 is on for a durative clock.
5. **Node budget.** UCI `go nodes N` still searches N new nodes on top of the adopted visits
   (`reusedVisitsCount = false`), and `info ... nodes` reports new nodes only (`agent.cc:7751`), so the arena's
   "searched" charge stays what the engine did this decision. Self-play with fix 1 tops the root up to its budget
   (`reusedVisitsCount = true`, `minNewNodes = treeReuseMinNewFraction x budget`, as `--tree-reuse` does,
   `selfplay.cc:1550`).
6. **After the search.** With fix 1 on and a durative clock, the end of `run_search` (`agent.cc:8455-8475`) stores the
   whole tree as the `TeamTree` instead of `store_next_root_candidates`, and leaves `nextRootCandidates_` empty. This
   store does not depend on `options.retainForReuse`, which self-play sets to false (`selfplay.cc` search options).
   A missed ponder still clears it, as it clears the candidates now.
7. **Other state that persists between searches.** In the PACE rows `ucinewgame` came before every search, so
   nothing survived from one search to the next. With fix 1 or fix 3 on, the arena stops sending it (arena section).
   To keep each row's only difference its fix, `run_search` under a durative clock with `teamTreeReuse` or
   `planCarry` on starts by clearing `mateContinuations_` (a legacy-flag cache that is not re-timed;
   `try_reuse_mate_continuation`, `agent.cc:7529`), `nextRootCandidates_` and `lastSearchHash_`. With `planCarry` on
   and `teamTreeReuse` off it also drops the retained root, as `reset_search_state` does, keeping only `plans_`.

### Data structures

```cpp
// agent.h, beside nextRootCandidates_
struct TeamTree {
    std::shared_ptr<Node> root;    // the last search's root; owns the tree
    std::shared_ptr<Board> board;  // its position, to replay edges from (added in the build)
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
| `tools/selfplay.cc` | `play_seat_clock_game` (`:904`): with `--team-tree-reuse`, no `agent.reset_search_state()` per decision (`:1045`; still once at game start, `:933`); the same in `tournament.cc` `play_seat_clock_match` (per-decision reset at `:564`, plus a new game-start reset), `searchOptions.teamTreeReuse`, top-up budget; summary line `selfplay team tree reuse: <hit %> of searches, <mean visits>` like `:1745`; the comment at `:903` updated |
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
  keeps the adopted root and re-noises it (`Node::refresh_root_noise`, as `run_search` already does at `:6273`),
  unless the adopted node is the old root itself, which already holds that search's noise (review fixes 1).
* **SitGate.** `can_sit` at each node is in the expansion key, so a shift that changes the gate or the sit-off rule
  re-expands that node.
* **FlagLoss.** Verdicts are dropped wherever the node's time state changed (3c), and their edge statistics go with
  them. A flag in the subtree is found again under `T'`.
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

**What this means under the headline flags.** Wave 2 plays `InstantTransfer false` (`WAVE2_FLAGS`,
`upstream.py:85`). A planned supplied drop is then possible only for a sitting partner. The v4 tree already wakes a
sitting partner 1 ms after the capture lands, and it acts at the next node of our team, one forced wait later. So
the drop rule adds little in the headline configuration. What fix 2 adds there is the pairing itself: the partner's
quiet move or capture is chosen together with ours, and a capture by the planned half feeds our immediate seat's
**next** decision when it lands before that decision starts. Test 2.4 therefore has a headline-flag variant
(2.4b). Do not read a pass of 2.4a (InstantTransfer true) as evidence for the headline rows.

**Priors.** Immediate halves as today. Planned half: the node's network policy for that board over the candidate list
(drop indices that are illegal before the capture take their raw logits, softmaxed with the rest at the node's policy
temperature). Then "decide later" gets `max(pass prior, JointPlanFreePermille / 1000)` and the moves share the rest in
proportion. Then each planned move that gives mate on its landing board (checked once per candidate at expansion, on
the board with the needed piece in the pocket) is raised to at least `SearchParams::JOINT_PLAN_MATE_PRIOR` = 0.25 and
the half is renormalised. The mate test runs in `collect_batch`, where the board is at hand (`process_batch` has no
board), and only on candidates that give check. It costs one make, a legal-move count and one unmake per checking
candidate at each planning leaf. The scaling probe records how much it changes nodes/ms.
The mate raise is there because the planned half's priors are read before the immediate half's capture reaches the
pocket; set the constant to 0 to ablate it. Joint factors (`jointFactorsA/B`,
`searchthread.cc:1105`) apply as for any half.

**Pair legality.** The generator (`JointCandidateGenerator`, `joint_action.h:220`) checks immediate halves exactly as
today (`is_joint_action_legal` with only the acting boards on turn). A pair with a planned drop that needs a supplied
type is legal only if the immediate half supplies that type. `legalPriorMass` gains the supplied-type categories (at
most 6 x 6 products) so widening by prior mass stays exact.

### Data structures

```cpp
// joint_action.h: JointActionCandidate. moveA/moveB stay the halves that are applied now; the planned half lives
// in its own field, in the 4 bytes of padding after gumbelScore (offset 28, before the 8-aligned idxA; the
// static_assert on sizeof stays 48).
Stockfish::Move planned = Stockfish::MOVE_NONE;   // the planned seat's move; MOVE_NONE = no plan / decide later

// node.h: Node. Which board a planning node plans on, -1 = none. One per node: at most one board can be on turn
// for the acting team without being ready, because the other board is the one it acts on.
// (As built: an int8_t in the candidate generator's padding, read through Node::planned_board(); the per-action
// needed/supplied types are packed into bits 1-3 of the generator's capture bytes, so sizeof(Node) stays 1040.)
int8_t plannedBoard_ = -1;

// time_state.h: DurativeConfig
bool jointPlans = false;
int32_t jointPlanFreePermille = 500;

// time_state.h: TimeState::Durative
bool jointPlans = false;
int16_t jointPlanFreePermille = 500;   // as built: the planning nodes' prior knob travels here; not in the key
std::array<Stockfish::Move, 4> committed{MOVE_NONE, MOVE_NONE, MOVE_NONE, MOVE_NONE};
```

`Durative::key()` mixes `committed` only when `jointPlans` is set, so keys with fix 2 off are exactly v4's.

**Why the planned half gets its own field (review change).** The draft stored the planned move in `moveA`/`moveB`
with a `plannedMask`, which made every `make_moves(action.moveA, action.moveB)` on a below-root edge wrong unless it
was rewritten. There are 11 such calls in `searchthread.cc` and 54 in `agent.cc`, and many in `agent.cc` walk
below-root edges: PV and mate-line extraction (`:2816`, `:2949`), certification and prover walks (`:3070`, `:5081`,
`:5422`), and the retained-candidate and reuse walks (`:8776-8878`, `:9075`, `:9250`, `:9305`). A missed site would
play the planned move early: one seat acting before its ready time, with the board and the durative `side` no longer
agreeing. With the separate field every existing site stays correct as it is.

The new rule is the other way round. Any code that looks up or compares a **below-root** action by
`(moveA, moveB)` must also compare `planned`:
* `JointCandidateGenerator::promote`;
* mate-hint matching (`mateCandidateTable`, `hintAction` in `select_and_expand`);
* `retain_played_line`;
* any child-by-action lookup.

The generator keeps the planned board's list in the slot that board's `{MOVE_NONE}` list has today, so the
`(idxA, idxB)` identity stays unique. When it builds a candidate it writes that list's entry into `planned` and
leaves that board's move field `MOVE_NONE`. `is_joint_action_legal` therefore sees that board as not acting, exactly
as today, and "decide later" is never judged as a sit. Only `time_after_action` (which passes `planned` and
`plannedBoard_` to `advance_action`), the expansion key, PV printing and fix 1's descent (which ignores it) read the
field.

**Invariant (debug builds and tests).** At every leaf in `collect_batch`, `board.side_to_move(b) ==
ctx.time.side_on(b)` for both boards under a durative clock. A planned move applied early, or a committed move never
applied, breaks it at once.

New `TimeState` API: `planned_boards(team)` (on-turn, not ready, not committed boards of `team`; empty outside a
durative state, at the root or with fix 2 off), `committed_move(seat)`, and `advance_action(team, movedA, movedB,
handA, handB, plannedA, plannedB)` with the two new arguments defaulting to `MOVE_NONE`: a planned move is stored as
the seat's committed move and changes no ready time; a seat that acts with its committed move clears it; a committed
seat counts as "deciding" for the pass wake-up rule (it lands at its ready time). A committed seat may not be planned
again. Flags run as today: a committed seat whose clock runs out before its landing flags (FlagLoss).

**A sitting seat that receives a plan.** This is the main planned-drop case under `InstantTransfer false`. The seat
keeps `sitting = true` for **timing and transfer**:
* its landing is its wake-up, and the rule "a pass on the other board ends 1 ms after this landing" still fires
  when the immediate half moves, so the planned drop lands 1 ms after the capture;
* `pending_in_hand` is still 0 for it, and `pending` is cleared when it acts.

For the **sit-off rules** a committed seat counts as not sitting:
* the cross-team sit-off `DRAW` check at the end of `advance_action` skips committed seats (otherwise an opponent
  sitting on the other board at an equal clock would end the line as a draw although our committed move is
  certain to land);
* in `sit_off_forces_move`, a committed seat is not the "other team's sitting seat".

Both checks read `committed`, so with fix 2 off they are unchanged.

### Files and functions

| File | Change |
|---|---|
| `environment/time_state.h` | `DurativeConfig::jointPlans`, `Durative::committed`, key, `planned_boards`, `committed_move`, `advance_action` planned arguments, `durative()` copies the switch |
| `environment/joint_action.h` | `planned` field; generator takes the planned board, per-action needed type (planned board) and supplied type (immediate board), writes the planned entry into `planned`; `promote` compares `planned`; pair legality and `legalPriorMass` |
| `search/node.h` | `plannedBoard_`; `try_init_and_expand` passes the planned board and the new per-action vectors |
| `search/searchthread.h` | `LeafContext` gains `plannedA/B`, `forcedA/B`, the per-action needed/supplied vectors; `TrajectoryEntry` unchanged (the candidate carries the mask) |
| `search/searchthread.cc` | `collect_batch` (`:914-965`): planned and forced boards, candidate lists, mate tags; `process_batch` (`:1055-1118`): planned-half priors; `select_and_expand` (`:1438-1590`): `time_after_action` passes `action.planned` and the node's `plannedBoard_` to `advance_action`; make/unmake sites and the per-board ply count (`:810-817`) are unchanged, because `moveA`/`moveB` are the applied halves; mate-hint matching compares `planned`; the debug invariant `side_to_move == side_on` |
| `search/searchthread.cc` `classify_terminal_position` / `has_unavoidable_waiting_board_mate` | a seat with a committed move is not credited with a waiting-board mate unless its committed move is that mate |
| `search/agent.cc` | PV printing and `extract_pv_from_child`: a planned half prints as `pass` at its step (it already does, the move field is `MOVE_NONE`), the move prints at its landing step; below-root child lookups by action compare `planned` |
| `interface/uci.cc` | option; `current_clock()` (`uci.cc:764`) sets `config.jointPlans`, `jointPlanFreePermille` |
| `tools/seat_clock_game.h`, `selfplay.cc`, `tournament.cc` | flags into `durative_config()` |
| `search/search_params.h` | `JOINT_PLAN_MATE_PRIOR` |

### Interactions

* **Root.** Unchanged (Common rules). Root edges never carry a `planned` move, so `bestmove`, `root_edge_stats`,
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
  with a committed move for **any** seat cannot be adopted as a root (reason `committed`, fix 1 step 2). For our
  seats this is because both our on-turn seats decide at the root; for the other team's seats it is because the
  real opponent has committed to nothing. Fix 3 then still carries our plan (below). Fix 1's descent follows edges
  by their applied moves only, which are `moveA`/`moveB` under the representation below.
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
   team, edgeQ, edgeVisitShare}`. Each search replaces both slots (an empty half clears its slot). A second source,
   used under fix 2: at the **start** of the next search, if fix 1 refused a board-matching candidate only because
   it held a committed move for one of **our** seats, that commitment is the tree's plan for that seat. It fills that
   board's slot only when the slot is empty, because the slot written at the end of the last search wins. It then
   goes through the same validity check (2). Commitments of the other team's seats are never recorded.
2. **Validity at the next search.** A slot applies when the team is the same, our team is on turn on that board, the
   board's position, side to move, castling and en passant equal the recorded FEN, our seat's pocket holds at least
   what it held, and the move is one of this root's actions on that board. That means it is legal now **and** it
   survives the held-back-drop filter (`prepare_actions`, `searchthread.cc:930`, InstantTransfer false). Otherwise the
   mix would put weight on an action that does not exist and the board's priors would no longer sum to 1. Anything
   else drops the slot. So the half that was played (its board
   changed) is never carried.
3. **Bias.** At the root, the carried board's prior is mixed: `p(move) <- (1 - w) p(move) + w`, every other action on
   that board `p <- (1 - w) p`, with `w = PlanCarryPermille / 1000`, after root noise. Then the best joint candidate
   containing the move (by joint prior) is promoted to be expanded first (`Node::promote_joint_action`, `node.h:578`),
   so it gets its first visit before any other new edge: that is the "initial visit". A fresh root gets the mix in
   `process_batch` at its expansion (`searchPly == 0`); an adopted, already expanded root (fix 1) gets it through a new
   `Node::apply_root_plan(board, move, w)` that reweights the generator's priors under the node lock, like
   `refresh_root_noise` (`node.cc:230`): mix, `resortBoard`, `legalPriorMass`.

   **Order matters.** `configure_root_search` calls `reprepareJointPolicyPool` -> `rebuildCandidateFrontier`, which
   recomputes every joint prior from the per-board lists and **resets `promotedCandidate`**
   (`joint_action.h:403-412`). So:
   * on an adopted root the order is: noise (`refresh_root_noise`), plan mix (`apply_root_plan`),
     `configure_root_search`, then the promotion;
   * on a fresh root, `configure_root_search` returns early because the root is not yet expanded. The mix goes into
     the per-board lists at the root's expansion in `process_batch`, after the root noise. The promotion comes after
     `try_init_and_expand` has built the frontier, still under the root's lock, before any worker can select a child.

   A promotion made earlier is silently lost. Test 3.1 checks the first expanded root edge, not only the prior.
4. **Telemetry.** Verbose UCI: `info string Plan carry: <board> <move> weight <w>` or `... dropped (<reason>)`.

### Self-play

`play_seat_clock_game` already plays only the deciding seats' halves (`selfplay.cc:1141-1152`). The undecided seat on
turn (if any) gets the other half recorded as above; when that seat's own `DECIDE` event comes, its search carries the
plan. In self-play a seat's move is committed at its search and lands later, so the team's other on-turn seat can be
one whose move is already decided and in flight (`committed[seat]`), not an undecided one. Its slot is not written,
because the plan would be moot: the in-flight move lands and changes the FEN. With `--team-tree-reuse` the agent is
not reset between decisions; without it, `reset_search_state()` must not clear the plan slots (they live outside the
tree). The game-start loop (`selfplay.cc:933`) calls `reset_search_state()`, which leaves plans alone by design, so it
must also call `clear_plans()` explicitly. Otherwise the last game's plans leak into the next game on the same
agent.

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

// SearchParams::RuntimeConfig (per-search copy, immutable during the search). One entry per board: both slots can
// be valid at once (both our seats on turn, both halves recorded, neither landed yet).
std::array<Stockfish::Move, 2> rootPlanMove{Stockfish::MOVE_NONE, Stockfish::MOVE_NONE};
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
| `tools/selfplay.cc`, `tournament.cc` | flag; `clear_plans()` beside the game-start `reset_search_state()` (`selfplay.cc:933`). `tournament.cc` `play_seat_clock_match` has no game-start reset, only a per-decision `agent.reset_search_state()` (`:564`). It gains `reset_search_state()` + `clear_plans()` for both agents at game start, and skips the per-decision reset under `--team-tree-reuse`, as self-play does. The contender and baseline agents swap teams between games, so the `team` checks on `TeamTree` and `CarriedPlan` are what keep a tree or plan from crossing to the other team there |

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
    is at `261081b`, measured first), and every generated candidate has `planned == MOVE_NONE` and every node `plannedBoard_ == -1` with fix 2 off.
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
    reuse: N visits recovered` where N is the root's visits **after** re-timing (cleared edges subtracted), and the
    search completes with N + 300 root visits at `go nodes 300`.
1.3 `TeamTreeReuse.RefusesBeyondTolerance`: the same with one clock `TeamTreeReuseToleranceMs + 1` off: fresh root,
    reason `clock`.
1.4 `TeamTreeReuse.RootShapeMismatchNotAdopted`: the matching node had one of our on-turn seats not ready; the live
    root (both our seats on turn) is not adopted, reason `shape`.
1.5 `TeamTreeReuse.RetimeRecomputesHashesAndPrunesShapeChanges`: an offset that flips a near-tie of ready times two
    plies below the adopted node: that node is replaced, every other retained node's hash equals
    `search_hash_key(team, pathTime)` under the new root time, and the transposition table holds exactly the walked
    nodes.
1.6 `TeamTreeReuse.ClockVerdictsDroppedWhenClocksMove`: a retained subtree with a FlagLoss-proven LOSS: dropped at a
    1 ms clock offset; kept at 0 offset with the same `RootMoveTimeMs` and paces; **dropped** at 0 clock offset when
    `RootMoveTimeMs` differs by 1 ms, and again when one opponent pace differs by 1 ms (the hash criterion, 3c).
1.7 `TeamTreeReuse.BothSeatsOfTheTeam`: a sequence Aw decides, landings, Bb decides: Bb's search adopts a node of the
    tree Aw's search built.
1.8 `SelfPlayWave2Test.TeamTreeReuseInSeatClockGames` (`test_selfplay_wave2.cc`): 3 games, uniform net, 4 s base,
    wave 2 preset plus `--team-tree-reuse true`: the summary reports reuse in more than 0 searches, both seats of each
    team among the reusing decisions, one `seat_clocks.tsv` row per record as before, no error.
    **As built:** 4 games at 20 s and 256 nodes with the 5000 ms tolerance (at 4 s and 16 nodes the uniform net's
    trees are too shallow to adopt anything), and "both seats of each team" is not asserted: which seats adopt
    varies run to run with four search workers (1 run in about 10 had one seat per team). Test 1.7 checks the
    cross-seat adoption deterministically.
1.9 `TeamTreeReuse.MemoryBounded`: an old tree of more than 100k nodes: after adoption at most
    `TEAM_TREE_MAX_RETAINED` nodes are reachable from the root, and `Node::s_liveNodes` returns to that after the GC
    thread drains. **As built:** the cap is a parameter of the walk; the test runs it at 60 on a 2000-node tree
    (the replacements are fresh unexpanded nodes on top of the cap).
1.10 `TeamTreeReuse.NoPermanentBrainOverTeamTree`: UCI `go` with fix 1 and a durative clock leaves the team tree in
    place after `bestmove` and `stop` (the next search adopts).
1.11 `TeamTreeReuse.OpeningNoiseDiscards` (UCI) and `TeamTreeReuse.SelfPlayNoiseRefreshes` (self-play): as Interactions.
1.12 `TeamTreeReuse.FourWorkersAfterAdoption`: 50 repeats of adopt + 4-worker search on the uniform net: no crash, root
    visits = adopted + searched.
1.13 `TeamTreeReuse.RetainedExpansionsMatchFreshOnes` (independent of `expansionKey_`): after adoption, for every
    retained expanded node, rebuild its action lists from scratch the way `collect_batch` does (acting boards,
    forced wait, `can_sit`, held-back drops) under its new path time, and compare them with the node's generator
    lists as move sets. Run it over 20 adoptions in self-play-like sequences with clock offsets of 0-200 ms.
    Otherwise an expansion key that leaves out an input would pass 1.5 and still keep wrong expansions. The same
    check runs as a debug-only `--team-tree-reuse-validate` inside 1.8. **As built:** 20 re-timings of searched
    trees (pockets, captures, near-tie paces) with random offsets of 0-200 ms; the debug flag is not built.
1.14 `TeamTreeReuse.StaleFlagEdgeCarriesNoValue`: the old tree proved a FlagLoss win along one of the adopted root's
    edges (hundreds of visits, Q near +1). The new clocks no longer flag. After adoption that edge has 0 visits and Q
    0, the root's visits fell by the same amount, and at `go nodes 64` the chosen action is not the stale line unless
    the fresh search proves it again. This fails if only the child node is replaced and the edge statistics are
    kept. **Correction:** an unsolved root cannot have a root edge proven won (one child proven lost makes the root
    a proven win, and a solved node is never adopted). So the test re-times a solved root directly
    (`retime_team_tree`): the edges into children proven lost for the flagging team have 0 visits and Q at
    `Q_INIT` afterwards, the children are fresh, and the root's visits fell by the removed visits. On an adopted
    root the same clearing applies to an edge into a child proven *won* for the opponent (Q near -1), and below.
    On the uniform net the stale edges hold a few visits, not hundreds; the `go nodes 64` half is not built.
1.15 `TeamTreeReuse.TranspositionSplitWhenPathsDiverge`: a retained node with two parents whose paths reach different
    times under `T'` (one path's pass is clamped by the flag, the other's is not) is split. Each parent's child has
    the hash of its own path, and the table holds both. **As built:** the shared node is made by hand (a
    grandchild of one root edge put under another), since the walk only compares the recomputed hashes; a
    time-only divergence was not constructed.
1.16 (built with fix 2 as `JointPlansSearch.TeamTreeReuseRefusesCommittedCandidate`) `TeamTreeReuse.RefusesCommittedCandidate` (fix 2 on): a board-matching node with a committed move for our
    partner is refused with reason `committed`, and so is one with a committed move for an **opponent** seat.
1.17 `TeamTreeReuse.NoOtherStateSurvives`: with fix 1 (and, separately, fix 3 alone) on under a durative clock and no
    `ucinewgame`, `mateContinuations_` and `nextRootCandidates_` are empty at the start of each search (7).
    `Tree reuse:` (the exact path) is never printed.

### 2. Fix 2

2.1 `JointPlans.PlannedHalfLandsAtItsReadyTime` (rules): a node where Aw is ready at 100 and Bb on turn at 400:
    `after_action` with a planned Bb move keeps Bb's ready time, stores the committed move, leaves board B unmoved; the
    node at 400 gives Bb only its committed move; the landing hands B to Bw as a normal move would.
2.2 `JointPlans.CommittedMovesStayLegal`: 2000 random lines with fix 2: every committed move is legal when it lands,
    and at every node `board.side_to_move(b) == time.side_on(b)` on both boards (the invariant under Data
    structures). Also run the search-level version: a 4000-node fix-2 search with the debug assertion on, followed by
    PV extraction, `retain_played_line` and a fix-1 adoption walk. The board's FENs equal the root's afterwards.
2.3 `JointPlans.SuppliedDropOnlyWhenTransferAllows`: a capture of a knight on A offers the partner's planned `N@`
    drops with InstantTransfer true, and with InstantTransfer false only for a sitting partner.
2.4a `JointPlans.FindsCaptureFeedingAMatingDropAtLowNodes` (needs it13). Construction:
    * Our team white (Aw, Bb). Board B: Bw's king on h1 smothered (rook g1, pawns g2, h2, nothing guarding f2), black
      (Bb) to move with an empty pocket: `N@f2#` needs a knight. Board A: Aw to move, `Qxf6` wins a black knight on f6
      that a pawn recaptures (a queen for a knight, so the capture looks bad on its own), and no mate for black on A
      within two plies.
    * The search root is a below-root state (test-only construction): a black-team root where Ab moves on A, so that
      after it white acts with Aw ready at 100 ms and Bb on turn but ready at 400 ms; Ab's reply on A lands at 150,
      between Aw's capture and Bb's landing. **Correction (fix 2 build):** below a black-team root every black action
      costs `RootMoveTimeMs`, so "Ab's pace 50 ms" is `RootMoveTimeMs` 50, and Bb needs pace 500 with elapsed 50 (not
      100) to be ready at 400 after the 50 ms root. Aw's own next landing then also falls due at 400 (100 + 50 + 100 +
      50 + 100), so white acts on both boards there, the B half forced. InstantTransfer true, FlagLoss true, SitGate
      false. Search `run_search` from that state for white with `SearchOptions::clock` set to it.
    * Expected: with fix 2 at N nodes, the most visited root edge is `(Qxf6, N@f2 planned)` and its child's value is a
      win (Q > 0.9 or solved WIN); with fix 2 off at the same N, the most visited edge's A half is not `Qxf6`. N is the
      smallest of 64, 128, 256, 512 where fix 2 passes; the test records N and the old tree's choice. If the old tree
      also finds `Qxf6` at that N, make the position harder (more Ab replies, a second plausible quiet move on A)
      until it misses, and say so in `docs/wave2_engine.md`.
    * Mirror for the black team (colours swapped) gives the same result.
    **As built (fix 2):** the old tree finds `Qxf6` too, at every N from 64 to 2048, with and without the root mate
    provers, and proves it within 7 root visits (fix 2: 9): after the capture black is on turn on one board while
    `N@f2#` waits on the other and survives every reply, which `has_unavoidable_waiting_board_mate` proves. A harder
    position the old tree misses would no longer be a mate that survives every reply, so not a forced win. The test
    records that both trees choose and prove the capture; fix 2's proven edge is `(Qxf6, decide later)`.
2.4b `JointPlans.HeadlineFlagsCoordination` (needs it13). Same expectations, with the headline flags
    (InstantTransfer **false**, FlagLoss true, SitGate false), so the test exercises what the PACE rows play. Here the
    transfer runs the other way, because a deciding partner cannot drop a piece that lands mid-decision:
    * the planned half (Bb, on turn, ready at 400) captures a piece that Aw needs;
    * the immediate half is Aw's preparatory move on A;
    * Ab's reply lands **after** 400, so Aw's next decision starts after the capture landed and the piece is not
      pending;
    * Aw then drops it for mate.
    The expected root edge is `(prep on A, capture on B planned)`. The same N rule and old-tree comparison apply as
    in 2.4a. If no such position can be built so that the old tree misses and fix 2 finds it, say so plainly in
    `docs/wave2_engine.md`: it would mean fix 2 has no tested benefit under the headline flags.
    **As built: not built, no such position found.** Ab's reply comes before Aw's next decision by construction, and
    Ab can always use it to defuse a drop mate (a pawn or rook move frees a square) or sit on A (SitGate off), so the
    line is never forced. Fix 2 has no tested benefit under the headline flags; the `c_joint_120` row is the test.
2.5 `JointPlans.DecideLaterKeepsOldBehaviourAvailable`: at a planning node the "decide later" pair exists for every
    immediate half, with prior share at least `JointPlanFreePermille / 1000`.
2.6 `JointPlans.RootUnchanged`: with fix 2 on, a search from a UCI root (atRoot) has no planned root edge and the same
    root action set as fix 2 off.
2.7 `JointPlans.PvPrintsPlannedHalvesAsPass`: an `info ... pv` line through a planned step parses with the arena's
    `_INFO_RE` and shows the move at its landing step.
2.8 `JointPlans.CommittedSittingSeatIsNoSitOff`: our partner sits on B and receives a planned move; the opponent on A
    then sits at exactly the partner's clock. The state is **not** a sit-off `DRAW`, `sit_off_forces_move` ignores
    the committed seat, and the committed move lands 1 ms after the immediate half's landing (the wake-up rule).
2.9 `JointPlans.ActionLookupsComparePlanned`: at a planning node, two candidates `(mA, -, planned x)` and
    `(mA, -, planned y)`: `promote`, the mate-hint match and the child-by-action lookups each pick the one asked
    for.

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
3.7 `PlanCarry.BothBoardsAndHeldBackDrop`: both slots valid at once are both mixed. A planned drop of a piece that is
    held back at the new root (InstantTransfer false, landed mid-decision) is dropped with reason `not a root
    action`, and the board's root priors still sum to 1. **Correction (fix 3 build):** no piece is ever held back
    for the root team at a live root (`TimeState::durative` holds pieces back only for the other team's seats; the
    root team decides anew), so the drop is a root action and is applied; the test checks that, and the priors sum.
3.8 `PlanCarry.NoLeakAcrossGamesOrTeams`: two consecutive self-play games on the same agents, and two tournament
    games where contender and baseline swap teams. No search of game 2 applies a plan recorded in game 1, and no
    team applies the other team's plan.
3.9 `PlanCarry.PromotionSurvivesRootConfiguration`: on a fresh root and on an adopted root, the first child expanded
    at the root is the promoted pair. It fails if the promotion is made before `configure_root_search` or before the
    root's frontier exists.

### 4. No change to flag and terminal semantics

4.1 `ContinuityRules.DurativeAcceptanceHoldsWithContinuity`: v4 acceptance tests 1, 3, 4, 5, 6 (`DurativeAcceptance.*`,
    `DurativeMateRace.*`, `DurativeRepetition.*`, the search halves in `DurativeSearchTest` and
    `DurativePart2SearchTest`) rerun as typed/parameterised tests with `Continuity` on: same verdicts (single loser
    at 100/600 ms, sit-off draw and its 1 s flip, at most 29 landed moves for a 3 s seat with planned moves counted
    on landing, mirror mate race WIN/LOSS, no repetition draw in the durative tree). A single search starts with no
    old tree and no plan, so fixes 1 and 3 would be inert in it and the test would pass on broken code. Each position
    is therefore searched twice. The first search is from a predecessor position one opponent landing earlier. The
    second, the checked one, comes after that landing with clocks 3 ms off. The test asserts that `Team tree reuse:
    N visits recovered` (N > 0) or `Plan carry:` fired where the construction allows it, and checks the verdict on
    the second search.
4.2 `ContinuityRules.OutcomeUnchangedByCommitment`: a committed seat that flags before its landing loses on the clock
    (FlagLoss) exactly as an uncommitted one; a sit-off draw state is reached the same way.
4.3 `tools/regression.py run --engine wave2 --engine-nodes 800` with the continuity flags: drop_mates and sit-off
    family at least the `261081b` numbers (`docs/wave2_local_check.md` section 7: 80/80, 20/20 ahead, 0/20 behind).
    Regression sets are single positions searched cold, so this checks fix 2 only. Fixes 1 and 3 are covered by 4.1's
    sequenced searches.

### 5. Python and arena

5.1 `arena/tests/test_upstream.py`: `test_wave2_continuity_flags` (new keys, `continuity` expands to the three, unknown
    keys refused), `test_wave2_continuity_implies_reuse_tree` (with `team_tree_reuse` or `plan_carry` on: no `ucinewgame`
    between searches, one at the controller's first search; with only `joint_plans` on: `ucinewgame` before every
    search, as in `ref_120`), `test_wave2_parses_reuse_line` (the `Team tree reuse` and `Plan carry` info strings
    reach `SearchResult`).
5.2 wave2-hivemind `tests/`: `Options` fields, `engine_args` emits only non-default flags, `--continuity`,
    `missing_engine_options` empty against the new `--help`.

## Order of implementation

One commit per numbered step at least, in this order; every step ends with the full C++ suite, `stock_parity.sh` and
`wave2_parity.sh` (8/8 each), and nothing is pushed.

1. Fix 1: options and plumbing (no behaviour), tests 0.1, 0.4, 0.5. Then `expansionKey_`, descent, re-timing,
   adoption, self-play; tests 1.x, 0.6, 0.7. **Gate:** the `continuity_probe.py reuse` hit-rate run (fix 1 step 2,
   "Expected hit rate"). Report the hit rate, the refusal reasons and the clock-difference distribution, and settle
   `TeamTreeReuseToleranceMs` from them before step 2.
2. Fix 2: `TimeState` and generator (rules tests 2.1-2.3, 2.5, 2.8, 0.2, 0.3); then search integration and PV
   (2.2 search half, 2.4a, 2.4b, 2.6, 2.7, 2.9); then 4.1, 4.2.
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
* `team_tree_reuse` or `plan_carry` on implies `reuse_tree=True` for the process (no `ucinewgame` before every search),
  and the controller sends one `ucinewgame` at its first search, so the tree and the plans start clean each game.
  `joint_plans` alone does **not** imply it: fix 2 works inside one search, and dropping the per-search
  `ucinewgame` would add a second difference from `c_ref_120`. Rows with the switches off keep `reuse_tree` false, as in
  the PACE round. The engine side (fix 1, step 7) clears every other piece of cross-search state, so `c_reuse` and
  `c_carry` differ from `c_ref_120` only by their fix (the PACE `ref_120` ran on the pre-D4 arena inputs).
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
| `c_ref_120` | upstream-it13 | `'{}'` | the durative-on reference on the new binary, required (review of the diagnosis fixes). With every fix off, tests 0.4 (one search per process, `engine/tests/wave2_parity.sh`) and 0.6 (whole self-play and tournament games and a 10-search UCI session in one process, `engine/tests/wave2_game_parity.sh`) match `261081b`, the binary `ref_120` ran, but that covers only the engine: the durative diagnosis fix D4 changed the arena's `SeatElapsedMs`/`SeatSitting` for every wave 2 arm with no flag, so the PACE `ref_120` (0.227) ran on inputs the new rows no longer get and does not stand as their reference. 0.6 compares self-play on its PGN and clock rows less node counts, with the mate probe off (see build corrections, review fixes) |

Reading: each fix-alone row against `c_ref_120` (not the PACE `ref_120`: D4); `c_all_120` against `c_nodur_120`. With about
1200 games a row's 95% range is about +-0.03, so a difference needs about 0.04. Read `c_same_120` first. The success
mark for this round is `c_all_120` at or above `c_nodur_120`; the PACE gap between durative on and off was 0.21. Each
row also reports reuse hits per search and plan carries per search (from the arena summary), so a fix that never fires
is visible. Cost: 6 rows x about 8 GPU-hours = about 48 GPU-hours (plus 8 for `c_ref_120`, required), behind the build with
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

## Review changes

Adversarial review against the code at `261081b`, 2026-11-20. Each item was a design that would not have worked as
written, or a test that would have passed on broken code.

1. **Fix 1: opponent commitments.** The draft refused only candidates where one of *our* seats had a committed move.
   A node holding an opponent commitment would have forced that opponent move throughout the adopted subtree. Now
   any commitment refuses (reason `committed`; test 1.16).
2. **Fix 1: transpositions.** The re-timing walk treated the tree as a tree. It is an MCGS graph
   (`canonicalize_child` merges on the durative key), and two paths that met under the old root can part under
   `T'`. The walk now keeps a visited map and splits a node whose recomputed hash differs between parents (test
   1.15). It is depth-first with make/unmake, because the path time needs `hand_piece_type`; a breadth-first
   pre-count applies the 100k cap.
3. **Fix 1: stale verdicts.**
   * "Keep verdicts when no clock moved" missed `RootMoveTimeMs` and pace changes. Both set every ready time of our
     team below the root, and the arena changes the budget every decision. Verdicts are now kept only when the
     node's recomputed hash equals its old hash.
   * The draft kept the parent's edge statistics where a child was replaced. A dropped flag proof would then live
     on as an edge with Q near +1 and hundreds of visits. Those edges are now cleared, and their visits come off the
     parent (tests 1.6, 1.14).
4. **Fix 1: expected hit rate.** The draft did not say that two things cut the 33% ceiling:
   * opponent landings are predicted at pace, so the opponent's clock often misses the 250 ms tolerance;
   * the `atRoot` shape rarely matches when both our seats are on turn.
   Step 1 now ends with a measured gate (hit rate, refusal reasons, clock-difference distribution at 250, 1000 and
   3000 ms) before fixes 2-3 or PACE. A FEN-only arena also means move counters are left out of the descent's FEN
   comparison.
5. **Fix 1: state that would survive without `ucinewgame`.**
   * `mateContinuations_` (not re-timed) and the exact-reuse candidates are now cleared at each search under fix 1
     or fix 3.
   * The team-tree store no longer depends on `retainForReuse`, which self-play sets to false.
   * Tournament seat-clock matches reset the agent every decision and swap teams between games. They are now
     covered like self-play.
6. **Fix 2: representation.** Storing the planned move in `moveA`/`moveB` behind a `plannedMask` would have needed
   every make/unmake site rewritten: 11 in `searchthread.cc` and 54 in `agent.cc`, of which the draft listed
   searchthread's and two in agent.cc. A missed site lets a seat act before its ready time. The planned move now has
   its own `planned` field in the candidate's existing padding (size still 48), with `Node::plannedBoard_`. Existing
   sites stay correct, and below-root lookups by action compare `planned` instead (test 2.9). A debug invariant
   `side_to_move == side_on` catches early or missing landings (test 2.2).
7. **Fix 2: sitting partner with a plan.** A committed seat that was sitting either lost its wake-up rule or counted
   as sitting for the cross-team sit-off `DRAW`, which would end lines as draws although the move was certain to
   land. It now keeps `sitting` for timing and transfer and is skipped by both sit-off checks (test 2.8).
8. **Fix 2: headline flags.**
   * Wave 2 plays `InstantTransfer false`. Under it a planned supplied drop exists only for a sitting partner, which
     the v4 tree already reaches 1 ms after the capture. So test 2.4 (InstantTransfer true) did not test the
     configuration the PACE rows play.
   * Added 2.4b under the headline flags, with the transfer running from the planned half to the immediate seat's
     next decision, and the instruction to report it plainly if no such case can be built.
   * The planned-mate tag runs only on checking moves, in `collect_batch`.
9. **Fix 3: promotion order.** `configure_root_search` rebuilds the frontier and resets `promotedCandidate`, so a
   promotion made before it, or before a fresh root is expanded, was silently lost. The order is now fixed (test
   3.9).
10. **Fix 3: smaller defects.**
    * `RuntimeConfig` held one plan board, but both slots can be valid; it now holds one per board.
    * "Legal now" is now "one of this root's actions", because a held-back drop is legal but not a root action, and
      mixing toward it unnormalises the priors (3.7).
    * The tree-committed source of a plan now has a defined precedence.
    * Self-play's game-start reset does not clear plans, so `clear_plans()` is now explicit there (3.8).
11. **Arena: `joint_plans` alone no longer implies `reuse_tree`**, which would have made `c_joint` differ from
    `ref_120` in two ways.
12. **Tests that would pass on broken code.**
    * 4.1 ran single cold searches, where fixes 1 and 3 never fire. It is now sequenced and asserts that they fired.
    * 4.3 is noted as testing fix 2 only.
    * 1.13 checks retained expansions against fresh ones, independently of `expansionKey_`.
    * 1.2's visit count accounts for cleared edges.
    * 1.17 checks that no other state survives between searches.

## Build corrections (fix 1, 2026-11-20)

Found while building fix 1 on `talisman/wave2`; the sections above are corrected in place.

1. **Expansion key.** It must also hold the time inputs of the terminal check (the other team's sit right, a clock
   outcome, MateRaceByLanding's landing order, early landings and last capturer): an expanded node is never
   classified as terminal again, so a node that a re-timing makes terminal has to be replaced.
2. **Cleared edges go back to the `Q_INIT` seed**, not 0: the first backup of an exact-backup edge adds to the seed.
3. **Fresh replacement nodes carry their path hash** and go into the new table; without that, test 1.15's "the
   table holds both" could not hold (a fresh node has hash 0 until visited).
4. **Test 1.14 as written cannot happen** on an adoptable root (a root edge proven won makes the root a proven win).
   The test checks the clearing on a re-timed solved root instead.
5. **Self-play flag needs seat clocks** (`--base-mix`), not only the durative tree: only the seat-clock game loops
   keep the agent between decisions.
6. **Hit rate.** A third structural cut besides the two in "Expected hit rate": the tree holds the other team's
   landings only in their predicted order. First self-play numbers there. The arena gate run is still owed.
7. **Options.** This step adds `TeamTreeReuse` and `TeamTreeReuseToleranceMs` only; the other four options and the
   `Continuity` preset come with fixes 2 and 3 (test 0.1 covers the two). Tests 0.2, 0.3, 0.6 and 1.16 belong to
   fix 2 or need recorded `261081b` self-play outputs and are not built yet (0.6 is built since, review fixes 7); 0.4 (`engine/tests/wave2_parity.sh`)
   and 0.5 are, 8/8 each.

## Build corrections (fix 2, 2026-11-20)

Found while building fix 2 on `talisman/wave2` (`docs/wave2_engine.md` there, "Fix 2"); the sections above are
corrected in place where they were wrong.

1. **Test 2.4a timing.** Below a black-team root every black action costs `RootMoveTimeMs`, so the construction
   needs `RootMoveTimeMs` 50 and Bb elapsed 50; at 400 ms white then acts on both boards (corrected in 2.4a).
2. **Test 2.4a cannot separate the trees.** The old tree's waiting-board mate rule already proves the capture that
   feeds the partner's mating drop (results in 2.4a). **2.4b is not built**: no forced position exists under
   InstantTransfer false, because Ab's reply (or a sit) always comes before Aw's next decision. So fix 2 has no
   tested benefit in a constructed position; only the arena rows can show one.
3. **Representation.** `plannedBoard_` lives in the candidate generator (an `int8_t` in its padding, read through
   `Node::planned_board()`), and the needed/supplied types are packed into bits 1-3 of the per-action capture bytes.
   `sizeof(Node)` stays 1040 (a first version with two extra vectors had 1088, with fix 2 off too).
   `TimeState` grows from 280 to 300 bytes (the spec said 17; the free share travels in `Durative` too, unkeyed).
   The UCI state is `durativeConfig.jointPlans`/`jointPlanFreePermille`, not separate fields in `uci.h`.
4. **A committed sitting seat beside a pass.** It keeps v4's pass timing in full: a pass on the other board, by its
   own team or the opponent, sends its wake-up, and so its landing, on to its cap (v4's leapfrog rule). Only a
   landing wakes it 1 ms later. A pass beside it waits for that landing.
5. **Fix 1's refusal order.** `committed` is checked after the clock and before the expansion key (which holds the
   commitments and so never matches a live root), and it ranks above `shape`. Otherwise the decide-later twin of a
   committed node, which acted on one board, would always report `shape`. `TeamTreeMatch::committed` carries the
   commitments of a node refused only for them, for fix 3's second plan source.
6. **Mate tag.** A planned move counts as mating if the victims are mated even when they may wait for a blocker
   (`is_checkmate(victims, true)`), so the raise only goes to mates no partner can block.
7. **Planned pairs need visits.** At the default share (500) a planned pair's joint prior is at most half its
   immediate move's. A 400-node UCI search had planning nodes below the root but no planned pair expanded (test
   2.6), so the probe's scaling run should compare shares before a PACE row is read.
8. **Tests.** 0.3's keys are one folded key per line, recorded from `261081b`'s `time_state.h` with
   `engine/tests/durative_key_lines.h` (the same file builds against both). The search half of 2.2 runs from a below-root
   state with the side invariant counted (`SearchThread::checkDurativeSides`), since the suite is a Release build
   where `assert` is off. 4.1 needs the `Continuity` preset and comes with fix 3; its fix-2 parts are the rules tests
   (`ContinuityRules.*`, v4 test 4 with plans). 4.3 runs with the arena plumbing (step 4).

## Build corrections (fix 3, 2026-11-20)

Found while building fix 3 on `talisman/wave2` (`docs/wave2_engine.md` there, "Fix 3"); corrected in place where the
sections above were wrong.

1. **Held-back drops cannot reach a root plan (3.7).** At a live root the root team decides anew: `TimeState::durative`
   fills `pending` only for the other team's seats. So no carried drop is ever dropped as `not a root action` for a
   held-back piece; the check stays as a guard. Side effect worth knowing (v4, not fix 3): under InstantTransfer false
   the root may play a drop of a piece that landed during that seat's current decision.
2. **Where the fresh-root mix happens.** Not in `process_batch` but in `Node::try_init_and_expand`, right after the
   root noise that already lives there; the promotion is made there too, after the frontier and the Gumbel pool are
   built and before the first child is taken (that function expands the first child itself, so a promotion made in
   `process_batch` after it would be one child late).
3. **No promotion when an expanded pair already holds the carried move** (an adopted root may have one): it has its
   visits. With both slots valid the pair holding both is promoted when legal, else the better single pair.
4. **Self-play flag needs seat clocks** (`--base-mix`), as fix 1's does (Options table corrected).
5. **Recording in self-play and tournament** goes through `Agent::record_plans` after the action is chosen (self-play
   samples one), for the team's on-turn seat that is neither deciding nor has a move in flight
   (`SeatClockGame::in_flight`). In those games that seat is always a sitting one, so plans fire in about 3-7% of
   searches (it13, 200 nodes, 30 s).
6. **UCI permanent brain** is not started under `PlanCarry` with a durative clock either: the next search drops the
   retained root anyway (Fix 1 step 7).
7. **Telemetry and reasons.** `info string Plan carry: <A|B> <move> weight <w>` or `... <A|B> <move> dropped
   (<reason>)` per slot that held a move; reasons `other team`, `not on turn`, `board changed`, `pocket`, `illegal`,
   `not a root action`, `weight 0` (PlanCarryPermille 0 drops the slot).
8. **Tests as built.** 3.1 with it13: A at the start, B after 1.e4; search 1 (200 nodes) chooses `(d2d4, e7e5)`, after
   d2d4 d7d5 the 64-node search with plan carry chooses `e7e5`, without it `b8c6`. 3.8 is checked at the agent level
   (other-team plans dropped, `reset_search_state` keeps plans, `clear_plans` empties them) plus self-play and
   tournament smoke runs; the game loops' clears are not observed directly. 3.9's adopted-root half is a node-level
   test (promotion before `configure_root_search` is lost, after it survives). 4.1 is built for v4 tests 1 and 6,
   each sequenced so that fix 1 (test 1) or fix 3 (test 6) fires; v4 tests 3 and 4 are fix 2's rules tests, and the
   mate race (test 5) is not rerun sequenced.
9. **Python.** `rl-loop --continuity` is a plain switch that sets the three `Options` fields; an explicit switch
   (`--joint-plans false`) overrides it.

## Build corrections (review fixes, 2026-11-20)

From the confirmed review of the continuity work (`talisman/wave2` `fe11956`, `8aff7b6`, `06143ba`, `c627626`; main
repo `a06db42`). The sections above are corrected in place where they were wrong.

1. **Fix 1 on the old root.** When no move landed since the last search (the partner decides while our move is in
   flight) the walk's match is the old root itself. Its priors already hold that search's root noise and plan mix, and
   both helpers mix into the current priors, so re-noising it compounded the noise (eps 0.25 became about 0.44) and
   the carried-move boost. It is still adopted (`TeamTreeReuseReport::oldRoot`), because in self-play it is a large
   share of all adoptions, but without `refresh_root_noise` or `apply_root_plan`; its carried halves report
   `old root` and are not mixed (its own visits carry the pair it chose). "Self-play keeps the adopted root and
   re-noises it" holds for adopted descendants only.
2. **Refusal ties.** Equal-rank refusals go to the node with more visits, so a `committed` refusal carries the most
   visited line's plan (siblings that differ only in the planned half reach the same live boards).
3. **Proven returns.** `return_proven_mate` and `return_proven_loss` record the returned pair and keep the tree, like
   the normal end of a search (UCI relies on that record alone).
4. **Missed ponders** record no plan, as their tree is dropped.
5. **Knobs need their switch** in self-play, tournament and `hivemind rl-loop`: a knob off its default without its
   switch is refused, and does not count as a search change (it used to move a stock run onto the clock time model).
6. **Arena:** the engine's raw `Continuity` option, passed through `engine_options`, now keeps the tree between
   searches and is checked in the startup advertisement.
7. **Test 0.6 is built** (`engine/tests/wave2_game_parity.sh`): 2 self-play games, 2 tournament games and a 10-search
   UCI session, base `261081b` vs ours, one worker, 5/5 identical. Self-play is not bit-reproducible run to run even on
   `261081b` itself: a search's node count can overshoot its target by a visit (so the `nodes` column and the
   training records' visit targets move), and the Fairy-Stockfish mate probe's proofs reach the tree at wall-clock
   times, which can change a move. 0.6 therefore runs self-play with the probe off and compares games.pgn and
   seat_clocks.tsv less `nodes`. The same timing noise makes `wave2_parity.sh` position 8 (a mate-in-2 loss, 100
   nodes) occasionally differ by two nodes, base against itself included: rerun before reading a 7/8.
8. **Tournament telemetry.** Seat-clock tournaments with TeamTreeReuse or PlanCarry print a per-game continuity line
   (adoptions, refusals, plans recorded/applied/other team, state left after the game-start clear);
   `TournamentWithContinuity` checks the clears and that fixes 1 and 3 fire (build correction 8 of fix 1 no longer
   applies).
9. **Fix 2 visits.** `JointPlansSearch.PlannedPairsGetVisits` checks that planned pairs hold over a tenth of the root
   visits and that forced landings are searched, so a prior bug that starves them fails a unit test. A behavioural
   benefit is still only measurable in the arena (`c_joint_120`); 2.4b stays unbuilt.

## Diagnosis fixes (2026-11-21)

From `docs/durative_diagnosis.md` (at `261081b`), built on `talisman/wave2` (`426339a`..`0e53f24`; details, tests and
checks in `docs/wave2_engine.md` there, "Diagnosis fixes"). Each is a UCI option and a `selfplay`/`tournament`/`rl-loop`
flag, off by default. With them off, `stock_parity.sh` 8/8, `wave2_parity.sh` 8/8 and `wave2_game_parity.sh` 5/5 are
identical.

| Finding | UCI option / flag | In `Continuity` | What it does |
|---|---|---|---|
| D1 | `OwnPaceBelowRoot` / `--own-pace-below-root` | yes | Below the root our seats act at their own `SeatPaceMs` (the budget where none is given); the root action and a pass made at the root keep `RootMoveTimeMs`. Composes with fix 1 (re-timing recomputes it, keyed) and fix 2 (planned halves land at the pace-based ready time) |
| D3 | `WaitNodeActingEval` / `--wait-node-acting-eval` | no | Forced-wait leaves are evaluated from the team that acts next, value (and WDL) negated. No gain was shown, so it is not in the preset |
| D5 | `BlockableCheckNotMate` / `--blockable-check-not-mate` | yes | A check a drop could block is never mate (the arena's `is_bughouse_mate`), on the board, so in every search, prover line and game-end check |

The `Continuity` preset (UCI, `--continuity` in self-play and tournament, Python `CONTINUITY_FIELDS`) now sets five
switches: the three fixes, D1 and D5. The arena expands `continuity: true` to the same five (`a9419ed`,
`arena/players/upstream.py` `WAVE2_PRESETS`); the single-switch PACE rows are `c_ownpace_120`, `c_waiteval_120`,
`c_mateblock_120` (`870b476`).

**D5 was confirmed before it was fixed.** Position: A `k7/pp6/8/8/8/8/8/4K2R[] w`, B `4k3/8/8/8/8/8/8/4K3[] w`, white
to play. After Rh1-h8 the arena plays on (Ab waits). The engine answered `score mate 1` in both trees.

**Residual, not fixed.** The engine still scores that position as lost wherever it denies the checked team the right
to sit, because a team that may not sit and has no move on a board it is on turn on has no legal action:
* the v3 tree at a lead of 0 or less;
* the durative tree's root provers whatever the lead. The sat-partner single-board proofs never let the defenders sit,
  and the reduced-partner proofs run on the v3 clock's lead gate.

So in the headline setting with the provers on, the root still says `mate 1` there. The tree alone does not. Fixing
it means a free sit for the provers under `SitGate false`, which changes every prover verdict in that setting.

D2 (no new work): on the 25 positions where we are on turn on one board, the continuity round's scaling rows give
continuity 12/14/17 of 25 against off 15/17/17 at 1k/4k/16k (it13 at 64k as reference, it13 fp32): fix 2 closes the gap
at 16k only. D4 and D7 (arena) and D6 (a Python test mock, fixed) change no engine default.
