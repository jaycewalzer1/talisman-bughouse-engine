# 06. Engines and AI for bughouse and crazyhouse

Written 2026-11-17 for the custom-engine project (one agent, both boards, real clocks as inputs, no hand-written
stalling rules; base weights it04 = `hivemind-it04-crossboard-risev33-loss1.556-p82.0`). Everything below was read from
the primary source named in brackets; a claim resting on a search snippet or a single weak source is marked as such.
"Stock" means the it04 engine in `third_party/hivemind` (the flag model we are modifying); "upstream" means the current
`aminwoo/hivemind` main branch and releases, which have moved on since our fork.

## 1. Summary

- **Upstream Hivemind already shipped a clock-aware engine (v3.0.0, 2026-10-16).** It reads two extra planes per board
  (time *lead* in units of the leader's move time, and a *speed* ratio), searches on a simulated clock in which each
  action costs the acting team its move time, and trained a new lineage (`twin-s-clock`) by self-play on sampled clocks.
  Reported gain: +156 Elo over its own iteration 0 at 100 ms/move; later iterations are flat (it13 vs it10: -1 Elo).
  [5][8][14][20] Our project is no longer "give Hivemind a clock"; it has to be "do it better than upstream did".
- **Upstream's clock model is one signed number, the lead; it has no absolute clocks, no flag loss, no notion of how long
  a side has been thinking, and sitting is still hard-gated by `lead > 0`.** [8][9][11] Those four gaps are exactly where
  our spec (four absolute clocks, flags as terminal losses, FreePass, elapsed-thinking planes) differs. [L1][L2]
- **The HiveMind paper (CoG 2023) is closed access.** Its abstract: supervised on human games, then AlphaZero-style tree
  search over both boards "using clock times to choose the move order", and >95% wins against Fairy-Stockfish at 2 s
  search, scaling better with time. No Elo, no human comparison. [2][3] Fairy-Stockfish's bughouse is a single-board
  engine with a "virtual drop" hack its own maintainer calls crude, so 95% against it says little about human-relative
  strength. [36][39]
- **No bughouse engine has a credible rating against strong humans.** Sjeng: ~2000 FICS bughouse, top 2200+ (self-report,
  undated) [29]; Sunsetter: crazyhouse 2722 FICS record, no bughouse figure [42]; Hivemind: only self-relative Elo at
  100 ms/move on an RTX 4070 [19][20]; a chess.com Fairy-Stockfish bridge claims "~3800 performance" with no evidence
  [55]. Human writers hold that engines are not stronger than top humans at bughouse (snippet only). [56][57]
- **Every classical engine ties sitting to two things: a mate/dead state and the diagonal clock comparison.** Sjeng sits when
  its score is in a losing band and the partner is alive [27]; Fairy-Stockfish sits when dead and waits for virtual
  material only while `own remaining time > partner's opponent's time` [35]; Sunsetter sits on "I am in trouble" and tells
  a partner who has sat for 5 moves to "go (watch our time)". [46] These are the behaviours the network must be able
  to express, which requires seeing the *diagonal* opponent's clock, not just the one on its own board.
- **Bughouse engines move far faster than chess engines.** Sunsetter allocates `initialTime/360 + mytime/360` per move in
  bughouse against `mytime/25` in chess, dropping to 40 ms below 4 s and 5 ms below 0.8 s, and does not ponder in
  bughouse at all. [44][46] CrazyAra's chess-style allocation (fixed time + 70% of increment, 5% of remaining after move
  40, stop at half time on "easy" positions, extend by half on a 0.1 Q drop) is the best-documented MCTS time manager
  and the natural baseline for our think-time head. [22]
- **Smaller nets won at fixed wall-clock.** Upstream distilled the 13.6M it04 into a 0.97M `twin-s` net that searched ~3x
  more nodes per second and beat the teacher at 100 ms/move; removing cross-board attention alone gave +32% speed.
  [19] Any gate at fixed time, not fixed nodes, must account for this.
- **Human data carries real clocks.** HiveMind's supervised data was scraped from chess.com (TCN-encoded games, rating
  filter ≥2250 in the documented example); CrazyAra used 569,537 lichess games with both players ≥2000. [16][22][L4]
  Corrected: per-move timestamps *were* kept (`scrape_cc.py` `moveTimestamps`) and are already used by
  `generate_planes.py` to set the flag plane and to derive `pass` labels from human sits; they are the only source of
  *human* sitting behaviour under real clocks (implication 12).

## 2. The knowledge

### 2.1 HiveMind (Woo, Sweetser, Aitchison; IEEE CoG 2023)

- Bibliographic: "HiveMind: Learning to Play the Cooperative Chess Variant Bughouse with DNNs and MCTS", CoG 2023,
  Boston, pp. 1-8, DOI 10.1109/CoG57401.2023.10333210. Affiliations: Australian National University (Aitchison also
  DeepMind). Cited once as of 2026-11. Closed access; no open PDF; ResearchGate has no full text. [1][2][3][4]
- Abstract (paraphrased from two index records): first supervised on human games; then an AlphaZero-style approach with
  domain knowledge, "using clock times to choose the move order", running a tree search across both boards that handles
  cross-board coordination "without hand-crafted heuristics" and "accounts for time"; beat Fairy-Stockfish, "the leading
  alpha-beta bughouse engine", with a win rate above 95% at 2 s search time, and scaled better with more time. [2][3]
- Everything else about the paper (planes, network, data size, exact rules of the search) is **unknown** to us. The
  "clock times choose the move order" phrase suggests the tree ordered which board moves next by the clocks, which is a
  different mechanism from the released engine's lead-gated passing (2.2). Treat as unverified.
- The authors' code is the `aminwoo/hivemind` repo (v1.2.1 README, GPL-3.0 at the time, no paper citation). [17] The
  licence later changed to MIT (v2.2.2 README and current LICENSE). [16][5][L3]

### 2.2 aminwoo/hivemind: stock (it04, flag model) and upstream (twin-s-clock, clock model)

**Weights and licence.** `aminwoo/bughouse-rise-v3`: "CrossBoardRiseV3", 15 bottleneck residual blocks at 384 channels
(operating width 256), kernel 5 at blocks 7, 11, 12, 13, ECA at 5, 8, 12, 13, 14; 2 cross-board attention blocks (dim
192, 6 heads) after the body, including pocket state; value head 16 ch -> 512 FC; two shared-weight policy heads
(192 ch); 13.57M params (ONNX, BN folded), 14.18M in the checkpoint. Inputs `[B,74,8,8]`; outputs `value`, `pi_a`,
`pi_b` (4672 each, AlphaZero chess encoding), `wdl_out`, `moves_left`. Checkpoint metadata: training_iteration 13315,
batch_steps 400. Val loss 1.556, policy accuracy 82.0%. **Licence: MIT, "matching the Hivemind repository."** No clocks
mentioned; no strength claim. [18] The repo pins revision `eac7b7a4...` and SHA-256s in `src/hivemind/network.py`. [L4]

**Stock input planes (37 per board, 74 total).** Per board block: 0-5 own pieces P,N,B,R,Q,K; 6-11 opponent pieces;
12-16 own pocket counts (constant plane `count/16`); 17-21 opponent pocket; 22-23 promoted pieces; 24 en passant;
25 "side to move equals expected turn" (board A: team, board B: opponent); 26 constant 1; 27-30 castling; **31 time
advantage flag** (`!single_board && hasTimeAdvantage`); 32-33 last move from/to; 34 halfmove clock `min(rule50,50)/50`;
35-36 repetition >=2, >=3. Board B's "first colour" is the opponent. [7][L5] So stock sees time as one bit, identical
on both boards, with no pace and no absolute values.

**Stock time rules (hand-written).** From `joint_action.h`: double pass is legal only with the time advantage and when
exactly one board is on turn; a single pass while the other board moves is legal if `teamHasTimeAdvantage ||
!(bothOnTurn) || partnerMoveIsCapture`; otherwise both boards must move. [L6][9] `TimeAdvantage` is a UCI option the
driver sets from the real clocks; `Mode sit|go` is a deprecated alias; `RequireMoveOn A|B` forces a move on a seat. [L7]
Mate legality depends on the flag: a blockable mate is terminal only when the defending partner cannot capture a piece
that enables a block, "a time-ahead team may still sit", and an unblockable mate-in-one always wins the race. [L8]
Self-play toggles the flag between teams per macro-ply (`hasTimeAdvantage = !hasTimeAdvantage`). [L9] The root mate
probe is gated on the time advantage. [L10]

**Upstream clock model (v3.0.0+).** `TimeState` holds **one signed number**, the white team's lead in ms over the
diagonal opponents ("absolute clock values never appear, because without increment the two boards' clock sums stay
equal"), plus each team's usual move time (`DEFAULT_MOVE_TIME_MS = 100`, "the fastest possible move") and an optional
one-shot override for the next move (`RootMoveTimeMs`, the budget of the move being searched, release v3.0.1 "charges
the searched move its clock budget"). `advance(team, onTurnOnBothBoards)`: the lead changes by the acting team's move
time **only when that team is on turn on both boards**; otherwise one clock of each team runs and the lead holds.
`can_sit(team) = lead_ms(team) > 0`; `can_sit_through(team, n)` requires the lead to cover the next move plus `n-1`
usual move times. **There is no flagging logic: a negative lead is not a loss.** [8][15]

- Planes 37 and 38 (both boards carry the same values): lead = `sign(lead) * min(1, log2(1 + |lead| / moveTime(leader)) / 10)`
  ("how many actions the leader can afford to wait"); speed = `clamp(log2(moveTime(team) / moveTime(opponent)) / 5, -1, 1)`.
  Under `TimeModel flag` the lead plane is +-1 and speed 0. Networks reading 74 planes still load. [6][7][20]
- UCI: `TimeModel clock|flag` (default clock), `TimeLeadMs`, `TeamMoveTimeMs`, `OpponentMoveTimeMs`, `RootMoveTimeMs`,
  `MaxMoveTimeMs`, `TimeAdvantage` (used only when no clock is active: `team_may_sit()` returns `clock ? can_sit :
  flag`). For two-board games `wtime/btime/winc/binc` are parsed but **ignored**; budget = `nodes`, else `movetime`,
  else 1000 ms. Single-board games use `remaining/30 + 3/4 increment`. [12]
- In the tree: `MOVE_NONE` (pass) is appended to every board's action list, "a team can pass even while in check";
  each action on the way down is charged to the acting team; the legality rules above are enforced with
  `teamHasTimeAdvantage := time.can_sit(team)` at each node, so sitting is still a hard rule, just re-evaluated per node.
  A mate on the waiting board is scored as a loss 3 plies out. A "frozen race" prover handles the case where the
  attackers sit on one board to force the defenders to move on the frozen board. Mate probes are skipped when the team
  `can_sit_through` the whole proof horizon. Bestmove format `bestmove (moveA,moveB)`, `pass` for a sit. [11][13]
- Self-play clock sampling: half the games start level, otherwise |lead| log-uniform in 100..10,000 ms with a fair
  sign; each team's move time log-uniform in 100..3,000 ms, **one value per team per game, no per-move jitter**. [10]
- Training loop (docs): 800 nodes +-5%/move, no mate search, no resignation, 100k games/iteration in 10k segments,
  1k validation games, replay of the previous 3 iterations (6 from it8), lr 1e-4, 1.5 passes new + 0.5 replay,
  100 openings from `engine/openings.tsv`, promotion unless the upper 95% CI vs the generator is below 0. [6][20]
- Results (all self-relative, 100 ms/move, 1,000 games unless noted): v3.0.0 twin-s-clock it3 +156 Elo over it0 (the
  net that reads the flag only); head-to-head vs it0, iterations 2-7 land +155..+200 ("conservative"); summing promoted
  wins it10 is ~+290 ("less conservative"); it9 and it11 not promoted (it11 -8 vs it10); it13 vs it10 -1 Elo (CI -33..+31,
  461 games); raw policy it13 vs it10 52.0% (+14 Elo). Caution from the card itself: it0 **"searches on the binary time
  advantage rather than the clock"**, so part of the it0 gap is the search change, not the net; and do not run it0 or
  earlier nets with `TimeModel clock`, "their evaluation jumps by about 26 pawns as the lead crosses zero." [14][20]
- Distillation lineage: `twin-s` (TwinBoardNet, 8 blocks x 128 ch per board, shared trunk, 2 pooled scale-and-shift
  exchanges, value head on A, B and |A-B|, swap-symmetric), 0.97M params, distilled from 30M it04-labelled positions
  with targets averaged over both board orders; "about 3x" the nodes/s of the teacher and stronger at 100 ms/move on an
  RTX 4070 (~72-77k nps both sides); no-attention variant +32% speed and beat attention variants at equal time.
  Self-play gains: it3 +68 Elo (CI +31..+107) over it0 in 160-game matches. [19]
- Supervised data (v2.2.2 README): human games, rating filter (example `sl_2250` = min 2250), "filtering for all four
  players", 98/2 whole-game split, board-swap augmentation. The scraper targets chess.com (`scrape_cc.py`, `tcn.py` is a
  port of chess.com's TCN move encoding). [16][L4] RL: HVM5 self-play format, openings from raw policy with sampled
  length (mean 8 macro-plies, max 30). [16]
- Other engine features worth knowing exist upstream: MCGS with transpositions, root Gumbel options, tree reuse (~13% of
  visits carried), ponder ("permanent brain", charging our own action to us when on turn on both boards), early stopping
  and time extension on eval drop / best-move change (chess-engine style, info strings), PUCT widening, mate oracle,
  `-O3` not `-Ofast` since v3.0.0 (we hit the same `-ffinite-math-only` selection bug: NOTES.md). [13][14][L11]

### 2.3 CrazyAra (Czech et al.) and its papers

- Paper: "Learning to Play the Chess Variant Crazyhouse Above World Champion Level with Deep Neural Networks and Human
  Data", Frontiers in AI 2020 (arXiv 1908.06660, 35 pp.). Supervised only. 569,537 lichess games, Jan 2016-Jun 2018,
  both players Elo >= 2,000, mean game Elo 2,279.61, very short time controls dominate, draws < 1%; 20 players account
  for 46% of games. Move accuracy 60.4% after 1.5 days. Beat 2017 world champion Justin Tan (JannLee) 4-1 (5+15 informal);
  beat 12 of 13 participants of the 2017 Crazyhouse Computer Championship on CPU (15+5); CrazyAraFish (further trained on
  121,571 Stockfish self-play games) scored 3 wins 1 draw of 10 vs Stockfish 10 at 30+30. [21][22]
- Inputs (Table 1): piece planes, **pocket counts as constant int planes** (max 32), promoted-pawn planes, colour, total
  move count (max 500), no-progress counter, all scaled to [0,1]. No clock input. [22]
- MCTS details: `cpuct-init 2.5`, `cpuct-base 19,652`, Dirichlet noise factor 25%; **Q of unvisited nodes initialised
  to -1** (treating them as draws caused over-exploration; a "pruning" patch for draws clipped a mate defence in a long
  game); "enhance checks": checking moves with prior < 0.1 get `+0.5 * max P` then renormalise; centipawn conversion
  `cp = -v/|v| * log(1-|v|)/log(1.2)`; U-value divisor parameterised. [22]
- **Time management (Sec. 9.2.4):** fixed move time per expected game length, remaining time and increment; add 70% of
  the increment per move; sudden death assumes 50 moves and switches to proportional at move 40, allocating 5% of
  remaining time ("the game will last 20 moves longer"); stop immediately on a single legal move; stop at **half** the
  allocation if the top prior move has probability > 90% and keeps the highest Q; **extend by half** of the extra
  preassigned time if the candidate's Q is 0.1 below the Q of the previous move played; vs humans multiply by a random
  factor in [-0.1, +0.1]. [22] TalkChess folklore from the same period: doubling search time is worth roughly 70 Elo,
  and spending time proportional to the opponent's think time was suggested by hgm as a tactics signal. [53][54]
- Follow-ups: "Improving AlphaZero Using Monte-Carlo Graph Search" (ICAPS 2021; DAG instead of tree, epsilon-greedy
  exploration, terminal solver, domain constraints; Elo figures not on the abstract page) and theses on RL for
  crazyhouse, MCTS-minimax hybrids, variance/uncertainty in MCTS. GPL-3.0. Lichess accounts CrazyAra and CrazyAraFish;
  a Dec 2018 rated game shows CrazyAra at 2613 beating JannLee at 2639 (snippet). [23][24][59]

### 2.4 Sjeng (Gian-Carlo Pascutto) and `partner.c`

- `partner.c` is a ptell command parser: `sit` -> "Ok, I sit next move. Tell me when to go." (`must_sit = TRUE,
  must_go = 0`); `go`/`move` -> `must_sit = FALSE, must_go = 4`; `fast`/`time` -> `go_fast = TRUE`; `slow`; `abort`;
  `flag`; `dead` and "i'll have to sit...(dead)" -> `go_fast = TRUE, partnerdead = TRUE`; `x`/`=`/"never mind" resets
  piece values. Piece-value requests: `+`/`-` with piece letters n b r q p d(iagonal: p,b,q) h(eavy: r,q); `+` adds
  150, `++` 1000, `+++` 50000 ("mates"), `-` the negatives; "trade" or no letter means trading is good/bad (pawn +-25,
  minor/rook +-50, queen +-100). `CheckBadFlow` virtually adds each missing piece type to the opponent's hand, runs move
  generation, and if a drop mates sends `tellics ptell ---p` (etc.) to the partner; the recovery messages are
  unreachable as written. No clocks in the file. [25]
- `sjeng.c`: the engine moves only when `!must_sit`; `must_go` decrements after each own move (so "go" buys 4 moves);
  it may ponder while sitting; `CheckBadFlow` runs after each own move in bughouse. [26]
- `search.c`: `must_sit` is set by the search when **the score lies in (-60000, -40000) in bughouse and the partner is
  alive**, printing "I'll have to sit...(lose piece that mates you)"; when forcedly mated with a live partner it prints
  "I am forcedly mated (dead). Tell me 'go' to start moving into it." (resigns if partner is dead); "Mate in %d, give him
  no more pieces." replaces the mate message; time extension on a fail-low is disabled in bughouse; `go_fast` caps the
  move time at 40 (centiseconds); the one-legal-move shortcut is skipped in bughouse; null move is reduced less
  (`depth-2-1`), aspiration window 100 vs 35, futility margin 200 vs 150, captured material counts double. [27]
- `eval.c`: zh board values P 100, N 210, B 230, R 250, Q 450, K 4000 (chess: N 310, R 500, Q 900); hand material added
  directly; king safety table `ksafety_scaled[15][9]` indexed by opponent hand material /100 and attacked squares around
  the king (row "nothing": -5..150; "queen": -10..500; "minor + pawn": -20..900). [28]
- Strength: "most Sjeng's on FICS have a rating of about 2000 points in bughouse, with a top of over 2200",
  "probably a bit weaker in bughouse than in crazyhouse" (crazyhouse ~2300, once #1 on ICC). Self-report, undated. [29]
  Release notes list "Sit if losing a piece" and "Automatically tell other Sjeng to go if something no longer mates us"
  (snippet of sjeng.org/news.html, not read in full). [30]

### 2.5 Fairy-Stockfish (ianfab): bughouse as a single board plus a partner protocol

- Variant definition: `bughouse = crazyhouse + twoBoards = true, capturesToHand = false, stalemateValue = -VALUE_MATE`.
  Nothing else differs; the engine evaluates **one board** and receives holdings from outside. [31]
- Holdings arrive via the XBoard `holding` command (inserted into the FEN; a running search is restarted). Full bughouse
  (sit/go, time management) exists **only in the CECP/XBoard protocol**; UCI can set up positions but the maintainer
  calls UCI "somewhat tricky" for real-time games because it is stateless. [32][39]
- **Virtual drops**: `allow_virtual_drop(c, pt)` = `pt != KING` and hand counts of P, N, B may go to -1 **only when the
  dropped piece is a pawn**, R and Q counts must stay >= 0. Virtual drops are generated only when they give check, never
  at the root/perft, and are stripped from the legal list; the search may still choose one as `bestMove`, and then the
  main thread tells the partner `fast` and **busy-waits** `while (!abort && !partnerDead && !fast && time[us] - elapsed
  > Partner.opptime)` before playing a real move. The maintainer describes the reserve as "a very crude hack" (one
  major, or one minor plus a pawn, or two pawns) that can suggest impossible drops, with the clock check applied only at
  the current move; he thinks alpha-beta fits this poorly and MCTS would handle it more naturally. [35][36][37][39]
- Partner protocol (`partner.cpp`): handshake "partner Fairy-Stockfish is an engine"; `sit` (ignored if partner is
  Fairy and we are winning, to avoid deadlock), `go` (stops search), `move [m]`, `fast`, `slow`, `dead`, `x`,
  `time n`/`otim n` (centiseconds -> ms). Search-side messages at `completedDepth >= 8`: sends its own clock
  (`time`) and its opponent's (`otim`) to a Fairy partner "relevant for sitting decisions"; declares `dead` and sits when
  `bestValue <= VALUE_MATED_IN_MAX_PLY`; tells the partner to `sit` when we win by force and `time[~us] < Partner.time`;
  signals `fast` when a *virtual* mate is available and `time[us] - elapsed > Partner.opptime`; on a virtual mate against
  us (`time[~us] > Partner.time`) tells the partner `sit` and plays fast itself. While a sit is pending or we are dead
  with a live partner, the time-out check is suppressed down to 1 s of own clock. [33][34][35]
- Rule variants requested by the bughouse.pro server (issue #811, open): forbid drop mates while allowing drop checks,
  per-colour pawn drop regions, "piece stealing" on promotion; the maintainer pointed to pychess's bughouse work. [40]
- FICS: runs under XBoard zippy (`-zippyVariants "bughouse" -zippyBughouse 2`, `set bugopen 1`), auto-accepts
  partnerships. No ratings published. [41]

### 2.6 Sunsetter (Georg von Zimmermann, Ben Dean-Kawamura) and other classical engines

- Sunsetter (GPLv2, C++) held the FICS crazyhouse records: Sunsetter 2569 (2001-10-27, PIII/500), Indalecio 2722
  (2002-09-08), Goober(C) earlier; "#1 on ICC for a short time". No bughouse rating. [42] CrazyAra's history section
  names Sunsetter 2 (2001) as a pioneer. [22]
- Piece values: P 100, N 192, B 195, R 200, Q 390 ("all evaluations use 100 points to equal a pawn"). In-hand bonus per
  piece: pawn `15 + 7*count`, minor/rook `20 + 10*count`, queen `40 + 20*count`; king safety is **multiplied** by a
  2..11 factor from the opponent's hand material (thresholds 80, 160, 260, 380, 530, 720, 960, 1260, 1630). Bughouse-only
  terms: `bughouseSitForEval` and `bughouseMateEval` add R/N/P values for pieces a side lacks in hand, switched off
  when the partner is sitting or has said go. [44][45]
- Search constants: check extension 3/4 ply, capture and forcing extensions 1/2 ply, null reduction 2. [44]
- Sitting: `sit()` sets `sitting = 1` and sends "sitting"; the search ignores stop requests while sitting unless
  pondering and loops `while (sitting && !reSearch) pollForInput()`. Messages: "sitting (...) Tell me 'sitting' if no
  stuff comes for me" on an illegal/winning-but-undecided situation; "sitting (I am mated)" then "go" when
  `bestValue <= -MATE_IN_ONE + 90`; "sit (I am in trouble)" when `bestValue <= -MATE` and the partner is not sitting,
  with a re-search; "go (I am ok)" on recovery; **"go (watch our time)" when the partner has sat for more than 5
  moves**; the main loop also tells the partner to sit when the opponent is waiting for something. Partner commands:
  `sitting`/`frozen`, `sit`, `go`, `flag`, `cancel`, `abort`. [43][44][46]
- Time: bughouse uses the lower of the two clocks; base `initialTime/360 + mytime/360`, `mytime/180` at <= 20 s,
  40 ms at <= 4 s, 5 ms at <= 0.8 s (chess: `mytime/25`, `/40`, 100 ms, 20 ms). "not pondering in Bug at all". The
  main loop polls input every 10 ms. [44][46]
- The 2016 CCC World Computer Crazyhouse Championship field: Imortal 2.5, Sunsetter 7g, TJChess 1.1, Sjeng 11.2, Pulsar
  2009, KKFChess 2.6.6, Nebiyu 1.45, Sjaak II 1.3.1a, at 90+30, 2 threads; no multithreaded crazyhouse engine existed. [52]
  Chessprogramming's Crazyhouse page also names CrazyWa (H.G. Muller) and Multi-Variant Stockfish (Dugovic), which it
  says improved by 800 Elo in 8 months. [49]

### 2.7 Servers, bots and the human comparison

- chess.com has no official bughouse bots (its bots are Komodo personalities for chess). The only chess.com bughouse
  engine found is `RobbieGM/chessdotcom-bughouse-engine`: a Tampermonkey script injecting into the defunct
  chess.com/live client plus a local Fairy-Stockfish server, "as in bughouse time is critical", claiming "a performance
  rating of around 3800" with no games or method. Single weak source. [55][51]
- Lichess has no bughouse. Crazyhouse bots: CrazyAra ~2600 in 2018 [59]; ToromBot shows 2753 crazyhouse (BOT status not
  confirmed from the snippet) [58]. Lichess pages were unreachable from this session.
- FICS/bughouse-db is the historical home of engine bughouse (Sjeng, Sunsetter, Fairy-Stockfish via zippy). [29][41][42]
- Human-side opinion: "bughouse engines are not stronger than the strongest humans"; "computers are not significantly
  better than humans at bughouse" (chess.com blog and club forum; snippets only, the pages blocked fetching). [56][57]
  A human time-management guide: "if you double sit for about 10 seconds, you can basically just resign" (snippet). [60]
- TalkChess 2022 (jorose): three engine setups (one engine both boards; two instances; engine + other agent), the
  observation that a player "may choose to not make a move in order to prevent the opponent from giving their partner
  material", and a proposal for a wait command that adjusts clocks when all four wait. No replies. [50] TalkChess 2025:
  "Fairy-Stockfish can already play bughouse" is the whole state of the art as far as that forum knows. [51]

### 2.8 Disagreements and open questions between sources

- *Is the lead enough?* Upstream argues one signed lead suffices without increment. [8] Every classical engine instead
  compares absolute clocks across the diagonal (`time[~us] > Partner.time`, `time[us] - elapsed > Partner.opptime`)
  and Sunsetter scales think time by the lower absolute clock. [35][46] Absolute values matter for flagging and for
  how much thinking a move can afford; the lead only says who can wait.
- *Should sitting be a rule or a learned action?* Upstream, Sjeng, Fairy-Stockfish and Sunsetter all gate it by rule.
  The HiveMind abstract claims the search works "without hand-crafted heuristics" for coordination, but the released
  code has a hand rule for passing. [2][9] No source has tested a free pass.
- *Fixed nodes or fixed time for gating?* Upstream reports all Elo at 100 ms/move and shows a 3x faster small net
  beating its teacher; our `beat_stock.md` compares fixed node budgets. [19][L12]
- *How strong is Fairy-Stockfish as a baseline?* The HiveMind abstract calls it the leading alpha-beta bughouse engine;
  its maintainer calls its bughouse layer a crude hack with a hard-coded reserve. [2][39]

## 3. Implications for the engine

Format: idea -> does stock it04 capture it (how we know) -> does upstream capture it -> concrete change for us.

1. **Clock as a continuous input.** Stock: no, one flag (plane 31) [7][L5]. Upstream: lead + speed planes, both boards
   identical, lead in units of the leader's move time [6][7]. Us: the v1 spec's four absolute clocks (planes 37-40,
   `min(t/300,1)` and `exp(-t/10)`) are a superset in information but a different parameterisation. **Add** the two
   upstream-style derived planes as well: lead in move-time units and the speed ratio, because upstream's it1->it7 gains
   came with exactly those planes (weak attribution: the model card itself says part of the it0 gap is the search
   change from flag to simulated clock, not the planes) and the network should not have to rediscover the division. Cost: 2 more channels (45 per
   board) in the one-time widening.

2. **Pace of the opponents.** Stock: no. Upstream: yes, as a per-game constant (`OpponentMoveTimeMs`) [8][10]. Us: v2
   adds elapsed thinking of the side to move (planes 41-42) but no history of how fast each seat has been moving.
   **Add** a per-seat running mean of recent move times (e.g. EMA over the last 5 moves on that board, `min(ema/5, 1)`)
   for all four seats, maintained by the arena and sent with `position`. This is what Sunsetter's "watch our time" and
   Fairy-Stockfish's `opptime` comparisons use.

3. **Flagging as a terminal.** Stock: no (time never ends a game in the tree). Upstream: no, explicitly ("a negative
   lead is not a loss") [8]. Us: `TreeMoveMs` makes a clock at or below 0 a terminal loss [L1]. Keep it; it is the one
   place where absolute clocks are strictly necessary. Also charge the opponents' clock inside the tree when they are
   on turn (upstream's `advance` charges only the acting team) so that a tree line can flag either team.

4. **Learned sitting.** Stock and upstream: hand rule, `can_sit = lead > 0` (upstream) or the flag (stock), evaluated
   at every node [9][11][L6]. Us: `FreePass` [L1]. Risk from the sources: the classical engines never sit for long
   without a reason, and the human rule "double sit 10 s = resign" [60] says the cost of a wrong sit is huge; training
   data must contain enough punished sits. **Add to the recorder** a per-sample label "seconds sat before the next own
   move" and report, per iteration, the distribution of sit lengths and the win rate of games containing sits > 5 s, so
   a collapse into sitting (or never sitting) is visible before the gate.

5. **Who moves next in the tree.** HiveMind's abstract says clocks chose the move order in the search [2]; upstream
   charges every action a fixed move time and advances the lead only when a team is on turn on both boards [8].
   Our `TreeMoveMs` gives every macro-ply the same duration on both boards. **Unknown** whether the paper's ordering
   was better; cheapest test: an option that, when the two boards' sides to move differ in elapsed thinking, expands
   the board with the longer elapsed first. Low priority until the clock build trains.

6. **Think-time control.** Stock: fixed nodes or `movetime`; two-board `wtime/btime` ignored [12]. Upstream: same, plus
   chess-style early stop and extension on eval drop / best-move change [13]. CrazyAra: 5% of remaining after move 40,
   stop at half if top prior > 0.9 and stable, extend by half on a 0.1 Q drop [22]. Us: v2 `think_gain` head [L2].
   **Baseline it** against CrazyAra's two heuristics (prior > 0.9 stable; Q drop 0.1) in the arena before trusting the
   head; the head should at least match them at equal cost. Also follow Sunsetter's floor: never allocate more than
   ~40 ms when under 4 s [46], i.e. clamp `clock/k` by remaining time, not only by k.

7. **Out-of-step boards.** Stock and upstream self-play move both boards in lockstep with one move time per team per
   game, no per-move jitter [10]. Us: jitter 0.5..2.0 per search and free waits [L1]. This is our clearest departure;
   upstream's flat it8-it13 curve suggests the lockstep regime saturates. Keep the jitter, and **record the realised
   move-time distribution per seat** so the training data's pace range can be matched to human data later.

8. **Opponent pool diversity and sanity baselines.** Stock gating is self-play vs stock. The sources give three cheap,
   different opponents: Fairy-Stockfish via XBoard with the `partner.cpp` protocol (virtual drops, sit/go) [34][41],
   Sjeng 11.2 (sits on a score band) [27], Sunsetter (fast mover, sits on trouble) [46]. **Add** a Fairy-Stockfish
   bughouse arm to the arena as a sanity floor: the HiveMind paper's 95% at 2 s is the only published cross-engine
   number for Hivemind, and we should reproduce it with the stock binary before claiming anything about our net.

9. **Network size at fixed time.** Stock: 13.57M params. Upstream: 0.97M `twin-s` beat the 13.6M teacher at 100 ms
   because of 3x nps, and dropping cross-board attention gave +32% speed [19]. Us: we widen it04. **Decide the gate
   metric now**: if the headline is "beats stock at equal wall-clock", a distilled small clock net may be the stronger
   route than widening it04, and the arena needs a fixed-time mode measured in ms on the same hardware. If the headline
   is "at equal nodes", say so in `beat_stock.md`.

10. **Mate rules under time.** Stock: `is_checkmate(team, hasTimeAdvantage)` and the RULES.md race rules [L8]; upstream
    adds frozen-race proofs and skips mate probes when `can_sit_through` the horizon [11][13]. Us: arena rule under
    `FreePass` (double check or no empty square between king and checker) and `MateProbeAnyTime` [L1]. Keep the probe:
    every classical engine's sit logic is really mate-threat logic (`CheckBadFlow`, virtual drops, `VALUE_VIRTUAL_MATE`),
    and the network at 100-800 nodes will not reliably see "my partner's capture gives me a mating drop in 1". Consider
    porting upstream's `can_sit_through`-style skip as a speed optimisation once clocks are in the tree.

11. **Piece values as a sanity check on the value head.** Classical bughouse values: Sunsetter P100 N192 B195 R200 Q390
    plus hand bonuses; Sjeng P100 N210 B230 R250 Q450 [28][45]. Neither is an input we need, but a probe of the it04
    value head's response to adding one piece to a pocket (median over positions) should land near these ratios; a large
    deviation after clock training would flag value drift.

12. **Human games with real clocks.** HiveMind's SL data is chess.com games via TCN [16][L4]. **Corrected (checked):**
    `scrape_cc.py` stores `moveTimestamps` per board; `game_reader.py` parses them, derives per-move deltas and uses
    them to order the two boards' moves (`verify_move_order`); `generate_planes.py` replays them with
    `BughouseBoard.update_time`, sets plane 31 from the real diagonal difference, and already derives `pass` labels
    exactly as described here (a team that did not move while the other board moved), while *dropping* samples where
    the team behind on time passed on an on-turn board. So a supervised warm-up of the clock planes and the pass
    action on human data needs no new scraping: fill planes 37-42 from the stored timestamps and stop dropping the
    behind-on-time sits. This is the only source of human sitting behaviour and would make the first RL iteration
    start from something other than zero-initialised clock weights.

13. **Licence and provenance.** it04 weights and the upstream repo are MIT [5][18][L3]; the paper-era repo was GPL-3.0
    [17]. Fairy-Stockfish and CrazyAra are GPL-3.0 [23][34]; Sunsetter GPLv2 [42]. Vendoring Fairy-Stockfish code into a
    distributed binary carries GPL obligations; upstream ships `--licenses` for this reason [5]. Record in the model
    sidecar: base `aminwoo/bughouse-rise-v3` rev `eac7b7a4...`, MIT.

14. **Protocol.** UCI is stateless and ignores clocks for two boards [12][39]; every real-time bughouse engine uses
    CECP/XBoard with `holding`, `partner`, `ptell`, `time`/`otim` [32][34]. Our `position ... clocks ... elapsed ...`
    extension [L1][L2] is the right shape; if the engine is ever to play on FICS/bughouse.pro, an XBoard shim mapping
    `holding`/`time`/`otim` onto it is the whole job, and the `partner.cpp` command set is the de-facto standard.

## 4. Sources

Web (fetched and read unless marked "snippet"):

- [1] https://researchr.org/publication/WooSA23 — HiveMind bibliographic record (CoG 2023, pp. 1-8); abstract missing.
- [2] https://api.openalex.org/works/doi:10.1109/CoG57401.2023.10333210 — OpenAlex record; abstract (reconstructed), closed access, 1 citation.
- [3] https://api.semanticscholar.org/graph/v1/paper/DOI:10.1109/CoG57401.2023.10333210?fields=title,abstract,tldr,openAccessPdf,citationCount — Semantic Scholar record; abstract, no open PDF.
- [4] https://api.crossref.org/works/10.1109/CoG57401.2023.10333210 — Crossref metadata; authors' affiliations (ANU; Aitchison also DeepMind), date.
- [5] https://github.com/aminwoo/hivemind — current README: twin-s-clock (78 planes), `TimeModel clock`, `TimeLeadMs`/`TeamMoveTimeMs`/`OpponentMoveTimeMs`/`RootMoveTimeMs`, MIT.
- [6] https://github.com/aminwoo/hivemind/blob/main/docs/training.md — channel table incl. lead/speed formulas, RL loop settings, promotion rule, clock sampling summary.
- [7] https://raw.githubusercontent.com/aminwoo/hivemind/main/engine/src/environment/planes.cc — exact per-channel encoding and `clock_lead_plane_value` / `clock_speed_plane_value` code.
- [8] https://raw.githubusercontent.com/aminwoo/hivemind/main/engine/src/environment/time_state.h — `TimeState`: single lead, `advance`, `can_sit`, `can_sit_through`, no flag loss.
- [9] https://raw.githubusercontent.com/aminwoo/hivemind/main/engine/src/environment/joint_action.h — joint action legality rules for passes (upstream copy).
- [10] https://raw.githubusercontent.com/aminwoo/hivemind/main/engine/src/tools/clock_sampling.h — self-play clock sampling constants (level 50%, lead 100..10,000 ms, move time 100..3,000 ms, no jitter).
- [11] https://raw.githubusercontent.com/aminwoo/hivemind/main/engine/src/search/searchthread.cc — clock charging along paths, pass always offered, waiting-board mate, frozen-board loss.
- [12] https://raw.githubusercontent.com/aminwoo/hivemind/main/engine/src/interface/uci.cc — UCI options/defaults, `team_may_sit`, two-board `go` ignores wtime/btime.
- [13] https://raw.githubusercontent.com/aminwoo/hivemind/main/engine/src/search/agent.cc — `bestmove (A,B)`/`pass` format, frozen race, `can_sit_through` probe skip, early stop/extension strings.
- [14] https://github.com/aminwoo/hivemind/releases — release notes v1.2.1..v3.0.3: clock model in v3.0.0, +156 Elo, -O3, bundled networks.
- [15] https://raw.githubusercontent.com/aminwoo/hivemind/main/docs/releases/v3.0.1.md — "charges the searched move its clock budget", #49 fix.
- [16] https://raw.githubusercontent.com/aminwoo/hivemind/v2.2.2/README.md — flag-era README: TimeAdvantage semantics, 74 planes, SL filter (min 2250, four players), RL data format, MIT.
- [17] https://raw.githubusercontent.com/aminwoo/hivemind/v1.2.1/README.md — paper-era README: GPL-3.0, no paper link, CUDA/TensorRT setup.
- [18] https://huggingface.co/aminwoo/bughouse-rise-v3 — it04 model card: architecture, 13.57M params, I/O, **licence MIT**, no clocks.
- [19] https://huggingface.co/aminwoo/bughouse-twin-s — distilled 0.97M net: speed vs size result, self-play Elo table.
- [20] https://huggingface.co/aminwoo/bughouse-twin-s-clock — clock lineage card: plane formulas, sampling, per-iteration Elo, caveats about it0.
- [21] https://arxiv.org/abs/1908.06660 — CrazyAra paper abstract and metadata.
- [22] https://arxiv.org/pdf/1908.06660 — CrazyAra full text (extracted locally with pypdf): input planes, MCTS constants, Sec. 9.2.4 time management, data and results.
- [23] https://github.com/QueensGambit/CrazyAra — README: variants, paper list, GPL-3.0, lichess accounts.
- [24] https://arxiv.org/abs/2012.11045 — MCGS for AlphaZero abstract.
- [25] https://sources.debian.org/src/sjeng/11.2-15/partner.c — Sjeng partner protocol, piece-value requests, `CheckBadFlow`.
- [26] https://sources.debian.org/src/sjeng/11.2-15/sjeng.c — `must_sit`/`must_go` consumers in the main loop.
- [27] https://sources.debian.org/src/sjeng/11.2-15/search.c — sit trigger band, partner messages, bughouse search tweaks.
- [28] https://sources.debian.org/src/sjeng/11.2-15/eval.c — zh piece values and hand-indexed king safety table.
- [29] https://sjeng.org/indexold.html — Sjeng FICS/ICC ratings (self-reported).
- [30] https://sjeng.org/news.html — release notes ("Sit if losing a piece"); **snippet only**.
- [31] https://raw.githubusercontent.com/fairy-stockfish/Fairy-Stockfish/master/src/variant.cpp — bughouse variant definition.
- [32] https://raw.githubusercontent.com/fairy-stockfish/Fairy-Stockfish/master/src/xboard.cpp — `holding`, `partner`, `ptell`, time commands.
- [33] https://raw.githubusercontent.com/fairy-stockfish/Fairy-Stockfish/master/src/partner.h — `PartnerHandler` fields.
- [34] https://raw.githubusercontent.com/fairy-stockfish/Fairy-Stockfish/master/src/partner.cpp — ptell command set and semantics.
- [35] https://raw.githubusercontent.com/fairy-stockfish/Fairy-Stockfish/master/src/search.cpp — sit/dead/fast logic, clock messages, virtual-drop wait loop (verbatim).
- [36] https://raw.githubusercontent.com/fairy-stockfish/Fairy-Stockfish/master/src/position.h — `allow_virtual_drop` / `virtual_drop` code.
- [37] https://raw.githubusercontent.com/fairy-stockfish/Fairy-Stockfish/master/src/movegen.cpp — virtual drops generated only as checks, stripped from legal list.
- [38] https://raw.githubusercontent.com/fairy-stockfish/Fairy-Stockfish/master/src/position.cpp — `pseudo_legal` drop check with virtual drops.
- [39] https://github.com/fairy-stockfish/Fairy-Stockfish/issues/812 — maintainer on virtual drops ("very crude hack"), XBoard-only bughouse support.
- [40] https://github.com/fairy-stockfish/Fairy-Stockfish/issues/811 — bughouse.pro rule-option requests; single-board scope confirmed.
- [41] https://github.com/fairy-stockfish/Fairy-Stockfish/wiki/Setting-up-Fairy-Stockfish-on-FICS — zippy bughouse setup.
- [42] https://sunsetter.sourceforge.net/results.html — Sunsetter FICS crazyhouse records.
- [43] https://raw.githubusercontent.com/georgvonzimmermann/Sunsetter/master/partner.cpp — partner command parser.
- [44] https://raw.githubusercontent.com/georgvonzimmermann/Sunsetter/master/bughouse.cpp — main loop, no pondering in bughouse, piece values, search constants.
- [45] https://raw.githubusercontent.com/georgvonzimmermann/Sunsetter/master/evaluate.cpp — in-hand bonuses, king-safety multiplier, bughouse eval terms.
- [46] https://raw.githubusercontent.com/georgvonzimmermann/Sunsetter/master/search.cpp — sit messages/conditions, bughouse time allocation.
- [47] https://raw.githubusercontent.com/georgvonzimmermann/Sunsetter/master/interface.cpp — `time`/`otim`, holdings parsing, partner setup.
- [48] https://raw.githubusercontent.com/georgvonzimmermann/Sunsetter/master/board.cpp — `timeToMove`, flag-fall handling.
- [49] https://www.chessprogramming.org/Crazyhouse — engine list, Multi-Variant Stockfish "+800 Elo", championships.
- [50] https://talkchess.com/viewtopic.php?p=937161 — jorose 2022: engine bughouse setups, wait command idea.
- [51] https://talkchess.com/viewtopic.php?p=981522 — "Is Engine Bughouse Chess Possible?" (2025): Fairy-Stockfish, chess.com bridge link.
- [52] https://talkchess.com/viewtopic.php?p=684933 — 1st CCC World Computer Crazyhouse Championship field and settings.
- [53] https://talkchess.com/viewtopic.php?p=343503 — time management thread: ~70 Elo per doubling, easy-move rule.
- [54] https://talkchess.com/viewtopic.php?p=985388 — ponderhit time management; hgm on scaling to opponent think time.
- [55] https://github.com/RobbieGM/chessdotcom-bughouse-engine — chess.com Tampermonkey + Fairy-Stockfish bridge; unverified "~3800" claim.
- [56] https://www.chess.com/blog/JarlCarlander/bughouse-and-ai-2 — "bughouse engines are not stronger than the strongest humans"; **snippet only** (page blocked).
- [57] https://www.chess.com/clubs/forum/bughouse — "computers are not significantly better than humans at bughouse"; **snippet only**.
- [58] https://lichess.org/@/ToromBot/perf/crazyhouse — 2753 crazyhouse; **snippet only**, BOT status unconfirmed.
- [59] https://lichess.org/forum/redirect/post/tLFVL8TR — CrazyAra 2613 vs JannLee 2639, Dec 2018; **snippet only**.
- [60] https://www.chess.com/blog/Sorsi/time-management-in-bughouse — "double sit for about 10 seconds ... resign"; **snippet only**.

Local (read in the repo):

- [L1] `docs/custom_engine_spec.md` — v1 contract: clock planes 37-40, FreePass, TreeMoveMs, flags terminal.
- [L2] `docs/custom_engine_spec_v2.md` — elapsed-thinking planes, think-time head, arrival head.
- [L3] `third_party/hivemind/LICENSE` — MIT, (c) 2026 aminwoo.
- [L4] `third_party/hivemind/src/hivemind/network.py`, `src/hivemind/data/scrape_cc.py`, `src/hivemind/data/tcn.py` — pinned weights/hashes; chess.com scraper and TCN decoder.
- [L5] `third_party/hivemind/engine/src/environment/planes.cc` — stock 37-channel layout plus our clock planes.
- [L6] `third_party/hivemind/engine/src/environment/joint_action.h` — `is_double_sit_legal`, `is_single_pass_legal`.
- [L7] `third_party/hivemind/engine/src/interface/uci.cc` — `TimeAdvantage`, `Mode` alias, `RequireMoveOn`, `pass` display.
- [L8] `third_party/hivemind/RULES.md` — mate-race rules under the time advantage.
- [L9] `third_party/hivemind/engine/src/tools/selfplay.cc` — flag alternates between teams each macro-ply.
- [L10] `third_party/hivemind/engine/src/search/agent.cc` — probe targets collected only when `!rootTeamHasTimeAdvantage`, and the selected-move certifier returns false when `teamHasTimeAdvantage` (the probe runs only for the team *behind* on time; v1's `MateProbeAnyTime` lifts this).
- [L11] `NOTES.md` — `-Ofast`/`-ffinite-math-only` selection bug and parity results.
- [L12] `docs/beat_stock.md` — fixed-node gating design.
