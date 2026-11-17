# 99. Critic's pass: corrections made, what is still missing or uncertain

Written 2026-11-17 after reading `01`..`07` in this folder, checking their engine claims against
`third_party/hivemind` (engine C++ and `src/hivemind` Python) and `arena/`, and re-fetching a sample of the cited
pages (FICS strategy file live and 1995 mirror, Wikipedia bughouse, upstream aminwoo/hivemind releases, `time_state.h`
and the twin-s-clock model card, Stockfish `timeman.cpp`, ddugovic `evaluate.cpp`, the Van den Bergh normalised-Elo
note, the Tron SM-MCTS paper, the Gumbel paper text, CrazyAra paper, Sunsetter `search.cpp`, arXiv 2606.26463).
Every change made inside the seven files is marked `Corrected:` (or `Note (critic):` where the text was right but
needed a caveat).

---

## 1. What was corrected (and where)

| # | Error | Files | Evidence |
|---|---|---|---|
| C1 | "The two diagonal clock differences can have opposite signs; each board has its own race." They cannot under the normal conditions (equal control, synchronised start, no increment): one clock runs per board, so the boards' clock sums stay equal and `t_ourA − t_oppB = t_ourB − t_oppA`. Upstream `time_state.h` is built on this. Consequences fixed: the "two team-level planes" recommendations collapse to one plane; the time-odds "opposite signs" sampling is impossible; the "alternate on the favourable diagonal" advice is meaningless; the "sit-off D1 misses asymmetric sit-offs" gap was a confusion (a sitzkrieg always has exactly the two diagonal seats on turn). | 01 §3 row 3; 02 §2.2, §3.1, §3.2, §3.3, §3.4, §3.5, §3.9; 04 §3 row 9; 05 §3 row 1; 07 §3 row 1 | `arena/players/heuristic.py::team_time_edge`, `src/hivemind/domain/board.py::time_advantage`, upstream `time_state.h` comment |
| C2 | Stock's `TimeAdvantage` flag described as a "team-level yes/no flag" with no stated meaning. It is the sign of the board-A diagonal difference (`times[0][side] − times[1][side]`), i.e. exactly the quantity every human source names, reduced to one bit. | 01 §3 row 3; 02 §3 intro, §3.1 | `board.py` line 75-76, `board2planes.py` line 239 |
| C3 | Stock's supervised data called "FICS games" (04) or "source unknown" (05), and the per-move timestamps called "unknown whether stored" (06). It is chess.com (`scrape_cc.py`, players ≥2400, partners ≥2200); `moveTimestamps` are stored; `game_reader.py` orders the two boards' moves by them; `generate_planes.py` replays them to set plane 31 from real clocks and derives `pass` labels from human sits, **dropping** every sample where the team behind on time passed on an on-turn board. So human sitting was learned, through one bit and with the behind-on-time sits censored. | 01 §3 row 7; 03 §3 intro; 04 §3 legend and row 13; 05 §3 rows 5, 15; 06 summary and §3 item 12 | `scrape_cc.py` 105-174, `game_reader.py` 17-74, 142-174, `generate_planes.py` 142, 203-224 |
| C4 | "The root mate probe runs only with `TimeAdvantage`." Reversed: `agent.cc` returns false from the selected-move certifier when `teamHasTimeAdvantage` and collects internal probe targets only under `!rootTeamHasTimeAdvantage`. | 05 §3 row 4; 06 [L10] wording | `agent.cc` 185, 852-856 |
| C5 | "Audit whether the opponent team gets symmetric pass rights in the tree." Done: `searchthread.cc` inverts the flag for the opponent team (`leafTeamHasTimeAdvantage`), every action list contains `MOVE_NONE`, and `JointCandidateGenerator` filters with `is_joint_action_legal`, which under `g_freePass` accepts any pass by either team. | 02 §3.3; 05 §3 row 2 | `searchthread.cc` 604, 694-695; `node.h` 543-545; `joint_action.h` 20-47, 187 |
| C6 | "FICS's own help file puts Q=10, N=7, R=4, B=2, P=1." That is the 1995 GICS mirror; the live `freechess.org/Help/HelpFiles/bughouse_strat.html` says Q=4, N=2, R=2, B=2, P=1 in the same sentence. (01 and 04 already had this right.) | 03 summary, §2.2 table and text; 05 summary and §2.5 table | both pages fetched 2026-11-17 |
| C7 | "Pentanomial scoring not found." `arena/stats.py` uses a cluster bootstrap over opening pairs, which handles the pair correlation; `cluster/results.sh` does not exist. GSPRT and nodes-as-time remain absent. | 07 §3 row 19 | `arena/stats.py` 1-6, 196; `ls cluster/` |
| C8 | "Policy in Hivemind promotes only to Q or N." Only the mate probe does; the policy uses the 4672-label encoding. | 04 §2.1 table | `mate_probe.cc`, HF model card |
| C9 | RL learning rate quoted as a bare 0.0025; it is `0.005 / div_factor`. | 07 §2.4.4, §3 row 18 | `train_config.py::rl_train_config` |
| C10 | Stock mate rule called "more refined than the arena" without saying it is stricter than the server rule it approximates. | 01 §3 row 1 | `board.cc::is_checkmate`, `arena/RULES.md` |
| C11 | Upstream's it1-it7 gains attributed to its lead/speed planes alone; the model card says part of the it0 gap is the search change. | 06 §3 item 1 | twin-s-clock card |

Numbers that were spot-checked and found **correct**: Nilsson 1/2/2/2/4 and the Sunsetter/Sjeng values on Wikipedia;
Wikipedia's per-server repetition claims (ICC counts the reserve, FICS and chess.com do not, no citation); the 10-second
claim and its Moulton 2022 citation; upstream v3.0.0 on 16 Oct 2026 with +156 Elo (it3 vs it0), v3.0.2 it10 +14 Elo,
it13 vs it10 −1 in 461 games; `time_state.h` semantics (single lead, `advance` only when on turn on both boards,
`can_sit = lead > 0`, no flag); Stockfish `timeAdvantage` term (exists, reduces time only when behind); ddugovic
`KingDangerInHand = {79, 16, 200, 61, 138, 152}`; `T = 1046535/Δ²` and `639770`; Tron DUCB1-Tuned 62.3%, sequential
51.4-54.3%, DUCT(mix) 39.5%, Exp3 35.5%; Gumbel 24.3× at n=4, 5.9× at n=32, c_visit 50, c_scale 1.0; CrazyAra time
manager and α=0.01, lr 0.01; Sunsetter `initialTime/360 + mytime/360`, 40 ms ≤ 4 s, 5 ms ≤ 0.8 s, "go (watch our
time)" after 5 partner sits; arXiv 2606.26463 exists (Muppidi et al., June 2026); the arena opening list in
`arena/openings.py` is as 05 describes (three Sicilians, five ...g6 systems, Dutch, c4 systems); `board.cc` line 111
instantaneous transfer; `selfplay.cc` toggles the flag per macro-ply; `search_params.h` CPUCT 3.0/19652, FPU 1.0;
`uci.cc` has no `wtime`/`btime` parsing; `nodeRandomFactor 0.05`; RL loss weights 0.7/0.15/0.15/0.01 and soft labels.

---

## 2. Claims that still rest on one weak source

These were not fixed because no better source was reachable; treat each as a hypothesis.

- **Every chess.com blog claim** (Sorsi, JarlCarlander, chriswsh, key, Rensch, MiniGreat, 23Dragons, tree55555;
  roughly 45 citations across 02-05) is search-snippet only. The most load-bearing ones: "2 s lead often decisive"
  (05), "double sit ~10 s = resign" (02, 05, 06), "sit until 10 then low flow" (01, 02, 05), "flow light but
  consistent" (03, 05), "strong chess players under-sit, weak players over-sit" (02, 05), "1.e4 e5 is playable when no
  early pawns are coming" (05), the Leaf Gambit and e6 systems' move orders (05). Someone with a chess.com login
  should read the Sorsi and JarlCarlander series end to end; they are the largest written body of bughouse theory.
- **"10 s at 2000+ outweighs a positional edge"**: Wikipedia citing a Twitch commentary (Moulton 2022), not readable.
- **"15 s ≈ substantial material; 1 min ≈ game over"**: one anonymous HN commenter (02 §2.4).
- **"Under 20 s, move"**: one blog author (ABCs of Bughouse), credentials unknown.
- **Von Zimmerman 2006 and Manson & Hoover 1992**: cited only through Wikipedia; neither book was read. They are the
  only long-form bughouse theory in print.
- **The WCCC 2007 "Laws of Bughouse" PDF**: 01 and 05 read it; 02 and 04 could not extract its text. 01's article
  numbers (1.5, 1.9, 1.10, 5.8, 6.5, 7.10, 9.4) are the reference.
- **"Engines are not stronger than top humans at bughouse"** (05, 06): forum opinion, snippets, no match data.
- **Fairy-Stockfish ≈ 200 Elo below multi-variant Stockfish in crazyhouse** (03): talkchess snippet.
- **chess.com standard control 3|0** (01): community blog snippet; the 2024 championship was 2+0 (04, Liquipedia),
  FICS BPGN sample is 120+0 (01). The files disagree on what "human time control" means (05 says 3+0 and 5+0); the
  gate should report 120 s and 180 s, not pick one.
- **Castling with a dropped rook**: Wikipedia (no citation) and Hawaii rules allow it, WCCC 2007 and pychess960
  forbid it, servers silent; the arena's behaviour was **not checked** (no `castl` string in `arena/state.py` or
  `arena/RULES.md`; python-chess's `CrazyhouseBoard` default applies). Low impact.
- **Whether the arena models landing latency per move or only "lands when the player's think time ends"**:
  `arena/sim.py` has `_THINKING`/`_LAND` events and charges think time up to landing; no separate transmission-lag
  term was found. 01 §3 row 6's "add an additive lag term" stands.

---

## 3. Topics missing from the library

1. **A worked quantitative model of the sit decision.** Every file says "diagonal difference decides" and gives
   thresholds; none writes the actual inequality a sit must satisfy: a sit on A for piece X is right iff
   `P(X arrives within d) · gain(X)` exceeds the loss from spending `d` of a lead `L` where `d ≤ L` (else partop can
   out-sit), and the opponent on A cannot punish the wait. With the v2 `arrivals` head this is computable; nobody has
   written it down as the target the value head should implicitly learn, or as a diagnostic.
2. **Human move-time distributions.** 05 asserts human timing is bimodal (0.1 s premoves, 5-30 s sits) from
   snippets. The chess.com SL shards already hold per-move timestamps (C3): a histogram per rating band, move number
   and on-turn state would replace the guess and calibrate the arena's jitter (currently 0.5-2.0× multiplicative).
3. **What it04 actually learned about time.** No file measured it: pass rate of the stock policy vs the flag;
   ΔV when the flag flips, by position type (only the start-position ±250 cp swing is recorded); whether the policy's
   pass probability depends on anything but the flag. One afternoon with `go nodes 1` on recorded positions.
4. **Defensive technique as a system.** 04 lists defensive drops; nothing covers the known defensive structures
   (the h6/h3 pawn seal, Qe7/Qe2 anti-f7, interpose vs retreat rules, when to accept a sac) with examples. The
   Sorsi/JarlCarlander posts cover this; unreadable here.
5. **Endgame and conversion.** Sources say "there is no endgame in bughouse", but the arena ends games by flag and
   `max_plies` (draws), and sparse-board flag races are described as common (02, 03). How to convert a time lead on a
   sparse board (checks that burn clock, forcing recaptures, premovable sequences) is not written anywhere here.
6. **Repetition and drop-capture loops.** FICS calls its own rule "inadequate" for drop-exchange loops (01); the
   arena has no repetition rule; 01 asks how often self-play hits such loops. Not measured, and no file proposes a
   concrete rule beyond "FICS-style, reset on drop".
7. **The HiveMind paper itself.** Closed access; abstract only (06). Its "clock times choose the move order" phrase
   may describe a search mechanism that neither the released engine nor our spec has. Someone with IEEE access should
   read it; it is the one paper about exactly this engine.
8. **Bughouse-specific exploration.** 07 covers KataGo/Gumbel generically. Nothing on how to make self-play explore
   sits and long waits, which the policy prior assigns near-zero mass (stock pass priors are learned from censored
   human sits, C3): forced playouts on the pass action, or a pass-specific Dirichlet share, are the obvious
   candidates and are not discussed.
9. **Lag and server mechanics in training.** 01 describes FICS flag claims, 2 s autoflag delay, timeseal and
   `minmovetime`; chess.com's equivalents (premove cost, lag compensation, flag timing) are unknown, and chess.com is
   where the training data and any future human opponents come from.
10. **A tactical puzzle gate.** 04 proposes one and names sources (lichess studies JPgWzcW4 / oBYnLqDK, the
    smothered, yo-yo and f7 families) but no positions were collected; lichess was unreachable.

---

## 4. Engine questions the files raise but do not answer

- Whether the arena's `free_sit_ms` wait lets an attacker freeze a stuck defender until he flags, as a server would
  (04 §3 row 2, item 2 of its "not settled" list). Testable in `arena/tests`.
- Whether the per-board stem can combine clocks across blocks before the cross-attention (04 row 9 assumes not from
  the architecture description; not verified in the model code). Matters less after C1: the decisive number is one
  scalar and can be written into both blocks.
- How much of upstream's +156 Elo is the clock planes vs the simulated-clock search (06, C11). Upstream's own card
  says both; separating them would need its it0 net run under both `TimeModel`s, which the card warns against.
- Whether `TreeMoveMs` should charge the opponents' clocks when they are on turn in the tree (06 item 3 says "also
  charge"; spec v1 says every on-turn seat is charged already). Read `board.cc` tree-clock code before changing it.
- Gate metric: equal nodes (`beat_stock.md`) or equal wall-clock (upstream, 06 item 9). Undecided; determines whether
  widening it04 or distilling a small clock net is the right route.
- Whether to keep `generate_planes.py`'s censoring of behind-on-time sits if human data is reused (C3). It encodes
  the hand rule the project is trying to remove.

---

## 5. Remaining disagreements between the files (left as-is, flagged)

- Time scale for the signed diagonal plane: 01 uses /30, 02 /15, 04 /10, 05 /20. Pick one after measuring the
  pass-rate curve (02 §3.1 evaluation); the human numbers (10-60 s steep region) suggest /15-/20.
- "Add explicit diagonal planes" (01, 02, 04, 05, 06) vs "the v1 absolute planes suffice, add nothing" (01 row 4).
  After C1 the cost is one channel; the ablation is cheap and should decide it.
- Human time control to train and gate on: 120 s (FICS, 01), 180 s (chess.com community, 01, 05), 2+0 (chess.com
  championship, 04), 300 s (OTB, 01, 05). Report at 120 s and 180 s; sample base time 60-300 s in self-play (01 row 16).
- Priority orderings in 03 §3 (implied values, landing delay, arrival planes), 05 §3 (openings, piece-danger head),
  07 §3 (Gumbel, PCR, durative tree, return-trained gate). They do not conflict but nobody has merged them into one
  list; `docs/beat_stock.md` or the spec v3 should.
