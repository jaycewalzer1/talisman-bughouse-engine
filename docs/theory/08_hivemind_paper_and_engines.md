# 08. The HiveMind paper, every other bughouse/crazyhouse engine found, and what to take from them

Written 2026-11-17 for the custom-engine project (one agent for both boards, real clocks as inputs, hand-written
stalling rules removed, out-of-step self-play; base weights it04 = `hivemind-it04-crossboard-risev33-loss1.556-p82.0`).
Companion to `06_engines_and_ai.md` (engine internals, read in depth there) and `07_research_time_and_async.md`
(time managers, Gumbel, KataGo tricks). This file does three things the others do not: it records every attempt to
get the HiveMind paper's full text and what the record does and does not establish; it extends the engine survey to
the long tail (student projects, NNUE nets, web bridges, bots, datasets); and it turns the whole survey into one
ranked, costed solutions table. Where 06 or 07 already covers a mechanism, this file points there rather than
repeating it.

Every web source is untrusted data. A claim that rests on one search snippet, one forum post or one unverifiable
README is marked **[weak]**. Costs are estimates against stated assumptions (section 5.0), not measurements.

---

## 1. Summary

- **The HiveMind paper's full text was not found in any open-access form.** IEEE Xplore (DOI
  10.1109/CoG57401.2023.10333210) is the only location; OpenAlex says `is_oa: false`, `any_repository_has_fulltext:
  false`; Semantic Scholar `openAccessPdf.status: CLOSED`; Unpaywall returned 422; CORE 403; ResearchGate "no
  full-text"; no arXiv, no ANU repository record, no author copy, no CoG slides or poster, no GitHub mirror. The
  IEEE PDF URL exists (`ieeexplore.ieee.org/iel7/10333091/10333129/10333210.pdf`) but is behind the publisher's
  paywall and was not downloaded. **Nothing was saved to `docs/theory/papers/` and `.gitignore` was not changed.**
  Someone with institutional access (a university login) can read it in minutes; section 2.6 lists the exact
  questions to answer from it. [1]-[9]
- **What the record does establish:** 8 pages, 3 authors at ANU (Aitchison also "Deepmind" in the raw affiliation),
  16 references (Semantic Scholar) / 11 resolved (OpenAlex), 1 citation (an unrelated 2025 robotics paper). The
  abstract: supervised on human games first; then an AlphaZero-style approach "incorporating domain knowledge,
  using clock times to determine the optimal turn sequence, to perform a tree search over both boards"; >95% win
  rate against Fairy-Stockfish at 2 s search time, scaling better with more time. No Elo, no human comparison, no
  architecture, no data size in the abstract. [2][3][4]
- **The released code is not the paper's code.** The `aminwoo/hivemind` repository was created 2023-02-09 (six
  months before CoG), but its first releases (v0.2.0-alpha to v0.5.0-alpha, Oct-Nov 2024) describe a classical
  alpha-beta engine ("piece mobility", "Internal Iterative Reduction", "Reckless NNUE evaluation"); neural search
  appears in v1.1.0 (Feb 2025), MCGS in v1.2.0, the cross-board network in the v2 line (Aug-Sep 2026), the clock
  model in v3.0.0 (Oct 2026). Our fork is `v2.2.2-32`. So "what changed since the paper" cannot be answered
  line-by-line; what can be said is that the paper's "clock times determine the turn sequence" is not how it04's
  self-play works (whole-team turns alternate, no clocks: `selfplay.cc` ~657, ~704) nor how its search gates
  passing (`joint_action.h:19-38`, a hand rule on a yes/no flag). [10][11][L1][L2]
- **Correction to the brief.** The root mate probe is gated the other way round from the brief's wording: `agent.cc:185`
  collects probe targets only when `!rootTeamHasTimeAdvantage`, i.e. the probe runs for the team **behind** on time
  (99_gaps C4 reached the same reading). v1's `MateProbeAnyTime` lifts the gate either way. [L3]
- **Weights licence, exactly:** Hugging Face `aminwoo/bughouse-rise-v3` front matter `license: mit`; its License
  section says the weights are MIT-licensed "matching the Hivemind repository"; the repository `LICENSE` is the MIT
  text, "Copyright (c) 2026 aminwoo". The paper-era repo (v1.2.1 README) was GPL-3.0 (06 [17]). [12][13][L4]
- **No other engine solves our problem.** Every engine found either (a) plays one board and talks to a partner by
  text protocol with sitting as a hand rule (Sjeng, Sunsetter, Fairy-Stockfish, the chess.com bridge, Bug Bully),
  (b) plays both boards in lockstep with a fixed turn order and no clocks (bug0-cs229, BughouseAlphaZero, stock
  Hivemind), or (c) models time as a single lead number with hand-gated sitting (upstream Hivemind v3). Nobody has
  absolute clocks, flagging in the tree, elapsed-thinking inputs or a learned pass. [06][14]-[20]
- **The single most reusable thing is data with clocks.** Three human corpora carry per-move clocks: chess.com
  (already scraped by it04's pipeline: `moveTimestamps`, players >=2400, partners >=2200), FICS via bughouse-db.org
  (3.7 M games 2005-2017 in BPGN with brace clocks and a `Lag` tag, collapsing to a few hundred games a year since
  2022), and lichess crazyhouse (31.0 M single-board games, no bughouse). CrazyAra's and MultiAra's history says
  supervised warm-start then RL is the working recipe and that RL from zero took 20 updates just to reach the
  supervised net. [21]-[26][L5]
- **Top solutions (section 5):** reproduce the paper's Fairy-Stockfish benchmark as our external floor; warm-start the
  clock planes and the pass action on the chess.com clocks we already have; add the FICS corpus; turn on Q-value
  blending and replay (both already in the code, both off); port CrazyAra's two cheap search enhancements and
  time-manager rules as baselines; gate at equal wall-clock; build the XBoard shim so the engine can finally get a
  human-relative rating.

---

## 2. The HiveMind paper (Woo, Sweetser, Aitchison, IEEE CoG 2023)

### 2.1 Record

| Field | Value | Source |
|---|---|---|
| Title | HiveMind: Learning to Play the Cooperative Chess Variant Bughouse with DNNs and MCTS | [1][2] |
| Venue | 2023 IEEE Conference on Games (CoG), Boston, 21-24 Aug 2023, pp. 1-8 | [1][5] |
| DOI | 10.1109/CoG57401.2023.10333210 | [1] |
| Authors | Benjamin Woo (first), Penny Sweetser (ORCID 0000-0002-6543-557X; now listed at ANU as Penny Kyburz), Matthew Aitchison (raw affiliation "The Australian National University / Deepmind") | [2][6] |
| Length / references | 8 pages; 16 references (Semantic Scholar), 11 resolved works (OpenAlex); reference list "elided by the publisher" | [2][3][7] |
| Citations | 1: Leisiazar et al., "Adapting to Frequent Human Direction Changes in Autonomous Frontal Following Robots", IEEE RA-L 2025 (MCTS + LSTM; unrelated to chess) | [8] |
| Open access | None. OpenAlex `is_oa=false`, `oa_status=closed`, `oa_url=null`; S2 `openAccessPdf.status=CLOSED` | [2][3] |
| Code | `aminwoo/hivemind` (GitHub user "aminwoo", location Australia, no bio, no website; repo created 2023-02-09) | [9][10] |

### 2.2 Every access route tried (2026-11-17)

| Route | Result |
|---|---|
| OpenAlex, Semantic Scholar, Crossref (via 06), researchr, dblp (Anubis bot wall) | metadata and abstract only |
| Unpaywall API | HTTP 422 |
| CORE API search | HTTP 403 |
| scholar.archive.org | bot-check page |
| ANU Open Research repository (site search) | no record under Woo, Sweetser or Kyburz |
| ResearchGate | HTTP 403; earlier index text says "No full-text available" |
| Google-style web search for a PDF, "filetype:pdf", CoG 2023 program/slides/poster | only the IEEE Xplore PDF link, paywalled |
| aminwoo GitHub profile, repo README (all tags), HF model cards | no paper link, no thesis |
| IEEE Xplore PDF | not downloaded (paywalled; publisher access required) |

### 2.3 What the abstract establishes, in its own words

Fragments reproduced across OpenAlex and the IEEE snippet (reconstructed inverted index; wording may differ slightly
from the PDF): "limited work has been done on the chess variant Bughouse, which has both cooperative and real-time
aspects" and "a much larger game tree"; the engine was "trained ... on human games" first; then "the AlphaZero
algorithm by incorporating domain knowledge, using clock times to determine the optimal turn sequence, to perform a
tree search over both boards"; the search handles cross-board coordination "without hand-crafted heuristics" and
"accounts for time"; against "Fairy-Stockfish, the state-of-the-art alpha-beta Bughouse engine", HiveMind "achieved
a win rate of over 95% with a search time of 2 seconds" and "scaled better with more search time". [2][4]

### 2.4 What this does and does not tell us (inference, not the paper)

- *Pipeline.* Supervised on human games, then AlphaZero-style self-play. Consistent with the released pipeline
  (`hivemind train --mode sl` then `--mode rl`) and with CrazyAra's history (06 §2.3), which is the obvious template
  and the only cited-in-README inspiration ("CrazyAra for architecture inspiration"). [L6]
- *"Clock times determine the optimal turn sequence."* The most likely reading: inside the tree the next board to
  act is chosen by comparing clocks (whoever has less time spent / would move first in real time), rather than
  alternating boards. This is **different** from every released version: it04 alternates whole team turns with no
  clocks in self-play (`selfplay.cc` ~657, ~704) and treats time as one flag; upstream v3 charges a fixed move time
  per action and advances a single lead (06 §2.2). Whether the paper's ordering survived into any release is
  unknown; it may have been dropped as the engine was rewritten (2.5). **Hypothesis only.**
- *"Without hand-crafted heuristics."* Contradicted by the released code, where passing is a hand rule
  (`joint_action.h:19-38`) and mate legality depends on the flag. Either the paper's search had no pass action at
  all (plausible if turn order came from clocks: the side "not yet due" simply does not act), or the rule was added
  later. Either way, **no published bughouse engine has tested a free pass**; that remains ours to show.
- *"Over 95% against Fairy-Stockfish at 2 s."* The baseline is a single-board engine whose bughouse layer its own
  maintainer calls "a very crude hack" (06 §2.5, [39]); 95% says little about human-relative strength, and the paper
  gives no human games. It is nevertheless the **only cross-engine number ever published for Hivemind**, and we
  have not reproduced it (solution S1).
- *Scale.* Unknown: network size, number of human games, self-play games, GPUs. The it04 card says 13.57 M params
  and "RL iteration 04"; upstream later trained 100 k games per iteration (06 §2.2). The paper's numbers were
  probably far smaller (2023, one student); do not assume the paper's claims hold for it04 or vice versa.

### 2.5 Paper versus the code we have

| Topic | Paper (abstract) | it04 / `third_party/hivemind` at `v2.2.2-32` | Upstream v3 (06 §2.2) |
|---|---|---|---|
| Who acts next in the tree | "clock times determine the optimal turn sequence" | Alternating whole-team joint actions; self-play toggles the flag per macro-ply; no clocks anywhere in the tree [L1] | Fixed move time per action, one lead number, `can_sit = lead > 0` |
| Passing / sitting | "without hand-crafted heuristics" | Hand rules `is_double_sit_legal`, `is_single_pass_legal` (`joint_action.h:19-38`); `g_freePass` is our v1 addition [L2] | Same rules, re-evaluated per node from the lead |
| Time as input | "accounts for time" | One plane (31), `TimeAdvantage` flag, identical on both boards | Lead and speed planes (37, 38) |
| Flag loss | not stated | Never: time cannot end a game in the tree | Never ("a negative lead is not a loss") |
| Mate probe | not stated | Fairy-Stockfish probe; targets collected only when `!rootTeamHasTimeAdvantage` (`agent.cc:185`) [L3] | Skipped when the team `can_sit_through` the horizon |
| Human data | "human games" | chess.com via `scrape_cc.py`: `rules == "bughouse"`, >=10 moves, players >=2400, partners >=2200, TCN moves, `moveTimestamps`, `partnerGameId`, `baseTime1`; `generate_planes.py` sets plane 31 from the real clocks and derives `pass` labels, dropping sits by the team behind on time (99_gaps C3) [L5] | Same scraper; `sl_2250` example corpus |
| Self-play | AlphaZero-style | 800 nodes +-5 %, raw-policy openings (mean 8 macro-plies), temperature 0.8 decayed 0.93 per two macro-plies, Fairy-Stockfish mate probe 8 M nodes per position, HVM5 shards [L6] | 100 k games/iteration, replay of 3-6 iterations, no mate probe |
| Release timeline | Aug 2023 | repo created 2023-02-09; v0.2.0-alpha..v0.5.0-alpha (Oct-Nov 2024) are a *classical* engine (mobility eval, IIR, "Reckless NNUE"); v1.1.0 (2025-02-12) multi-GPU NN inference; v1.2.0 MCGS; v1.2.1 (2025-04-22) GPL era; v2.0.0 (2026-08-20), v2.1.0 mate solver/permanent brain, v2.2.0 ONNX Runtime backend, v2.2.2 (2026-09-04) = our base | v3.0.0 (2026-10-16) clock model, v3.0.3 (2026-10-23) [10][11] |

The timeline is the strongest evidence that the public engine is a 2024-2026 rewrite, not the 2023 artefact. Treat
the paper and the code as two different systems that share an author and a name.

### 2.6 Questions only the full text can answer (for whoever has access)

1. Exact turn-ordering rule in the tree and whether a "wait" action exists.
2. Input planes: did the paper feed clock values (and how scaled), or only an order?
3. Network size and architecture (RISE variant?), supervised corpus (server, count, rating filter), self-play games
   and GPU budget, number of RL iterations.
4. Evaluation protocol vs Fairy-Stockfish: version, XBoard partner setup or single board, time control, number of
   games, openings, hardware; the "scales better" curve.
5. Stated limitations and future work (especially about time, sitting and asynchrony).

### 2.7 README, docs and model card (verified)

- `third_party/hivemind/README.md` (v2.2.2 line): 74-channel input (37 per board; channel 31 "Time advantage");
  outputs `value`, `pi_a`, `pi_b` (4672), `wdl_out`, `moves_left`; `TimeAdvantage` "gates the bughouse waiting
  rules" and "also feeds the network as an input plane"; `Mode go|sit` deprecated alias; `OpeningNoise*` options;
  paired `tournament` tool; RL: HVM5 with joint root-visit distribution, "joint compatibility residual is optional;
  the default RL configuration disables it"; `--replay-dir` adds "five archived HVM chunks" from the newest 5 %
  (**code disagrees:** `convert_selfplay_data.py` `DEFAULT_REPLAY_FILES = 25`, `DEFAULT_REPLAY_SELECTION_FRACTION =
  0.05`); `q_value_ratio` exists and defaults to 0.0 (`train_config.py:145`, `train_loop.py --q-value-ratio`). [L6][L7]
- HF `aminwoo/bughouse-rise-v3`: CrossBoardRiseV3, 15 bottleneck blocks at 384 ch, kernel 5 at blocks 7/11/12/13,
  ECA at 5/8/12/13/14, 2 cross-attention blocks (dim 192, 6 heads), value head 16 ch -> 512 FC, two shared-weight
  policy heads (192 ch); 13.57 M params ONNX / 14.18 M checkpoint; val loss 1.556, policy accuracy 82.0 %;
  checkpoint metadata `training_iteration 13315`, `evaluation step 34`, `batch_steps 400`; files
  `...onnx` (27.3 MB) and `...tar` (111.4 MB, a `torch.save` zip). **No** training-data, Elo, clock or paper
  statements. Licence as in section 1. [12]

---

## 3. Engine-by-engine survey

Deep internals for Sjeng, Sunsetter, Fairy-Stockfish, CrazyAra and upstream Hivemind are in 06 §2; here each row
is reduced to the eight questions in the brief plus what is new in this pass.

| Engine | Second board | Time | Sitting / partner comms | Search | Evaluation | Training data | Strength claim | Source quality |
|---|---|---|---|---|---|---|---|---|
| **Hivemind it04 (stock, ours)** | One network sees both boards (74 planes), two policy heads, joint action per macro-ply | One flag (plane 31); `wtime/btime` ignored; no clocks in tree | Hand rules on the flag; no comms (single agent) | MCGS, CPUCT 3.0/19652, FPU, `Q_VALUE_WEIGHT 1.0`, Fairy mate probe, no epsilon-greedy, no enhance-checks | value + WDL + moves-left | chess.com >=2400/2200 SL, then RL it01-04 at 800 nodes | none published | code [L1]-[L7] |
| **Hivemind upstream v3 (twin-s-clock)** | same | one signed lead + per-team move time; lead/speed planes; no flag loss | `can_sit = lead > 0` per node | as above + frozen-race prover, early stop | same | self-play on sampled clocks, 100 k games/it | +156 Elo vs it0 at 100 ms, self-relative | 06 [5]-[20] |
| **HiveMind paper (2023)** | tree search "over both boards" | "clock times determine the turn sequence" | "without hand-crafted heuristics" | AlphaZero-style | unknown | human games then self-play | >95 % vs Fairy-Stockfish at 2 s | abstract only [2][4] |
| **CrazyAra / ClassicAra / MultiAra** (Czech et al., TU Darmstadt) | n/a (crazyhouse, one board) | chess-style time manager (06 §2.3) | n/a | MCTS -> MCGS; Q-init -1, enhance checks, epsilon-greedy, terminal solver, Q-values for move selection | value (+WDL/plys in later versions) | SL: 569,537 lichess games >=2000; RL: ~1 M games in 18 days on 3 V100 (45 games/min/GPU at 800 nodes, batch 8); replay 5 % newest, 409,600 replay samples per update; Q-value ratio 0.15; cosine LR with 25 % warm-up; value loss weight 0.5; no resign; tree reuse | RL +370 Elo over SL net (2,750-game tournament); 61-3-36 vs MV-Stockfish dev 2019-12-03 at 15+10 (CrazyAra 8 K nps GPU vs SF 6.7 M nps, 8 threads); MCGS paper: MCGS +110, epsilon-greedy +100, all combined +310 Elo in crazyhouse at 5 s/move (100 k-800 k nodes), +69 in chess; MultiAra RL: crazyhouse +480 Elo over SL in 60 updates, ~17 days on 4 V100; Chess960 RL failed to beat its SL net; KOTH from zero needed 20 updates to match SL | papers and theses read [21][22][23][24] |
| **CrazyAraFish** | n/a | as above | n/a | as above | as above | 121,571 MV-Stockfish 10 self-play games at ~1 M nodes/move, 1,000 openings; 46.3 % -> 56.6 % val accuracy after 11 h on a 1080 Ti | 3-1-6 vs MV-SF (2018-11-29) at 30+30 | wiki + thesis [25][22] |
| **Fairy-Stockfish** (ianfab) | one board; holdings via XBoard `holding`; "virtual drops" of pieces that might arrive | own and partner's clocks via `time/otim` ptell; sit only while `time[us]-elapsed > Partner.opptime` | `partner.cpp` protocol: sit/go/move/fast/slow/dead/x/time/otim; sits when dead; tells partner to sit when winning by force and ahead on time | alpha-beta; classical or NNUE | classical, or NNUE nets: **bughouse-cd8cceab93fe.nnue +403 Elo vs classical (2023-01-24)**, crazyhouse-8ebf84784ad2.nnue +1136 (2022-09-05), both by "belzedar_ in discord" | NNUE training via `variant-nnue-pytorch` | crazyhouse classical -84 +-15.5 Elo vs MV-Stockfish on variantfishtest; "strong NNUE networks ... surpass all other engines"; no bughouse rating anywhere | code and wiki [14][15]; NNUE gains are self-reported single tests **[weak]** |
| **Multi-Variant Stockfish** (ddugovic) | n/a (crazyhouse) | Stockfish time manager | n/a | alpha-beta | classical, crazyhouse-tuned | n/a | "strongest classical engine for crazyhouse" in 2019 (Czech); +89 Elo 2018-11 -> 2019-12 (one user's 1+1 test) | README fetch 404; thesis [22] |
| **Sjeng 11.2** | one board; `partner.c` ptell parser | `go_fast` caps move time at 40 cs; no clocks in `partner.c` | `must_sit` when score in (-60000,-40000) and partner alive; `CheckBadFlow` virtual-drop mate scan -> "---p" messages; piece-value requests `+n`, `++q` | alpha-beta; bughouse tweaks (null move, aspiration, futility) | P100 N210 B230 R250 Q450; hand-indexed king safety | none (hand-tuned) | ~2000 FICS bughouse, top >2200 (self-report, undated) | code read in 06 [06: 25-30] |
| **Sunsetter** | one board | `initialTime/360 + mytime/360`, floors 40 ms / 5 ms; no pondering in bughouse | "sit (I am in trouble)", "go (watch our time)" after partner sits 5 moves | alpha-beta with extensions | P100 N192 B195 R200 Q390 + hand bonuses, king-safety multiplier | none | crazyhouse 2722 FICS record (2002); no bughouse figure | code read in 06 [06: 42-48] |
| **Bug Bully** (pawndini/bughouse-chess, fork of Ellipsoul) | one board at a time; "partner-capture lookahead": conditional lines if an immediate legal capture on the other board could supply a piece | clocks replay the recorded timeline, frozen in variations | none (analysis tool, not a player) | Fairy-Stockfish WASM, 3 PV at depth 11; supply searches capped at 750 ms | crazyhouse NNUE | chess.com games via published API + undocumented `/callback/live/game/{id}` | none claimed | README [16] |
| **chessdotcom-bughouse-engine** (RobbieGM) | one board (Tampermonkey script + local Fairy-Stockfish) | "time is critical" | none | Fairy-Stockfish | n/a | n/a | "performance rating of around 3800", no games | README **[weak]** (06 [55]) |
| **bug0-cs229** (Ardeshna, Selwyn, Singhal, Stanford CS229, 2021) | single agent for both boards; **imposed turn order** T1-A(W) -> T2-B(W) -> T1-B(B) -> T2-A(B); 60x8x8 planes (two boards x 29 + player + active board), 2272-move compressed policy | none ("simplifying assumptions about the order in which turns are taken") | none | AlphaZero (alpha-zero-general base), 5-25 simulations/move | value from self-play or regression | FICS BPGN from bughouse-db.org; RL: 10 games/checkpoint, batch 64, lr 0.002, 24 h on one GPU | RL agent lost to random (1-19); SL AlphaZero net beat random 19-1 and minimax (depth 1) 4-16 after 8 h; conclusion: self-play without compute overfits, SL on the archive works; future work: two agents per team "may allow the Bughouse timing problem to be more easily addressed" | writeup read [17] |
| **BughouseAlphaZero** (MbProg) | "Deep Convolutional Recurrent NN" + async MCTS; details absent | not described | not described | MCTS | n/a | supervised only; `filtered_dataset_small.csv`, source unnamed | none (says "tested against other engines", no numbers) | README only **[weak]** [18] |
| **Zinkelburger/hivemind** | fork of aminwoo/hivemind, 0 stars, README identical in substance | — | — | — | — | — | — | [19] |
| **CCC 2016 crazyhouse field** (Imortal, Sunsetter 7g, TJChess, Sjeng, Pulsar, KKFChess, Nebiyu, Sjaak II), CrazyWa | crazyhouse only | engine-specific | n/a | alpha-beta (Nebiyu: MCTS option) | hand-crafted | none | tournament results only | 06 [49][52] |
| **Lichess crazyhouse bots** | n/a | n/a | n/a | mostly Fairy-Stockfish ("the Lichess variant bot is nothing else than Fairy-Stockfish", forum) | NNUE | n/a | CrazyAra ~2600 (2018); ToromBot 2753 (unconfirmed BOT) | lichess unreachable from this machine (`ECONNREFUSED`); forum snippets **[weak]** [20] |
| **FICS "bots"** | XBoard's ICS ("zippy") mode drives Sjeng, Sunsetter or Fairy-Stockfish (`-zippyVariants bughouse -zippyBughouse 2`); FICS marks computers "(C)", forbids rated games with the operator or with the operator's own computer as partner | per engine | per engine (`ptell`) | per engine | per engine | — | — | FICS help pages, Fairy wiki [27][06: 41] |
| **TinyChessBot** | **not found.** Searches return Sebastian Lague's "Tiny Chess Bots" (no bughouse) and `tiny-chess-duck`; no bughouse engine of that name exists in the indexed web | — | — | — | — | — | — | [28] |
| **pychess.org / bughouse.pro** | pychess added bughouse (2023, "Bughouse chess has arrived") and uses Fairy-Stockfish NNUE for analysis; bughouse.pro is an open-source server in beta, no engine found; its rule requests to Fairy-Stockfish (#811) are open | — | — | — | — | — | — | snippets **[weak]** [29][06: 40] |

**Patterns worth stating plainly.**

1. Three architectures exist for the second board: text-protocol partner (classical engines, Fairy), joint single
   agent (Hivemind, bug0, BughouseAlphaZero), and single-board-plus-lookahead (Bug Bully's "supply search", Fairy's
   virtual drops, Sjeng's `CheckBadFlow`). The third is a cheap approximation of our `arrivals` head: "could a
   piece that mates arrive in one capture". Nobody learned it; everyone hand-codes it.
2. Every engine that plays in real time keeps the diagonal comparison (`time[us] - elapsed > Partner.opptime`) and
   never the absolute clock, except for Sunsetter's think-time floors. Flagging as an outcome exists nowhere in a
   search tree.
3. Every learned engine imposed a lockstep turn order for tractability (bug0 explicitly; it04 by alternating teams;
   upstream by fixed move times). Out-of-step training is untested in the literature.
4. Fairy-Stockfish is the universal baseline (HiveMind paper, pychess, lichess bots, chess.com bridges). Its bughouse
   NNUE (+403) exists since 2023 and is almost certainly stronger than whatever the paper beat in 2023 (the net is
   dated 2023-01-24; the paper's experiments were probably earlier). Any reproduction must state which net.
5. Strength claims are self-relative or against Fairy; no bughouse engine has a verified rating against humans.

---

## 4. Datasets

| Dataset | Content | Clocks | Size | Access | Use for us | Caveats |
|---|---|---|---|---|---|---|
| **chess.com bughouse games** (it04's SL corpus) | both boards per game, TCN moves, `moveTimestamps` per move, `baseTime1`, ratings, `partnerGameId` | **yes**, per move, per board | unknown total; filter players >=2400 and partners >=2200, >=10 moves | published API `/pub/player/{p}/games/archives` for discovery + undocumented `/callback/live/game/{id}` for board detail (also used by Bug Bully) | fill v1 planes 37-40 and v2 planes 41-42 from the stored timestamps; keep the behind-on-time sits that `generate_planes.py` drops; human pacing distribution | callback endpoint undocumented and may vanish (Bug Bully README); API forum says PGN export does not work for bughouse; terms-of-use not checked here [L5][16][30] |
| **FICS bughouse (bughouse-db.org)** | BPGN per year, `1A./1a./1B./1b.` labels, brace clocks per move, `Lag` tag, `TimeControl` (sample 120+0); ratings in tags | **yes** (remaining time in braces) | 2005: 385,012; 2006: 346,213; 2007: 307,383; 2008: 205,253; 2009: 187,290; 2010: 358,155; 2011: 346,244; 2012: 391,270; 2013: 341,670; 2014: 283,312; 2015: 269,758; 2016: 151,114; 2017: 104,798; 2018: 14,881; 2019: 12,547; 2020: 12,995; 2021: 2,756; 2022: 265; 2023: 37; 2024: 54; 2025: 255 (about 3.7 M total) | `http://www.bughouse-db.org/dl/export<year>.bpgn.bz2` (built by JamesBaud, run by Ludens); ficsgames.org excludes bughouse and points here | second human corpus with clocks and real lag; 2+0 and 3+0 controls match the gate's 120 s/180 s; a `Lag` term for the additive-jitter idea (01 §3 row 6) | FICS pool is weaker and older than chess.com 2400+; rating filter needed; BPGN parser to write (python-chess-bughouse exists on PyPI, unvetted); whether the archive since 2018 is complete is unknown [23][24][17][01: 35] |
| **lichess crazyhouse** | single-board crazyhouse PGN, monthly `.pgn.zst` | lichess PGNs normally carry `[%clk]`; **not verified on the page** this pass | 31,049,289 rated games, 2016-01 to 2026-10 | `database.lichess.org/crazyhouse/lichess_db_crazyhouse_rated_YYYY-MM.pgn.zst` | single-board pretraining of drop tactics / value (CrazyAra's recipe); not bughouse: no partner, no pocket inflow | would need a plane-layout bridge to our two-board input; crazyhouse pockets differ (captures go to the capturer) [26] |
| **CrazyAra Stockfish self-play** | 121,571 MV-SF 10 crazyhouse games at ~1 M nodes/move | no | 121 k | CrazyAra wiki links | precedent for engine-generated supervised data; not bughouse | [25] |
| **Hivemind HVM5 self-play** (ours) | planes, both policies, joint policy, outcome, moves-left; our recorder adds clocks, think-gain and arrivals labels | yes (our shards) | as generated | `arena/selfplay.py`, `convert_selfplay_data.py` | the RL corpus | out-of-step distribution is ours alone; nothing external to compare with [spec v1 §3, v2] |

---

## 5. Solutions

### 5.0 Cost assumptions

- Engineering: one "day" is one focused engineer-day including tests; ranges reflect unknowns in the code.
- GPU: one GPU-hour means one modern data-centre or RTX-4070-class GPU. Self-play throughput assumption: ~20 bughouse
  games/min/GPU at 800 nodes with the 13.6 M net (CrazyAra measured 45 crazyhouse games/min on a V100 at 800 nodes,
  batch 8, TensorRT off; bughouse has two boards and a bigger net). So 100 k games is roughly 80-100 GPU-h; a
  1,600-game arena row at 50-200 nodes is well under 1 GPU-h of inference but is wall-clock bound.
- "Spec" column: v1 = `docs/custom_engine_spec.md`, v2 = `docs/custom_engine_spec_v2.md`, 07 = `07_research...` §3.

### 5.1 Ranked table

| Rank | Solution (source) | Concrete change to our engine | Spec covers it? | Expected benefit | Cost | Risk |
|---|---|---|---|---|---|---|
| 1 | **Reproduce the paper's benchmark: Fairy-Stockfish bughouse as the external floor** (HiveMind abstract; Fairy `partner.cpp`; NNUE page) | Arena arm: two Fairy-Stockfish instances under XBoard protocol (`holding`, `partner`, `ptell`, `time/otim`) with `bughouse-cd8cceab93fe.nnue`, 2 s/move; adapter in `arena/players/`; run stock it04 and every candidate net against it at 60 s and 180 s; report win % with the pair-bootstrap CI | No (gate is self-play vs stock only) | The only published cross-engine number becomes reproducible; an opponent that is not us (sanity floor against self-play collapse); a number outsiders recognise | 3-5 days; ~20-50 GPU-h per full table | Fairy's bughouse layer is crude: a floor, not a ceiling; XBoard timing semantics must be mapped onto arena events carefully (its `sit` busy-waits) |
| 2 | **Warm-start clock planes and the pass action on chess.com clocks we already hold** (06 §3 item 12; `generate_planes.py`) | Extend `generate_planes.py` to fill planes 37-42 from `moveTimestamps`; stop dropping samples where the behind-on-time team passed; one short SL pass (trunk at low LR, new weights at full LR) before RL iteration 1; parity check on it04 outputs | Partly (v1 widens with zeros; nothing warm-starts) | RL starts from human sitting-under-clocks behaviour instead of zero clock weights; CrazyAra, MultiAra and bug0 all found SL-then-RL works and RL-from-zero is slow (20 updates to parity in KOTH) | 3-5 days; 10-30 GPU-h | Human sits at chess.com 2400 are a biased sample (censoring by result); label definition of a "pass" in human data is approximate; must not undo it04's policy (measure accuracy on the SL val shard before/after) |
| 3 | **Add the FICS BPGN corpus** (bughouse-db.org; bug0 used it) | BPGN reader (braces -> per-move clocks, `Lag`, ratings) into the same parquet schema as the chess.com reader; rating filter; dedupe; pool with (2) | No | ~3.7 M games with clocks and lag, 2+0/3+0, the controls the gate reports; more varied pacing; `Lag` gives the additive jitter term from 01 | 4-6 days; 10-20 GPU-h | Weaker pool; archive quality unknown since 2018; licence/terms of the archive unstated; parser edge cases (BPGN has no formal spec reachable) |
| 4 | **Turn on Q-value blending and replay memory** (Czech thesis §5.2: ratio 0.15 "lowered value loss and reduced convergence to +-1"; replay 5 % newest; both exist in our code, off) | `--q-value-ratio 0.15` in the RL loop; `--replay-dir` from iteration 2 on; fix README/code disagreement (5 vs 25 chunks) and decide | No | Smoother value targets, which matters more for us: flag outcomes make `z` noisier; replay fights forgetting during the out-of-step shift | 0.5-1 day; 0 GPU | Q blending with a frozen-clock root Q is slightly inconsistent in the first iteration; measure value loss on held-out shards |
| 5 | **CrazyAra's cheap search enhancements as SPRT candidates** (MCGS paper: epsilon-greedy ~+100, enhance checks, terminal solver, Q-values for move selection; combined +310 Elo in crazyhouse at 5 s) | Add `epsilon-greedy` leaf selection and `enhance checks` (prior < 0.1 checking moves get +0.5 max P) as UCI options off by default; MCGS and `Q_VALUE_WEIGHT` already exist; run each as a paired-opening SPRT at our node budgets | No | Up to tens of Elo each at low cost; checks matter more in bughouse than in chess | 2-4 days total; ~100-200 GPU-h for four SPRTs | The paper's gains were at 100 k-800 k nodes/move; at 50-800 nodes they may vanish or reverse; epsilon-greedy adds variance to self-play |
| 6 | **Time-manager baselines before trusting `think_gain`** (CrazyAra §9.2.4; Sunsetter floors; 07 item 6) | Arena controller options: stop at half budget if top prior > 0.9 and Q stable; extend by half on a 0.1 Q drop; hard floor 40 ms under 4 s and 5 ms under 0.8 s; compare to the v2 head at equal clock | Partly (v2 head; 06 item 6 names it) | A free baseline the head must beat; protects against flagging by one bad budget | 1-2 days; ~20 GPU-h | None beyond measurement cost |
| 7 | **Gate at equal wall-clock and keep distillation open** (upstream `twin-s`: 0.97 M net at 3x nps beat the 13.6 M teacher at 100 ms) | `beat_stock.md`: add a fixed-time mode (ms per move on identical hardware, nodes-as-time inside training); decide the headline metric now | Partly (`beat_stock.md` is fixed nodes) | Avoids declaring a win that disappears at real clocks; keeps the option of a small clock net that thinks faster, which is itself a time-management advantage | 2-3 days for the mode; distillation 5-8 days + 200-400 GPU-h if chosen | Hardware-dependence of wall-clock results; the cluster's shared GPUs leak contention into fixed-time runs (07 item 11) |
| 8 | **XBoard/FICS shim so the engine can get a human-relative rating** (Fairy `partner.cpp`, Sjeng `partner.c`, FICS computer-account rules) | Thin CECP layer mapping `holding`, `time/otim`, `partner`, `ptell` and the pass to our `position ... clocks ... elapsed` and `bestmove`; a registered FICS computer account partnered with a copy of itself (unrated with the operator, as FICS requires) or on bughouse.pro | No (spec is UCI only) | The one number no bughouse engine has: a rating against humans; also lets humans partner the engine later (06 scope note) | 3-5 days; 0 GPU | FICS bughouse pool is tiny now (265 archived games in 2022); bughouse.pro rules differ (#811); server policies on engines |
| 9 | **"Could a mating piece arrive in one capture" as an auxiliary label and sanity probe** (Sjeng `CheckBadFlow`, Fairy virtual drops, Bug Bully supply search) | In the recorder, compute per sample: for each missing piece type, does a legal capture on the partner board exist and would a drop of that type mate; store as 5 bits per side; train as a small BCE aux head (weight 0.05) or use only as an eval probe of the `arrivals` head | Partly (v2 `arrivals` counts arrivals in 5 s; this is the mate-relevance of one arrival) | Direct supervision of the one tactic every classical engine hand-codes; a check that the arrivals head learned the right thing | 2-3 days; negligible GPU | Label is a hand rule and may bias; keep it as a probe first |
| 10 | **Equal-eval opening suite for gating** (Czech used ianfab's `crazyhouse_mix_cp_130.epd`, 691 positions within +-130 cp; upstream uses 100 openings) | Score `arena/openings.py` lines with it04 at 800 nodes; keep lines with |Q| small; add drop-heavy openings from human data | Partly (paired openings exist) | Lower variance per game pair; fewer games per SPRT | 1-2 days; ~5 GPU-h | Selection by it04's own eval biases the suite toward it04's taste |
| 11 | **Test the paper's turn-ordering idea as an option** (abstract; 07 item 6 durative tree) | UCI option: when both boards are on turn, expand the board whose side to move has the larger elapsed time first; compare to `TreeMoveMs` lockstep | Partly (07 item 6 proposes the full durative tree) | Cheap first step toward asynchrony in the tree | 2-3 days; ~50 GPU-h | Unknown benefit; the paper's exact rule is unknown |
| 12 | **Data hygiene from CrazyAra/MultiAra** (20 players = 46 % of CrazyAra's games; MultiAra: mirroring hurt in racing kings) | Report player concentration in the chess.com corpus; cap games per player; verify the board-swap augmentation also swaps clocks and elapsed planes (otherwise the clock warm-start learns a mirrored lie) | No | Avoids overfitting a handful of strong humans' styles; prevents a silent augmentation bug when clocks are added | 0.5-1 day | None |
| 13 | **Fine-tuning learning rate** (CrazyAra SL 0.35 -> RL 0.005; 07 item 18) | First RL iteration with trunk LR <= 1e-4 and new heads at full rate | 07 covers | Protects it04 parity | 0.5 day | None |
| 14 | **Policy-only move when time is critical** (07 items 4-5; bug0's lesson that 5-simulation MCTS is worse than random) | Budget 0 (raw policy) in the budget set; never run MCTS below ~16 nodes | 07 covers | Avoids the pathological low-node regime | in 07's costs | — |
| 15 | **Sjeng-style piece-value requests / partner messages** | Not applicable: single agent, no partner (spec scope) | Out of scope by design | — | — | Revisit only for human-partner play (solution 8) |

### 5.2 Notes on the top eight

1. *Benchmark reproduction* is first because every strength claim in this project is currently self-relative. The
   paper's 95 % is the only external anchor; if stock it04 does not beat Fairy-Stockfish + bughouse NNUE at 2 s
   by a wide margin, the whole "beat stock" framing needs rethinking. Use the 2023 NNUE net and say so; the paper
   almost certainly used classical eval.
2. *Warm-start* is the cheapest way to make the first out-of-step RL iteration start from something: the data is on
   disk, the reader already orders moves by timestamps, and the censoring of behind-on-time sits is a one-line
   filter. Keep an SL validation shard and report policy accuracy and value loss before and after.
3. *FICS corpus* is the only second source; its decline since 2018 means it is historical data, but 2005-2017 is
   3.5 M games at the time controls we gate on. Write the parser defensively (no reachable spec).
4. *Q blending and replay* are flags. The only decision is the replay count (README 5 vs code 25); CrazyAra used
   409,600 samples from 5 % of the newest memory per update, which at 4,096-sample shards is 100 shards; pick by
   fraction of fresh data (CrazyAra: one third replay).
5. *Search enhancements* must be measured at our budgets, not assumed; the MCGS paper's hardware searched up to
   800 k nodes a move. Run them as independent SPRTs and expect some to fail.
6. *Time-manager baselines* cost almost nothing and define what "the head helps" means.
7. *Equal wall-clock gating* is a decision, not code: write it into `beat_stock.md` before the first gate.
8. *XBoard shim* is how this engine ever gets a number humans believe. Three to five days.

### 5.3 What the survey says not to do

- Do not add a partner protocol or messages to the single-agent engine (spec scope; 06 shows every protocol is a
  workaround for two programs that cannot see each other's board).
- Do not expect RL from a cold start to work: bug0's self-play agent lost to random play; MultiAra's from-zero run
  needed 20 updates to match its SL net; Chess960 RL never beat its SL net at all.
- Do not treat "95 % vs Fairy-Stockfish" as strength relative to humans; nobody has that number.
- Do not reuse upstream's single-lead parameterisation as a replacement for absolute clocks: it cannot express
  flagging or how much thinking a move can afford (06 §2.8). Adding it as two extra planes is fine (06 item 1).

---

## 6. Sources

Web (fetched and read unless marked snippet/summary):

- [1] https://ieeexplore.ieee.org/document/10333210 — IEEE Xplore record (DOI 10.1109/CoG57401.2023.10333210); page body not retrievable here; abstract text seen via search snippet. PDF link `https://ieeexplore.ieee.org/iel7/10333091/10333129/10333210.pdf` is paywalled, not downloaded.
- [2] https://api.openalex.org/works/doi:10.1109/CoG57401.2023.10333210 — closed access, no repository full text, 11 referenced works, 1 citation, authorships and raw affiliations, reconstructed abstract.
- [3] https://api.semanticscholar.org/graph/v1/paper/DOI:10.1109/CoG57401.2023.10333210?fields=... — `openAccessPdf.status CLOSED`, 16 references, 1 citation, DBLP key `conf/cig/WooSA23`.
- [4] Search-engine snippet of the IEEE abstract — the "incorporating domain knowledge, using clock times to determine the optimal turn sequence" and "win rate of over 95% with a search time of 2 seconds" phrases.
- [5] https://researchr.org/publication/WooSA23 and dblp `conf/cig/WooSA23` (dblp page blocked by Anubis) — venue, pages 1-8.
- [6] https://hmi.anu.edu.au/people-hmi/penny-kyburz — Sweetser now listed as Kyburz at ANU (snippet).
- [7] https://api.semanticscholar.org/graph/v1/paper/DOI:10.1109/CoG57401.2023.10333210/references — "references elided by the publisher".
- [8] https://api.semanticscholar.org/graph/v1/paper/DOI:10.1109/CoG57401.2023.10333210/citations — the one citing paper (Leisiazar et al., RA-L 2025).
- [9] https://github.com/aminwoo — profile: Australia, no bio/site, hivemind pinned.
- [10] https://api.github.com/repos/aminwoo/hivemind and `/releases?per_page=60` — created 2023-02-09, MIT, 7 stars; full release list v0.2.0-alpha (2024-10-30) .. v3.0.3 (2026-10-23) with notes.
- [11] https://github.com/aminwoo/hivemind/releases — v3.0.0 notes (clock model, +156 Elo, -O3), read in 06 [14].
- [12] https://huggingface.co/aminwoo/bughouse-rise-v3 (page and `raw/main/README.md`) — model card; front matter `license: mit`; "MIT-licensed, matching the Hivemind repository"; architecture, files, checkpoint metadata; no data/Elo/clock statements. (Fetch tool returned a summary, not verbatim text.)
- [13] https://github.com/aminwoo/hivemind — current README/LICENSE (MIT); v1.2.1 README GPL-3.0 per 06 [17].
- [14] https://fairy-stockfish.github.io/nnue/ — NNUE list: `bughouse-cd8cceab93fe.nnue` +403 Elo (2023-01-24), `crazyhouse-8ebf84784ad2.nnue` +1136 (2022-09-05), author "belzedar_ in discord"; gains are vs the built-in classical eval.
- [15] https://github.com/fairy-stockfish/Fairy-Stockfish/wiki/Playing-strength — crazyhouse classical -83.94 +-15.5 vs MV-Stockfish (variantfishtest); NNUE "surpass all other engines"; no bughouse entry.
- [16] https://github.com/pawndini/bughouse-chess — Bug Bully: Fairy-Stockfish WASM + crazyhouse NNUE, partner-capture lookahead, chess.com `/callback/live/game/{id}` dependence, no strength claim.
- [17] https://github.com/dardeshna/bug0-cs229 and `raw/.../writeup.pdf` (text extracted locally) — Stanford CS229 "Bug0": imposed turn order, 60-plane input, 2272-move policy, FICS BPGN data, RL vs SL results, Tables I-IV.
- [18] https://github.com/MbProg/BughouseAlphaZero — DCRNN + async MCTS, supervised only, tinyChessServer, no results **[weak]**.
- [19] https://github.com/Zinkelburger/hivemind — fork of aminwoo/hivemind, 0 stars.
- [20] https://talkchess.com/viewtopic.php?p=936518 and neighbours ("Crazyhouse status?") — "the Lichess variant bot is nothing else than Fairy-Stockfish"; NNUE "1136 Elo above classic" **[weak]**. `https://lichess.org/api/bot/online` unreachable (ECONNREFUSED).
- [21] https://arxiv.org/abs/1908.06660 — CrazyAra paper (details in 06 [21][22]).
- [22] https://ml-research.github.io/papers/czech2019deep.pdf — Czech, "Deep Reinforcement Learning for Crazyhouse", M.Sc. TU Darmstadt 2019 (58 pp., text extracted locally): §5.2 RL setting (3 V100, 45 games/min/GPU, 800 rollouts batch 8, 819,200 samples per update then 1,228,800, 409,600 replay samples from newest 5 %, Q-value ratio 0.15, value loss 0.5, cosine LR 25 % warm-up, no resign, tree reuse); §6.1 +370 Elo over SL after 973,044 games / 18 days, 2,750-game tournament; §6.3 Table 6.2 61-3-36 vs MV-SF dev at 15+10, hardware; §7.2 future work (time management "less time on obvious and more on complicated positions", from-zero replication, knowledge sharing between subtrees).
- [23] http://www.bughouse-db.org/ and `/dl` — FICS bughouse database (built by JamesBaud, run by Ludens), games since Feb 2005, yearly `export<year>.bpgn.bz2` with sizes and game counts (table in §4); `/dl/README.html` only says bzip2.
- [24] https://www.ficsgames.org/about.html — GameBot excludes bughouse; "Bughouse games are in the Bughouse database" -> bughouse-db.org; move times for all FICS games since Nov 2008.
- [25] https://github.com/QueensGambit/CrazyAra/wiki/Stockfish-10:-Crazyhouse-Self-Play — 121,571 MV-SF 10 games at ~1 M nodes, 1,000 openings, 46.29 % -> 56.6 % val accuracy after 11 h on a 1080 Ti.
- [26] https://database.lichess.org/ — crazyhouse 31,049,289 rated games, monthly `.pgn.zst` from 2016-01; the page text read did not mention `[%clk]` (unverified).
- [27] https://www.freechess.org/Help/HelpFiles/computers.html (and bughouse-db mirror) — computer accounts "(C)", no rated games vs operator or own computer as bughouse partner (snippet).
- [28] Searches for "TinyChessBot bughouse" — only Sebastian Lague's "Tiny Chess Bots" and `m0hossam/tiny-chess-duck`; no bughouse engine of that name.
- [29] https://www.pychess.org/blogs/@/gbtami/fairy-stockfish-on-pychess/L9dd2a6d and lichess ublog on bughouse.pro — pychess uses Fairy-Stockfish NNUE for analysis; bughouse.pro "open-source website ... in beta" (snippets) **[weak]**.
- [30] https://www.chess.com/clubs/forum/view/bughouse-fetch-partner-game-using-chess-com-api — forum: API has no bughouse PGN export; partner game fetch asked, no official answer (snippet) **[weak]**.
- [31] https://arxiv.org/abs/2012.11045 and PDF (text extracted locally) — Czech, Korus, Kersting, "Monte-Carlo Graph Search for AlphaZero" (ICAPS 2021): crazyhouse at 5 s/move on an RTX 2070 (100 k-800 k nodes): MCGS ~+110, epsilon-greedy ~+100, combined ~+310 Elo; chess +69; 691 equal openings from `ianfab/books crazyhouse_mix_cp_130.epd`; 30 % memory reduction; model trained on ~0.5 M lichess games then +436 +-30 Elo over 2.37 M self-play games.
- [32] https://ml-research.github.io/papers/gehrke2021assessing.pdf — Gehrke, "Assessing Popular Chess Variants Using Deep RL" (MultiAra), M.Sc. 2021 (text extracted locally): 2 DGX-2 (V100), SL thresholds (crazyhouse >=2000), RL 50-60 updates, 0.5-1.2 M games per variant, ~17 days on 4 GPUs, gains over SL: crazyhouse +480, KOTH +690, 3check +400, antichess +340, horde +300; Chess960 RL never beat SL; KOTH from zero matched SL after 20 updates; mirroring off helped racing kings.
- [33] https://arxiv.org/abs/2304.14918 — Czech, Blüml, Kersting, Steingrimsson, "Representation Matters for Mastering Chess" (v2 2024): input-representation plus value-loss changes up to +180 Elo, more than switching to transformer hybrids (~+30). Abstract only.
- [34] https://raw.githubusercontent.com/QueensGambit/CrazyAra/master/DeepCrazyhouse/configs/rl_config.py — CrazyAra RL defaults: 800 nodes, temperature 0.8 for 15 moves, Dirichlet alpha 0.3 eps 0.25, `rm_nb_files 5`, `rm_fraction_for_selection 0.05`, `arena_games 100`, resign prob 0.9, `nn_update_files 10`, 640x128 chunking.
- [35] https://github.com/QueensGambit/CrazyAra/wiki/Input-representation — 34-plane crazyhouse input, no history, no clock planes; motivation text.
- [36] https://raw.githubusercontent.com/QueensGambit/CrazyAra/master/README.md — papers/theses list, lichess accounts, GPLv3.

Local (read in the repo):

- [L1] `third_party/hivemind/engine/src/tools/selfplay.cc` (~657, ~704) — lockstep team turns, flag toggled, no clocks (verified by the project before this pass).
- [L2] `third_party/hivemind/engine/src/environment/joint_action.h:19-38` — `is_double_sit_legal`, `is_single_pass_legal`, `g_freePass` short-circuit.
- [L3] `third_party/hivemind/engine/src/search/agent.cc:185` — `if (!rootTeamHasTimeAdvantage)` gates probe-target collection (probe runs for the team behind on time).
- [L4] `third_party/hivemind/LICENSE` — MIT, "Copyright (c) 2026 aminwoo".
- [L5] `third_party/hivemind/src/hivemind/data/scrape_cc.py` (54, 105-174), `game_reader.py` (37-52) — chess.com endpoints, filters, `moveTimestamps`, `partnerGameId`, `baseTime1`, move ordering by timestamps.
- [L6] `third_party/hivemind/README.md` — planes, options, self-play and RL defaults, replay description ("five archived HVM chunks").
- [L7] `third_party/hivemind/src/hivemind/data/convert_selfplay_data.py:77-79` (`DEFAULT_REPLAY_FILES = 25`, `DEFAULT_REPLAY_SELECTION_FRACTION = 0.05`, `DEFAULT_Q_VALUE_RATIO = 0.0`), `src/hivemind/config/train_config.py:143-145, 238`, `training/train_loop.py:650-708`; `engine/src/search/search_params.h` (`CPUCT_INIT 3.0`, `Q_VALUE_WEIGHT 1.0`, no epsilon-greedy or enhance-checks symbols).
- [L8] `docs/custom_engine_spec.md`, `docs/custom_engine_spec_v2.md`, `docs/beat_stock.md` — the contracts the solutions are checked against.
- [L9] `docs/theory/06_engines_and_ai.md`, `07_research_time_and_async.md`, `01_rules_and_protocols.md` (BPGN sample [35], `Lag` tag), `99_gaps.md` (C3, C4) — cross-references; not edited.
