# 05. Openings and team strategy

Reference notes for the custom engine (one agent, both boards, real clocks; `docs/custom_engine_spec.md`,
`docs/custom_engine_spec_v2.md`, `docs/beat_stock.md`). Written 2026-11-17 from primary sources where they could be
read in full. **Access caveat:** chess.com, lichess.org and uschess.org could not be fetched from this machine (the
browser and the fetch tool both got error pages; the archive route was refused). The chess.com blog series by Sorsi,
JarlCarlander and chriswsh is the largest body of written bughouse theory, so where I cite them it is from search
snippets only, marked **[snippet]**. Treat every such claim as single-source until someone reads the post.

---

## 1. Summary

- **Roles are set by colour.** At high level White attacks and keeps pieces on; Black defends, trades and holds. The
  team's plan is "White mates, Black survives long enough" [2][5][15][snippet: MidnightFox 2024].
- **Time is diagonal.** What matters is your clock against your *partner's opponent's* clock, not your own opponent's.
  A sit only works while you have more time than the diagonal opponent; two diagonal sits ("sitzkrieg") are decided by
  that difference [1][2][5][14]. Quoted thresholds: a 10 s edge beats a positional edge at 2000+ [1]; a 2 s lead is
  "often decisive in an equal match" [snippet: Sorsi]; a 10 s double sit is "basically resign" [snippet: Sorsi]; under
  20 s "it's time to move" [12]; "sit until 10, then low flow" is a named plan [snippet: chriswsh].
- **Trades are a flow decision, not a material decision.** Every capture arms the diagonal opponent; every trade
  disarms your own board. Trade when your partner attacks, hold when your partner defends, and "whoever controls the
  flow controls the game" [1][12][13][16][snippet: Sorsi]. Light-but-steady flow beats dumping pieces [snippet].
- **Captures have a tempo.** "Make sure trades come on your tempo" [13]: the value of a capture depends on which side
  is on move on the other board when the piece lands. This is exactly what lockstep play hides and out-of-step play
  exposes.
- **Only contact, knight and double checks are real.** Any other check can be answered by a drop that may not even
  exist yet [3][4][7]. Attacks are built from those checks; "as long as he's in check, I'm winning" [2].
- **Openings are few and pawn-structural.** Move only the d- and e-pawns; never ...c5 (c7 hole), never fianchetto
  (g7/g2 hole), avoid f-pawn moves; castling is often skipped [1][2][12][15]. The standard Black setups are the
  Fortress 1.e4 e6 2.d4 d6 ("by far the most common"), the e6–Nf6 and e6–Nc6 systems, 1.e4 d5 2.exd5 Qxd5 3.Nc3 Qd8,
  and the Alekhine sacrifice 1.e4 Nf6 2.e5 d5 [2][13][15][snippet].
- **Piece values are flat and disputed.** Nilsson: P1 N2 B2 R2 Q4 [1]; FICS strat file, 1995 mirror: P1 B2 R4 N7 Q10 [2] (Corrected: the live freechess.org revision of the same file now says Q4 N2 R2 B2 P1);
  Sunsetter: P100 N192 B195 R200 Q390 [1]; "two pawns for a piece is a good deal" [12]. Everyone agrees pawns and
  knights are undervalued by chess values and long-range pieces overvalued.
- **Communication exists to transmit three facts:** what I need, what the diagonal opponent must not get ("p mates
  me"), and sit/go. Sjeng encodes exactly this: `+p/++p/+++p`, `-p/--p/---p`, `sit`, `go`, and an automatic
  1-ply "does giving them piece X mate me" probe [26]. One controller needs none of the messages but must compute all
  three facts.
- **Engines are not yet clearly better than humans at bughouse** (forum opinion [snippet]); the known engine gaps are
  sitting to starve, timing trades to tempo, and "strategical attacks" (Sunsetter's own admission [31]).

---

## 2. The knowledge

### 2.1 Why colour decides the roles

- White moves first, so the first drop threats come on the White boards; the White player "goes for mate" while the
  Black player "tries to hold it together" by exchanging [2][5]. Panickssery, summarising top play: "At high levels,
  white plays offense while black defends and trades" [15]. The 2024 Bughouse World Championship write-up describes
  the classic approach the same way **[snippet: MidnightFox]**.
- Consequence for trades: trades on the Black board feed the White board's attack; trades on the White board feed the
  opponents' White attacker. So "Black trades, White doesn't" is the default flow and both boards must deviate from
  it together when the roles flip [2][12].
- The roles are a default, not a law. Jerry (ABCs): "Defense is more important than attack"; attack only when mate is
  realistic or when it buys the partner time [12]. The bugmisc author: "Always defend before attacking unless you have
  mate" [13]. Gulamali (Chess Life 2012, **snippet**): the skill is intuition about "when to sack, when to defend, and
  when to tell a partner to stop moving".
- "Tournament Game strategy" (credited to a player called Ron [13]): when one board has both a time edge and the
  better game, the *other* board sits and the winning board plays it out. This is the clearest statement of asymmetric
  think-time allocation: one board gets all the clock, the other board's job is to not lose and not feed.

### 2.2 Time and sitting

Definitions. *Sitting* = not moving while on turn, to deny the diagonal opponent a piece or to wait for one [1][2].
*Double sit* = both partners thinking at once [snippet: Sorsi]. *Sitzkrieg* = both diagonal opponents sitting; the
one with less time on the diagonal is forced to move [1][5]. *Sacsitting* = sacrificing and then sitting [12].

Rules of thumb with numbers:

| Claim | Source | Strength |
|---|---|---|
| The diagonal time difference matters more than the difference on your own board | [1][2][14] | three sources |
| At 2000+ a 10-second edge can outweigh positional advantages | [1] | Wikipedia, uncited |
| A 2-second lead is often decisive in an equal match | Sorsi **[snippet]** | one |
| Double-sitting ~10 s "you can basically just resign" | Sorsi **[snippet]** | one |
| Under 20 seconds, stop sitting and move | [12] | one, self-described strong amateur |
| "Sit until 10, then low flow": Black's partner sits to 10 s then plays quiet moves | chriswsh **[snippet]** | one |
| The speed of a team is set by its slower player | [13] | one |
| Clubs that find sitting excessive use "no sit" rules or per-board scoring | [1] | |
| Without clocks, waiting is usually limited to the time the partner takes to make three moves | [1] | |

When to sit (consensus across [1][2][12][13][16][17]):
1. You are mated in one by *any* move you could make, and the partner cannot avoid giving the mating piece. Sitting
   is then "the only option" [1].
2. Your partner is one piece from mate; you sit so the trade that would give the defender a blocker never happens
   [16][17].
3. To force the opponent to move without the piece he wants: "tell your partner not to move until your opponent
   moves" [2][5].
4. Offensively early: "sit a few seconds for a pawn which will basically win the opening" (below top level, where
   openings are premoved) [snippet: Sorsi].

When not to sit: when down material waiting for pieces [12]; sacsitting as a habit ("every game it is downright bad"
[snippet: JarlCarlander]); "don't think in terms of defending by having your partner stall unless it's the only
alternative" [13]. Disagreement on frequency: chessbraingame says err towards saying "sit" too often, up to ten
times a game [14]; Sorsi says strong players sit too little and weak players too much **[snippet]**; JarlCarlander
says don't sit for pieces unless told to **[snippet]**.

Sitting is not thinking: "Sitting and thinking may look similar, but sitting is using a timing strategy" [12]. The
bugmisc author: "the longer you think, the worse your move becomes" and play bughouse like 1+0 bullet unless you have
to sit **[snippet]**. Top-level openings are premoved **[snippet: Sorsi]**, so early moves cost ~0 s.

Clock interaction with mate: FICS/WBN rules make the mate rule depend on *potential* drops, so a player under a
non-decisive check can sit and wait for a blocker to arrive [3][7]. OTB rule 5.3.1 (German Tandemschach): a mate is
averted if the partner's *very next* move makes a piece available [9].

### 2.3 Trades and flow

- Flow = the rate at which captured pieces pass between boards. Players ask for "high flow"/"low flow" [snippet].
  Sorsi: "whoever controls the flow controls the game"; "the best flow is generally light but consistent"
  **[snippet]**.
- Trades help the side with the initiative [12]. If your partner has many idle pieces and his opponent few, trades
  favour the opponent [13]. Trading removes your own defenders [13]. "Trade-all" is a style that neither attacks nor
  defends [12].
- Specific holds: "No queens!" when under attack, since a passed queen creates immediate threats [1][21]; "if your
  partner has castled, don't trade queens or your partner will get mated" **[snippet]**.
- Tempo of trades: "guard against checks and/or make sure trades come on your tempo" [13]. Read with [24] ("even them
  taking 3 seconds longer on a move can put you in a much tougher spot") and the sitzkrieg rule: a capture made while
  the partner is on move gives him a drop *now*; the same capture made while the partner's opponent is on move gives
  the partner a drop *after* the opponent has had a free move, and the recapture (if any) arms the opponent first.
- Don't feed the "partop" (partner's opponent) [12]. Automatic "free" captures are a listed beginner error [18].
- "If you win material the effect is doubled" (your board and your partner's hand) [14].
- Low flow is a weapon: strong bughouse players beat titled chess players "using low flow, since they play for future
  piece drops" **[snippet: JarlCarlander]**.

### 2.4 Checks, mates, attack and defence

- Decisive checks: only contact checks, knight checks and double checks cannot be blocked by a drop [3][4][7]. FICS'
  definition of mate requires that no drop interposition is *possible*, not merely none available [3].
- "Keep attacking with checks. As long as he's in check, I'm winning" [2]. A run of checks also burns the opponent's
  clock [16].
- Standard attack pattern: sacrifice on f7/f2, then knight check, then drops to drive the king out, then a mating net
  [2][16]. Named mate: Two Knights, N@h3 gxh3 N@f3 Kh1 Nxf2 **[snippet]**. g7 is as weak as f7 **[snippet:
  uschess]**; "once you lose g2, you're dead" [12].
- Defence rules: no holes near the king, especially the second rank and the c- and f-files, because a hole is filled
  by a drop at once [2][12]; interpose rather than retreat [12][13]; don't bring the king forward, give flight squares
  to the side [12]; Qe7 before the f7 sacrifice gives the king d8 [13]; drop a pawn on h6/h3 after castling to seal
  the king ("a very secure environment if black castles and drops a pawn on h6") [15]; fill holes by dropping [16].
- The defender must be more precise: "a defending mistake can have bigger consequences than an attacking mistake"
  [1]. Accepting a sacrifice can hand the sacrificer's partner a piece [1].
- "Material is less critical because there are no endgames" [1]; still "Material matters" [12]. "A piece in the hand
  is worth two on the board" [5]; "a pawn in hand is worth far more than a pawn on the board" [17].

### 2.5 Piece values (disagreement is the finding)

| Source | P | N | B | R | Q | Note |
|---|---|---|---|---|---|---|
| André Nilsson, via Wikipedia [1] | 1 | 2 | 2 | 2 | 4 | the most quoted scale |
| chessbraingame [14] | 1 | 2 | 2 | 2 | 4 | adds: B and R slightly above N at high level; two pieces > Q; two pawns > N |
| FICS strat file, 1995 GICS mirror [2] | 1 | 7 | 2 | 4 | 10 | bishops "often no better than a pawn". Corrected: the live freechess.org file gives 1/2/2/2/4 |
| Sunsetter engine [1] | 100 | 192 | 195 | 200 | 390 | hand-tuned engine values |
| ChessHost [16] | 1 | 2–3 | 2–3 | 2–3 | 4–5 | vendor page |
| Jerry [12] | | | | | | "Two pawns for a piece is a good deal"; Q for N can be right to end a check sequence |
| Panickssery [15] | | | | | | "Knights are more valuable than bishops" |

The spread (N from 2 to 7 on a pawn scale) shows the value is positional: a knight in hand against an exposed king is
decisive, a knight in hand against a closed fortress is a pawn-and-a-half.

### 2.6 Openings

General principles, stated by nearly every source:
- Far fewer openings than chess; many chess openings are "instantly unsound because they expose squares for pawn
  drops" [1][15]. Specifically unsound: early ...c5 (c7 hole; eliminates the Sicilian), early c4 (eliminates Queen's
  Gambit, Nimzo, English), fianchettos (g7/g2 hole: Modern, Pirc, KID, Grünfeld, Dutch g3), early ...b6 (b7 hole,
  though the Federkevic 2...b6 is a named exception) [2][12][15][snippet: JarlCarlander, Sorsi].
- "Never move any pawns except the e and d pawns" [2]; "unwise for either side to push his c-pawn or leave his
  f-pawn vulnerable" [15].
- Castling: "discouraged for both sides because it limits king escape routes" [2]; "it's typical not to castle" [15].
  ChessHost disagrees ("castle early") [16] but is the weakest source. Queenside castling is worse: b2 can be taken
  with check [12].
- Development matters because pieces arrive after the first few moves and developed pieces defend against early drops
  [1]. "The purpose of a bughouse opening isn't to throw pieces at f7 ... it's to create long-term positional
  potential" **[snippet: JarlCarlander]**.
- Top players vary 1.e4 and 1.d4 to be harder to prepare against **[snippet]**. Premoved openings are punished by
  "low-risk anti-premoves" (3.Bg5 as a premove "is rarely wise") **[snippet: JarlCarlander]**.
- If no early pawns are coming from the other board, 1.e4 e5 becomes playable for Black because the f7 sacrifice is
  weaker **[snippet: Sorsi]**. This is the clearest written statement that the right opening depends on the *other*
  board's flow.

Named lines (moves as given by the source):

| Name | Moves | Source | Comment |
|---|---|---|---|
| Fortress Defense | 1.e4 e6 2.d4 d6 | [2][15] | "By far the most common bughouse opening" [2]; Black waits, no holes |
| Federkevic Defense | 1.e4 e6 2.d4 b6 | [2] | FICS strat file; b7 hole accepted for ...Bb7 |
| Barbeau Counter Attack | 1.e4 e6 2.d4 Qh4 | [2] | early queen sortie |
| e6–Nf6 system | 1.e4 e6 2.d4 Nf6 (then ...d5 or ...Be7) | MiniGreat, Sorsi **[snippet]** | "Advanced variations" series |
| e6–Nc6 system | 1.e4 e6 2.d4 Nc6 | Sorsi **[snippet]** | |
| Scandinavian retreat | 1.e4 d5 2.exd5 Qxd5 3.Nc3 Qd8 | [13] | "best opening in terms of not having any weaknesses" |
| Alekhine sacrifice | 1.e4 Nf6 2.e5 d5 (gives a pawn) | [15] | author's unverified "most powerful response" |
| Leaf Gambit | 1.e4 d5 2.exd5 e6 | Sorsi **[snippet]** | "many people just premove Nc3 to avoid losing the d5 pawn" |
| Exchange French | 1.e4 e6 2.d4 d5 3.exd5 | Sorsi **[snippet]** | |
| French without ...c5 | 1.e4 e6 2.d4 d5 3.e5 (no ...c5) | **[snippet: JarlCarlander]** | recommended for beginners |
| Nima's Gambit | not given | MiniGreat **[snippet]** | title only |
| Mongolian Attack | White kingside set-up vs f7 | [2] | name only, moves not given |
| Qe7 anti-f7 | 1.e4 e6 2.d4 Qe7 idea | [13] | gives the king d8 after Bxf7+ |
| White plan vs Black king | Ng1–f3–g5xf7 to clear f7 | uschess **[snippet]** | |

Opening synchronisation: Sorsi has a post on choosing the two boards' openings together ("Bughouse Openings
Synchronization", unread **[snippet]**). The idea from [snippet: Sorsi, 1.e4 e5] is that Black's choice depends on
whether pawns will arrive early from the White board.

### 2.7 Communication conventions (what the messages encode)

- Permitted everywhere; OTB rules only ban touching the partner's board or clock [6][7][10]. German OTB rule 7.2 bans
  recommending two or more consecutive moves [9]. WBN rule 1.11: "the player can advise to partner even concrete
  moves" [7].
- Standard vocabulary [1][16][17][18][22]: "sit" / "go"; "trades are good" / "trades are bad" (= high/low flow);
  "no queens", "hold that bishop"; "knight mates", "I need pawns", "rook mates me" (= don't let him get a rook);
  "up"/"down" (time). FICS has a "Mates opponent" button: "if he gets you a N, you can mate" [snippet].
- Sjeng's protocol is the precise version [26]: `+n` (+150 eval units on knights in hand), `++n` (+1000), `+++n`
  (+50000, "mates"); `-p/--p/---p` symmetric; group letters `d` (diagonal: P,B,Q), `h` (heavy: R,Q); `sit`,
  `go`, `fast`, `slow`, `dead` ("fellow sjeng is dead -> give it all we've got"). Its `CheckBadFlow()` runs every move:
  for each piece type it adds that piece to the opponent's hand and checks whether any opponent move leaves no legal
  reply; if so it sends `---p` (etc.), and sends "p doesn't mate me anymore" when the danger clears, with a 3-move
  memory (`CANCEL_THRESH 3`).
- What two humans lose: online chat is "low-throughput" and the chat box too small to read [24]; a 14-second delay
  by the partner on a required capture lost the game on time [24]. Compatibility matters more than strength [12];
  a much stronger partner makes coordination harder [21].

### 2.8 What top players and teams are known for (thin, mostly snippets)

- Kazim Gulamali (FM, FICS): coordination is the key skill; watch both boards; "blow up the position and get the
  opponent into time pressure" **[snippet: Chess Life 2012/2014]**. Advice to learners: watch the top players.
- Dan Yeager (chickencrossroad) is a FICS legend; Levon Aronian is the strongest bughouse player among elite GMs
  [23][snippet]. Isaac Chiu and Janak Awatramani won the 2025 chess.com BWC and the 2026 WR championship
  **[snippet]**. Xiong and Liang won 50 consecutive games in a 2024 qualifier **[snippet]**.
- Advice for data use: "if you are fairly new to bughouse (below 3000 games), don't study top players' games from
  bughousedb.com" because top games are premoved, sit-heavy and context-dependent **[snippet]**.
- HiveMind (the paper, CoG 2023) was trained by supervised learning on human games then AlphaZero-style, "using clock
  times to determine the optimal turn sequence", and beat Fairy-Stockfish >95% at 2 s [28].

### 2.9 What one controller should do differently from two humans

Derived from the above; not stated as such by any source, except the talkchess thread, which lists "1 engine plays on
two boards and has full control" as the first architecture and notes a wait command and full clock visibility [27].

1. **Perfect, free communication.** Every "p mates me", "need N", "sit", "go" is an internal computation with zero
   latency. The 14-second miscoordination in [24] cannot happen. The controller should therefore sit *more* than
   human teams do in positions where the gain is a specific piece, because the cost of a sit for humans is partly the
   risk that the partner does not understand it.
2. **Exact diagonal arithmetic.** Humans estimate "am I up on the diagonal"; the controller knows all four clocks to
   the millisecond and can run a sitzkrieg to the last second.
3. **Asymmetric allocation by design.** "The speed of a team is set by its slower player" [13] becomes "the team's
   budget is one machine": thinking on board A is time board B does not get. The Tournament Game strategy [13]
   (one board plays, the other sits) is the natural extreme. A human pair cannot hand over clock; one controller can.
4. **Trades on tempo.** Because it sees both on-turn states and the clocks, it can capture exactly when the partner
   board can use the piece immediately, and withhold a capture that would land while the diagonal opponent is on move.
5. **Joint opening choice.** It can pick the two openings as a pair (attacking White set-up with a Fortress Black, or
   1.e4 e5 for Black when the White board promises no early pawns) rather than two independent repertoires.
6. **Deliberate double sits.** For humans a double sit is anxiety; for the controller it is a legal joint action to
   take when both diagonal opponents are also sitting and the team is up on both diagonals.
7. **Risks peculiar to one controller:** a search that is long on one board starves the other (the think_gain head in
   spec v2 B is the mitigation); sacsitting can look attractive to a value head that over-trusts future arrivals; and
   with no human partner there is no "sorry" channel, so a lost tempo is never recovered by talk.

### 2.10 Human strategies that engines are reported to miss

- Starving sits and sitzkrieg: classical engines have no pass move; Sjeng sits only when told to [26]; Hivemind sits
  only under a hand-written `TimeAdvantage` gate (section 3).
- Tempo-timed trades (2.3) need both boards' turn state and clocks in the input and out-of-step training data.
- "Strategical attacks": "Sunsetter is strong at tactics in open positions while clueless at generating strategical
  attacks" [31]. Jerry's "sacsitter" and "trade-all" styles are the human failure modes that a shallow engine
  reproduces [12].
- Planning for future drops rather than the current position ("you are dealing not just with the position, but with
  future drops") **[snippet: chess.com forum]**; forum opinion is that "computers are not significantly better than
  humans at bughouse" **[snippet]** (unverified).
- Opening choice conditioned on the other board (2.6): no engine source mentions it.

---

## 3. Implications for the engine

Reference for "does stock Hivemind capture it": the fork's README input table (37 planes per board: pieces, pockets,
promoted/ep, perspective/side-to-move/constant, castling, `TimeAdvantage` flag, last move, halfmove, repetition;
cross-board attention; joint policy over both boards) and `engine/src/environment/joint_action.h` (passing on an
on-turn board is legal only with `TimeAdvantage`, or when the partner board captures; double sit only with
`TimeAdvantage` and exactly one board on turn). The spec's clock build adds per-board planes 37–40 (our seat and the
same-board opponent's clock) and v2 adds elapsed-think planes 41–42, a think_gain head and a 20-value arrivals head.
Upstream `aminwoo/hivemind` has since moved the same way: its "twin-s-clock" model has 39 planes per board including
"the team's time lead and relative move speed", and its search "only allows waiting while the team's lead lasts" [29].

| # | Idea | Stock Hivemind | Concrete change |
|---|---|---|---|
| 1 | **Diagonal time is the quantity that matters** (2.2) | No. Stock has one team-level yes/no `TimeAdvantage` flag on both boards. The spec's planes put our-seat and same-board-opponent clocks in each board block, so the diagonal difference (our A vs their B) lives across the two blocks and must be assembled by the cross-board attention. | Add two explicit planes per board block: `tanh((t_us_this_board − t_opp_other_board)/20)` and `tanh((t_us_this_board − t_opp_this_board)/20)`. Note (critic): the first is the same number in both blocks, because with equal controls and synchronised starts the two boards' clock sums are equal and so the two diagonal differences coincide; stock's flag is already its sign (`board.py::time_advantage`). Cheap (still zero-initialised), removes an arithmetic the 2-layer attention may learn late. Log in training the correlation of sit frequency with the diagonal difference as a sanity metric. |
| 2 | **Sitting is legal whenever it is worth it** (2.2) | No. Hand-written gate in `joint_action.h`; sit frequency is a rule, not a learned choice. | Already in spec (`FreePass`). **Corrected (checked):** the opponent team in the tree *is* allowed to pass symmetrically: `searchthread.cc` inverts the flag for the opponent team (`leafTeamHasTimeAdvantage`), every action list carries `MOVE_NONE`, and `JointCandidateGenerator` filters with `is_joint_action_legal`, which under `g_freePass` accepts any pass by either team. Still add the regression test (a position where the opponent's best action is a pass) so this cannot silently break. |
| 3 | **Thresholds at 2–20 s** (2.2 table) | No clocks at all. | Spec planes `exp(−t/10)` and `min(t/300,1)` resolve 2–20 s well. Train on realistic controls: human bughouse is 3+0 and 5+0 [1][6][7][20]; the beat-stock runs used 60 s and one 180 s arm. Allocate part of the 10k GPU hours to 180 s and 300 s games, since sits of 10–30 s only exist there. |
| 4 | **"p mates me" / danger-by-piece** (2.7, Sjeng `CheckBadFlow`) | Partly, implicitly: the joint search sees partner captures and the resulting drops, but at ~100 nodes it rarely reaches "opponent gets a pawn → mate in 2". **Corrected:** the root mate probe runs only *without* `TimeAdvantage` (`agent.cc`: `if (teamHasTimeAdvantage) return false;` with the comment "Ahead on time, this team may sit the board a line is played on, so no single-board mate is forced against it"; `collect_internal_probe_targets` likewise runs under `if (!rootTeamHasTimeAdvantage)`). v1's `MateProbeAnyTime` lifts that gate. | Add a **piece-danger head**: 10 logits (5 piece types × 2 boards) = "if the opponent on this board received one more X now, is the side to move lost within 3 plies". Labels come from the recorder with a short search per piece type, exactly as `think_labels.py` does; masked BCE, weight 0.1. The head makes Sjeng's probe instantaneous and gives the policy a learned "no queens" signal. |
| 5 | **Trades on tempo** (2.3) | Partly. **Corrected:** the *supervised* phase saw real out-of-step human data (`game_reader.py` orders the two boards' moves by chess.com timestamps), so it04's prior could have picked tempo-timed trades up; it is the *RL self-play* (`selfplay.cc`, strictly alternating team macro-plies with the flag toggled each ply) that is lockstep by construction and could only have unlearned it. Stock's `is_single_pass_legal` even treats "partner captures" as the condition that *permits* a pass, a rule-of-thumb stand-in for the real timing. | Already the point of out-of-step self-play. Add a diagnostic: for each capture in recorded games, record who was on move on the other board and the time until that piece was dropped; the trained net should show captures clustering when the partner is on move. If it does not, the jitter range is too narrow. |
| 6 | **Premoves and bimodal human think times** (2.2) | No. Stock plays fixed nodes. | The think-time jitter (0.5–2.0×) is unimodal. Human timing is bimodal: ~0.1 s premoves in the opening and 5–30 s sits. Draw jitter from a mixture (e.g. 30% near-instant, 60% normal, 10% long) so the clock planes see realistic opponents, and so the net learns anti-premove play. |
| 7 | **Asymmetric think allocation / Tournament Game strategy** (2.1, 2.9) | No. Stock searches the joint action with one budget. | Spec v2 think_gain head is the mechanism. Add an evaluation metric: share of team think time spent on the board that eventually wins; expect it to rise with training. Consider a second label for think_gain: "does a 4× search change the move *on either board*", since a long think on A can change B's move through the joint policy. |
| 8 | **White attacks, Black defends** (2.1) | Yes, implicitly: colour is in the perspective plane and the SL data came from human games, so the prior carries it. Whether the RL loop keeps it is unknown. | Measure per-colour capture rate and drop-near-king rate in self-play by iteration. If both boards converge to the same style, that is a symptom of the value head ignoring flow; the arrivals head (spec v2 C) is the fix already planned. |
| 9 | **Flow control** (2.3) | Not as a concept. The arrivals head predicts flow; nothing rewards controlling it. | Keep arrivals as auxiliary only (it is a prediction, not a goal). Add a logged metric: captures per minute on each board, by team, by game phase. Use it to detect "trade-all" collapse. |
| 10 | **Decisive checks and the mate rule** (2.4) | Yes: the arena's `is_bughouse_mate` (double check, or no empty square between king and checker) matches FICS/WBN [3][7]; stock has a root mate probe. | None beyond `MateProbeAnyTime`. |
| 11 | **Opening set** (2.6) | Stock has no book; `OpeningNoise` Dirichlet noise for diversity. **The arena's `openings.py` is chess theory, and most of it is bughouse-unsound by the sources above**: Sicilian ×3 (…c5), Closed Sicilian/Pirc/Modern/KID/Grünfeld (…g6), Dutch (…f5), QGD/Slav/Nimzo/Queen's Indian/English (c4, …b6), Reti/English g3. Eval and training pairs therefore start from positions a strong bughouse team never reaches, and the "swap sides" fairness hides, rather than fixes, that the positions reward the attacker of a weakened structure. | Replace or augment `OPENINGS` with the bughouse canon: Fortress 1.e4 e6 2.d4 d6; e6–Nf6 (…d5 and …Be7 branches); e6–Nc6; 1.e4 d5 2.exd5 Qxd5 3.Nc3 Qd8; Leaf Gambit 1.e4 d5 2.exd5 e6; Alekhine 1.e4 Nf6 2.e5 d5; 1.e4 e5 2.Nf3 Nc6 3.Bc4 lines; Federkevic 1.e4 e6 2.d4 b6; Barbeau 2…Qh4; and 1.d4 d5 2.Nf3/Nc3 systems without c4. Keep two or three chess openings as controls. Gate against stock on *both* sets and report both; if the new net wins more on bughouse openings than on chess ones, that is a real (not an artefact) gain. |
| 12 | **Opening synchronisation** (2.6, 2.9) | Partly: the joint policy sees both boards from move 1, so it can condition. Whether it does is unknown. | Pair openings deliberately in eval (attacking White set-up with Fortress Black vs. both passive) and check that the net's early captures on the Black board rise when the White board is the attacking set-up. Also give self-play the standard start position for a share of games (no forced opening) so the net can learn its own pairings. |
| 13 | **Piece values** (2.5) | Learned; never explicit. | Use the spread as a diagnostic, not a target: regress the value head's change on hand composition in recorded positions and compare to the Nilsson/FICS scales. A learned N≈7 on a pawn scale against open kings and N≈2 against fortresses would match the human disagreement; a flat value would suggest the pocket planes are under-used. Weight the arrivals loss per piece type by roughly P1 N2 B2 R2 Q4 so pawn arrivals (the most frequent and most mate-relevant) are not drowned by rarer queens. |
| 14 | **Sacsitting is bad** (2.2) | Unknown. With `FreePass` the net may discover sac-then-sit if the arrivals head makes it over-optimistic about rescue pieces. | Log "sacrifice followed by own pass within 2 s" per 100 moves; expect low and falling. If it rises, lower the arrivals loss weight or add a small penalty on passes made while down material and not in check. |
| 15 | **Partner data quality** (2.8) | **Corrected:** stock SL used chess.com games scraped by `src/hivemind/data/scrape_cc.py` (players ≥2400, partners ≥2200), with per-move timestamps stored and used by `generate_planes.py` for plane 31 and for `pass` labels (sits by the team behind on time are dropped). | If more SL data is added from bughousedb, filter by rating and keep the clocks: top games are sit-heavy and premoved, which is exactly the behaviour the clock planes are meant to learn, but only if the move timestamps are kept in the training sample. |
| 16 | **Communication for a human partner** (2.7) | None (stock has `RequireMoveOn` style options, no messages). | Out of scope by spec. If ever needed, the Sjeng vocabulary [26] (`+++n`, `---p`, `sit`, `go`) is the de-facto protocol, and the piece-danger head (#4) and arrivals head give the engine the facts to say. |

---

## 4. Sources

Read in full (or in the quoted part):

1. Wikipedia, "Bughouse chess" — https://en.wikipedia.org/wiki/Bughouse_chess — strategy, communication, sitting,
   sitzkrieg, Nilsson and Sunsetter piece values, opening principles, book references (Zimmermann 2006, Manson &
   Hoover, Penn & Dizon).
2. ICS help "bughouse_strat" (FICS strategy file, mirrored at unix-ag) — https://www.unix-ag.uni-kl.de/~chess/gicshelp/node170.html —
   piece values Q10 N7 R4 B2 P1, Fortress/Federkevic/Barbeau/Mongolian names, "never move any pawns except the e and d
   pawns", stalling rules, White mates/Black holds.
3. FICS help "bughouse" — https://www.freechess.org/Help/HelpFiles/bughouse.html — decisive-check mate rule, ptell,
   match ending, same time control both boards.
4. ICS help "bughouse" (unix-ag mirror) — https://www.unix-ag.uni-kl.de/~chess/gicshelp/node169.html — same rules,
   mate-by-potential-drop wording.
5. WCCC 2004 bughouse tournament page — https://wccc2004.juliasfairies.com/bughouse_tourney.html — tips: king
   safety, contact/knight checks, "a piece in the hand is worth two", stall, sitzkrieg decided by diagonal time, Sjeng.
6. WCCC 2004 complete bughouse rules — https://wccc2004.juliasfairies.com/bughouse_rules.html — 5-minute control,
   no stalemate, communication allowed, no physical action on the other board.
7. WBN "Laws of Bughouse chess" PDF — https://www.chessfed.gr/old_site/wccc2007/files/bughouse_chess_complete_rules.pdf —
   1.5 mate only if no potential drop parries, 1.11 full communication, 1.9 last-move drop after partner's mate, 1.10
   simultaneous finishes.
8. Hawaii bughouse rules PDF — https://www.chessfed.gr/old_site/wccc2007/files/bughouse_chess_main_rules.pdf — clock
   start synchronisation, dropped rook may castle, dropped pawn keeps two-step.
9. Tandemschach-Regeln (German OTB rules) PDF — https://web1.sitebuilder.webspace-verkauf.de/gallery/Tandemschach.pdf —
   5.3.1 mate averted if partner's next move supplies a piece, 7.2 no two consecutive move recommendations, 6.2
   stalemated player stays on move.
10. Northwest Chess bughouse rules — https://nwchess.com/articles/rules/bughouse_rules.htm — Game/5, verbal
    communication and move suggestions legal, no touching partner's pieces.
11. USCF forum "Bughouse rules" — https://forum.uschess.org/t/bughouse-rules/16329 — no stalemate, clock runs when
    no legal move, drop-interposition negates mate.
12. Jerry, "The ABCs of Bughouse" — https://abcsofbughouse.blogspot.com/2021/12/the-abcs-of-bughouse.html — under
    20 s move, sitting vs thinking, sacsitting, partop, g2/b2 weaknesses, two pawns for a piece, defence first.
13. bugmisc, "Tips for Beginners that will Raise Your Ratings" — https://bugmisc.blogspot.com/2018/04/tips-for-beginners-that-will-raise-your.html —
    team speed set by slower player, defend before attacking, Tournament Game strategy, "trades come on your tempo",
    1.e4 d5 2.exd5 Qxd5 3.Nc3 Qd8, Qe7 anti-f7.
14. chessbraingame, "bughouse chess concept for beginner" — https://chessbraingame.blogspot.com/2017/05/bughouse-chess-concept-for-beginner.html —
    speed is for the diagonal opponent, say "sit" often, P1 B=N=R2 Q4 with refinements, doubled effect of material.
15. Arjun Panickssery, "Initial Thoughts on Bughouse Chess Openings" — https://arjunpanickssery.substack.com/p/initial-thoughts-on-bughouse-chess —
    Fortress most common, Alekhine sac, h6 pawn drop, no castling, White attacks/Black trades, knights > bishops. Weak
    source (no credentials, speculative).
16. ChessHost, "Bughouse Strategy: 12 Tips" — https://www.chesshost.app/blog/bughouse-strategy/ — trade rules by
    partner state, sit when one piece from mate, calls, rough values. Vendor page.
17. LichessDude bughouse guide — https://lichessdude.com/bughouse-chess-rules-strategy-guide/ — "I need a knight",
    "no trades", "sit", pawn in hand, 3+0/5+0.
18. Old School Chess, bughouse — https://oldschoolchess.com/learn/variants/bughouse — automatic captures as an
    error, "think about time diagonally", move fast when partner in danger.
19. chessworld.net bughouse guide — https://www.chessworld.net/chessclubs/openingguide/online-chess-bughouse.asp —
    sit to deny pieces, tell partner what the opponent needs. Weak.
20. Caissly, bughouse — https://caissly.pages.dev/variants/bughouse — tournament convention (talk only on your own
    turn), 3+0/5+0, named elite players. Weak.
21. Chesscul, "Pass the pieces" — https://chesscul.com/en/pass-the-pieces/ — play fast to send defensive material,
    "no queens", mismatched partner strength.
22. Alberto Chueca, bughouse teammate — https://www.albertochueca.com/?p=14575 — "give me a queen!", time advantage
    enables sitting. Weak.
23. St. Louis Public Radio, "On Chess: What's the deal with bughouse" — https://www.stlpr.org/arts/2019-04-18/on-chess-whats-the-deal-with-bughouse-chess —
    communicate needed pieces, sit only when ahead on clock, Aronian.
24. "The Bughouse Effect" (greaterwrong/LessWrong) — https://www.greaterwrong.com/posts/imnfJ9Ris7GgjkZbT/the-bughouse-effect-1 —
    first-hand account of a 14 s partner delay losing on time, low-throughput chat.
25. PyChess variants: Bughouse — https://www.pychess.org/variants/bughouse — rules, communication allowed, pointers
    to JarlCarlander and JannLee.
26. Sjeng 11.2 `partner.c` — https://sources.debian.org/src/sjeng/11.2-15/partner.c — ptell protocol (`+++n`,
    `---p`, sit/go/fast/slow/dead), eval deltas 150/1000/50000, `CheckBadFlow` per-piece mate probe with
    `CANCEL_THRESH 3`.
27. talkchess, J. Rosenthal "bughouse engine" thread — https://talkchess.com/viewtopic.php?p=937161 — three engine
    architectures, wait command, full clock visibility.
28. Woo, Sweetser, Aitchison, "HiveMind: Learning to Play the Cooperative Chess Variant Bughouse with DNNs and MCTS",
    IEEE CoG 2023 — https://ieeexplore.ieee.org/document/10333210/ — abstract only: SL on human games, AlphaZero with
    clock-based turn ordering, >95% vs Fairy-Stockfish at 2 s.
29. aminwoo/hivemind README (upstream) — https://github.com/aminwoo/hivemind — twin-s-clock 39-plane model with
    time lead and relative move speed, waiting allowed while lead lasts.
30. Chessprogramming wiki, Georg von Zimmermann — https://www.chessprogramming.org/Georg_von_Zimmermann — Sunsetter
    author, editor of *Bughouse Chess* (2006).
31. Sunsetter results page — https://sunsetter.sourceforge.net/results.html — "clueless at generating strategical
    attacks", FICS crazyhouse 2569.
32. Book listing, *Bughouse Chess* ed. G. von Zimmermann, BoD 2006, 254 pp. — https://akademibokhandeln.se/bok/bughouse-chess/9783833468117 —
    contents (openings overview, annotated games, Aronian interview). Listing only; the book itself was not read.

Local repository sources: `third_party/hivemind/README.md` (input planes, `TimeAdvantage`, `OpeningNoise`),
`third_party/hivemind/engine/src/environment/joint_action.h` (pass legality), `arena/openings.py` (eval openings),
`docs/custom_engine_spec.md`, `docs/custom_engine_spec_v2.md`, `docs/beat_stock.md`, `docs/phase1_scheduler.md`.

Snippet-only (search result text; the pages could not be opened from this machine; every claim marked **[snippet]**
above comes from one of these and should be verified by reading the post):
- Sorsi (chess.com): "Advanced bughouse tips", "Basic Bughouse Tips", "Time Management in Bughouse", "Decision Making
  in Bughouse", "Bughouse Openings Synchronization", "1 e4 e5 part 1", "Advanced Opening Analyses: The Leaf Gambit",
  "The Exchange French", "The e6 – Nc6 System", "Give Them Little and Make Them Pay a Lot" —
  https://www.chess.com/blog/Sorsi
- JarlCarlander (chess.com): "Bughouse for Beginners: Coordination and Strategy", "Partner Requests in Bughouse",
  "esoteric wisdom of the ancient bughouse masters", "survey of opening theory for black", "How to Theorize in
  Bughouse", "Bughouse and AI", "Bughouse Buttons", "The Bughouse Sicilian", "Controlled Chaos: Interview with a
  Bughouse Master" — https://www.chess.com/blog/JarlCarlander
- chriswsh (chess.com): "Let's Bughouse! Playing Against the Sicilian, Part 2" —
  https://www.chess.com/blog/chriswsh/let-s-bughouse-playing-against-the-sicilian-part-2
- MiniGreat (chess.com): "Bughouse Openings Part 1: Nima's Gambit", "Part 2: Advanced variations of e6 Nf6 system".
- MidnightFox (chess.com/lichess): "Chess.com Bughouse World Championship 2024".
- US Chess: Brooklyn bughouse articles https://uschess.org/content/view/11258/632 and the Kazim Gulamali interview
  https://www.uschess.org/content/view/11764/671 (403 from here).
- chess.com forum "Is there any actual strategy to winning in Bughouse?" and club forum "Bughouse tips. Anyone?".
