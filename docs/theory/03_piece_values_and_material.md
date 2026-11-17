# 03. Piece values and material flow in bughouse and crazyhouse

Written 2026-11-17 for the custom-engine project (`docs/custom_engine_spec.md`, `_v2.md`, `beat_stock.md`). Scope:
how much pieces are worth on the board and in hand, how that changes with position and time, trading principles
across boards, feeding and holding, material versus initiative, and what crazyhouse engines (Stockfish multi-variant,
Fairy-Stockfish, CrazyAra, Sunsetter, Sjeng, Droste & Fürnkranz) actually encode. Section 3 maps each idea onto what
original Hivemind (it04) already has and what would have to change.

Access note. `chess.com` and `lichess.org` resolve to `127.0.0.1` in this machine's `/etc/hosts`, and fetching them
through a proxy was refused by the permission system, so every chess.com and lichess.org item below is cited from
search-result snippets only and is marked **(snippet only)**. One lichess post ([3]) did come through and was read in
full. Everything else was read as a primary source.

---

## 1. Summary

- Every source agrees that drop variants **compress** piece values: pawns and knights go up, queen and rook go down.
  Engine-derived crazyhouse values put the queen at 3.8 to 4.5 pawns, not 9 ([3], [4], [5], [7], [8]). The most common
  human bughouse table is **P=1, N=B=R=2, Q=4** (attributed to André "Gnejs" Nilsson) [1]. **Corrected:** the
  outlier **Q=10, N=7, R=4, B=2, P=1** [2] is the 1995 GICS mirror of the FICS strategy file; the live file at
  freechess.org (`Help/HelpFiles/bughouse_strat.html`, checked 2026-11-17) gives **Q=4, N=2, R=2, B=2, P=1** in the
  same sentence. The 10/7/4/2/1 scale is therefore one old revision, not FICS's current advice. Nobody has a settled table and strong players say the number depends on
  the position [1], [2], [23].
- The reason given everywhere: a piece in hand can land on any empty square, so **in-hand pieces all have the same
  mobility**; what matters is what they do on arrival (check, fork, block) [1]. Hence the knight (unblockable check,
  the king cannot stand next to it) is the best attacking drop in every source, and the queen loses most because its
  long-range mobility is no longer special and it gets chased with tempo by minors [4], [16], [17].
- **In hand is worth more than on board** for pawns, knights and rooks, less for bishops and queens, in the only
  learned study that separated them (Droste & Fürnkranz 2008, Table 6.1) [4]. Fairy-Stockfish adds an explicit
  in-hand bonus, twice as large for non-sliders (pawn, knight) as for sliders [11]. Multi-variant Stockfish's
  king-danger weight for a **knight in hand (200) exceeds a queen in hand (152)** [9].
- The marginal value of a hand piece **depends on what else you hold** (a fourth pawn is worth less than a first) [4];
  Stockfish multi-variant encodes this as a quadratic "in-hand imbalance" table [10]. Human advice: keep a variety of
  pieces in hand, and spare pieces for defensive "air holes" ([28] snippet, [16]).
- Material is a **weak proxy** in these games: Droste & Fürnkranz's material-only crazyhouse evaluator won under 4% of
  games against the same engine with learned piece-square tables (96.4% for the tables), versus 12%+ in standard
  chess and suicide [4]. "Because new pieces come in, there is no endgame play in bughouse and material is therefore
  less important" [1]. "Initiative is more important than material" [16]; CrazyAra "attaches high importance to
  initiative" and sacrifices for activity [14]. The data-fitted value of **having the move** is 0.07 pawns in
  crazyhouse versus 0.01 in chess [3].
- Trading is a **cross-board decision**, not a material one. Attackers ask for trades ("+++", "trades are good");
  defenders ask for none ("---", "trades are bad") [1], [30]. Accepting a sacrifice gives your partner's opponent a
  piece [1]. Flow should be "light but consistent" and whoever controls flow wins ([24] snippet). "Don't feed your
  partop" (partner's opponent) [17]. Classical bughouse engines encode the mirror image: Sunsetter penalises
  positions whose plan needs a rook, knight or pawn it does not have ("the Malus for Pieces used we would have to wait
  for") and prefers mates that do not wait for a piece [6].
- **Time beats material** more often than in chess: "time is an even greater asset in bughouse than material"
  ([25] snippet); the diagonal clock difference matters more than the same-board one [1]; time matters most on
  sparse boards, when down material, or late ([17], also Fairy-Stockfish scales king danger by **material density**
  in drop and two-board variants [12]). "Don't pause to wait for material when you're down" [17].
- Human-data networks inherit human valuations and human aggression [14]; learned-from-self-play values differed
  from both human tables and hand-tuned engine values and beat them 66-68% in material-only matches [4]. Expect the
  RL rounds to move Hivemind's implied values and measure that drift directly (section 3.9).

---

## 2. The knowledge

### 2.1 Why values change in drop variants

- Wikipedia, citing von Zimmerman (2006, p. 17): "Bughouse piece values differ because pieces in reserve essentially
  have the same mobility as they can be dropped on any vacant square." Pawns gain relative importance because their
  limited board mobility does not restrict their drop; reserve pawns block non-contact checks and can land on the
  seventh rank [1].
- Long-range pieces lose value because the pawn structure changes constantly and they get hemmed in (Manson & Hoover
  1992, pp. 32-33 via [1]). Droste & Fürnkranz: "The low value of the queen probably results from the generally
  higher mobility on the board which is caused by the dropping moves. In comparison with these teleportation moves,
  the queens natural high mobility doesn't look so impressive anymore" [4].
- The lichess regression author: "the range of piece values generally is compressed compared to chess, since the
  ability to reintroduce pieces via drops makes weaker pieces more useful and valuable pieces quite exposed to
  attacks" [3].
- H.G. Muller (talkchess): a piece with no moves in a drop variant "should really have value 0. It might have a
  negative value", because it must be protected or it goes to the opponent's hand ([34] snippet, weak). The general
  point is sound: in drop variants a piece you cannot keep is a liability.
- "Contact checks and knight checks, which force the king to move as opposed to dropping pieces, are especially
  important" [1]. This is the mechanism behind every "knights up, queen down" table.

### 2.2 Human value tables

| Source | P | N | B | R | Q | Notes |
|---|---|---|---|---|---|---|
| Nilsson ("Gnejs"), via [1] | 1 | 2 | 2 | 2 | 4 | The most-cited bughouse table. Gnejs himself, interviewed, gives no numbers and says "Most of what I read about bughouse is crap" [21]. |
| FICS help file, 1995 GICS mirror (IanO / dogdog) [2] | 1 | 7 | 2 | 4 | 10 | "knights and queens dominate ... The bishops value is seriously diminished." Same numbers repeated in a chess.com forum thread ([26] snippet). **Corrected:** the live freechess.org revision of this file reads Q=4, N=2, R=2, B=2, P=1 (same as Nilsson). Treat 10/7/4/2/1 as one superseded revision; the ordering (N > R > B) is its only durable claim. |
| Sorsi (chess.com, bughouse coach) ([23] snippet only) | 1 | 2 | 2 | 2 | 4 | "depend very much on the position"; also "quantity beats quality" and "a few poor trades can decide a game" ([33] snippet). |
| okei, "Guide to Crazyhouse" [16] | 1 | 2 | 2 | 2 | 4 | Crazyhouse. "minor pieces including rooks are worth about 2 pawns"; queen "about two minors". |
| lichess "crazyhouse an overview" ([27] snippet only) | 2 | 3.5 | 3 | 4 | 6 | Crazyhouse rule of thumb; same ordering (N > B, Q ≈ 1.7 N). |
| lichess forum poster ([28] snippet only) | | 3-5 | 2-3 | | | Dropped knight "3-4 points when not developed or in closed positions and 4-5 points when dropped on a clear rank or open file"; dropped bishop 2-3. Weak, one anonymous poster. |
| ABCs of Bughouse (Jerry, 2021) [17] | | | | | | No table, but "Two pawns for a piece is a good deal", "give a queen for a knight to end a series of checks", "In a closed position pawns, knights and bishops are better" than the queen. |

Disagreement: the 1995 mirror of the FICS file puts the queen at 10 pawns, every other human source (including the current FICS file) at 4 to 6. The engine data (2.3) sides with
4. Knight-above-bishop is unanimous; knight-above-rook is claimed by FICS and the forum poster and is supported by the
Stockfish king-danger weights and Droste's in-hand values, but not by the lichess regression (R 3.05 > N 1.73, which
measures board material in engine games, not drop threat).

### 2.3 Engine value tables

**Hand-tuned classical bughouse/crazyhouse engines (centipawns).**

| Engine | P | N | B | R | Q | In-hand handling |
|---|---|---|---|---|---|---|
| Sunsetter v8 (bughouse and crazyhouse) [7] | 100 | 192 | 195 | 200 | 390 | `adjustInHand()`: base + per-piece bonus per type held: pawn 15 + 7/piece, N/B/R 20 + 10/piece, Q 40 + 20/piece [6]. King safety is multiplied by an opponent-hand bucket `getMaterialInHand()` in 2..11 (thresholds 80, 160, 260, 380, 530, 720, 960, 1260, 1630 cp) [6]. |
| Sjeng 11.2 [5] | 100 | 210 | 230 | 250 | 450 | Hand material added as a plain sum. King-safety table `ksafety_scaled[15][9]` indexed by opponent hand value/100 (clamped 0..14) and by attacked squares around the king; first row (empty hand) `{-5, 5, 10, 15, 50, 80, 150, 150, 150}`. Development tables are right-shifted by 2 when the opponent holds ≤ 200 cp ("2 pawns or less", not applied in bughouse), by 0 when ≥ 700 ("queen + minor, three minors or more"). |

So Sunsetter's queen is 3.9 pawns, Sjeng's 4.5; both put the rook a hair above the minors and the bishop at or above
the knight on the board. Both treat the opponent's hand as a multiplier on king danger, not as material.

**Multi-variant Stockfish (ddugovic), tuned by SPSA on lichess games** [8], [9], [10].

| | P | N | B | R | Q |
|---|---|---|---|---|---|
| Crazyhouse, midgame | 149 | 447 | 450 | 619 | 878 |
| Crazyhouse, endgame | 206 | 527 | 521 | 669 | 965 |
| Standard chess, midgame | 126 | 781 | 825 | 1276 | 2538 |
| Crazyhouse mg, pawn = 1 | 1 | 3.0 | 3.0 | 4.2 | 5.9 |
| Chess mg, pawn = 1 | 1 | 6.2 | 6.5 | 10.1 | 20.1 |

Queen/pawn falls from 20 to 5.9; knight/pawn from 6.2 to 3.0; knight ≈ bishop. Other crazyhouse-specific terms:
- `KingDangerInHand[PIECE_TYPE_NB] = { 79, 16, 200, 61, 138, 152 }`, indexed ALL, P, N, B, R, Q, each added per
  piece the opponent holds (the ALL entry is added for every piece). Total per piece in hand: **knight 279, queen 231,
  rook 217, bishop 140, pawn 95** [9]. The knight in hand is the most dangerous piece to face.
- Safe-check detection treats every empty square as a drop square for each piece type the opponent holds; pawn
  checks include drops [9].
- King danger is capped at `QueenValueMg` and reduced by 10% for the side to move [9].
- `material.cpp`: separate `QuadraticOursInHand` / `QuadraticTheirsInHand` tables with an "Empty hand" row
  (`-148` ours, `-40` theirs) and per-type pair terms, so the value of a hand piece depends on the rest of the hand [10].

**Fairy-Stockfish (generic rules engine; crazyhouse and bughouse share the drop path)** [11], [12], [13].
- Base values are the chess ones (P 126/208, N 781/854, B 825/915, R 1276/1380, Q 2538/2682 mg/eg) [13]. For
  `capturesToHand || twoBoards` the score is saturated: `value * 7000 / (7000 + value)`, then for `capturesToHand` the
  evaluation value is halved [11]. Saturated mg: P 124, N 702, B 738, R 1079, Q 1863, i.e. Q/P 15 and N/P 5.7. This is a
  much gentler compression than ddugovic's tuned table, consistent with Fichter's description of Fairy-Stockfish's
  crazyhouse evaluation as "only slightly adapted compared to chess" [19] and with the reported ~200 Elo gap to
  multi-variant Stockfish in crazyhouse (talkchess snippet, weak).
- **In-hand bonus**: `psq[pc][SQ_NONE] = score + make_score(35, 10) * (1 + !isSlider)`: +35/+10 for sliders,
  +70/+20 for pawns and knights [11]. No strongest-piece bonus in drop variants; king table doubled [11].
- `evaluate.cpp`: king danger adds `kingAttackersCountInHand` / `kingAttackersWeightInHand`; "Virtual piece drops" add
  `VirtualCheck * 500 / (500 + PieceValue[MG][pt])` (cheaper pieces count more) and 2000 for a presumed mate threat;
  "For drop games, king danger is independent of game phase, but dependent on material density" (scaled by
  `material_density() / 11000` when `captures_to_hand() || two_boards()`); `KingProximity` penalty `S(2, 6)` for pieces
  far from the kings in drop/two-board games; `DropMobility = S(10, 10)` per safe drop square in the opponent's half;
  the complexity/initiative term is skipped and replaced by a flat `+80 * captures_to_hand()` for the side to move;
  the king-danger mobility term is dropped entirely in captures-to-hand games [12].

**Learned values: Droste & Fürnkranz 2008 (TD self-play on a Sunsetter fork, depth 5)** [4].
- Table 6.1 (normalised so an on-board bishop = 3; "Avg. P.-Sq." = averaged from learned piece-square tables,
  "Piece-Value" = material-only run):

| | Avg. P.-Sq. | Piece-Value | Sunsetter 7e | Sjeng |
|---|---|---|---|---|
| Pawn | 1.26 | 1.34 | 1.54 | 1.30 |
| Knight | 2.54 | 2.35 | 2.95 | 2.74 |
| Bishop | 3.00 | 3.00 | 3.00 | 3.00 |
| Rook | 3.02 | 3.14 | 3.08 | 3.26 |
| Queen | 4.83 | 6.04 | 6.00 | 5.87 |
| Pawn (hand) | 1.03 | 1.49 | 1.54 | 1.30 |
| Knight (hand) | 2.48 | 2.96 | 2.95 | 2.74 |
| Bishop (hand) | 2.38 | 2.81 | 3.00 | 3.00 |
| Rook (hand) | 2.96 | 3.53 | 3.08 | 3.26 |
| Queen (hand) | 4.47 | 5.63 | 6.00 | 5.87 |

- Table 9.1 best-performing values, pawn = 1: **crazyhouse P 1.0, N 2.0, B 2.4, R 2.4, Q 3.8** (standard chess from
  the same method: 1.0, 2.7, 2.9, 4.3, 8.9) [4].
- "In particular knights, rooks and pawns are worth more on the hand than they are on the board, which is consistent
  with our practical experience with this game" [4]. In the material-only run: pawn 1.49 vs 1.34, knight 2.96 vs 2.35,
  rook 3.53 vs 3.14; bishop and queen are worth less in hand than on the board.
- Caveat stated by the authors: "we made the simplifying assumption that the value of an on-hand piece is independent
  of the other pieces that the player is holding. This is almost certainly not the case. If we do not hold any pawns
  yet, the value of a new pawn in hand will typically be higher than if we are already holding three pawns" [4].
- Queen-in-hand weights converged slowly because "the queen usually gets dropped again immediately after it has
  been taken to the hand" [4]: queens do not sit in pockets.
- Tournament (Table 6.2, 2000 games each): learnt vs traditional (chess values, hand = board) 67.65%; learnt vs
  Sunsetter 7e values 65.93%; learnt vs Sjeng values 66.20%; **piece-square vs learnt material-only 96.43%** [4].
- Piece-square findings: knight's best square moved from e5 to f6 ("A knight's check from f6 is often the start of a
  deadly attack, and clearly this is also the favourite spot for a dropping move"); the f-pawn "is clearly discouraged
  from moving because doing so will give the opponent the chance to start a dangerous attack with a pawn drop on
  f2"; rook usefulness restricted to g8/h8; king "clearly safest in the corner, where it has the least number of
  neighboring squares that need to be protected from drop pieces"; long castling nearly never happens; bishop and
  queen spread (max minus min square value) is much larger than in chess, i.e. **where** a piece stands matters more
  than **whether** you have it [4].

**Regression on engine games: lichess "Finding the value of pieces" (ianfab, 2022)** [3], method in [22].
- Logistic regression of game result on material counts over ~100k positions from Fairy-Stockfish NNUE self-play,
  positions after captures/promotions filtered out. Crazyhouse: **q 3.94, r 3.05, n 1.73, b 1.69, p 1.00, move 0.07**.
  Chess from the same pipeline: q 9.82, r 4.93, b 3.28, n 3.16, p 1.00, move 0.01 [3].
- In-hand and on-board pieces were **not** separated. Engine games, not human. Rook well above minors here, unlike
  the human 1/2/2/2/4 and unlike the king-danger evidence; the likely reading is that board rooks on g8/h8-type
  squares matter in engine play while hand rooks are not the ones that mate.

**Neural engines.**
- CrazyAra (Czech et al. 2019/2020): 34 input planes, pocket counts as constant planes normalised by a soft maximum
  of 32 per type, one promoted-piece plane per side; value target {-1, 0, 1} from the mover's view; no material
  term anywhere. "most engines' value evaluation is fundamentally based on material" while CrazyAra "seems to attach
  high importance to initiative" and "often seemingly, intuitively sacrifices pieces", attributed partly to the many
  aggressive human players in the 569,537-game lichess ≥2000 data set (45% one-minute games; 20 players in 46% of
  games). Risk named: a sacrifice "not sufficiently explored" [14].
- SixtyFour (Chi 2019): pocket counts as separate inputs capped at 16 P, 4 N, 4 B, 4 R, 2 Q per player; also no
  material term [15].
- Both support the lesson from [4]: a learned evaluator does not need, and should not be given, a material prior.

### 2.4 Pawns

- Drop targets: the seventh rank (one step from promotion) and squares next to the enemy king, to "draw the king
  out or yield a protected square next to the king which can be used for other pawns or pieces" [4]; blocking
  non-contact checks [1]; FICS: "Drop pawns near the back rank, so they can promote quickly" [2]. Pawns cannot be
  dropped on the first or last rank [1], [2].
- Weak squares: "Once you lose g2, you're dead"; "B2 is more problematic than g2 because the threat is to take it with
  check"; "Don't let them take on f7" [17]. Droste's learned pawn table says the same about f2 [4].
- Opening pawns: "never move any pawns except the e and d pawns" [2]; von Zimmerman: "not recommended to move pawns
  other than the d- and e-pawns" (pp. 21-24 via [1]).
- Value: hand pawn 1.49 vs board pawn 1.34 in [4]; "Two pawns for a piece is a good deal" [17]; "four pawns at hand are
  often worth a queen" ([28] snippet, weak). Diminishing returns in the hand are asserted by [4] and encoded by [10].
- Doubled pawns are less of a weakness than in chess because pawns in hand repair structure and a doubled pawn
  "controls squares of both colours" [16].

### 2.5 Knights

- "knights are the only pieces that can deliver [check] without being threatened by the king and without giving the
  opponent the chance to bar the check by dropping a piece" [4]; "it is impossible to defend against a knight check
  with a dropped piece" ([28] snippet); "Knights are wonderful attackers" [16].
- Engine weights: knight in hand is the largest king-danger contributor in multi-variant Stockfish (279 vs queen
  231) [9]; Fairy-Stockfish gives knights the doubled in-hand bonus [11]; H.G. Muller's test with "two N-B switchers in
  hand" found "The Knights seem to win that significantly" (Evert: "due to the always crowded board") [20].
- Best drop square f6/f3 (attacks the castled king's starting squares) [4]. "Knights don't belong on e7 or d7 unless
  you're white. They are subject to attack and they block your other pieces" [17].
- Sunsetter's waiting malus names rook, knight and pawn as the pieces a plan typically waits for [6]: the knight is
  the standard "I need a knight" request ("I need a knight" is the example partner message in [4]).

### 2.6 Queen

- Board value roughly halves: 3.8 to 4.5 pawns in every engine-derived table ([3], [4], [5], [7], [8]); "The queens
  value has almost halved" [4]. "Queens are worth less because minor pieces can attack them with tempo", about two
  minors [16]; intermediate players "lose the initiative by moving the queen to a square where it can be attacked
  with tempo" [16]; "it is often better to exchange the queen for a piece and positional compensation" than be chased
  [16].
- "A queen shouldn't always run from a knight" and "give a queen for a knight to end a series of checks or to keep the
  king from having to go up" [17]: the queen is a defensive currency.
- In hand the queen is worth slightly less than on the board (5.63 vs 6.04) and is dropped immediately on arrival [4];
  Sunsetter gives the queen the largest flat in-hand bonus (40 + 20/piece) [6]. Sources disagree on whether a queen
  in hand is a bigger threat than a knight in hand (Stockfish says no [9]).

### 2.7 Bishops and rooks

- FICS: "The bishops value is seriously diminished" [2]; it is the weakest in-hand piece in Droste (2.81 hand vs 3.00
  board) and ranks last in the lichess "queen > knight > rook > bishop" in-hand ordering ([28] snippet). But "bishops
  are good defenders" that "combine defence with attack" [16], and h3/h6 bishops "often lead to crushing attacks"
  (Sjeng comment) [5]; Sunsetter's development tables give bishops zero positional value [6].
- Rooks: 3.53 in hand vs 3.14 on board in [4] (hand rooks drop with check on the back rank; "R@g7"-type defensive
  drops in ([28] snippet)); on board their useful squares shrink to the g8/h8 corner [4]. Lichess regression has the
  rook at 3.05, clearly above the minors [3].

### 2.8 In hand versus on the board: hold or drop

- For: dropping at once. "I'm a big fan of putting down pieces as soon as you get them. As soon as you put them down,
  they control squares" [17]. Against: holding. "Some people like to save them for attacks and combinations" [17];
  "many players get caught up in 'dropping pieces' before using all of their 'regular pieces.' This is a mistake!"
  ([29] snippet); keep pieces for defensive air holes such as "P@g2 or R@g7", and "it is usually best to have a variety
  of pieces in hand" ([28] snippet).
- Engines side with holding being worth something: Fairy-Stockfish's `psq[pc][SQ_NONE]` bonus [11], Sunsetter's
  per-piece hand bonus [6], Droste's hand > board for P/N/R [4], Greg Strong's estimate that a hand piece's effective
  mobility is one third of the empty squares ([34] snippet, weak).
- Both sides are compatible: a piece's value in hand is its best future drop; dropping early converts that option into
  board control. Which is right depends on whether a better drop is coming, which is a search question, not a value
  question.

### 2.9 Trading across boards, feeding, flow

- Partner signalling: "trades are good" from an attacker, "trades are bad" from a player in trouble [1]; "+++" / "---"
  and "high flow" / "low flow" ([30] snippet); partners ask for a specific piece, for more trades, to hold a piece, or to
  stall [1]. FICS: "Black attempts to exchange pieces to reduce his opponents attack" while White keeps pieces on [2].
- Trades are one-sided: "Trades are not always a good idea. Usually they are good for only one side, the side that has
  the initiative. Don't initiate trades for no reason, only if they mean an advantage for your side" [17]. Trading can
  also "get rid of some of your opponent's pieces that may be used for an attack later" ([25] snippet).
- Sacrifices are a transfer to the enemy partner: accepting one "leaves the attacker with one fewer piece and the
  defender's partner with one more, giving that partner a chance to take the initiative" (von Zimmerman p. 113 via
  [1]); "Sacrifices are sometimes justified, but too many of them will cause you to lose on both boards"; "If you spend
  the whole game helping him mate you, he will mate you"; "Don't feed your partop" [17]. Early sacrifices are "a fairly
  parasitic way to play" and high-level play is "more reciprocal" ([24] snippet). On the other hand "if the opponent's
  king is exposed enough and you let your partner seek trades, two minor pieces is very little" ([24] snippet).
- Flow: "whoever controls the flow tends to win"; "the best flow is generally light but consistent"; neglecting
  development while demanding high flow "makes your own position worse and also constrains your partner"; strong
  players "can do a lot with just pawns and minor pieces" (low flow) ([24] snippet). "In a sparse position you can't
  feed" [17].
- "Don't trade what the opponent's partner needs" as such does not appear verbatim in any source read; it is the
  conjunction of the above (trades help the attacker; every capture is a drop for the diagonal opponent). The closest
  engine encoding is Sunsetter's *own-side* version: a malus when the plan needs a piece that is not in hand
  ("the Malus for Pieces used we would have to wait for in bughouse", skipped when the partner is sitting or has told
  it to go) and a preference for "mates where it does not have to wait for a piece" [6]. No classical engine was found
  that models what the diagonal opponent needs.
- Quantity versus quality: "quantity beats quality in bughouse and ... a few poor trades can decide a game" ([33]
  snippet). "Tactics that win material in standard chess don't win material in bughouse because the piece values are
  different" ([24] snippet).

### 2.10 Material versus initiative versus time

- "Initiative is more important than material. If a player has initiative they can drop their pieces in hand with
  check or with threats and dominate the board", and the initiative must be used quickly because if it runs out the
  opponent uses "their pocket full of pieces in hand" [16]. CrazyAra's observed style matches [14].
- Material-only evaluation is nearly worthless against positional knowledge in crazyhouse (3.6% vs 96.4%) [4].
  Stockfish multi-variant caps king danger at a queen's value and keeps a material imbalance table [9], [10];
  Fairy-Stockfish removes the chess initiative term and adds a flat 80 cp tempo bonus [12]; the lichess regression
  puts the move at 0.07 pawns (7x chess) [3]. "if both players have lots of material in hand a tempo might be the
  entire game ... With little in hand, tempo isn't that big, its more like normal chess" ([28] snippet).
- Time: "time is an even greater asset in bughouse than material" ([25] snippet); "This diagonal time advantage is
  more important than the difference on the clock between opponents on the same board"; above 2000 "a 10-second
  advantage can matter more than a positional edge" [1]; "Time will matter if there is a sparse board or a material
  advantage"; "Time is important, especially near the end of the game, if your side is down material"; "Don't pause
  to wait for material when you're down"; "Sitting and thinking may look similar, but sitting is using a timing
  strategy" [17]. "Stalling only works if you have more time than your partner's opponent, since otherwise you will
  flag first" [2]. "when you sit for a piece, make sure that piece hurts your opponent" ([33] snippet).
- Material density: Fairy-Stockfish scales king danger in drop and two-board games by board material density, not by
  game phase [12]. Human sources say the same thing from the other side: on sparse boards time, not material, decides
  [17].
- No source gives an exchange rate between seconds and pawns. That is an open empirical question for this project.

---

## 3. Implications for the engine

What original Hivemind (it04) has, from the code in `third_party/hivemind` (read, not inferred):
- Inputs per board (37 planes, `board2planes.py`, `planes.cc`): 12 piece planes; **10 pocket planes = count / 16 per
  piece type per side** (constant planes); 2 promoted-piece planes; en passant; on-turn; constant; 4 castling; the
  **boolean TimeAdvantage plane (31)**; last-move from/to; halfmove clock; repetition 2/3. No material feature, no
  history beyond the last move, no clocks (until spec v1), no in-flight pieces.
- Value: learned WDL + plys-to-end heads (`train_config.py`: `use_wdl`, `val_loss_factor` 0.15 in the RL preset); no
  material prior anywhere. Supervised on chess.com games with both players (and partners) ≥ 2200-2400
  (`game_reader.py`, `scrape_cc.py`); `time_control` is recorded per game but not an input. **Corrected/added:** per-move `moveTimestamps` are also
  stored and used: `generate_planes.py` replays them to set plane 31 from the real diagonal clock difference and to
  derive `pass` labels from human sits, dropping passes made by the team behind on time. So the human clocks shaped
  it04's sitting behaviour, through one bit.
- Search: joint MCGS over both boards. **A capture on one board adds the piece to the partner board's pocket in the
  same node** (`board.cc` line 111, `pos[1 - board_num]->add_to_hand_with_key(p)`), i.e. transfer is instantaneous
  inside the tree; pass legality and the bughouse mate rule depend on the boolean TimeAdvantage (`joint_action.h`,
  `board.cc` 264-430).
- Planned (spec v1/v2): 4 clock planes per board, 2 elapsed-think planes, `TreeMoveMs` clock decrement per macro-ply,
  `think_gain` head, `arrivals` head (pieces per type arriving in each of the 4 pockets within 5 s).

| # | Idea from the sources | Does it04 capture it? | Concrete change |
|---|---|---|---|
| 3.1 | Compressed, position-dependent piece values; no fixed table is right ([3], [4], [8]) | Yes in principle: pure learned value, no material term, same design as CrazyAra [14]. How well is unknown: never measured. | Measure it. Add `tools/implied_values.py`: for a sample of recorded positions, add or remove one piece of each type from each pocket (and from the board), record ΔV of the root value, fit per-type values. Report them per training round next to the tables in 2.3. Expect Q/P in 3.5-6 and knight ≥ bishop in hand; a round that breaks that ordering is a red flag. Zero engine change. |
| 3.2 | In-hand marginal value depends on the rest of the hand ([4], [10]) | Learnable from the 10 count planes. But a count encoded as `count / 16` puts the first knight at 0.0625 and the step from 0 to 1 (the step that matters most per [4]) is tiny relative to pawn counts. | Cheap input addition in the widened stem: 10 **"has at least one"** binary planes per board (or counts clipped to 4 as in [15]). Test with the parity-preserving zero-init path, ablate against the plain widened net after one round. Low cost, uncertain gain; mark as an experiment, not a fix. |
| 3.3 | Hand piece ≠ board piece; promoted pieces demote on capture ([4], [11]) | Yes: pocket planes are separate from board planes; promoted planes exist (channel 22/23). | None. |
| 3.4 | Every capture is a drop for the diagonal opponent; "don't feed your partop"; sacrifices are transfers ([1], [17]) | Yes, within the search horizon: the tree moves the captured piece into the other board's pocket at once (`board.cc` 111), so the value net on the next node sees the enemy partner's new piece. Beyond the horizon it is the value head's job and untested. | (a) The v2 `arrivals` head is the right auxiliary target: it forces the trunk to predict pocket inflow for all four pockets; keep it, and add it to the implied-value probe (3.1). (b) Known oversight: transfer in the tree is **instantaneous**, but in the arena a piece lands with latency and only when the partner actually moves. With `TreeMoveMs` the tree already charges clock per macro-ply; add a **landing delay** so a captured piece becomes droppable only after the next macro-ply on the receiving board, and ablate. This matters for exactly the "sit for the knight" lines the sources describe. |
| 3.5 | Trades help the side with initiative; attacker wants flow, defender wants none ([1], [17], [30]) | The joint policy chooses both boards' moves together, so "trade on A because B's attacker needs pieces" is representable. Whether it04 learned it from human games where partners decided separately is unknown. | Self-play with one agent per team (spec) is the correct fix: the credit for a trade on A that mates on B flows back through the shared value. Add a diagnostic: trade rate on the non-attacking board conditioned on the partner board's root value (should rise when the partner is winning, fall when losing). |
| 3.6 | Flow should be light but consistent; what matters is recent inflow, not static pockets ([24], [17]) | No: no capture history, only last-move squares. The net cannot see that a knight arrived 1 s ago versus 30 s ago. | Input experiment: per board, 5 planes "pieces of type X landed in this pocket in the last 5 s" (team-relative, 20 planes total, or 2 summary planes: arrivals in last 5 s to our pocket / to theirs). The arena recorder already knows landing times (it labels `arrivals`). Zero-init widening keeps parity. |
| 3.7 | Knight in hand is the most dangerous piece; king safety scales with opponent's hand ([9], [5], [6]) | Learnable; pocket planes plus king planes suffice. Whether it04 weights the opponent's hand correctly is unknown. | Covered by the probe in 3.1 (ΔV for adding a knight to the opponent's pocket should be the most negative of the five). Also a tactical test set: positions with mate-by-drop available only if a specific piece is in hand, from recorded games; policy top-1 accuracy per piece type. |
| 3.8 | Material matters less than initiative and tempo; the move is worth 7x more than in chess ([3], [4], [16]) | On-turn plane exists; TimeAdvantage boolean exists. Material is already not a term. The v2 elapsed-think planes and v1 clock planes make "who moves next and when" explicit. | Keep v1/v2. Add an evaluation: the WDL head's calibration split by (our pockets material minus theirs) buckets. If the net is miscalibrated at large positive material (the "I am up a queen but about to be mated" case the sources describe), add a loss weight on those buckets or sample them more in self-play. |
| 3.9 | Human-data networks inherit human values and human aggression; learned values beat human and engine tables ([14], [4]) | it04 is supervised on ≥2200-2400 chess.com humans, so its implied values are human values. | Expect drift during RL; the probe in 3.1 is the instrument. Guard against CrazyAra's failure mode ("a sacrifice not sufficiently explored" [14]): in the gate, log the share of lost games where the engine was ≥ 1 piece up by its own implied values 10 plies before mate. |
| 3.10 | Time beats material; diagonal clock gap matters more than same-board gap; time matters more on sparse boards and when down material ([1], [17], [25]) | No: the boolean TimeAdvantage plane is the only time input, and it is a hand-written rule we are removing. | v1 clock planes give the net all four clocks (both board blocks carry our seat's and the opponent's clock), so the diagonal difference is computable at the merge. Add to evaluation: ΔV for ±10 s on each of the four clocks, by board material density bucket. The sources predict the clock sensitivity is highest on sparse boards and when behind in material; if the trained net does not show that, the planes are not being used. |
| 3.11 | Holding versus dropping is a search question; hand pieces carry an option value ([11], [17], [28]) | Search decides drops; the value net implicitly prices the option. The FreePass option (spec) makes "hold and wait" a legal action without the `TimeAdvantage` gate, which is needed for "sit for the knight" to be learnable. | None beyond spec. Watch for the opposite failure in RL: a net that never drops and sits too long. The `think_gain` head and clock decrement in the tree (`TreeMoveMs`) are the counterweights. |
| 3.12 | Pawn-specific knowledge: 7th-rank drops, f2/g2/b2 weaknesses, e/d pawns only ([2], [4], [17]) | Policy learned from strong humans; nothing structural missing. | None. Could be a cheap sanity test set (P@-7th-rank mates, g2 collapses) for the gate. |
| 3.13 | Seconds-to-pawns exchange rate is unknown everywhere | Unknown, and nothing in it04 can answer it. | The clock net can: once trained, the ΔV-per-10-s and ΔV-per-pawn-in-hand probes from 3.1 and 3.10 give the first measured exchange rate in the literature. Worth a line in the gate report. |
| 3.14 | Pocket normalisation by 16 (`MAX_NUM_DROPS`); CrazyAra used a soft max of 32 [14], SixtyFour 16/4/4/4/2 per type [15] | Pockets above 16 of a type are impossible in bughouse (16 pawns exist per board), so no clipping loss. | None. The resolution question is 3.2. |

Priority if the 10k GPU hours have to be rationed: 3.1 (free, instruments everything else), 3.4(b) landing delay in
the tree (the one modelling error found that is outright wrong rather than merely unlearned), 3.6 recent-arrival
planes, 3.2 presence planes, then the diagnostics in 3.5, 3.8, 3.10.

---

## 4. Sources

Read in full unless marked (snippet only).

1. Wikipedia, "Bughouse chess", https://en.wikipedia.org/wiki/Bughouse_chess — strategy section: drop mobility
   argument (von Zimmerman 2006), Nilsson 1/2/2/2/4, Sunsetter and Sjeng values, trades-good/bad, sacrifice as
   transfer, diagonal time advantage, no endgame.
2. FICS help "Bughouse Strategy" (IanO, dogdog, ed. Friar, 1995), mirror
   https://www.unix-ag.uni-kl.de/~chess/gicshelp/node170.html — Q10 N7 R4 B2 P1, bishop diminished, Black trades
   down, drop pawns near the back rank, stalling only with more time than the diagonal opponent.
3. ianfab, "Finding the value of pieces", lichess blog, 2022-06-05, https://lichess.org/ublog/PByOBlNB/redirect —
   logistic-regression crazyhouse values q 3.94 r 3.05 n 1.73 b 1.69 p 1.00 move 0.07; chess comparison; compression
   argument.
4. S. Droste, J. Fürnkranz, "Learning of Piece Values for Chess Variants", TU Darmstadt TR TUD-KE-2008-07 (PDF via
   https://ke-tud.github.io/bibtex/attachments/single/185; journal version ICGA J. 31(4) 2008) — learned crazyhouse
   board and hand values (Tables 6.1, 9.1), tournament results (6.2), piece-square findings, hand-composition caveat.
5. Sjeng 11.2 `eval.c`, https://sources.debian.org/data/main/s/sjeng/11.2-6/eval.c — board values 100/210/230/250/450,
   `ksafety_scaled` by opponent hand, development scaling thresholds, h3/h6 bishop comment.
6. Sunsetter `evaluate.cpp`, https://raw.githubusercontent.com/georgvonzimmermann/Sunsetter/master/evaluate.cpp —
   `adjustInHand()` bonuses, `bughouseSitForEval()` waiting malus, `bughouseMateEval()`, king safety × hand bucket.
7. Sunsetter `bughouse.cpp`, https://raw.githubusercontent.com/georgvonzimmermann/Sunsetter/master/bughouse.cpp —
   piece values 100/192/195/200/390, NK/BC/DE factors, "ptell sit" behaviour.
8. ddugovic Stockfish `types.h`, https://raw.githubusercontent.com/ddugovic/Stockfish/master/src/types.h — crazyhouse
   `*ValueMgHouse/EgHouse` and chess values.
9. ddugovic Stockfish `evaluate.cpp`, https://raw.githubusercontent.com/ddugovic/Stockfish/master/src/evaluate.cpp —
   `KingDangerInHand`, drop-aware safe checks, king-danger cap and side-to-move reduction.
10. ddugovic Stockfish `material.cpp`, https://raw.githubusercontent.com/ddugovic/Stockfish/master/src/material.cpp —
    `QuadraticOursInHand` / `QuadraticTheirsInHand`, empty-hand terms.
11. Fairy-Stockfish `psqt.cpp`, https://raw.githubusercontent.com/fairy-stockfish/Fairy-Stockfish/master/src/psqt.cpp —
    saturation `7000/(7000+v)` for capturesToHand/twoBoards, halving, in-hand bonus `S(35,10)*(1+!isSlider)`.
12. Fairy-Stockfish `evaluate.cpp`, https://raw.githubusercontent.com/fairy-stockfish/Fairy-Stockfish/master/src/evaluate.cpp
    — in-hand king attackers, virtual drops, material-density scaling, KingProximity, DropMobility, +80 tempo, no
    initiative term.
13. Fairy-Stockfish `types.h`, https://raw.githubusercontent.com/fairy-stockfish/Fairy-Stockfish/master/src/types.h —
    base piece values.
14. J. Czech et al., "Learning to play the Chess Variant Crazyhouse above World Champion Level with Deep Neural
    Networks and Human Data", arXiv:1908.06660, https://ar5iv.labs.arxiv.org/html/1908.06660 — input planes, pocket
    normalisation, value target, initiative observation, data set statistics, Stockfish match results.
15. G. Chi, "Exploring the Performance of Deep Residual Networks in Crazyhouse Chess", arXiv:1908.09296,
    https://ar5iv.labs.arxiv.org/html/1908.09296 — per-type pocket caps 16/4/4/4/2, no material term.
16. okei (with MMichael, theLazyMD), "Guide to Crazyhouse",
    https://zhchess.blogspot.com/p/you-know-rules-of-crazyhouse-but-how-to.html — 1/2/4 values, knights attack,
    bishops defend, queen chased with tempo, initiative over material, defensive drops.
17. Jerry, "The ABCs of Bughouse", 2021, https://abcsofbughouse.blogspot.com/2021/12/the-abcs-of-bughouse.html —
    trades one-sided, don't feed partop, queen for knight, g2/b2/f7, time vs material, sitting; no credentials stated.
18. Chessprogramming wiki, "Crazyhouse", https://www.chessprogramming.org/Crazyhouse — engine list and references
    (Droste & Fürnkranz, Czech).
19. talkchess, "The Stockfish of shogi" thread, https://www.talkchess.com/viewtopic.php?p=825044 — F. Fichter on
    Fairy-Stockfish's crazyhouse evaluation being "only slightly adapted compared to chess" and SPSA tuning.
20. talkchess thread, https://talkchess.com/viewtopic.php?p=690047 — H.G. Muller: knights beat bishops in hand in a
    switcher test; promote only to Q or N in crazyhouse.
21. Chronatog, "The ICS Interviews No. 013 – Gnejs", https://chronatog.com/?p=484 — Gnejs on analysing "what are the
    pieces worth in bughouse" without giving numbers; distrust of written bughouse theory.
22. ianfab, `chess-variant-stats` README, https://github.com/ianfab/chess-variant-stats — method behind [3]
    (generate games, logistic regression, ≥100k positions).
23. Sorsi, "Weak and Strong Pieces in Bughouse", chess.com blog (snippet only) — N/B/R ≈ 2 pawns, Q ≈ 4, values
    position-dependent.
24. JarlCarlander, "Esoteric wisdom of the ancient bughouse masters" and "How to Theorize in Bughouse", chess.com
    blogs (snippet only) — flow control, light consistent flow, low-flow play, early sacrifices parasitic, "tactics
    that win material in chess don't win material in bughouse".
25. key, "The Basics of Bughouse", chess.com blog (snippet only) — "time is an even greater asset in bughouse than
    material"; trading removes attackers.
26. chess.com forum "Bughouse Strategy" (snippet only) — repeats Q10 N7 R4 B2 P1.
27. lichess blog "crazyhouse an overview" (snippet only) — P2 B3 N3.5 R4 Q6 rule of thumb.
28. lichess forum "Value of Pieces in Crazyhouse" and related (snippet only) — in-hand ordering Q > N > R > B, dropped
    knight 3-5, four pawns ≈ queen, air holes, variety in hand, tempo worth more with full pockets.
29. chess.com article "7 Doubles (Bughouse) Chess Tips For Beginners" (snippet only) — don't drop before using
    board pieces; sitting defined.
30. JarlCarlander, "Partner Requests in Bughouse", chess.com blog (snippet only) — "+++" / "---", high/low flow.
31. Wikipedia, "Crazyhouse", https://en.wikipedia.org/wiki/Crazyhouse — rules only (no strategy section); used for the
    pawn-drop rank rule.
32. B. Woo, P. Sweetser, M. Aitchison, "HiveMind: Learning to Play the Cooperative Chess Variant Bughouse with DNNs and
    MCTS", IEEE CoG 2023, https://researchr.org/publication/WooSA23 — bibliographic record only; abstract and PDF not
    reachable, so Hivemind facts above come from the code in `third_party/hivemind`.
33. Sorsi, "Advanced bughouse tips", chess.com blog (snippet only) — quantity over quality, poor trades decide games,
    sit only for a piece that hurts the opponent.
34. talkchess threads on hand-piece modelling (H.G. Muller, Greg Strong), https://talkchess.com/viewtopic.php?p=690065
    (snippet only) — hand mobility ≈ 1/3 of empty squares; immobile pieces worth ≤ 0 in drop games.
