# 04. Tactics and mating patterns in drop chess and bughouse

Written 2026-11-17 for the custom engine (one agent, both boards, real clocks; see `docs/custom_engine_spec.md`,
`docs/custom_engine_spec_v2.md`, `docs/beat_stock.md`). Covers: drop mates, smothered and back-rank patterns, f7/f2
attacks, pawn-drop attacks, defence by drops, king safety, blockable vs unblockable checks under bughouse rules, the
2024 Chess.com championship (Vasquez/Helmsknight vs Tang/Gurwich), and what annotated top-level material exists.

Access notes, so the evidence grade is clear. Every chess.com page (blogs by JarlCarlander, Sorsi, key, 23Dragons,
tree55555; the two 2024 championship news reports) refused automated fetching (WebFetch failed, Chrome tab showed an
error page three times, uschess.org and lichess.org also refused). For those pages only the search-engine snippets were
readable; every claim that rests on them is marked **[snippet]**. Everything else below was read in full from the
page itself. "Weak" marks a single informal or anonymous source.

---

## 1. Summary

- **The mate rule is the whole tactical grammar.** Online bughouse (FICS, ICC, chess.com) and the WCCC laws agree: a
  check that *could* be blocked by a dropped piece is not mate, even if the defender's hand is empty now [3][5][13][1].
  So a decisive mate needs a contact check, a knight check, a double check, or a check whose every interposition
  square is already occupied. Everything attackers do is aimed at reaching one of those four.
- **Checks that force a king move are the attack.** "As long as he's in check, I'm winning" (FICS strategy file) [1][4].
  Distance checks (rook/queen/bishop from afar) are "greatly reduced in value because the dropped pieces always provide
  blockers" [snippet, 30]; they still matter as tempo and as a way to make a defender with an empty hand *sit*.
- **The canonical mates**: (a) sacrifice on f7/f2, then knight check, then drop into the vacated or adjacent square
  ("knight check, then drop a queen or rook on the square the king just left") [snippet, 23]; (b) smothered mate
  `N@h6+ Kh8 R@g8+ Rxg8 Nxf7#` [9][10]; (c) g7/h7 clamp-and-mate with pawns and queen (`P@h6`, `Q@h7+`, `P@g7`,
  `R@g8#` family) [snippet, 24; weak 19]; (d) "yo-yo"/magnet mates started by a "hook check" that drags the king
  forward and back, ending in a contact drop (`N@h3+ ... N@f3 ... N@h4 Kh3 Q@g2#`) [12]; (e) back-rank mates that only
  work when a rook/queen is *adjacent* to the king, the king is smothered (often by a dropped 7th-rank pawn), and the
  defender's hand is empty [11].
- **Pawns are the best defenders and underrated attackers.** They are cheap, replaceable, drop on ranks 2-7, block
  sliding checks, make "pawn walls" (`P@c6`, `P@f6`) [snippet, 25], clamp flight squares (`P@e6` vs Ke8) [19], fork
  from the 7th, and promote (promoted pieces revert to pawns when captured, so a queen made from a pawn is a
  "counterfeit" that gives the enemy only a pawn) [3][5].
- **Piece values**: the most-cited scale is Gnejs's P=1, N=B=R=2, Q=4 [3][4][17]; Sunsetter and Sjeng used
  roughly 1 : 1.9-2.3 : 1.9-2.1 : 2-2.5 : 3.9-4.5 [3]; the FICS files put the knight above the rook and the bishop
  near a pawn [1][4]; one experienced author puts B and R slightly above N at high level [17]. There is no consensus
  beyond "pawn and knight up, long-range pieces down, two minors about a queen".
- **King safety is about holes, not castling.** Do not castle (it confines the king to one wing); avoid f2/f7 (KB2)
  weakness, holes on the 2nd/7th rank, fianchettos and c/f pawn moves, because any hole is filled by a drop at once
  [1][4][10]. Ask every move: "what piece drop mates me next move?" (weak, [19]).
- **Material is doubled and sacrifices must pay.** A capture helps you *and* feeds your partner [17]; sacrifices
  that do not reach a decisive check hand the other board the material [17][30]. A defender should play fast,
  consider returning material, and keep a piece in hand ("a knight in the hand is worth two on the board") [1][4].
- **Time is the diagonal clock.** What matters is your clock against your partner's opponent's ("partopp"), not your
  opponent's [3][27][28]; at 2000+ a 10-second edge can outweigh a positional advantage [3]; no increment, so every
  second sat is gone [snippet, 28]. Telling the partner to sit is itself a defensive move: "If you tell your partner
  to sit your opponent can't checkmate you" (Gulamali) [snippet, 26].
- **Top-level annotated material is thin.** The 2024 Chess.com championship (2+0, 8 teams, double elimination) was
  won by Liang/Xiong 11.5-8.5 over Chiu/Hauge [14]. Helmsknight/Vasquez beat Tang/Gurwich in qualifier 4 (73/80)
  [snippet, 20]; no game record or annotation of that match was reachable. Tang's partner in the knockout was
  Dubansky, Vazquez's was Ferguson [14].

---

## 2. The knowledge

### 2.1 Rules that create the tactics

| Rule | Text / source | Consequence |
|---|---|---|
| Drop with check and mate is legal | "Pieces in reserve may be dropped on any vacant square, including squares where the piece delivers check or checkmate" [7]; ICC same [snippet, 29]. Some OTB clubs (Canberra, c.1983) voted "no drop for mate" because "it requires greater skill to finish the game" [6]. | All drop-mate patterns below are legal online and in the arena. |
| Blockable check is not mate | FICS: "there must be no *possibility* of avoiding checkmate by a drop interposition ... he/she may later get a piece from his/her partner" [5]. chess.com: "A check that could be blocked by a piece drop is not considered checkmate, even if the player doesn't have a piece in the bank" [13]. WCCC laws (Art. 7-8): pieces not on the board "count as potentially in stock", so mate "usually requires a contact check, a knight check, or a multiple check" [15]; the 2007 PDF: "not considered to be checkmated if he could parry the check with dropping a piece potentially" [snippet, 16]. | Only contact, knight, double, and fully-occupied-line checks mate. A blockable check on an empty-handed defender stalls that board instead; the clock decides. |
| Pawn drops | Not on rank 1 or 8 [3][5][7]; a pawn dropped on rank 2 may advance two squares and be taken en passant [3][7]; a pawn dropped on the 7th promotes next move [5]. | 7th-rank pawn drops are a mate/promotion threat in one; 2nd-rank drops are a flexible shield. |
| Promoted pieces revert | Captured promoted pieces return as pawns [3][5][13]. | Promoting is cheap to lose; the opponent's team gains only a pawn. Corrected: it is Hivemind's *mate probe* (`mate_probe.cc`) that restricts promotions to Q or N; the policy head uses the 4672-label AlphaZero encoding, which includes under-promotions. |
| Dropped rook may castle | Rook dropped on a1/h1/a8/h8 may castle [3] (crazyhouse differs: "A dropped rook can't castle" [8]). | Rare; castling is discouraged anyway. |
| Stalemate | WCCC: stalemate does not apply; a player with no legal move waits [15]. FICS/engine: a stuck board is a loss unless the team is ahead on time (stock Hivemind) or waits (arena free mode). | See 2.8. |
| Clocks and end | Match ends when either board ends; win by mate, resignation, flag [3][7]. Championship 2+0, "two point advantage needed to win" a match [14]; WCCC 3 min [15]; nwchess 5 min [18]. No increment is the norm [snippet, 28]. | The whole game is a race between two mates and two flags. |

### 2.2 Blockable vs unblockable checks

Classification (derived from the rules above; the FICS file states the conclusion: "only contact and knight checks can
give a 'decisive' checkmate" [5]):

1. **Contact check**: checker adjacent to the king (any piece type including a pawn or a dropped queen next to the
   king). No interposition square. Mate iff the king has no flight and the checker cannot be captured. Attackers want
   the checker protected; "exclude undefended contact drops which the king can simply take" is the first filter in the
   engine's own supply-search heuristic (`supply_search.cc`).
2. **Knight check**: unblockable by definition; "the knight is a very powerful piece, often worth more than a rook"
   [1][4]; the knight "cannot be blocked by anything and its offensive value is more manifest" [8]. Knights also hit
   the queen with tempo from a drop (`N@e2+` forking Kg1/Qg3) [9][10].
3. **Double check**: only a king move helps [15]; Hivemind's own test suite uses "Nf7 and Qe8" double check on a
   smothered g8 king as a mate case (`test_mate_detection.cc`).
4. **Line check with every between-square occupied**: mate if no flight/capture. The crazyhouse back-rank analysis
   gives the practical conditions: a rook or queen *adjacent* to the king, the king smothered (e.g. by a pawn on the 7th),
   and the opponent with no pieces in hand to block [11]. The third condition is what the bughouse rule removes: in
   bughouse the *possibility* of a future piece counts, so a back-rank check from distance is never mate; only the
   adjacent (contact) version is.
5. **Blockable line check**: never mate online. Its uses: gain tempo, force a piece onto a bad square, force the
   defender to spend a piece from hand, or, against an empty hand, force the defender to *sit* while the diagonal clock
   runs. Dropped pieces "also block escape squares, which is why mates and attacks from a distance are greatly reduced in
   value" [snippet, 30].

Disagreement: Hivemind's stock mate detection is stricter than the FICS/chess.com rule. Stock treats a blockable check
as mate when the partner cannot *immediately* capture a piece that would enable a legal block (and, with time
advantage, when some opponent reply would leave no such capture) (`board.cc is_checkmate`,
`can_partner_provide_blocking_piece`; RULES.md: "A blockable mate is terminal only when the defending partner cannot
capture a piece that enables a legal block"). Free mode (`FreePass`) uses the arena rule, identical to FICS/chess.com: a
single sliding check with an empty between-square is never a loss. Fairy-Stockfish's `twoBoards` mode goes the other
way and lets a side *virtually drop* a piece it does not hold (`allow_virtual_drop`), assuming the partner will deliver
it; the engine excludes virtual drops from its own move lists and from the mate probe (`mate_probe.cc`, "both sides play
with the hands they actually have").

### 2.3 Piece values

| Source | P | N | B | R | Q | Notes |
|---|---|---|---|---|---|---|
| Gnejs (André Nilsson), via Wikipedia [3], FICS strat file [4], chessbraingame [17] | 1 | 2 | 2 | 2 | 4 | Most cited. "Two minor pieces slightly stronger than a queen, two pawns slightly stronger than a knight" [17]. |
| FICS gicshelp node170 [1] | 1 | 7 | 2 | 4 | 10 | Older scale; "bishops are often no better than pawns, and a pawn's promotion potential can be worth more than a bishop". The two FICS mirrors disagree numerically but agree on ordering N > R > B ~ P. |
| Sunsetter (engine) [3] | 100 | 192 | 195 | 200 | 390 | |
| Sjeng (engine) [3] | 100 | 210 | 230 | 250 | 450 | |
| Crazyhouse community guide [9][10] | 1 | 2 | 2 | 2 | 4 | "A queen is worth about two minors, because minors can hit it with tempo." |
| chessbraingame author [17] | | | | | | "bishops and rooks are slightly stronger than knights in high-level games" (contradicts FICS). Weak, one author. |

Why: reserve pieces have unrestricted mobility, so "the pawn relatively gains importance ... and long-range pieces like
the queen or the rook lose relative value, due to the constantly changing pawn structure" [3]. Winning material is
"effectively doubled" because it also arms the partner [17]. There are no endgames, so material matters less than
initiative [3][9].

### 2.4 Attacking patterns

**f7/f2 sacrifice family.** "Sacrifices on f7 or f2 are probably the most common tactics in Bughouse ... to lure your
opponent's king out" [snippet, 22]. The FICS file: "many mating attacks start with a sacrifice on KB7 followed by a knight
check" [1][4]; White's standard setup aims at f7 (the "Mongolian Attack": e4, d4, Bc4, Be3, Nf3, Nbd2, Qe2) [4]. The
follow-up motif: "first give check with a knight, and then on the next move drop a piece (usually a queen or a rook) on
the square just vacated by the opponent's king" [snippet, 23]. The attacker typically needs "a knight and queen" to mate
after `Bxf7+ Kxf7` [17][snippet, 22]. Costs when it fails: "losing double material, losing time waiting for pieces,
and pressuring the partner into fast, weaker moves" [17]; "with good communication, accepting an f7 sacrifice is not a
serious problem" [17]. One analysis says that after `Bxf7+` the routine `Ng5+` is "a common mistake" [snippet, 21].
Sorsi's attack-patterns post notes a related mistake, dropping a pawn on f2 when h2 was right [snippet, 2].

**Smothered mates.** Canonical: `N@h6+ Kh8 R@g8+ Rxg8 Nxf7#` [9][10] (JarlCarlander's "commonly missed" post gives the
same shape: king to h8, rook dropped on g8 forces the capture, `Nxf7#` [snippet, 23]). On the other wing/colour a knight
on h3 followed by `Nxf2#` [snippet, 31]. Shape: a knight check that drives the king into the corner, a queen or rook drop
that forces a self-block, a knight mate. Needs: the king's flight squares occupied by its own pieces (or by dropped
pawns), a knight in hand or in range, one more piece to force the block.

**g7/h7 clamp.** "One very common strategy in bughouse is to target the g7 square, even when your opponent's king isn't
castled" with a pawn on h6; a pawn dropped on g7 forces the king out and a queen mate follows; "wait until the Bf8 moves,
and then drop a knight on h5" [snippet, 24]. Against a castled king with Q and pawns in hand: `P@h6`, `Q@h7+`, `P@g7`,
`R@g8#`-type sequences (weak: anonymous AI-generated glossary [19]; the same family appears in the chess.com snippet).
Clamping drops: `P@e6` against Ke8 (or `P@e3` against Ke1) to remove flight squares, then `Q@f7` / `B@g6+` [19, weak].

**Yo-yo / magnet mates and the hook check.** From a crazyhouse example (Black has N,N,N,P,Q in hand, White king g1):
`N@h3+` (the "hook check" that starts the sequence; if Kh1 then `Q@g1` Rxg1 `Nxf2#`), after gxh3 `N@f3` (overloads the
g-pawn; the h3-knight covered f2), then whichever of Kh1/Kg2, `N@h4`, Kh3, `Q@g2#`. The author says this exact piece
combination "is not especially rare" [12]. The lichess guide names related patterns (magnet mate = "drawing the king to
a checking square", cookie-crumb, moon mate) without sequences [9].

**Piece storm.** "Dropping pieces one after another with check" [3] (von Zimmermann p.20). Pre-condition: a stock of
pieces in hand or arriving; pay-off: each check is a tempo that the defender cannot use to drop a defender.

**Pawn-drop attacks.** Drop pawns near the back rank so they promote quickly [1][4]; pawns on the 7th/2nd "gain
strength as the board empties" and make forks easy [snippet, 32]; a dropped pawn can fork or give check from deep in the
position [8]; "pawns can be dropped on any square, including the seventh rank, which makes them more valuable than in
regular chess" [9][10]. FICS slang "Themed" = two pawns promoted side by side [4].

**Quiet moves when checks run out.** "Use quiet moves that cut off the enemy king's escape when checks run out"
[snippet, 33]; keep attacking pieces safe from capture so the king stays boxed [snippet, 33].

### 2.5 Defence by drops

- **Pawn walls.** "The most useful and underestimated defensive tactic in bughouse", e.g. `P@c6`, `P@f6` in front of a
  king [snippet, 25]. Pawns "are great defenders ... because they aren't worth much, they're easily replaceable, and they
  shield the king well" [snippet, 33]. When a king is checked from distance by R/Q/B "dropping a pawn next to the king
  becomes defensively useful" [8].
- **Plug holes immediately.** "Plug any hole on g2/g7 as soon as possible, even if it means you have to sac a piece
  for only a pawn" [snippet, 30]. "Avoid holes anywhere ... opponents can drop pieces into them immediately" [1][4].
- **Keep a piece in hand.** "A knight in the hand is worth two on the board" [1][4]; "you can also drop pieces around
  your king to guard it from serious threats" [9][10]. Note the clock cost of holding (section 2.8).
- **Bishops as defenders.** "Ba5+ can cover c7, or Bxa8 continues to defend g2" [9][10]; "it is nearly always possible
  to prevent a sacrifice on your own King by dropping a Bishop, or moving the Queen" [snippet, 30].
- **Counterattack as defence.** "If you are worried about @h3, maybe play @h6" [9][10].
- **Accepting or returning material.** "Accepting a sacrifice can be lethal, but it gives the partner a chance to take
  initiative" [3]; "sometimes it is correct to return material to keep your king safe" [17].
- **Play fast when defending.** "A defender should communicate which pieces hurt most and play quickly" [3].
- **The one-move question.** Ask "what piece drop mates me next move?" every turn (weak, [19]); warn your partner
  before they feed a queen ("rook mates me", "no queens!") [3][19].
- **The partner's sit as defence.** "If you tell your partner to sit your opponent can't checkmate you" [snippet, 26];
  the partner "can sit (even if he or she didn't tell you to) if you see that a certain piece is going to hurt him or
  her" [snippet, 30].

### 2.6 King safety principles

- "King safety is paramount, so don't leave weak squares next to your king", especially KB2 (f2/f7), "since diagonal
  weaknesses are open to dropped pieces" [1][4].
- Castling "is generally bad in bughouse because it restricts the king's escape routes" [1][4]; the chess.com series has a
  whole post titled "Don't Castle" [7]. The crazyhouse guide is softer: "castling is not required" [9].
- Pawn moves other than d- and e-pawns are "not recommended" [3]; c- and f-file pawn moves are risky "which rules out
  lines like the Sicilian and Queen's Gambit"; fianchettos "are impractical because pawns can occupy them quickly" [4].
- Colour complexes: "if all your pawns are on light squares, you must take care to defend dark square weaknesses"
  [9][10].
- After an early queen trade "it is usually unwise to reintroduce the queen too soon" because dropped minors harass it
  with tempo [8]; "avoid letting the queen be chased with tempo" [10].
- Roles: "White generally aims to mate, and Black aims to hold together by trading pieces to reduce White's attack"
  [1][4]. Named Black defences: Federkevic (1.e4 e6 2.d4 b6), Barbeau Counter Attack (1.e4 e6 2.d4 Qh4), Fortress
  (1.e4 e6 2.d4 d6, "the most common bughouse opening") [4].
- Larry Kaufman on crazyhouse: "it is almost certain that White has a forced win" [16]; the crazyhouse Four Knights
  Italian with Bg5 is "often lost for Black because of tempo" [12]. First-move advantage is large in drop chess.

### 2.7 Initiative versus material

- "Initiative is paramount" [8]; "the key concept of bughouse" [1]. "Advantages of attacking: drops can lead to quick
  checkmate, the attacker holds the initiative, and defensive mistakes cost more than attacking ones" [3] (von
  Zimmermann p.109).
- "Beginners often chase material when they should keep attacking" [1]; conversely "sacrifices are generally a poor
  idea unless you have a plan to attack" and "if any sacrifice is made or accepted, the payoff should be proportional
  to the material given" [17][snippet, 30].
- Trades: "trades are good" / "trades are bad" are the standard partner calls [3]; "trade only when it helps your partner
  more than it helps the opponents" (weak, [19]).
- Sac-sitting (one board sacrifices, the partner stops moving): "no consensus on the definition"; "a weapon that can
  work occasionally, but every game, it is downright bad" (JarlCarlander) [snippet, 30].

### 2.8 Time, sitting, and the diagonal clock

- The clock that matters is your partner's opponent's: "If you are ahead of partopps, you win all sit/stall situations
  and time trouble races" [snippet, 28]; "the diagonal time difference matters more than the difference on the same
  board" [3]; "so long as you have more time than your opponent's partner, you don't need to make a single move"
  [snippet, 28].
- Magnitude: "Above 2000 strength, a time edge of as little as 10 seconds can outweigh positional advantages" [3]
  (Chuck Moulton, BWC 2022 commentary, cited by Wikipedia; the original commentary was not reachable).
- Sitting is forced defence when every move loses ("if you move, you are mated in one ... let your partner try to win"
  [snippet, 27]) and is the way to wait for a needed piece [1][4][20]; "stalling only works if you have more time than your
  opponent's partner, or you will flag first" [1][4]. "It is even possible for this to happen on both boards" (double
  sit, "sitzkrieg") [3][snippet, 28].
- No increment: "every second you spend sitting is a permanent cost" [snippet, 28]. Say "sit" often when even or ahead
  on time ("up to 10 times a game") [17]; "slow down when you have a window to use your time advantage" [17]; "if your
  partner is clearly lost, play faster and take more risks" [17].
- FICS-era pathology: "sealsitting" (exploiting the lag bug to sit past zero) was considered unethical [1]. Online lag
  distorts diagonal differences [3].
- Hambleton: "playing quickly is just as important as finding the best move" because sitting works only "provided you
  are ahead on the clock" [20].

### 2.9 Top-level games and the 2024 Chess.com Bughouse Championship

Facts established [14]: Aug 23 2024, online, 8 teams, double elimination, 2+0 with a 20-minute countdown per match (30
in the grand final), "two point advantage needed to win". Teams: Liang/Xiong, Chiu/Hauge, Tang/Dubansky, Zheng/Baker,
Wang/Plotkin, Vazquez/Ferguson, Zdybowicz/Reza, Erigaisi/Sarin. Results: Chiu/Hauge 7-2 Zheng/Baker; Vazquez/Ferguson
6-4 Zdybowicz/Reza; Chiu/Hauge 8-6 Vazquez/Ferguson; Liang/Xiong 11.5-1.5 Erigaisi/Sarin; Liang/Xiong 11-3 Chiu/Hauge
(upper final); lower bracket: Tang/Dubansky beat Erigaisi/Sarin 7-5, Vazquez/Ferguson 6-4, Zheng/Baker 9-2, lost 4-7 to
Chiu/Hauge; grand final Liang/Xiong 11.5-8.5 Chiu/Hauge ("overturned a deficit" [snippet, 34]).

The requested game "Vasquez+Helmsknight vs Tang+Gurwich" is from **qualifier 4**, not the knockout: Helmsknight and
Vazquez won the qualifier on their third attempt with 73/80, "despite dealing with distractions while streaming", after
a close contest with GM Andrew Tang and Daniil Gurwich, who "had taken an early lead" [snippet, 35]. Tang later qualified
with "BughouseKnight" (71/80 in qualifier 6) [snippet, 36]. In the knockout the pairs were Vazquez/Ferguson and
Tang/Dubansky, who met in the lower bracket (Tang/Dubansky 6-4) [14]. No move record, PGN/BPGN or annotation of any of
these games was reachable; the chess.com reports embed game viewers that the fetchers could not load. Qualifier-1/2
trivia: Xiong/Liang won 50 consecutive games [snippet, 37].

Other top-level material found: GM 6-player bughouse (Gukesh, Keymer, Maghsoodloo, Ganguly) featuring a "loophole
battery" where "they sacrificed 10 knights and bishops in a row", described as "an infinite ammo glitch" - the
three-board chain means every capture on one board arms the next, so a sacrifice sequence can be self-feeding [21]. No
moves given. 2025 championship: Awatramani/Chiu won; no annotations reachable [snippet, 38]. The lichess studies with
crazyhouse back-rank puzzles (`lichess.org/study/JPgWzcW4`, solutions `oBYnLqDK`) [11] and FM Crazy_Eight's materials
study [snippet, 39] are the closest thing to a tactical database; lichess refused the fetcher.

Books cited by Wikipedia and not readable here: Manson & Hoover (1992), von Zimmermann, *Bughouse Chess* (2006, 254 pp,
186 diagrams, annotated games and puzzles) [3][snippet, 40]. Gnejs: "Most of what I read about bughouse is crap, I am
afraid" [27].

---

## 3. Implications for the engine

Legend. "Stock" = original Hivemind it04 (`third_party/hivemind`, 37 planes per board, MCGS, FICS-rule mate detection
with time advantage, supervised start on human games then RL; README). Corrected: the supervised games are chess.com
games scraped with `src/hivemind/data/scrape_cc.py` (ratings ≥2400, partners ≥2200), not FICS, and they carry per-move
timestamps that `generate_planes.py` uses for plane 31 and for `pass` labels. "Custom" = the clock build and arena in
`docs/custom_engine_spec*.md`. "How I know" cites code read for this note: `src/hivemind/domain/board2planes.py`,
`engine/src/environment/board.cc`, `engine/src/search/mate_probe.cc`, `engine/src/search/supply_search.{h,cc}`,
`engine/src/Fairy-Stockfish/src/position.h`, `src/hivemind/constants.py`, `RULES.md`.

| # | Idea | Does stock capture it? | Concrete change |
|---|---|---|---|
| 1 | Blockable check is never mate; only contact/knight/double/occupied-line checks are decisive (2.1, 2.2). | Partly. Stock `is_checkmate` calls a blockable check mate when the partner has no immediate useful capture (stricter than FICS/chess.com). Free mode uses the arena rule, which matches FICS/chess.com exactly (`board.cc` `g_freePass` branch). The *network* has no plane saying "in check" or "check type"; it must infer from piece planes. | Keep the free-mode rule (it is the real rule). Add two cheap per-board planes in the clock build: `checkers` bitboard, and `empty interposition squares` (between_bb minus occupied; all-zero for knight/contact/double checks). The second is exactly the unblockability predicate and the thing attackers compute. Zero-init as with the clock planes; parity test holds. |
| 2 | A blockable check against an empty hand makes that board *sit*; the diagonal clock then decides (2.2, 2.8). | Custom captures it structurally: with `TreeMoveMs` a stuck board loses ms per macro-ply and a clock at 0 is a terminal loss. Stock could not (no clocks in tree, sit legality tied to the binary flag). Unknown whether the arena's "wait until the next move or piece lands on either board or `free_sit_ms`" lets the *attacker* keep the stuck defender frozen for as long as the real server would. | Add an arena test: defender in a blockable check with empty hand, partner board quiet, defender behind on the diagonal clock, assert the attacker's team flags them without moving. Add this "stall-mate" category to the gate stats (wins by flag while opponent board is in check). |
| 3 | Contact and knight checks are the attack; unprotected contact drops are refuted by Kxd (2.2, 2.4). | Search: `supply_search.cc` already weights real supply moves by geometric checking-drop potential and "excludes undefended contact drops which the king can simply take". The mate probe is single-board with `twoBoards=false`, so it certifies only mates with pieces actually in hand - i.e. only decisive mates, which is correct. Policy: 304 drop labels (all `X@sq`) so the net can express any drop. | No new rule needed. Measure: build a drop-mate puzzle set (crazyhouse back-rank studies [11], smothered `N@h6+ ... Nxf7#` [9], yo-yo `N@h3+ ... Q@g2#` [12], f7 sac then N-check then drop [snippet, 23]) as FENs with both pockets, and report policy top-1 / value sign on it per iteration. This is a tactical regression gate independent of Elo. |
| 4 | Mates often need a piece that has not arrived yet; the attacker times the sacrifice to the partner's capture (2.4 f7 family, "knight wins a queen"). | Stock sees both boards and both pockets (pocket counts /16 as constant planes, channels 12-21), so it can in principle plan across boards; its mate probe cannot (one board, no virtual drops). Fairy-Stockfish's virtual-drop machinery exists but is switched off everywhere in the engine. Custom v2 adds the `arrivals` head (pieces arriving per pocket within 5 s) as an auxiliary target only. | (a) Feed the `arrivals` prediction back as 20 input planes at the root in a later iteration (self-conditioning), or at least check that the value head's error drops on positions where a mating piece is about to arrive. (b) A second probe mode: allow virtual drops of exactly the piece types the partner can capture in one move (`has_useful_capture`), score as "mate pending supply", used only to order root moves, never as a terminal. |
| 5 | Pawns: 7th-rank drops, pawn walls, clamps, cheap blockers; promoted pieces revert (2.4, 2.5). | Yes in representation: pawn drops are policy labels on ranks 2-7, promoted-piece planes exist (channels 22-23), pocket counts per type. Whether the net *values* pawn drops as the humans do is unknown; stock's supervised start on FICS games imports human pawn-wall habits. | No input change. Add to the puzzle gate (3) a "defensive drop" subset: positions where the only non-losing move is `P@` or a quiet blocker, from recorded self-play losses. Track the share of policy mass on pawn drops when own king is attacked, per iteration, so a collapse toward all-out attack is visible. |
| 6 | Piece values differ (P=1, N=B=R=2, Q=4; two minors ~ queen) (2.3). | Learned by the value head; not hand-coded in stock's search. Hand-coded values do enter two places: Fairy-Stockfish `PieceValue[MG]` inside the mate probe (only to order moves and in the retired virtual-mate band) and the `pressure()` heuristic in `supply_search.cc` (unknown weights; not read in detail). | Leave values learned. Check `pressure()` and the probe's ordering do not use standard-chess values (B=N=3, R=5) for drop candidates; if they do, switch to the Gnejs scale. Low priority: these are exploration weights only. |
| 7 | King safety = holes near the king, f2/f7, no castling (2.6). | Learned. Castling-rights planes exist (27-30). Stock's human-game pretraining includes the "don't castle" culture. | Nothing structural. Diagnostic only: castling rate and mean king file per iteration in the gate report, to catch drift toward standard-chess habits. |
| 8 | Initiative over material; "defensive mistakes cost more than attacking ones"; first-move advantage is large (2.6, 2.7). | Learned by value/policy. The asymmetry argues for *not* symmetrising colours in training, which stock does not do (planes are team-relative but colour is implied by flip). | Report White-team win rate in self-play and gate; if it is far above 50% the opening book / pairing (800 opening pairs, both colours) already controls for it in the gate. Nothing else. |
| 9 | Time: the diagonal clock, 10-second edges, no increment, sit only when ahead of partopp (2.8). | Stock: one binary plane `TimeAdvantage` = sign(our A-seat clock minus the same colour's clock on B) i.e. the diagonal difference (`board.py time_advantage`), and hard rules that make sitting legal only with it. Custom v1: four absolute clock planes per board (ours/theirs, `min(t/300,1)` and `exp(-t/10)`), no explicit difference. The 10-s number supports the `exp(-t/10)` scale. The *diagonal* difference now has to be assembled across the two board blocks, which the shared per-board stem cannot do before the cross-board fusion. | Add one signed plane to each board block: `tanh((t_ours_on_this_board - t_diagonal_opponent) / 10)` (clamped), zero when no clocks. Note (critic): with equal time controls and synchronised starts the two boards' clock sums are equal, so this value is identical in both blocks; it is one number, as stock's flag already is in sign. Cheap, zero-init, and it is the single quantity every human source names. Keep the four absolute planes. |
| 10 | Sitting as a defensive act ("tell your partner to sit, they can't mate you") and sac-sit (2.5, 2.7, 2.8). | Custom: `FreePass` makes pass legal on either or both boards with no hand-written gate; the one-agent design has no partner to tell. Stock: pass only with time advantage. | Already the project's core bet. Add to the gate report: pass frequency, mean pass length, and outcome of games with a pass longer than 5 s, split by diagonal-clock sign, so "sits while behind on the diagonal" (always wrong per 2.8) is counted. |
| 11 | Blockable checks as tempo: force a defender to spend a piece from hand (2.2 item 5). | Learned; pocket planes change after the block. | None. |
| 12 | Forcing sequences are long (yo-yo mates, piece storms) and the human "quiet move when checks run out" (2.4). | MCGS depth depends on policy sharpness; no check extension. The mate probe (alpha-beta, one board, hands as held) finds pure forcing mates within its node budget. | Report probe hit rate and probe depth distribution in self-play; if most found mates are depth <= 3, raise `maxMateMoves` for the clock build and measure nodes/s cost. |
| 13 | Human knowledge is encoded in online games with premoves, sitting and lag artefacts (2.8). | **Corrected:** stock was pretrained on chess.com games (`scrape_cc.py`), not FICS, and the per-move timestamps were kept and used: `generate_planes.py` sets plane 31 from the real diagonal difference and labels human sits as `pass`, discarding sits by the team behind on time. Those games' clock behaviour (premoves, chess.com lag handling) is still not what the arena simulates. | For the clock build, the human shards can fill the v1 clock planes and v2 elapsed planes from the stored timestamps instead of zeros (parity only requires that the *widened weights* start at 0, not that the planes are 0). Keep human games as a source of positions and of human pacing; do not treat the filtered pass labels as ground truth for when sitting is right. |
| 14 | No annotated top-level database is readily available; the 2024 championship games exist only inside chess.com's viewer (2.9). | N/A. | If annotated games are wanted, they have to be pulled from chess.com by a logged-in person (the pages block fetchers), or from the FICS/BPGN archives that bughouse-db.org mirrors. Not needed for training; useful for the puzzle gate in (3). |

Three things this note did not settle and marks as unknown: the exact weights inside `supply_search.cc pressure()`;
whether the arena's `free_sit_ms` wait can hold a stuck defender indefinitely as a real server would (item 2); and the
per-board stem's ability to combine clocks across blocks (item 9 assumes it cannot, from the architecture description
"shared per-board input stem", not from reading the model code).

---

## 4. Sources

Read in full unless marked [snippet] (search-engine excerpt only; page refused fetching).

1. FICS help "bughouse_strat" mirror, https://www.unix-ag.uni-kl.de/~chess/gicshelp/node170.html - classic FICS strategy file: initiative, contact/knight checks, Q=10 N=7 R=4 B=2 P=1 scale, king safety, stalling, sealsitting.
2. [snippet] Sorsi, "Bughouse Attack Patterns", https://www.chess.com/blog/Sorsi/bughouse-attack-patterns - P@f2 vs P@h2 mistake.
3. Wikipedia, "Bughouse chess", https://en.wikipedia.org/wiki/Bughouse_chess - rules, Gnejs/Sunsetter/Sjeng values, piece storm, contact checks, 10-second edge at 2000+, sitzkrieg, references to von Zimmermann and Manson & Hoover.
4. FICS help "bughouse_strat" (bughouse-db mirror), https://bughouse-db.org/Help/HelpFiles/bughouse_strat.html - same file, Q=4 N=R=B=2 P=1 version, Mongolian Attack, named Black defences, slang.
5. FICS help "bughouse" (bughouse-db mirror), https://bughouse-db.org/Help/HelpFiles/bughouse.html - the exact mate rule ("no *possibility* of avoiding checkmate by a drop interposition"), drop notation, pawn-on-7th promotes next move.
6. chessexpress, "Why no drop for mate?", https://chessexpress.blogspot.com/2023/12/why-no-drop-for-mate.html - history of the OTB "no drop for mate" rule vs online standard.
7. pychess, "Bughouse", https://www.pychess.org/variants/bughouse - rules text (drops may mate), pointers to JarlCarlander series incl. "Don't Castle".
8. pychess, "Crazyhouse", https://www.pychess.org/variants/crazyhouse - pawn-drop defence against distance checks, knight unblockable, queen reintroduction, dropped rook can't castle.
9. zhchess, "You know the rules of crazyhouse, but how to...", https://zhchess.blogspot.com/p/you-know-rules-of-crazyhouse-but-how-to.html - smothered mate line, magnet/cookie-crumb/moon mates named, values 1/2/4, counterattack, colour complexes.
10. zhchess, "Beginner's guide to crazyhouse", https://zhchess.blogspot.com/2018/03/beginners-guide-to-crazyhouse.html - same material with N@e2+ queen fork, bishop defence examples.
11. zhchess, "Back rank mate", https://zhchess.blogspot.com/2019/10/back-rank-mate.html - three conditions for a drop-chess back-rank mate; lichess puzzle studies JPgWzcW4 / oBYnLqDK.
12. shogiandvariants, "Crazyhouse: chess with drops", https://shogiandvariants.blogspot.com/2023/11/crazyhouse-chess-with-drops.html - yo-yo mate, hook check, full N@h3+ ... Q@g2# sequence, Four Knights Italian assessment.
13. chess.com support, "What is Bughouse?", https://support.chess.com/en/articles/8615092-what-is-bughouse - "a check that could be blocked by a piece drop is not considered checkmate, even if the player doesn't have a piece in the bank"; counterfeits.
14. Liquipedia, "Chess.com Bughouse Chess Championship/2024", https://liquipedia.net/chess/Chess.com_Bughouse_Chess_Championship/2024 - format (2+0, double elimination), teams, every match score, prizes.
15. RulesPal, "Bughouse chess rulebook" (WCCC laws), https://rulespal.com/bughouse-chess/rulebook - Art. 7-8 potential-stock mate rule, 3-minute control, promotion by announcement, no stalemate.
16. [snippet] "Laws of BUGHOUSE chess" PDF, https://www.chessfed.gr/old_site/wccc2007/files/bughouse_chess_complete_rules.pdf - "not considered to be checkmated if he could parry the check with dropping a piece potentially" (PDF text not extractable here).
17. chessbraingame, "Bughouse chess concept for beginner", https://chessbraingame.blogspot.com/2017/05/bughouse-chess-concept-for-beginner.html - P=1 B=N=R=2 Q=4 "pretty accurate", doubled material, f7-sac costs, "say sit up to 10 times a game", timing advice.
18. Northwest Chess, "Bughouse rules", https://nwchess.com/articles/rules/bughouse_rules.htm - club OTB rules: 5 min, drop may mate, promoted pawn laid on its side, no sitting rule.
19. chessbio, "Siamese chess (Bughouse)", https://chessbio.com/term/siamese%20chess - WEAK (anonymous, AI-generated, contains an impossible line): P@h6/Q@h7+/P@g7/R@g8# family, P@e6 clamp, "what drop mates me next move?".
20. St. Louis Public Radio, "What's the deal with bughouse chess" (GM Aman Hambleton), https://www.stlpr.org/arts/2019-04-18/on-chess-whats-the-deal-with-bughouse-chess - sitting only when ahead on the clock, speed as important as move quality.
21. ChessBase, "6-player bughouse is pure chaos", https://en.chessbase.com/post/gukesh-keymer-maghsoodloo-ganguly-6-player-bughouse-is-pure-chaos - "loophole battery", 10 consecutive minor-piece sacrifices, chain feeding.
22. [snippet] key, "The Basics of Bughouse", https://www.chess.com/blog/key/the-basics-of-bughouse - f7/f2 sacrifices most common; knight + queen mate; queen sac for mating attack.
23. [snippet] JarlCarlander, "Commonly Missed Opportunities", https://www.chess.com/blog/JarlCarlander/bughouse-for-beginners-commonly-missed-opportunities - smothered mate with R@g8, knight-check-then-drop-on-vacated-square motif.
24. [snippet] tree55555 / Everything-about-chess posts, https://www.chess.com/blog/tree55555/bughouse-strategies-explained and https://www.chess.com/blog/Everything%20about%20chess/bughouse-chess-the-ultimate-chaos-mode-and-how-to-win-it - g7 target with h6 pawn, N@h5 after Bf8 moves, P@g7 lure.
25. [snippet] US Chess, bughouse article, https://uschess.org/content/view/11301/636 - "pawn walls" P@c6, P@f6 as the most underestimated defence (403 on fetch).
26. [snippet] US Chess, "Interview with a Bug Wizard: Kazim Gulamali", https://www.uschess.org/content/view/11764/671 - "If you tell your partner to sit your opponent can't checkmate you" (403 on fetch).
27. Chronatog, "ICS Interviews No. 013 - Gnejs", https://chronatog.com/?p=484 - Gnejs on bughouse writing quality; no values stated there (the valuation is attributed to him by [3][17]).
28. [snippet] chess.com club forum "Bughouse tips. Anyone?" and Sorsi "Basic Bughouse Tips", https://www.chess.com/clubs/forum/view/bughouse-tips-anyone?page=1 , https://www.chess.com/blog/Sorsi/basic-bughouse-tips - "look at partopp's clock", uptime definition, no increment.
29. [snippet] ICC help "bughouse", https://www.chessclub.com/help/bughouse - drops with check/mate allowed (page now redirects).
30. [snippet] JarlCarlander "Defending Against Sacrifices", "How to Theorize in Bughouse", "Esoteric wisdom..."; Sorsi "Advanced bughouse tips", https://www.chess.com/blog/JarlCarlander/bughouse-for-beginners-defending-against-sacrifices , https://www.chess.com/blog/Sorsi/advanced-bughouse-tips - plug g2/g7 at any cost, bishop drop stops a sac, sac-sit controversy, distance attacks devalued by blockers, proportional payoff.
31. [snippet] tree55555, "Bughouse Strategies Explained" - Nh3 / Nxf2# smothered pattern (same URL as 24).
32. [snippet] 23Dragons, "Tips for Bughouse (Part I)", https://www.chess.com/blog/23Dragons/tips-for-bughouse-part-i - pawn drops make forks easy; 7th/2nd rank pawns grow with an emptier board.
33. [snippet] JarlCarlander, "Being Safe" and chriswsh "Let's Bughouse" series, https://www.chess.com/blog/JarlCarlander/bughouse-for-beginners-being-safe , https://www.chess.com/blog/chriswsh/let-s-bughouse-playing-against-the-sicilian-part-2 - pawns as defenders, quiet moves when checks run out, keep attackers safe.
34. [snippet] chess.com news, "Comeback Kings Liang, Xiong Win Bughouse Championship", https://www.chess.com/news/view/2024-chesscom-bughouse-championship-knockouts-liang-xiong-win - 11.5-8.5 grand final, deficit overturned.
35. [snippet] chess.com news, "Helmsknight, Vasquez Overcome Team Tang To Qualify On 3rd Attempt", https://www.chess.com/news/view/2024-chesscom-bughouse-championship-qualifier-3-4 - 73/80, Tang/Gurwich early lead, streaming distractions.
36. [snippet] chess.com news, "Tang Scores 71/80 In 6th Qualifier With Mysterious New Bughouse Partner", https://www.chess.com/news/view/2024-chesscom-bughouse-championship-qualifier-5-6 - Tang/BughouseKnight.
37. [snippet] chess.com news, "Xiong, Liang Win 50 Consecutive Games", https://www.chess.com/news/view/2024-chesscom-bughouse-championship-qualifier-1-2 .
38. [snippet] chess.com news, "Awatramani, Chiu Dominate Bughouse Championship" (2025), https://www.chess.com/news/view/awatramani-chiu-win-bughouse-championship .
39. [snippet] lichess study by FM Crazy_Eight, "Materials for bughouse & crazyhouse", https://lichess.org/study/by/Crazy_Eight - links a "Bughouse Book" and training sites (lichess refused the fetcher).
40. [snippet] von Zimmermann (ed.), *Bughouse Chess* (2006), listing https://www.barnesandnoble.com/w/bughouse-chess-georg-von-zimmermann/1136934218 - 254 pp, 186 diagrams, annotated games and puzzles.
41. Wikipedia, "Crazyhouse", https://en.wikipedia.org/wiki/Crazyhouse - Kaufman's "almost certain that White has a forced win"; `1.N@e7+ Kh8 2.Bxg7#` caption.
42. Local code, `third_party/hivemind` (branch `talisman/custom-engine`): `src/hivemind/domain/board2planes.py`, `src/hivemind/domain/board.py` (`time_advantage` = diagonal difference), `engine/src/environment/board.cc` (`is_checkmate`, free-mode rule), `engine/src/search/mate_probe.cc` (single board, `twoBoards=false`, Q/N promotions), `engine/src/search/supply_search.{h,cc}`, `engine/src/Fairy-Stockfish/src/position.h` (`allow_virtual_drop`), `src/hivemind/constants.py` (304 drop policy labels), `RULES.md`, `README.md` - what stock Hivemind encodes and enforces.
