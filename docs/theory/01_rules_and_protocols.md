# 01. Rules and protocols: FICS, ICC, chess.com, lichess-style variants, OTB rulebooks, ptell

Written 2026-11-17 for the custom engine (one agent, both boards, real clocks, no hand-written stalling
rules; see `docs/custom_engine_spec.md`, `docs/custom_engine_spec_v2.md`, `docs/beat_stock.md`). Everything
below was read from the sources in section 4; local code is cited as [L1]..[L5]. Claims that rest on one
weak source (a search snippet, a forum post, a video) are marked **(weak)**.

Several pages could not be read directly: chess.com blog and forum pages refuse the fetcher, lichess.org
is unreachable from this machine, web.archive.org is blocked, and chessvariants.org returns 403. Where a
claim about those depends on a search snippet or on Wikipedia's citation of them, it is marked.

---

## 1. Summary

- **There is no single rulebook.** FICS/ICC (shared ICS heritage), chess.com, OTB federations (USCF 2004,
  WCCC 2004, WCCC 2007 "WBN laws") and the Dutch doorgeefschaak variant disagree on checkmate, repetition,
  castling with dropped rooks, both-flags-fall, and communication. An engine trained for one server's rules
  is playing a slightly different game on another [1][16][21][24][26][27][18].
- **The server mate rule is the one the arena already implements:** a check is mate only if no drop could
  ever block it, regardless of what the defender holds now. FICS, the 1995 ICS help (ICC's ancestor),
  chess.com, WCCC 2004 and WCCC 2007 all state this. USCF 2004 and the "Hawaii rules" do not; there a
  defender with nothing in hand is simply mated [1][16][21][24][26][27][25]. FICS itself documents a known
  inadequacy of the rule (drop-capture-drop loops) [15].
- **Sitting is legal and unbounded on every server with a clock.** The only limits are the clock (flag) and
  OTB-only penalties: a "three partner moves" convention for clockless play and WCCC 2007's 5-10 s
  "freezing" sanction [18][24][2][17]. No chess.com sitting rule beyond the clock was found.
- **Flagging is a claim, not automatic, on FICS**: `flag`/`autoflag` may take up to 2 s to fire, is checked
  before the increment, and a lagging timeseal user cannot be flagged at all. Both out of time: the server
  rules a draw. Clocks are kept in milliseconds; moves take at least 0.1 s when `minmovetime` is on
  [5][6][14][12][11]. These mechanics shape every FICS training game.
- **Promoted pieces revert to pawns when captured, everywhere.** OTB they physically stay pawns (laid on
  their side / declared aloud); WCCC 2007 even restricts promotion to Q or N [1][16][21][24][26][27][20].
- **Draws are near-impossible online**: FICS needs a draw agreed on both boards; a double flag is a draw;
  repetition handling differs (Wikipedia says FICS/chess.com ignore the reserve and ICC counts it, but gives
  no source; FICS's own crazyhouse rule resets the count on any drop; chess.com users report board-only
  counting; an old ICC help snippet says repetition draws were disabled there) [1][5][15][18][19][40].
- **Time controls**: FICS/ICC bughouse is typically 2 0 or 3 0, no increment (the BPGN sample is 120+0);
  chess.com's community calls 3|0 standard **(weak)** and says increments are virtually never used; OTB
  rulebooks use 5 0 (USCF, WCCC 2004) or 3/5 min without bonus (WCCC 2007) [35][41][27][26][24].
- **The ptell layer is small and formal**: FICS `partner`, `ptell`, `ptime`, `prefresh`, `pfollow`,
  `pobserve`, `bugwho`, channels 24/25; ICC: wild 24, channel 24, `say`, `kib`. The spoken vocabulary
  (sit/go, +/-/+++ by piece, trades good/bad, "knight mates", rock/injected, dead, partop, uptime, flow) is
  stable from 1995 to today and was literally parsed by the Sjeng engine with fixed numeric effects
  [1][2][3][7][8][16][17][29][38]. The **Transfer Communication Sheet** is a Singapore 2013 OTB invention:
  a cardboard divider through which partners had to communicate, not a historical standard [28].
- **Lichess has no bughouse** (variant list issue #441 open since 2015; the developer's stated reason was
  that drops cannot be encoded as PGN, lichess's storage format). Its crazyhouse rules match the common
  core and its FEN puts the reserve on a 9th rank with `~` marking promoted pieces [32][20].
- **BPGN** (FICS game records at bughouse-db.org) carries per-move remaining time in braces, a per-move
  `Lag` tag and `TimeControl 120+0` [35]. That is exactly the out-of-step timing data the custom engine
  needs; stock Hivemind's inputs cannot represent it.

---

## 2. The knowledge

### 2.1 Common core shared by every ruleset

- Two boards, four players, partners hold opposite colours; a capture passes the captured piece to the
  partner, who may drop it on any empty square of his board instead of moving [1][16][18][21][24][26][27].
- Pawns may not be dropped on rank 1 or 8 (all sources). A pawn dropped on rank 7 may promote next move
  [1][2]. A pawn dropped on its second rank keeps the double step and can be taken en passant (Wikipedia,
  Hawaii rules, WCCC 2007 art. 5.11, pychess, chess.com support) [18][25][24][30][21]. FICS/ICC/chess.com
  behaviour on this is not stated in their help files; the arena implements it [L1].
- A drop may not capture (WCCC 2007 5.10) and may not leave the dropper in check (WCCC 2004) [24][26].
- Drops may give check and mate on FICS, ICC, chess.com, pychess, USCF, WCCC, Hawaii rules
  [1][16][21][30][27][24][25]. One named exception: the Dutch *doorgeefschaak* rules forbid dropped pieces
  from giving check (Wikipedia, citing the Schaakbond regulation; the regulation itself was not reachable)
  [18][19] **(weak)**.
- A promoted pawn that is captured transfers as a pawn: FICS [1], ICS 1995/ICC [16], chess.com (it calls
  these "counterfeits") [21], pychess [30], USCF 10c [27], WCCC 2004 and 2007 [26][24], Hawaii rules [25],
  lichess crazyhouse [20].
- The match ends when either board ends by mate, resignation or flag; the losing team loses both games and
  all four ratings move (FICS, ICS 1995, chess.com, USCF 7, WCCC 2004 art. 4, WCCC 2007 1.3)
  [1][16][21][27][26][24]. OTB variants where the second board is played out and the scores added exist
  (Wikipedia via chessvariants; WCCC 2007 1.10 for simultaneous finishes) [18][24].
- Partners may talk freely and may not touch the other board (USCF 12, WCCC 2004, WCCC 2007 1.11, pychess)
  [27][26][24][30]. Caissly adds "only at your own turn" **(weak, unsourced summary page)** [33].

### 2.2 Checkmate and the blockable check

**The server rule (FICS, ICS/ICC, chess.com).** FICS: "checkmate requires that no drop interposition is
possible, not just that none is available at the moment"; the opponent may later receive a piece from a
partner; only contact checks and knight checks give a *decisive* checkmate [1]. The 1995 ICS help (the
ancestor shared by FICS and ICC) says the same, word for word in substance [16]. chess.com support: "a check
that could be blocked by a drop is not checkmate, even if the player has no piece in the bank" [21]. The FICS
crazyhouse description adds the practical reading: with a single sliding check, mate exists only if the
defender holds nothing, or, for a queen/rook check along the first or last rank, holds only pawns (pawns
cannot be dropped there) [15]. FICS itself notes the rule "may be inadequate for some situations where each
player drops a particular piece, which is immediately exchanged, and then dropped again" [15].

Consequences on a server: a player in a blockable check with nothing in hand is **not** mated and **not**
stalemated; he has no legal move and must wait. His clock runs. The game ends only when a piece arrives (he
blocks), the position changes on his board (it cannot, he is to move) or he flags [1][21][2]. Double check
is mate if the king cannot move (two checkers cannot both be blocked by one drop) [L1][L3].

**OTB rulebooks.** WCCC 2004: "a player is not mated if he could block or capture with a piece potentially
in his stock"; "a player is not stalemated"; FIDE 9.6 (draw when mate is impossible) is cancelled [26].
WCCC 2007 art. 1.5: not checkmated "if he could parry the check with dropping a piece potentially", so mate
"must almost always be given either with a contact check, either with a knight check, either with a multiple
check" [24]. Art. 1.9 adds a rule servers do not have: a player who is mated right after capturing may still
hand that piece to a partner who has the move, and the partner may drop it as his last move [24].

**Rulesets without the blockable-check rule.** USCF 2004 (rev. 2004-03-01, used at the 2006 National High
School Championships) defines wins by mating the king, allows capturing a king left in check after an
illegal move, and never mentions potential drops [27]. The Hawaii rules (RLP, 2002) say a check "can be
voided by dropping a captive between" but treat a king left in check as capturable [25]. Under these, a
defender with an empty hand facing a long-range check is mated outright. **Disagreement**: a position that
is a win under USCF/Hawaii is an ongoing sit under FICS/chess.com/WCCC.

**Side point on "stalemate".** FICS help does not define bughouse stalemate; WCCC 2004 says a player is
never stalemated [26]; the arena treats a stuck side as waiting, never a draw [L1]. Whether FICS adjudicates
a stalemate as a draw in bughouse is **unknown**.

### 2.3 Flagging, clocks, lag (FICS mechanics; ICC shares the heritage)

- FICS uses a Fischer clock for every game; time is kept in milliseconds; `time 0` gives 10 s; `0 0` is
  untimed [14]. Both bughouse boards share the same time control, but `moretime` can be given separately
  on each board [1][13].
- **Flagging is a claim.** `flag` asks the server to check both clocks: opponent out and you not -> win;
  both out -> "the server claims a draw"; opponent not out -> error [5]. `autoflag` makes the server claim
  for you, "may take up to 2 seconds to take effect", and is evaluated on the clock at the time of the move,
  *before* the increment is added [6]. Displayed 0:00 does not prove the opponent is out: the server's time
  is authoritative and is re-sent after each move [5].
- **Lag and timeseal.** Transmission delay is charged to the mover unless timeseal timestamps the move
  locally; then only thinking time is charged [11][12]. A lagging timeseal user cannot be flagged; the clock
  is adjusted on reconnect; if flagging is impossible, `adjourn` forces a courtesy adjournment (abort if a
  player is unregistered) [14][6]. In bughouse an adjournment or disconnection ends both games and the match
  cannot be resumed [1]. Flagging a lagging opponent is "not unethical" [11]. The 1995 strategy file records
  a timeseal bug (you cannot flag an opponent until he moves) and calls exploiting it ("sealsitting")
  unethical; accepted practice is to move or resign when out of time [17].
- `minmovetime 1` makes every move take at least 1/10 s [14]. Premove was "allowed but discouraged" on FICS
  [14]; chess.com has premove with a minimum charge per move (secondhand lichess-forum claim) **(weak)**.
- Wikipedia: lag changes the *diagonal* time difference; restarting when the gap grows is good
  sportsmanship (cited to an archived Ebenfelt page) [18][19]. Online partners use private messages, which
  makes coordination easier than OTB (von Zimmerman p. 240 via [19]).
- **Time controls.** BPGN sample from FICS: `TimeControl 120+0` [35]. chess.com: "the standard time control
  is 3|0" (community blog, snippet) **(weak)** [41]; bughouse "virtually never played with a time increment"
  (chess.com community, snippet) **(weak)** [38]. OTB: USCF G/5 [27]; WCCC 2004 5 min [26]; WCCC 2007 "3 or 5
  minute games without any time bonuses per move" [24].
- **Both flags down** = draw OTB (USCF 16b, WCCC 2007 7.10, Hawaii) [27][24][25] and on FICS via `flag`
  [5]. The arena instead picks the clock that had less time at the start of the interval as the loser [L1];
  stock Hivemind has no clocks at all [L2].

### 2.4 Sitting and stalling

- **Definition.** FICS 1995 strategy: "if you're being mated by force, you can choose not to move and let
  your partner try to win"; also delay until the opponent's next move so he must move without the piece he
  needs; both "only work if you have more time than your opponent", otherwise your team flags first [2][17].
- **Legal and unlimited with a clock** on FICS, ICC and chess.com; no server rule limits it beyond the clock
  (none found in [1][16][21][22][23]). "No sit" house rules exist in some clubs (snippet) **(weak)** [38].
- **OTB limits.** Clockless play: a player may not delay longer than the time his partner needs to make
  three moves (Wikipedia, citing chessvariants.org "Bughouse and Tandem Chess", 2002; the page returned
  403) [18][19]. WCCC 2007: no sitting rule, but arbiters may "freeze" a player for 5 or 10 s (he waits until
  his clock is 5 s lower) as a sanction, and a draw offer "at any other time" risks the annoyance clause
  [24].
- **Terms.** *Uptime*: the extra time you have relative to your partner's opponent, not your own opponent
  (the diagonal) [38][18]. *Sitzkrieg*: both diagonal players sit; decided by which of the two has more time
  [18]. Wikipedia (Manson & Hoover pp. 75-89 via [19]): the diagonal time difference matters more than the
  same-board difference. "At 2000+ a 10-second advantage can outweigh positional gains" (Twitch commentary,
  2022) **(weak)** [18][19]. "Sit until 10, then low flow" appears as a named plan in a chess.com blog
  **(weak)** [39]. ABCs of Bughouse: start moving when under 20 s; "sitting is a timing strategy", distinct
  from thinking; sacsitting (worrying about a small deficit) is a bad style; last-trench defence = play
  faster, tell partner to sit, tell partner to hold a piece [36].
- **Reading the rule from the engine side.** Because mate needs an unblockable check, and because a stuck
  defender simply waits, a *blockable* mating attack is only as good as the time it takes the partner board
  to deny a blocker. This is the structural reason "sit" exists: it converts the clock into a weapon.
  Hivemind encodes this as a legality gate, not as a learned trade-off [L2][L3].

### 2.5 Draws

| Rule | FICS | ICC | chess.com | WCCC 2004 | WCCC 2007 | USCF 2004 |
|---|---|---|---|---|---|---|
| Agreement | both boards must offer and accept ("a near impossibility") [1] | same ICS text [16] | not documented [21] | offer and acceptance from both players of a team decides the match [26] | per board, 2.2a [24] | between teams, 16a [27] |
| Repetition | crazyhouse: 3x same board position, same side to move, count resets on pawn move, capture **or drop** [15]; bughouse unstated; Wikipedia: board-only, reserve ignored (unsourced) [18][19] | Wikipedia: reserve counted (unsourced) [18][19]; an old chessclub.com help snippet says repetition draws disabled, agreement only **(weak, conflicting)** | users report board-only repetition being called in drop loops; claim via draw button, not automatic (forum snippets) **(weak)** [40] | 4 repetitions, no drop in between, claimable even if the opponent's stock grew [26] | 3 identical positions, counted aloud 1-2-3; a partner may count on the other board [24] | none; "insufficient losing chances" claims not allowed [27] |
| Both flags | draw via `flag` [5] | - | - | - | draw, 7.10 [24] | draw if no win claimed, 16b [27] |
| Simultaneous mates / both boards end | - | - | - | capture-then-mated while partner mates with the piece = draw [26] | 1.10 both finish: win both or draw; 2.2c same capture case [24] | - |
| Stalemate | undocumented | undocumented | undocumented | never stalemated [26] | not addressed | - |

The arena has no draws at all except `max_plies`/abort [L1]; stock Hivemind has twofold/threefold repetition
input planes and a WDL head, so it was trained with some notion of draws [L2].

### 2.6 Castling, en passant and other edge cases

- Castling with a dropped rook: allowed under the Hawaii rules ("considered not to have moved") [25] and in
  Wikipedia's description [18]; **forbidden** by WCCC 2007 5.8(1)c [24] and by pychess Bughouse960 [31]
  **(snippet)**. FICS/ICC/chess.com: not documented; python-chess's `CrazyhouseBoard` (the arena) does not
  grant castling rights to a dropped rook (no source read; check `arena/tests`).
- Illegal moves: OTB they lose (USCF 8, WCCC 2007 1.14, WCCC 2004 after the clock is pressed) [27][24][26];
  servers reject them. Clock-move OTB: a move is complete when the opponent's clock starts; the captured
  piece may not be passed before that (USCF 6/9, WCCC 2004, Hawaii) [27][26][25].
- Hidden pieces OTB: "invisible bishop" restriction, warnings, forfeits (WCCC 2007 1.7-1.8, USCF 11)
  [24][27]. No server analogue; pockets are public.
- OTB "non-disruptive premove": you may start your move while the opponent is still pressing his clock but
  may not complete it first (WCCC 2007 6.5) [24].

### 2.7 Communication: commands, vocabulary, Transfer Communication Sheet

**FICS commands.** `partner <user>` (needs `bugopen 1`, partnership not saved across logins), `ptell <msg>`
(only your partner hears it), `ptime` (your partner's game clocks), `prefresh` (partner's board), `pfollow`,
`pobserve`, `bugwho g|p|u` (games / idle partnerships / open users), `who B`; channel 24 for matches and
kibitzing, channel 25 for finding partners; a match is started as a "wild bughouse" challenge; aliases are
recommended for common ptells [1][2][3][4][7][8][16][17]. ICC: bughouse is wild 24, channel 24, `ptell`,
`prefresh`, `say` (other three players), `kib` (all four plus observers); `match <opp> <time> <inc> bughouse`
(search snippet of the now-redirected chessclub.com help) **(weak)** [16].

**Spoken vocabulary** (1995 FICS file [2][17], chess.com blogs [38][39], ABCs [36], Wikipedia [18]):

| Term | Meaning |
|---|---|
| sit / stall | do not move; wait (for a piece, or because you are mated by force) |
| go / move | stop sitting |
| uptime, "up time" | having more time than your partner's opponent (the diagonal) |
| partop | partner's opponent |
| sitzkrieg | both diagonal players sitting |
| +, ++, +++ (with piece letter) / -, --, --- | piece is good / very good / mates; bad / very bad / mates you (e.g. "+++n" = knight mates, "---p" = pawn mates me) |
| trades good / bad, "+++"/"---" alone, high/low flow | partner should trade a lot / little |
| "no queens", "no knights" | do not give the opponent that piece |
| "knight mates", "pawn mates", "rook mates me" | a single drop of that piece decides |
| rock | my position is solid |
| injected / inject | pieces dropped on squares that hurt the opponent; being so hit |
| dead | lost (or, Sjeng: "I'll have to sit (dead)") |
| feed, hang | give material / leave a mate or piece undefended |
| themed | two pawns side by side on the 7th, promoting |
| noodle / spoon | partner battle cries |
| annoy | pestering an opponent with text |
| hold | keep a piece in hand instead of dropping it |

**A machine-readable version of the vocabulary.** Sjeng's `partner.c` parses ptells and sets engine
state: `sit` -> `must_sit = TRUE`; `go`/`move` -> `must_sit = FALSE, must_go = 4`; `fast`/`time` -> `go_fast`;
`slow`; `abort`, `flag` relayed to the server; `x`/`=`/"never mind" resets hand values; `dead` and "(piece)"
set `partnerdead`/`piecedead` and go fast; `+` adds 150, `++` 1000, `+++` 50000 to the hand value of the
named piece (letters n b r q p, `d` = diagonal pieces, `h` = heavy pieces; `b` also raises q, `p` also raises
b and q), `-` the negatives; `trade +` adds 25 pawn / 50 minor and rook / 100 queen, `trade -` subtracts 20 /
50 / 100; a `CANCEL_THRESH 3` counter governs when it warns the partner with `---p` and later sends `x` and
`go` [29]. No time thresholds exist in that file.

**chess.com interface.** Partner communication is via chat and a row of preset buttons/icons; a red hand
means "sit", a clock icon "go" (blog snippets; the blog itself was unreadable) **(weak)** [37][38]. chess.com
support pages describe partnering (Custom Challenge, random or named partner, invite/cancel) but not the
buttons [22]. Gnejs (André Nilsson, the top FICS player credited with the 1/2/2/2/4 values) in 2019: "the
bughouse at chess.com seems buggy and communication is bad" [34].

**Transfer Communication Sheet (TCS).** Singapore's 1st National Transfer Chess Championship (2-3 Feb 2013,
Singapore Chess Federation + SMU Mindsports Club) used a cardboard divider between the two boards, one per
team, and "players were allowed to communicate only through a Transfer Communication Sheet"; the organisers
(Cyril Chua, Chua Wei Yuan) drafted their own rules because "there are no universally accepted rules" and
planned to propose them to FIDE [28]. A chess.com blog claims that "earlier rules" limited communication to
a small symbol set ("bishop", "no rook", ...) and that chess.com's partner icons are those symbols
**(weak, snippet of an unreadable page)** [37]. No standard TCS layout or symbol list was found anywhere; the
Wikipedia article does not mention it [18]. Treat the TCS as one tournament's device, not a protocol.

### 2.8 Piece values quoted alongside the rules

Not the topic of this file, but rule pages state them: FICS 1995 strategy Q4 N2 R2 B2 P1 ("bishops often
no better than a pawn", advanced pawns worth more than a bishop) [2]; the GICS copy of the same file reads
Q10 N7 R4 B2 P1 **(the two mirrors disagree; the FICS text is the live one)** [17]; Nilsson/Gnejs P1
B=N=R2 Q4 (von Zimmerman p. 17 via [19]); Sunsetter 100/195/192/200/390 and Sjeng 100/230/210/250/450
(P/B/N/R/Q) [18][19]. The structural reason given everywhere: a piece in hand has full mobility, so
long-range pieces lose relative value and pawns gain [18].

### 2.9 Game records and position encodings

- **BPGN** (FICS archive at bughouse-db.org; ficsgames.org excludes bughouse and points there [42]):
  tags `Event "FICS rated bughouse game"`, `Site`, `Date`, `Time`, `BughouseDBGameNo`, `WhiteA/BlackA/
  WhiteB/BlackB` with ratings, `TimeControl "120+0"`, a `Lag` tag with ~240 per-move values, `Result`;
  movetext `1A. d4{117.674} 1a. Nf6{119.900} 1B. e4{116.768} 2A. Nc3{117.529} ...`, drops as `N@g4+`,
  epilogue `{nonspiego resigns} 1-0`. The brace value is the mover's remaining clock after the move (the
  CCIF export labels the derived series `elapsedmovetime`) [35]. The original spec lived at
  bughouse.be/BPGN_Standard.txt (archive blocked) [19].
- Lichess crazyhouse FEN: reserve as a 9th rank, promoted pieces marked `~`; chess.com puts the reserve after
  the full-move number and tracks promoted pieces by coordinates (Wikipedia, flagged "failed verification")
  [20]. The arena keeps python-chess's `~` marker so a captured promoted piece transfers as a pawn [L1].

### 2.10 Lichess and crazyhouse

Lichess offers crazyhouse, not bughouse. Issue #441 ("definitive list of ... variants", opened 2015-04-26,
still open, label "prospect") lists bughouse as "probably not doable right now"; developer ornicar
(2016-01-14): drops "cannot be encoded as PGN, which is lichess de-facto storage format" [32]. Crazyhouse
rules: ordinary chess plus drops; the captured piece changes colour into the capturer's own pocket; pawns
not on ranks 1/8; a pawn dropped on its 2nd rank may double-step, elsewhere not; promoted pieces return as
pawns; dropping a pawn to mate is legal [20]. Crazyhouse has no partner, so there is no blockable-check
rule and no sitting: an ordinary checkmate ends it. pychess.org does host bughouse (rules "a shortened
version of the Wikipedia article") and Bughouse960 (no castling with a dropped rook) [30][31].

---

## 3. Implications for the engine

Legend for "Hivemind now": what stock it04 + the stock engine do, from [L2] (README: 37 input planes per
board incl. plane 31 "time advantage", planes 34-36 halfmove clock and repetition; value, WDL and
moves-left heads), [L3] (`joint_action.h` pass legality), [L4] (`board.cc::is_checkmate`), [L5]
(`RULES.md`).

| # | Idea from the sources | Hivemind now | Concrete change |
|---|---|---|---|
| 1 | Server mate rule: a blockable check is never mate; the defender waits [1][16][21] | **Corrected:** captured in a more elaborate but *stricter* form than the server rule (it calls some blockable checks mate; see the audit note at the end of this row). `is_checkmate` asks whether the *partner* can actually capture a piece that creates a legal block, with a `TimeAdvantage` branch that requires every opponent reply to leave such a capture; `assumePartnerCanBlock` gives the geometric model; `g_freePass` makes a stuck board a wait, not a loss [L4][L5]. | Keep the arena's geometric rule as the *terminal* rule (that is what servers do) and let the network learn the "can the partner actually feed a blocker" judgement from clocks + positions instead of hard-coding it. Verified: under `g_freePass`, `can_partner_provide_blocking_piece` short-circuits to the geometric test (`board.cc` ~361), so engine and arena agree on what a win is. Remaining audit item: the stock (non-FreePass) partner-capture search is a one-ply lookahead on the partner board; the sources' rule is about *ever* receiving a blocker, so stock search calls positions "mate" that a server would not. That is a stock weakness worth exploiting in the gate, not copying. |
| 2 | FICS's own caveat: drop-capture-drop loops make the rule "inadequate" [15] | Unknown handling; no loop detection in the arena beyond `max_plies` [L1]. | Log how often self-play hits long drop/capture cycles on a stuck board; if frequent, add a repetition-with-pocket input or let `max_plies` draws be scored as draws in training rather than discarded. |
| 3 | Sitting is unlimited; its value is set by the *diagonal* clock difference, not the same-board one [2][17][18][38] | Not captured as a quantity: plane 31 is one yes/no flag, and passing is *illegal* without it [L2][L3]. **Corrected:** the flag *is* the sign of the diagonal difference: `board.py::time_advantage` returns `times[0][side] - times[1][side]`, our board-A seat minus the same colour's seat on board B (the diagonal opponent). The lockstep tests showed the cost of having only the sign. | v1 clock planes give each board its own-seat and opponent-seat clocks; the diagonal difference is then only implicitly available across the two board blocks. **Corrected:** there is only *one* diagonal number, not two. With equal time controls, synchronised starts and no increment, exactly one clock runs per board, so `t_ourA + t_oppA = t_ourB + t_oppB` at all times and hence `t_ourA - t_oppB = t_ourB - t_oppA` (upstream `time_state.h` states the same invariant: "the two boards' clock sums stay equal and a team's lead over its diagonal opponents is one number"). The two differences diverge only through lag compensation (timeseal, §2.3), `moretime`, or unsynchronised starts. So add **one** signed plane, `tanh((t_ourA - t_oppB)/30)`, written into both board blocks (or as the shared time token, [L2] line 371), plus the absolute clock of the diagonal opponent if the per-block planes do not already carry it. Under FreePass this is what the policy must read to choose a pass. |
| 4 | "Sit until 10 then move", "start moving under 20 s" thresholds; a 10 s edge decides at 2000+ **(weak)** [36][39][18] | No absolute-time input; the flag is relative [L2]. | The v1 planes `min(t/300,1)` and `exp(-t/10)` cover the 0-30 s region well (`exp(-t/10)` at 10 s = 0.37, at 20 s = 0.14). Keep them; add nothing. Test: does the trained policy's pass rate change sharply around 10-20 s of own time? That is the behaviour the sources describe. |
| 5 | Flagging is a *claim* with up to 2 s delay, checked before increment; both out = draw; lagging timeseal users cannot be flagged [5][6][14] | Arena: a clock at 0 is an immediate loss; both out in one interval -> the one with less time loses [L1]. Stock: no clocks. | For FICS-faithful training data, model a flag as landing 0-2 s after time hits zero and score a double flag as a draw (`outcome 0`). For beating stock Hivemind in our arena the current rule is fine; make the arena's flag rule a profile option so one network can be evaluated under both. |
| 6 | Lag shifts the diagonal difference; timeseal removes transmission lag from the mover but not from the game [11][12][18] | Arena jitter (0.5-2.0x on think time) approximates lag plus speed variance [spec v1 §3]. | Add a second, additive jitter term per move drawn from a BPGN-like lag distribution (the `Lag` tag in [35] gives real per-move values) so the network sees both multiplicative and additive out-of-step noise. Record both in the sample. |
| 7 | Per-move remaining time is in every FICS game record (BPGN braces) [35] | **Corrected:** it04's supervised data is chess.com, not FICS (`src/hivemind/data/scrape_cc.py`, min rating 2400 for the players, 2200 for the partners), and it *does* carry per-move clocks: the scraper stores `moveTimestamps`, `game_reader.py` parses them into per-move remaining times and uses them to order the moves of the two boards, and `generate_planes.py` replays them through `BughouseBoard.update_time`, fills plane 31 from the real diagonal difference, derives `pass` labels from human sits (a team that did not move while the other team did) and *drops* every sample where the team behind on time passed on an on-turn board. So the human sitting data was not noise to it04: it was labelled, but with the clock reduced to one bit and the "sit while behind" cases filtered out. | If supervised pretraining or fine-tuning on human data is revisited, the existing chess.com shards already have the per-move clocks needed to fill the v1 clock planes and the v2 elapsed planes; BPGN braces give the same for FICS games. Stop filtering out sits by the team behind on time (the network should learn when those are right) and keep the human move-time distribution, which is the only source of human-like opponent pacing. |
| 8 | Promoted pieces revert to pawns [all] | Captured: planes 22-23 mark promoted pieces [L2]; arena transfers `Q~` as a pawn [L1]. | None. Verify a test exists for a promoted piece captured on A arriving as `p` in B's pocket. |
| 9 | Pawn dropped on rank 2/7 may double-step; en passant applies [18][24][25] | Arena: yes [L1]. Engine: Fairy-Stockfish bughouse rules, assumed yes; **unverified**. | One C++ test: drop P@e2, confirm e2-e4 is generated and e.p. follows. |
| 10 | Castling with a dropped rook: allowed (Hawaii, Wikipedia) vs forbidden (WCCC 2007, pychess960); servers silent [25][18][24][31] | Arena uses python-chess rules (no castling right for a dropped rook; **unverified**); Fairy-Stockfish bughouse: **unknown**. | Test both sides agree; pick "no castling with a dropped rook" (the stricter, pychess/WCCC reading) and document it in `arena/RULES.md`. Low impact on strength. |
| 11 | Draws: double-board agreement, double flag, repetition rules differ and are contested [1][5][15][18][40] | Stock has repetition planes and a WDL head but the arena never produces a draw except `max_plies` [L1][L2]. | Decide a target rule set (FICS-style: repetition on board only, count resets on any drop, per [15]) and implement it in the arena so the WDL head's draw class is not starved; score `max_plies` and double flags as draws in `outcome`. |
| 12 | Blockable-check wait is a *state with a running clock*: the stuck side can be flagged while waiting [1][21] | FreePass arena: a stuck board waits; its clock runs; flag ends it [L1][spec v1]. | Already right. Add the v2 "elapsed since last move" planes to the stuck board too, so the network can see how long the opponent has been stuck (how close to flagging) — it is the attacker's signal to keep denying blockers. |
| 13 | Communication vocabulary is a protocol between two agents who cannot see each other's board at full attention; a single agent sees everything [2][29][spec v1] | One agent, both boards: no messages needed [spec v1 §Scope]. | Nothing for the headline engine. For a future human-partner mode, Sjeng's mapping [29] is a ready spec: `sit` -> force pass on that board, `go` -> forbid pass, `+++n` -> add a bonus to the value of holding/getting a knight on the partner board. The piece-arrival head (spec v2 §C) is the natural source for *emitting* "+++n"/"---p" tells. |
| 14 | Simultaneous mates / capture-then-mated OTB draws [24][26] | Arena resolves by landing time: whichever event lands first wins [L1]. | Keep. Servers also resolve by arrival order, not simultaneity; OTB-only rule. |
| 15 | OTB rulesets where an empty-handed defender *is* mated (USCF, Hawaii) [27][25] | Not modelled. | Out of scope for beating Hivemind; note it as a rules profile if the engine is ever used for OTB analysis. |
| 16 | Typical TC is 2 0 / 3 0 online, 5 0 OTB, no increment [35][41][24][27] | Beat-stock runs used 60 s and 180 s clocks [beat_stock.md]. | Train with base time sampled across 60-300 s and increment fixed at 0; evaluate the gate at 120 s and 180 s (the FICS and chess.com norms), not only 60 s. |
| 17 | `minmovetime` 0.1 s floor; premove exists [14] | Arena think time = clock/k, jittered; no floor documented. | Add a 100 ms floor on every landing time so no move is faster than a server allows; otherwise self-play learns an impossible flag race. |

---

## 4. Sources

Web (fetched and read unless marked):

1. FICS Help: bughouse — https://www.freechess.org/Help/HelpFiles/bughouse.html — the server rule set: drop notation, pawn ranks, promoted-piece reversion, "no drop interposition possible" mate rule, same TC both boards, `moretime` per board, match end, disconnection ends both, draw needs both boards, `partner`/`ptell`.
2. FICS Help: bughouse_strat — https://www.freechess.org/Help/HelpFiles/bughouse_strat.html — 1995 strategy file: contact/knight checks, holes, initiative, Q4 N2 R2 B2 P1, stalling definition and its time precondition, `ptell`/`ptime`/`prefresh`/`pfollow`/`pobserve`, glossary (annoy, inject, noodle/spoon, rock, themed), openings.
3. FICS Help: ptell — https://www.freechess.org/Help/HelpFiles/ptell.html — `ptell string`, partner-only, 1995.
4. FICS Help: partner — https://www.freechess.org/Help/HelpFiles/partner.html — `bugopen`, partnership mechanics.
5. FICS Help: flag — https://www.freechess.org/Help/HelpFiles/flag.html — flag is a claim; both out = draw; displayed 0:00 not authoritative; abort if opponent lags.
6. FICS Help: v_autoflag — https://www.freechess.org/Help/HelpFiles/v_autoflag.html — up to 2 s delay; checked before increment; fails on disconnect.
7. FICS Help: bugwho — https://www.freechess.org/Help/HelpFiles/bugwho.html — `bugwho g|p|u`.
8. FICS Help: ptime — https://www.freechess.org/Help/HelpFiles/ptime.html — partner's clocks.
9. FICS Help: draw — https://www.freechess.org/Help/HelpFiles/draw.html — `draw` claims 50-move/threefold or offers; no bughouse specifics.
10. FICS Help: abort — https://www.freechess.org/Help/HelpFiles/abort.html — abort instead of flag when the opponent may be disconnected.
11. FICS Help: lag — https://www.freechess.org/Help/HelpFiles/lag.html — lag consumes clock unless timeseal; flagging a lagger is not unethical.
12. FICS Help: timeseal — https://www.freechess.org/Help/HelpFiles/timeseal.html — only thinking time charged with timeseal.
13. FICS Help: moretime — https://www.freechess.org/Help/HelpFiles/moretime.html — adds seconds to the opponent's clock.
14. FICS Help: clock — https://www.freechess.org/Help/HelpFiles/clock.html — Fischer clock, ms timing, `time 0` = 10 s, flag must be claimed, lagging timeseal users cannot be flagged, `minmovetime`, premove discouraged.
15. FICS Newsletter Nov 2001, "Introduction to crazyhouse Part 1" — https://freechess.org/Newsletter/Nov-2001/crazyhouse.html — practical reading of the interposition rule, its known inadequacy (drop-exchange loops), repetition count resets on drop.
16. ICS help "bughouse" (GICS mirror of the 1995 ICS help, common ancestor of FICS and ICC) — https://www.unix-ag.uni-kl.de/~chess/gicshelp/node169.html — same mate rule, promoted pieces, channels 24/25, `who B`, "wild bughouse" match; ICC-specific snippet (wild 24, `say`, `kib`) from search results of the now-redirected https://www.chessclub.com/help/bughouse (not readable; redirects to support.chessclub.com, which returned 403).
17. ICS help "bughouse strategy" (IanO, dogdog, Friar, 1995; GICS mirror) — https://www.unix-ag.uni-kl.de/~chess/gicshelp/node170.html — stalling, timeseal bug and "sealsitting", roles of White/Black, Q10 N7 R4 B2 P1 (differs from [2]).
18. Wikipedia: Bughouse chess — https://en.wikipedia.org/wiki/Bughouse_chess — rules survey (variants, repetition per server, three-moves clockless rule, both-flag draw, simultaneous mates), piece values, sitting/sitzkrieg/diagonal time, lag, communication conventions, online history.
19. Wikipedia: Bughouse chess, wikitext with references — https://en.wikipedia.org/w/index.php?title=Bughouse_chess&action=raw — which claims are sourced to what (von Zimmerman 2006, Manson & Hoover 1992, chessvariants.org 2002, Twitch 2022, Ebenfelt) and which are unsourced (the per-server repetition claims).
20. Wikipedia: Crazyhouse — https://en.wikipedia.org/wiki/Crazyhouse — crazyhouse rules; lichess 9th-rank FEN and `~`; chess.com encoding (flagged unverified).
21. chess.com Support: What is Bughouse? — https://support.chess.com/en/articles/8615092-what-is-bughouse — chess.com's rule statement: piece bank, pawn ranks, double step, "counterfeits", game end, blockable check is not mate even with an empty bank.
22. chess.com Support: How do I partner with someone in Bughouse? — https://support.chess.com/article/4750-how-do-i-partner-with-someone-in-bughouse — Custom Challenge, partner invite flow.
23. chess.com Support: Are there Bughouse tournaments? — https://support.chess.com/en/articles/8705941-are-there-bughouse-tournaments-how-do-i-join-one — random partner, partnership until a loss, schedule; no rules.
24. Laws of BUGHOUSE chess (OTB), v1.00, WBN / OG Bughouse CR, used at WCCC 2007 — https://www.chessfed.gr/old_site/wccc2007/files/bughouse_chess_complete_rules.pdf — 19-page FIDE-style law set: 1.5 potential-drop mate rule, 1.9 last-move drop, 1.10 simultaneous finish, 1.11 full communication, 2.2 draws, 5.8 no castling with dropped rook, 5.10-5.11 drops, 6.5 non-disruptive premove, 7.10 both flags, 9.4 freezing sanction.
25. "Bughouse Chess (Hawaii Rules)", RLP, updated 2002-06-15, distributed as WCCC 2007 "main rules" — https://www.chessfed.gr/old_site/wccc2007/files/bughouse_chess_main_rules.pdf — clock-move, king capture, both flags draw, castling with dropped rook allowed, double step for dropped pawn, start clocks together.
26. Complete ruleset for the WCCC 2004 bughouse tournament — https://wccc2004.juliasfairies.com/bughouse_rules.html — 5-minute FIDE blitz base, "potentially in his stock" mate rule, never stalemated, FIDE 9.6 cancelled, 4-fold repetition with no drop in between, both-team draw agreement, simultaneous-mate draw.
27. USCF Bughouse Chess Rules, rev. 2004-03-01 (2006 National High School Championships, Northwest Chess) — https://nwchess.com/articles/rules/bughouse_rules.htm — G/5, clock-move, flag claims, both flags draw, illegal move loses, no interposition rule, promoted pawn laid on its side, partners may suggest moves.
28. ChessBase, "Singapore's 1st Chess960 and Transfer Chess Championships" (Leung Weiwen, 2013-02-07) — https://en.chessbase.com/post/singapore-s-1st-che960-and-transfer-che-championships — the Transfer Communication Sheet: cardboard divider, only permitted communication channel, organisers' own rules, "no universally accepted rules".
29. Sjeng 11.2, partner.c — https://sources.debian.org/src/sjeng/11.2-15/partner.c — the ptell vocabulary as parsed by an engine: sit/go/fast/slow/abort/flag/x/dead/formula, +/++/+++ = 150/1000/50000 hand-value changes, trade +/- deltas, CANCEL_THRESH 3.
30. pychess.org: Bughouse — https://www.pychess.org/variants/bughouse — a live bughouse server besides FICS/chess.com; rules summary, double step/en passant, partner talk allowed, no physical interference.
31. pychess.org: Bughouse960 — https://www.pychess.org/variants/bughouse960 — search snippet only: no castling with a dropped rook.
32. lichess-org/lila issue #441 — https://github.com/lichess-org/lila/issues/441 — bughouse "probably not doable"; ornicar: drops cannot be encoded as PGN; open since 2015.
33. Caissly: Bughouse — https://caissly.pages.dev/variants/bughouse — unsourced summary page: "typically 3+0 or 5+0", "talk only at your own turn"; used only as a weak corroboration.
34. Chronatog, "ICS Interview – Gnejs" (2019-09-22) — https://chronatog.com/2019/09/ics-interview-gnejs/ — André Nilsson on FICS/ICC years, interest in piece values, chess.com communication "bad".
35. CCIF examples (BPGN game 3141831 from bughouse-db.org) — https://ccif.sourceforge.net/examples.html — concrete BPGN: tags, `1A./1a./1B./1b.` move labels, brace clock values, `Lag` tag, `TimeControl 120+0`.
36. ABCs of Bughouse (Jerry, 2021-12-18) — https://abcsofbughouse.blogspot.com/2021/12/the-abcs-of-bughouse.html — move when under 20 s, sitting is a timing strategy, sacsitting, partop, feed/hang, last-trench methods.
37. chess.com blog, JarlCarlander, "Bughouse Buttons" — https://www.chess.com/blog/JarlCarlander/bughouse-buttons — **search snippet only** (page unreadable): old rules limited communication to a symbol set, chess.com icons mirror it.
38. chess.com blog, Sorsi, "Basic Bughouse Tips" — https://www.chess.com/blog/Sorsi/basic-bughouse-tips — **search snippet only**: sit/go/uptime/+++/---/flow definitions, no-sit club rules, no increment.
39. chess.com blog, Sorsi, "Time Management in Bughouse" — https://www.chess.com/blog/Sorsi/time-management-in-bughouse — **search snippet only**: "sit until 10, then low flow"; avoid sitting without reason.
40. chess.com forum, "Problem with bughouse draw" and related repetition threads — https://www.chess.com/forum/view/chess-variants/problem-with-bughouse-draw — **search snippets only**: board-only repetition counted in drop loops; draw must be claimed.
41. chess.com blog, chessisgreat940, "Time Controls and Variants" — https://www.chess.com/blog/chessisgreat940/all-different-time-controls-and-variants-in-chess-com — **search snippet only**: "standard bughouse time control is 3|0".
42. FICS Games Database, About — https://www.ficsgames.org/about.html — **search snippet**: bughouse games excluded, held at the Bughouse database (bughouse-db.org).

Not readable (for the record): https://www.chess.com/terms/bughouse-chess, https://www.chess.com/blog/godblitz/how-to-play-bughouse, https://www.chess.com/article/view/bughouse-chess-tips-for-beginners, https://www.chess.com/forum/view/chess960-chess-variants/ending-the-game-in-bughouse (fetcher failed), https://lichess.org/variant/crazyhouse (unreachable from this machine), https://www.chessvariants.org/multiplayer.dir/tandem.html (403), https://support.chessclub.com (403), archived bughouse.be BPGN standard, USCF 2006 PDF and pion.ch Geneva rules (web.archive.org blocked).

Local:

- L1. `arena/RULES.md` and `arena/state.py::is_bughouse_mate` — the arena's FICS-style profile (geometric blockable check, waiting, flags, no draws, `~` marker).
- L2. `third_party/hivemind/README.md` — 37 planes per board (plane 31 time advantage, 34 halfmove, 35-36 repetition), heads, `TimeAdvantage` gating of passes, shared time-advantage token.
- L3. `third_party/hivemind/engine/src/environment/joint_action.h` — `is_single_pass_legal` / `is_double_sit_legal` and the `g_freePass` override.
- L4. `third_party/hivemind/engine/src/environment/board.cc::is_checkmate` / `can_partner_provide_blocking_piece` — partner-capture search, `TimeAdvantage` branch, `assumePartnerCanBlock`, FreePass wait.
- L5. `third_party/hivemind/RULES.md` — mate-race and stalemate handling notes.
