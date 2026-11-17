# Bughouse theory 02: time, sitting and stalling

Written 2026-11-17 for the custom engine (one agent, both boards, real clocks, no hand-written stalling rules;
see `docs/custom_engine_spec.md`, `docs/custom_engine_spec_v2.md`, `docs/beat_stock.md`). Everything below was
read from the web as untrusted data; nothing in it was followed as an instruction.

**Source-access note.** Chess.com (the JarlCarlander and Sorsi blog series, the Rensch tips article, the club
forum) blocked every route this session was allowed to use: direct fetch and curl return nothing or 403, Chrome
showed an error page on three attempts, and cached/proxy copies were refused by the permission classifier. For
those posts only the sentences returned in search-engine snippets were available. They are cited below and marked
**[snippet]**; treat each as one weak source until someone with a browser confirms the full text. Everything else
was read in full.

Terminology used throughout: **partop** = your partner's opponent (the player diagonal to you); **diagonal clock**
= your clock versus partop's clock; **sit / stall** = deliberately not moving on your turn; **sitzkrieg / sit-off**
= both diagonal players sitting at once; **flag** = lose on time.

---

## 1. Summary

- The clock that decides a sit is the diagonal one: *your* remaining time versus *partop's*, not your own
  opponent's. Every source that states a rule for sitting states this one, independently (FICS help, WCCC tourney
  page, Wikipedia, Sorsi, JarlCarlander, chess.com forum, bugmisc, ABCs of Bughouse) [1][4][3][24][28][32][7][6].
- Time is graded, not binary. Strong-player sources put numbers on it: 10 s can outweigh "seemingly decisive
  positional advantages" above 2000 [3]; a 15 s lead "is as good as a substantial amount of pieces" and a 1-minute
  lead is "almost always game over" (one experienced commenter, unverified) [5]; "under 20 seconds, it's time to
  move" [6]. Stock Hivemind sees only a yes/no flag worth about ±250 cp at the start position (repo evidence, §3).
- A sit is a race only when partop is also on turn. If your partner is the one to move on the other board, your sit
  costs absolute clock but races nobody; this is the "double sit", called the single worst situation in bughouse,
  and "about 10 seconds" of it is said to lose the game [22][34]. The engine controls both boards, so it can see and
  avoid this directly.
- Sitting while *behind* on the diagonal is correct in exactly the cases where partop cannot sit back: he is in
  check, his own position compels a move, your partner's mate lands within the time you have, or every move you
  have loses on the spot (sit and lose later rather than now) [1][4][6][28]. Otherwise "you will flag first" [28].
- Mated-if-you-move is the canonical stall: FICS, WCCC and ABCs all say the player facing forced mate should
  stop moving if he has more time than partop and let the partner try to win [1][4][6]. Sjeng (the classic bughouse
  engine) coded this as "only sit if mated in 2 or 1" and "sit if losing a --- piece" [10].
- "Move fast" is the default; thinking is an exception spent on the critical few moments and best done while
  partop is thinking [6][23][25]. The team's speed is set by its slower player [7]; bughouse is essentially never
  played with increment, so time is never recovered [27].
- Material and time trade against each other continuously: "time is more important when the material count is
  against your side on either board" and "a board with few pieces generally favors the fastest team" [7][6].
  A master with a won endgame and more time than partop can have the partner sit while he converts in under 30 s
  ("Tournament Game strategy") [7].
- The sit-until-10-then-flag tactic exists at club level: partner sits until 10 s remain, then plays fast moves to
  flag the opponent, who has been forced to keep moving [33]. The engine's tree must resolve such sit-offs
  analytically, not by rollout; the clock build's sit-off terminal rule (spec v2 D1) does this.
- Time controls disagree: WCCC rules 3+0 [15], Wikipedia "5–10 min" [3], chessbio 3+0/5+0/2+1 [12], and an
  experienced commenter says "zero good players play 5min" because it degenerates into minutes of sitting [5].
  The arena's 60 s games are shorter than any human standard; the 3-minute arm is the one humans would recognise.

---

## 2. The knowledge

### 2.1 What a sit is and what it is for

Three distinct purposes appear in the sources:

1. **Waiting for a piece** that mates or wins ("one piece from your partner would give you mate" [18]; "all you
   need is your partner to give you a bishop" [17]). Chess.com's glossary (via JarlCarlander's beginner post) defines
   sit as "to stall, or to not move (almost always when you have more time) so that one's partner's opponent
   doesn't get a piece he or she really needs" [27 snippet].
2. **Denying a piece**: not making the trade or capture that would hand partop the piece he needs ("waiting on the
   clock to avoid handing the opponents a critical piece via a trade" [12]; "This strategy keeps pieces in your
   partner's bank, allowing them to continue their attack without interruption" [19]).
3. **Refusing to lose now**: "you realize that if you move, you are mated in one. Therefore, you simply decide not to
   move and let your partner try to win the game" [1]; "if one player is about to lose, he can simply stop playing,
   in the hope that his partner can win before time runs out" [14].

FICS adds a fourth, offensive one: stall "to make your opponent move before a piece your partner says is coming
arrives" [1], i.e. force the opponent on *your* board to commit before the reinforcement lands.

ABCs of Bughouse insists on the distinction that matters for an engine: "Sitting and thinking may look similar,
but sitting is using a timing strategy. Thinking is not" [6]. A sit is a decision about the other board's clock; a
think is a decision about your own move quality.

### 2.2 The diagonal clock comparison

The rule, in the sources' own words:

- FICS help: stalling "only works with more time: you must have more time than your partner's opponent" [1].
- WCCC 2007 tourney page: "you must have more time on your clock than your partner's opponent", or "you will flag
  first"; if your partner stalls with less time than *his* opponent, "your partner will flag first" [4].
- Wikipedia: "This diagonal time advantage is more important than the difference on the clock between opponents on
  the same board" [3].
- Sorsi: "what matters is the time of one's partner's opponent – it is this time that will judge whether one can
  keep a piece from his or her partner's opponent" and defines **uptime** as the extra time you have over partop
  [24 snippet].
- Chess.com club forum: "look at partopps' clock and not your opponent's clock ... if you are ahead of partopps,
  you win all sit/stall situations and time trouble races" [32 snippet].
- bugmisc: "In general, you want to have more time than your partner's opponent"; and for defence, "Your opponent
  must have less time than your partner" when you have mate [7].

Why diagonal: when you sit on board A, the player whose clock also runs is whoever is on turn on board B. If that
is partop, the two of you are in a race and the lower clock must move or flag. Your own opponent's clock is
stopped, so it is irrelevant to the race (it matters later, for the flag race on your board).

**Corrected (the original paragraph here claimed the two diagonals can have opposite signs; they cannot under
normal conditions).** On each board exactly one clock runs at any moment (the side to move, sitting or not), so with
equal time controls, synchronised starts and no increment the two boards' clock sums stay equal:
A_us + A_them = B_us + B_them, hence **A_us − B_them = B_us − A_them**. There is one diagonal difference, not two, and
it equals half the team total (A_us − A_them) + (B_us − B_them). Upstream Hivemind's `time_state.h` relies on the
same invariant ("the two boards' clock sums stay equal and a team's lead over its diagonal opponents is one number").
The two diagonals diverge only through lag compensation (timeseal charges thinking time only, so a laggy board
loses less total clock; this is why Wikipedia says lag changes the diagonal difference [3]), `moretime`, or
unsynchronised starts. What differs between the two boards is not the sign of the race but *who is on turn* and
the *absolute* clocks (how close each seat is to flagging), which is what the engine must read per board (§3.1).

### 2.3 Sitzkrieg and sit-offs

"The situation where diagonal opponents sit at the same time is known as a 'sitzkrieg' ... The difference in time
between the diagonal opponents will eventually force one party to move" [3, citing Manson & Hoover 1992].
WCCC: "whether your partner or your opponent has the most time left on their clock" decides it; "A player who runs
out of time must move and immediately loses" [4]. The resolution is purely the diagonal clock difference, which is
why the diagonal comparison is the number to track.

A sit-off is not symmetric in cost: the side sitting *for* a piece is also delaying its own attack, while the side
sitting to *deny* is only burning clock. Sources frame who "wins" the sit-off purely by clocks, but the position
after the forced move decides the game; nobody quantifies this.

Degenerate sit-offs are a known pathology of long controls: "5min bug = 3+ min of sitting"; "If that was a 5min
game, it would have been a 4:48 wait on move 6"; "they tried to fix their problem by changing the time control
from 2min to 5-10min" (which made it worse) [5]. Clockless OTB bughouse used "a player may not delay his move beyond
the time that it takes for his partner to make three moves" [3, citing Bodlaender et al.]; the WCCC OTB laws use
3 minutes each and FIDE blitz rules [15]. Online, FICS had a timeseal bug that let a player "sealsit" forever,
which the help file calls "unethical and frustrating" [1]. Some casual groups "set house rules limiting sitting"
[18]. None of this applies to the arena (FICS mate rule, no sit cap in free mode), but it shows human data from
long controls will contain long sit-offs the engine should not learn as normal.

### 2.4 How much is time worth

Numbers given (all from single sources unless noted; none is measured):

| Claim | Source | Strength |
|---|---|---|
| ">2000 ... a time advantage of as little as 10 seconds can be more important than seemingly decisive positional advantages" | Wikipedia, citing Chuck Moulton's 2022 BWC annotation [3] | Secondary; the Moulton text itself was not reachable |
| "a 15 second time advantage in bughouse is as good as a substantial amount of pieces"; "A one minute advantage is almost always game over" | HN commenter xenophanes [5] | One anonymous but evidently experienced player |
| "When you are under 20 seconds, it's time to move" | ABCs of Bughouse [6] | One author |
| "even them taking 3 seconds longer on a move can put you in a much tougher spot"; teammate "spend[ing] 14 seconds before taking the Rook" cost the game on time | Bughouse Effect essay [16] | Anecdote |
| "A chess master should be able to win a won endgame in 30 seconds or less" | bugmisc [7] | One author |
| "Being down some time is not game losing, all other things being equal" | JarlCarlander [28 snippet] | Snippet |
| "time is an even greater asset in bughouse than material" | chess.com "Basics of Bughouse" [35 snippet] | Snippet |
| Stock Hivemind root eval swings about −270 cp (flag off) to +250 cp (flag on) at the start position | repo, `arena/players/hivemind.py` lines 27-28 | Measured in this repo |

Reconciling them: the human claims are all consistent with a value of time that is small near parity, steep
between roughly 10 and 60 s of diagonal difference, and saturating beyond a minute. The 10-s claim is specific
to strong players (whose moves are fast enough that 10 s is many moves); at lower levels the same 10 s is one think.
The stock engine's ±250 cp step at zero is roughly the magnitude humans assign to a *decisive* lead, applied to a
1 ms edge.

Where time matters more: "Time is more important when the material count is against your side on either board";
"A board with few pieces generally favors the fastest team" [7]; "time is more critical if you are down material or
there is a sparse board"; "Time is important, especially near the end of the game" [6]. Where it matters less:
"being on the right strategic page is more important than even not blundering material or being up time" (one
chess.com author, context unclear, possibly aggregate scoring) [30 snippet].

### 2.5 Piece values in bughouse (for the time-vs-material trade)

"Common bughouse values" are pawn = 1, bishop = knight = rook = 2, queen = 4 [3, citing von Zimmerman 2006 and the
Gnejs interview]; pawns gain and long-range pieces lose relative value because drops give every reserve piece the
same mobility [3]. Gnejs (first 2800+ on FICS) says his analytical work began with "What are the pieces worth in
bughouse?" [11], and Sjeng's changelog records "Modified piece values as per Gnejs request" [10], but neither
gives the numbers. This only matters here as the unit in which the "15 s ≈ substantial material" claim is denominated.

### 2.6 Move fast, think rarely: tempo of play

- "You must play bughouse like you play bullet chess (1 + 0) unless you have to sit for some reason" [34 snippet].
- "The longer you think, the worse your move becomes" [25 snippet]; "The most common problem that burns time is
  starting to think while under attack" [22 snippet].
- "Don't think during a time scramble"; "Get in the habit of moving when there isn't anything exciting to do";
  "Save your time for when you really need it"; "you can think a few times during the game in the most critical
  positions ... or when your partop is thinking" [6].
- "The speed of a team is determined by its slower player, not its faster one" [7].
- Attacking: "go faster than fast before the other side can counter-drop" [34 snippet]; when your king is exposed
  "you MUST keep attacking and checking his King" because a stall by partop will starve the attack [16]; "a run of
  checks burns their clock" [18].
- Counter-view for weaker players: "For new players taking 2-5 seconds to come up with a move is often superior";
  "A new player forced to move in 0.5 seconds ... will make terrible moves" [5]; JarlCarlander reportedly thinks
  slowing down is worth it below 2100 [31 snippet]. This is a level effect, not a disagreement about strong play.
- "Think when partop is thinking" [6] is the human version of a free think: while partop's clock runs, yours
  running too costs nothing on the diagonal.

Tempo in the chess sense is also stressed: "Tempo is important. Look for situations where you have a better move
than recapturing" [6]; "it is especially important to guard against checks and/or make sure trades come on your
tempo" [7]; "tempo management (your moves take time that your partner doesn't get)" [13]. The last line is the
clearest statement of the shared-resource view: on one machine, or in one human head, attention spent on one board
is time not spent on the other.

### 2.7 When to sit, when not to

**Sit** (sources agree):
- You are ahead on the diagonal and one specific piece mates or wins [17][18][24].
- You are facing forced mate and have more time than partop: "If you are getting mated, don't move if you have more
  time than your partner's opponent" [6]; FICS [1] and WCCC [4] identical.
- The piece you wait for actually hurts: "when you sit for a piece, make sure that piece hurts your opponent"
  [22 snippet]; "sitting for a piece just to start unclear attack will just kill your time for no good reason"
  [26 snippet].
- Small gains, once ahead: "sit for small gains once you are ahead on time"; "use extra seconds as soon as you gain
  them" [23 snippet]. Note the prescription is to *spend* a time edge immediately, not hoard it.
- Against stronger opponents: "more sitting means fewer chances for the stronger side to outplay the weaker side"
  [23 snippet].

**Do not sit**:
- Behind on the diagonal, for a piece: "Don't pause to wait for material when you're down" [6]; "you can't do it if
  you are down in time" [17].
- In check, unless a drop would block: "Don't sit when you're in check unless a blocker will help you" [6].
- As a beginner at all: "as a beginner, you should keep moving and avoid sitting" [24 snippet]; "don't sit for
  pieces unless your partner told you to do so" [24 snippet].
- Aimlessly: "even making an empty move is much better than just sitting and thinking what to do" [search snippet,
  source not individually confirmed].
- As a defensive plan: "Don't think in terms of defending by having your partner stall unless its the only
  alternative" [7]; "I don't recommend defending by playing faster, telling your partner to sit, or telling your
  partner to hold a certain piece. These should be used only as last trench methods" [6].

**The hardest judgement**: "The most difficult part of this game is to know when to sit and when to move. You need
to play really quickly in order to be uptime but you need to be uptime so you can sit for pieces that will hurt your
opponent" [23 snippet]. Two failure modes: "strong chess players don't know when to stop and sit for pieces. On the
other hand, those who are bad at chess usually sit way too much and try to play mostly with piece drops" [23
snippet]. Sorsi's stated resolution is a balance; no source gives a decision rule beyond the diagonal clock plus
"does the piece hurt".

### 2.8 Sitting while behind on the diagonal: when it is still right

No source states this as a rule, but the cases follow from the mechanics above and several are given as examples:

1. **Partop cannot sit back.** He is in check, or any sit costs him (his king is under a drop attack that lands on
   the next move). Then your sit is not a race; ABCs' "Don't sit when you're in check unless a blocker will help
   you" [6] is the mirror image: a checked player cannot afford the sit-off.
2. **Partner is on turn, and his next move mates or wins the piece.** The clock running on the other board is your
   partner's; the only cost is your absolute time. Correct if your absolute time covers it, wrong if it becomes a
   double sit [22][34].
3. **Any move loses on the spot.** Sitting converts "lose now" into "lose when my clock runs out unless partner
   wins first", which is strictly better; FICS and PBeM both describe it [1][14]. Whether you are behind on the
   diagonal only changes how long partner has.
4. **The flag race favours you anyway.** The "sit until 10, then low flow" plan [33 snippet]: partner sits deep into
   his own clock while the opponent on his board is forced to keep moving (he is not sitting, he is attacking), then
   plays fast moves to flag him. This is sitting while *ahead* on the diagonal but spending almost all of it; it
   shows the intended use of a time edge is to convert it into a flag, not to keep it.

Against: "sac-stallers" who "waste all their time and then realize they don't have time later" [28 snippet];
"Someone who is always worried about a small time deficit is a sacsitter"; "too many [sacrifices] will cause you
to lose on both boards" [6]. Sac-sitting (sacrifice to feed partner, then sit) is "not recommended" by ABCs [6],
is the pattern Rensch says you must learn to beat ("knowing how to play against the SS (Sac and sit) tactic will
seriously bring your rating up") [34 snippet], and JarlCarlander says its definition has "no consensus" [29
snippet]. Disagreement here is about style, not about the clock rule.

### 2.9 Partner-clock reasoning (what the two humans do that one agent must do alone)

- Humans exchange clock state: "Clock information is also quite good to tell your partner" [1]; call "Up" or "down"
  so your partner knows whether to speed up [18]; "constantly track your partner's opponent's pace" [22 snippet].
- Humans alternate: "sometimes what's required is for the team to sit and defend in alternation" [28 snippet]; a
  partner who is ahead can sit for you, one who is behind cannot ("you must have more time than your partner's
  opponent" applies to *his* diagonal, not yours) [4].
- The partner tells the sitter when to stop: FICS "go" = "stop sitting and move" [27 snippet]; Sjeng automated it:
  "Automatically tell other Sjeng to go if something no longer mates us" [10].
- Making up lost time is a joint problem: "when you're behind, ask your partner for advice on how to make up the
  time" [28 snippet]; the practical ways are the partner playing faster, forcing trades that give you premovable
  recaptures, or the partner sitting while you have a forced sequence.
- Resignation thresholds encode the same logic: resign only if "you are force mated, you have less time than your
  partner's opponent" and the other board cannot save you [7].

### 2.10 Classic engine practice (Sjeng)

Sjeng, named after Gnejs and written to play bughouse on FICS around 1999-2002, is the only open engine whose sitting
logic was readable this session [9][10]:

- Sits are rule-based, not searched: "Sjeng will now only sit if mated in 2 or 1" (release 5), "Sit if losing a ---
  piece" (release 7, i.e. a piece the partner says mates him).
- `sit` sets `must_sit`, `go`/`move` clears it and sets `must_go = 4`; `CheckBadFlow()` re-tests each piece type
  (`CANCEL_THRESH` = 3) and sends `go` when the mating threat disappears [9].
- The partner code reads **no clocks at all**: "time" is just an alias for the `fast` command; "Check whether we need
  to move faster than 1 second in fast mode"; "Go fast if fellow Sjeng is dead" [9][10].
- "Use ghost pieces to anticipate trades" (release 9) is an early piece-arrival model.

So the classic approach was: hand-written sit rules keyed to immediate mate, no diagonal-clock reasoning, and a
"fast mode" when the partner is in trouble. Stock Hivemind improved on this with a learned value but kept a
hand-written yes/no gate (§3).

### 2.11 Online-specific effects

Lag and premoves change effective clocks: FICS timeseal made flagging a sitter impossible until he moved [1];
Wikipedia says "it is good sportsmanship to restart the game when this difference gets too large" [3]; JarlCarlander
advises "anti-premoves" when "the time is extremely close and your position very desperate" [28 snippet] and sees
no reason to prohibit premoves. Lichess forum users report premoves cost little or no time and lag above about 1 s
is not fully compensated (forum claims, not documentation; not bughouse-specific) [search results, not cited].
Relevant only insofar as human game data from online servers has clock noise the arena does not.

---

## 3. Implications for the engine

How stock Hivemind handles time, from the repo (`third_party/hivemind`): one UCI option `TimeAdvantage true|false`
"gates the bughouse waiting rules: a team that is up on time may pass on a board it is on turn for, and may pass on
both. Without it, passing on an on-turn board is only legal when the partner board captures. The flag also feeds the
network as an input plane" (`README.md`, "Team and Time Advantage"; `engine/src/environment/joint_action.h`
`is_double_sit_legal`, `is_single_pass_legal`; plane 31, `planes.cc`). The crossboard-risev33 network carries "one
shared time-advantage token" (`README.md`). The arena sets the flag from `team_time_edge`, the sum over the team's
seats of (own clock − same-board opponent clock) (`arena/players/heuristic.py`); in training (`domain/board.py`,
`data/generate_planes.py`) the flag is `times[0][side] − times[1][side]`, the board-A diagonal difference, which has
the same sign (§2.2). The arena caps stock at
`max_consecutive_sits = 2` sits of `sit_ms = 1000` and otherwise sends `RequireMoveOn` (`arena/players/hivemind.py`).
The root mate probe runs only without `TimeAdvantage` (`agent.cc`, "Ahead on time, this team may sit the board a line
is played on, so no single-board mate is forced against it"), and checkmate itself is defined relative to the flag
(`is_checkmate(~teamSide, !teamHasTimeAdvantage)`). The flag swings the root eval by about 500 cp at the start
position (`arena/players/hivemind.py` comment). Inside the stock tree clocks are frozen.

### 3.1 The diagonal clock, per board

**Stock:** captures the *sign* and nothing else. **Corrected:** the stock flag is exactly the sign of the diagonal
difference (`board.py::time_advantage` = `times[0][side] − times[1][side]`, our A seat minus the diagonal opponent's
B seat), and the arena's team sum has the same sign because, by the §2.2 invariant, the sum is twice the single
diagonal difference. There is no "ahead on A's race, behind on B's" case to distinguish. What stock lacks is the
magnitude (1 ms and 2 minutes look the same) and the absolute clocks.
**v1 planes:** channels 37-40 give, per board block, our seat and the *same-board* opponent. The diagonal (our seat
on A vs their seat on B) is only available by combining block A's own-clock planes with block B's opponent-clock
planes across the two cross-attention layers. Learnable, but it is the single most important derived quantity in
this topic and every human source names it directly.
**Change (inputs):** add one signed plane, `tanh((t_ours_A − t_partop_A) / 15)` (15 s scale from [5]), written into
both board blocks (it is the same number for both), plus per block `min(t_partop / 300, 1)` so each block also sees
how close the diagonal opponent is to flagging. Keep the zero-init widening and it04 parity. Cheap; removes a learning burden exactly where the lockstep finding showed
the network was fragile. **Change (evaluation):** log, per sit decision in the arena, the diagonal difference at
that moment and the game outcome, and bin win rate by difference; the human claims predict a steep region between
about 10 and 60 s and near-indifference below 5 s. This is the first measurement of "what is a second worth" that
anyone will have for bughouse.

### 3.2 Graded time value instead of a 500 cp step

**Stock:** captures only the sign; the magnitude is a step function (measured ±250 cp). A 1 ms lead and a 2-minute
lead get the same input.
**v1 planes:** `min(t/300,1)` and `exp(−t/10)` per seat give absolute scale; the difference is implicit. Fine as
far as it goes.
**Change (training data):** the value head only learns a graded value of time if the data spans the range. Add
**time-odds openings** to self-play: start a fraction of games (suggest 25%) with asymmetric clocks, drawn so the
diagonal difference at move 1 covers ±5, ±15, ±30, ±60 s (Corrected: "opposite signs on the two diagonals" is
impossible while the two boards' clock sums are equal, §2.2; what *can* vary independently is the absolute level,
so also sample both teams low, e.g. 20 s vs 25 s, not only 60 s vs 65 s).
Without this the self-play distribution near the start is always symmetric and the steep region is reached only
late, in positions already decided. **Change (evaluation):** a time-odds match ladder against stock (stock given
+5/+15/+30/+60 s) measures the engine's Elo per second and directly tests the "15 s ≈ material" claim.

### 3.3 The double sit, and whose clock a sit races

**Stock:** partially. Both boards are in one joint action, so a double pass is one action, but it is legal only
with `TimeAdvantage` and alternating turns (`is_double_sit_legal`), and the arena caps consecutive sits. Nothing
tells the network *who else is on turn*: the on-turn planes (25) exist, but the clock that matters for a sit on A is
B's on-turn seat's clock, which §3.1 makes explicit.
**Custom engine:** `FreePass` lets both boards pass with no gate; `TreeMoveMs` charges every on-turn seat each
macro-ply, so a double sit is correctly priced as burning two of our clocks while the opponents' stand still. The
v2 elapsed-time planes (41-42) tell the network how long each on-turn side has already been thinking, which is the
human "think when partop is thinking" signal. **No further input change needed.** **Corrected (audit done):** the
opponent team's pass rights in the tree *do* equal ours. `searchthread.cc` (~line 695) sets
`leafTeamHasTimeAdvantage = (teamToPlay == rootTeam) ? flag : !flag`, every board's action list always contains
`MOVE_NONE` (line 604), and `node.h` hands that flag to `JointCandidateGenerator::initialize`, which filters joint
actions through `is_joint_action_legal` (`joint_action.h` line 187); under `g_freePass` that returns true for any
pass by either team. Line 65 is only the waiting-board mate check. The symmetry test is still worth having as a
regression guard. Add a C++ test: a symmetric sit-off position must evaluate to a draw with equal clocks and
flip sign when one diagonal clock is moved by a second.

### 3.4 Sit-offs must be resolved analytically, not rolled out

**Stock:** cannot. Clocks are frozen in the tree, so a sit-off is a repetition the search never terminates.
**Custom engine:** spec v2 D1 (implemented in `searchthread.cc`: "a full round with no move on either board is a
sit-off, decided by which seat on turn would flag first", `board.sit_off_losers()`), plus flags as terminal losses
(`board.flagged_teams()`). This is exactly the human rule from §2.3 (the diagonal difference decides). **Corrected:** the original
text claimed D1 misses an "asymmetric sit-off" where we sit on A, partop sits on B, and our partner on B and our A
opponent still move. That situation cannot exist: if we are on turn on A, our A opponent is not; if partop is on turn
on B, our partner is not. A sitzkrieg always has exactly the two diagonal seats on turn, both passing, so D1 covers
it. The real gap is different: a **one-sided sit** (we pass on A while board B keeps playing) is not decided by the
clocks at all but by what happens on B, and the tree reaches it only through many `TreeMoveMs` macro-plies of
B-moves. **Change (search):** make the pass duration event-driven, a **fast-forward pass**: a pass on A lasts until
the next B event (the next B move in the line, or A's clock reaching 0, or `TreeMoveMs` × k for a configurable k)
instead of one fixed `TreeMoveMs`, so that "sit for the knight" costs the right amount of clock and is resolved in
depth 1-2 instead of k. This is the tree-side version of the arena's event-driven wait (`free_sit_ms`).

### 3.5 Sitting while behind, and mated-if-you-move

**Stock:** hand-written and inverted in the dangerous direction. Without `TimeAdvantage` a team cannot sit at all
(except when the partner captures), so a stock seat facing forced mate while 1 ms behind on the flag must
move into mate even though 1 ms is no real disadvantage (Corrected: the earlier text said "even if its own diagonal
race is winning"; the flag *is* that race's sign, §2.2, so the two cannot disagree), and `is_checkmate(..., !teamHasTimeAdvantage)` scores a
blockable check as mate for the team behind on the flag. Sjeng's "only sit if mated in 2 or 1" was the opposite
hand rule.
**Custom engine:** `FreePass` plus the arena mate rule ("a stuck board or a blockable check is never a loss,
whatever `TimeAdvantage` or the partner board") removes the rule; the search with real clocks decides. The §2.8
cases (partop in check, partner on turn, every move loses, flag race favourable) are all representable: cases 1-3
appear within a few macro-plies, case 4 needs §3.4. **No new rule should be written**; this is the point of the
project. **Change (evaluation):** build a regression set of 20-50 hand positions for each §2.8 case with clocks
set each way, and assert the chosen joint action (sit vs move) matches the human verdict. The sign flips with a few
seconds of diagonal difference are the test, not the absolute values. Also a self-play stat: fraction of games
ending by flag, and fraction of flags where the flagged seat was sitting when it flagged versus moving; humans
describe both, and a network that only ever flags while sitting has learned to over-sit.

### 3.6 Move fast versus think: budget allocation across two boards

**Stock:** fixed node budget per search; the arena's `clock_k` arm (own clock / k) was the first time manager
(`docs/beat_stock.md`). `serial` charging means a search on one board delays the other, which is the "team speed is
set by the slower player" effect [7] inside one machine.
**v2:** `think_gain` head (B) scales the budget by whether thinking changes the move, and the elapsed-time planes let
the value head price time already spent. This captures "don't think during a scramble" and "save your time for when
you really need it" [6] if the labels are good. **Gaps and changes:** (a) the budget rule is per board; human
practice says allocate jointly, so when both boards are on turn the controller should ask `think_gain` for both and
search the board with higher gain first, giving the other a 16-node quick look and a move (the human "think while
partop is thinking" is the case where only one board is urgent). (b) Add a **minimum move latency** to the arena and
to `TreeMoveMs` so that a seat with 3 s left cannot make 30 moves; humans at 10 s make perhaps 10-20 moves and the
"sit until 10" tactic depends on that count. Without it the engine will learn that very low clocks are safer than
they are against a human or a stock engine with real latency. (c) `k_range` 60..300 at 60 s means 200-1000 ms
thinks at the start; humans at 3+0 average roughly 2-5 s early and well under 1 s late [5][6]. The jitter range
0.5..2.0 is good; consider a move-number dependence so early moves are slower, matching how human data looks.

### 3.7 The piece a sit is for

**Stock:** nothing explicit; the value head implicitly knows whether a pocket piece mates.
**v2:** the arrival head (C) predicts pieces arriving per pocket within 5 s. Humans sit for a *specific* piece and
need to know *when* it comes (FICS: stall so the opponent must move "before a piece your partner says is coming
arrives" [1]). **Change (outputs, training target):** add expected **time to first arrival** per pocket and piece
type (regression, masked when none arrives within the horizon), or replace the 5 s count with counts in two bins
(0-2 s, 2-5 s). This turns the sit decision into "will the knight arrive before partop's clock forces him" and is
cheap given the recorder already knows landing times.

### 3.8 Time controls and what the human knowledge transfers to

Human theory is written for 2+0 to 5+0, no increment [3][5][12][15][27]. The arena's main setting is 60 s. The
10 s / 15 s / 20 s thresholds are tied to human move latency (about 1 s per move at speed), so at 60 s with an
engine that moves in 200 ms the equivalent thresholds are smaller; the arena's 3-minute arm is where the numbers
above apply as stated. **Change (evaluation):** report every gate result at both 60 s and 180 s, as
`docs/beat_stock.md` already requires for the time-management arms, and do not read the human thresholds into 60 s
games without rescaling by measured moves-per-second.

### 3.9 What the single-agent design makes unnecessary

Everything in §2.9 (telling the partner your clock, "go", "up/down", requests, who sits for whom) is communication
between two players who each see one race. One agent with four clocks and both on-turn planes sees both races at
once; the spec's choice to drop messaging is consistent with this. The one thing humans do that the agent must
still do is **alternate**: sit on the board where a sit gains something (a piece denied or awaited, a mate avoided)
and move on the other. Corrected: this is not "the board whose diagonal is favourable", because both boards share
one diagonal sign (§2.2); it is a question of which board's position profits from the shared time edge. With §3.1
planes the edge is a direct read, and §3.5's regression set should include positions where a sit is right on exactly
one board.

---

## 4. Sources

Read in full unless marked [snippet]. Chess.com URLs were unreachable from this session (see note at top).

| n | URL | Title | Contribution |
|---|---|---|---|
| 1 | https://www.unix-ag.uni-kl.de/~chess/gicshelp/node170.html | FICS help: bughouse strategy (mirror) | The stall: mated-if-you-move, "must have more time than your partner's opponent", stall to force the opponent to move before a piece arrives, timeseal sealsitting, tell partner your clock |
| 2 | https://www.unix-ag.uni-kl.de/~chess/gicshelp/node169.html | FICS help: bughouse (mirror) | Same time control on both boards, match ends on any flag, ptell |
| 3 | https://en.wikipedia.org/wiki/Bughouse_chess | Wikipedia: Bughouse chess | Sitzkrieg, diagonal > same-board clock, 10 s claim (>2000, citing Moulton 2022), three-move clockless rule, piece values 1/2/2/2/4, simultaneous flag = draw |
| 4 | https://www.chessfed.gr/old_site/wccc2007/bughouse_tourney.html | WCCC 2007 bughouse tournament page | Stall and sitzkrieg mechanics in full, "your partner will flag first", initiative advice |
| 5 | https://news.ycombinator.com/item?id=3803105 | Hacker News thread on bughouse (2012) | xenophanes: 15 s ≈ substantial material, 1 min ≈ game over, 5-min games are 3+ min of sitting, 4:48 wait on move 6; Steko: new players 2-5 s per move |
| 6 | https://abcsofbughouse.blogspot.com/2021/12/the-abcs-of-bughouse.html | The ABCs of Bughouse (Jerry, 2021) | Under 20 s move; sitting is timing, thinking is not; don't wait for material when down; don't sit in check; if getting mated don't move if more time than partop; think when partop thinks; sacsitting not recommended; time matters more when down or sparse |
| 7 | https://bugmisc.blogspot.com/2018/04/tips-for-beginners-that-will-raise-your.html | Bughouse and Zhouse: Tips for Beginners (2018) and About Trades | More time than partop; team speed = slower player; time more important when down material or sparse; master wins won endgame in 30 s while partner sits ("Tournament Game strategy"); resign conditions; defence by partner stall only as last resort |
| 8 | https://www.chessfed.gr/old_site/wccc2007/files/bughouse_chess_complete_rules.pdf | Complete Bughouse Chess Rules (WCCC 2007 PDF) | Fetched but unreadable (binary); not used for claims |
| 9 | https://sources.debian.org/src/sjeng/11.2-15/partner.c | Sjeng partner.c | sit/go flags, must_go = 4, CheckBadFlow CANCEL_THRESH 3, no clock reads, "time" = alias of fast |
| 10 | https://sjeng.org/news.html | Sjeng release notes 1999-2002 | "only sit if mated in 2 or 1", "sit if losing a --- piece", auto "go" when mate threat gone, fast mode under 1 s, ghost pieces for trades, Gnejs piece values |
| 11 | https://chronatog.com/?p=484 | ICS Interviews No. 013: Gnejs (2019) | Little on time; confirms top players analysed piece values; background on Sjeng's namesake |
| 12 | https://chessbio.com/term/bughouse | Chessbio: Bughouse | Delaying a move to starve the opponent is legal and common; 3+0, 5+0, 2+1 standard; "sit!" calls |
| 13 | https://caissly.pages.dev/variants/bughouse | Caissly: Bughouse | "tempo management (your moves take time that your partner doesn't get)"; 3+0/5+0 |
| 14 | https://jpneto.github.io/world_abstract_games/rognlie/bughouse.html | Richard's PBeM Server: Bughouse | Stop playing and hope partner wins; time limits configurable |
| 15 | https://rulespal.com/bughouse-chess/rulebook | Laws of Bughouse (OTB) mirror | 3 min each, FIDE blitz rules, simultaneous end = draw, captured piece usable only after clock press |
| 16 | https://www.greaterwrong.com/posts/imnfJ9Ris7GgjkZbT/the-bughouse-effect-1 | The Bughouse Effect (essay) | Anecdotes: 3 s slower per move hurts, 14 s on a rook capture lost the game on time, partop's stall starves an attack |
| 17 | https://www.albertochueca.com/?p=14575 | Chueca: Play Bughouse Chess With a Friend (c. 2021) | Sitting defined; "you can't do it if you are down in time" |
| 18 | https://www.chesshost.app/blog/bughouse-strategy/ | ChessHost: 12 Tips (2026) | Sitting works with more time; "Up"/"down" calls; checks burn the clock; house rules limiting sitting |
| 19 | https://lichessdude.com/bughouse-chess-rules-strategy-guide/ | LichessDude guide (2025) | Sit = don't move until opponent does, keeps pieces in partner's bank; complicate when opponent is low |
| 20 | https://www.pychess.org/variants/bughouse | PyChess: Bughouse | Partners may "ask their partner to stall"; points to JarlCarlander series and JannLee videos |
| 21 | https://support.chess.com/en/articles/8615092-what-is-bughouse | Chess.com support: What is Bughouse | Timeout on either board wins for the team |
| 22 | https://www.chess.com/blog/Sorsi/time-management-in-bughouse | Sorsi: Time Management in Bughouse [snippet] | Uptime; thinking under attack burns time; sit only for a piece that hurts; double sit ~10 s = resign; track partop's pace; never double sit |
| 23 | https://www.chess.com/blog/Sorsi/advanced-bughouse-tips | Sorsi: Advanced bughouse tips [snippet] | When to sit is the hardest skill; play fast to be uptime so you can sit; strong chess players under-sit, weak over-sit; sit for small gains once ahead; spend extra seconds at once; flow light but consistent; sit more vs stronger opponents |
| 24 | https://www.chess.com/blog/Sorsi/basic-bughouse-tips | Sorsi: Basic Bughouse Tips [snippet] | Partop's time is what matters; uptime definition; beginners keep moving, don't sit unless asked |
| 25 | https://www.chess.com/blog/Sorsi/basic-bughouse-tips-2 | Sorsi: Basic Bughouse Tips 2 [snippet] | "The longer you think, the worse your move becomes" |
| 26 | https://www.chess.com/blog/Sorsi/decision-making-in-bughouse | Sorsi: Decision Making in Bughouse [snippet] | Sitting for a piece to start an unclear attack kills your time |
| 27 | https://www.chess.com/blog/JarlCarlander/beginning-at-bughouse-on-chess-com | JarlCarlander: Beginning at Bughouse [snippet] | Glossary (sit, go); no increment so time is never recovered; a player with more time than partop "doesn't need to make a single move" |
| 28 | https://www.chess.com/blog/JarlCarlander/bughouse-for-beginners-coordination | JarlCarlander: Coordination and Strategy [snippet] | Hole plan needs more time than partop "or he will also refuse to move and you will flag first"; sit and defend in alternation; being down some time is not losing; ask partner how to make up time; anti-premoves; sac-stallers waste time |
| 29 | https://www.chess.com/blog/JarlCarlander/esoteric-wisdom-of-the-ancient-bughouse-masters | JarlCarlander: Esoteric Wisdom [snippet] | Sac-sitting has no agreed definition |
| 30 | https://www.chess.com/blog/JarlCarlander/the-instructiveness-of-bughouses-unique-aspect | JarlCarlander: Instructiveness of Bughouse's Unique Aspect [snippet] | "Right strategic page" matters more than being up time (attribution and context uncertain) |
| 31 | https://www.chess.com/blog/JarlCarlander/bughouse-for-beginners-rapid-improvement | JarlCarlander: Rapid Improvement [snippet] | Slowing down is worth it below 2100 (attribution uncertain) |
| 32 | https://www.chess.com/clubs/forum/view/bughouse-tips-anyone?page=1 | Chess.com club forum: Bughouse tips. Anyone? [snippet] | "look at partopps' clock and not your opponent's clock"; ahead of partop wins all sit/stall and time-trouble races |
| 33 | https://www.chess.com/blog/chriswsh/let-s-bughouse-playing-against-the-sicilian-part-2 | chriswsh: Let's Bughouse, Sicilian part 2 [snippet] | "Sit until 10, then low flow": partner sits to 10 s then flags; opponents began resigning on move 1 |
| 34 | https://www.chess.com/article/view/bughouse-chess-tips-for-beginners | Rensch: 7 Doubles (Bughouse) Tips [snippet] | Play like 1+0 unless sitting; learn stalling; double sit is one of the worst situations; "go faster than fast" when attacking; learn to beat sac-and-sit |
| 35 | https://www.chess.com/blog/key/the-basics-of-bughouse | key: The Basics of Bughouse [snippet] | Stalling when you have more time; time a greater asset than material |

Not reachable (403/401/refused), listed so nobody repeats the attempt without a browser: uschess.org Kazim
Gulamali interview; chessvariants.org comments (id=228) and chessvariants.com bughouse pages; the Wikibooks
mirror; the lichess "house variants" forum page; archive.org and reader-proxy copies of chess.com posts.

Repo files used as evidence for stock behaviour: `third_party/hivemind/README.md` ("Team and Time Advantage",
architecture), `third_party/hivemind/engine/src/environment/joint_action.h`, `.../search/agent.cc` (mate probe gate),
`.../search/searchthread.cc` (flag and sit-off terminals, FreePass pass reply), `.../environment/planes.cc` (plane
31), `arena/players/hivemind.py` (eval swing comment, sit cap, free mode), `arena/players/heuristic.py`
(`team_time_edge`), `docs/custom_engine_spec.md`, `docs/custom_engine_spec_v2.md` (sections A-D), `docs/beat_stock.md`.
