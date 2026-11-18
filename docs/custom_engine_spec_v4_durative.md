# Custom engine spec v4: the durative tree (audit WP-E, part E2)

Draft, 2026-11-17. Wave 2. Builds on spec v3 and on the engine notes in `docs/custom_engine_spec_v3_engine.md`
("Notes for E2"), which list the hooks in our engine. Waits on the starting-network decision (audit D1): the design
below is the same for either network, but **where** it is written is not. On widened it04 it goes into our fork of
Hivemind v2.2.2 (`third_party/hivemind`); on upstream twin-s-clock it goes into upstream v3's engine or into ours
taught to run TwinBoardNet. Do not start the C++ until D1 is decided.

## The model of time

The tree stops alternating whole team turns. Each seat carries a **ready time**: when its next action lands.

- Ours: this search's budget (`TreeMoveMs`). Theirs: the `pace` suffix from the arena (EMA of observed landings),
  default ours.
- A node advances simulated time to the earliest ready time; only the team owning that seat acts there (one board, or
  both if both of its seats are ready at once). Exact ties alternate, starting with the opponent of the root team.
- Every clock on turn runs by the time advanced; elapsed planes and the node's time advantage use the clocks at the
  node's current time.
- A pass on a board lasts until the next event on the other board (the earliest other ready time), or until that
  seat's flag, capped at `k x TreeMoveMs` (k = 8 to start).
- A captured piece reaches the partner's hand at the capture's landing time and is droppable from the partner's next
  decision after that time, not at once (ablation flag `InstantTransfer` keeps the old behaviour).
- Simultaneous mate races are decided by landing time: the earlier action wins; equal times: the opponent of the
  side that captured last wins, or draw (pick one, document it, test both teams).
- Sit-offs: when only sits remain, the seat on turn with less clock must move (it no longer loses outright); D1 stays
  as the fallback when `pace` is absent.

## Acceptance tests (from the audit, WP-E E2)

1. Clocks 100 and 600 ms with `TreeMoveMs` 1000: a single loser.
2. "Sit for the knight": the partner wins a knight in 2 moves on B; with a 10 s diagonal lead the root plays the pass
   on A at depth <= 2, with a 10 s deficit it does not.
3. Symmetric sit-off is a draw at equal clocks and flips sign when one diagonal clock moves by 1 s
   (`test_custom_engine.cc:762-848` keep passing).
4. A seat with 3 s makes at most `3000 / min_move_ms` moves in any line.
5. Mirror test: a both-sides-mate-in-one position evaluated from each team's root gives `V(A) ~ -V(B)`.
6. A perpetual on A with mate in 2 on B evaluates near +1 and is never a solver draw.
7. `TreeMoveMs 0` and no `pace` reproduce v3 behaviour bit for bit; the stock build is unchanged.
8. Regression sets (`tools/regression.py`): the sit-off family reaches at least the raw net's 18/20 "sit when
   ahead" with 0/20 "sit when behind" at 200 nodes.

## Ablation flags (off = v3 behaviour)

`DurativeTree`, `InstantTransfer`, `MateRaceByLanding`, each a UCI option so one self-play round can switch one off.
