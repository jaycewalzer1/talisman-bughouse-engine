# Phase 1 scheduler: agreed design

Agreed 2026-11-02, revised 2026-11-09. Hivemind stays frozen and picks the moves. A small new policy decides
around it, trained with REINFORCE on team wins. The policy is `arena/players/scheduler.py`, the arena player
`arena/players/scheduler_player.py` and the training loop `arena/train_scheduler.py`. Tests:
`arena/tests_phase1/`.

## One decision, in order

| Step | Decision | Options | Sees |
|---|---|---|---|
| 1. Quick look | none, a small search | | Boards, clocks |
| 2. Gate | Move or wait | Move, wait for P/N/B/R/Q, or sit | Quick look, clocks, partner state |
| 3. Talk | What to say | Quiet, need P/N/B/R/Q, offer | The same inputs, plus the gate choice (one-hot) |
| 4. Budget | How long to think | x0.25, x0.5, x1, x2, x4 | The same inputs as the gate |
| 5. Long search | none | | |
| 6. Play | none, plays the search's move | | |

The chance of the whole action is the three chances multiplied, so their log-probabilities add.

- **Equal compute:** the quick look comes out of the same node budget as the long search.
- **Budget sees no gate choice:** the budget is only used when the gate says move, so the gate choice would tell
  it nothing.
- **Waiting:** there's no set length. The simulator asks again every 5 s and after any move or delivered
  message, and the policy's end-wait choice decides whether to stop. Keeping on waiting costs no search, and
  stopping goes back through the gate. A piece arriving ends the wait. End-wait choices are trained: each one's
  log-probability joins the game's list. The untrained policy stops, which is closest to how stock re-decides.
- **Requests:** choosing to wait for a piece also sends `need_X`. Sit is the same wait with no request, for when
  stock sits with no piece in mind.
- **Stale messages:** offers and requests expire after 15 s. A seat sends at most once per simulated instant, so
  two waiting partners can't wake each other forever.
- **Offers:** `have_X` with payload `{"piece": "X", "plies": n}`, read from the search's plan by
  `pv_offers`. Plies aren't converted to seconds; the policy sees the clocks and learns what they're worth.
- **Starting point:** the untrained policy behaves like stock: a fixed starting point plus a learned
  correction whose last layer starts at zero.
- **Exploration:** only when a wait is plausible (an offer, or the plan wants a piece). It starts at 5% and
  falls over training. REINFORCE uses the true sampling probabilities, including forced waits.

## Left for Phase 2

Offers can't change which move the partner plays. Making `need_X` change moves means new zero-initialized
network inputs and fine-tuning, so that waits until Phase 1 shows whether it's needed.

## First feature to try after the baseline

The cost of moving now: the eval with sitting allowed minus the eval with a move forced (`RequireMoveOn`). It would
say whether a piece we hold nearly does the job of the one we want. It needs a second search per measurement, so it
waits until the 16 features have a measured baseline to compare against.

## Training

```bash
.venv/bin/python -m arena.train_scheduler --iterations 200 --pairs 10 --workers 3 --out runs/phase1
```

Add `--resume` to continue a run. Results are in `<out>/train_log.jsonl`, the current policy in `<out>/latest.pt`.
Players are kind `agent`, setup `sharing`.

## Feature groups (added after the first 40-iteration run)

That run showed no learning: waits are rare, one win or loss is a thin signal, and the policy had little to tell a
good wait from a bad one. Four optional groups address that. All are off by default and turned on with
`--features`; every extra engine search they need comes out of the move's node budget, so total nodes stay equal to
stock's. Their inputs sit after the 32 base inputs and are 0 while a group is off, so every run has the same network
shape and runs can be combined later.

| Group | Adds | Extra searches |
|---|---|---|
| `tempo` | Legal move count, in check, gap between the engine's first and second choice in the quick look | None |
| `mate` | Per piece: eval gain from holding one more, forced mate or not, and whether the partner can capture one | Up to 2, only for pieces that could be dropped with check and could arrive |
| `danger` | Per piece: eval lost if our opponent got one more. Above 0.25 it sends the partner `hold_X` | Up to 1, same filter |
| `serve` | A new decision: when the partner asked for a piece and a capture of it is within 0.4 eval of the engine's choice, play it or not | None (the long search reports 4 lines) |

Training options that go with them:

- `--baseline eval` credits each decision with the result minus what the engine expected at that moment (a straight
  line fitted to the batch), instead of minus the running average.
- `--think-logit 2` lets an untrained policy try other think budgets about a third of the time. The gate keeps its
  strong stock starting point, because a random wait costs far more than a random think budget.
- `--exploration 0.15` raises the starting chance of a forced wait.

`cluster/launch_waves.sh` submits five runs side by side: `control`, `signal` (training options only) and one run
per wave. `tools/compare_runs.py runs/waves/*` prints them in one table.

Not built: teaching the gate from the wait probe's results (needs a probe run first), acting on `hold_X`, and a
search that concentrates on one board (needs an engine change).
