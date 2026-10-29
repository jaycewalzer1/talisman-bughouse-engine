# Arena

A bughouse match harness: four players (two teams, two boards) play on a shared simulated clock,
optionally talk to their partner, and every game is recorded as a stream of JSON events
(`docs/events.md`). The same stream is written to disk for analysis and broadcast to the viewers.

## How the pieces fit

```
                 interfaces.py  (Seat/Team constants, Observation, Decision, Message, GameResult)
                        |
   state.py ------> sim.py <------ players/
   BughouseState    Simulator      make_players(setup, kind) -> {seat: Player}
   (2 boards,       (event-driven  setups: centralized | silent | sharing
    pockets,         timeline,     kinds:  random | heuristic | hivemind
    4 clocks,        message
    rules)           channel,
                     emits events)
                        |
          +-------------+--------------+
          |                            |
       run.py  (batch)              serve.py  (live WebSocket)
       openings.py (paired openings)   plays games back to back,
       stats.py (summaries + CIs)      paces events in real time
          |                            |
   runs/<name>/game_XXXX.jsonl      viewer/ (ws://localhost:8765)
   runs/<name>/summary.json
```

- `state.py` knows the rules: legal moves incl. drops, captures going to the partner's pocket, clocks,
  checkmate/flag.
- `sim.py` asks players for `Decision`s given an `Observation`, charges think time on the clock, lands
  moves at the right simulated time, delivers team messages after `channel_delay_ms`, and emits events.
- `players/` builds the four players for a team setup: `centralized` (one brain plays both boards of a
  team), `silent` (partners never talk), `sharing` (partners exchange need/have/sit/go messages).
- `openings.py` has 24 opening pairs (one standard opening per board). `paired_schedule` plays each
  pair twice, swapping which team (T1 or T2) the contender controls, so opening luck cancels out.
- `stats.py` turns an event list into a per-game summary (`summarize_game`) and reports contender vs
  baseline metrics with 95% CIs from a bootstrap that resamples whole opening pairs (games sharing
  a pair are not independent).

In a match, the **contender** setup plays one team and the **baseline** setup the other; teams swap
every second game.

## Commands

Run from the repo root with the project env.

```bash
# tests
.venv/bin/python -m pytest arena/tests -q

# a batch: 20 games, sharing (contender) vs silent (baseline), 4 processes
.venv/bin/python -m arena.run --contender sharing --baseline silent --kind heuristic --games 20 \
    --base-ms 60000 --channel-delay-ms 0 --nodes 200 --seed 0 --workers 4 --out runs/sharing_vs_silent
```

The batch writes `runs/<name>/game_0001.jsonl` ... (one event per line), `runs/<name>/summary.json`
(config, per-game summaries, metrics with CIs) and prints a table such as:

```
contender score             0.550  [0.400, 0.700]
flag rate (any side)        0.150  [0.050, 0.300]
latency ms/decision   ...
```

Score is from the contender's side (win 1, draw 0.5, loss 0); 0.5 means no difference. A game that
crashes is logged with its traceback and left out of the statistics.

```bash
# live server for the viewers (ws://localhost:8765), games back to back, 5 s pause between games
.venv/bin/python -m arena.serve --port 8765 --kind heuristic --contender sharing --baseline silent \
    --speed 1.0 --base-ms 60000

# replay a recorded game instead (loops; add --once to play it once); --speed 4 = 4x real time
.venv/bin/python -m arena.serve --port 8765 --replay runs/sharing_vs_silent/game_0001.jsonl --speed 4

# then start a viewer
cd viewer && npm run dev   # http://localhost:5173
```

The server sends each event as one JSON text frame. A viewer that connects mid-game first gets every
event of the current game so far, then live events. A new game always starts with a `game_start` event,
so a viewer should reset its boards when it sees one.
