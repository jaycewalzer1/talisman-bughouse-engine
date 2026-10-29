# Bughouse Arena viewer

A web app for watching AI agents play bughouse: two boards, four seats, two teams that talk
to each other. There are two ways to look at a game, a **3D** street scene (night-time Brooklyn)
and a **2D** analytical table. Both show the same game.

## Run it

```bash
cd viewer
npm install      # first time only
npm run dev      # then open the URL it prints (usually http://localhost:5173)
```

By default you get the **3D view** playing a **mock** game, which is generated in the browser so
you don't need the engine. Use the sticker buttons at the top-left to switch:

- **2D | 3D**: which view to show.
- **Mock**: fake but plausible games, generated on the fly. A new one starts after each game ends.
- **Live ws**: the real arena over a WebSocket (see below).
- **Replay**: a recorded `.jsonl` game. The bundled `public/replays/sample.jsonl` loads
  automatically. Use 📂 to open your own file, for example `runs/<name>/game_1.jsonl`. You get
  play/pause, a scrub bar and speeds from 0.25x to 8x.

You can also set these from the URL: `?view=2d`, `?source=live`, `?source=replay&replay=/replays/sample.jsonl`,
`?ws=ws://otherhost:8765`.

## Watch the live arena

From the **repo root** (not `viewer/`), start the arena's event server:

```bash
python -m arena.serve        # serves events on ws://localhost:8765
```

Then choose **Live ws** in the viewer, or open `http://localhost:5173/?source=live`. If the
server isn't up yet, the viewer keeps retrying (1s, 2s, 4s, ... up to every 15s), so you can
start the two in either order.

## How the code fits together

```
 sources  ──events──▶  store  ──state──▶  views
 (mock / live / replay)   (zustand)        (2D / 3D)
```

Everything is driven by one **event stream**. Its schema is documented in
[`../docs/events.md`](../docs/events.md) and typed in `src/events/types.ts`. Events include
`game_start`, `thinking`, `move`, `pass`, `message`, `stall` and `game_end`.

| Folder / file | What it does |
|---|---|
| `src/events/types.ts` | TypeScript types for every event, plus small FEN helpers (`fenGrid`, `fenPocket`, ...). |
| `src/store/gameStore.ts` | The single **game store** ([zustand](https://zustand.docs.pmnd.rs/)). Sources push events in with `reset()` / `ingest()` / `ingestMany()` / `setPlayhead()`. It keeps the derived state (board FENs, clocks, who is thinking, stalls, messages, result). |
| `src/sources/mock.ts` | Mock games. `generateMockGame()` simulates a whole game up front (chess.js for legal moves, plus drops, piece passing, clocks, thinking, messages and stalls). `createMockSource()` plays it back in real time. |
| `src/sources/websocket.ts` | Live source: connects to the arena, reconnects with backoff, and moves the clocks forward in real time between events. |
| `src/sources/replay.ts` | Replay source: parses a `.jsonl` file and supports play, pause, seek and speed. |
| `src/sources/controller.ts` | Owns the **one** active source and runs the `requestAnimationFrame` loop that moves the playhead every frame, which keeps the clocks smooth. The UI uses its `usePlayback` store. |
| `src/views/two/` | The 2D view: boards, seat panels (clock, thinking, stall, pocket), the comms lanes and log, capture flights, and the result overlay. |
| `src/views/three/` | The 3D view (react-three-fiber). |
| `src/App.tsx` | The thin shell: view toggle, source picker and replay controls. The 3D view is lazy-loaded. |

### React beginner notes

- **Components** are functions that return JSX (HTML-like markup), for example `Board` in
  `src/views/two/Board.tsx`.
- **Reading the store**: `const fen = useGame(s => s.fens.A)`. The component re-renders only
  when that selected value changes. Select single values (numbers, strings, existing objects).
  Don't return a brand-new object from a selector.
- **Writing to the store** happens only in `src/sources/`. Views never change game state; they
  only read it.
- **Time**: event `ts` is game time in ms. `playheadTs` is "now" on that timeline.
  `displayClock()` subtracts the time that has passed since the last move, so a running clock ticks
  down between events.

### Bughouse in 30 seconds

There are two boards (A and B) and four players. Team **T1** is A_white + B_black and team
**T2** is A_black + B_white, so partners play opposite colours on different boards. When you
capture a piece, it goes to your partner's **pocket**, and they can later **drop** it on an
empty square as their move. Partners talk (`need_N`, `have_Q`, `sit`, `go`, `stall`, `ok`,
`no`), and a player may **sit** (stall) while waiting for a piece. Their clock keeps running
while they wait. Checkmate or a flag on either board ends the game for everyone.

In the 2D view, board B is drawn flipped (black at the bottom), as at a real bughouse table, so
each team's two players sit on the same side. Their messages travel straight across the lane
between them.

## Mock tools (optional)

`scripts/mock-tool.ts` runs the mock generator in Node, outside the browser:

```bash
npx esbuild scripts/mock-tool.ts --bundle --platform=node --format=esm --outfile=node_modules/.tmp/mock-tool.mjs
node node_modules/.tmp/mock-tool.mjs check 20     # simulate 20 games and assert the event invariants
node node_modules/.tmp/mock-tool.mjs sample 3     # regenerate public/replays/sample.jsonl (seed 3)
```

## Other commands

```bash
npm run build    # type-check (tsc -b) + production build into dist/
npm run lint     # eslint
```
