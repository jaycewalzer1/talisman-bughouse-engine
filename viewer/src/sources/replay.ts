// Replay source: plays a recorded `.jsonl` file (one event per line), e.g. runs/<name>/game_1.jsonl
// or public/replays/sample.jsonl.
//
// Timeline: event `ts` restarts at 0 for every game, so if a file holds several games we lay
// them end to end on one "global" timeline (with a 3 s gap) for the scrub bar.
// Seeking = reset the store + ingestMany(every event up to that point). Simple and always correct.

import type { GameEvent } from '../events/types'
import { useGame } from '../store/gameStore'
import type { Source } from './source'

const GAP_BETWEEN_GAMES_MS = 3000

export function parseJsonl(text: string): GameEvent[] {
  const out: GameEvent[] = []
  for (const line of text.split('\n')) {
    const t = line.trim()
    if (!t) continue
    try { out.push(JSON.parse(t)) } catch { console.warn('[replay] skipping bad line', t.slice(0, 120)) }
  }
  return out
}

export interface ReplaySource extends Source {
  duration: number                     // global ms
  position: () => number               // global ms
  seek: (globalMs: number) => void
  ended: () => boolean
}

export function createReplaySource(events: GameEvent[]): ReplaySource {
  // global time of each event + the offset of the game it belongs to
  const gts: number[] = []
  const offsets: number[] = []
  let offset = 0
  let lastG = 0
  events.forEach((e, i) => {
    if (e.type === 'game_start' && i > 0) offset = lastG + GAP_BETWEEN_GAMES_MS
    const g = offset + e.ts
    gts.push(g)
    offsets.push(offset)
    lastG = Math.max(lastG, g)
  })
  const duration = lastG

  let cursor = 0      // index of the next event to ingest
  let pos = 0         // current global time

  /** Store playhead = global time minus the offset of the game we're in. */
  function syncPlayhead() {
    const store = useGame.getState()
    const off = cursor > 0 ? offsets[cursor - 1] : 0
    const endTs = store.end ? store.end.ts : Infinity
    store.setPlayhead(Math.max(0, Math.min(pos - off, endTs)))
  }

  function seek(globalMs: number) {
    pos = Math.max(0, Math.min(duration, globalMs))
    let n = 0
    while (n < events.length && gts[n] <= pos) n++
    const store = useGame.getState()
    store.reset()
    store.ingestMany(events.slice(0, n))
    cursor = n
    syncPlayhead()
  }

  return {
    duration,
    position: () => pos,
    ended: () => pos >= duration,
    seek,
    start() { seek(0) },
    stop() { /* nothing to clean up */ },
    tick(dtMs) {
      if (dtMs <= 0 || pos >= duration) return
      pos = Math.min(duration, pos + dtMs)
      const batch: GameEvent[] = []
      while (cursor < events.length && gts[cursor] <= pos) batch.push(events[cursor++])
      if (batch.length) useGame.getState().ingestMany(batch)
      syncPlayhead()
    },
  }
}
