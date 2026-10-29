// Live source: listens to the arena's WebSocket server (`python -m arena.serve`, port 8765).
// Every text frame is one JSON event (we also accept a JSON array or several newline-separated
// events per frame, to be forgiving). game_start => reset the store; everything else => ingest.
//
// Clocks: events only carry clock values at their own `ts`. Between events we push the playhead
// forward in real time from the last event, so clocks visibly tick.

import type { GameEvent } from '../events/types'
import { useGame } from '../store/gameStore'
import type { Source } from './source'

export const DEFAULT_WS_URL = 'ws://localhost:8765'
const MAX_EXTRAPOLATE_MS = 60_000   // don't run clocks forever if the arena goes quiet

export function createLiveSource(url: string, onStatus: (s: string) => void): Source {
  let ws: WebSocket | null = null
  let stopped = false
  let attempt = 0
  let retryTimer: ReturnType<typeof setTimeout> | undefined
  // anchor for real-time playhead: last event's ts and the wall-clock time we received it
  let anchorTs = 0
  let anchorWall = 0
  let haveAnchor = false

  function handle(e: GameEvent) {
    const store = useGame.getState()
    if (e.type === 'game_start') store.reset()
    store.ingest(e)
    anchorTs = e.ts
    anchorWall = performance.now()
    haveAnchor = true
  }

  function parseFrame(text: string): GameEvent[] {
    const out: GameEvent[] = []
    for (const line of text.split('\n')) {
      const t = line.trim()
      if (!t) continue
      try {
        const v = JSON.parse(t)
        if (Array.isArray(v)) out.push(...v)
        else out.push(v)
      } catch {
        console.warn('[live] could not parse frame', t.slice(0, 200))
      }
    }
    return out
  }

  function connect() {
    if (stopped) return
    onStatus(attempt ? `reconnecting to ${url} ...` : `connecting to ${url} ...`)
    try {
      ws = new WebSocket(url)
    } catch {
      scheduleReconnect()
      return
    }
    ws.onopen = () => { attempt = 0; onStatus(`live: ${url}`) }
    ws.onmessage = (msg) => {
      if (typeof msg.data !== 'string') return
      for (const e of parseFrame(msg.data)) if (e && typeof e === 'object' && 'type' in e) handle(e)
    }
    ws.onclose = () => { ws = null; scheduleReconnect() }
    ws.onerror = () => { /* onclose follows and handles the retry */ }
  }

  function scheduleReconnect() {
    if (stopped) return
    // exponential backoff: 1s, 2s, 4s, 8s, then every 15s
    const delay = Math.min(15_000, 1000 * 2 ** attempt)
    attempt++
    onStatus(`no arena at ${url} — retry in ${Math.round(delay / 1000)}s`)
    retryTimer = setTimeout(connect, delay)
  }

  return {
    start() { connect() },
    stop() {
      stopped = true
      clearTimeout(retryTimer)
      if (ws) { ws.onclose = null; ws.close(); ws = null }
    },
    tick(_dt, wallNow) {
      if (!haveAnchor) return
      const s = useGame.getState()
      if (!s.start || s.end) return
      s.setPlayhead(anchorTs + Math.min(MAX_EXTRAPOLATE_MS, Math.max(0, wallNow - anchorWall)))
    },
  }
}
