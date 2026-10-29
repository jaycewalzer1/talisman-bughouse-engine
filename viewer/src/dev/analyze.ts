// Pure summaries of one game's event stream for the dev panel. Works on a finished game or one still
// in progress (live), so every field tolerates a missing game_end.

import { type GameEndEvent, type GameEvent, type GameStartEvent, type Seat, SEATS, TEAM_OF } from '../events/types'

export interface SeatStats {
  seat: Seat
  name: string
  kind: string
  moves: number
  drops: number
  captures: number
  searches: number
  nodes: number
  avgThinkMs: number | null
  borrowed: number
  sits: number
  sitMs: number
  messages: number
  clockMs: number | null
}

export interface EvalPoint { ts: number; seat: Seat; t1: number }   // eval from team T1's side, [-1, 1]

export interface GameSummary {
  start: GameStartEvent | null
  end: GameEndEvent | null
  lastTs: number
  plies: number
  seats: SeatStats[]
  evals: EvalPoint[]
  messages: number
  transfers: number
}

export function parseJsonl(text: string): GameEvent[] {
  const out: GameEvent[] = []
  for (const line of text.split('\n')) {
    const t = line.trim()
    if (!t) continue
    try { out.push(JSON.parse(t) as GameEvent) } catch { /* skip a torn line */ }
  }
  return out
}

export function summarize(events: GameEvent[]): GameSummary {
  const start = (events.find(e => e.type === 'game_start') as GameStartEvent | undefined) ?? null
  const end = (events.find(e => e.type === 'game_end') as GameEndEvent | undefined) ?? null
  const lastTs = events.length ? events[events.length - 1].ts : 0
  const stats = new Map<Seat, SeatStats>(SEATS.map(seat => [seat, {
    seat, name: start?.seats[seat]?.name ?? seat, kind: start?.seats[seat]?.kind ?? '?',
    moves: 0, drops: 0, captures: 0, searches: 0, nodes: 0, avgThinkMs: null, borrowed: 0,
    sits: 0, sitMs: 0, messages: 0, clockMs: start?.clocks[seat] ?? null,
  }]))
  const thinkSum = new Map<Seat, number>()
  const sittingSince = new Map<Seat, number>()
  const evals: EvalPoint[] = []
  let plies = 0, messages = 0, transfers = 0

  for (const e of events) {
    switch (e.type) {
      case 'move': {
        const s = stats.get(e.seat)!
        s.moves++; plies++
        if (e.isDrop) s.drops++
        if (e.capture) s.captures++
        for (const seat of SEATS) stats.get(seat)!.clockMs = e.clocks[seat]
        break
      }
      case 'thinking': {
        const s = stats.get(e.seat)!
        if (e.phase === 'start') {
          if ((e.nodes ?? 0) > 0) { s.searches++; s.nodes += e.nodes ?? 0 }
          if (e.think_ms != null) thinkSum.set(e.seat, (thinkSum.get(e.seat) ?? 0) + e.think_ms)
          if (e.borrowed) s.borrowed++
        } else if (e.eval != null) {
          evals.push({ ts: e.ts, seat: e.seat, t1: TEAM_OF[e.seat] === 'T1' ? e.eval : -e.eval })
        }
        break
      }
      case 'stall': {
        const s = stats.get(e.seat)!
        if (e.active) { s.sits++; sittingSince.set(e.seat, e.ts) }
        else if (sittingSince.has(e.seat)) { s.sitMs += e.ts - sittingSince.get(e.seat)!; sittingSince.delete(e.seat) }
        break
      }
      case 'message': stats.get(e.fromSeat)!.messages++; messages++; break
      case 'pass': transfers++; break
    }
  }
  for (const [seat, since] of sittingSince) stats.get(seat)!.sitMs += lastTs - since
  for (const s of stats.values()) {
    const starts = events.filter(e => e.type === 'thinking' && e.phase === 'start' && e.seat === s.seat).length
    s.avgThinkMs = starts ? Math.round((thinkSum.get(s.seat) ?? 0) / starts) : null
  }
  return { start, end, lastTs, plies, seats: SEATS.map(s => stats.get(s)!), evals, messages, transfers }
}

/** One readable line per event for the log table. */
export function describe(e: GameEvent): { seat: string; text: string } {
  switch (e.type) {
    case 'game_start': return { seat: '', text: `${e.game_id} · ${e.setup} · ${e.base_ms / 1000}s` }
    case 'thinking': return {
      seat: e.seat,
      text: e.phase === 'start'
        ? `start ${e.think_ms ?? '?'} ms, ${e.nodes ?? 0} nodes${e.borrowed ? ' (borrowed)' : ''}`
        : `end ${e.think_ms ?? '?'} ms${e.eval != null ? `, eval ${e.eval >= 0 ? '+' : ''}${e.eval.toFixed(2)}` : ''}`,
    }
    case 'move': return { seat: e.seat, text: `${e.board}: ${e.san ?? e.uci}${e.capture ? ` ×${e.capture}` : ''}` }
    case 'pass': return { seat: e.fromSeat, text: `${e.piece} → ${e.toSeat}` }
    case 'message': return { seat: e.fromSeat, text: `${e.token}${e.text ? ` "${e.text}"` : ''} → ${e.toSeat}` }
    case 'stall': return { seat: e.seat, text: e.active ? `sits (${e.reason ?? '?'})` : 'stops sitting' }
    case 'game_end': return { seat: e.loserSeat ?? '', text: `${e.result === 'draw' ? 'draw' : `${e.result} wins`} by ${e.reason}${e.board ? ` on ${e.board}` : ''}` }
  }
}
