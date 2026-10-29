// One store for both views. Event sources (mock, WebSocket, replay file) call ingest()/reset();
// views read the derived state. Views that animate individual events (piece slides, tosses,
// graffiti) should subscribe to `events` and react to new entries past the last seq they handled.

import { create } from 'zustand'
import {
  type BoardId, type Clocks, type GameEndEvent, type GameEvent, type GameStartEvent, type MessageEvent,
  type MoveEvent, type Seat, type ThinkingEvent, seatToMove,
} from '../events/types'

const START_FEN = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR[] w KQkq - 0 1'
const ZERO: Clocks = { A_white: 0, A_black: 0, B_white: 0, B_black: 0 }

export interface GameState {
  events: GameEvent[]          // everything ingested this game, in seq order
  start: GameStartEvent | null
  fens: Record<BoardId, string>
  clocks: Clocks               // as of clocksTs
  clocksTs: number
  playheadTs: number           // current simulated time; sources advance it for smooth clocks
  lastMove: Record<BoardId, MoveEvent | null>
  lastEventBoard: BoardId | null
  thinking: Partial<Record<Seat, ThinkingEvent>>   // seats currently thinking
  stalls: Partial<Record<Seat, string>>            // seats currently stalling -> reason
  messages: MessageEvent[]
  end: GameEndEvent | null

  reset: () => void
  ingest: (e: GameEvent) => void
  ingestMany: (es: GameEvent[]) => void
  setPlayhead: (ts: number) => void
}

const initial = () => ({
  events: [] as GameEvent[],
  start: null,
  fens: { A: START_FEN, B: START_FEN },
  clocks: { ...ZERO },
  clocksTs: 0,
  playheadTs: 0,
  lastMove: { A: null, B: null },
  lastEventBoard: null,
  thinking: {},
  stalls: {},
  messages: [] as MessageEvent[],
  end: null,
})

function apply(s: GameState, e: GameEvent): Partial<GameState> {
  const events = [...s.events, e]
  const playheadTs = Math.max(s.playheadTs, e.ts)
  switch (e.type) {
    case 'game_start':
      return { ...initial(), events: [e], start: e, fens: { ...e.fens }, clocks: { ...e.clocks }, clocksTs: e.ts }
    case 'move': {
      const thinking = { ...s.thinking }; delete thinking[e.seat]
      const stalls = { ...s.stalls }; delete stalls[e.seat]
      return {
        events, playheadTs, fens: { ...s.fens, [e.board]: e.fen }, clocks: { ...e.clocks }, clocksTs: e.ts,
        lastMove: { ...s.lastMove, [e.board]: e }, lastEventBoard: e.board, thinking, stalls,
      }
    }
    case 'thinking': {
      const thinking = { ...s.thinking }
      if (e.phase === 'start') thinking[e.seat] = e; else delete thinking[e.seat]
      return { events, playheadTs, thinking }
    }
    case 'stall': {
      const stalls = { ...s.stalls }
      if (e.active) stalls[e.seat] = e.reason ?? 'stall'; else delete stalls[e.seat]
      return { events, playheadTs, stalls }
    }
    case 'message':
      return { events, playheadTs, messages: [...s.messages, e] }
    case 'pass':
      return { events, playheadTs, lastEventBoard: e.fromBoard }
    case 'game_end':
      return { events, playheadTs, end: e, thinking: {}, stalls: {} }
  }
}

export const useGame = create<GameState>()((set) => ({
  ...initial(),
  reset: () => set(initial()),
  ingest: (e) => set((s) => apply(s, e)),
  ingestMany: (es) => set((s) => {
    let cur: GameState = s
    for (const e of es) cur = { ...cur, ...apply(cur, e) }
    return cur
  }),
  setPlayhead: (ts) => set({ playheadTs: ts }),
}))

/** Live remaining time for a seat at the playhead: only the side to move on each board is running. */
export function displayClock(s: Pick<GameState, 'clocks' | 'clocksTs' | 'playheadTs' | 'fens' | 'end' | 'start'>, seat: Seat): number {
  const base = s.clocks[seat]
  if (!s.start) return base
  if (s.end?.reason === 'flag' && s.end.loserSeat === seat) return 0
  const board = seat[0] as BoardId
  if (seatToMove(board, s.fens[board]) !== seat) return base
  // After game_end the clocks stop at the end time, not at the last move.
  const now = s.end ? Math.min(s.playheadTs, s.end.ts) : s.playheadTs
  return Math.max(0, base - (now - s.clocksTs))
}
