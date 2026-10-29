// Small helpers shared by the 2D view components.

import { type BoardId, type MessageEvent, type Seat, type TeamId, TEAM_OF, seatToMove } from '../../events/types'
import { type GameState, displayClock } from '../../store/gameStore'

// Same team colours as the 3D scene (src/views/three/layout.ts). Also defined as CSS vars --t1/--t2.
export const TEAM_COLOR: Record<TeamId, string> = { T1: '#ff8a1f', T2: '#27d8e6' }
export const seatColor = (s: Seat) => TEAM_COLOR[TEAM_OF[s]]

// Filled glyphs for both colours (we colour them with CSS). ︎ asks for the text (non-emoji) form.
const GLYPH: Record<string, string> = { k: '♚', q: '♛', r: '♜', b: '♝', n: '♞', p: '♟' }
export const glyph = (piece: string) => GLYPH[piece.toLowerCase()] + '︎'
export const isWhitePiece = (piece: string) => piece === piece.toUpperCase()

export const PIECE_NAME: Record<string, string> = { p: 'pawn', n: 'knight', b: 'bishop', r: 'rook', q: 'queen', k: 'king' }

/** Pretty seat label: "A ♔" etc. */
export const seatShort = (s: Seat) => `${s[0]}·${s.endsWith('white') ? 'W' : 'B'}`

/**
 * Board B is drawn flipped (black at the bottom), like a real bughouse table, so partners sit
 * on the same side: T1 (A_white + B_black) at the bottom, T2 (A_black + B_white) at the top.
 */
export const FLIPPED: Record<BoardId, boolean> = { A: false, B: true }
export const TOP_SEAT: Record<BoardId, Seat> = { A: 'A_black', B: 'B_white' }
export const BOTTOM_SEAT: Record<BoardId, Seat> = { A: 'A_white', B: 'B_black' }

/** Screen column/row (0..7, row 0 = top) of a square like "e4". */
export function squareXY(sq: string, flipped: boolean): { x: number; y: number } {
  const f = sq.charCodeAt(0) - 97
  const r = +sq[1] - 1
  return flipped ? { x: 7 - f, y: r } : { x: f, y: 7 - r }
}

/** Square name at screen column/row. */
export function xySquare(x: number, y: number, flipped: boolean): string {
  const f = flipped ? 7 - x : x
  const r = flipped ? y : 7 - y
  return 'abcdefgh'[f] + (r + 1)
}

/**
 * Clock to show for a seat. Like displayClock(), but after game_end it freezes the clocks at the
 * end time (and shows 0 for a flagged seat) instead of at the last move.
 */
export function viewClock(s: GameState, seat: Seat): number {
  if (!s.end) return displayClock(s, seat)
  if (s.end.reason === 'flag' && s.end.loserSeat === seat) return 0
  const board = seat[0] as BoardId
  if (s.start && seatToMove(board, s.fens[board]) === seat) return Math.max(0, s.clocks[seat] - (s.end.ts - s.clocksTs))
  return s.clocks[seat]
}

export function formatClock(ms: number): string {
  ms = Math.max(0, ms)
  if (ms < 10_000) return (Math.floor(ms / 100) / 10).toFixed(1)
  const total = Math.ceil(ms / 1000)
  return `${Math.floor(total / 60)}:${String(total % 60).padStart(2, '0')}`
}

export function formatTs(ms: number): string {
  const t = Math.max(0, ms) / 1000
  return `${Math.floor(t / 60)}:${(t % 60).toFixed(1).padStart(4, '0')}`
}

/** Sticker colour class for a message token. */
export function tokenKind(token: string): string {
  if (token.startsWith('need_')) return 'need'
  if (token.startsWith('have_')) return 'have'
  if (token === 'stall' || token === 'sit') return 'sit'
  if (token === 'go') return 'go'
  if (token === 'ok') return 'ok'
  if (token === 'no') return 'no'
  return 'other'
}

/** "need_N" -> "NEED ♞" style label. */
export function tokenLabel(token: string): string {
  const m = token.match(/^(need|have)_([PNBRQ])$/)
  if (m) return `${m[1]} ${glyph(m[2])}`
  return token
}

/**
 * "Sharing" setups send a status ping almost every decision: token "ok" with payload.kind="status"
 * (payload: best, eval, needs, pv). Those are shown low-key so real requests stand out.
 */
export function isStatusMsg(m: MessageEvent): boolean {
  return m.token === 'ok' && (m.payload as { kind?: unknown } | undefined)?.kind === 'status'
}

/** One-line summary of a status ping's payload. */
export function statusSummary(m: MessageEvent): string {
  const p = (m.payload ?? {}) as { best?: unknown; eval?: unknown; needs?: unknown }
  const parts: string[] = []
  if (p.best) parts.push(`best ${String(p.best)}`)
  if (typeof p.eval === 'number') parts.push(`eval ${p.eval >= 0 ? '+' : ''}${p.eval.toFixed(2)}`)
  if (Array.isArray(p.needs) ? p.needs.length : p.needs) parts.push(`needs ${Array.isArray(p.needs) ? p.needs.join('') : String(p.needs)}`)
  return parts.join(' · ') || (m.text ?? '')
}

/** game_start.setup may be "silent" (both teams) or combined like "T1=sharing,T2=silent". */
export function setupFor(setup: string | undefined, team: TeamId): string | null {
  if (!setup) return null
  const m = setup.match(new RegExp(`${team}=([^,;\\s]+)`))
  if (m) return m[1]
  return setup.includes('=') ? null : setup
}
