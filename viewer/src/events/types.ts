// Event schema shared by the 2D and 3D views. Source of truth: ../../../docs/events.md

export type Seat = 'A_white' | 'A_black' | 'B_white' | 'B_black'
export type BoardId = 'A' | 'B'
export type TeamId = 'T1' | 'T2'
export type Clocks = Record<Seat, number>

export const SEATS: Seat[] = ['A_white', 'A_black', 'B_white', 'B_black']
export const BOARD_OF: Record<Seat, BoardId> = { A_white: 'A', A_black: 'A', B_white: 'B', B_black: 'B' }
export const TEAM_OF: Record<Seat, TeamId> = { A_white: 'T1', B_black: 'T1', A_black: 'T2', B_white: 'T2' }
export const PARTNER: Record<Seat, Seat> = { A_white: 'B_black', B_black: 'A_white', A_black: 'B_white', B_white: 'A_black' }

interface Base { seq: number; ts: number }

export interface SeatInfo { name: string; team: TeamId; kind: string }

export interface GameStartEvent extends Base {
  type: 'game_start'
  game_id: string
  setup: string
  seats: Record<Seat, SeatInfo>
  fens: Record<BoardId, string>
  clocks: Clocks
  base_ms: number
  increment_ms: number
  channel_delay_ms: number
}

export interface ThinkingEvent extends Base {
  type: 'thinking'
  seat: Seat
  phase: 'start' | 'end'
  think_ms?: number | null
  nodes?: number | null
  eval?: number | null
  depth?: number | null
  /** the search also ran on the partner's idle machine (split setups, --lend) */
  borrowed?: boolean
}

export interface MoveEvent extends Base {
  type: 'move'
  board: BoardId
  seat: Seat
  uci: string
  san?: string
  isDrop: boolean
  capture: string | null
  fen: string
  clocks: Clocks
}

export interface PassEvent extends Base {
  type: 'pass'
  fromBoard: BoardId
  fromSeat: Seat
  toSeat: Seat
  piece: string
}

export interface MessageEvent extends Base {
  type: 'message'
  deliverTs: number
  fromSeat: Seat
  toSeat: Seat
  token: string
  text?: string | null
  payload?: Record<string, unknown>
}

export interface StallEvent extends Base {
  type: 'stall'
  seat: Seat
  active: boolean
  reason?: string | null
}

export interface GameEndEvent extends Base {
  type: 'game_end'
  result: TeamId | 'draw'
  reason: string
  board?: BoardId | null
  loserSeat?: Seat | null
}

export type GameEvent =
  | GameStartEvent | ThinkingEvent | MoveEvent | PassEvent | MessageEvent | StallEvent | GameEndEvent

/** Side to move ('w' | 'b') from a FEN. */
export function fenTurn(fen: string): 'w' | 'b' {
  return fen.split(' ')[1] === 'b' ? 'b' : 'w'
}

export function seatToMove(board: BoardId, fen: string): Seat {
  return `${board}_${fenTurn(fen) === 'w' ? 'white' : 'black'}` as Seat
}

/** Pocket letters from a bracket-form FEN, e.g. "...[Nnp] w" -> "Nnp". */
export function fenPocket(fen: string): string {
  const m = fen.split(' ')[0].match(/\[([^\]]*)\]/)
  return m ? m[1] : ''
}

/** 8x8 placement (rank 8 first) from a bracket-form FEN; null for empty squares. */
export function fenGrid(fen: string): (string | null)[][] {
  const placement = fen.split(' ')[0].replace(/\[.*\]/, '')
  return placement.split('/').slice(0, 8).map(row => {
    const out: (string | null)[] = []
    for (const ch of row.replace(/~/g, '')) {
      if (/\d/.test(ch)) for (let i = 0; i < +ch; i++) out.push(null)
      else out.push(ch)
    }
    return out
  })
}
