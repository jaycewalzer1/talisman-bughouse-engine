// Where everything sits in the world (units are meters, y is up).
// Board A is on the left (x < 0), board B on the right. Team T1 (A_white + B_black) stands on the
// far side (z < 0, facing the camera); team T2 (A_black + B_white) on the near side.
import { Vector3 } from 'three'
import type { BoardId, Seat, TeamId } from '../../events/types'

export const SQ = 0.09                      // one chess square
export const CRATE = 0.33                   // milk crate edge
export const TABLE_W = 1.5, TABLE_D = 1.0   // plywood sheet
export const TABLE_Y = CRATE * 2            // top of the crate stacks
export const BOARD_Y = TABLE_Y + 0.03       // surface the pieces stand on
export const WALL_Z = -4.2                  // graffiti wall
export const BOARD_X: Record<BoardId, number> = { A: -1.15, B: 1.15 }
// Board A is turned around so its white player (T1) stands on the far side.
export const BOARD_ROT: Record<BoardId, number> = { A: Math.PI, B: 0 }

export const TEAM_COLOR: Record<TeamId, string> = { T1: '#ff8a1f', T2: '#27d8e6' }

/** Board-local -> world. Board-local: white stands at +z, files run -x (a) to +x (h). */
export function boardToWorld(board: BoardId, x: number, y: number, z: number, out = new Vector3()): Vector3 {
  const t = BOARD_ROT[board], c = Math.cos(t), s = Math.sin(t)
  return out.set(BOARD_X[board] + x * c + z * s, y, -x * s + z * c)
}

/** 'e4' -> board-local x/z of the square center. */
export function squareLocal(sq: string): [number, number] {
  const f = sq.charCodeAt(0) - 97, r = sq.charCodeAt(1) - 49
  return [(f - 3.5) * SQ, (3.5 - r) * SQ]
}

export function squareWorld(board: BoardId, sq: string, y = BOARD_Y, out = new Vector3()): Vector3 {
  const [x, z] = squareLocal(sq)
  return boardToWorld(board, x, y, z, out)
}

export function seatColor(seat: Seat): 'w' | 'b' { return seat.endsWith('white') ? 'w' : 'b' }

/** Where a player stands and which way they face (rotation about y). Rigs face +z when rotY = 0. */
export function seatPose(seat: Seat): { pos: Vector3; rotY: number } {
  const board = seat[0] as BoardId
  const white = seatColor(seat) === 'w'
  const pos = boardToWorld(board, 0, 0, white ? 0.84 : -0.84)
  return { pos, rotY: (white ? Math.PI : 0) + BOARD_ROT[board] }
}

/** Reserve pile slot: color c's k-th piece of type `type` (P,N,B,R,Q) next to its owner on that board. */
const PILE_TYPES = 'PNBRQ'
export function reserveSlot(board: BoardId, c: 'w' | 'b', type: string, k: number, out = new Vector3()): Vector3 {
  const t = Math.max(0, PILE_TYPES.indexOf(type.toUpperCase()))
  const layer = Math.floor(k / 4), kk = k % 4
  const x = 0.44 + kk * 0.06 + (layer % 2) * 0.03
  const z = 0.36 - t * 0.075
  const sign = c === 'w' ? 1 : -1
  return boardToWorld(board, x * sign, BOARD_Y - 0.01 + 0.028 + layer * 0.045, z * sign, out)
}

export const CLOCK_LOCAL = { x: -0.6, z: 0.22 }
