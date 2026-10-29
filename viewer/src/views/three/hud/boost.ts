// Pure math for the boost meters (no React/three) so it can be checked in node.
//   material: pieces on both boards + both reserves, P1 N3 B3 R5 Q9, counted per team
//             (T1 = white on A + black on B, T2 = black on A + white on B).
//   clock edge: sum over the team's seats of (own clock - same-board opponent clock).
import { type BoardId, type Clocks, type GameEvent, type TeamId, fenGrid, fenPocket } from '../../../events/types'

export const VALUE: Record<string, number> = { p: 1, n: 3, b: 3, r: 5, q: 9, k: 0 }

/** Board pockets: FEN pocket plus passes that arrived after that board's last move. */
export function livePockets(fens: Record<BoardId, string>, events: GameEvent[]): Record<BoardId, string> {
  const v: Record<BoardId, string> = { A: fenPocket(fens.A).replace(/~/g, ''), B: fenPocket(fens.B).replace(/~/g, '') }
  const done: Record<BoardId, boolean> = { A: false, B: false }
  for (let i = events.length - 1; i >= 0 && !(done.A && done.B); i--) {
    const e = events[i]
    if (e.type === 'game_start') break
    if (e.type === 'move') done[e.board] = true
    else if (e.type === 'pass') { const to = e.toSeat[0] as BoardId; if (!done[to]) v[to] += e.piece }
  }
  return v
}

/** Material points per team across both boards and reserves. */
export function teamMaterial(fens: Record<BoardId, string>, pockets: Record<BoardId, string>): Record<TeamId, number> {
  const out: Record<TeamId, number> = { T1: 0, T2: 0 }
  const add = (board: BoardId, ch: string) => {
    const val = VALUE[ch.toLowerCase()] ?? 0
    const white = ch === ch.toUpperCase()
    // board A: white = T1; board B: white = T2
    const team: TeamId = (board === 'A') === white ? 'T1' : 'T2'
    out[team] += val
  }
  for (const board of ['A', 'B'] as BoardId[]) {
    for (const row of fenGrid(fens[board])) for (const ch of row) if (ch) add(board, ch)
    for (const ch of pockets[board]) if (/[a-z]/i.test(ch)) add(board, ch)
  }
  return out
}

/** T1's clock edge in ms (T2's is the negative). */
export function clockEdgeT1(c: Clocks): number {
  return (c.A_white - c.A_black) + (c.B_black - c.B_white)
}

const clamp1 = (x: number) => Math.max(-1, Math.min(1, x))

/**
 * T1's overall edge in [-1, 1] (T2's is the negative).
 * @param matDiff  T1 material minus T2 material (points)
 * @param clockMs  T1 clock edge (ms)
 */
export function blendEdge(matDiff: number, clockMs: number, baseMs: number, clockWeight: number, matScale: number): number {
  const m = clamp1(matDiff / Math.max(1, matScale))
  const c = clamp1(clockMs / Math.max(1000, baseMs * 0.6))
  return clamp1((1 - clockWeight) * m + clockWeight * c)
}

/** Meter fill 0..1 for a team from T1's edge. */
export const fillFor = (team: TeamId, edgeT1: number) => 0.5 + 0.5 * (team === 'T1' ? edgeT1 : -edgeT1)
