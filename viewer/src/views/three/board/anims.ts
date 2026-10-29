// Piece animation manager (plain module, no React). The director feeds it events; Pieces.tsx draws
// the active animations every frame and hides the squares/pile slots they cover.
//
// Safety for fast replays: a new move on a board instantly finishes that board's previous move
// animation, durations shrink when moves come quickly, and reset() drops everything.
import { Vector3 } from 'three'
import { fenGrid, fenPocket, type BoardId, type MoveEvent, type PassEvent, type Seat } from '../../../events/types'
import { useGame, type GameState } from '../../../store/gameStore'
import { reserveSlot, squareWorld } from '../layout'
import * as sfx from '../audio/sfx'
import { fx } from '../viewStore'

export type AnimKind = 'move' | 'ghost' | 'toss'

export interface PieceAnim {
  kind: AnimKind
  letter: string
  board: BoardId              // board this animation belongs to (move/ghost) or lands on (toss)
  hideSq?: string             // square on `board` whose static piece is hidden while animating
  pile?: string               // reserve pile key `${board}:${letter}` this toss lands in
  seat?: Seat                 // mover
  from: Vector3
  to: Vector3
  start: number               // seconds (performance clock)
  dur: number
  arc: number
  spin: number
  pos: Vector3                // current position (written by update)
  tilt: number
  landed: boolean
}

export const now = () => performance.now() / 1000

const anims: PieceAnim[] = []
const lastMoveAt: Partial<Record<BoardId, number>> = {}
const lastDest: Partial<Record<BoardId, { sq: string; landAt: number }>> = {}
let prevFen: Partial<Record<BoardId, string>> = {}

export function activeAnims(): readonly PieceAnim[] { return anims }

export function resetAnims() {
  anims.length = 0
  const s = useGame.getState()
  prevFen = { ...s.fens }
}

const dropFinished = (pred: (a: PieceAnim) => boolean) => {
  for (let i = anims.length - 1; i >= 0; i--) if (pred(anims[i])) anims.splice(i, 1)
}

function letterAt(fen: string | undefined, sq: string): string | null {
  if (!fen) return null
  const f = sq.charCodeAt(0) - 97, r = sq.charCodeAt(1) - 49
  return fenGrid(fen)[7 - r]?.[f] ?? null
}

// ---- reserves: the board FEN pocket plus passes that arrived after that board's last move ----
let resCache: { events: unknown; fens: unknown; v: Record<BoardId, string> } | null = null
export function reserves(s: Pick<GameState, 'events' | 'fens'>): Record<BoardId, string> {
  if (resCache && resCache.events === s.events && resCache.fens === s.fens) return resCache.v
  const v: Record<BoardId, string> = { A: fenPocket(s.fens.A).replace(/~/g, ''), B: fenPocket(s.fens.B).replace(/~/g, '') }
  const done: Record<BoardId, boolean> = { A: false, B: false }
  for (let i = s.events.length - 1; i >= 0 && !(done.A && done.B); i--) {
    const e = s.events[i]
    if (e.type === 'move' || e.type === 'game_start') {
      if (e.type === 'game_start') break
      done[e.board] = true
    } else if (e.type === 'pass') {
      const to = e.toSeat[0] as BoardId
      if (!done[to]) v[to] += e.piece
    }
  }
  resCache = { events: s.events, fens: s.fens, v }
  return v
}
const countIn = (pocket: string, letter: string) => [...pocket].filter(c => c === letter).length

export function onMove(e: MoveEvent) {
  const t = now()
  // finish whatever was still animating on this board
  dropFinished(a => a.board === e.board && a.kind !== 'toss')
  const gap = lastMoveAt[e.board] === undefined ? 9 : t - lastMoveAt[e.board]!
  lastMoveAt[e.board] = t
  const dur = Math.max(0.18, Math.min(0.55, gap * 0.7))
  const before = prevFen[e.board]
  prevFen[e.board] = e.fen
  const dest = e.isDrop ? e.uci.slice(-2) : e.uci.slice(2, 4)
  const letter = letterAt(e.fen, dest) ?? (e.isDrop ? e.uci[0] : 'P')
  const white = letter === letter.toUpperCase()
  let from: Vector3
  if (e.isDrop) {
    const pocketNow = reserves(useGame.getState())[e.board]
    from = reserveSlot(e.board, white ? 'w' : 'b', letter, countIn(pocketNow, letter))
  } else {
    from = squareWorld(e.board, e.uci.slice(0, 2))
  }
  const toV = squareWorld(e.board, dest)
  const knight = letter.toLowerCase() === 'n'
  anims.push({
    kind: 'move', letter, board: e.board, hideSq: dest, seat: e.seat, from, to: toV, start: t, dur,
    arc: e.isDrop ? 0.22 : knight ? 0.16 : 0.1, spin: 0, pos: from.clone(), tilt: 0, landed: false,
  })
  // captured piece stays put until the slam
  if (e.capture && !e.isDrop) {
    let capSq = dest
    if (before && !letterAt(before, dest)) capSq = dest[0] + e.uci[1] // en passant
    const capLetter = letterAt(before, capSq) ?? e.capture
    const p = squareWorld(e.board, capSq)
    anims.push({ kind: 'ghost', letter: capLetter, board: e.board, from: p, to: p, start: t, dur, arc: 0, spin: 0, pos: p.clone(), tilt: 0, landed: false })
  }
  lastDest[e.board] = { sq: dest, landAt: t + dur }
  // castling (king-to-target): slide the rook too
  if (!e.isDrop && letter.toLowerCase() === 'k' && Math.abs(e.uci.charCodeAt(0) - e.uci.charCodeAt(2)) === 2) {
    const rank = e.uci[1], kingside = e.uci[2] > e.uci[0]
    const rFrom = (kingside ? 'h' : 'a') + rank, rTo = (kingside ? 'f' : 'd') + rank
    anims.push({
      kind: 'move', letter: white ? 'R' : 'r', board: e.board, hideSq: rTo, from: squareWorld(e.board, rFrom),
      to: squareWorld(e.board, rTo), start: t + dur * 0.3, dur: dur * 0.7, arc: 0.06, spin: 0, pos: new Vector3(), tilt: 0, landed: false,
    })
  }
}

export function onPass(e: PassEvent) {
  const t = now()
  const src = lastDest[e.fromBoard]
  const from = squareWorld(e.fromBoard, src?.sq ?? 'e4', undefined)
  const start = Math.max(t, src?.landAt ?? t)
  const toBoard = e.toSeat[0] as BoardId
  const pile = reserves(useGame.getState())[toBoard]
  const white = e.piece === e.piece.toUpperCase()
  const k = Math.max(0, countIn(pile, e.piece) - 1)
  const to = reserveSlot(toBoard, white ? 'w' : 'b', e.piece, k)
  // keep at most 6 tosses in the air
  const tosses = anims.filter(a => a.kind === 'toss')
  if (tosses.length >= 6) anims.splice(anims.indexOf(tosses[0]), 1)
  anims.push({
    kind: 'toss', letter: e.piece, board: toBoard, pile: `${toBoard}:${e.piece}`, seat: e.fromSeat, from, to, start,
    dur: 0.95, arc: 1.4, spin: 9 + Math.random() * 6, pos: from.clone(), tilt: 0, landed: false,
  })
}

/** How many tossed pieces are still in the air toward a pile (they are hidden in the pile). */
export function inflight(board: BoardId, letter: string): number {
  let n = 0
  for (const a of anims) if (a.kind === 'toss' && !a.landed && a.pile === `${board}:${letter}`) n++
  return n
}

const easeInOut = (u: number) => (u < 0.5 ? 4 * u * u * u : 1 - Math.pow(-2 * u + 2, 3) / 2)

/** Advance all animations; called once per frame by Pieces. */
export function updateAnims(t = now()) {
  for (let i = anims.length - 1; i >= 0; i--) {
    const a = anims[i]
    const u = Math.min(1, Math.max(0, (t - a.start) / a.dur))
    if (a.kind === 'ghost') {
      a.pos.copy(a.from)
    } else if (a.kind === 'move') {
      const e = easeInOut(u)
      a.pos.lerpVectors(a.from, a.to, e)
      // lift, arc, then slam down hard at the end
      const lift = u < 0.85 ? Math.sin(Math.min(1, u / 0.85) * Math.PI * 0.5) : Math.pow(1 - (u - 0.85) / 0.15, 2)
      a.pos.y += a.arc * Math.sin(Math.PI * e) * 0.6 + lift * 0.06
      a.tilt = Math.sin(Math.PI * u) * 0.35
    } else {
      a.pos.lerpVectors(a.from, a.to, u)
      a.pos.y += a.arc * 4 * u * (1 - u)
      a.tilt = a.spin * u
    }
    if (u >= 1 && !a.landed) {
      a.landed = true
      if (a.kind === 'move') { sfx.slam(); fx.shake = Math.max(fx.shake, 0.12) }
      else if (a.kind === 'toss') sfx.clack()
    }
    if (t > a.start + a.dur) anims.splice(i, 1)
  }
}

/** Where the mover's hand should be right now (for the reach pose), or null. */
export function reachTarget(seat: Seat): Vector3 | null {
  for (const a of anims) if (a.kind === 'move' && a.seat === seat && now() >= a.start) return a.pos
  return null
}
