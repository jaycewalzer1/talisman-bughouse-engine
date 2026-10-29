// One chess board. Pieces are absolutely positioned <span>s that keep a stable `id` from one
// position to the next, so when a piece's square changes React just updates its CSS transform
// and a CSS transition slides it there (that's the whole "animation engine").

import { useMemo, useRef } from 'react'
import { Chess } from 'chess.js'
import { type BoardId, fenGrid } from '../../events/types'
import { useGame } from '../../store/gameStore'
import { FLIPPED, glyph, isWhitePiece, squareXY, xySquare } from './util'

interface TrackedPiece { id: number; piece: string; sq: string; dropped: boolean }

let nextId = 1

/**
 * Turn the previous piece list + the new FEN grid into a new piece list, re-using ids where a
 * piece plausibly stayed or moved (same letter), so it animates instead of popping.
 */
function reconcile(prev: TrackedPiece[], grid: (string | null)[][], hintFrom: string | null): TrackedPiece[] {
  const want = new Map<string, string>()   // square -> piece letter in the new position
  grid.forEach((row, y) => row.forEach((p, x) => { if (p) want.set(xySquare(x, y, false), p) }))

  const next: TrackedPiece[] = []
  const leftovers: TrackedPiece[] = []
  for (const p of prev) {
    if (want.get(p.sq) === p.piece) { next.push({ ...p, dropped: false }); want.delete(p.sq) }
    else leftovers.push(p)
  }
  // squares still needing a piece: take a leftover of the same kind (prefer the last move's origin)
  for (const [sq, piece] of want) {
    const cands = leftovers.filter(p => p.piece === piece)
    const pick = cands.find(p => p.sq === hintFrom) ?? cands[0]
    if (pick) {
      leftovers.splice(leftovers.indexOf(pick), 1)
      next.push({ ...pick, sq, dropped: false })
    } else {
      next.push({ id: nextId++, piece, sq, dropped: prev.length > 0 })   // drop or promotion
    }
  }
  return next   // leftovers that found no home were captured: they just disappear
}

function sideInCheck(fen: string): boolean {
  try {
    const plain = fen.replace(/\[[^\]]*\]/, '').replace(/~/g, '')
    const c = new Chess()
    c.load(plain, { skipValidation: true })
    return c.inCheck()
  } catch {
    return false
  }
}

export default function Board({ board }: { board: BoardId }) {
  const fen = useGame(s => s.fens[board])
  const lastMove = useGame(s => s.lastMove[board])
  const flipped = FLIPPED[board]

  // piece tracking survives re-renders in a ref; recomputed only when the FEN changes
  const track = useRef<{ fen: string; pieces: TrackedPiece[] }>({ fen: '', pieces: [] })
  if (track.current.fen !== fen) {
    const hint = lastMove && !lastMove.isDrop ? lastMove.uci.slice(0, 2) : null
    track.current = { fen, pieces: reconcile(track.current.pieces, fenGrid(fen), hint) }
  }
  const pieces = track.current.pieces

  const turn = fen.split(' ')[1] === 'b' ? 'b' : 'w'
  const inCheck = useMemo(() => sideInCheck(fen), [fen])
  const checkSq = inCheck ? pieces.find(p => p.piece === (turn === 'w' ? 'K' : 'k'))?.sq : undefined

  const from = lastMove && !lastMove.isDrop ? lastMove.uci.slice(0, 2) : null
  const to = lastMove ? lastMove.uci.slice(2, 4) : null   // works for "e2e4", "e7e8q" and "N@f3"

  const squares = []
  for (let y = 0; y < 8; y++) {
    for (let x = 0; x < 8; x++) {
      const sq = xySquare(x, y, flipped)
      const dark = (x + y) % 2 === 1
      const cls = ['sq', dark ? 'dark' : 'light']
      if (sq === from) cls.push('hl-from')
      if (sq === to) cls.push(lastMove?.isDrop ? 'hl-drop' : 'hl-to')
      if (sq === checkSq) cls.push('check')
      squares.push(
        <div key={sq} className={cls.join(' ')} data-sq={`${board}:${sq}`}>
          {x === 0 && <span className="coord rank">{sq[1]}</span>}
          {y === 7 && <span className="coord file">{sq[0]}</span>}
        </div>,
      )
    }
  }

  return (
    <div className={`board board-${board}`}>
      <div className="squares">{squares}</div>
      <div className="pieces">
        {pieces.map(p => {
          const { x, y } = squareXY(p.sq, flipped)
          return (
            <span
              key={p.id}
              className={`piece ${isWhitePiece(p.piece) ? 'pw' : 'pb'}`}
              style={{ transform: `translate(${x * 100}%, ${y * 100}%)` }}
            >
              <span className={p.dropped ? 'glyph dropped' : 'glyph'}>{glyph(p.piece)}</span>
            </span>
          )
        })}
      </div>
      <div className="board-tag">BOARD {board}</div>
    </div>
  )
}
