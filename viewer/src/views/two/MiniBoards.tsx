// Compact "picture-in-picture" version of the 2D table, shown in a corner card while the 3D view
// is full screen (App.tsx). Same layout as the full 2D view, shrunk to what reads at a glance:
//
//        [ A_black  pocket  clock ] [ B_white  pocket  clock ]   <- T2 on top
//        [        BOARD A         ] [     BOARD B (flipped)  ]
//        [ A_white  pocket  clock ] [ B_black  pocket  clock ]   <- T1 at the bottom
//
// It's a separate, much lighter renderer than Board.tsx (no piece tracking / slide animations /
// capture flyers / coords, no data-sq hooks that Flyers.tsx could latch onto): just a 64-cell grid
// redrawn when the FEN changes, plus last-move highlights. Every store selector returns a primitive
// (FEN string, formatted clock string, …) so the card only re-renders when what it shows changes.

import { type CSSProperties, memo, useMemo } from 'react'
import { type BoardId, type Seat, fenGrid, fenPocket, seatToMove } from '../../events/types'
import { useGame } from '../../store/gameStore'
import { BOTTOM_SEAT, FLIPPED, TOP_SEAT, formatClock, glyph, isWhitePiece, seatColor, viewClock } from './util'
import './mini.css'

const POCKET_ORDER = 'qrbnp'

const MiniBoard = memo(function MiniBoard({ board }: { board: BoardId }) {
  const fen = useGame(s => s.fens[board])
  const uci = useGame(s => s.lastMove[board]?.uci ?? null)
  const isDrop = useGame(s => !!s.lastMove[board]?.isDrop)
  const grid = useMemo(() => fenGrid(fen), [fen])
  const flipped = FLIPPED[board]
  const from = uci && !isDrop ? uci.slice(0, 2) : null
  const to = uci ? uci.slice(2, 4) : null

  const cells = []
  for (let y = 0; y < 8; y++) {
    for (let x = 0; x < 8; x++) {
      const gx = flipped ? 7 - x : x, gy = flipped ? 7 - y : y
      const sq = 'abcdefgh'[gx] + (8 - gy)
      const p = grid[gy]?.[gx] ?? null
      let cls = (x + y) % 2 ? 'mb-sq mb-dark' : 'mb-sq mb-light'
      if (sq === from) cls += ' mb-from'
      if (sq === to) cls += isDrop ? ' mb-drop' : ' mb-to'
      cells.push(
        <div key={sq} className={cls}>
          {p && <span className={isWhitePiece(p) ? 'mb-p mb-pw' : 'mb-p mb-pb'}>{glyph(p)}</span>}
        </div>,
      )
    }
  }
  return (
    <div className="mb-board">
      {cells}
      <span className="mb-board-tag">{board}</span>
    </div>
  )
})

function MiniPocket({ seat }: { seat: Seat }) {
  const pocket = useGame(s => fenPocket(s.fens[seat[0] as BoardId]))
  const white = seat.endsWith('white')
  const counts: Record<string, number> = {}
  for (const ch of pocket) {
    if ((ch === ch.toUpperCase()) !== white) continue
    counts[ch.toLowerCase()] = (counts[ch.toLowerCase()] ?? 0) + 1
  }
  const types = POCKET_ORDER.split('').filter(t => counts[t])
  return (
    <span className="mb-pocket" title="pocket: pieces this player can drop">
      {types.length === 0 ? <span className="mb-pocket-empty">–</span> : types.map(t => (
        <span key={t} className="mb-chip">
          <span className={white ? 'mb-pw' : 'mb-pb'}>{glyph(t)}</span>
          {counts[t] > 1 && <sub>{counts[t]}</sub>}
        </span>
      ))}
    </span>
  )
}

function MiniSeat({ seat }: { seat: Seat }) {
  const name = useGame(s => s.start?.seats[seat]?.name ?? seat.replace('_', ' '))
  const clock = useGame(s => formatClock(viewClock(s, seat)))
  const low = useGame(s => viewClock(s, seat) < 10_000)
  const running = useGame(s => !!s.start && !s.end && seatToMove(seat[0] as BoardId, s.fens[seat[0] as BoardId]) === seat)
  const sitting = useGame(s => !!s.stalls[seat])
  return (
    <div className={`mb-seat${sitting ? ' mb-sitting' : ''}`} style={{ '--team': seatColor(seat) } as CSSProperties}>
      <div className="mb-seat-row">
        <span className="mb-name" title={`${name} (${seat.replace('_', ' ')})`}>{sitting ? '⏸ ' : ''}{name}</span>
        <span className={`mb-clock${running ? ' mb-running' : ''}${low ? ' mb-low' : ''}`}>{clock}</span>
      </div>
      <MiniPocket seat={seat} />
    </div>
  )
}

function MiniResult() {
  const result = useGame(s => s.end?.result ?? null)
  if (!result) return null
  return <div className="mb-result">{result === 'draw' ? 'DRAW' : `${result} WINS`}</div>
}

/** Both boards, pockets and clocks in ~250 px. Pure display: the parent card handles clicks. */
export default function MiniBoards() {
  return (
    <div className="mb-root">
      {(['A', 'B'] as BoardId[]).map(b => (
        <div className="mb-table" key={b}>
          <MiniSeat seat={TOP_SEAT[b]} />
          <MiniBoard board={b} />
          <MiniSeat seat={BOTTOM_SEAT[b]} />
        </div>
      ))}
      <MiniResult />
    </div>
  )
}
