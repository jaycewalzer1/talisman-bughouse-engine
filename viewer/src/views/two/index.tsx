// The 2D view: both boards side by side like a real bughouse table.
//
//        [ T2 radio lane:  A_black  <──── messages ────>  B_white ]
//        [ A_black panel ]                      [ B_white panel ]
//        [    BOARD A    ]                      [ BOARD B (flipped) ]
//        [ A_white panel ]                      [ B_black panel ]
//        [ T1 radio lane:  A_white  <──── messages ────>  B_black ]
//
// Partners (same team) are on the same side, so messages travel straight across.
// Everything reads from the shared game store (src/store/gameStore.ts).

import type { CSSProperties } from 'react'
import { type BoardId, type Seat, TEAM_OF } from '../../events/types'
import { useGame } from '../../store/gameStore'
import Board from './Board'
import { CommsLane, CommsLog } from './Comms'
import Flyers from './Flyers'
import SeatPanel from './SeatPanel'
import { BOTTOM_SEAT, TEAM_COLOR, TOP_SEAT, formatTs, seatColor, setupFor } from './util'
import './two.css'

function LastMove({ board }: { board: BoardId }) {
  const mv = useGame(s => s.lastMove[board])
  // ply count from the FEN's move number + side to move (cheap; no need to scan events)
  const plies = useGame(s => {
    const f = s.fens[board].split(' ')
    return (+f[5] - 1) * 2 + (f[1] === 'b' ? 1 : 0)
  })
  return (
    <div className="lastmove">
      {mv ? <>last: <b>{mv.san ?? mv.uci}</b> <span className="dim">({mv.seat.replace('_', ' ')}, ply {plies})</span></> : <span className="dim">waiting for first move…</span>}
    </div>
  )
}

function Result() {
  const end = useGame(s => s.end)
  const seats = useGame(s => s.start?.seats)
  if (!end) return null
  const name = (s?: Seat | null) => (s && seats?.[s]?.name) || s?.replace('_', ' ') || ''
  const winners = end.result === 'draw' ? [] : (Object.keys(TEAM_OF) as Seat[]).filter(s => TEAM_OF[s] === end.result)
  const why = end.reason === 'checkmate' ? `${name(end.loserSeat)} checkmated on board ${end.board}`
    : end.reason === 'flag' ? `${name(end.loserSeat)} flagged on board ${end.board}`
      : end.reason.replace(/_/g, ' ')
  const color = end.result === 'draw' ? '#f4f1e8' : TEAM_COLOR[end.result]
  return (
    <div className="result">
      <div className="result-sticker" style={{ '--team': color } as CSSProperties}>
        <div className="result-big">{end.result === 'draw' ? 'DRAW' : `${end.result} WINS`}</div>
        {winners.length > 0 && <div className="result-names">{winners.map(name).join(' + ')}</div>}
        <div className="result-why">{why} · {formatTs(end.ts)}</div>
      </div>
    </div>
  )
}

function Header() {
  const start = useGame(s => s.start)
  const playhead = useGame(s => s.playheadTs)
  const seats = start?.seats
  const teamLine = (a: Seat, b: Seat) => `${seats?.[a]?.name ?? a} + ${seats?.[b]?.name ?? b}`
  const setupTag = (t: 'T1' | 'T2') => {
    const s = setupFor(start?.setup, t)
    return s ? <span className="legend-setup">{s}</span> : null
  }
  return (
    <header className="tv-header">
      <div className="tv-title"><span className="tag-title">BUGHOUSE</span><span className="tv-sub">2D table</span></div>
      <div className="legend">
        <span className="legend-team" style={{ '--team': TEAM_COLOR.T1 } as CSSProperties}>T1 · {teamLine('A_white', 'B_black')}{setupTag('T1')}</span>
        <span className="legend-team" style={{ '--team': TEAM_COLOR.T2 } as CSSProperties}>T2 · {teamLine('A_black', 'B_white')}{setupTag('T2')}</span>
      </div>
      <div className="tv-meta">
        {start ? <>
          {start.game_id} · {Math.round(start.base_ms / 1000)}s
          {start.channel_delay_ms ? ` · delay ${start.channel_delay_ms}ms` : ''} · <b>{formatTs(playhead)}</b>
        </> : 'waiting for game_start…'}
      </div>
    </header>
  )
}

export default function TwoView() {
  return (
    <div className="tv-root">
      <Header />
      <div className="tv-main">
        <div className="arena">
          <CommsLane team="T2" />
          <div className="tables">
            {(['A', 'B'] as BoardId[]).map(b => (
              <div className="table" key={b} style={{ '--top': seatColor(TOP_SEAT[b]), '--bottom': seatColor(BOTTOM_SEAT[b]) } as CSSProperties}>
                <SeatPanel seat={TOP_SEAT[b]} position="top" />
                <Board board={b} />
                <SeatPanel seat={BOTTOM_SEAT[b]} position="bottom" />
                <LastMove board={b} />
              </div>
            ))}
          </div>
          <CommsLane team="T1" />
          <Flyers />
          <Result />
        </div>
        <CommsLog />
      </div>
    </div>
  )
}
