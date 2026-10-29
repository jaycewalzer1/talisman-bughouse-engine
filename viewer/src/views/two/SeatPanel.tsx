// The strip above/below a board for one player: name sticker, live clock, thinking indicator,
// stall tape and the player's pocket (reserve pieces they can drop).

import { type CSSProperties, useMemo } from 'react'
import { type BoardId, type Seat, TEAM_OF, fenPocket, seatToMove } from '../../events/types'
import { useGame } from '../../store/gameStore'
import { formatClock, glyph, seatColor, viewClock } from './util'

const POCKET_ORDER = 'qrbnp'

function Clock({ seat }: { seat: Seat }) {
  const ms = useGame(s => viewClock(s, seat))
  const running = useGame(s => !!s.start && !s.end && seatToMove(seat[0] as BoardId, s.fens[seat[0] as BoardId]) === seat)
  const cls = ['clock']
  if (running) cls.push('running')
  if (ms < 10_000) cls.push('low')
  return <div className={cls.join(' ')}>{formatClock(ms)}</div>
}

function Thinking({ seat }: { seat: Seat }) {
  const think = useGame(s => s.thinking[seat])
  const playhead = useGame(s => s.playheadTs)
  if (!think) return <div className="thinking idle" />
  const elapsed = Math.max(0, playhead - think.ts)
  const plan = think.think_ms ?? 0
  const frac = plan > 0 ? Math.min(1, elapsed / plan) : 0
  return (
    <div className="thinking active" title={`thinking: ${plan} ms planned`}>
      <span className="pulse" />
      <span className="think-label">thinking {(elapsed / 1000).toFixed(1)}s</span>
      <span className="think-bar"><span style={{ width: `${frac * 100}%` }} /></span>
    </div>
  )
}

function Pocket({ seat }: { seat: Seat }) {
  const fen = useGame(s => s.fens[seat[0] as BoardId])
  const white = seat.endsWith('white')
  const counts = useMemo(() => {
    const c: Record<string, number> = {}
    for (const ch of fenPocket(fen)) {
      if ((ch === ch.toUpperCase()) !== white) continue
      c[ch.toLowerCase()] = (c[ch.toLowerCase()] ?? 0) + 1
    }
    return c
  }, [fen, white])
  const types = POCKET_ORDER.split('').filter(t => counts[t])
  return (
    <div className="pocket" data-pocket={seat} title="reserve: pieces this player can drop">
      {types.length === 0 && <span className="pocket-empty">empty pocket</span>}
      {types.map(t => (
        // key includes the count so the chip re-mounts (and pops) when it changes
        <span key={`${t}${counts[t]}`} className={`chip ${white ? 'pw' : 'pb'}`}>
          <span className="glyph">{glyph(t)}</span>
          {counts[t] > 1 && <span className="count">×{counts[t]}</span>}
        </span>
      ))}
    </div>
  )
}

/** Most recent eval this seat reported (thinking end), from its own perspective. */
function useLastEval(seat: Seat): number | null {
  const events = useGame(s => s.events)
  return useMemo(() => {
    for (let i = events.length - 1; i >= 0; i--) {
      const e = events[i]
      if (e.type === 'thinking' && e.seat === seat && e.phase === 'end' && typeof e.eval === 'number') return e.eval
    }
    return null
  }, [events, seat])
}

export default function SeatPanel({ seat, position }: { seat: Seat; position: 'top' | 'bottom' }) {
  const info = useGame(s => s.start?.seats[seat])
  const stall = useGame(s => s.stalls[seat])
  const isLoser = useGame(s => s.end?.loserSeat === seat)
  const ev = useLastEval(seat)
  const team = TEAM_OF[seat]

  return (
    <div className={`seat seat-${position} ${stall ? 'stalling' : ''} ${isLoser ? 'loser' : ''}`}
      style={{ '--team': seatColor(seat) } as CSSProperties}>
      <div className="seat-id">
        <span className="name-sticker">{info?.name ?? seat}</span>
        <span className="seat-meta">
          {seat.replace('_', ' ')} · {team}{info?.kind ? ` · ${info.kind}` : ''}
          {ev !== null && <span className={`eval ${ev >= 0 ? 'up' : 'down'}`}> eval {ev >= 0 ? '+' : ''}{ev.toFixed(2)}</span>}
        </span>
      </div>
      <Pocket seat={seat} />
      <div className="seat-status">
        {stall ? <div className="stall-tape">SITTING · {stall.replace(/_/g, ' ')}</div> : <Thinking seat={seat} />}
      </div>
      <Clock seat={seat} />
    </div>
  )
}
