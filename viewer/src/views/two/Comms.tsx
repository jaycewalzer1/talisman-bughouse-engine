// Agent communication, made obvious:
//  - <CommsLane team="T2|T1">: a strip that spans BOTH boards (partners sit on different boards),
//    showing each fresh message as a bubble riding an arrow from the sender's side to the partner's.
//  - <CommsLog>: a scrollable log of every message (plus stalls, piece passes and the result).

import { type CSSProperties, memo, useEffect, useMemo, useRef, useState } from 'react'
import { type GameEvent, type MessageEvent, type Seat, type TeamId, TEAM_OF } from '../../events/types'
import { useGame } from '../../store/gameStore'
import { formatTs, glyph, isStatusMsg, seatColor, statusSummary, tokenKind, tokenLabel } from './util'

const SHOW_MS = 4000      // how long a bubble stays in the lane after delivery
const MAX_IN_LANE = 2

// lane endpoints: left side = board A seat, right side = board B seat
const LANE_SEATS: Record<TeamId, { left: Seat; right: Seat }> = {
  T2: { left: 'A_black', right: 'B_white' },
  T1: { left: 'A_white', right: 'B_black' },
}

function Bubble({ m, inTransit }: { m: MessageEvent; inTransit: boolean }) {
  const rightward = m.fromSeat.startsWith('A')     // A -> B travels left to right
  return (
    <div className={`lane-msg ${rightward ? 'ltr' : 'rtl'} ${inTransit ? 'transit' : ''}`}>
      <svg className="lane-arrow" viewBox="0 0 100 10" preserveAspectRatio="none" aria-hidden>
        <line x1="2" y1="5" x2="98" y2="5" />
        {rightward ? <polyline points="94,1 98,5 94,9" /> : <polyline points="6,1 2,5 6,9" />}
      </svg>
      <div className="bubble">
        <span className={`token tk-${tokenKind(m.token)}`}>{tokenLabel(m.token)}</span>
        {m.text && <span className="bubble-text">{m.text}</span>}
      </div>
    </div>
  )
}

export function CommsLane({ team }: { team: TeamId }) {
  const messages = useGame(s => s.messages)
  const playhead = useGame(s => s.playheadTs)
  const { left, right } = LANE_SEATS[team]
  // in transit (sent, not yet delivered) or delivered within the last SHOW_MS
  const recent = messages.filter(m => TEAM_OF[m.fromSeat] === team && m.ts <= playhead && playhead < m.deliverTs + SHOW_MS)
  // status pings ("sharing" setups) don't get bubbles; just a small ticker line
  const visible = recent.filter(m => !isStatusMsg(m)).slice(-MAX_IN_LANE)
  const lastStatus = recent.filter(m => isStatusMsg(m) && playhead < m.deliverTs + 1500).slice(-1)[0]
  return (
    <div className={`lane lane-${team}`} style={{ '--team': seatColor(left) } as CSSProperties}>
      <span className="lane-end left">{left.replace('_', ' ')}</span>
      <div className="lane-track">
        {visible.length === 0 && !lastStatus && <span className="lane-quiet">{team} radio · quiet</span>}
        {visible.map(m => <Bubble key={m.seq} m={m} inTransit={m.deliverTs > playhead} />)}
        {lastStatus && visible.length < MAX_IN_LANE && (
          <span className="lane-status">📡 {lastStatus.fromSeat.replace('_', ' ')} status · {statusSummary(lastStatus)}</span>
        )}
      </div>
      <span className="lane-end right">{right.replace('_', ' ')}</span>
    </div>
  )
}

function SeatTag({ seat }: { seat: Seat }) {
  return <span className="seat-tag" style={{ color: seatColor(seat) }}>{seat.replace('_', ' ')}</span>
}

// memo: lines only re-render when their own props change, not on every clock frame
const LogLine = memo(function LogLine({ e, inTransit }: { e: GameEvent; inTransit: boolean }) {
  const t = <span className="log-ts">{formatTs(e.ts)}</span>
  switch (e.type) {
    case 'message':
      if (isStatusMsg(e)) return (
        <li className="log-status">
          {t}📡 <SeatTag seat={e.fromSeat} /> status <span className="log-text-inline">{statusSummary(e)}</span>
        </li>
      )
      return (
        <li className="log-msg">
          {t}<SeatTag seat={e.fromSeat} /><span className="log-arrow">→</span><SeatTag seat={e.toSeat} />
          <span className={`token tk-${tokenKind(e.token)}`}>{tokenLabel(e.token)}</span>
          {e.text && <span className="log-text">{e.text}</span>}
          {inTransit && <span className="log-transit">in transit…</span>}
        </li>
      )
    case 'stall':
      return (
        <li className="log-minor">
          {t}<SeatTag seat={e.seat} /> {e.active ? `sits (${(e.reason ?? 'stall').replace(/_/g, ' ')})` : 'stops sitting'}
        </li>
      )
    case 'pass':
      return (
        <li className="log-minor">
          {t}<SeatTag seat={e.fromSeat} /> passes <span className="log-glyph">{glyph(e.piece)}</span> to <SeatTag seat={e.toSeat} />
        </li>
      )
    case 'game_end':
      return <li className="log-end">{t} game over: {e.result === 'draw' ? 'draw' : `${e.result} wins`} ({e.reason})</li>
    default:
      return null
  }
})

export function CommsLog() {
  const events = useGame(s => s.events)
  const playhead = useGame(s => s.playheadTs)
  const listRef = useRef<HTMLUListElement>(null)
  const [showStatus, setShowStatus] = useState(false)
  const { items, msgCount, statusCount } = useMemo(() => {
    let msgCount = 0, statusCount = 0
    const items = events.filter(e => {
      if (e.type === 'message') {
        if (isStatusMsg(e)) { statusCount++; return showStatus }
        msgCount++
        return true
      }
      return e.type === 'stall' || e.type === 'pass' || e.type === 'game_end'
    }).slice(-250)
    return { items, msgCount, statusCount }
  }, [events, showStatus])

  // keep the newest line in view
  useEffect(() => {
    const el = listRef.current
    if (el) el.scrollTop = el.scrollHeight
  }, [items.length])

  return (
    <aside className="log">
      <div className="log-head"><span className="tape">TEAM RADIO</span><span className="log-count">{msgCount} msgs</span></div>
      {statusCount > 0 && (
        <label className="log-filter">
          <input type="checkbox" checked={showStatus} onChange={e => setShowStatus(e.target.checked)} />
          show 📡 status pings ({statusCount})
        </label>
      )}
      <ul ref={listRef}>
        {items.length === 0 && <li className="log-minor">no chatter yet…</li>}
        {items.map(e => <LogLine key={e.seq} e={e} inTransit={e.type === 'message' && e.deliverTs > playhead} />)}
      </ul>
    </aside>
  )
}
