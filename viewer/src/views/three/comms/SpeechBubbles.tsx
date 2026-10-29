// Comic speech bubbles at the screen edges: team T1 down the left edge, T2 down the right, so they
// never sit over the boards (which the broadcast camera keeps centred). Each bubble comes with the
// speaker's portrait cut-in. At most STYLE.comms.maxBubblesPerTeam visible per edge; extra lines
// queue (and shorten the visible ones' hold); stale or overflowing queued lines are dropped.
import { useEffect, useReducer, useRef } from 'react'
import { type Seat, type TeamId, TEAM_OF } from '../../../events/types'
import { useGame } from '../../../store/gameStore'
import { onFxReset } from '../director'
import { fxBus } from '../fxBus'
import { STYLE } from '../style.config'
import { PortraitCutIn } from './PortraitCutIn'

interface Bubble {
  id: number
  seat: Seat
  toSeat: Seat
  team: TeamId
  line: string
  status: boolean
  queuedAt: number
  shownAt: number
  leftAt: number      // 0 = still up
}

const EXIT_MS = 280
const STALE_MS = 5000
const TEAMS: TeamId[] = ['T1', 'T2']

export function SpeechBubbles() {
  const m = useRef({
    shown: { T1: [] as Bubble[], T2: [] as Bubble[] },
    queue: { T1: [] as Bubble[], T2: [] as Bubble[] },
    id: 0,
  })
  const [, rerender] = useReducer((x: number) => x + 1, 0)

  useEffect(() => {
    const st = m.current
    const pump = () => {
      const now = performance.now()
      let changed = false
      const C = STYLE.comms
      for (const team of TEAMS) {
        const shown = st.shown[team], queue = st.queue[team]
        // drop stale queued lines
        const fresh = queue.filter(b => now - b.queuedAt < STALE_MS)
        if (fresh.length !== queue.length) { st.queue[team] = fresh; changed = true }
        const q = st.queue[team]
        // retire bubbles whose hold is over (shorter hold when others are waiting)
        const hold = q.length ? Math.max(1000, C.bubbleHoldMs * 0.55) : C.bubbleHoldMs
        for (const b of shown) {
          if (!b.leftAt && now - b.shownAt > (b.status ? hold * 0.8 : hold)) { b.leftAt = now; changed = true }
        }
        const kept = shown.filter(b => !b.leftAt || now - b.leftAt < EXIT_MS)
        if (kept.length !== shown.length) { st.shown[team] = kept; changed = true }
        // promote from the queue
        while (st.shown[team].filter(b => !b.leftAt).length < C.maxBubblesPerTeam && q.length) {
          const b = q.shift()!
          b.shownAt = now
          st.shown[team] = [...st.shown[team], b]
          changed = true
        }
      }
      if (changed) rerender()
    }

    const offBus = fxBus.on(e => {
      if (e.kind !== 'talk') return
      const team = TEAM_OF[e.seat]
      const q = st.queue[team]
      q.push({
        id: ++st.id, seat: e.seat, toSeat: e.toSeat, team, line: e.line, status: e.status,
        queuedAt: performance.now(), shownAt: 0, leftAt: 0,
      })
      while (q.length > STYLE.comms.bubbleQueueMax) {
        const i = q.findIndex(b => b.status)
        q.splice(i >= 0 ? i : 0, 1)
      }
      pump()
    })
    const offReset = onFxReset(() => {
      st.shown = { T1: [], T2: [] }; st.queue = { T1: [], T2: [] }; rerender()
    })
    const iv = window.setInterval(pump, 100)
    return () => { offBus(); offReset(); window.clearInterval(iv) }
  }, [])

  return (
    <>
      {TEAMS.map(team => (
        <div key={team} className={`h3-bubbles h3-bubbles-${team === 'T1' ? 'left' : 'right'}`}>
          {m.current.shown[team].map(b => <BubbleRow key={b.id} b={b} side={team === 'T1' ? 'left' : 'right'} />)}
        </div>
      ))}
    </>
  )
}

function BubbleRow({ b, side }: { b: Bubble; side: 'left' | 'right' }) {
  const names = useGame(s => s.start?.seats)
  const from = names?.[b.seat]?.name ?? b.seat.replace('_', ' ')
  const to = names?.[b.toSeat]?.name ?? b.toSeat.replace('_', ' ')
  return (
    <div className={`h3-bubble-row h3-${b.team} h3-side-${side}${b.leftAt ? ' h3-leaving' : ''}${b.status ? ' h3-status' : ''}`}>
      {STYLE.comms.portraits && <PortraitCutIn seat={b.seat} team={b.team} side={side} />}
      <div className="h3-bubble">
        <div className="h3-bubble-tag">{from} <span className="h3-bubble-to">→ {to}</span></div>
        <div className="h3-bubble-line">{b.line}</div>
      </div>
    </div>
  )
}
