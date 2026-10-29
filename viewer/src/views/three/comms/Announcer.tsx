// Announcer layer: big hype text slammed across the top-centre for big moments (fxBus `announce`).
// One at a time, short hold, min gap between calls, at most one waiting (others dropped), so fast
// games don't turn into a wall of text. Game-over calls jump the queue.
import { useEffect, useRef, useState } from 'react'
import type { TeamId } from '../../../events/types'
import { onFxReset } from '../director'
import { fxBus } from '../fxBus'
import { STYLE } from '../style.config'
import { isEndText } from './lines'

interface Call { id: number; text: string; team?: TeamId; end: boolean }

export function Announcer() {
  const [cur, setCur] = useState<Call | null>(null)
  const st = useRef({ id: 0, pending: null as Call | null, busyUntil: 0, timer: 0 })

  useEffect(() => {
    const s = st.current
    const C = STYLE.comms
    const show = (c: Call) => {
      setCur(c)
      const hold = c.end ? C.announcerHoldMs * 1.6 : C.announcerHoldMs
      s.busyUntil = performance.now() + hold + C.announcerGapMs
      window.clearTimeout(s.timer)
      s.timer = window.setTimeout(() => {
        setCur(null)
        s.timer = window.setTimeout(next, C.announcerGapMs)
      }, hold)
    }
    const next = () => {
      const p = s.pending
      s.pending = null
      if (p) show(p)
    }
    const off = fxBus.on(e => {
      if (e.kind !== 'announce') return
      const c: Call = { id: ++s.id, text: e.text, team: e.team, end: isEndText(e.text) }
      if (c.end || performance.now() >= s.busyUntil) { s.pending = null; show(c); return }
      if (!s.pending || s.pending.text !== c.text) s.pending = c   // newest wins the single slot
    })
    const offReset = onFxReset(() => { s.pending = null; s.busyUntil = 0; window.clearTimeout(s.timer); setCur(null) })
    return () => { off(); offReset(); window.clearTimeout(s.timer) }
  }, [])

  if (!cur) return null
  return (
    <div className="h3-announcer" key={cur.id}>
      <div className={`h3-announce-text h3-display ${cur.team ? `h3-${cur.team}` : 'h3-neutral'}${cur.end ? ' h3-announce-end' : ''}`}>
        {cur.text}
      </div>
    </div>
  )
}
