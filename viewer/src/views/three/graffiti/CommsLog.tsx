// Scrollable comms log overlay (HTML, not 3D). Status chatter is shown dimmed.
// Off by default: comic speech bubbles (comms/SpeechBubbles.tsx) replace it. Flip
// STYLE.comms.showLog in style.config.ts to bring the old log back.
import { useEffect, useRef } from 'react'
import { TEAM_OF } from '../../../events/types'
import { useGame } from '../../../store/gameStore'
import { TEAM_COLOR } from '../layout'
import { useView } from '../viewStore'
import { isWallWorthy } from './sprayText'
import { STYLE } from '../style.config'

const fmtTs = (ms: number) => { const s = Math.floor(ms / 1000); return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}` }

export function CommsLog() {
  if (!STYLE.comms.showLog) return null
  return <CommsLogPanel />
}

function CommsLogPanel() {
  const messages = useGame(s => s.messages)
  const names = useGame(s => s.start?.seats)
  const show = useView(s => s.showLog)
  const setShow = useView(s => s.setShowLog)
  const box = useRef<HTMLDivElement>(null)
  const recent = messages.slice(-150)

  useEffect(() => {
    const el = box.current
    if (el && el.scrollHeight - el.scrollTop - el.clientHeight < 80) el.scrollTop = el.scrollHeight
  }, [recent.length, messages])

  return (
    <div style={{
      position: 'absolute', right: 14, bottom: 14, width: 'min(330px, calc(100% - 28px))', pointerEvents: 'auto',
      transform: 'rotate(0.6deg)', fontFamily: '"Courier New", monospace', fontSize: 12,
    }}>
      <button onClick={() => setShow(!show)} style={{
        background: '#e9e2c8', color: '#111', border: 'none', padding: '4px 10px', fontWeight: 800, fontSize: 12,
        fontFamily: 'inherit', transform: 'rotate(-2deg)', boxShadow: '2px 2px 0 rgba(0,0,0,.6)', cursor: 'pointer', letterSpacing: 1,
      }}>COMMS {show ? '▾' : '▸'} {messages.length}</button>
      {show && (
        <div ref={box} style={{
          marginTop: 6, maxHeight: 210, overflowY: 'auto', background: 'rgba(12,12,14,0.82)', color: '#ddd',
          padding: '6px 8px', borderLeft: '4px solid #e9e2c8', boxShadow: '3px 3px 0 rgba(0,0,0,.5)',
        }}>
          {recent.length === 0 && <div style={{ opacity: 0.5 }}>no messages yet…</div>}
          {recent.map(m => {
            const dim = !isWallWorthy(m)
            const from = names?.[m.fromSeat]?.name ?? m.fromSeat
            return (
              <div key={m.seq} style={{ opacity: dim ? 0.38 : 1, padding: '1px 0', lineHeight: 1.35 }}>
                <span style={{ opacity: 0.55 }}>{fmtTs(m.ts)} </span>
                <span style={{ color: TEAM_COLOR[TEAM_OF[m.fromSeat]], fontWeight: 700 }}>{from}</span>
                <span style={{ opacity: 0.55 }}> → {m.toSeat.replace('_', ' ')} </span>
                <b>{m.token}</b>{m.text ? <span style={{ opacity: 0.7 }}> “{m.text}”</span> : null}
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}
