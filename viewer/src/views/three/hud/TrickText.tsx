// SSX trick text: captures and drops pop in ("KNIGHT DROP x2!", "PAWN SNATCH!"), hold, fly out.
// Combo = consecutive tricks by the same team within STYLE.hud.comboWindowMs. Sits just above the
// bottom HUD strip, nudged toward the trick team's side, so it never sits on the boards for long.
import { useEffect, useRef, useState } from 'react'
import { type TeamId, TEAM_OF } from '../../../events/types'
import { onFxReset } from '../director'
import { fxBus } from '../fxBus'
import { STYLE } from '../style.config'
import { PIECE_VALUE, trickLabel } from '../comms/lines'

interface Trick { id: number; team: TeamId; text: string; combo: number; pts: number; clutch: boolean }

const LIFE_MS = 1500

export function TrickText() {
  const [items, setItems] = useState<Trick[]>([])
  const st = useRef({ id: 0, team: null as TeamId | null, at: 0, combo: 0 })

  useEffect(() => {
    const s = st.current
    const timers = new Set<number>()
    const off = fxBus.on(e => {
      if (e.kind !== 'capture' && e.kind !== 'drop') return
      const team = TEAM_OF[e.seat]
      const now = performance.now()
      s.combo = s.team === team && now - s.at <= STYLE.hud.comboWindowMs ? s.combo + 1 : 1
      s.team = team; s.at = now
      const t: Trick = {
        id: ++s.id, team, combo: s.combo,
        text: trickLabel(e.kind, e.piece),
        pts: e.kind === 'capture' ? PIECE_VALUE[e.piece.toUpperCase()] ?? 0 : 0,
        clutch: e.kind === 'drop' && e.blocksCheck,
      }
      setItems(prev => [...prev.slice(-1), t])     // newest + the one flying out
      const id = window.setTimeout(() => { timers.delete(id); setItems(prev => prev.filter(x => x.id !== t.id)) }, LIFE_MS)
      timers.add(id)
    })
    const offReset = onFxReset(() => { s.team = null; s.combo = 0; setItems([]) })
    return () => { off(); offReset(); timers.forEach(id => window.clearTimeout(id)) }
  }, [])

  return (
    <div className="h3-tricks">
      {items.map((t, i) => (
        <div
          key={t.id}
          className={`h3-trick h3-display h3-${t.team} h3-side-${t.team === 'T1' ? 'left' : 'right'}${i < items.length - 1 ? ' h3-trick-out' : ''}`}
        >
          {t.clutch && <span className="h3-trick-pre">CLUTCH </span>}
          {t.text}{t.combo > 1 ? <span className="h3-trick-combo"> x{t.combo}</span> : null}!
          {t.pts > 0 && <span className="h3-trick-pts">+{t.pts}</span>}
        </div>
      ))}
    </div>
  )
}
