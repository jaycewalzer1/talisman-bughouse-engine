// SSX-style boost meter per team: fills with the team's edge (material across both boards and
// reserves, blended with clock edge). Smooth CSS-animated fill in team colours; "BOOST!" when hot.
import { useMemo } from 'react'
import { type Clocks, type TeamId, SEATS } from '../../../events/types'
import { displayClock, useGame } from '../../../store/gameStore'
import { STYLE } from '../style.config'
import { blendEdge, clockEdgeT1, fillFor, livePockets, teamMaterial } from './boost'

/** Material diff (T1 - T2); recomputed only when the boards/events change. */
function useMaterialDiff(): number {
  const fens = useGame(s => s.fens)
  const events = useGame(s => s.events)
  return useMemo(() => {
    const m = teamMaterial(fens, livePockets(fens, events))
    return m.T1 - m.T2
  }, [fens, events])
}

export function BoostMeter({ team }: { team: TeamId }) {
  const matDiff = useMaterialDiff()
  // quantized so the meter re-renders a few times a second, not every frame
  const clockMs = useGame(s => {
    if (!s.start) return 0
    const c = {} as Clocks
    for (const seat of SEATS) c[seat] = displayClock(s, seat)
    return Math.round(clockEdgeT1(c) / 250) * 250
  })
  const baseMs = useGame(s => s.start?.base_ms ?? 60_000)
  const H = STYLE.hud
  const edge = blendEdge(matDiff, clockMs, baseMs, H.boostClockWeight, H.boostMaterialScale)
  const fill = fillFor(team, edge)
  const mat = team === 'T1' ? matDiff : -matDiff
  const hot = fill > 0.72
  const side = team === 'T1' ? 'left' : 'right'
  return (
    <div className={`h3-boost h3-${team} h3-side-${side}${hot ? ' h3-hot' : ''}`} title="material + clock edge">
      <div className="h3-boost-track">
        <div className="h3-boost-fill" style={{ transform: `scaleX(${fill.toFixed(3)})` }} />
        <div className="h3-boost-ticks" />
      </div>
      <div className="h3-boost-label h3-display">
        {hot ? 'BOOST!' : 'BOOST'}<span className="h3-boost-mat">{mat > 0 ? `+${mat}` : mat}</span>
      </div>
    </div>
  )
}
