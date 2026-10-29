// Arcade HUD (HTML overlay over the canvas). Layout keeps the middle of the screen — where the
// broadcast camera puts the boards — clear:
//   top-centre      announcer hype text (transient) / result banner (after game end)
//   left / right    comic speech bubbles + portrait cut-ins (T1 left edge, T2 right edge)
//   bottom corners  per-team boost meter + race clocks (T1 bottom-left, T2 bottom-right)
//   bottom centre   trick text (transient), camera hint (fades), camera/focus/sound buttons,
//                   "sitting" chips
// The whole HUD layer is pointer-events: none so the free camera can be dragged anywhere over it;
// only real buttons (.h3-btn) take the pointer.
// The app shell's sticker bar lives at the top-left, so nothing permanent goes there.
// Every piece can be switched off in style.config.ts (STYLE.comms / STYLE.hud).
import type { CSSProperties } from 'react'
import { type Seat, type TeamId, TEAM_OF } from '../../../events/types'
import { useGame } from '../../../store/gameStore'
import { useView } from '../viewStore'
import { setSoundEnabled } from '../audio/sfx'
import { STYLE } from '../style.config'
import { useMoments } from '../comms/moments'
import { Announcer } from '../comms/Announcer'
import { SpeechBubbles } from '../comms/SpeechBubbles'
import { END_FALLBACK, END_TEXT } from '../comms/lines'
import { TrickText } from './TrickText'
import { BoostMeter } from './BoostMeter'
import { RaceClock } from './RaceClock'
import './hud.css'

// Display fonts (free, OFL) — injected once as a <link> so it works regardless of CSS chunk order.
const FONT_HREF = 'https://fonts.googleapis.com/css2?family=Bowlby+One+SC&family=Russo+One&display=swap'
if (typeof document !== 'undefined' && !document.querySelector(`link[href="${FONT_HREF}"]`)) {
  const l = document.createElement('link'); l.rel = 'stylesheet'; l.href = FONT_HREF
  document.head.appendChild(l)
}

const TEAM_SEATS: Record<TeamId, Seat[]> = { T1: ['A_white', 'B_black'], T2: ['A_black', 'B_white'] }

function TeamPanel({ team }: { team: TeamId }) {
  const H = STYLE.hud
  if (!H.boostMeters && !H.raceClocks) return <div />
  return (
    <div className={`h3-team h3-${team} h3-side-${team === 'T1' ? 'left' : 'right'}`}>
      {H.boostMeters && <BoostMeter team={team} />}
      {H.raceClocks && (
        <div className="h3-clocks">
          {TEAM_SEATS[team].map(seat => <RaceClock key={seat} seat={seat} />)}
        </div>
      )}
    </div>
  )
}

/** One-line controls hint for the free camera; fades out after STYLE.camera.hintMs (CSS only). */
function CamHint() {
  const style = { '--h3-hint-delay': `${STYLE.camera.hintMs}ms` } as CSSProperties
  return (
    <div className="h3-hint" style={style}>
      DRAG to orbit · RIGHT-DRAG to pan · SCROLL to zoom · 1/2/0 focus
    </div>
  )
}

function Controls() {
  const mode = useView(s => s.cameraMode), setMode = useView(s => s.setCameraMode)
  const focus = useView(s => s.requestFocus)
  const soundOn = useView(s => s.soundOn), setSound = useView(s => s.setSoundOn)
  const stalls = useGame(s => s.stalls)
  const seats = useGame(s => s.start?.seats)
  const toggleSound = () => { const on = !soundOn; setSoundEnabled(on); setSound(on) }
  return (
    <div className="h3-center">
      <div className="h3-stalls">
        {(Object.keys(stalls) as Seat[]).map(seat => (
          <span key={seat} className={`h3-chip h3-${TEAM_OF[seat]}`}>
            ⏸ {seats?.[seat]?.name ?? seat} sitting{stalls[seat] && stalls[seat] !== 'stall' ? ` · ${stalls[seat]}` : ''}
          </span>
        ))}
      </div>
      {mode === 'free' && <CamHint />}
      <div className="h3-buttons">
        <button
          className={`h3-btn ${mode === 'auto' ? 'h3-btn-on' : ''}`}
          title={mode === 'auto' ? 'Auto director is driving: click to take the camera back' : 'Let the auto director cut between shots'}
          onClick={() => setMode(mode === 'auto' ? 'free' : 'auto')}
        >
          {mode === 'auto' ? '📺 AUTO CAM' : '🎥 FREE CAM'}
        </button>
        <button className="h3-btn h3-btn-sm" title="Focus board A (1)" onClick={() => focus('A')}>A</button>
        <button className="h3-btn h3-btn-sm" title="Focus board B (2)" onClick={() => focus('B')}>B</button>
        <button className="h3-btn h3-btn-sm" title="Both boards (0 / R)" onClick={() => focus('both')}>A+B</button>
        <button className={`h3-btn ${soundOn ? 'h3-btn-go' : 'h3-btn-off'}`} onClick={toggleSound}>
          {soundOn ? '🔊 SOUND ON' : '🔇 SOUND OFF'}
        </button>
      </div>
    </div>
  )
}

function ResultBanner() {
  const end = useGame(s => s.end)
  const seats = useGame(s => s.start?.seats)
  if (!end) return null
  const cls = end.result === 'draw' ? 'h3-neutral' : `h3-${end.result}`
  const winners = end.result === 'draw' ? null : TEAM_SEATS[end.result].map(s => seats?.[s]?.name ?? s).join(' + ')
  return (
    <div className="h3-result">
      <div className={`h3-result-card ${cls}`}>
        <div className="h3-result-title h3-display">{end.result === 'draw' ? 'DRAW' : `${end.result} WINS`}</div>
        <div className="h3-result-sub">
          {END_TEXT[end.reason] ?? END_FALLBACK}{end.board ? ` · board ${end.board}` : ''}
          {end.loserSeat ? ` · ${seats?.[end.loserSeat]?.name ?? end.loserSeat}` : ''}
          {winners ? <div>{winners}</div> : null}
        </div>
      </div>
    </div>
  )
}

export function Hud() {
  useMoments()
  const vars = {
    '--h3-t1': STYLE.palette.T1, '--h3-t2': STYLE.palette.T2, '--h3-ink': STYLE.palette.ink,
    '--h3-yellow': STYLE.palette.yellow, '--h3-magenta': STYLE.palette.magenta,
  } as CSSProperties
  return (
    <div className="h3-hud" style={vars}>
      {STYLE.comms.announcer && <Announcer />}
      <ResultBanner />
      {STYLE.comms.bubbles && <SpeechBubbles />}
      {STYLE.hud.trickText && <TrickText />}
      <div className="h3-bottom">
        <TeamPanel team="T1" />
        <Controls />
        <TeamPanel team="T2" />
      </div>
    </div>
  )
}
