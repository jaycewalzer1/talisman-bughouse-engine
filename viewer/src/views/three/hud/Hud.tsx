// Minimal HUD styled like stickers and tape (HTML overlay, bottom-left so the app shell's toggle
// on top stays clear): camera mode, sound, stall markers and the result banner.
import type { CSSProperties, ReactNode } from 'react'
import { TEAM_OF, type Seat } from '../../../events/types'
import { useGame } from '../../../store/gameStore'
import { TEAM_COLOR } from '../layout'
import { useView } from '../viewStore'
import { setSoundEnabled } from '../audio/sfx'

const sticker = (bg: string, rot: number, color = '#111'): CSSProperties => ({
  background: bg, color, border: 'none', padding: '6px 12px', fontWeight: 900, fontSize: 13, letterSpacing: 1,
  fontFamily: '"Arial Black", "Helvetica Neue", sans-serif', transform: `rotate(${rot}deg)`, cursor: 'pointer',
  boxShadow: '2px 3px 0 rgba(0,0,0,.65)', borderRadius: 3, textTransform: 'uppercase', pointerEvents: 'auto',
})

function Sticker({ onClick, bg, rot, children, color }: { onClick?: () => void; bg: string; rot: number; children: ReactNode; color?: string }) {
  return <button onClick={onClick} style={sticker(bg, rot, color)}>{children}</button>
}

export function Hud() {
  const mode = useView(s => s.cameraMode), setMode = useView(s => s.setCameraMode)
  const soundOn = useView(s => s.soundOn), setSound = useView(s => s.setSoundOn)
  const stalls = useGame(s => s.stalls)
  const end = useGame(s => s.end)
  const seats = useGame(s => s.start?.seats)
  const toggleSound = () => { const on = !soundOn; setSoundEnabled(on); setSound(on) }

  return (
    <>
      <div style={{ position: 'absolute', left: 14, bottom: 14, display: 'flex', gap: 10, alignItems: 'flex-end', flexWrap: 'wrap', pointerEvents: 'none', maxWidth: 'calc(100% - 380px)', minWidth: 200 }}>
        <Sticker bg={mode === 'broadcast' ? '#ffd23a' : '#e9e2c8'} rot={-3} onClick={() => setMode(mode === 'broadcast' ? 'orbit' : 'broadcast')}>
          {mode === 'broadcast' ? '📺 broadcast' : '🛹 free cam'}
        </Sticker>
        <Sticker bg={soundOn ? '#4fd16b' : '#3a3a3a'} color={soundOn ? '#111' : '#eee'} rot={2} onClick={toggleSound}>
          {soundOn ? '🔊 sound on' : '🔇 sound off'}
        </Sticker>
        {(Object.keys(stalls) as Seat[]).map((seat, i) => (
          <span key={seat} style={{ ...sticker('#e02424', i % 2 ? 2 : -2, '#fff'), cursor: 'default', borderLeft: `6px solid ${TEAM_COLOR[TEAM_OF[seat]]}` }}>
            ⏸ {seats?.[seat]?.name ?? seat} sitting{stalls[seat] && stalls[seat] !== 'stall' ? ` · ${stalls[seat]}` : ''}
          </span>
        ))}
      </div>
      {end && (
        <div style={{ position: 'absolute', left: '50%', top: '22%', transform: 'translateX(-50%) rotate(-3deg)', pointerEvents: 'none' }}>
          <div style={{
            ...sticker(end.result === 'draw' ? '#e9e2c8' : TEAM_COLOR[end.result as 'T1' | 'T2'], 0), cursor: 'default',
            fontSize: 'clamp(20px, 5vw, 44px)', padding: '10px 26px', boxShadow: '5px 6px 0 rgba(0,0,0,.7)',
          }}>
            {end.result === 'draw' ? 'DRAW' : `${end.result} WINS`}
            <div style={{ fontSize: 'clamp(11px, 2vw, 15px)', opacity: 0.8 }}>
              {end.reason}{end.board ? ` on board ${end.board}` : ''}{end.loserSeat ? ` · ${seats?.[end.loserSeat]?.name ?? end.loserSeat}` : ''}
            </div>
          </div>
        </div>
      )}
    </>
  )
}
