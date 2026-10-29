// Beat-up digital chess clocks taped to the tables. LCD text is a canvas texture redrawn only when
// the shown text changes. Under 10 s a side flickers red (and ticks, if sound is on).
import { useMemo } from 'react'
import { useFrame } from '@react-three/fiber'
import { CanvasTexture, Color } from 'three'
import { retroMaterial } from '../materials/retroMaterial'
import { STYLE } from '../style.config'
import { seatToMove, type BoardId, type Seat } from '../../../events/types'
import { displayClock, useGame } from '../../../store/gameStore'
import { BOARD_ROT, BOARD_X, BOARD_Y, CLOCK_LOCAL, TEAM_COLOR } from '../layout'
import { TEAM_OF } from '../../../events/types'
import * as sfx from '../audio/sfx'

const bodyMat = retroMaterial({ color: '#2a2b2e', chrome: 0.2 })
const buttonMat = retroMaterial({ color: '#b8bcc0', chrome: 0.6 })
const tapeMat = retroMaterial({ color: '#9a9d9f', chrome: 0.2 })
const LW = 128, LH = 48     // small LCD canvas, nearest-filtered

function fmt(ms: number): string {
  if (ms < 10000) return (Math.max(0, ms) / 1000).toFixed(1)
  const s = Math.ceil(ms / 1000)
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`
}

function Clock({ board }: { board: BoardId }) {
  const white = `${board}_white` as Seat, black = `${board}_black` as Seat
  const { canvas, tex, faceMat } = useMemo(() => {
    const canvas = document.createElement('canvas'); canvas.width = LW; canvas.height = LH
    const tex = new CanvasTexture(canvas)
    // unlit and a touch over 1 so the LCD digits catch the bloom
    const faceMat = retroMaterial({ map: tex, unlit: true, mipmaps: false, color: new Color(1.35, 1.35, 1.35) })
    return { canvas, tex, faceMat }
  }, [])
  const st = useMemo(() => ({ key: '', lastSec: { [white]: -1, [black]: -1 } as Record<string, number> }), [white, black])

  useFrame(({ clock }) => {
    const s = useGame.getState()
    const toMove = seatToMove(board, s.fens[board])
    const flick = Math.sin(clock.elapsedTime * 22) > 0
    const sides = [white, black].map(seat => {
      const ms = displayClock(s, seat)
      const low = ms < 10000 && !!s.start
      const running = !s.end && !!s.start && toMove === seat
      if (low && running) {
        const sec = Math.floor(ms / 1000)
        if (st.lastSec[seat] !== sec) { st.lastSec[seat] = sec; sfx.tick(ms < 5000) }
      }
      return { seat, txt: fmt(ms), low, running, red: low && (flick || !running) }
    })
    const key = sides.map(x => `${x.txt}|${x.red}|${x.running}`).join('/')
    if (key === st.key) return
    st.key = key
    const g = canvas.getContext('2d')!
    g.fillStyle = STYLE.palette.ink; g.fillRect(0, 0, LW, LH)
    sides.forEach((x, i) => {
      const ox = i * 64
      g.fillStyle = x.red ? '#5a0612' : '#10261c'; g.fillRect(ox + 4, 7, 56, 33)
      g.fillStyle = x.red ? '#ff2a4a' : x.running ? STYLE.palette.lime : '#5e8f6a'
      g.font = 'bold 20px monospace'; g.textAlign = 'center'; g.textBaseline = 'middle'
      g.fillText(x.txt, ox + 32, 25)
      g.fillStyle = TEAM_COLOR[TEAM_OF[x.seat]]; g.fillRect(ox + 4, 41, 56, 3)
      if (x.running) { g.fillStyle = STYLE.palette.yellow; g.fillRect(ox + 26, 2, 12, 4) } // the pressed button
    })
    tex.needsUpdate = true
  })

  // white's side of the clock on the left as white sees it
  return (
    <group position={[BOARD_X[board], 0, 0]} rotation-y={BOARD_ROT[board]}>
      <group position={[CLOCK_LOCAL.x, BOARD_Y, CLOCK_LOCAL.z]}>
        <mesh position={[0, 0.045, 0]} material={bodyMat}>
          <boxGeometry args={[0.22, 0.09, 0.08]} />
        </mesh>
        {/* one face toward each player */}
        <mesh position={[0, 0.05, 0.041]} material={faceMat}>
          <planeGeometry args={[0.2, 0.075]} />
        </mesh>
        <mesh position={[0, 0.05, -0.041]} rotation-y={Math.PI} material={faceMat}>
          <planeGeometry args={[0.2, 0.075]} />
        </mesh>
        {[-0.07, 0.07].map(x => (
          <mesh key={x} position={[x, 0.095, 0]} material={buttonMat}>
            <boxGeometry args={[0.035, 0.012, 0.03]} />
          </mesh>
        ))}
        {/* tape */}
        {[-0.118, 0.118].map(x => (
          <mesh key={x} position={[x, 0.04, 0]} rotation-z={x > 0 ? -0.12 : 0.12} material={tapeMat}>
            <boxGeometry args={[0.03, 0.1, 0.086]} />
          </mesh>
        ))}
      </group>
    </group>
  )
}

export function Clocks() {
  return <><Clock board="A" /><Clock board="B" /></>
}
