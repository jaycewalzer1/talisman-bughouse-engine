// Beat-up digital chess clocks taped to the tables. LCD text is a canvas texture redrawn only when
// the shown text changes. Under 10 s a side flickers red (and ticks, if sound is on).
import { useMemo } from 'react'
import { useFrame } from '@react-three/fiber'
import { CanvasTexture, SRGBColorSpace } from 'three'
import { seatToMove, type BoardId, type Seat } from '../../../events/types'
import { displayClock, useGame } from '../../../store/gameStore'
import { BOARD_ROT, BOARD_X, BOARD_Y, CLOCK_LOCAL, TEAM_COLOR } from '../layout'
import { TEAM_OF } from '../../../events/types'
import * as sfx from '../audio/sfx'

function fmt(ms: number): string {
  if (ms < 10000) return (Math.max(0, ms) / 1000).toFixed(1)
  const s = Math.ceil(ms / 1000)
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`
}

function Clock({ board }: { board: BoardId }) {
  const white = `${board}_white` as Seat, black = `${board}_black` as Seat
  const { canvas, tex } = useMemo(() => {
    const canvas = document.createElement('canvas'); canvas.width = 256; canvas.height = 96
    const tex = new CanvasTexture(canvas); tex.colorSpace = SRGBColorSpace
    return { canvas, tex }
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
    g.fillStyle = '#141414'; g.fillRect(0, 0, 256, 96)
    sides.forEach((x, i) => {
      const ox = i * 128
      g.fillStyle = x.red ? '#3a0606' : '#1e2a1c'; g.fillRect(ox + 8, 14, 112, 66)
      g.fillStyle = x.red ? '#ff2a2a' : x.running ? '#b8ff9a' : '#6e8f62'
      g.font = 'bold 40px monospace'; g.textAlign = 'center'; g.textBaseline = 'middle'
      g.fillText(x.txt, ox + 64, 50)
      g.fillStyle = TEAM_COLOR[TEAM_OF[x.seat]]; g.fillRect(ox + 8, 82, 112, 6)
      if (x.running) { g.fillStyle = '#ffd23a'; g.fillRect(ox + 52, 4, 24, 7) } // the pressed button
    })
    tex.needsUpdate = true
  })

  // white's side of the clock on the left as white sees it
  return (
    <group position={[BOARD_X[board], 0, 0]} rotation-y={BOARD_ROT[board]}>
      <group position={[CLOCK_LOCAL.x, BOARD_Y, CLOCK_LOCAL.z]}>
        <mesh position={[0, 0.045, 0]} castShadow>
          <boxGeometry args={[0.22, 0.09, 0.08]} />
          <meshStandardMaterial color="#2d2a26" roughness={0.6} />
        </mesh>
        {/* one face toward each player */}
        <mesh position={[0, 0.05, 0.041]}>
          <planeGeometry args={[0.2, 0.075]} />
          <meshBasicMaterial map={tex} toneMapped={false} />
        </mesh>
        <mesh position={[0, 0.05, -0.041]} rotation-y={Math.PI}>
          <planeGeometry args={[0.2, 0.075]} />
          <meshBasicMaterial map={tex} toneMapped={false} />
        </mesh>
        {[-0.07, 0.07].map(x => (
          <mesh key={x} position={[x, 0.095, 0]}>
            <boxGeometry args={[0.035, 0.012, 0.03]} />
            <meshStandardMaterial color="#cfcfcf" />
          </mesh>
        ))}
        {/* tape */}
        {[-0.118, 0.118].map(x => (
          <mesh key={x} position={[x, 0.04, 0]} rotation-z={x > 0 ? -0.12 : 0.12}>
            <boxGeometry args={[0.03, 0.1, 0.086]} />
            <meshStandardMaterial color="#b9b4a3" roughness={0.5} />
          </mesh>
        ))}
      </group>
    </group>
  )
}

export function Clocks() {
  return <><Clock board="A" /><Clock board="B" /></>
}
