// Broadcast mode: auto-cuts to the board with the latest event (store.lastEventBoard), holding each
// shot a minimum time so fast replays don't strobe, with slow dollies and occasional wide shots.
// Orbit mode: drei OrbitControls, starting from wherever the broadcast camera was.
import { useEffect, useMemo, useRef } from 'react'
import { useFrame, useThree } from '@react-three/fiber'
import { OrbitControls } from '@react-three/drei'
import { PerspectiveCamera, Vector3 } from 'three'
import type { BoardId } from '../../../events/types'
import { useGame } from '../../../store/gameStore'
import { BOARD_X } from '../layout'
import { useView } from '../viewStore'

interface Shot { from: [number, number, number]; to: [number, number, number]; drift: [number, number, number]; fov: number }

function boardShots(b: BoardId): Shot[] {
  const x = BOARD_X[b], out = b === 'A' ? -1 : 1
  // Players stand on the z axis of each table (z = ±0.84), so every shot looks in from the side or from
  // high above, never through a player's back.
  return [
    { from: [x + 1.7 * out, 1.9, 1.5], to: [x, 0.75, 0], drift: [0.2 * out, 0, -0.5], fov: 40 },                    // 3/4 high, T2 side
    { from: [x + 2.4 * out, 1.3, 0.2], to: [x, 0.85, 0], drift: [0, 0.1, -0.5], fov: 40 },                            // side, low
    { from: [x + 1.6 * out, 2.1, -1.6], to: [x, 0.75, 0], drift: [0.2 * out, -0.1, 0.4], fov: 42 },                  // 3/4 high, T1 side
    { from: [x + 0.6 * out, 3.6, 0.2], to: [x, 0.7, 0], drift: [0.15 * out, -0.2, -0.1], fov: 34 },                     // top-down on the board
  ]
}
const WIDE: Shot[] = [
  { from: [0, 2.6, 6.8], to: [0, 1.6, -1.5], drift: [1.2, 0.2, -0.6], fov: 48 },
  { from: [4.6, 0.35, 4.6], to: [0, 1.7, -1], drift: [-0.6, 0, 0], fov: 72 },                                           // skate-vid fisheye from the curb
  { from: [-5.5, 4.2, 3.8], to: [0, 1.0, -0.5], drift: [0.6, -0.3, 0], fov: 50 },
]
const MIN_HOLD = 2.4, MAX_HOLD = 7

export function CameraRig() {
  const mode = useView(s => s.cameraMode)
  const camera = useThree(s => s.camera) as PerspectiveCamera
  const st = useRef({ shot: WIDE[0], cutAt: -99, board: null as BoardId | null, hold: 4, lastWide: 0 })
  const tmp = useMemo(() => ({ p: new Vector3(), t: new Vector3() }), [])

  // place the camera on mount
  useEffect(() => {
    const s = WIDE[0]
    camera.position.set(...s.from); camera.lookAt(...s.to); camera.fov = s.fov; camera.updateProjectionMatrix()
  }, [camera])

  useFrame(({ clock }) => {
    if (mode !== 'broadcast') return
    const t = clock.elapsedTime, s = st.current
    const board = useGame.getState().lastEventBoard
    const since = t - s.cutAt
    const boardChanged = board !== s.board && board !== null
    if ((boardChanged && since > MIN_HOLD) || since > s.hold) {
      const goWide = board === null || (t - s.lastWide > 18 && Math.random() < 0.35)
      const pool = goWide ? WIDE : boardShots(board!)
      let next = pool[Math.floor(Math.random() * pool.length)]
      if (next === s.shot) next = pool[(pool.indexOf(next) + 1) % pool.length]
      s.shot = next; s.cutAt = t; s.board = board; s.hold = goWide ? 4 : MIN_HOLD + Math.random() * (MAX_HOLD - MIN_HOLD)
      if (goWide) s.lastWide = t
    }
    const u = Math.min(1, (t - s.cutAt) / (s.hold + 2))
    const sh = s.shot
    tmp.p.set(sh.from[0] + sh.drift[0] * u, sh.from[1] + sh.drift[1] * u, sh.from[2] + sh.drift[2] * u)
    // handheld wobble
    tmp.p.x += Math.sin(t * 0.9) * 0.02; tmp.p.y += Math.sin(t * 1.3 + 1) * 0.015
    camera.position.copy(tmp.p)
    camera.lookAt(tmp.t.set(...sh.to))
    if (camera.fov !== sh.fov) { camera.fov = sh.fov; camera.updateProjectionMatrix() }
  })

  if (mode !== 'orbit') return null
  return (
    <OrbitControls
      makeDefault target={[0, 0.9, 0]} enableDamping dampingFactor={0.08}
      minDistance={1.2} maxDistance={16} maxPolarAngle={Math.PI / 2 - 0.04}
    />
  )
}
