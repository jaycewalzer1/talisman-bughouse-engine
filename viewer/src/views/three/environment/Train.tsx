// An el train that rolls past overhead every so often: rumble sound + world shake. Brushed steel
// cars with the original purple stripe, glowing windows (bloom) and a faded abstract whole-car piece;
// instead of a real light, a flickering additive light pool slides across the ground under it.
import { useMemo, useRef } from 'react'
import { useFrame } from '@react-three/fiber'
import { CanvasTexture, Color, type Group } from 'three'
import { TRACK_Y, TRACK_Z } from './Tracks'
import { fx } from '../viewStore'
import * as sfx from '../audio/sfx'
import { retroMaterial } from '../materials/retroMaterial'
import { drawAbstractPiece } from '../graffiti/sprayText'
import { LightPool } from './BlobShadow'
import { lightPoolMaterial } from './decalMaterials'

const CARS = 5, CAR_LEN = 9, SPEED = 24, START = -70, END = 70

const body = retroMaterial({ color: '#8d9399', chrome: 0.6 })
const roof = retroMaterial({ color: '#5a5e64', chrome: 0.3 })
const windows = retroMaterial({ color: '#000000', emissive: '#fff2c4', emissiveIntensity: 1.5, unlit: true })
const stripeA = retroMaterial({ color: '#9b2fd6' })
const stripeB = retroMaterial({ color: '#3a3c42' })

export function Train() {
  const ref = useRef<Group>(null)
  const st = useRef({ x: START, running: false, nextAt: 9, dir: 1 })
  const piece = useMemo(() => {
    const m = retroMaterial({ map: new CanvasTexture(drawAbstractPiece(['#4f9a5b', '#3b6bb0', '#a03aa8'], 77, 256, 128)), transparent: true, mipmaps: false, color: '#c8c8c8' })
    m.depthWrite = false; m.polygonOffset = true; m.polygonOffsetFactor = -1
    return m
  }, [])
  const pool = useMemo(() => lightPoolMaterial('#d8e4ff', 0.0), [])
  const poolRef = useRef<Group>(null)
  const poolColor = useMemo(() => new Color('#d8e4ff'), [])

  useFrame(({ clock }, dt) => {
    const s = st.current, t = clock.elapsedTime, g = ref.current
    if (!g) return
    if (!s.running && t > s.nextAt) {
      s.running = true; s.dir = Math.random() < 0.5 ? 1 : -1; s.x = s.dir > 0 ? START : END
      sfx.train((END - START + CARS * CAR_LEN) / SPEED)
    }
    if (s.running) {
      s.x += s.dir * SPEED * Math.min(dt, 0.1)
      const done = s.dir > 0 ? s.x > END + CARS * CAR_LEN : s.x < START - CARS * CAR_LEN
      if (done) { s.running = false; s.nextAt = t + 30 + Math.random() * 35 }
    }
    g.visible = s.running
    g.position.x = s.x
    g.scale.x = s.dir
    const center = s.x - s.dir * (CARS * CAR_LEN) / 2
    fx.trainNear = s.running ? Math.max(0, 1 - Math.abs(center) / 35) : 0
    fx.shake = Math.max(fx.shake, fx.trainNear * 0.55)
    // window light strobing down through the ties
    const k = fx.trainNear * (0.35 + Math.random() * 0.25)
    ;(pool.uniforms.diffuse.value as Color).copy(poolColor).multiplyScalar(k)
    if (poolRef.current) { poolRef.current.position.x = Math.max(-8, Math.min(8, center)); poolRef.current.visible = k > 0.01 }
  })

  return (
    <group>
      <group ref={ref} position={[START, TRACK_Y + 1.65, TRACK_Z]} visible={false}>
        {Array.from({ length: CARS }, (_, i) => (
          <group key={i} position={[-i * (CAR_LEN + 0.4) - CAR_LEN / 2, 0, 0]}>
            <mesh material={body}>
              <boxGeometry args={[CAR_LEN, 3, 2.9]} />
            </mesh>
            <mesh position={[0, 1.55, 0]} material={roof}>
              <boxGeometry args={[CAR_LEN * 0.98, 0.15, 2.6]} />
            </mesh>
            {[1.46, -1.46].map(z => (
              <group key={z}>
                <mesh position={[0, 0.35, z]} material={windows}>
                  <boxGeometry args={[CAR_LEN * 0.86, 0.7, 0.02]} />
                </mesh>
                <mesh position={[0, -0.55, z]} material={stripeA}>
                  <boxGeometry args={[CAR_LEN, 0.16, 0.03]} />
                </mesh>
                <mesh position={[0, -0.75, z]} material={stripeB}>
                  <boxGeometry args={[CAR_LEN, 0.08, 0.03]} />
                </mesh>
              </group>
            ))}
            {i % 2 === 1 && (
              <mesh position={[0, -0.25, 1.49]} material={piece} renderOrder={1}>
                <planeGeometry args={[CAR_LEN * 0.7, CAR_LEN * 0.35]} />
              </mesh>
            )}
          </group>
        ))}
      </group>
      <group ref={poolRef} position={[0, 0, TRACK_Z]} visible={false}>
        <LightPool color="#d8e4ff" material={pool} radius={2.6} stretch={3} />
      </group>
    </group>
  )
}
