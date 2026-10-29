// A J/M/Z-style el train that rolls past overhead every so often: rumble sound + world shake +
// flickering window light spilling down through the ties.
import { useRef } from 'react'
import { useFrame } from '@react-three/fiber'
import type { Group, PointLight } from 'three'
import { TRACK_Y, TRACK_Z } from './Tracks'
import { fx } from '../viewStore'
import * as sfx from '../audio/sfx'

const CARS = 5, CAR_LEN = 9, SPEED = 24, START = -70, END = 70

export function Train() {
  const ref = useRef<Group>(null)
  const light = useRef<PointLight>(null)
  const st = useRef({ x: START, running: false, nextAt: 9, dir: 1 })

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
    if (light.current) light.current.intensity = fx.trainNear * (14 + Math.random() * 10)
  })

  return (
    <group>
      <group ref={ref} position={[START, TRACK_Y + 1.65, TRACK_Z]} visible={false}>
        {Array.from({ length: CARS }, (_, i) => (
          <group key={i} position={[-i * (CAR_LEN + 0.4) - CAR_LEN / 2, 0, 0]}>
            <mesh>
              <boxGeometry args={[CAR_LEN, 3, 2.9]} />
              <meshStandardMaterial color="#8d9399" metalness={0.8} roughness={0.35} />
            </mesh>
            {[1.46, -1.46].map(z => (
              <mesh key={z} position={[0, 0.3, z]}>
                <boxGeometry args={[CAR_LEN * 0.86, 0.75, 0.02]} />
                <meshBasicMaterial color="#fff2c4" toneMapped={false} />
              </mesh>
            ))}
            <mesh position={[0, -0.9, 1.47]}>
              <boxGeometry args={[CAR_LEN * 0.9, 0.12, 0.02]} />
              <meshBasicMaterial color="#9b2fd6" />
            </mesh>
          </group>
        ))}
      </group>
      <pointLight ref={light} position={[0, TRACK_Y - 1.2, TRACK_Z]} color="#d8e4ff" intensity={0} distance={18} decay={1.5} />
    </group>
  )
}
