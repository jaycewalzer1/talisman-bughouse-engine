// Lighting: warm sodium streetlights + cool moonlit fill (cool shadows). Only ONE light casts
// shadows (the lamp over the boards) to stay cheap.
import { useEffect, useRef } from 'react'
import { useFrame } from '@react-three/fiber'
import type { Object3D, PointLight, SpotLight } from 'three'

const SODIUM = '#ffa24a'

function StreetLamp({ pos, arm = 1.2, flicker = false }: { pos: [number, number, number]; arm?: number; flicker?: boolean }) {
  const light = useRef<PointLight>(null)
  useFrame(({ clock }) => {
    if (!flicker || !light.current) return
    const t = clock.elapsedTime
    const off = Math.sin(t * 13) > 0.97 || (Math.sin(t * 0.7) > 0.95 && Math.random() < 0.4)
    light.current.intensity = off ? 2 : 22
  })
  const [x, y, z] = pos
  return (
    <group position={[x, 0, z]}>
      <mesh position={[0, y / 2, 0]}>
        <cylinderGeometry args={[0.06, 0.09, y, 6]} />
        <meshStandardMaterial color="#3b4038" metalness={0.6} roughness={0.6} />
      </mesh>
      <mesh position={[arm / 2, y, 0]} rotation-z={Math.PI / 2}>
        <cylinderGeometry args={[0.04, 0.04, arm, 5]} />
        <meshStandardMaterial color="#3b4038" metalness={0.6} roughness={0.6} />
      </mesh>
      <mesh position={[arm, y - 0.08, 0]}>
        <boxGeometry args={[0.5, 0.12, 0.25]} />
        <meshBasicMaterial color="#ffcf8a" toneMapped={false} />
      </mesh>
      <pointLight ref={light} position={[arm, y - 0.3, 0]} color={SODIUM} intensity={22} distance={16} decay={1.6} />
    </group>
  )
}

export function Lights() {
  const spot = useRef<SpotLight>(null)
  const target = useRef<Object3D>(null)
  useEffect(() => { if (spot.current && target.current) spot.current.target = target.current }, [])
  return (
    <group>
      <hemisphereLight args={['#2a3d6e', '#07070c', 0.7]} />
      <directionalLight position={[-6, 12, -4]} intensity={0.35} color="#7f9be0" />
      {/* work lamp bolted under the el, right over the boards: the only shadow caster */}
      <object3D ref={target} position={[0, 0.7, 0.2]} />
      <spotLight
        ref={spot} position={[0.3, 5.3, 1.4]} color={SODIUM} intensity={70} angle={0.75} penumbra={0.65}
        distance={14} decay={1.4} castShadow shadow-mapSize={[1024, 1024]} shadow-bias={-0.0005}
      />
      <mesh position={[0.3, 5.45, 1.4]}>
        <cylinderGeometry args={[0.18, 0.3, 0.2, 8]} />
        <meshBasicMaterial color="#ffd9a0" toneMapped={false} />
      </mesh>
      <StreetLamp pos={[-6.5, 4.6, 2.6]} />
      <StreetLamp pos={[5.6, 4.6, -3.2]} arm={-1.2} flicker />
      {/* cold rim light from the far end of the block */}
      <pointLight position={[0, 2.5, 7]} color="#3d64ff" intensity={6} distance={12} decay={1.5} />
    </group>
  )
}
