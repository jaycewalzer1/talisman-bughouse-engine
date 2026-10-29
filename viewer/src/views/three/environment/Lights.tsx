// Lighting: warm sodium streetlights + cool blue night fill, as in the original (b16b231), but
// faked PS2-style: the real lighting is the shared retroLight uniforms (sodium key from the work
// lamp overhead + cool blue hemisphere ambient, per vertex in retroMaterial). The lamps are
// emissive bulbs (bloom) with additive pools of light on the ground and wall plus a faint haze
// cone, and a cool blue rim pool from the far end of the block. A cheap unshadowed hemi + key pair
// mirrors retroLight for any stock three.js material still in the scene.
import { useMemo } from 'react'
import { useFrame } from '@react-three/fiber'
import { Color, PlaneGeometry } from 'three'
import { STYLE } from '../style.config'
import { retroLight, retroMaterial } from '../materials/retroMaterial'
import { LightPool } from './BlobShadow'
import { lightConeMaterial, lightPoolMaterial } from './decalMaterials'
import { SODIUM, STREET_LAMPS, WORK_LAMP, lampState } from './lamps'
import { WALL_Z } from '../layout'

const GLOW = STYLE.scene.bulbGlow
const CONE = STYLE.scene.lampConeStrength

const poleMat = retroMaterial({ color: '#3b4038', chrome: 0.25 })
const headMat = retroMaterial({ color: '#2e322c', chrome: 0.3 })
const wallPlane = new PlaneGeometry(1, 1)

function StreetLamp({ base, arm, flicker = false }: { base: [number, number, number]; arm: number; flicker?: boolean }) {
  const bulb = useMemo(() => retroMaterial({ color: '#000000', emissive: '#ffcf8a', emissiveIntensity: GLOW, unlit: true, fog: false }), [])
  const pool = useMemo(() => lightPoolMaterial(SODIUM, 0.6), [])
  const wallGlow = useMemo(() => lightPoolMaterial(SODIUM, 0.28), [])
  const cone = useMemo(() => lightConeMaterial(SODIUM, CONE), [])
  const bases = useMemo(() => [pool, wallGlow].map(m => (m.uniforms.diffuse.value as Color).clone()), [pool, wallGlow])
  const coneBase = useMemo(() => (cone.uniforms.color.value as Color).clone(), [cone])
  useFrame(({ clock }) => {
    if (!flicker) return
    const t = clock.elapsedTime
    const off = Math.sin(t * 13) > 0.97 || (Math.sin(t * 0.7) > 0.95 && Math.random() < 0.4)
    const k = off ? 0.1 : 1
    lampState.flicker = k
    bulb.uniforms.emissiveIntensity.value = off ? 0.2 : GLOW
    ;[pool, wallGlow].forEach((m, i) => (m.uniforms.diffuse.value as Color).copy(bases[i]).multiplyScalar(k))
    ;(cone.uniforms.color.value as Color).copy(coneBase).multiplyScalar(k)
  })
  const [x, y, z] = base
  // how far the lamp head is from the back wall: close lamps also wash the brick
  const wallDist = z - WALL_Z
  return (
    <group position={[x, 0, z]}>
      <mesh position={[0, y / 2, 0]} material={poleMat}>
        <cylinderGeometry args={[0.06, 0.09, y, 6]} />
      </mesh>
      <mesh position={[arm / 2, y, 0]} rotation-z={Math.PI / 2} material={poleMat}>
        <cylinderGeometry args={[0.04, 0.04, Math.abs(arm), 5]} />
      </mesh>
      <mesh position={[arm, y - 0.02, 0]} material={headMat}>
        <boxGeometry args={[0.55, 0.1, 0.28]} />
      </mesh>
      <mesh position={[arm, y - 0.08, 0]} material={bulb}>
        <boxGeometry args={[0.5, 0.05, 0.25]} />
      </mesh>
      <LightPool color={SODIUM} material={pool} position={[arm, 0, 0]} radius={2.8} />
      {STYLE.scene.lampCones && (
        <mesh position={[arm, (y - 0.1) / 2, 0]} material={cone} renderOrder={3}>
          <cylinderGeometry args={[0.25, 2.2, y - 0.1, 12, 1, true]} />
        </mesh>
      )}
      {STYLE.scene.lightPools && wallDist < 2 && (
        <mesh geometry={wallPlane} material={wallGlow} position={[arm, y * 0.55, -wallDist + 0.03]} scale={[5, y * 1.1, 1]} renderOrder={2} />
      )}
    </group>
  )
}

export function Lights() {
  const lamp = useMemo(() => retroMaterial({ color: '#000000', emissive: '#ffd9a0', emissiveIntensity: GLOW, unlit: true }), [])
  const workCone = useMemo(() => lightConeMaterial('#ffd9a0', CONE * 0.6), [])
  const keyPos = useMemo(() => retroLight.keyDir.value.clone().multiplyScalar(12).toArray(), [])
  const [wx, wy, wz] = WORK_LAMP
  return (
    <group>
      {/* fallback for stock materials (no shadows) */}
      <hemisphereLight args={[retroLight.skyColor.value, retroLight.groundColor.value, STYLE.material.ambient * 2]} />
      <directionalLight position={keyPos} intensity={STYLE.material.keyStrength * 2.2} color={retroLight.keyColor.value} />

      {/* work lamp bolted under the el, right over the boards */}
      <mesh position={[wx, wy + 0.11, wz]} material={headMat}>
        <cylinderGeometry args={[0.18, 0.3, 0.2, 8]} />
      </mesh>
      <mesh position={[wx, wy, wz]} material={lamp}>
        <cylinderGeometry args={[0.27, 0.27, 0.02, 8]} />
      </mesh>
      {STYLE.scene.lampCones && (
        <mesh position={[wx, wy / 2, wz - 0.5]} rotation-x={0.18} material={workCone} renderOrder={3}>
          <cylinderGeometry args={[0.28, 2.6, wy, 12, 1, true]} />
        </mesh>
      )}
      <LightPool color="#ffc880" radius={3.0} stretch={1.35} intensity={0.32} position={[0, 0, 0.4]} />
      {STREET_LAMPS.map((l, i) => <StreetLamp key={i} base={l.base} arm={l.arm} flicker={l.flicker} />)}
      {/* cold rim light from the far end of the block */}
      <LightPool color="#3d64ff" radius={4.5} stretch={1.8} intensity={0.13} position={[0, 0, 6.5]} />
    </group>
  )
}
