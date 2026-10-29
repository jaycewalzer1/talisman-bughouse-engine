// The ground: the original wet asphalt (cracks, patches, puddles baked in), nearest-filtered and
// mip-mapped, running out into the fog, plus a glossy puddle overlay that reflects the lamps
// (wetMaterial.ts, a cheap stand-in for the original MeshReflectorMaterial) and the curb.
import { useFrame } from '@react-three/fiber'
import { STYLE } from '../style.config'
import { retroMaterial } from '../materials/retroMaterial'
import { asphaltTexture, puddleMaskTexture } from './textures'
import { wetMaterial } from './wetMaterial'
import { lampState } from './lamps'

const SIZE = 120, TILE = 8.8          // original: 5 x 4 tiles over 44 x 34 m
const REPEAT = SIZE / TILE
const ground = retroMaterial({ map: asphaltTexture(), repeat: [REPEAT, REPEAT], color: '#c4c8d0' })
const wet = wetMaterial(puddleMaskTexture(), REPEAT)
const curb = retroMaterial({ color: '#55534f' })
const curbPaint = retroMaterial({ color: '#8a7a3e' })   // faded yellow edge

function Puddles() {
  useFrame(({ clock }) => {
    wet.uniforms.time.value = clock.elapsedTime
    wet.uniforms.flicker.value = lampState.flicker
  })
  return (
    <mesh rotation-x={-Math.PI / 2} position-y={0.004} material={wet}>
      <planeGeometry args={[SIZE, SIZE]} />
    </mesh>
  )
}

export function Street() {
  return (
    <group>
      <mesh rotation-x={-Math.PI / 2} material={ground}>
        <planeGeometry args={[SIZE, SIZE]} />
      </mesh>
      {STYLE.world.wetGround && <Puddles />}
      {/* curb along the wall, faded painted edge */}
      <mesh position={[0, 0.08, -3.95]} material={curb}>
        <boxGeometry args={[30, 0.16, 0.5]} />
      </mesh>
      <mesh position={[0, 0.08, -3.695]} material={curbPaint}>
        <boxGeometry args={[30, 0.16, 0.012]} />
      </mesh>
    </group>
  )
}
