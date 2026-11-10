// The ground: the original wet asphalt (cracks, patches, puddles baked in), nearest-filtered and
// mip-mapped, running out into the fog, plus a glossy puddle overlay that reflects the lamps
// (wetMaterial.ts, a cheap stand-in for the original MeshReflectorMaterial) and the curb.
import { useFrame } from '@react-three/fiber'
import { useMemo } from 'react'
import { BufferAttribute, BufferGeometry } from 'three'
import { RETRO, STYLE } from '../style.config'
import { retroMaterial } from '../materials/retroMaterial'
import { asphaltTexture, crackedConcreteTexture, puddleMaskTexture } from './textures'
import { wetMaterial } from './wetMaterial'
import { lampState } from './lamps'

const SIZE = 120, TILE = 8.8          // original: 5 x 4 tiles over 44 x 34 m
const REPEAT = SIZE / TILE
const ground = retroMaterial({ map: asphaltTexture(), repeat: RETRO ? [REPEAT * 2, REPEAT * 2] : [REPEAT, REPEAT], color: RETRO ? '#c4c6dc' : '#c4c8d0' })
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

/**
 * Retro ground: same plane, but with a vertex grid that is dense around the play area and coarse far
 * away, so baked lighting has vertices where the shadows are. UVs stay in the plane's 0..1.
 */
function groundGeometry(): BufferGeometry {
  const lines = (inner: number, step: number) => {
    const out: number[] = []
    for (let x = -inner; x <= inner + 1e-6; x += step) out.push(x)
    for (let x = inner * 1.5; x < SIZE / 2; x *= 1.5) { out.push(x); out.unshift(-x) }
    out.push(SIZE / 2); out.unshift(-SIZE / 2)
    return out
  }
  const xs = lines(12, 0.42), zs = lines(8, 0.42)
  const pos = new Float32Array(xs.length * zs.length * 3), nor = new Float32Array(xs.length * zs.length * 3), uv = new Float32Array(xs.length * zs.length * 2)
  zs.forEach((z, j) => xs.forEach((x, i) => {
    const k = j * xs.length + i
    pos.set([x, 0, z], k * 3); nor.set([0, 1, 0], k * 3); uv.set([x / SIZE + 0.5, 0.5 - z / SIZE], k * 2)
  }))
  const idx: number[] = []
  for (let j = 0; j < zs.length - 1; j++) for (let i = 0; i < xs.length - 1; i++) {
    const a = j * xs.length + i, b = a + 1, c = a + xs.length, d = c + 1
    idx.push(a, c, b, b, c, d)
  }
  const g = new BufferGeometry()
  g.setAttribute('position', new BufferAttribute(pos, 3)); g.setAttribute('normal', new BufferAttribute(nor, 3)); g.setAttribute('uv', new BufferAttribute(uv, 2))
  g.setIndex(idx)
  return g
}

const SIDEWALK = { x: 30, z: 3.1 }
const sidewalk = RETRO ? retroMaterial({ map: crackedConcreteTexture(), repeat: [SIDEWALK.x / 1.6, SIDEWALK.z / 1.6] }) : null

export function Street() {
  const groundGeo = useMemo(() => RETRO ? groundGeometry() : null, [])
  return (
    <group>
      {groundGeo
        ? <mesh geometry={groundGeo} material={ground} userData={{ noOcclude: true, noTess: true }} />
        : <mesh rotation-x={-Math.PI / 2} material={ground}><planeGeometry args={[SIZE, SIZE]} /></mesh>}
      {/* cracked sidewalk slabs between the curb and the wall's foot, running under the rail + ledge */}
      {sidewalk && (
        <mesh rotation-x={-Math.PI / 2} position={[0, 0.17, -3.95 + 0.25 - 0.26]} material={sidewalk} userData={{ noOcclude: true }}>
          <planeGeometry args={[SIDEWALK.x, 0.54]} />
        </mesh>
      )}
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
