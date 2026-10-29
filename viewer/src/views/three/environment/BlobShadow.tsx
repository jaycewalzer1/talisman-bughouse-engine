// Fake ground decals instead of real shadows/lights (PS2 style):
//   <BlobShadow radius opacity />   soft dark blob under anything (characters agent may import it)
//   <BlobShadows items opacity />   many static blobs in one instanced draw
//   <LightPool color radius ... />  additive warm/cool pool of "light" on the ground
// All of them lie flat at y ~ 0.012 (pass `y` to put one on a table top or crate).
import { useMemo } from 'react'
import { PlaneGeometry, type ShaderMaterial } from 'three'
import type { ThreeElements } from '@react-three/fiber'
import { STYLE } from '../style.config'
import { retroMaterial } from '../materials/retroMaterial'
import { blobTexture } from './textures'
import { lightPoolMaterial } from './decalMaterials'
import { StaticInstances, type InstanceXform } from './StaticInstances'

const plane = new PlaneGeometry(1, 1).rotateX(-Math.PI / 2)
const shadowMats = new Map<number, ShaderMaterial>()

function shadowMat(opacity: number): ShaderMaterial {
  const key = Math.round(opacity * 100)
  let m = shadowMats.get(key)
  if (!m) {
    m = retroMaterial({ map: blobTexture(), color: STYLE.palette.ink, unlit: true, transparent: true, opacity, mipmaps: false })
    m.depthWrite = false
    m.polygonOffset = true; m.polygonOffsetFactor = -2; m.polygonOffsetUnits = -2
    shadowMats.set(key, m)
  }
  return m
}

type GroupProps = ThreeElements['group']

/** A soft dark blob on the ground. `radius` in metres; `stretch` makes it oval along x. */
export function BlobShadow({ radius = 0.4, opacity = 0.5, stretch = 1, y = 0.012, ...props }:
  { radius?: number; opacity?: number; stretch?: number; y?: number } & GroupProps) {
  if (!STYLE.world.blobShadows) return null
  return (
    <group {...props}>
      <mesh geometry={plane} material={shadowMat(opacity)} position-y={y} scale={[radius * 2 * stretch, 1, radius * 2]} renderOrder={1} />
    </group>
  )
}

export interface BlobItem { x: number; z: number; r: number; stretch?: number; rot?: number; y?: number }

/** Many static blob shadows in one draw call. */
export function BlobShadows({ items, opacity = 0.5 }: { items: BlobItem[]; opacity?: number }) {
  const xf = useMemo<InstanceXform[]>(() => items.map(b => ({
    pos: [b.x, b.y ?? 0.012, b.z], rot: [0, b.rot ?? 0, 0], scale: [b.r * 2 * (b.stretch ?? 1), 1, b.r * 2],
  })), [items])
  if (!STYLE.world.blobShadows || items.length === 0) return null
  return (
    <StaticInstances items={xf} material={shadowMat(opacity)} renderOrder={1}>
      <primitive object={plane} attach="geometry" />
    </StaticInstances>
  )
}

/** Additive pool of light on the ground (bodega spill, lamps, train windows). */
export function LightPool({ color, radius = 1.5, stretch = 1, intensity = 0.6, y = 0.014, material, ...props }:
  { color: string; radius?: number; stretch?: number; intensity?: number; y?: number; material?: ShaderMaterial } & GroupProps) {
  const mat = useMemo(() => material ?? lightPoolMaterial(color, intensity), [material, color, intensity])
  if (!STYLE.scene.lightPools) return null
  return (
    <group {...props}>
      <mesh geometry={plane} material={mat} position-y={y} scale={[radius * 2 * stretch, 1, radius * 2]} renderOrder={2} />
    </group>
  )
}
