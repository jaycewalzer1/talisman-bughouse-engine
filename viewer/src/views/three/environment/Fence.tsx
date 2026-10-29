// Chain-link fences: one alpha-cut plane per run (chunky 25 cm diamonds so links survive the
// low-res render) + chrome top rails + instanced posts + a few zip-tied stickers/plates.
import { useMemo } from 'react'
import { DoubleSide } from 'three'
import { retroMaterial } from '../materials/retroMaterial'
import { chainLinkTexture } from './textures'
import { StaticInstances, type InstanceXform } from './StaticInstances'
import { BlobShadows } from './BlobShadow'

const RUNS: { from: [number, number]; to: [number, number] }[] = [
  { from: [6.8, -4.1], to: [6.8, 3.6] },
  { from: [6.8, 3.6], to: [10, 3.6] },
  { from: [-10.9, 3.2], to: [-6.5, 5.2] },
]
const H = 2.8, CELL = 0.25

const steel = retroMaterial({ color: '#7d848a', chrome: 0.35 })

export function Fence() {
  const runs = useMemo(() => RUNS.map(r => {
    const dx = r.to[0] - r.from[0], dz = r.to[1] - r.from[1], len = Math.hypot(dx, dz)
    const mat = retroMaterial({
      map: chainLinkTexture(), repeat: [len / CELL, H / CELL], alphaTest: 0.5, mipmaps: false,
      side: DoubleSide, chrome: 0.25, color: '#9aa0a6',
    })
    return { r, len, rot: -Math.atan2(dz, dx), mat }
  }), [])
  const posts = useMemo(() => {
    const out: InstanceXform[] = []
    for (const r of RUNS) {
      const len = Math.hypot(r.to[0] - r.from[0], r.to[1] - r.from[1]), n = Math.ceil(len / 2.4)
      for (let i = 0; i <= n; i++) {
        const t = i / n
        out.push({ pos: [r.from[0] + (r.to[0] - r.from[0]) * t, H / 2, r.from[1] + (r.to[1] - r.from[1]) * t] })
      }
    }
    return out
  }, [])
  const blobs = useMemo(() => posts.map(p => ({ x: p.pos[0], z: p.pos[2], r: 0.16 })), [posts])
  return (
    <group>
      {runs.map(({ r, len, rot, mat }, i) => (
        <group key={i} position={[(r.from[0] + r.to[0]) / 2, 0, (r.from[1] + r.to[1]) / 2]} rotation-y={rot}>
          <mesh position={[0, H / 2, 0]} material={mat}>
            <planeGeometry args={[len, H]} />
          </mesh>
          <mesh position={[0, H, 0]} rotation-z={Math.PI / 2} material={steel}>
            <cylinderGeometry args={[0.035, 0.035, len, 6]} />
          </mesh>
        </group>
      ))}
      <StaticInstances items={posts} material={steel}>
        <cylinderGeometry args={[0.045, 0.045, H, 6]} />
      </StaticInstances>
      <BlobShadows items={blobs} opacity={0.4} />
    </group>
  )
}
