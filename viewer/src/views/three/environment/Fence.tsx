// Chain-link fences: one alpha-tested plane per run + instanced posts.
import { useMemo } from 'react'
import { DoubleSide } from 'three'
import { chainLinkTexture } from './textures'
import { StaticInstances, type InstanceXform } from './StaticInstances'

const RUNS: { from: [number, number]; to: [number, number] }[] = [
  { from: [6.8, -4.1], to: [6.8, 3.6] },
  { from: [6.8, 3.6], to: [10, 3.6] },
  { from: [-10.9, 3.2], to: [-6.5, 5.2] },
]
const H = 2.8

export function Fence() {
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
  return (
    <group>
      {RUNS.map((r, i) => {
        const dx = r.to[0] - r.from[0], dz = r.to[1] - r.from[1], len = Math.hypot(dx, dz)
        const map = chainLinkTexture().clone(); map.repeat.set(len * 6, H * 6); map.needsUpdate = true
        return (
          <group key={i} position={[(r.from[0] + r.to[0]) / 2, 0, (r.from[1] + r.to[1]) / 2]} rotation-y={-Math.atan2(dz, dx)}>
            <mesh position={[0, H / 2, 0]}>
              <planeGeometry args={[len, H]} />
              <meshStandardMaterial map={map} alphaTest={0.5} side={DoubleSide} metalness={0.8} roughness={0.4} color="#9aa0a6" />
            </mesh>
            <mesh position={[0, H, 0]} rotation-z={Math.PI / 2}>
              <cylinderGeometry args={[0.03, 0.03, len, 6]} />
              <meshStandardMaterial color="#7d848a" metalness={0.8} roughness={0.5} />
            </mesh>
          </group>
        )
      })}
      <StaticInstances items={posts}>
        <cylinderGeometry args={[0.04, 0.04, H, 6]} />
        <meshStandardMaterial color="#7d848a" metalness={0.8} roughness={0.5} />
      </StaticInstances>
    </group>
  )
}
