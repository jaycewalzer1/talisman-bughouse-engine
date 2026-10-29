// Street clutter (original set, grimy palette): dumpster, trash bags, cones, trash can. Cheap
// primitives, retro materials, fake blob shadows. (The boombox moved to SetDressing.tsx.)
import { StaticInstances } from './StaticInstances'
import { retroMaterial } from '../materials/retroMaterial'
import { dumpsterTexture } from './textures'
import { BlobShadow, BlobShadows } from './BlobShadow'

const dumpster = retroMaterial({ map: dumpsterTexture(), repeat: [1.6, 1] })
const lid = retroMaterial({ color: '#1d1d1d', chrome: 0.1 })
const bags = retroMaterial({ color: '#0e0e10', chrome: 0.4 })      // glossy black plastic
const cone = retroMaterial({ color: '#d0561e' })
const coneBand = retroMaterial({ color: '#c8c4b8' })
const can = retroMaterial({ color: '#3f4a3c', chrome: 0.35 })

export function Props() {
  return (
    <group>
      {/* dumpster in front of the roll-down gate */}
      <group position={[-9.3, 0, -3.35]} rotation-y={0.08}>
        <mesh position={[0, 0.7, 0]} material={dumpster}>
          <boxGeometry args={[2.2, 1.3, 1.2]} />
        </mesh>
        <mesh position={[0, 1.4, -0.1]} rotation-x={-0.25} material={lid}>
          <boxGeometry args={[2.25, 0.05, 1.3]} />
        </mesh>
        <BlobShadow radius={0.9} stretch={1.5} opacity={0.55} />
      </group>
      {/* trash bags */}
      <StaticInstances material={bags} items={[
        { pos: [-7.7, 0.25, -3.5], scale: [1, 0.8, 1] }, { pos: [-7.25, 0.22, -3.1], scale: [0.9, 0.7, 1.1] }, { pos: [-7.5, 0.55, -3.3], scale: [0.8, 0.6, 0.8] },
      ]}>
        <icosahedronGeometry args={[0.35, 0]} />
      </StaticInstances>
      {/* cones */}
      <StaticInstances material={cone} items={[{ pos: [3.9, 0.3, 2.4] }, { pos: [4.5, 0.3, 2.1] }, { pos: [2.8, 0.15, 3.1], rot: [Math.PI / 2, 0, 0.8] }]}>
        <coneGeometry args={[0.15, 0.6, 6]} />
      </StaticInstances>
      <StaticInstances material={coneBand} items={[{ pos: [3.9, 0.36, 2.4] }, { pos: [4.5, 0.36, 2.1] }]}>
        <cylinderGeometry args={[0.053, 0.075, 0.09, 6]} />
      </StaticInstances>
      {/* trash can */}
      <mesh position={[5.3, 0.45, -3.3]} material={can}>
        <cylinderGeometry args={[0.3, 0.27, 0.9, 8]} />
      </mesh>
      <BlobShadows opacity={0.5} items={[
        { x: -7.5, z: -3.3, r: 0.7, stretch: 1.2 }, { x: 3.9, z: 2.4, r: 0.2 }, { x: 4.5, z: 2.1, r: 0.2 },
        { x: 2.8, z: 3.1, r: 0.3, stretch: 1.4, rot: 0.8 }, { x: 5.3, z: -3.3, r: 0.38 },
      ]} />
    </group>
  )
}
