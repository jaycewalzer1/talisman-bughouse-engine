// Street clutter: dumpster, trash cans, cones, a boombox. Cheap primitives, no shadows except the dumpster.
import { StaticInstances } from './StaticInstances'
import { rustTexture } from './textures'

export function Props() {
  return (
    <group>
      {/* dumpster */}
      <group position={[-6.2, 0, -3.1]} rotation-y={0.08}>
        <mesh position={[0, 0.7, 0]} castShadow receiveShadow>
          <boxGeometry args={[2.2, 1.3, 1.2]} />
          <meshStandardMaterial map={rustTexture()} color="#4f7a4a" roughness={0.8} metalness={0.4} />
        </mesh>
        <mesh position={[0, 1.4, -0.1]} rotation-x={-0.25}>
          <boxGeometry args={[2.25, 0.05, 1.3]} />
          <meshStandardMaterial color="#1d1d1d" roughness={0.7} />
        </mesh>
      </group>
      {/* trash bags */}
      <StaticInstances items={[
        { pos: [-4.7, 0.25, -3.4], scale: [1, 0.8, 1] }, { pos: [-4.2, 0.22, -3.0], scale: [0.9, 0.7, 1.1] }, { pos: [-4.5, 0.55, -3.2], scale: [0.8, 0.6, 0.8] },
      ]}>
        <icosahedronGeometry args={[0.35, 0]} />
        <meshStandardMaterial color="#0e0e10" roughness={0.25} metalness={0.2} />
      </StaticInstances>
      {/* cones */}
      <StaticInstances items={[{ pos: [3.9, 0.3, 2.4] }, { pos: [4.5, 0.3, 2.1] }, { pos: [2.8, 0.15, 3.1], rot: [Math.PI / 2, 0, 0.8] }]}>
        <coneGeometry args={[0.15, 0.6, 6]} />
        <meshStandardMaterial color="#ff5a14" roughness={0.6} />
      </StaticInstances>
      {/* trash can */}
      <mesh position={[5.3, 0.45, -3.3]} castShadow>
        <cylinderGeometry args={[0.3, 0.27, 0.9, 8]} />
        <meshStandardMaterial color="#3f4a3c" metalness={0.6} roughness={0.6} />
      </mesh>
      {/* boombox on a crate */}
      <group position={[-3.3, 0.33, 1.6]} rotation-y={0.4}>
        <mesh position={[0, 0.12, 0]}>
          <boxGeometry args={[0.5, 0.24, 0.16]} />
          <meshStandardMaterial color="#202024" roughness={0.5} metalness={0.3} />
        </mesh>
        {[-0.14, 0.14].map(x => (
          <mesh key={x} position={[x, 0.12, 0.081]} rotation-x={Math.PI / 2}>
            <cylinderGeometry args={[0.075, 0.075, 0.01, 10]} />
            <meshStandardMaterial color="#555" />
          </mesh>
        ))}
        <mesh position={[0, 0.2, 0.082]}>
          <boxGeometry args={[0.08, 0.03, 0.005]} />
          <meshBasicMaterial color="#47ff9a" toneMapped={false} />
        </mesh>
      </group>
    </group>
  )
}
