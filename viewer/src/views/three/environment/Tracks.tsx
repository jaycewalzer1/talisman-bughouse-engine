// The elevated subway structure overhead: rusty columns, girders, cross ties (all instanced).
import { useMemo } from 'react'
import { rustTexture } from './textures'
import { StaticInstances, type InstanceXform } from './StaticInstances'

export const TRACK_Y = 6.2      // top of the deck the train runs on
export const TRACK_Z = -0.6     // centre line of the tracks
const COL_X = [-12, -4.5, 4.5, 12]
const COL_Z = [-3.4, 2.4]

export function Tracks() {
  const rust = rustTexture()
  const columns = useMemo<InstanceXform[]>(() => COL_X.flatMap(x => COL_Z.map(z => ({ pos: [x, TRACK_Y / 2 - 0.3, z] as [number, number, number] }))), [])
  const ties = useMemo<InstanceXform[]>(() => Array.from({ length: 80 }, (_, i) => ({ pos: [-40 + i, TRACK_Y - 0.15, TRACK_Z] as [number, number, number] })), [])
  const crossBeams = useMemo<InstanceXform[]>(() => COL_X.map(x => ({ pos: [x, TRACK_Y - 0.55, TRACK_Z] as [number, number, number] })), [])
  return (
    <group>
      <StaticInstances items={columns} castShadow>
        <boxGeometry args={[0.45, TRACK_Y - 0.6, 0.45]} />
        <meshStandardMaterial map={rust} roughness={0.85} metalness={0.3} />
      </StaticInstances>
      <StaticInstances items={crossBeams}>
        <boxGeometry args={[0.5, 0.6, 6.6]} />
        <meshStandardMaterial map={rust} roughness={0.85} metalness={0.3} />
      </StaticInstances>
      {/* longitudinal girders */}
      {[-3.4, -1.6, 0.4, 2.4].map(z => (
        <mesh key={z} position={[0, TRACK_Y - 0.55, z]} castShadow>
          <boxGeometry args={[80, 0.75, 0.3]} />
          <meshStandardMaterial map={rust} roughness={0.85} metalness={0.3} />
        </mesh>
      ))}
      <StaticInstances items={ties}>
        <boxGeometry args={[0.22, 0.14, 5.4]} />
        <meshStandardMaterial color="#2a2420" roughness={1} />
      </StaticInstances>
      {/* rails */}
      {[-1.9, -1.0, 0.0, 0.9].map(z => (
        <mesh key={z} position={[0, TRACK_Y, TRACK_Z + z + 0.45]}>
          <boxGeometry args={[80, 0.08, 0.07]} />
          <meshStandardMaterial color="#6b6b6b" metalness={0.9} roughness={0.35} />
        </mesh>
      ))}
    </group>
  )
}
