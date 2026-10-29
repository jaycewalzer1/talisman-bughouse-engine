// The elevated subway structure overhead: the original rusty dark-green columns and girders,
// cross ties, worn rails (repeats instanced). Fake blob shadows at the column feet.
import { useMemo } from 'react'
import { retroMaterial } from '../materials/retroMaterial'
import { rustTexture, stickerTexture } from './textures'
import { StaticInstances, type InstanceXform } from './StaticInstances'
import { BlobShadows } from './BlobShadow'

export const TRACK_Y = 6.2      // top of the deck the train runs on
export const TRACK_Z = -0.6     // centre line of the tracks
const COL_X = [-12, -4.5, 4.5, 12]
const COL_Z = [-3.4, 2.4]
const COL_H = TRACK_Y - 0.6

const columnMat = retroMaterial({ map: rustTexture(), repeat: [1, COL_H / 1.2] })
const columnFoot = retroMaterial({ map: stickerTexture(), repeat: [1, 1] })   // sticker-bombed bottom 1.4 m
const beamMat = retroMaterial({ map: rustTexture(), repeat: [6, 1] })
const girderMat = retroMaterial({ map: rustTexture(), repeat: [80 / 1.4, 1] })
const tieMat = retroMaterial({ color: '#2a2420' })
const railMat = retroMaterial({ color: '#6b6b6b', chrome: 0.5 })

export function Tracks() {
  const columns = useMemo<InstanceXform[]>(() => COL_X.flatMap(x => COL_Z.map(z => ({ pos: [x, COL_H / 2, z] as [number, number, number] }))), [])
  const feet = useMemo<InstanceXform[]>(() => COL_X.flatMap(x => COL_Z.map(z => ({ pos: [x, 0.7, z] as [number, number, number] }))), [])
  const ties = useMemo<InstanceXform[]>(() => Array.from({ length: 80 }, (_, i) => ({ pos: [-40 + i, TRACK_Y - 0.15, TRACK_Z] as [number, number, number] })), [])
  const crossBeams = useMemo<InstanceXform[]>(() => COL_X.map(x => ({ pos: [x, TRACK_Y - 0.55, TRACK_Z] as [number, number, number] })), [])
  const blobs = useMemo(() => columns.map(c => ({ x: c.pos[0], z: c.pos[2], r: 0.55 })), [columns])
  return (
    <group>
      <StaticInstances items={columns} material={columnMat}>
        <boxGeometry args={[0.45, COL_H, 0.45]} />
      </StaticInstances>
      <StaticInstances items={feet} material={columnFoot}>
        <boxGeometry args={[0.47, 1.4, 0.47]} />
      </StaticInstances>
      <BlobShadows items={blobs} opacity={0.45} />
      <StaticInstances items={crossBeams} material={beamMat}>
        <boxGeometry args={[0.5, 0.6, 6.6]} />
      </StaticInstances>
      {/* longitudinal girders */}
      {[-3.4, -1.6, 0.4, 2.4].map(z => (
        <mesh key={z} position={[0, TRACK_Y - 0.55, z]} material={girderMat}>
          <boxGeometry args={[80, 0.75, 0.3]} />
        </mesh>
      ))}
      <StaticInstances items={ties} material={tieMat}>
        <boxGeometry args={[0.22, 0.14, 5.4]} />
      </StaticInstances>
      {/* rails */}
      {[-1.9, -1.0, 0.0, 0.9].map(z => (
        <mesh key={z} position={[0, TRACK_Y, TRACK_Z + z + 0.45]} material={railMat}>
          <boxGeometry args={[80, 0.08, 0.07]} />
        </mesh>
      ))}
    </group>
  )
}
