// Two plywood sheets on stacked milk crates (crates instanced), with a worn roll-up board on each.
import { useMemo } from 'react'
import { DoubleSide } from 'three'
import type { BoardId } from '../../../events/types'
import { BOARD_ROT, BOARD_X, BOARD_Y, CRATE, SQ, TABLE_D, TABLE_W, TABLE_Y, boardToWorld } from '../layout'
import { boardTexture, crateTexture, plywoodTexture } from '../environment/textures'
import { StaticInstances, type InstanceXform } from '../environment/StaticInstances'

const CRATE_COLORS = ['#b8262b', '#1f4fa8', '#2b8a3e', '#d9a21b', '#b8262b', '#3c3c3c']
const BOARDS: BoardId[] = ['A', 'B']

export function Tables() {
  const crates = useMemo(() => {
    const out: InstanceXform[] = []
    let i = 0
    for (const b of BOARDS) for (const sx of [-1, 1]) for (const sz of [-1, 1]) for (let lvl = 0; lvl < 2; lvl++) {
      const p = boardToWorld(b, sx * 0.52, CRATE / 2 + lvl * CRATE, sz * 0.24)
      out.push({ pos: [p.x, p.y, p.z], rot: [0, ((i * 37) % 10 - 5) * 0.012, 0], color: CRATE_COLORS[i++ % CRATE_COLORS.length] })
    }
    // loose crates lying around (one is a seat, one tipped over)
    out.push({ pos: [-3.3, CRATE / 2, 1.6], rot: [0, 0.4, 0], color: '#1f4fa8' })
    out.push({ pos: [3.4, CRATE / 2, -1.9], rot: [0, -0.3, 0], color: '#d9a21b' })
    out.push({ pos: [3.4, CRATE * 1.5, -1.9], rot: [0, 0.2, 0], color: '#b8262b' })
    out.push({ pos: [-2.6, CRATE / 2, -2.6], rot: [Math.PI / 2, 0.6, 0], color: '#2b8a3e' })
    return out
  }, [])

  return (
    <group>
      <StaticInstances items={crates} castShadow receiveShadow>
        <boxGeometry args={[CRATE, CRATE, CRATE]} />
        <meshStandardMaterial map={crateTexture()} alphaTest={0.5} side={DoubleSide} roughness={0.6} />
      </StaticInstances>
      {BOARDS.map(b => (
        <group key={b} position={[BOARD_X[b], 0, 0]} rotation-y={BOARD_ROT[b]}>
          <mesh position={[0, TABLE_Y + 0.0125, 0]} castShadow receiveShadow rotation-y={0.02}>
            <boxGeometry args={[TABLE_W, 0.025, TABLE_D]} />
            <meshStandardMaterial map={plywoodTexture()} roughness={0.9} />
          </mesh>
          <mesh position={[0, BOARD_Y - 0.002, 0]} rotation-x={-Math.PI / 2} receiveShadow>
            <planeGeometry args={[SQ * 8 * 512 / 448, SQ * 8 * 512 / 448]} />
            <meshStandardMaterial map={boardTexture()} roughness={0.75} />
          </mesh>
          {/* duct tape holding the board flat */}
          {[[-0.38, 0.38], [0.38, -0.38]].map(([x, z], i) => (
            <mesh key={i} position={[x, BOARD_Y, z]} rotation={[-Math.PI / 2, 0, 0.8]}>
              <planeGeometry args={[0.16, 0.045]} />
              <meshStandardMaterial color="#9a9d9f" roughness={0.5} metalness={0.2} />
            </mesh>
          ))}
        </group>
      ))}
    </group>
  )
}
