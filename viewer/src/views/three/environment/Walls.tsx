// Grimy brick walls (one textured plane each, the original soot-streaked brick) plus the original
// static graffiti: made-up words in generic letter styles (no real writers, brands or logos) and a
// couple of faded abstract pieces. Positions nudged so nothing hides behind the bodega or gate.
import { useMemo } from 'react'
import { CanvasTexture } from 'three'
import { retroMaterial } from '../materials/retroMaterial'
import { brickTexture, gateTexture } from './textures'
import { drawAbstractPiece, drawGraffiti, type GraffitiStyle } from '../graffiti/sprayText'
import { WALL_Z } from '../layout'

const SIDE_X = -11

type Piece = {
  key: string; pos: [number, number]; w: number; seed: number
  text?: string; color?: string; style?: GraffitiStyle   // lettered piece
  colors?: string[]                                     // abstract piece
  side?: boolean                                        // on the left side wall (pos[0] = z)
  fade?: number                                         // multiplies the paint (older = darker)
}

// Original b16b231 pieces + colours.
const PIECES: Piece[] = [
  { key: 'bughaus', text: 'BUGHAUS', color: '#c63bd8', style: 'throwup', pos: [-0.1, 2.3], w: 4.2, seed: 1 },
  { key: 'drpz', text: 'DRPZ', color: '#4fd16b', style: 'throwup', pos: [5.4, 1.6], w: 2.8, seed: 2 },
  { key: 'zugz', text: 'ZUGZ', color: '#ffd23a', style: 'tag', pos: [-2.5, 0.95], w: 1.6, seed: 3, fade: 0.85 },
  { key: 'knght', text: 'KNGHT', color: '#e23b3b', style: 'tag', pos: [2.9, 3.6], w: 1.8, seed: 4 },
  { key: 'tlsmn', text: 'TLSMN', color: '#f2f2f2', style: 'throwup', pos: [8.6, 2.6], w: 3.0, seed: 5, fade: 0.8 },
  { key: 'skrt', text: 'SKRT', color: '#3b8bff', style: 'tag', pos: [-8.9, 4.3], w: 1.7, seed: 6 },
  { key: 'oxo', text: 'OXO', color: '#ff6a00', style: 'tag', pos: [0.2, 4.6], w: 1.3, seed: 7 },
  { key: 'blob2', colors: ['#3a6a8a', '#7a8a3a', '#c8c0aa'], pos: [1.6, 2.2], w: 3.8, seed: 21, side: true, fade: 0.75 },
  { key: 'zonk', text: 'ZONK', color: '#d0561e', style: 'throwup', pos: [-2.4, 1.6], w: 2.6, seed: 9, side: true, fade: 0.8 },
]

function Decal({ p }: { p: Piece }) {
  const mat = useMemo(() => {
    const cnv = p.colors ? drawAbstractPiece(p.colors, p.seed, 512, 256) : drawGraffiti(p.text!, p.color!, p.seed, p.style, 512, 256)
    const f = p.fade ?? 0.92
    const m = retroMaterial({ map: new CanvasTexture(cnv), transparent: true, color: `rgb(${Math.round(f * 255)},${Math.round(f * 250)},${Math.round(f * 240)})` })
    m.depthWrite = false
    m.polygonOffset = true; m.polygonOffsetFactor = -1; m.polygonOffsetUnits = -1
    return m
  }, [p])
  const pos: [number, number, number] = p.side ? [SIDE_X + 0.01, p.pos[1], p.pos[0]] : [p.pos[0], p.pos[1], WALL_Z + 0.01]
  return (
    <mesh position={pos} rotation-y={p.side ? Math.PI / 2 : 0} material={mat} renderOrder={1}>
      <planeGeometry args={[p.w, p.w / 2]} />
    </mesh>
  )
}

// original tiling: 12 x 3 brick tiles on the 26 x 6 m back wall
const backWall = retroMaterial({ map: brickTexture(), repeat: [12, 3] })
const sideWall = retroMaterial({ map: brickTexture(), repeat: [6, 3], color: '#9a8f88' })
const coping = retroMaterial({ color: '#2e2c2a' })
const gate = retroMaterial({ map: gateTexture(), repeat: [1, 1.2], chrome: 0.15, color: '#b8bcc0' })

export function Walls() {
  return (
    <group>
      <mesh position={[0, 3, WALL_Z]} material={backWall}>
        <planeGeometry args={[26, 6]} />
      </mesh>
      <mesh position={[0, 6.05, WALL_Z + 0.05]} material={coping}>
        <boxGeometry args={[26, 0.12, 0.2]} />
      </mesh>
      {/* side wall, left */}
      <mesh position={[SIDE_X, 3, 0.5]} rotation-y={Math.PI / 2} material={sideWall}>
        <planeGeometry args={[13, 6]} />
      </mesh>
      {PIECES.map(p => <Decal key={p.key} p={p} />)}
      {/* a roll-down gate in the wall */}
      <mesh position={[-9.2, 1.6, WALL_Z + 0.02]} material={gate}>
        <planeGeometry args={[2.6, 3.2]} />
      </mesh>
    </group>
  )
}
