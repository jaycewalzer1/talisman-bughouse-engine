// Brick walls (one textured plane each, with a bump map) plus original static graffiti decals.
import { useMemo } from 'react'
import { CanvasTexture, SRGBColorSpace } from 'three'
import { brickBump, brickTexture } from './textures'
import { drawGraffiti, type GraffitiStyle } from '../graffiti/sprayText'
import { WALL_Z } from '../layout'

// Made-up words only (no real writers' names or styles).
const PIECES: { text: string; color: string; style: GraffitiStyle; pos: [number, number]; w: number; seed: number }[] = [
  { text: 'BUGHAUS', color: '#c63bd8', style: 'throwup', pos: [-5.6, 2.1], w: 4.2, seed: 1 },
  { text: 'DRPZ', color: '#4fd16b', style: 'throwup', pos: [5.4, 1.6], w: 2.8, seed: 2 },
  { text: 'ZUGZ', color: '#ffd23a', style: 'tag', pos: [-2.6, 1.1], w: 1.6, seed: 3 },
  { text: 'KNGHT', color: '#e23b3b', style: 'tag', pos: [2.9, 3.4], w: 1.8, seed: 4 },
  { text: 'TLSMN', color: '#f2f2f2', style: 'throwup', pos: [8.6, 2.6], w: 3.0, seed: 5 },
  { text: 'SKRT', color: '#3b8bff', style: 'tag', pos: [-8.4, 1.0], w: 1.7, seed: 6 },
  { text: 'OXO', color: '#ff6a00', style: 'tag', pos: [0.2, 4.4], w: 1.3, seed: 7 },
]

function Decal({ text, color, style, pos, w, seed }: typeof PIECES[number]) {
  const map = useMemo(() => {
    const t = new CanvasTexture(drawGraffiti(text, color, seed, style)); t.colorSpace = SRGBColorSpace; return t
  }, [text, color, style, seed])
  return (
    <mesh position={[pos[0], pos[1], WALL_Z + 0.01]}>
      <planeGeometry args={[w, w / 2]} />
      <meshStandardMaterial map={map} transparent depthWrite={false} roughness={0.7} polygonOffset polygonOffsetFactor={-1} />
    </mesh>
  )
}

export function Walls() {
  const map = useMemo(() => { const t = brickTexture().clone(); t.repeat.set(12, 3); t.needsUpdate = true; return t }, [])
  const bump = useMemo(() => { const t = brickBump().clone(); t.repeat.set(12, 3); t.needsUpdate = true; return t }, [])
  const sideMap = useMemo(() => { const t = brickTexture().clone(); t.repeat.set(6, 3); t.needsUpdate = true; return t }, [])
  return (
    <group>
      <mesh position={[0, 3, WALL_Z]} receiveShadow>
        <planeGeometry args={[26, 6]} />
        <meshStandardMaterial map={map} bumpMap={bump} bumpScale={2} roughness={0.95} />
      </mesh>
      {/* side wall, left */}
      <mesh position={[-11, 3, 0.5]} rotation-y={Math.PI / 2}>
        <planeGeometry args={[13, 6]} />
        <meshStandardMaterial map={sideMap} bumpMap={bump} bumpScale={2} roughness={0.95} color="#9a8f88" />
      </mesh>
      {PIECES.map(p => <Decal key={p.text} {...p} />)}
      {/* a roll-down gate in the wall */}
      <mesh position={[-8.2, 1.6, WALL_Z + 0.005]}>
        <planeGeometry args={[3, 3.2, 1, 16]} />
        <meshStandardMaterial color="#5d6066" metalness={0.6} roughness={0.5} wireframe={false} />
      </mesh>
    </group>
  )
}
