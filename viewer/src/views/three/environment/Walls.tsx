// Grimy brick walls (one textured plane each), the coping, the roll-down gate, and the graffiti:
// each wall's whole paint layer is one transparent canvas (graffiti/wallArt.ts, painted once at
// load) on a single decal plane just in front of the brick. All lettering spells made-up words;
// no real writers, crews, brands or logos.
import { CanvasTexture, LinearFilter, LinearMipmapLinearFilter, SRGBColorSpace, type ShaderMaterial } from 'three'
import { useMemo } from 'react'
import { retroMaterial } from '../materials/retroMaterial'
import { brickTexture, gateTexture, wallGrimeTexture } from './textures'
import { backWallArt, sideWallArt } from '../graffiti/wallArt'
import { WALL_Z } from '../layout'
import { LOOK, RETRO, STYLE } from '../style.config'

const SIDE_X = -11
// Mid-2000s: the walls stop lower, so dusk sky and rooftops show between them and the el.
const WALL_H = RETRO ? 4.6 : 6

// original tiling: 12 x 3 brick tiles on the 26 x 6 m back wall
const backWall = retroMaterial({ map: brickTexture(), repeat: [12, 3 * WALL_H / 6] })
const sideWall = retroMaterial({ map: brickTexture(), repeat: [6, 3 * WALL_H / 6], color: '#9a8f88' })
const coping = retroMaterial({ color: '#2e2c2a' })
const gate = retroMaterial({ map: gateTexture(), repeat: [1, 1.2], chrome: 0.15, color: '#b8bcc0' })

let paint: [ShaderMaterial, ShaderMaterial] | null = null
function paintMaterials(): [ShaderMaterial, ShaderMaterial] {
  if (paint) return paint
  const make = (c: HTMLCanvasElement, tint: string) => {
    const map = new CanvasTexture(c)
    const m = retroMaterial({ map, transparent: true, color: tint })
    // smooth, mipmapped (cel look, no pixel crunch) — set after retroMaterial in case it overrides
    map.magFilter = LinearFilter; map.minFilter = LinearMipmapLinearFilter
    map.generateMipmaps = true; map.anisotropy = 8; map.colorSpace = SRGBColorSpace
    map.needsUpdate = true
    m.depthWrite = false
    m.polygonOffset = true; m.polygonOffsetFactor = -1; m.polygonOffsetUnits = -1
    return m
  }
  paint = [make(backWallArt(), STYLE.graffiti.paint), make(sideWallArt(), '#c8bfb4')]
  return paint
}

/** Retro: a wall's paint layer cut into <= 512 px tiles, each its own decal plane, with mortar lines
 *  showing through the paint so it sits on the brick rather than floating on it. */
function PaintTiles({ art, width, height, tint }: { art: HTMLCanvasElement; width: number; height: number; tint: string }) {
  // only the bottom of the art fits on the lower wall
  const artH = art.height * (height / 6), cropY = art.height - artH
  const tiles = useMemo(() => {
    const M = LOOK.maxTexture
    const nx = Math.ceil(art.width / M), ny = Math.ceil(artH / M)
    const brickRow = artH / (height / 0.125)
    const out: { x: number; y: number; w: number; h: number; mat: ShaderMaterial }[] = []
    for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
      const sx = (i / nx) * art.width, sy = cropY + (j / ny) * artH, sw = art.width / nx, sh = artH / ny
      const c = document.createElement('canvas'); c.width = Math.round(sw); c.height = Math.round(sh)
      const g = c.getContext('2d')!
      g.drawImage(art, -sx, -sy)
      g.globalCompositeOperation = 'destination-out'
      g.fillStyle = 'rgba(0,0,0,0.22)'
      for (let y = -((sy % brickRow)); y < c.height; y += brickRow) g.fillRect(0, y, c.width, 1)
      const map = new CanvasTexture(c)
      const mat = retroMaterial({ map, transparent: true, color: tint })
      map.colorSpace = SRGBColorSpace
      mat.depthWrite = false
      mat.polygonOffset = true; mat.polygonOffsetFactor = -1; mat.polygonOffsetUnits = -1
      out.push({ x: ((i + 0.5) / nx - 0.5) * width, y: (0.5 - (j + 0.5) / ny) * height, w: width / nx, h: height / ny, mat })
    }
    return out
  }, [art, width, height, tint, artH, cropY])
  return <>{tiles.map((t, i) => (
    <mesh key={i} position={[t.x, t.y, 0]} material={t.mat} renderOrder={1}><planeGeometry args={[t.w, t.h]} /></mesh>
  ))}</>
}

const grime = RETRO ? (() => {
  const m = retroMaterial({ map: wallGrimeTexture(), transparent: true, unlit: true, repeat: [3, 1] })
  m.depthWrite = false
  m.polygonOffset = true; m.polygonOffsetFactor = -2; m.polygonOffsetUnits = -2
  return m
})() : null

function RetroWalls() {
  const back = useMemo(() => backWallArt(), [])
  const side = useMemo(() => sideWallArt(), [])
  return (
    <>
      <group position={[0, WALL_H / 2, WALL_Z + 0.01]}><PaintTiles art={back} width={26} height={WALL_H} tint={STYLE.graffiti.paint} /></group>
      <group position={[SIDE_X + 0.01, WALL_H / 2, 0.5]} rotation-y={Math.PI / 2}><PaintTiles art={side} width={13} height={WALL_H} tint="#c8bfb4" /></group>
      <mesh position={[0, WALL_H / 2, WALL_Z + 0.02]} material={grime!} renderOrder={2}><planeGeometry args={[26, WALL_H]} /></mesh>
      <mesh position={[SIDE_X + 0.02, WALL_H / 2, 0.5]} rotation-y={Math.PI / 2} material={grime!} renderOrder={2}><planeGeometry args={[13, WALL_H]} /></mesh>
    </>
  )
}

export function Walls() {
  const art = STYLE.graffiti.enabled && !RETRO ? paintMaterials() : null
  return (
    <group>
      <mesh position={[0, WALL_H / 2, WALL_Z]} material={backWall}>
        <planeGeometry args={[26, WALL_H]} />
      </mesh>
      <mesh position={[0, WALL_H + 0.05, WALL_Z + 0.05]} material={coping}>
        <boxGeometry args={[26, 0.12, 0.2]} />
      </mesh>
      {/* side wall, left */}
      <mesh position={[SIDE_X, WALL_H / 2, 0.5]} rotation-y={Math.PI / 2} material={sideWall}>
        <planeGeometry args={[13, WALL_H]} />
      </mesh>
      {art && (
        <>
          <mesh position={[0, 3, WALL_Z + 0.01]} material={art[0]} renderOrder={1}>
            <planeGeometry args={[26, 6]} />
          </mesh>
          <mesh position={[SIDE_X + 0.01, 3, 0.5]} rotation-y={Math.PI / 2} material={art[1]} renderOrder={1}>
            <planeGeometry args={[13, 6]} />
          </mesh>
        </>
      )}
      {RETRO && STYLE.graffiti.enabled && <RetroWalls />}
      {/* a roll-down gate in the wall */}
      <mesh position={[-9.2, 1.6, WALL_Z + 0.02]} material={gate}>
        <planeGeometry args={[2.6, 3.2]} />
      </mesh>
    </group>
  )
}
