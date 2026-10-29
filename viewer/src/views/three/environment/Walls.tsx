// Grimy brick walls (one textured plane each), the coping, the roll-down gate, and the graffiti:
// each wall's whole paint layer is one transparent canvas (graffiti/wallArt.ts, painted once at
// load) on a single decal plane just in front of the brick. All lettering spells made-up words;
// no real writers, crews, brands or logos.
import { CanvasTexture, LinearFilter, LinearMipmapLinearFilter, SRGBColorSpace, type ShaderMaterial } from 'three'
import { retroMaterial } from '../materials/retroMaterial'
import { brickTexture, gateTexture } from './textures'
import { backWallArt, sideWallArt } from '../graffiti/wallArt'
import { WALL_Z } from '../layout'
import { STYLE } from '../style.config'

const SIDE_X = -11

// original tiling: 12 x 3 brick tiles on the 26 x 6 m back wall
const backWall = retroMaterial({ map: brickTexture(), repeat: [12, 3] })
const sideWall = retroMaterial({ map: brickTexture(), repeat: [6, 3], color: '#9a8f88' })
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

export function Walls() {
  const art = STYLE.graffiti.enabled ? paintMaterials() : null
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
      {/* a roll-down gate in the wall */}
      <mesh position={[-9.2, 1.6, WALL_Z + 0.02]} material={gate}>
        <planeGeometry args={[2.6, 3.2]} />
      </mesh>
    </group>
  )
}
