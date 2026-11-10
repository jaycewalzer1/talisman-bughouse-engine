// The DIY spot's personality, night/grimy palette: bodega storefront with fluorescent light spill,
// weathered sticker-bombed flat rail, beat-up boombox (speakers thump), skateboards, spray cans,
// warm bulbs strung from the el (a couple dead). All original; no real brands. Layout keeps clear of the boards/seats in layout.ts.
import { useMemo, useRef } from 'react'
import { useFrame, type ThreeElements } from '@react-three/fiber'
import { CatmullRomCurve3, Color, type Group, Vector3 } from 'three'
import { RETRO, STYLE } from '../style.config'
import { retroMaterial } from '../materials/retroMaterial'
import { WALL_Z } from '../layout'
import {
  awningTexture, bodegaSignTexture, bodegaWindowTexture, boomboxTexture, crackedConcreteTexture, deckTexture, gripTexture, stickerTexture,
} from './textures'
import { BlobShadow, BlobShadows, LightPool } from './BlobShadow'
import { StaticInstances, type InstanceXform } from './StaticInstances'
import { TRACK_Y } from './Tracks'
import { BODEGA_X } from './lamps'

const GLOW = STYLE.scene.bulbGlow
const chrome = retroMaterial({ color: '#a8acb0', chrome: 0.6 })

// ---------------------------------------------------------------- bodega
const frame = retroMaterial({ color: '#2a2a2e', chrome: 0.1 })
const shopWindow = retroMaterial({ map: bodegaWindowTexture(), unlit: true, color: new Color(1.05, 1.08, 1.0) })
const doorGlass = retroMaterial({ color: '#000000', emissive: '#e8ecd0', emissiveIntensity: 0.95, unlit: true })
const sign = retroMaterial({ map: bodegaSignTexture(), unlit: true, color: new Color(1.2, 1.15, 1.1) })
const awning = retroMaterial({ map: awningTexture(), repeat: [10, 1] })
const neon = retroMaterial({ color: '#000000', emissive: '#ff4a3a', emissiveIntensity: GLOW, unlit: true, fog: false })

function Bodega() {
  const z = WALL_Z
  return (
    <group position={[BODEGA_X, 0, 0]}>
      <mesh position={[0, 1.5, z + 0.07]} material={frame}>
        <boxGeometry args={[3.9, 3.0, 0.14]} />
      </mesh>
      <mesh position={[-0.75, 1.35, z + 0.145]} material={shopWindow}>
        <planeGeometry args={[2.1, 1.7]} />
      </mesh>
      <mesh position={[1.15, 1.12, z + 0.145]} material={doorGlass}>
        <planeGeometry args={[0.9, 2.1]} />
      </mesh>
      <mesh position={[1.15, 1.05, z + 0.16]} material={chrome}>
        <boxGeometry args={[0.7, 0.05, 0.04]} />
      </mesh>
      {/* sign + awning */}
      <mesh position={[0, 2.95, z + 0.2]} material={sign}>
        <boxGeometry args={[3.9, 0.62, 0.12]} />
      </mesh>
      <mesh position={[0, 2.48, z + 0.55]} rotation-x={0.38} material={awning}>
        <boxGeometry args={[3.9, 0.04, 0.95]} />
      </mesh>
      {/* a little neon squiggle in the window (abstract, not lettering) */}
      {[[-1.35, 1.75, 0.5], [-1.0, 1.85, -0.5], [-0.65, 1.75, 0.5]].map(([x, y, r], i) => (
        <mesh key={i} position={[x, y, z + 0.17]} rotation-z={r} material={neon}>
          <boxGeometry args={[0.42, 0.035, 0.02]} />
        </mesh>
      ))}
      {/* the spill: fluorescent pool on the pavement + a fainter wide one + a red neon tint */}
      <LightPool color="#eef0d4" radius={1.9} stretch={1.35} intensity={0.6} position={[0, 0, z + 1.7]} />
      <LightPool color="#a8c8b8" radius={3.2} stretch={1.2} intensity={0.16} position={[0, 0, z + 2.4]} />
      <LightPool color="#ff4a3a" radius={0.9} stretch={1.6} intensity={0.18} position={[-1.0, 0, z + 0.7]} />
    </group>
  )
}

// ---------------------------------------------------------------- sticker-bombed flat rail
const RAIL_LEN = 3.2
const railBar = retroMaterial({ map: stickerTexture(), repeat: [RAIL_LEN / 0.14, 1] })

function Rail() {
  return (
    <group position={[-1.7, 0, 2.35]} rotation-y={0.12}>
      <mesh position={[0, 0.38, 0]} material={railBar}>
        <boxGeometry args={[RAIL_LEN, 0.07, 0.07]} />
      </mesh>
      {[-1.25, 1.25].map(x => (
        <group key={x} position={[x, 0, 0]}>
          <mesh position={[0, 0.18, 0]} material={chrome}>
            <boxGeometry args={[0.05, 0.36, 0.05]} />
          </mesh>
          <mesh position={[0, 0.01, 0]} material={chrome}>
            <boxGeometry args={[0.32, 0.02, 0.2]} />
          </mesh>
        </group>
      ))}
      <BlobShadow radius={0.28} stretch={6} opacity={0.4} />
    </group>
  )
}

// ---------------------------------------------------------------- boombox
const boxBody = retroMaterial({ map: boomboxTexture(), chrome: 0.25 })
const speakerCone = retroMaterial({ color: '#141416', chrome: 0.2 })
const eq = retroMaterial({ color: '#000000', emissive: '#ffb040', emissiveIntensity: GLOW * 0.7, unlit: true })

function Boombox() {
  const cones = useRef<Group[]>([])
  const bar = useRef<Group>(null)
  useFrame(({ clock }) => {
    const t = clock.elapsedTime
    const beat = Math.pow(Math.max(0, Math.sin(t * Math.PI * 2 * 1.5)), 8)      // ~90 bpm thump
    for (const c of cones.current) c?.scale.set(1 + beat * 0.18, 1, 1 + beat * 0.18)
    if (bar.current) bar.current.scale.x = 0.3 + 0.7 * Math.abs(Math.sin(t * 7.3) * Math.sin(t * 3.1))
  })
  return (
    <group position={[-3.3, 0.33, 1.6]} rotation-y={0.4}>
      <mesh position={[0, 0.13, 0]} material={boxBody}>
        <boxGeometry args={[0.52, 0.26, 0.16]} />
      </mesh>
      {/* chrome handle + antenna */}
      <mesh position={[0, 0.3, 0]} material={chrome}>
        <boxGeometry args={[0.36, 0.025, 0.03]} />
      </mesh>
      {[-0.17, 0.17].map(x => (
        <mesh key={x} position={[x, 0.27, 0]} material={chrome}>
          <boxGeometry args={[0.025, 0.06, 0.03]} />
        </mesh>
      ))}
      <mesh position={[0.2, 0.42, -0.04]} rotation-z={-0.5} material={chrome}>
        <cylinderGeometry args={[0.005, 0.005, 0.4, 4]} />
      </mesh>
      {[-0.16, 0.16].map((x, i) => (
        <group key={x} position={[x, 0.12, 0.081]} rotation-x={Math.PI / 2}>
          <mesh material={chrome}>
            <cylinderGeometry args={[0.085, 0.085, 0.012, 10]} />
          </mesh>
          <group ref={el => { if (el) cones.current[i] = el }}>
            <mesh position={[0, 0.008, 0]} material={speakerCone}>
              <cylinderGeometry args={[0.07, 0.07, 0.012, 10]} />
            </mesh>
          </group>
        </group>
      ))}
      <group ref={bar} position={[0, 0.215, 0.082]}>
        <mesh material={eq}>
          <boxGeometry args={[0.1, 0.018, 0.004]} />
        </mesh>
      </group>
    </group>
  )
}

// ---------------------------------------------------------------- skateboards
const grip = retroMaterial({ map: gripTexture(), repeat: [1, 1] })
const wheel = retroMaterial({ color: '#d8d0b8' })

function Skateboard({ seed, ...props }: { seed: number } & ThreeElements['group']) {
  const deck = useMemo(() => retroMaterial({ map: deckTexture(seed), chrome: 0.15 }), [seed])
  return (
    <group {...props}>
      <mesh position={[0, 0.085, 0]} rotation-y={Math.PI / 2} material={deck}>
        <boxGeometry args={[0.2, 0.016, 0.8]} />
      </mesh>
      <mesh position={[0, 0.0945, 0]} rotation-x={-Math.PI / 2} material={grip}>
        <planeGeometry args={[0.78, 0.19]} />
      </mesh>
      {[-0.27, 0.27].map(x => (
        <group key={x} position={[x, 0, 0]}>
          <mesh position={[0, 0.06, 0]} material={chrome}>
            <boxGeometry args={[0.05, 0.035, 0.17]} />
          </mesh>
          {[-0.085, 0.085].map(z => (
            <mesh key={z} position={[0, 0.03, z]} rotation-x={Math.PI / 2} material={wheel}>
              <cylinderGeometry args={[0.028, 0.028, 0.03, 8]} />
            </mesh>
          ))}
        </group>
      ))}
    </group>
  )
}

// ---------------------------------------------------------------- spray cans (chrome, tinted)
const canMat = retroMaterial({ chrome: 0.8 })
const capMat = retroMaterial({})
const CANS: InstanceXform[] = [
  { pos: [-2.9, 0.1, -3.55], color: '#8a3a5a' }, { pos: [-2.75, 0.1, -3.6], color: '#6a8a3a' },
  { pos: [-2.6, 0.04, -3.4], rot: [Math.PI / 2, 0, 0.6], color: '#3a5a8a' }, { pos: [2.2, 0.1, 2.9], color: '#b8963a' },
]
const CAPS: InstanceXform[] = CANS.filter(c => !c.rot).map(c => ({ pos: [c.pos[0], 0.215, c.pos[2]], color: c.color }))

// ---------------------------------------------------------------- warm bulbs on the el
function bulbString(x0: number, x1: number, z: number, sag: number, seed: number) {
  const pts = [new Vector3(x0, TRACK_Y - 0.95, z), new Vector3((x0 + x1) / 2, TRACK_Y - 0.95 - sag, z), new Vector3(x1, TRACK_Y - 0.95, z)]
  const curve = new CatmullRomCurve3(pts)
  const colors = ['#ffd28a', '#ffb860', '#ffe0b0', '#ffc070', '#ffd28a', '#ffb860', '#2a2418']   // last one: dead bulb
  const n = Math.round(Math.abs(x1 - x0) / 0.5)
  const bulbs: InstanceXform[] = []
  for (let i = 1; i < n; i++) {
    const p = curve.getPoint(i / n)
    bulbs.push({ pos: [p.x, p.y - 0.06, p.z], color: colors[(i + seed) % colors.length] })
  }
  return { curve, bulbs }
}
const bulbMat = retroMaterial({ unlit: true, color: new Color(GLOW * 0.8, GLOW * 0.8, GLOW * 0.8), fog: false })
const wireMat = retroMaterial({ color: '#121212' })

function StringLights() {
  // (retro: only the string by the wall; the other hangs right in front of the default camera)
  const strings = useMemo(() => RETRO ? [bulbString(-4.5, 4.5, -3.25, 0.7, 2)] : [bulbString(-4.5, 4.5, 2.25, 0.9, 0), bulbString(-4.5, 4.5, -3.25, 0.7, 2)], [])
  const bulbs = useMemo(() => strings.flatMap(s => s.bulbs), [strings])
  return (
    <group>
      {strings.map((s, i) => (
        <mesh key={i} material={wireMat}>
          <tubeGeometry args={[s.curve, 16, 0.008, 3, false]} />
        </mesh>
      ))}
      <StaticInstances items={bulbs} material={bulbMat}>
        <icosahedronGeometry args={[0.055, 0]} />
      </StaticInstances>
    </group>
  )
}

// ---------------------------------------------------------------- mid-2000s additions
/** A deck stood on its tail and leaned back against the wall, art side out. */
function LeaningDeck({ seed, x }: { seed: number; x: number }) {
  return (
    <group position={[x, 0.41, WALL_Z + 0.2 + 0.09]} rotation-x={-0.2}>
      <group rotation-y={-Math.PI / 2}>
        <group rotation-z={Math.PI / 2}>
          <Skateboard seed={seed} position={[0, -0.05, 0]} />
        </group>
      </group>
    </group>
  )
}

const ledgeMat = RETRO ? retroMaterial({ map: crackedConcreteTexture(), repeat: [3.2, 0.4] }) : null
const ledgeEdge = retroMaterial({ color: '#77736c', chrome: 0.18 })    // waxed, grinded steel edge
const stoolTop = retroMaterial({ color: '#7a2a24', chrome: 0.3 })
const stoolLeg = retroMaterial({ color: '#6c6c6a', chrome: 0.4 })

function Ledge({ x, len }: { x: number; len: number }) {
  return (
    <group position={[x, 0, -3.42]}>
      <mesh position={[0, 0.25, 0]} material={ledgeMat!}><boxGeometry args={[len, 0.5, 0.55]} /></mesh>
      <mesh position={[0, 0.505, 0.27]} material={ledgeEdge} userData={{ noOcclude: true }}><boxGeometry args={[len, 0.018, 0.018]} /></mesh>
      <BlobShadow radius={0.4} stretch={len / 0.8} opacity={0.5} />
    </group>
  )
}

function Stool({ position }: { position: [number, number, number] }) {
  return (
    <group position={position}>
      <mesh position={[0, 0.46, 0]} material={stoolTop}><cylinderGeometry args={[0.17, 0.17, 0.05, 10]} /></mesh>
      {[0, 2.09, 4.19].map(a => (
        <mesh key={a} position={[Math.cos(a) * 0.1, 0.22, Math.sin(a) * 0.1]} rotation={[Math.sin(a) * 0.14, 0, -Math.cos(a) * 0.14]} material={stoolLeg} userData={{ noOcclude: true }}>
          <cylinderGeometry args={[0.012, 0.012, 0.46, 5]} />
        </mesh>
      ))}
      <BlobShadow radius={0.22} opacity={0.5} />
    </group>
  )
}

function RetroExtras() {
  return (
    <>
      <Ledge x={1.6} len={4.4} />
      <LeaningDeck seed={3} x={-2.1} />
      <LeaningDeck seed={4} x={-1.72} />
      <LeaningDeck seed={5} x={-1.28} />
      <Stool position={[-3.1, 0, -1.25]} />
      <Stool position={[3.15, 0, 1.05]} />
      <BlobShadows opacity={0.4} items={[{ x: -2.1, z: -3.9, r: 0.2 }, { x: -1.72, z: -3.9, r: 0.2 }, { x: -1.28, z: -3.9, r: 0.2 }]} />
    </>
  )
}

export function SetDressing() {
  return (
    <group>
      <Bodega />
      <Rail />
      <Boombox />
      {RETRO && <RetroExtras />}
      <Skateboard seed={1} position={[-0.6, 0, 2.95]} rotation-y={0.5} />
      <Skateboard seed={2} position={[-3.95, 0.1, 1.05]} rotation={[Math.PI, 0.3, 0]} />
      <StaticInstances items={CANS} material={canMat}>
        <cylinderGeometry args={[0.035, 0.035, 0.2, 8]} />
      </StaticInstances>
      <StaticInstances items={CAPS} material={capMat}>
        <cylinderGeometry args={[0.025, 0.03, 0.03, 6]} />
      </StaticInstances>
      <BlobShadows opacity={0.45} items={[{ x: -2.75, z: -3.5, r: 0.25, stretch: 1.4 }, { x: -0.6, z: 2.95, r: 0.22, stretch: 2.3, rot: 0.5 }, { x: -3.95, z: 1.05, r: 0.2, stretch: 2.3, rot: 0.3 }]} />
      {STYLE.scene.stringLights && <StringLights />}
    </group>
  )
}
