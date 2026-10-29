// Rounded low-poly hand: palm + cuff, 4 two-segment fingers and a two-segment thumb, all lofted
// (characters/loft.ts) with smooth normals — a soft octagonal palm, 6-sided capsule-ish fingers with
// rounded tips. ~198 tris, built as 5 vertex-coloured meshes so a hand costs 5 draw calls with ONE
// shared material (e.g. the skaters' matte retro material), +5 ink-outline hulls (materials/outline.ts):
//   palm+cuff | proximal finger bank | distal finger bank | thumb base | thumb tip
// The four fingers curl together (a bank), which is all a low-poly hand needs; `spread`
// widens the bank. Fingerless-glove look: cuff, palm and first finger segments in `glove`,
// fingertips and thumb tip in `skin` (glove = null -> bare hands).
//
// Hand-local frame (matches ik.ts defaults): the wrist is the origin, fingers point along -y, the palm
// faces +z, +x is the character's left. side = -1 builds a RIGHT hand (thumb on -x, anatomical pose),
// side = +1 a LEFT hand. Attach the returned `root` under the arm's wrist joint.
//
//   const hand = buildHand({ side: -1, skin: '#c58c63', glove: '#ff8a1f', material })
//   wrist.add(hand.root)
//   hand.setGrip(0.75)              // 0 open .. 1 fist; ~0.7 wraps a chess piece
//   hand.socket                     // Object3D where a held piece's grab point sits
import { BufferGeometry, Color, Float32BufferAttribute, type Material, Mesh, Object3D } from 'three'
import { mergeGeometries } from 'three/examples/jsm/utils/BufferGeometryUtils.js'
import { loftGeo, type LoftOpts, type Ring } from '../characters/loft'
import { withOutline } from '../materials/outline'

/** Ink hull width for hands (thinner than the bodies' so curled fingers don't clot into a blob). */
const HAND_INK_WIDTH = 1.5

export interface HandPose {
  grip: number      // 0 open, 1 fist
  spread?: number   // 0 together, 1 splayed (open-hand reach)
  pinch?: number    // 0..1 thumb opposition (thumb swings across the palm)
}

export interface Hand {
  root: Object3D
  socket: Object3D
  side: 1 | -1
  setGrip(g: number): void
  setPose(p: HandPose): void
  readonly pose: Required<HandPose>
  dispose(): void
}

export interface HandOptions {
  side: 1 | -1
  skin: string
  glove?: string | null
  ring?: string | null     // a ring band on the index finger
  material: Material       // should support vertex colours (skaters' matte retro material)
  size?: number            // uniform scale, default 1 (palm ~8.5 cm wide)
  castShadow?: boolean
}

// dimensions (meters)
const PALM_W = 0.085, PALM_L = 0.07, PALM_T = 0.034
const PALM_Y = -0.012            // palm top (just below the cuff)
const FINGER_W = 0.0185, FINGER_T = 0.022
const SEG1 = 0.03, SEG2 = 0.028
const FINGER_X = [-0.0315, -0.0105, 0.0105, 0.0315]
const TIP_K = [0.82, 1.0, 1.06, 0.92]          // distal length per finger, from -x to +x (right hand: pinky..index)
const THUMB_W = 0.022, THUMB1 = 0.03, THUMB2 = 0.024

/** Where a held piece's grab point sits, hand-local (between curled fingers and the palm). */
export const SOCKET_LOCAL: [number, number, number] = [0, -0.098, 0.03]

const _c = new Color()
function paint(g: BufferGeometry, color: string): BufferGeometry {
  _c.set(color)
  const n = g.attributes.position.count, col = new Float32Array(n * 3)
  for (let i = 0; i < n; i++) { col[i * 3] = _c.r; col[i * 3 + 1] = _c.g; col[i * 3 + 2] = _c.b }
  g.setAttribute('color', new Float32BufferAttribute(col, 3))
  return g
}
const cloft = (rings: Ring[], o: LoftOpts, color: string, x = 0) => paint(loftGeo(rings, o).translate(x, 0, 0), color)
/** A finger / thumb segment hanging from its joint along -y: 6-sided tube, optional rounded tip. */
function segment(len: number, rx: number, rz: number, tip: boolean, color: string, x = 0) {
  const rings: Ring[] = tip
    ? [{ y: 0.004, rx, rz }, { y: -len + rx * 0.7, rx: rx * 0.94, rz: rz * 0.94 }]
    : [{ y: 0.005, rx, rz }, { y: -len - 0.004, rx: rx * 0.97, rz: rz * 0.97 }]
  return cloft(rings, { seg: 6, phase: 0, bottom: tip ? rx * 0.75 : false, smoothCaps: true }, color, x)
}
const merge = (gs: BufferGeometry[]) => {
  const m = gs.length === 1 ? gs[0] : mergeGeometries(gs, false)!
  if (m !== gs[0]) gs.forEach(g => g.dispose())
  m.computeBoundingSphere()
  return m
}

interface HandGeo { palm: BufferGeometry; prox: BufferGeometry; dist: BufferGeometry; t1: BufferGeometry; t2: BufferGeometry }
const geoCache = new Map<string, HandGeo>()

function handGeo(side: 1 | -1, skin: string, glove: string, ring: string | null): HandGeo {
  const key = `${side}|${skin}|${glove}|${ring}`
  let g = geoCache.get(key)
  if (g) return g
  // palm + cuff: soft octagonal slab, slightly cupped toward +z, rounded knuckle edge
  const palm = merge([cloft([
    { y: 0.012, rx: PALM_W * 0.43, rz: PALM_T * 0.6 },                         // cuff
    { y: -0.03, rx: PALM_W * 0.5, rz: PALM_T * 0.55, z: 0.001 },
    { y: PALM_Y - PALM_L + 0.008, rx: PALM_W * 0.49, rz: PALM_T * 0.47, z: 0.002 },
  ], { seg: 8, sq: 0.75, top: 0, bottom: 0.008, smoothCaps: true }, glove)])
  // finger banks hang from their joint (y = 0) along -y
  const prox = merge(FINGER_X.map((x, i) => {
    const index = side < 0 ? i === 3 : i === 0
    return segment(SEG1, FINGER_W / 2, FINGER_T / 2, false, index && ring ? ring : glove, x)
  }))
  const dist = merge(FINGER_X.map((x, i) => {
    const k = TIP_K[side < 0 ? i : 3 - i]
    return segment(SEG2 * k, FINGER_W * 0.46, FINGER_T * 0.45, true, skin, x)
  }))
  const t1 = segment(THUMB1, THUMB_W / 2, THUMB_W / 2, false, glove)
  const t2 = segment(THUMB2, THUMB_W * 0.45, THUMB_W * 0.45, true, skin)
  g = { palm, prox, dist, t1, t2 }
  geoCache.set(key, g)
  return g
}

export function buildHand(o: HandOptions): Hand {
  const side = o.side
  const G = handGeo(side, o.skin, o.glove ?? o.skin, o.ring ?? null)
  const root = new Object3D()
  root.name = side < 0 ? 'handR' : 'handL'
  root.scale.setScalar(o.size ?? 1)
  const mesh = (geo: BufferGeometry) => {
    const m = new Mesh(geo, o.material)
    m.castShadow = o.castShadow ?? false
    return withOutline(m, { width: HAND_INK_WIDTH })
  }

  root.add(mesh(G.palm))
  const j1 = new Object3D()                        // knuckles
  j1.position.set(0, PALM_Y - PALM_L + 0.002, 0)
  j1.add(mesh(G.prox))
  const j2 = new Object3D()                        // middle joints
  j2.position.set(0, -SEG1, 0)
  j2.add(mesh(G.dist))
  j1.add(j2); root.add(j1)

  // thumb: base on the palm's side, angled down/forward; t1 swings across the palm, t2 curls
  const tBase = new Object3D()
  tBase.position.set(side * (PALM_W / 2 - 0.004), PALM_Y - 0.018, PALM_T * 0.3)
  tBase.rotation.set(-0.5, 0, side * 0.55)
  const t1 = new Object3D(); t1.add(mesh(G.t1))
  const t2 = new Object3D(); t2.position.set(0, -THUMB1, 0); t2.add(mesh(G.t2))
  t1.add(t2); tBase.add(t1); root.add(tBase)

  const socket = new Object3D()
  socket.name = 'gripSocket'
  socket.position.set(...SOCKET_LOCAL)
  root.add(socket)

  const pose: Required<HandPose> = { grip: 0, spread: 0, pinch: 0 }
  const apply = () => {
    const g = Math.max(0, Math.min(1, pose.grip))
    const sp = Math.max(0, Math.min(1, pose.spread)) * (1 - g)
    const pi = Math.max(0, Math.min(1, pose.pinch))
    // curl toward the palm (+z) = negative x rotation
    j1.rotation.set(-(0.12 + 1.3 * g), 0, 0)
    j1.scale.set(1 + 0.3 * sp, 1, 1)
    j2.rotation.set(-(0.1 + 1.45 * g), 0, 0)
    // thumb: opposes (swings across the palm) with grip/pinch, tip curls with grip
    const opp = Math.max(g * 0.8, pi)
    t1.rotation.set(-0.9 * opp, 0, side * (0.25 * sp - 0.95 * opp))
    t2.rotation.set(-0.9 * g, 0, -side * 0.3 * g)
  }
  apply()

  return {
    root, socket, side, pose,
    setGrip(g: number) { if (g !== pose.grip) { pose.grip = g; apply() } },
    setPose(p: HandPose) {
      pose.grip = p.grip
      if (p.spread !== undefined) pose.spread = p.spread
      if (p.pinch !== undefined) pose.pinch = p.pinch
      apply()
    },
    dispose() { root.removeFromParent() },   // geometry is cached and shared; the material belongs to the caller
  }
}

/** Triangle count of one hand's own geometry (for budgets/tests; ink hulls share it and aren't counted). */
export function handTriangles(h: Hand): number {
  let n = 0
  h.root.traverse(o => {
    const m = o as Mesh
    if (m.isMesh && !m.userData.inkHull) {
      const g = m.geometry
      n += (g.index ? g.index.count : g.attributes.position.count) / 3
    }
  })
  return n
}

/** Shaded meshes of one hand (each also has an ink hull child = one more draw when outlines are on). */
export function handMeshes(h: Hand): number {
  let n = 0
  h.root.traverse(o => { if ((o as Mesh).isMesh && !o.userData.inkHull) n++ })
  return n
}
