// Skater part builders, "Downtown" (MTV, 1999) look: flat cel animation in 3D. Skinny torsos with
// narrow sloping shoulders, long thin necks and arms, slightly big heads, very baggy wide-leg pants
// that flare and bunch over big chunky sneakers. Every body part (one per joint) is built from
// rounded, lofted forms (loft.ts: stacked elliptical / rounded-rect sections with smooth normals,
// optional fold ripples), smooth spheres and tubes, and a few hard-edged boxes (soles, deck, trucks).
// Flat per-piece vertex colours. Faces are drawn, not sculpted: thin dark ink strokes (tubes that sit
// just above the head surface) for lids, brows, the side of the nose and the mouth, plus small eye
// whites and pupils. A few ink strokes also mark the folds on pants and sleeves.
//
// Each joint's pieces are merged into at most three geometries — one per "finish" (matte cloth/skin,
// glossy plastic/rubber, chrome metal) — plus one `ink` hull geometry (position + normal of the matte
// and gloss pieces, minus face strokes, linings and tiny bits) drawn with the inverted-hull outline
// material (materials/outline.ts). A whole skater is ~35 draw calls. All characters share the same
// three toon materials (colour lives in the vertices), so there are no per-character materials.
//
// Joint frames (rig units, 1 ≈ 1.8 m tall before spec.height scaling) — the joint contract, see
// ../grab/README.md; do not move these, only the geometry hung on them:
//   hips   at (0, HIP_Y, 0) in the rig root          spine at (0, 0.06, 0) in hips
//   head   at (0, 0.52, 0) in spine                  shL/shR at (±shoulderX, 0.46, 0) in spine
//   elL/R  at (0, -UPPER_ARM, 0) in shoulder         wrist at (0, -WRIST, 0) in elbow (hands: ../grab/hand.ts)
//   hipL/R at (±hipX, 0, 0) in hips                  knee at (0, -0.45, 0), ankle at (0, -0.43, 0)
//   deck   in the rig root, origin on the ground, nose toward +z
// Narrow shoulders are torso geometry: the torso slopes down from the neck to a small yoke that the
// (small) shoulder cap of the upper arm overlaps. The long neck is head geometry reaching down into
// the collar; the skull sits a little higher and bigger within the head joint's own frame.
// Soft joints: every child limb part starts with a sphere at its joint (shoulder cap, elbow, knee)
// that the parent's open tube end overlaps, so bends never open a gap.
import {
  BoxGeometry, BufferGeometry, CatmullRomCurve3, Color, CylinderGeometry, Euler, Float32BufferAttribute,
  Matrix4, Quaternion, SphereGeometry, TorusGeometry, TubeGeometry, Vector3, type ShaderMaterial,
} from 'three'
import { mergeGeometries } from 'three/examples/jsm/utils/BufferGeometryUtils.js'
import { retroMaterial } from '../materials/retroMaterial'
import { STYLE } from '../style.config'
import { flipFaces, frontZ, loftGeo, ringAt, ringPoint, type LoftOpts, type Ring } from './loft'
import type { CharacterSpec } from './rig'

export type V3 = [number, number, number]
export type Finish = 'matte' | 'gloss' | 'chrome'
export const FINISHES: Finish[] = ['matte', 'gloss', 'chrome']
/** One geometry per finish, plus the merged ink-hull geometry (position + normal only). */
export type PartGeo = Record<Finish, BufferGeometry | null> & { ink: BufferGeometry | null }
export type Joint =
  | 'hips' | 'spine' | 'head' | 'shL' | 'elL' | 'shR' | 'elR'
  | 'hipL' | 'knL' | 'anL' | 'hipR' | 'knR' | 'anR' | 'deck'
export const JOINTS: Joint[] = ['hips', 'spine', 'head', 'shL', 'elL', 'shR', 'elR', 'hipL', 'knL', 'anL', 'hipR', 'knR', 'anR', 'deck']

export const HIP_Y = 0.92
export const UPPER_ARM = 0.29
export const FOREARM = 0.31          // elbow -> centre of the (old) fist; kept for pose maths
export const WRIST = 0.262           // elbow -> wrist joint where the procedural hand attaches
export const shoulderX = (s: CharacterSpec) => 0.42 * s.bulk * 0.56 + 0.045
export const hipX = (s: CharacterSpec) => 0.1 + 0.02 * s.bag

// ---------------------------------------------------------------------------------------------
// Shared materials (one set for all four skaters)

let mats: Record<Finish, ShaderMaterial> | null = null
export function skaterMaterials(): Record<Finish, ShaderMaterial> {
  if (!mats) {
    mats = {
      matte: retroMaterial({ vertexColors: true }),
      gloss: retroMaterial({ vertexColors: true, chrome: 0.22 }),
      chrome: retroMaterial({ vertexColors: true, chrome: 1 }),
    }
  }
  return mats
}

// ---------------------------------------------------------------------------------------------
// Builder

interface PieceOpts {
  rot?: V3; scale?: V3; finish?: Finish; inside?: boolean
  /** Include in the ink hull (default: matte/gloss pieces that aren't linings). */
  ink?: boolean
}

const _m = new Matrix4(), _q = new Quaternion(), _e = new Euler(), _p = new Vector3(), _s = new Vector3()
const _c = new Color()

/** Lighten (k > 0) or darken (k < 0) a hex colour. */
export function shade(hex: string, k: number): string {
  _c.set(hex)
  if (k >= 0) _c.lerp(new Color('#ffffff'), k)
  else _c.multiplyScalar(1 + k)
  return '#' + _c.getHexString()
}

/** A point just above a lofted surface at height y, column angle th (0 = front, +π/2 = +x). */
function onSurf(rings: Ring[], y: number, th: number, off = 0.004, sq = 1, waves = 5, wavePh = 0): V3 {
  const r = ringAt(rings, y)
  const p = ringPoint(r, th, sq, waves, wavePh)
  const dx = p[0] - (r.x ?? 0), dz = p[2] - (r.z ?? 0), l = Math.hypot(dx, dz) || 1
  return [p[0] + (dx / l) * off, y, p[2] + (dz / l) * off]
}

interface SurfOpts { off?: number; sq?: number; waves?: number; wavePh?: number }

class Part {
  private pieces: Record<Finish, BufferGeometry[]> = { matte: [], gloss: [], chrome: [] }
  private inkPieces: BufferGeometry[] = []
  private frames: Matrix4[] = []

  /** Run `fn` with every piece it adds transformed by (at, rot, uniform scale). */
  frame(at: V3, rot: V3, fn: () => void, scale = 1) {
    _e.set(rot[0], rot[1], rot[2])
    this.frames.push(new Matrix4().compose(new Vector3(...at), new Quaternion().setFromEuler(_e), new Vector3(scale, scale, scale)))
    fn()
    this.frames.pop()
  }

  add(g: BufferGeometry, color: string, at: V3, o: PieceOpts = {}) {
    const geo = g.index ? g.toNonIndexed() : g
    if (geo !== g) g.dispose()
    geo.deleteAttribute('uv')
    if (o.inside) flipFaces(geo)
    const r = o.rot ?? [0, 0, 0], s = o.scale ?? [1, 1, 1]
    _e.set(r[0], r[1], r[2]); _q.setFromEuler(_e)
    _m.compose(_p.set(...at), _q, _s.set(...s))
    for (let i = this.frames.length - 1; i >= 0; i--) _m.premultiply(this.frames[i])
    geo.applyMatrix4(_m)
    _c.set(color)
    const n = geo.attributes.position.count, col = new Float32Array(n * 3)
    for (let i = 0; i < n; i++) { col[i * 3] = _c.r; col[i * 3 + 1] = _c.g; col[i * 3 + 2] = _c.b }
    geo.setAttribute('color', new Float32BufferAttribute(col, 3))
    const finish = o.finish ?? 'matte'
    this.pieces[finish].push(geo)
    if (o.ink ?? (finish !== 'chrome' && !o.inside)) {
      const shell = new BufferGeometry()
      shell.setAttribute('position', geo.attributes.position)
      shell.setAttribute('normal', geo.attributes.normal)
      this.inkPieces.push(shell)
    }
    return this
  }

  /** Box (hard edges: soles, deck, trucks, tiny decals); `taper` scales the top face (x, z). */
  box(w: number, h: number, d: number, color: string, at: V3, o: PieceOpts & { taper?: [number, number] } = {}) {
    let g: BufferGeometry = new BoxGeometry(w, h, d)
    if (o.taper) {
      const pos = g.attributes.position
      for (let i = 0; i < pos.count; i++) {
        if (pos.getY(i) > 0) pos.setXYZ(i, pos.getX(i) * o.taper[0], pos.getY(i), pos.getZ(i) * o.taper[1])
      }
      g = g.toNonIndexed(); g.computeVertexNormals()
    }
    return this.add(g, color, at, { ...o, ink: o.ink ?? Math.min(w, h, d) > 0.015 })
  }

  /** Lofted form (see loft.ts), smooth normals. `inside` here is the loft's own (lining) flip. */
  loft(rings: Ring[], color: string, o: LoftOpts & PieceOpts & { at?: V3 } = {}) {
    // open patches (arcSeg) and linings stay out of the ink hull
    const ink = o.ink ?? (!o.inside && !o.arcSeg)
    return this.add(loftGeo(rings, o), color, o.at ?? [0, 0, 0], { ...o, inside: false, ink })
  }

  /** Cylinder along y (smooth sides, flat caps). Use `scale` for oval sections. */
  cyl(rt: number, rb: number, h: number, seg: number, color: string, at: V3, o: PieceOpts & { open?: boolean } = {}) {
    return this.add(new CylinderGeometry(rt, rb, h, seg, 1, o.open ?? false, Math.PI / seg), color, at, o)
  }

  /** Smooth sphere / ellipsoid (radius 1 — size it with `scale`), or a slice of one. */
  ball(color: string, at: V3, scale: V3, o: PieceOpts & { w?: number; h?: number; half?: boolean; phi?: [number, number]; theta?: [number, number] } = {}) {
    const ph = o.phi ?? [0, Math.PI * 2], th = o.theta ?? [0, o.half ? Math.PI / 2 : Math.PI]
    const g = new SphereGeometry(1, o.w ?? 12, o.h ?? 8, ph[0], ph[1], th[0], th[1])
    return this.add(g, color, at, { ...o, scale })
  }

  /** Smooth tube through points (chains, cables, drawstrings, headphone band). Thin ones skip the hull. */
  tube(pts: V3[], r: number, color: string, o: PieceOpts & { n?: number; seg?: number } = {}) {
    const curve = new CatmullRomCurve3(pts.map(p => new Vector3(...p)))
    return this.add(new TubeGeometry(curve, o.n ?? pts.length * 2, r, o.seg ?? 4, false), color, [0, 0, 0], { ...o, ink: o.ink ?? r > 0.01 })
  }

  /** Torus lying in the xz plane (radius 1, tube `t`) — size it with `scale`. */
  ring(t: number, color: string, at: V3, scale: V3, o: PieceOpts & { rs?: number; ts?: number } = {}) {
    const g = new TorusGeometry(1, t, o.rs ?? 4, o.ts ?? 10).rotateX(Math.PI / 2)
    return this.add(g, color, at, { ...o, scale })
  }

  /** An ink line drawn on a lofted surface: (y, θ) control points, densified and lifted off it. */
  stroke(rings: Ring[], yt: [number, number][], r: number, o: SurfOpts & { color?: string; n?: number } = {}) {
    const pts: V3[] = []
    const sub = 3
    for (let i = 0; i < yt.length - 1; i++) {
      for (let k = 0; k < sub; k++) {
        const t = k / sub, [y0, a0] = yt[i], [y1, a1] = yt[i + 1]
        pts.push(onSurf(rings, y0 + (y1 - y0) * t, a0 + (a1 - a0) * t, o.off ?? 0.004, o.sq, o.waves, o.wavePh))
      }
    }
    const [yl, al] = yt[yt.length - 1]
    pts.push(onSurf(rings, yl, al, o.off ?? 0.004, o.sq, o.waves, o.wavePh))
    const curve = new CatmullRomCurve3(pts.map(p => new Vector3(...p)))
    return this.add(new TubeGeometry(curve, o.n ?? Math.max(4, yt.length * 3), r, 4, false), o.color ?? INK, [0, 0, 0], { ink: false })
  }

  build(): PartGeo {
    const out = { matte: null, gloss: null, chrome: null, ink: null } as PartGeo
    if (this.inkPieces.length && STYLE.toon.outlines) {
      const ink = mergeGeometries(this.inkPieces, false)
      if (ink) { ink.computeBoundingSphere(); out.ink = ink }
    }
    this.inkPieces.length = 0
    for (const f of FINISHES) {
      const list = this.pieces[f]
      if (!list.length) continue
      const merged = list.length === 1 ? list[0] : mergeGeometries(list, false)
      if (merged !== list[0]) list.forEach(g => g.dispose())
      if (merged) { merged.computeBoundingSphere(); out[f] = merged }
    }
    return out
  }
}

/** Rings scaled out from a body loft between y0 and y1 (for patches that sit on it). */
const band = (body: Ring[], y0: number, y1: number, k: number, n = 2): Ring[] =>
  Array.from({ length: n }, (_, i) => ringAt(body, y0 + ((y1 - y0) * i) / (n - 1), k))

// ---------------------------------------------------------------------------------------------
// The skater

export interface BuiltSkater {
  parts: Record<Joint, PartGeo>
  tris: number          // shaded triangles (excluding the ink hulls)
  inkTris: number       // ink hull triangles
  drawCalls: number     // including the ink hulls
}

const INK = STYLE.toon.inkColor
const SILVER = '#aeb0b4'
const GOLD = '#b8954a'
const EYE_WHITE = '#e4ddcc'

/** Head geometry scale and lift inside the head joint's frame (bigger head, longer neck). */
const HEAD_K = 1.1, HEAD_LIFT = 0.016

export function buildSkater(s: CharacterSpec): BuiltSkater {
  const B = s.bag
  const SX = shoulderX(s), HX = hipX(s)
  const K = 1 + 0.15 * (s.bulk - 1)          // torso width: skinny for everyone, a touch wider for MOOK
  const tR = 0.1 * B                           // thigh radius at the hip
  const parts = {} as Record<Joint, PartGeo>
  const tee = s.top === 'tee' || s.top === 'layered'
  const xxl = s.top === 'tee'

  // ---- hips: seat of the baggy pants (sits low), belt, chain wallet ---------------------------
  const PELVIS: Ring[] = [
    { y: -0.13, rx: HX + tR * 0.86, rz: 0.155 },
    { y: -0.04, rx: HX + tR * 0.8, rz: 0.15 },
    { y: 0.05, rx: 0.19 * K + 0.015 * (B - 1), rz: 0.13 },
    { y: 0.12, rx: 0.175 * K, rz: 0.12 },
  ]
  {
    const p = new Part()
    p.loft(PELVIS, s.pantsColor, { seg: 14, sq: 0.9, bottom: 0.05, top: 0, smoothCaps: true })
    p.loft(band(PELVIS, 0.06, 0.095, 1.025), shade(s.pantsColor, -0.35), { seg: 14, sq: 0.9 })   // belt line
    // a fold line under the crotch on each side
    for (const x of [-1, 1]) p.stroke(PELVIS, [[-0.02, x * 0.25], [-0.07, x * 0.15], [-0.1, x * 0.05]], 0.0035, { sq: 0.9 })
    if (s.chainWallet) {
      const x = -PELVIS[1].rx * 0.92
      p.tube([[x + 0.03, 0.07, 0.1], [x - 0.02, -0.06, 0.1], [x - 0.04, -0.18, 0.02], [x - 0.02, -0.15, -0.08], [x + 0.05, -0.02, -0.12]],
        0.007, SILVER, { finish: 'chrome', n: 10, seg: 3 })
      p.ball('#3a2a1c', [x + 0.08, -0.04, -0.12], [0.05, 0.06, 0.016], { w: 8, h: 5, rot: [0.15, -0.35, 0] })
    }
    parts.hips = p.build()
  }

  // ---- spine: skinny torso, sloping shoulders, collar, prints, necklace, headphones -------------
  {
    const p = new Part()
    const dk = shade(s.topColor, -0.25)
    const seg = 16, sq = 0.86
    const pelvisAt = (yHips: number) => ringAt(PELVIS, yHips).rx
    let T: Ring[]
    if (xxl) {          // XXL tee: hangs straight off the shoulder tips to the thighs, boxy, rippled hem
      const hem = Math.max(SX - 0.03, pelvisAt(-0.13) + 0.03)
      T = [
        { y: -0.24, rx: hem, rz: 0.165, w: 0.05 },
        { y: -0.1, rx: hem - 0.01, rz: 0.16, w: 0.035 },
        { y: 0.14, rx: SX - 0.05, rz: 0.15, z: 0.004, w: 0.015 },
        { y: 0.34, rx: SX - 0.04, rz: 0.142 },
        { y: 0.4, rx: SX - 0.045, rz: 0.12 },
        { y: 0.435, rx: SX - 0.1, rz: 0.095 },
        { y: 0.466, rx: 0.115, rz: 0.075 },
        { y: 0.482, rx: 0.08, rz: 0.062 },
      ]
    } else if (tee) {   // layered tee: hip length, narrow chest
      const hem = pelvisAt(-0.04) + 0.02
      T = [
        { y: -0.11, rx: hem, rz: 0.14, w: 0.03 },
        { y: 0.04, rx: 0.19 * K, rz: 0.125 },
        { y: 0.22, rx: 0.172 * K, rz: 0.118, z: 0.004 },
        { y: 0.34, rx: 0.178 * K, rz: 0.114 },
        { y: 0.4, rx: SX - 0.04, rz: 0.1 },
        { y: 0.432, rx: SX - 0.08, rz: 0.088 },
        { y: 0.462, rx: 0.115, rz: 0.072 },
        { y: 0.482, rx: 0.07, rz: 0.058 },
      ]
    } else {            // hoodie / zip: bunches at the banded hem, skinny chest, narrow sloping yoke
      const hem = pelvisAt(0.0) + 0.02
      T = [
        { y: -0.07, rx: hem * 0.96, rz: 0.135 },
        { y: -0.02, rx: hem, rz: 0.14, w: 0.045 },
        { y: 0.08, rx: 0.188 * K, rz: 0.128, w: 0.02 },
        { y: 0.22, rx: 0.168 * K, rz: 0.118, z: 0.004 },
        { y: 0.34, rx: 0.172 * K, rz: 0.114 },
        { y: 0.398, rx: SX - 0.04, rz: 0.102 },
        { y: 0.43, rx: SX - 0.078, rz: 0.088 },
        { y: 0.46, rx: 0.12, rz: 0.074 },
        { y: 0.478, rx: 0.085, rz: 0.064 },
        { y: 0.49, rx: 0.066, rz: 0.055 },
      ]
    }
    const L = { seg, sq, waves: 6 }
    p.loft(T, s.topColor, { ...L, bottom: 0, top: 0.004, smoothCaps: true })
    const fz = (y: number) => frontZ(ringAt(T, y), seg, sq)
    const neckTop = T[T.length - 1].y
    if (tee) {
      p.loft([{ y: neckTop - 0.02, rx: 0.088, rz: 0.07 }, { y: neckTop + 0.006, rx: 0.074, rz: 0.06 }], s.trim, { seg: 14 })   // collar
      p.loft(band(T, 0.18, 0.3, 1.012), s.trim, { ...L, arcSeg: [-2, 1] })                    // chest print
      p.loft(band(T, 0.14, 0.162, 1.012), s.accent, { ...L, arcSeg: [-2, 1] })
      p.box(0.11, 0.04, 0.006, INK, [0, 0.24, fz(0.24) + 0.004])
      // drape folds hanging from the shoulders
      for (const x of [-1, 1]) {
        p.stroke(T, [[0.33, x * 0.95], [0.18, x * 0.8], [0.02, x * 0.7]], 0.0035, { sq, waves: 6 })
        if (xxl) p.stroke(T, [[-0.06, x * 0.42], [-0.2, x * 0.32]], 0.0035, { sq, waves: 6 })
      }
    } else {
      p.loft(band(T, -0.075, -0.025, 1.03), s.top === 'zip' ? dk : s.trim, L)                  // ribbed hem
      if (s.top === 'hoodie') {
        p.loft(band(T, 0.0, 0.14, 1.025, 3), dk, { ...L, arcSeg: [-3, 2] })                     // kangaroo pocket
        for (const x of [-1, 1]) {
          const z = fz(0.44)
          p.tube([[x * 0.035, 0.475, z - 0.02], [x * 0.04, 0.41, z + 0.008], [x * 0.045, 0.33, z + 0.006]], 0.006, s.trim, { n: 4, seg: 4 })
        }
      } else {
        p.loft(band(T, -0.075, 0.474, 1.012, 6), s.trim, { ...L, arcSeg: [-1, 0] })            // zip placket
        p.box(0.02, 0.04, 0.012, SILVER, [0, 0.33, fz(0.33) + 0.012], { finish: 'chrome' })
        p.loft(band(T, 0.0, 0.12, 1.02), dk, { ...L, arcSeg: [-4, -3] })                       // pockets
        p.loft(band(T, 0.0, 0.12, 1.02), dk, { ...L, arcSeg: [2, 3] })
      }
      // bunching folds over the hem
      for (const x of [-1, 1]) p.stroke(T, [[0.12, x * 0.85], [0.04, x * 0.6], [0.0, x * 0.42]], 0.0035, { sq, waves: 6 })
      if (s.hat !== 'hood') p.ball(s.topColor, [0, 0.452, -0.085], [0.13, 0.065, 0.085], { w: 14, h: 8 }) // hood down
    }
    if (s.hat === 'hood') p.ball(s.hatColor, [0, 0.47, -0.07], [0.115, 0.07, 0.085], { w: 14, h: 8 })   // hood up: drape
    if (s.necklace) {
      const c = s.necklace === 'gold' ? GOLD : SILVER
      const z = fz(0.38), top = neckTop - 0.01
      p.tube([[-0.07, top, z - 0.04], [-0.05, 0.42, z + 0.002], [0, 0.375, z + 0.01], [0.05, 0.42, z + 0.002], [0.07, top, z - 0.04]],
        0.0065, c, { finish: 'chrome', n: 10, seg: 3 })
      p.box(0.04, 0.04, 0.014, c, [0, 0.35, z + 0.016], { finish: 'chrome', rot: [0, 0, Math.PI / 4], ink: false })
    }
    if (s.headphones) {
      const bandC = '#26242a'
      p.tube([[-0.082, 0.472, 0.06], [-0.1, 0.5, -0.02], [0, 0.522, -0.085], [0.1, 0.5, -0.02], [0.082, 0.472, 0.06]], 0.011, bandC, { finish: 'gloss', n: 12, seg: 6 })
      for (const x of [-1, 1]) {
        const at: V3 = [x * 0.088, 0.468, 0.072], rot: V3 = [0.55, 0, -x * 0.95]
        p.cyl(0.04, 0.042, 0.032, 14, s.accent, at, { rot, finish: 'gloss' })
        p.cyl(0.025, 0.025, 0.036, 10, shade(s.accent, -0.45), at, { rot, ink: false })
      }
    }
    parts.spine = p.build()
  }

  // ---- head: long thin neck, bigger skull, drawn face, flat hair, hat ---------------------------
  {
    const p = new Part()
    const sk = s.skin
    const skD = shade(sk, -0.12)
    p.cyl(0.04, 0.045, 0.2, 14, sk, [0, 0.04, -0.005], { open: true })                         // neck
    p.frame([0, HEAD_LIFT, 0], [0, 0, 0], () => {
      const jaw = 0.094 + 0.012 * (s.bulk - 1)
      const H: Ring[] = [
        { y: 0.058, rx: 0.036, rz: 0.028, z: 0.082 },     // chin
        { y: 0.082, rx: 0.066, rz: 0.066, z: 0.054 },
        { y: 0.112, rx: jaw, rz: 0.103, z: 0.028 },        // jaw
        { y: 0.158, rx: 0.114, rz: 0.126, z: 0.01 },       // cheeks
        { y: 0.208, rx: 0.123, rz: 0.137, z: 0.0 },        // eyes
        { y: 0.258, rx: 0.125, rz: 0.14, z: -0.008 },
        { y: 0.304, rx: 0.11, rz: 0.125, z: -0.014 },
        { y: 0.338, rx: 0.077, rz: 0.09, z: -0.018 },
        { y: 0.358, rx: 0.032, rz: 0.038, z: -0.02 },
      ]
      const HL = { seg: 16, phase: 0 }
      p.loft(H, sk, { ...HL, bottom: 0.01, top: 0.01, smoothCaps: true })                     // skull + jaw
      // nose: a soft wedge (profile reads in the hull), drawn side line + nostril on the face
      p.ball(sk, [0, 0.178, 0.132], [0.019, 0.042, 0.034], { rot: [-0.22, 0, 0], w: 12, h: 8 })
      p.stroke(H, [[0.214, 0.1], [0.19, 0.15], [0.165, 0.19], [0.152, 0.16]], 0.0038, { off: 0.003, n: 10 })
      p.ball(INK, [0.012, 0.149, 0.158], [0.007, 0.004, 0.004], { w: 6, h: 4, ink: false })   // nostril dot
      // ears: simple flattened shapes, outlined by the hull
      for (const x of [-1, 1]) p.ball(skD, [x * 0.122, 0.19, -0.012], [0.022, 0.04, 0.03], { rot: [0, x * 0.3, 0], w: 10, h: 6 })
      if (s.shades) {
        p.loft(band(H, 0.192, 0.236, 1.065), '#1b1d23', { ...HL, arcSeg: [-3, 3], finish: 'gloss', ink: true })  // wraparounds
        p.loft(band(H, 0.226, 0.236, 1.08), s.hatTrim, { ...HL, arcSeg: [-3, 3], finish: 'gloss' })            // frame top edge
        for (const x of [-1, 1]) p.stroke(H, [[0.252, x * 0.2], [0.258, x * 0.36], [0.255, x * 0.52]], 0.0055)  // brows
      } else {
        for (const x of [-1, 1]) {
          const e = x * 0.36
          // almond eye white, pupil looking a touch inward, upper lid line, brow
          p.ball(EYE_WHITE, onSurf(H, 0.206, e, -0.002), [0.021, 0.0115, 0.008], { rot: [0, e, 0], w: 10, h: 6, ink: false })
          p.ball(INK, onSurf(H, 0.205, e - x * 0.03, 0.004), [0.0085, 0.0095, 0.005], { rot: [0, e, 0], w: 8, h: 6, ink: false })
          p.stroke(H, [[0.208, e - x * 0.2], [0.217, e - x * 0.04], [0.216, e + x * 0.1], [0.208, e + x * 0.19]], 0.0042, { off: 0.006 })
          p.stroke(H, [[0.2, e + x * 0.05], [0.197, e + x * 0.15]], 0.0028, { off: 0.005 })       // lower lid tick
          p.stroke(H, [[0.238, x * 0.17], [0.247, x * 0.32], [0.244, x * 0.5]], 0.0055)          // brow
        }
      }
      // mouth: a slightly crooked line and a short lower-lip mark
      p.stroke(H, [[0.122, -0.3], [0.118, -0.1], [0.119, 0.12], [0.125, 0.3]], 0.0042)
      p.stroke(H, [[0.103, -0.09], [0.101, 0.08]], 0.0032)
      // hair: flat-coloured shapes (show under/around caps), hairline tipped back off the brows
      if (s.hat !== 'hood') {
        p.frame([0, 0.212, -0.008], [-0.35, 0, 0], () =>
          p.ball(s.hair, [0, 0, 0], [0.134, 0.15, 0.146], { half: true, w: 16, h: 6 }))
        for (const x of [-1, 1]) p.ball(s.hair, [x * 0.118, 0.21, 0.035], [0.01, 0.04, 0.026], { w: 8, h: 5 })   // sideburns
      }

      switch (s.hat) {
        case 'hood': {
          // hood up: an ellipsoid shell with a round face opening, dark lining, a soft rolled rim; a
          // fringe of hair poking out under it
          p.ball(s.hair, [0, 0.29, 0.1], [0.09, 0.03, 0.04], { rot: [-0.5, 0, 0], w: 12, h: 6 })
          const tilt = 0.12, open = 0.95
          const shell = () => new SphereGeometry(1, 16, 9, 0, Math.PI * 2, open, Math.PI - open).rotateX(Math.PI / 2 + tilt)
          const C: V3 = [0, 0.215, -0.04], R: V3 = [0.168, 0.188, 0.17]
          p.add(shell(), s.hatColor, C, { scale: R })
          p.add(shell(), shade(s.hatColor, -0.5), C, { scale: [R[0] * 0.94, R[1] * 0.94, R[2] * 0.94], inside: true })
          const ro = Math.sin(open), rc = Math.cos(open)
          p.add(new TorusGeometry(1, 0.13, 6, 18), shade(s.hatColor, -0.1),
            [0, C[1] - R[1] * rc * Math.sin(tilt), C[2] + R[2] * rc * Math.cos(tilt)],
            { scale: [R[0] * ro, R[1] * ro, 0.14], rot: [tilt, 0, 0] })
          break
        }
        case 'beanie': {
          p.frame([0, 0.02, -0.005], [-0.12, 0, 0.07], () => {
            p.ball(s.hatColor, [0, 0.24, 0], [0.146, 0.135, 0.15], { half: true, w: 16, h: 6 })
            p.loft([{ y: 0.222, rx: 0.15, rz: 0.154 }, { y: 0.255, rx: 0.16, rz: 0.164 }, { y: 0.29, rx: 0.153, rz: 0.157 }],
              s.hatTrim, { seg: 16 })                                                                     // rolled cuff
            p.ball(s.hatTrim, [0, 0.39, 0], [0.05, 0.048, 0.05], { w: 10, h: 7 })                         // pom
          })
          break
        }
        case 'trucker': {
          // hair sticking out the back under the cap
          p.ball(s.hair, [0, 0.19, -0.11], [0.11, 0.07, 0.05], { rot: [0.3, 0, 0], w: 12, h: 6 })
          // twisted sideways and tipped back: "worn slightly off"
          p.frame([0, 0.015, 0], [-0.08, 0.95, 0.06], () => {
            p.ball(s.hatColor, [0, 0.235, 0], [0.143, 0.13, 0.148], { half: true, w: 16, h: 6 })
            p.ball(shade(s.trim, -0.05), [0, 0.235, 0], [0.147, 0.134, 0.152], { phi: [Math.PI / 2 - 0.75, 1.5], theta: [0.2, Math.PI / 2 - 0.2], w: 8, h: 4, ink: false }) // foam front
            p.box(0.06, 0.042, 0.008, s.hatTrim, [0, 0.31, 0.133], { rot: [-0.55, 0, 0] })                 // patch
            p.ball(s.hatTrim, [0, 0.245, 0.17], [0.102, 0.013, 0.11], { rot: [0.12, 0, 0], w: 14, h: 5, finish: 'gloss' }) // brim
            p.ball(s.hatTrim, [0, 0.365, 0], [0.018, 0.012, 0.018], { w: 8, h: 4, ink: false })
          })
          break
        }
        case 'slouch': {
          p.frame([0, 0.01, -0.01], [-0.18, 0, 0], () => {
            p.ball(s.hatColor, [0, 0.24, 0], [0.146, 0.13, 0.15], { half: true, w: 16, h: 6 })
            p.loft([{ y: 0.222, rx: 0.151, rz: 0.155 }, { y: 0.268, rx: 0.153, rz: 0.157 }], s.hatColor, { seg: 16 })
            p.loft([{ y: 0.268, rx: 0.155, rz: 0.159 }, { y: 0.288, rx: 0.153, rz: 0.157 }], s.hatTrim, { seg: 16 })
            p.ball(s.hatColor, [0, 0.33, -0.1], [0.125, 0.11, 0.15], { rot: [0.7, 0, 0], w: 14, h: 8 })   // the slouch
          })
          break
        }
      }
    }, HEAD_K)
    parts.head = p.build()
  }

  // ---- arms: long and thin; baggy sleeves with a few folds --------------------------------------
  for (const side of [1, -1] as const) {
    const up = new Part(), fo = new Part()
    const A = { seg: 12, waves: 4 }
    const longSleeve = !tee                       // hoodie / zip: long baggy sleeve over a thin arm
    if (tee) {
      const r0 = xxl ? 0.085 : 0.062
      up.ball(s.topColor, [0, -0.065, 0], [r0, r0 * 0.75, r0], { w: 14, h: 8 })                // shoulder (sits low: sloped)
      const SL: Ring[] = [{ y: -0.065, rx: r0 }, { y: -0.1, rx: r0 * 1.06, w: 0.04 }, { y: xxl ? -0.21 : -0.16, rx: r0 * 1.1, w: 0.06 }]
      up.loft(SL, s.topColor, A)                                                                // baggy short sleeve
      up.loft([{ y: SL[2].y, rx: r0 * 1.06, w: 0.06 }, { y: SL[2].y + 0.06, rx: r0 * 0.98 }], shade(s.topColor, -0.45), { ...A, inside: true })
      up.stroke(SL, [[-0.08, -side * 1.75], [-0.12, -side * 2.05], [SL[2].y + 0.01, -side * 2.2]], 0.0032, { waves: 4 })  // armpit fold
      const under = s.top === 'layered' ? s.sleeveColor : s.skin
      const ar = s.top === 'layered' ? 0.046 : 0.04
      up.loft([{ y: -0.06, rx: ar }, { y: -0.3, rx: ar * 0.93 }], under, A)
    } else {
      up.ball(s.sleeveColor, [0, -0.06, 0], [0.06, 0.05, 0.058], { w: 14, h: 8 })
      const SL: Ring[] = [{ y: -0.06, rx: 0.059 }, { y: -0.12, rx: 0.057 }, { y: -0.22, rx: 0.058, w: 0.07 }, { y: -0.3, rx: 0.061, w: 0.04 }]
      up.loft(SL, s.sleeveColor, A)
    }
    const fc = s.top === 'tee' ? s.skin : s.sleeveColor
    const r0 = longSleeve ? 0.06 : s.top === 'layered' ? 0.044 : 0.037
    fo.ball(fc, [0, 0, 0], [r0, r0, r0], { w: 12, h: 8 })                                      // elbow
    if (longSleeve) {
      const FS: Ring[] = [{ y: 0.0, rx: 0.058 }, { y: -0.09, rx: 0.054, w: 0.04 }, { y: -0.16, rx: 0.06, w: 0.09 },
        { y: -0.205, rx: 0.058, w: 0.07 }, { y: -0.235, rx: 0.052 }]
      fo.loft(FS, fc, A)
      fo.loft([{ y: -0.232, rx: 0.049 }, { y: -0.268, rx: 0.046 }], s.top === 'zip' ? s.trim : shade(fc, -0.2), { ...A, bottom: 0 })  // cuff
      fo.stroke(FS, [[-0.13, side * -0.5], [-0.15, side * 0.1], [-0.18, side * 0.6]], 0.0032, { waves: 4 })   // bunched fold
      fo.stroke(FS, [[-0.03, -side * 0.5], [-0.07, -side * 0.9]], 0.0032, { waves: 4 })                         // elbow crease
    } else if (s.top === 'layered') {
      const FS: Ring[] = [{ y: 0.0, rx: r0 }, { y: -0.14, rx: r0 * 0.96 }, { y: -0.2, rx: r0 * 1.08, w: 0.08 }, { y: -0.24, rx: r0 * 0.98 }]
      fo.loft(FS, fc, A)
      fo.loft([{ y: -0.236, rx: r0 * 0.95 }, { y: -0.266, rx: r0 * 0.92 }], shade(fc, -0.2), { ...A, bottom: 0 })
    } else {
      fo.loft([{ y: 0.0, rx: r0 }, { y: -0.15, rx: r0 * 0.94 }, { y: -0.265, rx: r0 * 0.82 }], fc, { ...A, bottom: 0 })
    }
    if (s.wristbands) {
      const wr = longSleeve ? 0.052 : r0 * 0.92 + 0.008
      fo.loft([{ y: -0.215, rx: wr }, { y: -0.255, rx: wr * 0.96 }], s.accent, { ...A, finish: 'gloss', bottom: 0 })
    }
    // hands are separate jointed meshes (grab/hand.ts), attached at the wrist by the rig
    parts[side > 0 ? 'shL' : 'shR'] = up.build()
    parts[side > 0 ? 'elL' : 'elR'] = fo.build()
  }

  // ---- legs: very baggy wide-leg pants that flare and pool over big chunky sneakers ------------
  for (const side of [1, -1] as const) {
    const th = new Part(), sh = new Part(), ft = new Part()
    const pc = s.pantsColor
    const A = { seg: 16, waves: 5, wavePh: side * 0.6 }
    const THIGH: Ring[] = [{ y: 0.05, rx: tR, rz: tR * 0.96 }, { y: -0.2, rx: tR * 1.05, z: 0.004, w: 0.035 }, { y: -0.47, rx: tR * 1.1, w: 0.02 }]
    th.loft(THIGH, pc, A)
    if (s.pants === 'cargo') {
      th.ball(shade(pc, -0.1), [side * tR * 1.04, -0.29, 0], [0.026, 0.08, 0.072], { w: 10, h: 6 })   // side pocket
      th.ball(shade(pc, -0.25), [side * tR * 1.09, -0.222, 0], [0.024, 0.02, 0.074], { w: 10, h: 4, ink: false }) // flap
    } else {
      th.loft(band(THIGH, -0.4, -0.25, 1.015), shade(pc, 0.14), { ...A, arcSeg: [-2, 1] })            // worn knee fade
    }
    th.stroke(THIGH, [[-0.05, side * -0.7], [-0.12, side * -0.45], [-0.2, side * -0.3]], 0.0036, A)      // crotch pull
    const kr = tR * 1.1
    sh.ball(pc, [0, 0, 0], [kr, kr, kr], { w: 14, h: 8 })                                           // knee
    // wide, outward-flared leg; the bottom bunches in a couple of soft rolls and pools over the shoe
    const hx = Math.min(tR * 1.45, HX + 0.03), fx = side * 0.012, hz = Math.min(tR * 1.6, 0.2)
    const SH: Ring[] = [
      { y: 0.02, rx: kr },
      { y: -0.14, rx: tR * 1.15, w: 0.03 },
      { y: -0.25, rx: tR * 1.24, rz: tR * 1.3, x: fx * 0.4, w: 0.05 },
      { y: -0.3, rx: tR * 1.36, rz: tR * 1.44, x: fx * 0.6, z: 0.006, w: 0.09 },
      { y: -0.335, rx: tR * 1.28, rz: tR * 1.38, x: fx * 0.7, z: 0.008, w: 0.05 },
      { y: -0.375, rx: hx * 0.98, rz: hz * 0.975, x: fx, z: 0.014, w: 0.1 },
      { y: -0.41, rx: hx * 0.94, rz: hz * 0.99, x: fx, z: 0.02, w: 0.07 },
      { y: -0.445, rx: hx * 0.98, rz: hz, x: fx, z: 0.024, w: 0.06 },
    ]
    sh.loft(SH, pc, A)
    sh.loft([{ y: -0.445, rx: hx * 0.94, rz: hz * 0.96, x: fx, z: 0.024, w: 0.06 }, { y: -0.39, rx: hx * 0.86, rz: hz * 0.88, x: fx, z: 0.018 }],
      shade(pc, -0.55), { ...A, inside: true })                                                     // hem lining
    // fold lines across the rolls
    sh.stroke(SH, [[-0.27, -0.55], [-0.29, -0.15], [-0.285, 0.25]], 0.0036, A)
    sh.stroke(SH, [[-0.36, -0.1], [-0.355, 0.35], [-0.37, 0.7]], 0.0036, A)
    sh.stroke(SH, [[-0.17, side * 1.2], [-0.24, side * 1.0]], 0.0032, A)
    // big chunky skate shoe (ankle = origin, sole bottom ≈ -0.047, toe toward +z)
    const UP: Ring[] = [
      { y: -0.008, rx: 0.098, rz: 0.196, z: 0.066 },
      { y: 0.045, rx: 0.1, rz: 0.19, z: 0.06 },
      { y: 0.092, rx: 0.089, rz: 0.148, z: 0.024 },
      { y: 0.128, rx: 0.076, rz: 0.098, z: -0.028 },
    ]
    const SO = { seg: 18, sq: 0.6 }
    ft.loft([{ y: -0.047, rx: 0.102, rz: 0.204, z: 0.068 }, { y: 0.0, rx: 0.106, rz: 0.21, z: 0.068 }],
      s.sole, { ...SO, bottom: 0, top: 0, finish: 'gloss' })                                     // fat cupsole (hard rim)
    ft.loft(UP, s.shoe, { seg: 16, sq: 0.72, top: 0.006, finish: 'gloss' })                     // toe box + upper
    for (const a of [[2, 5], [-6, -3]] as [number, number][]) {
      ft.loft(band(UP, 0.012, 0.06, 1.025), s.shoeTrim, { seg: 16, sq: 0.72, arcSeg: a, finish: 'gloss' }) // side stripe
    }
    ft.stroke(UP, [[0.035, -0.75], [0.05, 0], [0.035, 0.75]], 0.0034, { sq: 0.72, off: 0.003 })     // toe-cap seam
    ft.ball(s.shoeTrim, [0, 0.14, 0.06], [0.056, 0.07, 0.026], { rot: [-0.5, 0, 0], w: 10, h: 6 })  // fat tongue
    ft.ring(0.32, shade(s.shoe, -0.08), [0, 0.128, -0.04], [0.074, 0.08, 0.086], { ts: 16, rs: 6, finish: 'gloss' }) // padded collar
    parts[side > 0 ? 'hipL' : 'hipR'] = th.build()
    parts[side > 0 ? 'knL' : 'knR'] = sh.build()
    parts[side > 0 ? 'anL' : 'anR'] = ft.build()
  }

  // ---- deck -----------------------------------------------------------------------------------
  {
    const p = new Part()
    const grip = '#26252b'
    p.box(0.21, 0.022, 0.52, s.deck, [0, 0.095, 0], { finish: 'gloss' })
    p.box(0.205, 0.004, 0.52, grip, [0, 0.108, 0])
    p.box(0.212, 0.004, 0.3, s.deckArt, [0, 0.083, 0], { finish: 'gloss' })
    for (const z of [1, -1]) {
      p.frame([0, 0.095, z * 0.26], [-z * 0.32, 0, 0], () => {
        p.box(0.21, 0.022, 0.12, s.deck, [0, 0, z * 0.06], { finish: 'gloss' })
        p.cyl(0.105, 0.105, 0.022, 10, s.deck, [0, 0, z * 0.12], { finish: 'gloss', scale: [1, 1, 0.55] })
      })
      // truck: baseplate, hanger, axle — chrome
      p.box(0.07, 0.016, 0.09, SILVER, [0, 0.076, z * 0.25], { finish: 'chrome' })
      p.box(0.17, 0.03, 0.04, SILVER, [0, 0.052, z * 0.25], { finish: 'chrome', taper: [0.55, 0.8] })
      p.cyl(0.008, 0.008, 0.23, 5, SILVER, [0, 0.03, z * 0.25], { rot: [0, 0, Math.PI / 2], finish: 'chrome' })
      for (const x of [1, -1]) p.cyl(0.03, 0.03, 0.035, 10, s.wheels, [x * 0.1, 0.03, z * 0.25], { rot: [0, 0, Math.PI / 2], finish: 'gloss' })
    }
    parts.deck = p.build()
  }

  let tris = 0, inkTris = 0, drawCalls = 0
  for (const j of JOINTS) {
    for (const f of FINISHES) {
      const g = parts[j][f]
      if (g) { tris += g.attributes.position.count / 3; drawCalls++ }
    }
    const ink = parts[j].ink
    if (ink) { inkTris += ink.attributes.position.count / 3; drawCalls++ }
  }
  return { parts, tris, inkTris, drawCalls }
}

export function disposeSkater(b: BuiltSkater) {
  for (const j of JOINTS) {
    for (const f of FINISHES) b.parts[j][f]?.dispose()
    b.parts[j].ink?.dispose()
  }
}
