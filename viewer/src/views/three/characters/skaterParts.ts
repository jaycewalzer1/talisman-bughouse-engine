// Low-poly skater part builders. Every body part (one per joint) is built from rounded, lofted
// forms (loft.ts: stacked elliptical / rounded-rect sections with smooth normals), low spheres,
// tubes and a few hard-edged boxes where hard edges help (soles, deck, trucks, tiny decals). Flat
// per-piece vertex colours; each joint's pieces are merged into at most three geometries — one per
// "finish" (matte cloth/skin, glossy plastic/rubber, chrome metal) — so a whole skater is ~28 draw
// calls and ~2.5–3.3k triangles. All characters share the same three retro materials (colour lives
// in the vertices), so there are no per-character materials to compile.
//
// Joint frames (rig units, 1 ≈ 1.8 m tall before spec.height scaling):
//   hips   at (0, HIP_Y, 0) in the rig root          spine at (0, 0.06, 0) in hips
//   head   at (0, 0.52, 0) in spine                  shL/shR at (±shoulderX, 0.46, 0) in spine
//   elL/R  at (0, -UPPER_ARM, 0) in shoulder         wrist at (0, -WRIST, 0) in elbow (hands: ../grab/hand.ts)
//   hipL/R at (±hipX, 0, 0) in hips                  knee at (0, -0.45, 0), ankle at (0, -0.43, 0)
//   deck   in the rig root, origin on the ground, nose toward +z
// Soft joints: every child limb part starts with a sphere at its joint (shoulder cap, elbow, knee)
// that the parent's open tube end overlaps, so bends never open a gap.
import {
  BoxGeometry, BufferGeometry, CatmullRomCurve3, Color, CylinderGeometry, Euler, Float32BufferAttribute,
  Matrix4, Quaternion, SphereGeometry, TorusGeometry, TubeGeometry, Vector3, type ShaderMaterial,
} from 'three'
import { mergeGeometries } from 'three/examples/jsm/utils/BufferGeometryUtils.js'
import { retroMaterial } from '../materials/retroMaterial'
import { flipFaces, frontZ, loftGeo, ringAt, type LoftOpts, type Ring } from './loft'
import type { CharacterSpec } from './rig'

export type V3 = [number, number, number]
export type Finish = 'matte' | 'gloss' | 'chrome'
export const FINISHES: Finish[] = ['matte', 'gloss', 'chrome']
export type PartGeo = Record<Finish, BufferGeometry | null>
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

interface PieceOpts { rot?: V3; scale?: V3; finish?: Finish; inside?: boolean }

const _m = new Matrix4(), _q = new Quaternion(), _e = new Euler(), _p = new Vector3(), _s = new Vector3()
const _c = new Color()

/** Lighten (k > 0) or darken (k < 0) a hex colour. */
export function shade(hex: string, k: number): string {
  _c.set(hex)
  if (k >= 0) _c.lerp(new Color('#ffffff'), k)
  else _c.multiplyScalar(1 + k)
  return '#' + _c.getHexString()
}

class Part {
  private pieces: Record<Finish, BufferGeometry[]> = { matte: [], gloss: [], chrome: [] }
  private frames: Matrix4[] = []

  /** Run `fn` with every piece it adds transformed by (at, rot). */
  frame(at: V3, rot: V3, fn: () => void) {
    _e.set(rot[0], rot[1], rot[2])
    this.frames.push(new Matrix4().compose(new Vector3(...at), new Quaternion().setFromEuler(_e), new Vector3(1, 1, 1)))
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
    this.pieces[o.finish ?? 'matte'].push(geo)
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
    return this.add(g, color, at, o)
  }

  /** Lofted form (see loft.ts), smooth normals. `inside` here is the loft's own (lining) flip. */
  loft(rings: Ring[], color: string, o: LoftOpts & PieceOpts & { at?: V3 } = {}) {
    return this.add(loftGeo(rings, o), color, o.at ?? [0, 0, 0], { ...o, inside: false })
  }

  /** Cylinder along y (smooth sides, flat caps). Use `scale` for oval sections. */
  cyl(rt: number, rb: number, h: number, seg: number, color: string, at: V3, o: PieceOpts & { open?: boolean } = {}) {
    return this.add(new CylinderGeometry(rt, rb, h, seg, 1, o.open ?? false, Math.PI / seg), color, at, o)
  }

  /** Low sphere / ellipsoid (radius 1 — size it with `scale`), or a slice of one. */
  ball(color: string, at: V3, scale: V3, o: PieceOpts & { w?: number; h?: number; half?: boolean; phi?: [number, number]; theta?: [number, number] } = {}) {
    const ph = o.phi ?? [0, Math.PI * 2], th = o.theta ?? [0, o.half ? Math.PI / 2 : Math.PI]
    const g = new SphereGeometry(1, o.w ?? 10, o.h ?? 7, ph[0], ph[1], th[0], th[1])
    return this.add(g, color, at, { ...o, scale })
  }

  /** Smooth tube through points (chains, cables, drawstrings, headphone band). */
  tube(pts: V3[], r: number, color: string, o: PieceOpts & { n?: number; seg?: number } = {}) {
    const curve = new CatmullRomCurve3(pts.map(p => new Vector3(...p)))
    return this.add(new TubeGeometry(curve, o.n ?? pts.length * 2, r, o.seg ?? 4, false), color, [0, 0, 0], o)
  }

  /** Torus lying in the xz plane (radius 1, tube `t`) — size it with `scale`. */
  ring(t: number, color: string, at: V3, scale: V3, o: PieceOpts & { rs?: number; ts?: number } = {}) {
    const g = new TorusGeometry(1, t, o.rs ?? 4, o.ts ?? 10).rotateX(Math.PI / 2)
    return this.add(g, color, at, { ...o, scale })
  }

  build(): PartGeo {
    const out = { matte: null, gloss: null, chrome: null } as PartGeo
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
  tris: number
  drawCalls: number
}

const INK = '#141117'
const SILVER = '#b9bcc4'
const GOLD = '#c9a24a'

export function buildSkater(s: CharacterSpec): BuiltSkater {
  const W = 0.42 * s.bulk
  const B = s.bag
  const SX = shoulderX(s), HX = hipX(s)
  const parts = {} as Record<Joint, PartGeo>
  const tee = s.top === 'tee' || s.top === 'layered'

  // ---- hips: seat of the pants, chain wallet ------------------------------------------------
  {
    const p = new Part()
    const PELVIS: Ring[] = [
      { y: -0.11, rx: HX + 0.07 * B, rz: 0.135 },
      { y: 0.0, rx: Math.max(W * 0.52, HX + 0.09 * B), rz: 0.148 },
      { y: 0.12, rx: W * 0.47, rz: 0.132 },
    ]
    p.loft(PELVIS, s.pantsColor, { seg: 10, sq: 0.9, bottom: 0.05, top: 0, smoothCaps: true })
    p.loft(band(PELVIS, 0.065, 0.1, 1.025), shade(s.pantsColor, -0.3), { seg: 10, sq: 0.9 })   // belt line
    if (s.chainWallet) {
      const x = -W * 0.5
      p.tube([[x + 0.03, 0.08, 0.11], [x - 0.03, -0.06, 0.1], [x - 0.04, -0.18, 0.02], [x - 0.02, -0.15, -0.08], [x + 0.05, -0.02, -0.13]],
        0.007, SILVER, { finish: 'chrome', n: 10, seg: 3 })
      p.ball('#3a2a1c', [x + 0.08, -0.04, -0.13], [0.05, 0.06, 0.016], { w: 6, h: 4, rot: [0.15, -0.35, 0] })
    }
    parts.hips = p.build()
  }

  // ---- spine: torso, hood/collar, prints, necklace, headphones ------------------------------
  {
    const p = new Part()
    const dk = shade(s.topColor, -0.22)
    const seg = 12, sq = 0.85
    const T: Ring[] = tee
      ? [ // XXL tee: hangs past the hips, boxy but soft
        { y: -0.25, rx: W * 0.62, rz: 0.172 },
        { y: -0.1, rx: W * 0.6, rz: 0.168 },
        { y: 0.15, rx: W * 0.585, rz: 0.165, z: 0.004 },
        { y: 0.36, rx: W * 0.6, rz: 0.16 },
        { y: 0.45, rx: SX - 0.015, rz: 0.14 },
        { y: 0.505, rx: SX * 0.55, rz: 0.1 },
        { y: 0.53, rx: 0.09, rz: 0.075 },
      ]
      : [ // oversized hoodie / zip: blousy body, wide shoulders
        { y: -0.12, rx: W * 0.5, rz: 0.14 },
        { y: 0.0, rx: W * 0.535, rz: 0.15 },
        { y: 0.2, rx: W * 0.555, rz: 0.156, z: 0.005 },
        { y: 0.36, rx: W * 0.57, rz: 0.15 },
        { y: 0.45, rx: SX - 0.025, rz: 0.13 },
        { y: 0.505, rx: SX * 0.55, rz: 0.1 },
        { y: 0.53, rx: 0.09, rz: 0.075 },
      ]
    const L = { seg, sq }
    p.loft(T, s.topColor, { ...L, bottom: 0, top: 0.006, smoothCaps: true })
    const fz = (y: number) => frontZ(ringAt(T, y), seg, sq)
    if (tee) {
      p.loft([{ y: 0.512, rx: 0.1, rz: 0.086 }, { y: 0.538, rx: 0.088, rz: 0.074 }], s.trim, { seg: 10 })   // collar
      p.loft(band(T, 0.2, 0.34, 1.012), s.trim, { ...L, arcSeg: [-2, 1] })                   // chest print
      p.loft(band(T, 0.16, 0.185, 1.012), s.accent, { ...L, arcSeg: [-2, 1] })
      p.box(W * 0.26, 0.05, 0.006, INK, [0, 0.27, fz(0.27) + 0.003])
    } else {
      p.loft(band(T, -0.14, -0.07, 1.035), s.top === 'zip' ? dk : s.trim, L)                // ribbed hem
      if (s.top === 'hoodie') {
        p.loft(band(T, -0.04, 0.11, 1.03, 3), dk, { ...L, arcSeg: [-2, 1] })                 // kangaroo pocket
        for (const x of [-1, 1]) {
          const z = fz(0.47)
          p.tube([[x * 0.04, 0.5, z - 0.02], [x * 0.045, 0.43, z + 0.008], [x * 0.05, 0.35, z + 0.006]], 0.0075, s.trim, { n: 4, seg: 4 })
        }
      } else {
        p.loft(band(T, -0.07, 0.505, 1.012, 5), s.trim, { ...L, arcSeg: [-1, 0] })           // zip placket
        p.box(0.022, 0.045, 0.012, SILVER, [0, 0.36, fz(0.36) + 0.012], { finish: 'chrome' })
        p.loft(band(T, -0.05, 0.08, 1.02), dk, { ...L, arcSeg: [-3, -2] })                    // pockets
        p.loft(band(T, -0.05, 0.08, 1.02), dk, { ...L, arcSeg: [1, 2] })
      }
      if (s.hat !== 'hood') p.ball(s.topColor, [0, 0.48, -0.1], [0.16, 0.085, 0.11], { w: 10, h: 6 }) // hood down
    }
    if (s.hat === 'hood') p.ball(s.hatColor, [0, 0.5, -0.085], [0.14, 0.085, 0.1], { w: 10, h: 6 })    // hood up: drape
    if (s.necklace) {
      const c = s.necklace === 'gold' ? GOLD : SILVER
      const z = fz(0.4)
      p.tube([[-0.08, 0.51, z - 0.04], [-0.055, 0.43, z + 0.002], [0, 0.39, z + 0.01], [0.055, 0.43, z + 0.002], [0.08, 0.51, z - 0.04]],
        0.007, c, { finish: 'chrome', n: 10, seg: 3 })
      p.box(0.042, 0.042, 0.014, c, [0, 0.36, z + 0.016], { finish: 'chrome', rot: [0, 0, Math.PI / 4] })
    }
    if (s.headphones) {
      const bandC = '#24222a'
      p.tube([[-0.1, 0.49, 0.1], [-0.135, 0.525, -0.02], [0, 0.55, -0.11], [0.135, 0.525, -0.02], [0.1, 0.49, 0.1]], 0.012, bandC, { finish: 'gloss', n: 10, seg: 5 })
      for (const x of [-0.1, 0.1]) {
        p.cyl(0.05, 0.052, 0.04, 10, s.accent, [x, 0.49, 0.12], { rot: [1.2, 0, x * 2], finish: 'gloss' })
        p.cyl(0.03, 0.03, 0.045, 8, SILVER, [x, 0.49, 0.12], { rot: [1.2, 0, x * 2], finish: 'chrome' })
      }
    }
    parts.spine = p.build()
  }

  // ---- head: face, hair, hat ----------------------------------------------------------------
  {
    const p = new Part()
    const sk = s.skin
    const jaw = 0.092 + 0.02 * (s.bulk - 1)
    const H: Ring[] = [
      { y: 0.068, rx: 0.045, rz: 0.035, z: 0.074 },     // chin
      { y: 0.1, rx: jaw, rz: 0.095, z: 0.036 },         // jaw
      { y: 0.15, rx: 0.117, rz: 0.125, z: 0.012 },      // cheeks
      { y: 0.21, rx: 0.13, rz: 0.138, z: 0.002 },       // temples / eyes
      { y: 0.27, rx: 0.127, rz: 0.137, z: -0.008 },
      { y: 0.32, rx: 0.1, rz: 0.11, z: -0.014 },
      { y: 0.352, rx: 0.05, rz: 0.058, z: -0.016 },
    ]
    const HL = { seg: 12, phase: 0 }
    p.cyl(0.055, 0.06, 0.14, 10, sk, [0, 0.05, 0], { open: true })                         // neck
    p.loft(H, sk, { ...HL, bottom: 0.012, top: 0.012, smoothCaps: true })                   // skull + jaw
    p.ball(shade(sk, -0.08), [0, 0.178, 0.134], [0.02, 0.034, 0.03], { w: 6, h: 4 })         // nose
    for (const x of [-1, 1]) p.ball(shade(sk, -0.1), [x * 0.127, 0.195, -0.005], [0.02, 0.042, 0.032], { w: 6, h: 4 }) // ears
    if (s.shades) {
      p.loft(band(H, 0.197, 0.24, 1.07), s.hatTrim, { ...HL, arcSeg: [-2, 2], finish: 'chrome' })   // wraparounds
      for (const x of [-1, 1]) p.box(0.01, 0.014, 0.11, INK, [x * 0.138, 0.226, 0.03])
    } else {
      for (const x of [-1, 1]) {
        p.ball(INK, [x * 0.045, 0.213, 0.128], [0.015, 0.019, 0.01], { w: 6, h: 4 })                    // eyes
        p.box(0.048, 0.013, 0.012, s.hair, [x * 0.047, 0.25, 0.124], { rot: [0, 0, -x * 0.15] })       // brows
      }
    }
    p.box(0.045, 0.01, 0.01, shade(sk, -0.32), [0, 0.128, 0.131])                          // mouth
    // hair (shows under/around caps), hairline tipped back off the brows
    if (s.hat !== 'hood') {
      p.frame([0, 0.215, -0.006], [-0.35, 0, 0], () =>
        p.ball(s.hair, [0, 0, 0], [0.137, 0.15, 0.146], { half: true, w: 12, h: 4 }))
    }

    switch (s.hat) {
      case 'hood': {
        // hood up: an ellipsoid shell with a round face opening, dark lining, and a soft rolled rim
        const tilt = 0.12, open = 0.95
        const shell = () => new SphereGeometry(1, 12, 6, 0, Math.PI * 2, open, Math.PI - open).rotateX(Math.PI / 2 + tilt)
        const C: V3 = [0, 0.215, -0.04], R: V3 = [0.172, 0.19, 0.17]
        p.add(shell(), s.hatColor, C, { scale: R })
        p.add(shell(), shade(s.hatColor, -0.45), C, { scale: [R[0] * 0.94, R[1] * 0.94, R[2] * 0.94], inside: true })
        const ro = Math.sin(open), rc = Math.cos(open)
        p.add(new TorusGeometry(1, 0.14, 4, 14), shade(s.hatColor, -0.12),
          [0, C[1] - R[1] * rc * Math.sin(tilt), C[2] + R[2] * rc * Math.cos(tilt)],
          { scale: [R[0] * ro, R[1] * ro, 0.14], rot: [tilt, 0, 0] })
        break
      }
      case 'beanie': {
        p.frame([0, 0.02, -0.005], [-0.12, 0, 0.07], () => {
          p.ball(s.hatColor, [0, 0.24, 0], [0.148, 0.137, 0.152], { half: true, w: 12, h: 4 })
          p.loft([{ y: 0.222, rx: 0.152, rz: 0.156 }, { y: 0.255, rx: 0.162, rz: 0.166 }, { y: 0.29, rx: 0.155, rz: 0.159 }],
            s.hatTrim, { seg: 12 })                                                                     // rolled cuff
          p.ball(s.hatTrim, [0, 0.39, 0], [0.05, 0.048, 0.05], { w: 8, h: 5 })                          // pom
        })
        break
      }
      case 'trucker': {
        // twisted sideways and tipped back: "worn slightly off"
        p.frame([0, 0.015, 0], [-0.08, 0.95, 0.06], () => {
          p.ball(s.hatColor, [0, 0.235, 0], [0.145, 0.13, 0.15], { half: true, w: 12, h: 4 })
          p.ball('#d9d5cc', [0, 0.235, 0], [0.149, 0.134, 0.154], { phi: [Math.PI / 2 - 0.75, 1.5], theta: [0.2, Math.PI / 2 - 0.2], w: 6, h: 3 }) // foam front
          p.box(0.06, 0.042, 0.008, s.hatTrim, [0, 0.31, 0.135], { rot: [-0.55, 0, 0] })                 // patch
          p.ball(s.hatTrim, [0, 0.245, 0.17], [0.102, 0.013, 0.11], { rot: [0.12, 0, 0], w: 10, h: 4, finish: 'gloss' }) // brim
          p.ball(s.hatTrim, [0, 0.365, 0], [0.018, 0.012, 0.018], { w: 6, h: 3 })
        })
        break
      }
      case 'slouch': {
        p.frame([0, 0.01, -0.01], [-0.18, 0, 0], () => {
          p.ball(s.hatColor, [0, 0.24, 0], [0.148, 0.132, 0.152], { half: true, w: 12, h: 4 })
          p.loft([{ y: 0.222, rx: 0.153, rz: 0.157 }, { y: 0.268, rx: 0.155, rz: 0.159 }], s.hatColor, { seg: 12 })
          p.loft([{ y: 0.268, rx: 0.157, rz: 0.161 }, { y: 0.288, rx: 0.155, rz: 0.159 }], s.hatTrim, { seg: 12 })
          p.ball(s.hatColor, [0, 0.33, -0.1], [0.125, 0.11, 0.15], { rot: [0.7, 0, 0], w: 10, h: 6 })   // the slouch
        })
        break
      }
    }
    parts.head = p.build()
  }

  // ---- arms -----------------------------------------------------------------------------------
  for (const side of [1, -1] as const) {
    const up = new Part(), fo = new Part()
    const A = { seg: 10 }
    const hoodieSleeve = !tee                       // hoodie / zip: long baggy sleeve
    if (tee) {
      up.ball(s.topColor, [0, -0.01, 0], [0.098, 0.098, 0.098], { w: 10, h: 6 })             // shoulder
      up.loft([{ y: 0.0, rx: 0.098 }, { y: -0.12, rx: 0.105 }, { y: -0.19, rx: 0.11 }], s.topColor, A)   // baggy short sleeve
      up.loft([{ y: -0.19, rx: 0.106 }, { y: -0.13, rx: 0.1 }], shade(s.topColor, -0.4), { ...A, inside: true })
      const under = s.top === 'layered' ? s.sleeveColor : s.skin
      up.loft([{ y: -0.08, rx: 0.056 }, { y: -0.3, rx: 0.05 }], under, A)
    } else {
      up.ball(s.sleeveColor, [0, -0.01, 0], [0.088, 0.09, 0.088], { w: 10, h: 6 })
      up.loft([{ y: 0.0, rx: 0.086 }, { y: -0.14, rx: 0.082 }, { y: -0.3, rx: 0.073 }], s.sleeveColor, A)
    }
    const fc = s.top === 'tee' ? s.skin : s.sleeveColor
    const r0 = hoodieSleeve ? 0.073 : 0.05
    fo.ball(fc, [0, 0, 0], [r0, r0, r0], { w: 8, h: 6 })                                       // elbow
    if (hoodieSleeve) {
      fo.loft([{ y: 0.0, rx: 0.072 }, { y: -0.12, rx: 0.075 }, { y: -0.205, rx: 0.07 }], fc, A)
      fo.loft([{ y: -0.2, rx: 0.067 }, { y: -0.255, rx: 0.064 }], s.top === 'zip' ? s.trim : shade(fc, -0.2), { ...A, bottom: 0 })
    } else {
      fo.loft([{ y: 0.0, rx: r0 }, { y: -0.15, rx: r0 * 0.95 }, { y: -0.255, rx: r0 * 0.8 }], fc, { ...A, bottom: 0 })
    }
    if (s.wristbands) fo.loft([{ y: -0.195, rx: hoodieSleeve ? 0.073 : 0.056 }, { y: -0.25, rx: hoodieSleeve ? 0.07 : 0.052 }],
      s.accent, { ...A, finish: 'gloss', bottom: 0 })
    // hands are separate jointed meshes (grab/hand.ts), attached at the wrist by the rig
    parts[side > 0 ? 'shL' : 'shR'] = up.build()
    parts[side > 0 ? 'elL' : 'elR'] = fo.build()
  }

  // ---- legs: baggy pants that flare and stack over puffy shoes --------------------------------
  for (const side of [1, -1] as const) {
    const th = new Part(), sh = new Part(), ft = new Part()
    const pc = s.pantsColor, A = { seg: 10 }
    const THIGH: Ring[] = [{ y: 0.04, rx: 0.108 * B }, { y: -0.2, rx: 0.112 * B, z: 0.004 }, { y: -0.47, rx: 0.1 * B }]
    th.loft(THIGH, pc, A)
    if (s.pants === 'cargo') {
      th.ball(shade(pc, -0.12), [side * 0.105 * B, -0.29, 0], [0.028, 0.075, 0.07], { w: 8, h: 5 })   // side pocket
      th.ball(shade(pc, -0.25), [side * 0.11 * B, -0.225, 0], [0.026, 0.02, 0.072], { w: 8, h: 4 })   // flap
    } else {
      th.loft(band(THIGH, -0.4, -0.25, 1.015), shade(pc, 0.16), { ...A, arcSeg: [-2, 1] })            // worn knee fade
    }
    sh.ball(pc, [0, 0, 0], [0.1 * B, 0.1 * B, 0.1 * B], { w: 8, h: 6 })                                // knee
    sh.loft([
      { y: 0.02, rx: 0.1 * B }, { y: -0.15, rx: 0.106 * B }, { y: -0.27, rx: 0.116 * B, z: 0.004 },
      { y: -0.325, rx: 0.127 * B }, { y: -0.35, rx: 0.12 * B }, { y: -0.38, rx: 0.134 * B, z: 0.006 }, { y: -0.41, rx: 0.13 * B, z: 0.008 },
    ], pc, A)
    sh.loft([{ y: -0.41, rx: 0.126 * B, z: 0.008 }, { y: -0.36, rx: 0.114 * B }], shade(pc, -0.5), { ...A, inside: true }) // cuff lining
    // puffy skate shoe (ankle = origin, sole bottom ≈ -0.047, toe toward +z)
    const UP: Ring[] = [
      { y: -0.012, rx: 0.084, rz: 0.163, z: 0.048 },
      { y: 0.035, rx: 0.086, rz: 0.158, z: 0.043 },
      { y: 0.075, rx: 0.078, rz: 0.124, z: 0.012 },
      { y: 0.105, rx: 0.07, rz: 0.085, z: -0.035 },
    ]
    ft.loft([{ y: -0.047, rx: 0.088, rz: 0.172, z: 0.05 }, { y: -0.008, rx: 0.091, rz: 0.176, z: 0.05 }],
      s.sole, { seg: 14, sq: 0.55, bottom: 0, top: 0, finish: 'gloss' })                                // cupsole (hard rim)
    ft.loft(UP, s.shoe, { seg: 12, sq: 0.7, top: 0.004, finish: 'gloss' })                               // toe box + upper
    for (const a of [[1, 4], [-5, -2]] as [number, number][]) {
      ft.loft(band(UP, 0.0, 0.045, 1.025), s.shoeTrim, { seg: 12, sq: 0.7, arcSeg: a, finish: 'gloss' })  // side stripe
    }
    ft.ball(s.shoeTrim, [0, 0.12, 0.055], [0.05, 0.064, 0.024], { rot: [-0.5, 0, 0], w: 6, h: 4 })      // fat tongue
    ft.ring(0.3, shade(s.shoe, -0.06), [0, 0.112, -0.042], [0.07, 0.075, 0.08], { ts: 10, finish: 'gloss' }) // padded collar
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
        p.cyl(0.105, 0.105, 0.022, 8, s.deck, [0, 0, z * 0.12], { finish: 'gloss', scale: [1, 1, 0.55] })
      })
      // truck: baseplate, hanger, axle — chrome
      p.box(0.07, 0.016, 0.09, SILVER, [0, 0.076, z * 0.25], { finish: 'chrome' })
      p.box(0.17, 0.03, 0.04, SILVER, [0, 0.052, z * 0.25], { finish: 'chrome', taper: [0.55, 0.8] })
      p.cyl(0.008, 0.008, 0.23, 5, SILVER, [0, 0.03, z * 0.25], { rot: [0, 0, Math.PI / 2], finish: 'chrome' })
      for (const x of [1, -1]) p.cyl(0.03, 0.03, 0.035, 8, s.wheels, [x * 0.1, 0.03, z * 0.25], { rot: [0, 0, Math.PI / 2], finish: 'gloss' })
    }
    parts.deck = p.build()
  }

  let tris = 0, drawCalls = 0
  for (const j of JOINTS) for (const f of FINISHES) {
    const g = parts[j][f]
    if (g) { tris += g.attributes.position.count / 3; drawCalls++ }
  }
  return { parts, tris, drawCalls }
}

export function disposeSkater(b: BuiltSkater) {
  for (const j of JOINTS) for (const f of FINISHES) b.parts[j][f]?.dispose()
}
