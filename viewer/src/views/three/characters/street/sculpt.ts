// Turns a pack body into a dressed one, in the T-pose, before any skinning happens:
//   1. every triangle gets a region (skin, top, pants) by its dominant bone and where it sits; feet are
//      deleted (shoes are rigid meshes, accessories.ts)
//   2. vertices are detached between regions, so garments have real open hems
//   3. garment vertices are pushed out into baggy clothes (the skin weights come along, so the
//      clothes deform with the existing skeleton), with folds and stacked pant legs in the geometry
//   4. normals are rebuilt, welded across the UV seams so the seams don't show
// The body's own UVs are kept: texture.ts paints the outfit into them.
import { BufferAttribute, BufferGeometry, type Skeleton, Vector3 } from 'three'
import type { BodyMetrics, Outfit } from './outfits'

export type Region = 'skin' | 'top' | 'pants'
export type BoneClass = 'head' | 'torso' | 'uarm' | 'farm' | 'hand' | 'uleg' | 'lleg' | 'foot'

const classOf = (name: string): BoneClass => {
  if (name === 'Head' || name === 'neck_01') return 'head'
  if (/^(index|middle|ring|pinky|thumb)|^hand/.test(name)) return 'hand'
  if (name.startsWith('lowerarm')) return 'farm'
  if (name.startsWith('upperarm')) return 'uarm'
  if (name.startsWith('thigh')) return 'uleg'
  if (name.startsWith('calf')) return 'lleg'
  if (name.startsWith('foot') || name.startsWith('ball')) return 'foot'
  return 'torso'
}

const smooth = (a: number, b: number, x: number) => { const t = Math.max(0, Math.min(1, (x - a) / (b - a))); return t * t * (3 - 2 * t) }
const lerp = (a: number, b: number, k: number) => a + (b - a) * k

export interface Sculpted {
  geometry: BufferGeometry      // groups: 0 = cloth, 1 = skin
  /** Per-vertex paint inputs (rest pose before sculpting): region code 0 skin, 1 top, 2 pants. */
  region: Uint8Array
  rest: Float32Array
  cls: Uint8Array               // index into CLASSES
}
export const CLASSES: BoneClass[] = ['head', 'torso', 'uarm', 'farm', 'hand', 'uleg', 'lleg', 'foot']

/** How far down a top's sleeves go (|x| in the T-pose), or Infinity for full length. */
const sleeveEnd = (o: Outfit, b: BodyMetrics) => o.top === 'tee' ? b.shoulderRestX + 0.2 : Infinity

export function sculptBody(src: BufferGeometry, skeleton: Skeleton, o: Outfit, b: BodyMetrics): Sculpted {
  const pos = src.attributes.position, nor = src.attributes.normal, uv = src.attributes.uv
  const ji = src.attributes.skinIndex, jw = src.attributes.skinWeight
  const index = src.index!
  const boneClass = skeleton.bones.map(x => classOf(x.name))
  const bones = skeleton.bones.map(x => x.name)

  // dominant class per source vertex
  const vc: BoneClass[] = []
  for (let v = 0; v < pos.count; v++) {
    const sum: Partial<Record<BoneClass, number>> = {}
    for (let k = 0; k < 4; k++) { const c = boneClass[ji.getComponent(v, k)]; sum[c] = (sum[c] ?? 0) + jw.getComponent(v, k) }
    vc.push((Object.entries(sum).sort((a, z) => z[1] - a[1])[0][0]) as BoneClass)
  }

  // region per triangle
  const end = sleeveEnd(o, b)
  const out = { idxCloth: [] as number[], idxSkin: [] as number[] }
  const map = new Map<number, number>()
  const srcOf: number[] = [], regOf: number[] = []
  const P = new Vector3()
  const vert = (v: number, reg: number) => {
    const key = v * 3 + reg
    let n = map.get(key)
    if (n === undefined) { n = srcOf.length; map.set(key, n); srcOf.push(v); regOf.push(reg) }
    return n
  }
  for (let t = 0; t < index.count; t += 3) {
    const a = index.getX(t), c = index.getX(t + 1), d = index.getX(t + 2)
    const count: Partial<Record<BoneClass, number>> = {}
    for (const v of [a, c, d]) count[vc[v]] = (count[vc[v]] ?? 0) + 1
    const cls = (Object.entries(count).sort((x, z) => z[1] - x[1])[0][0]) as BoneClass
    if (cls === 'foot') continue
    P.set(0, 0, 0)
    for (const v of [a, c, d]) P.add(new Vector3().fromBufferAttribute(pos, v))
    P.divideScalar(3)
    const ax = Math.abs(P.x)
    let reg = 0
    if (cls === 'uleg' || cls === 'lleg') reg = 2
    else if (cls === 'torso') {
      const yMax = Math.max(pos.getY(a), pos.getY(c), pos.getY(d))
      // anything reaching up the neck stays skin; the neckline is cut below it
      reg = P.y < b.waistY ? 2 : yMax > b.neckY - 0.05 && ax < 0.11 ? 0 : 1
    }
    else if (cls === 'uarm' || cls === 'farm') reg = ax < end ? 1 : 0
    const tri = [vert(a, reg), vert(c, reg), vert(d, reg)]
    ;(reg === 0 ? out.idxSkin : out.idxCloth).push(...tri)
  }

  // Which vertices sit on a garment's open edge? (Edges used by one triangle in that region, keyed by
  // position so UV seams don't count.) Hems are snapped flat below; the raw edge is a zigzag of triangles.
  const allTris = [...out.idxCloth, ...out.idxSkin]
  const posKey = (i: number) => {
    const v = srcOf[i]
    return `${Math.round(pos.getX(v) * 400)},${Math.round(pos.getY(v) * 400)},${Math.round(pos.getZ(v) * 400)},${regOf[i]}`
  }
  const keyId = new Map<string, number>()
  const vid = srcOf.map((_, i) => { const k = posKey(i); let id = keyId.get(k); if (id === undefined) { id = keyId.size; keyId.set(k, id) } return id })
  const edgeUse = new Map<number, number>()
  const K = keyId.size + 1
  for (let t = 0; t < allTris.length; t += 3) for (let e = 0; e < 3; e++) {
    const a = vid[allTris[t + e]], c = vid[allTris[t + (e + 1) % 3]]
    const key = Math.min(a, c) * K + Math.max(a, c)
    edgeUse.set(key, (edgeUse.get(key) ?? 0) + 1)
  }
  const onEdge = new Uint8Array(keyId.size)
  for (const [key, count] of edgeUse) if (count === 1) { onEdge[Math.floor(key / K)] = 1; onEdge[key % K] = 1 }

  // sculpt
  const n = srcOf.length
  const fp = new Float32Array(n * 3), fu = new Float32Array(n * 2), fi = new Uint16Array(n * 4), fw = new Float32Array(n * 4)
  const rest = new Float32Array(n * 3), region = new Uint8Array(n), cls = new Uint8Array(n)
  const nrm = new Vector3()
  for (let i = 0; i < n; i++) {
    const v = srcOf[i], reg = regOf[i]
    P.fromBufferAttribute(pos, v); nrm.fromBufferAttribute(nor, v)
    rest.set([P.x, P.y, P.z], i * 3); region[i] = reg; cls[i] = CLASSES.indexOf(vc[v])
    const edge = onEdge[vid[i]] === 1
    if (reg === 2) pants(P, o, b, edge)
    else if (reg === 1) top(P, nrm, o, b, end, edge, vc[v] === 'uarm' || vc[v] === 'farm' || vc[v] === 'hand')
    else if (vc[v] === 'farm' || vc[v] === 'uarm') slimArm(P, b)
    else if (vc[v] === 'torso' && edge) neckRing(P, b, 0.094)
    fp.set([P.x, P.y, P.z], i * 3); fu.set([uv.getX(v), uv.getY(v)], i * 2)
    for (let k = 0; k < 4; k++) {
      // clothes don't follow the feet or hands: a pant hem hangs from the shin, a cuff from the forearm
      let joint = ji.getComponent(v, k)
      const c = boneClass[joint]
      if (reg === 2 && c === 'foot') joint = bones.indexOf(skeleton.bones[joint].name.endsWith('_l') ? 'calf_l' : 'calf_r')
      else if (reg === 1 && c === 'hand') joint = bones.indexOf(skeleton.bones[joint].name.endsWith('_l') ? 'lowerarm_l' : 'lowerarm_r')
      else if (reg === 1 && c === 'head') joint = bones.indexOf('spine_03')     // nor does a collar follow the head
      fi[i * 4 + k] = joint; fw[i * 4 + k] = jw.getComponent(v, k)
    }
    // Over the shoulder the pack's weights assume a tight T-pose; on a loose sleeve with the arm hanging
    // they fold into "horns". A simple ramp from chest to upper arm stays round.
    const shAx = Math.abs(rest[i * 3])
    if (reg === 1 && shAx > b.shoulderRestX - 0.06 && shAx < b.shoulderRestX + 0.1 && rest[i * 3 + 1] > b.armY - 0.16) {
      const w = smooth(b.shoulderRestX - 0.06, b.shoulderRestX + 0.1, shAx)
      fi[i * 4] = bones.indexOf(rest[i * 3] > 0 ? 'upperarm_l' : 'upperarm_r'); fw[i * 4] = w
      fi[i * 4 + 1] = bones.indexOf('spine_03'); fw[i * 4 + 1] = 1 - w
      fw[i * 4 + 2] = 0; fw[i * 4 + 3] = 0
    }
  }

  const g = new BufferGeometry()
  g.setAttribute('position', new BufferAttribute(fp, 3))
  g.setAttribute('uv', new BufferAttribute(fu, 2))
  g.setAttribute('skinIndex', new BufferAttribute(fi, 4))
  g.setAttribute('skinWeight', new BufferAttribute(fw, 4))
  g.setIndex([...out.idxCloth, ...out.idxSkin])
  g.addGroup(0, out.idxCloth.length, 0)
  g.addGroup(out.idxCloth.length, out.idxSkin.length, 1)
  weldedNormals(g, regOf)
  g.computeBoundingSphere()
  return { geometry: g, region, rest, cls }
}

/** The pack's arms are superhero arms: bare ones get slimmed a little around the bone. */
function slimArm(p: Vector3, b: BodyMetrics) {
  const k = 0.88
  p.y = b.armY + (p.y - b.armY) * k
  p.z = b.armZ + (p.z - b.armZ) * k
}

function pants(p: Vector3, o: Outfit, b: BodyMetrics, edge: boolean) {
  const side = p.x >= 0 ? 1 : -1
  const hipTop = b.waistY
  // the pant leg falls almost straight from the hip and stacks on the shoe; stretch the shin down to a low hem
  const y0 = p.y
  const hem = 0.028
  const y = edge && y0 < 0.19 ? hem : y0 < 0.3 ? lerp(hem + 0.012, 0.3, Math.max(0, (y0 - 0.12) / 0.18)) : y0
  const stack = 1 - smooth(0.05, 0.42, y)         // folds pile up toward the hem
  const dx = p.x - side * 0.118, dz = p.z + 0.035
  const r = Math.hypot(dx, dz) || 1e-4
  const th = Math.atan2(dz, dx * side)
  const bag = o.bag
  const fold = 1
    + 0.09 * stack * Math.sin(y * 42 + th * 2 + o.seed)
    + 0.05 * Math.sin(y * 17 - th * 3 + o.seed * 2)
    + 0.05 * smooth(0.42, 0.6, y) * (1 - smooth(0.6, 0.8, y)) * Math.cos(th)    // knee
  const tube = (0.094 + 0.024 * smooth(0.95, 0.5, y) + 0.016 * stack) * bag * fold
  // inner edge kept from crossing the other leg
  let nx = side * 0.122 + (dx / r) * tube
  const nz = -0.03 + (dz / r) * tube * 1.06
  if (nx * side < 0.012) nx = side * 0.012
  // the hips: one oval, low-slung, blended into the two legs through the drop crotch
  const w = smooth(0.74, 0.92, y)
  const hx = p.x, hz = p.z + 0.035
  const hr = Math.hypot(hx / b.hipRx, hz / 0.15) || 1e-4
  const ex = hx / hr * 1.05, ez = -0.035 + hz / hr * 1.05
  const blend = lerp(0.6, 0.9, 1 - smooth(hipTop - 0.05, hipTop, y))
  const ovalX = lerp(p.x, ex, blend), ovalZ = lerp(p.z, ez, blend)
  p.set(lerp(nx, ovalX, w), y, lerp(nz, ovalZ, w))
}

/** Snaps a neckline's zigzag edge onto a clean ring round the neck. */
function neckRing(p: Vector3, b: BodyMetrics, radius: number) {
  const a = Math.atan2(p.x, p.z + 0.03)
  p.set(Math.sin(a) * radius, b.neckY - 0.05, -0.03 + Math.cos(a) * radius * 0.92)
}

function top(p: Vector3, n: Vector3, o: Outfit, b: BodyMetrics, end: number, edge: boolean, isArm: boolean) {
  if (!isArm && p.y > b.neckY - 0.06) {
    // everything above the neckline: on the ring if it's the edge, otherwise just below it, never poking up
    const R = o.top === 'hoodie' ? 0.082 : 0.097
    if (edge) { neckRing(p, b, R); return }
    p.y = b.neckY - 0.062
    const a = Math.hypot(p.x, p.z + 0.03)
    if (a < R * 1.1) { const k = (R * 1.1) / (a || 1); p.x *= k; p.z = -0.03 + (p.z + 0.03) * k }
    return
  }
  const hoodie = o.top === 'hoodie'
  const side = p.x >= 0 ? 1 : -1
  let ax = Math.abs(p.x)
  const cuffAt = end === Infinity ? b.wristX + 0.012 : end
  if (isArm && ax > cuffAt - 0.15) ax = edge ? cuffAt : Math.min(ax, cuffAt - 0.012)     // flat sleeve hem
  if (!isArm && p.y < b.waistY + 0.12) p.y = edge ? b.waistY - 0.004 : Math.max(p.y, b.waistY + 0.012)   // flat bottom hem
  const armW = smooth(b.shoulderRestX - 0.04, b.shoulderRestX + 0.06, ax) * (isArm ? 1 : smooth(b.armY - 0.2, b.armY - 0.08, p.y))
  // torso: towards a loose oval, keeping some of the body's shape; the hem hangs over the pants
  const rx = b.torsoRx * (hoodie ? 1.2 : 1.03), rz = b.torsoRz * (hoodie ? 1.3 : 1.08)
  const cz = -0.03
  const q = Math.hypot(p.x / rx, (p.z - cz) / rz) || 1e-4
  const u = (p.y - b.waistY) / (b.armY - b.waistY)
  const blendT = (hoodie ? 0.78 : 0.68) * (1 - smooth(0.82, 1.05, u) * 0.75)
  const thick = hoodie ? 0.03 : 0.01
  const tx = lerp(p.x + n.x * thick, p.x / q, blendT)
  const tz = lerp(p.z + n.z * thick, cz + (p.z - cz) / q, blendT)
  const drop = (hoodie ? 0.1 : 0.07) * (1 - smooth(0, 0.17, u))
  const ripple = 1 + (hoodie ? 0.03 : 0.018) * Math.sin(p.y * 34 + Math.atan2(p.z, p.x) * 3 + o.seed)
  // sloped shoulders rather than a flat shelf out to the sleeve
  const slope = 0.05 * smooth(0.06, b.shoulderRestX, Math.abs(p.x)) * smooth(b.armY - 0.1, b.armY + 0.03, p.y)
  const ty = p.y - drop + n.y * thick * 0.6 - slope
  // sleeve: a loose tube around the arm bone, bunching near the cuff
  const dy = p.y - b.armY, dz = p.z - b.armZ
  const r = Math.hypot(dy, dz) || 1e-4
  const last = end === Infinity ? b.wristX : end
  const t = smooth(b.shoulderRestX, last, ax)
  const base = hoodie ? 0.08 : o.top === 'layered' ? 0.056 : 0.066
  const flare = end === Infinity ? 1 - 0.28 * smooth(last - 0.07, last, ax) : 1 + 0.16 * smooth(last - 0.03, last, ax)
  const R = (base - (hoodie ? 0.012 : 0.008) * t) * flare * lerp(0.72, 1, smooth(b.shoulderRestX, b.shoulderRestX + 0.13, ax)) * (1 + 0.1 * Math.sin(ax * 48 + o.seed + (dz > 0 ? 1 : 0)) * (hoodie ? 1 : 0.5))
  const sy = b.armY + (dy / r) * lerp(r, R, 0.85), sz = b.armZ + (dz / r) * lerp(r, R, 0.85)
  p.set(lerp(tx * ripple, side * ax, armW), lerp(ty, sy, armW), lerp(tz * ripple, sz, armW))
}

/** computeVertexNormals, then averaged across vertices at the same position within a region. */
function weldedNormals(g: BufferGeometry, regOf: number[]) {
  g.computeVertexNormals()
  const pos = g.attributes.position, nor = g.attributes.normal
  const bins = new Map<string, number[]>()
  for (let i = 0; i < pos.count; i++) {
    const key = `${Math.round(pos.getX(i) * 2000)},${Math.round(pos.getY(i) * 2000)},${Math.round(pos.getZ(i) * 2000)},${regOf[i]}`
    const b = bins.get(key); if (b) b.push(i); else bins.set(key, [i])
  }
  const v = new Vector3()
  for (const list of bins.values()) {
    v.set(0, 0, 0)
    for (const i of list) { v.x += nor.getX(i); v.y += nor.getY(i); v.z += nor.getZ(i) }
    // flattened hems can cancel out to nothing, and a zero normal shades to NaN, which bloom then
    // smears over the whole frame
    if (v.lengthSq() < 1e-8) v.set(0, -1, 0)
    v.normalize()
    for (const i of list) nor.setXYZ(i, v.x, v.y, v.z)
  }
}
