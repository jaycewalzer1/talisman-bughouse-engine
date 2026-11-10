// Fake baked lighting, PS2 style: at load, every static mesh in the set gets an `aBake` vertex
// attribute read by retroMaterial: x = ambient occlusion from short rays (corners, ledges, whatever
// stands on the ground go dark), y = how much of a streetlamp reaches it (with a visibility ray, so
// the tables and columns throw soft shadows). Instanced props get one value per instance.
//
// Rays only test boxes: each static mesh (or instance) counts as its oriented bounding box. That is
// crude but cheap, and at vertex resolution nobody can tell. Big faces are tessellated first so a
// vertex is actually near the things that should darken it.
import { BufferAttribute, type BufferGeometry, InstancedBufferAttribute, type InstancedMesh, Matrix4, type Mesh, type Object3D, Vector3 } from 'three'
import { TessellateModifier } from 'three/examples/jsm/modifiers/TessellateModifier.js'
import { LOOK } from '../style.config'
import { BODEGA_X, STREET_LAMPS, WORK_LAMP } from './lamps'
import { WALL_Z } from '../layout'

interface Box {
  inv: Float64Array       // world -> box space (rows 0..2 of a rigid-ish affine)
  min: [number, number, number]
  max: [number, number, number]
  c: Vector3; r: number   // bounding sphere
  owner: object
}

const TESS_EDGE = 0.75         // metres: faces longer than this get split
const MIN_OCCLUDER = 0.18      // anything smaller than this doesn't block light
const BAKE_RADIUS = 34         // far-away stuff (skyline, the ends of the el) stays unbaked
const LAMP_RANGE = 8.5

const LAMPS: { p: Vector3; power: number }[] = [
  { p: new Vector3(...WORK_LAMP), power: 1 },
  ...STREET_LAMPS.map(l => ({ p: new Vector3(l.base[0] + l.arm, l.base[1] - 0.1, l.base[2]), power: l.flicker ? 0.75 : 1.1 })),
  { p: new Vector3(BODEGA_X, 2.1, WALL_Z + 1.3), power: 0.55 },   // spill from the shop window
]

/** A fixed hemisphere ray set (cosine weighted, around +y). */
function raySet(n: number): Vector3[] {
  const out: Vector3[] = []
  for (let i = 0; i < n; i++) {
    const u = (i + 0.5) / n, a = i * 2.39996323
    const r = Math.sqrt(u)
    out.push(new Vector3(Math.cos(a) * r, Math.sqrt(1 - u), Math.sin(a) * r))
  }
  return out
}

function makeBox(worldMatrix: Matrix4, geo: BufferGeometry, owner: object): Box | null {
  if (!geo.boundingBox) geo.computeBoundingBox()
  const b = geo.boundingBox!
  const size = new Vector3(); b.getSize(size)
  const scale = new Vector3().setFromMatrixScale(worldMatrix)
  const dims = [size.x * scale.x, size.y * scale.y, size.z * scale.z].sort((a, c) => c - a)
  if (dims[0] < MIN_OCCLUDER) return null
  const inv = new Matrix4().copy(worldMatrix).invert().elements
  const c = b.getCenter(new Vector3()).applyMatrix4(worldMatrix)
  const pad = 0.001
  return {
    inv: Float64Array.from(inv),
    min: [b.min.x - pad, b.min.y - pad, b.min.z - pad], max: [b.max.x + pad, b.max.y + pad, b.max.z + pad],
    c, r: size.length() * 0.5 * Math.max(scale.x, scale.y, scale.z),
    owner,
  }
}

/** Slab test in box space. True if the segment from `o` along unit `d` hits the box within `len`. */
function hits(b: Box, o: Vector3, d: Vector3, len: number): boolean {
  const e = b.inv
  const ox = e[0] * o.x + e[4] * o.y + e[8] * o.z + e[12], oy = e[1] * o.x + e[5] * o.y + e[9] * o.z + e[13], oz = e[2] * o.x + e[6] * o.y + e[10] * o.z + e[14]
  const dx = e[0] * d.x + e[4] * d.y + e[8] * d.z, dy = e[1] * d.x + e[5] * d.y + e[9] * d.z, dz = e[2] * d.x + e[6] * d.y + e[10] * d.z
  let t0 = 0, t1 = len
  const slab = (oc: number, dc: number, lo: number, hi: number) => {
    if (Math.abs(dc) < 1e-9) return oc >= lo && oc <= hi
    let a = (lo - oc) / dc, c = (hi - oc) / dc
    if (a > c) { const t = a; a = c; c = t }
    if (a > t0) t0 = a
    if (c < t1) t1 = c
    return t0 <= t1
  }
  return slab(ox, dx, b.min[0], b.max[0]) && slab(oy, dy, b.min[1], b.max[1]) && slab(oz, dz, b.min[2], b.max[2])
}

const _tan = new Vector3(), _bit = new Vector3(), _dir = new Vector3(), _tl = new Vector3()

function sample(p: Vector3, n: Vector3, boxes: Box[], owner: object, rays: Vector3[]): [number, number] {
  const R = LOOK.aoRadius
  // only boxes near this point can occlude it
  const near = boxes.filter(b => b.owner !== owner && b.c.distanceToSquared(p) < (R + b.r) * (R + b.r))
  _tan.set(Math.abs(n.y) < 0.9 ? 0 : 1, Math.abs(n.y) < 0.9 ? 1 : 0, 0).cross(n).normalize()
  _bit.crossVectors(n, _tan)
  let blocked = 0
  for (const r of rays) {
    _dir.set(0, 0, 0).addScaledVector(_tan, r.x).addScaledVector(n, r.y).addScaledVector(_bit, r.z)
    if (p.y + _dir.y * R * 0.5 < 0) blocked += 0.6                    // heading into the ground
    else if (near.some(b => hits(b, p, _dir, R))) blocked += 1
  }
  const occ = Math.min(0.92, (blocked / rays.length) * LOOK.aoStrength)
  let lamp = 0
  for (const l of LAMPS) {
    _tl.subVectors(l.p, p)
    const d = _tl.length()
    if (d > LAMP_RANGE) continue
    _tl.divideScalar(d)
    const nl = n.dot(_tl)
    if (nl <= 0) continue
    const fall = (1 - d / LAMP_RANGE) ** 2
    const occluded = boxes.some(b => b.owner !== owner && hits(b, p, _tl, d - 0.3))
    lamp += l.power * nl * fall * (occluded ? 0.12 : 1)
  }
  return [occ, Math.min(1.6, lamp * LOOK.bakedLamp)]
}

const isStatic = (m: Mesh) => {
  const mat = m.material as { transparent?: boolean; blending?: number }
  return !mat.transparent
}

export function bakeWorldLighting(root: Object3D) {
  const t0 = performance.now()
  root.updateWorldMatrix(true, true)
  const meshes: Mesh[] = []
  root.traverse(o => { const m = o as Mesh; if (m.isMesh && isStatic(m)) meshes.push(m) })

  // occluders
  const boxes: Box[] = []
  const inst = new Matrix4(), wm = new Matrix4()
  for (const m of meshes) {
    const im = m as InstancedMesh
    if (m.userData.noOcclude) continue
    if (im.isInstancedMesh) {
      for (let i = 0; i < im.count; i++) {
        im.getMatrixAt(i, inst)
        const b = makeBox(wm.multiplyMatrices(m.matrixWorld, inst), m.geometry, { m, i })
        if (b) boxes.push(b)
      }
    } else {
      const b = makeBox(m.matrixWorld, m.geometry, m)
      if (b) boxes.push(b)
    }
  }

  const rays = raySet(LOOK.aoRays)
  const p = new Vector3(), n = new Vector3()
  const tess = new TessellateModifier(TESS_EDGE, 7)
  let verts = 0
  for (const m of meshes) {
    const im = m as InstancedMesh
    if (im.isInstancedMesh) {
      const data = new Float32Array(im.count * 2)
      for (let i = 0; i < im.count; i++) {
        im.getMatrixAt(i, inst)
        wm.multiplyMatrices(m.matrixWorld, inst)
        p.setFromMatrixPosition(wm)
        if (p.length() > BAKE_RADIUS) continue
        const owner = boxes.find(b => (b.owner as { m?: Mesh; i?: number }).m === m && (b.owner as { i?: number }).i === i)?.owner ?? im
        // a whole prop gets one value: sampled just above its centre, facing up + toward the lamps
        p.y = Math.max(p.y, 0.05)
        const [occ, lamp] = sample(p, n.set(0, 1, 0), boxes, owner, rays)
        data[i * 2] = occ * 0.8; data[i * 2 + 1] = lamp * 0.8
        verts++
      }
      m.geometry = m.geometry.clone()
      m.geometry.setAttribute('aBake', new InstancedBufferAttribute(data, 2))
      continue
    }
    const center = new Vector3().setFromMatrixPosition(m.matrixWorld)
    if (center.length() > BAKE_RADIUS + 15 || m.userData.noBake) continue
    let geo = m.geometry
    if (!m.userData.noTess) {
      if (!geo.boundingSphere) geo.computeBoundingSphere()
      const worldR = geo.boundingSphere!.radius * Math.max(...new Vector3().setFromMatrixScale(m.matrixWorld).toArray())
      geo = worldR > TESS_EDGE && worldR < 16 ? tess.modify(geo) : geo.clone()
    } else geo = geo.clone()
    const pos = geo.attributes.position, nor = geo.attributes.normal
    const nm = new Matrix4().copy(m.matrixWorld).invert().transpose()
    const data = new Float32Array(pos.count * 2)
    for (let i = 0; i < pos.count; i++) {
      n.fromBufferAttribute(nor, i).transformDirection(nm)
      p.fromBufferAttribute(pos, i).applyMatrix4(m.matrixWorld).addScaledVector(n, 0.02)
      const [occ, lamp] = sample(p, n, boxes, m, rays)
      data[i * 2] = occ; data[i * 2 + 1] = lamp
    }
    verts += pos.count
    geo.setAttribute('aBake', new BufferAttribute(data, 2))
    m.geometry = geo
  }
  if (import.meta.env.DEV) console.info(`[retro] baked lighting: ${verts} vertices, ${boxes.length} occluders, ${(performance.now() - t0).toFixed(0)} ms`)
}
