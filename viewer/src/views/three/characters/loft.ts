// Lofted low-poly forms: a stack of horizontal cross-sections (ellipses or rounded rectangles)
// skinned into one indexed geometry with SMOOTH normals (the ring seam is shared, so there is no
// lighting crease down the back). Used for every organic part of the skaters (torso, head, limbs,
// shoes) and for the procedural hands (grab/hand.ts).
//
// Cross-section angle θ: 0 = +z (front), π/2 = +x. `sq` < 1 squares the section off (superellipse),
// e.g. 0.6 for a shoe sole, 1 = ellipse. Caps: `bottom` / `top` are the apex offset of a fan cap
// (0 = flat, >0 domed outward), `false` leaves the end open. A cap is creased against the sides
// (hard rim, e.g. a sole) unless `smoothCaps` is set (round ends, e.g. a fingertip).
// `arcSeg: [k0, k1]` builds only columns k0..k1 of the full section: an open patch that lies exactly
// on (a scaled copy of) the same section — prints, pockets, stripes that follow the body.
// Folds: a ring's `w` (relative amplitude) ripples its radius as 1 + w·cos(waves·θ + wavePh), so
// baggy cloth gets soft vertical ridges (and a wavy silhouette the ink hull picks up).
import { BufferGeometry, Float32BufferAttribute, Uint16BufferAttribute, Uint32BufferAttribute } from 'three'

export interface Ring { y: number; rx: number; rz?: number; x?: number; z?: number; w?: number }

export interface LoftOpts {
  seg?: number
  sq?: number
  phase?: number               // default π/seg: a flat face looks forward
  bottom?: number | false
  top?: number | false
  smoothCaps?: boolean
  arcSeg?: [number, number]
  inside?: boolean             // flip faces (a lining seen from inside a sleeve / hood)
  waves?: number               // fold ridges around the section (with ring.w), default 5
  wavePh?: number
}

const sgnPow = (v: number, e: number) => Math.sign(v) * Math.pow(Math.abs(v), e)

/** Point on ring `r` at column angle θ. */
export function ringPoint(r: Ring, th: number, sq = 1, waves = 5, wavePh = 0): [number, number, number] {
  const k = r.w ? 1 + r.w * Math.cos(waves * th + wavePh) : 1
  return [(r.x ?? 0) + k * r.rx * sgnPow(Math.sin(th), sq), r.y, (r.z ?? 0) + k * (r.rz ?? r.rx) * sgnPow(Math.cos(th), sq)]
}

/** Linear interpolation of a ring stack at height y (for patches that follow a lofted body). */
export function ringAt(rings: Ring[], y: number, k = 1): Ring {
  const asc = rings[0].y <= rings[rings.length - 1].y ? rings : [...rings].reverse()
  let i = 0
  while (i < asc.length - 2 && asc[i + 1].y < y) i++
  const a = asc[i], b = asc[i + 1]
  const t = Math.max(0, Math.min(1, (y - a.y) / ((b.y - a.y) || 1)))
  const l = (p: number, q: number) => p + (q - p) * t
  return { y, rx: l(a.rx, b.rx) * k, rz: l(a.rz ?? a.rx, b.rz ?? b.rx) * k, x: l(a.x ?? 0, b.x ?? 0), z: l(a.z ?? 0, b.z ?? 0), w: l(a.w ?? 0, b.w ?? 0) }
}

/** z of the front-most face of a ring (where flat decals sit). */
export function frontZ(r: Ring, seg: number, sq = 1, phase = Math.PI / seg) {
  return (r.z ?? 0) + (r.rz ?? r.rx) * Math.pow(Math.cos(phase), sq) - 0.0005
}

export function loftGeo(rings: Ring[], o: LoftOpts = {}): BufferGeometry {
  const seg = o.seg ?? 8, sq = o.sq ?? 1, phase = o.phase ?? Math.PI / seg
  const wv = o.waves ?? 5, wp = o.wavePh ?? 0
  const step = (Math.PI * 2) / seg
  const arc = o.arcSeg
  const cols = arc ? arc[1] - arc[0] + 1 : seg
  const wrap = !arc
  const pos: number[] = []
  const idx: number[] = []
  // faces are wound for rings in ascending y; flip if they descend or for linings
  const flip = (rings[rings.length - 1].y < rings[0].y) !== !!o.inside
  const tri = (a: number, b: number, c: number) => { if (flip) idx.push(a, c, b); else idx.push(a, b, c) }
  const th = (j: number) => phase + ((arc ? arc[0] : 0) + j) * step
  for (const r of rings) for (let j = 0; j < cols; j++) pos.push(...ringPoint(r, th(j), sq, wv, wp))
  const n = wrap ? cols : cols - 1
  for (let i = 0; i < rings.length - 1; i++) for (let j = 0; j < n; j++) {
    const a = i * cols + j, b = i * cols + ((j + 1) % cols), c = a + cols, d = b + cols
    tri(a, b, c); tri(b, d, c)
  }
  const cap = (ri: number, off: number, top: boolean) => {
    const r = rings[ri]
    let base = ri * cols
    if (!o.smoothCaps) {            // duplicate the rim so the cap gets its own (creased) normals
      base = pos.length / 3
      for (let j = 0; j < cols; j++) pos.push(...ringPoint(r, th(j), sq, wv, wp))
    }
    const up = (rings[rings.length - 1].y >= rings[0].y) === top ? 1 : -1
    const apex = pos.length / 3
    pos.push(r.x ?? 0, r.y + up * off, r.z ?? 0)
    for (let j = 0; j < n; j++) {
      const a = base + j, b = base + ((j + 1) % cols)
      if (top) tri(a, b, apex); else tri(b, a, apex)
    }
  }
  if (o.bottom !== undefined && o.bottom !== false) cap(0, o.bottom, false)
  if (o.top !== undefined && o.top !== false) cap(rings.length - 1, o.top, true)
  const g = new BufferGeometry()
  g.setAttribute('position', new Float32BufferAttribute(pos, 3))
  g.setIndex(pos.length / 3 > 65535 ? new Uint32BufferAttribute(idx, 1) : new Uint16BufferAttribute(idx, 1))
  g.computeVertexNormals()
  return g
}

/** Reverse winding and normals (inner linings). Works on indexed or non-indexed geometry. */
export function flipFaces(g: BufferGeometry): BufferGeometry {
  if (g.index) {
    const a = g.index.array
    for (let i = 0; i < a.length; i += 3) { const t = a[i + 1]; a[i + 1] = a[i + 2]; a[i + 2] = t }
    g.index.needsUpdate = true
  } else {
    for (const name of Object.keys(g.attributes)) {
      const at = g.attributes[name], s = at.itemSize, arr = at.array as Float32Array
      for (let i = 0; i < at.count; i += 3) for (let c = 0; c < s; c++) {
        const t = arr[(i + 1) * s + c]; arr[(i + 1) * s + c] = arr[(i + 2) * s + c]; arr[(i + 2) * s + c] = t
      }
    }
  }
  const nrm = g.attributes.normal
  if (nrm) { const arr = nrm.array as Float32Array; for (let i = 0; i < arr.length; i++) arr[i] = -arr[i] }
  return g
}
