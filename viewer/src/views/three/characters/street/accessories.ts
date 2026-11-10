// Rigid low-poly bits that ride on the skeleton: chunky shoes, alpha-tested hair cards, hats, belt,
// wallet chain, hood, necklace. Each is authored in the body's T-pose (model space) and re-based onto
// its bone by attachRigid, so it follows that bone exactly. Textures are small canvases, <= 256 px.
import {
  BufferGeometry, CanvasTexture, Color, DoubleSide, Float32BufferAttribute, type Bone, Matrix4, Mesh, MeshPhongMaterial,
  RepeatWrapping, SRGBColorSpace, Vector3,
} from 'three'
import { mergeGeometries } from 'three/examples/jsm/utils/BufferGeometryUtils.js'
import type { BodyMetrics, Outfit } from './outfits'

function rng(seed: number) { let s = seed >>> 0; return () => { s = (s * 1664525 + 1013904223) >>> 0; return s / 4294967296 } }

const canvasTex = (w: number, h: number, paint: (g: CanvasRenderingContext2D) => void) => {
  const c = document.createElement('canvas'); c.width = w; c.height = h
  paint(c.getContext('2d')!)
  const t = new CanvasTexture(c)
  t.colorSpace = SRGBColorSpace; t.wrapS = t.wrapT = RepeatWrapping; t.anisotropy = 2
  return t
}

function geo(pos: number[], uv: number[], idx: number[]): BufferGeometry {
  const g = new BufferGeometry()
  g.setAttribute('position', new Float32BufferAttribute(pos, 3))
  g.setAttribute('uv', new Float32BufferAttribute(uv, 2))
  g.setIndex(idx)
  g.computeVertexNormals()
  return g
}

/** Rings of `m` points -> a closed-around tube with u around, v along. */
function ringsToGeo(rings: Vector3[][], cap = false): BufferGeometry {
  const pos: number[] = [], uv: number[] = [], idx: number[] = []
  const m = rings[0].length
  rings.forEach((r, j) => {
    for (let i = 0; i <= m; i++) {
      const p = r[i % m]
      pos.push(p.x, p.y, p.z); uv.push(i / m, j / (rings.length - 1))
    }
  })
  for (let j = 0; j < rings.length - 1; j++) for (let i = 0; i < m; i++) {
    const a = j * (m + 1) + i, b = a + 1, c = a + m + 1, d = c + 1
    idx.push(a, b, c, b, d, c)
  }
  if (cap) {
    const last = rings[rings.length - 1]
    const center = last.reduce((s, p) => s.add(p), new Vector3()).divideScalar(m)
    const ci = pos.length / 3
    pos.push(center.x, center.y, center.z); uv.push(0.5, 1)
    const base = (rings.length - 1) * (m + 1)
    for (let i = 0; i < m; i++) idx.push(base + i, base + i + 1, ci)
  }
  return geo(pos, uv, idx)
}

/** Makes `mesh` (geometry in model-space T-pose) a child of `bone`, staying where it was at rest. */
export function attachRigid(mesh: Mesh, bone: Bone, boneRestWorld: Matrix4) {
  mesh.geometry.applyMatrix4(new Matrix4().copy(boneRestWorld).invert())
  mesh.frustumCulled = false
  bone.add(mesh)
}

const phong = (o: { map?: CanvasTexture; color?: string; shininess?: number; specular?: string; alphaTest?: number; double?: boolean }) => new MeshPhongMaterial({
  map: o.map ?? null, color: new Color(o.color ?? '#ffffff'), shininess: o.shininess ?? 6, specular: new Color(o.specular ?? '#1a1a1a'),
  alphaTest: o.alphaTest ?? 0, side: o.double ? DoubleSide : 0,
})

// ------------------------------------------------------------------------------------------ shoes

/** A chunky skate shoe around the foot, with a sock stub inside the pant hem. u: 0 inner sole edge -> .5 top -> 1 outer sole edge. */
export function buildShoe(o: Outfit, side: 1 | -1): { geometry: BufferGeometry; material: MeshPhongMaterial } {
  const X = side * 0.114, Z0 = -0.135, Z1 = 0.175, H = 0.125
  const rings: Vector3[][] = []
  const ST = 11
  for (let j = 0; j <= ST; j++) {
    const t = j / ST, z = Z0 + (Z1 - Z0) * t
    // width + height profile: rounded heel, collar, fat rounded toe box
    const w = 0.068 * (t < 0.14 ? 0.62 + (t / 0.14) * 0.38 : t > 0.82 ? Math.sqrt(Math.max(0.12, 1 - ((t - 0.82) / 0.18) ** 2)) : 1)
    const top = t < 0.46 ? H * (t < 0.1 ? 0.82 + t * 1.8 : 1) : H - (H - 0.078) * Math.min(1, (t - 0.46) / 0.3) * (t > 0.76 ? 1 : 1)
    const toe = t > 0.86 ? 1 - (t - 0.86) * 2.4 : 1
    const r: Vector3[] = []
    const SEG = 12
    for (let i = 0; i < SEG; i++) {
      const a = (i / SEG) * Math.PI * 2 - Math.PI / 2       // starts underneath
      const c = Math.cos(a), sn = Math.sin(a)
      const sole = sn < -0.2
      // flat bottom, a thick sole bulging past the upper, then the upper domed over it
      const y = sole ? (sn < -0.6 ? 0.001 : 0.042 * ((sn + 0.6) / 0.4)) : 0.042 + ((sn + 0.2) / 1.2) * (top * toe - 0.042)
      r.push(new Vector3(X + c * w * (sole ? 1.1 : 1 - 0.1 * Math.max(0, sn)) * side, y, z))
    }
    rings.push(r)
  }
  // flip winding for the mirrored shoe by reversing ring order, so normals still face out
  if (side > 0) for (const r of rings) r.reverse()
  const shoe = ringsToGeo([rings[0].map(p => new Vector3(X, p.y * 0.4 + 0.03, p.z - 0.004)), ...rings], true)
  const S = o.shoe
  const map = canvasTex(256, 256, g => {
    // canvas x = around (0.5 = up top... see ring param), canvas y = heel (bottom) -> toe (top)
    const W = 256
    g.fillStyle = S.upper; g.fillRect(0, 0, W, W)
    // the sole wraps the lowest part of the ring: u near 0 / 1 and around 0.5+-... painted as bands
    const soleBand = (u0: number, u1: number, col: string) => { g.fillStyle = col; g.fillRect(u0 * W, 0, (u1 - u0) * W, W) }
    soleBand(0, 0.2, S.sole); soleBand(0.8, 1, S.sole)
    soleBand(0.2, 0.226, S.accent); soleBand(0.774, 0.8, S.accent)
    g.fillStyle = 'rgba(0,0,0,0.28)'; g.fillRect(0.11 * W, 0, 0.012 * W, W); g.fillRect(0.878 * W, 0, 0.012 * W, W)   // sole's seam line
    // panels: heel counter, side stripes, toe cap
    g.fillStyle = S.panel
    g.fillRect(W * 0.226, 0, W * 0.548, W * 0.22)     // heel counter (flipY off: canvas top = v 0 = heel)
    g.fillRect(W * 0.226, W * 0.82, W * 0.548, W * 0.18) // toe cap
    soleBand(0.226, 0.262, S.panel); soleBand(0.738, 0.774, S.panel)
    g.fillStyle = S.accent
    for (let i = 0; i < 4; i++) { g.fillRect(W * 0.27, W * (0.3 + i * 0.09), W * 0.1, W * 0.05); g.fillRect(W * 0.63, W * (0.3 + i * 0.09), W * 0.1, W * 0.05) }
    // tongue + laces straight up the top (u = .5)
    g.fillStyle = 'rgba(0,0,0,0.35)'; g.fillRect(W * 0.4, W * 0.42, W * 0.2, W * 0.38)
    g.fillStyle = S.lace
    for (let i = 0; i < 6; i++) g.fillRect(W * 0.41, W * (0.44 + i * 0.062), W * 0.18, W * 0.028)
    // creases + scuffs
    const r = rng(19)
    for (let i = 0; i < 90; i++) { g.fillStyle = `rgba(20,16,12,${r() * 0.16})`; g.fillRect(r() * W, r() * W, 2 + r() * 18, 1 + r() * 2) }
  })
  map.flipY = false
  // sock stub: fills the hem from below when the leg is seen from low down
  const sock: Vector3[][] = []
  for (const y of [0.1, 0.21]) {
    const r: Vector3[] = []
    for (let i = 0; i < 6; i++) { const a = (i / 6) * Math.PI * 2; r.push(new Vector3(X + Math.cos(a) * 0.05, y, -0.075 + Math.sin(a) * 0.05)) }
    sock.push(side > 0 ? r : r.reverse())
  }
  const sockGeo = ringsToGeo(sock)
  // the sock samples a dark corner of the shoe texture's collar
  const su = sockGeo.attributes.uv; for (let i = 0; i < su.count; i++) su.setXY(i, 0.5, 0.44)
  return { geometry: mergeGeometries([shoe, sockGeo])!, material: phong({ map, shininess: 22, specular: '#303030' }) }
}

// ------------------------------------------------------------------------------------------ hair

export interface HeadFit { c: Vector3; r: Vector3 }   // ellipsoid around the skull (model space)

const hairMats = new Map<string, MeshPhongMaterial>()
function hairMaterial(color: string): MeshPhongMaterial {
  const hit = hairMats.get(color); if (hit) return hit
  const r = rng(61)
  const base = new Color(color)
  const hex = (k: number) => '#' + base.clone().multiplyScalar(k).getHexString()
  // strands on transparent, thinning to wisps at the bottom of the card. v: 0 root (canvas top) -> 1 tip
  const map = canvasTex(128, 256, g => {
    for (let i = 0; i < 90; i++) {
      const x = r() * 128, w = 2 + r() * 7, len = 256 * (0.55 + r() * 0.45)
      g.fillStyle = hex(0.5 + r() * 1.5)
      g.beginPath(); g.moveTo(x - w / 2, 0); g.lineTo(x + w / 2, 0); g.lineTo(x + w * 0.1, len); g.closePath(); g.fill()
    }
  })
  map.flipY = false
  const m = phong({ map, alphaTest: 0.5, double: true, shininess: 28, specular: '#3a3a3a' })
  hairMats.set(color, m)
  return m
}

/** One ribbon: follows `path` (points), width across along `side`. */
function ribbon(path: Vector3[], width: (t: number) => number, across: (i: number) => Vector3): { p: number[]; uv: number[]; idx: number[] } {
  const p: number[] = [], uv: number[] = [], idx: number[] = []
  path.forEach((q, i) => {
    const t = i / (path.length - 1), w = width(t), s = across(i)
    p.push(q.x - s.x * w, q.y - s.y * w, q.z - s.z * w, q.x + s.x * w, q.y + s.y * w, q.z + s.z * w)
    uv.push(0, t, 1, t)
  })
  for (let i = 0; i < path.length - 1; i++) { const a = i * 2; idx.push(a, a + 1, a + 2, a + 1, a + 3, a + 2) }
  return { p, uv, idx }
}

function mergeRibbons(rs: { p: number[]; uv: number[]; idx: number[] }[]): BufferGeometry {
  const p: number[] = [], uv: number[] = [], idx: number[] = []
  for (const r of rs) {
    const base = p.length / 3
    p.push(...r.p); uv.push(...r.uv); idx.push(...r.idx.map(i => i + base))
  }
  return geo(p, uv, idx)
}

/** A point on the skull ellipsoid: azimuth `az` (0 = back, +pi/2 = the skater's left), `lat` (0 top .. pi/2 equator). */
function onSkull(f: HeadFit, az: number, lat: number, lift = 1.04): Vector3 {
  const s = Math.sin(lat), c = Math.cos(lat)
  return new Vector3(f.c.x + Math.sin(az) * s * f.r.x * lift, f.c.y + c * f.r.y * lift, f.c.z - Math.cos(az) * s * f.r.z * lift)
}

/** Strips running from the crown down over the skull, then hanging by `drop` metres. */
function skullStrips(f: HeadFit, opts: { n: number; azFrom: number; azTo: number; latEnd: number; drop: number; width: number; seed: number; lift?: number; latStart?: number }) {
  const r = rng(opts.seed)
  const out = []
  for (let i = 0; i < opts.n; i++) {
    const az = opts.azFrom + (opts.azTo - opts.azFrom) * ((i + 0.5) / opts.n) + (r() - 0.5) * 0.07
    const path: Vector3[] = []
    const SEG = 5
    for (let k = 0; k <= SEG; k++) {
      const t = k / SEG
      const start = opts.latStart ?? 0.12
      const q = onSkull(f, az, start + (opts.latEnd - start) * Math.min(1, t * 1.25), opts.lift ?? 1.05)
      if (t > 0.8) q.y -= opts.drop * (t - 0.8) / 0.2
      path.push(q)
    }
    const tangent = new Vector3(Math.cos(az), 0, Math.sin(az))
    out.push(ribbon(path, t => opts.width * (1 - t * 0.35), () => tangent))
  }
  return out
}

/** A plaited rope: two crossed ribbons, narrowing to a tied end. */
function plait(from: Vector3, to: Vector3, sag: Vector3, width: number) {
  const path: Vector3[] = []
  for (let k = 0; k <= 7; k++) {
    const t = k / 7
    path.push(new Vector3().lerpVectors(from, to, t).addScaledVector(sag, Math.sin(t * Math.PI)))
  }
  return [
    ribbon(path, t => width * (1 - 0.45 * t) * (0.82 + 0.3 * Math.abs(Math.sin(t * 20))), () => new Vector3(1, 0, 0)),
    ribbon(path, t => width * (1 - 0.45 * t) * (0.82 + 0.3 * Math.abs(Math.cos(t * 20))), () => new Vector3(0, 0, 1)),
  ]
}

export function buildHair(o: Outfit, f: HeadFit): { geometry: BufferGeometry; material: MeshPhongMaterial } | null {
  const hatted = o.hat !== 'none'
  const rs: ReturnType<typeof ribbon>[] = []
  const PI = Math.PI
  switch (o.hairKind) {
    case 'short':
      // a fade: short crown cards only
      rs.push(...skullStrips(f, { n: 30, azFrom: -PI, azTo: PI, latEnd: 1.25, drop: 0.0, width: 0.034, seed: 5, latStart: 0.02 }))
      break
    case 'locs': {
      // shoulder-length locs all round the back and sides, a few falling forward
      rs.push(...skullStrips(f, { n: 26, azFrom: -PI * 0.68, azTo: PI * 0.68, latEnd: 1.55, drop: 0.2, width: 0.03, seed: 9 }))
      rs.push(...skullStrips(f, { n: 24, azFrom: -PI, azTo: PI, latEnd: 1.05, drop: 0, width: 0.036, seed: 11, latStart: 0.02 }))
      break
    }
    case 'braids': {
      // pulled-back hair, with two plaits from behind the ears down over the front of the shoulders
      if (!hatted) rs.push(...skullStrips(f, { n: 20, azFrom: -PI * 0.75, azTo: PI * 0.75, latEnd: 1.2, drop: 0, width: 0.032, seed: 21 }))
      else rs.push(...skullStrips(f, { n: 18, azFrom: -PI * 0.6, azTo: PI * 0.6, latEnd: 1.75, drop: 0, width: 0.03, seed: 22, lift: 1.03, latStart: 1.3 }))
      for (const s of [1, -1]) {
        const from = new Vector3(f.c.x + s * f.r.x * 0.82, f.c.y - f.r.y * 0.28, f.c.z - f.r.z * 0.25)
        const to = new Vector3(s * 0.13, f.c.y - 0.42, 0.075)
        rs.push(...plait(from, to, new Vector3(s * 0.04, -0.01, 0.02), 0.032))
      }
      break
    }
    case 'ponytail': {
      rs.push(...skullStrips(f, { n: 24, azFrom: -PI * 0.72, azTo: PI * 0.72, latEnd: 1.72, drop: 0, width: 0.03, seed: 31, latStart: hatted ? 1.25 : 0.12, lift: 1.03 }))
      // pulled through the back of the cap, swinging to the shoulder blades
      const from = onSkull(f, 0, 1.15, 1.02)
      for (const s of [-1, 0, 1]) {
        const path: Vector3[] = []
        for (let k = 0; k <= 7; k++) {
          const t = k / 7
          path.push(new Vector3(from.x + s * 0.012 + s * 0.02 * t, from.y + 0.01 - 0.42 * t * t - 0.09 * t, from.z - 0.04 * Math.sin(t * PI) - 0.05 * t))
        }
        rs.push(ribbon(path, t => 0.04 * (1 - 0.72 * t) * (t < 0.1 ? 0.5 + t * 5 : 1), () => new Vector3(1, 0, s * 0.4).normalize()))
        rs.push(ribbon(path, t => 0.032 * (1 - 0.72 * t) * (t < 0.1 ? 0.5 + t * 5 : 1), () => new Vector3(0, 0.1, 1).normalize()))
      }
      break
    }
  }
  if (!rs.length) return null
  return { geometry: mergeRibbons(rs), material: hairMaterial(o.hair) }
}

// ------------------------------------------------------------------------------------------ hats

export function buildHat(o: Outfit, f: HeadFit): { geometry: BufferGeometry; material: MeshPhongMaterial } | null {
  if (o.hat === 'none') return null
  const seg = 12
  const dome = (lats: number[], lift: number[], tilt = 0.0): Vector3[][] => lats.map((lat, j) => {
    const r: Vector3[] = []
    for (let i = 0; i < seg; i++) {
      const a = (i / seg) * Math.PI * 2
      const s = Math.sin(lat), c = Math.cos(lat)
      const q = new Vector3(f.c.x + Math.sin(a) * s * f.r.x * lift[j], f.c.y + c * f.r.y * lift[j] + (Math.cos(a) * s) * tilt, f.c.z + Math.cos(a) * s * f.r.z * lift[j])
      r.push(q)
    }
    return r
  })
  if (o.hat === 'beanie') {
    // bottom up: cuff, knit dome, slouch at the crown
    const rings = dome([1.72, 1.56, 1.5, 1.3, 1.0, 0.7, 0.4, 0.14], [1.1, 1.1, 1.05, 1.06, 1.08, 1.12, 1.16, 1.18], 0.05)
    const map = canvasTex(128, 128, g => {
      g.fillStyle = o.hatColor; g.fillRect(0, 0, 128, 128)
      for (let x = 0; x < 128; x += 6) { g.fillStyle = 'rgba(0,0,0,0.22)'; g.fillRect(x, 0, 2, 128); g.fillStyle = 'rgba(255,255,255,0.08)'; g.fillRect(x + 3, 0, 1, 128) }
      // v runs cuff (0) -> crown (1); flipY means canvas top = crown
      g.fillStyle = o.hatTrim; g.fillRect(0, 128 - 36, 128, 7)
      g.fillStyle = 'rgba(0,0,0,0.3)'; g.fillRect(0, 128 - 30, 128, 3)
      const crown = g.createLinearGradient(0, 0, 0, 60); crown.addColorStop(0, 'rgba(0,0,0,0.25)'); crown.addColorStop(1, 'rgba(0,0,0,0)')
      g.fillStyle = crown; g.fillRect(0, 0, 128, 60)
    })
    return { geometry: ringsToGeo(rings, true), material: phong({ map, shininess: 3 }) }
  }
  // trucker cap: crown, plus a brim sticking out the front (model +z)
  const rings = dome([1.4, 1.2, 0.95, 0.65, 0.3], [1.07, 1.08, 1.09, 1.1, 1.1], 0.06)
  const crown = ringsToGeo(rings, true)
  const fp: number[] = [], fuv: number[] = [], idx: number[] = []
  const y0 = f.c.y + Math.cos(1.4) * f.r.y * 1.07 + 0.005, zf = f.c.z + f.r.z * 0.92
  for (let i = 0; i <= 6; i++) {
    const t = i / 6, a = (t - 0.5) * 2.1
    fp.push(f.c.x + Math.sin(a) * f.r.x * 1.03, y0, f.c.z + Math.cos(a) * f.r.z * 1.03)
    fp.push(f.c.x + Math.sin(a) * f.r.x * 1.1, y0 - 0.02, zf + Math.cos(a) * 0.1 + 0.02)
    fuv.push(t * 0.2, 0.92, t * 0.2, 0.98)
  }
  for (let i = 0; i < 6; i++) { const a = i * 2; idx.push(a, a + 1, a + 2, a + 1, a + 3, a + 2) }
  const brim = geo(fp, fuv, idx)
  const map = canvasTex(128, 128, g => {
    g.fillStyle = o.hatTrim; g.fillRect(0, 0, 128, 128)                    // mesh panels at the back
    g.fillStyle = 'rgba(0,0,0,0.2)'; for (let y = 0; y < 128; y += 4) for (let x = (y / 4) % 2 ? 0 : 2; x < 128; x += 4) g.fillRect(x, y, 2, 2)
    g.fillStyle = o.hatColor; g.fillRect(0, 0, 128, 128); g.clearRect(0, 0, 0, 0)
    g.fillStyle = o.hatTrim; g.fillRect(128 * 0.25, 128 * 0.3, 128 * 0.12, 128 * 0.5); g.fillRect(128 * 0.63, 128 * 0.3, 128 * 0.12, 128 * 0.5)
    g.fillStyle = o.hatColor; g.fillRect(0, 0, 128 * 0.2, 128 * 0.1)      // brim
  })
  return { geometry: mergeGeometries([crown, brim])!, material: phong({ map, shininess: 5, double: true }) }
}

// ------------------------------------------------------------------------------------------ waist

/** Everything hung on the pelvis: belt, buckle, loose straps, wallet chain. All in one mesh. */
export function buildWaist(o: Outfit, b: BodyMetrics): { geometry: BufferGeometry; material: MeshPhongMaterial } | null {
  const parts: BufferGeometry[] = []
  const y = b.waistY - 0.035
  const rx = b.hipRx * 1.075, rz = 0.163, cz = -0.035
  // belt ring
  const beltR: Vector3[][] = [y - 0.022, y + 0.022].map(yy => {
    const r: Vector3[] = []
    for (let i = 0; i < 12; i++) { const a = (i / 12) * Math.PI * 2; r.push(new Vector3(Math.sin(a) * rx, yy, cz + Math.cos(a) * rz)) }
    return r
  })
  const belt = ringsToGeo(beltR)
  const ub = belt.attributes.uv; for (let i = 0; i < ub.count; i++) ub.setXY(i, 0.1, 0.1)
  parts.push(belt)
  // buckle
  const bx = 0.026, bp = [[-bx, y - 0.02], [bx, y - 0.02], [-bx, y + 0.02], [bx, y + 0.02]]
  const bz = cz + rz + 0.004
  const buckle = geo(bp.flatMap(([x, q]) => [x, q, bz]), [0.3, 0.1, 0.4, 0.1, 0.3, 0.2, 0.4, 0.2], [0, 1, 2, 1, 3, 2])
  parts.push(buckle)
  const strap = (x0: number, z0: number, x1: number, y1: number, z1: number, u: number, w = 0.016) => {
    const path: Vector3[] = []
    for (let k = 0; k <= 6; k++) {
      const t = k / 6
      path.push(new Vector3(x0 + (x1 - x0) * t, y - 0.02 + (y1 - y) * t + 0.03 * Math.sin(t * Math.PI), z0 + (z1 - z0) * t + 0.03 * Math.sin(t * Math.PI * 2)))
    }
    const r = ribbon(path, () => w, () => new Vector3(0.6, 0.05, 0.6).normalize())
    const g = geo(r.p, r.uv.map((v, i) => i % 2 ? 0.05 + v * 0.1 : u), r.idx)
    parts.push(g)
  }
  if (o.straps) {
    // two long loose straps swinging off the hips
    strap(rx * 0.75, cz + rz * 0.7, 0.2, y - 0.5, cz + 0.1, 0.12)
    strap(-rx * 0.8, cz + rz * 0.5, -0.17, y - 0.42, cz + 0.13, 0.12)
    strap(rx * 0.9, cz - rz * 0.3, 0.25, y - 0.36, cz - 0.11, 0.12)
  }
  if (o.chain) {
    // a slack loop from the front belt loop round to the back pocket
    const path: Vector3[] = []
    for (let k = 0; k <= 9; k++) {
      const t = k / 9
      const a = 0.55 + t * 1.75
      path.push(new Vector3(Math.sin(a) * (rx + 0.012), y - 0.015 - Math.sin(t * Math.PI) * 0.21, cz + Math.cos(a) * (rz + 0.018)))
    }
    const r = ribbon(path, () => 0.006, () => new Vector3(0, 0, 1))
    parts.push(geo(r.p, r.uv.map((v, i) => i % 2 ? 0.3 + v * 0.3 : 0.45), r.idx))
    const r2 = ribbon(path, () => 0.006, () => new Vector3(0, 1, 0))
    parts.push(geo(r2.p, r2.uv.map((v, i) => i % 2 ? 0.3 + v * 0.3 : 0.45), r2.idx))
  }
  const map = canvasTex(128, 128, g => {
    g.fillStyle = o.belt; g.fillRect(0, 0, 128, 128)
    const r = rng(2)
    for (let i = 0; i < 120; i++) { g.fillStyle = `rgba(255,240,210,${r() * 0.09})`; g.fillRect(r() * 128, r() * 128, 4 + r() * 12, 1) }
    // canvas (flipY false): u right, v down. Buckle at u .3-.4, v .1-.2
    g.fillStyle = '#c8c4ba'; g.fillRect(38, 13, 13, 13)
    g.fillStyle = '#6c6860'; g.fillRect(41, 16, 7, 7)
    // chain: bright links
    for (let yy = 38; yy < 78; yy += 3) { g.fillStyle = yy % 2 ? '#e0dcd2' : '#8c8880'; g.fillRect(50, yy, 16, 2) }
  })
  map.flipY = false
  return { geometry: mergeGeometries(parts.map(p => { const q = p.index ? p.toNonIndexed() : p; q.computeVertexNormals(); return q }))!, material: phong({ map, shininess: 30, specular: '#404040', double: true }) }
}

/** A hood lying on the shoulders (hoodies), or a chain necklace. Rides on the top of the spine. */
export function buildCollar(o: Outfit, b: BodyMetrics): { geometry: BufferGeometry; material: MeshPhongMaterial } | null {
  const parts: BufferGeometry[] = []
  const y = b.neckY
  if (o.top === 'hoodie') {
    const rings: Vector3[][] = []
    const seg = 10
    // the bunched collar: wide ring round the neck, deep at the back
    for (let j = 0; j <= 4; j++) {
      const t = j / 4, r: Vector3[] = []
      for (let i = 0; i < seg; i++) {
        const a = (i / seg) * Math.PI * 2, back = Math.max(0, -Math.cos(a))
        const R = (0.078 + 0.042 * Math.sin(t * Math.PI) + 0.02 * back) * (b.torsoRx / 0.22 * 0.4 + 0.6)
        r.push(new Vector3(Math.sin(a) * R, y - 0.1 + 0.1 * t - 0.09 * back * Math.sin(t * Math.PI), -0.03 + Math.cos(a) * R * 0.95 - 0.05 * back * Math.sin(t * Math.PI)))
      }
      rings.push(r)
    }
    parts.push(ringsToGeo(rings))
  }
  const map = canvasTex(64, 64, g => {
    g.fillStyle = o.topColor; g.fillRect(0, 0, 64, 64)
    g.fillStyle = 'rgba(0,0,0,0.25)'; for (let i = 0; i < 6; i++) g.fillRect(0, i * 11 + 4, 64, 3)
    const grad = g.createLinearGradient(0, 0, 0, 64); grad.addColorStop(0, 'rgba(0,0,0,0.42)'); grad.addColorStop(1, 'rgba(0,0,0,0.05)')
    g.fillStyle = grad; g.fillRect(0, 0, 64, 64)
  })
  if (o.necklace) {
    const path: Vector3[] = []
    for (let k = 0; k <= 10; k++) {
      const t = k / 10, a = (t - 0.5) * 3.4
      path.push(new Vector3(Math.sin(a) * 0.075, y - 0.03 - Math.cos(a) * 0.075, 0.02 + Math.cos(a) * 0.075))
    }
    const r = ribbon(path, () => 0.0045, () => new Vector3(0, 0.4, 1).normalize())
    const g = geo(r.p, r.uv.map(() => 0.5), r.idx)
    if (o.top !== 'hoodie') { parts.push(g); return { geometry: g, material: phong({ color: '#d8c070', shininess: 60, specular: '#606060', double: true }) } }
  }
  if (!parts.length) return null
  return { geometry: mergeGeometries(parts)!, material: phong({ map, shininess: 4, double: true }) }
}

/** Fit the skull ellipsoid from the head-dominated vertices above the jaw. */
export function fitHead(positions: Float32Array, cls: Uint8Array, headCls: number, b: BodyMetrics): HeadFit {
  let x0 = 1e9, x1 = -1e9, y1 = -1e9, z0 = 1e9, z1 = -1e9
  const jaw = b.headY + 0.045
  for (let i = 0; i < cls.length; i++) {
    if (cls[i] !== headCls) continue
    const x = positions[i * 3], y = positions[i * 3 + 1], z = positions[i * 3 + 2]
    if (y < jaw) continue
    if (x < x0) x0 = x; if (x > x1) x1 = x; if (y > y1) y1 = y; if (z < z0) z0 = z; if (z > z1) z1 = z
  }
  const c = new Vector3((x0 + x1) / 2, jaw + (y1 - jaw) * 0.42, (z0 + z1) / 2)
  return { c, r: new Vector3((x1 - x0) / 2, y1 - c.y, (z1 - z0) / 2) }
}

