// Paints each skater's 512 px texture, PS2 style: everything is IN the texture, including the shading.
// Each texel of the body's UV layout is mapped back to its 3D rest position (rasterising the triangles in UV
// space) and asked "what are you?": skin from the pack's texture (re-toned), or a garment with its
// fold shadows, seams, pockets, cuffs, prints and grime painted in. Painting in 3D means garments line up
// across the UV seams without hand-laying anything out.
import { CanvasTexture, Color, LinearFilter, LinearMipmapLinearFilter, SRGBColorSpace } from 'three'
import { LOOK } from '../../style.config'
import { paintPrint } from './prints'
import type { BodyMetrics, Outfit } from './outfits'
import { CLASSES, type Sculpted } from './sculpt'

const S = LOOK.maxTexture

// --- small value noise
function hash3(x: number, y: number, z: number) {
  const s = Math.sin(x * 127.1 + y * 311.7 + z * 74.7) * 43758.5453
  return s - Math.floor(s)
}
function noise3(x: number, y: number, z: number) {
  const xi = Math.floor(x), yi = Math.floor(y), zi = Math.floor(z)
  const f = (t: number) => t * t * (3 - 2 * t)
  const u = f(x - xi), v = f(y - yi), w = f(z - zi)
  const lx = (a: number, c: number) => a + (c - a) * u
  const ly = (a: number, c: number) => a + (c - a) * v
  return ly(lx(hash3(xi, yi, zi), hash3(xi + 1, yi, zi)), lx(hash3(xi, yi + 1, zi), hash3(xi + 1, yi + 1, zi))) * (1 - w)
    + ly(lx(hash3(xi, yi, zi + 1), hash3(xi + 1, yi, zi + 1)), lx(hash3(xi, yi + 1, zi + 1), hash3(xi + 1, yi + 1, zi + 1))) * w
}

const smooth = (a: number, b: number, x: number) => { const t = Math.max(0, Math.min(1, (x - a) / (b - a))); return t * t * (3 - 2 * t) }
const band = (a: number, b: number, x: number, soft = 0.004) => smooth(a - soft, a + soft, x) * (1 - smooth(b - soft, b + soft, x))
type RGB = [number, number, number]
const rgb = (hex: string): RGB => { const c = new Color(hex).convertLinearToSRGB(); return [c.r * 255, c.g * 255, c.b * 255] }
const mix = (a: RGB, b: RGB, k: number): RGB => [a[0] + (b[0] - a[0]) * k, a[1] + (b[1] - a[1]) * k, a[2] + (b[2] - a[2]) * k]
const mul = (a: RGB, k: number): RGB => [a[0] * k, a[1] * k, a[2] * k]

interface Texel {
  x: number; y: number; z: number      // rest pose, before sculpting
  sx: number; sy: number; sz: number   // sculpted: where hems really are (rest-pose hems are zigzag)
  nx: number; ny: number; nz: number; cls: string; u: number; v: number
}

function pantsPaint(t: Texel, o: Outfit, b: BodyMetrics, base: RGB, stitch: RGB): RGB {
  const { x, y, z, nx, nz } = t, ax = Math.abs(x), side = x >= 0 ? 1 : -1
  const th = Math.atan2(z + 0.035, (x - side * 0.118) * side)
  const stack = 1 - smooth(0.09, 0.42, y)
  // big soft worn patches (that blotchy console-texture look), then folds that follow the sculpted ones
  let k = 0.88 + 0.34 * (noise3(x * 7 + o.seed, y * 5, z * 7) - 0.5)
  k += 0.18 * smooth(0.5, 0.2, Math.abs(y - 0.55)) * (nz > 0 ? 1 : 0.3) * (noise3(x * 12, y * 3, z * 12) - 0.2)    // thigh + shin fade
  const f1 = Math.sin(y * 42 + th * 2 + o.seed), f2 = Math.sin(y * 17 - th * 3 + o.seed * 2), f3 = Math.sin(y * 9 + th * 5 + o.seed)
  k *= 1 + 0.26 * stack * f1 + 0.13 * f2 + 0.08 * f3
  // creases: behind the knee, the groin, down the inside of the leg
  k *= 1 - 0.42 * band(0.44, 0.58, y, 0.05) * (nz < -0.2 ? 1 : 0)
  k *= 1 - 0.5 * smooth(0.82, 0.92, y) * (1 - smooth(0.92, 1.0, y)) * (1 - smooth(0.02, 0.07, ax)) * (nz > -0.5 ? 1 : 0.6)
  k *= 1 - 0.32 * (1 - smooth(0.03, 0.09, ax)) * (1 - smooth(0.7, 0.85, y))
  // baked top light + dirty hem
  k *= 0.86 + 0.2 * t.ny + 0.06 * nz
  k *= 0.6 + 0.4 * smooth(0.05, 0.2, y)
  let c = mul(base, k)
  if (o.pantsKind === 'jeans') {
    c = mix(c, mul(base, 1.3), 0.5 * band(0.3, 0.8, y, 0.09) * (nz > 0.1 ? 1 : 0) * smooth(0.18, 0.08, Math.abs(ax - 0.125)))
  }
  // outside seam
  const outer = (x - side * 0.118) * side > 0 ? Math.abs(z + 0.035) : 1
  c = mix(c, stitch, 0.55 * (1 - smooth(0.004, 0.009, outer)) * smooth(0.1, 0.14, y))
  // cargo pocket on the outside of each thigh
  if (o.pantsKind !== 'jeans') {
    const pz = z + 0.035, inPocket = ax > 0.1 && (x * nx > 0) && Math.abs(pz) < 0.07 && y > 0.48 && y < 0.7
    if (inPocket) {
      const edge = Math.min(0.07 - Math.abs(pz), Math.min(y - 0.48, 0.7 - y))
      c = mul(c, 1.1)
      if (edge < 0.005) c = mix(c, stitch, 0.42)
      else if (edge < 0.011) c = mul(c, 0.86)
      if (y > 0.65 && y < 0.655) c = mul(c, 0.45)
      if (y < 0.5) c = mul(c, 0.75)
    }
  }
  // back pockets
  if (nz < -0.3 && y > 0.84 && y < 0.97 && ax > 0.05 && ax < 0.16) {
    const edge = Math.min(y - 0.84, 0.97 - y, ax - 0.05, 0.16 - ax)
    if (edge < 0.005) c = mix(c, stitch, 0.42)
  }
  // waistband + fly
  if (y > b.waistY - 0.055) c = mix(mul(c, 0.8), stitch, 0.08)
  if (nz > 0.2 && ax < 0.008 && y > 0.85) c = mul(c, 0.45)
  return c
}

function topPaint(t: Texel, o: Outfit, b: BodyMetrics, base: RGB, sleeve: RGB, trim: RGB, print: ImageData): RGB {
  const { x, y, z, nz, ny } = t, ax = Math.abs(x)
  const onArm = t.cls === 'uarm' || t.cls === 'farm' || (ax > b.shoulderRestX + 0.03 && y > b.armY - 0.1)
  const hoodie = o.top === 'hoodie'
  const useSleeve = o.top === 'layered' && ax > b.shoulderRestX + 0.2
  const col = useSleeve ? sleeve : base
  let k = 0.92 + 0.26 * (noise3(x * 8 + o.seed, y * 6, z * 8) - 0.5)
  if (onArm) {
    // wrinkles spiral down the sleeve, crease inside the elbow
    k *= 1 + 0.2 * Math.sin(ax * 48 + o.seed + (z - b.armZ > 0 ? 1 : 0)) + 0.1 * Math.sin(ax * 19 + (y - b.armY) * 40)
    const elbow = (b.shoulderRestX + b.wristX) / 2
    k *= 1 - 0.3 * band(elbow - 0.03, elbow + 0.03, ax, 0.02)
    k *= 0.86 + 0.24 * ny
  } else {
    // drapes from the chest toward the hem, deep shadow under the arms
    k *= 1 + 0.1 * Math.sin(x * 40 + y * 9 + o.seed) * (1 - smooth(b.waistY + 0.1, b.armY - 0.1, y))
    k *= 1 + (hoodie ? 0.12 : 0.07) * Math.sin(y * 34 + Math.atan2(z, x) * 3 + o.seed)
    k *= 1 - 0.42 * smooth(b.torsoRx * 0.62, b.torsoRx * 0.95, ax) * band(b.armY - 0.26, b.armY - 0.04, y, 0.05)
    k *= 0.84 + 0.2 * ny + 0.1 * nz
  }
  k *= 0.72 + 0.28 * smooth(b.waistY - 0.02, b.waistY + 0.12, y)      // falls off toward the hem
  let c = mul(col, k)
  if (!onArm) {
    // the print, front projected, with a smaller one between the shoulders
    const front = nz > 0.12
    const w = b.torsoRx * 1.6, h = (b.armY + 0.02) - (b.waistY + 0.02)
    let u = -1, v = -1
    if (front) { u = x / w + 0.5; v = 1 - (y - (b.waistY + 0.02)) / h }
    else if (nz < -0.3) { u = (-x / (w * 0.5) + 0.5); v = 1 - (y - (b.armY - 0.24)) / (h * 0.42) }
    if (u > 0 && u < 1 && v > 0 && v < 1) {
      const px = Math.floor(u * (print.width - 1)), py = Math.floor(v * (print.height - 1)), i = (py * print.width + px) * 4
      const a = print.data[i + 3] / 255
      if (a > 0) c = mix(c, mul([print.data[i], print.data[i + 1], print.data[i + 2]], 0.65 + 0.35 * k), a * 0.92)
    }
    // trim at the hem and collar
    const hemY = b.waistY - 0.004 - (hoodie ? 0.1 : 0.07)
    const ribTop = hemY + (hoodie ? 0.05 : 0.016)
    if (t.sy < ribTop) c = mul(mix(c, mul(col, 0.8), 0.8), 1 + 0.12 * Math.sin(ax * 300))
    if (!hoodie) {
      // a rib collar around the sculpted neckline
      const collar = Math.hypot(t.sx, t.sz + 0.03)
      if (t.sy > b.neckY - 0.12) c = mix(c, trim, smooth(0.128, 0.114, collar))
    }
    if (hoodie && nz > 0.1) {
      // kangaroo pocket + drawstrings
      const py = y - (b.waistY + 0.03), half = 0.13 - py * 0.35
      if (py > 0 && py < 0.17 && ax < half) {
        const edge = Math.min(py, 0.17 - py, half - ax)
        c = mul(c, edge < 0.006 ? 0.78 : 1.03)
      }
      if (y > b.armY - 0.2 && y < b.neckY - 0.03 && Math.abs(ax - 0.03) < 0.005) c = mix(c, trim, 0.85)
    }
  } else {
    // cuffs / sleeve bands
    const end = o.top === 'tee' ? b.shoulderRestX + 0.2 : b.wristX + 0.012
    const dist = end - Math.abs(t.sx)
    if (o.top === 'tee') { if (dist < 0.012) c = mix(c, trim, 0.9) }
    else if (dist < 0.05) c = mul(mix(c, trim, o.top === 'hoodie' ? 0.1 : 0.2), 0.85 + 0.15 * Math.sin(ax * 330))
    if (o.top === 'layered' && Math.abs(ax - (b.shoulderRestX + 0.2)) < 0.012) c = mix(c, trim, 0.85)
  }
  return c
}

function skinPaint(t: Texel, o: Outfit, b: BodyMetrics, tex: ImageData, hair: RGB, gloves: RGB | null, bands: RGB[]): RGB {
  const px = Math.min(tex.width - 1, Math.floor(t.u * tex.width)), py = Math.min(tex.height - 1, Math.floor(t.v * tex.height))
  const i = (py * tex.width + px) * 4
  let c: RGB = [tex.data[i] * o.skin[0], tex.data[i + 1] * o.skin[1], tex.data[i + 2] * o.skin[2]]
  // keep what the top doesn't cover from being pale: a soft blush + contrast toward console-game skin
  const ax = Math.abs(t.x)
  if (t.cls === 'head') {
    // the scalp: painted hair colour with a soft hairline
    const up = t.y - b.headY
    const face = t.z > 0.03 && up < 0.145
    const hairline = smooth(0.13, 0.155, up) + (t.z < 0.02 ? smooth(0.045, 0.085, up) : 0) * smooth(0.02, -0.03, t.z)
    if (!face || hairline > 0) c = mix(c, hair, Math.min(1, hairline) * 0.94)
  }
  // gloves on the palm + first finger joints, wristbands above them
  if (gloves && t.cls === 'hand' && ax < b.fingerX) {
    const leather = mul(gloves, 0.8 + 0.5 * noise3(t.x * 60, t.y * 60, t.z * 60))
    const holes = Math.abs(ax - (b.wristX + 0.055)) < 0.006
    c = holes ? mul(leather, 0.55) : leather
  }
  if (bands.length && (t.cls === 'farm' || t.cls === 'hand')) {
    const d = b.wristX + 0.012 - ax
    const w = 0.017
    if (d > 0 && d < w * bands.length) {
      const idx = Math.floor(d / w)
      const s = d / w - idx
      c = mul(bands[idx], 0.72 + 0.4 * Math.sin(s * Math.PI))
    }
  }
  return c
}

/** Paints the texture for one skater. `base` = the pack's skin texture. */
export function paintSkaterTexture(s: Sculpted, o: Outfit, b: BodyMetrics, base: HTMLImageElement | HTMLCanvasElement): CanvasTexture {
  const baseC = document.createElement('canvas'); baseC.width = baseC.height = S
  const bg = baseC.getContext('2d', { willReadFrequently: true })!
  bg.drawImage(base, 0, 0, S, S)
  const skinData = bg.getImageData(0, 0, S, S)
  const print = paintPrint(o)

  const c = document.createElement('canvas'); c.width = c.height = S
  const g = c.getContext('2d')!
  const img = g.createImageData(S, S)
  const px = img.data
  const filled = new Uint8Array(S * S)

  const pantsBase = rgb(o.pants), stitch = mix(rgb(o.pants), [226, 196, 140], o.pantsKind === 'jeans' ? 0.75 : 0.25)
  const topBase = rgb(o.topColor), sleeve = rgb(o.sleeveColor), trim = rgb(o.trim)
  const hair = rgb(o.hair), gloves = o.gloves ? rgb(o.gloves) : null, bands = o.bands.map(rgb)

  const geo = s.geometry
  const uv = geo.attributes.uv, nor = geo.attributes.normal, sp = geo.attributes.position, idx = geo.index!
  const t: Texel = { x: 0, y: 0, z: 0, sx: 0, sy: 0, sz: 0, nx: 0, ny: 0, nz: 0, cls: '', u: 0, v: 0 }
  for (let f = 0; f < idx.count; f += 3) {
    const a = idx.getX(f), d = idx.getX(f + 1), e = idx.getX(f + 2)
    const reg = s.region[a]
    const cls = CLASSES[s.cls[a]]
    const ax = uv.getX(a) * S, ay = uv.getY(a) * S, dx = uv.getX(d) * S, dy = uv.getY(d) * S, ex = uv.getX(e) * S, ey = uv.getY(e) * S
    const den = (dy - ey) * (ax - ex) + (ex - dx) * (ay - ey)
    if (Math.abs(den) < 1e-6) continue
    const x0 = Math.max(0, Math.floor(Math.min(ax, dx, ex))), x1 = Math.min(S - 1, Math.ceil(Math.max(ax, dx, ex)))
    const y0 = Math.max(0, Math.floor(Math.min(ay, dy, ey))), y1 = Math.min(S - 1, Math.ceil(Math.max(ay, dy, ey)))
    for (let py = y0; py <= y1; py++) for (let qx = x0; qx <= x1; qx++) {
      const l0 = ((dy - ey) * (qx + 0.5 - ex) + (ex - dx) * (py + 0.5 - ey)) / den
      const l1 = ((ey - ay) * (qx + 0.5 - ex) + (ax - ex) * (py + 0.5 - ey)) / den
      const l2 = 1 - l0 - l1
      if (l0 < -0.02 || l1 < -0.02 || l2 < -0.02) continue
      const r = s.rest
      t.x = l0 * r[a * 3] + l1 * r[d * 3] + l2 * r[e * 3]
      t.y = l0 * r[a * 3 + 1] + l1 * r[d * 3 + 1] + l2 * r[e * 3 + 1]
      t.z = l0 * r[a * 3 + 2] + l1 * r[d * 3 + 2] + l2 * r[e * 3 + 2]
      t.sx = l0 * sp.getX(a) + l1 * sp.getX(d) + l2 * sp.getX(e)
      t.sy = l0 * sp.getY(a) + l1 * sp.getY(d) + l2 * sp.getY(e)
      t.sz = l0 * sp.getZ(a) + l1 * sp.getZ(d) + l2 * sp.getZ(e)
      t.nx = l0 * nor.getX(a) + l1 * nor.getX(d) + l2 * nor.getX(e)
      t.ny = l0 * nor.getY(a) + l1 * nor.getY(d) + l2 * nor.getY(e)
      t.nz = l0 * nor.getZ(a) + l1 * nor.getZ(d) + l2 * nor.getZ(e)
      t.cls = cls; t.u = (qx + 0.5) / S; t.v = (py + 0.5) / S
      const col = reg === 2 ? pantsPaint(t, o, b, pantsBase, stitch)
        : reg === 1 ? topPaint(t, o, b, topBase, sleeve, trim, print)
        : skinPaint(t, o, b, skinData, hair, gloves, bands)
      const o4 = (py * S + qx) * 4
      px[o4] = col[0]; px[o4 + 1] = col[1]; px[o4 + 2] = col[2]; px[o4 + 3] = 255
      filled[py * S + qx] = 1
    }
  }

  // bleed colours past the UV islands so mips and bilinear don't pick up the empty background
  for (let pass = 0; pass < 6; pass++) {
    const prev = filled.slice()
    for (let y = 1; y < S - 1; y++) for (let x = 1; x < S - 1; x++) {
      const i = y * S + x
      if (prev[i]) continue
      const from = prev[i - 1] ? i - 1 : prev[i + 1] ? i + 1 : prev[i - S] ? i - S : prev[i + S] ? i + S : -1
      if (from < 0) continue
      px[i * 4] = px[from * 4]; px[i * 4 + 1] = px[from * 4 + 1]; px[i * 4 + 2] = px[from * 4 + 2]; px[i * 4 + 3] = 255
      filled[i] = 1
    }
  }
  g.putImageData(img, 0, 0)

  const tex = new CanvasTexture(c)
  tex.flipY = false                // glTF UVs
  tex.colorSpace = SRGBColorSpace
  tex.magFilter = LinearFilter; tex.minFilter = LinearMipmapLinearFilter; tex.generateMipmaps = true
  tex.anisotropy = 2
  return tex
}
