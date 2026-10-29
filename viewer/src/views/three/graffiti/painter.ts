// Canvas painter for the wall graffiti: letter pieces built from the skeleton glyphs in glyphs.ts,
// plus the bits that fill a real wall between them (stars, crowns, arrows, buffs, characters,
// backdrops, drips). Everything is flat cel colour with a hard dark ink line — no gradients
// except the optional single soft "fade" fill, no noise, no overspray speckle.
//
// A piece is painted the way a writer does it: 3D block / drop first, then letter by letter the
// outer ink line, the light cut-back, the inner ink line and the fill (fill bands, shines and
// breaks are clipped to that letter's fill), so later letters overlap earlier ones.
import { rng } from '../environment/textures'
import { glyphPaths, glyphWidth, type Cmd } from './glyphs'

export const INK = '#1b1519'

/** Muted "Downtown" palette plus a couple of brighter accents. */
export const PAL = {
  rust: '#a4502f', brick: '#8f3b2f', mustard: '#c99f3c', teal: '#2f7b74', plum: '#6b3b5f',
  olive: '#737a3b', denim: '#3e5e8b', offwhite: '#e8dfc8', silver: '#c4c1b6', grey: '#8a8279',
  dgrey: '#57534e', sand: '#c9b48c', sky: '#86b2bd', moss: '#4f5f3a', wine: '#5a2a35',
  pink: '#d2577e', lime: '#a6c23f', orange: '#e0782b', // accents
} as const

export type Kind = 'bubble' | 'wild' | 'block' | 'tag'
type Pt = [number, number]
type Box = [number, number, number, number]

interface Stroke { path: Path2D; w: number }
interface Letter { strokes: Stroke[]; polys: Path2D[]; box: Box; bottoms: Pt[]; seed: number }

// ---------------------------------------------------------------- word layout

function hash(u: number, v: number, s: number): number {
  const x = Math.sin(u * 127.1 + v * 311.7 + s * 74.7) * 43758.5453
  return x - Math.floor(x)
}

interface WordOpts {
  kind: Kind; h: number; seed: number
  wild?: number          // 0..1 how much the letters warp, lean and grow arrows
  sx?: number            // horizontal squash/stretch
  weight?: number        // stroke width multiplier
  arrows?: number        // chance per open stroke end of growing an arrow (wild/block)
}

interface Word { letters: Letter[]; box: Box; W: number }

const KIND = {
  //        stroke  advance  shear  bulge  jitter rot    dy     scale
  bubble: { W: 0.46, adv: 0.3, shear: 0.06, bulge: 0.2, jit: 0.03, rot: 0.2, dy: 0.1, sc: 0.22 },
  wild: { W: 0.2, adv: 0.5, shear: 0.2, bulge: 0.0, jit: 0.07, rot: 0.32, dy: 0.16, sc: 0.3 },
  block: { W: 0.3, adv: 1.0, shear: 0.0, bulge: 0.0, jit: 0.0, rot: 0.0, dy: 0.0, sc: 0.0 },
  tag: { W: 0.1, adv: 0.0, shear: 0.5, bulge: 0.0, jit: 0.1, rot: 0.2, dy: 0.16, sc: 0.4 },
}

function buildWord(text: string, o: WordOpts): Word {
  const k = KIND[o.kind]
  const r = rng(o.seed * 9301 + 17)
  const h = o.h, sx = o.sx ?? 1, wild = o.wild ?? (o.kind === 'wild' ? 0.7 : 1)
  const W = k.W * h * (o.weight ?? 1)
  const letters: Letter[] = []
  let cursor = 0
  const chars = [...text.toUpperCase()]
  chars.forEach((ch, li) => {
    const gw = glyphWidth(ch)
    if (ch === ' ') { cursor += gw * h * sx; return }
    const sc = 1 + (r() - 0.5) * k.sc * wild + ((o.kind === 'wild' || o.kind === 'tag') && li === 0 ? (o.kind === 'tag' ? 0.3 : 0.12) : 0)
    const rot = (r() - 0.5) * k.rot * wild
    const dy = (r() - 0.5) * k.dy * h * wild
    const lw = gw * h * sx * sc
    const cx = cursor + lw / 2
    const lseed = o.seed * 31 + li * 7
    const cr = Math.cos(rot), sr = Math.sin(rot)
    const box: Box = [Infinity, Infinity, -Infinity, -Infinity]
    const bottoms: Pt[] = []
    const T = (u: number, v: number): Pt => {
      let x = (u - gw / 2) * h * sx * sc, y = (v - 0.5) * h * sc
      x *= 1 + k.bulge * (v - 0.35) * 2
      x += k.shear * (0.5 - v) * h
      x += (hash(u, v, lseed) - 0.5) * k.jit * wild * h
      y += (hash(v, u, lseed + 3) - 0.5) * k.jit * wild * h
      const X = cx + x * cr - y * sr, Y = h / 2 + dy + x * sr + y * cr
      if (X < box[0]) box[0] = X; if (Y < box[1]) box[1] = Y
      if (X > box[2]) box[2] = X; if (Y > box[3]) box[3] = Y
      return [X, Y]
    }
    const strokes: Stroke[] = []
    const polys: Path2D[] = []
    glyphPaths(ch).forEach((sub: Cmd[], si) => {
      const path = new Path2D()
      let w = W
      if (o.kind === 'wild') w *= si === 0 ? 1.15 + r() * 0.25 : 0.7 + r() * 0.5
      else if (o.kind === 'tag') w *= 0.8 + r() * 0.4
      let first: Pt = [0, 0], firstNext: Pt = [0, 0], last: Pt = [0, 0], lastPrev: Pt = [0, 0]
      let closed = false
      sub.forEach((cm, ci) => {
        const p = cm.p
        const pts: Pt[] = []
        for (let i = 0; i < p.length; i += 2) {
          pts.push(T(p[i], p[i + 1]))
          if (p[i + 1] >= 0.96) bottoms.push(pts[pts.length - 1])
        }
        if (cm.c === 'M') { path.moveTo(...pts[0]); first = pts[0] }
        else if (cm.c === 'L') path.lineTo(...pts[0])
        else if (cm.c === 'Q') path.quadraticCurveTo(...pts[0], ...pts[1])
        else if (cm.c === 'C') path.bezierCurveTo(...pts[0], ...pts[1], ...pts[2])
        else if (cm.c === 'Z') { path.closePath(); closed = true }
        if (cm.c !== 'Z' && cm.c !== 'M') {
          if (ci === 1) firstNext = pts[0]
          lastPrev = pts.length > 1 ? pts[pts.length - 2] : last
          last = pts[pts.length - 1]
        }
      })
      strokes.push({ path, w })
      // arrows / spikes on open ends (wildstyle extensions)
      const pa = o.arrows ?? 0
      if (!closed && pa > 0) {
        if (r() < pa) polys.push(arrowHead(last, lastPrev, w, r))
        if (r() < pa * 0.5) polys.push(arrowHead(first, firstNext, w, r))
      }
    })
    box[0] -= W; box[1] -= W; box[2] += W; box[3] += W
    letters.push({ strokes, polys, box, bottoms, seed: lseed })
    cursor += lw + (o.kind === 'block' ? W + h * 0.1 : o.kind === 'tag' ? h * (0.02 + r() * 0.06) : W * k.adv)
  })
  const box: Box = [Infinity, Infinity, -Infinity, -Infinity]
  letters.forEach(l => {
    box[0] = Math.min(box[0], l.box[0]); box[1] = Math.min(box[1], l.box[1])
    box[2] = Math.max(box[2], l.box[2]); box[3] = Math.max(box[3], l.box[3])
  })
  return { letters, box, W }
}

/** A sharp arrow head grown off a stroke end, pointing away from the stroke. */
function arrowHead(end: Pt, prev: Pt, w: number, r: () => number): Path2D {
  let dx = end[0] - prev[0], dy = end[1] - prev[1]
  const L = Math.hypot(dx, dy) || 1
  dx /= L; dy /= L
  const nx = -dy, ny = dx
  const len = w * (1.4 + r() * 1.2), half = w * (0.95 + r() * 0.35)
  const back = w * 0.2
  const p = new Path2D()
  const bx = end[0] - dx * back, by = end[1] - dy * back
  p.moveTo(bx + nx * half, by + ny * half)
  // a little barb kink on one side makes it read hand-made
  p.lineTo(end[0] + dx * len * 0.35 + nx * half * 0.35, end[1] + dy * len * 0.35 + ny * half * 0.35)
  p.lineTo(end[0] + dx * len, end[1] + dy * len)
  p.lineTo(bx - nx * half, by - ny * half)
  p.closePath()
  return p
}

function layer(g: CanvasRenderingContext2D, L: Letter, e: number, color: string, ox = 0, oy = 0) {
  g.save()
  g.translate(ox, oy)
  g.strokeStyle = color; g.fillStyle = color
  for (const s of L.strokes) { g.lineWidth = s.w + 2 * e; g.stroke(s.path) }
  for (const p of L.polys) { g.fill(p); if (e > 0) { g.lineWidth = 2 * e; g.stroke(p) } }
  g.restore()
}

// ---------------------------------------------------------------- scratch canvases

let scratchA: HTMLCanvasElement | null = null
let scratchB: HTMLCanvasElement | null = null
function scratch(w: number, h: number): [CanvasRenderingContext2D, CanvasRenderingContext2D] {
  const make = (c: HTMLCanvasElement | null) => {
    if (!c) { c = document.createElement('canvas'); c.width = 16; c.height = 16 }
    if (c.width < w || c.height < h) { c.width = Math.max(c.width, Math.ceil(w)); c.height = Math.max(c.height, Math.ceil(h)) }
    return c
  }
  scratchA = make(scratchA); scratchB = make(scratchB)
  const a = scratchA.getContext('2d')!, b = scratchB.getContext('2d')!
  for (const g of [a, b]) {
    g.setTransform(1, 0, 0, 1, 0, 0)
    g.globalCompositeOperation = 'source-over'; g.globalAlpha = 1
    g.clearRect(0, 0, w + 2, h + 2)
    g.lineJoin = 'round'; g.lineCap = 'round'; g.miterLimit = 3
    g.setLineDash([])
  }
  return [a, b]
}

// ---------------------------------------------------------------- pieces

export type FillMode = 'flat' | 'bands' | 'split' | 'fade' | 'bubbles'

export interface PieceSpec {
  text: string
  kind: Exclude<Kind, 'tag'>
  h: number                  // letter height, px
  seed: number
  fill: string[]             // fill colours (bands / split / fade / bubbles use 2-3)
  fillMode?: FillMode
  ink?: string               // outline
  cut?: string | null        // light cut-back outline (null: none)
  block?: string | null      // 3D block colour (null: just a dark drop)
  blockDir?: [number, number]// 3D depth as a fraction of h
  shine?: boolean
  drips?: number
  wild?: number
  arrows?: number
  weight?: number
  sx?: number
  breaks?: number            // light cut lines through each letter's fill
}

/** Paint a lettered piece centred at (cx, cy) on `target`. Returns its rough bounds. */
export function paintPiece(target: CanvasRenderingContext2D, s: PieceSpec, cx: number, cy: number, rot = 0, alpha = 1): Box {
  const word = buildWord(s.text, { kind: s.kind, h: s.h, seed: s.seed, wild: s.wild, arrows: s.arrows, weight: s.weight, sx: s.sx })
  const r = rng(s.seed * 7 + 3)
  const h = s.h
  const ink = s.ink ?? INK
  const eInk = h * (s.kind === 'bubble' ? 0.045 : 0.032)
  const eCut = s.cut ? h * (s.kind === 'bubble' ? 0.04 : 0.05) : 0
  const eOuter = s.cut ? eInk + eCut + h * 0.026 : eInk
  const bd = s.blockDir ?? (s.kind === 'bubble' ? [0.07, 0.06] : [0.1, 0.09])
  const dX = bd[0] * h, dY = bd[1] * h
  const m = Math.max(h * 0.75, eOuter * 2)
  const [x0, y0, x1, y1] = word.box
  const cw = x1 - x0 + m * 2 + Math.abs(dX), ch = y1 - y0 + m * 2 + Math.abs(dY)
  const [g, f] = scratch(cw, ch)
  const ox = m - x0 + Math.min(0, -dX), oy = m - y0 + Math.min(0, -dY)
  g.translate(ox, oy); f.translate(ox, oy)
  if (s.kind === 'block') { g.lineJoin = f.lineJoin = 'miter'; g.lineCap = f.lineCap = 'square' }
  else if (s.kind === 'wild') { g.lineJoin = f.lineJoin = 'miter'; g.lineCap = f.lineCap = 'butt' }

  // 1) 3D block / drop shadow, behind every letter: paint the silhouette once on the spare layer,
  //    then stamp it along the depth direction (cheap drawImage instead of re-stroking)
  const steps = Math.max(3, Math.ceil(Math.hypot(dX, dY) / Math.max(1.5, word.W * 0.06)))
  const stamp = (e: number, color: string, from: number, to: number) => {
    f.save(); f.setTransform(1, 0, 0, 1, 0, 0); f.clearRect(0, 0, cw + 2, ch + 2); f.restore()
    word.letters.forEach(L => layer(f, L, e, color))
    g.save(); g.setTransform(1, 0, 0, 1, 0, 0)
    for (let i = from; i <= to; i++) g.drawImage(f.canvas, 0, 0, cw, ch, dX * i / steps, dY * i / steps, cw, ch)
    g.restore()
  }
  stamp(eOuter, ink, 0, steps)
  if (s.block) {
    const eb = eOuter - h * 0.022
    stamp(eb, s.block, 0, steps - 1)
    // a lighter facet line along the block (the classic "3D edge")
    g.save(); g.globalAlpha = 0.5
    word.letters.forEach(L => layer(g, L, eb * 0.35, shade(s.block!, 1.35), dX * 0.45, dY * 0.45))
    g.restore()
  }
  f.save(); f.setTransform(1, 0, 0, 1, 0, 0); f.clearRect(0, 0, cw + 2, ch + 2); f.restore()

  // 2) letter by letter: outer ink, cut-back, inner ink, fill
  const mode = s.fillMode ?? 'flat'
  const bubbleR = rng(s.seed + 99)
  const bubbles = Array.from({ length: 22 }, () => [x0 + bubbleR() * (x1 - x0), y0 + bubbleR() * (y1 - y0), h * (0.04 + bubbleR() * 0.08)])
  word.letters.forEach(L => {
    if (s.cut) { layer(g, L, eOuter, ink); layer(g, L, eInk + eCut, s.cut) }
    layer(g, L, eInk, ink)
    // fill on its own layer so bands / shines clip to it
    const [lx0, ly0, lx1, ly1] = L.box
    f.save(); f.setTransform(1, 0, 0, 1, 0, 0); f.clearRect(lx0 + ox - 2, ly0 + oy - 2, lx1 - lx0 + 4, ly1 - ly0 + 4); f.restore()
    layer(f, L, 0, s.fill[0])
    f.globalCompositeOperation = 'source-atop'
    paintFill(f, mode, s.fill, word.box, h, s.seed, bubbles)
    if (s.breaks) {
      const lr = rng(L.seed)
      f.strokeStyle = s.cut ?? PAL.offwhite; f.lineWidth = h * 0.016; f.lineCap = 'butt'
      for (let b = 0; b < s.breaks; b++) {
        const yy = ly0 + (ly1 - ly0) * (0.25 + lr() * 0.5), sl = (lr() - 0.5) * (ly1 - ly0) * 0.9
        f.beginPath(); f.moveTo(lx0, yy - sl); f.lineTo(lx1, yy + sl); f.stroke()
      }
    }
    if (s.shine !== false) {
      f.save()
      f.translate(-word.W * 0.2, -word.W * 0.24)
      f.strokeStyle = '#fbf6ea'; f.lineCap = 'round'
      for (const st of L.strokes) {
        f.lineWidth = Math.max(2, st.w * 0.13)
        f.setLineDash([st.w * 0.7, st.w * 0.35, st.w * 0.08, 1e5])
        f.stroke(st.path)
      }
      f.restore()
    }
    f.globalCompositeOperation = 'source-over'; f.setLineDash([])
    g.save(); g.setTransform(1, 0, 0, 1, 0, 0)
    const sx0 = Math.max(0, lx0 + ox - 2), sy0 = Math.max(0, ly0 + oy - 2)
    const sw = Math.min(f.canvas.width - sx0, lx1 - lx0 + 4), sh = Math.min(f.canvas.height - sy0, ly1 - ly0 + 4)
    if (sw > 0 && sh > 0) g.drawImage(f.canvas, sx0, sy0, sw, sh, sx0, sy0, sw, sh)
    g.restore()
  })

  // 3) drips + sparkles
  const all = word.letters.flatMap(L => L.bottoms)
  for (let i = 0; i < (s.drips ?? 0) && all.length; i++) {
    const p = all[Math.floor(r() * all.length)]
    drip(g, p[0] + (r() - 0.5) * word.W * 0.6, p[1] + word.W * 0.5 + eInk * 0.5, h * (0.12 + r() * 0.4), Math.max(3, h * 0.045), s.fill[s.fill.length - 1], ink)
  }
  if (s.shine !== false) {
    for (let i = 0; i < 1; i++) {
      const L = word.letters[Math.floor(r() * word.letters.length)]
      sparkle(g, L.box[0] + (L.box[2] - L.box[0]) * (0.2 + r() * 0.6), L.box[1] + word.W * 0.3, h * (0.1 + r() * 0.06))
    }
  }

  // 4) composite onto the wall
  target.save()
  target.globalAlpha = alpha
  target.translate(cx, cy); target.rotate(rot)
  target.drawImage(g.canvas, 0, 0, cw, ch, -cw / 2, -ch / 2, cw, ch)
  target.restore()
  const pw = x1 - x0 + Math.abs(dX) + eOuter * 2, ph = y1 - y0 + Math.abs(dY) + eOuter * 2
  return [cx - pw / 2, cy - ph / 2, cx + pw / 2, cy + ph / 2]
}

function paintFill(f: CanvasRenderingContext2D, mode: FillMode, fill: string[], box: Box, h: number, seed: number, bubbles: number[][]) {
  const [x0, y0, x1, y1] = box
  const r = rng(seed + 5)
  if (mode === 'bands') {
    const n = fill.length
    const tilt = (r() - 0.5) * h * 0.4
    for (let i = 1; i < n; i++) {
      const t = 0.3 + (i - 1) * (0.5 / Math.max(1, n - 1)) + (r() - 0.5) * 0.06
      const ya = y0 + (y1 - y0) * t
      f.fillStyle = fill[i]
      f.beginPath(); f.moveTo(x0, ya); f.lineTo(x1, ya + tilt); f.lineTo(x1, y1 + h); f.lineTo(x0, y1 + h); f.fill()
    }
  } else if (mode === 'split') {
    f.fillStyle = fill[1] ?? fill[0]
    f.beginPath(); f.moveTo(x0, y1)
    const n = 7
    for (let i = 0; i <= n; i++) {
      const x = x0 + (x1 - x0) * i / n
      const y = y0 + (y1 - y0) * (i % 2 ? 0.42 : 0.62) + (r() - 0.5) * h * 0.1
      f.lineTo(x, y)
    }
    f.lineTo(x1, y1); f.closePath(); f.fill()
    if (fill[2]) {
      f.fillStyle = fill[2]
      f.fillRect(x0, y0 + (y1 - y0) * 0.82, x1 - x0, h)
    }
  } else if (mode === 'fade') {
    const gr = f.createLinearGradient(0, y0 + (y1 - y0) * 0.25, 0, y1 - (y1 - y0) * 0.2)
    gr.addColorStop(0, fill[0]); gr.addColorStop(1, fill[1] ?? fill[0])
    f.fillStyle = gr; f.fillRect(x0, y0, x1 - x0, y1 - y0)
    if (fill[2]) { f.fillStyle = fill[2]; f.fillRect(x0, y0 + (y1 - y0) * 0.8, x1 - x0, h) }
  } else if (mode === 'bubbles') {
    f.fillStyle = fill[1] ?? PAL.offwhite
    for (const [x, y, rr] of bubbles) { f.beginPath(); f.arc(x, y, rr, 0, 7); f.fill() }
    if (fill[2]) { f.fillStyle = fill[2]; f.fillRect(x0, y0 + (y1 - y0) * 0.78, x1 - x0, h) }
  }
}

// ---------------------------------------------------------------- tags (handstyles)

export interface TagSpec {
  text: string; h: number; seed: number; color: string
  halo?: string | null      // outline colour behind the tag ("tag with outline")
  weight?: number           // marker / nozzle fatness
  sx?: number
  crown?: boolean
  underline?: boolean       // swoosh + arrow under the word
  drips?: number
  lean?: number             // multiplies the default italic lean
}

/** Paint a handstyle tag whose left end sits at (x, y) (y = letter middle). Returns its bounds. */
export function paintTag(g: CanvasRenderingContext2D, t: TagSpec, x: number, y: number, rot = 0, alpha = 1): Box {
  const word = buildWord(t.text, { kind: 'tag', h: t.h, seed: t.seed, weight: t.weight, sx: t.sx ?? 0.8, wild: t.lean ?? 1 })
  const r = rng(t.seed * 3 + 1)
  const [x0, y0, x1, y1] = word.box
  g.save()
  g.globalAlpha = alpha
  g.translate(x - x0, y - (y0 + y1) / 2)
  g.rotate(rot)
  g.lineJoin = 'round'; g.lineCap = 'round'
  const swoosh = new Path2D()
  if (t.underline) {
    const yb = y1 - word.W * 0.6
    swoosh.moveTo(x0 + t.h * 0.1, yb + t.h * 0.2)
    swoosh.quadraticCurveTo((x0 + x1) / 2, yb + t.h * 0.48, x1 + t.h * 0.3, yb + t.h * 0.05)
    const ah = t.h * 0.18
    swoosh.moveTo(x1 + t.h * 0.3 - ah, yb - ah * 0.5); swoosh.lineTo(x1 + t.h * 0.3, yb + t.h * 0.05); swoosh.lineTo(x1 + t.h * 0.3 - ah * 0.9, yb + ah * 0.9)
  }
  const crown = new Path2D()
  if (t.crown) {
    const L = word.letters[0], cx = (L.box[0] + L.box[2]) / 2 + t.h * 0.15, cy = L.box[1] - t.h * 0.05, cwid = t.h * 0.5
    crown.moveTo(cx - cwid / 2, cy); crown.lineTo(cx - cwid / 2 - cwid * 0.1, cy - cwid * 0.55); crown.lineTo(cx - cwid * 0.2, cy - cwid * 0.25)
    crown.lineTo(cx, cy - cwid * 0.7); crown.lineTo(cx + cwid * 0.2, cy - cwid * 0.25); crown.lineTo(cx + cwid / 2 + cwid * 0.1, cy - cwid * 0.55)
    crown.lineTo(cx + cwid / 2, cy); crown.closePath()
  }
  // handstyle connections along the baseline + a flourish off the last letter
  const links = new Path2D()
  if (r() < 0.45) word.letters.slice(1).forEach((L, i) => {
    const P = word.letters[i], yb = Math.min(P.box[3], L.box[3]) - word.W * 1.4
    links.moveTo(P.box[2] - word.W * 1.6, yb); links.quadraticCurveTo((P.box[2] + L.box[0]) / 2, yb + t.h * 0.08, L.box[0] + word.W * 1.4, yb)
  })
  if (!t.underline && r() < 0.6) {
    const L = word.letters[word.letters.length - 1], ex = L.box[2] - word.W, ey = L.box[3] - word.W * 1.2
    links.moveTo(ex - t.h * 0.15, ey); links.bezierCurveTo(ex + t.h * 0.35, ey + t.h * 0.25, ex + t.h * 0.55, ey - t.h * 0.1, ex + t.h * 0.4, ey - t.h * 0.55)
  }
  const passes: [string, number][] = t.halo ? [[t.halo, t.h * 0.11], [t.color, 0]] : [[t.color, 0]]
  for (const [col, e] of passes) {
    g.strokeStyle = col
    word.letters.forEach(L => { for (const s of L.strokes) { g.lineWidth = s.w + e; g.stroke(s.path) } })
    g.lineWidth = word.W * 0.8 + e
    g.stroke(links)
    if (t.underline) g.stroke(swoosh)
    if (t.crown) g.stroke(crown)
  }
  // marker drips
  const all = word.letters.flatMap(L => L.bottoms)
  g.fillStyle = g.strokeStyle = t.color
  for (let i = 0; i < (t.drips ?? 0) && all.length; i++) {
    const p = all[Math.floor(r() * all.length)]
    const len = t.h * (0.15 + r() * 0.6), wd = Math.max(1.5, word.W * 0.35)
    g.lineWidth = wd; g.beginPath(); g.moveTo(p[0], p[1]); g.lineTo(p[0], p[1] + len); g.stroke()
    g.beginPath(); g.arc(p[0], p[1] + len, wd * 0.8, 0, 7); g.fill()
  }
  g.restore()
  const w = x1 - x0 + (t.underline ? t.h * 0.4 : 0)
  const hh = y1 - y0
  return [x, y - hh / 2 - (t.crown ? t.h * 0.4 : 0), x + w, y + hh / 2 + (t.underline ? t.h * 0.4 : 0)]
}

// ---------------------------------------------------------------- shapes

export function shade(hex: string, k: number): string {
  const n = parseInt(hex.slice(1), 16)
  const c = (v: number) => Math.max(0, Math.min(255, Math.round(k >= 1 ? v + (255 - v) * (k - 1) : v * k)))
  return `#${[n >> 16, (n >> 8) & 255, n & 255].map(v => c(v).toString(16).padStart(2, '0')).join('')}`
}

export function drip(g: CanvasRenderingContext2D, x: number, y: number, len: number, w: number, color: string, ink = INK) {
  g.save()
  g.lineCap = 'round'
  g.strokeStyle = ink; g.lineWidth = w + 3
  g.beginPath(); g.moveTo(x, y); g.lineTo(x, y + len); g.stroke()
  g.fillStyle = ink; g.beginPath(); g.arc(x, y + len, w * 0.8 + 1.5, 0, 7); g.fill()
  g.strokeStyle = color; g.lineWidth = w
  g.beginPath(); g.moveTo(x, y - w * 2.2); g.lineTo(x, y + len); g.stroke()
  g.fillStyle = color; g.beginPath(); g.arc(x, y + len, w * 0.8, 0, 7); g.fill()
  g.restore()
}

/** Four-point "shine" sparkle. */
export function sparkle(g: CanvasRenderingContext2D, x: number, y: number, s: number, color = '#fbf6ea') {
  g.save()
  g.fillStyle = color
  g.beginPath()
  for (let i = 0; i < 8; i++) {
    const a = i * Math.PI / 4, rr = i % 2 ? s * 0.14 : s * (i % 4 ? 0.6 : 1)
    g.lineTo(x + Math.sin(a) * rr, y - Math.cos(a) * rr)
  }
  g.closePath(); g.fill()
  g.restore()
}

export function star(g: CanvasRenderingContext2D, x: number, y: number, s: number, fill: string, rot = 0, ink = INK, pts = 5) {
  const p = new Path2D()
  for (let i = 0; i < pts * 2; i++) {
    const a = rot + i * Math.PI / pts, rr = i % 2 ? s * 0.45 : s
    p.lineTo(x + Math.sin(a) * rr, y - Math.cos(a) * rr)
  }
  p.closePath()
  g.save()
  g.lineJoin = 'miter'
  g.fillStyle = ink; g.save(); g.translate(s * 0.12, s * 0.12); g.fill(p); g.restore()
  g.strokeStyle = ink; g.lineWidth = Math.max(2, s * 0.16); g.stroke(p)
  g.fillStyle = fill; g.fill(p)
  g.restore()
}

export function crown(g: CanvasRenderingContext2D, x: number, y: number, w: number, fill: string, ink = INK) {
  const p = new Path2D()
  const h = w * 0.62
  p.moveTo(x - w / 2, y); p.lineTo(x - w * 0.58, y - h); p.lineTo(x - w * 0.25, y - h * 0.45)
  p.lineTo(x, y - h * 1.15); p.lineTo(x + w * 0.25, y - h * 0.45); p.lineTo(x + w * 0.58, y - h); p.lineTo(x + w / 2, y)
  p.closePath()
  g.save()
  g.lineJoin = 'miter'
  g.strokeStyle = ink; g.lineWidth = w * 0.09; g.stroke(p)
  g.fillStyle = fill; g.fill(p)
  g.fillStyle = INK
  for (const dx of [-0.58, 0, 0.58]) { g.beginPath(); g.arc(x + dx * w, y - h * (dx ? 1.08 : 1.25), w * 0.07, 0, 7); g.fill() }
  g.fillStyle = '#fbf6ea'; g.fillRect(x - w * 0.36, y - h * 0.3, w * 0.12, h * 0.1)
  g.restore()
}

/** A fat bent arrow along a polyline, with an ink outline. */
export function bigArrow(g: CanvasRenderingContext2D, pts: Pt[], w: number, fill: string, ink = INK) {
  const p = new Path2D()
  p.moveTo(...pts[0])
  for (let i = 1; i < pts.length - 1; i++) {
    const mx = (pts[i][0] + pts[i + 1][0]) / 2, my = (pts[i][1] + pts[i + 1][1]) / 2
    p.quadraticCurveTo(pts[i][0], pts[i][1], mx, my)
  }
  const end = pts[pts.length - 1], prev = pts[pts.length - 2]
  p.lineTo(...end)
  const head = arrowHead(end, prev, w, () => 0.5)
  g.save()
  g.lineCap = 'butt'; g.lineJoin = 'miter'
  g.strokeStyle = ink; g.lineWidth = w + w * 0.4; g.stroke(p)
  g.fillStyle = ink; g.fill(head); g.lineWidth = w * 0.4; g.stroke(head)
  g.strokeStyle = fill; g.lineWidth = w; g.stroke(p)
  g.fillStyle = fill; g.fill(head)
  g.restore()
}

/** A rough flat painted-over rectangle (the city's or another writer's buff). */
export function buff(g: CanvasRenderingContext2D, x0: number, y0: number, x1: number, y1: number, color: string, seed: number) {
  const r = rng(seed)
  const j = Math.min(x1 - x0, y1 - y0) * 0.05
  g.save()
  g.fillStyle = color
  g.beginPath()
  const edge = (ax: number, ay: number, bx: number, by: number) => {
    const n = 8
    for (let i = 0; i < n; i++) {
      const t = i / n
      g.lineTo(ax + (bx - ax) * t + (r() - 0.5) * j, ay + (by - ay) * t + (r() - 0.5) * j)
    }
  }
  edge(x0, y0, x1, y0); edge(x1, y0, x1, y1); edge(x1, y1, x0, y1); edge(x0, y1, x0, y0)
  g.closePath(); g.fill()
  // a roller-lap stripe and a thin run at the bottom edge
  g.fillStyle = shade(color, 0.92)
  g.fillRect(x0 + j, y0 + (y1 - y0) * (0.3 + r() * 0.3), (x1 - x0) - 2 * j, (y1 - y0) * 0.14)
  g.strokeStyle = color; g.lineWidth = 2.5; g.lineCap = 'round'
  for (let i = 0; i < 5; i++) {
    const x = x0 + (x1 - x0) * r()
    g.beginPath(); g.moveTo(x, y1 - 2); g.lineTo(x, y1 + j * (1 + r() * 3)); g.stroke()
  }
  g.restore()
}

/** A flat backdrop behind a burner: puffy cloud or jagged shards, with ink outline. */
export function backdrop(g: CanvasRenderingContext2D, x0: number, y0: number, x1: number, y1: number, fill: string, seed: number, kind: 'cloud' | 'shards' = 'cloud', ink = INK) {
  const r = rng(seed)
  const cx = (x0 + x1) / 2, cy = (y0 + y1) / 2, rx = (x1 - x0) / 2, ry = (y1 - y0) / 2
  const p = new Path2D()
  if (kind === 'cloud') {
    const n = 16
    for (let i = 0; i <= n; i++) {
      const a = (i / n) * Math.PI * 2
      const x = cx + Math.cos(a) * rx, y = cy + Math.sin(a) * ry
      if (i === 0) p.moveTo(x, y)
      else {
        const am = a - Math.PI / n, k = 1.18 + r() * 0.12
        p.quadraticCurveTo(cx + Math.cos(am) * rx * k, cy + Math.sin(am) * ry * k, x, y)
      }
    }
  } else {
    const n = 22
    for (let i = 0; i < n; i++) {
      const a = (i / n) * Math.PI * 2
      const k = i % 2 ? 0.82 + r() * 0.1 : 1.0 + r() * 0.3
      p.lineTo(cx + Math.cos(a) * rx * k, cy + Math.sin(a) * ry * k)
    }
  }
  p.closePath()
  g.save()
  g.lineJoin = 'miter'
  g.strokeStyle = ink; g.lineWidth = Math.max(4, ry * 0.04); g.stroke(p)
  g.fillStyle = fill; g.fill(p)
  g.restore()
}

/**
 * Original cartoon characters (no real person/brand): 'head' — a deadpan heavy-lidded guy in a
 * sideways cap; 'pawn' — a chess pawn in shades. Flat colours, ink line, drop shadow.
 */
export function character(g: CanvasRenderingContext2D, kind: 'head' | 'pawn', x: number, y: number, s: number, colors: { skin: string; cap: string; extra: string }) {
  const lw = s * 0.045
  const inked = (p: Path2D, fill: string) => {
    g.fillStyle = INK; g.save(); g.translate(s * 0.05, s * 0.05); g.fill(p); g.restore()
    g.strokeStyle = INK; g.lineWidth = lw; g.stroke(p)
    g.fillStyle = fill; g.fill(p)
  }
  g.save()
  g.lineJoin = 'round'; g.lineCap = 'round'
  if (kind === 'head') {
    const head = new Path2D(); head.ellipse(x, y, s * 0.42, s * 0.48, 0, 0, Math.PI * 2)
    const ear = new Path2D(); ear.ellipse(x - s * 0.42, y + s * 0.04, s * 0.09, s * 0.13, 0, 0, Math.PI * 2)
    inked(ear, colors.skin); inked(head, colors.skin)
    // sideways cap
    const cap = new Path2D()
    cap.moveTo(x - s * 0.45, y - s * 0.1)
    cap.bezierCurveTo(x - s * 0.5, y - s * 0.62, x + s * 0.42, y - s * 0.7, x + s * 0.44, y - s * 0.16)
    cap.closePath()
    const brim = new Path2D()
    brim.moveTo(x + s * 0.3, y - s * 0.22); brim.quadraticCurveTo(x + s * 0.75, y - s * 0.3, x + s * 0.82, y - s * 0.12)
    brim.quadraticCurveTo(x + s * 0.6, y - s * 0.06, x + s * 0.32, y - s * 0.1); brim.closePath()
    inked(cap, colors.cap); inked(brim, shade(colors.cap, 0.75))
    g.fillStyle = '#fbf6ea'; g.fillRect(x - s * 0.25, y - s * 0.45, s * 0.12, s * 0.05)
    // heavy-lidded eyes
    for (const ex of [-0.16, 0.12]) {
      const eye = new Path2D(); eye.ellipse(x + ex * s, y + s * 0.02, s * 0.12, s * 0.1, 0, 0, Math.PI * 2)
      g.fillStyle = '#f4efe2'; g.fill(eye); g.strokeStyle = INK; g.lineWidth = lw * 0.8; g.stroke(eye)
      g.fillStyle = INK; g.beginPath(); g.arc(x + ex * s + s * 0.04, y + s * 0.05, s * 0.035, 0, 7); g.fill()
      g.fillStyle = colors.skin
      g.beginPath(); g.ellipse(x + ex * s, y + s * 0.02, s * 0.13, s * 0.11, 0, Math.PI, Math.PI * 2); g.fill()
      g.strokeStyle = INK; g.lineWidth = lw; g.beginPath(); g.moveTo(x + ex * s - s * 0.13, y + s * 0.02); g.lineTo(x + ex * s + s * 0.13, y + s * 0.02); g.stroke()
    }
    // nose + smirk
    g.strokeStyle = INK; g.lineWidth = lw * 0.9
    g.beginPath(); g.moveTo(x - s * 0.01, y + s * 0.08); g.quadraticCurveTo(x + s * 0.08, y + s * 0.2, x - s * 0.04, y + s * 0.22); g.stroke()
    g.beginPath(); g.moveTo(x - s * 0.16, y + s * 0.32); g.quadraticCurveTo(x + s * 0.02, y + s * 0.37, x + s * 0.2, y + s * 0.26); g.stroke()
    g.fillStyle = shade(colors.skin, 0.8)
    g.beginPath(); g.ellipse(x + s * 0.25, y + s * 0.2, s * 0.06, s * 0.04, 0, 0, 7); g.fill()
  } else {
    const p = new Path2D()
    p.arc(x, y - s * 0.42, s * 0.24, Math.PI * 0.72, Math.PI * 2.28)
    p.lineTo(x + s * 0.24, y - s * 0.12); p.quadraticCurveTo(x + s * 0.18, y - s * 0.05, x + s * 0.16, y + s * 0.05)
    p.quadraticCurveTo(x + s * 0.22, y + s * 0.32, x + s * 0.36, y + s * 0.42); p.lineTo(x + s * 0.4, y + s * 0.56)
    p.lineTo(x - s * 0.4, y + s * 0.56); p.lineTo(x - s * 0.36, y + s * 0.42)
    p.quadraticCurveTo(x - s * 0.22, y + s * 0.32, x - s * 0.16, y + s * 0.05)
    p.quadraticCurveTo(x - s * 0.18, y - s * 0.05, x - s * 0.24, y - s * 0.12)
    p.closePath()
    inked(p, colors.skin)
    // collar band
    g.fillStyle = colors.cap; g.fillRect(x - s * 0.3, y - s * 0.17, s * 0.6, s * 0.07)
    g.strokeStyle = INK; g.lineWidth = lw * 0.8; g.strokeRect(x - s * 0.3, y - s * 0.17, s * 0.6, s * 0.07)
    // base band
    g.fillStyle = colors.extra; g.fillRect(x - s * 0.38, y + s * 0.42, s * 0.76, s * 0.14)
    g.strokeRect(x - s * 0.38, y + s * 0.42, s * 0.76, s * 0.14)
    // shades
    g.fillStyle = INK
    g.beginPath(); g.ellipse(x - s * 0.1, y - s * 0.44, s * 0.1, s * 0.06, 0.1, 0, 7); g.fill()
    g.beginPath(); g.ellipse(x + s * 0.11, y - s * 0.44, s * 0.1, s * 0.06, -0.1, 0, 7); g.fill()
    g.fillRect(x - s * 0.04, y - s * 0.47, s * 0.08, s * 0.02)
    g.fillStyle = '#fbf6ea'; g.fillRect(x - s * 0.15, y - s * 0.47, s * 0.04, s * 0.02); g.fillRect(x + s * 0.06, y - s * 0.47, s * 0.04, s * 0.02)
    // grin
    g.strokeStyle = INK; g.lineWidth = lw * 0.8
    g.beginPath(); g.moveTo(x - s * 0.09, y - s * 0.32); g.quadraticCurveTo(x + s * 0.02, y - s * 0.26, x + s * 0.11, y - s * 0.33); g.stroke()
    // head shine
    g.fillStyle = '#fbf6ea'; g.beginPath(); g.ellipse(x - s * 0.1, y - s * 0.58, s * 0.06, s * 0.03, -0.5, 0, 7); g.fill()
  }
  g.restore()
}
