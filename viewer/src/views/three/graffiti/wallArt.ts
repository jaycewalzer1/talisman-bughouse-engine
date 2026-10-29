// Composes the whole graffiti layer for each wall into ONE transparent canvas (laid over the
// brick as a single decal plane in environment/Walls.tsx). Painted once, cached.
//
// Layering, bottom to top, like a real yard wall:
//   1. old faded pieces + tags (washed back toward the wall colour)
//   2. buffs — flat grey/beige painted-over rectangles
//   3. backdrops, the big pieces in the middle band, throw-ups along the bottom, straight letters
//      and throw-ups up top ("heaven"), characters, stars / crowns / arrows
//   4. tags everywhere in the gaps (an occupancy grid steers them away from the pieces), a few
//      on the buffs and on top of older stuff
// All words are made up (bughouse / chess / skate flavoured); no real writers, crews or brands.
import { rng } from '../environment/textures'
import { STYLE } from '../style.config'
import {
  INK, PAL, backdrop, bigArrow, buff, character, crown, paintPiece, paintTag, sparkle, star,
  type PieceSpec,
} from './painter'

type Box = [number, number, number, number]

const TAG_WORDS = [
  'ZUG', 'DRP', 'SKRT', 'CHEK', 'PAWN', 'FORK', 'ROOK', 'MATE', 'OXO', 'ZONK', 'KNGHT', 'TLSMN',
  'BUG', 'HAUS', 'FLAG', 'PIN', 'BLITZ', 'TEMPO', 'EP', 'KRWN', 'SKWR', 'ZWSH', 'DBL', 'BRD', 'CLK',
  'TSK', 'BHC', 'NXT', 'SAC', 'GMBT', 'K9', 'ZUG9', 'BH2', 'FNCH', 'DROP', 'HOOK', 'GRND', 'KICK',
]
const TAG_COLORS = [INK, INK, INK, INK, PAL.silver, PAL.offwhite, PAL.plum, PAL.rust, PAL.denim, PAL.mustard, PAL.teal, PAL.wine, PAL.pink, PAL.lime, PAL.sky]

/** Wall-space helper: metres (x along the wall, y up from the ground) -> canvas px. */
class Wall {
  readonly c: HTMLCanvasElement
  readonly g: CanvasRenderingContext2D
  readonly occ: Float32Array
  readonly gw: number; readonly gh: number
  readonly wm: number; readonly hm: number; readonly ppm: number; readonly x0m: number; readonly cell: number
  constructor(wm: number, hm: number, ppm: number, x0m: number, cell = 0.25) {
    this.wm = wm; this.hm = hm; this.ppm = ppm; this.x0m = x0m; this.cell = cell
    this.c = document.createElement('canvas')
    this.c.width = Math.round(wm * ppm); this.c.height = Math.round(hm * ppm)
    this.g = this.c.getContext('2d')!
    this.gw = Math.ceil(wm / cell); this.gh = Math.ceil(hm / cell)
    this.occ = new Float32Array(this.gw * this.gh)
  }
  X(xm: number) { return (xm - this.x0m) * this.ppm }
  Y(ym: number) { return (this.hm - ym) * this.ppm }
  M(m: number) { return m * this.ppm }
  /** mark a px box in the occupancy grid */
  mark(b: Box, v = 1) {
    const s = this.ppm * this.cell
    for (let j = Math.max(0, Math.floor(b[1] / s)); j < Math.min(this.gh, Math.ceil(b[3] / s)); j++)
      for (let i = Math.max(0, Math.floor(b[0] / s)); i < Math.min(this.gw, Math.ceil(b[2] / s)); i++)
        this.occ[j * this.gw + i] = Math.max(this.occ[j * this.gw + i], v)
  }
  cover(b: Box): number {
    const s = this.ppm * this.cell
    let sum = 0, n = 0
    for (let j = Math.max(0, Math.floor(b[1] / s)); j < Math.min(this.gh, Math.ceil(b[3] / s)); j++)
      for (let i = Math.max(0, Math.floor(b[0] / s)); i < Math.min(this.gw, Math.ceil(b[2] / s)); i++) { sum += this.occ[j * this.gw + i]; n++ }
    return n ? sum / n : 1
  }
  /** a lettered piece, centre in metres, letter height in metres */
  piece(s: Omit<PieceSpec, 'h'> & { h: number }, xm: number, ym: number, rot = 0, alpha = 1, occ = 1) {
    const b = paintPiece(this.g, { ...s, h: this.M(s.h) }, this.X(xm), this.Y(ym), rot, alpha)
    if (occ) this.mark(b, occ)
    return b
  }
  /** wash everything painted so far back toward the wall colour (fades the old layer) */
  wash(color: string, a: number) {
    const g = this.g
    g.save(); g.globalCompositeOperation = 'source-atop'; g.globalAlpha = a
    g.fillStyle = color; g.fillRect(0, 0, this.c.width, this.c.height)
    g.restore()
  }
  /** scatter handstyle tags, preferring empty cells */
  tags(n: number, seed: number, o: { hMin?: number; hMax?: number; yMin?: number; yMax?: number; xMin?: number; xMax?: number; maxCover?: number; alpha?: number; colors?: string[] } = {}) {
    const r = rng(seed)
    const hMin = o.hMin ?? 0.2, hMax = o.hMax ?? 0.48
    const xMin = o.xMin ?? this.x0m, xMax = o.xMax ?? this.x0m + this.wm
    const yMin = o.yMin ?? 0.25, yMax = o.yMax ?? this.hm - 0.25
    const cols = o.colors ?? TAG_COLORS
    for (let k = 0; k < n; k++) {
      const word = TAG_WORDS[Math.floor(r() * TAG_WORDS.length)]
      const h = hMin + (hMax - hMin) * r() * r()
      const estW = h * word.length * 0.62
      let best: Box | null = null, bestCover = Infinity, bx = 0, by = 0
      for (let t = 0; t < 12; t++) {
        const xm = xMin + (xMax - xMin - estW) * r(), ym = yMin + (yMax - yMin) * r()
        const b: Box = [this.X(xm), this.Y(ym + h * 0.7), this.X(xm + estW), this.Y(ym - h * 0.7)]
        const cv = this.cover(b) + r() * 0.05
        if (cv < bestCover) { bestCover = cv; best = b; bx = xm; by = ym }
      }
      if (!best || bestCover > (o.maxCover ?? 0.55)) continue
      const color = cols[Math.floor(r() * cols.length)]
      const fat = r() < 0.2
      const b = paintTag(this.g, {
        text: word, h: this.M(h), seed: Math.floor(r() * 1e6), color,
        halo: r() < 0.18 ? (color === INK ? PAL.offwhite : INK) : null,
        weight: fat ? 1.9 : 0.8 + r() * 0.5, sx: 0.55 + r() * 0.45,
        crown: r() < 0.08, underline: r() < 0.22, drips: fat || r() < 0.25 ? 1 + Math.floor(r() * 3) : 0,
        lean: 0.5 + r() * 0.9,
      }, this.X(bx), this.Y(by), (r() - 0.5) * 0.2, o.alpha ?? 1)
      this.mark(b, 0.7)
    }
  }
}

/** big tag over something (a buff, an older piece) */
function overTag(w: Wall, text: string, xm: number, ym: number, hm: number, color: string, seed: number, extra: Partial<Parameters<typeof paintTag>[1]> = {}) {
  w.mark(paintTag(w.g, { text, h: w.M(hm), seed, color, weight: 1.1, sx: 0.75, ...extra }, w.X(xm), w.Y(ym), -0.04), 0.8)
}

// ---------------------------------------------------------------- back wall (26 x 6 m, x -13..13)

function paintBack(): HTMLCanvasElement {
  const ppm = STYLE.graffiti.pxPerMeter
  const w = new Wall(26, 6, ppm, -13)
  const G = w.g
  // hidden behind the bodega and the roll-down gate: don't waste paint there
  w.mark([w.X(-7.2), w.Y(3.25), w.X(-3.4), w.Y(0)], 2)
  w.mark([w.X(-10.45), w.Y(3.15), w.X(-7.95), w.Y(0)], 2)

  // 1) the old layer --------------------------------------------------------------------------
  if (STYLE.graffiti.oldLayer) {
    w.piece({ text: 'GAMBT', kind: 'wild', h: 1.05, seed: 41, fill: [PAL.olive, PAL.denim], fillMode: 'bands', cut: PAL.sand, block: PAL.moss, wild: 0.8, arrows: 0.35, sx: 0.8 }, 1.2, 4.15, -0.03, 1, 0)
    w.piece({ text: 'ROOK', kind: 'bubble', h: 0.9, seed: 42, fill: [PAL.sky], ink: PAL.wine, shine: false }, 4.6, 1.5, 0.04, 1, 0)
    w.piece({ text: 'STLMT', kind: 'wild', h: 0.95, seed: 43, fill: [PAL.brick, PAL.mustard], fillMode: 'split', cut: PAL.offwhite, block: PAL.dgrey, wild: 0.6, arrows: 0.3, sx: 0.75 }, 9.4, 4.4, 0.02, 1, 0)
    w.piece({ text: 'EP', kind: 'bubble', h: 1.1, seed: 44, fill: [PAL.mustard], shine: false }, -11.8, 3.7, -0.06, 1, 0)
    w.piece({ text: 'FLAG', kind: 'block', h: 0.8, seed: 45, fill: [PAL.grey], block: PAL.dgrey }, -1.6, 1.0, 0, 1, 0)
    w.piece({ text: 'BLITZ', kind: 'wild', h: 0.8, seed: 46, fill: [PAL.teal, PAL.plum], fillMode: 'bands', cut: PAL.offwhite, wild: 0.5, sx: 0.7 }, -5.3, 5.35, 0.03, 1, 0)
    w.piece({ text: 'KICK', kind: 'bubble', h: 0.7, seed: 47, fill: [PAL.pink], shine: false }, 12.1, 3.6, 0.08, 1, 0)
    w.tags(70, 401, { maxCover: 9, hMax: 0.55 })
    w.wash('#6b5b52', STYLE.graffiti.oldFade)
    w.occ.fill(0)
    w.mark([w.X(-7.2), w.Y(3.25), w.X(-3.4), w.Y(0)], 2)
    w.mark([w.X(-10.45), w.Y(3.15), w.X(-7.95), w.Y(0)], 2)
  }

  // 2) buffs ----------------------------------------------------------------------------------
  const buffs: [number, number, number, number, string][] = [
    [3.0, 3.55, 5.6, 4.6, '#7d756b'], [-12.8, 4.1, -10.9, 5.7, '#8a7e70'], [7.3, 0.15, 8.75, 1.75, '#6a6c66'], [-2.9, 4.4, -1.2, 5.6, '#857868'],
  ]
  buffs.forEach(([x0, y0, x1, y1, c], i) => buff(G, w.X(x0), w.Y(y1), w.X(x1), w.Y(y0), c, 300 + i))

  // 3) the main layer -------------------------------------------------------------------------
  // backdrops behind the burners
  backdrop(G, w.X(4.0), w.Y(3.75), w.X(8.9), w.Y(1.6), '#b8863a', 61, 'shards')
  backdrop(G, w.X(-3.6), w.Y(3.75), w.X(4.3), w.Y(1.55), '#3d6863', 62, 'cloud')
  // middle band: the burners
  w.piece({ text: 'BUGHAUS', kind: 'wild', h: 1.3, seed: 7, fill: [PAL.mustard, PAL.orange, PAL.rust], fillMode: 'bands', cut: PAL.offwhite, block: PAL.plum, wild: 0.75, arrows: 0.4, sx: 0.78, drips: 5, breaks: 1 }, 0.35, 2.65)
  w.piece({ text: 'ZUGZ', kind: 'wild', h: 1.3, seed: 8, fill: [PAL.teal, PAL.sky], fillMode: 'split', cut: PAL.offwhite, block: INK, wild: 0.9, arrows: 0.5, sx: 0.85, drips: 3, breaks: 1 }, 6.45, 2.7, 0.03)
  w.piece({ text: 'KNGHT', kind: 'wild', h: 1.05, seed: 9, fill: [PAL.offwhite, PAL.silver, PAL.denim], fillMode: 'fade', cut: PAL.brick, block: PAL.denim, wild: 0.35, arrows: 0.25, sx: 0.68, drips: 2 }, 11.05, 2.55, -0.02)
  // left of the gate
  w.piece({ text: 'DRP', kind: 'bubble', h: 0.75, seed: 10, fill: [PAL.silver], drips: 2 }, -11.75, 1.55, -0.05)
  // bottom band: throw-ups
  w.piece({ text: 'SKRT', kind: 'bubble', h: 0.85, seed: 11, fill: [PAL.silver], ink: INK, drips: 2 }, -1.7, 0.75, -0.03)
  w.piece({ text: 'FORK', kind: 'bubble', h: 0.9, seed: 12, fill: [PAL.mustard, PAL.offwhite], fillMode: 'bubbles', drips: 3 }, 2.25, 0.85, 0.02)
  w.piece({ text: 'OXO', kind: 'bubble', h: 0.85, seed: 13, fill: [PAL.offwhite], ink: PAL.denim, block: null }, 5.75, 0.85, -0.04)
  w.piece({ text: 'PAWN', kind: 'bubble', h: 0.8, seed: 14, fill: [PAL.rust, PAL.brick], fillMode: 'bands', drips: 2 }, 10.15, 0.85, 0.03)
  w.piece({ text: 'CHEK', kind: 'bubble', h: 0.45, seed: 15, fill: [PAL.plum], drips: 1 }, 12.25, 0.6, -0.06)
  // top band: straight letters + throw-ups ("heaven" spots)
  w.piece({ text: 'TLSMN', kind: 'block', h: 0.95, seed: 16, fill: [PAL.offwhite, PAL.sand], fillMode: 'bands', block: PAL.rust, blockDir: [0.13, 0.12], shine: true, sx: 0.8 }, 1.0, 5.0)
  w.piece({ text: 'MATE', kind: 'bubble', h: 0.85, seed: 17, fill: [PAL.teal], drips: 2 }, 6.25, 4.95, 0.04)
  w.piece({ text: 'ZONK', kind: 'wild', h: 0.85, seed: 18, fill: [PAL.pink, PAL.plum], fillMode: 'bands', cut: PAL.offwhite, block: INK, wild: 0.6, arrows: 0.4, sx: 0.75, drips: 2 }, 10.8, 5.05, -0.03)
  w.piece({ text: 'FORK', kind: 'wild', h: 0.95, seed: 19, fill: [PAL.olive, PAL.lime], fillMode: 'split', cut: PAL.offwhite, block: PAL.wine, wild: 0.7, arrows: 0.4, sx: 0.72 }, -5.3, 4.55, 0.02)
  w.piece({ text: 'BUG', kind: 'bubble', h: 0.55, seed: 20, fill: [PAL.offwhite], ink: PAL.brick, drips: 1 }, -11.75, 2.75, 0.05)
  // characters
  character(G, 'head', w.X(-9.15), w.Y(4.65), w.M(1.35), { skin: '#c99068', cap: PAL.denim, extra: PAL.mustard })
  w.mark([w.X(-10.1), w.Y(5.6), w.X(-8.0), w.Y(3.8)])
  character(G, 'pawn', w.X(8.55), w.Y(4.95), w.M(1.05), { skin: PAL.offwhite, cap: PAL.rust, extra: PAL.teal })
  w.mark([w.X(8.1), w.Y(5.6), w.X(9.0), w.Y(4.3)])
  // stars, crowns, arrows
  star(G, w.X(-3.25), w.Y(3.85), w.M(0.28), PAL.mustard, 0.2)
  star(G, w.X(4.05), w.Y(4.0), w.M(0.2), PAL.offwhite, -0.3)
  star(G, w.X(-7.6), w.Y(5.6), w.M(0.22), PAL.pink, 0.1)
  crown(G, w.X(9.55), w.Y(3.45), w.M(0.55), PAL.mustard)
  crown(G, w.X(-0.6), w.Y(1.55), w.M(0.4), PAL.offwhite)
  bigArrow(G, [[w.X(3.9), w.Y(5.55)], [w.X(4.6), w.Y(5.9)], [w.X(5.2), w.Y(5.45)]], w.M(0.11), PAL.rust)
  bigArrow(G, [[w.X(4.5), w.Y(1.75)], [w.X(6.4), w.Y(1.35)], [w.X(8.2), w.Y(1.7)]], w.M(0.1), PAL.offwhite)
  bigArrow(G, [[w.X(-12.7), w.Y(3.5)], [w.X(-11.7), w.Y(3.75)], [w.X(-10.8), w.Y(3.4)]], w.M(0.09), PAL.teal)
  for (const [x, y, s] of [[-2.6, 3.3, 0.16], [3.7, 3.55, 0.14], [8.5, 2.9, 0.15], [-1.2, 5.6, 0.12], [12.6, 5.5, 0.14]] as const)
    sparkle(G, w.X(x), w.Y(y), w.M(s))

  // 4) tags: on the buffs, over old stuff, then everywhere in the gaps -----------------------
  overTag(w, 'DBL', 3.2, 4.1, 0.55, INK, 501, { underline: true })
  overTag(w, 'SAC', 4.6, 3.85, 0.35, PAL.offwhite, 502, { halo: INK })
  overTag(w, 'KRWN', -12.6, 4.9, 0.55, PAL.wine, 503, { crown: true })
  overTag(w, 'NXT', 7.45, 1.15, 0.45, PAL.offwhite, 504, { drips: 2 })
  overTag(w, 'ZWSH', -2.8, 5.0, 0.45, INK, 505, { underline: true })
  w.tags(STYLE.graffiti.tagCount, 777)
  w.tags(Math.round(STYLE.graffiti.tagCount * 0.25), 778, { maxCover: 1.0, hMax: 0.3 })
  return w.c
}

// ---------------------------------------------------------------- side wall (13 x 6 m)
// Canvas x = 0 is the end nearest the camera (world z = 7); the last ~1.8 m sit behind the
// back wall and are never seen.

function paintSide(): HTMLCanvasElement {
  const ppm = STYLE.graffiti.pxPerMeter
  const w = new Wall(13, 6, ppm, 0)
  const G = w.g
  w.mark([w.X(11.2), 0, w.c.width, w.c.height], 2)
  if (STYLE.graffiti.oldLayer) {
    w.piece({ text: 'HOOK', kind: 'wild', h: 1.0, seed: 51, fill: [PAL.denim, PAL.sky], fillMode: 'bands', cut: PAL.offwhite, block: PAL.dgrey, wild: 0.6, sx: 0.8 }, 3.0, 4.4, 0.02, 1, 0)
    w.piece({ text: 'PIN', kind: 'bubble', h: 1.0, seed: 52, fill: [PAL.lime], shine: false }, 8.6, 1.2, 0.03, 1, 0)
    w.tags(30, 601, { maxCover: 9, xMax: 11.2 })
    w.wash('#6b5b52', STYLE.graffiti.oldFade)
    w.occ.fill(0)
    w.mark([w.X(11.2), 0, w.c.width, w.c.height], 2)
  }
  buff(G, w.X(0.3), w.Y(5.7), w.X(2.4), w.Y(4.2), '#7d756b', 311)
  buff(G, w.X(9.2), w.Y(2.2), w.X(10.9), w.Y(0.3), '#6a6c66', 312)
  backdrop(G, w.X(2.5), w.Y(3.8), w.X(10.4), w.Y(1.6), '#8e4a3a', 63, 'shards')
  w.piece({ text: 'GAMBIT', kind: 'wild', h: 1.3, seed: 31, fill: [PAL.offwhite, PAL.mustard, PAL.olive], fillMode: 'bands', cut: PAL.teal, block: INK, wild: 0.8, arrows: 0.45, sx: 0.72, drips: 4, breaks: 1 }, 6.45, 2.7)
  w.piece({ text: 'ZUG', kind: 'bubble', h: 0.9, seed: 32, fill: [PAL.silver], drips: 2 }, 1.6, 0.85, -0.03)
  w.piece({ text: 'DRPZ', kind: 'bubble', h: 0.85, seed: 33, fill: [PAL.plum, PAL.pink], fillMode: 'bands', drips: 2 }, 5.4, 0.8, 0.03)
  w.piece({ text: 'TEMPO', kind: 'block', h: 0.85, seed: 34, fill: [PAL.sky, PAL.denim], fillMode: 'bands', block: PAL.plum, blockDir: [0.12, 0.11], sx: 0.72 }, 6.4, 5.0)
  w.piece({ text: 'EP', kind: 'bubble', h: 0.9, seed: 35, fill: [PAL.mustard], ink: PAL.wine }, 9.9, 4.8, -0.05)
  character(G, 'pawn', w.X(1.25), w.Y(2.7), w.M(1.2), { skin: PAL.mustard, cap: PAL.plum, extra: PAL.denim })
  w.mark([w.X(0.7), w.Y(3.5), w.X(1.8), w.Y(1.9)])
  star(G, w.X(2.4), w.Y(3.7), w.M(0.25), PAL.offwhite, 0.25)
  crown(G, w.X(3.4), w.Y(3.85), w.M(0.45), PAL.mustard)
  bigArrow(G, [[w.X(7.4), w.Y(1.55)], [w.X(8.4), w.Y(1.2)], [w.X(9.1), w.Y(1.6)]], w.M(0.09), PAL.offwhite)
  overTag(w, 'CLK', 0.6, 5.0, 0.55, INK, 521, { underline: true })
  overTag(w, 'TSK', 9.5, 1.3, 0.5, PAL.offwhite, 522, { halo: INK })
  w.tags(Math.round(STYLE.graffiti.tagCount * 0.5), 888, { xMax: 11.2 })
  w.tags(Math.round(STYLE.graffiti.tagCount * 0.12), 889, { xMax: 11.2, maxCover: 1.0, hMax: 0.3 })
  return w.c
}

let back: HTMLCanvasElement | null = null
let side: HTMLCanvasElement | null = null
export function backWallArt(): HTMLCanvasElement { return back ??= paintBack() }
export function sideWallArt(): HTMLCanvasElement { return side ??= paintSide() }
