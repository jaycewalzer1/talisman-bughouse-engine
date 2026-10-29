// Procedural canvas textures (no downloads). Each one is generated once and cached.
// The core street assets (brick, wet asphalt + puddles, plywood, rusty el steel, chain-link, milk
// crate, worn vinyl board, grip) are the ORIGINAL generators at their original resolution; grime
// lives in the textures (soot streaks, cracks, stains), the palette stays muted. retroMaterial
// samples them nearest + mip-mapped. All art is original: abstract shapes and made-up words only.
// To use real CC0 photos later (Poly Haven / ambientCG), replace a function body with
// `new TextureLoader().load('/textures/....jpg')` and keep the same wrap/repeat settings.
import { CanvasTexture, LinearFilter, LinearMipmapLinearFilter, RepeatWrapping, SRGBColorSpace, type Texture } from 'three'

const cache = new Map<string, Texture>()

/** Muted street colours for stickers, decks, cans and the like (grimy, not grey). */
const GRIME = ['#9a4a32', '#3f6a8a', '#7a8a3a', '#b08a3a', '#6a4a7a', '#c8c0aa', '#2a2c30']

/** Seeded random so textures look the same on every load. */
export function rng(seed: number) {
  let s = seed >>> 0
  return () => { s = (s * 1664525 + 1013904223) >>> 0; return s / 4294967296 }
}

function canvas(w: number, h: number): [HTMLCanvasElement, CanvasRenderingContext2D] {
  const c = document.createElement('canvas'); c.width = w; c.height = h
  return [c, c.getContext('2d')!]
}

function tex(key: string, make: () => HTMLCanvasElement, opts: { color?: boolean; repeat?: [number, number] } = {}): Texture {
  const hit = cache.get(key); if (hit) return hit
  const t = new CanvasTexture(make())
  if (opts.color !== false) t.colorSpace = SRGBColorSpace
  t.wrapS = t.wrapT = RepeatWrapping
  if (opts.repeat) t.repeat.set(...opts.repeat)
  cache.set(key, t)
  return t
}

function pick<T>(r: () => number, a: readonly T[]): T { return a[Math.floor(r() * a.length)] }

function speckle(g: CanvasRenderingContext2D, w: number, h: number, n: number, r: () => number, colors: string[], size = 2) {
  for (let i = 0; i < n; i++) {
    g.fillStyle = colors[Math.floor(r() * colors.length)]
    g.fillRect(r() * w, r() * h, size * r() + 0.5, size * r() + 0.5)
  }
}

/** A blobby paint splat (hard edge, a few satellite drops). */
function splat(g: CanvasRenderingContext2D, x: number, y: number, rad: number, color: string, r: () => number) {
  g.fillStyle = color
  g.beginPath()
  for (let a = 0; a <= Math.PI * 2 + 0.01; a += 0.45) {
    const rr = rad * (0.65 + r() * 0.5)
    g.lineTo(x + Math.cos(a) * rr, y + Math.sin(a) * rr)
  }
  g.fill()
  for (let i = 0; i < 4; i++) {
    const a = r() * 7, d = rad * (1.2 + r())
    g.fillRect(x + Math.cos(a) * d, y + Math.sin(a) * d, 1 + r() * 2, 1 + r() * 2)
  }
}

/** Weathered stickers: rects, circles, stars, slap strips with tiny abstract glyphs. */
export function sticker(g: CanvasRenderingContext2D, x: number, y: number, s: number, r: () => number) {
  const bg = pick(r, GRIME)
  const fg = pick(r, GRIME.filter(c => c !== bg))
  g.save(); g.translate(x, y); g.rotate((r() - 0.5) * 0.9)
  const kind = Math.floor(r() * 4)
  g.fillStyle = bg
  if (kind === 0) g.fillRect(-s / 2, -s / 3, s, s * 0.66)
  else if (kind === 1) { g.beginPath(); g.arc(0, 0, s / 2, 0, 7); g.fill() }
  else if (kind === 2) {
    g.beginPath()
    for (let i = 0; i < 10; i++) { const a = i * Math.PI / 5, rr = i % 2 ? s * 0.22 : s * 0.5; g.lineTo(Math.cos(a) * rr, Math.sin(a) * rr) }
    g.fill()
  } else { g.fillRect(-s / 2, -s / 5, s, s * 0.4) }   // slap strip
  g.fillStyle = fg
  const glyph = Math.floor(r() * 3)                    // eye, dots or bar
  if (glyph === 0) { g.beginPath(); g.ellipse(0, 0, s * 0.22, s * 0.12, 0, 0, 7); g.fill() }
  else if (glyph === 1) { for (let i = -1; i <= 1; i++) g.fillRect(i * s * 0.2 - 1, -1, 2, 2) }
  else { g.fillRect(-s * 0.3, -s * 0.06, s * 0.6, s * 0.12) }
  // weathering: dirt over one corner
  g.fillStyle = 'rgba(20,18,16,0.35)'; g.fillRect(-s / 2, s * 0.1, s * 0.4, s * 0.25)
  g.restore()
}

// ---------- brick (original) ----------
function brickCanvas(bump: boolean): HTMLCanvasElement {
  const W = 512, H = 512, rows = 16, cols = 4
  const [c, g] = canvas(W, H); const r = rng(7)
  g.fillStyle = bump ? '#202020' : '#3b3632'; g.fillRect(0, 0, W, H)
  const bh = H / rows, bw = W / cols
  for (let y = 0; y < rows; y++) for (let x = -1; x < cols; x++) {
    const ox = (y % 2) * bw / 2
    const px = x * bw + ox + 3, py = y * bh + 3
    if (bump) { g.fillStyle = `rgb(${200 + r() * 40},${200 + r() * 40},${200 + r() * 40})` }
    else {
      const base = 70 + r() * 50, red = base + 40 + r() * 30
      g.fillStyle = `rgb(${red},${base * 0.55},${base * 0.45})`
    }
    g.fillRect(px, py, bw - 6, bh - 6)
  }
  if (!bump) {
    speckle(g, W, H, 5000, r, ['rgba(0,0,0,0.25)', 'rgba(255,255,255,0.06)', 'rgba(60,30,20,0.3)'])
    // soot + water streaks
    for (let i = 0; i < 40; i++) {
      const x = r() * W, w = 4 + r() * 20, len = 60 + r() * 300
      const grd = g.createLinearGradient(0, 0, 0, len)
      grd.addColorStop(0, 'rgba(10,10,12,0.35)'); grd.addColorStop(1, 'rgba(10,10,12,0)')
      g.fillStyle = grd; g.fillRect(x, 0, w, len)
    }
    const grime = g.createLinearGradient(0, H * 0.6, 0, H)
    grime.addColorStop(0, 'rgba(0,0,0,0)'); grime.addColorStop(1, 'rgba(5,5,5,0.55)')
    g.fillStyle = grime; g.fillRect(0, 0, W, H)
  }
  return c
}
export const brickTexture = () => tex('brick', () => brickCanvas(false))
export const brickBump = () => tex('brickBump', () => brickCanvas(true), { color: false })

// ---------- wet asphalt + puddles (original) ----------
/** Puddle blobs; `k` scales radii so a smaller mask canvas lines up with the 1024 px asphalt. */
function puddles(g: CanvasRenderingContext2D, W: number, H: number, r: () => number, fill: string, k = 1) {
  g.fillStyle = fill
  for (let i = 0; i < 9; i++) {
    const cx = r() * W, cy = r() * H, rad = (30 + r() * 110) * k
    g.beginPath()
    for (let a = 0; a <= Math.PI * 2 + 0.01; a += 0.3) {
      const rr = rad * (0.6 + 0.4 * Math.sin(a * 3 + i) * r() + 0.3)
      g.lineTo(cx + Math.cos(a) * rr * 1.6, cy + Math.sin(a) * rr)
    }
    g.fill()
  }
}
function asphaltCanvas(rough: boolean): HTMLCanvasElement {
  const W = 1024, H = 1024
  const [c, g] = canvas(W, H); const r = rng(11)
  g.fillStyle = rough ? '#d8d8d8' : '#2a2a2c'; g.fillRect(0, 0, W, H)
  speckle(g, W, H, 40000, r, rough ? ['#ffffff', '#b0b0b0', '#999'] : ['#3a3a3d', '#1c1c1e', '#47464a', '#252528'], 3)
  // patches + cracks
  for (let i = 0; i < 6; i++) {
    g.fillStyle = rough ? 'rgba(160,160,160,0.6)' : 'rgba(20,20,22,0.5)'
    g.fillRect(r() * W, r() * H, 80 + r() * 200, 60 + r() * 150)
  }
  g.strokeStyle = rough ? '#ffffff' : 'rgba(8,8,8,0.9)'; g.lineWidth = 2
  for (let i = 0; i < 14; i++) {
    let x = r() * W, y = r() * H; g.beginPath(); g.moveTo(x, y)
    for (let k = 0; k < 12; k++) { x += (r() - 0.5) * 60; y += (r() - 0.5) * 60; g.lineTo(x, y) }
    g.stroke()
  }
  // painted road line remnants
  if (!rough) { g.fillStyle = 'rgba(200,170,60,0.35)'; g.fillRect(0, H * 0.72, W, 18) }
  // puddles: dark + smooth (roughness map black = mirror-like)
  const r2 = rng(99)
  puddles(g, W, H, r2, rough ? '#141414' : 'rgba(8,10,14,0.85)')
  return c
}
export const asphaltTexture = () => tex('asphalt', () => asphaltCanvas(false), { repeat: [5, 4] })
export const asphaltRoughness = () => tex('asphaltR', () => asphaltCanvas(true), { color: false, repeat: [5, 4] })
/** Old name from the restyle; the ground is wet asphalt again. */
export const concreteTexture = asphaltTexture

/**
 * Puddle mask (white = water) lined up with the puddles baked into asphaltTexture, at 1/4 size.
 * Linear-filtered so the glossy overlay (WetGround.tsx) gets soft-ish edges.
 */
export const puddleMaskTexture = () => {
  const t = tex('puddleMask', () => {
    const S = 256; const [c, g] = canvas(S, S)
    g.fillStyle = '#000000'; g.fillRect(0, 0, S, S)
    puddles(g, S, S, rng(99), '#ffffff', S / 1024)
    return c
  }, { color: false })
  t.magFilter = LinearFilter; t.minFilter = LinearMipmapLinearFilter
  return t
}

// ---------- plywood (original) ----------
export const plywoodTexture = () => tex('plywood', () => {
  const W = 512, H = 512; const [c, g] = canvas(W, H); const r = rng(3)
  g.fillStyle = '#a7814f'; g.fillRect(0, 0, W, H)
  for (let y = 0; y < H; y += 2) {
    g.fillStyle = `rgba(${90 + r() * 40},${60 + r() * 20},${30},${0.15 + 0.15 * Math.sin(y * 0.09 + r())})`
    g.fillRect(0, y, W, 2)
  }
  for (let i = 0; i < 6; i++) { // knots
    const x = r() * W, y = r() * H
    g.fillStyle = 'rgba(70,40,20,0.6)'; g.beginPath(); g.ellipse(x, y, 18 + r() * 14, 6 + r() * 4, 0, 0, 7); g.fill()
  }
  for (let i = 0; i < 10; i++) { // water/soda stains
    g.strokeStyle = `rgba(40,25,10,${0.2 + r() * 0.3})`; g.lineWidth = 3
    g.beginPath(); g.arc(r() * W, r() * H, 15 + r() * 40, 0, 7); g.stroke()
  }
  speckle(g, W, H, 3000, r, ['rgba(0,0,0,0.2)', 'rgba(255,240,200,0.1)'])
  return c
})

// ---------- weathered sticker-bombed steel (rail, column feet) ----------
export const stickerTexture = () => tex('stickers', () => {
  const W = 128, H = 64; const [c, g] = canvas(W, H); const r = rng(41)
  g.fillStyle = '#34383a'; g.fillRect(0, 0, W, H)
  for (let i = 0; i < 40; i++) sticker(g, r() * W, r() * H, 8 + r() * 10, r)
  speckle(g, W, H, 500, r, ['rgba(0,0,0,0.35)', 'rgba(120,60,20,0.35)', 'rgba(255,255,255,0.08)'], 2)
  return c
})

// ---------- rusty painted steel (el columns, girders) (original) ----------
export const rustTexture = () => tex('rust', () => {
  const W = 256, H = 256; const [c, g] = canvas(W, H); const r = rng(21)
  g.fillStyle = '#2f3b33'; g.fillRect(0, 0, W, H)
  for (let i = 0; i < 70; i++) {
    const x = r() * W, y = r() * H, rad = 3 + r() * 22
    const grd = g.createRadialGradient(x, y, 0, x, y, rad)
    grd.addColorStop(0, 'rgba(140,60,20,0.95)'); grd.addColorStop(0.6, 'rgba(100,45,20,0.6)'); grd.addColorStop(1, 'rgba(80,40,20,0)')
    g.fillStyle = grd; g.fillRect(x - rad, y - rad, rad * 2, rad * 2)
  }
  for (let i = 0; i < 30; i++) { g.fillStyle = 'rgba(120,50,15,0.35)'; g.fillRect(r() * W, r() * H, 2, 20 + r() * 60) }
  speckle(g, W, H, 4000, r, ['rgba(0,0,0,0.3)', 'rgba(180,90,40,0.4)', 'rgba(200,200,180,0.08)'])
  return c
})

// ---------- chain-link fence (alpha-tested) (original) ----------
export const chainLinkTexture = () => tex('chain', () => {
  const W = 64, H = 64; const [c, g] = canvas(W, H)
  g.clearRect(0, 0, W, H)
  g.strokeStyle = '#b8bcc0'; g.lineWidth = 4
  g.beginPath(); g.moveTo(0, H / 2); g.lineTo(W / 2, 0); g.lineTo(W, H / 2); g.lineTo(W / 2, H); g.closePath(); g.stroke()
  return c
}, { repeat: [40, 12] })

// ---------- milk crate (grid of slots, alpha-tested; tinted by instance colour) (original) ----------
export const crateTexture = () => tex('crate', () => {
  const W = 128, H = 128; const [c, g] = canvas(W, H)
  g.fillStyle = '#ffffff'; g.fillRect(0, 0, W, H)
  g.globalCompositeOperation = 'destination-out'
  for (let y = 0; y < 4; y++) for (let x = 0; x < 6; x++) {
    g.fillRect(12 + x * 18, 16 + y * 26, 10, 18)
  }
  g.globalCompositeOperation = 'source-over'
  g.fillStyle = 'rgba(0,0,0,0.25)'; g.fillRect(0, 0, W, 10); g.fillRect(0, H - 10, W, 10)
  return c
})

// ---------- roll-up vinyl chess board, worn (original) ----------
export const boardTexture = () => tex('board', () => {
  const W = 512, H = 512, m = 32; const [c, g] = canvas(W, H); const r = rng(5)
  g.fillStyle = '#e9e1c9'; g.fillRect(0, 0, W, H)
  const s = (W - 2 * m) / 8
  for (let y = 0; y < 8; y++) for (let x = 0; x < 8; x++) {
    g.fillStyle = (x + y) % 2 === 1 ? '#3f6b45' : '#e7dfc4'
    g.fillRect(m + x * s, m + y * s, s, s)
  }
  g.fillStyle = '#2b2b2b'; g.font = 'bold 18px monospace'; g.textAlign = 'center'
  for (let i = 0; i < 8; i++) {
    g.fillText('abcdefgh'[i], m + i * s + s / 2, H - 9)
    g.fillText(String(8 - i), 14, m + i * s + s / 2 + 6)
  }
  speckle(g, W, H, 6000, r, ['rgba(0,0,0,0.12)', 'rgba(255,255,255,0.12)', 'rgba(90,60,20,0.15)'], 3)
  for (let i = 0; i < 5; i++) { g.strokeStyle = 'rgba(80,50,20,0.25)'; g.lineWidth = 4; g.beginPath(); g.arc(r() * W, r() * H, 20 + r() * 30, 0, 7); g.stroke() }
  g.fillStyle = 'rgba(0,0,0,0.07)'; g.fillRect(0, H / 2 - 3, W, 6) // fold crease
  return c
}, { repeat: [1, 1] })

// ---------- grip tape (original) ----------
export const gripTexture = () => tex('grip', () => {
  const W = 128, H = 128; const [c, g] = canvas(W, H); const r = rng(8)
  g.fillStyle = '#151515'; g.fillRect(0, 0, W, H)
  speckle(g, W, H, 2500, r, ['#2a2a2a', '#080808', '#333'], 2)
  return c
}, { repeat: [1, 3] })

// ---------- skateboard deck graphic (bottom), faded + scraped ----------
export const deckTexture = (seed = 1) => tex(`deck${seed}`, () => {
  const W = 32, H = 128; const [c, g] = canvas(W, H); const r = rng(60 + seed)
  const a = pick(r, GRIME), b = pick(r, GRIME.filter(x => x !== a))
  g.fillStyle = a; g.fillRect(0, 0, W, H)
  g.fillStyle = b
  for (let i = 0; i < 5; i++) { // flames
    const y = 30 + i * 18
    g.beginPath(); g.moveTo(0, y + 14); g.lineTo(W * (0.4 + r() * 0.5), y); g.lineTo(W, y + 10); g.lineTo(W, y + 18); g.lineTo(0, y + 22); g.fill()
  }
  g.fillStyle = '#16151a'; g.beginPath(); g.arc(W / 2, 18, 9, 0, 7); g.fill()
  g.fillStyle = '#c8c0aa'; g.fillRect(W / 2 - 4, 15, 3, 4); g.fillRect(W / 2 + 1, 15, 3, 4)
  speckle(g, W, H, 400, r, ['#a08a60', 'rgba(0,0,0,0.4)'], 2)   // scrapes down to the wood
  return c
})

// ---------- dumpster: dark green paint, rust, a faded scribble ----------
export const dumpsterTexture = () => tex('dumpster', () => {
  const W = 128, H = 128; const [c, g] = canvas(W, H); const r = rng(31)
  g.fillStyle = '#2f4a34'; g.fillRect(0, 0, W, H)
  g.fillStyle = '#263d2b'; for (let x = 0; x < W; x += 16) g.fillRect(x, 0, 4, H)
  for (let i = 0; i < 16; i++) splat(g, r() * W, r() * H, 3 + r() * 8, pick(r, ['#6a3a1c', '#4a2a14', '#1a2a1c']), r)
  for (let i = 0; i < 12; i++) { g.fillStyle = 'rgba(100,50,15,0.4)'; g.fillRect(r() * W, r() * H, 2, 10 + r() * 30) }
  g.strokeStyle = 'rgba(200,190,170,0.6)'; g.lineWidth = 2; g.beginPath(); g.moveTo(12, 80)
  for (let i = 0; i < 6; i++) g.quadraticCurveTo(20 + i * 16, 52 + r() * 20, 28 + i * 16, 80 - r() * 12)
  g.stroke()
  speckle(g, W, H, 1500, r, ['rgba(0,0,0,0.3)', 'rgba(140,70,30,0.3)'])
  return c
})

// ---------- roll-down gate (corrugated, grimy, an old faded tag) ----------
export const gateTexture = () => tex('gate', () => {
  const W = 128, H = 128; const [c, g] = canvas(W, H); const r = rng(77)
  for (let y = 0; y < H; y += 6) { g.fillStyle = '#6a6e74'; g.fillRect(0, y, W, 3); g.fillStyle = '#4c5056'; g.fillRect(0, y + 3, W, 3) }
  splat(g, 60, 60, 30, 'rgba(30,30,34,0.85)', r); splat(g, 60, 60, 25, 'rgba(70,110,150,0.7)', r); splat(g, 92, 80, 14, 'rgba(150,70,110,0.6)', r)
  speckle(g, W, H, 1500, r, ['rgba(0,0,0,0.3)', 'rgba(130,60,20,0.35)'])
  const grime = g.createLinearGradient(0, H * 0.5, 0, H)
  grime.addColorStop(0, 'rgba(0,0,0,0)'); grime.addColorStop(1, 'rgba(10,8,6,0.6)')
  g.fillStyle = grime; g.fillRect(0, 0, W, H)
  return c
})

// ---------- far skyline: dark towers, sparse lit windows ----------
export const windowsTexture = () => tex('windows', () => {
  const W = 32, H = 64; const [c, g] = canvas(W, H); const r = rng(13)
  g.fillStyle = '#0c0e14'; g.fillRect(0, 0, W, H)
  for (let y = 2; y < H - 2; y += 5) for (let x = 2; x < W - 2; x += 5) {
    const lit = r()
    g.fillStyle = lit > 0.9 ? '#d8a058' : lit > 0.86 ? '#8ea4c4' : '#11141b'
    g.fillRect(x, y, 3, 3)
  }
  return c
})

// ---------- radial gradient: blob shadows + fake light pools ----------
export const blobTexture = () => tex('blob', () => {
  const S = 64; const [c, g] = canvas(S, S)
  const grd = g.createRadialGradient(S / 2, S / 2, 0, S / 2, S / 2, S / 2)
  grd.addColorStop(0, 'rgba(255,255,255,1)'); grd.addColorStop(0.5, 'rgba(255,255,255,0.7)'); grd.addColorStop(1, 'rgba(255,255,255,0)')
  g.fillStyle = grd; g.fillRect(0, 0, S, S)
  return c
})

// ---------- bodega (fluorescent interior, faded sign, dirty awning) ----------
export const bodegaWindowTexture = () => tex('bodegaWin', () => {
  const W = 128, H = 64; const [c, g] = canvas(W, H); const r = rng(55)
  g.fillStyle = '#e4e2c8'; g.fillRect(0, 0, W, H)            // fluorescent interior
  for (let sy = 10; sy < H; sy += 16) {
    g.fillStyle = '#9a9a92'; g.fillRect(0, sy + 11, W, 2)      // shelf
    for (let x = 2; x < W - 4;) {                              // abstract product boxes/cans
      const w = 3 + Math.floor(r() * 5), h = 5 + Math.floor(r() * 6)
      g.fillStyle = pick(r, [...GRIME, '#c84a3a', '#d8c070'])
      g.fillRect(x, sy + 11 - h, w, h)
      x += w + 1
    }
  }
  for (let i = 0; i < 3; i++) sticker(g, 16 + i * 44, 12, 14, r)                               // window stickers
  speckle(g, W, H, 300, r, ['rgba(60,50,30,0.25)'], 2)                                           // dirty glass
  g.fillStyle = 'rgba(40,40,44,0.95)'; g.fillRect(0, 0, W, 2); g.fillRect(W / 2 - 1, 0, 2, H)   // mullions
  return c
})

export const bodegaSignTexture = () => tex('bodegaSign', () => {
  const W = 256, H = 48; const [c, g] = canvas(W, H); const r = rng(56)
  g.fillStyle = '#8a2a26'; g.fillRect(0, 0, W, H)
  g.fillStyle = '#c8b070'; g.fillRect(0, 0, W, 3); g.fillRect(0, H - 3, W, 3)
  g.font = '900 26px Impact, "Arial Black", sans-serif'; g.textAlign = 'center'; g.textBaseline = 'middle'
  g.fillStyle = '#1a1214'; g.fillText('DELI  GROCERY', W / 2 + 2, H / 2 + 2)
  g.fillStyle = '#ece4d0'; g.fillText('DELI  GROCERY', W / 2, H / 2)
  g.fillStyle = '#c8b070'; g.beginPath(); g.arc(18, H / 2, 10, 0, 7); g.fill(); g.beginPath(); g.arc(W - 18, H / 2, 10, 0, 7); g.fill()
  g.fillStyle = '#8a2a26'; g.font = 'bold 10px monospace'; g.fillText('24', 18, H / 2 + 1); g.fillText('24', W - 18, H / 2 + 1)
  speckle(g, W, H, 600, r, ['rgba(0,0,0,0.3)', 'rgba(255,255,255,0.08)'], 2)
  return c
})

export const awningTexture = () => tex('awning', () => {
  const W = 32, H = 8; const [c, g] = canvas(W, H); const r = rng(57)
  for (let x = 0; x < W; x += 8) { g.fillStyle = '#c8c0a8'; g.fillRect(x, 0, 4, H); g.fillStyle = '#3a5a44'; g.fillRect(x + 4, 0, 4, H) }
  speckle(g, W, H, 40, r, ['rgba(0,0,0,0.35)'], 1)
  return c
})

// ---------- boombox face ----------
export const boomboxTexture = () => tex('boombox', () => {
  const W = 64, H = 32; const [c, g] = canvas(W, H); const r = rng(58)
  g.fillStyle = '#2c2d32'; g.fillRect(0, 0, W, H)
  g.fillStyle = '#121216'; g.fillRect(24, 6, 16, 9)            // tape deck
  g.fillStyle = '#8a8c90'; g.fillRect(26, 8, 12, 2)
  g.fillStyle = '#b8a070'; for (let i = 0; i < 5; i++) g.fillRect(25 + i * 3, 18, 2, 2)
  g.fillStyle = '#9a9ca0'; g.fillRect(24, 24, 16, 2)
  sticker(g, 8, 6, 7, r)
  speckle(g, W, H, 80, r, ['rgba(255,255,255,0.08)', 'rgba(0,0,0,0.3)'], 1)
  return c
})
