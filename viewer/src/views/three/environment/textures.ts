// Procedural canvas textures (no downloads). Each one is generated once and cached.
// To use real CC0 photos later (Poly Haven / ambientCG), replace a function body with
// `new TextureLoader().load('/textures/....jpg')` and keep the same wrap/repeat settings.
import { CanvasTexture, RepeatWrapping, SRGBColorSpace, NearestFilter, type Texture } from 'three'

const cache = new Map<string, Texture>()

/** Seeded random so textures look the same on every load. */
export function rng(seed: number) {
  let s = seed >>> 0
  return () => { s = (s * 1664525 + 1013904223) >>> 0; return s / 4294967296 }
}

function canvas(w: number, h: number): [HTMLCanvasElement, CanvasRenderingContext2D] {
  const c = document.createElement('canvas'); c.width = w; c.height = h
  return [c, c.getContext('2d')!]
}

function tex(key: string, make: () => HTMLCanvasElement, opts: { color?: boolean; repeat?: [number, number]; nearest?: boolean } = {}): Texture {
  const hit = cache.get(key); if (hit) return hit
  const t = new CanvasTexture(make())
  if (opts.color !== false) t.colorSpace = SRGBColorSpace
  t.wrapS = t.wrapT = RepeatWrapping
  if (opts.repeat) t.repeat.set(...opts.repeat)
  if (opts.nearest) { t.magFilter = NearestFilter }
  t.anisotropy = 4
  cache.set(key, t)
  return t
}

function speckle(g: CanvasRenderingContext2D, w: number, h: number, n: number, r: () => number, colors: string[], size = 2) {
  for (let i = 0; i < n; i++) {
    g.fillStyle = colors[Math.floor(r() * colors.length)]
    g.fillRect(r() * w, r() * h, size * r() + 0.5, size * r() + 0.5)
  }
}

// ---------- brick ----------
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
    if (!bump) speckle(g, bw - 6, bh - 6, 0, r, [])
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

// ---------- wet asphalt + puddles ----------
function puddles(g: CanvasRenderingContext2D, W: number, H: number, r: () => number, fill: string) {
  g.fillStyle = fill
  for (let i = 0; i < 9; i++) {
    const cx = r() * W, cy = r() * H, rad = 30 + r() * 110
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

// ---------- plywood ----------
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

// ---------- rusty painted steel (el columns, girders) ----------
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

// ---------- chain-link fence (alpha-tested) ----------
export const chainLinkTexture = () => tex('chain', () => {
  const W = 64, H = 64; const [c, g] = canvas(W, H)
  g.clearRect(0, 0, W, H)
  g.strokeStyle = '#b8bcc0'; g.lineWidth = 4
  g.beginPath(); g.moveTo(0, H / 2); g.lineTo(W / 2, 0); g.lineTo(W, H / 2); g.lineTo(W / 2, H); g.closePath(); g.stroke()
  return c
}, { repeat: [40, 12] })

// ---------- milk crate (grid of slots, alpha-tested) ----------
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

// ---------- roll-up vinyl chess board, worn ----------
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

// ---------- grip tape / deck graphic ----------
export const gripTexture = () => tex('grip', () => {
  const W = 128, H = 128; const [c, g] = canvas(W, H); const r = rng(8)
  g.fillStyle = '#151515'; g.fillRect(0, 0, W, H)
  speckle(g, W, H, 2500, r, ['#2a2a2a', '#080808', '#333'], 2)
  return c
}, { repeat: [1, 3] })
