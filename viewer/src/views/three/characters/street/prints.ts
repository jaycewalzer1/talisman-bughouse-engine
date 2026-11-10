// Original T-shirt / hoodie prints, painted on a canvas and projected onto the chest by texture.ts.
// Chess-and-street themed; no real brands or artists.
import type { Outfit } from './outfits'

const S = 256

function rng(seed: number) { let s = seed >>> 0; return () => { s = (s * 1664525 + 1013904223) >>> 0; return s / 4294967296 } }

/** A rook on a rise, with bare branches sprawling up out of it across the whole chest. */
function rookTree(g: CanvasRenderingContext2D, o: Outfit) {
  const r = rng(77)
  g.strokeStyle = o.printInk; g.fillStyle = o.printInk; g.lineCap = 'round'
  const branch = (x: number, y: number, a: number, len: number, w: number, depth: number) => {
    const ex = x + Math.cos(a) * len, ey = y + Math.sin(a) * len
    g.lineWidth = w; g.beginPath(); g.moveTo(x, y)
    g.quadraticCurveTo(x + Math.cos(a + 0.3) * len * 0.5, y + Math.sin(a + 0.3) * len * 0.5, ex, ey); g.stroke()
    if (depth <= 0) return
    const n = depth > 4 ? 2 : 2 + Math.floor(r() * 2)
    for (let i = 0; i < n; i++) branch(ex, ey, a + (r() - 0.5) * 1.5, len * (0.6 + r() * 0.2), w * 0.66, depth - 1)
  }
  branch(128, 150, -Math.PI / 2 - 0.2, 58, 14, 7)
  branch(128, 150, -Math.PI / 2 + 0.6, 56, 11, 7)
  branch(128, 158, -Math.PI / 2 - 0.95, 50, 9, 6)
  // the rook
  const rx = 128, ry = 164
  g.fillRect(rx - 26, ry, 52, 10); g.fillRect(rx - 17, ry + 10, 34, 40); g.fillRect(rx - 26, ry + 50, 52, 10); g.fillRect(rx - 32, ry + 60, 64, 12)
  for (const dx of [-26, -8, 10]) g.fillRect(rx + dx, ry - 12, 16, 13)
}

/** A half sun over chunky made-up lettering, the way every mid-2000s skate tee did it. */
function hausSun(g: CanvasRenderingContext2D, o: Outfit) {
  g.fillStyle = o.printAccent
  g.beginPath(); g.arc(128, 118, 56, Math.PI, 0); g.closePath(); g.fill()
  for (let i = 0; i <= 8; i++) {
    const a = Math.PI + (i / 8) * Math.PI
    g.beginPath(); g.moveTo(128 + Math.cos(a - 0.08) * 64, 118 + Math.sin(a - 0.08) * 64); g.lineTo(128 + Math.cos(a + 0.08) * 64, 118 + Math.sin(a + 0.08) * 64)
    g.lineTo(128 + Math.cos(a) * 104, 118 + Math.sin(a) * 104); g.fill()
  }
  g.fillStyle = o.printInk; g.fillRect(20, 118, 216, 8)
  g.textAlign = 'center'; g.textBaseline = 'alphabetic'
  g.font = 'italic 900 58px "Arial Black", Impact, sans-serif'
  g.lineWidth = 8; g.strokeStyle = o.printInk; g.fillStyle = o.printAccent
  g.strokeText('BUG', 128, 178); g.fillText('BUG', 128, 178)
  g.strokeText('HAUS', 128, 232); g.fillText('HAUS', 128, 232)
}

/** A jagged starburst with a crown knocked out of the middle. */
function crownBurst(g: CanvasRenderingContext2D, o: Outfit) {
  g.fillStyle = o.printInk
  g.beginPath()
  for (let i = 0; i < 28; i++) {
    const a = (i / 28) * Math.PI * 2, R = i % 2 ? 66 : 108 + (i % 4) * 6
    const x = 128 + Math.cos(a) * R, y = 128 + Math.sin(a) * R * 0.86
    if (i) g.lineTo(x, y); else g.moveTo(x, y)
  }
  g.closePath(); g.fill()
  g.fillStyle = o.printAccent
  g.beginPath(); g.moveTo(72, 158); g.lineTo(62, 92); g.lineTo(98, 124); g.lineTo(128, 76); g.lineTo(158, 124); g.lineTo(194, 92); g.lineTo(184, 158); g.closePath(); g.fill()
  g.fillRect(72, 164, 112, 14)
}

/** A skull whose cranium is a pawn's head, over crossed bars. */
function pawnSkull(g: CanvasRenderingContext2D, o: Outfit) {
  g.strokeStyle = o.printAccent; g.lineWidth = 16; g.lineCap = 'butt'
  g.beginPath(); g.moveTo(48, 212); g.lineTo(208, 150); g.moveTo(208, 212); g.lineTo(48, 150); g.stroke()
  g.fillStyle = o.printInk
  g.beginPath(); g.arc(128, 96, 64, 0, 7); g.fill()
  g.fillRect(94, 146, 68, 46)
  g.fillStyle = '#10141c'
  g.beginPath(); g.ellipse(102, 102, 18, 21, 0.1, 0, 7); g.fill()
  g.beginPath(); g.ellipse(154, 102, 18, 21, -0.1, 0, 7); g.fill()
  g.beginPath(); g.moveTo(128, 122); g.lineTo(118, 144); g.lineTo(138, 144); g.closePath(); g.fill()
  for (let i = 0; i < 4; i++) g.fillRect(101 + i * 15, 164, 5, 28)
}

export function paintPrint(o: Outfit): ImageData {
  const c = document.createElement('canvas'); c.width = c.height = S
  const g = c.getContext('2d', { willReadFrequently: true })!
  switch (o.print) {
    case 'rookTree': rookTree(g, o); break
    case 'hausSun': hausSun(g, o); break
    case 'crownBurst': crownBurst(g, o); break
    case 'pawnSkull': pawnSkull(g, o); break
  }
  // screen-print wear: speckled holes
  const r = rng(13)
  g.globalCompositeOperation = 'destination-out'
  for (let i = 0; i < 700; i++) { g.fillStyle = `rgba(0,0,0,${0.3 + r() * 0.6})`; g.fillRect(r() * S, r() * S, 1 + r() * 2, 1 + r() * 2) }
  return g.getImageData(0, 0, S, S)
}
