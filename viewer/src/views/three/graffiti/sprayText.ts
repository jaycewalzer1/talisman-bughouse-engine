// Draws original graffiti (made-up words, generic letter styles) onto a canvas.
// Used for the static wall pieces and for the live message tags.
import { rng } from '../environment/textures'
import type { MessageEvent } from '../../../events/types'

export type GraffitiStyle = 'throwup' | 'tag'

const FONT = `'Arial Black', 'Impact', 'Helvetica Neue', sans-serif`

function shade(hex: string, k: number): string {
  const n = parseInt(hex.slice(1), 16)
  const f = (v: number) => Math.max(0, Math.min(255, Math.round(v * k)))
  return `rgb(${f(n >> 16)},${f((n >> 8) & 255)},${f(n & 255)})`
}

/**
 * Paint `text` into a new canvas. Returns the canvas; letters are jittered and rotated so it does
 * not read as a plain font, then outlined, filled, highlighted and given paint drips.
 */
export function drawGraffiti(text: string, color: string, seed: number, style: GraffitiStyle = 'throwup', w = 512, h = 256): HTMLCanvasElement {
  const c = document.createElement('canvas'); c.width = w; c.height = h
  const g = c.getContext('2d')!
  const r = rng(seed)
  const letters = [...text]
  const size = Math.min(h * 0.62, (w * 0.9) / Math.max(1, letters.length * 0.72))
  g.font = `900 ${style === 'tag' ? 'italic ' : ''}${size}px ${FONT}`
  g.textBaseline = 'middle'
  // measure + place letters with jitter
  const widths = letters.map(ch => g.measureText(ch).width * (style === 'throwup' ? 0.86 : 0.8))
  const total = widths.reduce((a, b) => a + b, 0)
  let x = (w - total) / 2
  const placed = letters.map((ch, i) => {
    const p = { ch, x: x + widths[i] / 2, y: h * 0.47 + (r() - 0.5) * size * 0.18, rot: (r() - 0.5) * 0.35, s: 0.9 + r() * 0.25 }
    x += widths[i]
    return p
  })
  const each = (fn: (p: typeof placed[number]) => void) => placed.forEach(p => {
    g.save(); g.translate(p.x, p.y); g.rotate(p.rot); g.scale(p.s, p.s); fn(p); g.restore()
  })
  g.lineJoin = 'round'; g.textAlign = 'center'

  if (style === 'throwup') {
    // overspray glow
    g.shadowColor = color; g.shadowBlur = 18
    g.strokeStyle = '#0c0c0e'; g.lineWidth = size * 0.3
    each(p => g.strokeText(p.ch, 0, 0))
    g.shadowBlur = 0
    g.strokeStyle = '#f4f1e8'; g.lineWidth = size * 0.16
    each(p => g.strokeText(p.ch, 0, 0))
  } else {
    g.strokeStyle = '#0c0c0e'; g.lineWidth = size * 0.1
    each(p => g.strokeText(p.ch, 0, 0))
  }
  const grd = g.createLinearGradient(0, h * 0.2, 0, h * 0.8)
  grd.addColorStop(0, shade(color, 1.25)); grd.addColorStop(0.55, color); grd.addColorStop(1, shade(color, 0.6))
  g.fillStyle = grd
  each(p => g.fillText(p.ch, 0, 0))
  // highlights / sparkles
  g.fillStyle = 'rgba(255,255,255,0.85)'
  each(() => { g.fillRect(-size * 0.18, -size * 0.25, size * 0.07, size * 0.07) })
  // drips below the letters
  g.strokeStyle = color; g.lineCap = 'round'
  const drips = 3 + Math.floor(r() * 5)
  for (let i = 0; i < drips; i++) {
    const p = placed[Math.floor(r() * placed.length)]
    const dx = p.x + (r() - 0.5) * size * 0.4, y0 = p.y + size * 0.3, len = size * (0.2 + r() * 0.6)
    g.lineWidth = 3 + r() * 4
    g.beginPath(); g.moveTo(dx, y0); g.lineTo(dx, y0 + len); g.stroke()
    g.fillStyle = color; g.beginPath(); g.arc(dx, y0 + len, g.lineWidth * 0.8, 0, 7); g.fill()
  }
  if (style === 'tag') { // swoosh underline with arrow
    g.strokeStyle = color; g.lineWidth = 6
    const y = h * 0.82
    g.beginPath(); g.moveTo(w * 0.12, y); g.quadraticCurveTo(w * 0.5, y + 22, w * 0.86, y - 10); g.stroke()
    g.beginPath(); g.moveTo(w * 0.86, y - 10); g.lineTo(w * 0.8, y - 26); g.moveTo(w * 0.86, y - 10); g.lineTo(w * 0.79, y + 4); g.stroke()
  }
  // speckle overspray
  g.fillStyle = color
  for (let i = 0; i < 160; i++) { g.globalAlpha = 0.3 * r(); g.fillRect(r() * w, r() * h, 2, 2) }
  g.globalAlpha = 1
  return c
}

/** Short wall text for a message token, e.g. need_N -> "NEED N". */
export function tokenToWall(token: string): string {
  const [head, piece] = token.split('_')
  switch (head) {
    case 'need': return `NEED ${piece ?? ''}!`
    case 'have': return `GOT ${piece ?? ''}`
    case 'stall': return 'SITTIN'
    case 'go': return 'GO GO'
    case 'sit': return 'SIT!'
    case 'ok': return 'OK'
    case 'no': return 'NAH'
    default: return token.toUpperCase().slice(0, 8)
  }
}

/** Status chatter ("ok" with payload.kind = "status") is too frequent to paint; real comms only. */
export function isWallWorthy(e: MessageEvent): boolean {
  if (e.token === 'ok' && (e.payload as { kind?: unknown } | undefined)?.kind === 'status') return false
  return true
}

