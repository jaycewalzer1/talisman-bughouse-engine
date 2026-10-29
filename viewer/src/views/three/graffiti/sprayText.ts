// Small graffiti helpers. The wall art itself is painted by graffiti/wallArt.ts (letters from
// glyphs.ts, rendered by painter.ts). Messages are no longer painted on the wall (comic speech
// bubbles replaced them); tokenToWall/isWallWorthy stay for the comms log, drawAbstractPiece for
// the train's side panel.
import { rng } from '../environment/textures'
import type { MessageEvent } from '../../../events/types'

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


/**
 * An abstract (letterless) piece: a fat outlined bubble blob with a 3D drop, star bursts, an arrow
 * and drips. Hot colours on a dark outline, transparent background. 100% original shapes.
 */
export function drawAbstractPiece(colors: string[], seed: number, w = 256, h = 128): HTMLCanvasElement {
  const c = document.createElement('canvas'); c.width = w; c.height = h
  const g = c.getContext('2d')!
  const r = rng(seed)
  const blob = (cx: number, cy: number, rx: number, ry: number) => {
    g.beginPath()
    const n = 9
    for (let i = 0; i <= n; i++) {
      const a = (i / n) * Math.PI * 2, k = 0.78 + r() * 0.35
      const x = cx + Math.cos(a) * rx * k, y = cy + Math.sin(a) * ry * k
      if (i === 0) g.moveTo(x, y); else g.quadraticCurveTo(cx + Math.cos(a - 0.35) * rx * 1.15, cy + Math.sin(a - 0.35) * ry * 1.15, x, y)
    }
    g.closePath()
  }
  const [a, b, d] = [colors[0], colors[1] ?? '#ffffff', colors[2] ?? colors[0]]
  // 3D drop shadow, outline, fill, inner highlight
  const lobes = 2 + Math.floor(r() * 2)
  const pts = Array.from({ length: lobes }, (_, i) => ({ x: w * (0.28 + i * (0.44 / Math.max(1, lobes - 1))), y: h * (0.48 + (r() - 0.5) * 0.15), rx: w * 0.17, ry: h * 0.32 }))
  g.fillStyle = '#120a1e'
  pts.forEach(p => { blob(p.x + 6, p.y + 6, p.rx + 6, p.ry + 6); g.fill() })
  g.fillStyle = '#ffffff'
  pts.forEach(p => { blob(p.x, p.y, p.rx + 4, p.ry + 4); g.fill() })
  pts.forEach((p, i) => {
    const grd = g.createLinearGradient(0, p.y - p.ry, 0, p.y + p.ry)
    grd.addColorStop(0, i % 2 ? b : a); grd.addColorStop(0.5, i % 2 ? b : a); grd.addColorStop(0.51, d); grd.addColorStop(1, d)
    g.fillStyle = grd; blob(p.x, p.y, p.rx, p.ry); g.fill()
  })
  // eyes / highlight dots on the blobs
  g.fillStyle = '#ffffff'
  pts.forEach(p => { g.fillRect(p.x - p.rx * 0.45, p.y - p.ry * 0.55, 8, 5) })
  // star bursts
  for (let i = 0; i < 3; i++) {
    const x = r() * w, y = r() * h * 0.5, s = 8 + r() * 10
    g.fillStyle = i === 0 ? '#ffffff' : b
    g.beginPath()
    for (let k = 0; k < 8; k++) { const ang = k * Math.PI / 4, rr = k % 2 ? s * 0.3 : s; g.lineTo(x + Math.cos(ang) * rr, y + Math.sin(ang) * rr) }
    g.fill()
  }
  // swoosh arrow
  g.strokeStyle = '#120a1e'; g.lineWidth = 9; g.lineCap = 'round'
  const y0 = h * 0.86
  const arrow = () => { g.beginPath(); g.moveTo(w * 0.1, y0); g.quadraticCurveTo(w * 0.5, y0 - 26, w * 0.9, y0 - 6); g.stroke()
    g.beginPath(); g.moveTo(w * 0.9, y0 - 6); g.lineTo(w * 0.82, y0 - 20); g.moveTo(w * 0.9, y0 - 6); g.lineTo(w * 0.81, y0 + 4); g.stroke() }
  arrow(); g.strokeStyle = b; g.lineWidth = 4; arrow()
  // drips
  g.fillStyle = d
  for (let i = 0; i < 5; i++) { const x = w * (0.2 + r() * 0.6), y = h * 0.7; g.fillRect(x, y, 3, 6 + r() * 18) }
  return c
}
