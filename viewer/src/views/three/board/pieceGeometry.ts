// Chunky low-poly chess pieces built from lathe profiles + boxes (8 sides = visibly faceted).
import { BoxGeometry, BufferGeometry, IcosahedronGeometry, LatheGeometry, Matrix4, Vector2 } from 'three'
import { mergeGeometries } from 'three/examples/jsm/utils/BufferGeometryUtils.js'

const SIDES = 8
const lathe = (pts: [number, number][]) => new LatheGeometry(pts.map(([r, y]) => new Vector2(r, y)), SIDES)
const box = (w: number, h: number, d: number, x: number, y: number, z: number, rx = 0) =>
  new BoxGeometry(w, h, d).applyMatrix4(new Matrix4().makeRotationX(rx).setPosition(x, y, z))
const ball = (r: number, y: number) => new IcosahedronGeometry(r, 0).applyMatrix4(new Matrix4().makeTranslation(0, y, 0))

function merge(parts: BufferGeometry[]): BufferGeometry {
  const g = mergeGeometries(parts.map(p => {
    const n = p.index ? p.toNonIndexed() : p
    // keep only the attributes every part has
    for (const k of Object.keys(n.attributes)) if (!['position', 'normal', 'uv'].includes(k)) n.deleteAttribute(k)
    return n
  }))!
  g.computeVertexNormals()
  g.computeBoundingSphere()
  return g
}

const BASE: [number, number][] = [[0, 0], [0.034, 0], [0.034, 0.012], [0.027, 0.018]]

function build(type: string): BufferGeometry {
  switch (type) {
    case 'p': return merge([lathe([...BASE, [0.014, 0.03], [0.012, 0.042], [0, 0.042]]), ball(0.016, 0.052)])
    case 'r': return merge([lathe([...BASE, [0.022, 0.03], [0.022, 0.055], [0.028, 0.058], [0.028, 0.072], [0, 0.072]]),
      ...[0, 1, 2, 3].map(i => box(0.012, 0.012, 0.012, Math.cos(i * Math.PI / 2) * 0.02, 0.078, Math.sin(i * Math.PI / 2) * 0.02))])
    case 'n': return merge([lathe([...BASE, [0.022, 0.03], [0, 0.03]]),
      box(0.026, 0.055, 0.032, 0, 0.055, -0.004, -0.25), box(0.022, 0.022, 0.04, 0, 0.074, 0.018, 0.35), box(0.008, 0.016, 0.01, 0, 0.09, -0.012)])
    case 'b': return merge([lathe([...BASE, [0.014, 0.035], [0.019, 0.055], [0.014, 0.075], [0.004, 0.085], [0, 0.085]]), ball(0.006, 0.09)])
    case 'q': return merge([lathe([...BASE, [0.015, 0.04], [0.016, 0.07], [0.026, 0.085], [0.02, 0.092], [0, 0.092]]), ball(0.01, 0.1)])
    case 'k': return merge([lathe([...BASE, [0.016, 0.045], [0.017, 0.075], [0.025, 0.088], [0.022, 0.097], [0, 0.097]]),
      box(0.008, 0.03, 0.008, 0, 0.11, 0), box(0.024, 0.008, 0.008, 0, 0.113, 0)])
  }
  return build('p')
}

const cache: Record<string, BufferGeometry> = {}
/** Geometry for a piece letter (case ignored). */
export function pieceGeometry(letter: string): BufferGeometry {
  const t = letter.toLowerCase()
  return (cache[t] ??= build(t))
}
export const PIECE_TYPES = ['p', 'n', 'b', 'r', 'q', 'k'] as const
