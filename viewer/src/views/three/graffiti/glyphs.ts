// Skeleton letter glyphs for the wall painter (graffiti/painter.ts).
//
// Each glyph is the CENTRE LINE of a letter in a unit box (x 0..w, y 0..1, y down), written as a
// tiny SVG-path subset (absolute M / L / Q / C / Z). The painter strokes these centre lines at
// different widths, joins and warps to get the three hand styles seen on real walls (see
// CREDITS.md for the photos the letter shapes were studied from):
//   bubble throw-ups — fat round strokes, letters squashed into each other, small counters;
//   semi-wild pieces — mitred strokes of varying width, lean, per-point warp, arrows on the ends;
//   handstyle tags   — thin round strokes, hard lean, connected, jittered by hand.
// Proportions were taken from tracing the letter skeletons in those photos (wide, low bowls on
// B/P/R, a dropped middle bar on E/H/A, a short-legged K, an S whose spine crosses steeply).

export interface Glyph { w: number; d: string }

export const GLYPHS: Record<string, Glyph> = {
  A: { w: 0.9, d: 'M0 1 L.45 0 L.9 1 M.2 .64 L.7 .64' },
  B: { w: 0.86, d: 'M0 1 L0 0 L.42 0 Q.78 0 .78 .24 Q.78 .47 .42 .47 L0 .47 M.42 .47 Q.86 .47 .86 .73 Q.86 1 .42 1 L0 1' },
  C: { w: 0.82, d: 'M.82 .17 Q.68 0 .45 0 Q0 0 0 .5 Q0 1 .45 1 Q.68 1 .82 .83' },
  D: { w: 0.88, d: 'M0 0 L0 1 L.38 1 Q.88 1 .88 .5 Q.88 0 .38 0 Z' },
  E: { w: 0.76, d: 'M.76 0 L0 0 L0 1 L.76 1 M0 .52 L.58 .52' },
  F: { w: 0.74, d: 'M.74 0 L0 0 L0 1 M0 .52 L.58 .52' },
  G: { w: 0.9, d: 'M.84 .16 Q.7 0 .46 0 Q0 0 0 .5 Q0 1 .46 1 Q.9 1 .9 .56 L.52 .56' },
  H: { w: 0.84, d: 'M0 0 L0 1 M.84 0 L.84 1 M0 .54 L.84 .54' },
  I: { w: 0.24, d: 'M.12 0 L.12 1' },
  J: { w: 0.7, d: 'M.7 0 L.7 .64 Q.7 1 .35 1 Q0 1 0 .7' },
  K: { w: 0.82, d: 'M0 0 L0 1 M.78 0 L0 .58 M.26 .4 L.82 1' },
  L: { w: 0.7, d: 'M0 0 L0 1 L.7 1' },
  M: { w: 1.02, d: 'M0 1 L0 0 L.51 .58 L1.02 0 L1.02 1' },
  N: { w: 0.84, d: 'M0 1 L0 0 L.84 1 L.84 0' },
  O: { w: 0.9, d: 'M.45 0 Q0 0 0 .5 Q0 1 .45 1 Q.9 1 .9 .5 Q.9 0 .45 0 Z' },
  P: { w: 0.82, d: 'M0 1 L0 0 L.42 0 Q.82 0 .82 .28 Q.82 .56 .42 .56 L0 .56' },
  Q: { w: 0.94, d: 'M.45 0 Q0 0 0 .5 Q0 1 .45 1 Q.9 1 .9 .5 Q.9 0 .45 0 Z M.56 .7 L.94 1.04' },
  R: { w: 0.86, d: 'M0 1 L0 0 L.42 0 Q.82 0 .82 .27 Q.82 .55 .42 .55 L0 .55 M.38 .55 L.86 1' },
  S: { w: 0.82, d: 'M.78 .13 Q.64 0 .42 0 Q.04 0 .04 .26 Q.04 .48 .41 .5 Q.8 .52 .8 .75 Q.8 1 .4 1 Q.14 1 0 .86' },
  T: { w: 0.84, d: 'M0 0 L.84 0 M.42 0 L.42 1' },
  U: { w: 0.84, d: 'M0 0 L0 .6 Q0 1 .42 1 Q.84 1 .84 .6 L.84 0' },
  V: { w: 0.86, d: 'M0 0 L.43 1 L.86 0' },
  W: { w: 1.06, d: 'M0 0 L.26 1 L.53 .38 L.8 1 L1.06 0' },
  X: { w: 0.84, d: 'M0 0 L.84 1 M.84 0 L0 1' },
  Y: { w: 0.84, d: 'M0 0 L.42 .52 L.84 0 M.42 .52 L.42 1' },
  Z: { w: 0.82, d: 'M0 0 L.82 0 L0 1 L.82 1' },
  0: { w: 0.7, d: 'M.35 0 Q0 0 0 .5 Q0 1 .35 1 Q.7 1 .7 .5 Q.7 0 .35 0 Z' },
  1: { w: 0.4, d: 'M0 .2 L.3 0 L.3 1' },
  2: { w: 0.74, d: 'M.02 .2 Q.12 0 .38 0 Q.72 0 .72 .28 Q.72 .5 .4 .7 L0 1 L.74 1' },
  9: { w: 0.7, d: 'M.68 .3 Q.68 0 .36 0 Q.04 0 .04 .3 Q.04 .58 .36 .58 Q.68 .58 .68 .3 L.6 1' },
  '!': { w: 0.24, d: 'M.12 0 L.12 .66 M.12 .92 L.12 .96' },
  '*': { w: 0.6, d: 'M.3 .1 L.3 .7 M0 .25 L.6 .55 M.6 .25 L0 .55' },
}

export type Cmd = { c: 'M' | 'L' | 'Q' | 'C' | 'Z'; p: number[] }

const cache = new Map<string, Cmd[][]>()

/** Parse a glyph's mini-path into sub-paths (one per M). */
export function glyphPaths(ch: string): Cmd[][] {
  const hit = cache.get(ch)
  if (hit) return hit
  const g = GLYPHS[ch]
  const out: Cmd[][] = []
  if (g) {
    const toks = g.d.match(/[MLQCZ]|-?\d*\.?\d+/g) ?? []
    let cur: Cmd[] | null = null
    let i = 0
    while (i < toks.length) {
      const c = toks[i++] as Cmd['c']
      const n = c === 'M' || c === 'L' ? 2 : c === 'Q' ? 4 : c === 'C' ? 6 : 0
      const p = toks.slice(i, i + n).map(Number); i += n
      if (c === 'M') { cur = []; out.push(cur) }
      cur?.push({ c, p })
    }
  }
  cache.set(ch, out)
  return out
}

export function glyphWidth(ch: string): number {
  return ch === ' ' ? 0.45 : GLYPHS[ch]?.w ?? 0.6
}
