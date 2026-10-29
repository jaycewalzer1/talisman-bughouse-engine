// Camera shot definitions + a geometric check that no player stands between the camera and a
// board. Players stand on each table's z axis at ±0.84 (layout.seatPose), so board shots come in
// from the sides, from 3/4 high angles, or from high above (y >= 3.4). Pure (no React).
import { Vector3 } from 'three'
import { type BoardId, type Seat, SEATS } from '../../../events/types'
import { BOARD_X, BOARD_Y, boardToWorld, seatPose } from '../layout'

export interface Shot { from: [number, number, number]; to: [number, number, number]; drift: [number, number, number]; fov: number }

export function boardShots(b: BoardId): Shot[] {
  const x = BOARD_X[b], out = b === 'A' ? -1 : 1
  return [
    { from: [x + 1.9 * out, 2.3, 1.3], to: [x, 0.72, 0], drift: [0.2 * out, 0, -0.5], fov: 40 },        // 3/4 high, near side
    { from: [x + 2.4 * out, 1.35, 0.15], to: [x, 0.8, 0], drift: [0, 0.1, -0.4], fov: 40 },              // side, low
    { from: [x + 1.9 * out, 2.4, -1.3], to: [x, 0.72, 0], drift: [0.2 * out, -0.1, 0.4], fov: 42 },      // 3/4 high, far side
    { from: [x + 0.55 * out, 3.7, 0.15], to: [x, 0.7, 0], drift: [0.15 * out, -0.2, -0.1], fov: 34 },     // top-down
  ]
}

/** Wide shots of both boards: high front, high back, long side. */
export const WIDE: Shot[] = [
  { from: [0, 7.0, 0.8], to: [0, 0.7, 0], drift: [0.6, -0.4, 0], fov: 50 },              // crane top-down over both
  { from: [6.2, 2.4, 0.6], to: [0, 0.8, 0], drift: [0, 0.3, -1.0], fov: 46 },            // long side, B end
  { from: [-6.2, 2.6, -0.6], to: [0, 0.8, 0], drift: [0, 0.2, 1.0], fov: 46 },           // long side, A end
  { from: [0.8, 6.8, -1.4], to: [0, 0.7, 0], drift: [-0.6, -0.3, 0.3], fov: 50 },        // high, over the wall side
]

// Idle orbit: a crane move around both boards. For each angle we precompute the lowest elevation
// at which no player blocks either board (checked with blockedBy), then smooth it, so the camera
// sweeps low along the sides and rises over the players' backs.
const ORBIT_N = 96, ORBIT_DIST = 5.2, ELEV_MIN = 0.45, ELEV_MAX = 1.4
let elevTable: Float32Array | null = null
const rawPose = (angle: number, elev: number, out: Vector3) => {
  const h = Math.cos(elev) * ORBIT_DIST
  return out.set(Math.sin(angle) * h * 1.15, 0.7 + Math.sin(elev) * ORBIT_DIST, Math.cos(angle) * h)
}
function buildElevTable(): Float32Array {
  const raw = new Float32Array(ORBIT_N), v = new Vector3()
  for (let i = 0; i < ORBIT_N; i++) {
    const a = (i / ORBIT_N) * Math.PI * 2
    let e = ELEV_MIN
    while (e < ELEV_MAX && (blockedBy(rawPose(a, e, v), 'A', 0.22).length || blockedBy(rawPose(a, e, v), 'B', 0.22).length)) e += 0.05
    raw[i] = e
  }
  // widen peaks (max over neighbours) then smooth, so the crane eases up before a blocked zone
  const out = new Float32Array(ORBIT_N)
  for (let i = 0; i < ORBIT_N; i++) {
    let m = 0
    for (let k = -4; k <= 4; k++) m = Math.max(m, raw[(i + k + ORBIT_N) % ORBIT_N])
    out[i] = m
  }
  const sm = new Float32Array(ORBIT_N)
  for (let i = 0; i < ORBIT_N; i++) {
    let s = 0
    for (let k = -3; k <= 3; k++) s += out[(i + k + ORBIT_N) % ORBIT_N]
    sm[i] = Math.max(raw[i], s / 7)
  }
  return sm
}
export function orbitElevation(angle: number): number {
  elevTable ??= buildElevTable()
  const f = (((angle / (Math.PI * 2)) % 1 + 1) % 1) * ORBIT_N
  const i = Math.floor(f), u = f - i
  // never dip below either neighbour's clear elevation
  const a = elevTable[i % ORBIT_N], b = elevTable[(i + 1) % ORBIT_N]
  return Math.max(a + (b - a) * u, Math.min(a, b))
}
/** Idle-orbit camera position for `angle` (0 = +z, PI/2 = +x side). */
export function orbitPose(angle: number, out = new Vector3()): Vector3 {
  return rawPose(angle, orbitElevation(angle), out)
}
export const ORBIT_TARGET: [number, number, number] = [0, 0.7, 0]

/** Close-up on a talking agent: front-side 3/4, from the outer side of their table. */
export function talkShot(seat: Seat): Shot {
  const { pos, rotY } = seatPose(seat)
  const f = new Vector3(Math.sin(rotY), 0, Math.cos(rotY))         // facing (towards the board)
  const side = new Vector3(Math.cos(rotY), 0, -Math.sin(rotY))
  // ~2 m out to the side and slightly in front: close enough to read the face and gesture, far
  // enough that the camera never ends up inside a skater (the old 0.9 m offset did).
  const a = pos.clone().addScaledVector(f, 0.55).addScaledVector(side, 1.85)
  const b = pos.clone().addScaledVector(f, 0.55).addScaledVector(side, -1.85)
  const cam = Math.abs(a.x) > Math.abs(b.x) ? a : b                 // outer side: the other table stays out of frame
  const to = pos.clone().addScaledVector(f, 0.05)
  return {
    from: [cam.x, 1.75, cam.z], to: [to.x, 1.3, to.z],
    drift: [(to.x - cam.x) * 0.08, -0.03, (to.z - cam.z) * 0.08],     // slow push-in
    fov: 32,
  }
}

// ---- free (user-driven) camera poses ----
// Orbit poses for the free camera: a target plus a camera position given as azimuth (degrees from
// +z towards +x), elevation and distance. Players stand on the tables' z axis, so from the front
// (+z) they block the boards unless the camera is nearly overhead; these angles were picked with
// blockedBy() (margin 0.3) so no player covers either board.
export interface OrbitPose { pos: [number, number, number]; target: [number, number, number]; fov: number }
function polar(target: [number, number, number], azDeg: number, elDeg: number, dist: number, fov: number): OrbitPose {
  const a = azDeg * Math.PI / 180, e = elDeg * Math.PI / 180, h = Math.cos(e) * dist
  return { pos: [target[0] + Math.sin(a) * h, target[1] + Math.sin(e) * dist, target[2] + Math.cos(a) * h], target, fov }
}
/** Default free view: high 3/4 from the B end, both boards in frame, nobody in the way. */
export const FREE_HOME: OrbitPose = polar([-0.15, 0.7, 0], 68, 44, 7.0, 45)
/** Focus poses: 3/4 high from each board's outer end. */
export const FREE_FOCUS: Record<'A' | 'B' | 'both' | 'home', OrbitPose> = {
  A: polar([BOARD_X.A, BOARD_Y, 0], -55, 46, 2.9, 45),
  B: polar([BOARD_X.B, BOARD_Y, 0], 55, 46, 2.9, 45),
  both: FREE_HOME,
  home: FREE_HOME,
}
/** The orbit target is clamped to this box (the play area), so panning can't leave the lot. */
export const FREE_BOUNDS = { min: [-4.5, 0.3, -3.4] as const, max: [4.5, 2.6, 3.4] as const }
/** Distance limits (metres from the target; with the target box that keeps the camera within ~18 m of the centre) and the lowest the camera may go. */
export const FREE_MIN_DIST = 2, FREE_MAX_DIST = 14, FREE_MIN_Y = 0.35

// ---- occlusion check (used by the node sanity check; cheap enough for runtime too) ----
const PLAYER_R = 0.3, PLAYER_H = 1.95

function segHitsCapsule(p: Vector3, q: Vector3, c: Vector3): boolean {
  // sample the segment; fine for a sanity check
  const N = 60
  for (let i = 1; i < N; i++) {
    const t = i / N
    const x = p.x + (q.x - p.x) * t, y = p.y + (q.y - p.y) * t, z = p.z + (q.z - p.z) * t
    if (y > PLAYER_H || y < 0) continue
    if (Math.hypot(x - c.x, z - c.z) < PLAYER_R) return true
  }
  return false
}

/**
 * Players that block part of `board` from camera position `from`. Checks the board centre and four
 * points `margin` out from it (0.32 = the corner squares; the board half-width is 0.36).
 */
export function blockedBy(from: Vector3, board: BoardId, margin = 0.32): Seat[] {
  const pts = [[0, 0], [-margin, -margin], [margin, -margin], [-margin, margin], [margin, margin]]
    .map(([x, z]) => boardToWorld(board, x, BOARD_Y, z))
  const hits: Seat[] = []
  for (const seat of SEATS) {
    const c = seatPose(seat).pos
    if (pts.some(p => segHitsCapsule(from, p, c))) hits.push(seat)
  }
  return hits
}
