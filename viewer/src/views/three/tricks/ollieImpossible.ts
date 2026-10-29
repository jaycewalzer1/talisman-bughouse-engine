// OLLIE IMPOSSIBLE — the deck wraps end-over-end (pitch 2π) around the rider's BACK foot.
//
// Mechanics: the back foot pops the tail and, instead of sliding up the board like an ollie,
// stays in contact with the tail and scoops; the board pivots around that foot like a wand around
// a finger (nose up -> over the top -> down behind -> under -> forward again) while the front foot
// lifts completely clear. The rider pops straight up (higher than an ollie, since the 0.8 m deck
// has to swing below the foot), catches with the front foot as the board comes level, and lands.
//
// Geometry: the back foot (sole) sits on the tail at x = -BACK_FOOT_X, y = rider.y. The deck centre
// is 0.02 below the sole and BACK_FOOT_X ahead of it. During the trick the deck centre orbits the
// back foot: centre = foot + R(pitch)·(BACK_FOOT_X, -0.02), so the tail end (0.4 m behind the centre)
// stays ~0.13 m from the foot the whole time. If the swinging deck would touch the ground, the
// whole rider is lifted (ground clamp), which also produces the tail-strike during the pop.
//
// Timeline (durationS = 1.7 s):
//   0.00–0.30  wind-up: crouch down, arms start out
//   0.30–0.34  pop: tail strikes, nose rises, legs extend, back foot starts scooping
//   0.34–1.16  airtime (ballistic arc, apex ≈ 0.83 m feet height at ~0.75 s)
//   0.30–1.02  wrap: pitch 0 -> 2π (Hermite ease: snappy pop, ~3π/2 at the apex, eased to 0 at catch)
//              front foot fully lifted away 0.45–0.88, tuck peaks ~0.65
//   0.90–1.16  catch: front foot comes back down onto the bolts, legs extend to meet the ground
//   1.16–1.40  landing absorb (crouch peaks ~1.24)
//   1.40–1.70  settle / roll away back to STANDING

import type { Trick, TrickPose } from './types'

const TAU = Math.PI * 2
const DUR = 1.7
const BACK_FOOT_X = 0.27     // back foot distance behind deck centre (on the tail)
const SOLE_TO_DECK = 0.02    // deck centre sits this far below the sole
const HALF_LEN = 0.4         // deck half-length (tail/nose ends)
const TRUCK_X = 0.28         // wheel contact points, relative to the deck centre
const WHEEL_DROP = 0.06
const END_CLEARANCE = 0.03   // min height of the deck ends above ground
const BASE_Y = 0.08          // STANDING rider.y

const WRAP_T0 = 0.30
const WRAP_T1 = 1.02
const AIR_T0 = 0.34
const AIR_T1 = 1.16
const AIR_PEAK = 0.75        // apex height above BASE_Y (≈ g·T²/8 for 0.82 s of air)

const clamp01 = (x: number) => (x < 0 ? 0 : x > 1 ? 1 : x)
const smooth = (x: number) => { const u = clamp01(x); return u * u * (3 - 2 * u) }

/** Piecewise keyframes [t, v], smoothstep-blended between keys (C1, flat at each key). */
function track(keys: ReadonlyArray<readonly [number, number]>) {
  return (t: number): number => {
    if (t <= keys[0][0]) return keys[0][1]
    for (let i = 1; i < keys.length; i++) {
      const [t1, v1] = keys[i]
      if (t <= t1) {
        const [t0, v0] = keys[i - 1]
        return v0 + (v1 - v0) * smooth((t - t0) / (t1 - t0))
      }
    }
    return keys[keys.length - 1][1]
  }
}

const crouch = track([[0, 0.1], [0.30, 0.65], [0.40, 0.12], [1.10, 0.12], [1.24, 0.7], [1.55, 0.1], [DUR, 0.1]])
const tuck = track([[0, 0], [0.36, 0], [0.62, 0.6], [0.95, 0.25], [1.14, 0], [DUR, 0]])
const armsOut = track([[0, 0.15], [0.30, 0.35], [0.50, 0.95], [1.05, 0.85], [1.30, 0.6], [1.65, 0.15], [DUR, 0.15]])
const frontFootFlick = track([[0, 0], [0.30, 0], [0.45, 1], [0.88, 1], [1.04, 0], [DUR, 0]])
const backFootScoop = track([[0, 0], [0.26, 0], [0.40, 1], [0.92, 1], [1.10, 0], [DUR, 0]])
const lean = track([[0, 0], [0.30, 0.15], [0.42, -0.1], [0.80, 0.05], [1.20, 0.2], [1.55, 0], [DUR, 0]])

/** Deck pitch 0 -> 2π over the wrap window: Hermite ease with a snappy start (the pop kicks the
 *  nose up at ~10 rad/s) and zero velocity at the catch (front foot stops the board level). */
const POP_SLOPE = 1.2
function pitchAt(t: number): number {
  const u = clamp01((t - WRAP_T0) / (WRAP_T1 - WRAP_T0))
  const u2 = u * u, u3 = u2 * u
  return TAU * (POP_SLOPE * (u3 - 2 * u2 + u) + (3 * u2 - 2 * u3))
}

/** Raw (unclamped) feet height: ballistic parabola during the air window. */
function rawRiderY(t: number): number {
  if (t <= AIR_T0 || t >= AIR_T1) return BASE_Y
  const tau = (t - AIR_T0) / (AIR_T1 - AIR_T0)
  return BASE_Y + 4 * AIR_PEAK * tau * (1 - tau)
}

/** Rotate (x, y) by angle a in the x-y plane (pitch about +z: nose +x goes toward +y). */
function rot(x: number, y: number, a: number): [number, number] {
  const c = Math.cos(a), s = Math.sin(a)
  return [x * c - y * s, x * s + y * c]
}

/** Lowest required feet height so the deck (ends and wheels) at this pitch stays above ground. */
function minRiderY(pitch: number): number {
  // Points relative to the back-foot sole, in deck-local coordinates around the deck centre.
  const cx = BACK_FOOT_X, cy = -SOLE_TO_DECK
  let need = -Infinity
  const pts: Array<[number, number, number]> = [
    [cx - HALF_LEN, cy, END_CLEARANCE],
    [cx + HALF_LEN, cy, END_CLEARANCE],
    [cx - TRUCK_X, cy - WHEEL_DROP, 0],
    [cx + TRUCK_X, cy - WHEEL_DROP, 0],
  ]
  for (const [px, py, clear] of pts) {
    const [, ry] = rot(px, py, pitch)
    need = Math.max(need, clear - ry)
  }
  return need
}

function sample(tIn: number): TrickPose {
  const t = Math.min(DUR, Math.max(0, Number.isFinite(tIn) ? tIn : 0))
  const pitch = pitchAt(t)
  const riderY = Math.max(rawRiderY(t), minRiderY(pitch))

  // Deck centre orbits the back foot.
  const footX = -BACK_FOOT_X
  const [ox, oy] = rot(BACK_FOOT_X, -SOLE_TO_DECK, pitch)
  const deckPos: [number, number, number] = [footX + ox, riderY + oy, 0]

  return {
    rider: { y: riderY, rotY: 0 },
    deck: { pos: deckPos, rot: { roll: 0, yaw: 0, pitch } },
    body: {
      crouch: crouch(t),
      tuck: tuck(t),
      armsOut: armsOut(t),
      frontFootFlick: frontFootFlick(t),
      backFootScoop: backFootScoop(t),
      lean: lean(t),
    },
  }
}

export const ollieImpossible: Trick = {
  name: 'OLLIE IMPOSSIBLE',
  durationS: DUR,
  sample,
}
