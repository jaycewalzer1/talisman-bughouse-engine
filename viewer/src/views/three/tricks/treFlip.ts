// TRE FLIP (360 flip): backside 360 shove-it + kickflip in one rotation. Pure function of time —
// see ./types.ts for the frame and axis conventions.
//
// Stance: rider faces +z, nose at +x = rider's left -> regular stance. Toe side is +z, heel side -z.
//   yaw  (backside 360 shuv): the back foot scoops the tail (-x) behind the heels (-z). A positive
//        rotation about +y would swing the tail toward +z, so backside = NEGATIVE yaw.
//   roll (kickflip): the front foot slides up and flicks off the nose diagonally toward the heel side,
//        so the toe edge (+z) rises and goes over. A positive roll about +x drops +z, so this is
//        NEGATIVE roll.
// Both end at exactly -2π (one full turn each), so they wrap seamlessly to 0.
//
// Timeline (s):
//   0.00–0.32  wind-up: crouch, lean forward, shoulders wind slightly frontside
//   0.28–0.40  pop: tail snaps (nose pitches up, levels by 0.40), legs extend; feet leave at 0.36
//   0.36–0.92  air (0.56 s): ballistic rider, peak ~0.55 m at 0.64
//   0.37–0.84  board: backside 360 scoop (yaw) + kickflip (roll), deck drops ~13 cm below the
//              feet, eased fast-then-slowing into the catch; knees tucked, arms out
//   0.84       catch: both feet on the bolts, ride the deck down
//   0.92       touch-down
//   0.92–1.16  landing absorb (deep crouch)
//   1.16–1.60  settle back to STANDING

import { STANDING, type Trick, type TrickPose } from './types'

const TAU = Math.PI * 2

const T_TAKEOFF = 0.36
const T_LAND = 0.92
const AIR_HEIGHT = 0.55 // peak feet rise above standing, metres (arcade-exaggerated)
const YAW_START = 0.37
const ROLL_START = 0.40
const T_CATCH = 0.84
const DURATION = 1.6

const DECK_Y = STANDING.deck.pos[1]           // 0.06, wheels on ground
const FEET_ON_DECK = STANDING.rider.y - DECK_Y // 0.02, deck surface offset under the feet
const TAIL_HALF = 0.4                          // centre-to-tail distance used for the pop lift

const clamp01 = (x: number): number => (x < 0 ? 0 : x > 1 ? 1 : x)
const smooth = (x: number): number => {
  const u = clamp01(x)
  return u * u * (3 - 2 * u)
}
const lerp = (a: number, b: number, u: number): number => a + (b - a) * u

/** Piecewise keyframe track with smoothstep blends (C0, zero velocity at each key). */
type Key = readonly [t: number, v: number]
function track(keys: readonly Key[], t: number): number {
  if (t <= keys[0][0]) return keys[0][1]
  for (let i = 1; i < keys.length; i++) {
    const [t1, v1] = keys[i]
    if (t <= t1) {
      const [t0, v0] = keys[i - 1]
      return lerp(v0, v1, smooth((t - t0) / (t1 - t0)))
    }
  }
  return keys[keys.length - 1][1]
}

/** Flip/shuv progress 0..1: already moving at the flick, fastest early-middle, slowing to 0 at catch. */
function spinProgress(t: number, start: number): number {
  const u = clamp01((t - start) / (T_CATCH - start))
  const easeOut = 1 - (1 - u) * (1 - u) * (1 - u)
  return 0.5 * smooth(u) + 0.5 * easeOut
}

/** Feet height: flat on the deck, ballistic parabola between takeoff and touch-down. */
function riderY(t: number): number {
  if (t <= T_TAKEOFF || t >= T_LAND) return STANDING.rider.y
  const s = (t - T_TAKEOFF) / (T_LAND - T_TAKEOFF)
  return STANDING.rider.y + AIR_HEIGHT * 4 * s * (1 - s)
}

// Pop: nose pitches up as the tail strikes, then levels as the flick takes over.
const PITCH: readonly Key[] = [[0.28, 0], [0.35, 0.4], [0.40, 0]]
// Board-to-feet separation during the flip (feet lift ~13 cm off the grip).
const GAP: readonly Key[] = [[0.36, 0], [0.46, 0.13], [0.72, 0.13], [T_CATCH, 0]]

const CROUCH: readonly Key[] = [
  [0, STANDING.body.crouch], [0.30, 0.72], [0.38, 0.15], [0.55, 0.35], [0.80, 0.35],
  [0.92, 0.45], [1.02, 0.8], [1.30, 0.3], [DURATION, STANDING.body.crouch],
]
const TUCK: readonly Key[] = [[0.38, 0], [0.56, 0.9], [0.78, 0.8], [0.92, 0.15], [1.05, 0]]
const ARMS: readonly Key[] = [
  [0, STANDING.body.armsOut], [0.30, 0.4], [0.50, 0.95], [0.86, 0.9], [1.05, 0.7],
  [DURATION, STANDING.body.armsOut],
]
const FLICK: readonly Key[] = [[0.34, 0], [0.43, 1], [0.56, 0]]
const SCOOP: readonly Key[] = [[0.28, 0], [0.37, 1], [0.50, 0]]
const LEAN: readonly Key[] = [
  [0, 0], [0.30, 0.25], [0.38, -0.12], [0.60, 0.05], [0.88, -0.08], [1.02, 0.2], [1.40, 0],
]
// Shoulders wind frontside, then counter-rotate against the backside scoop, then square up.
const ROT_Y: readonly Key[] = [[0, 0], [0.30, 0.14], [0.42, -0.18], [0.80, -0.06], [1.20, 0]]

export const treFlip: Trick = {
  name: 'TRE FLIP',
  durationS: DURATION,
  sample(tIn: number): TrickPose {
    const t = Math.min(Math.max(Number.isFinite(tIn) ? tIn : 0, 0), DURATION)

    const pitch = track(PITCH, t)
    const feet = riderY(t)
    // Deck surface follows the feet, minus the flip gap, plus the pop lift (tail stays on the ground
    // while the nose rises).
    const deckY = feet - FEET_ON_DECK - track(GAP, t) + TAIL_HALF * Math.sin(pitch)

    return {
      rider: { y: feet, rotY: track(ROT_Y, t) },
      deck: {
        pos: [0, deckY, 0],
        rot: {
          roll: -TAU * spinProgress(t, ROLL_START),
          yaw: -TAU * spinProgress(t, YAW_START),
          pitch,
        },
      },
      body: {
        crouch: track(CROUCH, t),
        tuck: track(TUCK, t),
        armsOut: track(ARMS, t),
        frontFootFlick: track(FLICK, t),
        backFootScoop: track(SCOOP, t),
        lean: track(LEAN, t),
      },
    }
  },
}
