// 360 SHUV (360 pop shove-it): the back foot pops the tail and scoops it backside, the board spins
// one full turn about the vertical axis under the rider with no flip, the rider jumps straight up
// (no body rotation), catches it with both feet and rolls away. Pure function of time — see types.ts.
//
// Timeline (durationS = 1.5 s):
//   0.00–0.27  wind-up: crouch down, arms come out, slight forward lean
//   0.27–0.34  pop + scoop: nose pitches up (tail strikes), back foot scoops, legs extend
//   0.34–0.42  take-off: deck levels out as it leaves the ground and starts turning
//   0.40–0.84  spin (0.44 s; rider airborne 0.34–0.86 = 0.52 s): full backside 360° yaw, deck flat and a bit below the feet,
//              knees pulled up / spread to let it spin, arms wide, small upper-body counter-twist
//   0.80–0.86  catch: feet come down onto the deck, deck rides back up under them
//   0.86–1.10  landing absorb: deep crouch
//   1.10–1.50  settle back to STANDING

import { STANDING, type Trick, type TrickPose } from './types'

const TAU = Math.PI * 2

/** Cosine ease 0..1 with zero slope at both ends (keeps every track C1-continuous). */
const ease = (u: number): number => 0.5 - 0.5 * Math.cos(Math.PI * Math.min(1, Math.max(0, u)))

/** Keyframe track: [time, value] pairs sorted by time, cosine-eased between keys, held outside. */
function track(keys: ReadonlyArray<readonly [number, number]>): (t: number) => number {
  return (t) => {
    if (t <= keys[0][0]) return keys[0][1]
    for (let i = 1; i < keys.length; i++) {
      const [t1, v1] = keys[i]
      if (t <= t1) {
        const [t0, v0] = keys[i - 1]
        return v0 + (v1 - v0) * ease((t - t0) / (t1 - t0))
      }
    }
    return keys[keys.length - 1][1]
  }
}

const DURATION = 1.5
const S = STANDING
const FEET_Y = S.rider.y          // 0.08
const DECK_Y = S.deck.pos[1]       // 0.06
const STAND_GAP = FEET_Y - DECK_Y  // 0.02 (shoe sole above deck centre)

// Jump arc: ballistic parabola for the feet from take-off to touchdown.
const T_TAKEOFF = 0.34
const T_TOUCH = 0.86
const JUMP_H = 0.5

// Yaw spin window (in the air). Backside for a regular rider facing +z: the tail (-x) swings
// behind the rider (-z), which is a negative rotation about +y.
const T_SPIN0 = 0.4
const T_SPIN1 = 0.84
const SPIN_DIR = -1

const pitchT = track([[0, 0], [0.27, 0], [0.33, 0.34], [0.42, 0]])
// Feet-to-deck gap: deck drops a bit below the feet while it spins, closes again for the catch.
const gapT = track([[0, STAND_GAP], [T_TAKEOFF, STAND_GAP], [0.46, 0.17], [0.74, 0.17], [T_TOUCH - 0.03, STAND_GAP]])
const riderRotT = track([[0, 0], [0.38, 0], [0.54, 0.14], [0.72, 0.05], [0.9, -0.04], [1.12, 0]])

const crouchT = track([
  [0, S.body.crouch], [0.27, 0.75], [0.35, 0.15], [0.5, 0.35], [0.8, 0.4],
  [T_TOUCH, 0.5], [1.0, 0.82], [1.25, 0.3], [DURATION, S.body.crouch],
])
const tuckT = track([[0, 0], [0.34, 0], [0.5, 0.6], [0.74, 0.55], [0.84, 0.12], [0.92, 0]])
const armsT = track([
  [0, S.body.armsOut], [0.27, 0.45], [0.45, 0.95], [0.82, 0.9], [1.05, 0.6], [DURATION, S.body.armsOut],
])
const scoopT = track([[0, 0], [0.24, 0], [0.33, 1], [0.5, 0.55], [0.64, 0]])
const leanT = track([
  [0, 0], [0.27, 0.25], [0.38, 0.05], [0.62, -0.06], [T_TOUCH, 0.08], [1.02, 0.22], [DURATION, 0],
])

function riderY(t: number): number {
  if (t <= T_TAKEOFF || t >= T_TOUCH) return FEET_Y
  const u = (t - T_TAKEOFF) / (T_TOUCH - T_TAKEOFF)
  return FEET_Y + JUMP_H * 4 * u * (1 - u)
}

function deckYaw(t: number): number {
  if (t <= T_SPIN0) return 0
  if (t >= T_SPIN1) return SPIN_DIR * TAU
  return SPIN_DIR * TAU * ease((t - T_SPIN0) / (T_SPIN1 - T_SPIN0))
}

export const shuv360: Trick = {
  name: '360 SHUV',
  durationS: DURATION,
  sample(tIn: number): TrickPose {
    const t = Math.min(DURATION, Math.max(0, Number.isFinite(tIn) ? tIn : 0))
    const ry = riderY(t)
    const pitch = pitchT(t)
    // Tail stays on the ground during the pop: lift the deck centre by the tail drop (half-length ~0.38).
    const popLift = 0.38 * Math.sin(Math.max(0, pitch))
    const deckY = ry - gapT(t) + popLift
    return {
      rider: { y: ry, rotY: riderRotT(t) },
      deck: {
        pos: [0, deckY, 0],
        rot: { roll: 0, yaw: deckYaw(t), pitch },
      },
      body: {
        crouch: crouchT(t),
        tuck: tuckT(t),
        armsOut: armsT(t),
        frontFootFlick: 0,
        backFootScoop: scoopT(t),
        lean: leanT(t),
      },
    }
  },
}
