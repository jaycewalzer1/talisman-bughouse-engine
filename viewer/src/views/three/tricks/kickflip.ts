// KICKFLIP — ollie pop off the tail, front foot slides up the grip and flicks off the toe-side
// edge of the nose, board does one full roll about its long axis while the rider jumps straight
// up with knees tucked, back foot catches first, front foot stomps, landing absorb, settle.
//
// Pure function of time (see ./types.ts). Rider faces +z, so the TOE-side edge is +z. The flick
// pushes that edge down and the heel-side edge comes up and over: in the contract's roll (about
// +x) a positive angle sends +z downward, so roll runs 0 -> +2π.
// NOTE: the exported pose negates it (0 -> -2π) to match treFlip.ts, whose flip direction follows
// the 360-flip how-to guides; both tricks must flip the same way.
//
// Timeline (s), durationS = 1.45:
//   0.00-0.28  wind-up: crouch down, lean in a little, arms start to open
//   0.28-0.36  pop: legs fire, tail strikes, nose rises (pitch up to ~0.5 rad)
//   0.36       takeoff (rider leaves the deck); 0.36-0.50 front foot drags up and levels the nose
//   0.40-0.50  flick off the toe-side corner of the nose
//   0.44-0.76  the flip: one full roll (fast off the flick, slowing into the catch)
//   0.41-0.80  feet separate from the deck (peak ~15 cm gap mid-flip); knees tucked high, arms out
//   0.74-0.84  catch: tail rises into the back foot first (nose-down pitch), then front foot levels it
//   0.84       touchdown (0.48 s airborne, ~45 cm feet height at the apex)
//   0.84-0.98  landing absorb (deep crouch)
//   0.98-1.45  settle back to STANDING and roll away

import { STANDING, type Trick, type TrickPose } from './types'

const TAU = Math.PI * 2
const DECK_Y = STANDING.deck.pos[1] // 0.06
const RIDER_Y = STANDING.rider.y    // 0.08 (feet on grip)
const HALF_DECK = 0.4               // nose/tail distance from deck centre (m)

const T_POP_START = 0.28
const T_STRIKE = 0.36        // tail hits the ground; rider leaves
const T_LEVEL = 0.5          // front foot has dragged the nose level
const T_TAKEOFF = 0.36
const T_LAND = 0.84
const JUMP_H = 0.45          // arcade-high apex of the feet above standing
const T_FLIP0 = 0.44
const T_FLIP1 = 0.76
const T_GAP0 = 0.41
const T_GAP1 = 0.8
const FEET_GAP = 0.15        // extra feet clearance above the deck mid-flip
const T_ABSORB = 0.98
const DURATION = 1.45

const clamp01 = (x: number) => (x < 0 ? 0 : x > 1 ? 1 : x)
/** Smoothstep from 0 at a to 1 at b. */
const ss = (a: number, b: number, t: number) => {
  const u = clamp01((t - a) / (b - a))
  return u * u * (3 - 2 * u)
}
/** 0 before a, rises to 1 over [a,b], holds, falls back to 0 over [c,d]. */
const plateau = (a: number, b: number, c: number, d: number, t: number) => ss(a, b, t) * (1 - ss(c, d, t))
const lerp = (a: number, b: number, k: number) => a + (b - a) * k

function riderY(t: number): number {
  if (t <= T_TAKEOFF || t >= T_LAND) return RIDER_Y
  const s = (t - T_TAKEOFF) / (T_LAND - T_TAKEOFF)
  return RIDER_Y + JUMP_H * 4 * s * (1 - s) // ballistic arc
}

function roll(t: number): number {
  const u = clamp01((t - T_FLIP0) / (T_FLIP1 - T_FLIP0))
  // Bias time forward so the board whips off the flick and slows as the feet come back to it.
  return TAU * ss(0, 1, Math.pow(u, 0.75))
}

function pitch(t: number): number {
  // Ollie pop: nose up as the tail strikes, then the front-foot drag levels it out.
  const pop = 0.5 * ss(T_POP_START, T_STRIKE, t) * (1 - ss(T_STRIKE, T_LEVEL, t))
  // Catch: tail rises into the back foot first (nose down), front foot then stomps it level.
  const catchTilt = -0.16 * plateau(0.74, 0.79, 0.79, 0.84, t)
  return pop + catchTilt
}

function feetGap(t: number): number {
  if (t <= T_GAP0 || t >= T_GAP1) return 0
  const u = (t - T_GAP0) / (T_GAP1 - T_GAP0)
  return FEET_GAP * Math.pow(Math.sin(Math.PI * u), 0.8)
}

function crouch(t: number): number {
  const base = STANDING.body.crouch
  if (t < T_POP_START) return lerp(base, 0.72, ss(0, T_POP_START, t))      // wind-up
  if (t < T_LAND) return lerp(0.72, 0.15, ss(T_POP_START, T_STRIKE, t))    // explode up (tuck takes over in the air)
  if (t < T_ABSORB) return lerp(0.15, 0.85, ss(T_LAND, 0.93, t))           // absorb
  return lerp(0.85, base, ss(T_ABSORB, DURATION, t))                       // settle
}

export const kickflip: Trick = {
  name: 'KICKFLIP',
  durationS: DURATION,
  sample(tIn: number): TrickPose {
    const t = Math.min(Math.max(Number.isFinite(tIn) ? tIn : 0, 0), DURATION)
    const ry = riderY(t)
    const p = pitch(t)
    // Deck centre: follows the feet, lifted while the nose is up (tail stays on the ground at the
    // strike), and dropped below the feet while it is flipping.
    const deckY = DECK_Y + (ry - RIDER_Y) + HALF_DECK * Math.sin(Math.max(p, 0)) - feetGap(t)
    const sb = STANDING.body
    return {
      rider: { y: ry, rotY: 0 },
      deck: { pos: [0, deckY, 0], rot: { roll: -roll(t), yaw: 0, pitch: p } },
      body: {
        crouch: crouch(t),
        tuck: 0.95 * plateau(0.38, 0.52, 0.72, 0.84, t),
        armsOut: lerp(sb.armsOut, 1, plateau(0.2, 0.42, 0.9, 1.35, t)),
        frontFootFlick: plateau(0.36, 0.47, 0.5, 0.74, t),
        backFootScoop: 0,
        lean: 0.18 * plateau(0.08, 0.28, 0.3, 0.42, t) - 0.12 * plateau(0.5, 0.62, 0.7, 0.82, t)
          + 0.22 * plateau(0.84, 0.94, 0.98, 1.4, t),
      },
    }
  },
}
