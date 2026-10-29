// Contract for win-celebration skate tricks. Each trick module exports a `Trick` that is a pure
// function of time: no three.js objects, no React — the character rig samples it every frame and
// applies the numbers to its own body parts and deck. That keeps tricks testable headlessly and
// independent of how the characters are modelled.
//
// Coordinates are in the RIDER's local frame (metres / radians), origin on the ground between the
// rider's feet, +y up, rider facing +z, deck long axis (nose) along +x when riding normally.
//
// Deck rotation axes (applied in this order: yaw Y, then pitch Z, then roll X — i.e. Euler 'YZX'):
//   roll  (x): rotation about the deck's long axis   -> kickflip / heelflip
//   yaw   (y): rotation about the vertical axis      -> shuvit / 360 shuv
//   pitch (z): end-over-end about the deck's width   -> impossible (wraps around the back foot)
// A full rotation is 2π. Tricks must start AND end at deck rotation (0,0,0) mod 2π with the deck
// back on the ground (pos.y = 0.06, wheels down) so the landing is clean.

import type { Seat } from '../../../events/types'

export interface TrickPose {
  /** Whole-rider offset: jump height and spin. y = feet height above ground. */
  rider: { y: number; rotY: number }
  /** Deck centre position relative to the rider origin, and rotation (see axes above). */
  deck: { pos: [number, number, number]; rot: { roll: number; yaw: number; pitch: number } }
  /** 0..1 body shaping the rig maps onto its joints. */
  body: {
    crouch: number          // knees bent / hips down (pop wind-up and landing absorb)
    tuck: number            // knees pulled up in the air
    armsOut: number         // arms out for balance
    frontFootFlick: number  // front foot sliding/flicking off the nose (kickflip, tre)
    backFootScoop: number   // back foot scooping the tail (shuv, tre, impossible)
    lean: number            // -1..1 torso lean back/forward
  }
}

export interface Trick {
  name: string              // e.g. "TRE FLIP" — also shown by the HUD announcer
  durationS: number         // one full attempt: wind-up, pop, air, catch, land, roll-away
  /** t in [0, durationS]. Must be continuous; pose at t=0 and t=durationS = standing on the deck. */
  sample(t: number): TrickPose
}

/** Who does which trick when their team wins (the user's choice). */
export const WIN_TRICK_OF: Record<Seat, 'treFlip' | 'kickflip' | 'shuv360' | 'ollieImpossible'> = {
  A_white: 'treFlip',
  B_black: 'kickflip',
  A_black: 'shuv360',
  B_white: 'ollieImpossible',
}

/** Neutral standing-on-deck pose; tricks start/end here. */
export const STANDING: TrickPose = {
  rider: { y: 0.06 + 0.02, rotY: 0 },
  deck: { pos: [0, 0.06, 0], rot: { roll: 0, yaw: 0, pitch: 0 } },
  body: { crouch: 0.1, tuck: 0, armsOut: 0.15, frontFootFlick: 0, backFootScoop: 0, lean: 0 },
}
