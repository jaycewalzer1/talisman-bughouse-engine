// Win celebration: map a trick's TrickPose (tricks/types.ts) onto the skater Pose.
//
//   0 .. MOUNT_S          step the board down in front, hop onto it
//   then, repeating:      trick.sample(0..durationS), then PAUSE_S standing on the deck
//
// Frames: the trick's rider frame IS the rig-root frame (origin between the feet on the ground,
// facing +z, +x = the skater's left). The trick deck's long axis (+x = nose) is mapped onto our deck
// model, whose nose points along +z, by a fixed +90° yaw. The trick's deck centre is the point 0.06
// above the wheel contact, which is (0, 0.06, 0) in our deck model. Our grip tape is 3 cm higher
// than the trick's (the model's DECK_TOP 0.11 vs STANDING rider.y 0.08), so soles sit at rider.y + 0.03.
// The deck is placed from deck.pos/rot every frame (snapDeck): it is never glued to the feet.
// Feet snap onto the grip when they are close to it. Otherwise they follow rider.y, lifted by the
// tuck, while the hips keep the rider's arc.
import { Euler, Quaternion, Vector3 } from 'three'
import type { Trick, TrickPose } from '../tricks/types'
import type { Pose, V3 } from './poses'

export const MOUNT_S = 0.75
export const PAUSE_S = 1.2

const SOLE_OFF = 0.03                 // our grip top (0.11) - trick STANDING rider.y (0.08)
const DECK_PIVOT_Y = 0.06             // trick deck centre, model coords
const GRIP_Y = 0.11                   // grip surface, model coords (= poses.DECK_TOP)
const STAND_HIP = 0.9                 // hip height above the soles when standing tall
const FRONT_X = 0.2, BACK_X = -0.2    // feet on the bolts (front = nose side = +x = left foot)
const TAU = Math.PI * 2

const _q = new Quaternion(), _q90 = new Quaternion().setFromAxisAngle(new Vector3(0, 1, 0), Math.PI / 2)
const _e = new Euler(), _v = new Vector3(), _w = new Vector3(), _up = new Vector3()

const clamp01 = (x: number) => Math.max(0, Math.min(1, x))
const smooth = (x: number) => { const c = clamp01(x); return c * c * (3 - 2 * c) }

/** Deck model transform for a trick deck pose: returns [pos, rot (YXZ euler)] and leaves _q = rotation. */
function deckFromTrick(tp: TrickPose): { pos: V3; rot: V3 } {
  const r = tp.deck.rot
  _e.set(r.roll % TAU, r.yaw % TAU, r.pitch % TAU, 'YZX')
  _q.setFromEuler(_e).multiply(_q90)
  _v.set(0, DECK_PIVOT_Y, 0).applyQuaternion(_q)
  const pos: V3 = [tp.deck.pos[0] - _v.x, tp.deck.pos[1] - _v.y, tp.deck.pos[2] - _v.z]
  _e.setFromQuaternion(_q, 'YXZ')
  return { pos, rot: [_e.x, _e.y, _e.z] }
}

/** Height of the grip under rig-local (x, z) for a deck at `pos` with rotation _q, or null when
 *  the foot is off the board or the board is not wheels-down. */
function gripUnder(pos: V3, x: number, z: number): number | null {
  _up.set(0, 1, 0).applyQuaternion(_q)
  if (_up.y < 0.6) return null
  // foot -> deck-model local
  _w.set(x - pos[0], 0, z - pos[2]).applyQuaternion(_q.clone().invert())
  if (Math.abs(_w.z) > 0.42 || Math.abs(_w.x) > 0.16) return null
  // point on the grip plane at that local (x, z): solve along the deck's up axis
  _v.set(_w.x, GRIP_Y, _w.z).applyQuaternion(_q)
  // the plane through that point with normal _up; height at (x, z)
  const px = pos[0] + _v.x, py = pos[1] + _v.y, pz = pos[2] + _v.z
  return py - ((x - px) * _up.x + (z - pz) * _up.z) / _up.y
}

function applyBody(p: Pose, tp: TrickPose) {
  const b = tp.body
  const sole = tp.rider.y + SOLE_OFF
  const { pos, rot } = deckFromTrick(tp)
  p.deckPos = pos; p.deckRot = rot; p.snapDeck = true

  // feet: front (left, +x) flicks off the nose toward the toe side, back (right) scoops the tail
  const fx = FRONT_X + 0.08 * b.frontFootFlick, fz = 0.06 * b.frontFootFlick
  const bx = BACK_X - 0.07 * b.backFootScoop, bz = -0.05 * b.backFootScoop
  const lift = 0.12 * b.tuck
  const foot = (x: number, z: number, extra: number): number => {
    const free = sole + lift + extra
    const g = gripUnder(pos, x, z)
    if (g === null) return free
    // close to the grip -> stand on it
    const d = free - g
    if (d < -0.06 || d > 0.04) return free
    return g
  }
  p.footL = [fz, foot(fx, fz, 0.05 * b.frontFootFlick)]
  p.footR = [bz, foot(bx, bz, 0)]
  p.footX = [fx, bx]

  // hips ride the rider's arc; crouch bends the knees
  p.hipsY = sole + STAND_HIP * (1 - 0.32 * b.crouch) - 0.92
  p.hips = [0, 0, 0]
  p.spine = [0.06 + 0.25 * b.crouch + 0.35 * b.lean, 0, 0]
  p.head = [0.25 - 0.15 * b.lean, -0.25, 0]          // eyes on the board (it's off to the left/front)
  const a = b.armsOut
  p.shL = [-0.2 * a, 0, 0.14 + 1.25 * a]; p.elL = [-0.25 - 0.35 * a, 0, 0]
  p.shR = [-0.2 * a, 0, -0.14 - 1.25 * a]; p.elR = [-0.25 - 0.35 * a, 0, 0]
  p.ikL = { w: 0, space: 'root', p: [0, 0, 0] }; p.ikR = { w: 0, space: 'root', p: [0, 0, 0] }
  p.spinY = tp.rider.rotY
}

export interface WinPhase { phase: 'mount' | 'trick' | 'pause'; loop: number; trickT: number }

/** Fill `p` with the win pose `u` seconds after the win started. */
export function winPose(p: Pose, trick: Trick, u: number): WinPhase {
  if (!(u >= 0)) u = 0
  if (u < MOUNT_S) {
    // step the deck down in front (the rig glides it there: no snap), hop on
    const stand = trick.sample(0)
    const { pos, rot } = deckFromTrick(stand)
    p.deckPos = pos; p.deckRot = rot; p.snapDeck = false
    const k = u / MOUNT_S
    const hop = smooth((k - 0.4) / 0.4)
    const air = Math.sin(Math.PI * clamp01((k - 0.4) / 0.4))
    const sole = (stand.rider.y + SOLE_OFF) * hop + 0.12 * air
    p.footL = [0, sole]; p.footR = [0, sole]
    p.footX = [FRONT_X * smooth(k / 0.4) + 0.12 * (1 - smooth(k / 0.4)), BACK_X * smooth(k / 0.4) - 0.12 * (1 - smooth(k / 0.4))]
    const crouch = k < 0.4 ? smooth(k / 0.4) * 0.5 : k > 0.85 ? 0.4 * (1 - smooth((k - 0.85) / 0.15)) : 0.1
    p.hipsY = sole + STAND_HIP * (1 - 0.32 * crouch) - 0.92 + 0.08 * air
    p.spine = [0.1 + 0.2 * crouch, 0, 0]
    p.head = [0.35, -0.2, 0]
    p.shL = [-0.1, 0, 0.3 + 0.6 * air]; p.elL = [-0.4, 0, 0]
    p.shR = [-0.1, 0, -0.3 - 0.6 * air]; p.elR = [-0.4, 0, 0]
    p.ikL = { w: 0, space: 'root', p: [0, 0, 0] }; p.ikR = { w: 0, space: 'root', p: [0, 0, 0] }
    return { phase: 'mount', loop: 0, trickT: 0 }
  }
  const period = trick.durationS + PAUSE_S
  const v = u - MOUNT_S
  const loop = Math.floor(v / period)
  const tt = v - loop * period
  const inTrick = tt < trick.durationS
  applyBody(p, trick.sample(inTrick ? tt : trick.durationS))
  return { phase: inTrick ? 'trick' : 'pause', loop, trickT: inTrick ? tt : trick.durationS }
}
