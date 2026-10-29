// Code-driven poses. Each state returns a target Pose for the current time; ProceduralCharacter
// blends toward it every frame (damped lerp), so switching states is always smooth. One-shot
// reactions (fist pump, head-in-hands, ...) are full poses mixed over the state pose with an
// ease-in/ease-out weight.
//
// Rig-local axes: +z = toward the board (forward), +x = the character's LEFT, +y = up.
// Arms hang along -y: rotation x < 0 swings forward/up, z > 0 swings the LEFT arm outward.
// Hands can instead be driven by 2-bone IK toward a point (`ikL` / `ikR`) given in the rig root,
// the spine, the deck, or world space. Legs are always foot-target IK in the sagittal plane.
// The deck uses Euler order YXZ: [pitch (x, < 0 = nose up), yaw (y), roll around its length (z)].
import { Euler, Vector3 } from 'three'
import type { Reaction } from '../fxBus'
import type { CharacterDriver } from './rig'
import { winPose } from './trickPose'

export type V3 = [number, number, number]
export type IkSpace = 'root' | 'spine' | 'deck' | 'world'
/** `pole`: direction the elbow should point (spine space, written for the LEFT arm; mirrored for the right). */
export interface Ik { w: number; space: IkSpace; p: V3; pole?: V3 }

export interface Pose {
  hipsY: number            // hip height offset (crouch < 0 < jump)
  hips: V3
  spine: V3                // x > 0 = lean forward
  head: V3
  shL: V3; elL: V3
  shR: V3; elR: V3
  footL: [number, number]  // foot target [forward z, sole height y]
  footR: [number, number]
  footX: [number, number] | null   // optional sideways foot targets [left, right] (stance on a deck); null = under the hips
  deckPos: V3
  deckRot: V3
  snapDeck: boolean        // deck follows the target exactly (fast tricks; angles may wrap)
  ikL: Ik
  ikR: Ik
  spinY: number            // whole-body yaw, applied unsmoothed (spins)
}

export const POSE_KEYS = ['hips', 'spine', 'head', 'shL', 'elL', 'shR', 'elR', 'deckPos', 'deckRot'] as const

export const DECK_TOP = 0.11                  // standing on a deck: sole height
const TAIL_TIP: V3 = [0, 0.12, -0.4]          // deck-local
const NOSE_GRAB: V3 = [0, 0.05, 0.36]         // deck-local: where a hand holds it
const ASIDE_POS: V3 = [-0.38, 0, 0.14]
const ASIDE_ROT: V3 = [0, 0.25, 0]
const FLIP_REST: V3 = [-0.14, 0, 0.36]
const CATCH: V3 = [-0.3, 0.62, 0.3]
const TAU = Math.PI * 2
// elbow pole hints (left-arm convention)
const POCKET: V3 = [0.8, -0.2, -0.5]
const CROSSED: V3 = [1, -0.45, 0.45]
const WATCH: V3 = [0.9, -0.5, 0.25]
const HOLD: V3 = [0.8, -0.5, -0.2]

export const REACTION_DUR: Record<Reaction, number> = {
  fistPump: 1.4, headInHands: 1.9, impatientTap: 2.0, celebrate: 2.3, slump: 2.3, talk: 1.6,
}
export const REACTION_PRIORITY: Record<Reaction, number> = {
  celebrate: 3, slump: 3, fistPump: 2, headInHands: 2, impatientTap: 1, talk: 1,
}

// ---------------------------------------------------------------------------------------------
// helpers

const clamp01 = (x: number) => Math.max(0, Math.min(1, x))
const smooth = (x: number) => { const c = clamp01(x); return c * c * (3 - 2 * c) }
const seg = (x: number, a: number, b: number) => clamp01((x - a) / (b - a))
const lerp = (a: number, b: number, k: number) => a + (b - a) * k
const lerpV = (a: V3, b: V3, k: number): V3 => [lerp(a[0], b[0], k), lerp(a[1], b[1], k), lerp(a[2], b[2], k)]
const ik = (space: IkSpace, p: V3, w = 1, pole?: V3): Ik => ({ w, space, p, pole })
const NO_IK = (): Ik => ({ w: 0, space: 'root', p: [0, 0, 0] })

const _e = new Euler(0, 0, 0, 'YXZ'), _v = new Vector3()
/** Deck position such that deck-local point `local` lands on rig-local `at` with rotation `rot`. */
export function deckAt(at: V3, rot: V3, local: V3): V3 {
  _e.set(rot[0], rot[1], rot[2], 'YXZ'); _v.set(local[0], local[1], local[2]).applyEuler(_e)
  return [at[0] - _v.x, at[1] - _v.y, at[2] - _v.z]
}

function base(): Pose {
  return {
    hipsY: -0.02, hips: [0, 0, 0], spine: [0.06, 0, 0], head: [0, 0, 0],
    shL: [0.02, 0, 0.12], elL: [-0.3, 0, 0], shR: [0.02, 0, -0.12], elR: [-0.3, 0, 0],
    footL: [0, 0], footR: [0, 0], footX: null, deckPos: [...ASIDE_POS], deckRot: [...ASIDE_ROT], snapDeck: false,
    ikL: NO_IK(), ikR: NO_IK(), spinY: 0,
  }
}

function clonePose(p: Pose): Pose {
  return {
    ...p, hips: [...p.hips], spine: [...p.spine], head: [...p.head],
    shL: [...p.shL], elL: [...p.elL], shR: [...p.shR], elR: [...p.elR],
    footL: [...p.footL], footR: [...p.footR], footX: p.footX ? [...p.footX] : null, deckPos: [...p.deckPos], deckRot: [...p.deckRot],
    ikL: { ...p.ikL, p: [...p.ikL.p] }, ikR: { ...p.ikR, p: [...p.ikR.p] },
  }
}

function mixIk(a: Ik, b: Ik, w: number): Ik {
  const useB = b.w > 0 && (a.w === 0 || w > 0.5)
  const src = useB ? b : a
  return { w: lerp(a.w, b.w, w), space: src.space, p: [...src.p], pole: src.pole }
}

function mixPose(a: Pose, b: Pose, w: number): Pose {
  const o = clonePose(a)
  for (const k of POSE_KEYS) o[k] = lerpV(a[k], b[k], w)
  o.hipsY = lerp(a.hipsY, b.hipsY, w)
  o.footL = [lerp(a.footL[0], b.footL[0], w), lerp(a.footL[1], b.footL[1], w)]
  o.footR = [lerp(a.footR[0], b.footR[0], w), lerp(a.footR[1], b.footR[1], w)]
  o.footX = w > 0.5 ? (b.footX ? [...b.footX] : null) : (a.footX ? [...a.footX] : null)
  o.ikL = mixIk(a.ikL, b.ikL, w)
  o.ikR = mixIk(a.ikR, b.ikR, w)
  o.spinY = lerp(a.spinY, b.spinY, w)
  return o
}

// ---------------------------------------------------------------------------------------------
// idle: cycles between little skater bits so the four never look frozen

const IDLE_SEG = 6
const IDLE_KINDS = ['nod', 'flip', 'roll', 'lean'] as const
type IdleKind = typeof IDLE_KINDS[number]

function idleKind(t: number, seed: number): { kind: IdleKind; u: number } {
  const tt = t + seed * 2.3
  const n = Math.floor(tt / IDLE_SEG)
  const h = Math.abs(Math.sin(n * 12.9898 + seed * 78.233) * 43758.5453)
  return { kind: IDLE_KINDS[Math.floor(h) % IDLE_KINDS.length], u: tt - n * IDLE_SEG }
}

/** Sharp dip on each beat, ~92 bpm. */
const nodBeat = (t: number, s: number) => Math.exp(-(((t * 1.53 + s) % 1) * 7))

function idlePose(p: Pose, d: CharacterDriver, t: number) {
  const s = d.seed
  const breathe = Math.sin(t * 1.7 + s) * 0.02
  p.hips = [0, 0.08, 0.03]
  p.spine = [0.07 + breathe, 0.06 * Math.sin(t * 0.5 + s), 0.02]
  p.head = [0.04, 0.3 * Math.sin(t * 0.33 + s * 2), 0]
  const { kind, u } = idleKind(t, s)
  switch (kind) {
    case 'roll': {
      // weight on the left leg, right foot rolling the deck back and forth
      const f = Math.sin(t * 1.5 + s)
      p.shL = [0.05, 0, 0.16]; p.elL = [-0.25, 0, 0]
      p.shR = [-0.05, 0, -0.18]; p.elR = [-0.35, 0, 0]
      p.footR = [0.08 + 0.13 * f, DECK_TOP]
      p.deckPos = [-0.13, 0, 0.12 + 0.13 * f]; p.deckRot = [0, 0.04, 0]
      break
    }
    case 'nod': {
      // hands in the hoodie pocket, nodding to whatever's in the headphones
      const n = nodBeat(t, s)
      p.hipsY = -0.02 - 0.035 * n
      p.hips = [0, 0.1 * Math.sin(t * Math.PI * 0.765 + s), 0.03]
      p.spine = [0.06 + 0.06 * n, 0.05 * Math.sin(t * 0.4 + s), 0]
      p.head = [0.02 + 0.26 * n, 0.12 * Math.sin(t * 0.4 + s), 0.04 * Math.sin(t * Math.PI * 0.765)]
      p.ikL = ik('spine', [0.08, 0.06, 0.2], 1, POCKET); p.ikR = ik('spine', [-0.08, 0.06, 0.2], 1, POCKET)
      p.footL = [0.06, 0.03 * n]
      break
    }
    case 'lean': {
      // deck stood on its tail at the right hip, hand resting on the nose
      const rot: V3 = [-Math.PI / 2 + 0.1, Math.PI / 2, 0]
      p.deckRot = rot; p.deckPos = deckAt([-0.46, 0, 0.05], rot, TAIL_TIP)
      p.ikR = ik('deck', [0, 0.02, 0.4], 1, [0.6, -0.4, -0.5])
      p.ikL = ik('spine', [0.1, 0.05, 0.2], 1, POCKET)
      p.hipsY = -0.05
      p.hips = [0, 0.15, 0.09]
      p.spine = [0.04 + breathe, -0.12, 0.07]
      p.head = [0.03 + 0.1 * nodBeat(t * 0.5, s), 0.25 * Math.sin(t * 0.3 + s), -0.08]
      p.footL = [0.12, 0]; p.footR = [-0.05, 0]
      break
    }
    case 'flip': flipUp(p, u); break
  }
}

/** Stomp the tail, pop the deck up with a full roll, catch it by the nose, hold, drop it. */
function flipUp(p: Pose, u: number) {
  p.deckPos = [...FLIP_REST]; p.deckRot = [0, 0, 0]
  const ph = (u - 0.7) / 2.8
  if (ph <= 0 || ph >= 1) { p.footR = [0.02, 0]; return }
  p.snapDeck = ph > 0.19
  const tail: V3 = [FLIP_REST[0], TAIL_TIP[1], FLIP_REST[2] + TAIL_TIP[2]]
  const stompPivot: V3 = [tail[0], 0.02, tail[2] - 0.03]
  const stompRot: V3 = [-1.25, 0, 0]
  p.head = [0.28, -0.3, 0]
  p.shL = [-0.1, 0, 0.35]; p.elL = [-0.5, 0, 0]          // left arm out for balance
  if (ph < 0.3) {
    // step onto the tail, then stomp: nose pitches up around the tail
    const a = smooth(seg(ph, 0.04, 0.19)), b = smooth(seg(ph, 0.2, 0.3))
    p.footR = [lerp(0.02, -0.12, a), lerp(0, DECK_TOP + 0.03, a) + 0.07 * Math.sin(Math.PI * a) - (DECK_TOP + 0.01) * b]
    const pivot = lerpV(tail, stompPivot, b)
    const rot: V3 = [stompRot[0] * b, 0, 0]
    p.deckRot = rot; p.deckPos = deckAt(pivot, rot, TAIL_TIP)
    p.hipsY = -0.02 - 0.06 * b
    p.shR = [-0.3, 0, -0.3]; p.elR = [-0.6, 0, 0]
  } else if (ph < 0.8) {
    // pop: the deck flies up into the right hand doing a full roll, then gets held up
    const c = seg(ph, 0.3, 0.46), cs = smooth(c)
    const p0 = deckAt(stompPivot, stompRot, TAIL_TIP)
    const pos = lerpV(p0, CATCH, cs)
    pos[1] += 0.3 * Math.sin(Math.PI * c) + (ph > 0.46 ? 0.025 * Math.sin((ph - 0.46) * 18) : 0)
    p.deckPos = pos
    p.deckRot = [lerp(stompRot[0], -Math.PI / 2, cs), 0, cs >= 1 ? 0 : TAU * cs]
    const f = seg(ph, 0.3, 0.42)
    p.footR = [lerp(-0.12, 0.02, smooth(f)), 0.05 * Math.sin(Math.PI * f)]
    p.hipsY = -0.02 + 0.03 * Math.sin(Math.PI * c)
    p.ikR = ik('deck', NOSE_GRAB, ph > 0.4 ? 1 : 0, HOLD)
    if (ph > 0.46) { p.shL = [0.02, 0, 0.14]; p.elL = [-0.3, 0, 0]; p.head = [0.12, -0.25, -0.06] }
  } else {
    // let it drop back flat next to the right foot
    const e = seg(ph, 0.8, 1)
    p.deckPos = lerpV(CATCH, FLIP_REST, e * e)
    p.deckRot = [lerp(-Math.PI / 2, 0, smooth(e)), 0, 0]
    p.ikR = ik('deck', NOSE_GRAB, ph < 0.84 ? 1 : 0, HOLD)
    p.footR = [0.02, 0]
  }
}

// ---------------------------------------------------------------------------------------------
// states

function statePose(d: CharacterDriver, t: number): Pose {
  const p = base()
  const s = d.seed
  const breathe = Math.sin(t * 1.7 + s) * 0.02
  switch (d.state) {
    case 'idle': idlePose(p, d, t); break
    case 'thinking': {
      // hunched over the board, chin on fist; fidgets faster as the clock runs down
      const k = d.pressure, speed = 1 + 5 * k
      p.hipsY = -0.05 - 0.05 * k
      p.spine = [0.38 + 0.22 * k + breathe, 0.07 * Math.sin(t * speed * 0.7 + s), 0]
      p.head = [0.22 + 0.1 * k, 0.1 * Math.sin(t * speed * 0.6 + s), 0.05 * Math.sin(t * speed)]
      const drum = Math.max(0, Math.sin(t * (4 + 14 * k) + s)) * 0.02 * k
      p.ikR = ik('spine', [-0.03, 0.57 + drum, 0.2], 1, [0.2, -1, 0.5])
      p.ikL = ik('spine', [-0.06, 0.24, 0.21], 1, [0.6, -0.6, 0.3])
      p.footL = [0.04, 0]
      p.footR = [-0.06, Math.max(0, Math.sin(t * (3 + 10 * k) + s)) * 0.04 * (0.3 + k)]
      break
    }
    case 'moving': {
      // lean in; the hands themselves are driven by the grab choreography (../grab) in the rig
      p.hipsY = -0.08
      p.spine = [0.55, -0.1, 0]
      p.head = [0.35, 0, 0]
      p.shL = [-0.35, 0, 0.25]; p.elL = [-0.6, 0, 0]
      p.shR = [-1.2, 0, 0]; p.elR = [-0.15, 0, 0]
      p.footL = [0.08, 0]; p.footR = [-0.1, 0]
      break
    }
    case 'stalling': {
      // waiting on the partner: arms crossed, glaring at their board, foot tapping;
      // every few seconds checks the (imaginary) watch and taps it
      const yaw = Math.max(-1.3, Math.min(1.3, d.lookYaw))
      p.hips = [0, 0, -0.04]
      p.spine = [-0.06 + breathe, yaw * 0.3, 0]
      p.head = [0.12, yaw * 0.7, 0]
      p.ikL = ik('spine', [-0.13, 0.3, 0.2], 1, CROSSED); p.ikR = ik('spine', [0.13, 0.33, 0.22], 1, CROSSED)
      const w = ((t - d.since + s) % 4.2) / 4.2
      if (w > 0.62 && w < 0.92) {
        const tap = Math.abs(Math.sin(t * 15))
        p.ikL = ik('spine', [0.02, 0.38, 0.3], 1, WATCH)
        p.ikR = ik('spine', [0.03, 0.43 + 0.03 * tap, 0.31], 1, WATCH)
        p.head = [0.4, 0.05, 0]; p.spine = [0.05, 0, 0]
      }
      p.footL = [0.02, 0]
      p.footR = [0.05, Math.max(0, Math.sin(t * 7 + s)) * 0.04]
      const rot: V3 = [-Math.PI / 2 + 0.25, Math.PI / 2, 0]
      p.deckRot = rot; p.deckPos = deckAt([-0.36, 0, 0.1], rot, TAIL_TIP)
      break
    }
    case 'win': {
      // ride onto the deck, then this skater's own trick on a loop (tricks/, see trickPose.ts)
      if (d.trick) { winPose(p, d.trick, t - d.since); break }
      // fallback: hop + kickflip loop with both arms up
      const P = 0.95, ph = ((t - d.since) % P) / P
      const air = Math.sin(Math.PI * Math.min(1, ph / 0.8))
      const hop = air * 0.42
      p.hipsY = hop - 0.12 * (1 - air)
      p.spine = [-0.1, 0, 0]
      p.head = [-0.3, 0, 0]
      p.shL = [0, 0, 2.5 + 0.2 * Math.sin(t * 9)]; p.elL = [-0.3, 0, 0]
      p.shR = [0, 0, -2.5 - 0.2 * Math.sin(t * 9)]; p.elR = [-0.3, 0, 0]
      p.footL = [0.02, DECK_TOP + hop + 0.12 * air]
      p.footR = [-0.02, DECK_TOP + hop + 0.12 * air]
      const flip = TAU * Math.min(1, Math.max(0, (ph - 0.1) / 0.55))
      p.deckPos = [0, hop * 0.92, 0]; p.deckRot = [0, Math.PI / 2, flip >= TAU ? 0 : flip]
      p.snapDeck = true
      break
    }
    case 'loss': {
      // slumped, head down, arms dangling
      p.hipsY = -0.13
      p.spine = [0.62 + breathe, 0, 0.05]
      p.head = [0.5, 0.15, 0.1]
      p.shL = [0.5, 0, 0.06]; p.elL = [-0.12, 0, 0]
      p.shR = [0.5, 0, -0.06]; p.elR = [-0.12, 0, 0]
      p.footL = [0.1, 0]; p.footR = [0.06, 0]
      break
    }
  }
  return p
}

// ---------------------------------------------------------------------------------------------
// reactions (u = seconds since the reaction started)

function reactionPose(r: Reaction, p: Pose, u: number): Pose {
  switch (r) {
    case 'fistPump': {
      // raise the fist, then yank it down twice: "YES!"
      const y = u < 0.2 ? 0 : u > 0.95 ? 1 : 0.5 - 0.5 * Math.cos(TAU * (u - 0.2) * 2)
      p.hipsY = -0.02 - 0.08 * y
      p.spine = [0.1 + 0.15 * y, 0.1, 0]
      p.head = [0.05 + 0.22 * y, 0.1, 0]
      p.shR = lerpV([-2.6, 0, -0.35], [-0.15, 0, -0.25], y)
      p.elR = lerpV([-0.6, 0, 0], [-2.3, 0, 0], y)
      p.shL = [-0.3, 0, 0.3]; p.elL = [-1.2, 0, 0]
      p.ikL = NO_IK(); p.ikR = NO_IK()
      p.footL = [0.08, 0]; p.footR = [-0.06, 0]
      return p
    }
    case 'headInHands': {
      // "no no no": hands clamp the head, which shakes between them
      const decay = 1 - seg(u, 0.6, 1.9)
      p.hipsY = -0.1
      p.spine = [0.5, 0, 0]
      p.head = [0.45, 0.3 * Math.sin(u * 9) * decay, 0]
      p.ikL = ik('spine', [0.15, 0.66, 0.11], 1, [0.6, -0.5, 0.7]); p.ikR = ik('spine', [-0.15, 0.66, 0.11], 1, [0.6, -0.5, 0.7])
      p.footL = [0.06, 0]; p.footR = [-0.08, 0]
      return p
    }
    case 'impatientTap': {
      // jab at the watch, shake the head, stomp the foot
      p.spine = [0.0, -0.05, 0]
      p.head = [0.35, 0.25 * Math.sin(u * 7), 0]
      p.ikL = ik('spine', [0.0, 0.4, 0.3], 1, WATCH)
      p.ikR = ik('spine', [0.01, 0.44 + 0.04 * Math.abs(Math.sin(u * 16)), 0.31], 1, WATCH)
      p.footL = [0.02, 0]
      p.footR = [0.05, 0.07 * Math.max(0, Math.sin(u * 12))]
      return p
    }
    case 'celebrate': {
      // crouch, jump with a 360 and both fists up, land pumping
      const a = seg(u, 0.25, 0.95)
      const hop = 0.45 * Math.sin(Math.PI * a)
      p.hipsY = u < 0.25 ? -0.12 * smooth(u / 0.25) : hop - 0.1 * (a >= 1 ? 1 - seg(u, 0.95, 1.3) : 0)
      p.spinY = a > 0 && a < 1 ? TAU * smooth(a) : 0
      const pump = u > 1.0 ? 0.3 * Math.sin((u - 1) * 14) : 0
      p.spine = [-0.15, 0, 0]
      p.head = [-0.35, 0, 0]
      p.shL = [-0.3, 0, 2.6 + pump]; p.elL = [-0.6, 0, 0]
      p.shR = [-0.3, 0, -2.6 + pump]; p.elR = [-0.6, 0, 0]
      p.ikL = NO_IK(); p.ikR = NO_IK()
      const tuck = 0.15 * Math.sin(Math.PI * a)
      p.footL = [0.04, hop + tuck]; p.footR = [-0.04, hop + tuck]
      return p
    }
    case 'slump': {
      // drop to the knees, head down, arms hanging
      const k = smooth(seg(u, 0, 0.5))
      p.hipsY = lerp(-0.02, -0.42, k)
      p.footL = [-0.2 * k, 0]; p.footR = [-0.24 * k, 0]
      p.spine = [0.55 + 0.04 * Math.sin(u * 12) * k, 0, 0]
      p.head = [0.65, 0, 0.1]
      p.shL = [-0.45, 0, 0.1]; p.elL = [-0.1, 0, 0]
      p.shR = [-0.45, 0, -0.1]; p.elR = [-0.1, 0, 0]
      p.ikL = NO_IK(); p.ikR = NO_IK()
      return p
    }
    case 'talk': {
      // head bob + palm-up hand gesture
      p.spine = [0.1, 0.1 * Math.sin(u * 3), 0]
      p.head = [0.05 + 0.12 * Math.abs(Math.sin(u * 10)), 0.15 * Math.sin(u * 3), 0.06 * Math.sin(u * 5)]
      p.shR = [-0.6, 0, -0.25]; p.elR = [-1.3 + 0.35 * Math.sin(u * 9), 0.5 * Math.sin(u * 7), 0]
      p.ikR = NO_IK()
      return p
    }
  }
}

/** The target pose for this frame (state pose, plus any active reaction mixed in). */
export function poseFor(d: CharacterDriver, t: number): Pose {
  const p = statePose(d, t)
  if (!d.reaction) return p
  // the win trick owns the whole body: no overlays on top of it
  if (d.state === 'win') { d.reaction = null; return p }
  const u = t - d.reactionAt, dur = REACTION_DUR[d.reaction]
  if (u >= dur || u < 0) { d.reaction = null; return p }
  const w = smooth(u / 0.15) * (1 - smooth(seg(u, dur - 0.35, dur)))
  return mixPose(p, reactionPose(d.reaction, clonePose(p), u), w)
}

/** 2-bone leg IK in the sagittal plane. Returns [hip x-rotation, knee x-rotation]. */
export function legIK(hipY: number, footZ: number, footY: number, l1 = 0.45, l2 = 0.47): [number, number] {
  const y = Math.max(0.05, hipY - footY), z = footZ
  const d = Math.min(l1 + l2 - 1e-3, Math.max(0.1, Math.hypot(y, z)))
  const knee = Math.PI - Math.acos(Math.max(-1, Math.min(1, (l1 * l1 + l2 * l2 - d * d) / (2 * l1 * l2))))
  const phi = Math.atan2(z, y)
  const alpha = Math.acos(Math.max(-1, Math.min(1, (l1 * l1 + d * d - l2 * l2) / (2 * l1 * d))))
  return [-(phi + alpha), knee]
}

/** Hip abduction (z rotation) that puts the foot at sideways offset `footX` (rig-local); side = +1
 *  left leg / -1 right leg. null -> the default slight splay. */
export function legAbduction(hipY: number, footY: number, footX: number | null | undefined, side: 1 | -1, hipX: number): number {
  if (footX === null || footX === undefined) return side * 0.05
  return Math.max(-0.5, Math.min(0.5, Math.atan2(footX - side * hipX, Math.max(0.2, hipY - footY))))
}

/** Leg joint angles [hip x, knee x, ankle x] for an abducted leg (the tilted leg needs more length). */
export function legAngles(hipY: number, foot: [number, number], abd: number): [number, number, number] {
  const drop = Math.max(0.05, hipY - foot[1]) / Math.cos(abd)
  const [h, k] = legIK(hipY, foot[0], hipY - drop)
  return [h, k, -(h + k)]
}

