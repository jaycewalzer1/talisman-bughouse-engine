// Code-driven poses. Each state returns target joint rotations (radians, Euler XYZ) for the
// current time; ProceduralCharacter blends toward them every frame (damped lerp).
// Rig-local axes: +z = toward the board (forward), +x = the character's LEFT, +y = up.
// Arms hang along -y: rotation x < 0 swings forward/up, z > 0 swings the LEFT arm outward.
// Legs are driven by foot targets (simple 2-bone IK), not by joint angles.
import type { CharacterDriver } from './rig'

export type V3 = [number, number, number]

export interface Pose {
  hipsY: number            // hip height offset (crouch < 0 < jump)
  hips: V3
  spine: V3                // x > 0 = lean forward
  head: V3
  shL: V3; elL: V3
  shR: V3; elR: V3
  footL: [number, number]  // foot target [forward z, height y] for the left foot
  footR: [number, number]
  deckPos: V3
  deckRot: V3
}

export const POSE_KEYS = ['hips', 'spine', 'head', 'shL', 'elL', 'shR', 'elR', 'deckPos', 'deckRot'] as const

const DECK_TOP = 0.095    // standing on a deck: sole height
const DECK_ASIDE: V3 = [-0.38, 0.0, 0.12]

function base(): Pose {
  return {
    hipsY: -0.02, hips: [0, 0, 0], spine: [0.06, 0, 0], head: [0, 0, 0],
    shL: [0.02, 0, 0.14], elL: [-0.3, 0, 0], shR: [0.02, 0, -0.14], elR: [-0.3, 0, 0],
    footL: [0, 0], footR: [0, 0], deckPos: [...DECK_ASIDE], deckRot: [0, 0.3, 0],
  }
}

export function poseFor(d: CharacterDriver, t: number): Pose {
  const p = base()
  const s = d.seed
  const breathe = Math.sin(t * 1.7 + s) * 0.02
  switch (d.state) {
    case 'idle': {
      // weight on the left leg, right foot rolling the deck back and forth, looking around
      const f = Math.sin(t * 1.5 + s)
      p.hips = [0, 0.1, 0.05]
      p.spine = [0.08 + breathe, 0.1 * Math.sin(t * 0.5 + s), 0.03]
      p.head = [0.05, 0.35 * Math.sin(t * 0.33 + s * 2), 0]
      p.shL = [0.05, 0, 0.16]; p.elL = [-0.25, 0, 0]
      p.shR = [-0.05, 0, -0.18]; p.elR = [-0.35, 0, 0]
      p.footL = [0, 0]
      p.footR = [0.12 + 0.13 * f, DECK_TOP]
      p.deckPos = [-0.12, 0, 0.12 + 0.13 * f]
      p.deckRot = [0, 0.05, 0]
      break
    }
    case 'thinking': {
      // hunched over the board, hand on chin; fidgets faster as the clock runs down
      const k = d.pressure, speed = 1 + 5 * k
      p.hipsY = -0.05 - 0.05 * k
      p.spine = [0.38 + 0.22 * k + breathe, 0.07 * Math.sin(t * speed * 0.7 + s), 0]
      p.head = [0.22 + 0.1 * k, 0.1 * Math.sin(t * speed * 0.6 + s), 0.05 * Math.sin(t * speed)]
      p.shR = [-1.15, 0.5, -0.05]; p.elR = [-2.05, 0, 0]
      p.shL = [-0.65, 0, 0.12]; p.elL = [-0.5, 0, 0]
      p.footL = [0.04, 0]
      p.footR = [-0.06, Math.max(0, Math.sin(t * (3 + 10 * k) + s)) * 0.04 * (0.3 + k)]
      break
    }
    case 'moving': {
      // lean in and reach (the right arm is aimed at the piece by the rig itself)
      p.hipsY = -0.08
      p.spine = [0.55, -0.1, 0]
      p.head = [0.35, 0, 0]
      p.shL = [-0.35, 0, 0.25]; p.elL = [-0.6, 0, 0]
      p.shR = [-1.2, 0, 0]; p.elR = [-0.15, 0, 0]
      p.footL = [0.08, 0]; p.footR = [-0.1, 0]
      break
    }
    case 'stalling': {
      // arms crossed, staring at the partner's board, tapping a foot
      const yaw = Math.max(-1.3, Math.min(1.3, d.lookYaw))
      p.hips = [0, 0, -0.04]
      p.spine = [-0.06 + breathe, yaw * 0.3, 0]
      p.head = [0.12, yaw * 0.7, 0]
      p.shL = [-0.45, -1.2, 0.1]; p.elL = [-1.8, 0, 0]
      p.shR = [-0.5, 1.2, -0.1]; p.elR = [-2.0, 0, 0]
      p.footL = [0.02, 0]
      p.footR = [0.05, Math.max(0, Math.sin(t * 6 + s)) * 0.035]
      p.deckPos = [-0.36, 0.41, 0.0]; p.deckRot = [-Math.PI / 2 + 0.18, 0.2, 0]
      break
    }
    case 'win': {
      // hop + kickflip loop with both arms up
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
      const flip = Math.PI * 2 * Math.min(1, Math.max(0, (ph - 0.1) / 0.55))
      p.deckPos = [0, hop * 0.92, 0]; p.deckRot = [0, Math.PI / 2, flip]
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

/** 2-bone leg IK in the sagittal plane. Returns [hip x-rotation, knee x-rotation]. */
export function legIK(hipY: number, footZ: number, footY: number, l1 = 0.45, l2 = 0.47): [number, number] {
  const y = Math.max(0.05, hipY - footY), z = footZ
  const d = Math.min(l1 + l2 - 1e-3, Math.max(0.1, Math.hypot(y, z)))
  const knee = Math.PI - Math.acos(Math.max(-1, Math.min(1, (l1 * l1 + l2 * l2 - d * d) / (2 * l1 * l2))))
  const phi = Math.atan2(z, y)
  const alpha = Math.acos(Math.max(-1, Math.min(1, (l1 * l1 + d * d - l2 * l2) / (2 * l1 * d))))
  return [-(phi + alpha), knee]
}
