// Analytic two-bone arm IK (shoulder -> elbow -> wrist) for plain Object3D hierarchies, plus a
// "reach budget" for far targets: optional root slide, spine lean (CCD-style), and arm stretch.
//
// Technique credits (no code copied; maths re-derived and written for Object3D chains):
//  - Two-joint IK with the law of cosines + a pole/hint vector for the bend plane, as described in
//    Daniel Holden, "Simple Two Joint IK", https://theorangeduck.com/page/simple-two-joint
//    (article, no licence stated, so we take the idea only, not code), and as implemented in Godot's
//    TwoBoneIK3D (MIT, https://github.com/godotengine/godot).
//  - The spine-lean pass is a single cyclic-coordinate-descent style rotation per joint, the same
//    idea as three.js examples/jsm/animation/CCDIKSolver.js (MIT, (c) three.js authors,
//    https://github.com/mrdoob/three.js/blob/r169/examples/jsm/animation/CCDIKSolver.js). That
//    solver needs a SkinnedMesh + Skeleton, so it is not used directly here.
//
// Model: run the FK pose first (it writes joint rotations every frame), then call solveArm(). The
// solver works in world space and writes local quaternions, blended with the FK pose by `weight`.
// Nothing accumulates across frames as long as FK rewrites the joints each frame (it does) and the
// root offset object is owned by the IK (we set its position absolutely).
import { Matrix4, Object3D, Quaternion, Vector3 } from 'three'

export interface ArmChain {
  shoulder: Object3D
  elbow: Object3D            // child (direct or nested) of shoulder
  wrist: Object3D            // child of elbow; the hand hangs off this
  /** Local axis of the joints about which the elbow bends, i.e. normalize(cross(upperDir, foreDir))
   *  expressed in shoulder space at rest. For a rig whose arm hangs along -y and whose elbow bends
   *  forward (+z) via negative x rotation (our skaters), this is (-1, 0, 0). */
  bendAxis?: Vector3
  /** Wrist-local direction the fingers point (default -y: hand continues the forearm). */
  handAxis?: Vector3
  /** Wrist-local palm normal (default +z). */
  palmAxis?: Vector3
}

export interface ReachBudget {
  /** Joints (lowest first, e.g. [hips, spine, chest]) that may bend toward a far target. */
  spine?: Object3D[]
  /** Max extra rotation per spine joint (radians). */
  maxLeanPerJoint?: number
  /** Object whose position the IK may set (absolutely) to slide the whole body toward the target,
   *  e.g. an empty group wrapping the character. Horizontal only, in its parent's frame. */
  rootOffset?: Object3D
  maxRootShift?: number      // meters (world), default 0.25
  /** Max arm stretch factor (1 = rigid). Bone offsets (elbow.position, wrist.position) are scaled. */
  maxStretch?: number        // default 1.08
  /** Fraction of the full arm length the solver aims to use before leaning/sliding (keeps a soft elbow). */
  comfort?: number           // default 0.9
}

export interface ArmGoal {
  /** World point the effector should reach. */
  target: Vector3
  /** Optional effector under the wrist (e.g. the hand's grip socket). The wrist target is offset so
   *  that this object lands on `target`. Requires handDir (wrist orientation must be known). */
  effector?: Object3D
  /** World direction the elbow should point toward (pole / hint), e.g. outward + down + back. */
  pole: Vector3
  /** Desired world direction of the fingers; omit to leave the wrist on its FK pose. */
  handDir?: Vector3
  /** Desired world palm normal (orthogonalised against handDir). */
  palmDir?: Vector3
  /** Max angle between forearm and fingers (radians), default 1.3. */
  maxWristBend?: number
  /** 0 = pure FK, 1 = pure IK. */
  weight: number
}

export interface ArmResult {
  reached: boolean
  /** Distance left between effector and target after solving (world meters). */
  error: number
  stretch: number
  rootShift: number
  lean: number               // total extra spine rotation applied (radians)
}

/** Per-chain state: rest bone offsets (for stretch) captured on first use. */
interface ChainState {
  elbowRest: Vector3
  wristRest: Vector3
}
const states = new WeakMap<ArmChain, ChainState>()

// scratch
const _s = new Vector3(), _e = new Vector3(), _w = new Vector3(), _t = new Vector3(), _v = new Vector3()
const _a = new Vector3(), _b = new Vector3(), _c = new Vector3(), _n = new Vector3(), _p = new Vector3()
const _q = new Quaternion(), _q2 = new Quaternion(), _qp = new Quaternion(), _qfk = new Quaternion()
const _m = new Matrix4(), _m2 = new Matrix4()
const _sc = new Vector3()
const _off = new Vector3()

const clamp = (x: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, x))
const worldPos = (o: Object3D, out: Vector3) => out.setFromMatrixPosition(o.matrixWorld)
function worldQuat(o: Object3D, out: Quaternion) { o.matrixWorld.decompose(_v, out, _sc); return out }
function worldScale(o: Object3D) { o.matrixWorld.decompose(_v, _qp, _sc); return (_sc.x + _sc.y + _sc.z) / 3 }

function anyPerp(v: Vector3, out: Vector3) {
  out.set(v.y, -v.x, 0)
  if (out.lengthSq() < 1e-10) out.set(0, v.z, -v.y)
  return out
}

/**
 * Rotation that maps the orthonormal frame (u0, n0) onto (u1, n1). u = primary axis (matched
 * exactly), n = secondary (matched after Gram-Schmidt). Inputs need not be normalised.
 */
export function frameToFrame(u0: Vector3, n0: Vector3, u1: Vector3, n1: Vector3, out = new Quaternion()): Quaternion {
  const basis = (u: Vector3, n: Vector3, m: Matrix4) => {
    const x = _a.copy(u).normalize()
    const z = _b.copy(n).addScaledVector(x, -n.dot(x))
    if (z.lengthSq() < 1e-10) anyPerp(x, z)
    z.normalize()
    const y = _c.crossVectors(z, x)
    return m.makeBasis(x, y, z)
  }
  basis(u0, n0, _m)
  basis(u1, n1, _m2)
  // R * B0 = B1  ->  R = B1 * B0^T
  _m.transpose()
  _m2.multiply(_m)
  return out.setFromRotationMatrix(_m2)
}

/** Set `o`'s local quaternion so that its world quaternion becomes `qWorld`, blended with its current
 *  (FK) local rotation by `w`. Updates o's matrixWorld (and children). */
function setWorldQuat(o: Object3D, qWorld: Quaternion, w: number) {
  _qfk.copy(o.quaternion)
  if (o.parent) worldQuat(o.parent, _qp).invert()
  else _qp.identity()
  o.quaternion.copy(_qp.multiply(qWorld))
  if (w < 1) o.quaternion.copy(_qfk.slerp(o.quaternion, w))
  o.updateMatrixWorld(true)
}

function rotateAboutWorld(o: Object3D, axisW: Vector3, angle: number) {
  // world rotation q applied on top: newWorld = q * oldWorld ; local = parentInv * newWorld
  worldQuat(o, _q2)
  _q.setFromAxisAngle(axisW, angle).multiply(_q2)
  setWorldQuat(o, _q, 1)
}

/** Total rest length of the arm (world units), measured from the current matrices. */
export function armLength(chain: ArmChain): number {
  chain.shoulder.updateWorldMatrix(true, true)
  return worldPos(chain.shoulder, _s).distanceTo(worldPos(chain.elbow, _e)) + _e.distanceTo(worldPos(chain.wrist, _w))
}

/**
 * Solve one arm. Call after FK has written this frame's pose. Mutates spine joints (adds lean),
 * rootOffset.position, elbow/wrist positions (stretch) and shoulder/elbow/wrist rotations.
 */
export function solveArm(chain: ArmChain, goal: ArmGoal, budget: ReachBudget = {}): ArmResult {
  const w = clamp(goal.weight, 0, 1)
  const res: ArmResult = { reached: false, error: 0, stretch: 1, rootShift: 0, lean: 0 }
  let st = states.get(chain)
  if (!st) { st = { elbowRest: chain.elbow.position.clone(), wristRest: chain.wrist.position.clone() }; states.set(chain, st) }
  // reset the parts the IK owns absolutely (FK doesn't touch them)
  chain.elbow.position.copy(st.elbowRest)
  chain.wrist.position.copy(st.wristRest)
  if (budget.rootOffset) budget.rootOffset.position.set(0, 0, 0)
  const top = budget.rootOffset ?? budget.spine?.[0] ?? chain.shoulder
  top.updateWorldMatrix(true, true)
  if (w <= 0) return res

  const bendAxis = chain.bendAxis ?? _n.set(-1, 0, 0)
  const handAxis = chain.handAxis ?? new Vector3(0, -1, 0)
  const palmAxis = chain.palmAxis ?? new Vector3(0, 0, 1)
  const comfort = budget.comfort ?? 0.9
  const maxStretch = budget.maxStretch ?? 1.08

  // ---- desired wrist orientation (world) and the wrist target that puts the effector on target
  const qHand = new Quaternion()
  const haveHand = !!goal.handDir
  if (haveHand) {
    const pd = goal.palmDir ?? new Vector3(0, -1, 0)
    frameToFrame(handAxis, palmAxis, goal.handDir!, pd, qHand)
  }
  // wrist target = target - (effector offset rotated into the hand frame)
  const effLocal = new Vector3()
  if (goal.effector && haveHand) {
    chain.wrist.updateWorldMatrix(true, true)
    chain.wrist.worldToLocal(effLocal.copy(worldPos(goal.effector, _v))).multiplyScalar(worldScale(chain.wrist))
  }
  const wristTarget = new Vector3()
  const setWristTarget = (q: Quaternion) => wristTarget.copy(goal.target).sub(_off.copy(effLocal).applyQuaternion(q))
  setWristTarget(qHand)

  // ---- lengths
  worldPos(chain.shoulder, _s); worldPos(chain.elbow, _e); worldPos(chain.wrist, _w)
  const L1 = _s.distanceTo(_e), L2 = _e.distanceTo(_w)
  const L = L1 + L2
  const want = L * comfort

  // ---- 1. spine lean, 2. root slide for what the lean can't cover, then lean again with what's left
  const used = budget.spine ? budget.spine.map(() => 0) : []
  const maxLean = budget.maxLeanPerJoint ?? 0.5
  const leanPass = () => {
    if (!budget.spine?.length) return
    // CCD-style: one rotation per joint, top joint first so the chest does most of the bend
    for (let i = budget.spine.length - 1; i >= 0; i--) {
      const j = budget.spine[i]
      worldPos(chain.shoulder, _s)
      if (_s.distanceTo(wristTarget) <= want) break
      const P = worldPos(j, _p)
      const sp = _a.subVectors(_s, P), tp = _b.subVectors(wristTarget, P)
      const r = sp.length(), D = tp.length()
      if (r < 1e-5 || D < 1e-5) continue
      const theta0 = sp.angleTo(tp)
      // angle (at P) between shoulder and target that makes |shoulder - target| == want
      const theta1 = Math.acos(clamp((r * r + D * D - want * want) / (2 * r * D), -1, 1))
      const delta = clamp(theta0 - theta1, 0, maxLean - used[i]) * w
      if (delta < 1e-5) continue
      const axis = _c.crossVectors(sp, tp)
      if (axis.lengthSq() < 1e-12) continue
      rotateAboutWorld(j, axis.normalize(), delta)
      used[i] += delta
      res.lean += delta
    }
  }
  leanPass()

  if (budget.rootOffset && (budget.maxRootShift ?? 0.25) > 0) {
    const ro = budget.rootOffset
    worldPos(chain.shoulder, _s)
    const d = _v.subVectors(wristTarget, _s)
    const excess = d.length() - want
    const horiz = _a.set(d.x, 0, d.z)
    const hl = horiz.length()
    if (excess > 0 && hl > 1e-6) {
      const shift = Math.min(budget.maxRootShift ?? 0.25, excess) * w
      horiz.multiplyScalar(shift / hl)
      if (ro.parent) {
        // world displacement -> ro.parent space
        ro.parent.updateWorldMatrix(true, false)
        const p0 = worldPos(ro, _b)
        const p1 = _c.copy(p0).add(horiz)
        ro.parent.worldToLocal(p0); ro.parent.worldToLocal(p1)
        ro.position.add(p1.sub(p0))
      } else ro.position.add(horiz)
      ro.updateMatrixWorld(true)
      res.rootShift = shift
      leanPass()
    }
  }

  // ---- 3-5. stretch, two-bone solve, wrist. If the wrist had to be clamped, the effector moved:
  // re-aim the wrist target with the clamped hand frame and solve again (converges in 2-3 passes).
  const qUsed = new Quaternion().copy(qHand)
  const fkS = chain.shoulder.quaternion.clone(), fkE = chain.elbow.quaternion.clone(), fkW = chain.wrist.quaternion.clone()
  for (let iter = 0; iter < 3; iter++) {
    // restore FK so the weight blend never compounds across passes
    chain.shoulder.quaternion.copy(fkS); chain.elbow.quaternion.copy(fkE); chain.wrist.quaternion.copy(fkW)
    // ---- 3. stretch if still out of reach
    worldPos(chain.shoulder, _s)
    let dist = _s.distanceTo(wristTarget)
    let stretch = 1
    if (dist > L * 0.999) {
      stretch = clamp(dist / (L * 0.999), 1, maxStretch)
      stretch = 1 + (stretch - 1) * w
    }
    chain.elbow.position.copy(st.elbowRest).multiplyScalar(stretch)
    chain.wrist.position.copy(st.wristRest).multiplyScalar(stretch)
    chain.shoulder.updateMatrixWorld(true)
    res.stretch = stretch
    const l1 = L1 * stretch, l2 = L2 * stretch

    // ---- 4. analytic two-bone solve
    const dir = _t.subVectors(wristTarget, _s)
    dist = clamp(dir.length(), Math.abs(l1 - l2) + 1e-4, l1 + l2 - 1e-5)
    dir.normalize()
    // perpendicular toward the pole
    const perp = _p.copy(goal.pole).addScaledVector(dir, -goal.pole.dot(dir))
    if (perp.lengthSq() < 1e-8) anyPerp(dir, perp)
    perp.normalize()
    const a = (l1 * l1 - l2 * l2 + dist * dist) / (2 * dist)         // along dir to the elbow foot
    const h = Math.sqrt(Math.max(0, l1 * l1 - a * a))                 // elbow offset from the line
    const elbowW = new Vector3().copy(_s).addScaledVector(dir, a).addScaledVector(perp, h)
    const wristW = new Vector3().copy(_s).addScaledVector(dir, dist)
    const upper = new Vector3().subVectors(elbowW, _s)
    const fore = new Vector3().subVectors(wristW, elbowW)
    // bend-plane normal: cross(upper, fore); if the arm is straight, use cross(dir, perp)
    const bendW = new Vector3().crossVectors(upper, fore)
    if (bendW.lengthSq() < 1e-10) bendW.crossVectors(dir, perp)
    bendW.normalize()

    // shoulder: map (rest upper-arm dir, bend axis) in shoulder-local space to world
    const upperLocal = new Vector3().copy(chain.elbow.position)
    const qS = frameToFrame(upperLocal, bendAxis, upper, bendW)
    setWorldQuat(chain.shoulder, qS, w)
    // elbow: its local forearm dir = wrist.position; same bend axis in its own frame
    const foreLocal = new Vector3().copy(chain.wrist.position)
    const qE = frameToFrame(foreLocal, bendAxis, fore, bendW)
    setWorldQuat(chain.elbow, qE, w)

    // ---- 5. wrist orientation (clamped against the forearm so it never folds back)
    if (haveHand) {
      const foreW = _a.subVectors(worldPos(chain.wrist, _w), worldPos(chain.elbow, _e)).normalize()
      const hd = _b.copy(goal.handDir!).normalize()
      const maxBend = goal.maxWristBend ?? 1.3
      const ang = foreW.angleTo(hd)
      let qW = qHand
      if (ang > maxBend) {
        // rotate the desired hand frame back toward the forearm until the bend is maxBend
        const axis = _c.crossVectors(hd, foreW)
        if (axis.lengthSq() > 1e-12) {
          axis.normalize()
          qW = _q2.setFromAxisAngle(axis, ang - maxBend).multiply(qHand)
        }
      }
      setWorldQuat(chain.wrist, qW, w)
      if (!effLocal.lengthSq() || qW.angleTo(qUsed) < 1e-3) break
      qUsed.copy(qW)
      setWristTarget(qUsed)
    } else break
  }

  // ---- result
  chain.shoulder.updateMatrixWorld(true)
  const eff = goal.effector && haveHand ? worldPos(goal.effector, _v) : worldPos(chain.wrist, _v)
  res.error = eff.distanceTo(goal.target)
  res.reached = res.error < 0.01
  return res
}

/** Convenience pole: in the character root's frame, `side` = +1 left arm / -1 right arm (rig-local +x
 *  is the character's left). Elbow points out, down and a bit back. */
export function armPole(root: Object3D, side: 1 | -1, out = new Vector3()): Vector3 {
  root.updateWorldMatrix(true, false)
  worldQuat(root, _qp)
  return out.set(side * 0.8, -0.55, -0.35).applyQuaternion(_qp).normalize()
}
