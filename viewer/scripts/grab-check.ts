// Headless check for src/views/three/grab (IK reach coverage, pole sanity, FK blend, hand, timeline).
// Run from viewer/:
//
//   npx esbuild scripts/grab-check.ts --bundle --platform=node --format=esm --outfile=node_modules/.tmp/grab-check.mjs \
//     && node node_modules/.tmp/grab-check.mjs
//
// Exits non-zero on failure. Uses a mock arm rig with the same proportions as the skaters
// (ProceduralCharacter / skaterParts): hips 0.92 m, spine pivot +0.06, shoulders +0.46, upper arm 0.29,
// elbow -> wrist 0.262.
import { Object3D, Quaternion, Vector3, MeshBasicMaterial } from 'three'
import { SEATS, type BoardId, type Seat } from '../src/events/types'
import { reserveSlot, seatColor, seatPose, squareWorld } from '../src/views/three/layout'
import { armPole, solveArm, type ArmChain, type ReachBudget } from '../src/views/three/grab/ik'
import { buildHand, handMeshes, handTriangles } from '../src/views/three/grab/hand'
import { grabBus, grabHeight, PHASE_ORDER, defaultPath, type GrabPhase } from '../src/views/three/grab/grabChoreo'

let failures = 0
const fail = (msg: string) => { failures++; console.error('FAIL', msg) }
const ok = (msg: string) => console.log('ok  ', msg)

function mockRig(seat: Seat, height = 1) {
  const { pos, rotY } = seatPose(seat)
  const seatG = new Object3D(); seatG.position.copy(pos); seatG.rotation.y = rotY
  const reach = new Object3D(); seatG.add(reach)               // IK-owned root offset
  const body = new Object3D(); body.scale.setScalar(height); reach.add(body)
  const hips = new Object3D(); hips.position.set(0, 0.92 - 0.08, 0); body.add(hips)
  const spine = new Object3D(); spine.position.set(0, 0.06, 0); hips.add(spine)
  const shoulder = new Object3D(); shoulder.position.set(-0.28, 0.46, 0); spine.add(shoulder)
  const elbow = new Object3D(); elbow.position.set(0, -0.29, 0); shoulder.add(elbow)
  const wrist = new Object3D(); wrist.position.set(0, -0.262, 0); elbow.add(wrist)
  const mat = new MeshBasicMaterial()
  const hand = buildHand({ side: -1, skin: '#c58c63', glove: '#ff8a1f', material: mat })
  wrist.add(hand.root)
  hand.setGrip(0.72)
  const fk = () => {   // the 'moving' FK pose, rewritten every frame like the real rig
    hips.rotation.set(0, 0, 0)
    spine.rotation.set(0.55, -0.1, 0)
    shoulder.rotation.set(-1.2, 0, 0); elbow.rotation.set(-0.15, 0, 0); wrist.rotation.set(0, 0, 0)
  }
  fk()
  seatG.updateMatrixWorld(true)
  const chain: ArmChain = { shoulder, elbow, wrist }
  const budget: ReachBudget = { spine: [hips, spine], maxLeanPerJoint: 0.6, rootOffset: reach, maxRootShift: 0.15, maxStretch: 1.08 }
  return { seatG, reach, hips, spine, shoulder, elbow, wrist, hand, chain, budget, fk }
}

const FILES = 'abcdefgh'
const wp = (o: Object3D) => new Vector3().setFromMatrixPosition(o.matrixWorld)

// ---------------------------------------------------------------- 1. IK coverage
{
  let n = 0, worst = 0, worstAt = '', maxShift = 0, maxLean = 0, maxStretch = 1, poleBad = 0, hyper = 0
  for (const seat of SEATS) for (const height of [0.92, 1.0, 1.08]) {
    const r = mockRig(seat, height)
    const board = seat[0] as BoardId
    const targets: [string, Vector3][] = []
    for (const f of FILES) for (let rank = 1; rank <= 8; rank++) for (const L of ['p', 'k']) {
      const v = squareWorld(board, f + rank); v.y += grabHeight(L); targets.push([`${f}${rank}/${L}`, v])
    }
    for (const t of 'PNBRQ') for (let k = 0; k < 6; k++) {
      const v = reserveSlot(board, seatColor(seat), t, k); v.y += grabHeight(t, true); targets.push([`res ${t}${k}`, v])
    }
    for (const [name, target] of targets) {
      r.fk(); r.seatG.updateMatrixWorld(true)
      const pole = armPole(r.seatG, -1)
      const dir = new Vector3(target.x - r.seatG.position.x, 0, target.z - r.seatG.position.z).normalize()
      const handDir = new Vector3(dir.x * Math.cos(0.95), -Math.sin(0.95), dir.z * Math.cos(0.95))
      const palmDir = new Vector3(0, -1, 0).addScaledVector(handDir, handDir.y).normalize()
      const res = solveArm(r.chain, { target, effector: r.hand.socket, pole, handDir, palmDir, weight: 1 }, r.budget)
      n++
      if (res.error > worst) { worst = res.error; worstAt = `${seat} h=${height} ${name}` }
      maxShift = Math.max(maxShift, res.rootShift); maxLean = Math.max(maxLean, res.lean); maxStretch = Math.max(maxStretch, res.stretch)
      // pole sanity: elbow sits on the pole side of the shoulder->wrist line
      const S = wp(r.shoulder), E = wp(r.elbow), W = wp(r.wrist)
      const line = W.clone().sub(S).normalize()
      const off = E.clone().sub(S); off.addScaledVector(line, -off.dot(line))
      if (off.length() > 0.01 && off.normalize().dot(pole) < 0.2) poleBad++
      // elbow bends the anatomical way: forearm rotated about the bend axis by a positive angle (no hyperextension)
      const up = E.clone().sub(S), fo = W.clone().sub(E)
      const bendAxisWorld = new Vector3(-1, 0, 0).applyQuaternion(r.shoulder.getWorldQuaternion(new Quaternion()))
      if (new Vector3().crossVectors(up, fo).dot(bendAxisWorld) < -1e-6) hyper++
    }
  }
  console.log(`     IK: ${n} targets, worst error ${(worst * 1000).toFixed(2)} mm at ${worstAt}`)
  console.log(`     max root shift ${maxShift.toFixed(3)} m, max lean ${maxLean.toFixed(2)} rad, max stretch ${maxStretch.toFixed(3)}`)
  if (worst > 0.01) fail(`IK misses a target by ${(worst * 1000).toFixed(1)} mm (${worstAt})`); else ok('IK reaches every square + reserve slot within 1 cm (4 seats x 3 heights)')
  if (poleBad) fail(`${poleBad} solves put the elbow away from the pole`); else ok('elbow always on the pole side')
  if (hyper) fail(`${hyper} solves hyperextend the elbow`); else ok('no elbow hyperextension')
}

// ---------------------------------------------------------------- 2. weight blending / no accumulation
{
  const r = mockRig('B_white')
  const target = squareWorld('B', 'e5'); target.y += 0.07
  const pole = armPole(r.seatG, -1)
  r.fk(); r.seatG.updateMatrixWorld(true)
  const fkQ = r.shoulder.quaternion.clone(), fkW = wp(r.wrist)
  solveArm(r.chain, { target, pole, weight: 0 }, r.budget)
  if (r.shoulder.quaternion.angleTo(fkQ) > 1e-6 || wp(r.wrist).distanceTo(fkW) > 1e-6) fail('weight 0 changed the pose'); else ok('weight 0 = pure FK')
  r.fk(); r.seatG.updateMatrixWorld(true)
  solveArm(r.chain, { target, pole, weight: 1 }, r.budget)
  const ikQ = r.shoulder.quaternion.clone()
  r.fk(); r.seatG.updateMatrixWorld(true)
  solveArm(r.chain, { target, pole, weight: 0.5 }, r.budget)
  const a = r.shoulder.quaternion.angleTo(fkQ), b = r.shoulder.quaternion.angleTo(ikQ)
  if (!(a > 0.01 && b > 0.01)) fail(`weight 0.5 not between FK and IK (${a.toFixed(3)}, ${b.toFixed(3)})`); else ok('weight 0.5 blends between FK and IK')
  // repeated frames must be stable (nothing accumulates)
  const errs: number[] = []
  for (let i = 0; i < 30; i++) { r.fk(); r.seatG.updateMatrixWorld(true); errs.push(solveArm(r.chain, { target, pole, weight: 1 }, r.budget).error) }
  if (Math.max(...errs) - Math.min(...errs) > 1e-6) fail('IK result drifts across frames'); else ok('stable across 30 frames (no accumulation)')
}

// ---------------------------------------------------------------- 3. hand
{
  const mat = new MeshBasicMaterial()
  const h = buildHand({ side: -1, skin: '#c58c63', material: mat })
  const tris = handTriangles(h), calls = handMeshes(h)
  if (tris < 100 || tris > 200 || calls > 5) fail(`hand has ${tris} tris / ${calls} meshes`); else ok(`hand: ${tris} triangles, ${calls} meshes (draw calls)`)
  const tipDist = (g: number) => {
    h.setGrip(g); h.root.updateMatrixWorld(true)
    const s = wp(h.socket)
    let d = 0, n = 0
    h.root.traverse(o => { if (o.children.length === 1 && (o.children[0] as { isMesh?: boolean }).isMesh && o.parent && o.parent !== h.root && o.position.y < -0.02) {
      // distal segment tip
      const tip = new Vector3(0, -0.025, 0).applyMatrix4(o.matrixWorld); d += tip.distanceTo(s); n++ } })
    return d / Math.max(1, n)
  }
  const open = tipDist(0), held = tipDist(0.72)
  if (!(held < open && held < 0.05)) fail(`fingertips don't close around the socket (open ${open.toFixed(3)}, held ${held.toFixed(3)})`)
  else ok(`fingertips close on the socket: ${(open * 100).toFixed(1)} cm open -> ${(held * 100).toFixed(1)} cm gripped`)
}

// ---------------------------------------------------------------- 4. timeline
function runTimeline(label: string, lead: number, dur: number, capture: boolean) {
  grabBus.reset()
  const seat: Seat = 'A_white'
  const from = squareWorld('A', 'e2'), to = squareWorld('A', 'e4')
  const start = lead, path = defaultPath(from, to, start, dur, 0.1)
  grabBus.requestMove({ seat, letter: 'P', from, to, start, dur, now: 0, path, capture: capture ? squareWorld('A', 'd5') : null, captureLetter: 'p' })
  const seen: GrabPhase[] = []
  let order = true, holdOk = true, trackErr = 0, capOk = true
  const p = new Vector3()
  for (let t = -0.05; t < start + dur + 1; t += 0.005) {
    const s = grabBus.sample(seat, t)
    if (seen[seen.length - 1] !== s.phase) {
      if (seen.length && PHASE_ORDER.indexOf(s.phase) < PHASE_ORDER.indexOf(seen[seen.length - 1]) && s.phase !== 'idle') order = false
      seen.push(s.phase)
    }
    if ((s.phase === 'carry' || s.phase === 'place') && !s.holding) holdOk = false
    if ((s.phase === 'reach' || s.phase === 'release' || s.phase === 'return') && s.holding) holdOk = false
    if (s.phase === 'carry' || s.phase === 'place') {
      path(t, p); const u = Math.min(1, (t - start) / dur)
      trackErr = Math.max(trackErr, Math.abs(s.target.y - p.y - grabHeight('P')) * (u > 0.3 ? 1 : 0) + Math.hypot(s.target.x - p.x, s.target.z - p.z))
    }
    if (capture) {
      const off = grabBus.sampleOff(seat, t), cp = grabBus.capturedPos(seat, t)
      if (off.holding && cp && off.target.distanceTo(cp) - grabHeight('p') > 1e-6) capOk = false
    }
  }
  const expected = lead > 0 ? ['idle', 'reach', 'grip', 'carry', 'place', 'release', 'return', 'idle'] : ['idle', 'carry', 'place', 'release', 'return', 'idle']
  const got = seen.join('>')
  if (!order || got !== expected.join('>')) fail(`${label}: phases ${got}`)
  else ok(`${label}: phases ${got}`)
  if (!holdOk) fail(`${label}: holding flag wrong for some phase`)
  if (trackErr > 1e-6) fail(`${label}: hand target strays from the piece path by ${trackErr}`)
  else ok(`${label}: hand target follows the piece path exactly while holding`)
  if (capture) { if (!capOk) fail(`${label}: captured piece not in the off hand`); else ok(`${label}: captured piece rides in the off hand`) }
}
runTimeline('normal move', grabBus.reachLead(3), 0.5, false)
runTimeline('capture', grabBus.reachLead(1), 0.45, true)
runTimeline('fast replay (no lead)', 0, 0.18, false)

// ---------------------------------------------------------------- 5. end to end: IK + timeline, socket on the piece
{
  grabBus.reset()
  const seat: Seat = 'B_black'
  const r = mockRig(seat)
  const from = squareWorld('B', 'g8'), to = squareWorld('B', 'f6')
  const lead = grabBus.reachLead(2), dur = 0.5
  const path = defaultPath(from, to, lead, dur, 0.16)
  grabBus.requestMove({ seat, letter: 'n', from, to, start: lead, dur, now: 0, path })
  let worst = 0
  const p = new Vector3()
  for (let t = 0; t < lead + dur + 0.6; t += 1 / 60) {
    const s = grabBus.sample(seat, t)
    r.fk(); r.seatG.updateMatrixWorld(true)
    r.hand.setPose({ grip: s.grip, spread: s.spread, pinch: s.pinch })
    if (!s.active) continue
    solveArm(r.chain, { target: s.target, effector: r.hand.socket, pole: armPole(r.seatG, -1), handDir: s.handDir, palmDir: s.palmDir, weight: s.weight }, r.budget)
    if (s.holding && s.phase !== 'grip') {
      path(t, p); p.y += grabHeight('n')
      worst = Math.max(worst, wp(r.hand.socket).distanceTo(p))
    }
  }
  if (worst > 0.012) fail(`end-to-end: socket ${(worst * 1000).toFixed(1)} mm off the carried knight`)
  else ok(`end-to-end: grip socket stays within ${(worst * 1000).toFixed(2)} mm of the carried piece`)
}

if (failures) { console.error(`\n${failures} failure(s)`); process.exit(1) }
console.log('\nall grab checks passed')
