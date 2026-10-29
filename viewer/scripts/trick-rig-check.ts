// Headless check: every seat's win trick sampled through the rig's mapping (characters/trickPose.ts
// + poses.ts leg maths), over the mount and two full loops. Run from viewer/:
//
//   npx esbuild scripts/trick-rig-check.ts --bundle --platform=node --format=esm --outfile=node_modules/.tmp/trick-rig-check.mjs \
//     && node node_modules/.tmp/trick-rig-check.mjs
//
// Fails on any NaN/Infinity, on a deck that isn't back at rest between loops, or on feet that sink
// through the grip while the board is wheels-down under them.
import { Euler, Vector3 } from 'three'
import { SEATS } from '../src/events/types'
import { poseFor, legAbduction, legAngles, POSE_KEYS, type Pose } from '../src/views/three/characters/poses'
import { MOUNT_S, PAUSE_S, winPose } from '../src/views/three/characters/trickPose'
import { winTrickOf } from '../src/views/three/tricks'
import type { CharacterDriver } from '../src/views/three/characters/rig'

let failures = 0
const fail = (m: string) => { failures++; if (failures < 20) console.error('FAIL', m) }
const HIP_Y = 0.92, HX = 0.12

function finitePose(p: Pose): boolean {
  const nums: number[] = [p.hipsY, p.spinY, ...p.footL, ...p.footR, ...(p.footX ?? [])]
  for (const k of POSE_KEYS) nums.push(...p[k])
  return nums.every(Number.isFinite)
}

for (const seat of SEATS) {
  const trick = winTrickOf(seat)
  const period = trick.durationS + PAUSE_S
  const end = MOUNT_S + 2 * period
  let n = 0, minClear = Infinity, maxLift = 0
  const d: CharacterDriver = {
    state: 'win', since: 0, pressure: 0, lookYaw: 0, reach: null, seed: 1, reaction: 'celebrate', reactionAt: 0,
    pending: null, pendingAt: 0, trick,
  }
  for (let u = 0; u <= end; u += 1 / 240) {
    const p = poseFor(d, u)                     // full path: state pose + (suppressed) reaction
    n++
    if (!finitePose(p)) { fail(`${seat} ${trick.name}: non-finite pose at u=${u.toFixed(3)}`); continue }
    const hipY = HIP_Y + p.hipsY
    for (const [foot, fx, side] of [[p.footL, p.footX?.[0], 1], [p.footR, p.footX?.[1], -1]] as const) {
      const a = legAbduction(hipY, foot[1], fx, side, HX)
      const ang = legAngles(hipY, [foot[0], foot[1]], a)
      if (![a, ...ang].every(Number.isFinite)) fail(`${seat}: non-finite leg at u=${u.toFixed(3)}`)
    }
    maxLift = Math.max(maxLift, p.hipsY)
    // standing / pausing on the deck: feet on the grip (0.11 above the deck origin when it's flat)
    const r = winPose(p, trick, u)
    if (r.phase !== 'mount') {
      const e = new Euler(p.deckRot[0], p.deckRot[1], p.deckRot[2], 'YXZ')
      const up = new Vector3(0, 1, 0).applyEuler(e)
      if (up.y > 0.95 && Math.abs(p.deckPos[1]) < 0.01) {
        minClear = Math.min(minClear, p.footL[1] - 0.11, p.footR[1] - 0.11)
      }
      if (r.phase === 'pause') {
        const flat = Math.abs(p.deckPos[0]) + Math.abs(p.deckPos[1]) + Math.abs(p.deckPos[2]) < 1e-3 && up.y > 0.9999
        if (!flat) fail(`${seat}: deck not at rest during the pause (u=${u.toFixed(2)})`)
        if (Math.abs(p.footL[1] - 0.11) > 0.005 || Math.abs(p.footR[1] - 0.11) > 0.005) fail(`${seat}: feet off the grip while paused`)
      }
    }
  }
  if (minClear < -0.02) fail(`${seat}: feet sink ${(-minClear * 100).toFixed(1)} cm into the grip`)
  console.log(`ok   ${seat.padEnd(8)} ${trick.name.padEnd(17)} ${n} samples, peak hip lift ${maxLift.toFixed(2)} m, min sole-grip ${(minClear * 100).toFixed(1)} cm`)
}

if (failures) { console.error(`\n${failures} failure(s)`); process.exit(1) }
console.log('\nall trick-rig checks passed')
