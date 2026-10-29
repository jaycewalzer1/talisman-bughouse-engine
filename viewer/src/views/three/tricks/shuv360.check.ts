// Headless sanity check for the 360 shuv trick. Not part of the app bundle; run with esbuild + node:
//   node -e "require('esbuild').buildSync({entryPoints:['src/views/three/tricks/shuv360.check.ts'],
//     bundle:true,platform:'node',outfile:'/private/tmp/claude-501/shuv360.check.cjs'})"
//   node /private/tmp/claude-501/shuv360.check.cjs
// Throws (non-zero exit) on the first failed assertion summary.

import { STANDING, type TrickPose } from './types'
import { shuv360 } from './shuv360'

const TAU = Math.PI * 2
const failures: string[] = []
const check = (ok: boolean, msg: string) => {
  if (!ok) failures.push(msg)
}
const near = (a: number, b: number, eps = 1e-6) => Math.abs(a - b) <= eps
const angNear = (a: number, b: number, eps = 1e-6) => {
  const d = (((a - b) % TAU) + TAU) % TAU
  return d <= eps || TAU - d <= eps
}

function flat(p: TrickPose): number[] {
  return [
    p.rider.y, p.rider.rotY, ...p.deck.pos, p.deck.rot.roll, p.deck.rot.yaw, p.deck.rot.pitch,
    p.body.crouch, p.body.tuck, p.body.armsOut, p.body.frontFootFlick, p.body.backFootScoop, p.body.lean,
  ]
}
const LABELS = [
  'rider.y', 'rider.rotY', 'deck.x', 'deck.y', 'deck.z', 'roll', 'yaw', 'pitch',
  'crouch', 'tuck', 'armsOut', 'frontFootFlick', 'backFootScoop', 'lean',
]
const ANGLE_IDX = new Set([1, 5, 6, 7])

function matchesStanding(p: TrickPose, label: string) {
  const a = flat(p)
  const b = flat(STANDING)
  a.forEach((v, i) => {
    const ok = ANGLE_IDX.has(i) ? angNear(v, b[i]) : near(v, b[i])
    check(ok, `${label}: ${LABELS[i]}=${v} expected ${b[i]}`)
  })
}

const D = shuv360.durationS
check(shuv360.name === '360 SHUV', `name is ${shuv360.name}`)
check(D >= 1.3 && D <= 1.6, `durationS ${D} outside 1.3–1.6`)

matchesStanding(shuv360.sample(0), 't=0')
matchesStanding(shuv360.sample(D), 't=duration')

const end = shuv360.sample(D)
check(near(Math.abs(end.deck.rot.yaw), TAU), `total yaw ${end.deck.rot.yaw} != ±2π`)

const N = 3000
const dt = D / N
let prev = flat(shuv360.sample(0))
let minDeckY = Infinity
let maxRoll = 0
let maxRiderRotY = 0
let maxStep = 0
let maxStepLabel = ''
let minSpinGap = Infinity
let spinStart = -1
let spinEnd = -1
let maxH = 0
let maxPitch = 0
let maxYawAbsDuringPitch = 0
for (let i = 0; i <= N; i++) {
  const t = i * dt
  const p = shuv360.sample(t)
  const f = flat(p)
  f.forEach((v, j) => check(Number.isFinite(v), `NaN/inf ${LABELS[j]} at t=${t.toFixed(3)}`))
  minDeckY = Math.min(minDeckY, p.deck.pos[1])
  maxRoll = Math.max(maxRoll, Math.abs(p.deck.rot.roll))
  maxRiderRotY = Math.max(maxRiderRotY, Math.abs(p.rider.rotY))
  maxH = Math.max(maxH, p.rider.y)
  maxPitch = Math.max(maxPitch, Math.abs(p.deck.rot.pitch))
  if (Math.abs(p.deck.rot.pitch) > 0.05) maxYawAbsDuringPitch = Math.max(maxYawAbsDuringPitch, Math.abs(p.deck.rot.yaw))
  const spinning = Math.abs(p.deck.rot.yaw) > 1e-4 && Math.abs(p.deck.rot.yaw) < TAU - 1e-4
  if (spinning) {
    if (spinStart < 0) spinStart = t
    spinEnd = t
    minSpinGap = Math.min(minSpinGap, p.rider.y - p.deck.pos[1])
  }
  if (i > 0) {
    // Continuity: per-step change bounded (angles: yaw allowed to be fast, ~ up to 25 rad/s).
    f.forEach((v, j) => {
      const step = Math.abs(v - prev[j])
      // Generous rate caps (units or rad per second): fast but finite motion is fine, jumps are not.
      const limit = (j === 6 ? 30 : 40) * dt
      if (step > maxStep) {
        maxStep = step
        maxStepLabel = `${LABELS[j]} @t=${t.toFixed(3)}`
      }
      check(step <= limit, `discontinuity in ${LABELS[j]} at t=${t.toFixed(3)}: Δ=${step}`)
    })
  }
  prev = f
}

check(minDeckY >= 0.05, `deck dips below ground: min deck.y=${minDeckY}`)
check(maxRoll < 1e-9, `deck rolls (flip) max |roll|=${maxRoll}`)
check(minSpinGap > 0, `feet not above deck during spin: min gap=${minSpinGap}`)
check(maxRiderRotY < 0.3, `rider body rotates too much: max |rotY|=${maxRiderRotY}`)

const report = {
  durationS: D,
  totalYaw: end.deck.rot.yaw,
  totalYawOverTau: end.deck.rot.yaw / TAU,
  spinWindowS: [Number(spinStart.toFixed(3)), Number(spinEnd.toFixed(3))],
  minDeckY: Number(minDeckY.toFixed(4)),
  minFeetAboveDeckDuringSpin: Number(minSpinGap.toFixed(4)),
  peakFeetHeight: Number(maxH.toFixed(3)),
  maxPopPitch: Number(maxPitch.toFixed(3)),
  maxYawWhilePitched: Number(maxYawAbsDuringPitch.toFixed(3)),
  maxRiderRotY: Number(maxRiderRotY.toFixed(3)),
  maxRoll,
  largestStep: `${maxStep.toFixed(4)} (${maxStepLabel})`,
}
console.log(JSON.stringify(report, null, 2))

if (failures.length) {
  console.log(`FAIL (${failures.length}):\n  ` + failures.slice(0, 20).join('\n  '))
  throw new Error('shuv360 check failed')
}
console.log('PASS: shuv360 all checks')
