// Headless sanity check for the tre flip. Not part of the app bundle (nothing imports it).
// Run (from viewer/):
//   node -e "require('esbuild').buildSync({entryPoints:['src/views/three/tricks/treFlip.check.ts'],
//     bundle:true,platform:'node',outfile:'/private/tmp/claude-501/treFlip.check.cjs'})"
//   node /private/tmp/claude-501/treFlip.check.cjs

import { treFlip } from './treFlip'
import { STANDING, type TrickPose } from './types'

declare const process: { exitCode?: number }

const TAU = Math.PI * 2
const EPS = 1e-6
const failures: string[] = []
const check = (ok: boolean, msg: string): void => {
  if (!ok) failures.push(msg)
}

/** Wrap an angle into (-π, π]. */
const wrap = (a: number): number => {
  const w = ((a % TAU) + TAU) % TAU
  return w > Math.PI ? w - TAU : w
}

const flat = (p: TrickPose): number[] => [
  p.rider.y, p.rider.rotY,
  ...p.deck.pos,
  p.deck.rot.roll, p.deck.rot.yaw, p.deck.rot.pitch,
  p.body.crouch, p.body.tuck, p.body.armsOut, p.body.frontFootFlick, p.body.backFootScoop, p.body.lean,
]
const LABELS = [
  'rider.y', 'rider.rotY', 'deck.x', 'deck.y', 'deck.z', 'roll', 'yaw', 'pitch',
  'crouch', 'tuck', 'armsOut', 'frontFootFlick', 'backFootScoop', 'lean',
]

function nearStanding(p: TrickPose, label: string): void {
  const a = flat(p)
  const b = flat(STANDING)
  for (let i = 0; i < a.length; i++) {
    const isRot = i >= 5 && i <= 7
    const d = isRot ? wrap(a[i] - b[i]) : a[i] - b[i]
    check(Math.abs(d) < 1e-4, `${label}: ${LABELS[i]} = ${a[i]} (standing ${b[i]})`)
  }
}

const D = treFlip.durationS
check(treFlip.name === 'TRE FLIP', `name ${treFlip.name}`)
check(D >= 1.4 && D <= 1.8, `durationS ${D}`)

const p0 = treFlip.sample(0)
const pEnd = treFlip.sample(D)
nearStanding(p0, 't=0')
nearStanding(pEnd, 't=end')

// Exactly one full turn on each axis, same direction (backside shuv + kickflip, regular stance).
const dRoll = pEnd.deck.rot.roll - p0.deck.rot.roll
const dYaw = pEnd.deck.rot.yaw - p0.deck.rot.yaw
check(Math.abs(Math.abs(dRoll) - TAU) < EPS, `total roll ${dRoll}`)
check(Math.abs(Math.abs(dYaw) - TAU) < EPS, `total yaw ${dYaw}`)
check(Math.sign(dRoll) === Math.sign(dYaw), `roll/yaw signs differ: ${dRoll} ${dYaw}`)
check(Math.abs(pEnd.deck.rot.pitch - p0.deck.rot.pitch) < EPS, 'pitch net rotation non-zero')

// Sweep at 1 ms.
const STEP = 0.001
const N = Math.round(D / STEP)
const maxStep: Record<string, number> = { pos: 0, rot: 0, body: 0 }
const LIMIT = { pos: 0.006, rot: 0.04, body: 0.03 } // 6 m/s, 40 rad/s, 30 /s
let minDeckY = Infinity
let minFootClear = Infinity
let maxFootClear = -Infinity
let maxRiderY = -Infinity
let flipStart = -1
let flipEnd = -1
let prev = flat(p0)
let prevRoll = 0
let prevYaw = 0
let monotone = true
for (let i = 0; i <= N; i++) {
  const t = Math.min(i * STEP, D)
  const p = treFlip.sample(t)
  const a = flat(p)
  check(a.every(Number.isFinite), `non-finite at t=${t}`)
  for (const k of [8, 9, 10, 11, 12]) check(a[k] >= -EPS && a[k] <= 1 + EPS, `${LABELS[k]} out of 0..1 at t=${t}`)
  check(a[13] >= -1 && a[13] <= 1, `lean out of range at t=${t}`)
  minDeckY = Math.min(minDeckY, p.deck.pos[1])
  maxRiderY = Math.max(maxRiderY, p.rider.y)

  // "During the flip" = while the deck is rotating on either spin axis (and not yet caught).
  const spinning = Math.abs(p.deck.rot.roll) > 1e-9 && Math.abs(p.deck.rot.roll) < TAU - 1e-9
  if (spinning) {
    if (flipStart < 0) flipStart = t
    flipEnd = t
    const clear = p.rider.y - (p.deck.pos[1] + (STANDING.rider.y - STANDING.deck.pos[1]))
    minFootClear = Math.min(minFootClear, clear)
    maxFootClear = Math.max(maxFootClear, clear)
  }
  if (p.deck.rot.roll > prevRoll + EPS || p.deck.rot.yaw > prevYaw + EPS) monotone = false
  prevRoll = p.deck.rot.roll
  prevYaw = p.deck.rot.yaw

  if (i > 0) {
    for (let k = 0; k < a.length; k++) {
      const d = Math.abs(a[k] - prev[k])
      const group = k <= 4 && k !== 1 ? 'pos' : k <= 7 ? 'rot' : 'body'
      maxStep[group] = Math.max(maxStep[group], d)
    }
  }
  prev = a
}

check(minDeckY >= 0.05 - EPS, `deck below ground: min deck y ${minDeckY}`)
check(minFootClear >= -EPS, `feet inside the deck during the flip: min clearance ${minFootClear}`)
check(maxFootClear >= 0.10 && maxFootClear <= 0.16, `peak foot clearance ${maxFootClear} not in 10–15 cm`)
check(monotone, 'spin not monotone')
for (const g of Object.keys(LIMIT) as (keyof typeof LIMIT)[]) {
  check(maxStep[g] <= LIMIT[g], `discontinuity: max per-ms ${g} change ${maxStep[g]} > ${LIMIT[g]}`)
}
// Determinism.
check(JSON.stringify(treFlip.sample(0.6123)) === JSON.stringify(treFlip.sample(0.6123)), 'not deterministic')
// Out-of-range input clamps instead of exploding.
check(flat(treFlip.sample(-1)).every(Number.isFinite) && flat(treFlip.sample(99)).every(Number.isFinite), 'bad clamp')

const r = (x: number): string => x.toFixed(4)
console.log(`TRE FLIP  duration ${D}s`)
console.log(`  total roll ${r(dRoll)}  total yaw ${r(dYaw)}  (2π = ${r(TAU)})`)
console.log(`  spin window ${r(flipStart)}–${r(flipEnd)} s, peak feet y ${r(maxRiderY)} m`)
console.log(`  min deck y ${r(minDeckY)} m, foot clearance over grip during flip: min ${r(minFootClear)} / max ${r(maxFootClear)} m`)
console.log(`  max per-1ms change: pos ${r(maxStep.pos)} m, rot ${r(maxStep.rot)} rad, body ${r(maxStep.body)}`)
if (failures.length) {
  console.log(`FAIL (${failures.length})`)
  for (const f of failures.slice(0, 40)) console.log('  - ' + f)
  process.exitCode = 1
} else {
  console.log('PASS')
}
