// Headless sanity check for the kickflip. Bundle + run with node, e.g.:
//   node -e "require('esbuild').buildSync({entryPoints:['src/views/three/tricks/kickflip.check.ts'],
//     bundle:true,platform:'node',outfile:'/private/tmp/claude-501/kickflip.check.cjs'})"
//   node /private/tmp/claude-501/kickflip.check.cjs
// Throws (non-zero exit) on the first batch of failures.

import { kickflip } from './kickflip'
import { STANDING, type TrickPose } from './types'

const TAU = Math.PI * 2
const failures: string[] = []
const check = (ok: boolean, msg: string) => {
  if (!ok) failures.push(msg)
}
const modTau = (a: number) => {
  const m = ((a % TAU) + TAU) % TAU
  return Math.min(m, TAU - m) // distance to nearest multiple of 2π
}

function flat(p: TrickPose): number[] {
  const b = p.body
  return [
    p.rider.y, p.rider.rotY, ...p.deck.pos, p.deck.rot.roll, p.deck.rot.yaw, p.deck.rot.pitch,
    b.crouch, b.tuck, b.armsOut, b.frontFootFlick, b.backFootScoop, b.lean,
  ]
}
const NAMES = ['rider.y', 'rider.rotY', 'deck.x', 'deck.y', 'deck.z', 'roll', 'yaw', 'pitch',
  'crouch', 'tuck', 'armsOut', 'frontFootFlick', 'backFootScoop', 'lean']
const ROT_IDX = new Set([1, 5, 6, 7])

function closeToStanding(p: TrickPose, label: string) {
  const a = flat(p)
  const s = flat(STANDING)
  a.forEach((v, i) => {
    const d = ROT_IDX.has(i) ? modTau(v - s[i]) : Math.abs(v - s[i])
    check(d < 1e-6, `${label}: ${NAMES[i]} = ${v} (standing ${s[i]})`)
  })
}

const T = kickflip.durationS
check(kickflip.name === 'KICKFLIP', `name ${kickflip.name}`)
check(T >= 1.3 && T <= 1.6, `durationS ${T} out of 1.3–1.6`)
closeToStanding(kickflip.sample(0), 't=0')
closeToStanding(kickflip.sample(T), 't=durationS')

const end = kickflip.sample(T)
check(Math.abs(Math.abs(end.deck.rot.roll) - TAU) < 1e-6, `total roll ${end.deck.rot.roll} != ±2π`)
check(Math.abs(end.deck.rot.yaw) < 1e-6, `end yaw ${end.deck.rot.yaw}`)

// Dense sweep
const N = 3000
const dt = T / N
const MAX_STEP = [0.02, 0.02, 0.02, 0.02, 0.02, 0.05, 0.05, 0.05, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02]
let prev = flat(kickflip.sample(0))
let minDeckY = Infinity
let maxGap = -Infinity
let minFlipClear = Infinity
let maxPitchAfterFlip = 0
let rollDoneT = -1
let airStart = -1
let airEnd = -1
let maxRiderY = 0
for (let i = 0; i <= N; i++) {
  const t = i * dt
  const p = kickflip.sample(t)
  const v = flat(p)
  v.forEach((x, k) => check(Number.isFinite(x), `NaN/Inf ${NAMES[k]} at t=${t.toFixed(3)}`))
  v.forEach((x, k) => {
    const d = Math.abs(x - prev[k])
    check(d <= MAX_STEP[k], `discontinuity ${NAMES[k]} Δ=${d.toFixed(4)} at t=${t.toFixed(3)}`)
  })
  for (const k of [8, 9, 10, 11, 12]) check(v[k] >= 0 && v[k] <= 1, `${NAMES[k]} out of 0..1 at ${t}`)
  check(p.body.lean >= -1 && p.body.lean <= 1, `lean out of range at ${t}`)
  prev = v

  const [dx, dy] = p.deck.pos
  minDeckY = Math.min(minDeckY, dy)
  maxRiderY = Math.max(maxRiderY, p.rider.y)
  if (p.rider.y > STANDING.rider.y + 1e-9) {
    if (airStart < 0) airStart = t
    airEnd = t
  }
  const roll = p.deck.rot.roll
  if (rollDoneT < 0 && Math.abs(Math.abs(roll) - TAU) < 1e-9) rollDoneT = t
  // While the board is mid-flip (well away from wheels-down), feet must be above the deck's
  // highest point: deck centre + rotated half-width (0.105) / half-thickness (0.02).
  if (modTau(roll) > 0.3) {
    const deckTop = dy + Math.abs(0.105 * Math.sin(roll)) + Math.abs(0.02 * Math.cos(roll))
    const clear = p.rider.y - deckTop
    minFlipClear = Math.min(minFlipClear, clear)
    maxGap = Math.max(maxGap, p.rider.y - (dy + 0.02))
    void dx
  }
  // Ollie pop pitch must be gone before the flip finishes (catch tilt is negative, separate).
  if (roll > Math.PI && p.deck.rot.pitch > 1e-3) maxPitchAfterFlip = Math.max(maxPitchAfterFlip, p.deck.rot.pitch)
}
check(minDeckY >= 0.05, `deck dips below ground: min deck.y ${minDeckY}`)
check(minFlipClear > 0, `feet intersect deck mid-flip: min clearance ${minFlipClear}`)
check(maxGap >= 0.1 && maxGap <= 0.2, `mid-flip foot/deck gap ${maxGap} not ~10–15 cm`)
check(maxPitchAfterFlip === 0, `pop pitch still positive in 2nd half of flip: ${maxPitchAfterFlip}`)
check(rollDoneT > 0 && rollDoneT < airEnd, `roll completes (${rollDoneT}) after touchdown (${airEnd})`)

console.log('KICKFLIP check')
console.log(`  durationS            ${T}`)
console.log(`  airborne             ${airStart.toFixed(3)}–${airEnd.toFixed(3)} s (${(airEnd - airStart).toFixed(3)} s)`)
console.log(`  apex feet height     ${maxRiderY.toFixed(3)} m`)
console.log(`  roll complete at     ${rollDoneT.toFixed(3)} s, total ${end.deck.rot.roll.toFixed(6)} rad`)
console.log(`  min deck.y           ${minDeckY.toFixed(4)} m`)
console.log(`  max foot/deck gap    ${maxGap.toFixed(3)} m (mid-flip)`)
console.log(`  min flip clearance   ${minFlipClear.toFixed(3)} m (feet above rotated deck extent)`)
if (failures.length) {
  console.log(`FAIL (${failures.length})`)
  for (const f of failures.slice(0, 30)) console.log('  - ' + f)
  throw new Error('kickflip check failed')
}
console.log('PASS')
