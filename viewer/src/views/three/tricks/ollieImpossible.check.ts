// Headless check for the ollie impossible. Bundle with esbuild (platform node) and run with node:
//   node -e "require('esbuild').buildSync({entryPoints:['src/views/three/tricks/ollieImpossible.check.ts'],
//     bundle:true,platform:'node',outfile:'/private/tmp/claude-501/oi-check.cjs'})" && node /private/tmp/claude-501/oi-check.cjs
// Throws on the first batch of failures; prints a summary otherwise.

import { STANDING, type TrickPose } from './types'
import { ollieImpossible as trick } from './ollieImpossible'

const TAU = Math.PI * 2
const HALF_LEN = 0.4
const BACK_FOOT_X = -0.27
const failures: string[] = []
const fail = (msg: string) => { if (failures.length < 30) failures.push(msg) }

const wrap = (a: number) => { const r = ((a % TAU) + TAU) % TAU; return r > Math.PI ? r - TAU : r }

function flatten(p: TrickPose): number[] {
  return [
    p.rider.y, p.rider.rotY, ...p.deck.pos, p.deck.rot.roll, p.deck.rot.yaw, p.deck.rot.pitch,
    p.body.crouch, p.body.tuck, p.body.armsOut, p.body.frontFootFlick, p.body.backFootScoop, p.body.lean,
  ]
}
const ROT_IDX = new Set([1, 5, 6, 7]) // rider.rotY, roll, yaw, pitch

function assertNearStanding(p: TrickPose, label: string) {
  const a = flatten(p), b = flatten(STANDING)
  a.forEach((v, i) => {
    const d = ROT_IDX.has(i) ? wrap(v - b[i]) : v - b[i]
    if (Math.abs(d) > 1e-6) fail(`${label}: field #${i} = ${v}, STANDING = ${b[i]}`)
  })
}

/** World-space (rider-frame) positions of the deck ends; Euler YZX with roll/yaw ≈ 0 => pitch about z. */
function deckEnds(p: TrickPose) {
  const { pitch } = p.deck.rot
  const [x, y] = p.deck.pos
  const ax = Math.cos(pitch), ay = Math.sin(pitch)
  return { nose: [x + HALF_LEN * ax, y + HALF_LEN * ay], tail: [x - HALF_LEN * ax, y - HALF_LEN * ay] }
}

const D = trick.durationS
if (trick.name !== 'OLLIE IMPOSSIBLE') fail(`name = ${trick.name}`)
if (D < 1.5 || D > 1.9) fail(`durationS = ${D}`)

assertNearStanding(trick.sample(0), 't=0')
assertNearStanding(trick.sample(D), 't=D')

const endPitch = trick.sample(D).deck.rot.pitch
if (Math.abs(Math.abs(endPitch) - TAU) > 1e-9) fail(`total pitch = ${endPitch}, expected ±2π`)

const N = 4000
const dt = D / N
let prev: number[] | null = null
let minEnd = Infinity, minEndT = 0, maxTailDist = 0, maxRiderY = 0, maxRiderYT = 0
let maxStep = 0, maxStepIdx = -1
let prevPitch = 0
for (let i = 0; i <= N; i++) {
  const t = i * dt
  const p = trick.sample(t)
  const v = flatten(p)
  if (v.some((x) => !Number.isFinite(x))) fail(`NaN/Inf at t=${t.toFixed(3)}`)
  if (Math.abs(p.deck.rot.roll) > 1e-9 || Math.abs(p.deck.rot.yaw) > 1e-9) fail(`roll/yaw non-zero at t=${t}`)
  if (p.deck.rot.pitch < prevPitch - 1e-12) fail(`pitch not monotonic at t=${t.toFixed(3)}`)
  prevPitch = p.deck.rot.pitch
  for (const k of ['crouch', 'tuck', 'armsOut', 'frontFootFlick', 'backFootScoop'] as const) {
    if (p.body[k] < -1e-9 || p.body[k] > 1 + 1e-9) fail(`body.${k} out of [0,1] at t=${t}`)
  }

  const { nose, tail } = deckEnds(p)
  const lo = Math.min(nose[1], tail[1])
  if (lo < minEnd) { minEnd = lo; minEndT = t }
  if (lo < 0.02 - 1e-9) fail(`deck end below 0.02 m at t=${t.toFixed(3)} (y=${lo.toFixed(4)})`)

  // Back foot: on the tail at x=-0.27, sole at rider.y. Check during the wrap (pitch moving).
  const pitch = p.deck.rot.pitch
  if (pitch > 1e-6 && pitch < TAU - 1e-6) {
    const d = Math.hypot(tail[0] - BACK_FOOT_X, tail[1] - p.rider.y)
    maxTailDist = Math.max(maxTailDist, d)
    if (d > 0.15) fail(`tail ${d.toFixed(3)} m from back foot at t=${t.toFixed(3)}`)
  }
  if (p.rider.y > maxRiderY) { maxRiderY = p.rider.y; maxRiderYT = t }

  if (prev) {
    v.forEach((x, j) => {
      const step = Math.abs(x - prev![j])
      if (step > maxStep) { maxStep = step; maxStepIdx = j }
    })
  }
  prev = v
}
// Continuity: with dt ≈ 0.4 ms, the fastest channel (pitch, ~11 rad/s peak) moves < 0.01 per step.
if (maxStep > 0.01) fail(`discontinuity: max per-step change ${maxStep} on field #${maxStepIdx}`)

// Out-of-range times clamp rather than explode.
for (const t of [-1, D + 1, NaN]) {
  if (flatten(trick.sample(t)).some((x) => !Number.isFinite(x))) fail(`non-finite pose at t=${t}`)
}

const at = (t: number) => trick.sample(t)
const summary = [
  `durationS=${D}`,
  `pitch(D)=${endPitch.toFixed(6)} (2π=${TAU.toFixed(6)})`,
  `lowest deck end=${minEnd.toFixed(4)} m @ t=${minEndT.toFixed(3)}`,
  `max tail↔back-foot distance during wrap=${maxTailDist.toFixed(4)} m`,
  `apex rider.y=${maxRiderY.toFixed(3)} m @ t=${maxRiderYT.toFixed(3)}`,
  `max per-step change=${maxStep.toFixed(5)} (field #${maxStepIdx}, dt=${dt.toFixed(5)} s)`,
  ...[0.3, 0.34, 0.5, 0.72, 0.9, 1.02, 1.16, 1.24, 1.7].map((t) => {
    const p = at(t)
    return `  t=${t.toFixed(2)} riderY=${p.rider.y.toFixed(3)} pitch=${p.deck.rot.pitch.toFixed(3)} ` +
      `deck=(${p.deck.pos[0].toFixed(3)},${p.deck.pos[1].toFixed(3)}) crouch=${p.body.crouch.toFixed(2)} ` +
      `ff=${p.body.frontFootFlick.toFixed(2)} scoop=${p.body.backFootScoop.toFixed(2)}`
  }),
]

if (failures.length) throw new Error(`ollieImpossible check FAILED:\n${failures.join('\n')}\n${summary.join('\n')}`)
console.log(`ollieImpossible check PASSED\n${summary.join('\n')}`)
