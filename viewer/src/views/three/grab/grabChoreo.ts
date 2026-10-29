// Grab choreography: a tiny per-seat timeline that tells a character's hands where to be, how
// closed the fingers are and how much IK to apply, synchronised with the piece animation.
//
// Who drives timing: the piece animation (board/anims.ts) keeps owning the piece path. When a move
// arrives it asks `grabBus.reachLead(gap)` how long the hand needs to get to the piece, starts the
// piece flight that much later, and registers the move with `grabBus.requestMove(...)` passing the
// exact path function it uses to draw the piece. The rig then calls `grabBus.sample(seat)` each frame
// and runs IK to the returned target; because target and piece come from the same path(t), the
// piece sits in the hand while carried.
//
//   reach   hand travels from its idle pose to above the source, fingers spread, then dips down
//   grip    fingers close on the piece (holding = true from here)
//   carry   hand follows path(t) + grab offset
//   place   last 15% of the flight: the slam, with a wrist flick
//   release fingers open, hand pops up a little
//   return  IK weight eases back to 0 (FK pose takes over)
//
// Captures (bonus): the off hand plucks the captured piece during the reach, flicks it off the
// board edge mid-carry. `capturedPos(seat, t)` says where to draw it while it is held.
import { Vector3 } from 'three'
import type { Seat } from '../../../events/types'
import { seatPose } from '../layout'

export type GrabPhase = 'idle' | 'reach' | 'grip' | 'carry' | 'place' | 'release' | 'return'
export const PHASE_ORDER: GrabPhase[] = ['idle', 'reach', 'grip', 'carry', 'place', 'release', 'return']

export interface MoveRequest {
  seat: Seat
  letter: string             // piece letter (case = colour; used for grab height)
  from: Vector3              // piece base at the source square / reserve slot (world)
  to: Vector3                // piece base at the destination square (world)
  start: number              // seconds (same clock as anims.now()): when the piece lifts
  dur: number                // seconds of the piece flight (lift -> slam)
  drop?: boolean             // grabbed from the reserve pile (piece lying on its side)
  arc?: number               // arc height the default path uses (anims: 0.1 move, 0.16 knight, 0.22 drop)
  capture?: Vector3 | null   // captured piece base (world) -> off-hand flick
  captureLetter?: string
  /** Exact piece base position at time t (the one Pieces draws). Defaults to a copy of anims' curve. */
  path?: (t: number, out: Vector3) => Vector3
  now?: number               // request time (defaults to the clock)
}

export interface GrabSample {
  phase: GrabPhase
  active: boolean            // false -> leave the arm on FK
  weight: number             // IK weight 0..1
  grip: number               // finger curl 0..1 (hand.setPose)
  spread: number
  pinch: number
  holding: boolean           // the piece is in this hand
  target: Vector3            // world point for the hand's grip socket
  handDir: Vector3           // world direction the fingers point (includes the wrist flick)
  palmDir: Vector3           // world palm normal
  flick: number              // current wrist flick angle (radians), informational
}

// ---- tuning
export const TUNE = {
  reachMin: 0.0, reachMax: 0.32, reachOfGap: 0.35,  // lead time = clamp(gap * reachOfGap)
  gripFrac: 0.35,                                     // last part of the lead spent closing the fingers
  hover: 0.07,                                        // height above the grab point while approaching
  holdGrip: 0.72,
  release: 0.14, ret: 0.38,                           // seconds
  placeFrac: 0.85,                                    // matches anims' slam start
  flickAmp: 0.65,
  handPitch: 0.95,                                    // fingers' downward angle (rad) while grabbing
}

/** Piece heights (base -> top) from pieceGeometry.ts; the hand wraps just below the top. */
const HEIGHT: Record<string, number> = { p: 0.068, n: 0.098, b: 0.096, r: 0.084, q: 0.11, k: 0.13 }
export function grabHeight(letter: string, lying = false): number {
  if (lying) return 0.022
  return (HEIGHT[letter.toLowerCase()] ?? 0.08) - 0.02
}

export const clockNow = () => performance.now() / 1000

const easeInOut = (u: number) => (u < 0.5 ? 4 * u * u * u : 1 - Math.pow(-2 * u + 2, 3) / 2)
const smooth = (u: number) => { const x = Math.max(0, Math.min(1, u)); return x * x * (3 - 2 * x) }
const clamp01 = (u: number) => Math.max(0, Math.min(1, u))

/** Same curve as board/anims.ts updateAnims() for kind 'move' (piece base position). */
export function defaultPath(from: Vector3, to: Vector3, start: number, dur: number, arc: number) {
  return (t: number, out: Vector3) => {
    const u = clamp01((t - start) / dur)
    const e = easeInOut(u)
    out.lerpVectors(from, to, e)
    const lift = u < 0.85 ? Math.sin(Math.min(1, u / 0.85) * Math.PI * 0.5) : Math.pow(1 - (u - 0.85) / 0.15, 2)
    out.y += arc * Math.sin(Math.PI * e) * 0.6 + lift * 0.06
    return out
  }
}

interface Timeline {
  req: MoveRequest
  path: (t: number, out: Vector3) => Vector3
  t0: number                 // reach start
  gripAt: number             // fingers start closing
  start: number; end: number
  handStart: Vector3 | null  // where the hand was when the request came in (reported by the rig)
  fwd: Vector3               // seat forward (world, horizontal)
  left: Vector3              // seat's left (world)
  seatPos: Vector3
}

const lines = new Map<Seat, Timeline>()
const rest = new Map<Seat, { pos: Vector3; at: number }>()
/** Where the rig's grip sockets actually ended up this frame (after IK). */
const sockets = new Map<string, { pos: Vector3; at: number }>()
const FRESH_S = 0.1
const CAPTURE_RELEASE = 0.6       // fraction of the flight when the off hand lets go of a capture

const _v = new Vector3(), _w = new Vector3(), _h = new Vector3()

function newSample(): GrabSample {
  return {
    phase: 'idle', active: false, weight: 0, grip: 0, spread: 0, pinch: 0, holding: false,
    target: new Vector3(), handDir: new Vector3(0, -1, 0), palmDir: new Vector3(0, 0, 1), flick: 0,
  }
}

/** Fingers point toward `toward` (horizontal) and pitched down; palm faces down-ish. Flick pitches more. */
function aimHand(tl: Timeline, target: Vector3, flick: number, s: GrabSample) {
  const dirH = _h.set(target.x - tl.seatPos.x, 0, target.z - tl.seatPos.z)
  if (dirH.lengthSq() < 1e-6) dirH.copy(tl.fwd)
  dirH.normalize()
  const pitch = TUNE.handPitch + flick
  s.handDir.set(dirH.x * Math.cos(pitch), -Math.sin(pitch), dirH.z * Math.cos(pitch))
  // palm normal: "down" orthogonalised against the fingers
  s.palmDir.set(0, -1, 0).addScaledVector(s.handDir, s.handDir.y).normalize()
}

function flickAt(tl: Timeline, t: number): number {
  const { start, end } = tl
  const placeAt = start + (end - start) * TUNE.placeFrac
  if (t < placeAt) return 0
  if (t <= end) return TUNE.flickAmp * smooth((t - placeAt) / Math.max(1e-3, end - placeAt))
  // decaying wobble after the slam
  const k = (t - end) / 0.18
  return k >= 1 ? 0 : TUNE.flickAmp * (1 - k) * Math.cos(k * Math.PI * 1.5)
}

export const grabBus = {
  /** How long before the piece lifts the hand should start reaching, given the time since the
   *  previous move on that board (fast replays get ~0 lead). */
  reachLead(gap: number): number {
    return Math.max(TUNE.reachMin, Math.min(TUNE.reachMax, gap * TUNE.reachOfGap))
  },

  requestMove(req: MoveRequest) {
    const now = req.now ?? clockNow()
    const { pos, rotY } = seatPose(req.seat)
    const fwd = new Vector3(Math.sin(rotY), 0, Math.cos(rotY))
    const left = new Vector3(Math.cos(rotY), 0, -Math.sin(rotY))   // rig-local +x in world
    const lead = Math.max(0, req.start - now)
    const r = rest.get(req.seat)
    lines.set(req.seat, {
      req,
      path: req.path ?? defaultPath(req.from, req.to, req.start, req.dur, req.arc ?? (req.drop ? 0.22 : 0.1)),
      t0: now,
      gripAt: req.start - lead * TUNE.gripFrac,
      start: req.start, end: req.start + req.dur,
      handStart: r && now - r.at < 0.5 ? r.pos.clone() : null,
      fwd, left, seatPos: pos.clone(),
    })
  },

  /** Rigs may report where their (FK) grip socket is while idle so reaches start from there. */
  reportRest(seat: Seat, worldPos: Vector3, t = clockNow()) {
    const r = rest.get(seat)
    if (r) { r.pos.copy(worldPos); r.at = t } else rest.set(seat, { pos: worldPos.clone(), at: t })
  },

  /** Rigs report their solved grip-socket world positions every frame (after IK). */
  reportSocket(seat: Seat, hand: 'main' | 'off', worldPos: Vector3, t = clockNow()) {
    const k = seat + hand, r = sockets.get(k)
    if (r) { r.pos.copy(worldPos); r.at = t } else sockets.set(k, { pos: worldPos.clone(), at: t })
  },

  /** Base position for the moving piece while the main hand holds it: the piece rides in the actual
   *  hand (so it stays put even if the IK came up short). null -> draw it on its own path. */
  heldBase(seat: Seat, t = clockNow(), out = new Vector3()): Vector3 | null {
    const tl = lines.get(seat)
    const r = sockets.get(seat + 'main')
    if (!tl || !r || t - r.at > FRESH_S) return null
    const s = grabBus.sample(seat, t, scratch)
    if (!s.holding) return null
    // piece = socket + (piece path - hand target)
    tl.path(t, out).sub(s.target).add(r.pos)
    return out
  },

  cancel(seat: Seat) { lines.delete(seat) },
  reset() { lines.clear(); sockets.clear() },

  /** When and where the off hand lets go of a captured piece (for chaining the pass toss). */
  captureRelease(seat: Seat): { at: number; pos: Vector3 } | null {
    const tl = lines.get(seat)
    if (!tl?.req.capture) return null
    return { at: tl.start + (tl.end - tl.start) * CAPTURE_RELEASE, pos: capturedTarget(tl, 1, new Vector3()) }
  },

  /** Main (right) hand sample for this frame. */
  sample(seat: Seat, t = clockNow(), out: GrabSample = newSample()): GrabSample {
    const tl = lines.get(seat)
    out.flick = 0; out.pinch = 0
    if (!tl) { out.phase = 'idle'; out.active = false; out.weight = 0; out.grip = 0; out.spread = 0; out.holding = false; return out }
    const { req, start, end, gripAt, t0 } = tl
    const placeAt = start + (end - start) * TUNE.placeFrac
    const relEnd = end + TUNE.release, retEnd = relEnd + TUNE.ret
    if (t >= retEnd) { lines.delete(seat); return grabBus.sample(seat, t, out) }
    if (t < t0) { out.phase = 'idle'; out.active = false; out.weight = 0; out.grip = 0; out.spread = 0; out.holding = false; return out }

    const gh0 = grabHeight(req.letter, !!req.drop), gh1 = grabHeight(req.letter)
    const grabPt = _w.copy(req.from); grabPt.y += gh0
    out.active = true
    out.pinch = 0
    if (t < gripAt) {
      // reach: from the reported rest (or straight at the hover point) -> hover -> dip
      out.phase = 'reach'
      const u = clamp01((t - t0) / Math.max(1e-3, gripAt - t0))
      out.weight = smooth(u / 0.45)
      out.grip = 0.05; out.spread = smooth(u / 0.5)
      const hover = _v.copy(grabPt); hover.y += TUNE.hover
      if (u < 0.7) {
        const k = easeInOut(u / 0.7)
        if (tl.handStart) out.target.lerpVectors(tl.handStart, hover, k)
        else out.target.copy(hover)
        out.target.y += Math.sin(Math.PI * k) * 0.05   // little arc over the board edge
      } else {
        out.target.lerpVectors(hover, grabPt, smooth((u - 0.7) / 0.3))
      }
      out.holding = false
    } else if (t < start) {
      out.phase = 'grip'
      const u = clamp01((t - gripAt) / Math.max(1e-3, start - gripAt))
      out.weight = 1
      out.grip = TUNE.holdGrip * smooth(u); out.spread = 1 - u; out.pinch = u
      out.target.copy(grabPt)
      out.holding = u > 0.6
    } else if (t <= end) {
      out.phase = t < placeAt ? 'carry' : 'place'
      out.weight = 1
      out.grip = TUNE.holdGrip; out.spread = 0; out.pinch = 1
      tl.path(t, out.target)
      const u = clamp01((t - start) / Math.max(1e-3, end - start))
      out.target.y += gh0 + (gh1 - gh0) * smooth(u / 0.3)
      out.holding = true
      out.flick = flickAt(tl, t)
    } else if (t < relEnd) {
      out.phase = 'release'
      const u = (t - end) / TUNE.release
      out.weight = 1
      out.grip = TUNE.holdGrip * (1 - smooth(u)); out.spread = smooth(u) * 0.6
      out.target.copy(req.to); out.target.y += gh1 + 0.05 * smooth(u)
      out.holding = false
      out.flick = flickAt(tl, t)
    } else {
      out.phase = 'return'
      const u = (t - relEnd) / TUNE.ret
      out.weight = 1 - smooth(u)
      out.grip = 0.1; out.spread = 0.6 * (1 - u)
      out.target.copy(req.to); out.target.y += gh1 + 0.05 + 0.04 * smooth(u)
      out.holding = false
      out.flick = flickAt(tl, t)
    }
    aimHand(tl, out.target, out.flick, out)
    return out
  },

  /** Off (left) hand: only active for captures. Plucks the captured piece and flicks it off the board. */
  sampleOff(seat: Seat, t = clockNow(), out: GrabSample = newSample()): GrabSample {
    const tl = lines.get(seat)
    out.flick = 0; out.pinch = 0
    const cap = tl?.req.capture
    if (!tl || !cap) { out.phase = 'idle'; out.active = false; out.weight = 0; out.grip = 0; out.spread = 0; out.holding = false; return out }
    const { start, end, t0 } = tl
    const gh = grabHeight(tl.req.captureLetter ?? 'p')
    const lead = Math.max(0.12, start - t0)
    const a0 = start - lead                      // reach in parallel with the main hand
    const grabAt = start
    const tossAt = start + (end - start) * CAPTURE_RELEASE   // let go with a flick, well before the slam
    const back = tossAt + 0.4
    if (t < a0 || t >= back) { out.phase = 'idle'; out.active = false; out.weight = 0; out.holding = false; return out }
    out.active = true
    const grabPt = _w.copy(cap); grabPt.y += gh
    if (t < grabAt) {
      out.phase = t < grabAt - lead * 0.3 ? 'reach' : 'grip'
      const u = clamp01((t - a0) / lead)
      out.weight = smooth(u / 0.5)
      out.target.copy(grabPt); out.target.y += TUNE.hover * (1 - smooth(u))
      out.grip = TUNE.holdGrip * smooth((u - 0.7) / 0.3); out.spread = 1 - u
      out.holding = false
    } else if (t < tossAt) {
      out.phase = 'carry'
      out.weight = 1
      out.grip = TUNE.holdGrip
      capturedTarget(tl, (t - grabAt) / Math.max(1e-3, tossAt - grabAt), out.target)
      out.target.y += gh
      out.holding = true
    } else {
      out.phase = t < tossAt + TUNE.release ? 'release' : 'return'
      const u = clamp01((t - tossAt) / 0.4)
      out.weight = 1 - smooth((u - 0.3) / 0.7)
      out.grip = TUNE.holdGrip * (1 - smooth(u / 0.35)); out.spread = 0.5
      capturedTarget(tl, 1, out.target); out.target.y += gh + 0.04 * u
      out.flick = TUNE.flickAmp * Math.sin(Math.PI * clamp01(u / 0.4))
      out.holding = false
    }
    aimHand(tl, out.target, out.flick, out)
    return out
  },

  /** Where to draw the captured piece (base) while the off hand holds it; null when not held. */
  capturedPos(seat: Seat, t = clockNow(), out = new Vector3()): Vector3 | null {
    const tl = lines.get(seat)
    if (!tl?.req.capture) return null
    const tossAt = tl.start + (tl.end - tl.start) * CAPTURE_RELEASE
    if (t < tl.start) return out.copy(tl.req.capture)
    if (t >= tossAt) return null
    // ride in the actual off hand when the rig reported it this frame
    const r = sockets.get(seat + 'off')
    if (r && t - r.at <= FRESH_S && grabBus.sampleOff(seat, t, scratch).holding) {
      out.copy(r.pos); out.y -= grabHeight(tl.req.captureLetter ?? 'p')
      return out
    }
    return capturedTarget(tl, (t - tl.start) / Math.max(1e-3, tossAt - tl.start), out)
  },

  /** Main-hand grip-socket target this frame, or null when the arm should stay on FK. */
  getHandTarget(seat: Seat, t = clockNow()): Vector3 | null {
    const s = grabBus.sample(seat, t, scratch)
    return s.active ? s.target : null
  },
  isHolding(seat: Seat, t = clockNow()): boolean { return grabBus.sample(seat, t, scratch).holding },
  /** Debug/tests: the active timeline's key times. */
  timesOf(seat: Seat) {
    const tl = lines.get(seat)
    return tl ? { t0: tl.t0, gripAt: tl.gripAt, start: tl.start, end: tl.end } : null
  },
}
const scratch = newSample()

/** Off-hand carry path for a captured piece: lift and swing out past the board edge on the off side. */
function capturedTarget(tl: Timeline, u: number, out: Vector3): Vector3 {
  const k = smooth(u)
  const cap = tl.req.capture!
  // destination: out to the mover's left, toward the mover, a bit above the table
  out.copy(cap).addScaledVector(tl.left, 0.16 * k).addScaledVector(tl.fwd, -0.12 * k)
  out.y += Math.sin(Math.PI * Math.min(1, u * 1.2)) * 0.07 + 0.04 * k
  return out
}
