// Camera rig. Two modes (viewStore.cameraMode):
//
// FREE (default) — the user drives: drei OrbitControls (left-drag orbit, right-drag / two-finger
//   pan, wheel / pinch zoom, damped), plus keyboard: WASD / arrows pan the target, Q/E orbit,
//   R resets, 1 / 2 / 0 glide to board A / board B / both (then control goes straight back).
//   Limits: never under the ground, 2–14 m from the target, target clamped to the play area.
//   Nothing moves this camera on its own. Capture/drop "juice" (STYLE.camera.punchIn) is a tiny
//   FOV punch (freePunchFov, capped at 3°) plus the world shake (fx.shake, scaled) — the camera
//   position never changes.
//
// AUTO (opt-in "AUTO CAM", the old broadcast director):
//   - follows the board with the latest event using stable side / 3-4 high / top-down shots that
//     never put a player between the camera and the board (see shots.ts), with occasional wides;
//   - quick ~1.2 s close-up on an agent who talks (fxBus `talk`, not status shares or pass calls);
//   - FOV punch-in + shake on captures and drops (fxBus `capture` / `drop`);
//   - slow orbit around both boards when nothing has happened for a few seconds;
//   - every cut respects a minimum hold so fast games don't strobe.
// Switching AUTO -> FREE keeps the camera exactly where it is (the orbit target becomes the point
// the broadcast shot was looking at).
import { useEffect, useMemo, useRef } from 'react'
import { useFrame, useThree } from '@react-three/fiber'
import { OrbitControls } from '@react-three/drei'
import type { OrbitControls as OrbitControlsImpl } from 'three-stdlib'
import { PerspectiveCamera, Vector3 } from 'three'
import type { BoardId, Seat } from '../../../events/types'
import { useGame } from '../../../store/gameStore'
import { onFxReset, onGameEvent } from '../director'
import { fxBus } from '../fxBus'
import { STYLE } from '../style.config'
import { type CameraFocus, fx, useView } from '../viewStore'
import {
  FREE_BOUNDS, FREE_FOCUS, FREE_HOME, FREE_MAX_DIST, FREE_MIN_DIST, FREE_MIN_Y, ORBIT_TARGET,
  type OrbitPose, type Shot, WIDE, boardShots, orbitPose, talkShot,
} from './shots'

type Kind = 'board' | 'wide' | 'talk' | 'orbit'
const MAX_HOLD = 7
const UP = new Vector3(0, 1, 0)

const nowS = () => performance.now() / 1000
const easeInOut = (u: number) => (u < 0.5 ? 4 * u * u * u : 1 - Math.pow(-2 * u + 2, 3) / 2)

interface Glide { to: OrbitPose; init: boolean; t0: number; fromP: Vector3; fromT: Vector3; fromFov: number }

// keyboard (KeyboardEvent.code)
const MOVE_KEYS: Record<string, [number, number]> = {     // [forward, right]
  KeyW: [1, 0], ArrowUp: [1, 0], KeyS: [-1, 0], ArrowDown: [-1, 0],
  KeyA: [0, -1], ArrowLeft: [0, -1], KeyD: [0, 1], ArrowRight: [0, 1],
}
const ROT_KEYS: Record<string, number> = { KeyQ: -1, KeyE: 1 }
const FOCUS_KEYS: Record<string, CameraFocus> = {
  Digit1: 'A', Numpad1: 'A', Digit2: 'B', Numpad2: 'B', Digit0: 'both', Numpad0: 'both', KeyR: 'home',
}

function typingInField(e: KeyboardEvent): boolean {
  const el = e.target as HTMLElement | null
  if (!el || !el.tagName) return false
  const tag = el.tagName
  return tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT' || el.isContentEditable
}

function clampTarget(t: Vector3): Vector3 {
  const { min, max } = FREE_BOUNDS
  return t.set(
    Math.min(max[0], Math.max(min[0], t.x)),
    Math.min(max[1], Math.max(min[1], t.y)),
    Math.min(max[2], Math.max(min[2], t.z)),
  )
}

export function CameraRig() {
  const mode = useView(s => s.cameraMode)
  const thumb = useView(s => s.thumb)   // 3D is the small PiP thumbnail: no user camera input
  const camera = useThree(s => s.camera) as PerspectiveCamera
  const controls = useRef<OrbitControlsImpl>(null)
  const st = useRef({
    kind: 'wide' as Kind, shot: WIDE[0] as Shot, cutAt: -99, board: null as BoardId | null, hold: 4, lastWide: 0,
    lastEventAt: nowS(), orbitAngle: 0, punch: 0, talkReq: null as { seat: Seat; at: number } | null, lastTalkCut: -99,
    lastTarget: new Vector3(...FREE_HOME.target),   // where the camera is looking (for AUTO -> FREE)
    baseFov: FREE_HOME.fov,                         // free-mode FOV before the punch
    glide: null as Glide | null,
    keys: new Set<string>(),
    prevMode: null as string | null,
  })
  const tmp = useMemo(() => ({ p: new Vector3(), t: new Vector3(), d: new Vector3(), r: new Vector3() }), [])

  // mode changes (and the initial placement)
  useEffect(() => {
    const s = st.current, c = controls.current
    const first = s.prevMode === null
    s.prevMode = mode
    if (mode === 'free') {
      if (first) {
        camera.position.set(...FREE_HOME.pos)
        s.lastTarget.set(...FREE_HOME.target)
        s.baseFov = FREE_HOME.fov
      } else {
        // keep the camera where the director left it; orbit around what it was looking at
        s.baseFov = Math.min(60, Math.max(28, camera.fov))
      }
      clampTarget(s.lastTarget)
      if (c) { c.target.copy(s.lastTarget); c.update() }
      camera.lookAt(s.lastTarget)
      camera.fov = s.baseFov; camera.updateProjectionMatrix()
    } else {
      s.glide = null
      s.kind = 'wide'; s.cutAt = -99   // first director cut straight away
      if (first) {
        const w = WIDE[0]
        camera.position.set(...w.from); camera.lookAt(...w.to); camera.fov = w.fov; camera.updateProjectionMatrix()
      }
    }
  }, [mode, camera])

  // user grabs the controls -> any glide stops immediately
  useEffect(() => {
    const c = controls.current
    if (!c) return
    const stop = () => { st.current.glide = null }
    c.addEventListener('start', stop)
    return () => c.removeEventListener('start', stop)
  }, [])

  // focus requests (HUD buttons, 1/2/0/R) -> glide; the start pose is sampled on the first frame
  useEffect(() => useView.subscribe((now, prev) => {
    const req = now.focusReq
    if (!req || req === prev.focusReq) return
    st.current.glide = { to: FREE_FOCUS[req.to], init: false, t0: 0, fromP: new Vector3(), fromT: new Vector3(), fromFov: 45 }
  }), [])

  // keyboard
  useEffect(() => {
    const keys = st.current.keys
    const down = (e: KeyboardEvent) => {
      if (e.ctrlKey || e.metaKey || e.altKey || typingInField(e) || useView.getState().thumb) return
      const focus = FOCUS_KEYS[e.code]
      if (focus) { if (!e.repeat) useView.getState().requestFocus(focus); return }
      if (MOVE_KEYS[e.code] || ROT_KEYS[e.code] !== undefined) {
        if (e.code.startsWith('Arrow')) e.preventDefault()   // don't scroll the page
        const v = useView.getState()
        if (v.cameraMode !== 'free') v.setCameraMode('free')
        keys.add(e.code)
        st.current.glide = null
      }
    }
    const up = (e: KeyboardEvent) => { keys.delete(e.code) }
    const clear = () => keys.clear()
    window.addEventListener('keydown', down)
    window.addEventListener('keyup', up)
    window.addEventListener('blur', clear)
    return () => {
      window.removeEventListener('keydown', down); window.removeEventListener('keyup', up); window.removeEventListener('blur', clear)
    }
  }, [])

  // shrinking into the PiP thumbnail: drop any held WASD/QE keys so the camera doesn't keep drifting
  useEffect(() => { if (thumb) st.current.keys.clear() }, [thumb])

  // game activity, punches and talk requests
  useEffect(() => {
    const s = st.current
    const offE = onGameEvent(() => { s.lastEventAt = nowS() })
    const offR = onFxReset(() => { s.lastEventAt = nowS(); s.talkReq = null; s.punch = 0 })
    const offB = fxBus.on(e => {
      const C = STYLE.camera
      if ((e.kind === 'capture' || e.kind === 'drop') && C.punchIn) {
        s.punch = 1
        const k = useView.getState().cameraMode === 'free' ? C.freeShakeScale : 1
        fx.shake = Math.max(fx.shake, (e.kind === 'capture' ? 0.38 : 0.28) * k)
      } else if (e.kind === 'talk' && C.talkCutIns && !e.status && e.token !== 'pass') {
        s.talkReq = { seat: e.seat, at: nowS() }
      }
    })
    return () => { offE(); offR(); offB() }
  }, [])

  // ---------------- FREE ----------------
  const freeFrame = (dt: number) => {
    const c = controls.current
    if (!c) return
    const C = STYLE.camera, s = st.current, t = nowS()
    const target = c.target, pos = camera.position

    const g = s.glide
    if (g) {
      if (!g.init) {
        g.init = true; g.t0 = t
        g.fromP.copy(pos); g.fromT.copy(target); g.fromFov = s.baseFov
      }
      const u = Math.min(1, (t - g.t0) / Math.max(0.05, C.focusGlideMs / 1000)), e = easeInOut(u)
      pos.lerpVectors(g.fromP, tmp.p.set(...g.to.pos), e)
      target.lerpVectors(g.fromT, tmp.t.set(...g.to.target), e)
      s.baseFov = g.fromFov + (g.to.fov - g.fromFov) * e
      if (u >= 1) s.glide = null
    } else if (s.keys.size) {
      // WASD / arrows: slide target + camera together over the ground plane; Q/E: orbit
      let f = 0, r = 0, rot = 0
      for (const k of s.keys) {
        const m = MOVE_KEYS[k]
        if (m) { f += m[0]; r += m[1] }
        rot += ROT_KEYS[k] ?? 0
      }
      if (f || r) {
        tmp.d.subVectors(target, pos).setY(0)
        if (tmp.d.lengthSq() < 1e-6) tmp.d.set(0, 0, -1)   // looking straight down
        tmp.d.normalize()
        tmp.r.set(-tmp.d.z, 0, tmp.d.x)                     // right of forward
        const step = C.keyPanSpeed * Math.max(2, pos.distanceTo(target)) * dt
        tmp.p.copy(tmp.d).multiplyScalar(f * step).addScaledVector(tmp.r, r * step)
        target.add(tmp.p); pos.add(tmp.p)
      }
      if (rot) {
        tmp.d.subVectors(pos, target).applyAxisAngle(UP, rot * C.keyRotateSpeed * dt)
        pos.copy(target).add(tmp.d)
      }
    }

    // limits: target inside the play area (the camera follows the correction), camera above ground
    tmp.t.copy(target)
    clampTarget(target)
    tmp.t.subVectors(target, tmp.t)
    if (tmp.t.lengthSq() > 0) pos.add(tmp.t)
    if (pos.y < FREE_MIN_Y) pos.y = FREE_MIN_Y
    camera.lookAt(target)
    s.lastTarget.copy(target)

    // capture/drop punch: FOV only, never the position
    s.punch *= Math.exp(-dt * 3.2)
    const want = s.baseFov - (C.punchIn ? Math.min(3, C.freePunchFov) : 0) * s.punch
    if (Math.abs(camera.fov - want) > 0.01) { camera.fov = want; camera.updateProjectionMatrix() }
  }

  // ---------------- AUTO (broadcast director) ----------------
  const autoFrame = (dt: number) => {
    const C = STYLE.camera
    const t = nowS(), s = st.current
    const since = t - s.cutAt
    const minHold = C.minHoldMs / 1000
    const cut = (kind: Kind, shot: Shot, hold: number) => {
      s.kind = kind; s.shot = shot; s.cutAt = t; s.hold = hold
    }

    // 1) talk close-up (rate-limited; drop requests that waited too long)
    const req = s.talkReq
    if (req && t - req.at > 0.8) s.talkReq = null
    else if (req && since >= minHold && t - s.lastTalkCut >= C.talkCutCooldownMs / 1000) {
      cut('talk', talkShot(req.seat), C.talkCutMs / 1000)
      s.lastTalkCut = t; s.talkReq = null
    }

    const idle = C.idleOrbit && t - s.lastEventAt > C.idleAfterMs / 1000
    if (s.kind === 'talk' && t - s.cutAt < s.hold) {
      // hold the close-up
    } else if (idle) {
      if (s.kind !== 'orbit') {
        s.kind = 'orbit'; s.cutAt = t
        s.orbitAngle = Math.atan2(camera.position.x, camera.position.z)
      }
      s.orbitAngle += C.idleOrbitSpeed * dt
    } else {
      // 2) follow the active board
      const board = useGame.getState().lastEventBoard
      const boardChanged = board !== null && board !== s.board
      const leaving = s.kind === 'orbit' || s.kind === 'talk'
      if (leaving || (boardChanged && since > C.boardSwitchHoldMs / 1000) || since > s.hold) {
        const goWide = board === null || (!leaving && t - s.lastWide > 18 && Math.random() < 0.35)
        const pool = goWide ? WIDE : boardShots(board!)
        let next = pool[Math.floor(Math.random() * pool.length)]
        if (next === s.shot || (next.from[0] === s.shot.from[0] && next.from[2] === s.shot.from[2])) next = pool[(pool.indexOf(next) + 1) % pool.length]
        cut(goWide ? 'wide' : 'board', next, goWide ? 4 : 2.6 + Math.random() * (MAX_HOLD - 2.6))
        s.board = board
        if (goWide) s.lastWide = t
      }
    }

    // pose
    let fov: number
    if (s.kind === 'orbit') {
      orbitPose(s.orbitAngle, tmp.p)
      tmp.t.set(...ORBIT_TARGET)
      fov = 46
    } else {
      const sh = s.shot
      const u = Math.min(1, (t - s.cutAt) / (s.hold + 2))
      tmp.p.set(sh.from[0] + sh.drift[0] * u, sh.from[1] + sh.drift[1] * u, sh.from[2] + sh.drift[2] * u)
      tmp.t.set(...sh.to)
      fov = sh.fov
    }
    s.lastTarget.copy(tmp.t)
    // handheld wobble + a little camera-side shake on top of the world shake
    tmp.p.x += Math.sin(t * 0.9) * 0.02; tmp.p.y += Math.sin(t * 1.3 + 1) * 0.015
    if (fx.shake > 0.01) { const k = fx.shake * 0.025; tmp.p.x += (Math.random() - 0.5) * k; tmp.p.y += (Math.random() - 0.5) * k }
    camera.position.copy(tmp.p)
    camera.lookAt(tmp.t)

    // FOV punch: snap in, ease back out
    s.punch *= Math.exp(-dt * 3.2)
    const want = fov - C.punchFov * s.punch
    if (Math.abs(camera.fov - want) > 0.01) { camera.fov = want; camera.updateProjectionMatrix() }
  }

  // runs after drei's controls.update() (priority -1)
  useFrame((_, delta) => {
    const dt = Math.min(delta, 0.1)
    if (mode === 'free') freeFrame(dt)
    else autoFrame(dt)
  })

  return (
    <OrbitControls
      ref={controls}
      makeDefault
      enabled={mode === 'free' && !thumb}
      enableDamping dampingFactor={0.08}
      rotateSpeed={0.7} zoomSpeed={0.9} panSpeed={0.9}
      screenSpacePanning={false}
      minDistance={FREE_MIN_DIST} maxDistance={FREE_MAX_DIST}
      maxPolarAngle={Math.PI / 2 - 0.06}
    />
  )
}
