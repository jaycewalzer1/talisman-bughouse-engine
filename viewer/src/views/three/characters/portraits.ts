// One-time head-and-shoulders portraits of each skater, for the speech-bubble cut-ins.
//
//   const url = usePortraits(s => s.urls[seat])      // PNG data URL (128x128) or undefined
//
// <PortraitCapture /> (mounted inside <Characters />) waits until the rigs have rendered a few
// frames, then renders each skater alone (other skaters hidden, environment excluded via a
// dedicated layer) from a 3/4 front camera into a small offscreen target, composites it over a
// flat sunset gradient in the team colour with a chunky ink outline, and stores a data URL.
// It runs once per mount — no per-frame cost afterwards.
import { useRef } from 'react'
import { useFrame, useThree } from '@react-three/fiber'
import { create } from 'zustand'
import {
  Color, PerspectiveCamera, Vector3, WebGLRenderTarget, type Object3D, type Scene, type WebGLRenderer,
} from 'three'
import { SEATS, TEAM_OF, type Seat } from '../../../events/types'
import { STYLE } from '../style.config'
import { SPECS } from './specs'

export const PORTRAIT_SIZE = 128
/** Layer every skater mesh is also on; the portrait camera only sees this layer. */
export const PORTRAIT_LAYER = 7

export const usePortraits = create<{ urls: Partial<Record<Seat, string>> }>()(() => ({ urls: {} }))

interface Target { root: Object3D; head: Object3D }
const targets: Partial<Record<Seat, Target>> = {}

/** Called by the rig. Returns an unregister function. */
export function registerPortraitTarget(seat: Seat, t: Target) {
  targets[seat] = t
  return () => { if (targets[seat] === t) delete targets[seat] }
}

// ---------------------------------------------------------------------------------------------

const toSrgb = (() => {
  const lut = new Uint8ClampedArray(256)
  for (let i = 0; i < 256; i++) {
    const c = i / 255
    lut[i] = Math.round(255 * (c <= 0.0031308 ? 12.92 * c : 1.055 * Math.pow(c, 1 / 2.4) - 0.055))
  }
  return lut
})()

function capture(gl: WebGLRenderer, scene: Scene) {
  const S = PORTRAIT_SIZE
  const rt = new WebGLRenderTarget(S, S)
  const cam = new PerspectiveCamera(26, 1, 0.05, 20)
  cam.layers.set(PORTRAIT_LAYER)
  const buf = new Uint8Array(S * S * 4)
  const prevTarget = gl.getRenderTarget()
  const prevClear = gl.getClearColor(new Color()), prevAlpha = gl.getClearAlpha()
  const prevBg = scene.background
  const vis = new Map<Seat, boolean>()
  for (const s of SEATS) { const t = targets[s]; if (t) vis.set(s, t.root.visible) }
  const urls: Partial<Record<Seat, string>> = {}
  const head = new Vector3(), fwd = new Vector3(), side = new Vector3(), up = new Vector3(0, 1, 0)

  try {
    scene.background = null
    gl.setClearColor(0x000000, 0)
    for (const seat of SEATS) {
      const t = targets[seat]
      if (!t) continue
      for (const s of SEATS) { const o = targets[s]; if (o) o.root.visible = s === seat }
      t.root.updateWorldMatrix(true, true)
      const scale = t.root.getWorldScale(new Vector3()).y
      t.head.localToWorld(head.set(0, 0.17, 0))
      t.root.getWorldDirection(fwd); fwd.y = 0; fwd.normalize()
      side.crossVectors(up, fwd).normalize()
      const dir = fwd.clone().multiplyScalar(Math.cos(0.38)).addScaledVector(side, Math.sin(0.38))
      cam.position.copy(head).addScaledVector(dir, 0.95 * scale).addScaledVector(up, 0.04 * scale)
      cam.lookAt(head.x, head.y - 0.1 * scale, head.z)
      cam.updateMatrixWorld()
      gl.setRenderTarget(rt)
      gl.clear(true, true, true)
      gl.render(scene, cam)
      gl.readRenderTargetPixels(rt, 0, 0, S, S, buf)
      urls[seat] = composite(buf, seat)
    }
  } finally {
    for (const [s, v] of vis) { const o = targets[s]; if (o) o.root.visible = v }
    scene.background = prevBg
    gl.setClearColor(prevClear, prevAlpha)
    gl.setRenderTarget(prevTarget)
    rt.dispose()
  }
  usePortraits.setState({ urls })
}

function composite(buf: Uint8Array, seat: Seat): string {
  const S = PORTRAIT_SIZE
  // character layer (flip rows, linear -> sRGB)
  const fig = document.createElement('canvas'); fig.width = fig.height = S
  const fg = fig.getContext('2d')!
  const img = fg.createImageData(S, S)
  for (let y = 0; y < S; y++) {
    for (let x = 0; x < S; x++) {
      const i = ((S - 1 - y) * S + x) * 4, o = (y * S + x) * 4
      img.data[o] = toSrgb[buf[i]]; img.data[o + 1] = toSrgb[buf[i + 1]]; img.data[o + 2] = toSrgb[buf[i + 2]]
      img.data[o + 3] = buf[i + 3] > 8 ? 255 : 0
    }
  }
  fg.putImageData(img, 0, 0)
  // ink silhouette for the outline
  const ink = document.createElement('canvas'); ink.width = ink.height = S
  const ig = ink.getContext('2d')!
  ig.drawImage(fig, 0, 0)
  ig.globalCompositeOperation = 'source-in'
  ig.fillStyle = STYLE.palette.ink; ig.fillRect(0, 0, S, S)

  const out = document.createElement('canvas'); out.width = out.height = S
  const g = out.getContext('2d')!
  const team = TEAM_OF[seat] === 'T1' ? STYLE.palette.T1 : STYLE.palette.T2
  const accent = SPECS[seat].accent
  // flat sunset gradient in the team colour
  const grad = g.createLinearGradient(0, 0, 0, S)
  grad.addColorStop(0, STYLE.world.skyTop)
  grad.addColorStop(0.5, team)
  grad.addColorStop(0.78, '#fff1c8')
  grad.addColorStop(1, team)
  g.fillStyle = grad; g.fillRect(0, 0, S, S)
  g.fillStyle = 'rgba(255,255,255,0.35)'
  for (let k = 0; k < 4; k++) g.fillRect(0, S * (0.6 + k * 0.08), S, 2 + k)
  for (const [dx, dy] of [[-3, 0], [3, 0], [0, -3], [0, 3], [-2, -2], [2, 2], [-2, 2], [2, -2]]) g.drawImage(ink, dx, dy)
  g.drawImage(fig, 0, 0)
  g.lineWidth = 6; g.strokeStyle = accent; g.strokeRect(3, 3, S - 6, S - 6)
  return out.toDataURL('image/png')
}

/** Mounted once inside <Characters />. Renders the portraits a few frames after the rigs appear. */
export function PortraitCapture() {
  const gl = useThree(s => s.gl)
  const scene = useThree(s => s.scene)
  const frame = useRef(0)
  const done = useRef(false)
  useFrame(() => {
    if (done.current) return
    frame.current++
    const ready = SEATS.every(s => targets[s])
    if (frame.current < 5 || (!ready && frame.current < 120)) return
    done.current = true
    try { capture(gl, scene) } catch (err) { console.warn('[characters] portrait capture failed', err) }
  })
  return null
}
