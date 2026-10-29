// Entry point of the 3D view (lazy-loaded by App.tsx). Canvas + HTML overlays.
// The scene renders at the full device pixel ratio (capped by STYLE.render.maxDpr), antialiased by
// the composer's MSAA: a clean cel-cartoon frame. No shadow maps, no tone mapping (the palette is
// set directly in the materials).
import { lazy, Suspense, useEffect, useLayoutEffect, useRef, useState } from 'react'
import { Canvas, useFrame, useThree } from '@react-three/fiber'
import { STYLE } from './style.config'
import { Scene } from './Scene'
import { Hud } from './hud/Hud'
import { CommsLog } from './graffiti/CommsLog'
import { useView } from './viewStore'
import { outlineShared } from './materials/outline'
import { retroLight } from './materials/retroMaterial'

const Effects = lazy(() => import('./Effects'))

/** Full device pixel ratio, capped. */
function renderDpr(): number {
  return Math.min(window.devicePixelRatio || 1, STYLE.render.maxDpr)
}

/** Keeps the outline hulls' pixel width and the retro resolution uniform in sync with the buffer. */
function RenderScale() {
  const size = useThree(s => s.size)
  const dpr = useThree(s => s.viewport.dpr)
  useEffect(() => {
    outlineShared.viewHeight.value = size.height * dpr
    retroLight.resolution.value = [size.width * dpr, size.height * dpr]
  }, [size.width, size.height, dpr])
  return null
}

/** While the view is the PiP thumbnail (frameloop "demand"), redraw a few times a second. */
function ThumbTicker({ fps }: { fps: number }) {
  const invalidate = useThree(s => s.invalidate)
  useEffect(() => {
    invalidate()
    const id = window.setInterval(() => invalidate(), 1000 / fps)
    return () => window.clearInterval(id)
  }, [fps, invalidate])
  return null
}

/**
 * r3f restarts its clock (elapsedTime = 0) whenever `frameloop` changes. Lights, the train
 * schedule, characters etc. key off clock.elapsedTime, so carry it across the switch instead.
 * Runs before every other useFrame (lowest priority number; negative = doesn't take over rendering).
 */
function ClockKeeper() {
  const last = useRef(0)
  useFrame(({ clock }, delta) => {
    if (clock.elapsedTime + 1e-6 < last.current) clock.elapsedTime = last.current + Math.min(delta, 0.1)
    last.current = clock.elapsedTime
  }, -1000)
  return null
}

/** Thumbnail redraw rate (fps) while 3D sits in the picture-in-picture corner. */
const THUMB_FPS = 4
/** Matches the App's swap animation: keep full-rate rendering while the canvas is shrinking. */
const THUMB_SETTLE_MS = 320

interface ThreeViewProps {
  /** Shown as the small PiP thumbnail (2D mode): HUD hidden, camera input off, ~4 fps. */
  thumb?: boolean
  /** Thumbnail collapsed / hidden: stop rendering entirely (scene stays mounted). */
  paused?: boolean
}

export default function ThreeView({ thumb = false, paused = false }: ThreeViewProps) {
  const [initialDpr] = useState(renderDpr)
  // low-rate rendering only kicks in once the shrink animation has finished
  const [settled, setSettled] = useState(thumb)
  useEffect(() => {
    if (!thumb) { setSettled(false); return }
    const t = window.setTimeout(() => setSettled(true), THUMB_SETTLE_MS)
    return () => window.clearTimeout(t)
  }, [thumb])
  useLayoutEffect(() => { useView.getState().setThumb(thumb) }, [thumb])
  useEffect(() => () => useView.getState().setThumb(false), [])

  // The canvas is never unmounted or resized for the thumbnail: App.tsx just CSS-scales this whole
  // layer into the corner, so the scene, characters, composer buffers and GL context all survive.
  const low = thumb && settled
  const frameloop = thumb && paused ? 'never' : low ? 'demand' : 'always'
  return (
    <div style={{ position: 'relative', width: '100%', height: '100%', minHeight: 400, background: STYLE.world.skyTop, overflow: 'hidden' }}>
      <Canvas
        flat
        frameloop={frameloop}
        // measure the layout size (offsetWidth/Height), not the transformed on-screen box: the PiP
        // thumbnail is a CSS scale of this layer, and must not shrink the drawing buffer / composer
        resize={{ offsetSize: true }}
        dpr={initialDpr}
        // the composer does the MSAA; the default framebuffer itself needs none
        gl={{ antialias: false, powerPreference: 'high-performance', stencil: false }}
        camera={{ fov: 45, near: 0.05, far: 140, position: [0, 2.6, 6.8] }}
      >
        <RenderScale />
        <ClockKeeper />
        {frameloop === 'demand' && <ThumbTicker fps={THUMB_FPS} />}
        <Scene />
        <Suspense fallback={null}><Effects /></Suspense>
      </Canvas>
      {/* overlays are kept mounted (bubble/announcer state survives) but hidden in the thumbnail */}
      <div style={{ display: thumb ? 'none' : 'contents' }}>
        <Hud />
        <CommsLog />
      </div>
    </div>
  )
}
