// Post-processing chain (lazy-loaded), in this order:
//   (1) the scene renders at full device-pixel-ratio (capped by STYLE.render.maxDpr, see index.tsx),
//       multisampled by the composer (STYLE.render.msaa samples),
//   (2) bloom, thresholded high so only lamp bulbs / neon / train windows glow,
//   (3) ink edges from the depth buffer (post/InkEdgeEffect.ts) — the cel outline for the set,
//   (4) SMAA on top if MSAA is off (STYLE.render.smaa),
//   (5) an optional gentle vignette.
// The cartoon look has no dither, quantize, grain or pixelation. The mid-2000s look drops the ink and
// adds (6) a console-style grade: crushed blacks, warm balance, ordered dither (post/RetroGradeEffect.ts).
// Every stage is toggleable / tunable in style.config.ts (STYLE.post, STYLE.render).
import { useEffect, useMemo, useRef, type JSX } from 'react'
import { useThree } from '@react-three/fiber'
import { Bloom, EffectComposer, SMAA, Vignette } from '@react-three/postprocessing'
import { type EffectComposer as EffectComposerImpl } from 'postprocessing'
import { RETRO, STYLE } from './style.config'
import { InkEdgeEffect } from './post/InkEdgeEffect'
import { RetroGradeEffect } from './post/RetroGradeEffect'

const P = STYLE.post

export default function Effects() {
  const composer = useRef<EffectComposerImpl>(null)
  const dpr = useThree(s => s.viewport.dpr)
  const size = useThree(s => s.size)
  // The composer only resizes on CSS-size changes by itself, so also resize on dpr changes.
  useEffect(() => { composer.current?.setSize(size.width, size.height) }, [dpr, size])

  const ink = useMemo(() => new InkEdgeEffect({
    color: STYLE.toon.inkColor, width: P.inkWidth, silhouette: P.inkSilhouette, crease: P.inkCrease,
    strength: P.inkStrength, fadeNear: P.inkFadeNear, fadeFar: P.inkFadeFar,
  }), [])
  useEffect(() => { ink.width = P.inkWidth * Math.max(1, dpr) }, [ink, dpr])
  useEffect(() => () => ink.dispose(), [ink])
  const grade = useMemo(() => new RetroGradeEffect(), [])
  useEffect(() => () => grade.dispose(), [grade])

  const fx: JSX.Element[] = []
  if (P.bloom) fx.push(
    <Bloom key="bloom" mipmapBlur intensity={P.bloomIntensity} luminanceThreshold={P.bloomThreshold}
      luminanceSmoothing={P.bloomSmoothing} radius={P.bloomRadius} />,
  )
  if (P.inkEdges) fx.push(<primitive key="ink" object={ink} />)
  if (STYLE.render.smaa && !STYLE.render.msaa) fx.push(<SMAA key="smaa" />)
  if (P.vignette) fx.push(<Vignette key="vignette" offset={P.vignetteOffset} darkness={P.vignetteDarkness} />)
  if (RETRO) fx.push(<primitive key="grade" object={grade} />)

  return (
    <EffectComposer ref={composer} multisampling={STYLE.render.msaa}>
      {fx}
    </EffectComposer>
  )
}
