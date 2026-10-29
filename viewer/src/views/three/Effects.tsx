// Post-processing chain (lazy-loaded), in this order:
//   (1) the scene is already rendered at ~STYLE.render.width x height (see index.tsx) and upscaled
//       nearest-neighbour by the browser (CSS image-rendering: pixelated),
//   (2) bloom, thresholded high so only lamp bulbs / neon / train windows glow,
//   (3) film grain (as in the original look),
//   (4) light colour quantization + subtle ordered Bayer dither (post/QuantizeDitherEffect.ts),
//   (5) a heavy vignette (as in the original look).
// Every stage is toggleable / tunable in style.config.ts (STYLE.post).
import { useEffect, useMemo, useRef, type JSX } from 'react'
import { useThree } from '@react-three/fiber'
import { Bloom, EffectComposer, Noise, Vignette } from '@react-three/postprocessing'
import { BlendFunction, type EffectComposer as EffectComposerImpl } from 'postprocessing'
import { STYLE } from './style.config'
import { QuantizeDitherEffect } from './post/QuantizeDitherEffect'

const P = STYLE.post

export default function Effects() {
  const composer = useRef<EffectComposerImpl>(null)
  const dpr = useThree(s => s.viewport.dpr)
  const size = useThree(s => s.size)
  // The internal resolution is driven by dpr (index.tsx). The composer only resizes on CSS-size
  // changes by itself, so also resize its buffers whenever dpr changes.
  useEffect(() => { composer.current?.setSize(size.width, size.height) }, [dpr, size])

  const dither = useMemo(() => new QuantizeDitherEffect({ strength: P.ditherStrength, levels: P.colorLevels }), [])
  useEffect(() => () => dither.dispose(), [dither])

  const fx: JSX.Element[] = []
  if (P.bloom) fx.push(
    <Bloom key="bloom" mipmapBlur intensity={P.bloomIntensity} luminanceThreshold={P.bloomThreshold}
      luminanceSmoothing={P.bloomSmoothing} radius={P.bloomRadius} />,
  )
  if (P.grain) fx.push(<Noise key="grain" premultiply blendFunction={BlendFunction.SCREEN} opacity={P.grainOpacity} />)
  if (P.dither) fx.push(<primitive key="dither" object={dither} />)
  if (P.vignette) fx.push(<Vignette key="vignette" offset={P.vignetteOffset} darkness={P.vignetteDarkness} />)

  return (
    <EffectComposer ref={composer} multisampling={0}>
      {fx}
    </EffectComposer>
  )
}
