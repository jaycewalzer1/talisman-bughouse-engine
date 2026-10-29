// Post-processing stack (lazy-loaded): bloom on lights, chromatic aberration, light pixelation,
// film grain and a heavy vignette. Lower `granularity` / Noise opacity here to tone it down.
import { Bloom, ChromaticAberration, EffectComposer, Noise, Pixelation, Vignette } from '@react-three/postprocessing'
import { BlendFunction } from 'postprocessing'
import { Vector2 } from 'three'

const CA_OFFSET = new Vector2(0.0014, 0.0009)

export default function Effects() {
  return (
    <EffectComposer multisampling={0}>
      <Bloom mipmapBlur intensity={0.85} luminanceThreshold={0.55} luminanceSmoothing={0.25} />
      <ChromaticAberration offset={CA_OFFSET} radialModulation modulationOffset={0.25} />
      <Pixelation granularity={2} />
      <Noise premultiply blendFunction={BlendFunction.SCREEN} opacity={0.32} />
      <Vignette offset={0.22} darkness={0.82} />
    </EffectComposer>
  )
}
