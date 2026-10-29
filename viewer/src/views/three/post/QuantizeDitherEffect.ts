// Light colour quantization + ordered 4x4 Bayer dithering (PS2-era framebuffer feel), written as a
// pmndrs/postprocessing Effect so it merges into the same EffectPass as the vignette and runs AFTER
// bloom. Quantizes in sRGB (perceptual) space so dark tones don't band harder than light ones.
// gl_FragCoord is in drawing-buffer pixels, which are the low-res (~640x448) pixels, so the
// pattern lands exactly on the chunky upscaled pixels.
import { BlendFunction, Effect } from 'postprocessing'
import { Uniform } from 'three'

const fragmentShader = /* glsl */`
  uniform float strength;   // 0..1 how much the Bayer threshold offsets each pixel
  uniform float levels;     // steps per channel

  const float BAYER[16] = float[16](
     0.0,  8.0,  2.0, 10.0,
    12.0,  4.0, 14.0,  6.0,
     3.0, 11.0,  1.0,  9.0,
    15.0,  7.0, 13.0,  5.0
  );

  vec3 toSRGB(vec3 c) {
    return mix(c * 12.92, 1.055 * pow(c, vec3(1.0 / 2.4)) - 0.055, step(vec3(0.0031308), c));
  }
  vec3 toLinear(vec3 c) {
    return mix(c / 12.92, pow((c + 0.055) / 1.055, vec3(2.4)), step(vec3(0.04045), c));
  }

  void mainImage(const in vec4 inputColor, const in vec2 uv, out vec4 outputColor) {
    ivec2 p = ivec2(mod(gl_FragCoord.xy, 4.0));
    float threshold = (BAYER[p.y * 4 + p.x] + 0.5) / 16.0 - 0.5;    // -0.47 .. +0.47
    float steps = max(levels - 1.0, 1.0);
    vec3 c = toSRGB(clamp(inputColor.rgb, 0.0, 1.0));
    c = floor(c * steps + 0.5 + threshold * strength) / steps;
    outputColor = vec4(toLinear(clamp(c, 0.0, 1.0)), inputColor.a);
  }
`

export class QuantizeDitherEffect extends Effect {
  constructor({ strength = 0.5, levels = 24 }: { strength?: number; levels?: number } = {}) {
    super('QuantizeDitherEffect', fragmentShader, {
      blendFunction: BlendFunction.SRC,
      uniforms: new Map<string, Uniform>([
        ['strength', new Uniform(strength)],
        ['levels', new Uniform(levels)],
      ]),
    })
  }
  get strength(): number { return this.uniforms.get('strength')!.value }
  set strength(v: number) { this.uniforms.get('strength')!.value = v }
  get levels(): number { return this.uniforms.get('levels')!.value }
  set levels(v: number) { this.uniforms.get('levels')!.value = v }
}
