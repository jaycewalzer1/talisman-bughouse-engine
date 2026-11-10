// The console-era grade, as one pmndrs/postprocessing Effect at the end of the chain: crushed blacks,
// warm balance, a touch of saturation, then a 4x4 ordered dither into a reduced number of levels per
// channel (16-bit-framebuffer feel, subtle). Works in display (sRGB) space because that's where a
// console's framebuffer lived. Knobs: LOOK in style.config.ts.
import { BlendFunction, Effect } from 'postprocessing'
import { Uniform } from 'three'
import { LOOK } from '../style.config'

const fragmentShader = /* glsl */`
  uniform float crush; uniform float warmth; uniform float saturation; uniform float levels;

  float bayer4(vec2 p) {
    int x = int(mod(p.x, 4.0)), y = int(mod(p.y, 4.0));
    int i = x + y * 4;
    int m[16] = int[16](0, 8, 2, 10, 12, 4, 14, 6, 3, 11, 1, 9, 15, 7, 13, 5);
    return (float(m[i]) + 0.5) / 16.0;
  }
  vec3 toDisplay(vec3 c) {
    return mix(c * 12.92, 1.055 * pow(max(c, vec3(0.0)), vec3(1.0 / 2.4)) - 0.055, step(0.0031308, c));
  }
  vec3 toLinear(vec3 c) {
    return mix(c / 12.92, pow((max(c, vec3(0.0)) + 0.055) / 1.055, vec3(2.4)), step(0.04045, c));
  }

  void mainImage(const in vec4 inputColor, const in vec2 uv, out vec4 outputColor) {
    vec3 c = toDisplay(clamp(inputColor.rgb, 0.0, 1.0));
    c = clamp((c - crush) / (1.0 - crush), 0.0, 1.0);
    c = mix(vec3(dot(c, vec3(0.3, 0.59, 0.11))), c, saturation);
    c *= vec3(1.0 + warmth, 1.0 + warmth * 0.15, 1.0 - warmth);
    c = floor(c * levels + bayer4(gl_FragCoord.xy)) / levels;
    outputColor = vec4(toLinear(clamp(c, 0.0, 1.0)), inputColor.a);
  }
`

export class RetroGradeEffect extends Effect {
  constructor() {
    super('RetroGradeEffect', fragmentShader, {
      blendFunction: BlendFunction.SRC,
      uniforms: new Map<string, Uniform>([
        ['crush', new Uniform(LOOK.crush)],
        ['warmth', new Uniform(LOOK.warmth)],
        ['saturation', new Uniform(LOOK.saturation)],
        ['levels', new Uniform(LOOK.ditherLevels)],
      ]),
    })
  }
}
