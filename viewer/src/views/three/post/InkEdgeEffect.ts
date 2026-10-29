// Ink lines from the depth buffer (cel look), as a pmndrs/postprocessing Effect so it merges into
// the same EffectPass as the vignette. Per pixel it reads linear depth at the centre and 4
// neighbours `width` pixels away and draws ink where
//   - depth jumps (silhouettes: object vs. wall, sky, ground), relative to the centre depth, or
//   - the depth slope bends (creases: box edges, kerbs, steps — a 2nd-difference test).
// Lines fade out with distance so the far skyline stays a soft painted background.
// Characters / pieces also get inverted-hull outlines (materials/outline.ts); this covers the set.
import { BlendFunction, Effect, EffectAttribute } from 'postprocessing'
import { Color, Uniform } from 'three'

const fragmentShader = /* glsl */`
  uniform vec3 ink;
  uniform float width;        // pixels
  uniform float silhouette;   // relative depth jump that counts as an edge
  uniform float crease;       // relative 2nd difference that counts as a crease
  uniform float strength;     // 0..1 line opacity
  uniform float fadeNear; uniform float fadeFar;

  float lin(vec2 uv) {
    float d = readDepth(uv);
    // background (sky dome doesn't write depth): treat as very far
    return d >= 0.9999 ? cameraFar : -getViewZ(d);
  }

  void mainImage(const in vec4 inputColor, const in vec2 uv, out vec4 outputColor) {
    vec2 o = texelSize * width;
    float c = lin(uv);
    float l = lin(uv - vec2(o.x, 0.0)), r = lin(uv + vec2(o.x, 0.0));
    float d = lin(uv - vec2(0.0, o.y)), u = lin(uv + vec2(0.0, o.y));
    float near = min(c, min(min(l, r), min(u, d)));
    // silhouettes: biggest jump to a neighbour, relative to the nearer surface
    float jump = max(max(abs(l - c), abs(r - c)), max(abs(u - c), abs(d - c))) / max(near, 0.05);
    float e = smoothstep(silhouette, silhouette * 1.6, jump);
    // creases: second difference along x / y (flat planes and steady slopes give ~0)
    float k = (abs(l + r - 2.0 * c) + abs(u + d - 2.0 * c)) / max(c, 0.05);
    e = max(e, smoothstep(crease, crease * 1.8, k) * 0.85);
    e *= 1.0 - smoothstep(fadeNear, fadeFar, near);
    outputColor = vec4(mix(inputColor.rgb, ink, e * strength), inputColor.a);
  }
`

export interface InkEdgeOptions {
  color?: string; width?: number; silhouette?: number; crease?: number; strength?: number
  fadeNear?: number; fadeFar?: number
}

export class InkEdgeEffect extends Effect {
  constructor(o: InkEdgeOptions = {}) {
    super('InkEdgeEffect', fragmentShader, {
      blendFunction: BlendFunction.SRC,
      attributes: EffectAttribute.DEPTH,
      uniforms: new Map<string, Uniform>([
        ['ink', new Uniform(new Color(o.color ?? '#14110f'))],
        ['width', new Uniform(o.width ?? 1.5)],
        ['silhouette', new Uniform(o.silhouette ?? 0.06)],
        ['crease', new Uniform(o.crease ?? 0.02)],
        ['strength', new Uniform(o.strength ?? 1)],
        ['fadeNear', new Uniform(o.fadeNear ?? 14)],
        ['fadeFar', new Uniform(o.fadeFar ?? 40)],
      ]),
    })
  }
  /** Line width in drawing-buffer pixels (Effects.tsx scales it with the device pixel ratio). */
  set width(v: number) { this.uniforms.get('width')!.value = v }
}
