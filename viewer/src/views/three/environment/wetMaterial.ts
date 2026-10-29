// Cheap fake wet-asphalt puddles (instead of the original MeshReflectorMaterial, which rendered the
// whole scene a second time): one transparent plane over the ground, masked to the puddles baked
// into asphaltTexture. Per pixel it computes a rippled reflection vector and adds
//   - a dark night-sky reflection with a fresnel ramp (puddles read glossy at grazing angles),
//   - streaky highlights stretched toward the viewer for each lamp in lamps.ts.
// No render target, no extra scene pass; ~5 dot products per puddle pixel.
import { Color, ShaderMaterial, UniformsLib, UniformsUtils, Vector2, Vector3, type Texture } from 'three'
import { STYLE } from '../style.config'
import { LAMPS } from './lamps'

const N = LAMPS.length

const vertexShader = /* glsl */`
  #include <common>
  #include <fog_pars_vertex>
  uniform vec2 repeatUv;
  varying vec2 vUv; varying vec3 vWorld;
  void main() {
    vUv = uv * repeatUv;
    vec4 wp = modelMatrix * vec4(position, 1.0);
    vWorld = wp.xyz;
    vec4 mvPosition = viewMatrix * wp;
    gl_Position = projectionMatrix * mvPosition;
    #include <fog_vertex>
  }
`

const fragmentShader = /* glsl */`
  #include <common>
  #include <fog_pars_fragment>
  #define NL ${N}
  uniform sampler2D mask;
  uniform vec3 lampPos[NL]; uniform vec3 lampCol[NL];
  uniform float flicker; uniform float shine; uniform float time;
  uniform vec3 skyLo; uniform vec3 skyHi;
  varying vec2 vUv; varying vec3 vWorld;
  void main() {
    float a = smoothstep(0.3, 0.7, texture2D(mask, vUv).r);
    if (a < 0.02) discard;
    vec3 V = normalize(vWorld - cameraPosition);
    vec2 rp = vec2(sin(vWorld.x * 9.0 + time * 1.3 + sin(vWorld.z * 7.0)),
                   sin(vWorld.z * 11.0 - time * 1.1 + sin(vWorld.x * 5.0))) * 0.012;
    vec3 Nn = normalize(vec3(rp.x, 1.0, rp.y));
    vec3 R = reflect(V, Nn);
    float fres = 0.3 + 0.7 * pow(1.0 - max(dot(-V, Nn), 0.0), 3.0);
    vec3 col = mix(skyLo, skyHi, clamp(R.y * 1.4, 0.0, 1.0)) * fres;
    for (int i = 0; i < NL; i++) {
      vec3 L = normalize(lampPos[i] - vWorld);
      vec3 dl = R - L;
      float h = dot(dl.xz, dl.xz), v = dl.y * dl.y;
      float s = exp(-h * 700.0 - v * 40.0) + 0.22 * exp(-h * 90.0 - v * 8.0);
      col += lampCol[i] * s * (i == 2 ? flicker : 1.0);
    }
    gl_FragColor = vec4(col * shine, a * 0.88);
    #include <colorspace_fragment>
    #include <fog_fragment>
  }
`

export function wetMaterial(mask: Texture, repeat: number): ShaderMaterial {
  const m = new ShaderMaterial({
    uniforms: UniformsUtils.merge([UniformsLib.fog, {
      repeatUv: { value: new Vector2(repeat, repeat) },
      flicker: { value: 1 },
      shine: { value: STYLE.world.puddleShine },
      time: { value: 0 },
      // what a puddle mirrors: the dark underside of the el / night sky, a hint of city glow low down
      skyLo: { value: new Color(STYLE.world.skyHorizon).multiplyScalar(0.9) },
      skyHi: { value: new Color(STYLE.world.fogColor).multiplyScalar(0.6) },
    }]),
    vertexShader, fragmentShader, fog: true, transparent: true, depthWrite: false,
  })
  m.uniforms.mask = { value: mask }
  m.uniforms.lampPos = { value: LAMPS.map(l => new Vector3(...l.pos)) }
  m.uniforms.lampCol = { value: LAMPS.map(l => new Color(l.color).multiplyScalar(l.power * 1.6)) }
  m.polygonOffset = true; m.polygonOffsetFactor = -1; m.polygonOffsetUnits = -1
  return m
}
