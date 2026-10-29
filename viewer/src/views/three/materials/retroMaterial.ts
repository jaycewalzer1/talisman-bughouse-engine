// The one material everything in the 3D view uses: a 2-tone cel (toon) shader in the look of a
// late-90s TV cartoon. Per-pixel N.L from the single key light is cut into a lit band and one
// cool shadow band with a hard (but antialiased) edge; ambient is flat, not a gradient, so colour
// fields stay flat. Vertex colours, smooth-filtered textures, optional (banded) chrome shine,
// emissive and scene fog all still work. STYLE.toon.enabled = false falls back to smooth lighting.
// Ink outlines are separate: materials/outline.ts (hulls) + post/InkEdgeEffect.ts (depth edges).
//
// Usage:  const m = retroMaterial({ color: '#ff7a1a', chrome: 0.6 })
//         <mesh material={m} />   or   <primitive object={m} attach="material" />
// Cache materials (module-level or useMemo) — they're cheap but each one is a GPU program use.

import {
  Color, LinearFilter, LinearMipmapLinearFilter, Matrix3, ShaderMaterial, type Texture, UniformsLib,
  UniformsUtils, Vector3, SRGBColorSpace,
} from 'three'
import { STYLE } from '../style.config'
import { chromeTexture } from './chrome'

export interface RetroOptions {
  color?: string | Color
  map?: Texture | null
  vertexColors?: boolean
  chrome?: number           // 0..1 env-map shine
  emissive?: string | Color
  emissiveIntensity?: number
  transparent?: boolean
  opacity?: number
  side?: 0 | 1 | 2          // FrontSide | BackSide | DoubleSide
  fog?: boolean
  unlit?: boolean           // skip lighting (signs, neon, sky-facing decals)
  repeat?: [number, number] // UV repeat for `map` (else the texture's own offset/repeat is used)
  alphaTest?: number        // discard below this alpha (default 0.04; 0.5 for cut-out fences/crates)
  mipmaps?: boolean         // default true: smooth (trilinear) filtering, no shimmer far away
}

// Shared lighting uniforms: change once, every retro material follows.
export const retroLight = {
  // night: warm sodium key from the work lamp overhead (slightly toward the camera side), cool
  // blue sky ambient, near-black ground bounce. Colours live in STYLE.world.
  keyDir: { value: new Vector3(0.12, 1, 0.38).normalize() },
  keyColor: { value: new Color(STYLE.world.keyColor) },
  skyColor: { value: new Color(STYLE.world.ambientSky) },
  groundColor: { value: new Color(STYLE.world.ambientGround) },
  ambient: { value: STYLE.material.ambient },
  keyStrength: { value: STYLE.material.keyStrength },
  snap: { value: STYLE.material.vertexSnap },
  resolution: { value: [1280, 720] as [number, number] },
  toon: { value: STYLE.toon.enabled ? 1 : 0 },
  toonThreshold: { value: STYLE.toon.threshold },
  toonSoft: { value: STYLE.toon.softness },
  toonShadow: { value: STYLE.toon.shadowTone },
  toonShadowTint: { value: new Color(STYLE.toon.shadowTint) },
}

const vertexShader = /* glsl */`
  #include <common>
  #include <fog_pars_vertex>
  uniform vec3 keyDir; uniform vec3 keyColor; uniform vec3 skyColor; uniform vec3 groundColor;
  uniform float ambient; uniform float keyStrength; uniform float snap; uniform float unlit;
  uniform vec2 resolution; uniform mat3 mapTransform;
  varying vec3 vLight; varying vec2 vUv; varying vec2 vEnvUv; varying vec3 vWN;
  #ifdef USE_VCOLOR
    attribute vec3 color; varying vec3 vColor;
  #endif
  #ifdef USE_INSTANCING_COLOR
    varying vec3 vInstColor;
  #endif
  void main() {
    vUv = (mapTransform * vec3(uv, 1.0)).xy;
    vec4 localPos = vec4(position, 1.0);
    vec3 objNormal = normal;
    #ifdef USE_INSTANCING
      localPos = instanceMatrix * localPos;
      objNormal = mat3(instanceMatrix) * objNormal;
    #endif
    vec3 wn = normalize(mat3(modelMatrix) * objNormal);
    vWN = wn;
    // smooth fallback (toon off): lighting evaluated here, interpolated across the face.
    float nl = max(dot(wn, keyDir), 0.0);
    vec3 hemi = mix(groundColor, skyColor, wn.y * 0.5 + 0.5);
    vLight = mix(hemi * ambient + keyColor * nl * keyStrength, vec3(1.0), unlit);
    // sphere-map coordinates for chrome
    vec3 vn = normalize(normalMatrix * objNormal);
    vEnvUv = vn.xy * 0.5 + 0.5;
    #ifdef USE_VCOLOR
      vColor = color;
    #endif
    #ifdef USE_INSTANCING_COLOR
      vInstColor = instanceColor;
    #endif
    vec4 mvPosition = modelViewMatrix * localPos;
    gl_Position = projectionMatrix * mvPosition;
    if (snap > 0.0) {
      vec2 grid = resolution / max(snap / 160.0, 0.001) * 0.5;
      gl_Position.xy = floor(gl_Position.xy / gl_Position.w * grid + 0.5) / grid * gl_Position.w;
    }
    #include <fog_vertex>
  }
`

const fragmentShader = /* glsl */`
  #include <common>
  #include <fog_pars_fragment>
  uniform vec3 diffuse; uniform float opacity; uniform vec3 emissive; uniform float emissiveIntensity;
  uniform float chrome; uniform sampler2D envMap2; uniform float alphaTest;
  uniform vec3 keyDir; uniform vec3 keyColor; uniform vec3 skyColor; uniform vec3 groundColor;
  uniform float ambient; uniform float keyStrength; uniform float unlit;
  uniform float toon; uniform float toonThreshold; uniform float toonSoft; uniform float toonShadow;
  uniform vec3 toonShadowTint;
  #ifdef USE_RETRO_MAP
    uniform sampler2D map;
  #endif
  varying vec3 vLight; varying vec2 vUv; varying vec2 vEnvUv; varying vec3 vWN;
  #ifdef USE_VCOLOR
    varying vec3 vColor;
  #endif
  #ifdef USE_INSTANCING_COLOR
    varying vec3 vInstColor;
  #endif
  void main() {
    vec4 base = vec4(diffuse, opacity);
    #ifdef USE_VCOLOR
      base.rgb *= vColor;
    #endif
    #ifdef USE_INSTANCING_COLOR
      base.rgb *= vInstColor;
    #endif
    #ifdef USE_RETRO_MAP
      vec4 t = texture2D(map, vUv);
      base *= t;
    #endif
    if (base.a < alphaTest) discard;
    vec3 light = vLight;
    float lit = 1.0;
    if (toon > 0.5 && unlit < 0.5) {
      // two flat tones: lit = flat ambient + key; shadow = lit * tone * cool tint. Hard edge, a
      // hair of smoothstep (screen-space via fwidth) so the band boundary stays antialiased.
      vec3 n = normalize(vWN);
      if (!gl_FrontFacing) n = -n;
      float nl = dot(n, keyDir);
      float aa = max(toonSoft, fwidth(nl) * 0.75);
      lit = smoothstep(toonThreshold - aa, toonThreshold + aa, nl);
      vec3 amb = mix(groundColor, skyColor, 0.62) * ambient;
      vec3 bright = amb + keyColor * keyStrength;
      light = mix(bright * toonShadow * toonShadowTint, bright, lit);
    }
    vec3 col = base.rgb * light;
    if (chrome > 0.0) {
      vec3 env = texture2D(envMap2, vEnvUv).rgb;
      if (toon > 0.5) {
        // cel highlight: one flat glint band instead of a smooth env reflection
        float gl = step(0.62, dot(env, vec3(0.333))) * lit;
        col = mix(col, col * 1.25 + vec3(0.06), chrome * 0.6) + vec3(0.55, 0.52, 0.45) * gl * chrome * 0.35;
      } else {
        col = mix(col, env * (0.35 + base.rgb * 0.9), chrome * 0.75) + env * env * chrome * 0.25;
      }
    }
    col += emissive * emissiveIntensity;
    gl_FragColor = vec4(col, base.a);
    #include <colorspace_fragment>
    #include <fog_fragment>
  }
`

export function retroMaterial(o: RetroOptions = {}): ShaderMaterial {
  const color = o.color instanceof Color ? o.color : new Color(o.color ?? '#ffffff')
  const emissive = o.emissive instanceof Color ? o.emissive : new Color(o.emissive ?? '#000000')
  if (o.map) {
    const mips = o.mipmaps ?? true
    o.map.magFilter = LinearFilter
    o.map.minFilter = mips ? LinearMipmapLinearFilter : LinearFilter
    o.map.anisotropy = 4
    o.map.generateMipmaps = mips
    o.map.colorSpace = SRGBColorSpace
    o.map.needsUpdate = true
  }
  let mapTransform = new Matrix3()
  if (o.map && o.repeat) mapTransform.setUvTransform(0, 0, o.repeat[0], o.repeat[1], 0, 0, 0)
  else if (o.map) { o.map.updateMatrix(); mapTransform = o.map.matrix }
  const m = new ShaderMaterial({
    uniforms: UniformsUtils.merge([UniformsLib.fog, {
      diffuse: { value: color },
      opacity: { value: o.opacity ?? 1 },
      emissive: { value: emissive },
      emissiveIntensity: { value: o.emissiveIntensity ?? (o.emissive ? 1 : 0) },
      chrome: { value: (o.chrome ?? 0) * STYLE.material.chrome },
      envMap2: { value: chromeTexture() },
      map: { value: o.map ?? null },
      unlit: { value: o.unlit ? 1 : 0 },
      alphaTest: { value: o.alphaTest ?? 0.04 },
    }]),
    vertexShader,
    fragmentShader,
    fog: o.fog ?? true,
    transparent: o.transparent ?? (o.opacity !== undefined && o.opacity < 1),
    side: o.side ?? 0,
    defines: {
      ...(o.vertexColors ? { USE_VCOLOR: '' } : {}),
      ...(o.map ? { USE_RETRO_MAP: '' } : {}),
    },
  })
  // Lighting uniforms are shared objects, so editing retroLight.* updates every material.
  Object.assign(m.uniforms, retroLight)
  // Keep the map/envMap textures out of UniformsUtils.merge's clone (it clones textures).
  m.uniforms.map = { value: o.map ?? null }
  m.uniforms.envMap2 = { value: chromeTexture() }
  m.uniforms.mapTransform = { value: mapTransform }
  return m
}

/**
 * Same as retroMaterial. Per-instance colours (InstancedMesh.setColorAt) are picked up by every
 * retro material automatically via three's USE_INSTANCING_COLOR define, which three only sets when
 * the mesh actually has an instanceColor attribute (a custom define here used to reference an
 * undeclared `instanceColor` attribute and fail to compile on meshes without colours).
 */
export function retroInstancedMaterial(o: RetroOptions = {}): ShaderMaterial {
  return retroMaterial(o)
}
