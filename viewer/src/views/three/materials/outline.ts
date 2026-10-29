// Inverted-hull ink outlines (cel look). A back-face copy of the mesh is pushed out along its
// normals in clip space by a constant number of screen pixels, filled with flat ink colour.
// Cheap: one extra draw per mesh, shares the geometry, no extra render target.
//
// Usage:  withOutline(mesh)                 // adds a hull child that shares mesh.geometry
//         withOutline(mesh, { width: 1.5 }) // thinner (pixels at 1080p, scaled by drawing-buffer height)
// The post edge pass (post/InkEdgeEffect.ts) draws depth silhouettes for everything else.
import { Color, Mesh, ShaderMaterial, UniformsLib, UniformsUtils, BackSide, type Object3D } from 'three'
import { STYLE } from '../style.config'

/** Shared uniforms: drawing-buffer height (set by index.tsx) so widths stay constant in pixels. */
export const outlineShared = {
  viewHeight: { value: 1080 },
}

const vertexShader = /* glsl */`
  #include <common>
  #include <fog_pars_vertex>
  uniform float width; uniform float viewHeight;
  void main() {
    vec4 localPos = vec4(position, 1.0);
    vec3 n = normal;
    #ifdef USE_INSTANCING
      localPos = instanceMatrix * localPos;
      n = mat3(instanceMatrix) * n;
    #endif
    vec4 mvPosition = modelViewMatrix * localPos;
    vec4 clip = projectionMatrix * mvPosition;
    vec3 vn = normalize(normalMatrix * n);
    vec2 dir = normalize((projectionMatrix * vec4(vn, 0.0)).xy + 1e-6);
    // constant pixel width (relative to a 1080 px tall buffer), thinner far away is fine
    float px = width * (viewHeight / 1080.0);
    clip.xy += dir * px * 2.0 / viewHeight * clip.w;
    // nudge back a little so the hull never pokes through the front faces at creases
    clip.z += 0.0004 * clip.w;
    gl_Position = clip;
    #include <fog_vertex>
  }
`

const fragmentShader = /* glsl */`
  #include <common>
  #include <fog_pars_fragment>
  uniform vec3 ink;
  void main() {
    gl_FragColor = vec4(ink, 1.0);
    #include <colorspace_fragment>
    #include <fog_fragment>
  }
`

const cache = new Map<string, ShaderMaterial>()

export interface OutlineOptions { width?: number; color?: string }

export function outlineMaterial(o: OutlineOptions = {}): ShaderMaterial {
  const width = o.width ?? STYLE.toon.outlineWidth
  const color = o.color ?? STYLE.toon.inkColor
  const key = `${width}|${color}`
  const hit = cache.get(key); if (hit) return hit
  const m = new ShaderMaterial({
    uniforms: UniformsUtils.merge([UniformsLib.fog, {
      width: { value: width },
      ink: { value: new Color(color) },
    }]),
    vertexShader, fragmentShader, side: BackSide, fog: true,
  })
  m.uniforms.viewHeight = outlineShared.viewHeight
  cache.set(key, m)
  return m
}

/** Adds an ink hull as a child of `mesh` (shares its geometry). No-op if outlines are off. */
export function withOutline<T extends Object3D>(obj: T, o: OutlineOptions = {}): T {
  if (!STYLE.toon.outlines) return obj
  const mesh = obj as unknown as Mesh
  if (!mesh.isMesh || !mesh.geometry) return obj
  const hull = new Mesh(mesh.geometry, outlineMaterial(o))
  hull.name = 'inkHull'
  hull.raycast = () => {}
  hull.userData.inkHull = true
  hull.renderOrder = mesh.renderOrder
  mesh.add(hull)
  return obj
}

/** Outline every mesh under `root` (skips hulls, transparent / unlit decals and tiny meshes if asked). */
export function outlineTree(root: Object3D, o: OutlineOptions = {}): void {
  const targets: Mesh[] = []
  root.traverse(c => {
    const m = c as Mesh
    if (!m.isMesh || m.userData.inkHull || m.userData.noOutline) return
    const mat = m.material as { transparent?: boolean }
    if (mat && mat.transparent) return
    targets.push(m)
  })
  for (const m of targets) withOutline(m, o)
}
