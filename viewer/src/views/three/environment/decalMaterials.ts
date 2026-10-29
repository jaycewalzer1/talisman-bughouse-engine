// Shared materials for fake ground decals (see BlobShadow.tsx). Separate file so the component
// module only exports components (React fast refresh).
import { AdditiveBlending, Color, ShaderMaterial } from 'three'
import { retroMaterial } from '../materials/retroMaterial'
import { blobTexture } from './textures'
import { STYLE } from '../style.config'

/** Additive "light" decal material (radial falloff); tweak `uniforms.diffuse` to animate it. */
export function lightPoolMaterial(color: string, intensity = 1): ShaderMaterial {
  const m = retroMaterial({
    map: blobTexture(), color: new Color(color).multiplyScalar(intensity * STYLE.scene.lightPoolScale), unlit: true, transparent: true, mipmaps: false,
  })
  m.blending = AdditiveBlending
  m.depthWrite = false
  m.polygonOffset = true; m.polygonOffsetFactor = -3; m.polygonOffsetUnits = -3
  return m
}

/**
 * Faint additive haze cone under a lamp (open cylinder, uv.y = 1 at the top). Fades toward the
 * ground and toward the silhouette edges (view-facing ramp), so it reads as light in damp air.
 */
export function lightConeMaterial(color: string, intensity = 0.07): ShaderMaterial {
  return new ShaderMaterial({
    uniforms: { color: { value: new Color(color).multiplyScalar(intensity * STYLE.scene.lightPoolScale) } },
    vertexShader: /* glsl */`
      varying float vY; varying float vFace;
      void main() {
        vY = uv.y;
        vec3 n = normalize(normalMatrix * normal);
        vec4 mv = modelViewMatrix * vec4(position, 1.0);
        vFace = abs(dot(n, normalize(-mv.xyz)));
        gl_Position = projectionMatrix * mv;
      }
    `,
    fragmentShader: /* glsl */`
      uniform vec3 color; varying float vY; varying float vFace;
      void main() {
        float k = pow(vY, 1.6) * vFace * vFace;
        gl_FragColor = vec4(color * k, 1.0);
        #include <colorspace_fragment>
      }
    `,
    transparent: true, depthWrite: false, blending: AdditiveBlending, fog: false,
  })
}
