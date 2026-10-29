// Night sky dome that follows the camera: dark navy overhead, a faint orange city glow low on the
// horizon, a few dim low clouds lit from below, the odd star. Plus a ring of far tower silhouettes
// with sparse lit windows (one instanced draw). The horizon band equals the fog colour, so fogged
// geometry melts straight into the sky.
import { useMemo, useRef } from 'react'
import { useFrame } from '@react-three/fiber'
import { BackSide, Color, type Mesh, ShaderMaterial } from 'three'
import { STYLE } from '../style.config'
import { retroMaterial } from '../materials/retroMaterial'
import { rng, windowsTexture } from './textures'
import { StaticInstances, type InstanceXform } from './StaticInstances'

const W = STYLE.world

const vert = /* glsl */`
  varying vec3 vDir;
  void main() {
    vDir = position;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  }
`
const frag = /* glsl */`
  uniform vec3 top; uniform vec3 mid; uniform vec3 glowC; uniform vec3 bottom; uniform vec3 fogC; uniform vec3 cloudC;
  varying vec3 vDir;
  float hash(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
  void main() {
    vec3 d = normalize(vDir);
    float y = d.y;
    float az = atan(d.z, d.x);
    vec3 col;
    if (y >= 0.0) {
      col = mix(glowC, mid, smoothstep(0.02, 0.22, y));       // city glow -> navy
      col = mix(col, top, smoothstep(0.2, 0.75, y));
      // a few dim low clouds, underlit orange-grey by the city
      for (int i = 0; i < 2; i++) {
        float fi = float(i);
        float cy = 0.09 + fi * 0.08 + 0.015 * sin(az * 3.0 + fi * 2.3);
        float band = smoothstep(0.03, 0.0, abs(y - cy)) * smoothstep(-0.2, 0.6, sin(az * 2.0 + fi * 1.9));
        col = mix(col, cloudC, band * 0.6);
      }
      // sparse faint stars high up
      vec2 g = floor(vec2(az * 60.0, y * 60.0));
      float st = step(0.996, hash(g)) * smoothstep(0.3, 0.6, y);
      col += vec3(0.5, 0.55, 0.65) * st * 0.35;
    } else {
      col = mix(fogC, bottom, clamp(-y * 4.0, 0.0, 1.0));
    }
    // melt into the fog colour right at the horizon
    col = mix(col, fogC, 1.0 - smoothstep(0.0, 0.05, abs(y)));
    gl_FragColor = vec4(col, 1.0);
    #include <colorspace_fragment>
  }
`

function skylineItems(): InstanceXform[] {
  const r = rng(404), out: InstanceXform[] = []
  // near-black silhouettes against the city glow; the window texture adds sparse lit windows
  const tints = ['#7a7f8a', '#6a6f7a', '#8a8a90', '#5e6270']
  for (let i = 0; i < 46; i++) {
    if (r() < 0.4) continue           // gaps between towers
    const a = (i / 46) * Math.PI * 2 + r() * 0.08
    const dist = 58 + r() * 18
    const w = 5 + r() * 9, h = 6 + r() * 16 * (r() < 0.2 ? 1.6 : 1), d = 5 + r() * 7
    out.push({
      pos: [Math.cos(a) * dist, h / 2 - 0.5, Math.sin(a) * dist],
      rot: [0, -a + (r() - 0.5) * 0.3, 0],
      scale: [w, h, d],
      color: tints[Math.floor(r() * tints.length)],
    })
  }
  return out
}

export function Sky() {
  const dome = useRef<Mesh>(null)
  const mat = useMemo(() => new ShaderMaterial({
    uniforms: {
      top: { value: new Color(W.skyTop) },
      mid: { value: new Color(W.skyTop).lerp(new Color(W.fogColor), 0.55) },
      glowC: { value: new Color(W.skyHorizon) },
      bottom: { value: new Color(W.skyBottom) },
      fogC: { value: new Color(W.fogColor) },
      cloudC: { value: new Color(W.skyHorizon).lerp(new Color(W.fogColor), 0.4) },
    },
    vertexShader: vert, fragmentShader: frag, side: BackSide, depthWrite: false, fog: false,
  }), [])

  const skyline = useMemo(() => STYLE.scene.skyline ? skylineItems() : [], [])
  const skylineMat = useMemo(() => retroMaterial({ map: windowsTexture(), repeat: [2, 4], unlit: true, fog: false }), [])

  useFrame(({ camera }) => { dome.current?.position.copy(camera.position) })

  return (
    <group>
      <mesh ref={dome} material={mat} renderOrder={-1000} frustumCulled={false}>
        <sphereGeometry args={[100, 24, 16]} />
      </mesh>
      {skyline.length > 0 && (
        <StaticInstances items={skyline} material={skylineMat}>
          <boxGeometry args={[1, 1, 1]} />
        </StaticInstances>
      )}
    </group>
  )
}
