// Live message graffiti: each team message is spray-painted on the wall behind the sender
// (revealed left-to-right with a ragged edge), held for a few seconds, then fades out.
import { useEffect, useMemo, useRef, useState } from 'react'
import { useFrame } from '@react-three/fiber'
import { CanvasTexture, ShaderMaterial, SRGBColorSpace } from 'three'
import { TEAM_OF } from '../../../events/types'
import { onFxReset, onGameEvent } from '../director'
import { seatPose, TEAM_COLOR, WALL_Z } from '../layout'
import { drawGraffiti, isWallWorthy, tokenToWall } from './sprayText'
import * as sfx from '../audio/sfx'

const MAX_TAGS = 8
const PAINT = 0.7, LIFE = 5.0, FADE = 1.2

interface Tag { id: number; x: number; y: number; w: number; rot: number; born: number; mat: ShaderMaterial; tex: CanvasTexture }

const vert = /* glsl */`varying vec2 vUv; void main(){ vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position,1.0); }`
const frag = /* glsl */`
uniform sampler2D map; uniform float reveal; uniform float opacity; varying vec2 vUv;
float h(float n){ return fract(sin(n*91.7)*43758.5); }
void main(){
  vec4 c = texture2D(map, vUv);
  float edge = reveal * 1.15 - 0.075 + (h(floor(vUv.y*40.0)) - 0.5) * 0.06;
  float a = c.a * smoothstep(edge + 0.02, edge - 0.02, vUv.x) * opacity;
  if (a < 0.01) discard;
  gl_FragColor = vec4(c.rgb, a);
  #include <colorspace_fragment>
}`

let nextId = 1

export function SprayWall() {
  const [tags, setTags] = useState<Tag[]>([])
  const live = useRef<Tag[]>([])
  const slots = useMemo(() => ({ T1: 0, T2: 0 }), [])

  useEffect(() => {
    const clear = () => { live.current.forEach(t => { t.tex.dispose(); t.mat.dispose() }); live.current = []; setTags([]) }
    const offR = onFxReset(clear)
    const offE = onGameEvent(e => {
      if (e.type !== 'message' || !isWallWorthy(e)) return
      const team = TEAM_OF[e.fromSeat]
      const color = TEAM_COLOR[team]
      const seed = nextId * 13 + e.seq
      const tex = new CanvasTexture(drawGraffiti(tokenToWall(e.token), color, seed, nextId % 3 === 0 ? 'tag' : 'throwup'))
      tex.colorSpace = SRGBColorSpace
      const mat = new ShaderMaterial({
        uniforms: { map: { value: tex }, reveal: { value: 0 }, opacity: { value: 1 } },
        vertexShader: vert, fragmentShader: frag, transparent: true, depthWrite: false,
      })
      // far-side team paints high, near-side team lower; rotate through a few slots so tags don't stack
      const slot = slots[team]++ % 3
      const sx = seatPose(e.fromSeat).pos.x
      const tag: Tag = {
        id: nextId++, x: sx + (slot - 1) * 1.3 + (Math.random() - 0.5) * 0.3,
        y: (team === 'T1' ? 3.1 : 1.9) + (slot % 2) * 0.35 + (Math.random() - 0.5) * 0.2,
        w: 1.5 + Math.random() * 0.3, rot: (Math.random() - 0.5) * 0.12, born: performance.now() / 1000, mat, tex,
      }
      const all = [...live.current, tag]
      while (all.length > MAX_TAGS) { const old = all.shift()!; old.tex.dispose(); old.mat.dispose() }
      live.current = all
      setTags(all)
      sfx.spray(PAINT + 0.2)
    })
    return () => { offR(); offE(); clear() }
  }, [slots])

  useFrame(() => {
    const t = performance.now() / 1000
    let expired = false
    for (const tag of live.current) {
      const age = t - tag.born
      tag.mat.uniforms.reveal.value = Math.min(1, age / PAINT)
      tag.mat.uniforms.opacity.value = age > LIFE - FADE ? Math.max(0, (LIFE - age) / FADE) : 1
      if (age > LIFE) expired = true
    }
    if (expired) {
      const keep = live.current.filter(tag => t - tag.born <= LIFE)
      live.current.filter(tag => t - tag.born > LIFE).forEach(tag => { tag.tex.dispose(); tag.mat.dispose() })
      live.current = keep
      setTags(keep)
    }
  })

  return (
    <group>
      {tags.map((tag, i) => (
        <mesh key={tag.id} position={[tag.x, tag.y, WALL_Z + 0.03 + i * 0.002]} rotation-z={tag.rot} material={tag.mat} renderOrder={2}>
          <planeGeometry args={[tag.w, tag.w / 2]} />
        </mesh>
      ))}
    </group>
  )
}
