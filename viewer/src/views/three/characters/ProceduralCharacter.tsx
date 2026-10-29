// A low-poly skater built from boxes, posed entirely in code. Implements CharacterRig.
// Joint hierarchy: hips -> spine -> (head, shoulders -> elbows), hips -> legs (IK) ; deck separate.
import { useLayoutEffect, useMemo, useRef } from 'react'
import { useFrame } from '@react-three/fiber'
import { Group, MeshStandardMaterial, Vector3 } from 'three'
import type { CharacterRigProps } from './rig'
import { legIK, poseFor, POSE_KEYS, type Pose, type V3 } from './poses'
import { gripTexture } from '../environment/textures'

const HIP_Y = 0.92
const BLEND_RATE = 9

type Joint = 'hips' | 'spine' | 'head' | 'shL' | 'elL' | 'shR' | 'elR' | 'hipL' | 'knL' | 'anL' | 'hipR' | 'knR' | 'anR' | 'deck'

function useMats(spec: CharacterRigProps['spec'], team: string) {
  return useMemo(() => {
    const m = (color: string, rough = 0.85) => new MeshStandardMaterial({ color, roughness: rough, flatShading: true })
    return {
      skin: m(spec.skin, 0.7), top: m(spec.top), sleeves: m(spec.sleeves), accent: m(spec.accent), pants: m(spec.pants),
      shoes: m(spec.shoes, 0.6), hat: m(spec.hatColor), deck: m(spec.deck, 0.5), dark: m('#111111', 0.4),
      white: m('#e8e6df', 0.6), team: new MeshStandardMaterial({ color: team, emissive: team, emissiveIntensity: 0.35, flatShading: true }),
      metal: new MeshStandardMaterial({ color: '#b9b9b9', metalness: 0.9, roughness: 0.3 }),
      gold: new MeshStandardMaterial({ color: '#d9b440', metalness: 0.9, roughness: 0.25 }),
      grip: new MeshStandardMaterial({ map: gripTexture(), roughness: 1 }),
    }
  }, [spec, team])
}

export function ProceduralCharacter({ spec, driver, teamColor }: CharacterRigProps) {
  const M = useMats(spec, teamColor)
  const J = useRef<Partial<Record<Joint, Group | null>>>({})
  const root = useRef<Group>(null)
  const set = (k: Joint) => (g: Group | null) => { J.current[k] = g }
  const cur = useRef<Pose | null>(null)
  const tmp = useMemo(() => ({ v: new Vector3(), stretch: 1 }), [])
  const W = 0.4 * spec.bulk, B = spec.bagginess

  useLayoutEffect(() => {
    root.current?.traverse(o => { (o as { castShadow?: boolean }).castShadow = true })
  }, [])

  useFrame(({ clock }, dtRaw) => {
    const dt = Math.min(dtRaw, 0.1)
    const d = driver.current
    const target = poseFor(d, clock.elapsedTime)
    const j = J.current
    // aim the right arm at the piece being moved
    let stretchTarget = 1
    if (d.state === 'moving' && d.reach && j.shR?.parent) {
      const v = j.shR.parent.worldToLocal(tmp.v.copy(d.reach)).sub(j.shR.position)
      const len = v.length() || 1
      target.shR = [Math.atan2(-v.z, -v.y), 0, Math.asin(Math.max(-1, Math.min(1, v.x / len)))]
      target.elR = [-0.1, 0, 0]
      stretchTarget = Math.max(1, Math.min(1.8, len / (0.6 * spec.height)))
    }
    if (!cur.current) cur.current = structuredClone(target)
    const c = cur.current
    const k = 1 - Math.exp(-dt * BLEND_RATE)
    const lerp3 = (a: V3, b: V3, kk = k) => { a[0] += (b[0] - a[0]) * kk; a[1] += (b[1] - a[1]) * kk; a[2] += (b[2] - a[2]) * kk }
    for (const key of POSE_KEYS) lerp3(c[key], target[key], key === 'shR' && d.state === 'moving' ? Math.min(1, k * 2) : k)
    if (d.state === 'win') c.deckRot = [...target.deckRot]       // flipping: no smoothing (angle wraps)
    c.hipsY += (target.hipsY - c.hipsY) * k
    c.footL[0] += (target.footL[0] - c.footL[0]) * k; c.footL[1] += (target.footL[1] - c.footL[1]) * k
    c.footR[0] += (target.footR[0] - c.footR[0]) * k; c.footR[1] += (target.footR[1] - c.footR[1]) * k
    tmp.stretch += (stretchTarget - tmp.stretch) * k

    const hipY = HIP_Y + c.hipsY
    j.hips?.position.set(0, hipY, 0); j.hips?.rotation.set(...c.hips)
    j.spine?.rotation.set(...c.spine)
    j.head?.rotation.set(...c.head)
    j.shL?.rotation.set(...c.shL); j.elL?.rotation.set(...c.elL)
    j.shR?.rotation.set(...c.shR); j.elR?.rotation.set(...c.elR)
    j.shR?.scale.set(1, tmp.stretch, 1)
    const [hl, kl] = legIK(hipY, c.footL[0], c.footL[1])
    const [hr, kr] = legIK(hipY, c.footR[0], c.footR[1])
    j.hipL?.rotation.set(hl, 0, 0.05); j.knL?.rotation.set(kl, 0, 0); j.anL?.rotation.set(-(hl + kl), 0, 0)
    j.hipR?.rotation.set(hr, 0, -0.05); j.knR?.rotation.set(kr, 0, 0); j.anR?.rotation.set(-(hr + kr), 0, 0)
    j.deck?.position.set(...c.deckPos); j.deck?.rotation.set(...c.deckRot)
  })

  const arm = (side: 1 | -1) => (
    <group ref={set(side > 0 ? 'shL' : 'shR')} position={[side * (W / 2 + 0.06), 0.47, 0]}>
      <mesh position={[0, -0.14, 0]} material={M.sleeves}><boxGeometry args={[0.13, 0.31, 0.14]} /></mesh>
      {side > 0 && <mesh position={[0, -0.1, 0]} material={M.team}><boxGeometry args={[0.14, 0.05, 0.15]} /></mesh>}
      <group ref={set(side > 0 ? 'elL' : 'elR')} position={[0, -0.29, 0]}>
        <mesh position={[0, -0.13, 0]} material={spec.shortSleeves ? M.skin : M.sleeves}><boxGeometry args={[0.11, 0.27, 0.11]} /></mesh>
        <mesh position={[0, -0.3, 0.01]} material={M.skin}><boxGeometry args={[0.085, 0.09, 0.07]} /></mesh>
      </group>
    </group>
  )

  const leg = (side: 1 | -1) => (
    <group ref={set(side > 0 ? 'hipL' : 'hipR')} position={[side * 0.1, 0, 0]}>
      <mesh position={[0, -0.22, 0]} material={M.pants}><boxGeometry args={[0.15 * B, 0.46, 0.16 * B]} /></mesh>
      <group ref={set(side > 0 ? 'knL' : 'knR')} position={[0, -0.45, 0]}>
        {spec.shorts ? (
          <>
            <mesh position={[0, -0.04, 0]} material={M.pants}><boxGeometry args={[0.16 * B, 0.1, 0.17 * B]} /></mesh>
            <mesh position={[0, -0.18, 0]} material={M.skin}><boxGeometry args={[0.09, 0.22, 0.09]} /></mesh>
            <mesh position={[0, -0.35, 0]} material={M.white}><boxGeometry args={[0.1, 0.16, 0.1]} /></mesh>
          </>
        ) : (
          <mesh position={[0, -0.2, 0]} material={M.pants}><boxGeometry args={[0.16 * B, 0.42, 0.17 * B]} /></mesh>
        )}
        <group ref={set(side > 0 ? 'anL' : 'anR')} position={[0, -0.43, 0]}>
          <mesh position={[0, 0, 0.05]} material={M.shoes}><boxGeometry args={[0.13, 0.08, 0.28]} /></mesh>
          <mesh position={[0, -0.035, 0.05]} material={M.white}><boxGeometry args={[0.135, 0.02, 0.285]} /></mesh>
        </group>
      </group>
    </group>
  )

  const hat = () => {
    switch (spec.hat) {
      case 'hood': return (
        <group position={[0, 0.15, -0.02]}>
          <mesh position={[0, 0.15, 0]} material={M.hat}><boxGeometry args={[0.3, 0.06, 0.29]} /></mesh>
          <mesh position={[0.135, 0, 0]} material={M.hat}><boxGeometry args={[0.04, 0.3, 0.29]} /></mesh>
          <mesh position={[-0.135, 0, 0]} material={M.hat}><boxGeometry args={[0.04, 0.3, 0.29]} /></mesh>
          <mesh position={[0, 0, -0.13]} material={M.hat}><boxGeometry args={[0.3, 0.3, 0.04]} /></mesh>
        </group>
      )
      case 'beanie': return (
        <group>
          <mesh position={[0, 0.3, 0]} material={M.hat}><boxGeometry args={[0.24, 0.12, 0.24]} /></mesh>
          <mesh position={[0, 0.25, 0]} material={M.dark}><boxGeometry args={[0.25, 0.04, 0.25]} /></mesh>
          <mesh position={[0, 0.38, 0]} material={M.hat}><boxGeometry args={[0.07, 0.05, 0.07]} /></mesh>
        </group>
      )
      case 'cap-back': return (
        <group>
          <mesh position={[0, 0.29, 0]} material={M.hat}><boxGeometry args={[0.235, 0.08, 0.235]} /></mesh>
          <mesh position={[0, 0.26, -0.17]} material={M.dark}><boxGeometry args={[0.19, 0.015, 0.13]} /></mesh>
        </group>
      )
      case 'bucket': return (
        <group>
          <mesh position={[0, 0.3, 0]} material={M.hat}><boxGeometry args={[0.24, 0.1, 0.24]} /></mesh>
          <mesh position={[0, 0.255, 0]} material={M.hat} rotation-x={0.05}><boxGeometry args={[0.36, 0.015, 0.36]} /></mesh>
        </group>
      )
    }
  }

  return (
    <group ref={root} scale={spec.height}>
      <group ref={set('hips')} position={[0, HIP_Y, 0]}>
        <mesh material={M.pants}><boxGeometry args={[W * 0.85, 0.16, 0.22]} /></mesh>
        <group ref={set('spine')} position={[0, 0.06, 0]}>
          <mesh position={[0, 0.26, 0]} material={M.top}><boxGeometry args={[W, 0.5, 0.25]} /></mesh>
          <mesh position={[0, 0.1, 0.13]} material={M.accent}><boxGeometry args={[W * 0.6, 0.12, 0.02]} /></mesh>
          {spec.shortSleeves && <mesh position={[0, 0.32, 0.126]} material={M.accent}><boxGeometry args={[W, 0.05, 0.01]} /></mesh>}
          {spec.chain && <mesh position={[0, 0.4, 0.13]} material={M.gold}><boxGeometry args={[0.12, 0.02, 0.02]} /></mesh>}
          <group ref={set('head')} position={[0, 0.52, 0]}>
            <mesh position={[0, 0.03, 0]} material={M.skin}><boxGeometry args={[0.1, 0.08, 0.1]} /></mesh>
            <mesh position={[0, 0.16, 0]} material={M.skin}><boxGeometry args={[0.22, 0.25, 0.22]} /></mesh>
            {spec.shades
              ? <mesh position={[0, 0.18, 0.112]} material={M.dark}><boxGeometry args={[0.2, 0.05, 0.01]} /></mesh>
              : [-0.05, 0.05].map(x => <mesh key={x} position={[x, 0.18, 0.111]} material={M.dark}><boxGeometry args={[0.03, 0.035, 0.01]} /></mesh>)}
            {spec.headphones && (
              <group>
                <mesh position={[0, 0.31, 0]} material={M.dark}><boxGeometry args={[0.27, 0.025, 0.04]} /></mesh>
                {[-0.125, 0.125].map(x => <mesh key={x} position={[x, 0.17, 0]} material={M.team}><boxGeometry args={[0.04, 0.1, 0.09]} /></mesh>)}
              </group>
            )}
            {hat()}
          </group>
          {arm(1)}{arm(-1)}
        </group>
        {leg(1)}{leg(-1)}
      </group>
      <group ref={set('deck')}>
        <mesh position={[0, 0.075, 0]} material={M.deck}><boxGeometry args={[0.21, 0.02, 0.62]} /></mesh>
        <mesh position={[0, 0.0861, 0]} rotation-x={-Math.PI / 2} material={M.grip}><planeGeometry args={[0.2, 0.6]} /></mesh>
        {[1, -1].map(z => (
          <mesh key={z} position={[0, 0.095, z * 0.36]} rotation-x={z * 0.35} material={M.deck}><boxGeometry args={[0.2, 0.02, 0.14]} /></mesh>
        ))}
        {[1, -1].map(z => (
          <group key={z} position={[0, 0, z * 0.24]}>
            <mesh position={[0, 0.052, 0]} material={M.metal}><boxGeometry args={[0.17, 0.03, 0.05]} /></mesh>
            {[1, -1].map(x => (
              <mesh key={x} position={[x * 0.085, 0.028, 0]} rotation-z={Math.PI / 2} material={M.white}>
                <cylinderGeometry args={[0.028, 0.028, 0.03, 8]} />
              </mesh>
            ))}
          </group>
        ))}
      </group>
    </group>
  )
}
