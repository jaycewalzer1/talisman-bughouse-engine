// The visible body of a skater in the mid-2000s look: a CC0 rigged human (public/models, see ASSETS.md),
// dressed by sculpt.ts + texture.ts + accessories.ts, and driven by the existing procedural rig.
//
// The rig keeps doing everything it always did (poses, reactions, grab IK, tricks) on its invisible
// joint groups. Every frame, after the IK, this copies the result onto the glTF skeleton:
//   hips / spine / neck / head / legs: rotations; the spine's bend is shared over the 3 spine bones
//   arms: rotations + joint positions, so hands land exactly where the IK put them (the bones stretch a
//         few percent, since the rig's arms are slightly longer). The T-pose arm is first re-based
//         onto the rig's rest pose (hanging, palm forward).
//   fingers: curled from the rig's hand pose (grip / spread / pinch).
import { useEffect, useMemo } from 'react'
import { useFrame, useLoader } from '@react-three/fiber'
import {
  type Bone, Group, ImageLoader, Matrix4, Mesh, MeshPhongMaterial, Quaternion, type SkinnedMesh, Vector3, Color,
  SRGBColorSpace, TextureLoader,
} from 'three'
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js'
import { clone as cloneSkinned } from 'three/examples/jsm/utils/SkeletonUtils.js'
import type { Seat } from '../../../../events/types'
import type { Joint } from '../skaterParts'
import { PORTRAIT_LAYER } from '../portraits'
import type { HandPose } from '../../grab/hand'
import { attachRigid, buildCollar, buildHair, buildHat, buildShoe, buildWaist, fitHead } from './accessories'
import { BODIES, OUTFITS } from './outfits'
import { CLASSES, sculptBody } from './sculpt'
import { paintSkaterTexture } from './texture'

const url = (p: string) => `${import.meta.env.BASE_URL}models/${p}`
/** Runs after the rig (-1), before Pieces (0). */
export const SKIN_FRAME_PRIORITY = -0.5

/** T-pose -> rig rest pose (arm hanging, palm forward): +x -> -y, -y -> +z, for the left arm. */
function restRebase(side: 1 | -1): Quaternion {
  if (side > 0) return new Quaternion().setFromRotationMatrix(new Matrix4().makeBasis(new Vector3(0, -1, 0), new Vector3(0, 0, -1), new Vector3(1, 0, 0)))
  return new Quaternion().setFromRotationMatrix(new Matrix4().makeBasis(new Vector3(0, 1, 0), new Vector3(0, 0, -1), new Vector3(-1, 0, 0)))
}

interface Link { bone: Bone; parent: number; restLocal: Quaternion; restPos: Vector3; restWorld: Quaternion; restWorldPos: Vector3; q: Quaternion; p: Vector3 }
interface Curl { bone: Bone; rest: Quaternion; axis: Vector3; k: number; thumb: boolean; spreadAxis: Vector3; spread: number; side: 1 | -1 }

export interface SkinProxies {
  joints: Partial<Record<Joint, Group | null>>
  wrists: { wrL: Group | null; wrR: Group | null }
  handPoses: { L: Required<HandPose>; R: Required<HandPose> }
  hipY: number
}

function buildScene(seat: Seat, gltf: { scene: Group }, skinImage: HTMLImageElement, eyeMap: MeshPhongMaterial['map']) {
  const o = OUTFITS[seat], b = BODIES[o.body]
  const t0 = performance.now()
  const scene = cloneSkinned(gltf.scene) as Group
  scene.updateMatrixWorld(true)
  const meshes: SkinnedMesh[] = []
  scene.traverse(c => { const m = c as SkinnedMesh; if (m.isSkinnedMesh) meshes.push(m) })
  const bodyMesh = meshes.reduce((a, c) => c.geometry.attributes.position.count > a.geometry.attributes.position.count ? c : a)
  const skeleton = bodyMesh.skeleton
  const byName = new Map(skeleton.bones.map(x => [x.name, x]))

  const sc = sculptBody(bodyMesh.geometry, skeleton, o, b)
  const map = paintSkaterTexture(sc, o, b, skinImage)
  // the texture already has the shading painted in, so it also glows a little by itself: dark corners of
  // the set must not turn the clothes black
  const cloth = new MeshPhongMaterial({ map, emissiveMap: map, emissive: new Color('#ffffff'), emissiveIntensity: 0.1, shininess: 5, specular: new Color('#141414'), side: 2 })
  // skin: Phong with a little plastic sheen
  const skin = new MeshPhongMaterial({ map, emissiveMap: map, emissive: new Color('#ffffff'), emissiveIntensity: 0.16, shininess: 34, specular: new Color('#4e4038') })
  bodyMesh.geometry = sc.geometry
  bodyMesh.material = [cloth, skin]

  for (const m of meshes) {
    m.frustumCulled = false
    if (m === bodyMesh) continue
    const eye = m.material && (m.material as { name?: string }).name === 'MI_Eyes'
    m.material = eye
      ? new MeshPhongMaterial({ map: eyeMap, shininess: 80, specular: new Color('#aaaaaa') })
      : new MeshPhongMaterial({ color: new Color(o.hair), shininess: 2 })
  }

  // rigid accessories
  const world = (name: string) => byName.get(name)!.matrixWorld
  const add = (name: string, part: { geometry: Mesh['geometry']; material: MeshPhongMaterial } | null) => {
    if (!part) return
    attachRigid(new Mesh(part.geometry, part.material), byName.get(name)!, world(name))
  }
  const fit = fitHead(sc.rest, sc.cls, CLASSES.indexOf('head'), b)
  add('Head', buildHair(o, fit))
  add('Head', buildHat(o, fit))
  add('foot_l', buildShoe(o, 1))
  add('foot_r', buildShoe(o, -1))
  add('pelvis', buildWaist(o, b))
  add('spine_03', buildCollar(o, b))
  scene.traverse(c => c.layers.enable(PORTRAIT_LAYER))

  // retarget tables
  const names = ['pelvis', 'spine_01', 'spine_02', 'spine_03', 'neck_01', 'Head', 'clavicle_l', 'upperarm_l', 'lowerarm_l', 'hand_l', 'clavicle_r', 'upperarm_r', 'lowerarm_r', 'hand_r', 'thigh_l', 'calf_l', 'foot_l', 'thigh_r', 'calf_r', 'foot_r']
  const index = new Map(names.map((n, i) => [n, i]))
  const links: Link[] = names.map(n => {
    const bone = byName.get(n)!
    const restWorld = new Quaternion(), restWorldPos = new Vector3()
    bone.getWorldQuaternion(restWorld); bone.getWorldPosition(restWorldPos)
    return { bone, parent: index.get(bone.parent?.name ?? '') ?? -1, restLocal: bone.quaternion.clone(), restPos: bone.position.clone(), restWorld, restWorldPos, q: new Quaternion(), p: new Vector3() }
  })
  const rootBone = byName.get('root')!
  const rootWorld = rootBone.getWorldQuaternion(new Quaternion()), rootPos = rootBone.getWorldPosition(new Vector3())

  // fingers
  const curls: Curl[] = []
  const tmpQ = new Quaternion()
  for (const [suffix, side] of [['l', 1], ['r', -1]] as const) for (const finger of ['index', 'middle', 'ring', 'pinky', 'thumb']) {
    const spread = { index: 1, middle: 0.3, ring: -0.3, pinky: -1, thumb: 0 }[finger]!
    for (let n = 1; n <= 3; n++) {
      const bone = byName.get(`${finger}_0${n}_${suffix}`)
      if (!bone) continue
      bone.getWorldQuaternion(tmpQ)
      const inv = tmpQ.clone().invert()
      // T-pose: fingers along +-x, palm toward -y, so they curl about -+z
      const axis = new Vector3(0, 0, -side).applyQuaternion(inv).normalize()
      const spreadAxis = new Vector3(0, side, 0).applyQuaternion(inv).normalize()
      curls.push({ bone, rest: bone.quaternion.clone(), axis, spreadAxis, spread: n === 1 ? spread : 0, k: finger === 'thumb' ? [0.5, 0.45, 0.5][n - 1] : [1.25, 1.45, 0.85][n - 1], thumb: finger === 'thumb', side })
    }
  }
  let tris = 0
  scene.traverse(c => { const m = c as Mesh; if (m.isMesh) tris += (m.geometry.index?.count ?? m.geometry.attributes.position.count) / 3 })
  if (import.meta.env.DEV) console.info(`[retro] ${seat}: ${tris} tris in all, dressed in ${(performance.now() - t0).toFixed(0)} ms`)
  return { scene, links, index, curls, rootWorld, rootPos, b }
}

const _q = new Quaternion(), _q2 = new Quaternion(), _qs = new Quaternion(), _v = new Vector3(), _v2 = new Vector3()
const IDENT = new Quaternion()

export function StreetSkin({ seat, proxies }: { seat: Seat; proxies: { current: SkinProxies } }) {
  const o = OUTFITS[seat]
  const gltf = useLoader(GLTFLoader, url(o.body === 'male' ? 'skater_male.glb' : 'skater_female.glb'))
  const image = useLoader(ImageLoader, url(o.body === 'male' ? 'skin_male.png' : 'skin_female.png'))
  const eye = useLoader(TextureLoader, url('eye.png'))
  const built = useMemo(() => {
    eye.colorSpace = SRGBColorSpace; eye.flipY = false
    return buildScene(seat, gltf as unknown as { scene: Group }, image, eye)
  }, [seat, gltf, image, eye])
  useEffect(() => () => {
    built.scene.traverse(c => {
      const m = c as Mesh
      if (m.isMesh && m.geometry) m.geometry.dispose()
    })
  }, [built])
  const rebase = useMemo(() => ({ L: restRebase(1), R: restRebase(-1) }), [])

  useFrame(() => {
    const px = proxies.current, j = px.joints, w = px.wrists
    if (!j.hips || !j.spine || !j.head || !j.shL || !j.elL || !j.shR || !j.elR || !j.hipL || !j.knL || !j.anL || !j.hipR || !j.knR || !j.anR || !w.wrL || !w.wrR) return
    const { links: L, index: I, b, rootPos } = built
    const S = 1 / b.scale
    // rig-root-space joint rotations + positions, from the local chain
    const qHips = j.hips.quaternion, pHips = j.hips.position
    const qSpine = _q.copy(qHips).multiply(j.spine.quaternion).clone()
    const pSpine = _v.copy(j.spine.position).applyQuaternion(qHips).add(pHips).clone()

    const setBone = (i: number, wq: Quaternion, wp?: Vector3) => {
      const l = L[i], par = l.parent >= 0 ? L[l.parent] : null
      const pq = par ? par.q : built.rootWorld, pp = par ? par.p : rootPos
      _q2.copy(pq).invert()
      l.bone.quaternion.copy(_q2).multiply(wq)
      if (wp) l.bone.position.copy(_v2.copy(wp).sub(pp).applyQuaternion(_q2))
      l.q.copy(wq)
      if (wp) l.p.copy(wp)
      else l.p.copy(l.restPos).applyQuaternion(pq).add(pp)
    }
    const rebased = (rig: Quaternion, l: Link, out = new Quaternion()) => out.copy(rig).multiply(l.restWorld)

    // pelvis follows the hips' offset from their rest height
    const pel = L[I.get('pelvis')!]
    const h0 = _v2.set(0, px.hipY * S, pel.restWorldPos.z)   // the rig's hips pivot, where the pelvis swings from
    const pelP = new Vector3().subVectors(pel.restWorldPos, h0).applyQuaternion(qHips).add(h0)
    pelP.x += pHips.x * S; pelP.y += (pHips.y - px.hipY) * S; pelP.z += pHips.z * S
    setBone(I.get('pelvis')!, rebased(qHips, pel), pelP)

    // spine shared over three bones
    const sp = j.spine.quaternion
    const spineStep = (name: string, k: number) => {
      const l = L[I.get(name)!]
      _qs.copy(IDENT).slerp(sp, k)
      setBone(I.get(name)!, rebased(new Quaternion().copy(qHips).multiply(_qs), l))
    }
    spineStep('spine_01', 0.34); spineStep('spine_02', 0.67); spineStep('spine_03', 1)

    // neck + head
    const hd = j.head.quaternion
    const neck = L[I.get('neck_01')!]
    _qs.copy(IDENT).slerp(hd, 0.4)
    setBone(I.get('neck_01')!, rebased(new Quaternion().copy(qSpine).multiply(_qs), neck))
    setBone(I.get('Head')!, rebased(new Quaternion().copy(qSpine).multiply(hd), L[I.get('Head')!]))

    // arms
    const arm = (suffix: 'l' | 'r', sh: Group, el: Group, wr: Group, rb: Quaternion) => {
      const clav = I.get(`clavicle_${suffix}`)!
      setBone(clav, rebased(new Quaternion().copy(qSpine), L[clav]))
      const qS = new Quaternion().copy(qSpine).multiply(sh.quaternion)
      const pS = new Vector3().copy(sh.position).applyQuaternion(qSpine).add(pSpine)
      const qE = new Quaternion().copy(qS).multiply(el.quaternion)
      const pE = new Vector3().copy(el.position).applyQuaternion(qS).add(pS)
      const qW = new Quaternion().copy(qE).multiply(wr.quaternion)
      const pW = new Vector3().copy(wr.position).applyQuaternion(qE).add(pE)
      const off = new Vector3(0, 0, -built.b.zOffset * S)
      const go = (name: string, q: Quaternion, p: Vector3) => {
        const i = I.get(`${name}_${suffix}`)!
        setBone(i, new Quaternion().copy(q).multiply(rb).multiply(L[i].restWorld), p.multiplyScalar(S).add(off))
      }
      go('upperarm', qS, pS); go('lowerarm', qE, pE); go('hand', qW, pW)
    }
    arm('l', j.shL, j.elL, w.wrL, rebase.L)
    arm('r', j.shR, j.elR, w.wrR, rebase.R)

    // legs: rotations only, using the model's own bone lengths
    const leg = (suffix: 'l' | 'r', hip: Group, kn: Group, an: Group) => {
      const q1 = new Quaternion().copy(qHips).multiply(hip.quaternion)
      const q2 = new Quaternion().copy(q1).multiply(kn.quaternion)
      const q3 = new Quaternion().copy(q2).multiply(an.quaternion)
      setBone(I.get(`thigh_${suffix}`)!, rebased(q1, L[I.get(`thigh_${suffix}`)!]))
      setBone(I.get(`calf_${suffix}`)!, rebased(q2, L[I.get(`calf_${suffix}`)!]))
      setBone(I.get(`foot_${suffix}`)!, rebased(q3, L[I.get(`foot_${suffix}`)!]))
    }
    leg('l', j.hipL, j.knL, j.anL)
    leg('r', j.hipR, j.knR, j.anR)

    // fingers
    for (const c of built.curls) {
      const h = c.side > 0 ? px.handPoses.L : px.handPoses.R
      const grip = c.thumb ? Math.max(h.grip * 0.7, h.pinch) : h.grip
      c.bone.quaternion.copy(c.rest).multiply(_q.setFromAxisAngle(c.axis, grip * c.k))
      if (c.spread) c.bone.quaternion.multiply(_q.setFromAxisAngle(c.spreadAxis, c.spread * h.spread * 0.22))
    }
  }, SKIN_FRAME_PRIORITY)

  return (
    <group scale={built.b.scale} position={[0, 0, built.b.zOffset]}>
      <primitive object={built.scene} />
    </group>
  )
}

