// A rounded low-poly skater (merged vertex-coloured lofted parts from skaterParts.ts, shared retro materials),
// posed entirely in code. Implements CharacterRig.
// Joint hierarchy: reach (IK-owned slide) -> root -> hips -> spine -> (head, shoulders -> elbows ->
// wrists -> hands), hips -> legs (IK) ; deck separate.
// Arms: FK pose, then grab/ik.ts two-bone IK whenever the pose asks for it (chin on fist, pockets,
// holding the deck; target smoothed in rig space), then the grab choreography (grab/grabChoreo.ts):
// the right hand reaches, grips, carries and slams the piece, the left plucks captured pieces.
// Hands are jointed procedural meshes (grab/hand.ts) attached at the wrists.
// Win: the skater rides onto the deck and loops their trick (tricks/, mapped in trickPose.ts).
import { useEffect, useLayoutEffect, useMemo, useRef } from 'react'
import { useFrame } from '@react-three/fiber'
import { Group, Quaternion, Vector3 } from 'three'
import type { CharacterRigProps } from './rig'
import { legAbduction, legAngles, poseFor, POSE_KEYS, type Ik, type Pose, type V3 } from './poses'
import {
  buildSkater, disposeSkater, FINISHES, FOREARM, HIP_Y, hipX, shoulderX, skaterMaterials, UPPER_ARM, WRIST,
  type Joint, type PartGeo,
} from './skaterParts'
import { PORTRAIT_LAYER, registerPortraitTarget } from './portraits'
import { BlobShadow } from '../environment/BlobShadow'
import { armPole, solveArm, type ArmChain, type ReachBudget } from '../grab/ik'
import { buildHand } from '../grab/hand'
import { clockNow, grabBus, type GrabSample } from '../grab/grabChoreo'

const BLEND_RATE = 9
const IK_FOLLOW = 14
const DEFAULT_POLE: V3 = [0.7, -0.6, -0.25]   // elbow out, down, a bit back (left arm)
/** Runs before Pieces (priority 0) so a held piece is drawn in this frame's hand. Negative = no
 *  render takeover in R3F. Characters.tsx's driver runs at -2, before this. */
export const RIG_FRAME_PRIORITY = -1
/** Far squares: lean (spine + hips), roll forward on the deck a little, stretch a touch. */
const GRAB_BUDGET = { maxLeanPerJoint: 0.6, maxRootShift: 0.15, maxStretch: 1.08 }

const loggedTris = new Set<string>()
const HAND_TRIS = 198

function PartMeshes({ g }: { g: PartGeo }) {
  const M = skaterMaterials()
  return <>{FINISHES.map(f => g[f] && <mesh key={f} geometry={g[f]!} material={M[f]} />)}</>
}

interface ArmState { p: Vector3; w: number; live: boolean }
const newSample = (): GrabSample => ({
  phase: 'idle', active: false, weight: 0, grip: 0, spread: 0, pinch: 0, holding: false,
  target: new Vector3(), handDir: new Vector3(), palmDir: new Vector3(), flick: 0,
})

export function ProceduralCharacter({ seat, spec, driver, teamColor }: CharacterRigProps) {
  const built = useMemo(() => buildSkater(spec), [spec])
  useEffect(() => () => disposeSkater(built), [built])
  if (import.meta.env.DEV && !loggedTris.has(spec.id)) {
    loggedTris.add(spec.id)
    console.info(`[characters] ${spec.id}: ${built.tris} tris (+${2 * HAND_TRIS} hands), ${built.drawCalls} draw calls (+10 hands)`)
  }
  // rounded jointed hands: fingerless gloves in the (washed) team colour, rings if the look has them
  const hands = useMemo(() => {
    const material = skaterMaterials().matte
    const glove = spec.team || teamColor
    return {
      L: buildHand({ side: 1, skin: spec.skin, glove, ring: spec.rings ? '#c9a24a' : null, material }),
      R: buildHand({ side: -1, skin: spec.skin, glove, ring: spec.rings ? '#b9bcc4' : null, material }),
    }
  }, [spec, teamColor])

  const J = useRef<Partial<Record<Joint, Group | null>>>({})
  const W = useRef<{ wrL: Group | null; wrR: Group | null; palmL: Group | null; palmR: Group | null }>({ wrL: null, wrR: null, palmL: null, palmR: null })
  const root = useRef<Group>(null)
  const reach = useRef<Group>(null)        // IK-owned: slides the skater toward far squares
  const shadow = useRef<Group>(null)
  const set = (k: Joint) => (g: Group | null) => { J.current[k] = g }
  const cur = useRef<Pose | null>(null)
  const S = useMemo(() => ({
    L: { p: new Vector3(), w: 0, live: false } as ArmState,
    R: { p: new Vector3(), w: 0, live: false } as ArmState,
    v: new Vector3(), pole: new Vector3(), sock: new Vector3(), qh: new Quaternion(), qd: new Quaternion(),
    abd: [0.05, -0.05] as [number, number],
    g: newSample(), o: newSample(),
    chains: null as null | { L: ArmChain; R: ArmChain; pL: ArmChain; pR: ArmChain; budget: ReachBudget },
  }), [])
  const SX = shoulderX(spec), HX = hipX(spec)

  useLayoutEffect(() => {
    root.current?.traverse(o => o.layers.enable(PORTRAIT_LAYER))
    if (J.current.deck) J.current.deck.rotation.order = 'YXZ'
    if (root.current && J.current.head) return registerPortraitTarget(seat, { root: root.current, head: J.current.head })
  }, [seat, built])

  // attach hands at the wrists and build the IK chains
  useLayoutEffect(() => {
    const j = J.current, w = W.current
    if (!w.wrL || !w.wrR || !w.palmL || !w.palmR || !j.shL || !j.elL || !j.shR || !j.elR || !j.hips || !j.spine) return
    w.wrL.add(hands.L.root); w.wrR.add(hands.R.root)
    for (const h of [hands.L, hands.R]) h.root.traverse(o => o.layers.enable(PORTRAIT_LAYER))
    S.chains = {
      // grab chains end at the wrist joint (the hand socket is the effector)
      L: { shoulder: j.shL, elbow: j.elL, wrist: w.wrL },
      R: { shoulder: j.shR, elbow: j.elR, wrist: w.wrR },
      // pose chains (chin on fist, pockets, deck holds) aim the middle of the hand like the old fists
      pL: { shoulder: j.shL, elbow: j.elL, wrist: w.palmL },
      pR: { shoulder: j.shR, elbow: j.elR, wrist: w.palmR },
      budget: { ...GRAB_BUDGET, spine: [j.hips, j.spine], rootOffset: reach.current ?? undefined },
    }
    return () => { hands.L.dispose(); hands.R.dispose(); S.chains = null }
  }, [hands, built, S])

  /** IK target -> rig-root-local point (written to `out`). */
  const toRoot = (t: Ik, out: Vector3) => {
    const j = J.current, r = root.current!
    out.set(t.p[0], t.p[1], t.p[2])
    if (t.space === 'root') return out
    if (t.space === 'spine' && j.spine) j.spine.localToWorld(out)
    else if (t.space === 'deck' && j.deck) j.deck.localToWorld(out)
    return r.worldToLocal(out)
  }

  /** Pose-driven hand IK (smoothed target in rig space), run through grab/ik.ts. */
  const poseArm = (side: 1 | -1, st: ArmState, t: Ik, chain: ArmChain, k: number, dt: number) => {
    const j = J.current, r = root.current!
    st.w += (t.w - st.w) * k
    if (t.w > 0) {
      toRoot(t, S.v)
      if (!st.live) { st.p.copy(S.v); st.live = true }
      else st.p.lerp(S.v, 1 - Math.exp(-dt * IK_FOLLOW))
    } else if (st.w < 0.003) st.live = false
    if (!st.live || st.w <= 0.003 || !j.spine) return
    const pole = t.pole ?? DEFAULT_POLE
    S.pole.set(pole[0] * side, pole[1], pole[2]).transformDirection(j.spine.matrixWorld)
    solveArm(chain, { target: r.localToWorld(S.v.copy(st.p)), pole: S.pole, weight: Math.min(1, st.w) }, { maxStretch: 1 })
  }

  useFrame(({ clock }, dtRaw) => {
    const dt = Math.min(dtRaw, 0.1)
    const d = driver.current
    const target = poseFor(d, clock.elapsedTime)
    const j = J.current, r = root.current, ch = S.chains
    if (!r) return
    if (!cur.current) cur.current = structuredClone(target)
    const c = cur.current
    const k = 1 - Math.exp(-dt * BLEND_RATE)
    const lerp3 = (a: V3, b: V3) => { a[0] += (b[0] - a[0]) * k; a[1] += (b[1] - a[1]) * k; a[2] += (b[2] - a[2]) * k }
    for (const key of POSE_KEYS) lerp3(c[key], target[key])
    // tricks are fast and exact: the deck, feet and hips follow them with no smoothing
    const exact = target.snapDeck && d.state === 'win'
    const kk = exact ? 1 : k
    if (target.snapDeck) { c.deckPos = [...target.deckPos]; c.deckRot = [...target.deckRot] }
    c.hipsY += (target.hipsY - c.hipsY) * kk
    c.footL[0] += (target.footL[0] - c.footL[0]) * kk; c.footL[1] += (target.footL[1] - c.footL[1]) * kk
    c.footR[0] += (target.footR[0] - c.footR[0]) * kk; c.footR[1] += (target.footR[1] - c.footR[1]) * kk

    const hipY = HIP_Y + c.hipsY
    r.rotation.y = target.spinY
    j.hips?.position.set(0, hipY, 0); j.hips?.rotation.set(...c.hips)
    j.spine?.rotation.set(...c.spine)
    j.head?.rotation.set(...c.head)
    j.deck?.position.set(...c.deckPos); j.deck?.rotation.set(c.deckRot[0], c.deckRot[1], c.deckRot[2], 'YXZ')
    // legs: sagittal 2-bone IK plus hip abduction for a sideways stance (footX)
    S.abd[0] += (legAbduction(hipY, c.footL[1], target.footX?.[0], 1, HX) - S.abd[0]) * kk
    S.abd[1] += (legAbduction(hipY, c.footR[1], target.footX?.[1], -1, HX) - S.abd[1]) * kk
    const [hl, kl, al] = legAngles(hipY, c.footL, S.abd[0])
    const [hr, kr, ar] = legAngles(hipY, c.footR, S.abd[1])
    j.hipL?.rotation.set(hl, 0, S.abd[0]); j.knL?.rotation.set(kl, 0, 0); j.anL?.rotation.set(al, 0, -S.abd[0])
    j.hipR?.rotation.set(hr, 0, S.abd[1]); j.knR?.rotation.set(kr, 0, 0); j.anR?.rotation.set(ar, 0, -S.abd[1])

    // arms: FK first, then pose IK, then the grab (which may also lean the spine / slide the body)
    if (j.shL && j.elL && j.shR && j.elR) {
      j.shL.rotation.set(...c.shL); j.elL.rotation.set(...c.elL)
      j.shR.rotation.set(...c.shR); j.elR.rotation.set(...c.elR)
      j.shL.scale.set(1, 1, 1); j.shR.scale.set(1, 1, 1)
      // bone offsets the IK may have stretched last frame
      j.elL.position.set(0, -UPPER_ARM, 0); j.elR.position.set(0, -UPPER_ARM, 0)
    }
    W.current.wrL?.position.set(0, -WRIST, 0); W.current.wrR?.position.set(0, -WRIST, 0)
    W.current.wrL?.quaternion.identity(); W.current.wrR?.quaternion.identity()
    reach.current?.position.set(0, 0, 0)
    reach.current?.updateWorldMatrix(true, true)
    if (!ch) return
    poseArm(1, S.L, target.ikL, ch.pL, k, dt)
    poseArm(-1, S.R, target.ikR, ch.pR, k, dt)

    const tn = clockNow()
    const g = grabBus.sample(seat, tn, S.g)
    const o = grabBus.sampleOff(seat, tn, S.o)
    if (g.active && j.hips) {
      S.qh.copy(j.hips.quaternion)
      solveArm(ch.R, {
        target: g.target, effector: hands.R.socket, pole: armPole(r, -1, S.pole),
        handDir: g.handDir, palmDir: g.palmDir, weight: g.weight,
      }, ch.budget)
      // the hips may have leaned: counter-rotate the legs so the feet stay planted
      if (S.qh.angleTo(j.hips.quaternion) > 1e-5) {
        S.qd.copy(j.hips.quaternion).invert().multiply(S.qh)
        j.hipL?.quaternion.premultiply(S.qd); j.hipR?.quaternion.premultiply(S.qd)
      }
    }
    if (o.active) {
      solveArm(ch.L, {
        target: o.target, effector: hands.L.socket, pole: armPole(r, 1, S.pole),
        handDir: o.handDir, palmDir: o.palmDir, weight: o.weight,
      }, { maxStretch: 1.12 })
    }

    // fingers: relaxed / clenched from the pose, grab choreography overrides while active
    const handPose = (h: typeof hands.L, st: ArmState, s: GrabSample) => {
      const base = 0.5 + 0.45 * Math.min(1, st.w)
      if (!s.active) { h.setPose({ grip: base, spread: 0, pinch: 0 }); return }
      const m = s.phase === 'return' ? s.weight : 1
      h.setPose({ grip: base + (s.grip - base) * m, spread: s.spread * m, pinch: s.pinch * m })
    }
    handPose(hands.R, S.R, g)
    handPose(hands.L, S.L, o)

    // tell the grab bus where the hands really are (Pieces draws held pieces there)
    if (g.active) grabBus.reportSocket(seat, 'main', hands.R.socket.getWorldPosition(S.sock), tn)
    else grabBus.reportRest(seat, hands.R.socket.getWorldPosition(S.sock), tn)
    if (o.active) grabBus.reportSocket(seat, 'off', hands.L.socket.getWorldPosition(S.sock), tn)

    if (shadow.current) {
      const lift = Math.max(0, c.hipsY)
      shadow.current.scale.setScalar(Math.max(0.45, 1 - lift * 1.4))
    }
  }, RIG_FRAME_PRIORITY)

  const P = built.parts
  const arm = (side: 1 | -1) => (
    <group ref={set(side > 0 ? 'shL' : 'shR')} position={[side * SX, 0.46, 0]}>
      <PartMeshes g={P[side > 0 ? 'shL' : 'shR']} />
      <group ref={set(side > 0 ? 'elL' : 'elR')} position={[0, -UPPER_ARM, 0]}>
        <PartMeshes g={P[side > 0 ? 'elL' : 'elR']} />
        <group ref={g => { W.current[side > 0 ? 'wrL' : 'wrR'] = g }} position={[0, -WRIST, 0]} />
        <group ref={g => { W.current[side > 0 ? 'palmL' : 'palmR'] = g }} position={[0, -FOREARM, 0]} />
      </group>
    </group>
  )
  const leg = (side: 1 | -1) => (
    <group ref={set(side > 0 ? 'hipL' : 'hipR')} position={[side * HX, 0, 0]}>
      <PartMeshes g={P[side > 0 ? 'hipL' : 'hipR']} />
      <group ref={set(side > 0 ? 'knL' : 'knR')} position={[0, -0.45, 0]}>
        <PartMeshes g={P[side > 0 ? 'knL' : 'knR']} />
        <group ref={set(side > 0 ? 'anL' : 'anR')} position={[0, -0.43, 0]}>
          <PartMeshes g={P[side > 0 ? 'anL' : 'anR']} />
        </group>
      </group>
    </group>
  )

  return (
    <group ref={reach}>
      <group ref={root} scale={spec.height}>
        <group ref={set('hips')} position={[0, HIP_Y, 0]}>
          <PartMeshes g={P.hips} />
          <group ref={set('spine')} position={[0, 0.06, 0]}>
            <PartMeshes g={P.spine} />
            <group ref={set('head')} position={[0, 0.52, 0]}>
              <PartMeshes g={P.head} />
            </group>
            {arm(1)}{arm(-1)}
          </group>
          {leg(1)}{leg(-1)}
        </group>
        <group ref={set('deck')}>
          <PartMeshes g={P.deck} />
        </group>
      </group>
      <group ref={shadow}>
        <BlobShadow radius={0.4 * Math.max(1, spec.bulk) * spec.height} opacity={0.55} />
      </group>
    </group>
  )
}
