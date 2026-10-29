// All chess pieces: on both boards, in the four reserve piles, and in flight.
// Static pieces use 12 instanced meshes (6 types x 2 colors) rebuilt every frame from the store
// (cheap: < 100 matrices). Animated pieces use a small pool of ordinary meshes on top.
// Look: faceted chrome (retroMaterial sphere-map shine): white = pearl chrome, black = dark chrome
// with a cool violet tint (team-agnostic). No shadow maps.
import { useEffect, useMemo } from 'react'
import { useFrame } from '@react-three/fiber'
import { AdditiveBlending, Euler, InstancedMesh, Matrix4, Mesh, PlaneGeometry, Quaternion, Vector3 } from 'three'
import { fenGrid, type BoardId } from '../../../events/types'
import { useGame } from '../../../store/gameStore'
import { BOARD_ROT, BOARD_Y, reserveSlot, squareWorld } from '../layout'
import { PIECE_TYPES, pieceGeometry } from './pieceGeometry'
import { activeAnims, inflight, now, onMove, onPass, reserves, resetAnims, updateAnims } from './anims'
import { grabBus } from '../grab/grabChoreo'
import { onFxReset, onGameEvent } from '../director'
import { retroMaterial } from '../materials/retroMaterial'
import { outlineMaterial, withOutline } from '../materials/outline'
import { STYLE } from '../style.config'

const CAP = 48
const POOL = 14
const BOARDS: BoardId[] = ['A', 'B']
const FILES = 'abcdefgh'

const gridCache = new Map<string, (string | null)[][]>()
function grid(fen: string) {
  let g = gridCache.get(fen)
  if (!g) { if (gridCache.size > 64) gridCache.clear(); g = fenGrid(fen); gridCache.set(fen, g) }
  return g
}
const jitter = (i: number) => (((i * 2654435761) >>> 0) % 1000) / 1000 - 0.5

export function Pieces() {
  const mats = useMemo(() => ({
    w: retroMaterial({ color: '#efe6d2', chrome: 0.3 }),
    b: retroMaterial({ color: '#4a3a5e', chrome: 0.35 }),
  }), [])
  const meshes = useMemo(() => {
    const out: Record<string, InstancedMesh> = {}
    for (const t of PIECE_TYPES) for (const c of ['w', 'b'] as const) {
      const m = new InstancedMesh(pieceGeometry(t), mats[c], CAP)
      m.count = 0; m.frustumCulled = false
      out[c === 'w' ? t.toUpperCase() : t] = m
    }
    return out
  }, [mats])
  // ink hulls for the instanced pieces: same geometry + instance matrices, count mirrored per frame
  const hulls = useMemo(() => {
    if (!STYLE.toon.outlines) return {} as Record<string, InstancedMesh>
    const out: Record<string, InstancedMesh> = {}
    for (const k in meshes) {
      const src = meshes[k]
      const h = new InstancedMesh(src.geometry, outlineMaterial({ width: STYLE.toon.outlineWidth * 0.75 }), CAP)
      h.instanceMatrix = src.instanceMatrix
      h.count = 0; h.frustumCulled = false; h.raycast = () => {}
      out[k] = h
    }
    return out
  }, [meshes])
  const pool = useMemo(() => Array.from({ length: POOL }, () => {
    const m = new Mesh(pieceGeometry('p'), mats.w); m.visible = false
    return withOutline(m, { width: STYLE.toon.outlineWidth * 0.75 })
  }), [mats])

  useEffect(() => {
    resetAnims()
    const offE = onGameEvent(e => { if (e.type === 'move') onMove(e); else if (e.type === 'pass') onPass(e) })
    const offR = onFxReset(resetAnims)
    return () => { offE(); offR() }
  }, [])

  const tmp = useMemo(() => ({ m: new Matrix4(), q: new Quaternion(), e: new Euler(), p: new Vector3(), s: new Vector3(1, 1, 1) }), [])

  useFrame(() => {
    updateAnims()
    const s = useGame.getState()
    const anims = activeAnims()
    for (const k in meshes) meshes[k].count = 0
    const put = (letter: string, pos: Vector3, yaw: number, lie: number) => {
      const m = meshes[letter]; if (!m || m.count >= CAP) return
      tmp.e.set(lie, yaw, 0); tmp.q.setFromEuler(tmp.e)
      m.setMatrixAt(m.count++, tmp.m.compose(pos, tmp.q, tmp.s))
    }
    // pieces on the boards
    for (const b of BOARDS) {
      const hidden = new Set<string>()
      for (const a of anims) if (a.board === b && a.hideSq && a.kind === 'move') hidden.add(a.hideSq)
      const g = grid(s.fens[b])
      for (let r = 0; r < 8; r++) for (let f = 0; f < 8; f++) {
        const L = g[r]?.[f]; if (!L) continue
        const sq = FILES[f] + (8 - r)
        if (hidden.has(sq)) continue
        const white = L === L.toUpperCase()
        squareWorld(b, sq, BOARD_Y, tmp.p)
        put(L, tmp.p, (white ? Math.PI : 0) + BOARD_ROT[b] + jitter(r * 8 + f) * 0.3, 0)
      }
    }
    // reserve piles (minus pieces still flying toward them)
    const res = reserves(s)
    for (const b of BOARDS) {
      const counts: Record<string, number> = {}
      for (const L of res[b]) counts[L] = (counts[L] ?? 0) + 1
      for (const L in counts) {
        const n = counts[L] - inflight(b, L)
        const white = L === L.toUpperCase()
        for (let k = 0; k < n; k++) {
          reserveSlot(b, white ? 'w' : 'b', L, k, tmp.p)
          put(L, tmp.p, jitter(k * 7 + L.charCodeAt(0)) * 2.5, Math.PI / 2)
        }
      }
    }
    for (const k in meshes) {
      meshes[k].instanceMatrix.needsUpdate = true
      if (hulls[k]) hulls[k].count = meshes[k].count
    }
    // animated pieces. Characters run their IK before this (negative useFrame priority) and report
    // where their hands are, so a held piece is drawn in the actual hand.
    let i = 0
    const t = now()
    for (const a of anims) {
      if (i >= POOL) break
      let lie = 0
      if (a.kind === 'move' && a.seat) {
        if (!grabBus.heldBase(a.seat, t, tmp.p)) tmp.p.copy(a.pos)
        // reserve pieces lie on their side in the pile: stand them up as they lift
        if (a.drop) lie = (Math.PI / 2) * (1 - Math.min(1, Math.max(0, (t - a.start) / (a.dur * 0.3))))
      } else if (a.kind === 'ghost' && a.seat && grabBus.captureRelease(a.seat)) {
        // the off hand plucks the captured piece and flicks it away (null = already let go)
        if (!grabBus.capturedPos(a.seat, t, tmp.p)) continue
      } else tmp.p.copy(a.pos)
      const m = pool[i++]
      const white = a.letter === a.letter.toUpperCase()
      m.geometry = pieceGeometry(a.letter)
      for (const c of m.children) if ((c as Mesh).isMesh) (c as Mesh).geometry = m.geometry
      m.material = white ? mats.w : mats.b
      m.position.copy(tmp.p)
      const yaw = (white ? Math.PI : 0) + BOARD_ROT[a.board]
      if (a.kind === 'toss') m.rotation.set(a.tilt, a.tilt * 0.4, a.landed ? Math.PI / 2 : 0)
      else m.rotation.set(a.tilt + lie, yaw, 0)
      m.visible = true
    }
    for (; i < POOL; i++) pool[i].visible = false
  })

  return (
    <group>
      {Object.entries(meshes).map(([k, m]) => <primitive key={k} object={m} />)}
      {Object.entries(hulls).map(([k, m]) => <primitive key={`ink${k}`} object={m} />)}
      {pool.map((m, i) => <primitive key={`pool${i}`} object={m} />)}
      <LastMoveMarks />
    </group>
  )
}

/** Faint highlight on the last move's from/to squares on each board. */
function LastMoveMarks() {
  const marks = useMemo(() => {
    const geo = new PlaneGeometry(0.088, 0.088).rotateX(-Math.PI / 2)
    const mat = retroMaterial({ color: STYLE.palette.yellow, unlit: true, transparent: true, opacity: 0.5, mipmaps: false })
    mat.blending = AdditiveBlending
    mat.depthWrite = false
    return BOARDS.flatMap(() => [new Mesh(geo, mat), new Mesh(geo, mat)])
  }, [])
  useFrame(() => {
    const s = useGame.getState()
    BOARDS.forEach((b, bi) => {
      const mv = s.lastMove[b]
      const [from, to] = marks.slice(bi * 2, bi * 2 + 2)
      if (!mv) { from.visible = to.visible = false; return }
      const fromSq = mv.isDrop ? null : mv.uci.slice(0, 2)
      const toSq = mv.isDrop ? mv.uci.slice(-2) : mv.uci.slice(2, 4)
      squareWorld(b, toSq, BOARD_Y + 0.001, to.position); to.visible = true
      if (fromSq) { squareWorld(b, fromSq, BOARD_Y + 0.001, from.position); from.visible = true } else from.visible = false
    })
  })
  return (
    <group>
      {marks.map((m, i) => <primitive key={i} object={m} />)}
    </group>
  )
}
