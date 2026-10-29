// All chess pieces: on both boards, in the four reserve piles, and in flight.
// Static pieces use 12 instanced meshes (6 types x 2 colors) rebuilt every frame from the store
// (cheap: < 100 matrices). Animated pieces use a small pool of ordinary meshes on top.
import { useEffect, useMemo } from 'react'
import { useFrame } from '@react-three/fiber'
import { Euler, InstancedMesh, Matrix4, Mesh, MeshStandardMaterial, PlaneGeometry, Quaternion, Vector3 } from 'three'
import { fenGrid, type BoardId } from '../../../events/types'
import { useGame } from '../../../store/gameStore'
import { BOARD_ROT, BOARD_Y, reserveSlot, squareWorld } from '../layout'
import { PIECE_TYPES, pieceGeometry } from './pieceGeometry'
import { activeAnims, inflight, onMove, onPass, reserves, resetAnims, updateAnims } from './anims'
import { onFxReset, onGameEvent } from '../director'

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
    w: new MeshStandardMaterial({ color: '#efe3c6', roughness: 0.55, flatShading: true }),
    b: new MeshStandardMaterial({ color: '#25202a', roughness: 0.45, metalness: 0.1, flatShading: true, emissive: '#120c18' }),
  }), [])
  const meshes = useMemo(() => {
    const out: Record<string, InstancedMesh> = {}
    for (const t of PIECE_TYPES) for (const c of ['w', 'b'] as const) {
      const m = new InstancedMesh(pieceGeometry(t), mats[c], CAP)
      m.castShadow = true; m.receiveShadow = true; m.count = 0; m.frustumCulled = false
      out[c === 'w' ? t.toUpperCase() : t] = m
    }
    return out
  }, [mats])
  const pool = useMemo(() => Array.from({ length: POOL }, () => {
    const m = new Mesh(pieceGeometry('p'), mats.w); m.castShadow = true; m.visible = false; return m
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
    for (const k in meshes) meshes[k].instanceMatrix.needsUpdate = true
    // animated pieces
    let i = 0
    for (const a of anims) {
      if (i >= POOL) break
      const m = pool[i++]
      const white = a.letter === a.letter.toUpperCase()
      m.geometry = pieceGeometry(a.letter)
      m.material = white ? mats.w : mats.b
      m.position.copy(a.pos)
      const yaw = (white ? Math.PI : 0) + BOARD_ROT[a.board]
      if (a.kind === 'toss') m.rotation.set(a.tilt, a.tilt * 0.4, a.landed ? Math.PI / 2 : 0)
      else m.rotation.set(a.tilt, yaw, 0)
      m.visible = true
    }
    for (; i < POOL; i++) pool[i].visible = false
  })

  return (
    <group>
      {Object.entries(meshes).map(([k, m]) => <primitive key={k} object={m} />)}
      {pool.map((m, i) => <primitive key={`pool${i}`} object={m} />)}
      <LastMoveMarks />
    </group>
  )
}

/** Faint highlight on the last move's from/to squares on each board. */
function LastMoveMarks() {
  const marks = useMemo(() => {
    const geo = new PlaneGeometry(0.088, 0.088).rotateX(-Math.PI / 2)
    const mat = new MeshStandardMaterial({ color: '#ffd23a', transparent: true, opacity: 0.35, emissive: '#ffb000', emissiveIntensity: 0.4, depthWrite: false })
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
