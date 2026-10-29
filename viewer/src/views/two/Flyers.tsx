// Captured pieces visibly fly from the capture square on one board to the partner's pocket on the
// other board. We watch the store's event list for new `pass` events, look up the two DOM
// elements (square + pocket, found via data-* attributes), and animate a glyph along an arc with
// the Web Animations API (element.animate), then remove it.

import { type CSSProperties, useEffect, useRef, useState } from 'react'
import type { GameEvent, PassEvent } from '../../events/types'
import { useGame } from '../../store/gameStore'
import { glyph, isWhitePiece, seatColor } from './util'

interface Flyer { id: number; piece: string; color: string; x0: number; y0: number; x1: number; y1: number }

const FLIGHT_MS = 800
const LIVE_WINDOW_MS = 1500   // ignore passes that are "old news" (e.g. ingested in bulk by a seek)
let flyId = 0

function center(el: Element, origin: DOMRect) {
  const r = el.getBoundingClientRect()
  return { x: r.left + r.width / 2 - origin.left, y: r.top + r.height / 2 - origin.top }
}

function FlyingPiece({ f, onDone }: { f: Flyer; onDone: (id: number) => void }) {
  const ref = useRef<HTMLSpanElement>(null)
  useEffect(() => {
    const el = ref.current
    if (!el) return
    // points on a quadratic bezier arcing above both ends
    const cx = (f.x0 + f.x1) / 2
    const cy = Math.min(f.y0, f.y1) - 140
    const frames: Keyframe[] = []
    for (let i = 0; i <= 12; i++) {
      const t = i / 12
      const x = (1 - t) ** 2 * f.x0 + 2 * (1 - t) * t * cx + t * t * f.x1
      const y = (1 - t) ** 2 * f.y0 + 2 * (1 - t) * t * cy + t * t * f.y1
      const scale = 1 + Math.sin(Math.PI * t) * 0.9
      frames.push({ transform: `translate(${x}px, ${y}px) translate(-50%, -50%) scale(${scale}) rotate(${t * 360}deg)` })
    }
    const anim = el.animate(frames, { duration: FLIGHT_MS, easing: 'ease-in-out', fill: 'forwards' })
    anim.onfinish = () => onDone(f.id)
    return () => anim.cancel()
  }, [f, onDone])
  return (
    <span ref={ref} className={`flyer ${isWhitePiece(f.piece) ? 'pw' : 'pb'}`} style={{ '--trail': f.color } as CSSProperties}>
      {glyph(f.piece)}
    </span>
  )
}

export default function Flyers() {
  const events = useGame(s => s.events)
  const layer = useRef<HTMLDivElement>(null)
  const seen = useRef<{ first: GameEvent | undefined; lastSeq: number }>({ first: undefined, lastSeq: -1 })
  const [flyers, setFlyers] = useState<Flyer[]>([])

  useEffect(() => {
    const st = seen.current
    if (events[0] !== st.first) st.lastSeq = -1           // new game / reset: start over
    st.first = events[0]
    const playhead = useGame.getState().playheadTs
    const origin = layer.current?.getBoundingClientRect()
    const fresh: Flyer[] = []
    for (let i = events.length - 1; i >= 0 && events[i].seq > st.lastSeq; i--) {
      const e = events[i]
      if (e.type !== 'pass' || !origin || playhead - e.ts > LIVE_WINDOW_MS) continue
      const p = e as PassEvent
      // the capture square = destination of the move right before this pass
      const prev = events[i - 1]
      const sq = prev && prev.type === 'move' ? prev.uci.slice(2, 4) : null
      const from = sq ? document.querySelector(`[data-sq="${p.fromBoard}:${sq}"]`) : null
      const to = document.querySelector(`[data-pocket="${p.toSeat}"]`)
      if (!from || !to) continue
      const a = center(from, origin), b = center(to, origin)
      fresh.push({ id: flyId++, piece: p.piece, color: seatColor(p.toSeat), x0: a.x, y0: a.y, x1: b.x, y1: b.y })
    }
    if (events.length) st.lastSeq = events[events.length - 1].seq
    if (fresh.length) setFlyers(fl => [...fl, ...fresh])
  }, [events])

  // stable callback so FlyingPiece's effect doesn't restart
  const done = useRef((id: number) => setFlyers(fl => fl.filter(f => f.id !== id))).current

  return (
    <div className="flyers" ref={layer} aria-hidden>
      {flyers.map(f => <FlyingPiece key={f.id} f={f} onDone={done} />)}
    </div>
  )
}
