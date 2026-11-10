// Move-by-move replay of one finished game inside the dev panel. Shows the normal 2D view, which reads
// the shared game store, and drives that store with a replay source: stepping seeks to a stop
// (analyze.ts replayStops), playing ticks the source like the viewer's controller does.
// Keys: left/right step, Home/End jump, space plays or pauses.

import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { type GameEvent, TEAM_OF } from '../events/types'
import { SPEEDS } from '../sources/controller'
import { createReplaySource } from '../sources/replay'
import { useGame } from '../store/gameStore'
import TwoView from '../views/two'
import { replayStops } from './analyze'

const fmtEval = (v: number) => `${v >= 0 ? '+' : ''}${v.toFixed(2)}`

export default function GameReplay({ events, onCursor }: { events: GameEvent[]; onCursor: (ts: number) => void }) {
  const stops = useMemo(() => replayStops(events), [events])
  const source = useMemo(() => createReplaySource(events), [events])
  const [cur, setCur] = useState(0)
  const [playing, setPlaying] = useState(false)
  const [speed, setSpeed] = useState(1)
  const list = useRef<HTMLOListElement>(null)
  const last = stops.length - 1

  const goto = useCallback((n: number) => {
    const next = Math.max(0, Math.min(last, n))
    setPlaying(false)
    source.seekEvent(stops[next].index)
    setCur(next)
  }, [last, source, stops])

  const togglePlay = useCallback(() => {
    if (!playing && cur >= last) { source.seekEvent(stops[0].index); setCur(0) }
    setPlaying(!playing)
  }, [cur, last, playing, source, stops])

  useEffect(() => {
    if (stops.length) source.seekEvent(stops[0].index)
    return () => useGame.getState().reset()
  }, [source, stops])

  useEffect(() => {
    if (!playing) return
    let raf = 0
    let prev = 0
    const frame = (now: number) => {
      source.tick(prev ? Math.min(250, now - prev) * speed : 0, now)
      prev = now
      const pos = source.position()
      setCur(c => { let n = c; while (n < last && stops[n + 1].ts <= pos) n++; return n })
      if (source.ended()) { setCur(last); setPlaying(false); return }
      raf = requestAnimationFrame(frame)
    }
    raf = requestAnimationFrame(frame)
    return () => cancelAnimationFrame(raf)
  }, [last, playing, source, speed, stops])

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement).tagName
      if (tag === 'INPUT' || tag === 'SELECT' || tag === 'TEXTAREA') return
      if (e.key === 'ArrowRight') goto(cur + 1)
      else if (e.key === 'ArrowLeft') goto(cur - 1)
      else if (e.key === 'Home') goto(0)
      else if (e.key === 'End') goto(last)
      else if (e.key === ' ' && tag !== 'BUTTON') togglePlay()
      else return
      e.preventDefault()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [cur, goto, last, togglePlay])

  const ts = stops[cur]?.ts
  useEffect(() => { if (ts != null) onCursor(ts) }, [onCursor, ts])

  useEffect(() => {
    const box = list.current
    const row = box?.children[cur] as HTMLElement | undefined
    if (!box || !row) return
    if (row.offsetTop < box.scrollTop) box.scrollTop = row.offsetTop
    else if (row.offsetTop + row.offsetHeight > box.scrollTop + box.clientHeight) box.scrollTop = row.offsetTop + row.offsetHeight - box.clientHeight
  }, [cur])

  if (!stops.length) return <p className="dp-empty">This game has no events to replay.</p>

  return (
    <div className="dp-replay">
      <div className="dp-controls" role="group" aria-label="Replay controls">
        <button onClick={() => goto(0)} disabled={cur === 0} aria-label="First move" title="First (Home)">«</button>
        <button onClick={() => goto(cur - 1)} disabled={cur === 0} aria-label="Previous move" title="Previous (←)">‹</button>
        <button className="primary" onClick={togglePlay} title="Play or pause (space)">{playing ? 'Pause' : 'Play'}</button>
        <button onClick={() => goto(cur + 1)} disabled={cur === last} aria-label="Next move" title="Next (→)">›</button>
        <button onClick={() => goto(last)} disabled={cur === last} aria-label="Last move" title="Last (End)">»</button>
        <input type="range" min={0} max={last} value={cur} onChange={e => goto(+e.target.value)} aria-label="Replay position" />
        <span className="dp-stepcount">{cur} / {last}</span>
        <select value={speed} onChange={e => setSpeed(+e.target.value)} aria-label="Playback speed">
          {SPEEDS.map(s => <option key={s} value={s}>{s}x</option>)}
        </select>
      </div>
      <div className="dp-stage">
        <TwoView />
        <ol className="dp-moves" ref={list}>
          {stops.map((s, i) => (
            <li key={i}>
              <button className={i === cur ? 'on' : ''} aria-current={i === cur ? 'step' : undefined} onClick={() => goto(i)}>
                <span className="n">{i}</span>
                <span className={`dp-dot ${s.seat ? (TEAM_OF[s.seat] === 'T1' ? 't1' : 't2') : 'none'}`} />
                <span className="what">{s.label}</span>
                {s.t1Eval != null && <span className="ev" title="Evaluation from T1's side">{fmtEval(s.t1Eval)}</span>}
              </button>
            </li>
          ))}
        </ol>
      </div>
    </div>
  )
}
