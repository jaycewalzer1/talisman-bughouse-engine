// Dev panel (dev.html): look inside games. Left: the live arena game (and earlier live games this
// session) plus every recorded run under runs/ (served by vite.devApi.ts). Right: the selected game's
// per-seat numbers, the evaluation timeline and the raw event log.

import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { type GameEvent, type Seat, TEAM_OF } from '../events/types'
import { type EvalPoint, type GameSummary, describe, parseJsonl, summarize } from './analyze'

const WS_URL = new URLSearchParams(window.location.search).get('ws') ?? 'ws://localhost:8765'
const MAX_LIVE_HISTORY = 20
const LOG_TYPES = ['game_start', 'thinking', 'move', 'pass', 'message', 'stall', 'game_end'] as const

interface RunInfo { name: string; mtime: number; games: string[]; files: string[] }
type Selection = { kind: 'live' } | { kind: 'past'; index: number } | { kind: 'file'; run: string; file: string }

// ---------------------------------------------------------------- live feed

function useLiveGames() {
  const [current, setCurrent] = useState<GameEvent[]>([])
  const [past, setPast] = useState<GameEvent[][]>([])
  const [status, setStatus] = useState('connecting…')
  const cur = useRef<GameEvent[]>([])
  useEffect(() => {
    let ws: WebSocket | null = null
    let stopped = false
    let retry: ReturnType<typeof setTimeout> | undefined
    const connect = () => {
      ws = new WebSocket(WS_URL)
      ws.onopen = () => setStatus(`live · ${WS_URL}`)
      ws.onclose = () => { if (!stopped) { setStatus('arena not running · retrying'); retry = setTimeout(connect, 2000) } }
      ws.onmessage = msg => {
        const events = parseJsonl(String(msg.data))
        for (const e of events) {
          if (e.type === 'game_start') {
            const finished = cur.current
            if (finished.length) setPast(p => [finished, ...p].slice(0, MAX_LIVE_HISTORY))
            cur.current = [e]
          } else {
            cur.current = [...cur.current, e]
          }
        }
        setCurrent(cur.current)
      }
    }
    connect()
    return () => { stopped = true; clearTimeout(retry); ws?.close() }
  }, [])
  return { current, past, status }
}

// ---------------------------------------------------------------- recorded runs

function useRuns() {
  const [runs, setRuns] = useState<RunInfo[] | null>(null)
  const [error, setError] = useState<string | null>(null)
  const refresh = useCallback(() => {
    fetch('/api/runs')
      .then(r => (r.ok ? r.json() : Promise.reject(new Error(`HTTP ${r.status}`))))
      .then((data: RunInfo[]) => { setRuns(data); setError(null) })
      .catch(() => setError('Could not list runs/. Start the viewer with npm run dev from viewer/.'))
  }, [])
  useEffect(refresh, [refresh])
  return { runs, error, refresh }
}

function useGameFile(sel: Selection) {
  const [events, setEvents] = useState<GameEvent[] | null>(null)
  const [error, setError] = useState<string | null>(null)
  const key = sel.kind === 'file' ? `${sel.run}/${sel.file}` : null
  useEffect(() => {
    if (!key) return
    setEvents(null); setError(null)
    let live = true
    fetch(`/api/runs/${key}`)
      .then(r => (r.ok ? r.text() : Promise.reject(new Error(`HTTP ${r.status}`))))
      .then(t => { if (live) setEvents(parseJsonl(t)) })
      .catch(() => { if (live) setError(`Could not load ${key}.`) })
    return () => { live = false }
  }, [key])
  return { events, error }
}

// ---------------------------------------------------------------- pieces

const fmtClock = (ms: number | null) => {
  if (ms == null) return '–'
  const s = Math.max(0, ms) / 1000
  return `${Math.floor(s / 60)}:${(s % 60).toFixed(1).padStart(4, '0')}`
}
const fmtS = (ms: number) => `${(ms / 1000).toFixed(1)}s`

function SeatTable({ sum }: { sum: GameSummary }) {
  return (
    <div className="dp-scroll">
      <table className="dp-table">
        <thead>
          <tr>
            <th>Seat</th><th>Player</th><th className="num">Moves</th><th className="num">Drops</th>
            <th className="num">Captures</th><th className="num">Searches</th><th className="num">Nodes</th>
            <th className="num">Avg think</th><th className="num">Borrowed</th><th className="num">Sits</th>
            <th className="num">Sat for</th><th className="num">Msgs</th><th className="num">Clock</th>
          </tr>
        </thead>
        <tbody>
          {sum.seats.map(s => (
            <tr key={s.seat} className={TEAM_OF[s.seat] === 'T1' ? 't1' : 't2'}>
              <td><span className="dp-dot" />{s.seat}</td>
              <td className="dp-name" title={`${s.name} (${s.kind})`}>{s.name}</td>
              <td className="num">{s.moves}</td><td className="num">{s.drops}</td><td className="num">{s.captures}</td>
              <td className="num">{s.searches}</td><td className="num">{s.nodes.toLocaleString()}</td>
              <td className="num">{s.avgThinkMs == null ? '–' : `${s.avgThinkMs} ms`}</td>
              <td className="num">{s.borrowed}</td><td className="num">{s.sits}</td>
              <td className="num">{fmtS(s.sitMs)}</td><td className="num">{s.messages}</td>
              <td className="num">{fmtClock(s.clockMs)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

const SEAT_STYLE: Record<Seat, string> = { A_white: 't1', B_black: 't1 dashed', A_black: 't2', B_white: 't2 dashed' }

function EvalChart({ evals, t0, t1 }: { evals: EvalPoint[]; t0: number; t1: number }) {
  const W = 800, H = 220, L = 44, R = 12, T = 12, B = 28
  const span = Math.max(1, t1 - t0)
  const x = (ts: number) => L + ((ts - t0) / span) * (W - L - R)
  const y = (v: number) => T + ((1 - v) / 2) * (H - T - B)
  const ticks = useMemo(() => {
    const stepS = [5, 10, 15, 30, 60, 120][[5, 10, 15, 30, 60, 120].findIndex(s => span / 1000 / s <= 8)] ?? 300
    const out: number[] = []
    for (let s = 0; s * 1000 <= span; s += stepS) out.push(s)
    return out
  }, [span])
  if (!evals.length) return <p className="dp-empty">No evaluations yet. Engine searches log one each time a move lands.</p>
  const bySeat = (['A_white', 'B_black', 'A_black', 'B_white'] as Seat[]).map(seat => ({ seat, pts: evals.filter(p => p.seat === seat) }))
  return (
    <div className="dp-scroll">
      <svg className="dp-chart" viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Evaluation over time, from team T1's side">
        {[-1, -0.5, 0, 0.5, 1].map(v => (
          <g key={v}>
            <line x1={L} x2={W - R} y1={y(v)} y2={y(v)} className={v === 0 ? 'zero' : 'grid'} />
            <text x={L - 6} y={y(v) + 4} textAnchor="end" className="axis">{v > 0 ? `+${v}` : v}</text>
          </g>
        ))}
        {ticks.map(s => (
          <text key={s} x={x(t0 + s * 1000)} y={H - 8} textAnchor="middle" className="axis">{s}s</text>
        ))}
        <text x={W - R} y={T + 10} textAnchor="end" className="axis label">T1 ahead</text>
        <text x={W - R} y={H - B - 4} textAnchor="end" className="axis label">T2 ahead</text>
        {bySeat.map(({ seat, pts }) => pts.length > 0 && (
          <g key={seat} className={`series ${SEAT_STYLE[seat]}`}>
            <polyline fill="none" points={pts.map(p => `${x(p.ts)},${y(p.t1)}`).join(' ')} />
            {pts.map((p, i) => <circle key={i} cx={x(p.ts)} cy={y(p.t1)} r={2.5} />)}
          </g>
        ))}
      </svg>
      <div className="dp-legend">
        {bySeat.map(({ seat }) => <span key={seat} className={SEAT_STYLE[seat]}><i />{seat}</span>)}
      </div>
    </div>
  )
}

function EventLog({ events, follow }: { events: GameEvent[]; follow: boolean }) {
  const [hidden, setHidden] = useState<Set<string>>(() => new Set())
  const box = useRef<HTMLDivElement>(null)
  const shown = useMemo(() => events.filter(e => !hidden.has(e.type)), [events, hidden])
  const t0 = events[0]?.ts ?? 0
  useEffect(() => { if (follow && box.current) box.current.scrollTop = box.current.scrollHeight }, [shown.length, follow])
  const toggle = (t: string) => setHidden(h => { const n = new Set(h); if (n.has(t)) n.delete(t); else n.add(t); return n })
  return (
    <>
      <div className="dp-chips" role="group" aria-label="Event types">
        {LOG_TYPES.map(t => (
          <button key={t} className={hidden.has(t) ? '' : 'on'} aria-pressed={!hidden.has(t)} onClick={() => toggle(t)}>
            {t} <span>{events.filter(e => e.type === t).length}</span>
          </button>
        ))}
      </div>
      <div className="dp-log" ref={box}>
        <table className="dp-table">
          <tbody>
            {shown.map(e => {
              const d = describe(e)
              const team = d.seat in TEAM_OF ? (TEAM_OF[d.seat as Seat] === 'T1' ? 't1' : 't2') : ''
              return (
                <tr key={e.seq}>
                  <td className="num dim">{((e.ts - t0) / 1000).toFixed(2)}</td>
                  <td className={`type ${e.type}`}>{e.type}</td>
                  <td className={team}>{d.seat}</td>
                  <td>{d.text}</td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
    </>
  )
}

function GameView({ events, title, live }: { events: GameEvent[]; title: string; live?: boolean }) {
  const sum = useMemo(() => summarize(events), [events])
  const t0 = sum.start?.ts ?? 0
  const result = sum.end
    ? `${sum.end.result === 'draw' ? 'Draw' : `${sum.end.result} wins`} by ${sum.end.reason}${sum.end.board ? ` on board ${sum.end.board}` : ''}`
    : live ? 'In progress' : 'No result recorded'
  return (
    <div className="dp-game">
      <header className="dp-gamehead">
        <h2>{title}</h2>
        <div className="dp-facts">
          <span className={`dp-pill ${sum.end ? (sum.end.result === 'T1' ? 't1' : sum.end.result === 'T2' ? 't2' : '') : 'live'}`}>{result}</span>
          {sum.start && <span>{sum.start.setup}</span>}
          <span>{sum.plies} plies</span>
          <span>{fmtS(sum.lastTs - t0)} game time</span>
          <span>{sum.transfers} pieces passed</span>
          <span>{sum.messages} messages</span>
        </div>
      </header>
      <section><h3>Seats</h3><SeatTable sum={sum} /></section>
      <section><h3>Evaluation <small>from T1's side, one point per search</small></h3><EvalChart evals={sum.evals} t0={t0} t1={sum.lastTs} /></section>
      <section className="grow"><h3>Event log</h3><EventLog events={events} follow={!!live} /></section>
    </div>
  )
}

// ---------------------------------------------------------------- page

export default function DevPanel() {
  const live = useLiveGames()
  const { runs, error: runsError, refresh } = useRuns()
  const [sel, setSel] = useState<Selection>({ kind: 'live' })
  const [open, setOpen] = useState<string | null>(null)
  const file = useGameFile(sel)

  const isSel = (s: Selection) => JSON.stringify(s) === JSON.stringify(sel)
  const gameLabel = (evs: GameEvent[]) => {
    const st = evs.find(e => e.type === 'game_start')
    const end = evs.find(e => e.type === 'game_end')
    return `${st && st.type === 'game_start' ? st.game_id : 'game'}${end && end.type === 'game_end' ? ` · ${end.result}` : ''}`
  }

  let main
  if (sel.kind === 'live') {
    main = live.current.length
      ? <GameView events={live.current} title={`Live · ${gameLabel(live.current)}`} live />
      : <div className="dp-empty big">Waiting for a live game. Start one with <code>python -m arena.serve</code>.</div>
  } else if (sel.kind === 'past') {
    main = <GameView events={live.past[sel.index] ?? []} title={`Earlier live game · ${gameLabel(live.past[sel.index] ?? [])}`} />
  } else {
    main = file.error ? <div className="dp-empty big">{file.error}</div>
      : file.events ? <GameView events={file.events} title={`${sel.run} / ${sel.file}`} />
      : <div className="dp-empty big">Loading {sel.file}…</div>
  }

  return (
    <div className="dp">
      <header className="dp-top">
        <h1>Dev panel</h1>
        <span className="dp-status">{live.status}</span>
        <nav className="dp-links">
          <a href="/" target="_blank" rel="noopener">Viewer ↗</a>
          <a href="https://claude.ai/artifact/B8Wfe56MCf1xFekB2RJymf" target="_blank" rel="noopener noreferrer">Progress ↗</a>
        </nav>
      </header>
      <aside className="dp-side">
        <h3>Live</h3>
        <button className={isSel({ kind: 'live' }) ? 'on' : ''} onClick={() => setSel({ kind: 'live' })}>
          ● Current {live.current.length ? `· ${gameLabel(live.current)}` : ''}
        </button>
        {live.past.map((g, i) => (
          <button key={i} className={isSel({ kind: 'past', index: i }) ? 'on' : ''} onClick={() => setSel({ kind: 'past', index: i })}>
            {gameLabel(g)}
          </button>
        ))}
        <h3>Recorded runs <button className="dp-refresh" onClick={refresh} aria-label="Refresh runs">↻</button></h3>
        {runsError && <p className="dp-empty">{runsError}</p>}
        {runs?.length === 0 && <p className="dp-empty">No runs yet. Batches write to runs/.</p>}
        {runs?.map(r => (
          <div key={r.name} className="dp-run">
            <button className="dp-runname" aria-expanded={open === r.name} onClick={() => setOpen(open === r.name ? null : r.name)}>
              {open === r.name ? '▾' : '▸'} {r.name} <span>{r.games.length}</span>
            </button>
            {open === r.name && (
              <div className="dp-games">
                {r.games.length === 0 && <p className="dp-empty">No game files in this run.</p>}
                {r.games.map(g => (
                  <button key={g} className={isSel({ kind: 'file', run: r.name, file: g }) ? 'on' : ''}
                    onClick={() => setSel({ kind: 'file', run: r.name, file: g })}>
                    {g.replace(/^game_|\.jsonl$/g, '')}
                  </button>
                ))}
              </div>
            )}
          </div>
        ))}
      </aside>
      <main className="dp-main">{main}</main>
    </div>
  )
}
