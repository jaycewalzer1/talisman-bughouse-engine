// App shell: a thin row of "sticker" controls floating over a full-screen view.
//   - view toggle: 2D | 3D (3D is lazy-loaded so its big three.js bundle only loads when needed)
//   - source picker: Mock | Live ws | Replay file, plus replay controls
// URL options: ?view=2d|3d  ?source=mock|live|replay  ?ws=ws://host:port  ?replay=/path/to/game.jsonl

import { Suspense, lazy, useEffect, useRef, useState } from 'react'
import { SPEEDS, type SourceKind, startController, usePlayback } from './sources/controller'
import TwoView from './views/two'

const ThreeView = lazy(() => import('./views/three'))

type ViewKind = '2d' | '3d'

// read the URL once at startup
const params = new URLSearchParams(window.location.search)
const initialView: ViewKind = params.get('view') === '2d' ? '2d' : '3d'
const sourceParam = params.get('source')
const initialSource: SourceKind = sourceParam === 'live' || sourceParam === 'replay' ? sourceParam : 'mock'

function fmt(ms: number) {
  const s = Math.floor(ms / 1000)
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`
}

function SpeedSelect() {
  const speed = usePlayback(s => s.speed)
  return (
    <select className="btn" value={speed} onChange={e => usePlayback.getState().setSpeed(+e.target.value)} title="playback speed">
      {SPEEDS.map(s => <option key={s} value={s}>{s}x</option>)}
    </select>
  )
}

function ReplayControls() {
  const playing = usePlayback(s => s.playing)
  const duration = usePlayback(s => s.replayDuration)
  const position = usePlayback(s => s.replayPosition)
  const name = usePlayback(s => s.replayName)
  const { togglePlay, seek, loadReplayFile } = usePlayback.getState()
  const fileInput = useRef<HTMLInputElement>(null)

  return (
    <div className="sticker replay">
      <button className="btn" onClick={() => fileInput.current?.click()} title="open a .jsonl replay file">
        📂 {name ?? 'open .jsonl'}
      </button>
      <input
        ref={fileInput} type="file" accept=".jsonl,.json,.txt" hidden
        onChange={e => { const f = e.target.files?.[0]; if (f) void loadReplayFile(f); e.target.value = '' }}
      />
      <button className="btn" onClick={togglePlay} disabled={!duration}>{playing ? '❚❚' : '▶'}</button>
      <input
        className="scrub" type="range" min={0} max={duration || 1} step={100} value={position}
        onChange={e => seek(+e.target.value)} disabled={!duration}
      />
      <span className="time">{fmt(position)} / {fmt(duration)}</span>
      <SpeedSelect />
    </div>
  )
}

function LiveControls() {
  const wsUrl = usePlayback(s => s.wsUrl)
  const [draft, setDraft] = useState(wsUrl)
  return (
    <form className="sticker live" onSubmit={e => { e.preventDefault(); usePlayback.getState().setWsUrl(draft) }}>
      <input className="url" value={draft} onChange={e => setDraft(e.target.value)} spellCheck={false} />
      <button className="btn" type="submit">connect</button>
    </form>
  )
}

function MockControls() {
  const playing = usePlayback(s => s.playing)
  return (
    <div className="sticker replay">
      <button className="btn" onClick={usePlayback.getState().togglePlay}>{playing ? '❚❚' : '▶'}</button>
      <SpeedSelect />
    </div>
  )
}

export default function App() {
  const [view, setView] = useState<ViewKind>(initialView)
  const source = usePlayback(s => s.source)
  const status = usePlayback(s => s.status)

  // start the event source + animation loop once; stop it when the app unmounts
  useEffect(() => startController({
    source: initialSource,
    wsUrl: params.get('ws') ?? undefined,
    replayUrl: params.get('replay') ?? undefined,
  }), [])

  // keep the URL in sync so a reload (or a shared link) keeps the same view + source
  useEffect(() => {
    const p = new URLSearchParams(window.location.search)
    p.set('view', view)
    p.set('source', source)
    window.history.replaceState(null, '', `?${p.toString()}`)
  }, [view, source])

  return (
    <>
      <main className="view">
        {view === '2d' ? <TwoView /> : (
          <Suspense fallback={<div className="loading">loading 3D…</div>}>
            <ThreeView />
          </Suspense>
        )}
      </main>

      <nav className="shell">
        <div className="sticker seg" role="group" aria-label="view">
          {(['2d', '3d'] as ViewKind[]).map(v => (
            <button key={v} className={view === v ? 'on' : ''} onClick={() => setView(v)}>{v.toUpperCase()}</button>
          ))}
        </div>
        <div className="sticker seg tilt" role="group" aria-label="event source">
          {([['mock', 'Mock'], ['live', 'Live ws'], ['replay', 'Replay']] as [SourceKind, string][]).map(([k, label]) => (
            <button key={k} className={source === k ? 'on' : ''} onClick={() => usePlayback.getState().setSource(k)}>{label}</button>
          ))}
        </div>
        {source === 'replay' && <ReplayControls />}
        {source === 'live' && <LiveControls />}
        {source === 'mock' && <MockControls />}
        {status && <span className="status-tape">{status}</span>}
      </nav>
    </>
  )
}
