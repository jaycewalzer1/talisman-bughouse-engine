// App shell: a thin row of "sticker" controls floating over a full-screen view.
//   - view toggle: 2D | 3D (3D is lazy-loaded so its big three.js bundle only loads when needed)
//   - source picker: Mock | Live ws | Replay file, plus replay controls
//   - picture-in-picture corner card (top-right): the view you're NOT in, live and clickable.
//     In 3D mode it's a compact 2D table (views/two/MiniBoards); in 2D mode it's the 3D scene itself,
//     shrunk into the corner. Clicking it (or the 2D|3D toggle — same state) swaps the two.
//   - page links: Progress (build log tracker) and Dev panel (dev.html), each in a new tab
// URL options: ?view=2d|3d  ?source=mock|live|replay  ?ws=ws://host:port  ?replay=/path/to/game.jsonl
//
// How the 3D thumbnail works: the 3D <Canvas> is never unmounted once loaded (remounting would
// rebuild the scene, characters and portraits and can lose the WebGL context). Its layer always
// keeps its full-window layout size; for the thumbnail it is only CSS-transformed
// (translate + scale) into the card. So the canvas never resizes — RenderScale's dpr and the
// post-processing composer buffers are untouched — and the swap animation is a single
// compositor-only transform transition. ThreeView itself drops to ~4 fps ("demand" frameloop),
// hides its HUD and ignores camera input while `thumb` is set, and stops rendering entirely
// ("never") while the card is collapsed.

import { type CSSProperties, type KeyboardEvent, Suspense, lazy, useCallback, useEffect, useLayoutEffect, useRef, useState } from 'react'
import { SPEEDS, type SourceKind, startController, usePlayback } from './sources/controller'
import TwoView from './views/two'
import MiniBoards from './views/two/MiniBoards'
import { STYLE } from './views/three/style.config'   // plain object, no three.js: safe in the main bundle

const ThreeView = lazy(() => import('./views/three'))

/** Build log tracker (progress + learning notes), opened in a new tab from the shell. */
const PROGRESS_URL = 'https://claude.ai/artifact/B8Wfe56MCf1xFekB2RJymf'

type ViewKind = '2d' | '3d'

// read the URL once at startup
const params = new URLSearchParams(window.location.search)
const initialView: ViewKind = params.get('view') === '2d' ? '2d' : '3d'
const sourceParam = params.get('source')
const initialSource: SourceKind = sourceParam === 'live' || sourceParam === 'replay' ? sourceParam : 'mock'

// ---------------- picture-in-picture geometry ----------------
const PIP_TOP = 10          // same top as the shell bar; the shell's right edge is pulled in to make room
const PIP_RIGHT = 12
const PIP_BORDER = 3
const PIP_LABEL_H = 22
const PIP_GAP = 10          // breathing room kept clear around the card (shell / bubbles / 2D log)
const SWAP_MS = 280         // swap animation (keep in sync with index.css .layer-3d transition)
const COLLAPSE_KEY = 'bughouse.pipCollapsed'
/** The card wears the 3D HUD's team colours (sodium orange / steel blue). */
const PIP_VARS = { '--pip-t1': STYLE.palette.T1, '--pip-t2': STYLE.palette.T2, '--pip-yellow': STYLE.palette.yellow } as CSSProperties

function readCollapsed(): boolean {
  try { return window.localStorage.getItem(COLLAPSE_KEY) === '1' } catch { return false }
}
function writeCollapsed(on: boolean) {
  try { window.localStorage.setItem(COLLAPSE_KEY, on ? '1' : '0') } catch { /* private mode etc. */ }
}

function useWindowSize() {
  const [size, setSize] = useState(() => ({ w: window.innerWidth, h: window.innerHeight }))
  useEffect(() => {
    const on = () => setSize({ w: window.innerWidth, h: window.innerHeight })
    window.addEventListener('resize', on)
    return () => window.removeEventListener('resize', on)
  }, [])
  return size
}

/** Width of the 3D thumbnail's "screen" (its height follows the window's aspect). */
const thumbWidth = (vw: number) => Math.round(Math.min(280, Math.max(170, vw * 0.2)))

function onActivate(fn: () => void) {
  return {
    role: 'button', tabIndex: 0, onClick: fn,
    onKeyDown: (e: KeyboardEvent) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); fn() } },
  } as const
}

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

  // ---- picture-in-picture state ----
  const [collapsed, setCollapsedState] = useState(readCollapsed)
  const setCollapsed = useCallback((on: boolean) => { setCollapsedState(on); writeCollapsed(on) }, [])
  // the 3D bundle loads the first time it's needed (3D view, or the 3D thumbnail), then stays mounted
  const [threeMounted, setThreeMounted] = useState(initialView === '3d' || !readCollapsed())
  useEffect(() => { if (view === '3d' || !collapsed) setThreeMounted(true) }, [view, collapsed])
  // the full 2D view stays mounted for the length of the swap animation so the 3D layer can grow over it
  const [twoLingers, setTwoLingers] = useState(false)
  useEffect(() => {
    if (view === '2d') { setTwoLingers(true); return }
    const t = window.setTimeout(() => setTwoLingers(false), SWAP_MS + 60)
    return () => window.clearTimeout(t)
  }, [view])
  const showTwo = view === '2d' || twoLingers

  const { w: vw, h: vh } = useWindowSize()
  const thumb = view === '2d'
  const tw = thumbWidth(vw), scale = tw / Math.max(1, vw), th = Math.round(vh * scale)
  const layerTransform = thumb
    ? `translate(${vw - PIP_RIGHT - PIP_BORDER - tw}px, ${PIP_TOP + PIP_BORDER}px) scale(${scale})`
    : 'translate(0px, 0px) scale(1)'

  // measure whatever sits in the corner and publish it as CSS vars, so the shell bar, the 3D HUD's
  // right-hand bubbles and the 2D view's header/log keep clear of it
  const cornerRef = useRef<HTMLElement | null>(null)
  const setCorner = useCallback((el: HTMLElement | null) => { cornerRef.current = el }, [])
  useLayoutEffect(() => {
    const el = cornerRef.current
    const root = document.documentElement.style
    const publish = () => {
      // layout box (offset*), not getBoundingClientRect: ignores the card's pop-in transform
      root.setProperty('--pip-w', el ? `${el.offsetWidth + PIP_GAP}px` : '0px')
      root.setProperty('--pip-bottom', el ? `${el.offsetTop + el.offsetHeight + PIP_GAP}px` : '0px')
    }
    publish()
    if (!el) return
    const ro = new ResizeObserver(publish)
    ro.observe(el)
    return () => ro.disconnect()
  }, [view, collapsed, vw, vh])

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

  const collapseBtn = (
    <button className="pip-collapse" title="hide the inset" aria-label="hide the inset"
      onClick={e => { e.stopPropagation(); setCollapsed(true) }}
      onKeyDown={e => e.stopPropagation()}>–</button>
  )

  return (
    <>
      <main className="view" style={PIP_VARS}>
        {showTwo && <div className={`layer-2d ${view === '2d' ? 'is-in' : 'is-out'}`}><TwoView /></div>}
        {threeMounted && (
          <div
            className={`layer-3d ${thumb ? 'is-thumb' : 'is-full'} ${thumb && collapsed ? 'is-hidden' : ''}`}
            style={{ transform: layerTransform }}
            aria-hidden={thumb}
          >
            <Suspense fallback={<div className="loading">loading 3D…</div>}>
              <ThreeView thumb={thumb} paused={thumb && collapsed} />
            </Suspense>
          </div>
        )}

        {collapsed ? (
          <button ref={setCorner} className="pip-tab"
            title={`show the ${thumb ? '3D' : '2D'} inset`} onClick={() => setCollapsed(false)}>
            ▾ {thumb ? '3D' : '2D'}
          </button>
        ) : thumb ? (
          // 2D mode: a transparent frame around the scaled-down 3D layer (which sits just under it)
          <div ref={setCorner} key="pip3d" className="pip-card pip-3d" {...onActivate(() => setView('3d'))}
            title="switch to the 3D view"
            style={{ width: tw + 2 * PIP_BORDER, height: th + 2 * PIP_BORDER + PIP_LABEL_H, '--pip-label-h': `${PIP_LABEL_H}px` } as CSSProperties}>
            <div className="pip-label"><b>3D</b> · click to enter</div>
            {collapseBtn}
          </div>
        ) : (
          // 3D mode: a compact live 2D table
          <div ref={setCorner} key="pip2d" className="pip-card pip-2d" {...onActivate(() => setView('2d'))}
            title="switch to the 2D view">
            <MiniBoards />
            <div className="pip-label"><b>2D</b> · click to enter</div>
            {collapseBtn}
          </div>
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
        <div className="sticker seg pages" role="group" aria-label="other pages">
          <a className="btn" href={PROGRESS_URL} target="_blank" rel="noopener noreferrer">Progress ↗</a>
          <a className="btn" href="/dev.html" target="_blank" rel="noopener">Dev panel ↗</a>
        </div>
      </nav>
    </>
  )
}
