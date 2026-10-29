// The controller owns the ONE active event source (mock | live | replay) and runs the animation
// loop that moves the store's playhead every frame (that's what makes clocks tick smoothly).
//
// UI code talks to it only through the `usePlayback` zustand store below:
//   const source = usePlayback(s => s.source)
//   usePlayback.getState().setSource('replay')
// and calls startController() once (App does this in a useEffect).

import { create } from 'zustand'
import { useGame } from '../store/gameStore'
import { createMockSource } from './mock'
import { createReplaySource, parseJsonl, type ReplaySource } from './replay'
import type { Source } from './source'
import { DEFAULT_WS_URL, createLiveSource } from './websocket'

export type SourceKind = 'mock' | 'live' | 'replay'
export const SPEEDS = [0.25, 0.5, 1, 2, 4, 8]
export const SAMPLE_REPLAY_URL = `${import.meta.env.BASE_URL}replays/sample.jsonl`

export interface PlaybackState {
  source: SourceKind
  status: string                 // human readable: "live: ws://...", "mock game #3", ...
  playing: boolean               // mock + replay only (live always follows the arena)
  speed: number
  wsUrl: string
  replayName: string | null      // file name / URL of the loaded replay
  replayDuration: number         // ms on the replay's global timeline
  replayPosition: number         // ms, updated ~10x per second for the scrub bar

  setSource: (kind: SourceKind) => void
  setWsUrl: (url: string) => void
  play: () => void
  pause: () => void
  togglePlay: () => void
  setSpeed: (speed: number) => void
  seek: (ms: number) => void
  loadReplayFile: (file: File) => Promise<void>
  loadReplayUrl: (url: string) => Promise<void>
}

// ---- module-level (non-React) state ---------------------------------------
let active: Source | null = null
let replay: ReplaySource | null = null   // same object as `active` when source === 'replay'
let rafId = 0
let lastFrame = 0
let lastUiUpdate = 0

function setStatus(status: string) { usePlayback.setState({ status }) }

function stopActive() {
  active?.stop()
  active = null
  replay = null
}

function startSource(kind: SourceKind) {
  stopActive()
  useGame.getState().reset()
  if (kind === 'mock') {
    active = createMockSource(setStatus)
  } else if (kind === 'live') {
    active = createLiveSource(usePlayback.getState().wsUrl, setStatus)
  } else {
    // replay with nothing chosen yet: load the bundled sample game
    void usePlayback.getState().loadReplayUrl(SAMPLE_REPLAY_URL)
    return
  }
  active.start()
}

function activateReplay(events: ReturnType<typeof parseJsonl>, name: string) {
  stopActive()
  if (!events.length) { setStatus(`replay: ${name} has no events`); return }
  replay = createReplaySource(events)
  active = replay
  replay.start()
  usePlayback.setState({
    replayName: name, replayDuration: replay.duration, replayPosition: 0, playing: true,
    status: `replay: ${name} (${events.length} events)`,
  })
}

function frame(now: number) {
  const dtWall = lastFrame ? Math.min(250, now - lastFrame) : 0   // clamp: tab was in background
  lastFrame = now
  const pb = usePlayback.getState()
  const dt = pb.playing ? dtWall * pb.speed : 0
  active?.tick(dt, now)

  if (replay && now - lastUiUpdate > 100) {
    lastUiUpdate = now
    const pos = replay.position()
    if (pos !== pb.replayPosition) usePlayback.setState({ replayPosition: pos })
    if (replay.ended() && pb.playing) usePlayback.setState({ playing: false })
  }
  rafId = requestAnimationFrame(frame)
}

// ---- the store ------------------------------------------------------------
export const usePlayback = create<PlaybackState>()((set, get) => ({
  source: 'mock',
  status: '',
  playing: true,
  speed: 1,
  wsUrl: DEFAULT_WS_URL,
  replayName: null,
  replayDuration: 0,
  replayPosition: 0,

  setSource: (kind) => {
    set({ source: kind, playing: true })
    startSource(kind)
  },
  setWsUrl: (url) => {
    set({ wsUrl: url })
    if (get().source === 'live') startSource('live')
  },
  play: () => {
    // pressing play at the end of a replay restarts it
    if (replay?.ended()) replay.seek(0)
    set({ playing: true })
  },
  pause: () => set({ playing: false }),
  togglePlay: () => (get().playing ? get().pause() : get().play()),
  setSpeed: (speed) => set({ speed }),
  seek: (ms) => {
    if (!replay) return
    replay.seek(ms)
    set({ replayPosition: replay.position() })
  },
  loadReplayFile: async (file) => {
    set({ source: 'replay', playing: true })
    activateReplay(parseJsonl(await file.text()), file.name)
  },
  loadReplayUrl: async (url) => {
    set({ source: 'replay', playing: true })
    setStatus(`replay: loading ${url} ...`)
    try {
      const res = await fetch(url)
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      const text = await res.text()
      if (get().source !== 'replay') return     // user switched away while we were loading
      activateReplay(parseJsonl(text), url.split('/').pop() || url)
    } catch (err) {
      setStatus(`replay: could not load ${url} (${String(err)})`)
    }
  },
}))

/**
 * Start the animation loop and the initial source. Safe to call more than once (React
 * StrictMode runs effects twice in development); returns a cleanup function.
 */
export function startController(initial: { source: SourceKind; wsUrl?: string; replayUrl?: string }) {
  if (initial.wsUrl) usePlayback.setState({ wsUrl: initial.wsUrl })
  const pb = usePlayback.getState()
  if (initial.source === 'replay') {
    void pb.loadReplayUrl(initial.replayUrl ?? SAMPLE_REPLAY_URL)
  } else {
    pb.setSource(initial.source)
  }
  lastFrame = 0
  rafId = requestAnimationFrame(frame)
  return () => {
    cancelAnimationFrame(rafId)
    stopActive()
  }
}
