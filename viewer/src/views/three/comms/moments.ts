// Turns raw game events (director.onGameEvent) into fxBus "moments" that the HUD, comms, camera and
// characters react to: capture, drop, talk, reaction, announce, gameEnd. Mount useMoments() ONCE
// (Hud.tsx does). Pure UI: never writes to the game store.
import { useEffect } from 'react'
import { type BoardId, type GameEvent, type Seat, type TeamId, SEATS, TEAM_OF } from '../../../events/types'
import { displayClock, useGame } from '../../../store/gameStore'
import { onFxReset, onGameEvent } from '../director'
import { fxBus } from '../fxBus'
import { STYLE } from '../style.config'
import { ANNOUNCER, END_FALLBACK, END_TEXT, isStatusShare, lineFor, statusBest, statusLine } from './lines'

const DOUBLE_DROP_MS = 3000        // same team, two drops within this much game time
const PRESSURE_MS = 10_000         // a clock first going under this -> TIME PRESSURE!

export const opponentOf = (seat: Seat): Seat =>
  (seat.endsWith('white') ? seat.replace('white', 'black') : seat.replace('black', 'white')) as Seat

const st = {
  lastSan: { A: null, B: null } as Record<BoardId, string | null>,
  lastDropTs: {} as Partial<Record<TeamId, number>>,
  status: {} as Partial<Record<Seat, { at: number; best: string }>>,
  lowFired: new Set<Seat>(),
  primed: false,
  gameId: '' as string,
}

function resetState() {
  const s = useGame.getState()
  st.lastSan = { A: s.lastMove.A?.san ?? null, B: s.lastMove.B?.san ?? null }
  st.lastDropTs = {}
  st.status = {}
  st.lowFired.clear()
  st.primed = false           // next clock tick marks already-low clocks silently
  st.gameId = s.start?.game_id ?? ''
}

const announce = (text: string, team?: TeamId) => fxBus.emit({ kind: 'announce', text, team })

function talk(seat: Seat, toSeat: Seat, token: string, line: string, status: boolean) {
  fxBus.emit({ kind: 'talk', seat, toSeat, token, line, status })
  fxBus.emit({ kind: 'reaction', seat, reaction: 'talk' })
}

/** Exported for tests: handle one game event. */
export function handleGameEvent(e: GameEvent, nowMs = performance.now()) {
  switch (e.type) {
    case 'move': {
      const prevSan = st.lastSan[e.board]
      st.lastSan[e.board] = e.san ?? null
      const team = TEAM_OF[e.seat]
      if (e.capture) {
        const piece = e.capture.replace(/~/g, '').toUpperCase()
        fxBus.emit({ kind: 'capture', board: e.board, seat: e.seat, victim: opponentOf(e.seat), piece })
        if (piece === 'Q') announce(ANNOUNCER.queenSnatch, team)
      }
      if (e.isDrop) {
        const piece = e.uci[0].toUpperCase()
        const square = e.uci.slice(2, 4)
        // Heuristic: the opponent's previous SAN on this board gave check -> this drop answered it.
        const blocksCheck = !!prevSan && /[+]\W*$/.test(prevSan)
        fxBus.emit({ kind: 'drop', board: e.board, seat: e.seat, piece, square, blocksCheck })
        const prev = st.lastDropTs[team]
        if (blocksCheck) {
          announce(ANNOUNCER.clutchSave, team)
          st.lastDropTs[team] = e.ts
        } else if (prev !== undefined && e.ts - prev <= DOUBLE_DROP_MS) {
          announce(ANNOUNCER.doubleDrop, team)
          delete st.lastDropTs[team]          // a third drop starts a new pair
        } else {
          st.lastDropTs[team] = e.ts
        }
      }
      break
    }
    case 'pass':
      if (STYLE.comms.passBubbles) talk(e.fromSeat, e.toSeat, 'pass', lineFor('pass'), false)
      break
    case 'message': {
      if (isStatusShare(e.token, e.payload)) {
        const best = statusBest(e.payload)
        const prev = st.status[e.fromSeat]
        if (prev && (prev.best === best || nowMs - prev.at < STYLE.comms.statusBubbleCooldownMs)) break
        st.status[e.fromSeat] = { at: nowMs, best }
        talk(e.fromSeat, e.toSeat, e.token, statusLine(best), true)
      } else {
        talk(e.fromSeat, e.toSeat, e.token, lineFor(e.token), false)
      }
      break
    }
    case 'game_end': {
      announce(END_TEXT[e.reason] ?? END_FALLBACK, e.result === 'draw' ? undefined : e.result)
      fxBus.emit({ kind: 'gameEnd', result: e.result, reason: e.reason })
      break
    }
    default:
      break
  }
}

/** Clock watcher: first time any clock dips under 10 s -> TIME PRESSURE! (once per seat per game). */
function clockTick() {
  const s = useGame.getState()
  if (!s.start || s.end) return
  if (s.start.game_id !== st.gameId) resetState()
  if (s.start.base_ms <= PRESSURE_MS) return
  for (const seat of SEATS) {
    if (st.lowFired.has(seat)) continue
    if (displayClock(s, seat) < PRESSURE_MS) {
      st.lowFired.add(seat)
      if (st.primed) announce(ANNOUNCER.timePressure, TEAM_OF[seat])
    }
  }
  st.primed = true
}

/** Mount once (Hud.tsx). */
export function useMoments() {
  useEffect(() => {
    resetState()
    const offE = onGameEvent(e => handleGameEvent(e))
    const offR = onFxReset(resetState)
    const iv = window.setInterval(clockTick, 250)
    return () => { offE(); offR(); window.clearInterval(iv) }
  }, [])
}
