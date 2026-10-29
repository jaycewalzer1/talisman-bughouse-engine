// The director turns the store's growing `events` array into one-shot "fx" callbacks
// (piece slides, tosses, graffiti, sounds). Each event is handled exactly once.
//
// - New game (`game_start`) or the array shrinking/being replaced (replay scrubbing) -> `reset`.
// - Many events arriving at once (fast replay, seeking, switching from the 2D view) -> `reset`
//   too: we skip the animations and the scene simply shows the store's final state.
import { useEffect } from 'react'
import { useGame } from '../../store/gameStore'
import type { GameEvent } from '../../events/types'

const SNAP_THRESHOLD = 6

type Handler = (e: GameEvent) => void
const handlers = new Set<Handler>()
const resetHandlers = new Set<() => void>()

/** Subscribe to events as they happen. Returns an unsubscribe function (use inside useEffect). */
export function onGameEvent(h: Handler) { handlers.add(h); return () => { handlers.delete(h) } }
/** Called when transient effects should be cleared (new game, scrub, snap). */
export function onFxReset(h: () => void) { resetHandlers.add(h); return () => { resetHandlers.delete(h) } }

function emitReset() { resetHandlers.forEach(h => h()) }
function emit(e: GameEvent) {
  handlers.forEach(h => {
    try { h(e) } catch (err) { console.error('[three] fx handler failed', err) }
  })
}

/** Mount once (in Scene). */
export function useEventDirector() {
  useEffect(() => {
    let handled = 0
    let first: GameEvent | undefined

    const process = (events: GameEvent[]) => {
      if (events.length < handled || (events.length > 0 && first !== undefined && events[0] !== first)) {
        handled = 0
        emitReset()
      }
      first = events[0]
      const fresh = events.slice(handled)
      handled = events.length
      if (fresh.length === 0) return
      if (fresh.length > SNAP_THRESHOLD) { emitReset(); return }
      for (const e of fresh) {
        if (e.type === 'game_start') emitReset()
        else emit(e)
      }
    }

    process(useGame.getState().events)
    return useGame.subscribe((s, prev) => { if (s.events !== prev.events) process(s.events) })
  }, [])
}
