// A tiny in-view event bus so separate parts of the 3D view can react to game moments without
// importing each other. director.ts (which sees every game event once) emits; characters, camera,
// HUD and comms subscribe. Pure UI: never feeds back into the game store.

import type { BoardId, Seat, TeamId } from '../../events/types'

export type Reaction = 'fistPump' | 'headInHands' | 'impatientTap' | 'celebrate' | 'slump' | 'talk'

export type FxEvent =
  | { kind: 'reaction'; seat: Seat; reaction: Reaction }
  | { kind: 'capture'; board: BoardId; seat: Seat; victim: Seat; piece: string }    // piece = taken piece letter
  | { kind: 'drop'; board: BoardId; seat: Seat; piece: string; square: string; blocksCheck: boolean }
  | { kind: 'talk'; seat: Seat; toSeat: Seat; token: string; line: string; status: boolean }
  | { kind: 'announce'; text: string; team?: TeamId }
  | { kind: 'gameEnd'; result: TeamId | 'draw'; reason: string }

type Listener = (e: FxEvent) => void
const listeners = new Set<Listener>()

export const fxBus = {
  emit(e: FxEvent) { for (const l of listeners) l(e) },
  /** Subscribe; returns an unsubscribe function (use it as a useEffect cleanup). */
  on(l: Listener) { listeners.add(l); return () => { listeners.delete(l) } },
}
