// Decides what each of the 4 players is doing (from the store) and renders their rigs, plus the
// one-shot reactions (fist pump on a capture, head-in-hands when losing a piece, impatient tap when
// stalling, talk gesture on messages, celebrate / slump at the end).
//
// Reactions come from fxBus (emitted by the comms/FX layer) and, for robustness, are also derived
// straight from game events. Duplicates are dropped: a derived reaction is skipped if fxBus sent
// any reaction for that seat in the last 300 ms, and the same reaction for the same seat is never
// started twice within 300 ms (whichever source fires first wins).
// To swap in a glTF/Mixamo rig, change `Rig` below (see README).
import { useEffect, useMemo, useRef } from 'react'
import { useFrame } from '@react-three/fiber'
import { Vector3 } from 'three'
import { BOARD_OF, PARTNER, SEATS, TEAM_OF, type GameEvent, type Seat, type TeamId } from '../../../events/types'
import { displayClock, useGame } from '../../../store/gameStore'
import { BOARD_X, TEAM_COLOR, seatPose } from '../layout'
import { reachTarget } from '../board/anims'
import { onFxReset, onGameEvent } from '../director'
import { fxBus, type Reaction } from '../fxBus'
import type { CharacterDriver, CharacterRig, CharacterState } from './rig'
import { ProceduralCharacter, RIG_FRAME_PRIORITY } from './ProceduralCharacter'
import { clockNow, grabBus } from '../grab/grabChoreo'
import { winTrickOf } from '../tricks'
import { MOUNT_S } from './trickPose'
import { REACTION_DUR, REACTION_PRIORITY } from './poses'
import { SPECS } from './specs'
import { NameTag } from './NameTag'
import { PortraitCapture } from './portraits'

const Rig: CharacterRig = ProceduralCharacter
const MOVE_LINGER = 0.3
const DEDUPE_S = 0.3
const PENDING_MAX_S = 2.5
const DERIVED_TALK_COOLDOWN_S = 4

// ---------------------------------------------------------------------------------------------
// reaction plumbing (module level: event handlers run outside the frame loop)

const drivers: Partial<Record<Seat, CharacterDriver>> = {}
const lastFx: Partial<Record<Seat, number>> = {}
const lastKind: Partial<Record<Seat, Partial<Record<Reaction, number>>>> = {}
const now = () => performance.now() / 1000
const busy: Partial<Record<Seat, { r: Reaction; until: number }>> = {}

function trigger(seat: Seat, r: Reaction, source: 'fx' | 'derived') {
  const d = drivers[seat]
  if (!d) return
  const t = now()
  if (source === 'derived' && t - (lastFx[seat] ?? -1e9) < DEDUPE_S) return
  const kinds = lastKind[seat] ?? (lastKind[seat] = {})
  if (t - (kinds[r] ?? -1e9) < DEDUPE_S) return
  if (source === 'fx') lastFx[seat] = t
  kinds[r] = t
  // don't let a chatty "talk" stomp on a fist pump that's still playing / queued
  const b = busy[seat]
  if (b && t < b.until && REACTION_PRIORITY[b.r] > REACTION_PRIORITY[r]) return
  busy[seat] = { r, until: t + REACTION_DUR[r] * 0.7 }
  d.pending = r
  d.pendingAt = t
}

function victimOf(seat: Seat): Seat {
  return `${BOARD_OF[seat]}_${seat.endsWith('white') ? 'black' : 'white'}` as Seat
}

function endReactions(result: TeamId | 'draw', source: 'fx' | 'derived') {
  if (result === 'draw') return
  for (const s of SEATS) trigger(s, TEAM_OF[s] === result ? 'celebrate' : 'slump', source)
}

const lastDerivedTalk: Partial<Record<Seat, number>> = {}

function onEvent(e: GameEvent) {
  switch (e.type) {
    case 'move':
      if (e.capture) { trigger(e.seat, 'fistPump', 'derived'); trigger(victimOf(e.seat), 'headInHands', 'derived') }
      break
    case 'stall':
      if (e.active) trigger(e.seat, 'impatientTap', 'derived')
      break
    case 'message': {
      const t = now()
      if (t - (lastDerivedTalk[e.fromSeat] ?? -1e9) > DERIVED_TALK_COOLDOWN_S) {
        lastDerivedTalk[e.fromSeat] = t
        trigger(e.fromSeat, 'talk', 'derived')
      }
      break
    }
    case 'game_end':
      endReactions(e.result, 'derived')
      break
  }
}

function useReactionFeeds() {
  useEffect(() => {
    const offFx = fxBus.on(e => {
      switch (e.kind) {
        case 'reaction': trigger(e.seat, e.reaction, 'fx'); break
        case 'capture': trigger(e.seat, 'fistPump', 'fx'); trigger(e.victim, 'headInHands', 'fx'); break
        case 'talk': trigger(e.seat, 'talk', 'fx'); break
        case 'gameEnd': endReactions(e.result, 'fx'); break
      }
    })
    const offGame = onGameEvent(onEvent)
    const offReset = onFxReset(() => {
      for (const s of SEATS) { delete busy[s]; const d = drivers[s]; if (d) { d.reaction = null; d.pending = null } }
    })
    return () => { offFx(); offGame(); offReset() }
  }, [])
}

// ---------------------------------------------------------------------------------------------

function Player({ seat, index }: { seat: Seat; index: number }) {
  const { pos, rotY } = useMemo(() => seatPose(seat), [seat])
  const driver = useRef<CharacterDriver>({
    state: 'idle', since: 0, pressure: 0, lookYaw: 0, reach: null, seed: index * 1.7,
    reaction: null, reactionAt: 0, pending: null, pendingAt: 0, trick: winTrickOf(seat),
  })
  const st = useMemo(() => ({ lastReach: -9, reachPos: new Vector3(), v: new Vector3(), up: new Vector3(0, 1, 0), announced: false }), [])

  useEffect(() => {
    const d = driver.current
    drivers[seat] = d
    return () => { if (drivers[seat] === d) delete drivers[seat] }
  }, [seat])

  useFrame(({ clock }) => {
    const t = clock.elapsedTime
    const s = useGame.getState()
    const d = driver.current
    const reach = reachTarget(seat)
    if (reach) { st.lastReach = t; st.reachPos.copy(reach) }
    // the hand is on its way to / holding / putting down a piece
    if (grabBus.getHandTarget(seat, clockNow())) st.lastReach = t
    let next: CharacterState = 'idle'
    if (s.end) next = s.end.result === 'draw' ? 'idle' : s.end.result === TEAM_OF[seat] ? 'win' : 'loss'
    else if (t - st.lastReach < MOVE_LINGER) next = 'moving'
    else if (s.stalls[seat]) next = 'stalling'
    else if (s.thinking[seat]) next = 'thinking'
    if (next !== d.state) { d.state = next; d.since = t; st.announced = false }
    // name the trick once, as this winner pops it for the first time (the announcer rate-limits)
    if (d.state === 'win' && d.trick && !st.announced && t - d.since >= MOUNT_S + 0.25) {
      st.announced = true
      fxBus.emit({ kind: 'announce', text: d.trick.name, team: TEAM_OF[seat] })
    }
    d.reach = next === 'moving' ? st.reachPos : null
    const ms = s.start ? displayClock(s, seat) : 60000
    d.pressure = Math.max(0, Math.min(1, (20000 - ms) / 18000))
    // yaw toward the partner's board, in this rig's local frame
    const pb = BOARD_OF[PARTNER[seat]]
    st.v.set(BOARD_X[pb] - pos.x, 0, 0 - pos.z).applyAxisAngle(st.up, -rotY)
    d.lookYaw = Math.atan2(st.v.x, st.v.z)
    // start a queued reaction once the hand is free (not mid-move)
    if (d.pending) {
      const age = now() - d.pendingAt
      if (age > PENDING_MAX_S) d.pending = null
      else if (d.state !== 'moving') { d.reaction = d.pending; d.reactionAt = t; d.pending = null }
    }
  }, RIG_FRAME_PRIORITY - 1)   // before the rig, which runs before Pieces

  return (
    <group position={pos} rotation-y={rotY}>
      <Rig seat={seat} spec={SPECS[seat]} driver={driver} teamColor={TEAM_COLOR[TEAM_OF[seat]]} />
      <NameTag seat={seat} height={2.08 * SPECS[seat].height} />
    </group>
  )
}

export function Characters() {
  useReactionFeeds()
  return (
    <>
      {SEATS.map((s, i) => <Player key={s} seat={s} index={i} />)}
      <PortraitCapture />
    </>
  )
}
