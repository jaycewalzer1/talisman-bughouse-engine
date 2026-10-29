// Decides what each of the 4 players is doing (from the store) and renders their rigs.
// To swap in a glTF/Mixamo rig, change `Rig` below (see README).
import { useMemo, useRef } from 'react'
import { useFrame } from '@react-three/fiber'
import { Vector3 } from 'three'
import { BOARD_OF, PARTNER, SEATS, TEAM_OF, type Seat } from '../../../events/types'
import { displayClock, useGame } from '../../../store/gameStore'
import { BOARD_X, TEAM_COLOR, seatPose } from '../layout'
import { reachTarget } from '../board/anims'
import type { CharacterDriver, CharacterRig, CharacterState } from './rig'
import { ProceduralCharacter } from './ProceduralCharacter'
import { SPECS } from './specs'
import { NameTag } from './NameTag'

const Rig: CharacterRig = ProceduralCharacter
const MOVE_LINGER = 0.3

function Player({ seat, index }: { seat: Seat; index: number }) {
  const { pos, rotY } = useMemo(() => seatPose(seat), [seat])
  const driver = useRef<CharacterDriver>({ state: 'idle', since: 0, pressure: 0, lookYaw: 0, reach: null, seed: index * 1.7 })
  const st = useMemo(() => ({ lastReach: -9, reachPos: new Vector3(), v: new Vector3() }), [])

  useFrame(({ clock }) => {
    const t = clock.elapsedTime
    const s = useGame.getState()
    const d = driver.current
    const reach = reachTarget(seat)
    if (reach) { st.lastReach = t; st.reachPos.copy(reach) }
    let next: CharacterState = 'idle'
    if (s.end) next = s.end.result === 'draw' ? 'idle' : s.end.result === TEAM_OF[seat] ? 'win' : 'loss'
    else if (t - st.lastReach < MOVE_LINGER) next = 'moving'
    else if (s.stalls[seat]) next = 'stalling'
    else if (s.thinking[seat]) next = 'thinking'
    if (next !== d.state) { d.state = next; d.since = t }
    d.reach = next === 'moving' ? st.reachPos : null
    const ms = s.start ? displayClock(s, seat) : 60000
    d.pressure = Math.max(0, Math.min(1, (20000 - ms) / 18000))
    // yaw toward the partner's board, in this rig's local frame
    const pb = BOARD_OF[PARTNER[seat]]
    st.v.set(BOARD_X[pb] - pos.x, 0, 0 - pos.z).applyAxisAngle(new Vector3(0, 1, 0), -rotY)
    d.lookYaw = Math.atan2(st.v.x, st.v.z)
  })

  return (
    <group position={pos} rotation-y={rotY}>
      <Rig spec={SPECS[seat]} driver={driver} teamColor={TEAM_COLOR[TEAM_OF[seat]]} />
      <NameTag seat={seat} height={2.15 * SPECS[seat].height} />
    </group>
  )
}

export function Characters() {
  return <>{SEATS.map((s, i) => <Player key={s} seat={s} index={i} />)}</>
}
