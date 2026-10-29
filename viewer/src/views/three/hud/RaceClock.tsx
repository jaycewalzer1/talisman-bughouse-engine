// Race-timer clock for one seat: italic digits in team colour, the running clock highlighted,
// flashing under 10 s, plus the agent handle and which board/colour they play.
import { type BoardId, type Seat, TEAM_OF, seatToMove } from '../../../events/types'
import { displayClock, useGame } from '../../../store/gameStore'

export function formatRace(ms: number): string {
  ms = Math.max(0, ms)
  if (ms < 10_000) return (Math.floor(ms / 100) / 10).toFixed(1)
  const total = Math.ceil(ms / 1000)
  return `${Math.floor(total / 60)}:${String(total % 60).padStart(2, '0')}`
}

export function RaceClock({ seat }: { seat: Seat }) {
  const board = seat[0] as BoardId
  // 100 ms resolution so this re-renders at most 10x/s
  const ms = useGame(s => Math.floor(displayClock(s, seat) / 100) * 100)
  const running = useGame(s => !!s.start && !s.end && seatToMove(board, s.fens[board]) === seat)
  const name = useGame(s => s.start?.seats?.[seat]?.name) ?? seat.replace('_', ' ')
  const stalled = useGame(s => !!s.stalls[seat])
  const started = useGame(s => !!s.start)
  const low = started && ms < 10_000
  const white = seat.endsWith('white')
  return (
    <div className={`h3-clock h3-${TEAM_OF[seat]}${running ? ' h3-running' : ''}${low ? ' h3-low' : ''}`}>
      <div className="h3-clock-head">
        <span className={`h3-clock-pip ${white ? 'h3-pip-w' : 'h3-pip-b'}`} />
        <span className="h3-clock-board">{board}</span>
        <span className="h3-clock-name">{name}</span>
        {stalled && <span className="h3-clock-sit">SIT</span>}
      </div>
      <div className="h3-clock-time">{started ? formatRace(ms) : '--:--'}</div>
    </div>
  )
}
