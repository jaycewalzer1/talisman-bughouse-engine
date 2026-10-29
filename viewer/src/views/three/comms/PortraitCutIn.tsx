// THUG-style portrait cut-in: the speaker's head close-up in a skewed, team-coloured frame that
// slides in from the screen edge. Falls back to a drawn initial badge until the portrait exists.
import type { Seat, TeamId } from '../../../events/types'
import { useGame } from '../../../store/gameStore'

// characters/portraits.ts is written by another part of the view. import.meta.glob resolves at
// build time to {} if the file isn't there, so this compiles with or without it.
type PortraitState = { urls: Partial<Record<Seat, string>> }
type PortraitHook = <T>(sel: (s: PortraitState) => T) => T
const mods = import.meta.glob('../characters/portraits.ts', { eager: true }) as Record<string, { usePortraits?: PortraitHook }>
const portraitHook = Object.values(mods)[0]?.usePortraits

export const usePortraitUrl: (seat: Seat) => string | undefined = portraitHook
  ? (seat) => portraitHook(s => s.urls?.[seat])
  : () => undefined

export function PortraitCutIn({ seat, team, side }: { seat: Seat; team: TeamId; side: 'left' | 'right' }) {
  const url = usePortraitUrl(seat)
  const name = useGame(s => s.start?.seats?.[seat]?.name) ?? seat.replace('_', ' ')
  return (
    <div className={`h3-portrait h3-${team} h3-from-${side}`} title={name}>
      <div className="h3-portrait-inner">
        {url
          ? <img src={url} alt={name} draggable={false} />
          : <span className="h3-portrait-badge">{initialOf(name)}</span>}
      </div>
    </div>
  )
}

function initialOf(name: string): string {
  const m = name.match(/[A-Za-z0-9]/)
  return m ? m[0].toUpperCase() : '?'
}
