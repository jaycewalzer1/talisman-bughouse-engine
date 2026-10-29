// A strip of tape floating over each player: AI name in team color, plus a red "SITTING" sticker
// while stalling and "..." while thinking. Canvas sprite, redrawn only when the text changes.
import { useMemo, useRef } from 'react'
import { useFrame } from '@react-three/fiber'
import { CanvasTexture, SRGBColorSpace, type Sprite } from 'three'
import { TEAM_OF, type Seat } from '../../../events/types'
import { useGame } from '../../../store/gameStore'
import { TEAM_COLOR } from '../layout'
import { SPECS } from './specs'

export function NameTag({ seat, height }: { seat: Seat; height: number }) {
  const sprite = useRef<Sprite>(null)
  const { canvas, tex } = useMemo(() => {
    const canvas = document.createElement('canvas'); canvas.width = 256; canvas.height = 96
    const tex = new CanvasTexture(canvas); tex.colorSpace = SRGBColorSpace
    return { canvas, tex }
  }, [])
  const key = useRef('')

  useFrame(({ clock }) => {
    const s = useGame.getState()
    // Character handle + player kind; the arena's full names are too long for a floating tag.
    const kind = s.start?.seats[seat]?.kind
    const name = kind ? `${SPECS[seat].id} · ${kind.slice(0, 5)}` : SPECS[seat].id
    const stall = s.stalls[seat]
    const thinking = !!s.thinking[seat]
    const k = `${name}|${stall ?? ''}|${thinking}`
    if (sprite.current && stall) sprite.current.position.y = height + Math.abs(Math.sin(clock.elapsedTime * 4)) * 0.04
    if (k === key.current) return
    key.current = k
    const g = canvas.getContext('2d')!
    g.clearRect(0, 0, 256, 96)
    // tape strip
    g.save(); g.translate(128, 30); g.rotate(-0.03)
    g.fillStyle = 'rgba(232,226,205,0.95)'; g.fillRect(-118, -20, 236, 40)
    g.fillStyle = TEAM_COLOR[TEAM_OF[seat]]; g.fillRect(-118, -20, 10, 40)
    g.fillStyle = '#151515'; g.font = 'bold 24px "Courier New", monospace'; g.textAlign = 'center'; g.textBaseline = 'middle'
    g.fillText((name + (thinking && !stall ? ' ...' : '')).slice(0, 16), 4, 2)
    g.restore()
    if (stall) {
      g.save(); g.translate(128, 72); g.rotate(0.05)
      g.fillStyle = '#e02424'; g.fillRect(-80, -17, 160, 34)
      g.fillStyle = '#fff'; g.font = 'bold 22px "Arial Black", sans-serif'; g.textAlign = 'center'; g.textBaseline = 'middle'
      g.fillText('⏸ SITTING', 0, 1)
      g.restore()
    }
    tex.needsUpdate = true
  })

  return (
    <sprite ref={sprite} position={[0, height, 0]} scale={[0.55, 0.21, 1]}>
      <spriteMaterial map={tex} transparent depthWrite={false} toneMapped={false} />
    </sprite>
  )
}
