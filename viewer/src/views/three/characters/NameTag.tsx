// A small arcade-style tag over each skater: slanted team-colour plate with the handle, an accent
// underline, "..." while thinking and a red WAIT sticker while stalling. Canvas sprite, redrawn
// only when the text changes. (The HUD shows the full AI names.)
import { useMemo, useRef } from 'react'
import { useFrame } from '@react-three/fiber'
import { CanvasTexture, NearestFilter, SRGBColorSpace, type Sprite } from 'three'
import { TEAM_OF, type Seat } from '../../../events/types'
import { useGame } from '../../../store/gameStore'
import { STYLE } from '../style.config'
import { SPECS } from './specs'

const W = 192, H = 72

export function NameTag({ seat, height }: { seat: Seat; height: number }) {
  const sprite = useRef<Sprite>(null)
  const { canvas, tex } = useMemo(() => {
    const canvas = document.createElement('canvas'); canvas.width = W; canvas.height = H
    const tex = new CanvasTexture(canvas); tex.colorSpace = SRGBColorSpace
    tex.magFilter = NearestFilter; tex.minFilter = NearestFilter; tex.generateMipmaps = false
    return { canvas, tex }
  }, [])
  const key = useRef('')

  useFrame(({ clock }) => {
    const s = useGame.getState()
    const stall = s.stalls[seat]
    const thinking = !!s.thinking[seat]
    if (sprite.current) sprite.current.position.y = height + (stall ? Math.abs(Math.sin(clock.elapsedTime * 5)) * 0.03 : 0)
    const k = `${stall ?? ''}|${thinking}`
    if (k === key.current) return
    key.current = k
    const spec = SPECS[seat]
    const team = TEAM_OF[seat] === 'T1' ? STYLE.palette.T1 : STYLE.palette.T2
    const g = canvas.getContext('2d')!
    g.clearRect(0, 0, W, H)
    // slanted plate
    const plate = (x: number, y: number, w: number, h: number, sl: number) => {
      g.beginPath(); g.moveTo(x + sl, y); g.lineTo(x + w, y); g.lineTo(x + w - sl, y + h); g.lineTo(x, y + h); g.closePath()
    }
    g.fillStyle = STYLE.palette.ink; plate(14, 6, 168, 38, 12); g.fill()
    g.fillStyle = team; plate(10, 2, 168, 38, 12); g.fill()
    g.fillStyle = spec.accent; plate(18, 36, 140, 6, 3); g.fill()
    g.font = 'italic 900 26px "Arial Black", Impact, sans-serif'
    g.textAlign = 'center'; g.textBaseline = 'middle'
    const label = spec.id + (thinking && !stall ? '…' : '')
    g.lineWidth = 5; g.strokeStyle = STYLE.palette.ink; g.strokeText(label, 94, 22)
    g.fillStyle = '#ffffff'; g.fillText(label, 94, 22)
    if (stall) {
      g.save(); g.translate(150, 56); g.rotate(-0.08)
      g.fillStyle = '#e02424'; g.fillRect(-36, -12, 72, 24)
      g.fillStyle = '#fff'; g.font = 'italic 900 16px "Arial Black", Impact, sans-serif'
      g.fillText('WAIT', 0, 1)
      g.restore()
    }
    tex.needsUpdate = true
  })

  return (
    <sprite ref={sprite} position={[0, height, 0]} scale={[0.4, 0.15, 1]}>
      <spriteMaterial map={tex} transparent depthWrite={false} toneMapped={false} />
    </sprite>
  )
}
