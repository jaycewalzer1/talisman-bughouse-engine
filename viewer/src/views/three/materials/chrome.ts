// Chrome / env-map shine, ported in spirit from the PSX-style metallic materials: a tiny generated
// sphere map (sunset sky up top, a hard bright horizon band, dark ground) sampled by view-space
// normal. One shared 128px texture, nearest-filtered.
import { CanvasTexture, NearestFilter, SRGBColorSpace } from 'three'
import { STYLE } from '../style.config'

let tex: CanvasTexture | null = null

export function chromeTexture(): CanvasTexture {
  if (tex) return tex
  const S = 128
  const c = document.createElement('canvas'); c.width = c.height = S
  const g = c.getContext('2d')!
  const grad = g.createLinearGradient(0, 0, 0, S)
  grad.addColorStop(0, STYLE.world.skyTop)
  grad.addColorStop(0.42, STYLE.palette.magenta)
  grad.addColorStop(0.5, '#fff6d8')     // hot horizon highlight
  grad.addColorStop(0.53, STYLE.world.skyHorizon)
  grad.addColorStop(0.62, '#3a1d3f')
  grad.addColorStop(1, '#0d0a14')
  g.fillStyle = grad; g.fillRect(0, 0, S, S)
  // a couple of bright "window" glints so chrome reads as chrome even when static
  g.fillStyle = 'rgba(255,255,255,0.9)'
  g.fillRect(S * 0.18, S * 0.2, S * 0.08, S * 0.16)
  g.fillRect(S * 0.3, S * 0.24, S * 0.04, S * 0.1)
  // round mask: sphere maps only use the inscribed circle
  tex = new CanvasTexture(c)
  tex.magFilter = NearestFilter; tex.minFilter = NearestFilter; tex.generateMipmaps = false
  tex.colorSpace = SRGBColorSpace
  return tex
}
