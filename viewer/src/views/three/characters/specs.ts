// The four looks. Team T1 wears warm oranges/yellows, team T2 teals/cyans.
import type { Seat } from '../../../events/types'
import type { CharacterSpec } from './rig'

export const SPECS: Record<Seat, CharacterSpec> = {
  // T1 — tall and lanky, hood up, oversized hoodie
  A_white: {
    id: 'RAZ', skin: '#c58c63', top: '#ff7a1a', sleeves: '#ff7a1a', accent: '#1b1b1b', pants: '#1f1f22',
    shoes: '#f0f0f0', hat: 'hood', hatColor: '#ff7a1a', deck: '#c4262c', height: 1.06, bulk: 1.15, bagginess: 1.35,
  },
  // T1 — short and stocky, beanie, puffer vest, cargo shorts + tall socks
  B_black: {
    id: 'MOOK', skin: '#6e4a33', top: '#151515', sleeves: '#ffb21f', accent: '#ffb21f', pants: '#8a7a55', shorts: true,
    shoes: '#2b2b2b', hat: 'beanie', hatColor: '#ffd23a', deck: '#f2f2f2', height: 0.92, bulk: 1.3, bagginess: 1.2, chain: true,
  },
  // T2 — backwards snapback, flannel, wide jeans, headphones
  A_black: {
    id: 'JUNO', skin: '#e2b48f', top: '#126b72', sleeves: '#126b72', accent: '#0a2f33', pants: '#6f8fb5',
    shoes: '#d9d2c0', hat: 'cap-back', hatColor: '#1fc9d6', deck: '#2b2b80', height: 1.0, bulk: 1.0, bagginess: 1.45, headphones: true,
  },
  // T2 — bucket hat, long white tee with cyan stripe, track pants, shades
  B_white: {
    id: 'VEX', skin: '#8d5a3b', top: '#ecebe6', sleeves: '#ecebe6', shortSleeves: true, accent: '#27d8e6', pants: '#14213d',
    shoes: '#27d8e6', hat: 'bucket', hatColor: '#27d8e6', deck: '#1d9a50', height: 0.98, bulk: 0.95, bagginess: 1.15, shades: true,
  },
}
