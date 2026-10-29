// Where the scene's light sources are, shared by the lamp meshes (Lights.tsx, SetDressing.tsx) and
// the wet-ground puddle shader (WetGround.tsx), which reflects them as streaky highlights.
import { WALL_Z } from '../layout'

export const SODIUM = '#ffa24a'

export interface Lamp { pos: [number, number, number]; color: string; power: number }

export const WORK_LAMP: [number, number, number] = [0.3, 5.31, 1.4]
export const STREET_LAMPS: { base: [number, number, number]; arm: number; flicker?: boolean }[] = [
  { base: [-6.5, 4.6, 2.6], arm: 1.2 },
  { base: [5.6, 4.6, -3.2], arm: -1.2, flicker: true },
]
export const BODEGA_X = -5.3

/** Reflected in puddles. Index 2 is the flickering street lamp (see `lampState`). */
export const LAMPS: Lamp[] = [
  { pos: WORK_LAMP, color: '#ffd9a0', power: 1.0 },
  { pos: [STREET_LAMPS[0].base[0] + STREET_LAMPS[0].arm, 4.5, STREET_LAMPS[0].base[2]], color: SODIUM, power: 1.0 },
  { pos: [STREET_LAMPS[1].base[0] + STREET_LAMPS[1].arm, 4.5, STREET_LAMPS[1].base[2]], color: SODIUM, power: 1.0 },
  { pos: [BODEGA_X, 2.95, WALL_Z + 0.25], color: '#ff6a50', power: 0.55 },     // sign
  { pos: [BODEGA_X - 0.75, 1.35, WALL_Z + 0.2], color: '#e8ecd0', power: 0.5 }, // fluorescent window
]

/** Written by the flickering lamp each frame, read by the puddle shader (0 = off, 1 = on). */
export const lampState = { flicker: 1 }
