// The four skaters. All original designs (early-2000s skate fashion, no real people / brands).
// Team colour dominates (T1 orange, T2 blue) but every garment colour is "washed" — desaturated and
// a touch darker, like worn, faded fabric — so they sit in the grimy night scene and still read.
// Each agent gets its own accent from STYLE.palette so partners still read apart. Silhouettes differ
// at a glance:
//   RAZ  tall + lanky, hood up, headphones round the neck
//   MOOK short + wide, XXL tee to the thighs, pom beanie, cargos, gold chain
//   JUNO medium, layered tee, trucker cap twisted sideways, wide light jeans
//   VEX  slim, zip hoodie, tall slouch beanie + mirrored shades, chain wallet
import { Color } from 'three'
import type { Seat } from '../../../events/types'
import { STYLE } from '../style.config'
import type { CharacterSpec } from './rig'

const P = STYLE.palette

/** Worn-fabric version of a colour: `desat` (0..1) pulls saturation down, `dark` (0..1) lightness. */
function wash(hex: string, desat = 0.25, dark = 0.1): string {
  const c = new Color(hex), hsl = { h: 0, s: 0, l: 0 }
  c.getHSL(hsl)
  c.setHSL(hsl.h, hsl.s * (1 - desat), hsl.l * (1 - dark))
  return '#' + c.getHexString()
}

const T1 = wash(P.T1, 0.22, 0.08), T2 = wash(P.T2, 0.22, 0.08)
const MAG = wash(P.magenta, 0.3, 0.05), LIME = wash(P.lime, 0.35, 0.12), YEL = wash(P.yellow, 0.3, 0.08)
const OFFWHITE = '#d3cdc0', DIRTY = '#bdb6a7'

export const SPECS: Record<Seat, CharacterSpec> = {
  // T1 — accent magenta
  A_white: {
    id: 'RAZ', skin: '#c58c63', hair: '#2a1a12', team: T1, accent: MAG,
    top: 'hoodie', topColor: T1, sleeveColor: T1, trim: MAG,
    pants: 'jeans', pantsColor: '#283149',
    shoe: OFFWHITE, shoeTrim: MAG, sole: '#29262c',
    hat: 'hood', hatColor: T1, hatTrim: MAG,
    deck: MAG, deckArt: T1, wheels: '#d8ccb0',
    height: 1.07, bulk: 1.0, bag: 1.3, headphones: true, wristbands: true, rings: true,
  },
  // T1 — accent yellow
  B_black: {
    id: 'MOOK', skin: '#6e4a33', hair: '#120c08', team: T1, accent: YEL,
    top: 'tee', topColor: wash(P.T1, 0.3, 0.12), sleeveColor: wash(P.T1, 0.3, 0.12), trim: YEL,
    pants: 'cargo', pantsColor: '#5e5440',
    shoe: '#222026', shoeTrim: YEL, sole: DIRTY,
    hat: 'beanie', hatColor: wash(P.T1, 0.35, 0.32), hatTrim: YEL,
    deck: YEL, deckArt: '#1d1b22', wheels: T1,
    height: 0.93, bulk: 1.3, bag: 1.45, chainWallet: true, necklace: 'gold', wristbands: true,
  },
  // T2 — accent lime
  A_black: {
    id: 'JUNO', skin: '#e2b48f', hair: '#6b3b1e', team: T2, accent: LIME,
    top: 'layered', topColor: T2, sleeveColor: LIME, trim: OFFWHITE,
    pants: 'jeans', pantsColor: '#7287a0',
    shoe: wash(P.T2, 0.3, 0.25), shoeTrim: LIME, sole: OFFWHITE,
    hat: 'trucker', hatColor: wash(P.T2, 0.25, 0.3), hatTrim: LIME,
    deck: LIME, deckArt: T2, wheels: OFFWHITE,
    height: 1.0, bulk: 1.0, bag: 1.5, headphones: true, rings: true, necklace: 'silver',
  },
  // T2 — accent magenta
  B_white: {
    id: 'VEX', skin: '#8d5a3b', hair: '#151515', team: T2, accent: MAG,
    top: 'zip', topColor: wash(P.T2, 0.25, 0.15), sleeveColor: wash(P.T2, 0.25, 0.15), trim: MAG,
    pants: 'cargo', pantsColor: '#202329',
    shoe: wash(P.magenta, 0.3, 0.15), shoeTrim: OFFWHITE, sole: OFFWHITE,
    hat: 'slouch', hatColor: '#1e2129', hatTrim: MAG,
    deck: T2, deckArt: MAG, wheels: MAG,
    height: 1.03, bulk: 0.92, bag: 1.2, chainWallet: true, shades: true, wristbands: true,
  },
}
