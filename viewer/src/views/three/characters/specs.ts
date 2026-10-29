// The four skaters. All original designs (late-90s / early-2000s street fashion, no real people /
// brands), styled after the flat cel look of MTV's "Downtown" (1999): muted, slightly desaturated
// urban colours — rust, brick, olive, mustard, plum, denim, teal, off-white, warm greys. Team colour
// still dominates (T1 → rust / burnt orange, T2 → denim / teal); every colour is "washed" from
// STYLE.palette here so the toon bands stay calm. Silhouettes differ at a glance:
//   RAZ  tall + lanky, hood up, headphones round the neck
//   MOOK shorter, XXL tee to the thighs, pom beanie, cargos, gold chain
//   JUNO medium, layered tee over long sleeves, trucker cap twisted sideways + headphones, wide light jeans
//   VEX  slim, zip hoodie, tall slouch beanie + shades, chain wallet
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

const RUST = wash(P.T1, 0.4, 0.2)                 // T1 hoodie: burnt orange gone rusty
const BRICK = wash('#b04a32', 0.3, 0.12)
const DENIM = wash(P.T2, 0.38, 0.18)               // T2: washed denim blue
const TEAL = wash('#2f8a8a', 0.4, 0.2)
const PLUM = wash(P.magenta, 0.45, 0.25)
const MUSTARD = wash(P.yellow, 0.35, 0.14)
const OLIVE = wash(P.lime, 0.55, 0.38)
const OFFWHITE = '#d6cfbf', WARMGREY = '#8c857a', DIRTY = '#b7ae9d'

export const SPECS: Record<Seat, CharacterSpec> = {
  // T1 — accent plum
  A_white: {
    id: 'RAZ', skin: '#c08a64', hair: '#2a1a12', team: RUST, accent: PLUM,
    top: 'hoodie', topColor: RUST, sleeveColor: RUST, trim: MUSTARD,
    pants: 'jeans', pantsColor: '#33405a',
    shoe: OFFWHITE, shoeTrim: PLUM, sole: '#3a363a',
    hat: 'hood', hatColor: RUST, hatTrim: PLUM,
    deck: PLUM, deckArt: RUST, wheels: '#d2c6aa',
    height: 1.07, bulk: 1.0, bag: 1.3, headphones: true, wristbands: true, rings: true,
  },
  // T1 — accent mustard
  B_black: {
    id: 'MOOK', skin: '#6e4a33', hair: '#120c08', team: RUST, accent: MUSTARD,
    top: 'tee', topColor: wash('#c8703c', 0.38, 0.14), sleeveColor: wash('#c8703c', 0.38, 0.14), trim: OFFWHITE,
    pants: 'cargo', pantsColor: OLIVE,
    shoe: '#2e2c30', shoeTrim: MUSTARD, sole: DIRTY,
    hat: 'beanie', hatColor: BRICK, hatTrim: MUSTARD,
    deck: MUSTARD, deckArt: '#25232a', wheels: RUST,
    height: 0.93, bulk: 1.3, bag: 1.45, chainWallet: true, necklace: 'gold', wristbands: true,
  },
  // T2 — accent olive
  A_black: {
    id: 'JUNO', skin: '#dcb18e', hair: '#6b3b1e', team: DENIM, accent: MUSTARD,
    top: 'layered', topColor: TEAL, sleeveColor: OLIVE, trim: OFFWHITE,
    pants: 'jeans', pantsColor: '#8396ab',
    shoe: WARMGREY, shoeTrim: OFFWHITE, sole: OFFWHITE,
    hat: 'trucker', hatColor: wash(P.T2, 0.4, 0.4), hatTrim: OFFWHITE,
    deck: OLIVE, deckArt: DENIM, wheels: OFFWHITE,
    height: 1.0, bulk: 1.0, bag: 1.5, headphones: true, rings: true, necklace: 'silver',
  },
  // T2 — accent plum
  B_white: {
    id: 'VEX', skin: '#8d5a3b', hair: '#151515', team: DENIM, accent: PLUM,
    top: 'zip', topColor: DENIM, sleeveColor: DENIM, trim: PLUM,
    pants: 'cargo', pantsColor: '#34363a',
    shoe: PLUM, shoeTrim: OFFWHITE, sole: OFFWHITE,
    hat: 'slouch', hatColor: '#2a2c33', hatTrim: PLUM,
    deck: DENIM, deckArt: PLUM, wheels: PLUM,
    height: 1.03, bulk: 0.92, bag: 1.2, chainWallet: true, shades: true, wristbands: true,
  },
}
