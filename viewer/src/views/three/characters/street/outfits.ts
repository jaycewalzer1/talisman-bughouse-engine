// The four skaters in their mid-2000s console look: base body, outfit, hair, accessories. All original
// designs. T1 wears burnt orange up top, T2 steel blue, so the teams read at a glance; pants, hair, hats
// and prints tell the partners apart.
import type { Seat } from '../../../../events/types'

export type Body = 'male' | 'female'
export type TopKind = 'tee' | 'hoodie' | 'layered'    // layered = tee over a long-sleeved shirt
export type HairKind = 'braids' | 'ponytail' | 'short' | 'locs'
export type HatKind = 'none' | 'beanie' | 'cap'
export type PrintKind = 'rookTree' | 'hausSun' | 'crownBurst' | 'pawnSkull'

export interface Outfit {
  body: Body
  skin: [number, number, number]     // multiplies the pack's skin texture (1 = as painted)
  hair: string
  hairKind: HairKind
  hat: HatKind
  hatColor: string
  hatTrim: string
  top: TopKind
  topColor: string
  sleeveColor: string                // layered: the long sleeves under the tee
  trim: string                       // collar, cuffs, sleeve bands
  print: PrintKind
  printInk: string
  printAccent: string
  pants: string
  pantsKind: 'cargo' | 'jeans' | 'work'
  bag: number                        // pant-leg width multiplier
  belt: string
  straps: boolean                    // loose belt straps hanging off the hips
  chain: boolean                     // wallet chain
  necklace: boolean
  shoe: { upper: string; panel: string; accent: string; sole: string; lace: string }
  gloves: string | null              // fingerless gloves
  bands: string[]                    // wristbands, wrist to forearm
  seed: number
}

export const OUTFITS: Record<Seat, Outfit> = {
  // T1: tall, zip-less pullover hoodie, dark baggy jeans, no hat, short fade + stubble
  A_white: {
    body: 'male', skin: [0.98, 0.9, 0.82], hair: '#1c1410', hairKind: 'short', hat: 'none', hatColor: '#000000', hatTrim: '#000000',
    top: 'hoodie', topColor: '#c85a20', sleeveColor: '#c85a20', trim: '#e8b860', print: 'hausSun', printInk: '#2a1208', printAccent: '#f4d48a',
    pants: '#2e3442', pantsKind: 'jeans', bag: 1.08, belt: '#191614', straps: false, chain: true, necklace: false,
    shoe: { upper: '#d8d4cc', panel: '#c8501e', accent: '#2a2a30', sole: '#efe9dc', lace: '#f4f0e6' },
    gloves: null, bands: [], seed: 1.3,
  },
  // T1: shorter, orange graphic tee with a sprawling ink print, olive cargos with hanging straps, braids + beanie
  B_black: {
    body: 'female', skin: [0.56, 0.42, 0.34], hair: '#150d09', hairKind: 'braids', hat: 'beanie', hatColor: '#7c2a1a', hatTrim: '#e3a13c',
    top: 'tee', topColor: '#f08a2a', sleeveColor: '#e07a2c', trim: '#8a2a18', print: 'rookTree', printInk: '#1d1a3c', printAccent: '#1d1a3c',
    pants: '#6c6c38', pantsKind: 'cargo', bag: 1.2, belt: '#3a2a1c', straps: true, chain: false, necklace: true,
    shoe: { upper: '#c9ccd0', panel: '#e08030', accent: '#4a86c8', sole: '#e8e4dc', lace: '#ffffff' },
    gloves: '#3a3a3e', bands: ['#c83a3a', '#e8c040', '#3a9a6a', '#4a70c0'], seed: 4.1,
  },
  // T2: blue hoodie, washed light jeans, long ponytail, trucker cap
  A_black: {
    body: 'female', skin: [1.04, 0.98, 0.94], hair: '#7a3e18', hairKind: 'ponytail', hat: 'cap', hatColor: '#26466e', hatTrim: '#e6e2d6',
    top: 'hoodie', topColor: '#3a78b4', sleeveColor: '#3a78b4', trim: '#d8d4c6', print: 'crownBurst', printInk: '#e8e6da', printAccent: '#f2c84a',
    pants: '#8ea2b8', pantsKind: 'jeans', bag: 1.14, belt: '#2a2422', straps: false, chain: false, necklace: true,
    shoe: { upper: '#3a3c44', panel: '#e6e2d6', accent: '#3a78b4', sole: '#f0ece2', lace: '#e6e2d6' },
    gloves: null, bands: [], seed: 7.7,
  },
  // T2: blue tee over a pale long-sleeve, black work pants, chain, shoulder-length locs
  B_white: {
    body: 'male', skin: [0.68, 0.54, 0.44], hair: '#0e0c0c', hairKind: 'locs', hat: 'none', hatColor: '#000000', hatTrim: '#000000',
    top: 'layered', topColor: '#2c6aa8', sleeveColor: '#c8ccd0', trim: '#123a66', print: 'pawnSkull', printInk: '#ecebe2', printAccent: '#6cc2e8',
    pants: '#26262a', pantsKind: 'work', bag: 1.12, belt: '#111112', straps: false, chain: true, necklace: false,
    shoe: { upper: '#a22a22', panel: '#1e1e22', accent: '#e8e4da', sole: '#e8e4da', lace: '#1e1e22' },
    gloves: '#1e1e22', bands: [], seed: 10.2,
  },
}

/** Rest-pose measurements of the two base bodies (metres, in the GLB's own T-pose). */
export interface BodyMetrics {
  scale: number            // brings the hips to the rig's HIP_Y
  zOffset: number          // centres the model over the rig's spine
  shoulderX: number; shoulderY: number; hipX: number    // rig-space joint positions the proxy skeleton adopts
  pelvisY: number; waistY: number; neckY: number; headY: number; topY: number
  armY: number; armZ: number; shoulderRestX: number; wristX: number; fingerX: number
  torsoRx: number; torsoRz: number; hipRx: number
}

export const BODIES: Record<Body, BodyMetrics> = {
  male: {
    scale: 0.948, zOffset: 0.04, shoulderX: 0.201, shoulderY: 0.402, hipX: 0.108,
    pelvisY: 0.949, waistY: 1.02, neckY: 1.52, headY: 1.6, topY: 1.81,
    armY: 1.455, armZ: -0.068, shoulderRestX: 0.212, wristX: 0.706, fingerX: 0.85,
    torsoRx: 0.22, torsoRz: 0.15, hipRx: 0.19,
  },
  female: {
    scale: 0.975, zOffset: 0.04, shoulderX: 0.148, shoulderY: 0.402, hipX: 0.108,
    pelvisY: 0.932, waistY: 1.0, neckY: 1.485, headY: 1.55, topY: 1.767,
    armY: 1.418, armZ: -0.056, shoulderRestX: 0.152, wristX: 0.641, fingerX: 0.76,
    torsoRx: 0.175, torsoRz: 0.135, hipRx: 0.19,
  },
}
