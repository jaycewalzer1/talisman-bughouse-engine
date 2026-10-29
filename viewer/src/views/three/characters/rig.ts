// The contract between "game logic" (Characters.tsx decides WHAT a player is doing) and a
// character rig (HOW it looks doing it). Any component with these props can be a rig:
//   - ProceduralCharacter.tsx (current): merged low-poly skater parts + code-driven poses
//   - a future GltfCharacter.tsx: Mixamo model via useGLTF + useAnimations (see README)
import type { JSX } from 'react'
import type { Vector3 } from 'three'
import type { Seat } from '../../../events/types'
import type { Reaction } from '../fxBus'
import type { Trick } from '../tricks/types'

export type CharacterState = 'idle' | 'thinking' | 'moving' | 'stalling' | 'win' | 'loss'

/** Mutable, updated every frame by Characters.tsx. Rigs READ it in useFrame (no React re-render). */
export interface CharacterDriver {
  state: CharacterState
  since: number           // seconds (clock.elapsedTime) when `state` began
  pressure: number        // 0..1 clock pressure (1 = almost flagging)
  lookYaw: number         // radians, rig-local yaw toward the partner's board (used when stalling)
  reach: Vector3 | null   // world point the hand should reach for (piece being moved), if any
  seed: number            // per-character random offset so idles don't sync up
  /** One-shot overlay (fist pump, head-in-hands, ...) blended over the state pose. */
  reaction: Reaction | null
  reactionAt: number      // clock seconds when `reaction` started
  /** Queued by event handlers (outside the frame loop); Characters.tsx starts it when sensible. */
  pending: Reaction | null
  pendingAt: number       // performance.now() / 1000 when queued
  /** This skater's win trick (tricks/), performed in the 'win' state. */
  trick: Trick | null
}

export type TopKind = 'hoodie' | 'tee' | 'zip' | 'layered'
export type HatKind = 'hood' | 'beanie' | 'trucker' | 'slouch'

export interface CharacterSpec {
  id: string               // look / handle (not the AI's name)
  skin: string
  hair: string
  team: string             // dominant team colour
  accent: string           // per-agent secondary colour (partners read apart)
  top: TopKind
  topColor: string
  sleeveColor: string      // long sleeves (layered: the undershirt)
  trim: string             // cuffs, hem, drawstrings, prints
  pants: 'jeans' | 'cargo'
  pantsColor: string
  shoe: string
  shoeTrim: string
  sole: string
  hat: HatKind
  hatColor: string
  hatTrim: string
  deck: string             // deck bottom colour
  deckArt: string          // stripe on the deck bottom
  wheels: string
  height: number           // overall scale (1 = 1.8 m)
  bulk: number             // torso width multiplier
  bag: number              // pant-leg width multiplier
  headphones?: boolean     // worn around the neck
  chainWallet?: boolean
  necklace?: 'silver' | 'gold'
  shades?: boolean
  wristbands?: boolean
  rings?: boolean
}

export interface CharacterRigProps {
  seat: Seat
  spec: CharacterSpec
  driver: { current: CharacterDriver }
  teamColor: string
}

export type CharacterRig = (props: CharacterRigProps) => JSX.Element
