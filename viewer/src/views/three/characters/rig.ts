// The contract between "game logic" (Characters.tsx decides WHAT a player is doing) and a
// character rig (HOW it looks doing it). Any component with these props can be a rig:
//   - ProceduralCharacter.tsx (current): low-poly primitives + code-driven poses
//   - a future GltfCharacter.tsx: Mixamo model via useGLTF + useAnimations (see README)
import type { JSX } from 'react'
import type { Vector3 } from 'three'

export type CharacterState = 'idle' | 'thinking' | 'moving' | 'stalling' | 'win' | 'loss'

/** Mutable, updated every frame by Characters.tsx. Rigs READ it in useFrame (no React re-render). */
export interface CharacterDriver {
  state: CharacterState
  since: number           // seconds (clock.elapsedTime) when `state` began
  pressure: number        // 0..1 clock pressure (1 = almost flagging)
  lookYaw: number         // radians, rig-local yaw toward the partner's board (used when stalling)
  reach: Vector3 | null   // world point the hand should reach for (piece being moved), if any
  seed: number            // per-character random offset so idles don't sync up
}

export type HatKind = 'hood' | 'beanie' | 'cap-back' | 'bucket'

export interface CharacterSpec {
  id: string               // look name (not the AI's name)
  skin: string
  top: string              // hoodie / tee / vest body
  sleeves: string
  shortSleeves?: boolean   // forearms show skin
  accent: string           // pocket, stripe, flannel lines, hat accent
  pants: string
  shorts?: boolean         // cargo shorts + tall socks
  shoes: string
  hat: HatKind
  hatColor: string
  deck: string             // deck bottom color
  height: number           // overall scale (1 = 1.8 m)
  bulk: number             // torso width multiplier
  bagginess: number        // leg thickness multiplier
  headphones?: boolean
  chain?: boolean
  shades?: boolean
}

export interface CharacterRigProps {
  spec: CharacterSpec
  driver: { current: CharacterDriver }
  teamColor: string
}

export type CharacterRig = (props: CharacterRigProps) => JSX.Element
