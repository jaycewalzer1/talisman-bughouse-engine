// Everything inside the <Canvas>. The world group shakes (train rumble, slams) instead of the
// camera, so it works the same in broadcast and orbit modes.
import { useRef } from 'react'
import { useFrame } from '@react-three/fiber'
import type { Group } from 'three'
import { useEventDirector } from './director'
import { fx } from './viewStore'
import { Street } from './environment/Street'
import { Walls } from './environment/Walls'
import { Fence } from './environment/Fence'
import { Tracks } from './environment/Tracks'
import { Train } from './environment/Train'
import { Lights } from './environment/Lights'
import { Props } from './environment/Props'
import { Tables } from './board/Tables'
import { Pieces } from './board/Pieces'
import { Clocks } from './board/Clocks'
import { Characters } from './characters/Characters'
import { SprayWall } from './graffiti/SprayWall'
import { CameraRig } from './camera/CameraRig'

export function Scene() {
  useEventDirector()
  const world = useRef<Group>(null)

  useFrame((_, dt) => {
    const g = world.current; if (!g) return
    const a = fx.shake * 0.035
    g.position.set((Math.random() - 0.5) * a, (Math.random() - 0.5) * a, (Math.random() - 0.5) * a * 0.5)
    fx.shake *= Math.exp(-Math.min(dt, 0.1) * 5)
  })

  return (
    <>
      <color attach="background" args={['#06080f']} />
      <fog attach="fog" args={['#0a0e1a', 9, 34]} />
      <CameraRig />
      <group ref={world}>
        <Lights />
        <Street />
        <Walls />
        <Fence />
        <Tracks />
        <Train />
        <Props />
        <Tables />
        <Pieces />
        <Clocks />
        <Characters />
        <SprayWall />
      </group>
    </>
  )
}
