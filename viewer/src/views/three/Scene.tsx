// Everything inside the <Canvas>. The world group shakes (train rumble, slams) instead of the
// camera, so it works the same in broadcast and orbit modes. The night sky dome sits outside the
// shaking group (it follows the camera), and the fog colour matches the sky's horizon so distant
// geometry melts into it.
import { useRef } from 'react'
import { useFrame } from '@react-three/fiber'
import type { Group } from 'three'
import { useEventDirector } from './director'
import { fx } from './viewStore'
import { STYLE } from './style.config'
import { Sky } from './environment/Sky'
import { Street } from './environment/Street'
import { Walls } from './environment/Walls'
import { Fence } from './environment/Fence'
import { Tracks } from './environment/Tracks'
import { Train } from './environment/Train'
import { Lights } from './environment/Lights'
import { Props } from './environment/Props'
import { SetDressing } from './environment/SetDressing'
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
      <color attach="background" args={[STYLE.world.fogColor]} />
      <fog attach="fog" args={[STYLE.world.fogColor, STYLE.world.fogNear, STYLE.world.fogFar]} />
      <Sky />
      <CameraRig />
      <group ref={world}>
        <Lights />
        <Street />
        <Walls />
        <Fence />
        <Tracks />
        <Train />
        <Props />
        <SetDressing />
        <Tables />
        <Pieces />
        <Clocks />
        <Characters />
        <SprayWall />
      </group>
    </>
  )
}
