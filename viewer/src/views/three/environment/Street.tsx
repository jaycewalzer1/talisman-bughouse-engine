// Wet asphalt with cheap puddle reflections (low-res reflector, puddles = low roughness).
import { MeshReflectorMaterial } from '@react-three/drei'
import { asphaltRoughness, asphaltTexture } from './textures'

export function Street() {
  return (
    <group>
      <mesh rotation-x={-Math.PI / 2} receiveShadow>
        <planeGeometry args={[44, 34]} />
        <MeshReflectorMaterial
          map={asphaltTexture()}
          roughnessMap={asphaltRoughness()}
          roughness={1}
          metalness={0.35}
          color="#8a8a90"
          resolution={256}
          blur={[300, 80]}
          mixBlur={1.2}
          mixStrength={3.5}
          mixContrast={1.1}
          mirror={0.6}
          depthScale={0.6}
          minDepthThreshold={0.3}
          maxDepthThreshold={1.4}
        />
      </mesh>
      {/* curb along the wall */}
      <mesh position={[0, 0.08, -3.95]} receiveShadow>
        <boxGeometry args={[30, 0.16, 0.5]} />
        <meshStandardMaterial color="#55534f" roughness={0.95} />
      </mesh>
    </group>
  )
}
