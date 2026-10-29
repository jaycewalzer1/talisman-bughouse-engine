// Entry point of the 3D view (lazy-loaded by App.tsx). Canvas + HTML overlays.
import { lazy, Suspense } from 'react'
import { Canvas } from '@react-three/fiber'
import { ACESFilmicToneMapping } from 'three'
import { Scene } from './Scene'
import { Hud } from './hud/Hud'
import { CommsLog } from './graffiti/CommsLog'

const Effects = lazy(() => import('./Effects'))

export default function ThreeView() {
  return (
    <div style={{ position: 'relative', width: '100%', height: '100%', minHeight: 400, background: '#06080f', overflow: 'hidden' }}>
      <Canvas
        shadows
        dpr={[1, 1.5]}
        gl={{ antialias: false, powerPreference: 'high-performance', toneMapping: ACESFilmicToneMapping, toneMappingExposure: 1.1 }}
        camera={{ fov: 45, near: 0.05, far: 120, position: [0, 2.6, 6.8] }}
      >
        <Scene />
        <Suspense fallback={null}><Effects /></Suspense>
      </Canvas>
      <Hud />
      <CommsLog />
    </div>
  )
}
