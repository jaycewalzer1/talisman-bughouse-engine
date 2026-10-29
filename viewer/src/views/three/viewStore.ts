// Settings that belong only to the 3D view (camera mode, sound), plus a few shared mutable
// numbers that several scene parts poke every frame (kept outside React state on purpose:
// changing them must not re-render anything).
import { create } from 'zustand'

export type CameraMode = 'broadcast' | 'orbit'

interface ViewState {
  cameraMode: CameraMode
  soundOn: boolean
  showLog: boolean
  setCameraMode: (m: CameraMode) => void
  setSoundOn: (on: boolean) => void
  setShowLog: (on: boolean) => void
}

export const useView = create<ViewState>()((set) => ({
  cameraMode: 'broadcast',
  soundOn: false,          // browsers block audio until the user clicks, so start muted
  showLog: true,
  setCameraMode: (cameraMode) => set({ cameraMode }),
  setSoundOn: (soundOn) => set({ soundOn }),
  setShowLog: (showLog) => set({ showLog }),
}))

/** Per-frame shared values. `shake` decays on its own (see Scene.tsx). */
export const fx = {
  shake: 0,        // 0..1 camera/world shake (train rumble, piece slams)
  trainNear: 0,    // 0..1 how close the train is overhead
}
