// Settings that belong only to the 3D view (camera mode, sound), plus a few shared mutable
// numbers that several scene parts poke every frame (kept outside React state on purpose:
// changing them must not re-render anything).
import { create } from 'zustand'

/** 'free' = user-driven orbit camera (default); 'auto' = the old broadcast director (opt-in). */
export type CameraMode = 'free' | 'auto'
/** Where a focus glide should go: one board, both boards, or the default home view. */
export type CameraFocus = 'A' | 'B' | 'both' | 'home'

interface ViewState {
  cameraMode: CameraMode
  soundOn: boolean
  showLog: boolean
  /** True while the 3D view is the small picture-in-picture thumbnail (2D mode): camera input
   *  (orbit + keyboard) is off and the HUD is hidden. Set by index.tsx from its `thumb` prop. */
  thumb: boolean
  /** Latest focus request (n bumps on every request so repeats still fire). */
  focusReq: { to: CameraFocus; n: number } | null
  setCameraMode: (m: CameraMode) => void
  setSoundOn: (on: boolean) => void
  setShowLog: (on: boolean) => void
  setThumb: (on: boolean) => void
  /** Glide the free camera to a board / both boards (switches AUTO CAM off). */
  requestFocus: (to: CameraFocus) => void
}

export const useView = create<ViewState>()((set) => ({
  cameraMode: 'free',
  soundOn: false,          // browsers block audio until the user clicks, so start muted
  showLog: true,
  thumb: false,
  focusReq: null,
  setCameraMode: (cameraMode) => set({ cameraMode }),
  setSoundOn: (soundOn) => set({ soundOn }),
  setShowLog: (showLog) => set({ showLog }),
  setThumb: (thumb) => set({ thumb }),
  requestFocus: (to) => set(s => ({ cameraMode: 'free', focusReq: { to, n: (s.focusReq?.n ?? 0) + 1 } })),
}))

/** Per-frame shared values. `shake` decays on its own (see Scene.tsx). */
export const fx = {
  shake: 0,        // 0..1 camera/world shake (train rumble, piece slams)
  trainNear: 0,    // 0..1 how close the train is overhead
}
