# 3D view — "under the el" (Bushwick, night)

Lazy-loaded by `src/App.tsx` via `import('./views/three')`. It only **reads** the game store
(`useGame` in `src/store/gameStore.ts`); event sources live in `src/sources/`.

## How the files fit together

```
index.tsx            <Canvas> + HTML overlays (HUD, comms log). Post-processing is lazy-loaded.
Scene.tsx            Everything inside the canvas. Mounts the director, shakes the world group.
director.ts          Turns the store's growing `events` array into one-shot callbacks
                     (onGameEvent / onFxReset). Handles every event once; on new game, scrubbing,
                     or >6 events arriving at once it RESETS effects and the scene just shows the
                     store's final state (no animation pile-ups at 8x replay).
layout.ts            All positions: boards, squares, seats, reserve piles, wall. Start here.
viewStore.ts         3D-only settings (camera mode, sound, log) + shared per-frame numbers (shake).
Effects.tsx          Bloom, chromatic aberration, pixelation, film grain, vignette.

environment/         Street (wet asphalt reflector), Walls (+ static graffiti), Fence, Tracks (el),
                     Train (passes every 30–65 s: sound + shake), Lights, Props,
                     textures.ts (all procedural canvas textures), StaticInstances.tsx (instancing helper).
board/               Tables (plywood on milk crates), Clocks (live, red flicker < 10 s, ticks),
                     Pieces (instanced pieces on boards + reserve piles, pooled meshes for animations),
                     anims.ts (move lift/arc/slam, drops from the pile, capture tosses to the partner's
                     pile, castling rook), pieceGeometry.ts (low-poly lathe pieces).
characters/          rig.ts (CharacterRig interface + driver), specs.ts (the 4 looks), poses.ts
                     (per-state target poses + leg IK), ProceduralCharacter.tsx (box-built skater),
                     Characters.tsx (decides each player's state from the store), NameTag.tsx.
graffiti/            sprayText.ts (canvas graffiti painter), SprayWall.tsx (messages painted on the
                     wall, fade after 5 s), CommsLog.tsx (scrollable overlay).
camera/CameraRig.tsx Broadcast auto-cut (follows `lastEventBoard`, min 2.4 s hold, random wide shots)
                     or free orbit (drei OrbitControls).
hud/Hud.tsx          Sticker buttons: camera mode, sound; stall stickers; result banner.
audio/sfx.ts         WebAudio synth: slam, clack, spray hiss, clock tick, train rumble. Muted until
                     the sound sticker is clicked (browser autoplay rules).
```

### Character states
`Characters.tsx` picks one state per player each frame:
`win`/`loss` (game over) → `moving` (their piece is animating) → `stalling` (store `stalls`) →
`thinking` (store `thinking`; fidget speed scales with clock pressure) → `idle`.
`ProceduralCharacter` blends joint rotations toward the state's pose with a damped lerp, so
switches are smooth.

### Reserves
A board's pocket comes from its FEN brackets, but the partner board's FEN only updates when that
board moves. So piles show `pocket(fen) + passes that arrived since that board's last move`
(`reserves()` in `board/anims.ts`).

### Graffiti
Only real comms are painted; `ok` with `payload.kind === "status"` is skipped (and shown dimmed in
the log). All wall art is generated from made-up words with canvas strokes (no real artists' work).

## Swapping in a Mixamo / glTF character
1. Put e.g. `public/models/skater.glb` (with clips like Idle, Thinking, Reach, ArmsCrossed,
   Celebrate, Defeated) in the project.
2. Create `characters/GltfCharacter.tsx` implementing `CharacterRigProps`:
   ```tsx
   const { scene, animations } = useGLTF('/models/skater.glb')
   const clone = useMemo(() => SkeletonUtils.clone(scene), [scene])   // one per player
   const { actions } = useAnimations(animations, group)
   const CLIP = { idle: 'Idle', thinking: 'Thinking', moving: 'Reach', stalling: 'ArmsCrossed', win: 'Celebrate', loss: 'Defeated' }
   useFrame(() => {               // react to driver.current.state changes
     const next = actions[CLIP[driver.current.state]]
     if (next !== current) { current?.fadeOut(0.3); next?.reset().fadeIn(0.3).play(); current = next }
     // optional: driver.current.pressure -> next.timeScale; driver.current.reach -> hand IK
   })
   return <group ref={group}><primitive object={clone} /></group>
   ```
3. In `characters/Characters.tsx` change `const Rig: CharacterRig = ProceduralCharacter` to
   `GltfCharacter` (wrap the canvas content in `<Suspense>` while it loads).

## Replacing procedural textures
Every texture is one function in `environment/textures.ts`. To use CC0 photos (Poly Haven,
ambientCG: "red brick", "asphalt wet", "plywood", "rusty metal"), drop files in `public/textures/`
and return `new TextureLoader().load('/textures/brick_diff.jpg')` with the same wrap/repeat
(set `colorSpace = SRGBColorSpace` on color maps only).

## Performance notes
- Instanced: pieces (12 meshes for all pieces and piles), crates, el columns/ties, fence posts, props.
- Walls are textured planes with a bump map (not individual bricks); fence is one alpha-tested plane per run.
- One shadow-casting light (1024² map). Other lights are unshadowed.
- Ground reflector renders at 256 px. dpr capped at 1.5; MSAA off (pixelation hides aliasing).
- Canvas textures (clocks, name tags) redraw only when their text changes.
- Post-processing is a separate lazy chunk. To go faster: lower `dpr`, drop `Bloom`, or set the
  reflector `resolution` to 128 in `environment/Street.tsx`.
