# 3D view — "under the el" (Bushwick, night)

Lazy-loaded by `src/App.tsx` via `import('./views/three')` and kept mounted once loaded (it shrinks
into the corner thumbnail when the 2D view is active). It only **reads** the game store
(`useGame` in `src/store/gameStore.ts`); event sources live in `src/sources/`.

Look: a flat late-90s TV-cartoon style (cel shading, ink outlines, muted urban palette, painted
backgrounds) rendered in 3D. **Every look/feel knob is in `style.config.ts`**: render dpr and
antialiasing, toon bands, outline widths, ink edge pass, bloom, vignette, fog/sky/lighting colours,
palette, comms, HUD and camera settings. Start there.

## How the files fit together

```
index.tsx          <Canvas> (full-dpr render), HUD overlay, thumbnail mode.
Scene.tsx          Everything inside the canvas; mounts the director; shakes the world group.
Effects.tsx, post/ Bloom -> depth ink edges (post/InkEdgeEffect.ts) -> SMAA -> gentle vignette.
style.config.ts    All settings.
director.ts        Turns the store's events into one-shot callbacks (onGameEvent / onFxReset).
fxBus.ts           In-view bus for game "moments" (capture, drop, talk, announce, reaction).
layout.ts          Where boards, seats, reserve piles and the wall sit in world space.
viewStore.ts       View-only state: camera mode, sound, thumbnail flag; shared `fx.shake`.

materials/         retroMaterial (2-tone cel shading, fog, banded chrome) used by everything;
                   outline.ts (inverted-hull ink outlines: withOutline / outlineTree); chrome map.
environment/       Street + wet puddles, brick walls + graffiti, fence, el tracks, train, lamps,
                   sky, bodega/boombox/rail props, blob shadows, light pools, textures.ts.
board/             Milk-crate tables, chess pieces + move/drop/toss animation, chess clocks.
characters/        The four skaters: smooth low-poly parts (loft.ts, skaterParts.ts), specs,
                   pose state machine (poses.ts), rig (ProceduralCharacter.tsx), reactions,
                   win tricks (trickPose.ts), name tags, one-time portrait snapshots.
grab/              Two-bone arm IK, hands with curling fingers, and the reach -> grip -> carry ->
                   slam -> release choreography synced to board/anims.ts. See grab/README.md.
tricks/            Win tricks as pure functions of time (tre flip, kickflip, 360 shuv, ollie
                   impossible); `WIN_TRICK_OF` in tricks/types.ts picks who does which.
comms/             Street-talk line table (lines.ts — edit the lines here), moments.ts (game
                   events -> fxBus), speech bubbles, portrait cut-ins, announcer.
hud/               Trick text + combos, team boost meters, race clocks, camera/sound buttons.
camera/            Free camera (default: drag/pan/zoom, WASD/QE/R, 1/2/0 focus) and the opt-in
                   AUTO CAM director; shots.ts has the shot list + occlusion check.
graffiti/          Wall graffiti: glyphs.ts (letter skeletons), painter.ts (pieces, throw-ups, tags, extras),
                   wallArt.ts (wall layouts, one canvas per wall), CREDITS.md; the old comms log.
audio/sfx.ts       WebAudio-synthesised sounds (muted until the HUD sound button is clicked).
```

## Checks

From `viewer/`: `npx tsc -b`, `npx vite build`, and the headless scripts —
`scripts/grab-check.ts`, `scripts/trick-rig-check.ts`, `src/views/three/tricks/*.check.ts`
(bundle with esbuild, run with node; each file's header has the command).

## Swapping in real models later

Characters are procedural. To use rigged glTF models instead, keep the joint contract in
`grab/README.md` (joint names/axes the IK, hands and tricks rely on) and build the rig from
`useGLTF` + bone lookups instead of `skaterParts.ts`.
