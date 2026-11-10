# 3D view — "under the el" (Bushwick)

Lazy-loaded by `src/App.tsx` via `import('./views/three')` and kept mounted once loaded (it shrinks
into the corner thumbnail when the 2D view is active). It only **reads** the game store
(`useGame` in `src/store/gameStore.ts`); event sources live in `src/sources/`.

Two looks, both in `style.config.ts` (**every look/feel knob is there, start there**):

- **Mid-2000s console (default):** PS2 / early-360 at dusk. Renders at 854x480 (`LOOK.internalWidth/Height`), upscaled
  by the browser with bilinear filtering; Lambert / Phong only; lighting baked into vertices at load; textures
  capped at 512 px; bloom on lights only, crushed blacks, warm grade, ordered dither.
- **Late-90s cartoon (`?retro=0`):** cel shading, ink outlines, night. Kept for comparison.

The look is chosen once at load from the URL (`RETRO` in `style.config.ts`).

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
characters/street/ Mid-2000s bodies: CC0 glTF models dressed at load. outfits.ts (who wears what), sculpt.ts (baggy
                   clothes pushed out of the body mesh), texture.ts + prints.ts (paints the 512 px texture in 3D),
                   accessories.ts (shoes, hair cards, hats, belt, chain), StreetSkin.tsx (drives the skeleton).
environment/bakeLighting.ts   Raycast AO + streetlamp reach, baked into an `aBake` vertex attribute at load.
post/RetroGradeEffect.ts      Crushed blacks, warm balance, ordered dither.
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

## How the glTF bodies are driven

`ProceduralCharacter.tsx` still computes everything (poses, reactions, grab IK, tricks) on its joint groups, which
are invisible in the mid-2000s look; only the deck is drawn from them. `street/StreetSkin.tsx` then copies the result
onto the skeleton every frame (priority -0.5: after the rig, before Pieces). So a new pose in `poses.ts` works
in both looks. The joint contract is in `grab/README.md`. Model sources and licences: `../../../../ASSETS.md`.

To regenerate the models: `node scripts/prepare-models.mjs "<unzipped pack>"`.
To look at the characters on their own (dev server only): `/?view=3d&source=live&lineup=1`.
To screenshot + measure FPS: `node scripts/shot-3d.mjs out.png --fps 5` (header lists the options).
