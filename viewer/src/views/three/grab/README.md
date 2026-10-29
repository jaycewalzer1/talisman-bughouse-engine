# grab/ — hands that pick up, carry and slam chess pieces

Self-contained: plain modules, no React, no new dependencies.

| file | what |
|---|---|
| `ik.ts` | `solveArm()`: analytic two-bone IK (shoulder → elbow → wrist) on plain `Object3D`s with a pole vector, wrist aim, and a reach budget for far squares (spine lean → root slide → arm stretch). Blends with the FK pose by `weight`. `armPole()` helper. |
| `hand.ts` | `buildHand({ side, skin, glove, ring, material })`: rounded ~198-tri fingerless-glove hand, built as 5 vertex-coloured meshes (palm+cuff, proximal finger bank, distal finger bank, 2 thumb segments) sharing one material (the skaters' matte retro material). `setGrip(0..1)` / `setPose({grip, spread, pinch})`, and a `socket` Object3D where a held piece's grab point sits. |
| `grabChoreo.ts` | `grabBus`: per-seat timeline reach → grip → carry → place (slam + wrist flick) → release → return, plus an off-hand that plucks captured pieces. |
| `../../../../scripts/grab-check.ts` | headless check (see below). |

## How it works

**Who owns timing.** `board/anims.ts` keeps owning the piece path. On a move it asks
`grabBus.reachLead(gap)` for a lead time (0 to 0.32 s, shrinking to 0 in fast replays), starts the
piece flight that much later (the piece waits on its source square while the hand comes over), and
calls `grabBus.requestMove({ seat, letter, from, to, start, dur, drop, capture, path })`. `path(t, out)`
is the same function Pieces uses to draw the piece, so target and piece always agree. There is no
one-frame lag and no reparenting.

**Each frame the rig** runs its FK pose / state machine first, then:

```ts
const s = grabBus.sample(seat)              // phase, weight, grip, target, handDir, palmDir, holding
hand.setPose({ grip: s.grip, spread: s.spread, pinch: s.pinch })
if (s.active) solveArm(
  { shoulder: shR, elbow: elR, wrist: wrR },
  { target: s.target, effector: hand.socket, pole: armPole(root, -1), handDir: s.handDir, palmDir: s.palmDir, weight: s.weight },
  { spine: [hips, spine], maxLeanPerJoint: 0.6, rootOffset: reachGroup, maxRootShift: 0.15, maxStretch: 1.08 },
)
// same with grabBus.sampleOff(seat) + the left arm (armPole(root, +1)) for captures
```

The hand's grip socket then sits on the piece's grab point (base + `grabHeight(letter)`, a bit below
the top so the fingers wrap the head) for the whole carry.

**IK details.**
- The elbow goes in the plane of target and pole, using the law of cosines. Joint rotations come from
  mapping each bone's rest frame (bone direction + bend axis) onto the solved frame, so the code
  works for any rest orientation. For our skaters the arm hangs along -y and bends about -x, which
  is the default `bendAxis`.
- Far targets: (1) lean the spine joints one CCD-style rotation each, capped per joint; (2) slide
  `rootOffset` horizontally for the remaining distance (the skater rolls forward on the deck);
  (3) lean again; (4) stretch the bone offsets up to `maxStretch`.
- Wrist bend is capped at `maxWristBend`. If the cap kicks in, the effector offset is recomputed and
  the arm is solved again (at most 3 passes).
- `weight` slerps each joint between its FK and IK rotations. Nothing accumulates: FK rewrites the
  joints each frame and the IK sets `rootOffset` and the bone offsets absolutely. So
  **`rootOffset` must be an empty group the IK owns** (wrap the rig in it).

**Hand frame.** Wrist-local: fingers along -y, palm normal +z, +x = character's left. `side = -1`
builds a right hand. The socket is at `SOCKET_LOCAL` (between the curled fingers and the palm).
`grip ≈ 0.72` wraps a piece.

**Captures (bonus).** `sampleOff` drives the left hand to the captured piece during the reach. It
grips it at lift-off, swings it off the board edge on the mover's left, and lets go with a flick at 60%
of the flight. `capturedPos(seat)` tells Pieces where to draw the captured ("ghost") piece while it is
held (null once released).

## How it is wired in (current state)

- `characters/ProceduralCharacter.tsx` builds both hands and attaches them to wrist joints at
  `(0, -WRIST, 0)` under each elbow. A per-skater `reach` group wraps the rig and is the IK's
  `rootOffset`. Each frame it runs: FK → pose IK (`solveArm` toward the pose's `ikL/ikR`, aiming
  a palm point at `(0, -FOREARM, 0)`) → grab IK on the right arm (`grabBus.sample`, with a lean/slide
  budget) → off-hand IK on the left (`sampleOff`) → finger poses → `reportSocket` / `reportRest`.
  If the IK leaned the hips, the legs are counter-rotated so the feet stay planted.
- Frame order: Characters' driver runs at useFrame priority -2, the rig at -1, Pieces at 0. So
  `Pieces` draws the held piece at the hand's real socket (`grabBus.heldBase`) and the plucked
  capture at the off hand (`grabBus.capturedPos`) in the same frame.
- `board/anims.ts` adds the reach lead before lift-off (`grabBus.reachLead(gap)`), exports
  `pathAt()`, registers each move with `grabBus.requestMove(...)` and its exact path, and starts the
  pass toss from the off hand's release point (`grabBus.captureRelease`).

## Headless check

```
npx esbuild scripts/grab-check.ts --bundle --platform=node --format=esm --outfile=node_modules/.tmp/grab-check.mjs \
  && node node_modules/.tmp/grab-check.mjs
```

It runs 1896 IK solves: every square for king and pawn heights, plus reserve slots, from all 4 seats
at 3 character heights. Pass condition: error < 1 cm (actual worst is 0.19 mm). It also checks:
- the elbow stays on the pole side and never hyperextends
- weight 0 / 0.5 / 1 blending, and stability across frames
- the hand has 100 to 200 tris and its fingertips close on the socket
- the phase order and holding flags for a normal move, a capture and a fast replay
- an end-to-end run where the socket stays on the carried piece

## Credits / licences

No third-party code is copied. The techniques come from:
- Analytic two-joint IK with a pole/hint vector: D. Holden, *Simple Two Joint IK*,
  https://theorangeduck.com/page/simple-two-joint (article with no licence, so only the maths is
  used), and Godot `TwoBoneIK3D` (MIT, https://github.com/godotengine/godot).
- The CCD-style spine lean follows three.js `examples/jsm/animation/CCDIKSolver.js` (MIT, ships with
  three r169). It is not used directly because it needs a `SkinnedMesh` + `Skeleton`.
