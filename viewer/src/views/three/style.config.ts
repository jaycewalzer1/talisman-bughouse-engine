// Every art-direction knob for the 3D view in one place. Flip things off here if a laptop
// struggles or you want to compare looks. Nothing in here touches game logic.

export const STYLE = {
  render: {
    maxDpr: 2,              // full device-pixel-ratio render, capped (2 = retina; 1.5 helps weak iGPUs)
    msaa: 0,                // composer multisample count (0 = off -> SMAA). >0 rendered a blank frame on Apple M-series + postprocessing 6.37; SMAA is cheaper anyway
    smaa: true,             // post antialiasing when msaa is 0
  },
  post: {
    bloom: true,            // only things brighter than ~1.0 (lamp bulbs, neon, train windows) bloom
    bloomIntensity: 0.35,
    bloomThreshold: 0.9,
    bloomSmoothing: 0.1,
    bloomRadius: 0.5,
    inkEdges: true,         // ink lines from the depth buffer (silhouettes + creases of the set)
    inkWidth: 1.75,         // CSS pixels (scaled by dpr)
    inkSilhouette: 0.035,   // relative depth jump that draws a line
    inkCrease: 0.025,       // relative depth bend that draws a crease line
    inkStrength: 0.92,
    inkFadeNear: 16,        // lines fade with distance (far skyline stays soft)
    inkFadeFar: 42,
    vignette: true,         // gentle, optional
    vignetteOffset: 0.35,
    vignetteDarkness: 0.4,
  },
  toon: {
    enabled: true,          // false = smooth Gouraud lighting (old look) instead of 2-tone cel bands
    shadowTone: 0.58,       // shadow band brightness relative to the lit band (0..1)
    threshold: 0.32,        // N.L above this = lit band
    softness: 0.03,         // width of the band edge (0 = razor; a hair keeps it antialiased)
    shadowTint: '#7d7fb8',  // the shadow band is tinted cool (cel painters' blue-violet shadows)
    outlines: true,         // inverted-hull ink outlines on characters / pieces / big props
    outlineWidth: 2.4,      // pixels at 1080p
    inkColor: '#14110f',
    paintedTextures: true,  // set textures painted cel-style (flat fields + ink lines); false = old grimy noise
  },
  material: {
    vertexSnap: 0,          // 0 = off (PS2-clean). PS1 wobble would be ~ 160
    ambient: 0.95,          // cool blue night ambient (colour: STYLE.world.ambientSky/Ground)
    keyStrength: 0.85,      // warm sodium key from the work lamp overhead (colour: STYLE.world.keyColor)
    chrome: 0.5,            // global multiplier for chrome/env shine (grimy: less bling)
  },
  world: {
    fogNear: 16,            // painted-background haze, not murk
    fogFar: 62,
    skyTop: '#1d2442',      // soft painted night gradient (top -> horizon -> bottom)
    skyHorizon: '#8c5d4a',  // dusty rust city glow at the horizon
    skyBottom: '#23252e',
    fogColor: '#4a4454',    // warm grey-plum; the sky melts into it right at the horizon
    keyColor: '#f2c48c',    // sodium key light (shared retroLight uniform), softened for cel bands
    ambientSky: '#8890b8',  // cool flat ambient (toon uses a fixed mix of sky/ground, no gradient)
    ambientGround: '#3c3644',
    blobShadows: true,
    wetGround: true,        // glossy puddle overlay that reflects the lamps (one cheap transparent plane)
    puddleShine: 1.0,
  },
  palette: {
    T1: '#c0632f',          // rust orange team (A_white + B_black)
    T2: '#4a79a6',          // denim blue team (A_black + B_white)
    magenta: '#8c4a72',     // plum
    lime: '#8a9a48',        // olive
    yellow: '#c99b3c',      // mustard
    ink: '#14110f',
    // Downtown-ish set colours: flat muted fields
    brick: '#8a4a3a', teal: '#3f7a78', olive: '#6e7040', offWhite: '#d8d0bc', warmGrey: '#7a726a',
  },
  comms: {
    bubbles: true,
    portraits: true,
    announcer: true,
    statusBubbleCooldownMs: 4000,   // "PEEP THIS LINE" at most this often per agent
    bubbleHoldMs: 2200,
    showLog: false,                 // old chat-log overlay (off: bubbles replace it)
    maxBubblesPerTeam: 2,           // visible at once per screen edge; the rest queue
    bubbleQueueMax: 3,              // queued bubbles per team beyond this are dropped (oldest status first)
    passBubbles: true,              // "HERE, CATCH!" bubble when a captured piece is sent
    announcerHoldMs: 1500,          // how long a hype text stays up
    announcerGapMs: 900,            // min gap between two hype texts
  },
  hud: {
    trickText: true,
    boostMeters: true,
    raceClocks: true,
    comboWindowMs: 4000,            // same-team tricks within this window stack a combo (x2, x3...)
    boostClockWeight: 0.3,          // boost meter = (1-w) * material edge + w * clock edge
    boostMaterialScale: 12,         // material lead (points) that fills the meter completely
  },
  camera: {
    idleOrbit: true,
    idleOrbitSpeed: 0.06,   // rad/s
    punchIn: true,          // FOV punch + shake on captures and drops
    talkCutIns: true,       // quick close-up on the agent who speaks
    idleAfterMs: 5000,      // no game event for this long -> slow orbit
    minHoldMs: 1500,        // never cut faster than this
    boardSwitchHoldMs: 2200,// min hold before following the other board
    talkCutMs: 1200,        // talk close-up length
    talkCutCooldownMs: 4500,// at most one talk close-up per this long
    punchFov: 7,            // degrees of FOV punch-in on capture/drop
    // free (user-driven) camera — the default. The keys above only drive AUTO CAM, except punchIn.
    freePunchFov: 2.5,      // degrees of FOV punch in free mode (never moves the camera itself)
    freeShakeScale: 0.6,    // scales the world shake on captures/drops in free mode
    keyPanSpeed: 0.55,      // WASD/arrows pan speed, in camera-distances per second
    keyRotateSpeed: 1.4,    // Q/E orbit speed, rad/s
    focusGlideMs: 900,      // 1/2/0/R glide duration
    hintMs: 7000,           // how long the controls hint stays on screen
  },
  scene: {
    skyline: true,          // far dark tower silhouettes with sparse lit windows (one instanced draw)
    stringLights: true,     // warm bulbs hung from the el
    bulbGlow: 1.4,          // >1 so emissive bulbs/signs cross the bloom threshold
    lightPoolScale: 0.55,   // global multiplier on the light-pool decals + cones (cel: flat, not glowing)
    lightPools: true,       // fake additive light decals on the ground/wall (bodega, lamps, train)
    lampCones: true,        // faint additive light cones under the lamps (haze)
    lampConeStrength: 0.07,
  },
  graffiti: {
    enabled: true,          // the painted wall layer (one decal plane per wall)
    pxPerMeter: 160,        // canvas resolution: back wall 26 m -> 4160 x 960, side wall 2080 x 960
    tagCount: 140,          // handstyle tags scattered in the gaps on the back wall (side gets ~half)
    oldLayer: true,         // older faded pieces + tags underneath the current ones
    oldFade: 0.42,          // 0..1 how far the old layer is washed back toward the wall colour
    paint: '#ece6da',       // tint on the paint layer (dims it a touch so it sits on the brick)
  },
} as const

export type StyleConfig = typeof STYLE
