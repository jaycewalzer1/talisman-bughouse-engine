// Every art-direction knob for the 3D view in one place. Flip things off here if a laptop
// struggles or you want to compare looks. Nothing in here touches game logic.

export const STYLE = {
  render: {
    width: 800,             // internal render resolution; upscaled nearest-neighbour to the window
    height: 560,            // (640x448 read too crunchy with the grimy textures)
    pixelated: true,        // false = smooth (bilinear) upscale
  },
  post: {
    bloom: true,            // only things brighter than ~1.0 (lamp bulbs, neon, train windows) bloom
    bloomIntensity: 0.55,
    bloomThreshold: 0.86,
    bloomSmoothing: 0.12,
    bloomRadius: 0.6,
    grain: true,            // film grain (postprocessing Noise, screen-blended)
    grainOpacity: 0.16,
    dither: true,
    ditherStrength: 0.3,    // 0..1, ordered 4x4 Bayer
    colorLevels: 32,        // quantization steps per channel (lower = crunchier)
    vignette: true,
    vignetteOffset: 0.22,
    vignetteDarkness: 0.78,
  },
  material: {
    vertexSnap: 0,          // 0 = off (PS2-clean). PS1 wobble would be ~ 160
    ambient: 0.75,          // cool blue night ambient (colour: STYLE.world.ambientSky/Ground)
    keyStrength: 0.9,       // warm sodium key from the work lamp overhead (colour: STYLE.world.keyColor)
    chrome: 0.6,            // global multiplier for chrome/env shine (grimy: less bling)
  },
  world: {
    fogNear: 9,
    fogFar: 36,
    skyTop: '#04060d',      // night sky gradient (top -> horizon -> bottom)
    skyHorizon: '#3b2a24',  // faint orange city glow just above the horizon
    skyBottom: '#0b0e14',
    fogColor: '#151a23',    // dark blue-grey; the sky melts into it right at the horizon
    keyColor: '#ffb469',    // sodium key light (shared retroLight uniform)
    ambientSky: '#5468a0',  // cool blue hemisphere ambient, from above
    ambientGround: '#17161c',
    blobShadows: true,
    wetGround: true,        // glossy puddle overlay that reflects the lamps (one cheap transparent plane)
    puddleShine: 1.0,
  },
  palette: {
    T1: '#d9772e',          // sodium orange team (A_white + B_black)
    T2: '#3f88c8',          // steel blue team (A_black + B_white)
    magenta: '#a8467e',
    lime: '#8db54a',
    yellow: '#d8b04a',
    ink: '#0c0d12',
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
    bulbGlow: 2.2,          // >1 so emissive bulbs/signs cross the bloom threshold
    lightPools: true,       // fake additive light decals on the ground/wall (bodega, lamps, train)
    lampCones: true,        // faint additive light cones under the lamps (haze)
    lampConeStrength: 0.07,
  },
} as const

export type StyleConfig = typeof STYLE
