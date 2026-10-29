// All sounds are synthesized with WebAudio (no files). The AudioContext is only created after the
// user clicks the HUD sound toggle (browser autoplay rules). Every function is a no-op while muted.

let ctx: AudioContext | null = null
let master: GainNode | null = null
let noiseBuf: AudioBuffer | null = null
let brownBuf: AudioBuffer | null = null
let enabled = false
let lastTick = 0
let trainNodes: { stop: () => void } | null = null

function makeNoise(c: AudioContext, brown: boolean): AudioBuffer {
  const len = c.sampleRate * 2
  const buf = c.createBuffer(1, len, c.sampleRate)
  const d = buf.getChannelData(0)
  let last = 0
  for (let i = 0; i < len; i++) {
    const w = Math.random() * 2 - 1
    if (brown) { last = (last + 0.02 * w) / 1.02; d[i] = last * 3.5 } else d[i] = w
  }
  return buf
}

/** Turn sound on/off. Must be called from a click handler the first time. */
export function setSoundEnabled(on: boolean) {
  enabled = on
  if (on && !ctx) {
    const AC = window.AudioContext ?? (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext
    ctx = new AC()
    master = ctx.createGain()
    master.gain.value = 0.6
    master.connect(ctx.destination)
    noiseBuf = makeNoise(ctx, false)
    brownBuf = makeNoise(ctx, true)
  }
  if (ctx) {
    if (on) void ctx.resume()
    else { trainNodes?.stop(); trainNodes = null; void ctx.suspend() }
  }
}

function ready(): AudioContext | null { return enabled && ctx && ctx.state === 'running' ? ctx : null }

function noiseSource(c: AudioContext, brown = false): AudioBufferSourceNode {
  const s = c.createBufferSource()
  s.buffer = brown ? brownBuf : noiseBuf
  s.loop = true
  s.loopStart = Math.random()
  return s
}

/** Heavy piece slammed onto plywood. */
export function slam(strength = 1) {
  const c = ready(); if (!c || !master) return
  const t = c.currentTime
  const o = c.createOscillator(); o.type = 'sine'
  o.frequency.setValueAtTime(140, t); o.frequency.exponentialRampToValueAtTime(45, t + 0.12)
  const g = c.createGain(); g.gain.setValueAtTime(0.7 * strength, t); g.gain.exponentialRampToValueAtTime(0.001, t + 0.18)
  o.connect(g).connect(master); o.start(t); o.stop(t + 0.2)
  const n = noiseSource(c); const f = c.createBiquadFilter(); f.type = 'bandpass'; f.frequency.value = 1800; f.Q.value = 0.8
  const ng = c.createGain(); ng.gain.setValueAtTime(0.35 * strength, t); ng.gain.exponentialRampToValueAtTime(0.001, t + 0.06)
  n.connect(f).connect(ng).connect(master); n.start(t, Math.random()); n.stop(t + 0.08)
}

/** Small clack (tossed piece landing in a reserve pile). */
export function clack() { slam(0.35) }

/** Spray can hiss with a rattle at the start. */
export function spray(duration = 0.8) {
  const c = ready(); if (!c || !master) return
  const t = c.currentTime
  for (let i = 0; i < 3; i++) { // can rattle
    const o = c.createOscillator(); o.type = 'square'; o.frequency.value = 2400 + i * 300
    const g = c.createGain(); const ti = t + i * 0.07
    g.gain.setValueAtTime(0.05, ti); g.gain.exponentialRampToValueAtTime(0.001, ti + 0.03)
    o.connect(g).connect(master); o.start(ti); o.stop(ti + 0.04)
  }
  const n = noiseSource(c); const f = c.createBiquadFilter(); f.type = 'highpass'; f.frequency.value = 3500
  const g = c.createGain(); const t0 = t + 0.22
  g.gain.setValueAtTime(0.0001, t0); g.gain.linearRampToValueAtTime(0.16, t0 + 0.05)
  g.gain.setValueAtTime(0.16, t0 + duration - 0.1); g.gain.linearRampToValueAtTime(0.0001, t0 + duration)
  n.connect(f).connect(g).connect(master); n.start(t0, Math.random()); n.stop(t0 + duration + 0.05)
}

/** Clock tick (used under 10 seconds). Throttled. */
export function tick(urgent = false) {
  const c = ready(); if (!c || !master) return
  const now = performance.now(); if (now - lastTick < 120) return; lastTick = now
  const t = c.currentTime
  const o = c.createOscillator(); o.type = 'square'; o.frequency.value = urgent ? 1500 : 1100
  const g = c.createGain(); g.gain.setValueAtTime(0.06, t); g.gain.exponentialRampToValueAtTime(0.001, t + 0.03)
  o.connect(g).connect(master); o.start(t); o.stop(t + 0.04)
}

/** Elevated train passing overhead: low rumble + wheel clacks, swelling and fading over `duration` s. */
export function train(duration: number) {
  const c = ready(); if (!c || !master) return
  trainNodes?.stop()
  const t = c.currentTime
  const env = c.createGain()
  env.gain.setValueAtTime(0.0001, t)
  env.gain.linearRampToValueAtTime(0.9, t + duration * 0.45)
  env.gain.linearRampToValueAtTime(0.9, t + duration * 0.55)
  env.gain.linearRampToValueAtTime(0.0001, t + duration)
  env.connect(master)
  const rumble = noiseSource(c, true)
  const lp = c.createBiquadFilter(); lp.type = 'lowpass'; lp.frequency.value = 220
  rumble.connect(lp).connect(env)
  const hum = c.createOscillator(); hum.type = 'sawtooth'; hum.frequency.value = 48
  const humLp = c.createBiquadFilter(); humLp.type = 'lowpass'; humLp.frequency.value = 120
  const humG = c.createGain(); humG.gain.value = 0.25
  hum.connect(humLp).connect(humG).connect(env)
  // wheel clacks: a bandpassed noise gated by a square LFO
  const clk = noiseSource(c); const bp = c.createBiquadFilter(); bp.type = 'bandpass'; bp.frequency.value = 900; bp.Q.value = 2
  const gate = c.createGain(); gate.gain.value = 0
  const lfo = c.createOscillator(); lfo.type = 'square'; lfo.frequency.value = 7
  const lfoG = c.createGain(); lfoG.gain.value = 0.18
  lfo.connect(lfoG).connect(gate.gain)
  clk.connect(bp).connect(gate).connect(env)
  const nodes = [rumble, hum, clk, lfo]
  nodes.forEach(n => n.start(t))
  nodes.forEach(n => n.stop(t + duration + 0.1))
  trainNodes = { stop: () => { try { env.gain.cancelScheduledValues(c.currentTime); env.gain.setValueAtTime(0.0001, c.currentTime); nodes.forEach(n => n.stop()) } catch { /* already stopped */ } } }
}
