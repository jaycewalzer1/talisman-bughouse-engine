// Screenshots the 3D view with Playwright and reports FPS. Needs the dev server (npx vite --port 5199).
//   node scripts/shot-3d.mjs <out.png> [--query "view=3d&source=mock"] [--wait 6000] [--fps 5]
//     [--cam "x,y,z,tx,ty,tz"] [--hud 0] [--size 1440x810] [--js "<expression>"] [--throttle 4]
// --cam moves the free camera (x,y,z looking at tx,ty,tz); --hud 0 hides the DOM overlays.
import { chromium } from 'playwright'

const args = process.argv.slice(2)
const out = args[0]
const opt = (k, d) => { const i = args.indexOf(`--${k}`); return i > 0 ? args[i + 1] : d }
const [w, h] = opt('size', '1440x810').split('x').map(Number)
const b = await chromium.launch({ headless: true, args: ['--use-gl=angle', '--use-angle=metal', '--enable-gpu', '--ignore-gpu-blocklist'] })
const p = await b.newPage({ viewport: { width: w, height: h }, deviceScaleFactor: 2 })
const logs = []
if (opt('throttle')) await (await p.context().newCDPSession(p)).send('Emulation.setCPUThrottlingRate', { rate: +opt('throttle') })
p.on('console', m => { if (['error', 'warning'].includes(m.type()) || m.text().startsWith('[retro')) logs.push(`${m.type()}: ${m.text()}`.slice(0, 500)) })
p.on('pageerror', e => logs.push('pageerror: ' + String(e).slice(0, 800)))
await p.goto(`http://localhost:5199/?${opt('query', 'view=3d&source=mock')}`)
await p.waitForTimeout(+opt('wait', 6000))
const cam = opt('cam')
if (cam) {
  await p.evaluate(v => {
    const s = window.__three3d(), [x, y, z, tx, ty, tz] = v.split(',').map(Number)
    s.camera.position.set(x, y, z); s.controls.target.set(tx, ty, tz); s.controls.minDistance = 0.1; s.controls.update()
  }, cam)
  await p.waitForTimeout(400)
}
if (opt('hud') === '0') await p.addStyleTag({ content: 'nav.shell,.pip-card,.pip-tab,[class*=hud],[class*=Hud],[class*=bubble]{display:none!important}' })
const js = opt('js')
if (js) console.log('js:', JSON.stringify(await p.evaluate(js)))
const fpsS = +opt('fps', 0)
if (fpsS > 0) {
  const r = await p.evaluate(s => new Promise(res => {
    let n = 0, worst = 0; const t0 = performance.now(); let last = t0
    const f = t => { n++; worst = Math.max(worst, t - last); last = t; if (t - t0 < s * 1000) requestAnimationFrame(f); else res({ fps: +(n / ((t - t0) / 1000)).toFixed(1), worstFrameMs: +worst.toFixed(1) }) }
    requestAnimationFrame(f)
  }), fpsS)
  console.log('fps', JSON.stringify(r))
}
await p.screenshot({ path: out })
if (logs.length) console.log(logs.slice(0, 30).join('\n'))
await b.close()
