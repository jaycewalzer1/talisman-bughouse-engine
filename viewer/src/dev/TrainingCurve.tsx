// Training step of durative.html: each lineage's gate games as win rate per 100-game block, coloured by iteration,
// with what changed in each iteration. Data: /api/training (vite.devApi.ts), which runs cluster/training_curve.py on
// PACE over runs/full/* and the ab2 arms.

import { useEffect, useState } from 'react'

interface Block { games: number; score: number }
interface Match { games: number; score: number; elo: number | null; ci: number[] | null; finished: boolean; same_as_generator: boolean; blocks: Block[] }
interface Iteration {
  iteration: number; done: boolean; selfplay_games: number; positions: number; segments: number; flag_games: number
  repeat_records: number; promoted: boolean | null; generator: string | null
  train: { steps: number; replay_fraction: number; new_positions: number; replay_positions: number } | null
  matches: { generator?: Match; it0?: Match }
}
interface Lineage { root: string; name: string; launch: string | null; loop_settings: string | null; iterations: Iteration[] }
interface Curve { block: number; lineages: Lineage[] }

// One colour per iteration, cycling; distinct in both lightness and hue on the panel's grey ground.
const IT_COLORS = ['#3f7355', '#a8473c', '#3d5f8f', '#b07a24', '#6e4b8b', '#2f7f86', '#8a5a3b', '#5b6b2e', '#a34d79', '#4c4f54']
const itColor = (n: number) => IT_COLORS[(n - 1) % IT_COLORS.length]
const label = (l: Lineage) => l.root.includes('/ab2/') ? `ab2 ${l.name}` : l.name
const pct = (v: number) => `${(100 * v).toFixed(1)}%`
const k = (v: number) => v >= 1000 ? `${(v / 1000).toFixed(v >= 10000 ? 0 : 1)}k` : `${v}`

function changes(it: Iteration, prev: Iteration | undefined): string[] {
  const out: string[] = []
  if (!prev) out.push('starts from it13 (the generator for the first self-play)')
  else if (prev.promoted) out.push(`self-play on a NEW generator: it${prev.iteration} was promoted`)
  else if (prev.promoted === false) out.push(`same generator as it${prev.iteration} (it was not promoted)`)
  if (it.train && it.train.replay_positions) out.push(`trained on ${k(it.train.new_positions)} new + ${k(it.train.replay_positions)} replayed positions (replay share ${pct(it.train.replay_fraction)})`)
  else if (it.train) out.push(`trained on ${k(it.train.new_positions)} new positions, no replay yet`)
  if (prev && it.selfplay_games && prev.selfplay_games && it.flag_games / it.selfplay_games - prev.flag_games / prev.selfplay_games > 0.01)
    out.push(`more flag losses in self-play: ${pct(it.flag_games / it.selfplay_games)} vs ${pct(prev.flag_games / prev.selfplay_games)}`)
  return out
}

function Chart({ lin, block }: { lin: Lineage; block: number }) {
  // x: the it13 gate's blocks in play order, iteration after iteration; the generator gate's blocks beside them.
  const pts: { x: number; it: number; s: number; vs: 'it13' | 'gen'; g: number }[] = []
  const spans: { it: number; x0: number; x1: number; score: number; ci: [number, number] | null }[] = []
  let x = 0
  for (const it of lin.iterations) {
    const m13 = it.matches.it0, mg = it.matches.generator
    const n = Math.max(m13?.blocks.length ?? 0, mg?.blocks.length ?? 0)
    if (!n) continue
    m13?.blocks.forEach((b, i) => pts.push({ x: x + i, it: it.iteration, s: b.score, vs: 'it13', g: b.games }))
    mg?.blocks.forEach((b, i) => { if (!mg.same_as_generator) pts.push({ x: x + i + 0.18, it: it.iteration, s: b.score, vs: 'gen', g: b.games }) })
    const main = m13 ?? mg!
    const half = main.games ? 1.96 * Math.sqrt(main.score * (1 - main.score) / main.games) : 0
    spans.push({ it: it.iteration, x0: x - 0.3, x1: x + n - 0.7, score: main.score, ci: main.games ? [main.score - half, main.score + half] : null })
    x += n + 0.8
  }
  if (!pts.length) return <p className="dt-empty">No gate games yet: the first iteration is still in self-play or training.</p>
  const W = 940, H = 330, L = 56, R = 16, T = 16, B = 40
  const xs = (v: number) => L + (v + 0.5) / Math.max(x, 1) * (W - L - R)
  const yMin = 0.2, yMax = 0.8
  const ys = (v: number) => T + (1 - (Math.min(Math.max(v, yMin), yMax) - yMin) / (yMax - yMin)) * (H - T - B)
  return (
    <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="xMinYMin meet" style={{ height: '40vh' }} role="img"
      aria-label={`Gate win rate per ${block} games for ${label(lin)}, coloured by iteration`}>
      {[0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8].map(v => (
        <g key={v}>
          <line className={v === 0.5 ? 'dt-half' : 'dt-grid'} x1={L} x2={W - R} y1={ys(v)} y2={ys(v)} />
          <text className="dt-tick" x={L - 24} y={ys(v) + 4}>{v.toFixed(1)}</text>
        </g>
      ))}
      {spans.map(s => (
        <g key={s.it}>
          {s.ci && <rect x={xs(s.x0)} width={xs(s.x1) - xs(s.x0)} y={ys(s.ci[1])} height={ys(s.ci[0]) - ys(s.ci[1])} fill={itColor(s.it)} opacity={0.12} />}
          <line x1={xs(s.x0)} x2={xs(s.x1)} y1={ys(s.score)} y2={ys(s.score)} stroke={itColor(s.it)} strokeWidth={2.5} />
          <text className="dt-xlabel" x={(xs(s.x0) + xs(s.x1)) / 2} y={H - B + 22} fill={itColor(s.it)}>it{s.it}</text>
        </g>
      ))}
      {pts.map((p, i) => p.vs === 'it13'
        ? <circle key={i} cx={xs(p.x)} cy={ys(p.s)} r={5.5} fill={itColor(p.it)}><title>it{p.it} vs it13, block of {p.g}: {pct(p.s)}</title></circle>
        : <circle key={i} cx={xs(p.x)} cy={ys(p.s)} r={4.5} fill="var(--raised)" stroke={itColor(p.it)} strokeWidth={1.8}><title>it{p.it} vs its generator, block of {p.g}: {pct(p.s)}</title></circle>)}
    </svg>
  )
}

export default function TrainingCurve() {
  const [curve, setCurve] = useState<Curve | null>(null)
  const [err, setErr] = useState('')
  const [pick, setPick] = useState(0)
  const [loading, setLoading] = useState(false)
  const load = () => {
    setLoading(true)
    fetch('/api/training').then(r => r.json()).then((j: { ok: boolean; text: string }) => {
      if (!j.ok) { setErr(j.text.slice(0, 300)); return }
      setCurve(JSON.parse(j.text) as Curve); setErr('')
    }).catch(e => setErr(String(e))).finally(() => setLoading(false))
  }
  useEffect(load, [])
  const lins = curve?.lineages ?? []
  const lin = lins[Math.min(pick, Math.max(lins.length - 1, 0))]
  return (
    <div className="dt-train">
      <div className="nn-try">
        {lins.map((l, i) => (
          <button key={l.root} className="dp-btn" aria-pressed={lin === l} onClick={() => setPick(i)}>{label(l)}</button>
        ))}
        <button className="dp-btn" onClick={load} disabled={loading}>{loading ? 'Reading PACE…' : 'Refresh'}</button>
      </div>
      {err && <pre className="dt-err">{err}</pre>}
      {!curve && !err && <p className="dt-empty">Reading the runs on PACE…</p>}
      {lin && (
        <>
          <figure className="dt-bars">
            <figcaption>{label(lin)} · gate win rate per {curve!.block} games · filled = vs it13, hollow = vs its generator · band = the iteration's 95% range</figcaption>
            <Chart lin={lin} block={curve!.block} />
          </figure>
          <div className="dt-its">
            {lin.iterations.map((it, i) => {
              const m13 = it.matches.it0, mg = it.matches.generator
              return (
                <div key={it.iteration} className="dt-it" style={{ borderLeftColor: itColor(it.iteration) }}>
                  <b style={{ color: itColor(it.iteration) }}>it{it.iteration}</b>{it.done ? '' : ' · running'}
                  <div>{k(it.selfplay_games)} self-play games ({it.segments} segments), {k(it.positions)} positions, {it.selfplay_games ? pct(it.flag_games / it.selfplay_games) : '–'} flag games</div>
                  {it.train && <div>training {it.train.steps} steps</div>}
                  <div>
                    {m13 && <>vs it13 <b>{pct(m13.score)}</b> over {m13.games}{m13.elo != null ? ` (${m13.elo >= 0 ? '+' : ''}${m13.elo.toFixed(0)} Elo)` : ''}{m13.finished ? '' : ', unfinished'}</>}
                    {m13 && mg && !mg.same_as_generator && ' · '}
                    {mg && !mg.same_as_generator && <>vs generator {pct(mg.score)} over {mg.games}</>}
                    {it.promoted != null && <> · {it.promoted ? 'promoted' : 'not promoted'}</>}
                  </div>
                  <ul>{changes(it, lin.iterations[i - 1]).map(c => <li key={c}>{c}</li>)}</ul>
                </div>
              )
            })}
          </div>
          {(lin.launch || lin.loop_settings) && <p className="dt-settings"><code>{lin.launch ?? lin.loop_settings}</code></p>}
        </>
      )}
    </div>
  )
}
