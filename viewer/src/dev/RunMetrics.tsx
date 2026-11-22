// Metrics step of durative.html: one record per finished iteration of a run (cluster/run_metrics.py, with the probe
// from cluster/probe_net.py merged in as "network"), so we can see what moves Elo. Elo vs fixed anchors is the
// target; every other KPI is a diagnostic or an alarm. Data: /api/metrics (vite.devApi.ts). A KPI the collector
// could not measure arrives as null with a reason under "unavailable"; it is drawn greyed with that reason, never
// dropped.

import { type ReactNode, useEffect, useMemo, useState } from 'react'

interface Block { games: number; score: number }
interface Match {
  games: number; score: number | null; elo: number | null; ci: [number, number] | null; wins: number; losses: number; draws: number
  movetime: number | null; clock: unknown; blocks: Block[]
}
interface Alert { level: 'warn' | 'stop'; code: string; message: string; iteration?: number }
interface CurveRow { [k: string]: number | undefined }
interface Probe {
  heldout?: Record<string, number | string | null>
  calibration?: { buckets: { lo: number; hi: number; n: number; predicted: number | null; actual: number | null }[]; ece: number | null }
  input_groups?: Record<string, { policy_kl: number | null; value_delta: number | null; planes: number[] }>
  agreement?: Record<string, number | null>
  layers?: { name: string; delta_rel: number | null; dead_frac: number | null }[]
  worst?: { fen: string; predicted: number; actual: number; source?: string; category?: string }[]
  swings?: { fen: string; prev: number; now: number; actual: number; source?: string; category?: string }[]
  probe_set?: { positions: number }
}
interface Iter {
  run: string; iteration: number; done: boolean; generated_at: string
  outcome: { gate_generator: Match | null; gate_it13: Match | null; promoted: boolean | null; anchors: Record<string, Match>; elo_vs_it13: number | null }
  cost: Record<string, number | null>
  selfplay: Record<string, unknown>
  training: { curve: CurveRow[]; final: CurveRow | null } & Record<string, unknown>
  search: Record<string, unknown>
  network: Probe | null
  changes: Record<string, [unknown, unknown]>
  alerts: Alert[]
  unavailable: Record<string, string>
}
interface Run { run: string; root: string; settings: string | null; iterations: Iter[]; alerts: Alert[] }
interface Summary { schema: number; generated_at: string; runs: Run[]; sample?: string }
interface Reply { at: number; ok: boolean; text: string; source: string }

// Same per-iteration colours as TrainingCurve.tsx, so an iteration reads the same on both steps.
const IT_COLORS = ['#3f7355', '#a8473c', '#3d5f8f', '#b07a24', '#6e4b8b', '#2f7f86', '#8a5a3b', '#5b6b2e', '#a34d79', '#4c4f54']
const itColor = (n: number) => IT_COLORS[(n - 1) % IT_COLORS.length]

const num = (v: unknown): number | null => typeof v === 'number' && Number.isFinite(v) ? v : null
function dig(o: unknown, key: string): unknown {
  return key.split('.').reduce<unknown>((x, k) => x && typeof x === 'object' ? (x as Record<string, unknown>)[k] : undefined, o)
}
const signed = (v: number, d = 0) => `${v >= 0 ? '+' : ''}${v.toFixed(d)}`
const pct = (v: number) => `${(100 * v).toFixed(1)}%`
const fix = (d: number) => (v: number) => v.toFixed(d)
const kfmt = (v: number) => Math.abs(v) >= 1000 ? `${(v / 1000).toFixed(Math.abs(v) >= 10000 ? 0 : 1)}k` : v.toFixed(0)

// Why a key is null: the collector's reason for it or for its nearest parent ("network" covers "network.heldout.loss").
function reason(it: Iter, key: string): string | null {
  const parts = key.split('.')
  for (let i = parts.length; i > 0; i--) {
    const r = it.unavailable?.[parts.slice(0, i).join('.')]
    if (r) return r
  }
  return null
}
const whyNull = (it: Iter, key: string) => reason(it, key) ?? (it.done ? 'no value and no reason from the collector' : 'iteration still running')

interface Kpi { key: string; label: string; caption: string; good: 'up' | 'down' | 'none'; fmt: (v: number) => string; get?: (it: Iter) => number | null; zero?: boolean }
const GROUPS: { name: string; kpis: Kpi[] }[] = [
  {
    name: 'Outcome', kpis: [
      { key: 'outcome.elo_vs_it13', label: 'Elo vs it13 (gate)', good: 'up', fmt: v => signed(v), zero: true,
        caption: 'Gate Elo against the starting it13 net at 100 ms. Higher is better; this is the number training should push up.' },
      { key: 'outcome.gate_generator.score', label: 'Score vs generator', good: 'up', fmt: fix(3),
        caption: 'Score against the net that played this self-play. Above 0.5 promotes it to generator.' },
      { key: 'cost.elo_per_gpu_hour', label: 'Elo per GPU-hour', good: 'up', fmt: fix(2), zero: true,
        caption: 'Elo vs it13 gained this iteration per GPU-hour spent. Higher is better; near 0 for long = the run is flat.' },
      { key: 'cost.gpu_hours', label: 'GPU-hours', good: 'down', fmt: fix(1),
        caption: 'GPU time this iteration (self-play, training, gates). Lower is cheaper.' },
      { key: 'cost.wall_hours', label: 'Wall hours', good: 'down', fmt: fix(1),
        caption: 'Clock time from the previous iteration finishing to this one finishing.' },
    ],
  },
  {
    name: 'Self-play', kpis: [
      { key: 'selfplay.flag_rate', label: 'Flag rate', good: 'down', fmt: pct,
        caption: 'Self-play games lost on time. Lower is better; a jump means the net is burning clock.' },
      { key: 'selfplay.repeat_rate', label: 'Repeat rate', good: 'down', fmt: pct,
        caption: 'Positions that repeat. Lower is better; high means shuffling instead of playing.' },
      { key: 'selfplay.game_length.mean', label: 'Game length (plies)', good: 'none', fmt: fix(0),
        caption: 'Plies over both boards per game. No good direction; a sudden shift means the style changed.' },
      { key: 'selfplay.policy_target_entropy.mean', label: 'Policy target entropy', good: 'up', fmt: fix(3),
        caption: 'How spread the search targets are. A fast fall means targets collapsed onto one move (alarm).' },
      { key: 'selfplay.value_target.win', label: 'Win share of targets', good: 'none', fmt: pct,
        caption: 'Positions labelled a win for the side to move. Should stay near 50%; drift = a seat bias.' },
      { key: 'selfplay.value_target.draw', label: 'Draw share of targets', good: 'none', fmt: pct,
        caption: 'Positions labelled a draw. Bughouse rarely draws; a rise means repetition or stalling.' },
      { key: 'selfplay.sit_rate', label: 'Sit rate', good: 'none', fmt: pct,
        caption: 'Searches that chose to sit and wait for the partner. Tells whether the hybrid uses waiting.' },
      { key: 'selfplay.hybrid_wait_rate', label: 'Hybrid wait trigger', good: 'none', fmt: pct,
        caption: 'Searches where the hybrid wait rule was considered (an opponent move due soon).' },
      { key: 'selfplay.pass_rate', label: 'Pass targets', good: 'none', fmt: pct,
        caption: 'On-turn targets whose best action is to pass. Should track the sit rate.' },
      { key: 'selfplay.drop_rate', label: 'Drop rate', good: 'none', fmt: pct,
        caption: 'Moves that drop a piece from the pocket. A bughouse style signal, no good direction.' },
      { key: 'selfplay.pocket_mean', label: 'Pocket pieces', good: 'none', fmt: fix(1),
        caption: 'Pieces held across all four pockets on average. Rising = pieces are being hoarded.' },
    ],
  },
  {
    name: 'Training', kpis: [
      { key: 'training.final.search_loss', label: 'Val loss (end)', good: 'down', fmt: fix(3),
        caption: 'Validation loss (policy + value) when training stopped. Lower is better.' },
      { key: 'training.final.search_wdl', label: 'Val value loss', good: 'down', fmt: fix(3),
        caption: 'Validation win/draw/loss cross-entropy. Lower is better; rising = the value head is getting worse.' },
      { key: 'training.final.search_top1_a', label: 'Val top-1 (board A)', good: 'up', fmt: pct,
        caption: 'How often the net\'s first choice matches the search target. Higher is better.' },
      { key: 'training.final.train_loss', label: 'Train loss (end)', good: 'down', fmt: fix(3),
        caption: 'Loss on the training batches. Far below the validation loss means overfitting.' },
      { key: 'training.replay_fraction', label: 'Replay share', good: 'none', fmt: pct,
        caption: 'Share of training positions replayed from earlier iterations (a setting, not a result).' },
      { key: 'training.steps', label: 'Training steps', good: 'none', fmt: kfmt,
        caption: 'Optimizer steps this iteration (a setting).' },
      { key: 'training.grad_norm', label: 'Gradient norm', good: 'down', fmt: fix(2),
        caption: 'Size of the training gradients. Spikes mean unstable training.' },
    ],
  },
  {
    name: 'Search', kpis: [
      { key: 'search.nps', label: 'Nodes per second', good: 'up', fmt: kfmt, get: it => num((dig(it, 'search.nps') as unknown[] | null)?.[0]),
        caption: 'The trained net\'s search speed in the gate. Higher is better; a drop costs Elo at a fixed clock.' },
      { key: 'search.time_use', label: 'ms per search', good: 'none', fmt: fix(1), get: it => num(dig(it, 'search.time_use.contender_ms_per_search')),
        caption: 'Average time per gate search against a 100 ms budget. Far under = time left on the table.' },
      { key: 'search.sit_gate_rate', label: 'Gate sit rate', good: 'none', fmt: pct,
        caption: 'Searches in the gate that sat. Compare with self-play\'s sit rate.' },
      { key: 'search.depth', label: 'Search depth', good: 'up', fmt: fix(1),
        caption: 'Average depth reached. Deeper at equal time is better.' },
    ],
  },
  {
    name: 'Network', kpis: [
      { key: 'network.heldout.loss', label: 'Held-out loss', good: 'down', fmt: fix(3),
        caption: 'Loss on the frozen probe set, the same positions every iteration. Lower is better.' },
      { key: 'network.heldout.value_loss', label: 'Held-out value loss', good: 'down', fmt: fix(3),
        caption: 'Win/draw/loss error on the probe set. Lower is better.' },
      { key: 'network.heldout.top1', label: 'Held-out top-1', good: 'up', fmt: pct,
        caption: 'Probe positions where the net\'s first choice is the search\'s. Higher is better.' },
      { key: 'network.heldout.value_mae', label: 'Held-out value error', good: 'down', fmt: fix(3),
        caption: 'Mean gap between predicted value and the game result. Lower is better.' },
      { key: 'network.calibration.ece', label: 'Calibration error', good: 'down', fmt: fix(3),
        caption: 'Gap between predicted win chance and how often it actually wins. Lower is better.' },
      { key: 'network.agreement.prev_top1', label: 'Same move as previous net', good: 'none', fmt: pct,
        caption: 'Probe positions where the top move did not change. Low = the net changed a lot this iteration.' },
      { key: 'network.agreement.prev_policy_kl', label: 'Policy shift (KL)', good: 'none', fmt: fix(3),
        caption: 'How far the move distribution moved from the previous net. Zero = training changed nothing.' },
      { key: 'network.layers', label: 'Dead channels (worst layer)', good: 'down', fmt: pct,
        get: it => { const v = (it.network?.layers ?? []).map(l => num(l.dead_frac)).filter((x): x is number => x != null); return v.length ? Math.max(...v) : null },
        caption: 'Largest share of never-firing channels in any layer. Rising = capacity being lost.' },
      { key: 'network.layers.delta_rel', label: 'Weight change', good: 'none', fmt: v => v.toExponential(1),
        get: it => { const v = (it.network?.layers ?? []).map(l => num(l.delta_rel)).filter((x): x is number => x != null); return v.length ? v.reduce((a, b) => a + b) / v.length : null },
        caption: 'Average relative weight change from the previous net over layers. Tells how hard training pushed.' },
    ],
  },
]
const kpiValue = (k: Kpi, it: Iter) => k.get ? k.get(it) : num(dig(it, k.key))

function Mini({ kpi, its }: { kpi: Kpi; its: Iter[] }) {
  const vals = its.map(it => ({ it, v: kpiValue(kpi, it) }))
  const have = vals.filter(p => p.v != null) as { it: Iter; v: number }[]
  const last = have[have.length - 1]
  const missing = vals.filter(p => p.v == null)
  if (!have.length) {
    const why = missing.length ? whyNull(missing[missing.length - 1].it, kpi.key) : 'no iterations'
    return (
      <div className="rm-card rm-na">
        <div className="rm-card-head"><b>{kpi.label}</b><span>n/a</span></div>
        <p className="rm-cap">{kpi.caption}</p>
        <p className="rm-why">Not measured: {why}</p>
      </div>
    )
  }
  const W = 260, H = 96, L = 6, R = 6, T = 8, B = 18
  let lo = Math.min(...have.map(p => p.v)), hi = Math.max(...have.map(p => p.v))
  if (kpi.zero) { lo = Math.min(lo, 0); hi = Math.max(hi, 0) }
  if (hi - lo < 1e-12) { const pad = Math.abs(hi) * .05 || 1; lo -= pad; hi += pad }
  const pad = (hi - lo) * .12; lo -= pad; hi += pad
  const n = its.length
  const x = (i: number) => L + (n === 1 ? .5 : i / (n - 1)) * (W - L - R)
  const y = (v: number) => T + (1 - (v - lo) / (hi - lo)) * (H - T - B)
  const idx = new Map(its.map((it, i) => [it.iteration, i]))
  const segs: string[] = []
  let cur = ''
  vals.forEach((p, i) => { if (p.v == null) { if (cur) segs.push(cur); cur = '' } else cur += `${cur ? 'L' : 'M'}${x(i)},${y(p.v)}` })
  if (cur) segs.push(cur)
  const arrow = kpi.good === 'up' ? '↑ better' : kpi.good === 'down' ? '↓ better' : ''
  return (
    <div className="rm-card">
      <div className="rm-card-head"><b>{kpi.label}</b><span>{kpi.fmt(last.v)} <small>it{last.it.iteration}</small></span></div>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`${kpi.label} per iteration: ${have.map(p => `it${p.it.iteration} ${kpi.fmt(p.v)}`).join(', ')}`}>
        {kpi.zero && lo < 0 && hi > 0 && <line className="dt-half" x1={L} x2={W - R} y1={y(0)} y2={y(0)} />}
        {segs.map((d, i) => <path key={i} className="rm-link" d={d} />)}
        {have.map(p => {
          const i = idx.get(p.it.iteration)!
          return (
            <g key={p.it.iteration}>
              <circle cx={x(i)} cy={y(p.v)} r={4.5} fill={itColor(p.it.iteration)} stroke="var(--raised)" strokeWidth={2}
                opacity={p.it.done ? 1 : .55} />
              <circle className="rm-hit" cx={x(i)} cy={y(p.v)} r={12}>
                <title>{`it${p.it.iteration}${p.it.done ? '' : ' (running)'}: ${kpi.fmt(p.v)}`}</title>
              </circle>
            </g>
          )
        })}
        {its.map((it, i) => (n <= 12 || i % Math.ceil(n / 12) === 0 || i === n - 1) &&
          <text key={it.iteration} className="rm-xt" x={x(i)} y={H - 4}>{it.iteration}</text>)}
      </svg>
      <p className="rm-cap">{kpi.caption}{arrow && <i> {arrow}</i>}</p>
      {missing.length > 0 && <p className="rm-why">n/a for {missing.map(p => `it${p.it.iteration}`).join(', ')}: {whyNull(missing[missing.length - 1].it, kpi.key)}</p>}
    </div>
  )
}

const tc = (m: Match) => m.movetime != null ? `${m.movetime} ms/move` : typeof m.clock === 'number' ? `${Math.round(m.clock / 1000)} s clock` : 'clock'
function EloChart({ its }: { its: Iter[] }) {
  // Gate Elo vs it13 with its 95% range per iteration (filled), fixed anchors beside it (hollow, one shape each).
  const anchorNames = [...new Set(its.flatMap(it => Object.keys(it.outcome.anchors ?? {})))].filter(a => a !== 'it0')
  const pts: { it: Iter; dx: number; m: Match; name: string }[] = []
  for (const it of its) {
    if (it.outcome.gate_it13?.elo != null) pts.push({ it, dx: 0, m: it.outcome.gate_it13, name: 'gate vs it13' })
    anchorNames.forEach((a, j) => { const m = it.outcome.anchors?.[a]; if (m?.elo != null) pts.push({ it, dx: .16 * (j + 1), m, name: a }) })
  }
  if (!pts.length) return <p className="dt-empty">No finished gate yet, so no Elo.</p>
  const W = 900, H = 260, L = 56, R = 16, T = 14, B = 34
  const ys = pts.flatMap(p => [p.m.elo!, ...(p.m.ci ?? [])]).filter(v => Math.abs(v) < 300)
  let lo = Math.min(0, ...ys), hi = Math.max(0, ...ys)
  const pad = Math.max(10, (hi - lo) * .1); lo -= pad; hi += pad
  const n = its.length
  const x = (i: number) => L + (i + .5) / n * (W - L - R)
  const y = (v: number) => T + (1 - (Math.min(Math.max(v, lo), hi) - lo) / (hi - lo)) * (H - T - B)
  const step = (hi - lo) / 5, mag = 10 ** Math.floor(Math.log10(step)), tick = [1, 2, 5, 10].map(m => m * mag).find(m => m >= step)!
  const ticks: number[] = []
  for (let v = Math.ceil(lo / tick) * tick; v <= hi; v += tick) ticks.push(v)
  const idx = new Map(its.map((it, i) => [it.iteration, i]))
  const shape = (name: string, cx: number, cy: number, color: string) => {
    const j = anchorNames.indexOf(name)
    if (j < 0) return <circle cx={cx} cy={cy} r={6} fill={color} stroke="var(--raised)" strokeWidth={2} />
    if (j % 2 === 0) return <rect x={cx - 5.5} y={cy - 5.5} width={11} height={11} fill="var(--raised)" stroke={color} strokeWidth={2} />
    return <path d={`M${cx} ${cy - 7} L${cx + 6.5} ${cy + 5} L${cx - 6.5} ${cy + 5} Z`} fill="var(--raised)" stroke={color} strokeWidth={2} />
  }
  return (
    <>
      <svg viewBox={`0 0 ${W} ${H}`} className="rm-wide" role="img" aria-label="Elo versus it13 per iteration with 95% ranges, and fixed anchors at 120 s">
        {ticks.map(v => (
          <g key={v}>
            <line className={v === 0 ? 'dt-half' : 'dt-grid'} x1={L} x2={W - R} y1={y(v)} y2={y(v)} />
            <text className="dt-tick" x={L - 26} y={y(v) + 4}>{signed(v)}</text>
          </g>
        ))}
        {its.map((it, i) => <text key={it.iteration} className="dt-xlabel" x={x(i)} y={H - 10} fill={itColor(it.iteration)}>it{it.iteration}</text>)}
        {pts.map((p, k) => {
          const cx = x(idx.get(p.it.iteration)!) + p.dx * (W - L - R) / n, c = itColor(p.it.iteration)
          return (
            <g key={k}>
              {p.m.ci && <line x1={cx} x2={cx} y1={y(p.m.ci[0])} y2={y(p.m.ci[1])} stroke={c} strokeWidth={2} strokeLinecap="round" opacity={.6} />}
              {shape(p.name, cx, y(p.m.elo!), c)}
              <circle className="rm-hit" cx={cx} cy={y(p.m.elo!)} r={13}>
                <title>{`it${p.it.iteration} ${p.name} (${tc(p.m)}): ${signed(p.m.elo!)} Elo` +
                  `${p.m.ci ? ` [${signed(p.m.ci[0])}, ${signed(p.m.ci[1])}]` : ''}, score ${p.m.score?.toFixed(3)} over ${p.m.games} games`}</title>
              </circle>
            </g>
          )
        })}
      </svg>
      <div className="nn-legend">
        <span><svg className="rm-key" viewBox="0 0 14 14"><circle cx={7} cy={7} r={5.5} fill="var(--text)" /></svg> gate vs it13, 100 ms</span>
        {anchorNames.map(a => {
          const m = its.map(it => it.outcome.anchors?.[a]).find(Boolean)
          return <span key={a}><svg className="rm-key" viewBox="0 0 14 14">{shape(a, 7, 7.5, 'var(--text)')}</svg> {a}{m ? `, ${tc(m)}` : ''}</span>
        })}
        <span>line = 95% range · colour = iteration</span>
      </div>
    </>
  )
}

const CURVE_KEYS: [string, string][] = [['search_loss', 'val loss'], ['train_loss', 'train loss'], ['search_wdl', 'val value loss'],
  ['search_top1_a', 'val top-1 A'], ['search_value_mae', 'val value error']]
function LossCurves({ its }: { its: Iter[] }) {
  const [key, setKey] = useState('search_loss')
  const lines = its.map(it => ({ it, pts: (it.training?.curve ?? []).map((r, i) => ({ x: num(r.epoch) ?? i + 1, y: num(r[key]) }))
    .filter((p): p is { x: number; y: number } => p.y != null) })).filter(l => l.pts.length)
  const keys = CURVE_KEYS.filter(([k]) => its.some(it => (it.training?.curve ?? []).some(r => num(r[k]) != null)))
  const all = lines.flatMap(l => l.pts)
  const W = 900, H = 240, L = 56, R = 46, T = 12, B = 32
  const xMax = Math.max(...all.map(p => p.x), 1)
  let lo = Math.min(...all.map(p => p.y)), hi = Math.max(...all.map(p => p.y))
  const pad = (hi - lo) * .08 || .01; lo -= pad; hi += pad
  const x = (v: number) => L + v / xMax * (W - L - R)
  const y = (v: number) => T + (1 - (v - lo) / (hi - lo)) * (H - T - B)
  const f = key.includes('top1') ? pct : fix(3)
  return (
    <>
      <div className="nn-try">
        {keys.map(([k, l]) => <button key={k} className="dp-btn" aria-pressed={k === key} onClick={() => setKey(k)}>{l}</button>)}
      </div>
      {!lines.length ? <p className="dt-empty">No training curve yet (C.log has no eval lines).</p> : (
        <svg viewBox={`0 0 ${W} ${H}`} className="rm-wide" role="img" aria-label={`Training ${key} by epoch, one line per iteration`}>
          {[0, .25, .5, .75, 1].map(t => lo + t * (hi - lo)).map(v => (
            <g key={v}>
              <line className="dt-grid" x1={L} x2={W - R} y1={y(v)} y2={y(v)} />
              <text className="dt-tick" x={L - 26} y={y(v) + 4}>{f(v)}</text>
            </g>
          ))}
          {[0, xMax / 2, xMax].map(v => <text key={v} className="dt-tick" x={x(v)} y={H - 10}>{v.toFixed(1)} ep</text>)}
          {lines.map(({ it, pts }) => {
            const c = itColor(it.iteration), e = pts[pts.length - 1]
            return (
              <g key={it.iteration}>
                <path d={pts.map((p, i) => `${i ? 'L' : 'M'}${x(p.x)},${y(p.y)}`).join('')} fill="none" stroke={c} strokeWidth={2} />
                {pts.map((p, i) => (
                  <circle key={i} cx={x(p.x)} cy={y(p.y)} r={3} fill={c}>
                    <title>{`it${it.iteration}, epoch ${p.x}: ${f(p.y)}`}</title>
                  </circle>
                ))}
                <text className="rm-end" x={x(e.x) + 7} y={y(e.y) + 4}>it{it.iteration}</text>
              </g>
            )
          })}
        </svg>
      )}
    </>
  )
}

function Calibration({ it }: { it: Iter }) {
  const cal = it.network?.calibration
  const b = (cal?.buckets ?? []).filter(k => k.n > 0 && k.predicted != null && k.actual != null)
  if (!b.length) return <p className="rm-why">No calibration for it{it.iteration}: {whyNull(it, 'network.calibration')}</p>
  const S = 300, P = 40
  const s = (v: number) => P + v * (S - P - 10)
  const maxN = Math.max(...b.map(k => k.n)), c = itColor(it.iteration)
  return (
    <figure className="rm-cal">
      <svg viewBox={`0 0 ${S} ${S}`} role="img" aria-label={`Calibration of it${it.iteration}: predicted win chance against actual win rate`}>
        {[0, .25, .5, .75, 1].map(v => (
          <g key={v}>
            <line className="dt-grid" x1={s(0)} x2={s(1)} y1={S - s(v)} y2={S - s(v)} />
            <text className="dt-tick" x={P - 16} y={S - s(v) + 4}>{v}</text>
            <text className="dt-tick" x={s(v)} y={S - P + 22}>{v}</text>
          </g>
        ))}
        <line className="dt-half" x1={s(0)} y1={S - s(0)} x2={s(1)} y2={S - s(1)} />
        {b.map(k => (
          <circle key={k.lo} cx={s(k.predicted!)} cy={S - s(k.actual!)} r={3 + 7 * Math.sqrt(k.n / maxN)} fill={c} fillOpacity={.75} stroke="var(--raised)" strokeWidth={1.5}>
            <title>{`predicted ${k.lo}–${k.hi}: mean ${pct(k.predicted!)}, actual ${pct(k.actual!)} over ${k.n} positions`}</title>
          </circle>
        ))}
      </svg>
      <figcaption>x = predicted win chance, y = how often it won · dashed = perfect · dot size = positions · ECE {cal?.ece != null ? cal.ece.toFixed(3) : 'n/a'}</figcaption>
    </figure>
  )
}

function Heatmap({ its }: { its: Iter[] }) {
  const [field, setField] = useState<'policy_kl' | 'value_delta'>('policy_kl')
  const cols = its.filter(it => it.network?.input_groups)
  const rows = [...new Set(cols.flatMap(it => Object.keys(it.network!.input_groups!)))]
  if (!cols.length) return <p className="rm-why">No input-group ablation yet: {its.length ? whyNull(its[its.length - 1], 'network.input_groups') : 'no iterations'}</p>
  const v = (it: Iter, g: string) => num(it.network?.input_groups?.[g]?.[field])
  const max = Math.max(...cols.flatMap(it => rows.map(g => v(it, g) ?? 0)), 1e-9)
  // sqrt so the small groups (pockets, clocks) still show shade next to the whole-board ones.
  const shade = (x: number) => Math.round(100 * Math.sqrt(x / max))
  const f = field === 'policy_kl' ? fix(3) : fix(3)
  return (
    <>
      <div className="nn-try">
        <button className="dp-btn" aria-pressed={field === 'policy_kl'} onClick={() => setField('policy_kl')}>move choice (policy KL)</button>
        <button className="dp-btn" aria-pressed={field === 'value_delta'} onClick={() => setField('value_delta')}>evaluation (value change)</button>
      </div>
      <div className="rm-scroll">
        <table className="rm-heat">
          <thead>
            <tr><th>input group</th>{cols.map(it => <th key={it.iteration} style={{ color: itColor(it.iteration) }}>it{it.iteration}</th>)}<th>change</th></tr>
          </thead>
          <tbody>
            {rows.map(g => {
              const first = v(cols[0], g), last = v(cols[cols.length - 1], g)
              const rel = first != null && last != null && first > 0 ? last / first - 1 : null
              return (
                <tr key={g}>
                  <th><code>{g}</code></th>
                  {cols.map(it => {
                    const x = v(it, g)
                    if (x == null) return <td key={it.iteration} className="rm-na-cell" title={whyNull(it, `network.input_groups.${g}`)}>n/a</td>
                    const p = shade(x)
                    return (
                      <td key={it.iteration} title={`${g}, it${it.iteration}: ${f(x)}`}
                        style={{ background: `color-mix(in oklab, var(--text) ${p}%, var(--raised))`, color: p > 52 ? 'var(--bg)' : 'var(--text)' }}>{f(x)}</td>
                    )
                  })}
                  <td className="rm-delta">{rel == null || cols.length < 2 ? '' : `${rel >= 0 ? '+' : ''}${(100 * rel).toFixed(0)}%`}</td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
      <p className="rm-cap">Each cell blanks one group of input planes on the frozen probe set and measures how much the net's
        {field === 'policy_kl' ? ' move distribution moves (KL divergence)' : ' evaluation moves (mean |Δ value|)'}. Darker = the net leans on
        that input more. A row darkening over iterations means training taught the net to use it. Change = last vs first column.</p>
    </>
  )
}

function Positions({ it }: { it: Iter }) {
  const w = it.network?.worst ?? [], s = it.network?.swings ?? []
  if (!it.network) return <p className="rm-why">No probe for it{it.iteration}: {whyNull(it, 'network')}</p>
  const v = (x: number) => signed(x, 2)
  return (
    <div className="rm-tables">
      <div>
        <h4>Worst value predictions, it{it.iteration}</h4>
        <div className="rm-scroll">
          <table className="dp-table rm-pos">
            <thead><tr><th>position</th><th>predicted</th><th>result</th></tr></thead>
            <tbody>{w.slice(0, 10).map((r, i) => <tr key={i}><td><code>{r.fen}</code>{r.category && <small> {r.category}</small>}</td><td>{v(r.predicted)}</td><td>{v(r.actual)}</td></tr>)}</tbody>
          </table>
        </div>
      </div>
      <div>
        <h4>Biggest swings from the previous net</h4>
        <div className="rm-scroll">
          <table className="dp-table rm-pos">
            <thead><tr><th>position</th><th>before</th><th>now</th><th>result</th></tr></thead>
            <tbody>{s.slice(0, 10).map((r, i) => <tr key={i}><td><code>{r.fen}</code>{r.category && <small> {r.category}</small>}</td><td>{v(r.prev)}</td><td>{v(r.now)}</td><td>{v(r.actual)}</td></tr>)}</tbody>
          </table>
        </div>
      </div>
      <p className="rm-cap">Value is the side to move's expected result, −1 loss to +1 win. Positions are written from our team's view
        (our pieces uppercase, pockets as [ours|theirs]), not as real FENs.</p>
    </div>
  )
}

function Tile({ label, value, sub, na }: { label: string; value: ReactNode; sub: ReactNode; na?: boolean }) {
  return <div className={na ? 'rm-tile rm-na' : 'rm-tile'}><span>{label}</span><b>{value}</b><small>{sub}</small></div>
}

const show = (v: unknown) => v == null ? '–' : typeof v === 'object' ? JSON.stringify(v) : String(v)

export default function RunMetrics() {
  const [reply, setReply] = useState<Reply | null>(null)
  const [data, setData] = useState<Summary | null>(null)
  const [err, setErr] = useState('')
  const [loading, setLoading] = useState(false)
  const [sample, setSample] = useState(() => new URLSearchParams(location.search).get('sample') === '1')
  const [pick, setPick] = useState(0)
  const [probeIt, setProbeIt] = useState<number | null>(null)
  const load = (useSample = sample) => {
    setLoading(true)
    fetch(`/api/metrics${useSample ? '?sample=1' : ''}`).then(r => r.json()).then((j: Reply) => {
      setReply(j)
      if (!j.ok) { setErr(j.text.slice(0, 600)); setData(null); return }
      setData(JSON.parse(j.text) as Summary); setErr('')
    }).catch(e => setErr(String(e))).finally(() => setLoading(false))
  }
  useEffect(() => load(), []) // eslint-disable-line react-hooks/exhaustive-deps

  const runs = data?.runs ?? []
  const run = runs[Math.min(pick, Math.max(runs.length - 1, 0))]
  const its = useMemo(() => [...(run?.iterations ?? [])].sort((a, b) => a.iteration - b.iteration), [run])
  const probed = its.filter(it => it.network)
  const pIt = probed.find(it => it.iteration === probeIt) ?? probed[probed.length - 1]
  const finished = its.filter(it => it.done)
  const latest = finished[finished.length - 1]
  const lastIt = its[its.length - 1]
  const alerts = [...(run?.alerts ?? its.flatMap(it => it.alerts.map(a => ({ ...a, iteration: it.iteration }))))]
    .sort((a, b) => (a.level === b.level ? 0 : a.level === 'stop' ? -1 : 1) || (b.iteration ?? 0) - (a.iteration ?? 0))
  const anchorNames = [...new Set(its.flatMap(it => Object.keys(it.outcome.anchors ?? {})))].filter(a => a !== 'it0')
  const lastAnchor = (a: string) => [...its].reverse().map(it => ({ it, m: it.outcome.anchors?.[a] })).find(p => p.m?.elo != null)
  const e = latest?.outcome.gate_it13
  const lastWith = (k: string) => [...its].reverse().find(it => num(dig(it, k)) != null)

  return (
    <div className="dt-train rm">
      <div className="nn-try">
        {runs.map((r, i) => <button key={r.root} className="dp-btn" aria-pressed={r === run} onClick={() => { setPick(i); setProbeIt(null) }}>{r.run}</button>)}
        <button className="dp-btn" onClick={() => load()} disabled={loading}>{loading ? 'Reading…' : 'Refresh'}</button>
        <button className="dp-btn" aria-pressed={sample} onClick={() => { setSample(!sample); setPick(0); setProbeIt(null); load(!sample) }}>Sample data</button>
      </div>
      {reply && <p className="rm-src">{reply.source === 'sample' ? <b className="rm-sample">Synthetic sample: every number is made up.</b> : `Source: ${reply.source}`}
        {data && ` · generated ${data.generated_at}`}</p>}
      {err && <pre className="dt-err">{err}</pre>}
      {!data && !err && <p className="dt-empty">Running the metrics collector…</p>}
      {data && !run && <p className="dt-empty">No runs under runs/full yet.</p>}
      {run && (
        <>
          <section className="rm-tiles">
            <Tile label="Elo vs it13 (gate, 100 ms)" na={latest?.outcome.elo_vs_it13 == null}
              value={latest?.outcome.elo_vs_it13 != null ? signed(latest.outcome.elo_vs_it13) : 'n/a'}
              sub={latest?.outcome.elo_vs_it13 != null ? <>it{latest.iteration}{e?.ci ? ` · [${signed(e.ci[0])}, ${signed(e.ci[1])}]` : ''} · {e?.games} games</>
                : latest ? whyNull(latest, 'outcome.elo_vs_it13') : 'no finished iteration'} />
            {anchorNames.length ? anchorNames.map(a => {
              const p = lastAnchor(a)
              return <Tile key={a} label={`Anchor ${a}`} na={!p} value={p ? signed(p.m!.elo!) : 'n/a'}
                sub={p ? <>it{p.it.iteration} · {tc(p.m!)}{p.m!.ci ? ` · [${signed(p.m!.ci[0])}, ${signed(p.m!.ci[1])}]` : ''} · {p.m!.games} games</> : 'not played'} />
            }) : <Tile label="Anchors at 120 s" na value="n/a" sub="no anchor match has finished (cluster/launch_anchor_match.sh, every 2nd iteration)" />}
            {(() => { const it = lastWith('cost.elo_per_gpu_hour'); return (
              <Tile label="Elo per GPU-hour" na={!it} value={it ? fix(2)(it.cost.elo_per_gpu_hour!) : 'n/a'}
                sub={it ? `it${it.iteration}` : lastIt ? whyNull(lastIt, 'cost.elo_per_gpu_hour') : ''} />) })()}
            {(() => { const it = lastWith('cost.gpu_hours_cum'); return (
              <Tile label="GPU-hours used" na={!it} value={it ? it.cost.gpu_hours_cum!.toFixed(0) : 'n/a'}
                sub={it ? `through it${it.iteration}${it.done ? '' : ' (running)'}` : lastIt ? whyNull(lastIt, 'cost.gpu_hours_cum') : ''} />) })()}
          </section>

          <h3 className="rm-h">Alerts</h3>
          {alerts.length ? (
            <ul className="rm-alerts">
              {alerts.map((a, i) => (
                <li key={i} className={a.level === 'stop' ? 'rm-stop' : 'rm-warn'}>
                  <span className="rm-level">{a.level === 'stop' ? '■ STOP' : '▲ warn'}</span>
                  {a.iteration != null && <b style={{ color: itColor(a.iteration) }}>it{a.iteration}</b>} <code>{a.code}</code> {a.message}
                </li>
              ))}
            </ul>
          ) : <p className="dt-empty">No alerts.</p>}

          <h3 className="rm-h">Elo per iteration</h3>
          <EloChart its={its} />

          {GROUPS.map(g => (
            <section key={g.name}>
              <h3 className="rm-h">{g.name}</h3>
              <div className="rm-grid">{g.kpis.map(k => <Mini key={k.key} kpi={k} its={its} />)}</div>
            </section>
          ))}

          <h3 className="rm-h">Training curves within each iteration</h3>
          <LossCurves its={its} />

          <h3 className="rm-h">How the network uses its inputs</h3>
          <Heatmap its={its} />

          <h3 className="rm-h">Probe of one iteration</h3>
          {probed.length ? (
            <>
              <div className="nn-try">
                {probed.map(it => <button key={it.iteration} className="dp-btn" aria-pressed={it === pIt} onClick={() => setProbeIt(it.iteration)}>it{it.iteration}</button>)}
              </div>
              <Calibration it={pIt} />
              <Positions it={pIt} />
            </>
          ) : <p className="rm-why">No probe yet: {lastIt ? whyNull(lastIt, 'network') : 'no iterations'}</p>}

          <h3 className="rm-h">What changed, iteration by iteration</h3>
          <div className="dt-its">
            {its.map(it => {
              const ch = Object.entries(it.changes ?? {}), un = Object.entries(it.unavailable ?? {})
              return (
                <div key={it.iteration} className="dt-it" style={{ borderLeftColor: itColor(it.iteration) }}>
                  <b style={{ color: itColor(it.iteration) }}>it{it.iteration}</b>{it.done ? '' : ' · running'}
                  {it.outcome.promoted != null && <> · {it.outcome.promoted ? 'promoted' : 'not promoted'}</>}
                  {ch.length ? <ul>{ch.map(([k, [a, b]]) => <li key={k}><code>{k}</code>: {show(a)} → {show(b)}</li>)}</ul>
                    : <ul><li>no setting changes</li></ul>}
                  {un.length > 0 && (
                    <details className="rm-un">
                      <summary>{un.length} not measured</summary>
                      <ul>{un.map(([k, r]) => <li key={k}><code>{k}</code>: {r}</li>)}</ul>
                    </details>
                  )}
                </div>
              )
            })}
          </div>
          {run.settings && <p className="dt-settings"><code>{run.settings}</code></p>}
        </>
      )}
    </div>
  )
}
