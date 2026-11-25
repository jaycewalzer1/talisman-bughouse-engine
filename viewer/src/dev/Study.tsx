// Game study page (study.html): every synced match row side by side, each row's games, and one game's key moments
// with its replay. Data: tools/study_games.py writes runs/study/study.json and games_<group>.json, served by
// /api/study (vite.devApi.ts) together with the synced game logs. Everything is from our engine's side ("us").

import { useCallback, useEffect, useMemo, useState } from 'react'
import { type GameEvent } from '../events/types'
import { parseJsonl } from './analyze'
import GameReplay from './GameReplay'

type Pt = [number, number]
interface Agg {
  n: number; score?: number; ci?: [number, number]; [k: string]: unknown
}
interface Group { id: string; file: string; label: string; k: [number | null, number | null][]; all: Agg; wins: Agg; losses: Agg }
interface Swing { ts0: number; ts: number; from: number; to: number; delta: number; seat: string; events: Ev[] }
interface Ev { ts: number; k: 'move' | 'pass' | 'sit'; us: boolean; board?: string; san?: string; cap?: string | null; piece?: string; to?: string; seat?: string; ms?: number }
interface Sit { seat: string; us: boolean; ts: number; ms: number; why: string | null }
interface Game {
  file: string; game: string; pair: number; opening: string; swap: boolean; us_team: string; k_us: number | null; k_them: number | null
  score: number; reason: string; board: string; plies: number; dur_s: number
  think_per_dec_us: number; think_per_dec_them: number; ahead_us: number | null; ahead_them: number | null
  sits_us: number; sits_them: number; sit_s_us: number; sit_s_them: number; stale_us: number; stale_them: number
  mat_diff: number | null; pocket_us: number | null; pocket_them: number | null; lead_mean_s: number; lead_end_s: number
  clock_left_us_s: number | null; clock_left_them_s: number | null
  turn: { ts: number; frac: number } | null; swings: Swing[]; sits: Sit[]
  mate: { ts: number; board: string; san: string; drop: boolean; by_us: boolean; loser: string; loser_us: boolean | null; reason: string; lead_ms: number } | null
  curve_us: Pt[]; curve_them: Pt[]; lead_curve: Pt[]; mat_curve: Pt[]
}

const US = '#2f62b0'
const THEM = '#b07a24'
const num = (v: unknown, d = 2) => typeof v === 'number' ? v.toFixed(d) : '–'
const pct = (v: unknown) => typeof v === 'number' ? `${Math.round(100 * v)}%` : '–'
const kLabel = (g: Group) => g.k.map(([a, b]) => `${a ?? '?'} v ${b ?? '?'}`).join(', ')

// Columns of the match table: [key, header, format, title]
const COLS: [string, string, (v: unknown) => string, string][] = [
  ['ahead_us', 'ahead us', pct, 'share of our turns on which we led on the clock (arena clock_teams)'],
  ['ahead_them', 'ahead them', pct, 'share of their turns on which they led'],
  ['sits_us', 'sits us', v => num(v, 1), 'sits (stalls) per game, ours'],
  ['sits_them', 'sits them', v => num(v, 1), 'sits per game, theirs'],
  ['sit_s_us', 'sit s us', v => num(v, 1), 'seconds sat per game, ours'],
  ['sit_s_them', 'sit s them', v => num(v, 1), 'seconds sat per game, theirs'],
  ['think_per_dec_us', 'ms/move us', v => num(v, 0), 'think ms per decision, ours'],
  ['think_per_dec_them', 'ms/move them', v => num(v, 0), 'think ms per decision, theirs'],
  ['stale_us', 'stale us', pct, 'share of our searches whose position changed (pocket) before the move was played'],
  ['stale_them', 'stale them', pct, 'same, theirs'],
  ['mat_diff', 'material', v => num(v, 1), 'mean material (board + pocket) ours minus theirs, sampled at every move'],
  ['pocket_us', 'pocket us', v => num(v, 1), 'mean pocket material, ours'],
  ['pocket_them', 'pocket them', v => num(v, 1), 'mean pocket material, theirs'],
  ['mate_on_our_seat', 'mated us', pct, 'share of games that ended with one of OUR seats mated'],
  ['mate_by_drop', 'by drop', pct, 'share of games ended by a drop'],
  ['turn_frac', 'turn at', pct, 'when the game turned for good (mean eval of both engines last crosses 0), share of game time'],
  ['dur_s', 'length s', v => num(v, 0), 'game length in simulated seconds'],
]

function Findings() {
  return (
    <section className="st-find">
      <h2>What the games say so far</h2>
      <ol>
        <li><b>Who leads on the clock decides the match, not who searches more.</b> Under the v3 rules only the team ahead
          on the clock may sit and wait for a piece. With our clock_k at 160 (moves 1.45x faster than it13 at 110) we led on
          98% of our turns, sat 8.4 times a game to their 1.3, and scored 0.994. At clock_k 1000 we searched 16x fewer
          nodes per move and still scored 0.907. At clock_k 55 (slower) the mirror: 0.004.</li>
        <li><b>The harness is fair between the two engines</b> (docs/fairness_audit_2026-11-25.md): same charging, rules, information
          and pairing for both kinds; T1 scores 0.50; wave2 vs itself 0.503. The k 55 row is upstream winning 0.996 the same
          way. So the effect is real under these rules, against an opponent whose pace is fixed. Open: wave2 guesses opponent
          sits with its own clock_k (a confound inside the sweep), and mate-prover wall time is not charged.</li>
        <li><b>The trained net (hybrid1 it2) lost for a different reason.</b> Same think time per move as it13 (1,024 vs 1,023 ms),
          same clock-lead share, yet 0.365: it holds MORE material (+3.1, 12.1 in pocket vs 8.6) and is mated on its own seat
          in 63% of games. That looks like learned piece hoarding, not a timing problem; the speed charge only cost it ~20% nodes.</li>
        <li><b>In even matches the same lever shows up in miniature:</b> winners led on ~66% of turns, losers ~52%.</li>
      </ol>
      <p className="st-note">Pending on PACE: chk1/sym160 (both at 160, expect ~0.5) and chk1/rev_opp160 (it13 faster, expect
        it13 to win). Re-sync and rerun <code>python3 tools/study_games.py</code> to add them.</p>
    </section>
  )
}

function MatchTable({ groups, part, setPart, sel, setSel }: {
  groups: Group[]; part: 'all' | 'wins' | 'losses'; setPart: (p: 'all' | 'wins' | 'losses') => void; sel: string | null; setSel: (id: string) => void
}) {
  return (
    <section>
      <div className="st-head">
        <h2>Matches</h2>
        <div className="st-seg" role="group" aria-label="Which games">
          {(['all', 'wins', 'losses'] as const).map(p => (
            <button key={p} className={`dp-btn ${part === p ? 'on' : ''}`} aria-pressed={part === p} onClick={() => setPart(p)}>{p}</button>
          ))}
        </div>
      </div>
      <p className="st-note">Each row is one match folder; columns are means per game over {part === 'all' ? 'all games' : `our ${part}`}. Click a row to list its games.</p>
      <div className="dp-scroll">
        <table className="dp-table st-table">
          <thead>
            <tr>
              <th>match</th><th>k us v them</th><th className="num">games</th><th className="num">score [95%]</th>
              {COLS.map(([k, h, , t]) => <th key={k} className="num" title={t}>{h}</th>)}
            </tr>
          </thead>
          <tbody>
            {groups.map(g => {
              const a = g[part]
              return (
                <tr key={g.id} className={sel === g.id ? 'on' : ''} onClick={() => setSel(g.id)} tabIndex={0}
                  onKeyDown={e => { if (e.key === 'Enter') setSel(g.id) }}>
                  <td>{g.id}</td><td className="dim">{kLabel(g)}</td><td className="num">{a.n}</td>
                  <td className="num">{num(g.all.score, 3)} <span className="dim">[{num(g.all.ci?.[0], 2)}, {num(g.all.ci?.[1], 2)}]</span></td>
                  {COLS.map(([k, , f]) => <td key={k} className="num">{f(a[k])}</td>)}
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
    </section>
  )
}

// Score against clock-lead share, one dot per match row.
function LeadScatter({ groups }: { groups: Group[] }) {
  const [hover, setHover] = useState<Group | null>(null)
  const W = 640, H = 300, L = 44, R = 16, T = 14, B = 36
  const x = (v: number) => L + v * (W - L - R)
  const y = (v: number) => T + (1 - v) * (H - T - B)
  const pts = groups.filter(g => typeof g.all.ahead_us === 'number' && typeof g.all.score === 'number')
  return (
    <section>
      <h2>Score follows the clock lead</h2>
      <p className="st-note">x: share of our turns on which we led on the clock; y: match score. One dot per match row.</p>
      <div className="st-chartwrap">
        <svg className="dp-chart" viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Score against clock-lead share">
          {[0, 0.25, 0.5, 0.75, 1].map(v => (
            <g key={v}>
              <line className={v === 0.5 ? 'zero' : 'grid'} x1={L} x2={W - R} y1={y(v)} y2={y(v)} />
              <text className="axis" x={L - 6} y={y(v) + 4} textAnchor="end">{v}</text>
              <line className="grid" x1={x(v)} x2={x(v)} y1={T} y2={H - B} />
              <text className="axis" x={x(v)} y={H - B + 16} textAnchor="middle">{Math.round(v * 100)}%</text>
            </g>
          ))}
          <text className="axis" x={(L + W - R) / 2} y={H - 4} textAnchor="middle">turns ahead on the clock (us)</text>
          {pts.map(g => (
            <g key={g.id} onMouseEnter={() => setHover(g)} onMouseLeave={() => setHover(null)}>
              <circle cx={x(g.all.ahead_us as number)} cy={y(g.all.score!)} r={14} fill="transparent" />
              <circle cx={x(g.all.ahead_us as number)} cy={y(g.all.score!)} r={5} fill={US} stroke="#fcfcfb" strokeWidth={2} />
              <text className="axis" x={x(g.all.ahead_us as number) + 8} y={y(g.all.score!) - 7}>{g.id.split('/').pop()}</text>
            </g>
          ))}
        </svg>
        {hover && (
          <div className="st-tip" role="status">
            <b>{hover.id}</b> ({kLabel(hover)})<br />score {num(hover.all.score, 3)} over {hover.all.n} games<br />
            ahead {pct(hover.all.ahead_us)} of our turns, sits {num(hover.all.sits_us, 1)} v {num(hover.all.sits_them, 1)}
          </div>
        )}
      </div>
    </section>
  )
}

type Sort = 'pair' | 'turn' | 'swing' | 'material' | 'sits'
function GameList({ games, part, open, sel }: { games: Game[]; part: 'all' | 'wins' | 'losses'; open: (g: Game) => void; sel: string | null }) {
  const [sort, setSort] = useState<Sort>('turn')
  const rows = useMemo(() => {
    const f = games.filter(g => part === 'all' || (part === 'wins' ? g.score === 1 : g.score === 0))
    const key: Record<Sort, (g: Game) => number> = {
      pair: g => g.pair,
      turn: g => g.turn?.frac ?? 1,
      swing: g => -Math.max(0, ...g.swings.map(s => Math.abs(s.delta))),
      material: g => -(Math.abs(g.mat_diff ?? 0)),
      sits: g => -(g.sits_us + g.sits_them),
    }
    return [...f].sort((a, b) => key[sort](a) - key[sort](b)).slice(0, 300)
  }, [games, part, sort])
  return (
    <section>
      <div className="st-head">
        <h2>Games <small>{rows.length} shown{rows.length === 300 ? ' (first 300)' : ''}</small></h2>
        <label className="st-sort">sort by{' '}
          <select value={sort} onChange={e => setSort(e.target.value as Sort)}>
            <option value="turn">earliest turning point</option>
            <option value="swing">biggest eval swing</option>
            <option value="material">biggest material gap</option>
            <option value="sits">most sits</option>
            <option value="pair">pair (opening) order</option>
          </select>
        </label>
      </div>
      <div className="dp-scroll st-list">
        <table className="dp-table">
          <thead><tr><th>game</th><th>result</th><th>end</th><th className="num">turn at</th><th className="num">ahead us</th>
            <th className="num">sits us/them</th><th className="num">material</th><th className="num">pocket us/them</th><th>opening</th></tr></thead>
          <tbody>
            {rows.map(g => (
              <tr key={g.file} className={sel === g.file ? 'on' : ''} onClick={() => open(g)} tabIndex={0}
                onKeyDown={e => { if (e.key === 'Enter') open(g) }}>
                <td>{g.file.split('/').slice(-2).join('/')}</td>
                <td className={g.score === 1 ? 'st-win' : g.score === 0 ? 'st-loss' : ''}>{g.score === 1 ? 'win' : g.score === 0 ? 'loss' : 'draw'}</td>
                <td className="dim">{g.reason} {g.board}{g.mate?.drop ? ' drop' : ''}</td>
                <td className="num">{pct(g.turn?.frac)}</td><td className="num">{pct(g.ahead_us)}</td>
                <td className="num">{g.sits_us} / {g.sits_them}</td><td className="num">{num(g.mat_diff, 1)}</td>
                <td className="num">{num(g.pocket_us, 1)} / {num(g.pocket_them, 1)}</td>
                <td className="dp-name" title={g.opening}>{g.opening}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  )
}

// One small chart on the shared time axis: our series (and optionally theirs), with moment markers.
function Strip({ title, us, them, yMin, yMax, unit, dur, marks, cursor, onPick }: {
  title: string; us: Pt[]; them?: Pt[]; yMin: number; yMax: number; unit: string; dur: number
  marks: { ts: number; kind: string }[]; cursor: number | null; onPick: (ts: number) => void
}) {
  const W = 720, H = 120, L = 44, R = 12, T = 8, B = 18
  const x = (s: number) => L + (s / Math.max(1, dur)) * (W - L - R)
  const y = (v: number) => T + (1 - (Math.max(yMin, Math.min(yMax, v)) - yMin) / (yMax - yMin)) * (H - T - B)
  const line = (p: Pt[]) => p.map(([a, b]) => `${x(a).toFixed(1)},${y(b).toFixed(1)}`).join(' ')
  return (
    <figure className="st-strip">
      <figcaption>{title}{them ? <> <span className="st-key" style={{ background: US }} /> us <span className="st-key" style={{ background: THEM }} /> them</> : null}</figcaption>
      <svg className="dp-chart" viewBox={`0 0 ${W} ${H}`} role="img" aria-label={title}
        onClick={e => {
          const r = (e.currentTarget as SVGSVGElement).getBoundingClientRect()
          const sx = ((e.clientX - r.left) / r.width) * W
          onPick(Math.max(0, ((sx - L) / (W - L - R)) * dur))
        }}>
        <line className="zero" x1={L} x2={W - R} y1={y(0)} y2={y(0)} />
        <text className="axis" x={L - 6} y={y(yMax) + 8} textAnchor="end">{yMax}{unit}</text>
        <text className="axis" x={L - 6} y={y(yMin)} textAnchor="end">{yMin}{unit}</text>
        {marks.map((m, i) => <line key={i} className={`st-mark ${m.kind}`} x1={x(m.ts)} x2={x(m.ts)} y1={T} y2={H - B} />)}
        {them && <polyline fill="none" stroke={THEM} strokeWidth={2} points={line(them)} />}
        <polyline fill="none" stroke={US} strokeWidth={2} points={line(us)} />
        {cursor != null && <line className="cursor" x1={x(cursor)} x2={x(cursor)} y1={T} y2={H - B} />}
        {[0, 0.25, 0.5, 0.75, 1].map(f => <text key={f} className="axis" x={x(f * dur)} y={H - 4} textAnchor="middle">{Math.round(f * dur)}s</text>)}
      </svg>
    </figure>
  )
}

function evText(e: Ev) {
  if (e.k === 'move') return `${e.board}: ${e.san}${e.cap ? ` (takes ${e.cap})` : ''}`
  if (e.k === 'pass') return `${e.piece} passed to ${e.to}`
  return `${e.seat} sits ${((e.ms ?? 0) / 1000).toFixed(1)} s`
}

function GameDetail({ g }: { g: Game }) {
  const [events, setEvents] = useState<GameEvent[] | null>(null)
  const [cursor, setCursor] = useState<number | null>(null)
  const [jump, setJump] = useState<{ ts: number; n: number } | undefined>(undefined)
  useEffect(() => {
    let live = true
    setEvents(null)
    fetch(`/api/study/${g.file.replace(/^runs\/study\//, '')}`).then(r => r.text()).then(t => { if (live) setEvents(parseJsonl(t)) })
    return () => { live = false }
  }, [g.file])
  const go = useCallback((ms: number) => setJump(j => ({ ts: ms, n: (j?.n ?? 0) + 1 })), [])
  const onCursor = useCallback((ts: number) => setCursor(ts / 1000), [])
  const marks = [
    ...g.sits.map(s => ({ ts: s.ts / 1000, kind: s.us ? 'sit-us' : 'sit-them' })),
    ...(g.turn ? [{ ts: g.turn.ts / 1000, kind: 'turn' }] : []),
    ...g.swings.map(s => ({ ts: s.ts / 1000, kind: 'swing' })),
  ]
  const moments: { ts: number; what: string; detail: string; cls: string }[] = [
    ...(g.turn ? [{ ts: g.turn.ts, what: 'Turning point', detail: `game turns for good at ${pct(g.turn.frac)} of its length`, cls: 'turn' }] : []),
    ...g.swings.map(s => ({
      ts: s.ts0, what: `Swing ${s.delta > 0 ? '+' : ''}${s.delta.toFixed(2)}`,
      detail: `our eval ${s.from.toFixed(2)} → ${s.to.toFixed(2)} (${s.seat}). Between: ${s.events.map(evText).join('; ') || 'nothing logged'}`,
      cls: 'swing',
    })),
    ...g.sits.filter(s => s.ms >= 1000).map(s => ({ ts: s.ts, what: `${s.us ? 'We sit' : 'They sit'} ${(s.ms / 1000).toFixed(1)} s`, detail: `${s.seat} (${s.why ?? '?'})`, cls: s.us ? 'sit-us' : 'sit-them' })),
    ...(g.mate ? [{
      ts: g.mate.ts, what: g.mate.reason === 'checkmate' ? `Mate: ${g.mate.san}` : g.mate.reason,
      detail: `board ${g.mate.board}, ${g.mate.drop ? 'a drop' : 'a move'} by ${g.mate.by_us ? 'us' : 'them'}; ${g.mate.loser_us ? 'OUR' : 'their'} seat ${g.mate.loser} lost; clock lead then ${(g.mate.lead_ms / 1000).toFixed(1)} s (us minus them, both seats)`,
      cls: 'mate',
    }] : []),
  ].sort((a, b) => a.ts - b.ts)
  return (
    <section className="st-detail">
      <h2>{g.file.split('/').slice(-3).join('/')} <small>{g.score === 1 ? 'win' : g.score === 0 ? 'loss' : 'draw'}, {g.reason} on {g.board}, {g.plies} plies, {g.dur_s} s</small></h2>
      <p className="dp-facts">
        <span>k <b>{g.k_us}</b> v <b>{g.k_them}</b></span>
        <span>ms/move <b>{g.think_per_dec_us}</b> v <b>{g.think_per_dec_them}</b></span>
        <span>ahead on clock <b>{pct(g.ahead_us)}</b> v <b>{pct(g.ahead_them)}</b></span>
        <span>sits <b>{g.sits_us}</b> ({g.sit_s_us} s) v <b>{g.sits_them}</b> ({g.sit_s_them} s)</span>
        <span>stale searches <b>{pct(g.stale_us)}</b> v <b>{pct(g.stale_them)}</b></span>
        <span>clock left <b>{g.clock_left_us_s}</b> v <b>{g.clock_left_them_s}</b> s</span>
        <span>opening <b>{g.opening}</b></span>
      </p>
      <div className="st-strips">
        <Strip title="Eval, each engine's own search, our side (+1 = we win)" us={g.curve_us} them={g.curve_them} yMin={-1} yMax={1} unit="" dur={g.dur_s} marks={marks} cursor={cursor} onPick={s => go(s * 1000)} />
        <Strip title="Clock lead: our two clocks minus theirs" us={g.lead_curve} yMin={-Math.max(5, ...g.lead_curve.map(p => Math.abs(p[1])))} yMax={Math.max(5, ...g.lead_curve.map(p => Math.abs(p[1])))} unit="s" dur={g.dur_s} marks={marks} cursor={cursor} onPick={s => go(s * 1000)} />
        <Strip title="Material (board + pocket): ours minus theirs" us={g.mat_curve} yMin={-Math.max(5, ...g.mat_curve.map(p => Math.abs(p[1])))} yMax={Math.max(5, ...g.mat_curve.map(p => Math.abs(p[1])))} unit="" dur={g.dur_s} marks={marks} cursor={cursor} onPick={s => go(s * 1000)} />
        <p className="st-note">Lines on the charts: <span className="st-legend turn">turning point</span> <span className="st-legend swing">swing</span> <span className="st-legend sit-us">we sit</span> <span className="st-legend sit-them">they sit</span>. Click a chart or a moment to jump the replay there.</p>
      </div>
      <h3>Key moments</h3>
      <ol className="st-moments">
        {moments.map((m, i) => (
          <li key={i} className={m.cls}>
            <button onClick={() => go(m.ts)}><span className="t">{(m.ts / 1000).toFixed(1)} s</span><b>{m.what}</b><span className="d">{m.detail}</span></button>
          </li>
        ))}
      </ol>
      <h3>Replay <small>T1 = {g.us_team === 'T1' ? 'us' : 'them'} (dark dots)</small></h3>
      {events ? <GameReplay key={g.file} events={events} onCursor={onCursor} jump={jump} /> : <p className="dp-empty">Loading the game…</p>}
    </section>
  )
}

export default function Study() {
  const [groups, setGroups] = useState<Group[] | null>(null)
  const [err, setErr] = useState<string | null>(null)
  const [part, setPart] = useState<'all' | 'wins' | 'losses'>('all')
  const [sel, setSel] = useState<string | null>(null)
  const [games, setGames] = useState<Game[] | null>(null)
  const [game, setGame] = useState<Game | null>(null)
  useEffect(() => {
    fetch('/api/study/study.json').then(r => r.ok ? r.json() : Promise.reject(new Error(`study.json: ${r.status}`)))
      .then(d => setGroups(d.groups)).catch(e => setErr(String(e)))
  }, [])
  const group = groups?.find(g => g.id === sel) ?? null
  useEffect(() => {
    if (!group) return
    let live = true
    setGames(null); setGame(null)
    fetch(`/api/study/${group.file}`).then(r => r.json()).then(d => { if (live) setGames(d) })
    return () => { live = false }
  }, [group])
  return (
    <div className="st-page">
      <header className="dp-top"><h1>Game study</h1><span className="dp-status">us = our engine (wave2 hybrid); them = upstream it13 unless the row says otherwise</span></header>
      <main className="st-main">
        {err && <p className="dp-empty">Could not load study data ({err}). Run <code>python3 tools/study_games.py</code> from the repo root.</p>}
        <Findings />
        {groups && <MatchTable groups={groups} part={part} setPart={setPart} sel={sel} setSel={setSel} />}
        {groups && <LeadScatter groups={groups} />}
        {group && (games ? <GameList games={games} part={part} open={setGame} sel={game?.file ?? null} /> : <p className="dp-empty">Loading games…</p>)}
        {game && <GameDetail g={game} />}
      </main>
    </div>
  )
}
