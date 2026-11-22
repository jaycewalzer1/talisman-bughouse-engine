// Durative tree page (durative.html): a step-through of wave 2's search-time model, from the v3 lockstep tree
// through the durative tree (spec v4), its diagnosis, the continuity fixes (v5) and the hybrid (v6), with the
// live PACE round from /api/pace (vite.devApi.ts). Sources: docs/custom_engine_spec_v4_durative.md,
// docs/durative_diagnosis.md, docs/custom_engine_spec_v5_continuity.md, docs/custom_engine_spec_v6_hybrid.md.

import { type ReactNode, useEffect, useState } from 'react'

type Mode = 'v3' | 'durative' | 'hybrid'
type Lane = 0 | 1 | 2 | 3
interface Row { name: string; games: number; score: number; lo: number; hi: number }
interface Pace { at: number; ok: boolean; running: number; pending: number; rows: Row[]; long: Row[]; prev: Row[]; verdict: string; raw: string }

const LANES = ['A · us', 'A · them', 'B · partner', 'B · them']
const US = (l: Lane) => l === 0 || l === 2
const T0 = 150, T1 = 960, MS = 2500
const tx = (ms: number) => T0 + (ms / MS) * (T1 - T0)
const ly = (l: Lane) => 54 + l * 58

// One tree level per entry: who acts, on which lanes, when the action lands. 'wait' = a forced-wait node.
interface Level { lanes: Lane[]; ms: number; wait?: boolean; tag?: string }
const PICTURES: Record<Mode, Level[]> = {
  v3: [
    { lanes: [0, 2], ms: 0 }, { lanes: [1, 3], ms: 600 }, { lanes: [0, 2], ms: 1200 }, { lanes: [1, 3], ms: 1800 },
  ],
  durative: [
    { lanes: [0], ms: 0, tag: 'plan ½' }, { lanes: [3], ms: 300 }, { lanes: [], ms: 300, wait: true }, { lanes: [1], ms: 550 },
    { lanes: [2], ms: 620, tag: 'plan ½' }, { lanes: [3], ms: 920 }, { lanes: [0], ms: 1100 }, { lanes: [1], ms: 1170 },
    { lanes: [2], ms: 1170 }, { lanes: [3], ms: 1540 }, { lanes: [0], ms: 1650 }, { lanes: [1], ms: 1790 },
    { lanes: [], ms: 1790, wait: true }, { lanes: [3], ms: 2160 }, { lanes: [0], ms: 2200 },
  ],
  hybrid: [
    { lanes: [0], ms: 0, tag: 'sit' }, { lanes: [3], ms: 400, tag: 'event lands' }, { lanes: [2], ms: 650, tag: 'takes N' },
    { lanes: [0], ms: 700, tag: 'N@f6' }, { lanes: [1, 3], ms: 1300 }, { lanes: [0, 2], ms: 1900 },
  ],
}

interface Step { label: string; title: string; mode: Mode | 'chart'; body: ReactNode; refs: string[] }

const STEPS: Step[] = [
  {
    label: 'Lockstep', title: 'Where we started: the v3 tree takes turns', mode: 'v3',
    body: <>
      <p>Upstream Hivemind v3 (and it04 before it) searches as if the two teams alternate <b>whole turns</b>: our team
        moves on every board where it is on turn, then their team does, each turn costing a fixed move time.</p>
      <p>Each dot is one tree level. Our two seats act in the <b>same</b> node (the bracket), so a two-board plan
        such as “partner trades on B, I drop the piece on A” is one joint action, one ply deep.</p>
      <p>Real bughouse does not take turns. Four clocks run, seats land moves whenever they finish thinking, and a
        piece captured on B shows up in a hand on A a moment later. v3 bolts time on with a lead plane and a hard
        rule: sitting is legal only when the diagonal lead is positive (<code>SitGate</code>).</p>
    </>,
    refs: ['docs/theory/06 (upstream clock model)', 'joint_action.h:183'],
  },
  {
    label: 'Durative', title: 'The durative tree: every seat on its own clock', mode: 'durative',
    body: <>
      <p>Spec v4 replaced the alternation with <b>ready times</b>. Each seat knows when its next action lands: ours at
        this search's budget (<code>RootMoveTimeMs</code>), theirs at their observed pace (<code>SeatPaceMs</code>).
        A node jumps to the earliest ready time and only that seat acts.</p>
      <ul>
        <li>A pass lasts until the next event on the other board, capped at <code>PassCapK</code> × budget.</li>
        <li>A captured piece lands in the partner's hand at the capture's landing time, not instantly.</li>
        <li>Mate races go to whoever lands first; flags are real losses.</li>
        <li>No <code>SitGate</code>: the tree should <i>discover</i> when sitting pays.</li>
      </ul>
      <p>Look at the picture: the same 2.5 s now takes ~15 levels instead of 4, plus grey forced-wait nodes where
        the team “to move” has nobody ready yet.</p>
    </>,
    refs: ['docs/custom_engine_spec_v4_durative.md', 'time_state.h turn_at_node'],
  },
  {
    label: 'It lost', title: 'First PACE round (2026-11-19): it lost badly', mode: 'chart',
    body: <>
      <p>Against upstream it13 at 120 s, equal clocks, ~270–300 games per row:</p>
      <ul>
        <li><b>durative on</b> (ref_120) 0.227</li>
        <li><b>durative on + gate</b> (sitgate) 0.206</li>
        <li><b>durative off + gate</b> (nodur) 0.436</li>
      </ul>
      <p>It wasn't speed: sitgate searched 15.7 nodes/ms, almost the same as nodur's 16.0. It wasn't eval calibration
        either: Brier scores matched. The durative engine's own eval <b>slid</b> through the game (−0.17 by decision
        20–30, −0.49 after 45) while nodur's stayed level. So it was choosing worse moves the whole game.</p>
      <p>Then a 40-position probe: as nodes go 1k→4k→16k, agreement with it13@64k is 57/50/55% with durative on
        versus 57/62/68% off. <b>More search did not help the durative tree.</b></p>
    </>,
    refs: ['runs/wave2 rows (below, “previous round”)', 'docs/continuity_baseline.md'],
  },
  {
    label: 'Split plans', title: 'Diagnosis D2: plans are torn in half', mode: 'durative',
    body: <>
      <p>The read-only diagnosis looked for a sign or backup bug and found none. Control: give all four seats the
        same pace (so time is synchronous) and the durative tree picks v3's move on <b>14 of 15</b> positions where we
        are on turn on both boards. The machinery is sound.</p>
      <p>The gap is structural. Look at the two “plan ½” dots: our A move and our B move now sit at levels 1 and 5,
        with opponent moves and a wait node between them. A coordinated two-board plan that v3 sees at depth 1 is
        now 4+ plies deep, and the network's policy prior is spread over each half separately. On positions where
        we are on turn on one board only, sync timing matches v3 just 13/25.</p>
      <p>That is why more nodes didn't converge: the tree spends them on interleavings, not on plans.</p>
    </>,
    refs: ['docs/durative_diagnosis.md D0, D2'],
  },
  {
    label: 'Value errors', title: 'Three smaller errors (D1, D3, D4)', mode: 'durative',
    body: <>
      <p><b>D1, we were modelled 15% faster.</b> Below the root every one of our actions cost the root budget (546 ms
        average) while theirs cost their real pace (623 ms). Our real pace was 629 ms. So in the tree we got free
        moves and q started optimistic. Fixing it removed ~0.05 of optimism but did not change move agreement.</p>
      <p><b>D3, wait nodes are evaluated by the wrong side.</b> The net gives the side to move a tempo bonus:
        V(us to move) + V(them to move) averages +0.26 on the same boards. At a grey wait node the waiting team is
        “to move”, so it gets that bonus wrongly. Real, but the patch didn't move agreement.</p>
      <p><b>D4, “elapsed” meant two things.</b> The arena sent time since the last landing; self-play (what
        training sees) sent time since the decision began. It changed the root move on 2 of 3 “sitting” positions.</p>
      <p>All three got fixed behind options (<code>OwnPaceBelowRoot</code>, <code>WaitNodeActingEval</code>, D4 in
        the arena) and all three are rows in the current round.</p>
    </>,
    refs: ['docs/durative_diagnosis.md D1, D3, D4', 'time_state.h action_time_ms'],
  },
  {
    label: 'Continuity', title: 'Spec v5 continuity fixes: none of them closed the gap', mode: 'chart',
    body: <>
      <p>v5 added three fixes, each behind its own switch:</p>
      <ul>
        <li><b>Team tree reuse</b> (c_reuse): keep the subtree after real landings. Before this, reuse fired 0 of 522
          times because the clock is in the node key.</li>
        <li><b>Joint plans</b> (c_joint): below the root our two seats pick a joint pair again, each half landing at
          its own time. This is the direct answer to D2.</li>
        <li><b>Plan carry</b> (c_carry): hand our plan to the partner's next search.</li>
      </ul>
      <p>Today's round r4: every durative row, all fixes on or one at a time, sits at <b>0.19–0.27</b> against it13.
        nodur sits around 0.41–0.45. Joint plans is the best single fix (0.265) and still nowhere near.
        Conclusion: the full durative tree is a dead end at these node budgets.</p>
    </>,
    refs: ['docs/custom_engine_spec_v5_continuity.md', 'cluster/launch_checks.sh:240-268'],
  },
  {
    label: 'Hybrid', title: 'Spec v6 hybrid: v3 picks the move, the clock picks when', mode: 'hybrid',
    body: <>
      <p>The hybrid keeps v3's joint tree for move choice (so plans stay one ply deep) and keeps exactly one durative
        idea: <b>waiting for an event that's about to land</b>.</p>
      <p>When we are on turn on A alone, an opponent seat is thinking on B and its landing is due within
        1.5 × our budget, the root may <b>sit</b> on A. In that search the opponent on B must land a move at depth 1
        (<code>HybridEventLands all</code>). That rule applies to every root child so the comparison stays fair.
        A sit is kept only if it beats the best move by a margin (20‰).</p>
      <p>In the picture: we sit on A, their B move lands at 400 ms, our partner takes a knight, and the knight
        reaches our hand in time for N@f6. That's the “sit for the knight” case v4 was built to find, without
        tearing every plan in half.</p>
      <p>Locally it looked inert (0.502 vs nodur over 273 games, ~0.35 sits per game). On PACE it doesn't.</p>
    </>,
    refs: ['docs/custom_engine_spec_v6_hybrid.md (H3, review items 1-11)'],
  },
  {
    label: 'Now', title: 'Where it stands', mode: 'chart',
    body: null,
    refs: ['bash cluster/status_wave2c.sh', '.venv/bin/python cluster/decide_full_run.py'],
  },
]

// Snapshot read 2026-11-22 12:17 EST, shown until /api/pace answers.
const SNAPSHOT: Pace = {
  at: Date.parse('2026-11-22T17:17:05Z'), ok: false, running: 29, pending: 0, verdict: '', raw: '',
  rows: [
    ['c_all_120', 293, .251, .201, .290], ['c_all_nod1_120', 265, .249, .199, .307], ['c_carry_120', 269, .247, .190, .316],
    ['c_joint_120', 251, .265, .216, .321], ['c_mateblock_120', 302, .220, .179, .278], ['c_nodur_120', 245, .408, .337, .461],
    ['c_ownpace_120', 269, .243, .199, .310], ['c_ref_120', 282, .257, .210, .321], ['c_reuse_120', 259, .222, .177, .287],
    ['c_same_120', 289, .540, .495, .592], ['c_waiteval_120', 268, .185, .136, .257], ['h_best_120', 268, .521, .466, .586],
  ].map(r => ({ name: r[0], games: r[1], score: r[2], lo: r[3], hi: r[4] }) as Row),
  long: [
    ['l_hybrid_120', 1252, .511, .489, .545], ['l_nodur_120', 1140, .454, .420, .481], ['l_same_120', 1224, .498, .478, .521],
  ].map(r => ({ name: r[0], games: r[1], score: r[2], lo: r[3], hi: r[4] }) as Row),
  prev: [
    ['ref_120', 300, .227, .180, .285], ['sitgate', 248, .206, .149, .250], ['nodur', 266, .436, .361, .491],
  ].map(r => ({ name: r[0], games: r[1], score: r[2], lo: r[3], hi: r[4] }) as Row),
}

const ROW_RE = /^(\w+)\s+(\d+)\s+([\d.]+) \[([\d.]+), ([\d.]+)\]/
function parsePace(at: number, ok: boolean, text: string): Pace {
  const [queue, status = ''] = text.split('== STATUS')
  const p: Pace = { at, ok, running: 0, pending: 0, rows: [], long: [], prev: [], verdict: '', raw: text }
  for (const line of queue.split('\n')) {
    if (/ RUNNING /.test(line)) p.running++
    else if (/ PENDING /.test(line)) p.pending++
  }
  let section: Row[] | null = null
  for (const line of status.split('\n')) {
    if (line.startsWith('==')) section = line.includes('r4/rows') || /wave2c\/r\d+\/rows/.test(line) ? p.rows
      : /\/long/.test(line) ? p.long : line.includes('previous round') ? p.prev : null
    const m = section && ROW_RE.exec(line)
    if (section && m) section.push({ name: m[1], games: +m[2], score: +m[3], lo: +m[4], hi: +m[5] })
  }
  return p
}

function Timeline({ mode }: { mode: Mode }) {
  const levels = PICTURES[mode]
  return (
    <svg viewBox="0 0 1020 300" role="img" aria-label={`Search timeline, ${mode} model: tree levels placed on four seat lanes by landing time`}>
      {LANES.map((name, l) => (
        <g key={name}>
          <line className="dt-lane" x1={T0} x2={T1} y1={ly(l as Lane)} y2={ly(l as Lane)} />
          <text className="dt-lane-label" x={T0 - 14} y={ly(l as Lane) + 5}>{name}</text>
        </g>
      ))}
      {[0, 500, 1000, 1500, 2000, 2500].map(ms => (
        <text key={ms} className="dt-tick" x={tx(ms)} y={286}>{ms} ms</text>
      ))}
      {mode === 'hybrid' && (
        <>
          <rect className="dt-sit" x={tx(0)} y={ly(0) - 9} width={tx(400) - tx(0)} height={18} rx={4} />
          <path className="dt-flight" d={`M${tx(650)} ${ly(2)} C ${tx(690)} ${ly(1)}, ${tx(690)} ${ly(1)}, ${tx(700)} ${ly(0) + 14}`} markerEnd="url(#dt-head)" />
        </>
      )}
      <defs>
        <marker id="dt-head" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" orient="auto">
          <path className="dt-arrow" d="M0 0 L8 4 L0 8 Z" />
        </marker>
      </defs>
      {levels.map((lv, i) => {
        const x = tx(lv.ms) + (lv.wait ? 22 : 0)
        if (lv.wait) {
          return (
            <g key={i}>
              <rect className="dt-wait" x={x - 11} y={ly(1) + 18} width={22} height={22} transform={`rotate(45 ${x} ${ly(1) + 29})`} />
              <text className="dt-num" x={x} y={ly(1) + 34}>{i + 1}</text>
            </g>
          )
        }
        const ys = lv.lanes.map(ly)
        return (
          <g key={i}>
            {ys.length > 1 && <line className="dt-joint" x1={x} x2={x} y1={Math.min(...ys)} y2={Math.max(...ys)} />}
            {lv.lanes.map(l => (
              <g key={l}>
                <circle className={US(l) ? 'dt-dot us' : 'dt-dot them'} cx={x} cy={ly(l)} r={13} />
                <text className={US(l) ? 'dt-num us' : 'dt-num'} x={x} y={ly(l) + 5}>{i + 1}</text>
              </g>
            ))}
            {lv.tag && <text className="dt-tag" x={x} y={ly(lv.lanes[0]) - 19}>{lv.tag}</text>}
          </g>
        )
      })}
      <text className="dt-caption" x={T1} y={22}>
        {mode === 'v3' ? '4 tree levels in 2.4 s' : mode === 'durative' ? `${levels.length} tree levels in 2.2 s` : 'one sit, then v3 joint turns'}
      </text>
    </svg>
  )
}

function Bars({ title, rows, highlight }: { title: string; rows: Row[]; highlight: (r: Row) => string }) {
  if (rows.length === 0) return null
  const X0 = 170, X1 = 640, w = (v: number) => X0 + v * (X1 - X0)
  const H = 26
  return (
    <figure className="dt-bars">
      <figcaption>{title}</figcaption>
      <svg viewBox={`0 0 760 ${rows.length * H + 30}`} preserveAspectRatio="xMinYMin meet" style={{ height: `${(rows.length * H + 30) * 0.14}vh` }} role="img" aria-label={title}>
        <line className="dt-half" x1={w(.5)} x2={w(.5)} y1={4} y2={rows.length * H + 6} />
        {[0, .25, .5, .75, 1].map(v => <text key={v} className="dt-tick" x={w(v)} y={rows.length * H + 24}>{v}</text>)}
        {rows.map((r, i) => {
          const y = 18 + i * H, cls = highlight(r)
          return (
            <g key={r.name} className={cls}>
              <text className="dt-row-name" x={X0 - 10} y={y + 5}>{r.name}</text>
              <line className="dt-ci" x1={w(r.lo)} x2={w(r.hi)} y1={y} y2={y} />
              <circle className="dt-pt" cx={w(r.score)} cy={y} r={5.5} />
              <text className="dt-row-val" x={X1 + 14} y={y + 5}>{r.score.toFixed(3)} · {r.games} g</text>
            </g>
          )
        })}
      </svg>
    </figure>
  )
}

const kind = (r: Row) => /hybrid|h_best/.test(r.name) ? 'k-hybrid' : /nodur/.test(r.name) ? 'k-nodur' : /same/.test(r.name) ? 'k-same' : 'k-dur'

export default function DurativeTree() {
  const [index, setIndex] = useState(0)
  const [pace, setPace] = useState<Pace>(SNAPSHOT)
  const [loading, setLoading] = useState(false)
  const step = STEPS[index]

  const refresh = () => {
    setLoading(true)
    fetch('/api/pace').then(r => r.json()).then((j: { at: number; ok: boolean; text: string }) => {
      if (j.ok) setPace(parsePace(j.at, true, j.text))
      else setPace(p => ({ ...p, ok: false, raw: j.text }))
    }).catch(() => setPace(p => ({ ...p, ok: false }))).finally(() => setLoading(false))
  }
  useEffect(refresh, [])

  const hy = pace.long.find(r => r.name === 'l_hybrid_120'), nd = pace.long.find(r => r.name === 'l_nodur_120')
  const sm = pace.long.find(r => r.name === 'l_same_120')
  const now = (
    <>
      <p><b>The hybrid is the first wave 2 engine to hold it13 even.</b> Long test, ~1,200 games per row at 120 s:
        hybrid {hy ? hy.score.toFixed(3) : '?'} [{hy?.lo}, {hy?.hi}] vs nodur {nd ? nd.score.toFixed(3) : '?'}
        [{nd?.lo}, {nd?.hi}]. The intervals don't overlap, so <code>decide_full_run.py</code> picks
        <b> hybrid, narrow</b>. The same-engine control reads {sm ? sm.score.toFixed(3) : '?'}, so there's no seat bias.</p>
      <p>Its verdict is still <b>NO-GO</b>, but only for bookkeeping: the round's rows are still running (they end at
        their 8 h limit), and the width A/B needs a second gated narrow iteration. ab2's narrow it2 finished self-play
        and training, then timed out before its gate, so it needs one resubmit.</p>
      <p>If those clear, the full run would be 10 iterations × 50k games at R ≈ 726 games/GPU-h, ≈ 708 GPU-h.
        That's 59 chained 16 h jobs on one GPU, about 30 days, because multi-GPU self-play isn't built.</p>
      <p>Self-play smokes all pass. One iteration each: hybrid narrow 0.60 vs it13, continuity narrow 0.575, nodur
        0.50 (200 games each, so noisy).</p>
    </>
  )

  return (
    <main className="nn-page dt-page">
      <h1>The durative tree</h1>
      <p className="nn-sub">Wave 2's time model, what went wrong, and the hybrid that replaced it. Model in every number:
        twin-s-clock it13 (HF aminwoo/bughouse-twin-s-clock@edaa246) and its wave 2 builds, vs upstream it13.</p>

      <ol className="nn-steps">
        {STEPS.map((s, i) => (
          <li key={s.label}>
            <button className="dp-btn" aria-current={i === index ? 'step' : undefined} onClick={() => setIndex(i)}>{s.label}</button>
          </li>
        ))}
      </ol>

      <div className="nn-layout">
        <div>
          {step.mode !== 'chart' && (
            <div className="nn-stage">
              <Timeline mode={step.mode} />
              <div className="nn-legend">
                <span><i className="dt-key us" /> our team</span>
                <span><i className="dt-key" /> their team</span>
                <span><i className="dt-key wait" /> forced-wait node</span>
                <span>number = tree depth</span>
              </div>
            </div>
          )}
          {step.mode === 'chart' && (
            <div className="nn-stage dt-charts">
              {index === 2 && <Bars title="Round 1, 2026-11-19 · score vs upstream it13, 120 s" rows={pace.prev.filter(r => ['ref_120', 'sitgate', 'nodur', 'noflag', 'racewin', 'instant'].includes(r.name))} highlight={kind} />}
              {index !== 2 && <Bars title="Long test (~1,200 games per row)" rows={pace.long} highlight={kind} />}
              {index !== 2 && <Bars title="Round r4 rows · score vs upstream it13 (c_same_120 is self-play control)" rows={[...pace.rows].sort((a, b) => b.score - a.score)} highlight={kind} />}
              <div className="nn-legend">
                <span><i className="dt-key k-hybrid" /> hybrid</span>
                <span><i className="dt-key k-nodur" /> durative off + gate</span>
                <span><i className="dt-key k-dur" /> durative on (variants)</span>
                <span><i className="dt-key k-same" /> control</span>
                <span>bar = 95% CI · 0.5 = even</span>
              </div>
            </div>
          )}
          <div className="dt-live">
            <span>{pace.ok ? 'PACE live' : 'PACE snapshot'} · {new Date(pace.at).toLocaleString()} · {pace.running} jobs running{pace.pending ? `, ${pace.pending} pending` : ''}</span>
            <button className="dp-btn" onClick={refresh} disabled={loading}>{loading ? 'Reading PACE…' : 'Refresh from PACE'}</button>
          </div>
          {!pace.ok && pace.raw && <pre className="dt-err">{pace.raw.slice(0, 400)}</pre>}
        </div>

        <aside className="nn-panel">
          <h2>{step.title}</h2>
          {step.body ?? now}
          <ul className="nn-refs">{step.refs.map(r => <li key={r}><code>{r}</code></li>)}</ul>
          <div className="nn-nav">
            <button className="dp-btn" disabled={index === 0} onClick={() => setIndex(index - 1)}>← Back</button>
            <button className="dp-btn" disabled={index === STEPS.length - 1} onClick={() => setIndex(index + 1)}>Next →</button>
          </div>
        </aside>
      </div>
    </main>
  )
}
