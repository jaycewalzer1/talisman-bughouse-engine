// Scheduler network page (network.html): a step-through diagram of the Phase 1 policy described in
// docs/phase1_scheduler.md and implemented in arena/players/scheduler.py.

import { type ReactNode, useState } from 'react'

const GATE = ['move', 'wait_P', 'wait_N', 'wait_B', 'wait_R', 'wait_Q', 'sit']
const GATE_CHIPS = ['move', 'P', 'N', 'B', 'R', 'Q', 'sit']
const TALK_CHIPS = ['quiet', 'P', 'N', 'B', 'R', 'Q', 'offer']
const THINK_CHIPS = ['×¼', '×½', '×1', '×2', '×4']
const END_CHIPS = ['keep_waiting', 'stop']

const FEATURE_GROUPS = [
  { id: 'Quick look', y: 40, fields: ['eval', 'eval_change', 'search_passed', 'wants', 'can_offer'] },
  { id: 'Clocks', y: 167, fields: ['own_clock_ms', 'partner_clock_ms', 'own_opponent_clock_ms', 'partner_opponent_clock_ms'] },
  { id: 'From the partner', y: 277, fields: ['partner_waiting', 'offered_piece', 'offered_plies', 'requested_piece'] },
  { id: 'Our state', y: 387, fields: ['holding', 'sits_in_row', 'nodes_used_fraction'] },
]

interface Step {
  label: string
  title: string
  what: ReactNode
  why: ReactNode
  refs: string[]
  lit: string[] | 'all'
}

const STEPS: Step[] = [
  {
    label: 'Overview', title: 'One decision, left to right', lit: 'all',
    what: 'The policy reads what this seat knows, condenses it into a shared summary, then makes three choices in order: gate, talk, think.',
    why: 'Hivemind stays frozen and still picks the actual move. This network only decides around it: whether to move at all, what to tell the partner, and how long to search.',
    refs: ['docs/phase1_scheduler.md:7-18'],
  },
  {
    label: 'Features', title: 'Everything it knows: 16 fields', lit: ['features'],
    what: 'Features is a plain record filled in at each decision: what the quick search found, all four clocks, what the partner has said, and our own state.',
    why: 'This list is the network’s whole world, so it can’t use anything left out. We’re keeping these 16 for the baseline. The cost of moving now versus waiting is the first candidate to add afterwards.',
    refs: ['arena/players/scheduler.py:34-55'],
  },
  {
    label: 'to_tensor', title: 'Everything becomes numbers', lit: ['features', 'e-ft', 'tensor'],
    what: 'A network can only multiply and add numbers, so to_tensor turns each field into one or more numbers, kept roughly between -1 and 1.',
    why: 'A 60,000 ms clock next to a 0.3 eval would drown the eval out, so clocks are divided by 60,000. Fields that name a piece get one slot per piece, for the same reason gate_choice is one-hot. In total: 32 numbers.',
    refs: ['to_tensor in arena/players/scheduler.py', 'arena/tests_phase1/test_scheduler.py:99-116'],
  },
  {
    label: 'Body', title: 'One shared body', lit: ['tensor', 'e-tb', 'body'],
    what: 'A few layers combine the inputs into a compact summary of the situation. Each layer multiplies by learned weights, adds, then applies a bend so that later layers can combine inputs in ways a straight line can’t.',
    why: 'Your decision. Time pressure and being behind matter to every head, and each game’s single win or loss now trains the same layers. There are two hidden layers of 64 units. The circles in the drawing are only illustrative.',
    refs: ['SchedulerPolicy in arena/players/scheduler.py'],
  },
  {
    label: 'Start + correction', title: 'Each score is a start plus a correction',
    lit: ['body', 'e-bg', 'e-bt', 'e-bk', 'e-be', 'gate', 'talk', 'think', 'end'],
    what: 'Each head gives one score, called a logit, per option. Higher means more likely. Each logit is a fixed starting value plus a correction learned from the body’s summary.',
    why: 'The correction’s last layer starts at exactly zero, so the untrained policy behaves like stock Hivemind. The starting value adds 6 to whatever stock would pick: move, or if the search passed, wait for the wanted piece, or sit when no piece is wanted. That makes stock’s choice about 98.5% likely. Training then only has to learn when to depart from it.',
    refs: ['docs/phase1_scheduler.md:26-27', 'arena/tests_phase1/test_scheduler.py:142-162'],
  },
  {
    label: 'Gate', title: 'Gate: move, or wait for a piece', lit: ['body', 'e-bg', 'gate', 'e-gc', 'gatechips'],
    what: 'Seven logits. Forbidden options are removed (gate_mask: no waiting for a piece we hold), softmax turns the rest into probabilities, exploration is mixed in, and one option is sampled.',
    why: 'It goes first because talk depends on it. Waiting for a piece also sends need_X, while sit waits silently, which is what stock does when it sits with no piece in mind.',
    refs: ['gate_mask and sample_action in arena/players/scheduler.py'],
  },
  {
    label: 'One-hot', title: 'The gate choice becomes seven inputs', lit: ['gatechips', 'e-gh', 'onehot'],
    what: 'The chosen option becomes seven numbers: a 1 in its slot and 0 in the others. Before the gate has been chosen (gate_choice=None) all seven are 0, meaning no gate information.',
    why: 'Your decision. A raw integer would make wait_R seem closer to wait_Q than wait_P is. With one-hot, every option has its own weight, so need_N can respond to wait_N alone.',
    refs: ['SchedulerPolicy.forward in arena/players/scheduler.py'],
  },
  {
    label: 'Talk', title: 'Talk: what to say', lit: ['body', 'e-bt', 'onehot', 'e-ht', 'talk', 'e-tc', 'talkchips'],
    what: 'Seven logits, computed from the body’s summary plus the one-hot gate choice. "offer" is masked unless the plan has something to offer, then one option is sampled.',
    why: 'Talk sees the gate choice so it can’t contradict it. Otherwise a seat waiting for a knight would be talking without knowing that it’s waiting.',
    refs: ['talk_mask in arena/players/scheduler.py', 'arena/tests_phase1/test_scheduler.py:172-181'],
  },
  {
    label: 'Think', title: 'Think: how long to search', lit: ['body', 'e-bk', 'think', 'e-kc', 'thinkchips'],
    what: 'Five multipliers of the base node budget. plan_nodes subtracts the quick look from the total, so the quick look isn’t free.',
    why: 'It reads only the shared summary. The budget is only used when the gate says move, so the gate choice would tell it nothing. It’s still sampled every time, which keeps every action the same shape.',
    refs: ['plan_nodes and sample_action in arena/players/scheduler.py'],
  },
  {
    label: 'Learning', title: 'One action, one number to learn from', lit: ['gatechips', 'talkchips', 'thinkchips', 'logprob'],
    what: 'The chance of the whole action is the three chances multiplied, so their logs add. After the game, team_reward gives +1, 0 or -1, and reinforce_loss pushes all of this seat’s decisions up or down together.',
    why: 'One result is shared by every decision in the game. That’s why the signal is scarce: good and bad decisions in a won game all get pushed the same way.',
    refs: ['reinforce_loss in arena/players/scheduler.py', 'arena/train_scheduler.py'],
  },
  {
    label: 'End wait', title: 'End wait: a separate moment', lit: ['body', 'e-be', 'end', 'e-ec', 'endchips'],
    what: 'Once a seat is waiting, the simulator asks again every 5 s and after any move or delivered message. This head chooses between keep waiting, which costs no search, and stop, which goes back through the gate. A piece arriving ends the wait regardless.',
    why: 'There’s no set wait length, so the policy has to learn when waiting has stopped being worth it. It starts at stop, which is closest to how stock re-decides, and each choice’s log-prob joins the game’s list so it trains.',
    refs: ['sample_end_wait in arena/players/scheduler.py', 'arena/players/scheduler_player.py'],
  },
]

const STEP_OF_PART: Record<string, number> = {
  features: 1, tensor: 2, body: 3, gate: 5, gatechips: 5, onehot: 6, talk: 7, talkchips: 7,
  think: 8, thinkchips: 8, logprob: 9, end: 10, endchips: 10,
}

const curve = (x1: number, y1: number, x2: number, y2: number) => {
  const mid = (x1 + x2) / 2
  return `M${x1} ${y1} C${mid} ${y1}, ${mid} ${y2}, ${x2} ${y2}`
}

const BODY_COLUMNS = [480, 536, 592]
const BODY_ROWS = [176, 218, 260, 302, 344]

export default function NetworkDiagram() {
  const [index, setIndex] = useState(0)
  const [gateChoice, setGateChoice] = useState(2)
  const step = STEPS[index]

  const isLit = (id: string) => step.lit === 'all' || step.lit.includes(id)
  const Part = ({ id, children }: { id: string; children: ReactNode }) => {
    const target = STEP_OF_PART[id]
    return (
      <g className={`nn-part${isLit(id) ? '' : ' dim'}${target === undefined ? '' : ' nn-click'}`}
        onClick={target === undefined ? undefined : () => setIndex(target)}>
        {children}
      </g>
    )
  }
  const Edge = ({ id, d }: { id: string; d: string }) => (
    <Part id={id}><path className="nn-edge" d={d} markerEnd="url(#nn-head)" /></Part>
  )
  const Chips = ({ x, y, labels, width, selected }: { x: number; y: number; labels: string[]; width: number; selected?: number }) => (
    <>
      {labels.map((label, i) => (
        <g key={label}>
          <rect className={`nn-chip${i === selected ? ' on' : ''}`} x={x + i * width} y={y} width={width - 4} height={40} rx={5} />
          <text className={`mid${i === selected ? ' on' : ''}`} x={x + i * width + (width - 4) / 2} y={y + 25}>{label}</text>
        </g>
      ))}
    </>
  )
  const Head = ({ id, y, name, count }: { id: string; y: number; name: string; count: string }) => (
    <Part id={id}>
      <rect className="nn-box" x={690} y={y} width={130} height={80} rx={7} />
      <text className="title mid" x={755} y={y + 30}>{name}</text>
      <text className="note mid" x={755} y={y + 50}>{count}</text>
      <text className="note mid" x={755} y={y + 68}>start + correction</text>
    </Part>
  )

  return (
    <main className="nn-page">
      <h1>Scheduler network</h1>
      <p className="nn-sub">The Phase 1 policy as built. Step through it, or click a part of the diagram.</p>

      <ol className="nn-steps">
        {STEPS.map((s, i) => (
          <li key={s.label}>
            <button className="dp-btn" aria-current={i === index ? 'step' : undefined} onClick={() => setIndex(i)}>{s.label}</button>
          </li>
        ))}
      </ol>

      <div className="nn-layout">
        <div>
          <div className="nn-stage">
            <svg viewBox="0 0 1176 596" role="img" aria-label="Scheduler network diagram: features feed a tensor, a shared body, and four decision heads">
              <defs>
                <marker id="nn-head" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" orient="auto">
                  <path className="nn-arrow" d="M0 0 L8 4 L0 8 Z" />
                </marker>
              </defs>

              <Part id="features">
                {FEATURE_GROUPS.map(g => (
                  <g key={g.id}>
                    <rect className="nn-box" x={16} y={g.y} width={250} height={32 + g.fields.length * 17} rx={7} />
                    <text className="title" x={28} y={g.y + 21} style={{ fontSize: 13 }}>{g.id}</text>
                    {g.fields.map((name, i) => <text key={name} className="field" x={28} y={g.y + 40 + i * 17}>{name}</text>)}
                  </g>
                ))}
              </Part>

              <Part id="e-ft">
                {FEATURE_GROUPS.map(g => (
                  <path key={g.id} className="nn-edge" d={curve(266, g.y + (32 + g.fields.length * 17) / 2, 306, 260)} markerEnd="url(#nn-head)" />
                ))}
              </Part>

              <Part id="tensor">
                <rect className="nn-box decided" x={306} y={194} width={100} height={132} rx={7} />
                <text className="title mid" x={356} y={238}>to_tensor</text>
                <text className="note mid" x={356} y={262}>one list of</text>
                <text className="note mid" x={356} y={279}>numbers</text>
                <text className="note mid" x={356} y={304}>32 numbers</text>
              </Part>
              <Edge id="e-tb" d="M406 260 L440 260" />

              <Part id="body">
                <rect className="nn-box decided" x={440} y={110} width={190} height={310} rx={9} />
                <text className="title mid" x={535} y={138}>Shared body</text>
                {BODY_COLUMNS.slice(0, -1).flatMap((x, c) => BODY_ROWS.flatMap(y1 => BODY_ROWS.map(y2 => (
                  <line key={`${c}-${y1}-${y2}`} className="nn-edge faint" x1={x} y1={y1} x2={BODY_COLUMNS[c + 1]} y2={y2} />
                ))))}
                {BODY_COLUMNS.flatMap(x => BODY_ROWS.map(y => <circle key={`${x}-${y}`} className="nn-node" cx={x} cy={y} r={9} />))}
                <text className="note mid" x={535} y={388}>2 layers of 64 units</text>
                <text className="note mid" x={535} y={406}>(drawing is illustrative)</text>
              </Part>

              <Edge id="e-bg" d={curve(630, 260, 690, 70)} />
              <Edge id="e-bt" d={curve(630, 260, 690, 260)} />
              <Edge id="e-bk" d={curve(630, 260, 690, 380)} />
              <Edge id="e-be" d={curve(630, 260, 690, 500)} />

              <Head id="gate" y={30} name="Gate" count="7 logits" />
              <Head id="talk" y={220} name="Talk" count="7 logits" />
              <Head id="think" y={340} name="Think" count="5 logits" />
              <Head id="end" y={460} name="End wait" count="2 logits" />

              <Edge id="e-gc" d="M820 70 L858 70" />
              <Part id="gatechips">
                <text className="note" x={860} y={42}>1. Gate decides first</text>
                <Chips x={860} y={50} labels={GATE_CHIPS} width={43} selected={gateChoice} />
              </Part>

              <Edge id="e-gh" d={`M${860 + gateChoice * 43 + 19.5} 90 L${860 + gateChoice * 43 + 19.5} 144`} />
              <Part id="onehot">
                {GATE_CHIPS.map((label, i) => (
                  <g key={label}>
                    <rect className="nn-box decided" x={860 + i * 43} y={146} width={39} height={30} rx={4} />
                    <text className="mid" x={860 + i * 43 + 19.5} y={166} style={{ fontWeight: i === gateChoice ? 700 : 400 }}>{i === gateChoice ? 1 : 0}</text>
                  </g>
                ))}
                <text className="note" x={860} y={194}>One-hot: seven extra inputs for Talk</text>
              </Part>
              <Edge id="e-ht" d="M858 161 L755 161 L755 218" />

              <Edge id="e-tc" d="M820 260 L858 260" />
              <Part id="talkchips">
                <text className="note" x={860} y={232}>2. Talk, knowing the gate choice</text>
                <Chips x={860} y={240} labels={TALK_CHIPS} width={43} />
              </Part>


              <Edge id="e-kc" d="M820 380 L858 380" />
              <Part id="thinkchips">
                <text className="note" x={860} y={352}>3. Think</text>
                <Chips x={860} y={360} labels={THINK_CHIPS} width={50} />
              </Part>

              <Edge id="e-ec" d="M820 500 L858 500" />
              <Part id="endchips">
                <text className="note" x={860} y={472}>Asked again while waiting</text>
                <Chips x={860} y={480} labels={END_CHIPS} width={120} />
              </Part>

              <Part id="logprob">
                <rect className="nn-box" x={690} y={552} width={470} height={34} rx={6} />
                <text className="mid" x={925} y={574}>log_prob = log P(gate) + log P(talk) + log P(think)</text>
              </Part>
            </svg>
          </div>
          <div className="nn-legend">
            <span><i className="nn-swatch" /> Fixed by the spec or tests</span>
            <span><i className="nn-swatch decided" /> Decided in our discussion</span>
            
          </div>
        </div>

        <aside className="nn-panel">
          <h2>{step.title}</h2>
          {step.label === 'One-hot' && (
            <>
              <div className="nn-try">
                {GATE.map((g, i) => (
                  <button key={g} className="dp-btn" aria-pressed={i === gateChoice} onClick={() => setGateChoice(i)}>{g}</button>
                ))}
              </div>
              <code className="nn-vector">{GATE[gateChoice]} → [{GATE.map((_, i) => (i === gateChoice ? 1 : 0)).join(', ')}]</code>
            </>
          )}
          <p>{step.what}</p>
          <h3>Why</h3>
          <p>{step.why}</p>
          <ul className="nn-refs">{step.refs.map(r => <li key={r}><code>{r}</code></li>)}</ul>
          <div className="nn-nav">
            <button className="dp-btn" disabled={index === 0} onClick={() => setIndex(index - 1)}>Back</button>
            <button className="dp-btn" disabled={index === STEPS.length - 1} onClick={() => setIndex(index + 1)}>Next</button>
          </div>
        </aside>
      </div>
    </main>
  )
}
