// Headless helper for the mock generator. Run from viewer/:
//
//   npx esbuild scripts/mock-tool.ts --bundle --platform=node --format=esm --outfile=node_modules/.tmp/mock-tool.mjs \
//     && node node_modules/.tmp/mock-tool.mjs check 20        # sanity-check 20 mock games
//   ...same build... && node node_modules/.tmp/mock-tool.mjs sample 7   # write public/replays/sample.jsonl (seed 7)
//
// (esbuild ships with Vite, so nothing extra to install.)

import { writeFileSync, mkdirSync } from 'node:fs'
import { Chess } from 'chess.js'
import { generateMockGame } from '../src/sources/mock'
import { type GameEvent, type Seat, PARTNER, fenPocket } from '../src/events/types'

function fail(msg: string): never { throw new Error(msg) }

/** Assert the invariants a viewer relies on. Returns a one-line summary. */
function check(events: GameEvent[]): string {
  if (events[0]?.type !== 'game_start') fail('first event is not game_start')
  const last = events[events.length - 1]
  if (last.type !== 'game_end') fail('last event is not game_end')
  if (events.filter(e => e.type === 'game_end').length !== 1) fail('more than one game_end')
  const pockets: Record<string, string> = { A: '', B: '' }
  let captures = 0, drops = 0, messages = 0, stalls = 0
  const stallOpen = new Set<Seat>()
  events.forEach((e, i) => {
    if (e.seq !== i) fail(`seq ${e.seq} at index ${i}`)
    if (i && e.ts < events[i - 1].ts) fail(`ts goes backwards at seq ${e.seq}`)
    if (e.type === 'move') {
      // the FEN must load in chess.js once pockets and ~ markers are stripped
      const plain = e.fen.replace(/\[[^\]]*\]/, '').replace(/~/g, '')
      try { new Chess().load(plain, { skipValidation: true }) } catch { fail(`bad fen at seq ${e.seq}: ${e.fen}`) }
      for (const v of Object.values(e.clocks)) if (v < 0) fail('negative clock')
      if (e.isDrop) {
        drops++
        if (!/^[PNBRQ]@[a-h][1-8]$/.test(e.uci)) fail(`bad drop uci ${e.uci}`)
        if (e.uci[0] === 'P' && /[18]$/.test(e.uci)) fail('pawn dropped on back rank')
      }
      if (e.capture) {
        captures++
        const nx = events[i + 1]
        if (nx?.type !== 'pass' || nx.piece !== e.capture || nx.fromSeat !== e.seat || nx.toSeat !== PARTNER[e.seat] || nx.ts !== e.ts)
          fail(`capture at seq ${e.seq} not followed by a matching pass`)
      }
      pockets[e.board] = fenPocket(e.fen)
    }
    if (e.type === 'pass') {
      const prev = events[i - 1]
      if (prev.type !== 'move' || !prev.capture) fail(`pass at seq ${e.seq} without capture`)
      // the receiving board's next FEN must contain the piece (checked loosely: pocket letter case)
      const isWhiteRecv = e.toSeat.endsWith('white')
      if ((e.piece === e.piece.toUpperCase()) !== isWhiteRecv) fail(`pass piece case wrong at seq ${e.seq}`)
    }
    if (e.type === 'message') {
      messages++
      if (PARTNER[e.fromSeat] !== e.toSeat) fail('message not between partners')
      if (e.deliverTs < e.ts) fail('deliverTs < ts')
    }
    if (e.type === 'stall') {
      if (e.active) { stalls++; stallOpen.add(e.seat) } else stallOpen.delete(e.seat)
    }
  })
  const end = last as Extract<GameEvent, { type: 'game_end' }>
  return `${(end.ts / 1000).toFixed(1)}s ${events.length} ev, ${captures} caps, ${drops} drops, ` +
    `${messages} msgs, ${stalls} stalls -> ${end.result} by ${end.reason} on ${end.board ?? '-'}`
}

const [mode = 'check', arg = '10'] = process.argv.slice(2)

if (mode === 'check') {
  const n = +arg
  const reasons: Record<string, number> = {}
  let maxTs = 0
  const t0 = Date.now()
  for (let seed = 1; seed <= n; seed++) {
    const ev = generateMockGame({ seed, gameId: `check-${seed}` })
    const line = check(ev)
    const end = ev[ev.length - 1] as Extract<GameEvent, { type: 'game_end' }>
    reasons[end.reason] = (reasons[end.reason] ?? 0) + 1
    maxTs = Math.max(maxTs, end.ts)
    console.log(`seed ${seed}: ${line}`)
  }
  console.log(`OK ${n} games, reasons ${JSON.stringify(reasons)}, longest ${(maxTs / 1000).toFixed(1)}s, ` +
    `${((Date.now() - t0) / n).toFixed(0)} ms/game to generate`)
} else if (mode === 'sample') {
  const ev = generateMockGame({ seed: +arg, gameId: 'sample-0001' })
  console.log(check(ev))
  mkdirSync('public/replays', { recursive: true })
  writeFileSync('public/replays/sample.jsonl', ev.map(e => JSON.stringify(e)).join('\n') + '\n')
  console.log('wrote public/replays/sample.jsonl')
} else {
  console.error('usage: mock-tool.mjs check [n] | sample [seed]')
  process.exit(1)
}
