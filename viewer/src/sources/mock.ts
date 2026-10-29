// Mock event source: plays fake-but-plausible bughouse games with no engine.
//
// How it works (two layers):
//  1. generateMockGame() runs a small discrete-event simulation of ONE whole game up front and
//     returns the complete list of schema-correct events (see ../../../docs/events.md).
//     chess.js gives us legal normal moves; we add bughouse drops, piece passing, clocks,
//     thinking, partner messages and stalls on top.
//  2. createMockSource() "plays" that list in (simulated) real time into the game store, then waits
//     a few seconds after game_end and starts a fresh game.
//
// Generating up front keeps the logic simple, makes games reproducible from a seed, and lets
// scripts/mock-tool.ts dump a game to public/replays/sample.jsonl.

import { Chess, type Color, type Move, type PieceSymbol, type Square } from 'chess.js'
import {
  type BoardId, type Clocks, type GameEvent, type Seat, type TeamId, PARTNER, TEAM_OF,
} from '../events/types'
import { useGame } from '../store/gameStore'
import type { Source } from './source'

// ---------------------------------------------------------------- tunables

const BASE_MS = 120_000        // 2:00 per seat: a board can't last longer than 2 * BASE_MS
const MAX_PLIES = 600          // safety cap (both boards together)
const VALUE: Record<PieceSymbol, number> = { p: 1, n: 3, b: 3, r: 5, q: 9, k: 0 }
const NAME: Record<PieceSymbol, string> = { p: 'pawn', n: 'knight', b: 'bishop', r: 'rook', q: 'queen', k: 'king' }
const POCKET_ORDER = 'qrbnp'
const SEAT_NAMES: Record<Seat, string> = {
  A_white: 'Kickflip', A_black: 'Heelflip', B_white: 'Nollie', B_black: 'Boneless',
}

// ---------------------------------------------------------------- small helpers

/** Tiny seeded PRNG (mulberry32) so a seed always produces the same game. */
function makeRng(seed: number) {
  let a = seed >>> 0
  return () => {
    a = (a + 0x6d2b79f5) >>> 0
    let t = a
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

const other = (c: Color): Color => (c === 'w' ? 'b' : 'w')
const seatOf = (board: BoardId, c: Color) => `${board}_${c === 'w' ? 'white' : 'black'}` as Seat
const colorOfSeat = (s: Seat): Color => (s.endsWith('white') ? 'w' : 'b')
const fileOf = (sq: Square) => sq.charCodeAt(0) - 97
const rankOf = (sq: Square) => +sq[1]
const dist = (a: Square, b: Square) => Math.max(Math.abs(fileOf(a) - fileOf(b)), Math.abs(rankOf(a) - rankOf(b)))
const letter = (t: PieceSymbol, c: Color) => (c === 'w' ? t.toUpperCase() : t)
const ALL_SQUARES: Square[] = []
for (let r = 1; r <= 8; r++) for (let f = 0; f < 8; f++) ALL_SQUARES.push(`${'abcdefgh'[f]}${r}` as Square)

/** Swap the side to move in a plain FEN (used after a drop, which chess.js can't do itself). */
function flipTurn(fen: string, mover: Color): string {
  const p = fen.split(' ')
  p[1] = other(mover)
  p[3] = '-'
  p[4] = '0'
  if (mover === 'b') p[5] = String(+p[5] + 1)
  return p.join(' ')
}

function kingSquare(chess: Chess, c: Color): Square {
  for (const row of chess.board()) for (const p of row) if (p && p.type === 'k' && p.color === c) return p.square
  throw new Error('no king')
}

/**
 * Arena rule (arena/RULES.md): a check by a single slider with an empty square in between is NOT
 * mate even if there is no legal move, because a partner might still supply a piece to block.
 */
function checkIsBlockable(chess: Chess): boolean {
  const me = chess.turn()
  const k = kingSquare(chess, me)
  const checkers = chess.attackers(k, other(me))
  if (checkers.length !== 1) return false
  const t = chess.get(checkers[0])?.type
  if (t === 'n' || t === 'p') return false
  return dist(k, checkers[0]) > 1
}

/** True bughouse mate for the side to move. */
function isBughouseMate(chess: Chess): boolean {
  return chess.isCheckmate() && !checkIsBlockable(chess)
}

// ---------------------------------------------------------------- simulation types

type Candidate =
  | { kind: 'move'; m: Move; score: number }
  | { kind: 'drop'; piece: PieceSymbol; sq: Square; score: number }

interface BoardSim {
  id: BoardId
  chess: Chess
  pockets: Record<Color, PieceSymbol[]>
  promoted: Set<Square>          // squares holding promoted pieces (they transfer as pawns)
  ply: number
  turnStart: number              // ts when the side to move started its turn
  turnToken: number              // bumped every turn; stale scheduled callbacks check it
  stallToken: number
  stall: { reason: string } | null
  thinkStart: number
  thinkPlan: { think: number; nodes: number; depth: number }
}

interface Request { piece: PieceSymbol; deadlineTs: number; accepted: boolean | null }

interface Scheduled { ts: number; order: number; fn: () => void }

export interface MockOptions {
  seed?: number
  gameId?: string
  baseMs?: number
}

// ---------------------------------------------------------------- the simulation

class Sim {
  rand: () => number
  queue: Scheduled[] = []
  order = 0
  now = 0
  out: GameEvent[] = []
  ended = false
  plies = 0
  clock: Clocks
  boards: Record<BoardId, BoardSim>
  requests: Partial<Record<Seat, Request>> = {}
  scratch = new Chess()
  baseMs: number

  constructor(opts: MockOptions) {
    this.rand = makeRng(opts.seed ?? Math.floor(Math.random() * 2 ** 31))
    this.baseMs = opts.baseMs ?? BASE_MS
    this.clock = { A_white: this.baseMs, A_black: this.baseMs, B_white: this.baseMs, B_black: this.baseMs }
    const mk = (id: BoardId): BoardSim => ({
      id, chess: new Chess(), pockets: { w: [], b: [] }, promoted: new Set(), ply: 0,
      turnStart: 0, turnToken: 0, stallToken: 0, stall: null, thinkStart: 0,
      thinkPlan: { think: 0, nodes: 0, depth: 0 },
    })
    this.boards = { A: mk('A'), B: mk('B') }
  }

  // --- scheduling -------------------------------------------------------
  at(ts: number, fn: () => void) { this.queue.push({ ts: Math.max(Math.round(ts), this.now), order: this.order++, fn }) }

  run(gameId: string): GameEvent[] {
    const seats = {} as Record<Seat, { name: string; team: TeamId; kind: string }>
    for (const s of Object.keys(SEAT_NAMES) as Seat[]) seats[s] = { name: SEAT_NAMES[s], team: TEAM_OF[s], kind: 'mock' }
    this.emit({
      type: 'game_start', seq: 0, ts: 0, game_id: gameId, setup: 'mock', seats,
      fens: { A: this.fen(this.boards.A), B: this.fen(this.boards.B) }, clocks: { ...this.clock },
      base_ms: this.baseMs, increment_ms: 0, channel_delay_ms: 0,
    })
    this.beginTurn(this.boards.A)
    this.beginTurn(this.boards.B)

    while (!this.ended && this.queue.length) {
      // pop the earliest scheduled callback (ties: first scheduled first)
      let bi = 0
      for (let i = 1; i < this.queue.length; i++) {
        const q = this.queue[i], b = this.queue[bi]
        if (q.ts < b.ts || (q.ts === b.ts && q.order < b.order)) bi = i
      }
      const item = this.queue.splice(bi, 1)[0]
      this.now = item.ts
      if (this.now > 4 * this.baseMs) { this.end('draw', 'aborted', null, null); break }
      item.fn()
    }
    // seq strictly increasing in emit order (which is already ts order)
    this.out.forEach((e, i) => { e.seq = i })
    return this.out
  }

  emit(e: GameEvent) { e.ts = Math.round(this.now); this.out.push(e) }

  // --- state views --------------------------------------------------------
  mover(b: BoardSim): Seat { return seatOf(b.id, b.chess.turn()) }

  live(seat: Seat): number {
    const b = this.boards[seat[0] as BoardId]
    if (this.mover(b) !== seat) return this.clock[seat]
    return Math.max(0, this.clock[seat] - (this.now - b.turnStart))
  }

  liveClocks(): Clocks {
    return { A_white: this.live('A_white'), A_black: this.live('A_black'), B_white: this.live('B_white'), B_black: this.live('B_black') }
  }

  /** Bracket-form FEN with pockets and `~` markers on promoted pieces (same as the arena). */
  fen(b: BoardSim): string {
    const rows = b.chess.board().map(row => {
      let s = '', empty = 0
      for (const p of row) {
        if (!p) { empty++; continue }
        if (empty) { s += empty; empty = 0 }
        s += letter(p.type, p.color) + (b.promoted.has(p.square) ? '~' : '')
      }
      return empty ? s + empty : s
    })
    const sortP = (a: PieceSymbol, c: PieceSymbol) => POCKET_ORDER.indexOf(a) - POCKET_ORDER.indexOf(c)
    const pocket = [...b.pockets.w].sort(sortP).map(t => t.toUpperCase()).join('') + [...b.pockets.b].sort(sortP).join('')
    return `${rows.join('/')}[${pocket}] ${b.chess.fen().split(' ').slice(1).join(' ')}`
  }

  material(b: BoardSim, c: Color): number {
    let m = 0
    for (const row of b.chess.board()) for (const p of row) if (p && p.color === c) m += VALUE[p.type]
    for (const t of b.pockets[c]) m += VALUE[t] * 0.9
    return m
  }

  /** Thinker-perspective eval in [-1, 1]: material on the board + pocket, plus noise. */
  evalFor(b: BoardSim, c: Color): number {
    const d = this.material(b, c) - this.material(b, other(c))
    const v = Math.tanh(d / 8) + (this.rand() - 0.5) * 0.15
    return Math.round(Math.max(-1, Math.min(1, v)) * 100) / 100
  }

  // --- message helper -----------------------------------------------------
  say(from: Seat, token: string, text: string, payload: Record<string, unknown> = {}) {
    this.emit({ type: 'message', seq: 0, ts: 0, deliverTs: Math.round(this.now), fromSeat: from, toSeat: PARTNER[from], token, text, payload })
  }

  // --- turn flow ----------------------------------------------------------
  beginTurn(b: BoardSim) {
    b.turnStart = this.now
    b.turnToken++
    b.stall = null
    const tok = b.turnToken
    const seat = this.mover(b)
    // the flag fires unless a move happens first (moves bump turnToken)
    this.at(this.now + this.clock[seat], () => { if (b.turnToken === tok) this.flag(seat) })

    if (this.now > 1500 && this.maybeStallAtTurnStart(b, seat)) return
    this.at(this.now + 60 + this.rand() * 200, () => this.thinkStart(b, tok))
  }

  /** Sit-stalls and need-stalls. Returns true when the seat starts stalling instead of thinking. */
  maybeStallAtTurnStart(b: BoardSim, seat: Seat): boolean {
    const partner = PARTNER[seat]
    const pb = this.boards[partner[0] as BoardId]
    const myClock = this.live(seat)
    const oppClock = this.live(seatOf(b.id, other(b.chess.turn())))

    // (1) partner is in check on the other board: "sit! don't feed them pieces"
    if (this.mover(pb) === partner && pb.chess.inCheck() && myClock > 15_000 && this.rand() < 0.35) {
      this.say(partner, 'sit', 'sit! I\'m in check, don\'t trade', {})
      this.startStall(b, seat, 'partner_sit', Math.min(1500 + this.rand() * 3000, myClock * 0.2))
      return true
    }

    // (2) a drop of some piece I don't have would mate: ask partner for it and sit for it
    this.expireRequests()
    if (!this.requests[seat] && !b.chess.inCheck() && this.rand() < 0.7) {
      const want = this.findDropMateWant(b)
      if (want) {
        const P = want.piece.toUpperCase()
        const deadlineTs = Math.round(this.now + 6000)
        this.say(seat, `need_${P}`, `need a ${NAME[want.piece]}, mate on ${want.sq}`, { piece: P, deadlineTs, value: 0.9 })
        this.requests[seat] = { piece: want.piece, deadlineTs, accepted: null }
        // partner answers a moment later: ok if they can take one right now, no otherwise
        this.at(this.now + 300 + this.rand() * 600, () => this.answerRequest(seat))
        if (myClock > 10_000 && myClock > oppClock - 8000 && this.rand() < 0.8) {
          this.say(seat, 'stall', `sitting until the ${NAME[want.piece]} comes`)
          this.startStall(b, seat, 'need_piece', Math.min(2500 + this.rand() * 4000, myClock * 0.3))
          return true
        }
      }
    }
    return false
  }

  answerRequest(requester: Seat) {
    const req = this.requests[requester]
    if (!req || this.ended) return
    const helper = PARTNER[requester]
    const hb = this.boards[helper[0] as BoardId]
    const canTake = hb.chess.turn() === colorOfSeat(helper) &&
      hb.chess.moves({ verbose: true }).some(m => m.captured === req.piece)
    if (canTake && this.rand() < 0.85) {
      req.accepted = true
      this.say(helper, 'ok', `ok, taking a ${NAME[req.piece]}`)
    } else {
      req.accepted = false
      this.say(helper, 'no', `can't get a ${NAME[req.piece]} now`)
      // no point sitting any longer
      const rb = this.boards[requester[0] as BoardId]
      if (rb.stall?.reason === 'need_piece') {
        if (this.rand() < 0.6) this.say(helper, 'go', 'go, don\'t wait')
        this.wakeStall(rb, 300 + this.rand() * 400)
      }
    }
  }

  expireRequests() {
    for (const s of Object.keys(this.requests) as Seat[]) {
      if ((this.requests[s]?.deadlineTs ?? 0) < this.now) delete this.requests[s]
    }
  }

  startStall(b: BoardSim, seat: Seat, reason: string, durMs: number) {
    b.stall = { reason }
    b.stallToken++
    const st = b.stallToken
    this.emit({ type: 'stall', seq: 0, ts: 0, seat, active: true, reason })
    if (Number.isFinite(durMs)) {
      this.at(this.now + durMs, () => {
        if (st !== b.stallToken) return
        // a partner often tells a long-sitting player to go
        if (reason !== 'no_moves' && this.rand() < 0.4) this.say(PARTNER[seat], 'go', 'go go go, clock!')
        this.endStall(b, st)
      })
    }
  }

  /** End the current stall of board b after `delayMs` (if it is still the same stall then). */
  wakeStall(b: BoardSim, delayMs: number) {
    const st = b.stallToken
    this.at(this.now + delayMs, () => this.endStall(b, st))
  }

  endStall(b: BoardSim, st: number) {
    if (this.ended || st !== b.stallToken || !b.stall) return
    const reason = b.stall.reason
    b.stall = null
    b.stallToken++
    this.emit({ type: 'stall', seq: 0, ts: 0, seat: this.mover(b), active: false, reason })
    const tok = b.turnToken
    this.at(this.now + 80 + this.rand() * 150, () => this.thinkStart(b, tok))
  }

  thinkStart(b: BoardSim, tok: number) {
    if (this.ended || tok !== b.turnToken || b.stall) return
    const seat = this.mover(b)
    const remaining = this.live(seat)
    // log-normal-ish think time around ~1.3 s, with the occasional long think
    let think = 2000 * Math.exp((this.rand() + this.rand() + this.rand() - 1.5) * 1.4)
    if (b.ply < 6) think *= 0.5                       // opening moves are quick
    if (this.rand() < 0.06) think *= 3                // deep think
    think = Math.round(Math.max(150, Math.min(think, remaining * 0.12, 9000)))
    const nodes = Math.round(think * (300 + this.rand() * 500))
    const depth = Math.max(2, Math.round(Math.log2(nodes) / 2.2))
    b.thinkStart = this.now
    b.thinkPlan = { think, nodes, depth }
    this.emit({ type: 'thinking', seq: 0, ts: 0, seat, phase: 'start', think_ms: think, nodes, eval: null, depth: null })

    // sometimes the partner sees a capture available here and asks for that piece
    const partner = PARTNER[seat]
    this.expireRequests()
    if (!this.requests[partner] && think > 500 && this.rand() < 0.2) {
      const caps = b.chess.moves({ verbose: true }).filter(m => m.captured)
      if (caps.length) {
        caps.sort((x, y) => VALUE[y.captured!] - VALUE[x.captured!])
        const piece = caps[0].captured!
        const P = piece.toUpperCase()
        this.at(this.now + think * 0.25, () => {
          if (this.ended || this.requests[partner]) return
          const deadlineTs = Math.round(this.now + 5000)
          this.say(partner, `need_${P}`, `need a ${NAME[piece]} in 5s`, { piece: P, deadlineTs, value: Math.round((0.3 + this.rand() * 0.4) * 100) / 100 })
          this.requests[partner] = { piece, deadlineTs, accepted: null }
        })
        this.at(this.now + think * 0.6, () => this.answerRequest(partner))
      }
    }

    this.at(this.now + think, () => this.doMove(b, tok))
  }

  doMove(b: BoardSim, tok: number) {
    if (this.ended || tok !== b.turnToken) return
    const seat = this.mover(b)
    const color = b.chess.turn()
    const best = this.choose(b, seat)
    this.emit({
      type: 'thinking', seq: 0, ts: 0, seat, phase: 'end', think_ms: Math.round(this.now - b.thinkStart),
      nodes: b.thinkPlan.nodes, eval: this.evalFor(b, color), depth: b.thinkPlan.depth,
    })

    if (!best) {
      // No legal move but not mated (blockable check / stalemate): sit until a piece arrives or flag.
      this.say(seat, 'stall', 'stuck, send me anything!')
      this.startStall(b, seat, 'no_moves', Infinity)
      return
    }

    // charge the clock, then apply the move
    this.clock[seat] = this.live(seat)
    b.turnStart = this.now
    let uci: string, san: string, captured: PieceSymbol | null = null, isDrop = false

    if (best.kind === 'move') {
      const m = best.m
      const wasPromoted = b.promoted.has(m.to)
      const done = b.chess.move({ from: m.from, to: m.to, promotion: m.promotion })
      uci = m.from + m.to + (m.promotion ?? '')
      san = done.san
      if (done.captured) captured = wasPromoted ? 'p' : done.captured
      b.promoted.delete(m.to)
      if (b.promoted.delete(m.from)) b.promoted.add(m.to)
      if (m.promotion) b.promoted.add(m.to)
    } else {
      isDrop = true
      b.chess.put({ type: best.piece, color }, best.sq)
      b.chess.load(flipTurn(b.chess.fen(), color), { skipValidation: true })
      b.pockets[color].splice(b.pockets[color].indexOf(best.piece), 1)
      uci = `${best.piece.toUpperCase()}@${best.sq}`
      san = uci + (b.chess.inCheck() ? '+' : '')
    }
    const mate = isBughouseMate(b.chess)
    // chess.js doesn't know drops can block, so fix up its '#'
    san = san.replace(/[+#]$/, '') + (mate ? '#' : b.chess.inCheck() ? '+' : '')
    b.ply++
    this.plies++

    // The receiving pocket's colour = the captured piece's colour = the opposite of the mover.
    const passLetter = captured ? letter(captured, other(color)) : null
    this.emit({
      type: 'move', seq: 0, ts: 0, board: b.id, seat, uci, san, isDrop, capture: passLetter,
      fen: this.fen(b), clocks: this.liveClocks(),
    })

    if (captured && passLetter) this.passPiece(b, seat, captured, other(color), passLetter)

    if (mate) return this.end(TEAM_OF[seat], 'checkmate', b.id, this.mover(b))
    if (this.plies >= MAX_PLIES) return this.end('draw', 'max_plies', null, null)

    // if I had told my partner to sit, release them now
    const pb = this.boards[PARTNER[seat][0] as BoardId]
    if (pb.stall?.reason === 'partner_sit') {
      this.at(this.now + 150, () => { if (pb.stall?.reason === 'partner_sit') this.say(seat, 'go', 'ok go!') })
      this.wakeStall(pb, 300)
    }

    this.beginTurn(b)
  }

  passPiece(fromBoard: BoardSim, fromSeat: Seat, t: PieceSymbol, recvColor: Color, pieceLetter: string) {
    const toSeat = PARTNER[fromSeat]
    const tb = this.boards[toSeat[0] as BoardId]
    tb.pockets[recvColor].push(t)
    this.emit({ type: 'pass', seq: 0, ts: 0, fromBoard: fromBoard.id, fromSeat, toSeat, piece: pieceLetter })
    const req = this.requests[toSeat]
    if (req && req.piece === t) delete this.requests[toSeat]
    else if (t !== 'p' && this.rand() < 0.2) this.say(fromSeat, `have_${t.toUpperCase()}`, `${NAME[t]} coming your way`, { piece: t.toUpperCase() })

    // the receiver was sitting for a piece and it's their turn: wake them up
    if (this.mover(tb) === toSeat && tb.stall && (tb.stall.reason === 'no_moves' || (tb.stall.reason === 'need_piece' && (!req || req.piece === t)))) {
      if (tb.stall.reason === 'need_piece') this.at(this.now + 120, () => this.say(toSeat, 'ok', 'got it, thx'))
      this.wakeStall(tb, 250 + this.rand() * 350)
    }
  }

  flag(seat: Seat) {
    if (this.ended) return
    const b = this.boards[seat[0] as BoardId]
    this.clock[seat] = 0
    this.end(TEAM_OF[seat] === 'T1' ? 'T2' : 'T1', 'flag', b.id, seat)
  }

  end(result: TeamId | 'draw', reason: string, board: BoardId | null, loserSeat: Seat | null) {
    if (this.ended) return
    this.ended = true
    this.emit({ type: 'game_end', seq: 0, ts: 0, result, reason, board, loserSeat })
  }

  // --- move choice ----------------------------------------------------------

  /** Legal drops for the side to move (pieces from its pocket onto empty squares). */
  dropSquares(b: BoardSim, t: PieceSymbol): Square[] {
    const c = b.chess
    const me = c.turn()
    const inCheck = c.inCheck()
    const myKing = inCheck ? kingSquare(c, me) : null
    const out: Square[] = []
    for (const sq of ALL_SQUARES) {
      if (c.get(sq)) continue
      if (t === 'p' && (sq[1] === '1' || sq[1] === '8')) continue
      if (inCheck) {
        // a drop is only legal if it blocks the check
        c.put({ type: t, color: me }, sq)
        const ok = !c.isAttacked(myKing!, other(me))
        c.remove(sq)
        if (!ok) continue
      }
      out.push(sq)
    }
    return out
  }

  /** Would dropping `t` on `sq` mate? (put -> flip turn -> test on a scratch board) */
  dropMates(b: BoardSim, t: PieceSymbol, sq: Square): { check: boolean; mate: boolean } {
    const c = b.chess
    const me = c.turn()
    c.put({ type: t, color: me }, sq)
    const check = c.isAttacked(kingSquare(c, other(me)), me)
    let mate = false
    if (check) {
      try {
        this.scratch.load(flipTurn(c.fen(), me), { skipValidation: true })
        mate = isBughouseMate(this.scratch)
      } catch { /* odd position: ignore */ }
    }
    c.remove(sq)
    return { check, mate }
  }

  /** Scan for "if only I had an X, I'd drop-mate": returns the cheapest such piece. */
  findDropMateWant(b: BoardSim): { piece: PieceSymbol; sq: Square } | null {
    const c = b.chess
    const saved = c.fen()
    const oppKing = kingSquare(c, other(c.turn()))
    let found: { piece: PieceSymbol; sq: Square } | null = null
    for (const t of ['p', 'n', 'b', 'r', 'q'] as PieceSymbol[]) {
      if (b.pockets[c.turn()].includes(t)) continue
      for (const sq of ALL_SQUARES) {
        if (dist(sq, oppKing) > 2 || c.get(sq)) continue
        if (t === 'p' && (sq[1] === '1' || sq[1] === '8')) continue
        if (this.dropMates(b, t, sq).mate) { found = { piece: t, sq }; break }
      }
      if (found) break
    }
    c.load(saved, { skipValidation: true })
    return found
  }

  /** Score every legal move/drop with simple heuristics + noise, return the best one. */
  choose(b: BoardSim, seat: Seat): Candidate | null {
    const c = b.chess
    const me = c.turn()
    const opp = other(me)
    const saved = c.fen()
    const req = this.requests[PARTNER[seat]]
    const wanted = req && req.accepted !== false && req.deadlineTs >= this.now ? req.piece : null
    const early = b.ply < 20
    const cands: Candidate[] = []

    for (const m of c.moves({ verbose: true })) {
      const pieceVal = VALUE[m.promotion ?? m.piece]
      const capVal = m.captured ? (b.promoted.has(m.to) ? 1 : VALUE[m.captured]) : 0
      const fromAttacked = c.isAttacked(m.from, opp)
      c.move({ from: m.from, to: m.to, promotion: m.promotion })
      const gives = c.inCheck()
      const mate = gives && isBughouseMate(c)
      const attacked = c.isAttacked(m.to, opp)
      const defended = c.isAttacked(m.to, me)
      c.undo()
      const hang = attacked ? (defended ? Math.max(0, pieceVal - capVal) * 0.6 : pieceVal * 0.95) : 0
      let s = capVal * 1.1 - hang + (gives ? 1.4 : 0) + (mate ? 100 : 0)
      if (m.promotion) s += m.promotion === 'q' ? 7 : -2
      if (fromAttacked && !attacked && m.piece !== 'k') s += pieceVal * 0.5   // rescue a threatened piece
      if (wanted && m.captured === wanted) s += 4                               // partner asked for this
      if (early) {
        if ((m.piece === 'n' || m.piece === 'b') && (m.from[1] === '1' || m.from[1] === '8')) s += 0.7
        if (m.flags.includes('k') || m.flags.includes('q')) s += 1.2
        if (m.piece === 'p' && (m.from[0] === 'd' || m.from[0] === 'e')) s += 0.5
        if (m.piece === 'q') s -= 0.3
        if (m.piece === 'r') s -= 0.3
      }
      if (m.piece === 'k' && !m.flags.includes('k') && !m.flags.includes('q')) s -= 0.8
      s += this.rand() * 1.1
      cands.push({ kind: 'move', m, score: s })
    }

    const oppKing = kingSquare(c, opp)
    for (const t of new Set(b.pockets[me])) {
      for (const sq of this.dropSquares(b, t)) {
        const { check, mate } = this.dropMates(b, t, sq)
        c.put({ type: t, color: me }, sq)
        const attacked = c.isAttacked(sq, opp)
        const defended = c.isAttacked(sq, me)
        c.remove(sq)
        const hang = attacked ? (defended ? 0.3 * VALUE[t] : 0.95 * VALUE[t]) : 0
        let s = 0.1 - hang + (check ? 1.6 : 0) + (mate ? 100 : 0) + (dist(sq, oppKing) <= 2 ? 0.5 : 0)
        if (t === 'p') s += (me === 'w' ? rankOf(sq) : 9 - rankOf(sq)) * 0.04
        s += this.rand() * 1.0
        cands.push({ kind: 'drop', piece: t, sq, score: s })
      }
    }
    c.load(saved, { skipValidation: true })   // put/remove may have touched ep/castling state

    if (!cands.length) return null
    // Look at the best few: usually (not always - they're fallible) avoid allowing a mate in one.
    cands.sort((x, y) => y.score - x.score)
    for (const cand of cands.slice(0, 6)) {
      if (cand.score > 50 || this.rand() > 0.85 || !this.allowsMateInOne(b, cand, saved)) return cand
    }
    return cands[0]
  }

  /** Does the opponent have a normal-move mate right after `cand`? (opponent drops ignored) */
  allowsMateInOne(b: BoardSim, cand: Candidate, saved: string): boolean {
    const c = b.chess
    const me = c.turn()
    if (cand.kind === 'move') c.move({ from: cand.m.from, to: cand.m.to, promotion: cand.m.promotion })
    else {
      c.put({ type: cand.piece, color: me }, cand.sq)
      c.load(flipTurn(c.fen(), me), { skipValidation: true })
    }
    let found = false
    for (const r of c.moves({ verbose: true })) {
      c.move({ from: r.from, to: r.to, promotion: r.promotion })
      found = c.inCheck() && isBughouseMate(c)
      c.undo()
      if (found) break
    }
    c.load(saved, { skipValidation: true })
    return found
  }
}

/** Simulate one complete game and return all its events (seq from 0, ts in ms). */
export function generateMockGame(opts: MockOptions = {}): GameEvent[] {
  return new Sim(opts).run(opts.gameId ?? 'mock-0001')
}

// ---------------------------------------------------------------- streaming source

const PAUSE_AFTER_END_MS = 7000

/**
 * Plays generated games into the store. `tick(dtMs)` is called every animation frame by the
 * controller with the elapsed *simulated* time (already multiplied by speed, 0 when paused).
 */
export function createMockSource(onStatus: (s: string) => void): Source {
  let events: GameEvent[] = []
  let cursor = 0
  let simTs = 0
  let gameNo = 0

  function newGame() {
    gameNo++
    const id = `mock-${String(gameNo).padStart(4, '0')}`
    events = generateMockGame({ gameId: id })
    cursor = 0
    simTs = 0
    useGame.getState().reset()
    onStatus(`mock game #${gameNo}`)
  }

  return {
    start() { newGame() },
    stop() { events = [] },
    tick(dtMs) {
      if (!events.length) return
      simTs += dtMs
      const batch: GameEvent[] = []
      while (cursor < events.length && events[cursor].ts <= simTs) batch.push(events[cursor++])
      const store = useGame.getState()
      if (batch.length) store.ingestMany(batch)
      const endTs = events[events.length - 1].ts
      store.setPlayhead(Math.min(simTs, endTs))
      if (cursor >= events.length && simTs > endTs + PAUSE_AFTER_END_MS) newGame()
    },
  }
}
