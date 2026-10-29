// =====================================================================================
//  STREET-TALK LINE TABLE  (edit me!)
// =====================================================================================
// Every agent message token maps to a list of short speech-bubble lines. The bubble picks the
// variants in rotation (round-robin per token) so the same line doesn't repeat back to back.
//
// How to edit:
//   - Add/remove/reword strings in the arrays below. Keep them SHORT: max ~6 words, ALL CAPS,
//     punchy (the bubbles are sized for a glance, not for reading).
//   - Keys are the arena message tokens (docs/events.md): need_P|N|B|R|Q, have_P|N|B|R|Q,
//     stall, go, sit, ok, no. Two extra keys are view-only:
//       status  -> an `ok` message whose payload.kind === "status" (search-status share)
//       pass    -> a captured piece being sent to the partner ("HERE, CATCH!")
//   - Unknown tokens fall back to FALLBACK.
//   - ANNOUNCER holds the big top-centre hype texts; END_TEXT the game-over calls.
// This file is pure data + tiny helpers (no React / three imports) so it can be unit-checked in node.
// =====================================================================================

export const LINES: Record<string, string[]> = {
  need_P: ['YO I NEED A PAWN!', 'PAWN ME, QUICK!', 'GIMME A PAWN!'],
  need_N: ['YO I NEED A KNIGHT!', 'HORSE! NEED A HORSE!', 'KNIGHT ME UP!'],
  need_B: ['NEED A BISHOP, FAST!', 'BISHOP! WHO GOT ONE?', 'SLIDE ME A BISHOP!'],
  need_R: ['YO I NEED A ROOK!', 'ROOK ME, ROOK ME!', 'GOTTA HAVE A ROOK!'],
  need_Q: ['I NEED THE QUEEN!', 'QUEEN ME UP, NOW!', 'GET ME A QUEEN!'],

  have_P: ['GOT A PAWN FOR YOU', 'PAWN COMING THROUGH', 'SPARE PAWN, WANT IT?'],
  have_N: ['GOT A KNIGHT FOR YOU', 'HORSE ON THE WAY', 'KNIGHT IN MY POCKET'],
  have_B: ['GOT A BISHOP FOR YOU', 'BISHOP ON DECK', 'BISHOP READY TO SEND'],
  have_R: ['GOT A ROOK FOR YOU', 'ROOK ON THE WAY', 'HOLDING A ROOK, SAY WORD'],
  have_Q: ['GOT A QUEEN FOR YOU', 'QUEEN ON DECK!', 'SHE IS ALL YOURS'],

  stall: ['HOLD UP, HOLD UP...', "I'M SITTING. CHILL.", 'WAITING ON YOU...'],
  go: ['GO GO GO!', 'MOVE! MOVE! MOVE!', "DON'T WAIT, PLAY!"],
  sit: ['SIT TIGHT!', "DON'T MOVE YET!", 'CHILL. WAIT FOR IT.'],
  ok: ['BET.', 'SAY LESS.', 'GOT YOU.'],
  no: ["CAN'T, MY BAD", 'NAH, NOT NOW', 'NO CAN DO'],

  status: ['PEEP THIS LINE', 'CHECK MY LINE', "HERE'S THE PLAN"],
  pass: ['HERE, CATCH!', 'INCOMING!', 'TAKE THIS!'],
}

export const FALLBACK = ['YO!']

/** Big hype texts for the announcer layer (top-centre). */
export const ANNOUNCER = {
  doubleDrop: 'DOUBLE DROP!',
  timePressure: 'TIME PRESSURE!',
  clutchSave: 'CLUTCH SAVE!',
  queenSnatch: 'QUEEN SNATCH!',
} as const

/** Game-over calls by `game_end.reason`. The announcer shows these with priority. */
export const END_TEXT: Record<string, string> = {
  checkmate: 'CHECKMATE!',
  flag: 'FLAGGED!',
  stalemate: 'STALEMATE!',
  max_plies: "TIME'S UP!",
  aborted: 'ABORTED',
}
export const END_FALLBACK = 'GAME OVER!'
export const isEndText = (t: string) => t === END_FALLBACK || Object.values(END_TEXT).includes(t)

export const PIECE_NAME: Record<string, string> = { P: 'PAWN', N: 'KNIGHT', B: 'BISHOP', R: 'ROOK', Q: 'QUEEN', K: 'KING' }
export const PIECE_VALUE: Record<string, number> = { P: 1, N: 3, B: 3, R: 5, Q: 9, K: 0 }

const rot: Record<string, number> = {}

/** Next line for a token (round-robin over its variants). `key` overrides the table key. */
export function lineFor(token: string, key?: string): string {
  const k = key ?? token
  let list = LINES[k]
  if (!list) {
    // need_X / have_X with an odd piece letter -> generic
    const m = /^(need|have)_(\w)$/.exec(k)
    list = m ? [m[1] === 'need' ? `NEED A ${PIECE_NAME[m[2].toUpperCase()] ?? 'PIECE'}!` : `GOT A ${PIECE_NAME[m[2].toUpperCase()] ?? 'PIECE'} FOR YOU`] : FALLBACK
  }
  const i = rot[k] ?? 0
  rot[k] = i + 1
  return list[i % list.length]
}

/** True for a search-status share: token `ok` with payload.kind === "status". */
export function isStatusShare(token: string, payload?: Record<string, unknown>): boolean {
  return token === 'ok' && payload?.kind === 'status'
}

/** The "best" field of a status payload as a short string ('' if missing). */
export function statusBest(payload?: Record<string, unknown>): string {
  const b = payload?.best
  return typeof b === 'string' ? b.slice(0, 10) : b == null ? '' : String(b).slice(0, 10)
}

/** Bubble text for a status share: "PEEP THIS LINE: Nf3" (the move keeps its case). */
export function statusLine(best: string): string {
  const head = lineFor('ok', 'status')
  return best ? `${head}: ${best}` : head
}

/** Trick-text label for a capture/drop: "KNIGHT DROP", "PAWN SNATCH". */
export function trickLabel(kind: 'capture' | 'drop', piece: string): string {
  const name = PIECE_NAME[piece.toUpperCase()] ?? 'PIECE'
  return `${name} ${kind === 'drop' ? 'DROP' : 'SNATCH'}`
}
