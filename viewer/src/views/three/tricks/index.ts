// Registry: trick id -> Trick (wiring only; the tricks themselves live in their own files).
import type { Seat } from '../../../events/types'
import { WIN_TRICK_OF, type Trick } from './types'
import { treFlip } from './treFlip'
import { kickflip } from './kickflip'
import { shuv360 } from './shuv360'
import { ollieImpossible } from './ollieImpossible'

export const TRICKS: Record<(typeof WIN_TRICK_OF)[Seat], Trick> = { treFlip, kickflip, shuv360, ollieImpossible }

/** The trick this seat performs when its team wins. */
export const winTrickOf = (seat: Seat): Trick => TRICKS[WIN_TRICK_OF[seat]]
