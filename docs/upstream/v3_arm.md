# upstream-v3 arm, Fairy-Stockfish arm, bughouse openings (WP-G)

Written 2026-11-17 for audit WP-G (`docs/audit_2026-11-17.md`, D3) and spec v3.

## 1. upstream-v3 (aminwoo/hivemind v3, twin-s-clock)

| Item | Value |
|---|---|
| Source | `https://github.com/aminwoo/hivemind` tag **v3.0.3** (commit 7a4dbdb, 2026-10-23), the latest v3.0.x tag on 2026-11-17 |
| Location | `third_party/upstream-hivemind/` (plain clone, not a submodule; our `third_party/hivemind` is untouched) |
| Build | `sh tools/build_upstream.sh` (`TAG`, `JOBS` default 2): `cmake --preset ninja-release -DHIVEMIND_BACKEND=onnxruntime -DONNXRuntime_ROOT=/opt/homebrew/opt/onnxruntime`, upstream's `-O3` release preset; binary `engine/build-ninja/hivemind.bin` (`hivemind` is a shell wrapper) |
| Network | `twin-s-clock.onnx` from HF `aminwoo/bughouse-twin-s-clock` revision `edaa246c` = **iteration 13**, the card's "latest promoted network"; sha256 `f4ac25e32a993776f26b42d79cc5c1e48b77a16a13d1e2a2db30528f43c9af73` (identical to `twin-s-clock-it13.onnx`). Converted to `twin-s-clock-fp32.onnx` with upstream's `convert_onnx_fp32.py` |
| Note | Release v3.0.3's own `fetch-network` pins revision `30995e7`, whose file is **iteration 10** (sha256 `40b1f9b6...` = `twin-s-clock-it10.onnx`). The card puts it13 at -1 Elo vs it10 (CI -33..+31, interim). Use `HF_REV=30995e780f192d56393cce94acdae234f25f6331 NET_SHA256=40b1f9b6c6e875a068372a6510d060d14acc5c85a8c88527d39d0194e12b2ab4` to fetch the release's network instead |
| Licence | MIT (repo and weights) |

### UCI mapping (`arena/players/upstream.py`)

* Every search: `Team white|black`, `TimeModel clock`, `TimeLeadMs`, `TeamMoveTimeMs`, `OpponentMoveTimeMs`,
  `RootMoveTimeMs`, `ucinewgame` (unless `reuse_tree`), `position fen <A> | <B>`, `go nodes N`, then `stop` after
  `bestmove` (upstream starts a permanent-brain search after every move). `Ponder false`, `BackgroundSearch false`.
  `RequireMoveOn` and `TimeAdvantage` are never sent: upstream's own `can_sit = lead > 0` decides sitting.
* **TimeLeadMs** = our team's lead over the diagonal opponents: T1 (= upstream's white team, A_white + B_black)
  `((Aw - Bw) + (Bb - Ab)) / 2`, T2 the negation (upstream `time_state.h`: `Aw - Bw = Bb - Ab` without increment).
  Equals `team_time_edge / 2`. Sign unit tests: `arena/tests/test_upstream.py::test_lead_sign_*`.
* **TeamMoveTimeMs / OpponentMoveTimeMs**: per-team EMA (alpha 0.3) of the time one action costs, read off the clocks
  (`MoveTimeTracker`): a seat's clock drop between two observations where it is not on turn, divided by the moves it
  made in between (sits included, which is what upstream's move time means). Before a team has a sample, this
  search's own budget stands in (equal speeds, speed plane 0).
* **RootMoveTimeMs** = this search's charged budget in ms (`nodes x ms_per_node`).
* Budget: fixed `nodes` per search (default 800, upstream's self-play budget) or `clock_k` (remaining / k).
  Upstream has no two-board time manager of its own (`wtime`/`btime` are parsed and ignored for bughouse).
* Charge: `nodes_searched x ms_per_node` (spec v3 contract 5); `charge="requested"` for the old rule.

### Speed on this Mac (M4 Max, `measure_upstream_ms_per_node`, startpos, wall ms per searched node)

| Provider | 100 nodes | 400 | 1600 |
|---|---|---|---|
| ONNX Runtime CPU (`HIVEMIND_COREML=off`) | 0.78 | 0.69 | 0.62 |
| Core ML (engine default on macOS) | 0.26 | 0.21 | 0.20 |

`UPSTREAM_MS_PER_NODE = 0.65` (CPU, the provider stock's `MS_PER_NODE = 5.5` per requested node was measured on;
stock searches ~4.2 ms per searched node at 200-400 nodes). So on this CPU upstream searches **~6.5x** the nodes of
it04 per second (upstream's card: ~3x on an RTX 4070). Spec v3 D2 wants both numbers measured on the gate GPU.

### Raw policy parity (`tools/upstream_check.py parity`)

The engine's own `policy` command (under exactly the time options a search would send) against an independent
evaluation: upstream's Python plane encoder (`src/hivemind/domain/board2planes.py`, 37 planes per board) plus the
two clock planes computed from the model card's formulas, ONNX Runtime CPU on the same FP32 file, softmax over the
legal moves plus pass with the engine's policy indices (castling king->rook square).

| Run | Positions | On-turn boards | Top-1 agree | max abs prior diff |
|---|---|---|---|---|
| seed 0, CPU provider | 20 | 26 | 26/26 | 1.65e-6 |
| seed 0, Core ML provider | 20 | 26 | 26/26 | 1.09e-6 |
| seed 1, CPU provider | 20 | 27 | 27/27 | 1.78e-6 |

Positions: 5 fixed (castling both sides, promotions, full pockets) + random walks with pocket gifts; leads
log-uniform in +-50..20,000 ms or 0, move times 80..3,000 ms.

### Smoke (`tools/upstream_check.py smoke`)

See section 4.

## 2. Fairy-Stockfish arm (`arena/players/fairy.py`)

Binary `third_party/Fairy-Stockfish/src/stockfish` (already built, "Fairy-Stockfish 181026"; bench ~1.8 M nps single
thread). CECP per seat; protocol and the sitting translation are in the module docstring. Findings while probing:

* `nodestime` (nodes as time) stops after a few hundred nodes in bughouse on this build, so Fairy runs on its own
  wall-clock time manager and the arena charges the measured wall time (`charge="wall"`; `charge="nodes"` with
  `FAIRY_MS_PER_NODE = 0.0005` is the alternative). Wall time is noisy under load; ms/node per search is logged.
* Fairy's sitting is wall-clock busy-waiting (after `ptell dead`, while `sitRequested`, and while waiting for a
  "virtual drop" of a piece it does not hold). The wrapper releases the engine (`ptell dead` / `ptell fast` sent back
  at once; `?` must not be used, it blocks Fairy's input thread until the search returns) and turns each into an arena
  wait. Partner messages (`fairy_sit`, `fairy_dead`, `fairy_x`, ...) travel over the arena channel.

Not done for Fairy (exact list in section 5).

## 3. Bughouse openings (`arena/openings_bughouse.py`)

16 canon lines from theory 05 section 2.6 (Fortress, Federkevic, Barbeau, e6-Nf6 ...d5 / ...Be7, e6-Nc6, Qe7, Scandinavian
retreat, Leaf Gambit, Exchange French, French without ...c5, Alekhine sacrifice, Two Knights, Hungarian, 1.d4 d5 2.Nf3
and 2.Nc3 without c4), 3 chess controls (Sicilian Alapin, QGD, King's Indian), the bare start. Captures go to the
partner's pocket on the other board. Sets: `bughouse` (240 canon pairs + 6 control pairs + bare start at 25%: 328
entries, 247 clusters), `bughouse_canon`, `chess_controls`, `bare`. Where a line needed a completing move the source
does not give, the module says so (`3.Nc3` filler in the e6-Nf6 lines, `3...exd5`, `3...Nc6`).

Scoring: `python -m arena.openings_bughouse --score --nodes 800` searches each pair with stock it04 for both teams
(both are on turn), `TimeAdvantage false`. it04 rates every such position about -0.6 for the team searching (its
no-advantage bias), so a one-team Q cannot be filtered; `balance = (Q_T1 - Q_T2) / 2` cancels it. Pairs with
|balance| > 0.35 are dropped from `bughouse` once `arena/openings_bughouse_scores.json` exists.

## 4. Results of the runs made for this package (2026-11-17, M4 Max shared with other agents' jobs)

**it04 vs upstream-v3 smoke** (`tools/upstream_check.py smoke --games 6`, `runs/upstream/smoke6/`): 60 s, increment 0,
`bughouse` openings (3 pairs, teams swapped), single process. it04 = stock binary + `hivemind-fp32.onnx`, centralized,
100 nodes at 5.5 ms per requested node (550 ms/move); upstream = twin-s-clock it13, 800 nodes at 0.65 ms per searched
node (~520 ms/move), Core ML provider for wall speed (simulated time is nodes-based, so the provider does not change
results beyond FP rounding). **Upstream 6-0, all by checkmate**, 66-124 plies, 3-9 upstream passes per game, 0
`bestmove (none)`, 0 fallback moves, 0 aborts; 20 min wall. The direction matches upstream's card (its small clock net
beats the it04 lineage at equal time). 6 games is a smoke, not a measurement: the audit's 200-game version is owed.

**it04 vs Fairy-Stockfish** (2 games, script in this session's scratchpad, not committed): 60 s, `bughouse` openings,
Fairy on its own wall-clock time manager charged its wall time. **Fairy 2-0 by checkmate** (57 and 80 plies), 23 Fairy
sits (dead / partner-sit / virtual-drop waits), 57 ptell messages relayed, 0 fallback moves; measured ~0.8-1.0 us per
Fairy node under load. This is the opposite of the paper's ">95% at 2 s" for Hivemind and of the audit's expected
"it04 beats Fairy clearly": it04 at 100 nodes (~550 ms of simulated time) is not the paper's 2 s/move engine, and Fairy
here is charged real wall time on a loaded machine. Not a result; the 20-game smoke at matched time is owed.

**Opening scoring** (`--score --nodes 800`, both teams): stopped at **80 of 247 pairs** (the shared machine made it
~1 pair/min). Partial numbers in `docs/upstream/openings_scores_partial.json`: mean Q_T1 -0.59 (it04's no-advantage
bias, the reason the balance is used), |balance| mean 0.043, max 0.117, so no scored pair comes near the 0.35 cut.
No `arena/openings_bughouse_scores.json` was written, so `bughouse` is currently unfiltered (all 247 clusters).

## 5. Not done / open

* 200-game it04-vs-upstream smoke (audit WP-G test 2): only 6 games run here. Run
  `python tools/upstream_check.py smoke --games 200 --out runs/upstream/smoke200` on PACE or overnight.
* Full opening scoring (167 pairs left; ~3 h on a free M4 Max at 800 nodes): rerun `python -m arena.openings_bughouse
  --score`, which writes `arena/openings_bughouse_scores.json` and so applies the |balance| <= 0.35 filter.
* Fairy: the 20-game "it04 beats Fairy clearly" smoke; a matched-time setting (Fairy on wall time vs simulated
  it04 time is not equal time: either measure it04's real wall time with `charge_wall_time`, or make Fairy's
  `nodestime` work and charge nodes); its `fast`/`slow` partner hints only change Fairy's own time manager, the
  arena does not act on them; a Fairy kind in `setups.py` (see the integration notes in the WP-G report).
* `upstream` and `fairy` kinds are not yet in `arena/players/setups.py` (owned by WP-C) and the `--openings` choices
  in `arena/run.py`, `arena/pool.py`, `arena/selfplay.py` are hard-coded to all/train/eval (WP-A/WP-C files): the
  new sets are in `OPENING_SETS` but those CLIs reject the names until integration.
* Upstream ms/node on the gate GPU (TensorRT build of upstream) for the D2 equal-time headline: only this Mac's CPU
  and Core ML numbers exist.
* `third_party/upstream-hivemind/` is an untracked clone; add it to `.gitignore` (integration).
