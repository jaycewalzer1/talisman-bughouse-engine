# Custom engine spec v7: league self-play, learned time policy, real-clock gate

Draft, 2026-11-27. Implements phases 1 and 2 of `docs/plan_league_2026-11-25.md` (approved: "ship it all"). Phase 3
(scale) and the final match are out of scope. Builds on spec v6 (`docs/custom_engine_spec_v6_hybrid.md`) as built in
`third_party/wave2-hivemind` (branch `talisman/wave2`, HEAD and pin `d5802ee`, `cluster/wave2_engine.ref`) and on the
fairness audit (`docs/fairness_audit_2026-11-25.md`, cited as "audit #N"). Line numbers are at main `6e81cd9` and fork
`d5802ee`. Fork paths are written `W2:<path>` (relative to `third_party/wave2-hivemind/`).

Five work packages (WP1-WP5) build this in parallel. Each owns a disjoint set of files (section 9). Where one WP needs
code in another's file, this spec defines the hook: the owner writes the call site exactly as specified, the other
writes the callee in its own file.

## 0. Ground rules (all WPs)

- **Default off.** Every new flag, option and controller keyword defaults to the old behaviour. With none of them set,
  rl-loop, self-play, the arena, the tune harness, anchors, the final match and the running `hybrid1` config must
  produce the same commands, markers, logs and games as today. Tests check this ("defaults unchanged" tests).
- **Commits.** Small progress commits of your own files only (`git add <your paths>`). Main repo: branch
  `custom-engine`. Fork: branch `talisman/wave2` in `third_party/wave2-hivemind` (a separate clone, not a submodule).
  If `index.lock` exists, wait a few seconds and retry. Never reset, check out or delete other agents' files. Do not
  push.
- **Machine.** 36 GB M4 Max shared with other agents and kokopelli jobs. C++ builds in your own dir,
  `W2:engine/build-<wp>` (e.g. `build-wp3`), configured like `W2:src/hivemind/cli/selfplay.py:BUILD_COMMAND`
  (`cmake -S engine -B engine/build-wp3 -G Ninja -DCMAKE_BUILD_TYPE=Release`; ONNX Runtime backend on the Mac), built
  with `nice cmake --build ... -j4`. One heavy build or long match at a time. Python: main repo `.venv/bin/python`
  (uv-managed), fork `W2:.venv` (`cd third_party/wave2-hivemind && uv run ...`).
- **PACE is not reachable.** Cluster-side work is a repo script with short command lines (long pasted one-liners get
  truncated), checked locally with `DRY_RUN=1` and the fake Slurm in `tests/cluster/fake_slurm`. TensorRT code paths
  cannot be run here: write them by mirroring an existing path (the arrivals head) and give them a GPU check job.
- **Tests stay fast** (seconds, not minutes). Main: `.venv/bin/python -m pytest arena/tests tests/cluster -q`. Fork:
  `uv run pytest tests -q`. Engine: `engine/build-<wp>/tests/hivemind_tests --gtest_filter=...`.
- Style: match surrounding code and comment density (docstrings that say what and why, plain sentences).

## 1. What exists today (reading notes)

- **rl-loop** (`W2:src/hivemind/cli/rl_loop.py`, 1935 lines). `Loop.iteration` (1471): generator marker (balance plan
  and `selfplay_mode` frozen before it), self-play segments (local, planned, or shared via `Claims`), `val`, `train`,
  `balance_after_train`, `match("generator")` at 100 ms per move (C++ tournament, `MATCH_ENGINE`), `promote`
  (1154: score rule, overridden by a sequential gate's H0/H1), `match` vs the fixed baseline (it0 = it13). Stage
  markers in `rl-itN/stages/*.done`. `planned_segments()` (723) returns the search balance plan's segment table
  `{segNN: {theta, seed, games, engine_args}}`; `next_claim` and `selfplay` honour it (engine args with
  `{segment_dir}` substituted). `--search-balance` controller discovery: `search_balance_controller()` (311) looks for
  `cluster/search_balance.py` above `PROJECT_ROOT`. `WAVE2_DEFAULTS` (296) gives defaults to workers/old namespaces.
- **Self-play** (`W2:engine/src/tools/selfplay.cc`, `play_seat_clock_game` 1139). One network for both teams
  (`Agent agents[2]`, both on the same `threadEngines`). Budget per search: `randomized_node_budget` = `--nodes` (800)
  x (1 +- 0.05), then the playout cap (`cap_decision`). Time: `SeatClockGame::sample_start`
  (`seat_clock_game.h`) draws a base from the mix (60/120/180/300 s) and, with `--seat-move-times` (on in the wave 2
  preset), **a move time per seat, log-uniform in [100, 3000] ms** (`ClockSampling`, `clock_sampling.h`);
  `plan_cost` charges that fixed time (plus jitter and lag) whatever the search's size. So self-play already has
  random paces, but pace and search size are unrelated: a slow seat pays clock and gets nothing for it, and no seat
  ever chooses its pace. Records: HDST distill chunks (`--distill-output`), both teams' searched decisions.
  `--record-state-output` dumps one JSON line per record so `hivemind reanalyse` can rebuild each root exactly.
- **Arena wave2 controller** (`arena/players/upstream.py`). Budget `clock / clock_k / ms_per_node` (511-516).
  `seat_decisions` (728-779) infers whether an opponent seat on turn is sitting with threshold
  `1.5 x min(pace EMA, clock_at_start / clock_k)`, where `clock_k` is **our own** (called at 910 with
  `clock_k=self.clock_k`): audit #3. Self-play's C++ side is not affected: `SeatClockGame` hands the engine the true
  `SeatSitting`/`SeatElapsedMs` and observed paces, so nothing in self-play mirrors this estimate (checked:
  `tools/strength_harness.py` is the only other caller, and it keeps the default).
- **think_gain (spec v2 B)** exists only in the OLD stack: `third_party/hivemind` (clock engine, `rise_mobile_v3`,
  trainer, `engine.cc` binding), `arena/players/hivemind.py` (`parse_think_gain` 184, `think_head` 736-1050),
  `arena/selfplay.py`, `tools/think_labels.py`. The wave2 engine, the twin-board network (`W2:src/hivemind/
  architectures/twin_board.py`) and the distill trainer (`W2:src/hivemind/distill/train.py`, `data.py`) have none of
  it. The wave2 engine reads extra outputs by name (`arrivals` in `W2:engine/src/nn/engine.cc` 686 and
  `engine_ort.cc` 341): the pattern to copy.
- **Cluster.** `cluster/launch_full_run.sh` (chain of `wave2_iterate.sbatch` jobs `full_<RUN>`, workers `sp_<RUN>`,
  watchdog with `GPU_HOURS` cap; `WAVE2_EXTRA` is word-split into rl-loop's arguments, workers inherit it).
  `cluster/launch_anchor_match.sh` is the idempotent submit-and-merge pattern for arena matches (eval.sbatch parts,
  warm measurement `MEASURE_WARM_MS=3000 MEASURE_REPEATS=10` since 4ffe3f6, `summary.json` per anchor).
  `cluster/full_run_watchdog.sh` counts `full_`, `sp_`, `mon_`, `am_` jobs toward the cap.
- **Data on this Mac.** `runs/study/full/hybrid1/anchors/it2/hybrid-it13/part{1,2}`: 400 games at 120 s, contender
  hybrid1 it2 vs `wave2-self` with it13 (= it0) under the same hybrid options, per-event game logs (FENs with pockets,
  drops, captures, clocks, per-think eval and nodes). Not here: the it2 ONNX, hybrid1 self-play chunks (PACE only).

## 2. Concepts and numbers fixed by this spec

| name | value | meaning |
|---|---|---|
| `k` (pace) | per team, per game | think time per move = seat clock / k (the arena's clock_k rule) |
| self-play pace range | log-uniform k in [80, 400] | each team draws its own k per game (WP3) |
| `K_REF` | 110 | the pace at which a self-play search gets `--nodes` (800) at a full clock |
| self-play nodes | `clamp(round(N x (K_REF / k) x s x clock_now / base), 200, 3200)` | N = jittered `--nodes`; s = think scale |
| self-play cost | `max(min_move_ms, round(clock_now / k x s)) + jitter/lag` | replaces the fixed per-seat move time |
| think scale `s` | `clamp(0.5 + 1.5 p, 0.5, 2.0)` | p = think_gain probability for the board (spec v2 B) |
| think head init | weights 0, bias `-ln 2` | p = 1/3, s = 1.0: a new head changes nothing at the start |
| league shares | self 0.5, past 0.25, fixed 0.25 | segment shares per iteration (WP3) |
| fixed opponents | it13 at k in {80, 110, 160, 250, 400} | frozen it13 under the same engine options |
| gate base clock | 60 s | both gate arms (plan allows 60-120 s; 60 s halves the cost) |
| gate games | 400 per arm per network (200 colour-swapped pairs), 4 parts | seed `7000 + part` for every network |
| gate equal arm | ours k 110 vs upstream-it13 k 110, think head off | strength at equal pace |
| gate free arm | ours k 160 + think head vs upstream-it13 k 110 | our time policy (160 = tune r1 best) |
| drop | `d < -0.02` and `d + 1.645 SE < 0` | per arm, paired over shared pairs |

Expected mean self-play cost per decision at the defaults: `E[K_REF/k]` for log-uniform k in [80, 400] is
`110 x (1/80 - 1/400) / ln 5 = 0.68`, times a mean clock fraction of about 0.7, so about half today's nodes per
decision (more games per GPU-hour, fewer nodes each). League segments record only the learner's decisions, so they
yield about half the records per game.

## 3. WP1: the real-clock two-score gate (`--gate2`)

### 3.1 Behaviour

With `--gate2`, after `match("generator")` (still played, unchanged: it is cheap, keeps the old curve and feeds
`launch_anchor_match.sh` the `wave2` dict), rl-loop runs a stage `gate2` and `promote()` decides from it instead of
the 100 ms score. Without `--gate2` nothing changes.

The gate scores two networks, the candidate `rl-itN/models/C.onnx` and the incumbent (the generator this iteration
played, sha in `stages/generator`), each in two arms against stock `upstream-it13`:

- `equal`: our wave2 contender (iteration's engine options, see 3.3) at `clock_k 110`, no think head, vs
  `upstream-it13` at 110. Base `GATE2_BASE_MS` (60000).
- `free`: our contender at `clock_k GATE2_FREE_K` (160) with `think_head true` (WP2/WP4; a network without the head
  falls back to the clock rule, counted as `think_head_missing`) vs `upstream-it13` at 110.

Both arms set `opp_sit_pace` to `GATE2_OPP_SIT` (`observed`, WP2's fix) on our side, for candidate and incumbent
alike. All arms: pool `arena/pools/wave2.json`, `SETUP=pool`, bughouse openings, rules v3, `MEASURE=1
MEASURE_NODES=800 WARM_KINDS="upstream wave2" MEASURE_WARM_MS=3000 MEASURE_REPEATS=10` (4ffe3f6, audit #4), equal
clocks, part k plays seed `GATE2_SEED_BASE + k` (7000; off the tune 5000, final 3000 and long-test 1000 schedules),
`GATE2_GAMES / 2 / GATE2_PARTS` pairs per part. Every network meets the same openings in the same colours, pair by
pair, so candidate and incumbent are compared paired.

Results are cached per network: `RUN_ROOT/gate2/nets/<sha16>/` (first 16 hex of the ONNX sha256). The incumbent's
results from an earlier iteration (or the launcher's pre-submitted it13 run, WP5) are reused, so a normal iteration
pays only the candidate's two arms (about 10.7 GPU-hours at 60 s, 0.8 GPU-min a game).

### 3.2 Decision

Per arm, over the pairs both networks finished (key `(part, pair_index)`): `x_p`, `y_p` = candidate and incumbent pair
scores in {0, 0.5, 1}; `d = mean(x - y)`, `SE = sd(x - y) / sqrt(n)` (n >= 2). Also each network's own score with a
95% normal range over pairs, and the clock-lead share `clock_teams.turns_ahead` (audit fix 3).

- `drop(arm) = d < -GATE2_DROP_MARGIN (0.02) and d + 1.645 x SE < 0` (one-sided 95%: clearly worse, by more than
  noise-level 0.02). At 200 pairs, SE is about 0.035, so a drop of about 0.06 (40 Elo) is detected.
- `complete(arm)`: at least `GATE2_MIN_PAIRS` (150) shared pairs.
- **Promote iff** both arms complete, neither drops, and `d_equal + d_free >= 0`.
- Not complete by the timeout (`--gate2-timeout-h`, 10): not promoted, logged as `incomplete`.
- The free arm sits near a 0.99 ceiling (tune r1); `d` there is driven by the few losses. That is intended: it
  guards the time-management win against regressions, while `equal` is where strength must not drop.

`stages/gate2.done` (JSON): `{"candidate": {"sha16", "equal": {...}, "free": {...}}, "incumbent": {...},
"arms": {"equal": {"pairs", "d", "se", "drop", "complete"}, "free": {...}}, "promoted": bool, "reason": str,
"settings": {...}}` where each network-arm entry is the merged `summary.json` (score, score_ci, elo, games, pairs,
turns_ahead, ms_per_node per part, think_head_missing). `promote.done` gains `"gate2": {...arms, reason}`; its
`score` field stays the 100 ms score (old readers keep working).

### 3.3 Cluster side: `cluster/gate2_match.sh` (main repo, WP1)

`bash cluster/gate2_match.sh RUN_ROOT LABEL ONNX` (short; settings come from the environment and the run's files).
Modelled on `launch_anchor_match.sh`: idempotent, safe to call every few minutes; each call submits what is missing
and merges what is finished; prints one line per action, nothing when there is nothing to do.

- Network dir `RUN_ROOT/gate2/nets/<sha16>/`: `model.onnx` (copy), `config.json` written once (label, sha256, engine
  ref, settings, contender kw per arm, opponent), `jobs.tsv`, per arm `<arm>/part<k>/` (eval rows, resumable),
  merged `<arm>/summary.json` and `<arm>/pairs.tsv` (`part, pair_index, opening_id, contender_team, score_pair,
  game_ids`), and `done` once both arms are merged. A net dir whose `config.json` settings differ from the current
  settings is refused with a message (never mixed).
- Contender options: `RUN_ROOT/gate2/wave2.json` (an `Options.as_dict()`, written by rl-loop on its first gate2 use, or
  by WP5's launcher via `python -m hivemind.gate2 wave2-json`), mapped onto arena kw exactly as
  `launch_anchor_match.sh` maps the gate's `wave2` dict (`WAVE2_FLAGS`, only values off the arena's defaults), plus
  `model`, `clock_k`, `opp_sit_pace`, and for `free` `think_head: true`.
- Jobs: `cluster/eval.sbatch`, `-J g2_<RUN>_<sha8>_<arm>_p<k>`, `-t ${GATE2_TIME:-05:00:00}`, `ENGINE_REF` the
  checkout's HEAD (with the same engine/ check as anchors), `MAX_TRIES` 3 then `GAVE_UP`. A failed `squeue` means merge
  only.
- Settings (env, defaults in 2): `GATE2_GAMES` 400, `GATE2_PARTS` 4, `GATE2_BASE_MS` 60000, `GATE2_FREE_K` 160,
  `GATE2_EQUAL_K` 110, `GATE2_SEED_BASE` 7000, `GATE2_OPP_SIT` observed, `GATE2_ARMS` "equal free", `GATE2_TIME`,
  `DRY_RUN`, `NO_SUBMIT`.

### 3.4 rl-loop side (fork, WP1)

Flags (all need `--gate2`; parser error otherwise): `--gate2`, `--gate2-script PATH` (default: `cluster/
gate2_match.sh` found above `PROJECT_ROOT`, like `search_balance_controller`), `--gate2-games` 400,
`--gate2-parts` 4, `--gate2-base-ms` 60000, `--gate2-free-k` 160, `--gate2-equal-k` 110, `--gate2-seed-base` 7000,
`--gate2-opp-sit` {observed, own_k} observed, `--gate2-drop-margin` 0.02, `--gate2-min-pairs` 150,
`--gate2-poll-s` 300, `--gate2-timeout-h` 10. No values with spaces (they travel in `WAVE2_EXTRA`).

Stage `gate2` (after `match("generator")`, before `promote`): write `RUN_ROOT/gate2/wave2.json` if absent (refuse
when it exists with other content), `RUN_ROOT/gate2/settings.json` likewise; then loop: run the script (env
`GATE2_*` from the flags, `cwd` = main repo root = the script's repo) for the incumbent and the candidate; if both net
dirs have `done`, decide (3.2); else `check_stop()`, sleep `--gate2-poll-s`, until the timeout (measured from
`stages/gate2.started`, written on first entry, so a resumed job keeps the clock). Script failures are logged and
retried each poll; never fatal. `RUN_ROOT` = `args.runs.parent` (the launcher layout `runs/full/<RUN>/runs`).

New module `W2:src/hivemind/gate2.py` (standard library only): `read_pairs(net_dir, arm) -> dict[(part, idx),
float]`, `decide(cand, inc, margin, min_pairs) -> dict` (3.2), `summary(net_dir) -> dict`, and CLI
`python -m hivemind.gate2 wave2-json [rl-loop wave 2 flags] > wave2.json` (parses with `wave2.add_arguments`, prints
`Options.as_dict()`), `python -m hivemind.gate2 decide CAND_DIR INC_DIR` (prints the decision JSON; WP5's status
script uses it).

### 3.5 Hooks WP1 writes into rl-loop for WP3 and WP4

rl-loop is WP1's file. WP1 adds these call sites; the callees live in WP3's `W2:src/hivemind/league.py` and WP4's
`W2:src/hivemind/think.py` with exactly these signatures (WP1 tests them with stubs/monkeypatch; import lazily so a
missing module only matters when its flag is on):

```python
# league.py (WP3)
def add_arguments(parser) -> None            # --league, --pace and their options (section 5.3)
def validate(args, parser) -> None           # parser.error(...) on bad combinations
def active(args) -> bool                     # --league or --pace given
def plan(args, runs: Path, n: int, generator: Path, fixed: Path, segment_games: int) -> dict
    # {"segments": {segNN: {"theta": str, "seed": int, "games": int, "engine_args": [str],
    #                       "opponent_model": str | None, "league": {...}}}, "summary": str}
def segment_summary(log_path: Path, spec: dict) -> dict   # learner W/D/L etc. from the segment's log
# think.py (WP4)
def add_arguments(parser) -> None            # --think-labels, --think-policy, --think-weight ... (section 6.4)
def validate(args, parser) -> None
def selfplay_args(args) -> list[str]         # engine args for non-val segments ({segment_dir} allowed)
def label_command(args, n: int, it: Path, generator: Path, engine: Path) -> list[str] | None
    # None when labels are off; else a command printing one JSON line, writing labels next to the chunks
def trainer_args(args) -> list[str]          # e.g. ["--think-weight", "0.1"]; [] when off
```

- `main()`: `league.add_arguments(parser)`, `think.add_arguments(parser)` after `wave2.add_arguments`; then
  `league.validate`, `think.validate`; `--league/--pace` with `--search-balance` is a parser error; `--league`/`--pace`
  need `--wave2` and a base mix.
- `iteration()`: before the generator marker, when `league.active(args)`: freeze `league.plan(...)` into
  `stages/league_plan.done` (same freezing rules as `balance_plan`: written once, before the generator marker, read by
  workers whatever their flags; a failed plan is a failsafe marker with plain segments and no league/pace args, logged).
- `planned_segments()`: the balance table, else the league table, else None. `next_claim`/`shared_segment`/the planned
  branch of `iteration` then work unchanged (claims carry `theta`, `engine_args`).
- `selfplay()`, non-val segments only: `extra += think.selfplay_args(args)` (when think is on) and the planned
  segment's `engine_args`; substitute `{segment_dir}` (as today) and `{opponent_model}` with a per-process copy of
  `spec["opponent_model"]` made by `selfplay.one_worker_copy` + `refresh_copy` (TensorRT caches are per file; a
  worker uses its own dir like `generator_sp1`). Validation never gets league, pace or think args (losses stay
  comparable). The segment marker gains `"league": league.segment_summary(log, spec)` for league segments and the
  CAP_FLAGS stock check is extended to the league/pace flags for `val`.
- After all self-play and before `train`: stage `think_labels` when `think.label_command(...)` is not None: run it
  (with `attempt`-style retries but **non-fatal**: after the retries, mark `{"failed": reason}` and train without
  labels), mark the JSON it printed.
- `train()`: `extra += think.trainer_args(args)` (when on).

### 3.6 Tests (WP1)

Fork `tests/test_gate2.py`: `decide` on hand-built pair tables (drop, no drop, incomplete, sum rule, shared-pair
intersection), `wave2-json` round trip. Fork `tests/test_rl_loop_gate2.py`: rl-loop with a fake gate script (writes
`done`, summaries and pairs.tsv; or never finishes -> timeout -> not promoted), promotion follows gate2, markers as in
3.2, defaults unchanged (command lines and markers without `--gate2`/`--league`/`--think-*` equal today's; extend the
existing defaults test), league/think hooks called with stub modules (league table claimed by a worker, val stock,
`{opponent_model}` substitution, labels stage non-fatal). Main `tests/cluster/test_gate2_match.py`: DRY_RUN with fake
Slurm (sbatch lines, job names, warm env, seeds), merge of fake part rows into summary/pairs, settings mismatch refused,
idempotence (second call submits nothing).

## 4. WP2: the opponent-sit confound and the hoarding diagnosis

### 4.1 Opponent pace from observation (`arena/players/upstream.py`)

New wave2 controller keywords (add to `WAVE2_CONTROLLER_KW`; validate values):

- `opp_sit_pace`: `"own_k"` (default: today's code, bit for bit), `"observed"`, `"fixed"`.
- `opp_clock_k`: float, used by `"fixed"`, and as the first fallback of `"observed"`.

`seat_decisions` gains a keyword `opp_k: dict[str, float] | None = None` (per opponent seat); when given it replaces
`clock_k` in `threshold()` for that seat (`own = clock_at_start / opp_k[seat]`). `None` keeps today's behaviour.

`"observed"`: the controller keeps, per opponent seat, its landed decisions as `(clock_at_handoff_ms, think_ms)` where
handoff = the landing that put the seat on turn (the board's previous landing) and think = its landing time minus
that; a decision during which `seat_decisions` marked the seat sitting is tagged `sat`. `k_obs(seat)` = median of
`clock_at_handoff / max(1, think)` over the last 12 untagged samples; with fewer than 3 untagged, the 75th
percentile of all its samples (sits read slow, so a high quantile skips them); with fewer than 3 samples, the
opponent team's pooled samples; then `opp_clock_k` if set; then our own `clock_k` (today's value). The estimate
includes lag and minimum-move time, as the observed pace does (the threshold compares like with like). Telemetry:
`opp_k_estimates` (last value per seat) and counts per fallback level, in the controller's summary like the hybrid
counters.

Self-play: nothing to change (section 1); say so in the module comment at `seat_decisions`.

### 4.2 Think-head call site (`upstream.py`, for WP4)

WP2 also owns the arena side of the learned time policy, written against WP4's `arena/think_policy.py`:

```python
# arena/think_policy.py (WP4)
QUICK_LOOK_NODES = 16
def scale(p: float) -> float                 # clamp(0.5 + 1.5 p, 0.5, 2.0)
def budget(clock_budget_nodes: int, p: float | None, min_nodes: int, max_nodes: int) -> int
    # p None (no head): clock_budget_nodes unchanged; else round(clock_budget_nodes x scale(p)) - QUICK_LOOK_NODES,
    # clamped to [min_nodes, max_nodes]
```

- Wave2 controller keyword `think_head` (default False). On: before a (non-cached) search, a quick look
  `search_clock(..., nodes=QUICK_LOOK_NODES)` with the same options; read `think_gain` for the deciding seat's board
  (`info string think_gain <pa> <pb>`, parsed with `arena.players.hivemind.parse_think_gain` into
  `SearchResult.think_gain`); main budget `think_policy.budget(...)`; both searches charged (nodes and time) as
  searched. No `think_gain` line: clock budget unchanged, `think_head_missing += 1`. Telemetry `think_looks`,
  `think_head_missing`, `think_gain_sum`, `think_scale_sum`.
- Tests that need `arena/think_policy.py` use `pytest.importorskip("arena.think_policy")` until WP4's first commit
  lands (WP4 commits it first).

### 4.3 Hoarding diagnosis (`tools/hoarding_diagnosis.py` -> `docs/hoarding_diagnosis.md`)

Question: does hybrid1 it2 hoard pieces (plan: +3.1 material, pocket 12.1 vs 8.6, mated on its own seat 63%), and is
that a training-target problem? Data: `runs/study/full/hybrid1/anchors/it2/hybrid-it13/part*` (it2 vs it0 = it13 in
the same games, same options), with `runs/study/wave2c/r4/long/l_hybrid_120` (it13 hybrid vs upstream, if present)
and `runs/study/tune/r1/base` as references. Standard library + numpy; may import `tools/study_games.py` helpers
(do not edit it). Per side (it2 seats vs it0 seats), wins and losses split, with counts and 95% ranges:

1. drop rate: drops / own moves made with a non-empty pocket; also by game phase (move number terciles).
2. pocket holding time: per piece that lands in a seat's pocket, ms and own moves until it is dropped (censored at
   game end; report median, mean, p90, share never dropped), per piece type.
3. pocket value at own decisions (P1 N3 B3 R5 Q9), and material + pocket balance over time.
4. engine eval on held-piece positions: mean search eval at own thinks binned by pocket value (0, 1-4, 5-9, 10+),
   against the game result (calibration: does it2 think holding is winning more than the results say).
5. how games end: mated on own seat / partner's seat / flag / sit-off, by side; drop-mates given and received.
6. Network targets (value and drop policy mass on held-piece positions): locally for it13 (`W2:engine/models/
   twin-s-clock-it13.onnx`) on positions sampled from these games if the planes can be built from the logs with the
   fork's Python encoder; it2's ONNX and hybrid1's self-play value targets are on PACE only, so the tool takes
   `--model` and the doc gives the exact PACE command (`runs/full/hybrid1/anchors/it2/C.onnx`) and marks those
   numbers "pending (PACE)". If the encoder cannot reproduce the engine's planes, say so and skip item 6 locally.

`docs/hoarding_diagnosis.md`: real numbers from these files, the method in a few lines, a verdict (hoarding yes/no,
how big, what it implies for the value targets or the league), and what is pending on PACE.

### 4.4 Tests (WP2)

`arena/tests/test_opp_pace.py`: `own_k` equals today's output on recorded observations (golden from the current
function), `observed` estimate on synthetic landings (sits skipped, fallbacks in order), `fixed`, kw validation,
think-head quick look with a fake process (budget scaled, missing head counted, both searches charged).
`arena/tests/test_hoarding_diagnosis.py`: metrics on two tiny synthetic game logs.

## 5. WP3: pace randomization and league opponents in self-play

### 5.1 Engine (fork C++)

New `selfplay` options (all off by default; off, the random streams and outputs are byte-identical to today):

- `--pace-k-min K --pace-k-max K` (0 = off): pace mode. Per game, each team draws `k` log-uniform in [min, max]
  (`sample_log_uniform` on the game's stream, drawn after everything drawn today so off-mode streams are unchanged).
  Overrides `--seat-move-times` for cost. `--pace-ref-k` (110), `--pace-min-nodes` (200), `--pace-max-nodes` (3200).
- Cost and budget per decision (section 2): `SeatClockGame::plan_cost` gains a pace path: cost from the seat's
  current clock, the team's k and the think scale `s` (`moveTimeMs_[seat]` is set to the unjittered `clock / k x s`
  so pass caps follow the pace); nodes `clamp(round(N x (K_REF/k) x s x clock/base), min, max)` with N the jittered
  base, then the playout cap as today. Both deciding seats of a joint decision use the search's `s`.
- `--opponent-model PATH`: league game. The opponent team's agent searches with a second engine set built from PATH
  (same device, batch and backend as the main one; one per parallel game, like the main engines). The learner team is
  drawn per game (`--learner-team random`, or white/black for tests). Raw-policy openings use the learner's network for
  both teams. **Only the learner team's searched decisions are written** (HDST records, `--record-state-output` lines,
  seat rows); outcomes, WDL and moves-left are from the learner's side as for any record. `--opponent-k K` fixes the
  opponent team's k (pace mode); 0 draws it like the learner's.
- Think policy (WP4 provides the functions, 6.3): `--think-policy true` (default false). Before each searched
  decision of a team whose engine has a think head: `p` per deciding board from `evaluate_think_gain`, `s =
  think_budget scale` (max over the deciding boards); without a head `s = 1` and a `think_missing` counter. In pace
  mode `s` scales cost and nodes; without pace mode it scales nodes only.
- Per game summary line and the PGN/seat rows gain `k_white_team`, `k_black_team`, `learner_team` (league), mean `s`.
  `selfplay complete` keeps its format. The cap counts / progress / checkpoint logic must still resume a segment.

Hook to WP4 (compile guard so WP3 builds before WP4 lands): in `selfplay.cc`, `#if __has_include("tools/
think_policy.h")` include it and call the functions; else `--think-policy true` is refused with a clear error.
`engine/tests/CMakeLists.txt` (WP3's) lists `test_league_pace.cc` and, guarded by
`if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/test_think_policy.cc)`, WP4's `test_think_policy.cc`.

### 5.2 Engine tests (WP3, `engine/tests/test_league_pace.cc`)

Off-mode identity (same seed: same start, costs, budgets as before the change); k within range and log-uniform-ish;
cost and nodes follow the formula at full and half clock; clamps; pass cap follows the pace; league: only learner
records written, learner team split near 50/50 over seeds, opponent engine used for the opponent's searches (a fake
engine or a counter), `--opponent-k`. A smoke run of `hivemind selfplay` on the Mac (ORT, 4 games, 50 nodes, it13 as
both model and opponent) writing chunks the Python reader accepts.

### 5.3 Python (fork, `W2:src/hivemind/league.py`)

Flags: `--pace` (on: pace args on every non-val segment), `--pace-k-range 80:400`, `--pace-ref-k 110`,
`--pace-nodes 200:3200`; `--league` (on: league segments), `--league-shares 0.5,0.25,0.25` (self, past, fixed),
`--league-past 3` (last promoted generators, excluding the current), `--league-fixed-k 80,110,160,250,400`,
`--league-fixed-model PATH` (default: the lineage's fixed baseline, `<lineage>-it0.onnx` = it13), `--league-pfsp`
{on, off} (on).

`plan(...)`: `S = ceil(games / segment_games)` segments `seg00..`, seed `1000 n + k` (as today), games as today.
Allocation: `round` the shares onto whole segments (self gets the remainder; at least one self segment). Past: the
promoted iterations' `rl-itM/models/C.onnx` (`promote.done` promoted true) newest first, up to `--league-past`, not
the current generator's sha; none -> their share goes to fixed. Fixed: cycle the k list. PFSP (AlphaStar "hard"):
with the previous iteration's league segment summaries, each opponent's weight is `max(0.05, 1 - w)^2` (w = learner
score vs it) when it has >= 200 learner games, else 1; segments within a share are dealt in proportion. Each league
segment's `engine_args`: pace args, `--opponent-model {opponent_model}`, `--opponent-k K` for fixed, and
`opponent_model` = the source path; `theta` = `self`, `past:itM`, `fixed:k110`. Self segments get pace args only.
`segment_summary` parses the segment's game lines (learner W/D/L, mean k, think scale) for the marker and PFSP.

Fork tests `tests/test_league.py`: allocation and seeds, iteration 1 fallback (no past), PFSP weights, args per
segment, validation errors, summary parsing.

## 6. WP4: the think_gain head and the learned time policy

### 6.1 Network and trainer (fork Python)

- `twin_board.py`: optional `_ThinkGainHead` (global pool of the shared board features, one hidden layer of 64, two
  logits [A, B]), last layer weights 0 and bias `-ln 2` (p = 1/3, s = 1.0: section 2); `TwinBoardNet(..., think_gain=False)`;
  forward appends it last when present; `load_twin_board_checkpoint(..., think_gain=None)` adds a fresh head to a
  checkpoint without one (as `wide` adds arrivals; only `think_gain_head.*` may be missing). ONNX output name
  `think_gain`, shape [B, 2], logits. Parity: with the fresh head every other output equals the source network's.
- `distill/data.py`: a data dir may hold `think_labels.npz` next to its HDST chunks (`record_index`, `label` [N, 2]
  float, `mask` [N, 2] bool; record_index = position among the segment's records in chunk order). Batches gain
  `think_gain` [2] and `has_think_gain` [2] (masked when absent). `distill/train.py`: term `think_gain`, masked BCE on
  logits, `--think-weight` (default 0: off; rl-loop passes 0.1), the head trains at the heads' rate (`HEADS`).
  Training a checkpoint without the head and `--think-weight > 0` adds the head (fresh). Export keeps the output.

### 6.2 Labels (`W2:src/hivemind/think.py`, `python -m hivemind.think labels`)

Uses the reanalyse machinery, no new engine command: for each non-val segment with a `states.jsonl` (written because
`think.selfplay_args` adds `--record-state-output {segment_dir}/states.jsonl` when labels are on), pick a hashed
fraction (`--think-label-fraction`, 0.02) of records, run `hivemind reanalyse` at the recorded budget and at
`--think-factor` (4) x it, same generator and wave 2 options, no noise; label per board on turn = 1 if the two
searches' most-visited entries differ (a pass counts), else 0; boards not on turn masked. A repeat search at the base
budget on 5% of them gives `noise_rate`. Writes `think_labels.npz` into the segment's training data dir and prints
one JSON line: `{records, labelled, rate_a, rate_b, noise_rate, gpu_s}`. Cost at the defaults: about 29k labels per
10k-game iteration, (1 + 4) x the recorded budget each, about 1.5 GPU-hours.
Under `--think-policy` the "recorded budget" is the state line's `think_ref_nodes` (the budget at think scale 1: pace,
clock and playout cap as for `target_nodes`), not `target_nodes`, which already carries the generator head's scale:
otherwise a position the head rated high would be labelled 2B against 8B and one rated low 0.5B against 2B, and the
head's targets would be pulled back toward its own mean (review of 2026-11-27). Without `--think-policy` the state
lines and labels are as before.

### 6.3 Engine (fork C++)

- `nn/engine.h`, `engine.cc` (TensorRT), `engine_ort.cc` (ONNX Runtime): bind an output named `think_gain` with 2
  values per row when present (copy the arrivals path); `hasThinkGain()`; ignore otherwise. A network without it
  behaves exactly as today.
- New header-only `tools/think_policy.h` (commit it FIRST, with this interface, so WP3 can compile against it):

```cpp
struct ThinkPolicy { double minScale = 0.5, maxScale = 2.0; };
inline double think_scale(const ThinkPolicy&, float p);   // clamp(0.5 + 1.5 p, min, max)
// One network evaluation of `board` for `team` on `clock`: think_gain probabilities (sigmoid) per board.
// False when the engine's network has no think_gain output.
bool evaluate_think_gain(Engine& engine, Board& board, Stockfish::Color team, const TimeState& clock,
                         std::array<float, 2>& out);
```

- UCI (`search/agent.cc` or where the search's final `info` lines are printed): after each search, when the network
  has the head, print `info string think_gain <pa> <pb>` (probabilities, 4 decimals) for the search's root (one extra
  network evaluation, not counted in nodes). Nothing printed otherwise.
- `cluster/think_gain_check.sbatch` (main repo): on a GPU node, a headed copy of it13 (`python -m hivemind.think
  add-head`), the engine on TensorRT prints `think_gain` on 50 positions and matches ONNX Runtime in Python within
  fp16 tolerance; every other output unchanged vs it13. WP5's launcher submits it before the chain.

### 6.4 rl-loop hooks (callees for WP1, `think.py`)

Flags: `--think-labels` (labels stage + states dump), `--think-label-fraction 0.02`, `--think-factor 4`,
`--think-policy` (self-play budgets scaled by the generator's head: engine `--think-policy true`), `--think-weight
0.1` (trainer; only with `--think-labels`). `selfplay_args`, `label_command`, `trainer_args` as in 3.5.
`python -m hivemind.think add-head --checkpoint X --out DIR` writes a headed checkpoint and ONNX (parity checked).

### 6.5 Arena (main repo `arena/think_policy.py`)

The pure functions of 4.2 (WP2 writes the call site). Commit first. Tests `arena/tests/test_think_policy.py`.

### 6.6 Tests (WP4)

Fork: head shapes, zero-start parity (all outputs but think_gain identical, p = 1/3), loader adds the head,
masked BCE (no labels -> no loss, gradient only on labelled boards), npz attach by record index, labels on a fake
reanalyse (differs/same/pass/masked/noise), `add-head` round trip, ONNX export has the output. Engine
`test_think_policy.cc`: scale clamps, ORT engine reads the output of a tiny headed ONNX and reports none for it13.

## 7. WP5: pilot launcher, status, watchdog hook, phase 0

### 7.1 `cluster/launch_league_pilot.sh`

Short command: `RUN=league1 bash cluster/launch_league_pilot.sh` (`DRY_RUN=1` for checks). It wraps
`launch_full_run.sh` with the pilot's settings, after preflight:

- Pilot: `CONFIG=hybrid` (twin-s it13 narrow, `--durative-tree false --sit-gate true --hybrid`), `WIDE=0`, `ITERS=2`,
  `GAMES_PER_ITER=10000`, `WORKERS=3`, `SEGMENT_GAMES=500`, `GPU_HOURS=100`, `CHAIN=2`, `WAVE2_EXTRA` = the config's
  flags plus `--pace --league --think-labels --think-policy --think-weight 0.1 --gate2` (each switch can be dropped
  with `PACE=0 LEAGUE=0 THINK=0 GATE2=0`; `EXTRA` appends). Recorded in `launch_settings.txt` as usual.
- Engine pin: `cluster/wave2_engine.ref` must contain `d5802ee` as an ancestor (`git merge-base --is-ancestor`), else
  refuse. The pinned engines must list the new options in `--help` (`--pace-k-min`, `--opponent-model`,
  `--think-policy` for the switches that are on) and rl-loop `--help` must list `--gate2`, `--league`,
  `--think-labels`: else print the build command (the readiness build, `cluster/build.sh` for build-ninja and
  `cluster/build_sp1.sh` for build-sp1, as `launch_wave2c.sh` does) and stop; `BUILD=1` submits the build job and
  makes the chain depend on it (`afterok`). The PACE W2 checkout is at 0839774: the build is expected.
- With `THINK=1`: submit `cluster/think_gain_check.sbatch` first; the chain depends on it (`afterok`).
- With `GATE2=1`: write `runs/full/<RUN>/gate2/wave2.json` (`python -m hivemind.gate2 wave2-json <flags>`) and
  pre-submit the incumbent (it13 lineage file `models/twin-s-clock-w2.onnx` after init, or the it13 ONNX itself:
  same sha) through `cluster/gate2_match.sh` with the pilot's `GATE2_*` settings, so iteration 1 waits only for the
  candidate.
- Cost printed up front: about 41 GPU-hours for iteration 1 (self-play ~13, val 0.7, labels 1.5, train 1.7, 100 ms
  gate 1.2, gate2 candidate 10.7 + incumbent 10.7, main job idle while the gate runs ~1.5) and about 30 for iteration
  2; cap 100.

### 7.2 `cluster/league_pilot_status.py RUN_ROOT [--json]` (standard library)

Per iteration: stage markers present, league plan (segments by theta, PFSP weights), self-play games per kind, mean
k and think scale, think label rates and noise, train steps, 100 ms gate score, gate2 per network and arm (score,
range, pairs, turns_ahead, d, SE, drop) and the decision; GPU-hours spent (sacct over `full_`, `sp_`, `g2_`, `mon_`,
`am_` jobs of the run when `sacct` exists, else "n/a off-cluster") vs the cap.

### 7.3 Watchdog hook (`cluster/full_run_watchdog.sh`)

Count `g2_<RUN>_*` (and `tg_<RUN>` think check, if named so) jobs toward `GPU_HOURS`; at the cap also cancel pending
`g2_<RUN>_*` jobs; with `STOP_WORKERS` set, rl-loop's gate2 stage stops submitting (the script honours
`RUN_ROOT/STOP_WORKERS`, WP1's script reads it: WP1 adds that check, WP5 creates the file as today). Defaults
unchanged for runs without gate2 jobs.

### 7.4 `cluster/phase0_chk1.sh` (runs on the Mac)

`bash cluster/phase0_chk1.sh`: rsync `pace:/storage/project/ps-bughouse-0/jwalzer3/talisman-bughouse-engine/runs/tune/
chk1/` to `runs/study/tune/chk1/` (results.jsonl, game_*.jsonl, summary.json, jobs.tsv, candidate json, round.json,
specs.jsonl, ms_per_node.json; no engines, ONNX or big logs), then `python3 cluster/tune_status.py
runs/study/tune/chk1 > runs/study/tune/chk1/status.txt` (cat it), then `python3 tools/study_games.py runs/study`.
`SRC=<dir>` replaces the rsync source (tests, or a copy already here); `REMOTE=pace` host alias; `DRY_RUN=1` prints
the rsync. chk1's rows (sym160, rev_opp160) may not have a `round.json`: if `tune_status.py` needs one, the script
writes a minimal one only into the local copy, or reads each row on its own; it never edits `tune_status.py`.

### 7.5 Tests (WP5)

`tests/cluster/test_league_pilot.py` (fake Slurm): DRY_RUN prints the launch_full_run call with the pilot settings,
the think check and incumbent gate2 submissions, dependencies; refuses a pin without d5802ee; switches off; missing
options -> build message. `tests/cluster/test_phase0_chk1.py`: with `SRC` a tiny fake round, status.txt and study.json
are written. Watchdog: a fake sacct with `g2_` jobs counts toward the cap.

## 8. Acceptance (done means)

1. Everything new off: existing test suites pass; the defaults tests show unchanged rl-loop commands/markers,
   self-play outputs (same seed, byte-identical HDST chunks on a tiny run), arena decisions (`own_k`).
2. WP1: a local rl-loop smoke run with a fake gate script promotes or not by the rule; `gate2_match.sh` DRY_RUN prints
   correct sbatch lines.
3. WP2: `docs/hoarding_diagnosis.md` with real numbers from the anchors games; opp-pace tests pass.
4. WP3: a Mac self-play run with `--pace-k-min 80 --pace-k-max 400 --opponent-model it13` writes learner-only chunks
   the trainer reads.
5. WP4: a headed it13 trains a few steps on fake labels; the Mac engine prints `think_gain` with the headed ONNX and
   nothing with it13; the arena wave2 controller with `think_head` scales budgets (with WP2).
6. WP5: `DRY_RUN=1 RUN=league1 bash cluster/launch_league_pilot.sh` against the fake Slurm prints the whole launch;
   `phase0_chk1.sh` works on a local fake round.
7. Integration (orchestrator, after all WPs): bump `cluster/wave2_engine.ref` to the fork HEAD containing WP1/3/4,
   run all suites, the dry-run launcher, then on PACE: build, think check, launch.

## 9. File ownership

| WP | owns (create or edit) |
|---|---|
| WP1 gate | `W2:src/hivemind/cli/rl_loop.py`, `W2:src/hivemind/gate2.py`, `W2:tests/test_gate2.py`, `W2:tests/test_rl_loop_gate2.py`, `cluster/gate2_match.sh`, `tests/cluster/test_gate2_match.py` |
| WP2 confound + diagnosis | `arena/players/upstream.py`, `arena/tests/test_opp_pace.py`, `tools/hoarding_diagnosis.py`, `arena/tests/test_hoarding_diagnosis.py`, `docs/hoarding_diagnosis.md` |
| WP3 pace + league | `W2:engine/src/tools/selfplay.cc`, `selfplay.h`, `seat_clock_game.h`, `clock_sampling.h`, `W2:engine/src/main.cc`, `W2:engine/CMakeLists.txt`, `W2:engine/tests/CMakeLists.txt`, `W2:engine/tests/test_league_pace.cc`, `W2:src/hivemind/league.py`, `W2:src/hivemind/cli/selfplay.py` (if needed), `W2:tests/test_league.py` |
| WP4 think policy | `W2:engine/src/nn/engine.h`, `engine.cc`, `engine_ort.cc`, `W2:engine/src/search/agent.cc` (and `interface/uci.cc` if the print lives there), `W2:engine/src/tools/think_policy.h`, `W2:engine/tests/test_think_policy.cc`, `W2:src/hivemind/architectures/twin_board.py`, `W2:src/hivemind/distill/train.py`, `W2:src/hivemind/distill/data.py`, `W2:src/hivemind/think.py`, `W2:tests/test_think.py`, `arena/think_policy.py`, `arena/tests/test_think_policy.py`, `cluster/think_gain_check.sbatch` |
| WP5 launcher | `cluster/launch_league_pilot.sh`, `cluster/league_pilot_status.py`, `cluster/full_run_watchdog.sh`, `cluster/phase0_chk1.sh`, `tests/cluster/test_league_pilot.py`, `tests/cluster/test_phase0_chk1.py`, `cluster/README.md` (a "League pilot" section) |

Shared-file hooks: rl-loop (WP1) calls `league.py` (WP3) and `think.py` (WP4), 3.5. `selfplay.cc` (WP3) calls
`think_policy.h` (WP4) behind `__has_include`, 5.1. `upstream.py` (WP2) calls `arena/think_policy.py` (WP4), 4.2.
`engine/tests/CMakeLists.txt` (WP3) lists WP4's test behind `if(EXISTS)`. `gate2_match.sh` (WP1) honours
`STOP_WORKERS` that the watchdog (WP5) writes. Nobody edits `cluster/wave2_engine.ref`, `launch_full_run.sh`,
`wave2_iterate.sbatch`, `arena/pool.py`, `tools/study_games.py` or `cluster/tune_status.py`; if a WP finds it must,
it stops and reports instead.

## 10. Open decisions (defaults chosen; the user may change them)

- Free-arm pace 160 (tune r1's best) vs 110 (pure learned allocation at it13's pace). 160 keeps the headline row;
  the promotion comparison is fair either way (both networks get the same setting).
- Gate base 60 s (plan allows 60-120). 120 s doubles the gate cost to about 21 GPU-hours a network.
- Self-play nodes now follow pace (about half today's mean). If policy targets suffer, raise `--nodes` (N_ref).
- League shares 0.5/0.25/0.25 and PFSP hard; AlphaStar's main agent used about 35/50/15.
