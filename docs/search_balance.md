# Search balance: deep search targets vs more self-play games

Status: FINAL SPEC (design only). Nothing here is built yet. Objective: Elo per GPU-hour.
Fork paths are relative to `third_party/wave2-hivemind/`. Line numbers are as of fork ff835c8e.

## 0. Verdict on the two candidate designs

| Criterion | A: Thompson sampling + GP | B: compass search + paired Z-test | Notes |
|---|---|---|---|
| 1. Expected Elo/GPU-h net of overhead | 3 | 3 | Same arms, same ~8-9% overhead, same compute-matched mini-trains. A explores more widely (better on long runs); B moves along neighbours (better on 4-10 iteration runs). |
| 2. Robustness to noise | 3 | 4 | A depends on a calibrated β (probe→Elo) and GP hyperparameters that 4 iterations cannot pin down. B's Z ≥ 2 over ≥ 2 observations, plus a mini-gate veto, is transparent and hard to fool. |
| 3. Implementation risk here | 2 | 4 | A needs a GP, a hierarchical β regression and a 5-D kernel. B is about 300 lines of plain Python. |
| 4. Resumability/determinism | 4 | 4 | Both are pure functions of recorded files. A's GP refit in float64 is harder to keep byte-identical across numpy/BLAS versions. |

**Winner: B (compass search).** Ideas taken from A:
- the cost model with a raw-policy term and measured η/γ, with planning always on realised cost;
- explicit training-step accounting;
- the overhead ledger and its drop order;
- `failsafe` and `maintenance` modes;
- a forced revert test after a promotion;
- λ arms judged by mini-trains alone, because they need no extra self-play;
- the mini-gate only when it can change a decision;
- evidence decay over time, in place of the GP's temporal kernel;
- the offline sweep used as a prior.

Both designs fixed the scout's cost error. **One visit costs 4.24e-5 GPU-s and one 800-node search about 0.034 GPU-s, not 0.48.**

## 1. Knobs and grid

θ = (f, k, b, λ, src). The incumbent at start is θ0 = (0, –, 800, 0, cap), which is exactly today's recipe.

| Knob | Meaning | Grid | Default |
|---|---|---|---|
| f | Share of searched decisions that get a deep search | {0, 1/16, 1/8, 1/4, 1/2} | 0 |
| k | Deep budget = k·800 nodes. The deep search also picks the move played (in-game playout cap). | {2, 4, 8} (8 only after the η bench passes on 24 GB with 6 parallel games) | – |
| b | Budget of the other decisions. b = 800 keeps their policy targets. b < 800 writes them with **empty policy lists** (WDL and moves-left only, KataGo-style) and Dirichlet ε = 0. | {800, 400, 200} | 800 |
| λ | Q-vs-outcome value blend in the trainer, on policy-carrying records only. Trainer-only, so it costs no self-play. | {0, 0.25, 0.5} | 0 |
| src | Where deep targets come from | {cap, reanalyse}. reanalyse is masked until the §6 fidelity test passes. | cap |

- Exploration order of directions: f+, k+, f−, k−, b−, b+, λ+, λ−, then src (when unmasked).
- Validation self-play always runs at θ0 (no cap flags), so heldout losses stay comparable across iterations.

## 2. Cost model (measured numbers)

**Measured** (hybrid1 it1 seg00, ab2/narrow it1, RTX6000, 6 parallel games, sp1, TRT fp16):

- 768-776 games/GPU-h at 800 nodes.
- 145.3 searched + 7.5 raw-policy decisions per game.
- 760.8 mean root visits per search.
- 84.8M visits/GPU-h, i.e. v = 4.24e-5 GPU-s/visit.
- Per iteration (≈16.5 GPU-h):

| Stage | GPU-h |
|---|---|
| Self-play, 10k games | 12.9 |
| Val, 500 games | 0.66 |
| Train, about 2114 steps at 2.9 s/step | 1.72 |
| Gate, 400 games at 100 ms | 1.22 |

**Relative cost per game:**

c(θ) = (1−f)·(b/800)·γ(b) + f·k·γ(k·800) + ρ

- ρ ≈ 0.01 is the raw-policy opening and per-game overhead.
- γ(n) is the slowdown factor at n nodes (nps(800)/nps(n)). It starts at 1.0.
- After each iteration, γ is replaced by the realised value from the challenger segment:
  - γ̂ = (segment GPU-seconds / Σ seat_clocks.tsv nodes) / v_inc, where v_inc is the same ratio for the twin incumbent segment.
  - It is kept as a running median per n in the controller state.
- The controller plans with c computed from γ̂, never from the nominal value alone.

**Self-play is budgeted in GPU-hours, not games:**

- Each train segment gets the GPU time of `segment_games` stock games: S_seg = segment_games / 770 GPU-h (2000 games ≈ 2.6 GPU-h).
- A segment playing θ gets games(θ) = round(segment_games / c(θ)).

Examples:

| θ | c | Games per 2000-game segment |
|---|---|---|
| f=1/8, k=4, b=800 | 1.385 | 1444 |
| f=1/4, k=2, b=800 | 1.26 | 1587 |
| f=1/8, k=8, b=800 | 1.885 | 1061 |
| f=1/4, k=4, b=200 | 1.198 | 1669 |
| f=1/2, k=4, b=200 | 2.135 | 937 |

**Training steps.** train.plan counts every record, so fast (masked) records would inflate steps.

- Rule (recorded in the balance file as `steps_rule: "policy_positions"`): steps = plan(policy_positions_new, policy_positions_replay, new_passes, replay_passes, batch).
- policy_positions = records − masked records. The masked count comes from the engine's per-segment `cap_counts.json` (§5).
- At θ0 this equals today's count exactly.

**Reanalyse cost** (src = reanalyse) is extra on top of self-play: R = f_r · k · records · 800 · v / η_batch.

- 1.44M records at f_r = 1/8, k = 4 gives ≈ 6.8/η_batch GPU-h.
- It is paid by shrinking self-play games at equal total GPU-h, so it is only competitive if offline batching beats self-play nps by η_batch ≳ 1.5 (unmeasured).

**Overhead per iteration** (§4 steps 4-6):

| Item | GPU-h | Frequency |
|---|---|---|
| Mini-trains, 4 × 200 steps × 2.9 s | 0.65 | Every iteration with a challenger |
| Probe deep-target refresh | 0.55 | Every 2nd iteration, so ≈ 0.28 amortised |
| Probe scoring, 4 nets × ~3k positions | 0.05 | |
| Mini-gate, 100 games at 100 ms | 0.3 | Conditional |
| **Typical** | **≈ 1.0-1.3 (6-8% of 16.5)** | |

- Hard cap: cumulative overhead ≤ 10% of the run's GPU-h.
- Drop order when the cap binds: mini-gate (the challenger cannot be promoted that iteration), then the probe refresh (stale targets are reused), then the second seed pair, then exploration entirely (mode `frozen`).
- The challenger segment's own GPU time is NOT overhead, because its data trains the main net. Its regret is accounted in the simulator only.

## 3. Signals and probe deep targets

### Primary signal: the paired, compute-matched mini-train score difference d

- Both nets start from the same init.
- The data cost the same self-play GPU time, comes from the same generator net and from common openings.
- The only difference between the two datasets is θ.

### Probe set for scoring: `runs/full/<RUN>/balance/probe/`

- **`deep_vV.dst`**: faithful deep targets made by a fixed-seed *probe-mode self-play*. These are real seat-clock games, so the roots are exact.
  - Settings: 150 games at f = 1/8, k = 16 (12,800 nodes), b = 800, Dirichlet ε = 0 on deep decisions, `--fast-policy-mask true`. Only deep records keep a policy.
  - Played by the current best net. Output: about 2,700 deep positions, plus outcome WDL on every record, at about 0.55 GPU-h.
  - Seed: 1000·N + 998, which leaves 999 to val.
  - Sidecar `deep_vV.json` holds {net sha256, model path, seed, settings, record count}.
- **Leakage guard:** probe depth (16·800) is always ≥ 2 × the largest arm k. The score always includes the outcome-based WDL term. A challenger can never be promoted on policy fit alone (§4 mini-gate veto).
- **Staleness:** the probe is regenerated every 2nd iteration, or when the best net's sha changes and the last refresh is ≥ 2 iterations old. The previous version is also scored for one iteration, so versions can be chained: the evidence ledger stores d together with its probe version.
- **Second WDL source:** the parallel swarm's frozen probe set `runs/full/<RUN>/probe/` (via probe_net.py heldout losses) supplies a WDL log-loss and calibration term on positions frozen across the run. It goes into the score as a secondary term when present (weight α₂ = 0.5·α). Its sha256 is recorded.

### Score of a mini-net

score = CE_policy(deep_vV) + α · WDL_logloss(deep_vV ∪ swarm probe) + 0.1 · moves_left_loss

- α = 1.0 is fixed for the whole run and written in it1's plan.
- Lower is better. d = score(incumbent mini) − score(challenger mini). d > 0 means the challenger is better.

### Guard metrics (logged per arm, never decisive)

- From game lines and seat_clocks.tsv: hybrid triggered/played per game, sit rate, flag rate, decisions per game, mean visits.
- A shift of more than 20% versus the twin sets `review: true` in the plan file and the panel.

### Audit signals (main net, never used to pick arms)

- Main gate result and the swarm's `--gate-sprt` decision, 120 s anchors, and metrics/itN.json. These are used only for the revert rule.
- **Calibration of δ_min (probe units → Elo).** Regress main-gate Elo gains on main-net probe-score gains over finished iterations (ab2/narrow plus the run so far). δ_min is the probe gain worth about +5 Elo per iteration at equal GPU.
- Until there are 3 calibration points, δ_min = 1.5·σ̂, with σ̂ the pooled seed-pair spread.

## 4. Per-iteration algorithm

### Controller state

The state is carried inside balance/itN.json (§5), so there is no mutable side file:

```
theta_star, step (grid indices, starts 2), dir_idx, tabu {theta: until_iter}, evidence {theta_key: [obs...]},
gamma {nodes: [samples]}, overhead {cum_gpu_h, run_gpu_h}, mode, last_promotion {from, to, iter}, revert_test
```

### Plan before iteration N's self-play

```
def plan(N):                                  # pure function of files; hivemind balance plan
    S = load(balance/it{N-1}.json).state  if N > first else init_state(prior=balance/prior.json)
    if missing/corrupt inputs: return write_plan(N, S, challenger=None, mode="failsafe")   # run theta_star
    if S.revert_test: chal = S.last_promotion.from                    # forced revert test (from A)
    elif S.mode == "maintenance" and N % 3 != 0: chal = None
    elif overhead_projection(S) > 0.10: chal = None; S.mode = "frozen"
    else: chal = propose(S)                                           # next legal neighbour, see below
    segs = {f"seg{k:02d}": S.theta_star for k in train_segments}
    if chal: segs[X] = chal                       # X = last train segment; twin T = seg00
    games = {s: round(segment_games / c(theta, S.gamma)) for s, theta in segs.items()}
    write_atomic(balance/it{N}.plan.json, {...segs, games, engine_args per seg, val_args=STOCK,
                 trainer_args(theta_star.lambda), steps_rule, probe_refresh: bool, inputs: shas})

def propose(S):
    for i in range(len(DIRS)):
        d = DIRS[(S.dir_idx + i) % len(DIRS)]
        cand = move(S.theta_star, d, S.step)
        if on_grid(cand) and not tabu(cand, N) and fits_bench(cand) and available(cand.src):
            S.dir_idx = (S.dir_idx + i) % len(DIRS); return cand
    if S.step > 1: S.step //= 2; S.dir_idx = 0; return propose(S)
    S.mode = "maintenance"; return rotate_neighbour(S, N)            # deterministic ±1 re-test
```

- First challenger from θ0 with step 2: f 0 → 1/8 at k = 4.
- λ-only challengers (same f, k, b) need no challenger segment. They are judged on the twin's data with two trainer settings, so their only cost is the mini-trains.

### Self-play

- Train segments use their plan's engine args and game counts.
- **Common random numbers:** segment X uses the **same seed as its twin T** (1000·N + 0). The cap coin never consumes randomEngine, so both arms draw identical opening lengths and raw-policy openings for their shared game prefix.
- Val runs stock.

### Main training

- Data: all segments, including X, at steps_rule.
- The main net's effective recipe (the record-weighted mix of θ's) is logged as `main_mix`.

### Balance stages (after `train`, before `match`)

```
balance_minitrain:   for arm in (inc, chal), for s in (0, 1):                 # 4 runs, ~0.65 GPU-h
    hivemind distill-train --init <iteration-start generator checkpoint> \
        --search-data <arm data: X for chal; T's first-cost-matched games for inc = all of T> \
        --replay-data <same fixed old slice for all four, 50%> --replay-fraction 0.5 \
        --steps 200 --seed 1000*N + 10*s + 7 (same for both arms) --lr <loop lr> \
        --search-value-weight 0 --search-wdl-weight 1 --search-moves-left-weight 0.1 \
        [--search-q-blend λ] --export last --eval-every 200 --out balance/itN/mini/{arm}_s{s}
    (for a λ challenger: data = T for both arms, only --search-q-blend differs)
balance_score:  hivemind balance score --nets balance/itN/mini/*/last.pt --probe balance/probe/deep_vV.dst \
        [--swarm-probe runs/full/<RUN>/probe] -> balance/itN/score.json
        d_s = score(inc_s) - score(chal_s); d = mean(d_s); sigma_n = |d_0 - d_1| / sqrt(2)
balance_probe (when plan.probe_refresh): probe-mode self-play with the CURRENT best net -> deep_v{V+1}.dst
```

- Compute matching is built in: X and T had the same GPU time, and X just holds fewer games.
- The mini-train data spans whole segments, so the incumbent mini sees more positions. That is the point: it is the equal-GPU comparison.

### Decision (after `match`/`promote`, so the gate and the swarm's SPRT are known)

```
def decide(N):                                     # hivemind balance decide
    S = plan_state(N); E = S.evidence[key(chal)]
    E.append({iter:N, d, sigma_n, probe_v, weight:1.0})
    for o in all evidence: o.weight = 0.5 if N - o.iter > 3 else 1.0           # drift decay
    mean, se = weighted_mean_and_pooled_se(E)       # se floor: sigma_hat_pooled / sqrt(n_obs)
    Z = mean / se
    eligible = (Z >= 2.0 and mean >= delta_min and len(E) >= 2) or (len(E) == 1 and Z >= 3.0)
    if eligible and overhead allows: run balance_minigate (100 games, 50 openings, seats swapped,
        100 ms, mini chal_s0 vs inc_s0); veto if one-sided p < 0.10 that chal is worse
    if eligible and not veto and not sprt_regressed(N):
        promote: S.last_promotion = {from: theta_star, to: chal, iter: N}; theta_star = chal;
                 S.revert_test = True (next iteration re-tests the old incumbent as challenger)
    elif Z <= -1.0 or len(E) >= 3:
        reject: tabu[chal] = N + 3; dir_idx += 1; a full failed cycle of DIRS halves step
    else: continue (same challenger next iteration)
    revert: if S.revert_test result has P(old better) i.e. Z_old >= 1.5, OR the next two main gates
            after a promotion are both negative, OR swarm alert "probe regression" -> theta_star = from,
            tabu[to] = N + 3, freeze promotions for 2 iterations (mode "frozen")
    converged: step == 1 and every legal neighbour rejected -> mode "maintenance" (one ±1 neighbour
            every 3rd iteration, by deterministic rotation hash(run_seed, N))
    update gamma from realised segment cost; update overhead ledger; write_atomic(balance/itN.json)
```

### Realistic outcome for hybrid1

- The controller can attach at the first iteration after integration.
- With 2-3 controlled iterations, expect one challenger (f = 1/8, k = 4) tested twice and at most one promotion.
- "No evidence, no move" is the designed outcome when signals are weak.

## 5. File formats

**`runs/full/<RUN>/balance/itN.plan.json`** is immutable once written, with an atomic tmp + rename write:

```json
{"schema": 1, "run_seed": 0, "iteration": N, "mode": "explore|maintenance|frozen|failsafe",
 "theta_star": {"f":0,"k":null,"b":800,"lambda":0,"src":"cap"},
 "challenger": {"f":0.125,"k":4,"b":800,"lambda":0,"src":"cap"} ,
 "segments": {"seg00": {"theta":"inc","seed":1000,"games":2000,"engine_args":[]},
              "seg04": {"theta":"chal","seed":1000,"games":1444,"twin":"seg00",
                        "engine_args":["--deep-fraction","0.125","--deep-multiplier","4"]}},
 "val": {"engine_args": [], "assert_stock": true},
 "trainer_args": [], "steps_rule": "policy_positions",
 "probe": {"version": 2, "refresh": false, "path": "balance/probe/deep_v2.dst", "sha256": "..."},
 "cost": {"v_gpu_s_per_visit": 4.24e-5, "gamma": {"3200": 1.04}, "c_inc": 1.0, "c_chal": 1.385},
 "model": {"generator": "models/...onnx", "sha256": "..."},
 "inputs": {"balance/it{N-1}.json": "sha256", "metrics/it{N-1}.json": "sha256|null", "prior": "sha256"}}
```

**`runs/full/<RUN>/balance/itN.json`** is written by `decide`. It is the plan plus results plus the new state, and run_metrics.py merges it:

```json
{"schema": 1, "iteration": N, "plan_sha256": "...",
 "results": {"d": 0.0042, "sigma_n": 0.0019, "scores": {"inc_s0": 2.113, "chal_s0": 2.108, "...": 0},
             "minigate": {"games":100, "score":0.53, "p_worse":0.27} ,
             "realised": {"seg04": {"gpu_s": 9400, "visits": 2.1e8, "games": 1444, "records": 209000,
                          "deep": 26100, "masked": 0, "sit_rate": 0.11, "flag_rate": 0.05}}},
 "decision": {"action": "promote|reject|continue|revert|freeze|none", "reason": "Z=2.3 n=2 ...",
              "Z": 2.3, "mean": 0.0039, "delta_min": 0.0021},
 "state": {"theta_star": {}, "step": 2, "dir_idx": 0, "tabu": {}, "evidence": {}, "gamma": {},
           "overhead": {"this_iter_gpu_h": 1.05, "cum_gpu_h": 2.1, "run_gpu_h": 33.0, "frac": 0.064},
           "last_promotion": null, "revert_test": false, "frozen_until": null, "mode": "explore"},
 "main_mix": {"inc": 0.83, "chal": 0.17}}
```

Other files:
- **Per-stage results:** `balance/itN/mini/<arm>_s<s>/{last.pt, *.onnx, log}`, `balance/itN/score.json`, `balance/itN/minigate.json`.
- **Stage markers:** `rl-itN/stages/balance_{minitrain,score,probe,minigate,decide}.done`.
- **`balance/prior.json`:** the offline sweep result (§7). It is the initial σ̂, the γ table and directions masked by the bench.
- **Engine sidecar `<distill-output>/cap_counts.json`:** written only when cap flags are on. It holds {records, deep, masked, deep_visits, fast_visits}, is rewritten on each checkpoint, and is recomputed from records on resume.

**Determinism and resume:**
- `plan` and `decide` are pure functions of their listed inputs and the run seed. Tie-breaks use hash(run_seed, N, dir_idx).
- No floats go through BLAS: weighted means are plain Python floats, and JSON is dumped with sort_keys and fixed rounding (6 significant digits).
- An existing plan or result file is never recomputed. On rerun, the recorded input shas are checked, and a mismatch is a hard error.
- A missing result triggers a re-run of only that stage, by its `.done` marker.
- Unit test: delete itN.json, rerun `decide`, and require a byte-identical file.

## 6. Engine changes (fork engine/, all OFF by default)

### Playout cap: `hivemind selfplay` new flags (SelfPlayConfig fields)

| Flag | Default | Meaning |
|---|---|---|
| `--deep-fraction F` | 0 | Probability that a searched decision is deep |
| `--deep-multiplier K` | 1 | Deep budget = K × base |
| `--fast-nodes B` | 0 | 0 means the same as `--nodes`. Budget of non-deep decisions. |
| `--fast-policy-mask true/false` | true when B < nodes | Write non-deep records with empty policy on both boards |
| `--fast-dirichlet true/false` | false | Dirichlet noise on non-deep searches when B < nodes |
| `--deep-dirichlet true/false` | true | Dirichlet noise on deep searches. Probe mode passes false. |
| `--cap-seed-salt S` | 0x5EED_CA9 | Salt for the coin |

Implementation points:
- **Seat path:** selfplay.cc:1076-1077. **Legacy path:** 1711-1712.
  - Call `randomized_node_budget(config, randomEngine)` unconditionally, exactly as now, then scale: `target = base * (deep ? K : (B ? B/nodes : 1))`.
  - Coin: `deep = unit(selfplay_seed(runId ^ salt, gameIndex*maxMacroPlies + decision)) < F`. It never touches randomEngine.
- **Dirichlet:** set `searchOptions.search.rootDirichletEpsilon` per decision (selfplay.cc:1102-1105, 1733-1735).
- **Policy mask:** at the record write (1179-1190, 1777-1790), a masked record gets `policy[0]` and `policy[1]` cleared. The record, the TrainingSample and the seat row are all still written, so the 1:1 alignment that writeGame, label_report and attach_timing rely on holds.
- **Team tree reuse:** `minNewNodes = 0.25 × chosen target`, not the base target. Legacy tree reuse uses the same rule.
- **Resign:** when F > 0 or B < nodes, only deep or 800+ searches may feed the resign counter. The loop keeps resign off anyway.
- **Counts:** cap_counts.json (§5).
- **Required tests:**
  - (1) Flags absent: same seed, byte-identical .dst, games.pgn and seat_clocks.tsv versus the stock build, built into `engine/build-balance` only.
  - (2) C++ read_chunk and Python data.py round-trip records with policyCount 0 on both boards, including through checkpoint/resume (selfplay.cc:1381-1391, 1479-1519).
  - (3) label_report and attach_timing pass on a capped segment.
  - (4) With F = 1/8, K = 2, the deep share in cap_counts is within the binomial CI.
  - Locally: ≤ 4 games, parallel-games ≤ 2.

### State dump for reanalyse: `--record-state-output PATH` (off by default)

- Writes one JSONL line per record, at writeGame time so the outcome is known: {game, decision, record_index, fenA, fenB, side/team to move, TimeState (clocked lead, paces, plan_cost budget for this decision, sit gate), durative/hybrid clock incl. in-flight committed moves, wave2 option hash, root noise seed, nodes, outcome, wdl, moves_left}.
- Checkpoint/resume must round-trip it like records.

### `hivemind reanalyse` (new subcommand, modelled on tools/search_diag.cc)

```
hivemind reanalyse --model M.onnx --states S.jsonl --source-dst <dir|files> --nodes N \
    [--dirichlet false] [--replay-noise true] [--select F] --wave2 <same options as the source run> \
    --parallel-engines P --output OUT_DIR
```

What it does, per state:
- builds the board pair, TimeState and hybrid clock;
- applies `apply_board_rules` for config.wave2;
- runs a fresh search (Gumbel off, mate probe off);
- writes HDST via `distill::ChunkWriter`, with planes recomputed and asserted equal to the source record's planes, policy through `distill_search_policy` with the same selfplay/hybrid edge filters, root Q as value, and outcome/WDL/moves-left copied from the state line;
- batches across P engines.

**Fidelity gate (must pass before src = reanalyse is unmasked):**
- `--nodes 800 --replay-noise true` on a segment made by the same build and net reproduces the original records' policy and value bit-exactly on ≥ 99% of records, with one search worker.
- Plane equality must hold on 100% of records.

### Reanalyse pipeline (src = reanalyse arm)

1. Self-play runs with `--record-state-output`, which costs only disk.
2. After self-play, `reanalyse --select F` picks a deterministic hash-chosen fraction F of states and searches them at k·800 with the generator net.
3. The output dir `search/reanalyse/segNN` is passed to the trainer as additional search data. The originals keep their 800-node policy; the trainer's `--dedupe-reanalysed` (new, off by default) masks the policy of originals whose (seg, record_index) was reanalysed.
4. Reanalyse GPU time is charged to self-play, so games shrink to keep the iteration's GPU-h fixed.

The same tool, run on the parallel swarm's frozen probe positions (if they carry state lines), is the long-term replacement for probe-mode self-play.

### As built (engine side, 2026-11-22; differences from the text above)

Files: `engine/src/tools/selfplay.{h,cc}` (cap + state dump), `engine/src/tools/reanalyse.{h,cc}` (new), `engine/src/main.cc` (flags, `reanalyse` subcommand, help text), `engine/CMakeLists.txt`, `engine/tests/test_search_balance.cc`. Local build: `engine/build-balance` (one search worker, ONNX Runtime).

**Playout cap flags** are as in the table, with these details:
- `--fast-policy-mask` is tri-state in SelfPlayConfig (`fastPolicyMask` -1/0/1): unset means "true exactly when B < nodes". The cap counts as on when F > 0, B ≠ nodes, or the mask is forced true.
- Coin: `unit(selfplay_seed(0, selfplay_seed(runId ^ salt, gameIndex*maxMacroPlies + decision))) < F` (one extra mix so the coin bits are not the noise seed xor the salt). `decision` is the seat path's `decisions` counter / the legacy path's `macroPly`.
- Fast budget is `max(1, round(base · B / nodes))`, so the base's ±5% jitter carries over.
- Dirichlet: deep searches keep ε unless `--deep-dirichlet false`; non-deep searches keep ε when B = nodes and drop it when B < nodes unless `--fast-dirichlet true`.
- Resign counter: only searches with `deep || B >= nodes` update it (both the real and the exempt-game check).
- Masked records: the HDST record keeps planes, root Q, ply, outcome, WDL and moves-left; both policy lists are empty. HVM6 training chunks (`--training-chunks true`, unused by the loop) are not masked.
- `cap_counts.json` is written next to the HDST chunks (`--distill-output`, else `--output`) at every checkpoint and at the end, only when the cap is on. On resume the counts are **restored from the checkpoint's progress file** (new optional keys `cap_records cap_deep cap_masked cap_deep_visits cap_fast_visits`, written only when the cap is on), not recomputed from records: a record alone cannot tell a deep search from a stock one.
- Per-game counts go on a **separate stdout line** `selfplay cap game N records R deep D masked M`, plus a final `selfplay cap: R records, D deep, M masked, visits deep X other Y`. The `selfplay game` line is unchanged, because labels.py and run_metrics.py end the termination field at known keywords.
- Deep searches do not cost more simulated seat-clock time: `plan_cost` is drawn before the search and does not depend on its node count, so the games' clocks and flag rates stay comparable between arms.

**State dump** (`--record-state-output PATH`, must not exist on a fresh run; truncated back to the checkpoint on resume via the optional progress key `state_bytes`). It is JSONL, written in game order at writeGame time:
- header `{"type": "header", "version": 1, "seed", "nodes", "max_macro_plies", "seat_clocks", "input_channels", "time_state_bytes"}`;
- per game `{"type": "game", "game", "winner", "termination", "moves": [[board, Stockfish::Move int], ...]}` with every move pushed on either board, in order;
- per record `{"type": "record", "game", "record", "prefix", "ply", "team", "time_advantage", "clock_set", "clock", "hybrid_clock", "noise_seed", "base_nodes", "target_nodes", "min_new_nodes", "epsilon_bits", "deep", "masked", "visits", "outcome", "wdl", "moves_left"}`.
- The board is rebuilt by replaying `moves[:prefix]` under the wave 2 board rules, which is exact (pockets and repetition history included). `clock` and `hybrid_clock` are the bytes of `TimeState` in hex (same build only; the header's `time_state_bytes` and `input_channels` are checked). Its 37 padding bytes are zeroed: they hold stack garbage that differs between runs, and are found at startup by flipping the low bit of each byte and checking `operator==`. Zeroing them makes the file byte-identical across a kill and resume. There are no FENs.
- Size is about 1.2 KB per record, so roughly 0.35 GB per 2,000-game segment.

**`hivemind reanalyse`**:
```
hivemind reanalyse --model M.onnx --states S.jsonl --output OUT [--source-dst DIR|FILE ...] \
    [--nodes N] [--replay-noise BOOL] [--dirichlet BOOL] [--dirichlet-alpha A] [--dirichlet-epsilon E] \
    [--select F] [--select-salt S] [--skip-deep BOOL] [--fairy-stockfish-mate-nodes 0] [--root-scans BOOL] \
    [--parallel-engines P] [--batch-size B] [--distill-chunk-positions N] <selfplay wave 2 options of the source run>
```
- `--nodes 0` (the default) uses each record's recorded target. N > 0 uses `round(base_nodes · N / run_nodes)`.
- Noise: `--replay-noise true` uses the recorded seed and ε. Otherwise ε = 0, or the configured ε with the recorded seed if `--dirichlet true`.
- Every search starts from an empty tree (`reset_search_state` + `clear_plans`). Runs made with `--team-tree-reuse` or `--plan-carry` therefore cannot be reproduced bit for bit. hybrid1 uses neither.
- Mate probe nodes default to 0, as in the loop.
- Edge filter as in the seat path: `hybrid_policy_edges` when the hybrid triggered, else `selfplay_policy_edges`.
- With `--source-dst`, the recomputed planes must equal the source record's (any mismatch is a hard error), and policy and value equality is counted.
- Outputs:
  - `reanalyse_<seed>_NNNNN.dst` (outcome, WDL, moves-left and ply copied from the state line);
  - `reanalysed.tsv` (game, record, source_index), which the trainer's `--dedupe-reanalysed` should key on;
  - `reanalyse.json` (records, reanalysed, visits, compared, planes_equal, policy_equal, value_equal, policy_and_value_equal).
- Selection: `reanalyse_selected(seed, salt, game, record, F)`, a hash on `(game << 20) ^ record`.

**Determinism finding (affects the byte-identity test, the fidelity gate and the CRN twin pairing).** Stock self-play is **not** reproducible on this Mac with the default `--root-scans true`, even with one search worker and `--parallel-games 1`. Two stock runs with the same seed gave different games.pgn and .dst files. The cause is the time-budgeted root mate scans (agent.cc `MateSearchBudget` deadlines, mate_probe.cc). With `--root-scans false`, runs are byte-identical. So:
- (1) the byte-identity test runs with `--root-scans false`;
- (2) the fidelity gate must use a segment made with `--root-scans false` and reanalyse with `--root-scans false`;
- (3) in the live loop, twin segments X and T share seeds but diverge wherever a root scan's timing differs.

Common random numbers then hold only up to the first divergence in each game. The paired d is still unbiased, but its variance reduction is smaller than assumed.

**`hivemind reanalyse` driver (`src/hivemind/cli/reanalyse.py`), as fixed in the review round (§10):** it reads the
dump format above (header, game and record lines; a record's `record_index` is its position among the dump's record
lines, i.e. among the segment's HDST records). Each shard's states file is the header, the game lines of the games
it touches and its chosen record lines, in dump order. `--source-dst` is **not** passed: the engine numbers source
records by their order in the file it is given, so on a subset it would compare against the wrong records. The
driver instead checks every output record in order against its own source record (planes, WDL, outcome,
moves-left), checks the engine's `reanalysed.tsv` (game, record) against the shard, and counts policy/value
equality itself (`fidelity` in the segment marker; `fidelity_valid` is true only with `--replay-noise true`,
`--root-scans false` in `--engine-extra`, and a replay made here with `--root-scans false` in `--replay-extra`).
`selected.tsv` now has columns `record_index shard game record`.

## 7. Trainer changes (fork src/hivemind/distill, all OFF by default)

- `distill-train --search-q-blend λ` (λ in [0, 1], default 0; as built in distill/train.py `wdl_target` / `q_wdl`).
  - It changes the **WDL-head target**, not a scalar-value term, and has no separate weight flag (the WDL term keeps
    its usual `--search-wdl-weight`): target = (1−λ)·outcome + λ·[(1−q)/2, 0, (1+q)/2], with q the record's root Q
    (the `value` field, same team's side, so q = +1 is a certain win) in [loss, draw, win] order. The Q part puts
    no mass on the draw.
  - Only `q_blend_rows` are blended: search rows (source > 0) with an outcome WDL, a policy on at least one board
    and a finite q in [−1, 1]. Macro-ply-limit records (zero WDL) and playout-cap fast records (masked, no policy:
    their Q comes from the small search) keep the outcome target.
  - Validation (`evaluate`) keeps the outcome target, so runs with different λ report comparable losses.
  - λ = 0 returns `batch["wdl"]` untouched and `main()` calls `losses()` exactly as before the flag existed.
  - The prior's capability check (`distill-train --help` lists `--search-q-blend`) is what makes λ > 0 arms legal.
- `train.policy_positions(dirs)` (cli/train.py): records minus `cap_counts.json` masked counts. It falls back to `positions()` when no sidecar exists, so it is identical at θ0.
- Mini-train mode needs no new trainer code: `distill-train --init --steps 200 --seed --export last` already exists. It is wrapped by `hivemind balance minitrain`, which only builds the command (reusing `train.distill_command`) and writes `.done`.
- `hivemind balance score`: loads each last.pt, evaluates on the probe .dst (policy CE on policy-carrying records, WDL log-loss, moves-left) and on the swarm probe when present, and writes score.json.

As built there is no `hivemind/balance/` package: the controller (plan, realise, decide, status, simulate) is
talisman `cluster/search_balance.py` (stdlib only), and the GPU stages are the fork's `src/hivemind/cli/balance.py`,
`hivemind balance {minitrain,score,probe,accounting}` (it also passes `plan`/`decide`/`realise`/`status` through to
the controller). rl-loop calls the stages as `python -m hivemind.cli.balance …` because `python -m hivemind …`
drops a command's return code (exit 2 = a stage error).

## 8. Offline validation (before spending GPU)

### Simulator: `hivemind balance sim`, CPU only, repo .venv

- **True gain per iteration:**
  - g(θ, n) = A·log N(θ) + B(n)·Q(k)·f_eff(f, b) − P_hybrid(b) + ε
  - Cost from §2, with γ shocks (γ(6400) ∈ {1, 1.5}, OOM at k = 8).
- **Observations:**
  - d ~ N(κ·g(θ_c) − κ·g(θ_inc) + bias_KL(k), σ²);
  - σ measured locally (below);
  - mini-gate binomial over 100 games;
  - main gate ±35 Elo (binomial over 400 games);
  - cost jitter ±6%.
- **Surfaces:**
  - (a) flat;
  - (b) optimum at θ0;
  - (c) interior optimum f = 1/8-1/4, k = 4;
  - (d) edge optimum k = 8;
  - (e) drifting, with B(n) rising;
  - (f) proxy-adversarial: bias_KL > 0 but no Elo gain;
  - (g) a pure λ win.
- **Runs:** 500 seeds × each surface × run lengths {4, 10, 20}. Compare against the oracle, fixed θ0, and design A's Thompson controller (kept in sim.py as a baseline).
- **Pass criteria:**
  - false departure from θ0 on (a), (b) and (f) ≤ 5%;
  - regret ≤ 10% of the oracle's gain by iteration 8 on (c) and (e);
  - overhead ≤ 10% in every run;
  - determinism: kill midway, resume, and get byte-identical files.
- If the pass criteria fail at the measured σ, the thresholds (Z, n_obs) are tuned in the sim, never on the live run.

### Local measurements (int-shared fixture: copy first, small runs only)

- σ of d: copy `runs/wave2/int-shared`, then run 4 tiny mini-trains (2 seeds × 2 data subsets, a few hundred steps) with the fork .venv.
- End-to-end dry run:
  - engine/build-balance selfplay with `--deep-fraction 0.125 --deep-multiplier 2`, 4 games, parallel-games ≤ 2;
  - `balance plan/minitrain/score/decide` on the copy;
  - delete-and-resume test.
- KL scale prior: searchdiag at budgets 800, 1600, 3200, 6400 on FEN-dumped fixture positions. This is approximate (legacy time state) and used only for the prior's scale.

### On PACE (each needs the user's explicit approval; a scratch run dir, never hybrid1)

- One η/γ bench: 6 parallel games at 800, 1600, 3200 and 6400 nodes for ~10 minutes each, with memory recorded. This sets `fits_bench` for k = 8.
- One replicate mini-train round for a first σ̂ on real-size data. The results go into `balance/prior.json`.

## 9. Integration (after both swarms finish; this swarm does not edit these files)

### rl_loop.py (`src/hivemind/cli/rl_loop.py`)

**New flags (as built):**
- `--search-balance` (default off; off is byte-identical to today's behaviour);
- `--search-balance-controller PATH` (default `$HIVEMIND_SEARCH_BALANCE`, else talisman `cluster/search_balance.py`);
- `--search-balance-prior PATH` (default `RUN/balance/prior.json`; passed to `plan`, `decide` and `decide --dry`,
  and recorded in the plan as `prior_path`);
- `--search-balance-mini-steps 200`, `--search-balance-probe-games 150`, `--search-balance-minigate-games 100`,
  `--search-balance-gpus 1.0`, `--search-balance-device {auto,cpu,cuda}`.
- The explore cap (10%) is the controller's `OVERHEAD_CAP`, not a flag.
- Cluster side: `wave2_iterate.sbatch BALANCE=1` (with `BALANCE_PRIOR`, `BALANCE_ARGS`) and
  `launch_full_run.sh BALANCE=1` (`RESUME=1 REPLACE_PENDING=1` to switch a running chain).

**`iteration(n)`:**
- Before any self-play, call `hivemind balance plan --run <RUN> --iteration n` once, guarded by `stages/balance_plan.done`. It writes `balance/itN.plan.json`.
- On failure, log the error and write a failsafe plan: θ* everywhere, no challenger.

**Segment settings:**
- In the local loop (`iteration`, lines ~1061-1071) and in `shared_segment` and `distributed_selfplay` (workers), look the segment name up in the plan.
- Use its `games`, in place of `min(segment_games, ...)`, and its `seed`: the challenger segment reuses the twin's seed.
- Append its `engine_args` in `Loop.selfplay` (line 544) on **train segments only**.
- `games_played()` and the `while played < args.games` loop must count segments to completion by the plan, not by 10,000 games. The loop ends when every planned segment is `.done`.
- Assert that the val command contains no cap flags (`plan.val.assert_stock`).

**`cli/selfplay.py:engine_command`:** take `nodes` as a parameter (default 800) in place of the hard-coded `--nodes 800`. The plan's engine_args carry the cap flags, and nodes stay 800.

**`train()`:**
- When balance is on, use `train.policy_positions` for `npos`/`opos`, per `steps_rule`.
- Append the plan's `trainer_args` (e.g. `--search-q-blend`) via `extra`.
- Record `steps_rule` and `main_mix` in train.done.

**New stages (each under `self.attempt`):**
1. After `train`: `balance_probe` (if `plan.probe.refresh`), `balance_minitrain`, `balance_score`.
2. After `match` and `promote`: `balance_minigate` (only if `decide --dry` says eligible), then `balance_decide`.

Rules for these stages:
- They are skipped when there is no challenger.
- They run on the same GPU sequentially and use `wait_for_quiet`.
- Their GPU seconds are logged into the ledger.
- The minigate uses `self.tournament(...)` with the mini nets' ONNX, 100 games at 100 ms, and openings rotated from the gate set.

**Inputs passed to decide:**
- the gate's `match_generator.done` (with the `sprt` field if `--gate-sprt`);
- `metrics/it{N}.json` if the collector already wrote it (sha recorded either way).

**Mid-run attach (hybrid1):** with `--search-balance` on a run that started without it, `plan` for the first controlled iteration initialises from `prior.json`. Iterations already played count as θ0 evidence only for δ_min calibration.

**Deployment safety:** hybrid1's chained jobs rebuild build-sp1 when the fork HEAD changes. The engine change must be merged with the cap flags off and pass the byte-identical test before any chained job can pick it up.

### cluster/full_run_watchdog.sh

- Treat `balance_*` stages as normal stages for staleness and heartbeat.
- Alert (via the metrics swarm's alert channel) when:
  - `mode == failsafe`;
  - overhead frac > 0.10;
  - a challenger segment's realised c is more than 1.3 × planned;
  - a challenger segment hits OOM or crashes;
  - `review: true` (hybrid/flag shift > 20%);
  - `balance_decide` is missing when the next iteration's self-play starts.
- Never restart a balance stage more than twice. After that, write a failsafe plan for the next iteration.

### cluster/run_metrics.py

- Merge `balance/itN.json` into `metrics/itN.json` as a top-level `"balance"` section with these fields:
  - `theta_star`, `challenger`, `mode`
  - `decision.{action,reason,Z,mean,delta_min}`
  - `results.d`, `results.sigma_n`
  - `overhead.frac`
  - `cost.{c_inc,c_chal}`
  - per-arm realised {games/GPU-h, visits/s, deep share, masked share, sit_rate, flag_rate}
  - `main_mix`
  - `probe.version`
- Add `cost.selfplay_gpu_h_by_arm` to the existing cost section.
- Elo per GPU-hour in the outcome section must use the total iteration GPU-h *including* balance overhead.

### Panel (viewer/)

Add a "Search balance" card per run:
- the current θ* and challenger as a small grid (f × k, with b and λ as badges);
- an evidence table (arm, n_obs, mean d ± se, Z, status: testing, promoted, tabu, rejected);
- the decision history over iterations;
- overhead frac against the 10% cap line;
- games/GPU-h and visits/s per arm;
- hybrid sit and flag rates per arm, with the review flag;
- the probe version and net sha;
- a mode badge (explore, maintenance, frozen, failsafe).

### cluster/README.md

Document `--search-balance`, the balance/ directory, the PACE bench job (user-submitted), and the rule that val is always stock.

## Simulation results

**Superseded in part (2026-11-22 review round):** the tables in this section predate the shared per-iteration noise
term τ, the measured (not true) prior noise, the prior flags and the Z re-tune. The current numbers, and what each
fix changed, are in §10.

Status: controller and simulator BUILT in `cluster/search_balance.py` (stdlib only) and tested in `tests/cluster/test_search_balance.py`. Nothing below used a GPU or a network: every number is from the synthetic world, with no model or weights. Reproduce with:

```
.venv/bin/python cluster/search_balance.py simulate --iterations 40 --seeds 200 --sigma 4 --markdown
```

(about 75 s of CPU per σ value).

### Synthetic world

- **Equal GPU per arm.** Every arm gets the same self-play GPU time, because games are budgeted in GPU-h, so ranking arms by the true per-iteration gain g ranks them by Elo per GPU-h.
- **Base gain.** g = 30 Elo/iteration at θ0 + 6·log(value positions) + 9·log(policy positions) + a deep-target benefit B·(1−e^(−f/s))·q(k) + a λ term. Positions are relative to θ0 at equal GPU, using the true cost.
- **Surfaces.** Seven surfaces, (a)-(g) of §8:
  - (b) and (f) share the same true g, whose best arm is within 0.5 Elo of θ0;
  - (c) peaks at f = 1/4, k = 4, with f = 1/8, k = 4 within 1.3 Elo of it;
  - (d) peaks at k = 8, f = 1/4;
  - (e) has B rising from 4 to 22 over 20 iterations;
  - (f) adds a probe bias of up to +12 units to deep arms that buys no Elo;
  - (g) has λ = 0.25 worth +7 Elo.
- **Noise.**
  - Mini-train d per seed is N(true gap + bias, σ²), in Elo-equivalent probe units (κ = 1).
  - The main gate has ±35 Elo of noise (sd).
  - The mini-gate is binomial over 100 games, on half the true gap.
  - Realised cost has ±6% jitter.
  - The true γ is γ(200) = 1.3, γ(400) = 1.15, γ(3200) = 1.03, and γ(6400) ∈ {1, 1.5} per seed.
  - k = 8 fits the bench in 50% of seeds, and in all seeds on (d). The controller starts with γ = 1 and learns it from the realised segments.
- **Comparisons.**
  - The **oracle** is the best legal fixed recipe, chosen per iteration on (e), with the true γ and no overhead.
  - **Fixed recipes** plan with nominal γ = 1, so they pay any cost overrun.
  - **TS (A-lite)** is design A without the GP: an independent Gaussian prior per arm, a Thompson-sampled challenger, promotion at P(better) ≥ 0.95, and the same overhead minus the mini-gate. It is weaker than the full design A.
- **Assumed σ.** σ of d has not been measured (§8's local σ run was not done). The tables use σ = 4 Elo-equivalent units per seed, and σ = 2 and σ = 8 are summarised below. The prior gives the controller σ̂ = σ.

### σ = 4, 200 seeds × 40 iterations
Elo per GPU-h as % of the oracle (oracle = best legal fixed recipe, per iteration on e_drift), at iteration 4/8/10/20/40:

| surface | controller (B) | TS baseline (A-lite) | fixed theta0 | fixed f1/8 k4 | fixed f1/4 k2 | fixed f1/4 k4 b200 |
|---|---|---|---|---|---|---|
| a_flat | 94.4/94.4/94.6/96.0/96.7 | 93.2/92.9/92.8/92.3/91.5 | 99.9/100.0/100.0/100.0/100.0 | 99.1/99.2/99.2/99.2/99.2 | 99.9/100.0/100.0/100.0/100.0 | 94.6/94.6/94.6/94.7/94.7 |
| b_theta0 | 93.6/94.2/94.5/96.0/96.6 | 86.1/86.1/86.0/86.0/86.0 | 100.0/100.0/100.0/100.0/100.0 | 90.1/90.0/90.0/90.1/90.1 | 93.6/93.5/93.5/93.6/93.5 | 52.6/52.6/52.6/52.6/52.6 |
| c_interior | 82.2/85.1/86.1/88.0/90.4 | 69.9/72.3/73.0/75.7/78.7 | 73.8/73.8/73.8/73.8/73.7 | 95.9/95.9/95.9/95.9/95.9 | 86.9/86.9/86.9/86.9/86.9 | 77.2/77.2/77.2/77.2/77.2 |
| d_edge_k8 | 79.0/80.9/81.8/83.6/86.0 | 72.0/73.5/74.2/76.5/78.5 | 76.2/76.2/76.2/76.3/76.2 | 92.6/92.6/92.6/92.6/92.6 | 87.9/87.8/87.8/87.9/87.8 | 71.4/71.4/71.4/71.4/71.4 |
| e_drift | 93.3/92.1/91.0/85.2/82.2 | 86.5/85.1/84.1/79.8/79.8 | 99.2/97.1/95.6/87.4/80.0 | 95.5/97.3/97.7/97.7/96.8 | 96.3/96.6/96.3/93.2/89.8 | 60.7/64.2/65.6/70.9/74.3 |
| f_proxy | 90.4/89.7/89.7/90.0/90.7 | 85.3/84.5/84.1/82.4/80.5 | 100.1/100.0/100.1/100.1/100.0 | 90.1/90.1/90.1/90.1/90.1 | 93.6/93.6/93.6/93.6/93.6 | 52.7/52.7/52.7/52.7/52.6 |
| g_lambda | 76.3/78.1/79.3/85.0/89.5 | 71.2/72.2/72.7/75.2/78.4 | 81.2/81.1/81.1/81.1/81.1 | 73.2/73.1/73.1/73.1/73.0 | 76.0/75.9/75.9/75.9/75.9 | 42.7/42.7/42.7/42.7/42.7 |

| surface | left theta0 by it4/8/10/20/40 | theta* = optimum at end of it8/20/40 | median first iteration at optimum (share of runs) | promotions/run | reverts/run | overhead mean / max |
|---|---|---|---|---|---|---|---|
| a_flat | 2%/3%/4%/4%/6% | 0%/0%/0% | -/-/- | - (0%) | 0.07 | 0.01 | 3.3% / 7.3% |
| b_theta0 | 0%/2%/2%/2%/4% | 98%/98%/98% | -/-/- | - (0%) | 0.04 | 0.01 | 3.1% / 7.3% |
| c_interior | 95%/97%/97%/98%/99% | 0%/4%/6% | 43%/54%/64% | 11.5 (6%) | 1.07 | 0.01 | 4.0% / 8.9% |
| d_edge_k8 | 71%/82%/85%/88%/94% | 0%/1%/2% | 20%/31%/41% | 18.5 (2%) | 1.11 | 0.01 | 4.2% / 8.9% |
| e_drift | 2%/6%/8%/22%/78% | 0%/0%/0% | -171%/-18%/11% | 5.0 (3%) | 0.96 | 0.04 | 3.8% / 7.3% |
| f_proxy | 52%/60%/62%/64%/74% | 42%/36%/26% | -/-/- | - (0%) | 0.77 | 0.03 | 3.8% / 8.9% |
| g_lambda | 21%/52%/74%/86%/98% | 18%/59%/77% | -16%/21%/44% | 10.0 (78%) | 1.14 | 0.01 | 3.7% / 7.3% |

### Sensitivity to σ: regret (1 − Elo/GPU-h ÷ oracle) at iterations 8 and 20, departure from θ0 by iteration 20

| surface | σ=2 regret it8 / it20 | σ=4 regret it8 / it20 | σ=8 regret it8 / it20 | left θ0 by it20 (σ=2/4/8) | TS (A-lite) regret it8, σ=4 |
|---|---|---|---|---|---|
| a_flat | 5.6 / 3.9% | 5.6 / 4.0% | 5.6 / 4.1% | 2 / 4 / 12% | 7.1% |
| b_theta0 | 4.9 / 3.5% | 5.8 / 4.0% | 6.2 / 4.4% | 0 / 2 / 6% | 13.9% |
| c_interior | 13.4 / 10.6% | 14.9 / 12.0% | 22.4 / 19.9% | 100 / 98 / 64% | 27.7% |
| d_edge_k8 | 15.7 / 13.0% | 19.1 / 16.4% | 23.5 / 21.8% | 100 / 88 / 52% | 26.5% |
| e_drift | 7.8 / 14.6% | 7.9 / 14.8% | 8.1 / 15.0% | 33 / 22 / 20% | 14.9% |
| f_proxy | 13.7 / 13.7% | 10.3 / 10.0% | 7.1 / 5.9% | 98 / 64 / 24% | 15.5% |
| g_lambda | 18.7 / 12.6% | 21.9 / 15.0% | 23.2 / 18.7% | 100 / 86 / 54% | 27.8% |

### Against the §8 pass criteria (σ = 4)

- **Overhead ≤ 10% in every run: PASS.**
  - The worst cumulative fraction at any iteration of any run is 8.9%.
  - The mean is 3.1-4.2%, which is lower than §2's 6-8% because maintenance mode tests only every 3rd iteration.
- **False departure from θ0 ≤ 5% on (a) and (b): PASS at σ ≤ 4** (2-4% by iteration 20). It fails at σ = 8 (6-16%).
- **False departure on (f): FAIL.** 52-64% of runs leave θ0 by iteration 8-20 at σ = 4.
  - The mini-gate cannot see a ~0.7 Elo true gap.
  - The revert test re-measures the same biased probe.
  - Two negative main gates (±35 Elo) rarely both happen.
  - The real cost is modest: θ* moves to a recipe within ~1 Elo of θ0, so the regret is 10% versus 6% on (b), and most of that is overhead.
  - The spec's actual defence is limiting bias_KL: the outcome-WDL score term and probe depth ≥ 2× arm k. The sim cannot test that defence. Measure the bias on real data before trusting promotions of deep arms.
- **Regret ≤ 10% of the oracle by iteration 8 on (c) and (e).**
  - (e) PASSES at 7.9%. θ0 is the optimum for the first ~10 iterations, and the controller then lags the drift: 14.8% at iteration 20.
  - (c) FAILS at 14.9%, and is about 12% at iteration 20. This is structural, not noise:
    - about 2 iterations at θ0 before the first promotion, because n ≥ 2 observations or a single-shot Z ≥ 3 is needed plus a mini-gate;
    - about 4% overhead;
    - one challenger segment in five playing worse neighbours while compass search probes them.
  - The controller almost never reaches the exact optimum (f = 1/4) from f = 1/8, because the 1.3 Elo gap is below δ_min (≈ 1.5σ̂ = 6). It does capture 43/54/64% of the oracle's gain over θ0 at iterations 8/20/40.
- **Against fixed guesses.**
  - The controller beats θ0 whenever there is something to find: (c), (d), (g), and (e) after the drift.
  - It loses about 4-6% to θ0 when θ0 is optimal, which is the price of exploring.
  - It loses to a lucky fixed guess: fixed f = 1/8, k = 4 is within 4% of the oracle on (c) and 7% on (d).
  - It beats every fixed guess on (g) after about 20 iterations, and is the only policy that is within 18% of the oracle on every surface by iteration 40 (worst: (e) at 82%).
  - It beats TS (A-lite) on every surface and length, which agrees with §0's choice of B.
- **Determinism: PASS.**
  - Same seed gives an identical trajectory.
  - In tests, deleting `it2.json`, `it3.plan.json` and `it3.json` and rerunning `decide 2` / `plan 3` / `decide 3` gives byte-identical files, even after `metrics/it2.json` is regenerated.

**Tuning candidates (to tune in the sim, not on the live run; none applied):**
1. Start from the offline sweep's best arm when `prior.json` holds one. This removes the iterations at θ0.
2. Skip maintenance mode on runs shorter than about 15 iterations.
3. Let a 2-observation Z ≥ 2 promote without the mini-gate when the probe bias has been measured as small. The mini-gate never vetoed a true gain here, and it costs an iteration.

### Implementation notes and deviations from §1-§5

- **c(θ) is normalised by (1 + ρ)**, so θ0 costs exactly 1.0 and plays exactly `segment_games`. As a result f = 1/8, k = 4 is c = 1.3713 (1458 games), not 1.385 (1444).
- **b < 800 requires f > 0.** With f = 0 every policy target would be masked, so that arm is off-grid.
- **Mini-gate veto → `continue`**, following §4's flow: a veto blocks the promotion, and rejection still needs Z ≤ −1 or n ≥ 3. Evidence for a pair is cleared on rejection, so a later re-test starts fresh. Evidence is keyed by the (incumbent, challenger) pair, because d is relative to the incumbent.
- **Single-shot promotion** (n = 1, Z ≥ 3) also requires mean ≥ δ_min and a σ̂ from the prior or from ≥ 3 seed-pair spreads.
- **δ_min clamp.** The calibrated δ_min (5 Elo / β, regressed through the origin) is clamped to [0.5, 3]·σ̂. Mini-net d and main-net probe gains are not on the same scale, so an unclamped value could block every promotion or allow every one.
- **SPRT.** `sprt_regressed` means the gate's SPRT decision is `H0` with score < 0.5.
- **Probe-regression revert.** It fires on the run_metrics alert codes `heldout_value_loss_up` or `probe_regression` within 3 iterations after a promotion.
- **Re-run checks.** These cover only balance-owned inputs: prior and it{N−1}.json for `plan`; plan, score.json and minigate.json for `decide`. metrics/*.json is regenerated with timestamps, so only a digest of the fields actually used is recorded for it.
- **Failsafe.** A corrupt or missing it{N−1}.json gives a failsafe plan: θ* everywhere, no challenger, with state taken from the last readable file. `decide` with a challenger but no score.json exits non-zero and writes nothing, so the loop retries the stage, and after the retries the next `plan` is failsafe.

## 10. Review round fixes (2026-11-22)

Five review findings were reproduced and handled. Nothing in this round touched rl_loop.py, the watchdog, the
metrics collector, the panel or PACE; no engine source was changed. Local runs used the int-shared fixture copy
(generator `C.onnx`, sha256 f4ac25e3…, ONNX Runtime Core ML backend, fp32, one worker) and `engine/build-balance`.

### What changed

1. **reanalyse.py read a dump format the engine does not write (blocker): FIXED.** See the driver paragraph at the
   end of §6. Reproduced first (`not a state line ('record_index')` after the seg00 replay). After the fix, on the
   fixture copy with the real engine:
   - `-k 2 -f 0.25`: 106 of 424 records reanalysed and verified record by record;
   - fidelity, `-k 1 -f 1 --replay-noise true`, root scans off in both the replay and the reanalysis: policy and
     value equal for **424/424** records; the same with `-f 0.25` in 4 shards of ≤ 30 states: **106/106**. So the
     subset files, the selection and the record alignment are exact.
   - The test stub now reads and writes the engine's real formats (header/game/record lines, records in file order,
     `reanalysed.tsv`) and refuses a file without a header or with a record before its game line, as the engine
     does.
2. **The controller could not reach a decision on a live run (blocker): PARTLY FIXED.**
   - Fixed: `search_balance.py realise RUN N [--accounting FILE]` writes `balance/itN/realised.json` from the
     segment marker (games, records, flag rate), `seat_clocks.tsv` (sum of the per-record node budgets = visits),
     `cap_counts.json`, and the self-play log's hybrid line (sit rate). GPU seconds come only from the job
     accounting (`balance/itN/accounting.json {segNN: gpu_s}`); without it no γ sample is taken rather than a guess.
   - Fixed: prior flags. `score_stage` false (the default) → no challenger is planned at all (mode `frozen`, reason
     says why) and `decide` records `none` without needing score.json. `search_q_blend` false → λ > 0 arms are
     illegal (the trainer has no `--search-q-blend`). `policy_positions` false → the plan's `steps_rule` is
     `positions` (there is no `train.policy_positions`).
   - Not built in this round, **built in the integration round after it**: the minitrain/score/probe/accounting
     stages (`hivemind balance …`, src/hivemind/cli/balance.py), `--search-q-blend` (§7) and
     `train.policy_positions` (cli/train.py), probe-mode self-play, and the rl_loop wiring (§9 flags). Until
     `balance/prior.json` sets `score_stage` (and the other measured flags), the controller still holds θ0 and
     only logs realised costs.
3. **σ̂ ignored the noise from which games were played (major): FIXED in the controller and the simulator.**
   - Each observation's se is now √(s²/seeds + τ̂²). τ̂ pools the prior's `tau_hat` (with `tau_df`) and the spread
     of each (incumbent, challenger) pair's d across iterations beyond its seed noise (rejected pairs' evidence is
     kept in `state.tau_pool` for this). No promotion until τ̂ has ≥ 3 df (`TAU_DF_MIN`).
   - Mini-train seed s now trains on the games with index % 2 = s (plan `minitrain.data_half`), so a seed pair's
     spread includes data noise too. X and T share seeds, so their halves stay paired.
   - `Z_PROMOTE` 2.0 → 3.0 and no single-observation promotion (re-tuned in the sim with τ; 2.0/3.0 gave 12-38%
     false departures at τ = σ).
   - The prior's σ̂/τ̂ must come from a replicate round on different data: θ0 vs θ0 mini-trains on two θ0
     segments of each of R = 10 past iterations, two seeds each (≈ 6.5 GPU-h at the §2 mini-train cost). The
     simulator now gives the controller that round's **estimates**, not the true σ, and adds `--tau`.
   - Synthetic decide case from the review (d = 0.105 then 0.02, tight seed pairs): before, Z = 20 and eligible;
     now τ̂ ≈ 0.06 from the swing itself and not eligible (test `test_between_data_noise_blocks_promotion`).
   - False departure from θ0 by iteration 20 (150 seeds, σ = 4), before → after:

     | τ | a_flat | b_theta0 |
     |---|---|---|
     | 0 | 5% → 0% | 2% → 0% |
     | 2 | 13% → 2% | 2% → 1% |
     | 4 | 38% → 3% | 22% → 2% |
     | 8 | 77% → 3% | 65% → 0% |

     The price is slower real gains: c_interior leaves θ0 by it20 in 94% (τ = 0) / 67% (τ = 4) / 30% (τ = 8) of
     runs, and its Elo/GPU-h at it8 is 83% / 79% of the oracle (it was 85% with the old, over-confident test).
4. **Self-play is not reproducible with root scans on (major): NOT FIXED in the engine; mitigated.** Reproduced
   again (`build-balance`, seed 1000, 2 games: .dst and games.pgn differ with the default root scans, identical
   with `--root-scans false`). Making the scans node-budgeted would not be enough: the scan thread runs beside the
   MCTS workers and stops the search when it proves a mate, so the race remains. A deterministic mode needs the
   scans run synchronously before the workers, a change inside agent.cc's search loop that this round did not
   attempt. Mitigations: (a) the fidelity gate only counts with root scans off (`fidelity_valid`), and passes there
   (424/424); (b) the live X/T pairing loss is part of τ, which the replicate round measures with live settings
   and the controller now uses.
5. **The probe favours deep arms whether or not they buy Elo (major): FIXED by masking.** f > 0 arms are illegal
   until `prior.probe_bias_measured` is set, after the bias measurement (θ0-only mini-nets trained with and without
   deep targets, scored on a probe from a different net or depth). With the flag off, f_proxy leaves θ0 in 0% (τ =
   0) / 2% (τ = 4) of runs. With the flag on, f_proxy still leaves θ0 in 14-43% by it8-it40, down from 52-74% (the
   stricter test helps but cannot remove a bias); that is exactly the case the measurement must rule out first.

### Simulation, all flags on, σ = 4, 100 seeds × 40 iterations

τ = 0:

Sim: 100 seeds x 40 iterations per surface, per-seed mini-train noise sigma = 4, shared per-iteration noise tau = 0 Elo-equivalent probe units; prior noise from a simulated 10-replicate round; prior flags ['probe_bias_measured', 'score_stage', 'search_q_blend'].

Elo per GPU-h as % of the oracle (oracle = best legal fixed recipe, per iteration on e_drift), at iteration 4/8/10/20/40:

| surface | controller (B) | TS baseline (A-lite) | fixed theta0 | fixed f1/8 k4 | fixed f1/4 k2 | fixed f1/4 k4 b200 |
|---|---|---|---|---|---|---|
| a_flat | 94.4/94.4/94.6/96.0/96.7 | 93.4/93.1/93.0/92.4/91.7 | 99.9/100.0/99.9/100.0/100.0 | 99.1/99.2/99.1/99.2/99.2 | 99.9/100.0/100.0/100.0/100.0 | 94.6/94.6/94.6/94.7/94.7 |
| b_theta0 | 93.6/94.2/94.5/96.0/96.7 | 86.3/86.2/86.0/85.9/85.9 | 100.0/100.0/100.0/100.0/100.0 | 90.0/90.0/90.1/90.1/90.1 | 93.5/93.5/93.5/93.6/93.5 | 52.6/52.6/52.6/52.7/52.6 |
| c_interior | 79.2/83.1/84.3/86.6/89.3 | 69.8/72.1/72.9/75.7/78.7 | 73.9/73.8/73.8/73.8/73.7 | 96.1/96.0/96.0/96.0/95.9 | 87.1/87.0/87.0/87.0/86.9 | 77.4/77.3/77.3/77.3/77.2 |
| d_edge_k8 | 77.0/78.5/79.2/80.8/82.6 | 71.7/73.3/74.1/76.8/78.8 | 76.0/75.9/75.9/75.9/75.9 | 92.3/92.3/92.3/92.3/92.3 | 87.5/87.5/87.5/87.5/87.5 | 71.2/71.1/71.1/71.1/71.1 |
| e_drift | 93.2/92.0/90.9/84.6/79.8 | 86.6/85.1/84.0/79.6/79.8 | 99.1/97.0/95.5/87.4/80.0 | 95.4/97.2/97.6/97.7/96.7 | 96.2/96.5/96.2/93.2/89.8 | 60.6/64.2/65.6/70.9/74.3 |
| f_proxy | 92.2/91.1/91.2/91.9/92.7 | 84.9/84.1/83.7/82.1/80.1 | 100.1/100.0/100.0/100.1/100.0 | 90.1/90.1/90.1/90.2/90.1 | 93.6/93.5/93.5/93.6/93.6 | 52.7/52.6/52.6/52.7/52.6 |
| g_lambda | 75.9/76.9/77.7/82.4/86.6 | 71.1/72.4/72.8/75.2/78.5 | 81.4/81.2/81.2/81.1/81.1 | 73.3/73.1/73.1/73.0/73.0 | 76.1/76.0/75.9/75.9/75.8 | 42.8/42.7/42.7/42.7/42.7 |

| surface | left theta0 by it4/8/10/20/40 | theta* = optimum at end of it8/20/40 | share of the oracle's gain over theta0 captured, it8/20/40 | median first iteration at optimum (share of runs) | promotions/run | reverts/run | overhead mean / max |
|---|---|---|---|---|---|---|---|
| a_flat | 0%/0%/0%/0%/1% | 0%/0%/0% | -/-/- | - (0%) | 0.01 | 0.00 | 3.2% / 7.3% |
| b_theta0 | 0%/0%/0%/0%/1% | 100%/100%/99% | -/-/- | - (0%) | 0.01 | 0.00 | 3.1% / 7.3% |
| c_interior | 91%/93%/94%/94%/96% | 0%/0%/0% | 35%/49%/59% | - (0%) | 0.97 | 0.01 | 4.0% / 7.3% |
| d_edge_k8 | 57%/62%/64%/65%/73% | 0%/0%/0% | 11%/20%/28% | - (0%) | 0.76 | 0.02 | 3.9% / 7.3% |
| e_drift | 0%/0%/0%/5%/50% | 0%/0%/0% | -165%/-22%/-1% | - (0%) | 0.52 | 0.00 | 3.6% / 7.3% |
| f_proxy | 37%/40%/41%/42%/43% | 60%/58%/57% | -/-/- | - (0%) | 0.43 | 0.00 | 3.6% / 7.3% |
| g_lambda | 11%/25%/47%/59%/82% | 10%/44%/69% | -23%/7%/29% | 11 (69%) | 0.84 | 0.00 | 3.6% / 7.3% |

τ = 4 (shared per-iteration noise equal to σ):

Sim: 100 seeds x 40 iterations per surface, per-seed mini-train noise sigma = 4, shared per-iteration noise tau = 4 Elo-equivalent probe units; prior noise from a simulated 10-replicate round; prior flags ['probe_bias_measured', 'score_stage', 'search_q_blend'].

Elo per GPU-h as % of the oracle (oracle = best legal fixed recipe, per iteration on e_drift), at iteration 4/8/10/20/40:

| surface | controller (B) | TS baseline (A-lite) | fixed theta0 | fixed f1/8 k4 | fixed f1/4 k2 | fixed f1/4 k4 b200 |
|---|---|---|---|---|---|---|
| a_flat | 94.2/94.4/94.6/96.0/96.7 | 93.4/93.1/93.0/92.4/91.7 | 99.9/100.0/99.9/100.0/100.0 | 99.1/99.2/99.1/99.2/99.2 | 99.9/100.0/100.0/100.0/100.0 | 94.6/94.6/94.6/94.7/94.7 |
| b_theta0 | 93.5/93.9/94.2/95.8/96.5 | 86.3/86.2/86.0/85.9/85.9 | 100.0/100.0/100.0/100.0/100.0 | 90.0/90.0/90.1/90.1/90.1 | 93.5/93.5/93.5/93.6/93.5 | 52.6/52.6/52.6/52.7/52.6 |
| c_interior | 76.4/78.9/79.6/81.5/83.8 | 69.8/72.1/72.9/75.7/78.7 | 73.9/73.8/73.8/73.8/73.7 | 96.1/96.0/96.0/96.0/95.9 | 87.1/87.0/87.0/87.0/86.9 | 77.4/77.3/77.3/77.3/77.2 |
| d_edge_k8 | 76.0/76.8/77.2/78.4/79.8 | 71.7/73.3/74.1/76.8/78.8 | 76.0/75.9/75.9/75.9/75.9 | 92.3/92.3/92.3/92.3/92.3 | 87.5/87.5/87.5/87.5/87.5 | 71.2/71.1/71.1/71.1/71.1 |
| e_drift | 93.2/91.8/90.7/84.6/79.3 | 86.6/85.1/84.0/79.6/79.8 | 99.1/97.0/95.5/87.4/80.0 | 95.4/97.2/97.6/97.7/96.7 | 96.2/96.5/96.2/93.2/89.8 | 60.6/64.2/65.6/70.9/74.3 |
| f_proxy | 92.7/92.6/92.7/94.3/95.0 | 84.9/84.1/83.7/82.1/80.1 | 100.1/100.0/100.0/100.1/100.0 | 90.1/90.1/90.1/90.2/90.1 | 93.6/93.5/93.5/93.6/93.6 | 52.7/52.6/52.6/52.7/52.6 |
| g_lambda | 75.9/76.5/76.9/80.2/83.7 | 71.1/72.4/72.8/75.2/78.5 | 81.4/81.2/81.2/81.1/81.1 | 73.3/73.1/73.1/73.0/73.0 | 76.1/76.0/75.9/75.9/75.8 | 42.8/42.7/42.7/42.7/42.7 |

| surface | left theta0 by it4/8/10/20/40 | theta* = optimum at end of it8/20/40 | share of the oracle's gain over theta0 captured, it8/20/40 | median first iteration at optimum (share of runs) | promotions/run | reverts/run | overhead mean / max |
|---|---|---|---|---|---|---|---|
| a_flat | 2%/4%/4%/5%/8% | 0%/0%/0% | -/-/- | - (0%) | 0.10 | 0.05 | 3.2% / 7.3% |
| b_theta0 | 0%/1%/1%/2%/4% | 99%/98%/96% | -/-/- | - (0%) | 0.04 | 0.00 | 3.2% / 7.3% |
| c_interior | 56%/60%/64%/66%/76% | 0%/0%/0% | 19%/29%/38% | 36 (1%) | 0.77 | 0.01 | 3.9% / 7.3% |
| d_edge_k8 | 36%/41%/42%/44%/56% | 0%/0%/0% | 4%/10%/16% | - (0%) | 0.61 | 0.01 | 3.8% / 7.3% |
| e_drift | 1%/3%/3%/7%/39% | 1%/0%/0% | -172%/-22%/-3% | 6 (1%) | 0.40 | 0.01 | 3.5% / 7.3% |
| f_proxy | 12%/14%/17%/17%/21% | 86%/84%/80% | -/-/- | - (0%) | 0.22 | 0.02 | 3.4% / 7.3% |
| g_lambda | 5%/15%/26%/41%/61% | 4%/30%/50% | -25%/-5%/14% | 18.0 (50%) | 0.64 | 0.02 | 3.6% / 7.3% |

Reproduce: `.venv/bin/python cluster/search_balance.py simulate --iterations 40 --seeds 100 --sigma 4 --tau 4
--markdown` (about a minute of CPU).

### Before integration (in addition to §9)

1. Build the minitrain/score stages and set `score_stage`; until then the controller only logs.
2. Run the replicate round on PACE (with the user's approval) and write `sigma_hat`, `tau_hat`, `tau_df` into
   `balance/prior.json`.
3. Measure the probe bias; only then set `probe_bias_measured`.
4. Have the loop write `balance/itN/accounting.json` (sacct GPU seconds per segment) and call `search_balance.py
   realise RUN N` before `decide`.


### Deployment review fixes (2026-11-22, before hybrid1 it2)

- `decide` takes `--prior` (rl-loop passes `--search-balance-prior` to `decide` and `decide --dry`; without it decide
  reads the plan's `prior_path`, else `balance/prior.json`), so plan and decide read the same prior.
- `decide --no-score`: when a challenger's minitrain or score stage was skipped after its attempts, rl-loop runs
  decide once with it. The result is `none: no mini-train scores: no evidence, no move`, and the overhead spent
  (`overhead.json`), the realised costs and a probe version that was built (its `balance_probe.done` marker) still
  reach the state, so the next plan is not failsafe and the probe is not rebuilt.
- The overhead ledger takes the iteration's GPU-hours from `metrics/itN.json` only when that record is `done`, net of
  the iteration's balance overhead (the collector counts it already); otherwise the nominal `ITER_GPU_H`. At decide
  time the iteration is not done yet, so in practice the nominal figure is used.
- rl-loop checks the `accounting` call's status and logs a failure.
- `wave2_iterate.sbatch BALANCE=1`: on a main job a failed precondition (build-sp1 rebuild, no `--deep-fraction`, no
  controller file, rl-loop without `--search-balance`) turns balance off for that job with a `WARNING search balance
  OFF` line instead of failing it; a broken engine binary still fails the job.
- `launch_full_run.sh REPLACE_PENDING=1` submits the new chain first and cancels the old pending jobs only after every
  sbatch succeeded (singleton keeps the new jobs behind them); `launch_settings.txt` is rewritten after that.
