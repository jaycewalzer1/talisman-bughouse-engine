# Plan: league self-play with learned time management (agreed 2026-11-25)

User: "ship it all". Phases 1 and 2 are approved to build and run; phase 3 waits for the phase 2 result.

## Why (evidence, 2026-11-25)
- tune r1: our clock_k 160 vs it13 at 110 scores 0.994; 1000 scores 0.907 with 16x fewer nodes; 55 scores 0.004.
  The faster team leads on the clock ~98% of turns, and under v3 only the side ahead may sit (SitGate / upstream
  can_sit), so it waits for pieces. The harness is symmetric (docs/fairness_audit_2026-11-25.md).
- This beats it13's fixed TIME MANAGEMENT. It does not yet show better PLAY at equal pace: runs/tune/chk1/sym160
  (both at 160, jobs 16153523) answers that. Early: sym160 4-0, rev_opp160 0-6 (n tiny).
- Training so far gained nothing: hybrid1 it1 flat; it2 passed the 100 ms gate (0.525) but scored 0.365 vs it13 at
  120 s with equal think time, +3.1 material, 12.1 pocket vs 8.6, mated on its own seat 63% (looks like hoarding).
- Plain self-play cannot teach the clock lead (both sides same pace, so lead is noise), and the 100 ms gate
  cannot see it. More of the current hybrid1 loop = noise.

## Phases
0. Read chk1 (sym160, rev_opp160): `python3 cluster/tune_status.py runs/tune/chk1`; sync to runs/study and rerun
   `python3 tools/study_games.py`. ~5 GPU-h, already running.
1. Gate + diagnosis (~30 GPU-h)
   - Real-clock two-score gate in rl-loop: (a) equal pace, both clock_k 110, 60-120 s; (b) free pace, our learned
     time policy vs it13 at its own pace. Promote only if neither drops. Warm ms/node measurement (4ffe3f6).
   - Hoarding diagnosis on hybrid1 it2 vs it0: drop rate, pocket holding time, value/policy targets on held-piece
     positions (study page, games in runs/study/full/hybrid1/anchors).
   - Fix the wave2 confound: opponent-sit estimate uses our own clock_k (arena/players/upstream.py:760-767, 910);
     estimate from the opponent's observed pace instead.
2. Pilot (~80-100 GPU-h): 2 iterations on twin-s it13 (narrow, hybrid config) with
   - per-game pace randomization: each team's clock_k log-uniform 80-400 in self-play;
   - think_gain head (v2 B) trained and used to choose per-move budget (learned time policy);
   - league opponents: past generators + frozen it13 at several fixed paces (AlphaStar-style mix);
   - two-score gate from phase 1.
   Engine pin must be d5802ee+ (raw-policy crash fix); PACE W2 checkout is at 0839774 and needs a build.
3. Scale (~2,500 GPU-h, about half the remaining budget) only if BOTH gate scores rise in phase 2. Use
   launch_full_run.sh WORKERS / GPU_HOURS cap; gpu_budget.py for the real rate.
4. Final match: two scores, warm-measured, plus reserve (~50 GPU-h).

Skip: durative tree work, a bigger network (speed is the currency), resuming hybrid1 as configured.

Headline aimed for: "beats the strongest public bughouse engine at equal speed, and learned its own time management,
which wins ~99% against it". If phase 2 shows no equal-pace gain, publish the time-management result honestly.

## State at handoff (2026-11-25)
- Pushed: fork talisman/wave2 d5802ee; main custom-engine d34e9dd (pin ccb8c1f, warm-measure launchers 4ffe3f6,
  study tool + page). PACE main pulled to 4ffe3f6 (not d34e9dd yet).
- hybrid1 stopped 11-24 00:05 (48.3 / 450 GPU-h). Nothing else queued except chk1.
- Study page: `cd viewer && npx vite --port 5180` -> http://localhost:5180/study.html
