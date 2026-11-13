"""Train the Phase 1 scheduler with REINFORCE on team wins, against the opponent pool.

    python -m arena.train_scheduler --iterations 200 --pairs 10 --workers 3 --out runs/phase1
    python -m arena.train_scheduler --iterations 200 --pairs 10 --workers 3 --out runs/phase1 --resume

Each iteration saves the policy, plays a pool schedule with it, replays the recorded decisions to get log
probabilities with gradients, and takes one optimizer step. Writes to --out: latest.pt (the policy), state.pt (to
resume), pool.json (the pool plus snapshots), train_log.jsonl (one line per iteration), and itNNNN/ (that
iteration's games). The pool file passed with --pool is never modified.

--features turns on the player's optional feature groups (arena/players/scheduler_player.py: tempo, mate, danger,
serve). --baseline eval judges each decision against what the engine expected at that moment instead of against
the average result, which takes most of the luck out of the reward.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import sys
from dataclasses import replace

import torch

from arena.players.scheduler import (
    MOVE_INDEX,
    QUIET_INDEX,
    SERVE_OPTIONS,
    START_EXPLORATION,
    STOCK_LOGIT,
    THINK_MULTIPLIERS,
    Features,
    SchedulerPolicy,
    action_log_prob,
    end_wait_log_prob,
    reinforce_loss,
    serve_log_prob,
    team_reward,
)
from arena.players.scheduler_player import FEATURE_GROUPS
from arena.pool import Pool, make_schedule, run_schedule

BASELINE_DECAY = 0.9
MAX_GRAD_NORM = 1.0


def features_from_dict(d: dict) -> Features:
    return Features(**{k: tuple(v) if isinstance(v, list) else v for k, v in d.items()})


def step_log_prob(policy: SchedulerPolicy, step: dict) -> torch.Tensor:
    """The log probability of one recorded decision, with gradients attached."""
    f = features_from_dict(step["features"])
    if step["type"] == "end_wait":
        return end_wait_log_prob(policy, f, step["choice"])
    if step["type"] == "serve":
        return serve_log_prob(policy, f, step["choice"])
    return action_log_prob(policy, f, step["gate"], step["talk"], step["think"], step["epsilon"])


def fit_eval_baseline(games: list[tuple[list[dict], float]]) -> tuple[float, float]:
    """(a, k) of the straight line reward = a + k * eval that best fits every recorded decision in the batch.
    eval is the engine's quick-look value at that decision, so a + k * eval is the result the engine expected."""
    xs = [s["features"]["eval"] for steps, _ in games for s in steps]
    ys = [reward for steps, reward in games for _ in steps]
    n = len(xs)
    if n < 2:
        return (sum(ys) / n if n else 0.0), 0.0
    mx, my = sum(xs) / n, sum(ys) / n
    var = sum((x - mx) ** 2 for x in xs)
    k = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / var if var > 1e-9 else 0.0
    return my - k * mx, k


def eval_baseline_loss(policy: SchedulerPolicy, steps: list[dict], reward: float, a: float, k: float) -> torch.Tensor:
    """REINFORCE for one seat's game with a baseline per decision: each action is credited with the reward minus
    what the engine expected when it was taken. The baseline depends only on the position, never on the action,
    so the gradient still points the right way on average; it only gets less noisy."""
    terms = [(reward - (a + k * s["features"]["eval"])) * step_log_prob(policy, s) for s in steps]
    return -torch.stack(terms).sum()


def learn(policy: SchedulerPolicy, optimizer: torch.optim.Optimizer, games: list[tuple[list[dict], float]],
          baseline: float, eval_baseline: bool = False) -> dict:
    """One optimizer step from (recorded decisions, reward) pairs, one pair per seat per game."""
    extra = {}
    if eval_baseline:
        a, k = fit_eval_baseline(games)
        losses = [eval_baseline_loss(policy, steps, reward, a, k) for steps, reward in games if steps]
        extra = {"baseline_slope": round(k, 4)}
    else:
        losses = [reinforce_loss([step_log_prob(policy, s) for s in steps], reward, baseline)
                  for steps, reward in games if steps]
    if not losses:
        return {"loss": 0.0, "grad_norm": 0.0}
    loss = torch.stack(losses).mean()
    optimizer.zero_grad()
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), MAX_GRAD_NORM)
    optimizer.step()
    return {"loss": round(loss.item(), 4), "grad_norm": round(float(grad_norm), 4), **extra}


def prepare_specs(specs: list[dict], policy_path: str, iteration: int, total: int, trajectory_dir: str,
                  start_exploration: float = START_EXPLORATION) -> None:
    """Point each game's contender at the current policy and tell it where to record its decisions."""
    for s in specs:
        team = s["teams"][s["contender_team"]]
        team["path"] = policy_path
        team["player_kw"].update(train_step=iteration, total_steps=total, start_exploration=start_exploration,
                                 trajectory_prefix=os.path.join(trajectory_dir, s["game_id"]))


def collect_games(specs: list[dict], summaries: list[dict], trajectory_dir: str) -> tuple[list[tuple[list[dict], float]], list[float]]:
    """(decisions, reward) per contender seat, and one reward per finished game. Crashed games are left out."""
    by_id = {s["game_id"]: s for s in specs}
    games, rewards = [], []
    for summary in summaries:
        if "error" in summary:
            continue
        reward = team_reward(summary["result"], by_id[summary["game_id"]]["contender_team"])
        rewards.append(reward)
        for path in sorted(glob.glob(os.path.join(trajectory_dir, f"{summary['game_id']}_*.json"))):
            with open(path) as f:
                games.append((json.load(f)["steps"], reward))
    return games, rewards


def behavior(games: list[tuple[list[dict], float]]) -> dict:
    """How often the policy chose each kind of action, to see what training is changing."""
    decides = [s for steps, _ in games for s in steps if s["type"] == "decide"]
    n = len(decides) or 1
    moves = [s for s in decides if s["gate"] == MOVE_INDEX]
    serves = [s for steps, _ in games for s in steps if s["type"] == "serve"]
    mates = [s for s in decides if s["features"].get("mate_with")]
    return {"decisions": len(decides), "wait_rate": round(sum(s["gate"] != MOVE_INDEX for s in decides) / n, 4),
            "talk_rate": round(sum(s["talk"] != QUIET_INDEX for s in decides) / n, 4),
            "end_waits": sum(s["type"] == "end_wait" for steps, _ in games for s in steps),
            # the average think budget on moves, as a multiple of the normal one
            "think_mean": round(sum(THINK_MULTIPLIERS[s["think"]] for s in moves) / (len(moves) or 1), 3),
            # decisions where one more piece gave a forced mate, and how often the policy then waited
            "mate_seen": len(mates), "mate_waited": sum(s["gate"] != MOVE_INDEX for s in mates),
            # chances to play a capture the partner asked for, and how often it did
            "serve_offers": len(serves), "served": sum(SERVE_OPTIONS[s["choice"]] == "serve" for s in serves)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m arena.train_scheduler", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool", default="arena/pools/phase1.json")
    ap.add_argument("--iterations", type=int, default=200, help="total iterations, including any being resumed")
    ap.add_argument("--pairs", type=int, default=10, help="opening pairs per iteration (2 games each)")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--snapshot-every", type=int, default=20, help="add a frozen copy to the pool this often; 0 = never")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--base-ms", type=int, default=60000)
    ap.add_argument("--channel-delay-ms", type=int, default=0)
    ap.add_argument("--max-plies", type=int, default=400)
    ap.add_argument("--out", default="runs/phase1")
    ap.add_argument("--resume", action="store_true", help="continue from --out/state.pt")
    ap.add_argument("--features", default="", help=f"comma-separated feature groups from {', '.join(FEATURE_GROUPS)}")
    ap.add_argument("--exploration", type=float, default=START_EXPLORATION,
                    help="chance of a forced wait where one is plausible, at the first iteration")
    ap.add_argument("--think-logit", type=float, default=STOCK_LOGIT,
                    help="how strongly an untrained policy sticks to the normal think budget (6 = 98.5%%, 2 = 65%%)")
    ap.add_argument("--baseline", default="mean", choices=("mean", "eval"),
                    help="judge games against the running average result, or against the engine's eval per decision")
    ap.add_argument("--target", default="stock-centralized", help="opponent whose score gets its own log column")
    args = ap.parse_args(argv)
    features = [f for f in args.features.split(",") if f]
    if set(features) - set(FEATURE_GROUPS):
        ap.error(f"--features must come from {', '.join(FEATURE_GROUPS)}")

    os.makedirs(args.out, exist_ok=True)
    pool_path, state_path = os.path.join(args.out, "pool.json"), os.path.join(args.out, "state.pt")
    latest = os.path.abspath(os.path.join(args.out, "latest.pt"))
    torch.manual_seed(args.seed)
    policy = SchedulerPolicy(think_logit=args.think_logit)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)
    start, baseline = 0, 0.0
    pool = Pool.load(args.pool)
    if args.resume:
        state = torch.load(state_path, weights_only=True)
        policy.load_state_dict(state["policy"])
        optimizer.load_state_dict(state["optimizer"])
        start, baseline = int(state["iteration"]), float(state["baseline"])
        pool = Pool.load(pool_path)
    # Snapshots copy the contender's player settings, so a frozen copy plays with the features it was trained with.
    pool.contender = replace(pool.contender, kind="agent",
                             player_kw={**pool.contender.player_kw, "features": features})

    for iteration in range(start, args.iterations):
        iter_dir = os.path.join(args.out, f"it{iteration + 1:04d}")
        trajectory_dir = os.path.join(iter_dir, "trajectories")
        os.makedirs(trajectory_dir, exist_ok=True)
        torch.save(policy.state_dict(), latest)

        specs = make_schedule(pool, args.pairs, seed=args.seed + iteration, base_ms=args.base_ms,
                              channel_delay_ms=args.channel_delay_ms, max_plies=args.max_plies,
                              skip_pairs=iteration * args.pairs)
        prepare_specs(specs, latest, iteration, args.iterations, trajectory_dir, args.exploration)
        report = run_schedule(specs, iter_dir, workers=args.workers, n_boot=200, seed=args.seed)

        games, rewards = collect_games(specs, report["games"], trajectory_dir)
        stats = learn(policy, optimizer, games, baseline, eval_baseline=args.baseline == "eval")
        shutil.rmtree(trajectory_dir)
        if rewards:
            baseline = BASELINE_DECAY * baseline + (1 - BASELINE_DECAY) * sum(rewards) / len(rewards)

        target = report["results"]["by_opponent"].get(args.target)
        line = {"iteration": iteration + 1, "games": len(rewards), "errors": report["errors"],
                "score": round((sum(rewards) / len(rewards) + 1) / 2, 4) if rewards else None,
                "target_score": round(target["score"], 4) if target else None,
                "target_games": target["games"] if target else 0,
                "baseline": round(baseline, 4), **stats, **behavior(games), "wall_s": report["wall_s"]}
        with open(os.path.join(args.out, "train_log.jsonl"), "a") as f:
            f.write(json.dumps(line) + "\n")
        print(json.dumps(line), flush=True)

        if args.snapshot_every and (iteration + 1) % args.snapshot_every == 0:
            path = os.path.abspath(os.path.join(args.out, f"it{iteration + 1:04d}.pt"))
            torch.save(policy.state_dict(), path)
            pool.add_snapshot(f"it{iteration + 1:04d}", path)
        pool.save(pool_path)
        torch.save({"policy": policy.state_dict(), "optimizer": optimizer.state_dict(),
                    "iteration": iteration + 1, "baseline": baseline}, state_path)

    torch.save(policy.state_dict(), latest)
    print(f"done: {latest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
