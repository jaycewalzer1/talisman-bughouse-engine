"""Your definition of done for arena/players/scheduler.py. They all fail until you write it.

    .venv/bin/python -m pytest arena/tests_phase1 -q        # everything
    .venv/bin/python -m pytest arena/tests_phase1 -q -k comm   # one piece at a time

These live outside arena/tests so the main suite stays green while you work.
"""

from __future__ import annotations

import pytest
import torch

from arena.interfaces import Message, Observation
from arena.players.scheduler import (
    BASE_THINK_INDEX,
    END_WAIT_OPTIONS,
    FEATURE_SIZE,
    GATE_OPTIONS,
    START_EXPLORATION,
    TALK_OPTIONS,
    THINK_MULTIPLIERS,
    CommState,
    Features,
    SchedulerPolicy,
    exploration_prob,
    gate_mask,
    mixture_probs,
    plan_nodes,
    plausible_wait,
    reinforce_loss,
    sample_action,
    talk_mask,
    team_reward,
    to_tensor,
)

MOVE = GATE_OPTIONS.index("move")
WAIT_N = GATE_OPTIONS.index("wait_N")
QUIET = TALK_OPTIONS.index("quiet")
OFFER = TALK_OPTIONS.index("offer")


def features(**kw) -> Features:
    base = dict(
        eval=0.0, eval_change=0.0, search_passed=False, wants=(), can_offer=False,
        own_clock_ms=60_000, partner_clock_ms=60_000, own_opponent_clock_ms=60_000,
        partner_opponent_clock_ms=60_000, partner_waiting=False, offered_piece=None, offered_plies=0,
        requested_piece=None, holding=(), sits_in_row=0, nodes_used_fraction=0.0,
    )
    return Features(**{**base, **kw})


def obs(seat="A_white", inbox=(), to_move=None) -> Observation:
    return Observation(
        now_ms=1000, seat=seat, fens={"A": "a", "B": "b"},
        clocks_ms={"A_white": 1, "A_black": 1, "B_white": 1, "B_black": 1},
        to_move=to_move or {"A": "A_white", "B": "B_black"}, legal_moves=["a2a3"], inbox=inbox,
    )


def msg(token, **payload) -> Message:
    return Message("B_black", "A_white", token, 0, payload=payload)


def gate_probs(policy, f):
    logits = policy(to_tensor(f)).gate.masked_fill(~gate_mask(f), float("-inf"))
    return torch.softmax(logits, -1)


# ------------------------------------------------------------------------------------------ CommState

def test_comm_partner_waiting_set_by_stall_and_cleared_when_not_its_turn():
    c = CommState("A_white")
    c.update(obs())
    assert c.partner_waiting is False
    c.update(obs(inbox=(msg("stall"),)))
    assert c.partner_waiting is True
    c.update(obs())
    assert c.partner_waiting is True, "no new message and still the partner's turn: still waiting"
    c.update(obs(to_move={"A": "A_white", "B": "B_white"}))
    assert c.partner_waiting is False, "the partner has moved, so it is no longer waiting"


def test_comm_records_offers_and_requests():
    c = CommState("A_white")
    c.update(obs(inbox=(msg("have_N", piece="N", plies=2), msg("need_Q", piece="Q"))))
    assert (c.offered_piece, c.offered_plies, c.requested_piece) == ("N", 2, "Q")


def test_comm_ignores_status_messages():
    c = CommState("A_white")
    c.update(obs(inbox=(msg("ok", kind="status"),)))
    assert (c.partner_waiting, c.offered_piece, c.requested_piece) == (False, None, None)


# ------------------------------------------------------------------------------------------ to_tensor

def test_tensor_has_declared_shape_and_sane_scale():
    x = to_tensor(features(own_clock_ms=120_000, eval=-1.0, wants=("N",), holding=("P", "P")))
    assert FEATURE_SIZE > 0
    assert x.shape == (FEATURE_SIZE,)
    assert x.dtype == torch.float32
    assert torch.isfinite(x).all()
    assert x.abs().max() <= 5.0, "a 120000 ms clock must not reach the network as 120000"


@pytest.mark.parametrize("change", [
    dict(partner_waiting=True), dict(search_passed=True), dict(eval=0.5), dict(own_clock_ms=30_000),
    dict(partner_opponent_clock_ms=30_000), dict(offered_piece="N", offered_plies=2),
    dict(requested_piece="Q"), dict(wants=("N",)), dict(holding=("R",)), dict(can_offer=True),
    dict(sits_in_row=2),
], ids=lambda c: next(iter(c)))
def test_tensor_changes_with_each_input(change):
    assert not torch.equal(to_tensor(features()), to_tensor(features(**change))), \
        "if an input does not change the tensor, the policy can't see it"


# ------------------------------------------------------------------------------------------ masks

def test_gate_mask_forbids_waiting_for_a_piece_we_hold():
    m = gate_mask(features(holding=("N",)))
    assert m[MOVE] and not m[WAIT_N] and m[GATE_OPTIONS.index("wait_Q")]


def test_talk_mask_only_allows_offer_with_something_to_offer():
    assert not talk_mask(features(can_offer=False))[OFFER]
    assert talk_mask(features(can_offer=True))[OFFER]
    assert talk_mask(features())[QUIET]


# ------------------------------------------------------------------------------------------ policy

def test_policy_output_shapes():
    h = SchedulerPolicy()(to_tensor(features()))
    assert h.gate.shape == (len(GATE_OPTIONS),)
    assert h.talk.shape == (len(TALK_OPTIONS),)
    assert h.think.shape == (len(THINK_MULTIPLIERS),)
    assert h.end_wait.shape == (len(END_WAIT_OPTIONS),)


def test_untrained_policy_moves_like_stock_when_search_does_not_pass():
    assert gate_probs(SchedulerPolicy(), features(search_passed=False))[MOVE] > 0.95


def test_untrained_policy_waits_like_stock_when_search_passes():
    assert gate_probs(SchedulerPolicy(), features(search_passed=True, wants=("N",)))[MOVE] < 0.5


def test_untrained_policy_uses_the_base_budget_and_stays_quiet():
    h = SchedulerPolicy()(to_tensor(features()))
    assert torch.softmax(h.think, -1)[BASE_THINK_INDEX] > 0.9
    assert torch.softmax(h.talk, -1)[QUIET] > 0.9


def test_untrained_policy_does_not_depend_on_its_random_init():
    f = features(search_passed=True, eval=0.3)
    torch.manual_seed(1)
    a = gate_probs(SchedulerPolicy(), f)
    torch.manual_seed(2)
    b = gate_probs(SchedulerPolicy(), f)
    assert torch.allclose(a, b, atol=1e-6), "the correction should start at exactly zero"


def test_correction_can_learn_from_the_start():
    policy = SchedulerPolicy()
    policy(to_tensor(features())).gate.sum().backward()
    grads = [p.grad for p in policy.parameters() if p.grad is not None and p.grad.abs().sum() > 0]
    assert grads, "zeroing the correction must not cut off the gradient"


def test_talk_head_sees_the_gate_choice():
    # Train the talk head for one step so it isn't still at its zero start, then compare gate choices.
    policy = SchedulerPolicy()
    optimizer = torch.optim.SGD(policy.parameters(), lr=0.5)
    x = to_tensor(features())
    for _ in range(3):
        optimizer.zero_grad()
        policy(x, gate_choice=WAIT_N).talk[TALK_OPTIONS.index("need_N")].neg().backward()
        optimizer.step()
    assert not torch.allclose(policy(x, gate_choice=MOVE).talk, policy(x, gate_choice=WAIT_N).talk)


# ------------------------------------------------------------------------------------------ exploration

def test_plausible_wait_needs_an_offer_or_a_wanted_piece():
    assert not plausible_wait(features())
    assert plausible_wait(features(offered_piece="N", offered_plies=2))
    assert plausible_wait(features(wants=("N",)))


def test_exploration_starts_at_5_percent_and_falls():
    assert exploration_prob(0, 1000, True) == pytest.approx(START_EXPLORATION)
    assert START_EXPLORATION == pytest.approx(0.05)
    steps = [exploration_prob(s, 1000, True) for s in (0, 250, 500, 1000)]
    assert steps == sorted(steps, reverse=True) and steps[-1] < steps[0] and steps[-1] >= 0


def test_no_exploration_when_a_wait_is_not_plausible():
    assert exploration_prob(0, 1000, False) == 0


def test_mixture_is_a_distribution_that_counts_forced_actions():
    p = torch.tensor([0.9, 0.1, 0.0])
    explore = torch.tensor([False, True, True])
    q = mixture_probs(p, 0.1, explore)
    assert q.sum() == pytest.approx(1.0)
    assert q[0] == pytest.approx(0.81)
    assert q[1] == pytest.approx(0.09 + 0.05)
    assert q[2] == pytest.approx(0.05), "the policy gives this 0, but exploration can still choose it"
    assert torch.allclose(mixture_probs(p, 0.0, explore), p)


# ------------------------------------------------------------------------------------------ sampling

def test_sample_never_picks_a_masked_option():
    policy, rng = SchedulerPolicy(), torch.Generator().manual_seed(0)
    f = features(holding=("N",), wants=("Q",), can_offer=False)
    for _ in range(300):
        a = sample_action(policy, f, 0.5, rng)
        assert a.gate != WAIT_N and a.talk != OFFER


def test_sample_is_repeatable_with_the_same_generator():
    policy = SchedulerPolicy()
    f = features(wants=("N",))
    run = lambda: [(a.gate, a.talk, a.think) for a in
                   (sample_action(policy, f, 0.3, g) for g in [torch.Generator().manual_seed(5)] for _ in range(50))]
    assert run() == run()


def test_sample_rates_follow_the_mixture():
    # Search doesn't want to pass, so the policy moves about always. With epsilon 0.5 about half the gates
    # are forced waits.
    policy, rng = SchedulerPolicy(), torch.Generator().manual_seed(0)
    f = features(wants=("N",))
    n = sum(sample_action(policy, f, 0.5, rng).gate != MOVE for _ in range(2000))
    assert 0.42 < n / 2000 < 0.58


def test_log_prob_uses_the_mixture_and_adds_the_three_decisions():
    policy, rng = SchedulerPolicy(), torch.Generator().manual_seed(0)
    f = features(wants=("N",))
    x = to_tensor(f)
    a = sample_action(policy, f, 0.2, rng)
    h = policy(x)
    gate_p = torch.softmax(h.gate.masked_fill(~gate_mask(f), float("-inf")), -1)
    explore = gate_mask(f).clone()
    explore[MOVE] = False
    mixed = mixture_probs(gate_p, 0.2, explore)
    talk_h = policy(x, gate_choice=a.gate).talk.masked_fill(~talk_mask(f), float("-inf"))
    expected = (torch.log(mixed[a.gate])
                + torch.log_softmax(talk_h, -1)[a.talk]
                + torch.log_softmax(h.think, -1)[a.think])
    assert a.log_prob.item() == pytest.approx(expected.item(), abs=1e-5)


# ------------------------------------------------------------------------------------------ budget

def test_plan_nodes_charges_the_quick_look_to_the_same_budget():
    assert plan_nodes(100, BASE_THINK_INDEX, 16) == 84
    assert plan_nodes(100, THINK_MULTIPLIERS.index(4.0), 16) == 384
    assert plan_nodes(100, THINK_MULTIPLIERS.index(0.25), 16) == 9


def test_plan_nodes_never_drops_below_one():
    assert plan_nodes(10, THINK_MULTIPLIERS.index(0.25), 16) == 1


# ------------------------------------------------------------------------------------------ learning

@pytest.mark.parametrize("result,team,reward", [("T1", "T1", 1), ("T2", "T1", -1), ("T1", "T2", -1), ("draw", "T1", 0)])
def test_team_reward(result, team, reward):
    assert team_reward(result, team) == reward


def wait_prob_after_update(reward):
    torch.manual_seed(0)
    policy = SchedulerPolicy()
    f = features(wants=("N",))

    def p_wait():
        return gate_probs(policy, f)[WAIT_N].item()

    before = p_wait()
    optimizer = torch.optim.SGD(policy.parameters(), lr=0.1)
    log_p = torch.log(gate_probs(policy, f)[WAIT_N])
    reinforce_loss([log_p], reward).backward()
    optimizer.step()
    return before, p_wait()


def test_winning_makes_the_action_more_likely():
    before, after = wait_prob_after_update(1.0)
    assert after > before


def test_losing_makes_the_action_less_likely():
    before, after = wait_prob_after_update(-1.0)
    assert after < before


def test_baseline_is_subtracted_and_sums_over_decisions():
    a, b = torch.tensor(-1.0), torch.tensor(-2.0)
    assert reinforce_loss([a, b], 1.0).item() == pytest.approx(3.0)
    assert reinforce_loss([a, b], 1.0, baseline=1.0).item() == pytest.approx(0.0)
    assert reinforce_loss([a, b], -1.0).item() == pytest.approx(-3.0)
