"""The parts test_scheduler.py doesn't cover: the sit option, end-wait, the arena player and one training step.
The engine is faked, so these run anywhere."""

from __future__ import annotations

import json

import pytest
import torch

import arena.players.scheduler_player as sp
from arena.interfaces import Message, Observation
from arena.players.hivemind import SearchResult
from arena.players.scheduler import (
    BASE_THINK_INDEX,
    END_WAIT_OPTIONS,
    GATE_OPTIONS,
    MOVE_INDEX,
    QUIET_INDEX,
    SIT_INDEX,
    STOP_INDEX,
    TALK_OPTIONS,
    THINK_MULTIPLIERS,
    Action,
    CommState,
    Features,
    SchedulerPolicy,
    action_log_prob,
    end_wait_log_prob,
    sample_action,
    sample_end_wait,
    to_tensor,
)
from arena.players.setups import team_players
from arena.train_scheduler import learn, step_log_prob

START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR[] w KQkq - 0 1"
WITH_KNIGHT = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR[N] w KQkq - 0 1"
WAIT_N = GATE_OPTIONS.index("wait_N")
NEED_Q = TALK_OPTIONS.index("need_Q")


def features(**kw) -> Features:
    base = dict(
        eval=0.0, eval_change=0.0, search_passed=False, wants=(), can_offer=False,
        own_clock_ms=60_000, partner_clock_ms=60_000, own_opponent_clock_ms=60_000,
        partner_opponent_clock_ms=60_000, partner_waiting=False, offered_piece=None, offered_plies=0,
        requested_piece=None, holding=(), sits_in_row=0, nodes_used_fraction=0.0,
    )
    return Features(**{**base, **kw})


def obs(now=1000, fen=START, inbox=(), own_clock=60_000) -> Observation:
    return Observation(
        now_ms=now, seat="A_white", fens={"A": fen, "B": START},
        clocks_ms={"A_white": own_clock, "A_black": 60_000, "B_white": 60_000, "B_black": 60_000},
        to_move={"A": "A_white", "B": "B_black"}, legal_moves=["e2e4", "g1f3"], inbox=inbox,
    )


class FakeProcess:
    def __init__(self, move="e2e4", pv=()):
        self.move, self.pv, self.calls = move, list(pv), []
        self.released = False

    def search(self, fen_a, fen_b, *, team, nodes, time_advantage=False, require=None):
        self.calls.append({"nodes": nodes, "require": require})
        move = self.move if require is None else (self.move or "e2e4")
        return SearchResult(move, None, 0.1, 10, None, nodes, 1, pv=self.pv)

    def release(self):
        self.released = True


def player(process=None, **kw) -> sp.SchedulerPlayer:
    return sp.SchedulerPlayer("A_white", process or FakeProcess(), SchedulerPolicy(), nodes=100, ms_per_node=10.0, **kw)


def force(monkeypatch, gate=MOVE_INDEX, talk=QUIET_INDEX, think=BASE_THINK_INDEX):
    monkeypatch.setattr(sp, "sample_action", lambda *a: Action(gate, talk, think, torch.tensor(0.0)))


def force_end_wait(monkeypatch, name):
    monkeypatch.setattr(sp, "sample_end_wait", lambda *a: (END_WAIT_OPTIONS.index(name), torch.tensor(0.0)))


# ------------------------------------------------------------------------------------------ policy

def test_untrained_sits_silently_when_search_passes_and_no_piece_is_wanted():
    logits = SchedulerPolicy()(to_tensor(features(search_passed=True))).gate
    assert logits.argmax() == SIT_INDEX


def test_untrained_waits_for_the_wanted_piece():
    logits = SchedulerPolicy()(to_tensor(features(search_passed=True, wants=("N",)))).gate
    assert logits.argmax() == WAIT_N


def test_untrained_end_wait_prefers_stop():
    logits = SchedulerPolicy()(to_tensor(features())).end_wait
    assert logits.argmax() == STOP_INDEX


def test_none_gate_choice_is_different_from_every_real_choice_after_training():
    policy = SchedulerPolicy()
    optimizer = torch.optim.SGD(policy.parameters(), lr=0.02)
    x = to_tensor(features())
    for _ in range(3):
        optimizer.zero_grad()
        sum(policy(x, gate_choice=g).talk[NEED_Q] for g in range(len(GATE_OPTIONS))).neg().backward()
        optimizer.step()
    none = policy(x).talk
    assert not any(torch.allclose(none, policy(x, gate_choice=g).talk) for g in range(len(GATE_OPTIONS)))


def test_think_does_not_depend_on_the_gate_choice():
    policy = SchedulerPolicy()
    x = to_tensor(features())
    assert torch.equal(policy(x).think, policy(x, gate_choice=WAIT_N).think)


def test_replayed_log_prob_matches_the_sampled_one():
    policy, rng = SchedulerPolicy(), torch.Generator().manual_seed(3)
    f = features(wants=("N",), holding=("B",), can_offer=True)
    for _ in range(20):
        a = sample_action(policy, f, 0.3, rng)
        assert action_log_prob(policy, f, a.gate, a.talk, a.think, 0.3).item() == pytest.approx(a.log_prob.item())
    choice, lp = sample_end_wait(policy, f, rng)
    assert end_wait_log_prob(policy, f, choice).item() == pytest.approx(lp.item())


def test_gradients_are_finite_with_masked_options():
    policy, rng = SchedulerPolicy(), torch.Generator().manual_seed(1)
    f = features(holding=("N", "Q"), wants=("B",))
    sample_action(policy, f, 0.2, rng).log_prob.backward()
    assert all(torch.isfinite(p.grad).all() for p in policy.parameters() if p.grad is not None)


def test_old_offers_and_requests_expire():
    c = CommState("A_white")
    m = lambda token, **p: Message("B_black", "A_white", token, 0, payload=p)
    c.update(obs(now=1000, inbox=(m("have_N", piece="N", plies=2), m("need_Q", piece="Q"))))
    c.update(obs(now=10_000))
    assert (c.offered_piece, c.requested_piece) == ("N", "Q")
    c.update(obs(now=20_000))
    assert (c.offered_piece, c.offered_plies, c.requested_piece) == (None, 0, None)


# ------------------------------------------------------------------------------------------ player

def test_a_move_costs_the_chosen_budget_including_the_quick_look(monkeypatch):
    force(monkeypatch, think=THINK_MULTIPLIERS.index(4.0))
    process = FakeProcess()
    d = player(process, quick_nodes=16).decide(obs())
    assert [c["nodes"] for c in process.calls] == [16, 384]
    assert (d.kind, d.uci, d.nodes, d.think_ms) == ("move", "e2e4", 400, 4000)
    assert process.calls[1]["require"] == "A", "after the gate says move, the long search may not pass"


def test_waiting_for_a_piece_sends_need_and_stall_and_skips_the_long_search(monkeypatch):
    force(monkeypatch, gate=WAIT_N)
    process = FakeProcess()
    d = player(process, quick_nodes=16).decide(obs())
    assert len(process.calls) == 1
    assert d.kind == "wait" and d.stall_reason == "need_piece" and d.think_ms == 160
    assert [m.token for m in d.messages] == ["need_N", "stall"]


def test_sit_sends_only_stall(monkeypatch):
    force(monkeypatch, gate=SIT_INDEX)
    d = player().decide(obs())
    assert d.kind == "wait" and [m.token for m in d.messages] == ["stall"]


def test_quick_look_may_only_pass_under_stock_rules():
    process = FakeProcess()
    player(process).decide(obs())
    assert process.calls[0]["require"] == "A", "not ahead on time, so stock would have to move"
    process = FakeProcess()
    player(process).decide(obs(own_clock=70_000))
    assert process.calls[0]["require"] is None


def test_keep_waiting_costs_no_search(monkeypatch):
    force(monkeypatch, gate=WAIT_N)
    force_end_wait(monkeypatch, "keep_waiting")
    process = FakeProcess()
    p = player(process)
    p.decide(obs())
    d = p.decide(obs(now=6000))
    assert len(process.calls) == 1 and d.kind == "wait" and d.wait_ms == sp.WAIT_RECHECK_MS and d.nodes == 0
    assert [s["type"] for s in p.steps] == ["decide", "end_wait"]


def test_stop_goes_back_through_the_gate(monkeypatch):
    force(monkeypatch, gate=WAIT_N)
    force_end_wait(monkeypatch, "stop")
    process = FakeProcess()
    p = player(process)
    p.decide(obs())
    force(monkeypatch)
    assert p.decide(obs(now=6000)).kind == "move"
    assert [s["type"] for s in p.steps] == ["decide", "end_wait", "decide"]


def test_an_arriving_piece_ends_the_wait_without_asking(monkeypatch):
    force(monkeypatch, gate=WAIT_N)
    p = player()
    p.decide(obs())
    force(monkeypatch)
    assert p.decide(obs(now=2000, fen=WITH_KNIGHT)).kind == "move"
    assert [s["type"] for s in p.steps] == ["decide", "decide"]


def test_offer_carries_piece_and_plies(monkeypatch):
    force(monkeypatch, talk=TALK_OPTIONS.index("offer"))
    monkeypatch.setattr(sp, "pv_offers", lambda *a: [("R", 2)])
    have = player().decide(obs()).messages[0]
    assert have.token == "have_R" and have.payload == {"piece": "R", "plies": 2}


def test_one_send_per_instant_so_waiting_partners_cannot_wake_each_other_forever(monkeypatch):
    force(monkeypatch, gate=SIT_INDEX)
    force_end_wait(monkeypatch, "stop")
    p = player()
    assert p.decide(obs(now=1000)).messages
    assert p.decide(obs(now=1000)).messages == ()
    assert p.decide(obs(now=1001)).messages


def test_lending_doubles_the_budget_and_keeps_the_think_time(monkeypatch):
    force(monkeypatch)
    process = FakeProcess()
    d = player(process, quick_nodes=16, lend=True).decide(
        Observation(**{**obs().__dict__, "partner_machine_free": True}))
    assert sum(c["nodes"] for c in process.calls) == 200 and d.borrow and d.think_ms == 1000


def test_trajectory_is_written_on_close_and_replays(tmp_path, monkeypatch):
    process = FakeProcess(pv=[("g1f3", None)])
    p = player(process, trajectory_prefix=str(tmp_path / "g1"), train_step=0, total_steps=10)
    p.decide(obs())
    p.close()
    p.close()
    assert process.released
    saved = json.loads((tmp_path / "g1_A_white.json").read_text())
    assert saved["seat"] == "A_white" and len(saved["steps"]) == 1
    assert torch.isfinite(step_log_prob(p.policy, saved["steps"][0]))


def test_agent_team_only_builds_for_sharing():
    with pytest.raises(ValueError, match="sharing"):
        team_players("T1", "centralized", "agent", seed=0, nodes=200)
    players = team_players("T1", "sharing", "agent", seed=0, nodes=200, compute="throughput", lend=True)
    assert set(players) == {"A_white", "B_black"}
    assert all(p.kind == "agent" and p.nodes == 100 and p.lend for p in players.values())
    for p in players.values():
        p.process.close()


# ------------------------------------------------------------------------------------------ training

def recorded_wait() -> dict:
    return {"type": "decide", "features": json.loads(json.dumps(features(wants=("N",)).__dict__)),
            "gate": WAIT_N, "talk": QUIET_INDEX, "think": BASE_THINK_INDEX, "epsilon": 0.05}


def wait_probability(policy) -> float:
    return step_log_prob(policy, recorded_wait()).exp().item()


@pytest.mark.parametrize("reward,sign", [(1.0, 1), (-1.0, -1)])
def test_one_learning_step_moves_the_recorded_action_the_right_way(reward, sign):
    torch.manual_seed(0)
    policy = SchedulerPolicy()
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-2)
    before = wait_probability(policy)
    stats = learn(policy, optimizer, [([recorded_wait()], reward)], baseline=0.0)
    assert (wait_probability(policy) - before) * sign > 0 and stats["grad_norm"] > 0


def test_learning_from_no_decisions_is_a_no_op():
    policy = SchedulerPolicy()
    assert learn(policy, torch.optim.Adam(policy.parameters()), [([], 1.0)], 0.0)["loss"] == 0.0
