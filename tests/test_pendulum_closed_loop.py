"""Sustained MPC stabilization against an independently written pendulum plant."""

import pytest
import torch

from crocoddyl_batched_mpc import BatchedMPC, DDPProblem, MPCController
from crocoddyl_batched_mpc.models.pendulum import PendulumCost, PendulumDynamics

pytestmark = pytest.mark.ddp_acceptance


@pytest.mark.parametrize("backend", ["torch", "fddp"])
def test_swingup_holds_upright_and_recovers_from_local_disturbance(backend):
    dynamics = PendulumDynamics(dtype=torch.float64)
    problem = DDPProblem.from_models(
        dynamics, PendulumCost(dtype=torch.float64), batch_size=3, horizon=30,
        max_iterations=6, cost_tolerance=1e-6,
    )
    controller = MPCController(BatchedMPC(problem, backend=backend))
    state = torch.tensor([[0.0, 0.0], [0.4, -0.2], [-0.4, 0.2]], dtype=torch.float64)
    zero_control_state = state.clone()
    held_before_disturbance = torch.ones(3, dtype=torch.bool)
    held_after_disturbance = torch.ones(3, dtype=torch.bool)
    usable = torch.ones(3, dtype=torch.bool)

    def plant(x, action):
        # theta=0 is downward: gravity must oppose positive displacement.
        velocity = x[:, 1] + 0.05 * (-9.81 * x[:, 0].sin() - 0.1 * x[:, 1] + action[:, 0])
        return torch.stack((x[:, 0] + 0.05 * velocity, velocity), dim=-1)

    for step in range(100):
        if step == 60:
            # New measured state must override the shifted FDDP state guess.
            state[0] += state.new_tensor([0.2, -0.3])
        action, result = controller.compute(state)
        usable &= result.usable & torch.isfinite(action).all(-1)
        state = plant(state, action)
        zero_control_state = plant(zero_control_state, torch.zeros_like(action))
        stable = (
            torch.isfinite(state).all(-1)
            & ((state[:, 0] - torch.pi).abs() < 0.05)
            & (state[:, 1].abs() < 0.1)
        )
        if 40 <= step < 60:
            held_before_disturbance &= stable
        if step >= 80:
            held_after_disturbance &= stable

    assert usable.all(), "Every environment must produce a usable action on every cycle"
    assert held_before_disturbance.all(), "Hold upright for 20 cycles after swing-up"
    assert held_after_disturbance.all(), "Recover and hold upright after a local disturbance"
    assert (state[:, 0] - torch.pi).abs().max() < 0.002
    assert state[:, 1].abs().max() < 0.01
    assert (zero_control_state[:, 0] - torch.pi).abs().min() > 2.0
