"""Standing acceptance against the independent Pinocchio KKT plant."""

import pytest
import torch

from crocoddyl_batched_mpc import BatchedMPC, MPCController
from crocoddyl_batched_mpc.models import Go2StandingCost, Go2TorchDynamics, go2_standing_problem
from examples.go2_standing_mpc import run_standing


def test_standing_cost_includes_equilibrium_feedforward():
    model = Go2TorchDynamics()
    cost = Go2StandingCost(model)
    x = model.standing_state[None]
    u = cost.u_reference[None]
    torch.testing.assert_close(cost.calc(x, u), x.new_zeros(1), atol=1e-20, rtol=0)
    lx, lu, _, _, _ = cost.calc_diff(x, u)
    assert lx.abs().max() < 1e-12
    assert lu.abs().max() < 1e-12
    assert cost.calc(x, torch.zeros_like(u))[0] > 0.1


@pytest.mark.parametrize("backend", ["torch", "fddp"])
def test_go2_solver_warm_start_reset_and_invalid_environment_isolation(backend):
    problem = go2_standing_problem(3, horizon=2, max_iterations=1)
    controller = MPCController(BatchedMPC(problem, backend=backend))
    x = problem.dynamics.standing_state.expand(3, -1).clone()
    x[1, 3:7] = 0
    action, result = controller.compute(x)
    assert result.usable.tolist() == [True, False, True]
    assert torch.isfinite(action).all()
    torch.testing.assert_close(action[0], problem.cost.u_reference, atol=1e-8, rtol=1e-8)
    torch.testing.assert_close(action[2], action[0], atol=1e-10, rtol=1e-10)
    old = controller._warm_controls[0].clone()
    controller.reset(torch.tensor([False, True, False]))
    torch.testing.assert_close(controller._warm_controls[0], old)
    assert controller._warm_valid.tolist() == [True, False, True]
    assert controller._last_action[1].abs().max() == 0


@pytest.mark.crocoddyl
def test_go2_standing_recovers_and_holds_against_independent_plant():
    pytest.importorskip("pinocchio")
    report, state, history, _ = run_standing(plant="pinocchio")
    assert report["all_actions_usable"]
    assert max(report["last_five_state_error_max"]) < 0.015
    assert max(report["final_configuration_error"]) < 0.005
    assert max(report["final_velocity_norm"]) < 0.015
    assert max(report["final_foot_position_error_m"]) < 1e-4
    assert max(report["max_torque_limit_fraction"]) < 1
    assert min(report["min_normal_force_N"]) > 0
    assert max(report["max_tangential_normal_ratio"]) < 0.6
    assert max(report["max_contact_acceleration_residual"]) < 1e-9
    assert torch.isfinite(state).all()
    assert history[-5:].norm(dim=-1).max() < history[0].norm(dim=-1).min() / 5
    print(report)
