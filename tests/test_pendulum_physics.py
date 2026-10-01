"""Physical convention gates independent of matching analytical derivatives."""

import pytest
import torch

from crocoddyl_batched_mpc.models.pendulum import (
    PendulumDynamics,
    PendulumDynamicsParameters,
)

pytestmark = pytest.mark.ddp_acceptance


@pytest.mark.parametrize("batched_parameters", [False, True])
def test_gravity_restores_downward_and_destabilizes_upright(batched_parameters):
    parameters = (
        PendulumDynamicsParameters.create(4, dtype=torch.float64)
        if batched_parameters else None
    )
    dynamics = PendulumDynamics(dtype=torch.float64, parameters=parameters)
    epsilon = 0.01
    state = torch.tensor(
        [[epsilon, 0.0], [-epsilon, 0.0], [torch.pi + epsilon, 0.0],
         [torch.pi - epsilon, 0.0]], dtype=torch.float64
    )
    next_state = dynamics.calc(state, torch.zeros(4, 1, dtype=torch.float64))
    # A positive torque increases theta; gravity points toward theta=0 near down
    # and away from theta=pi near up, regardless of the solver's cost function.
    assert next_state[0, 1] < 0 < next_state[1, 1]
    assert next_state[3, 1] < 0 < next_state[2, 1]
    jacobian, _ = dynamics.calc_diff(state, torch.zeros(4, 1, dtype=torch.float64))
    assert (jacobian[:2, 1, 0] < 0).all()
    assert (jacobian[2:, 1, 0] > 0).all()


def test_torque_balances_gravity_and_damping_with_nonunit_parameters():
    parameters = PendulumDynamicsParameters.create(
        2, mass=[0.8, 1.7], length=[0.6, 1.2], damping=[0.1, 0.3],
        dtype=torch.float64,
    )
    dynamics = PendulumDynamics(parameters=parameters)
    state = torch.tensor([[0.7, 0.2], [-0.4, -0.3]], dtype=torch.float64)
    torque = (
        parameters.mass * parameters.gravity * parameters.length * state[:, 0].sin()
        + parameters.damping * state[:, 1]
    )[:, None]
    expected = state.clone()
    expected[:, 0] += dynamics.dt * state[:, 1]
    torch.testing.assert_close(dynamics.calc(state, torque), expected, atol=1e-14, rtol=0)
