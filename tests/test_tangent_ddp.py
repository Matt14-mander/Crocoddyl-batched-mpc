"""Solver gates with nx!=ndx; kinematic test models do not represent contact dynamics."""

from dataclasses import replace

import pytest
import torch

from crocoddyl_batched_mpc import BatchedMPC, DDPProblem, MPCController
from crocoddyl_batched_mpc.costs import ManifoldQuadraticCost
from crocoddyl_batched_mpc.ddp_utils import initialize_trajectory_lqr_linearization
from crocoddyl_batched_mpc.dynamics import tangent_dynamics_jacobians
from crocoddyl_batched_mpc.manifolds import FloatingBaseManifold, SO3Manifold

pytestmark = pytest.mark.ddp_acceptance


class LocalIntegrator:
    """Command all tangent velocities directly, solely to test local derivatives."""

    def __init__(self, manifold):
        self.manifold = manifold
        self.nx, self.ndx, self.nu = manifold.nx, manifold.ndx, manifold.ndx
        self.device, self.dtype, self.dt = torch.device("cpu"), torch.float64, 0.2

    def calc(self, x, u):
        return self.manifold.integrate(x, self.dt * u)

    def calc_diff(self, x, u):
        return tangent_dynamics_jacobians(self, self.manifold, x, u)


def make_problem(manifold, batch=2, horizon=4, iterations=5):
    reference = manifold.neutral(torch.zeros(manifold.nx, dtype=torch.float64))
    cost = ManifoldQuadraticCost(
        manifold,
        reference,
        torch.eye(manifold.ndx, dtype=reference.dtype),
        torch.eye(manifold.ndx, dtype=reference.dtype) * 0.1,
    )
    return DDPProblem.from_models(
        LocalIntegrator(manifold),
        cost,
        manifold=manifold,
        batch_size=batch,
        horizon=horizon,
        max_iterations=iterations,
    )


def test_tangent_dynamics_and_cost_derivatives_match_independent_perturbations():
    manifold = FloatingBaseManifold(1)
    problem = make_problem(manifold)
    torch.manual_seed(4)
    x = manifold.integrate(
        manifold.neutral(torch.zeros(2, manifold.nx, dtype=torch.float64)),
        torch.randn(2, manifold.ndx, dtype=torch.float64) * 0.3,
    )
    u = torch.randn(2, manifold.ndx, dtype=torch.float64) * 0.4
    fx, fu = problem.dynamics.calc_diff(x, u)
    lx, lu, lxx, _, _ = problem.cost.calc_diff(x, u)
    predicted = problem.dynamics.calc(x, u)
    state_columns, control_columns, cost_columns = [], [], []
    for index in range(manifold.ndx):
        step = torch.zeros_like(u)
        step[:, index] = 1e-6
        xp, xm = manifold.integrate(x, step), manifold.integrate(x, -step)
        state_columns.append(
            (
                manifold.diff(problem.dynamics.calc(xp, u), predicted)
                - manifold.diff(problem.dynamics.calc(xm, u), predicted)
            )
            / 2e-6
        )
        control_columns.append(
            (
                manifold.diff(problem.dynamics.calc(x, u + step), predicted)
                - manifold.diff(problem.dynamics.calc(x, u - step), predicted)
            )
            / 2e-6
        )
        cost_columns.append((problem.cost.calc(xp, u) - problem.cost.calc(xm, u)) / 2e-6)
    torch.testing.assert_close(fx, torch.stack(state_columns, -1), atol=3e-10, rtol=3e-8)
    torch.testing.assert_close(fu, torch.stack(control_columns, -1), atol=3e-10, rtol=3e-8)
    torch.testing.assert_close(lx, torch.stack(cost_columns, -1), atol=3e-10, rtol=3e-8)
    torch.testing.assert_close(lu, u * 0.1)
    assert (torch.linalg.eigvalsh(lxx) > 0).all()


@pytest.mark.parametrize("backend", ["torch", "fddp"])
def test_quaternion_ddp_reduces_cost_and_handles_infeasible_guess(backend):
    manifold = SO3Manifold()
    problem = make_problem(manifold)
    x0 = manifold.integrate(
        manifold.neutral(torch.zeros(2, 4, dtype=torch.float64)),
        torch.tensor([[0.4, -0.2, 0.1], [-0.1, 0.3, 0.2]], dtype=torch.float64),
    )
    initial = x0[:, None].expand(-1, problem.horizon + 1, -1).clone()
    guess = manifold.integrate(
        initial, torch.ones(2, problem.horizon + 1, 3, dtype=x0.dtype) * 0.15
    )
    problem = replace(problem, x_init=guess, gap_penalty=100)
    result = BatchedMPC(problem, backend=backend).solve(x0)
    assert result.usable.all()
    assert manifold.is_valid(result.xs).all()
    torch.testing.assert_close(result.xs[:, 0], x0)
    assert (result.cost < (problem.horizon + 1) * problem.cost.calc(x0, None)).all()
    for t in range(problem.horizon):
        torch.testing.assert_close(
            manifold.diff(
                result.xs[:, t + 1], problem.dynamics.calc(result.xs[:, t], result.us[:, t])
            ),
            torch.zeros(2, 3, dtype=x0.dtype),
            atol=1e-10,
            rtol=0,
        )
    for index in range(2):
        single = replace(problem, batch_size=1, x_init=guess[index : index + 1])
        oracle = BatchedMPC(single, backend=backend).solve(x0[index : index + 1])
        torch.testing.assert_close(result.xs[index], oracle.xs[0], atol=1e-10, rtol=1e-10)
        torch.testing.assert_close(result.us[index], oracle.us[0], atol=1e-10, rtol=1e-10)


def test_fddp_backward_pulls_derivatives_into_nonzero_gap_chart():
    problem = make_problem(SO3Manifold(), batch=1, horizon=1, iterations=1)
    m = problem.manifold
    x0 = m.integrate(
        problem.cost.reference[None], torch.tensor([[0.2, -0.1, 0.3]], dtype=problem.dtype)
    )
    xn = m.integrate(
        problem.cost.reference[None], torch.tensor([[-0.3, 0.4, 0.2]], dtype=problem.dtype)
    )
    xs, us = torch.stack((x0, xn), 1), torch.zeros(1, 1, 3, dtype=problem.dtype)
    backend = BatchedMPC(problem, backend="fddp").backend
    gaps = backend._compute_gaps(problem, xs, us)
    gains, offsets, _, ok, finite = backend._backward_pass(
        problem, xs, us, x0.new_full((1,), problem.regularization_init), gaps
    )
    jx, ju = [], []
    for index in range(3):
        step = torch.zeros(1, 3, dtype=problem.dtype)
        step[:, index] = 1e-6
        jx.append(
            (
                m.diff(problem.dynamics.calc(m.integrate(x0, step), us[:, 0]), xn)
                - m.diff(problem.dynamics.calc(m.integrate(x0, -step), us[:, 0]), xn)
            )
            / 2e-6
        )
        ju.append(
            (
                m.diff(problem.dynamics.calc(x0, us[:, 0] + step), xn)
                - m.diff(problem.dynamics.calc(x0, us[:, 0] - step), xn)
            )
            / 2e-6
        )
    jx, ju = torch.stack(jx, -1), torch.stack(ju, -1)
    vx, _, vxx, _, _ = problem.cost.calc_diff(xn, None)
    _, lu, _, luu, lxu = problem.cost.calc_diff(x0, us[:, 0])
    gradient = vx + (vxx @ gaps[:, 0, :, None]).squeeze(-1)
    qu = lu + (ju.transpose(-1, -2) @ gradient[..., None]).squeeze(-1)
    quu = luu + ju.transpose(-1, -2) @ vxx @ ju
    qux = lxu.transpose(-1, -2) + ju.transpose(-1, -2) @ vxx @ jx
    quu += problem.regularization_init * torch.eye(3, dtype=problem.dtype)
    assert ok.all() and finite.all()
    torch.testing.assert_close(
        offsets[:, 0], -torch.linalg.solve(quu, qu[..., None]).squeeze(-1), atol=1e-9, rtol=1e-8
    )
    torch.testing.assert_close(gains[:, 0], -torch.linalg.solve(quu, qux), atol=1e-9, rtol=1e-8)


def test_floating_state_controller_warm_start_reset_and_invalid_quaternion_isolation():
    problem = make_problem(FloatingBaseManifold(0), horizon=2, iterations=2)
    m = problem.manifold
    x = m.integrate(
        m.neutral(torch.zeros(2, m.nx, dtype=problem.dtype)),
        torch.ones(2, m.ndx, dtype=problem.dtype) * 0.05,
    )
    controller = MPCController(BatchedMPC(problem, backend="fddp"))
    action, result = controller.compute(x)
    assert result.usable.all()
    next_state = problem.dynamics.calc(x, action)
    controller.reset(torch.tensor([True, False]))
    _, second = controller.compute(next_state)
    assert second.usable.all()
    bad = next_state.clone()
    bad[0, 3:7] = 0
    _, third = controller.compute(bad)
    assert not third.usable[0] and third.usable[1]
    assert torch.isnan(third.xs[0]).all()
    xs, us = initialize_trajectory_lqr_linearization(problem, x, n_iterations=1)
    assert xs.shape == (2, 3, m.nx) and us.shape == (2, 2, m.ndx)
    assert m.is_valid(xs).all()


def test_problem_rejects_ambient_derivative_contract_for_quaternion_state():
    problem = make_problem(SO3Manifold())
    problem.dynamics.ndx = 4
    with pytest.raises(ValueError, match="dynamics ndx"):
        replace(problem)
