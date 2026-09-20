"""External nonlinear FDDP oracle using the same semi-implicit pendulum model."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from crocoddyl_batched_mpc import BatchedMPC
from tests.test_ddp_acceptance import make_pendulum

pytestmark = pytest.mark.fddp_acceptance


def _initial_guess(problem, x0):
    us = torch.zeros(1, problem.horizon, 1, dtype=torch.float64)
    xs = torch.empty(1, problem.horizon + 1, 2, dtype=torch.float64)
    xs[:, 0] = x0
    for t in range(problem.horizon):
        xs[:, t + 1] = problem.dynamics.calc(xs[:, t], us[:, t])
    # Exercise FDDP's infeasible-trajectory branch without changing x[0].
    xs[:, 1:, 0] += 0.08
    xs[:, 1:, 1] -= 0.04
    return xs, us


def _gap_max(problem, xs, us):
    return max(
        np.max(
            np.abs(
                problem.dynamics.calc(
                    torch.as_tensor(xs[t : t + 1], dtype=torch.float64),
                    torch.as_tensor(us[t : t + 1], dtype=torch.float64),
                ).numpy()[0]
                - xs[t + 1]
            )
        )
        for t in range(problem.horizon)
    )


def _action_model_class(crocoddyl):
    class PendulumAction(crocoddyl.ActionModelAbstract):
        def __init__(self, *, terminal=False, state=None):
            super().__init__(crocoddyl.StateVector(2) if state is None else state, 1, 0)
            self.terminal = terminal
            self.dt = 0.05
            self.gravity = 9.81
            self.damping = 0.1
            self.reference = np.array([np.pi, 0.0])
            self.Q = np.diag([100.0, 10.0] if terminal else [10.0, 1.0])
            self.R = np.array([[0.1]])

        def calc(self, data, x, u=None):
            error = x - self.reference
            data.cost = 0.5 * error @ self.Q @ error
            if u is not None and not self.terminal:
                acceleration = self.gravity * np.sin(x[0]) - self.damping * x[1] + u[0]
                velocity = x[1] + self.dt * acceleration
                data.xnext[:] = [x[0] + self.dt * velocity, velocity]
                data.cost += 0.5 * u @ self.R @ u

        def calcDiff(self, data, x, u=None):
            data.Lx[:] = self.Q @ (x - self.reference)
            data.Lxx[:, :] = self.Q
            if u is not None and not self.terminal:
                dt = self.dt
                velocity_theta = dt * self.gravity * np.cos(x[0])
                velocity_velocity = 1 - dt * self.damping
                data.Fx[:, :] = [
                    [1 + dt * velocity_theta, dt * velocity_velocity],
                    [velocity_theta, velocity_velocity],
                ]
                data.Fu[:, :] = [[dt * dt], [dt]]
                data.Lu[:] = self.R @ u
                data.Luu[:, :] = self.R
                data.Lxu[:, :] = 0.0

    return PendulumAction


def _make_oracle(crocoddyl, x0, horizon):
    action = _action_model_class(crocoddyl)
    state = crocoddyl.StateVector(2)
    running = [action(state=state) for _ in range(horizon)]
    shooting = crocoddyl.ShootingProblem(x0, running, action(terminal=True, state=state))
    return crocoddyl.SolverFDDP(shooting)


def _solve_oracle(crocoddyl, x0, xs, us, iterations):
    solver = _make_oracle(crocoddyl, x0, len(us))
    solver.th_stop = 1e-9
    solver.solve([row.copy() for row in xs], [row.copy() for row in us], iterations, False, 1e-6)
    return solver


def _assert_action_values(model):
    problem = make_pendulum(batch=1, horizon=12)
    data = model.createData()
    x = np.array([2.6, -0.3])
    u = np.array([0.4])
    model.calc(data, x, u)
    model.calcDiff(data, x, u)
    tx = torch.from_numpy(x).unsqueeze(0)
    tu = torch.from_numpy(u).unsqueeze(0)
    fx, fu = problem.dynamics.calc_diff(tx, tu)
    lx, lu, lxx, luu, lxu = problem.cost.calc_diff(tx, tu)
    np.testing.assert_allclose(data.xnext, problem.dynamics.calc(tx, tu).numpy()[0], atol=1e-12)
    np.testing.assert_allclose(data.cost, problem.cost.calc(tx, tu).item(), atol=1e-12)
    for actual, expected in (
        (data.Fx, fx[0]),
        (data.Fu, fu[0]),
        (data.Lx, lx[0]),
        (data.Lu, lu[0]),
        (data.Lxx, lxx[0]),
        (data.Luu, luu[0]),
        (data.Lxu, lxu[0]),
    ):
        np.testing.assert_allclose(actual, expected.numpy(), atol=1e-12)


def _assert_terminal_values(model):
    problem = make_pendulum(batch=1, horizon=12)
    data = model.createData()
    x = np.array([2.6, -0.3])
    model.calc(data, x)
    model.calcDiff(data, x)
    tx = torch.from_numpy(x).unsqueeze(0)
    lx, _, lxx, _, _ = problem.cost.calc_diff(tx, None)
    np.testing.assert_allclose(data.cost, problem.cost.calc(tx, None).item(), atol=1e-12)
    np.testing.assert_allclose(data.Lx, lx[0].numpy(), atol=1e-12)
    np.testing.assert_allclose(data.Lxx, lxx[0].numpy(), atol=1e-12)


def test_oracle_action_math_without_crocoddyl_bindings():
    class FakeActionModelAbstract:
        def __init__(self, state, nu, nr):
            self.state, self.nu, self.nr = state, nu, nr

        def createData(self):
            return SimpleNamespace(
                xnext=np.zeros(2),
                cost=0.0,
                Fx=np.zeros((2, 2)),
                Fu=np.zeros((2, 1)),
                Lx=np.zeros(2),
                Lu=np.zeros(1),
                Lxx=np.zeros((2, 2)),
                Luu=np.zeros((1, 1)),
                Lxu=np.zeros((2, 1)),
            )

    fake = SimpleNamespace(ActionModelAbstract=FakeActionModelAbstract, StateVector=int)
    action = _action_model_class(fake)
    _assert_action_values(action())
    _assert_terminal_values(action(terminal=True))


@pytest.mark.crocoddyl
@pytest.mark.fddp_oracle
def test_pendulum_action_values_and_derivatives_match_torch():
    crocoddyl = pytest.importorskip("crocoddyl")
    action = _action_model_class(crocoddyl)
    _assert_action_values(action())
    _assert_terminal_values(action(terminal=True))


@pytest.mark.crocoddyl
@pytest.mark.fddp_oracle
def test_initial_infeasible_feedback_gains_match_crocoddyl():
    crocoddyl = pytest.importorskip("crocoddyl")
    problem = make_pendulum(batch=1, horizon=12, max_iterations=1)
    x0 = torch.tensor([[2.6, -0.3]], dtype=torch.float64)
    xs, us = _initial_guess(problem, x0)
    assert _gap_max(problem, xs[0].numpy(), us[0].numpy()) > 0.01
    with torch.enable_grad():
        gains, _, _, ok, finite = BatchedMPC(problem, backend="fddp").backend._backward_pass(
            problem,
            xs,
            us,
            torch.full((1,), problem.regularization_init, dtype=torch.float64),
            BatchedMPC(problem, backend="fddp").backend._compute_gaps(problem, xs, us),
        )
    assert ok.all() and finite.all()
    oracle = _solve_oracle(crocoddyl, x0[0].numpy(), xs[0].numpy(), us[0].numpy(), 1)
    oracle_gains = np.asarray(oracle.K)
    assert oracle_gains.shape == (problem.horizon, problem.nu, problem.nx)
    # Crocoddyl stores positive K and applies u -= K dx; Torch stores the signed update.
    np.testing.assert_allclose(gains[0].detach().numpy(), -oracle_gains, atol=3e-3, rtol=3e-3)


@pytest.mark.crocoddyl
@pytest.mark.fddp_oracle
def test_nonlinear_trajectory_and_budget_progress_against_crocoddyl():
    crocoddyl = pytest.importorskip("crocoddyl")
    print(f"Crocoddyl version: {getattr(crocoddyl, '__version__', 'unknown')}")
    base = make_pendulum(batch=1, horizon=12, max_iterations=25)
    x0 = torch.tensor([[2.6, -0.3]], dtype=torch.float64)
    xs, us = _initial_guess(base, x0)
    initial_gap = _gap_max(base, xs[0].numpy(), us[0].numpy())
    torch_results = []
    oracle_results = []
    for budget in (1, 2, 5, 25):
        problem = replace(base, x_init=xs, u_init=us, max_iterations=budget, gap_penalty=100.0)
        torch_results.append(BatchedMPC(problem, backend="fddp").solve(x0))
        oracle_results.append(
            _solve_oracle(crocoddyl, x0[0].numpy(), xs[0].numpy(), us[0].numpy(), budget)
        )

    torch_gaps = [
        _gap_max(base, result.xs[0].numpy(), result.us[0].numpy()) for result in torch_results
    ]
    oracle_gaps = [
        _gap_max(base, np.asarray(result.xs), np.asarray(result.us)) for result in oracle_results
    ]
    for budget, torch_result, oracle_result, torch_gap, oracle_gap in zip(
        (1, 2, 5, 25), torch_results, oracle_results, torch_gaps, oracle_gaps, strict=True
    ):
        print(
            f"budget={budget}: torch cost={torch_result.cost.item():.9g} "
            f"gap={torch_gap:.3g} iterations={torch_result.iterations.item()}; "
            f"crocoddyl cost={oracle_result.cost:.9g} gap={oracle_gap:.3g} "
            f"iteration_index={oracle_result.iter}"
        )
    assert torch_gaps[0] < initial_gap and oracle_gaps[0] < initial_gap
    assert all(later <= earlier + 1e-12 for earlier, later in zip(torch_gaps, torch_gaps[1:]))
    assert all(later <= earlier + 1e-12 for earlier, later in zip(oracle_gaps, oracle_gaps[1:]))
    assert torch_gaps[-1] <= base.gap_tolerance
    assert oracle_gaps[-1] <= base.gap_tolerance
    assert torch_results[-1].feasible.item()
    assert torch_results[-1].cost.item() <= torch_results[0].cost.item() + 1e-8
    assert oracle_results[-1].cost <= oracle_results[0].cost + 1e-8
    np.testing.assert_allclose(
        torch_results[-1].xs[0].numpy(), np.asarray(oracle_results[-1].xs), atol=3e-3, rtol=3e-3
    )
    np.testing.assert_allclose(
        torch_results[-1].us[0].numpy(), np.asarray(oracle_results[-1].us), atol=3e-3, rtol=3e-3
    )
    np.testing.assert_allclose(
        torch_results[-1].cost.item(), oracle_results[-1].cost, atol=3e-3, rtol=3e-3
    )
