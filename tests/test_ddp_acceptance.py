"""Executable correctness gates for the nonlinear DDP backend."""

import pytest
import torch

from crocoddyl_batched_mpc import BatchedMPC, DDPProblem, SolveStatus
from crocoddyl_batched_mpc.manifolds import EuclideanManifold
from crocoddyl_batched_mpc.models.pendulum import PendulumCost, PendulumDynamics
from tests.test_ddp import linear_problems

pytestmark = pytest.mark.ddp_acceptance


def make_pendulum(
    *, batch: int, horizon: int, device: str = "cpu", max_iterations: int = 25
) -> DDPProblem:
    dtype = torch.float64
    dynamics = PendulumDynamics(device=device, dtype=dtype, dt=0.05)
    cost = PendulumCost(device=device, dtype=dtype)
    return DDPProblem.from_models(
        dynamics,
        cost,
        batch_size=batch,
        horizon=horizon,
        manifold=EuclideanManifold(2),
        max_iterations=max_iterations,
        cost_tolerance=1e-6,
        gradient_tolerance=1e-6,
    )


def independent_rollout_and_cost(
    problem: DDPProblem, x0: torch.Tensor, us: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Recompute feasibility and objective without backend helper methods."""
    xs = x0.new_empty(problem.batch_size, problem.horizon + 1, problem.nx)
    xs[:, 0] = x0
    total = x0.new_zeros(problem.batch_size)
    for t in range(problem.horizon):
        total += problem.cost.calc(xs[:, t], us[:, t])
        xs[:, t + 1] = problem.dynamics.calc(xs[:, t], us[:, t])
    return xs, total + problem.cost.calc(xs[:, -1], None)


def control_stationarity_residual(
    problem: DDPProblem, xs: torch.Tensor, us: torch.Tensor
) -> torch.Tensor:
    """Independent first-order control residual from a discrete adjoint sweep."""
    costate = problem.cost.calc_diff(xs[:, -1], None)[0]
    residuals = []
    for t in reversed(range(problem.horizon)):
        Fx, Fu = problem.dynamics.calc_diff(xs[:, t], us[:, t])
        lx, lu, _, _, _ = problem.cost.calc_diff(xs[:, t], us[:, t])
        assert lu is not None
        residuals.append(lu + (Fu.transpose(-1, -2) @ costate[..., None]).squeeze(-1))
        costate = lx + (Fx.transpose(-1, -2) @ costate[..., None]).squeeze(-1)
    return torch.stack(residuals, dim=1).abs().flatten(1).amax(-1)


def test_pendulum_solution_meets_feasibility_cost_and_stationarity_gates():
    problem = make_pendulum(batch=1, horizon=40)
    x0 = torch.tensor([[0.2, 0.0]], dtype=torch.float64)
    zero_us = torch.zeros(1, problem.horizon, problem.nu, dtype=torch.float64)
    _, initial_cost = independent_rollout_and_cost(problem, x0, zero_us)

    result = BatchedMPC(problem).solve(x0)
    recomputed_xs, recomputed_cost = independent_rollout_and_cost(problem, x0, result.us)
    stationarity = control_stationarity_residual(problem, result.xs, result.us)
    angle_error = (result.xs[:, -1, 0] - torch.pi).abs()

    assert result.status.item() == SolveStatus.SUCCESS
    torch.testing.assert_close(result.xs, recomputed_xs, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(result.cost, recomputed_cost, atol=1e-10, rtol=1e-10)
    assert result.cost.item() <= 0.65 * initial_cost.item()
    assert stationarity.item() < 3e-3
    assert angle_error.item() < 0.05
    assert result.xs[0, -1, 1].abs().item() < 0.25


def test_nonlinear_batch_matches_independent_environment_solves():
    x0 = torch.tensor([[0.2, 0.0], [0.5, -0.1], [-0.4, 0.2]], dtype=torch.float64)
    batched = BatchedMPC(make_pendulum(batch=3, horizon=20, max_iterations=15)).solve(x0)
    assert batched.usable.all()
    for index in range(3):
        single = BatchedMPC(make_pendulum(batch=1, horizon=20, max_iterations=15)).solve(
            x0[index : index + 1]
        )
        assert batched.status[index] == single.status[0]
        assert batched.iterations[index] == single.iterations[0]
        torch.testing.assert_close(batched.xs[index], single.xs[0], atol=1e-9, rtol=1e-9)
        torch.testing.assert_close(batched.us[index], single.us[0], atol=1e-9, rtol=1e-9)
        torch.testing.assert_close(batched.cost[index], single.cost[0], atol=1e-9, rtol=1e-9)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_nonlinear_cpu_cuda_agree_within_float64_tolerance():
    x0 = torch.tensor([[0.2, 0.0], [0.5, -0.1]], dtype=torch.float64)
    cpu = BatchedMPC(make_pendulum(batch=2, horizon=15, max_iterations=12)).solve(x0)
    gpu = BatchedMPC(make_pendulum(batch=2, horizon=15, device="cuda", max_iterations=12)).solve(
        x0.cuda()
    )
    assert torch.equal(cpu.status, gpu.status.cpu())
    assert torch.equal(cpu.iterations, gpu.iterations.cpu())
    torch.testing.assert_close(cpu.xs, gpu.xs.cpu(), atol=1e-8, rtol=1e-8)
    torch.testing.assert_close(cpu.us, gpu.us.cpu(), atol=1e-8, rtol=1e-8)
    torch.testing.assert_close(cpu.cost, gpu.cost.cpu(), atol=1e-8, rtol=1e-8)


@pytest.mark.crocoddyl
def test_ddp_linear_solution_matches_crocoddyl_reference():
    pytest.importorskip("crocoddyl")
    lqr, ddp = linear_problems(batch=2, horizon=5)
    x0 = torch.tensor([[1.0, -0.2], [-0.5, 0.4]], dtype=torch.float64)
    reference = BatchedMPC(lqr, backend="crocoddyl").solve(x0)
    actual = BatchedMPC(ddp).solve(x0)
    assert reference.success.all() and actual.success.all()
    torch.testing.assert_close(actual.xs, reference.xs, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(actual.us, reference.us, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(actual.cost, reference.cost, atol=1e-6, rtol=1e-6)
