"""Acceptance gates for feasibility-driven DDP and dynamic gaps."""

from dataclasses import replace

import pytest
import torch

from crocoddyl_batched_mpc import BatchedMPC, MPCController, SolveStatus
from tests.test_ddp import linear_problems
from tests.test_m2_warm_start import RecordingBackend

pytestmark = [
    pytest.mark.ddp_acceptance,
    pytest.mark.m2_acceptance,
    pytest.mark.fddp_acceptance,
]


def infeasible_linear_problem(batch: int = 3, *, device: str = "cpu", max_iterations: int = 10):
    lqr, problem = linear_problems(
        batch=batch, horizon=5, device=device, max_iterations=max_iterations
    )
    generator = torch.Generator(device=device).manual_seed(7)
    xs = 2.0 * torch.randn(batch, 6, 2, generator=generator, device=device, dtype=torch.float64)
    us = torch.zeros(batch, 5, 1, device=device, dtype=torch.float64)
    return lqr, replace(problem, x_init=xs, u_init=us, gap_penalty=100.0)


def dynamic_gap(problem, xs, us):
    return torch.stack(
        [
            problem.manifold.diff(problem.dynamics.calc(xs[:, t], us[:, t]), xs[:, t + 1])
            for t in range(problem.horizon)
        ],
        dim=1,
    )


def test_fddp_closes_infeasible_linear_gaps_and_matches_exact_lqr():
    lqr, problem = infeasible_linear_problem()
    x0 = torch.tensor([[1.0, -0.2], [-0.5, 0.4], [0.2, 0.7]], dtype=torch.float64)
    exact = BatchedMPC(lqr).solve(x0)
    result = BatchedMPC(problem, backend="fddp").solve(x0)
    assert result.success.all() and result.feasible.all() and result.usable.all()
    assert dynamic_gap(problem, result.xs, result.us).abs().max() <= problem.gap_tolerance
    torch.testing.assert_close(result.xs, exact.xs, atol=1e-9, rtol=1e-9)
    torch.testing.assert_close(result.us, exact.us, atol=1e-9, rtol=1e-9)
    torch.testing.assert_close(result.cost, exact.cost, atol=1e-9, rtol=1e-9)


def test_fddp_uses_current_x0_instead_of_stale_initial_first_state():
    _, problem = infeasible_linear_problem(batch=1)
    problem.x_init[:, 0] = 99.0
    x0 = torch.tensor([[0.3, -0.4]], dtype=torch.float64)
    result = BatchedMPC(problem, backend="fddp").solve(x0)
    torch.testing.assert_close(result.xs[:, 0], x0)
    assert result.feasible.all()


def test_infeasible_budget_exhaustion_is_not_exposed_as_usable_action():
    _, problem = linear_problems(batch=1, horizon=5, max_iterations=1)
    x0 = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
    xs = torch.zeros(1, 6, 2, dtype=torch.float64)
    xs[:, 0] = x0
    problem = replace(
        problem,
        x_init=xs,
        u_init=torch.zeros(1, 5, 1, dtype=torch.float64),
        gap_penalty=1e-12,
    )
    result = BatchedMPC(problem, backend="fddp").solve(x0)
    assert result.status.item() == SolveStatus.MAX_ITERATIONS
    assert not result.feasible.item()
    assert not result.usable.item()
    assert result.action.eq(0).all()


def test_fddp_accepts_cost_increase_when_partial_step_contracts_gap():
    _, problem = linear_problems(batch=1, horizon=3, max_iterations=1)
    x0 = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
    xs = torch.zeros(1, 4, 2, dtype=torch.float64)
    xs[:, 0] = x0
    us = torch.zeros(1, 3, 1, dtype=torch.float64)
    problem = replace(problem, x_init=xs, u_init=us, gap_penalty=3.0)
    initial_cost = sum(problem.cost.calc(xs[:, t], us[:, t]) for t in range(3))
    initial_cost += problem.cost.calc(xs[:, -1], None)
    initial_gap = dynamic_gap(problem, xs, us).abs().max()

    result = BatchedMPC(problem, backend="fddp").solve(x0)
    final_gap = dynamic_gap(problem, result.xs, result.us).abs().max()
    assert result.cost > initial_cost
    torch.testing.assert_close(final_gap, 0.6 * initial_gap, atol=1e-12, rtol=1e-12)
    assert not result.feasible.item() and not result.usable.item()


def test_fddp_batch_with_different_gaps_matches_independent_solves():
    _, problem = infeasible_linear_problem()
    x0 = torch.tensor([[1.0, -0.2], [-0.5, 0.4], [0.2, 0.7]], dtype=torch.float64)
    batched = BatchedMPC(problem, backend="fddp").solve(x0)
    for index in range(problem.batch_size):
        single_problem = replace(
            problem,
            batch_size=1,
            x_init=problem.x_init[index : index + 1].clone(),
            u_init=problem.u_init[index : index + 1].clone(),
        )
        single = BatchedMPC(single_problem, backend="fddp").solve(x0[index : index + 1])
        assert batched.status[index] == single.status[0]
        assert batched.feasible[index] == single.feasible[0]
        assert batched.iterations[index] == single.iterations[0]
        torch.testing.assert_close(batched.xs[index], single.xs[0], atol=1e-9, rtol=1e-9)
        torch.testing.assert_close(batched.us[index], single.us[0], atol=1e-9, rtol=1e-9)


class RecordingStateBackend(RecordingBackend):
    uses_state_guess = True

    def __init__(self) -> None:
        super().__init__()
        self.initial_states = []

    def solve(self, problem, x0):
        self.initial_states.append(problem.x_init.clone())
        result = super().solve(problem, x0)
        return replace(result, feasible=torch.ones_like(result.status, dtype=torch.bool))


def test_controller_shifts_state_and_control_trajectory_for_fddp():
    _, problem = linear_problems(batch=2, horizon=3)
    backend = RecordingStateBackend()
    controller = MPCController(BatchedMPC(problem, backend=backend))
    first_state = torch.tensor([[1.0, 0.0], [0.5, -0.2]], dtype=torch.float64)
    controller.compute(first_state)
    next_state = torch.tensor([[0.9, -0.1], [0.4, -0.3]], dtype=torch.float64)
    controller.compute(next_state)
    torch.testing.assert_close(backend.initial_states[1][:, 0], next_state)
    torch.testing.assert_close(
        backend.initial_states[1][:, 1:-1],
        first_state[:, None].expand(-1, problem.horizon - 1, -1),
    )
    torch.testing.assert_close(
        backend.initial_controls[1], torch.ones_like(backend.initial_controls[1])
    )


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_fddp_cpu_cuda_agree_and_cuda_hot_path_does_not_sync():
    _, cpu_problem = infeasible_linear_problem(batch=4)
    _, gpu_problem = infeasible_linear_problem(batch=4, device="cuda")
    x0 = torch.tensor([[1.0, -0.2], [-0.5, 0.4], [0.2, 0.7], [-0.3, -0.1]], dtype=torch.float64)
    expected = BatchedMPC(cpu_problem, backend="fddp").solve(x0)
    solver = BatchedMPC(gpu_problem, backend="fddp")
    gpu_x0 = x0.cuda()
    solver.solve(gpu_x0)
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
        actual = solver.solve(gpu_x0)
    assert torch.equal(actual.status.cpu(), expected.status)
    assert torch.equal(actual.feasible.cpu(), expected.feasible)
    torch.testing.assert_close(actual.xs.cpu(), expected.xs, atol=1e-8, rtol=1e-8)
    torch.testing.assert_close(actual.us.cpu(), expected.us, atol=1e-8, rtol=1e-8)
    forbidden = {
        "cudaStreamSynchronize",
        "cudaDeviceSynchronize",
        "aten::item",
        "aten::_local_scalar_dense",
    }
    assert not forbidden.intersection(event.key for event in profile.key_averages())
