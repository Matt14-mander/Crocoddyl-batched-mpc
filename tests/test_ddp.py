from dataclasses import replace

import pytest
import torch

from crocoddyl_batched_mpc import BatchedMPC, DDPProblem, LQRProblem, MPCController, SolveStatus


class LinearDynamics:
    def __init__(self, A: torch.Tensor, B: torch.Tensor) -> None:
        self.A, self.B = A, B
        self.nx, self.nu = A.shape[0], B.shape[1]
        self.dt = 1.0
        self.device, self.dtype = A.device, A.dtype

    def calc(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return x @ self.A.T + u @ self.B.T

    def calc_diff(self, x: torch.Tensor, u: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch = x.shape[0]
        return self.A.expand(batch, -1, -1), self.B.expand(batch, -1, -1)


class QuadraticCost:
    def __init__(self, Q: torch.Tensor, R: torch.Tensor, Qf: torch.Tensor) -> None:
        self.Q, self.R, self.Qf = Q, R, Qf
        self.nx, self.nu = Q.shape[0], R.shape[0]
        self.device, self.dtype = Q.device, Q.dtype

    def calc(self, x: torch.Tensor, u: torch.Tensor | None = None) -> torch.Tensor:
        matrix = self.Qf if u is None else self.Q
        value = 0.5 * (x * (x @ matrix.T)).sum(-1)
        if u is not None:
            value += 0.5 * (u * (u @ self.R.T)).sum(-1)
        return value

    def calc_diff(self, x: torch.Tensor, u: torch.Tensor | None = None):
        batch = x.shape[0]
        matrix = self.Qf if u is None else self.Q
        lx = x @ matrix.T
        lxx = matrix.expand(batch, -1, -1)
        if u is None:
            return lx, None, lxx, None, None
        lu = u @ self.R.T
        luu = self.R.expand(batch, -1, -1)
        lxu = x.new_zeros(batch, self.nx, self.nu)
        return lx, lu, lxx, luu, lxu


def linear_problems(
    *, batch: int = 3, horizon: int = 5, device: str = "cpu", max_iterations: int = 5
) -> tuple[LQRProblem, DDPProblem]:
    dtype = torch.float64
    A = torch.tensor([[1.0, 0.1], [0.0, 1.0]], device=device, dtype=dtype)
    B = torch.tensor([[0.005], [0.1]], device=device, dtype=dtype)
    Q = torch.diag(torch.tensor([2.0, 0.5], device=device, dtype=dtype))
    R = torch.tensor([[0.2]], device=device, dtype=dtype)
    Qf = 5.0 * Q
    lqr = LQRProblem.from_lti(A, B, Q, R, Qf, batch_size=batch, horizon=horizon)
    ddp = DDPProblem.from_models(
        LinearDynamics(A, B),
        QuadraticCost(Q, R, Qf),
        batch_size=batch,
        horizon=horizon,
        max_iterations=max_iterations,
        cost_tolerance=1e-8,
        gradient_tolerance=1e-8,
        regularization_init=0.0,
        regularization_min=0.0,
    )
    return lqr, ddp


def test_ddp_matches_exact_lqr_and_converges_per_environment():
    lqr, ddp = linear_problems()
    x0 = torch.tensor([[1.0, -0.2], [-0.5, 0.4], [0.2, 0.7]], dtype=torch.float64)
    exact = BatchedMPC(lqr).solve(x0)
    result = BatchedMPC(ddp).solve(x0)
    assert result.success.all()
    torch.testing.assert_close(result.xs, exact.xs, atol=1e-9, rtol=1e-9)
    torch.testing.assert_close(result.us, exact.us, atol=1e-9, rtol=1e-9)
    torch.testing.assert_close(result.cost, exact.cost, atol=1e-9, rtol=1e-9)
    assert (result.iterations == 2).all()


def test_nominal_states_are_rebuilt_from_current_x0():
    _, ddp = linear_problems(batch=1, horizon=3, max_iterations=1)
    stale_states = torch.full((1, 4, 2), 99.0, dtype=torch.float64)
    controls = torch.zeros(1, 3, 1, dtype=torch.float64)
    ddp = replace(ddp, x_init=stale_states, u_init=controls)
    x0 = torch.tensor([[0.4, -0.1]], dtype=torch.float64)
    result = BatchedMPC(ddp).solve(x0)
    torch.testing.assert_close(result.xs[:, 0], x0)
    expected_next = ddp.dynamics.calc(x0, result.us[:, 0])
    torch.testing.assert_close(result.xs[:, 1], expected_next)


def test_fixed_budget_returns_a_usable_best_candidate():
    _, ddp = linear_problems(batch=1, max_iterations=1)
    x0 = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
    result = BatchedMPC(ddp).solve(x0)
    assert result.status.item() == SolveStatus.MAX_ITERATIONS
    assert result.usable.item()
    assert result.action.abs().sum() > 0
    controller = MPCController(BatchedMPC(ddp))
    action, controlled = controller.compute(x0)
    torch.testing.assert_close(action, controlled.us[:, 0])


class NonfiniteDerivativeCost(QuadraticCost):
    def calc_diff(self, x: torch.Tensor, u: torch.Tensor | None = None):
        values = list(super().calc_diff(x, u))
        bad = x[:, 0] < 0
        values[0] = values[0].masked_fill(bad[:, None], float("nan"))
        return tuple(values)


def test_derivative_failure_is_isolated_and_never_raises_unbound_local():
    _, ddp = linear_problems(batch=2)
    ddp = replace(ddp, cost=NonfiniteDerivativeCost(ddp.cost.Q, ddp.cost.R, ddp.cost.Qf))
    result = BatchedMPC(ddp).solve(torch.tensor([[1.0, 0.0], [-1.0, 0.0]], dtype=torch.float64))
    assert result.status[0] == SolveStatus.SUCCESS
    assert result.status[1] == SolveStatus.NUMERICAL_FAILURE
    assert result.iterations.tolist() == [2, 1]
    assert torch.isfinite(result.us[0]).all() and torch.isnan(result.us[1]).all()


class RejectingCost(QuadraticCost):
    def calc(self, x: torch.Tensor, u: torch.Tensor | None = None) -> torch.Tensor:
        return x.new_zeros(x.shape[0])

    def calc_diff(self, x: torch.Tensor, u: torch.Tensor | None = None):
        values = list(super().calc_diff(x, u))
        values[0] = torch.ones_like(values[0])
        if u is not None:
            values[1] = torch.ones_like(values[1])
        return tuple(values)


def test_line_search_rejection_exhausts_budget_without_numerical_failure():
    _, ddp = linear_problems(batch=1, max_iterations=3)
    ddp = replace(ddp, cost=RejectingCost(ddp.cost.Q, ddp.cost.R, ddp.cost.Qf))
    result = BatchedMPC(ddp).solve(torch.ones(1, 2, dtype=torch.float64))
    assert result.status.item() == SolveStatus.MAX_ITERATIONS
    assert result.iterations.item() == 3
    assert result.usable.item()
    assert torch.isfinite(result.us).all()


class RetryFactorizationCost(QuadraticCost):
    def calc_diff(self, x: torch.Tensor, u: torch.Tensor | None = None):
        values = list(super().calc_diff(x, u))
        if u is not None:
            values[3] = -torch.ones_like(values[3])
        return tuple(values)


def test_factorization_failure_increases_regularization_and_retries():
    dtype = torch.float64
    A, B = torch.eye(1, dtype=dtype), torch.zeros(1, 1, dtype=dtype)
    Q, R = torch.eye(1, dtype=dtype), torch.eye(1, dtype=dtype)
    problem = DDPProblem.from_models(
        LinearDynamics(A, B),
        RetryFactorizationCost(Q, R, Q),
        batch_size=1,
        horizon=1,
        max_iterations=4,
        regularization_init=0.1,
        regularization_min=0.1,
        regularization_max=10.0,
        regularization_factor=10.0,
    )
    result = BatchedMPC(problem).solve(torch.ones(1, 1, dtype=dtype))
    assert result.status.item() == SolveStatus.SUCCESS
    assert result.iterations.item() == 3


def test_problem_rejects_inconsistent_model_dtype():
    _, ddp = linear_problems(batch=1)
    bad_cost = QuadraticCost(ddp.cost.Q.float(), ddp.cost.R.float(), ddp.cost.Qf.float())
    with pytest.raises(ValueError, match="cost dtype"):
        replace(ddp, cost=bad_cost)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_ddp_cuda_device_inference_and_hot_path_has_no_host_sync():
    _, ddp = linear_problems(batch=4, horizon=2, device="cuda", max_iterations=2)
    controller = MPCController(BatchedMPC(ddp))
    state = torch.ones(4, 2, device="cuda", dtype=torch.float64)
    controller.compute(state)
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
        action, result = controller.compute(state)
    assert action.is_cuda and result.status.is_cuda
    forbidden = {
        "cudaStreamSynchronize",
        "cudaDeviceSynchronize",
        "aten::item",
        "aten::_local_scalar_dense",
    }
    assert not forbidden.intersection(event.key for event in profile.key_averages())
