from dataclasses import replace

import pytest
import torch

from crocoddyl_batched_mpc import BatchedMPC, LQRProblem, MPCController, SolveStatus
from crocoddyl_batched_mpc.models import double_integrator


def random_problem(batch=3, horizon=4, nx=3, nu=2):
    gen = torch.Generator().manual_seed(42)

    def rand(*shape):
        return torch.randn(*shape, generator=gen, dtype=torch.float64)

    def spd(*shape):
        m = rand(*shape)
        return m @ m.transpose(-1, -2) + torch.eye(shape[-1])

    return LQRProblem(
        A=0.2 * rand(batch, horizon, nx, nx),
        B=rand(batch, horizon, nx, nu),
        Q=spd(batch, horizon, nx, nx),
        R=spd(batch, horizon, nu, nu),
        Qf=spd(batch, nx, nx),
        f=0.1 * rand(batch, horizon, nx),
        q=rand(batch, horizon, nx),
        r=rand(batch, horizon, nu),
        qf=rand(batch, nx),
    )


def dense_oracle(p, b, x0, reg):
    """Eliminate states and solve the full control Hessian, without Riccati recursion."""

    def objective(flat_u):
        controls = flat_u.reshape(p.horizon, p.nu)
        x, cost = x0, flat_u.new_zeros(())
        for t, u in enumerate(controls):
            cost = cost + 0.5 * (x @ p.Q[b, t] @ x + u @ p.R[b, t] @ u)
            cost = cost + p.q[b, t] @ x + p.r[b, t] @ u + 0.5 * reg * (u @ u)
            x = p.A[b, t] @ x + p.B[b, t] @ u + p.f[b, t]
        return cost + 0.5 * x @ p.Qf[b] @ x + p.qf[b] @ x

    zero = torch.zeros(p.horizon * p.nu, dtype=x0.dtype, requires_grad=True)
    gradient = torch.autograd.grad(objective(zero), zero)[0]
    hessian = torch.autograd.functional.hessian(objective, zero)
    optimum = torch.linalg.solve(hessian, -gradient)
    return optimum.reshape(p.horizon, p.nu), objective(optimum).detach()


@pytest.mark.parametrize("horizon", [1, 5])
@pytest.mark.parametrize("regularization", [0.0, 0.2])
def test_riccati_matches_independent_dense_optimum(horizon, regularization):
    p = random_problem(horizon=horizon)
    x0 = torch.tensor([[0.4, -0.2, 0.7]] * p.batch_size, dtype=p.A.dtype)
    result = BatchedMPC(p, regularization=regularization).solve(x0)
    assert result.success.all()
    for b in range(p.batch_size):
        us, cost = dense_oracle(p, b, x0[b], regularization)
        torch.testing.assert_close(result.us[b], us, rtol=1e-9, atol=1e-9)
        torch.testing.assert_close(result.cost[b], cost, rtol=1e-9, atol=1e-9)
    expected = (p.A @ result.xs[:, :-1, :, None] + p.B @ result.us[..., None]).squeeze(-1)
    torch.testing.assert_close(result.xs[:, 1:], expected + p.f)
    torch.testing.assert_close(result.xs[:, 0], x0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_batch_matches_separate_solves_and_preserves_inputs(dtype):
    p = double_integrator(batch_size=3, horizon=8, dtype=dtype)
    # Noncontiguous input and broadcast model views are supported.
    x0 = torch.arange(12, dtype=dtype).reshape(3, 4)[:, ::2]
    before = x0.clone()
    result = BatchedMPC(p).solve(x0.requires_grad_())
    for b in range(3):
        single = double_integrator(batch_size=1, horizon=8, dtype=dtype)
        expected = BatchedMPC(single).solve(x0[b : b + 1])
        torch.testing.assert_close(result.us[b], expected.us[0])
    torch.testing.assert_close(x0, before)
    assert result.us.dtype == dtype and not result.us.requires_grad
    assert result.status.dtype == torch.int64


def test_numerical_failure_is_per_environment_and_action_is_finite():
    p = double_integrator(batch_size=3, horizon=3)
    R = p.R.clone()
    R[1] = -1e6
    p = replace(p, R=R)
    x0 = torch.ones(3, 2)
    x0[2, 0] = float("nan")
    result = BatchedMPC(p).solve(x0)
    assert result.status.tolist() == [0, 1, 1]
    assert torch.isnan(result.us[1:]).all()
    assert torch.isfinite(result.action).all()
    assert (result.action[1:] == 0).all()


def test_nonfinite_model_is_not_silently_successful():
    p = double_integrator(batch_size=2)
    q = p.q.clone()
    q[1, 0, 0] = float("inf")
    result = BatchedMPC(replace(p, q=q)).solve(torch.ones(2, 2))
    assert result.status.tolist() == [SolveStatus.SUCCESS, SolveStatus.NUMERICAL_FAILURE]


def test_controller_hold_and_selective_reset():
    p = double_integrator(batch_size=3, horizon=3)
    controller = MPCController(BatchedMPC(p))
    state = torch.ones(3, 2)
    good_action, _ = controller.compute(state)
    controller.reset(torch.tensor([True, False, False]))
    state[:] = float("nan")
    fallback, result = controller.compute(state)
    assert not result.success.any()
    torch.testing.assert_close(fallback[0], torch.zeros(1))
    torch.testing.assert_close(fallback[1:], good_action[1:])
    controller.reset()
    fallback, _ = controller.compute(state)
    assert (fallback == 0).all()
    # Returning an action must not expose internal fallback storage.
    fallback.fill_(7)
    assert (controller.compute(state)[0] == 0).all()


@pytest.mark.parametrize(
    "bad", [torch.ones(2), torch.ones(1, 2), torch.ones(2, 2, dtype=torch.float64)]
)
def test_rejects_bad_state_contract(bad):
    with pytest.raises(ValueError):
        BatchedMPC(double_integrator(batch_size=2)).solve(bad)


def test_problem_validation():
    p = double_integrator(batch_size=2)
    with pytest.raises(ValueError, match="Q"):
        replace(p, Q=torch.eye(2))
    with pytest.raises(ValueError, match="float32"):
        replace(p, A=p.A.half())
    with pytest.raises(ValueError):
        double_integrator(horizon=0)
    with pytest.raises(ValueError, match="Unknown backend"):
        BatchedMPC(p, backend="cuda")
    with pytest.raises(ValueError, match="regularization"):
        BatchedMPC(p, regularization=-0.1)
    with pytest.raises(TypeError, match="bool"):
        MPCController(BatchedMPC(p)).reset(torch.tensor([0, 1]))


def test_closed_loop_reduces_error():
    p = double_integrator(batch_size=4)
    solver = BatchedMPC(p)
    state = torch.tensor([[1.0, 0.0]] * 4)
    for _ in range(100):
        result = solver.solve(state)
        assert result.success.all()
        state = (p.A[:, 0] @ state[..., None] + p.B[:, 0] @ result.action[..., None]).squeeze(-1)
    assert state[:, 0].abs().max() < 0.05


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cuda_matches_cpu_on_nondefault_stream(dtype):
    cpu = double_integrator(batch_size=16, horizon=5, dtype=dtype)
    x = torch.tensor([[0.7, -0.4]] * 16, dtype=dtype)
    expected = BatchedMPC(cpu).solve(x)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        gpu = double_integrator(batch_size=16, horizon=5, device="cuda", dtype=dtype)
        result = BatchedMPC(gpu).solve(x.cuda())
        action = result.action * 2  # Immediate consumer on the same stream.
    stream.synchronize()
    for name in ("xs", "us", "cost", "status", "iterations"):
        actual = getattr(result, name)
        assert actual.is_cuda
        torch.testing.assert_close(actual.cpu(), getattr(expected, name), rtol=2e-4, atol=2e-5)
    torch.testing.assert_close(action.cpu(), 2 * expected.action, rtol=2e-4, atol=2e-5)


@pytest.mark.crocoddyl
def test_crocoddyl_reference_matches_torch():
    pytest.importorskip("crocoddyl")
    p = random_problem(batch=2, horizon=5)
    state = torch.ones(2, p.nx, dtype=p.A.dtype)
    expected = BatchedMPC(p).solve(state)
    actual = BatchedMPC(p, backend="crocoddyl").solve(state)
    assert actual.success.all()
    torch.testing.assert_close(actual.us, expected.us, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(actual.xs, expected.xs, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(actual.cost, expected.cost, atol=1e-6, rtol=1e-6)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_hot_path_does_not_synchronize_with_host():
    solver = BatchedMPC(double_integrator(batch_size=16, horizon=3, device="cuda"))
    state = torch.ones(16, 2, device="cuda")
    solver.solve(state)  # Initialize libraries before profiling steady-state solve.
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
        solver.solve(state)
    forbidden = {
        "cudaStreamSynchronize",
        "cudaDeviceSynchronize",
        "aten::item",
        "aten::_local_scalar_dense",
    }
    assert not forbidden.intersection(event.key for event in profile.key_averages())
