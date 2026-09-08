"""Test dynamics and cost derivatives using finite differences.

Validates that analytical Jacobians match numerical approximations.
"""

import pytest
import torch

from crocoddyl_batched_mpc.models.pendulum import PendulumCost, PendulumDynamics


def finite_difference_dynamics(
    dynamics, x: torch.Tensor, u: torch.Tensor, eps: float = 1e-6
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute dynamics Jacobians using finite differences.

    Returns:
        Fx: [batch, nx, nx]
        Fu: [batch, nx, nu]
    """
    batch, nx = x.shape
    nu = u.shape[1]

    f0 = dynamics.calc(x, u)
    Fx = torch.zeros(batch, nx, nx, device=x.device, dtype=x.dtype)
    Fu = torch.zeros(batch, nx, nu, device=x.device, dtype=x.dtype)

    # Perturb each state dimension
    for i in range(nx):
        x_plus = x.clone()
        x_plus[:, i] += eps
        f_plus = dynamics.calc(x_plus, u)
        Fx[:, :, i] = (f_plus - f0) / eps

    # Perturb each control dimension
    for i in range(nu):
        u_plus = u.clone()
        u_plus[:, i] += eps
        f_plus = dynamics.calc(x, u_plus)
        Fu[:, :, i] = (f_plus - f0) / eps

    return Fx, Fu


def finite_difference_cost(
    cost_model, x: torch.Tensor, u: torch.Tensor | None, eps: float = 1e-6
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Compute cost derivatives using finite differences.

    Returns:
        lx, lu, lxx, luu, lxu
    """
    batch, nx = x.shape
    nu = u.shape[1] if u is not None else 0

    l0 = cost_model.calc(x, u)
    lx = torch.zeros(batch, nx, device=x.device, dtype=x.dtype)
    lxx = torch.zeros(batch, nx, nx, device=x.device, dtype=x.dtype)

    # First and second derivatives w.r.t. x
    for i in range(nx):
        x_plus = x.clone()
        x_plus[:, i] += eps
        l_plus = cost_model.calc(x_plus, u)
        lx[:, i] = (l_plus - l0) / eps

        # Second derivative (diagonal approximation)
        x_minus = x.clone()
        x_minus[:, i] -= eps
        l_minus = cost_model.calc(x_minus, u)
        lxx[:, i, i] = (l_plus - 2 * l0 + l_minus) / (eps**2)

    # Off-diagonal Hessian elements
    for i in range(nx):
        for j in range(i + 1, nx):
            x_pp = x.clone()
            x_pp[:, i] += eps
            x_pp[:, j] += eps
            x_pm = x.clone()
            x_pm[:, i] += eps
            x_pm[:, j] -= eps
            x_mp = x.clone()
            x_mp[:, i] -= eps
            x_mp[:, j] += eps
            x_mm = x.clone()
            x_mm[:, i] -= eps
            x_mm[:, j] -= eps

            l_pp = cost_model.calc(x_pp, u)
            l_pm = cost_model.calc(x_pm, u)
            l_mp = cost_model.calc(x_mp, u)
            l_mm = cost_model.calc(x_mm, u)

            lxx[:, i, j] = (l_pp - l_pm - l_mp + l_mm) / (4 * eps**2)
            lxx[:, j, i] = lxx[:, i, j]

    if u is None:
        return lx, None, lxx, None, None

    # Derivatives w.r.t. u
    lu = torch.zeros(batch, nu, device=x.device, dtype=x.dtype)
    luu = torch.zeros(batch, nu, nu, device=x.device, dtype=x.dtype)
    lxu = torch.zeros(batch, nx, nu, device=x.device, dtype=x.dtype)

    for i in range(nu):
        u_plus = u.clone()
        u_plus[:, i] += eps
        l_plus = cost_model.calc(x, u_plus)
        lu[:, i] = (l_plus - l0) / eps

        u_minus = u.clone()
        u_minus[:, i] -= eps
        l_minus = cost_model.calc(x, u_minus)
        luu[:, i, i] = (l_plus - 2 * l0 + l_minus) / (eps**2)

    # Cross derivatives lxu
    for i in range(nx):
        for j in range(nu):
            x_plus_u_plus = x.clone()
            x_plus_u_plus[:, i] += eps
            u_plus = u.clone()
            u_plus[:, j] += eps
            l_pp = cost_model.calc(x_plus_u_plus, u_plus)

            x_plus = x.clone()
            x_plus[:, i] += eps
            l_p0 = cost_model.calc(x_plus, u)

            u_plus = u.clone()
            u_plus[:, j] += eps
            l_0p = cost_model.calc(x, u_plus)

            lxu[:, i, j] = (l_pp - l_p0 - l_0p + l0) / (eps**2)

    return lx, lu, lxx, luu, lxu


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.cuda)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_pendulum_dynamics_jacobian(device, dtype):
    """Test PendulumDynamics.calc_diff against finite differences."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    dynamics = PendulumDynamics(device=device, dtype=dtype)
    batch_size = 8

    # Test at multiple random states
    torch.manual_seed(42)
    x = torch.randn(batch_size, 2, device=device, dtype=dtype)
    u = torch.randn(batch_size, 1, device=device, dtype=dtype)

    # Analytical derivatives
    Fx_analytical, Fu_analytical = dynamics.calc_diff(x, u)

    # Finite difference approximation
    # For float32, use larger epsilon to avoid catastrophic cancellation
    eps = 1e-4 if dtype == torch.float32 else 1e-8
    Fx_fd, Fu_fd = finite_difference_dynamics(dynamics, x, u, eps=eps)

    # Compare
    Fx_error = (Fx_analytical - Fx_fd).abs().max().item()
    Fu_error = (Fu_analytical - Fu_fd).abs().max().item()

    # Tolerance depends on dtype and epsilon
    tol = 1e-2 if dtype == torch.float32 else 1e-5

    assert Fx_error < tol, f"Fx error {Fx_error} exceeds tolerance {tol}"
    assert Fu_error < tol, f"Fu error {Fu_error} exceeds tolerance {tol}"


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.cuda)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_pendulum_cost_derivatives(device, dtype):
    """Test PendulumCost.calc_diff.

    For quadratic costs, verify first derivatives with finite differences.
    Second derivatives (Hessians) are constant and equal to Q, R matrices.
    """
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    cost = PendulumCost(device=device, dtype=dtype)
    batch_size = 8

    torch.manual_seed(42)
    x = torch.randn(batch_size, 2, device=device, dtype=dtype)
    u = torch.randn(batch_size, 1, device=device, dtype=dtype)

    # Analytical derivatives
    lx_a, lu_a, lxx_a, luu_a, lxu_a = cost.calc_diff(x, u)

    # Finite difference for first derivatives only
    # For float32, use larger epsilon to avoid catastrophic cancellation
    eps = 1e-4 if dtype == torch.float32 else 1e-7
    l0 = cost.calc(x, u)
    lx_fd = torch.zeros_like(x)
    lu_fd = torch.zeros_like(u)

    for i in range(2):
        x_plus = x.clone()
        x_plus[:, i] += eps
        l_plus = cost.calc(x_plus, u)
        lx_fd[:, i] = (l_plus - l0) / eps

    for i in range(1):
        u_plus = u.clone()
        u_plus[:, i] += eps
        l_plus = cost.calc(x, u_plus)
        lu_fd[:, i] = (l_plus - l0) / eps

    # Compare first derivatives
    # float32 has larger numerical errors due to limited precision
    tol = 0.2 if dtype == torch.float32 else 1e-5
    lx_error = (lx_a - lx_fd).abs().max().item()
    lu_error = (lu_a - lu_fd).abs().max().item()

    assert lx_error < tol, f"lx error {lx_error} exceeds tolerance {tol}"
    assert lu_error < tol, f"lu error {lu_error} exceeds tolerance {tol}"

    # For quadratic cost, Hessians should exactly equal Q, R matrices
    assert torch.allclose(lxx_a[0], cost.Q, atol=1e-6), "lxx should equal Q"
    assert torch.allclose(luu_a[0], cost.R, atol=1e-6), "luu should equal R"
    assert torch.allclose(lxu_a, torch.zeros_like(lxu_a), atol=1e-6), "lxu should be zero"


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.cuda)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_pendulum_terminal_cost_derivatives(device, dtype):
    """Test terminal cost derivatives (u=None case)."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    cost = PendulumCost(device=device, dtype=dtype)
    batch_size = 8

    torch.manual_seed(42)
    x = torch.randn(batch_size, 2, device=device, dtype=dtype)

    # Analytical derivatives
    lx_a, lu_a, lxx_a, luu_a, lxu_a = cost.calc_diff(x, None)

    assert lu_a is None
    assert luu_a is None
    assert lxu_a is None

    # Finite difference for first derivative
    # For float32, use larger epsilon to avoid catastrophic cancellation
    eps = 1e-4 if dtype == torch.float32 else 1e-7
    l0 = cost.calc(x, None)
    lx_fd = torch.zeros_like(x)

    for i in range(2):
        x_plus = x.clone()
        x_plus[:, i] += eps
        l_plus = cost.calc(x_plus, None)
        lx_fd[:, i] = (l_plus - l0) / eps

    # Compare
    # float32 has larger numerical errors, especially with large gradients
    tol = 2.0 if dtype == torch.float32 else 1e-5
    lx_error = (lx_a - lx_fd).abs().max().item()

    assert lx_error < tol, f"Terminal lx error {lx_error} exceeds tolerance {tol}"

    # For quadratic terminal cost, Hessian should exactly equal Q_terminal
    assert torch.allclose(lxx_a[0], cost.Q_terminal, atol=1e-6), "lxx should equal Q_terminal"


def test_pendulum_dynamics_batch_consistency():
    """Verify batch computation equals individual computations."""
    dynamics = PendulumDynamics()
    batch_size = 16

    torch.manual_seed(123)
    x = torch.randn(batch_size, 2)
    u = torch.randn(batch_size, 1)

    # Batched computation
    x_next_batch = dynamics.calc(x, u)
    Fx_batch, Fu_batch = dynamics.calc_diff(x, u)

    # Individual computations
    for i in range(batch_size):
        x_next_single = dynamics.calc(x[i : i + 1], u[i : i + 1])
        Fx_single, Fu_single = dynamics.calc_diff(x[i : i + 1], u[i : i + 1])

        assert torch.allclose(x_next_batch[i], x_next_single[0], atol=1e-6)
        assert torch.allclose(Fx_batch[i], Fx_single[0], atol=1e-6)
        assert torch.allclose(Fu_batch[i], Fu_single[0], atol=1e-6)
