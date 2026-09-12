"""Helper functions for DDP initialization and warm start."""

import torch
from torch import Tensor

from .ddp_problem import DDPProblem


def initialize_trajectory_lqr_linearization(
    problem: DDPProblem, x0: Tensor, n_iterations: int = 3
) -> tuple[Tensor, Tensor]:
    """Initialize trajectory using iterative LQR around nominal trajectory.

    This provides a better warm start than zero control rollout.

    Args:
        problem: DDPProblem instance
        x0: Initial state [batch, nx]
        n_iterations: Number of LQR iterations (default 3)

    Returns:
        xs: Initial state trajectory [batch, T+1, nx]
        us: Initial control trajectory [batch, T, nu]
    """
    batch = problem.batch_size
    horizon = problem.horizon
    nx, nu = problem.nx, problem.nu
    device, dtype = x0.device, x0.dtype

    # Start with zero control rollout
    xs = torch.zeros(batch, horizon + 1, nx, device=device, dtype=dtype)
    us = torch.zeros(batch, horizon, nu, device=device, dtype=dtype)
    xs[:, 0] = x0

    for t in range(horizon):
        xs[:, t + 1] = problem.dynamics.calc(xs[:, t], us[:, t])

    # Iteratively improve with LQR
    for _ in range(n_iterations):
        us = _lqr_feedback_sweep(problem, xs, us)
        xs = _forward_rollout(problem, x0, us)

    return xs, us


def _lqr_feedback_sweep(problem: DDPProblem, xs: Tensor, us: Tensor) -> Tensor:
    """Compute improved controls via LQR feedback around trajectory."""
    batch, horizon, nx, nu = problem.batch_size, problem.horizon, problem.nx, problem.nu
    device, dtype = xs.device, xs.dtype

    # Backward pass: compute value function
    lx_T, _, lxx_T, _, _ = problem.cost.calc_diff(xs[:, horizon], None)
    Vx = lx_T
    Vxx = lxx_T

    gains = torch.zeros(batch, horizon, nu, nx, device=device, dtype=dtype)
    offsets = torch.zeros(batch, horizon, nu, device=device, dtype=dtype)

    eye_nu = torch.eye(nu, device=device, dtype=dtype)
    reg = 1e-6  # Small regularization for stability

    for t in reversed(range(horizon)):
        with torch.enable_grad():
            Fx, Fu = problem.dynamics.calc_diff(xs[:, t], us[:, t])

        lx, lu, lxx, luu, lxu = problem.cost.calc_diff(xs[:, t], us[:, t])

        # Q-function
        Qx = lx + torch.matmul(Fx.transpose(-1, -2), Vx.unsqueeze(-1)).squeeze(-1)
        Qu = lu + torch.matmul(Fu.transpose(-1, -2), Vx.unsqueeze(-1)).squeeze(-1)
        Quu = luu + torch.matmul(torch.matmul(Fu.transpose(-1, -2), Vxx), Fu)
        Qux = lxu.transpose(-1, -2) + torch.matmul(torch.matmul(Fu.transpose(-1, -2), Vxx), Fx)

        # Regularize and solve
        Quu_reg = Quu + reg * eye_nu
        L = torch.linalg.cholesky(Quu_reg)

        rhs = torch.cat((-Qux, -Qu.unsqueeze(-1)), dim=-1)
        intermediate = torch.linalg.solve_triangular(L, rhs, upper=False)
        solution = torch.linalg.solve_triangular(L.transpose(-1, -2), intermediate, upper=True)

        gains[:, t] = solution[..., :nx]
        offsets[:, t] = solution[..., nx]

        # Update value function
        Qxx = lxx + torch.matmul(torch.matmul(Fx.transpose(-1, -2), Vxx), Fx)
        Vx = Qx + torch.matmul(gains[:, t].transpose(-1, -2), Qu.unsqueeze(-1)).squeeze(-1)
        Vxx = Qxx + torch.matmul(torch.matmul(gains[:, t].transpose(-1, -2), Quu), gains[:, t])
        Vxx = 0.5 * (Vxx + Vxx.transpose(-1, -2))

    # Apply feedback to get new controls
    us_new = torch.zeros_like(us)
    for t in range(horizon):
        dx = problem.manifold.diff(xs[:, t], xs[:, t])  # Zero for initialization
        us_new[:, t] = (
            us[:, t] + offsets[:, t] + torch.matmul(gains[:, t], dx.unsqueeze(-1)).squeeze(-1)
        )

    return us_new


def _forward_rollout(problem: DDPProblem, x0: Tensor, us: Tensor) -> Tensor:
    """Forward rollout with given controls."""
    batch, horizon, nx = problem.batch_size, problem.horizon, problem.nx
    device, dtype = x0.device, x0.dtype

    xs = torch.zeros(batch, horizon + 1, nx, device=device, dtype=dtype)
    xs[:, 0] = x0

    for t in range(horizon):
        xs[:, t + 1] = problem.dynamics.calc(xs[:, t], us[:, t])

    return xs
