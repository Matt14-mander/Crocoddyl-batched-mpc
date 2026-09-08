"""Batched Riccati recursion: parallel environments, sequential horizon."""

import math

import torch
from torch import Tensor

from ..problem import LQRProblem
from ..result import MPCResult, SolveStatus


def mv(matrix: Tensor, vector: Tensor) -> Tensor:
    return (matrix @ vector.unsqueeze(-1)).squeeze(-1)


def trajectory_cost(p: LQRProblem, xs: Tensor, us: Tensor, regularization: float) -> Tensor:
    x = xs[:, :-1]
    running = (0.5 * x * mv(p.Q, x) + p.q * x).sum(-1)
    running += (0.5 * us * mv(p.R, us) + p.r * us).sum(-1)
    running += 0.5 * regularization * us.square().sum(-1)
    terminal = (0.5 * xs[:, -1] * mv(p.Qf, xs[:, -1]) + p.qf * xs[:, -1]).sum(-1)
    return running.sum(-1) + terminal


class TorchLQRBackend:
    """Exact affine LQR solve using Torch on the input device.

    regularization adds 0.5 * regularization * ||u||^2 to the objective.
    This is a fixed control penalty, not adaptive DDP damping. No host scalar
    reads or tensor transfers occur in solve; CUDA uses PyTorch's current stream.
    """

    def __init__(self, *, regularization: float = 0.0) -> None:
        if not math.isfinite(regularization) or regularization < 0:
            raise ValueError("regularization must be finite and nonnegative")
        self.regularization = regularization

    @torch.no_grad()
    def solve(self, problem: LQRProblem, x0: Tensor) -> MPCResult:
        p = problem
        p.validate_state(x0)
        batch, horizon, nx, nu = p.batch_size, p.horizon, p.nx, p.nu
        eye = torch.eye(nu, device=x0.device, dtype=x0.dtype)
        valid = torch.isfinite(x0).all(-1)
        for name in ("A", "B", "Q", "R", "Qf", "f", "q", "r", "qf"):
            valid = valid & torch.isfinite(getattr(p, name)).flatten(1).all(-1)
        gains = x0.new_empty(batch, horizon, nu, nx)
        offsets = x0.new_empty(batch, horizon, nu)
        Vxx, Vx = p.Qf, p.qf
        for t in reversed(range(horizon)):
            A, B = p.A[:, t], p.B[:, t]
            At, Bt = A.transpose(-1, -2), B.transpose(-1, -2)
            next_gradient = mv(Vxx, p.f[:, t]) + Vx
            H = p.R[:, t] + Bt @ Vxx @ B + self.regularization * eye
            G = Bt @ Vxx @ A
            g = p.r[:, t] + mv(Bt, next_gradient)
            # check_errors=False leaves failure information on device.
            L, info = torch.linalg.cholesky_ex(H, check_errors=False)
            good = (info == 0) & torch.isfinite(L).flatten(1).all(-1)
            valid = valid & good
            L = torch.where(good[:, None, None], L, eye)
            rhs = torch.cat((G, g.unsqueeze(-1)), dim=-1)
            # cholesky_solve can synchronize CUDA for its internal error checks.
            # The factor is already checked on device, so use triangular solves.
            intermediate = torch.linalg.solve_triangular(L, rhs, upper=False)
            policy = -torch.linalg.solve_triangular(L.transpose(-1, -2), intermediate, upper=True)
            K, k = policy[..., :nx], policy[..., nx]
            gains[:, t], offsets[:, t] = K, k
            Vxx = p.Q[:, t] + At @ Vxx @ A + G.transpose(-1, -2) @ K
            Vxx = 0.5 * (Vxx + Vxx.transpose(-1, -2))
            Vx = p.q[:, t] + mv(At, next_gradient) + mv(G.transpose(-1, -2), k)

        xs = x0.new_empty(batch, horizon + 1, nx)
        us = x0.new_empty(batch, horizon, nu)
        xs[:, 0] = x0
        for t in range(horizon):
            us[:, t] = mv(gains[:, t], xs[:, t]) + offsets[:, t]
            xs[:, t + 1] = mv(p.A[:, t], xs[:, t]) + mv(p.B[:, t], us[:, t]) + p.f[:, t]
        cost = trajectory_cost(p, xs, us, self.regularization)
        valid = valid & torch.isfinite(cost)
        valid = valid & torch.isfinite(xs).flatten(1).all(-1)
        valid = valid & torch.isfinite(us).flatten(1).all(-1)
        status = torch.where(valid, SolveStatus.SUCCESS, SolveStatus.NUMERICAL_FAILURE)
        return MPCResult(
            torch.where(valid[:, None, None], xs, float("nan")),
            torch.where(valid[:, None, None], us, float("nan")),
            torch.where(valid, cost, float("nan")),
            status,
            torch.ones(batch, dtype=torch.int64, device=x0.device),
        )
