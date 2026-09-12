"""Small analytic model for smoke tests and benchmarks, not robot dynamics."""

import math

import torch

from ..problem import LQRProblem


def double_integrator(
    *,
    batch_size: int = 64,
    horizon: int = 20,
    dt: float = 0.02,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> LQRProblem:
    """State [position, velocity], control acceleration, zero target, exact ZOH."""
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be finite and positive")
    A = torch.tensor([[1.0, dt], [0.0, 1.0]], device=device, dtype=dtype)
    B = torch.tensor([[0.5 * dt * dt], [dt]], device=device, dtype=dtype)
    Q = torch.diag(torch.tensor([10.0, 1.0], device=device, dtype=dtype))
    R = torch.full((1, 1), 0.1, device=device, dtype=dtype)
    return LQRProblem.from_lti(A, B, Q, R, 10 * Q, batch_size=batch_size, horizon=horizon)
