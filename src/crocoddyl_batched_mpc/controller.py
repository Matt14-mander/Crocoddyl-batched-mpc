"""Stateful tensor adapter for vectorized environments, including Isaac Lab."""

import torch
from torch import Tensor

from .ddp_problem import DDPProblem
from .problem import LQRProblem
from .result import MPCResult
from .solver import BatchedMPC


class MPCController:
    """Compute first action; hold the last successful action on numerical failure.

    Reset terminated environments to clear fallback state. This policy is only
    a generic training fallback; the robot task defines physical action limits.
    Exact LQR has no warm start. This cache is solely for failure handling.
    """

    def __init__(self, solver: BatchedMPC) -> None:
        self.solver = solver
        p = solver.problem

        # Get dimensions based on problem type
        if isinstance(p, LQRProblem):
            batch_size, nu = p.batch_size, p.nu
            device, dtype = p.A.device, p.A.dtype
        elif isinstance(p, DDPProblem):
            batch_size, nu = p.batch_size, p.nu
            device, dtype = p.device, p.dtype
        else:
            raise TypeError(f"Unsupported problem type: {type(p)}")

        self._last_action = torch.zeros(batch_size, nu, device=device, dtype=dtype)

    @torch.no_grad()
    def compute(self, state: Tensor) -> tuple[Tensor, MPCResult]:
        result = self.solver.solve(state)
        action = torch.where(result.usable[:, None], result.action, self._last_action)
        self._last_action.copy_(action)
        return action, result

    @torch.no_grad()
    def reset(self, mask: Tensor | None = None) -> None:
        """Clear all environments, or a bool [batch] mask on the problem device."""
        if mask is None:
            self._last_action.zero_()
            return
        if not isinstance(mask, Tensor) or mask.dtype != torch.bool:
            raise TypeError("reset mask must be a bool tensor")
        if mask.shape != self._last_action.shape[:1] or mask.device != self._last_action.device:
            raise ValueError("reset mask must have shape [batch] on the problem device")
        self._last_action.masked_fill_(mask[:, None], 0.0)
