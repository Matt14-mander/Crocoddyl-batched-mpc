from dataclasses import dataclass
from enum import IntEnum

from torch import Tensor


class SolveStatus(IntEnum):
    SUCCESS = 0
    NUMERICAL_FAILURE = 1
    MAX_ITERATIONS = 2


@dataclass(frozen=True)
class MPCResult:
    """All tensors stay on the problem device, including per-environment status.

    Numerically failed trajectories/cost are NaN. action is zero for those environments.
    No gradients: RL can use MPC as a controller/teacher, not a differentiable layer.
    """

    xs: Tensor  # [batch, horizon + 1, nx]
    us: Tensor  # [batch, horizon, nu]
    cost: Tensor  # [batch]
    status: Tensor  # int64 [batch]
    iterations: Tensor  # int64 [batch]

    @property
    def success(self) -> Tensor:
        return self.status == SolveStatus.SUCCESS

    @property
    def usable(self) -> Tensor:
        """A finite candidate remains usable when a fixed iteration budget expires."""
        import torch

        return (self.status != SolveStatus.NUMERICAL_FAILURE) & torch.isfinite(self.us[:, 0]).all(
            -1
        )

    @property
    def action(self) -> Tensor:
        import torch

        return torch.where(self.usable[:, None], self.us[:, 0], 0.0)
