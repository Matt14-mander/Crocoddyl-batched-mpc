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

    Failed trajectories/cost are NaN. action is zero for failed environments.
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
    def action(self) -> Tensor:
        import torch

        return torch.where(self.success[:, None], self.us[:, 0], 0.0)
