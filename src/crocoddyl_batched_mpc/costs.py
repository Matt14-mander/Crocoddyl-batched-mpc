"""Batched cost model interface for MPC objective functions.

Supports quadratic and general nonlinear costs with first and second derivatives.
"""

from typing import Protocol

from torch import Tensor


class CostModel(Protocol):
    """Protocol for stage cost: l(x, u) and terminal cost: l_terminal(x).

    Derivatives are required for DDP backward pass.
    """

    @property
    def nx(self) -> int:
        """State dimension."""
        ...

    @property
    def nu(self) -> int:
        """Control dimension."""
        ...

    def calc(self, x: Tensor, u: Tensor | None = None) -> Tensor:
        """Evaluate cost.

        Args:
            x: State [batch, nx]
            u: Control [batch, nu] or None for terminal cost

        Returns:
            cost: Scalar cost per environment [batch]
        """
        ...

    def calc_diff(
        self, x: Tensor, u: Tensor | None = None
    ) -> tuple[Tensor, Tensor | None, Tensor, Tensor | None, Tensor | None]:
        """Compute cost derivatives.

        Args:
            x: State [batch, nx]
            u: Control [batch, nu] or None for terminal cost

        Returns:
            lx: Gradient w.r.t. state [batch, nx]
            lu: Gradient w.r.t. control [batch, nu] or None if terminal
            lxx: Hessian w.r.t. state [batch, nx, nx]
            luu: Hessian w.r.t. control [batch, nu, nu] or None if terminal
            lxu: Cross Hessian [batch, nx, nu] or None if terminal
        """
        ...
