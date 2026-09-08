"""Batched dynamics model interface for nonlinear systems.

All methods operate on batch-major tensors: [batch, ...].
Implementations must support both CPU and CUDA devices.
"""

from typing import Protocol

from torch import Tensor


class DynamicsModel(Protocol):
    """Protocol for discrete-time dynamics: x[t+1] = f(x[t], u[t], dt).

    State dimension nx and control dimension nu are inferred from tensor shapes.
    Models may be time-invariant (shared parameters) or time-varying (per-timestep).
    """

    @property
    def nx(self) -> int:
        """State dimension."""
        ...

    @property
    def nu(self) -> int:
        """Control dimension."""
        ...

    @property
    def dt(self) -> float:
        """Integration timestep in seconds."""
        ...

    def calc(self, x: Tensor, u: Tensor) -> Tensor:
        """Forward dynamics: compute next state.

        Args:
            x: Current state [batch, nx]
            u: Control input [batch, nu]

        Returns:
            x_next: Next state [batch, nx]
        """
        ...

    def calc_diff(self, x: Tensor, u: Tensor) -> tuple[Tensor, Tensor]:
        """Compute dynamics Jacobians.

        Args:
            x: Current state [batch, nx]
            u: Control input [batch, nu]

        Returns:
            Fx: State Jacobian ∂f/∂x [batch, nx, nx]
            Fu: Control Jacobian ∂f/∂u [batch, nx, nu]
        """
        ...
