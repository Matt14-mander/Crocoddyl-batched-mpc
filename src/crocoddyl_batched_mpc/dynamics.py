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
    def ndx(self) -> int:
        """Tangent dimension; legacy Euclidean models may omit this (= nx)."""
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

        Derivatives use input increments at x and output increments at calc(x,u).
        Equivalently differentiate diff(calc(integrate(x,dx),u+du), calc(x,u)).

        Returns:
            Fx: Local state Jacobian [batch, ndx, ndx]
            Fu: Local control Jacobian [batch, ndx, nu]
        """
        ...


def tangent_dynamics_jacobians(dynamics, manifold, x: Tensor, u: Tensor):
    """Forward-mode local derivatives for a batch-independent Torch model.

    A correctness baseline, not an optimized rigid-body derivative kernel.
    Each JVP perturbs the same coordinate in every independent environment;
    models coupling batch elements are outside the dynamics contract.
    """
    import torch

    predicted = dynamics.calc(x, u).detach()
    delta = x.new_zeros(x.shape[0], manifold.ndx)
    control_delta = torch.zeros_like(u)

    def local(dx, du):
        output = dynamics.calc(manifold.integrate(x, dx), u + du)
        return manifold.diff(output, predicted)

    state_columns, control_columns = [], []
    for index in range(manifold.ndx):
        direction = torch.zeros_like(delta)
        direction[:, index] = 1
        _, derivative = torch.func.jvp(local, (delta, control_delta), (direction, control_delta))
        state_columns.append(derivative)
    for index in range(u.shape[-1]):
        direction = torch.zeros_like(u)
        direction[:, index] = 1
        _, derivative = torch.func.jvp(local, (delta, control_delta), (delta, direction))
        control_columns.append(derivative)
    return torch.stack(state_columns, -1), torch.stack(control_columns, -1)
