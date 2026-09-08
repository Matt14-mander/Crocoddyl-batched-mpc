"""Nonlinear optimal control problem for DDP/iLQR solvers.

Extends LQRProblem to support general nonlinear dynamics and costs.
"""

from dataclasses import dataclass

import torch
from torch import Tensor

from .costs import CostModel
from .dynamics import DynamicsModel
from .manifolds import EuclideanManifold, StateManifold


@dataclass(frozen=True)
class DDPProblem:
    """Nonlinear finite-horizon optimal control problem.

    minimize sum_{t=0}^{T-1} l(x[t], u[t]) + l_T(x[T])
    subject to x[t+1] = f(x[t], u[t])

    All instances in the batch share the same dynamics and cost models,
    but may have different initial states and reference trajectories.
    """

    dynamics: DynamicsModel
    cost: CostModel
    manifold: StateManifold
    batch_size: int
    horizon: int
    x_init: Tensor | None = None  # Optional initial trajectory guess [batch, T+1, nx]
    u_init: Tensor | None = None  # Optional initial control guess [batch, T, nu]

    # Solver parameters
    max_iterations: int = 100
    cost_tolerance: float = 1e-6
    gradient_tolerance: float = 1e-6
    regularization_init: float = 1e-6
    regularization_min: float = 1e-9
    regularization_max: float = 1e9
    regularization_factor: float = 10.0

    def __post_init__(self) -> None:
        """Validate problem dimensions and parameters."""
        if self.batch_size < 1 or self.horizon < 1:
            raise ValueError("batch_size and horizon must be positive")

        if self.dynamics.nx != self.cost.nx:
            raise ValueError(f"Dynamics nx={self.dynamics.nx} != cost nx={self.cost.nx}")

        if self.dynamics.nu != self.cost.nu:
            raise ValueError(f"Dynamics nu={self.dynamics.nu} != cost nu={self.cost.nu}")

        if self.manifold.nx != self.dynamics.nx:
            raise ValueError(f"Manifold nx={self.manifold.nx} != dynamics nx={self.dynamics.nx}")

        # Validate initial trajectories if provided
        if self.x_init is not None:
            expected_shape = (self.batch_size, self.horizon + 1, self.nx)
            if tuple(self.x_init.shape) != expected_shape:
                raise ValueError(f"x_init shape {tuple(self.x_init.shape)} != {expected_shape}")

        if self.u_init is not None:
            expected_shape = (self.batch_size, self.horizon, self.nu)
            if tuple(self.u_init.shape) != expected_shape:
                raise ValueError(f"u_init shape {tuple(self.u_init.shape)} != {expected_shape}")

        # Validate solver parameters
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be positive")
        if not (0 < self.cost_tolerance < 1):
            raise ValueError("cost_tolerance must be in (0, 1)")
        if not (0 < self.gradient_tolerance < 1):
            raise ValueError("gradient_tolerance must be in (0, 1)")

    @property
    def nx(self) -> int:
        return self.dynamics.nx

    @property
    def nu(self) -> int:
        return self.dynamics.nu

    @property
    def ndx(self) -> int:
        return self.manifold.ndx

    @property
    def device(self) -> torch.device:
        """Infer device from initial trajectory or dynamics parameters."""
        if self.x_init is not None:
            return self.x_init.device
        if self.u_init is not None:
            return self.u_init.device
        # Default to CPU if no initial guess provided
        return torch.device("cpu")

    @property
    def dtype(self) -> torch.dtype:
        """Infer dtype from initial trajectory."""
        if self.x_init is not None:
            return self.x_init.dtype
        if self.u_init is not None:
            return self.u_init.dtype
        return torch.float32

    def validate_state(self, x0: Tensor) -> None:
        """Check that initial state has correct shape and device."""
        expected_shape = (self.batch_size, self.nx)
        if tuple(x0.shape) != expected_shape:
            raise ValueError(f"x0 shape {tuple(x0.shape)} != {expected_shape}")

        if self.x_init is not None:
            if x0.device != self.x_init.device:
                raise ValueError(f"x0 device {x0.device} != problem device {self.x_init.device}")
            if x0.dtype != self.x_init.dtype:
                raise ValueError(f"x0 dtype {x0.dtype} != problem dtype {self.x_init.dtype}")

    @classmethod
    def from_models(
        cls,
        dynamics: DynamicsModel,
        cost: CostModel,
        batch_size: int,
        horizon: int,
        manifold: StateManifold | None = None,
        **kwargs,
    ) -> "DDPProblem":
        """Convenience constructor with automatic manifold inference.

        Defaults to Euclidean manifold if not specified.
        """
        if manifold is None:
            manifold = EuclideanManifold(dynamics.nx)

        return cls(
            dynamics=dynamics,
            cost=cost,
            manifold=manifold,
            batch_size=batch_size,
            horizon=horizon,
            **kwargs,
        )
