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
    x_init: Tensor | None = None  # Compatibility hint; states are rebuilt from x0/u_init
    u_init: Tensor | None = None  # Optional initial control guess [batch, T, nu]
    device: torch.device | str | None = None
    dtype: torch.dtype | None = None

    # Solver parameters
    max_iterations: int = 100
    cost_tolerance: float = 1e-4  # More relaxed from 1e-6
    gradient_tolerance: float = 1e-4  # More relaxed from 1e-6
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

        if self.manifold.ndx != self.dynamics.nx:
            raise ValueError("DDP currently requires manifold.ndx == dynamics.nx")

        inferred_device = self.device
        inferred_dtype = self.dtype
        for source in (self.x_init, self.u_init, self.dynamics, self.cost):
            if source is None:
                continue
            if inferred_device is None and getattr(source, "device", None) is not None:
                inferred_device = source.device
            if inferred_dtype is None and getattr(source, "dtype", None) is not None:
                inferred_dtype = source.dtype
        if inferred_device is None or inferred_dtype is None:
            raise ValueError(
                "DDPProblem needs device/dtype explicitly or from its initial guess/models"
            )
        resolved_device = torch.device(inferred_device)
        if resolved_device.type == "cuda" and resolved_device.index is None:
            resolved_device = torch.device("cuda", torch.cuda.current_device())
        object.__setattr__(self, "device", resolved_device)
        object.__setattr__(self, "dtype", inferred_dtype)
        if self.dtype not in (torch.float32, torch.float64):
            raise ValueError("DDP currently supports float32 and float64")
        for name, model in (("dynamics", self.dynamics), ("cost", self.cost)):
            model_device = getattr(model, "device", None)
            model_dtype = getattr(model, "dtype", None)
            if model_device is not None:
                resolved_model_device = torch.device(model_device)
                if resolved_model_device.type == "cuda" and resolved_model_device.index is None:
                    resolved_model_device = torch.device("cuda", torch.cuda.current_device())
                if resolved_model_device != self.device:
                    raise ValueError(
                        f"{name} device {model_device} != problem device {self.device}"
                    )
            if model_dtype is not None and model_dtype != self.dtype:
                raise ValueError(f"{name} dtype {model_dtype} != problem dtype {self.dtype}")

        # Validate initial trajectories if provided
        if self.x_init is not None:
            if not isinstance(self.x_init, Tensor):
                raise TypeError("x_init must be a torch.Tensor")
            expected_shape = (self.batch_size, self.horizon + 1, self.nx)
            if tuple(self.x_init.shape) != expected_shape:
                raise ValueError(f"x_init shape {tuple(self.x_init.shape)} != {expected_shape}")
            self._validate_tensor_metadata("x_init", self.x_init)

        if self.u_init is not None:
            if not isinstance(self.u_init, Tensor):
                raise TypeError("u_init must be a torch.Tensor")
            expected_shape = (self.batch_size, self.horizon, self.nu)
            if tuple(self.u_init.shape) != expected_shape:
                raise ValueError(f"u_init shape {tuple(self.u_init.shape)} != {expected_shape}")
            self._validate_tensor_metadata("u_init", self.u_init)

        # Validate solver parameters
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be positive")
        if not (0 < self.cost_tolerance < 1):
            raise ValueError("cost_tolerance must be in (0, 1)")
        if not (0 < self.gradient_tolerance < 1):
            raise ValueError("gradient_tolerance must be in (0, 1)")
        if not 0 <= self.regularization_min <= self.regularization_init:
            raise ValueError("regularization_min must be nonnegative and <= regularization_init")
        if self.regularization_init > self.regularization_max:
            raise ValueError("regularization_init must be <= regularization_max")
        if self.regularization_factor <= 1:
            raise ValueError("regularization_factor must be greater than 1")

    @property
    def nx(self) -> int:
        return self.dynamics.nx

    @property
    def nu(self) -> int:
        return self.dynamics.nu

    @property
    def ndx(self) -> int:
        return self.manifold.ndx

    def _validate_tensor_metadata(self, name: str, value: Tensor) -> None:
        if value.device != self.device or value.dtype != self.dtype:
            raise ValueError(
                f"{name} must have device={self.device} and dtype={self.dtype}, "
                f"got device={value.device} and dtype={value.dtype}"
            )

    def validate_state(self, x0: Tensor) -> None:
        """Check that initial state has correct shape and device."""
        if not isinstance(x0, Tensor):
            raise TypeError("x0 must be a torch.Tensor")
        expected_shape = (self.batch_size, self.nx)
        if tuple(x0.shape) != expected_shape:
            raise ValueError(f"x0 shape {tuple(x0.shape)} != {expected_shape}")

        self._validate_tensor_metadata("x0", x0)

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
