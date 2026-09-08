"""Explicit batch-major, finite-horizon affine LQR problem contract."""

from dataclasses import dataclass

import torch
from torch import Tensor


def check_tensor(name: str, value: Tensor, shape: tuple[int, ...], like: Tensor) -> None:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name}: expected shape {shape}, got {tuple(value.shape)}")
    if value.device != like.device or value.dtype != like.dtype:
        raise ValueError(f"{name} must have device={like.device} and dtype={like.dtype}")


@dataclass(frozen=True)
class LQRProblem:
    """Minimize sum(0.5*x'Q*x + q'x + 0.5*u'R*u + r'u) + terminal cost.

    Dynamics: x[t+1] = A[t] x[t] + B[t] u[t] + f[t].
    A/B/Q/R are [batch, horizon, rows, cols]; Qf is [batch, nx, nx].
    f/q/r are [batch, horizon, dimension]; qf is [batch, nx].
    Q/Qf must be symmetric positive semidefinite, R symmetric positive definite.
    Optional linear terms default to zero. Tensors are borrowed, never migrated.
    """

    A: Tensor
    B: Tensor
    Q: Tensor
    R: Tensor
    Qf: Tensor
    f: Tensor | None = None
    q: Tensor | None = None
    r: Tensor | None = None
    qf: Tensor | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.A, Tensor) or self.A.ndim != 4:
            raise ValueError("A must have shape [batch, horizon, nx, nx]")
        if self.A.dtype not in (torch.float32, torch.float64):
            raise ValueError("Only float32 and float64 are supported")
        if self.A.device.type not in ("cpu", "cuda"):
            raise ValueError("Only CPU and CUDA devices are supported")
        batch, horizon, nx, last = self.A.shape
        if min(batch, horizon, nx) < 1 or last != nx:
            raise ValueError("batch, horizon and nx must be positive; A must be square")
        if not isinstance(self.B, Tensor) or self.B.ndim != 4 or self.B.shape[-1] < 1:
            raise ValueError("B must have shape [batch, horizon, nx, nu] with nu > 0")
        nu = self.B.shape[-1]
        shapes = {
            "B": (batch, horizon, nx, nu),
            "Q": (batch, horizon, nx, nx),
            "R": (batch, horizon, nu, nu),
            "Qf": (batch, nx, nx),
            "f": (batch, horizon, nx),
            "q": (batch, horizon, nx),
            "r": (batch, horizon, nu),
            "qf": (batch, nx),
        }
        for name, shape in shapes.items():
            value = getattr(self, name)
            if value is None and name in ("f", "q", "r", "qf"):
                value = self.A.new_zeros(shape)
                object.__setattr__(self, name, value)
            check_tensor(name, value, shape, self.A)

    @property
    def batch_size(self) -> int:
        return self.A.shape[0]

    @property
    def horizon(self) -> int:
        return self.A.shape[1]

    @property
    def nx(self) -> int:
        return self.A.shape[-1]

    @property
    def nu(self) -> int:
        return self.B.shape[-1]

    def validate_state(self, x0: Tensor) -> None:
        check_tensor("x0", x0, (self.batch_size, self.nx), self.A)

    @classmethod
    def from_lti(
        cls,
        A: Tensor,
        B: Tensor,
        Q: Tensor,
        R: Tensor,
        Qf: Tensor,
        *,
        batch_size: int,
        horizon: int,
    ) -> "LQRProblem":
        """Expand shared 2-D LTI matrices as views; use the constructor for varying models."""
        for name, value in (("A", A), ("B", B), ("Q", Q), ("R", R), ("Qf", Qf)):
            if not isinstance(value, Tensor) or value.ndim != 2:
                raise ValueError(f"{name} must be a 2-D tensor in from_lti")
        if batch_size < 1 or horizon < 1:
            raise ValueError("batch_size and horizon must be positive")

        def expand(value: Tensor) -> Tensor:
            return value[None, None].expand(batch_size, horizon, *value.shape)

        return cls(
            expand(A), expand(B), expand(Q), expand(R), Qf[None].expand(batch_size, *Qf.shape)
        )
