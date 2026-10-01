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
    def ndx(self) -> int:
        """Local state dimension; Euclidean legacy models may omit it (= nx)."""
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
            lx: Gradient w.r.t. state [batch, ndx]
            lu: Gradient w.r.t. control [batch, nu] or None if terminal
            lxx: Hessian w.r.t. state [batch, ndx, ndx]
            luu: Hessian w.r.t. control [batch, nu, nu] or None if terminal
            lxu: Cross Hessian [batch, ndx, nu] or None if terminal
        """
        ...


class ManifoldQuadraticCost:
    """Quadratic local state residual with Gauss-Newton state Hessians.

    reference is [nx] or [B,nx]. Shared Q/Q_terminal live in its tangent chart;
    R is [nu,nu]. Gradients are exact locally, Hessians omit residual curvature.
    References and weights must remain on the model device/dtype.
    """

    def __init__(
        self, manifold, reference: Tensor, Q: Tensor, R: Tensor, Q_terminal: Tensor | None = None
    ) -> None:
        import torch

        self.manifold = manifold
        self.nx, self.ndx = manifold.nx, manifold.ndx
        if reference.ndim not in (1, 2) or reference.shape[-1] != self.nx:
            raise ValueError("reference must have shape [nx] or [batch,nx]")
        if reference.dtype not in (torch.float32, torch.float64):
            raise ValueError("reference requires float32 or float64")
        self.device, self.dtype = reference.device, reference.dtype
        if R.ndim != 2 or R.shape[0] < 1 or R.shape[0] != R.shape[1]:
            raise ValueError("R must have shape [nu,nu]")
        self.nu = R.shape[0]
        terminal = Q if Q_terminal is None else Q_terminal
        for name, value, dimension in (
            ("Q", Q, self.ndx),
            ("R", R, self.nu),
            ("Q_terminal", terminal, self.ndx),
        ):
            if value.shape != (dimension, dimension):
                raise ValueError(f"{name} has incorrect tangent/control dimensions")
            if value.device != self.device or value.dtype != self.dtype:
                raise ValueError("weights and reference must share device/dtype")
            if not torch.isfinite(value).all() or not torch.allclose(value, value.T):
                raise ValueError(f"{name} must be finite and symmetric")
        if not manifold.is_valid(reference).all():
            raise ValueError("reference must be a valid manifold state")
        self.reference, self.Q, self.R, self.Q_terminal = reference, Q, R, terminal

    @property
    def parameter_batch_size(self) -> int | None:
        return self.reference.shape[0] if self.reference.ndim == 2 else None

    def calc(self, x: Tensor, u: Tensor | None = None) -> Tensor:
        residual = self.manifold.diff(x, self.reference)
        weight = self.Q_terminal if u is None else self.Q
        value = 0.5 * (residual * (residual @ weight.T)).sum(-1)
        if u is not None:
            value = value + 0.5 * (u * (u @ self.R.T)).sum(-1)
        return value

    def calc_diff(self, x: Tensor, u: Tensor | None = None):
        residual = self.manifold.diff(x, self.reference)
        jacobian = self.manifold.diff_jacobian(x, self.reference)
        weight = self.Q_terminal if u is None else self.Q
        lx = (jacobian.transpose(-1, -2) @ (residual @ weight.T)[..., None]).squeeze(-1)
        lxx = jacobian.transpose(-1, -2) @ weight @ jacobian
        if u is None:
            return lx, None, lxx, None, None
        return (
            lx,
            u @ self.R.T,
            lxx,
            self.R.expand(x.shape[0], -1, -1),
            x.new_zeros(x.shape[0], self.ndx, self.nu),
        )
