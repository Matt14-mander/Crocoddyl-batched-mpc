"""State manifolds for integrate/diff operations.

Euclidean space uses simple addition/subtraction.
Non-Euclidean manifolds (SO(2), SO(3), etc.) require special operations.
"""

import torch
from torch import Tensor


class StateManifold:
    """Base class for state space manifolds.

    Distinguishes representation dimension (nx) from tangent space dimension (ndx).
    For Euclidean spaces, nx == ndx. For Lie groups, they may differ.
    """

    def __init__(self, nx: int, ndx: int | None = None) -> None:
        self.nx = nx
        self.ndx = ndx if ndx is not None else nx

    def integrate(self, x: Tensor, dx: Tensor) -> Tensor:
        """Integrate tangent vector into manifold: x ⊕ dx.

        Args:
            x: State on manifold [batch, nx]
            dx: Tangent vector [batch, ndx]

        Returns:
            x_new: Integrated state [batch, nx]
        """
        raise NotImplementedError

    def diff(self, x1: Tensor, x2: Tensor) -> Tensor:
        """Compute tangent vector from x2 to x1: x1 ⊖ x2.

        Args:
            x1: Target state [batch, nx]
            x2: Base state [batch, nx]

        Returns:
            dx: Tangent vector such that x1 ≈ integrate(x2, dx) [batch, ndx]
        """
        raise NotImplementedError


class EuclideanManifold(StateManifold):
    """Standard Euclidean space: R^n.

    integrate: x + dx
    diff: x1 - x2
    """

    def __init__(self, n: int) -> None:
        super().__init__(nx=n, ndx=n)

    def integrate(self, x: Tensor, dx: Tensor) -> Tensor:
        return x + dx

    def diff(self, x1: Tensor, x2: Tensor) -> Tensor:
        return x1 - x2


class SO2Manifold(StateManifold):
    """2D rotation manifold: angles with periodic wrapping.

    State is represented as angle in radians.
    integrate: angle addition with normalization to [-π, π]
    diff: angle difference with wrapping
    """

    def __init__(self) -> None:
        super().__init__(nx=1, ndx=1)

    def integrate(self, x: Tensor, dx: Tensor) -> Tensor:
        """Add angles and wrap to [-π, π]."""
        result = x + dx
        # Wrap to [-π, π]
        return torch.atan2(torch.sin(result), torch.cos(result))

    def diff(self, x1: Tensor, x2: Tensor) -> Tensor:
        """Compute shortest angular difference."""
        diff = x1 - x2
        # Wrap to [-π, π]
        return torch.atan2(torch.sin(diff), torch.cos(diff))


class ProductManifold(StateManifold):
    """Cartesian product of multiple manifolds.

    Example: SE(2) = R^2 × SO(2) for planar position + orientation.
    """

    def __init__(self, manifolds: list[StateManifold]) -> None:
        self.manifolds = manifolds
        nx = sum(m.nx for m in manifolds)
        ndx = sum(m.ndx for m in manifolds)
        super().__init__(nx=nx, ndx=ndx)

    def integrate(self, x: Tensor, dx: Tensor) -> Tensor:
        """Integrate each component manifold separately."""
        results = []
        x_offset = 0
        dx_offset = 0
        for m in self.manifolds:
            x_slice = x[..., x_offset : x_offset + m.nx]
            dx_slice = dx[..., dx_offset : dx_offset + m.ndx]
            results.append(m.integrate(x_slice, dx_slice))
            x_offset += m.nx
            dx_offset += m.ndx
        return torch.cat(results, dim=-1)

    def diff(self, x1: Tensor, x2: Tensor) -> Tensor:
        """Diff each component manifold separately."""
        results = []
        x_offset = 0
        for m in self.manifolds:
            x1_slice = x1[..., x_offset : x_offset + m.nx]
            x2_slice = x2[..., x_offset : x_offset + m.nx]
            results.append(m.diff(x1_slice, x2_slice))
            x_offset += m.nx
        return torch.cat(results, dim=-1)
