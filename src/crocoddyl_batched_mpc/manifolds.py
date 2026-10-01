"""State manifolds for integrate/diff operations.

Euclidean space uses simple addition/subtraction.
Non-Euclidean manifolds (SO(2), SO(3), etc.) require special operations.
"""

import torch
from torch import Tensor

from .lie import (
    quaternion_conjugate,
    quaternion_exp,
    quaternion_log,
    quaternion_matrix,
    quaternion_multiply,
    quaternion_normalize,
    se3_right_jacobian_inverse,
    so3_left_jacobian,
    so3_left_jacobian_inverse,
)


class StateManifold:
    """Base class for state space manifolds.

    Distinguishes representation dimension (nx) from tangent space dimension (ndx).
    For Euclidean spaces, nx == ndx. For Lie groups, they may differ.
    """

    is_flat = False

    def __init__(self, nx: int, ndx: int | None = None) -> None:
        self.nx = nx
        self.ndx = ndx if ndx is not None else nx

    def neutral(self, like: Tensor) -> Tensor:
        """Valid identity state with the same leading dimensions/device/dtype."""
        return like.new_zeros(*like.shape[:-1], self.nx)

    def is_valid(self, x: Tensor) -> Tensor:
        return torch.isfinite(x).all(-1)

    def diff_jacobian(self, target: Tensor, base: Tensor) -> Tensor:
        """d diff(integrate(target, delta), base) / d delta at zero.

        Maps derivatives at target into the difference chart based at base.
        Custom manifolds must provide this operation for infeasible FDDP.
        """
        raise NotImplementedError

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

    is_flat = True

    def __init__(self, n: int) -> None:
        super().__init__(nx=n, ndx=n)

    def diff_jacobian(self, target: Tensor, base: Tensor) -> Tensor:
        return torch.eye(self.ndx, device=target.device, dtype=target.dtype).expand(
            *target.shape[:-1], self.ndx, self.ndx
        )

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

    is_flat = True

    def __init__(self) -> None:
        super().__init__(nx=1, ndx=1)

    def diff_jacobian(self, target: Tensor, base: Tensor) -> Tensor:
        return target.new_ones(*target.shape[:-1], 1, 1)

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
        self.is_flat = all(m.is_flat for m in manifolds)
        nx = sum(m.nx for m in manifolds)
        ndx = sum(m.ndx for m in manifolds)
        super().__init__(nx=nx, ndx=ndx)

    def neutral(self, like: Tensor) -> Tensor:
        return torch.cat([m.neutral(like) for m in self.manifolds], -1)

    def is_valid(self, x: Tensor) -> Tensor:
        valid = torch.ones(x.shape[:-1], dtype=torch.bool, device=x.device)
        offset = 0
        for m in self.manifolds:
            valid &= m.is_valid(x[..., offset : offset + m.nx])
            offset += m.nx
        return valid

    def diff_jacobian(self, target: Tensor, base: Tensor) -> Tensor:
        result = target.new_zeros(*target.shape[:-1], self.ndx, self.ndx)
        offset, tangent = 0, 0
        for m in self.manifolds:
            block = m.diff_jacobian(
                target[..., offset : offset + m.nx], base[..., offset : offset + m.nx]
            )
            result[..., tangent : tangent + m.ndx, tangent : tangent + m.ndx] = block
            offset += m.nx
            tangent += m.ndx
        return result

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


class SO3Manifold(StateManifold):
    """Unit quaternion [x,y,z,w]; right/body rotation increments in radians.

    diff(target, base) = Log(base^-1 target); the principal log cuts at pi.
    """

    def __init__(self) -> None:
        super().__init__(nx=4, ndx=3)

    def neutral(self, like: Tensor) -> Tensor:
        state = super().neutral(like)
        state[..., 3] = 1
        return state

    def is_valid(self, x: Tensor) -> Tensor:
        return super().is_valid(x) & ((x.square().sum(-1) - 1).abs() < 1e-5)

    def integrate(self, x: Tensor, dx: Tensor) -> Tensor:
        return quaternion_normalize(quaternion_multiply(x, quaternion_exp(dx)))

    def diff(self, x1: Tensor, x2: Tensor) -> Tensor:
        return quaternion_log(quaternion_multiply(quaternion_conjugate(x2), x1))

    def diff_jacobian(self, target: Tensor, base: Tensor) -> Tensor:
        return so3_left_jacobian_inverse(-self.diff(target, base))


class SE3Manifold(StateManifold):
    """Pose [world position, xyzw quaternion], body [linear,angular] increments.

    Translation uses the SE(3) exponential, matching Pinocchio's free flyer;
    it is not independent world-position addition.
    """

    def __init__(self) -> None:
        super().__init__(nx=7, ndx=6)
        self.rotation = SO3Manifold()

    def neutral(self, like: Tensor) -> Tensor:
        state = super().neutral(like)
        state[..., 6] = 1
        return state

    def is_valid(self, x: Tensor) -> Tensor:
        return super().is_valid(x) & self.rotation.is_valid(x[..., 3:7])

    def integrate(self, x: Tensor, dx: Tensor) -> Tensor:
        local_translation = (so3_left_jacobian(dx[..., 3:]) @ dx[..., :3, None]).squeeze(-1)
        translation = (quaternion_matrix(x[..., 3:]) @ local_translation[..., None]).squeeze(-1)
        return torch.cat(
            (x[..., :3] + translation, self.rotation.integrate(x[..., 3:], dx[..., 3:])), -1
        )

    def diff(self, x1: Tensor, x2: Tensor) -> Tensor:
        angular = self.rotation.diff(x1[..., 3:], x2[..., 3:])
        translation = (
            quaternion_matrix(x2[..., 3:]).transpose(-1, -2)
            @ (x1[..., :3] - x2[..., :3])[..., None]
        ).squeeze(-1)
        linear = (so3_left_jacobian_inverse(angular) @ translation[..., None]).squeeze(-1)
        return torch.cat((linear, angular), -1)

    def diff_jacobian(self, target: Tensor, base: Tensor) -> Tensor:
        return se3_right_jacobian_inverse(self.diff(target, base))


class FloatingBaseManifold(ProductManifold):
    """State [base position, xyzw, joint positions, generalized velocities].

    Supports Euclidean scalar joints. nq=7+nj, nv=6+nj, nx=13+2nj,
    ndx=12+2nj. Configuration perturbations use body free-flyer increments;
    generalized velocity coordinates are added as in Crocoddyl StateMultibody.
    Other joint types require their own configuration manifolds.
    """

    def __init__(self, joint_count: int = 0) -> None:
        if not isinstance(joint_count, int) or isinstance(joint_count, bool) or joint_count < 0:
            raise ValueError("joint_count must be a nonnegative integer")
        self.nq, self.nv = 7 + joint_count, 6 + joint_count
        super().__init__(
            [SE3Manifold(), EuclideanManifold(joint_count), EuclideanManifold(self.nv)]
        )
