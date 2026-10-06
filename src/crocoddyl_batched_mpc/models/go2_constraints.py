"""Hard torque and conservative Coulomb friction constraints for flat support."""

import math

import torch


class Go2ContactConstraints:
    """World +z normal, fixed support set, |fx|+|fy| <= mu*fz, fz>=minimum.

    The diamond is inscribed in the circular Coulomb cone. Negative normal
    force is forbidden; zero force does not automatically remove a fixed foot.
    """

    def __init__(
        self,
        dynamics,
        *,
        friction=0.6,
        minimum_normal_force=1.0,
        maximum_normal_force=200.0,
        torque_limits=None,
        tolerance=1e-7,
    ):
        if not math.isfinite(friction) or friction <= 0:
            raise ValueError("friction must be finite and positive")
        if not (
            math.isfinite(minimum_normal_force)
            and math.isfinite(maximum_normal_force)
            and 0 <= minimum_normal_force < maximum_normal_force
        ):
            raise ValueError("normal force bounds must be finite, ordered and nonnegative")
        if not math.isfinite(tolerance) or tolerance <= 0:
            raise ValueError("tolerance must be finite and positive")
        limits = (
            dynamics.torque_limits
            if torque_limits is None
            else torch.as_tensor(torque_limits, device=dynamics.device, dtype=dynamics.dtype)
        )
        if limits.shape != (12,) or not torch.isfinite(limits).all() or (limits <= 0).any():
            raise ValueError("torque_limits must be 12 finite positive values")
        if (limits > dynamics.torque_limits).any():
            raise ValueError("torque_limits cannot exceed URDF effort limits")
        self.torque_limits = limits.clone()
        self.friction, self.minimum_normal_force = friction, minimum_normal_force
        self.maximum_normal_force, self.tolerance = maximum_normal_force, tolerance
        face = limits.new_tensor(
            [
                [1.0, 1.0, -friction],
                [1.0, -1.0, -friction],
                [-1.0, 1.0, -friction],
                [-1.0, -1.0, -friction],
                [0.0, 0.0, -1.0],
                [0.0, 0.0, 1.0],
            ]
        )
        self.force_matrix = torch.block_diag(*([face] * len(dynamics.feet)))
        self.force_bound = limits.new_tensor(
            [0.0, 0.0, 0.0, 0.0, -minimum_normal_force, maximum_normal_force]
        ).repeat(len(dynamics.feet))

    def violations(self, torques, forces):
        torque = (torques.abs() - self.torque_limits).clamp_min(0).amax(-1)
        force = (forces.flatten(-2) @ self.force_matrix.T - self.force_bound).clamp_min(0).amax(-1)
        return torque, force

    def feasible(self, torques, forces):
        tv, fv = self.violations(torques, forces)
        return (
            torch.isfinite(torques).all(-1)
            & torch.isfinite(forces).flatten(-2).all(-1)
            & (tv <= self.tolerance)
            & (fv <= self.tolerance)
        )
