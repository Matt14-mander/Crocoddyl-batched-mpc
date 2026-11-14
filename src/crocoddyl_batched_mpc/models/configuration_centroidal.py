"""C1 future-configuration momentum maps and static support capability.

World [linear, angular] momentum is about the instantaneous system COM.
Velocities use canonical local base twists + FL/FR/RL/RR/Z1 joint order.
The optional point payload is a STATIC gravity probe, never a dynamics model.
"""

import time
from dataclasses import dataclass

import numpy as np

from ..guidance import _array


@dataclass(frozen=True)
class MomentumMap:
    com: np.ndarray
    matrix: np.ndarray
    base_inverse: np.ndarray
    condition: float

    def base_velocity(self, momentum, joint_velocity):
        return self.base_inverse @ (
            _array(momentum, (6,), "momentum")
            - self.matrix[:, 6:] @ _array(joint_velocity, (18,), "joint velocity")
        )


def momentum_map(robot, q):
    import pinocchio as pin

    data = robot.model.createData()
    matrix = pin.computeCentroidalMap(robot.model, data, robot.to_native_q(q))[
        :, robot.v_indices
    ].copy()
    condition = float(np.linalg.cond(matrix[:, :6]))
    if not np.isfinite(condition) or condition > 1e8:
        raise ValueError("singular/ill-conditioned centroidal base map")
    return MomentumMap(
        _array(data.com[0], (3,), "COM"),
        _array(matrix, (6, 24), "centroidal map"),
        _array(np.linalg.inv(matrix[:, :6]), (6, 6), "base inverse"),
        condition,
    )


@dataclass(frozen=True)
class CapabilitySettings:
    friction: float = 0.6
    normal_min: float = 2.0
    normal_max: float = 200.0
    tolerance: float = 1e-6
    foot_tolerance: float = 1e-3
    qp_iterations: int = 60
    qp_tolerance: float = 1e-8

    def __post_init__(self):
        if type(self.qp_iterations) is not int or self.qp_iterations < 1:
            raise ValueError("invalid capability QP budget")
        if any(not np.isfinite(v) or v <= 0 for v in vars(self).values()):
            raise ValueError("capability settings must be positive and finite")
        if self.normal_min >= self.normal_max:
            raise ValueError("invalid support-force range")


@dataclass
class CapabilityResult:
    accepted: bool
    status: str
    forces: np.ndarray
    torques: np.ndarray
    diagnostics: dict


class QuasiStaticCapability:
    """Minimize peak normalized joint torque under exact floating-base balance.

    Decision (f12, t): |g_j - J_j.T f| <= t * torque_limits. Minimizing t
    provides an overload score for bounded posture redistribution without relaxing
    the physical acceptance limit t<=1. Original 24-DoF equilibrium is rechecked.
    No dynamic acceleration, collision, actuator thermal or hardware certificate.
    """

    def __init__(self, robot, anchors, settings=None, *, torque_limits=None):
        self.robot = robot
        self.anchors = _array(anchors, (4, 3), "anchors")
        self.settings = CapabilitySettings() if settings is None else settings
        if np.ptp(self.anchors[:, 2]) > 1e-8:
            raise ValueError("capability check requires horizontal support")
        self.torque_limits = _array(
            robot.torque_limits if torque_limits is None else torque_limits, (18,), "torque limits"
        )
        if np.any(self.torque_limits <= 0) or np.any(self.torque_limits > robot.torque_limits):
            raise ValueError("capability limits may derate but never increase model limits")

    def quantities(self, q, *, payload_mass_kg=0.0, payload_offset_tcp=(0.0, 0.0, 0.0)):
        if not np.isfinite(payload_mass_kg) or payload_mass_kg < 0:
            raise ValueError("payload mass must be finite and nonnegative")
        offset = _array(payload_offset_tcp, (3,), "payload offset")
        value = self.robot.evaluate(q, np.zeros(24))
        gravity = value.gravity.copy()
        # This model contains no added payload. The probe's world gravity wrench
        # is applied once, at an offset expressed in the TCP frame.
        force = payload_mass_kg * self.robot.model.gravity.linear
        wrench = np.r_[force, np.cross(value.tcp_pose[:3, :3] @ offset, force)]
        gravity -= value.tcp_jacobian.T @ wrench
        return value, value.foot_jacobians.reshape(12, 24), gravity

    def evaluate(self, q, *, payload_mass_kg=0.0, payload_offset_tcp=(0.0, 0.0, 0.0)):
        import torch

        from ..qp import solve_inequality_qp

        started = time.perf_counter()
        q = self.robot._q(q)
        s = self.settings
        value, jac, gravity = self.quantities(
            q, payload_mass_kg=payload_mass_kg, payload_offset_tcp=payload_offset_tcp
        )
        A = np.c_[jac[:, :6].T, np.zeros(6)]
        u, singular, vt = np.linalg.svd(A, full_matrices=True)
        rank = int(np.sum(singular > 1e-10 * singular[0]))
        particular = vt[:rank].T @ ((u[:, :rank].T @ gravity[:6]) / singular[:rank])
        null = vt[rank:].T
        cone = np.array(
            [[a, b, -s.friction] for a, b in ((1, 1), (1, -1), (-1, 1), (-1, -1))]
            + [[0, 0, -1], [0, 0, 1]]
        )
        G = np.vstack(
            [
                np.c_[np.kron(np.eye(4), cone), np.zeros(24)],
                np.c_[-jac[:, 6:].T, -self.torque_limits],
                np.c_[jac[:, 6:].T, -self.torque_limits],
                np.r_[np.zeros(12), -1][None],
            ]
        )
        bound = np.r_[
            np.tile([0, 0, 0, 0, -s.normal_min, s.normal_max], 4),
            -gravity[6:],
            gravity[6:],
            0,
        ]
        H, g = np.diag(np.r_[np.full(12, 1e-8), 1e-6]), np.r_[np.zeros(12), 1.0]
        reduced_G, reduced_b = G @ null, bound - G @ particular
        scale = np.maximum(np.maximum(np.linalg.norm(reduced_G, axis=1), abs(reduced_b)), 1e-8)
        result = solve_inequality_qp(
            torch.from_numpy(null.T @ H @ null),
            torch.from_numpy((null.T @ (H @ particular + g))[None]),
            torch.from_numpy(reduced_G / scale[:, None]),
            torch.from_numpy((reduced_b / scale)[None]),
            iterations=s.qp_iterations,
            tolerance=s.qp_tolerance,
        )
        solution = particular + null @ result.x[0].numpy()
        forces = solution[:12].reshape(4, 3)
        torques = gravity[6:] - jac[:, 6:].T @ solution[:12]
        finite = bool(np.isfinite(solution).all() and np.isfinite(torques).all())
        lower, upper = self.robot.position_limits
        metrics = dict(
            equilibrium_residual=float(
                np.max(
                    abs(gravity - jac.T @ solution[:12] - self.robot.actuation_matrix() @ torques)
                )
            ),
            force_violation=float(max(0, np.max(G[:24] @ solution - bound[:24]))),
            torque_violation=float(max(0, np.max(abs(torques) - self.torque_limits))),
            peak_utilization=float(np.max(abs(torques) / self.torque_limits)),
            foot_error=float(np.max(abs(value.foot_positions - self.anchors))),
            joint_violation=float(max(0, np.max(lower - q[7:]), np.max(q[7:] - upper))),
            epigraph_residual=float(max(0, np.max(G @ solution - bound))),
        )
        solved = (
            finite
            and bool(result.converged[0])
            and rank == 6
            and metrics["equilibrium_residual"] <= s.tolerance
            and metrics["force_violation"] <= s.tolerance
            and metrics["epigraph_residual"] <= s.tolerance
        )
        geometry_ok = (
            metrics["joint_violation"] <= s.tolerance and metrics["foot_error"] <= s.foot_tolerance
        )
        accepted = solved and geometry_ok and metrics["torque_violation"] <= s.tolerance
        if accepted:
            status = "SUCCESS"
        elif not geometry_ok:
            status = "REFERENCE_GEOMETRY_INVALID"
        else:
            status = "QUASI_STATIC_LIMIT" if solved else "NO_ACCEPTED_CAPABILITY_QP"
        return CapabilityResult(
            accepted,
            status,
            forces.copy(),
            torques.copy(),
            dict(
                **metrics,
                qp_converged=bool(result.converged[0]),
                qp_iterations=int(result.iterations[0]),
                equality_rank=rank,
                payload_mass_kg=float(payload_mass_kg),
                payload_scope="static gravity probe only; not added to centroidal dynamics",
                solve_time_s=time.perf_counter() - started,
            ),
        )
