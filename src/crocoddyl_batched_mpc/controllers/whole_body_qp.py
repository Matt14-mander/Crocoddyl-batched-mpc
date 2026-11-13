"""Fixed-support inverse-dynamics QP for the canonical Go2-Z1 CPU model.

Decision [a24, f12, tau18]. Equalities are eliminated by SVD; the existing
Torch primal-dual QP solves the remaining hard inequalities. Every accepted
solution is checked again in the original physical variables. No installation,
old-controller changes, hidden PD, torque clipping or zero-torque fallback.
"""

import math
import time
from dataclasses import dataclass

import numpy as np

from ..guidance import GuidanceSample, _array
from ..references.standing_arm import model_identity


@dataclass(frozen=True)
class WBCSettings:
    dt: float = 0.005
    friction: float = 0.6
    normal_min: float = 2.0
    normal_max: float = 200.0
    contact_kp: float = 100.0
    contact_kd: float = 20.0
    task_kp: float = 100.0
    task_kd: float = 20.0
    joint_kp: float = 40.0
    joint_kd: float = 12.0
    tcp_weight: float = 200.0
    com_weight: float = 30.0
    base_position_weight: float = 5.0
    base_rotation_weight: float = 30.0
    posture_weight: float = 0.1
    force_weight: float = 1e-4
    torque_change_weight: float = 1e-3
    regularization: float = 1e-5
    qp_iterations: int = 40
    qp_tolerance: float = 1e-9
    constraint_tolerance: float = 1e-6

    def __post_init__(self):
        if type(self.qp_iterations) is not int or self.qp_iterations < 1:
            raise ValueError("qp_iterations must be a positive integer")
        for name, value in vars(self).items():
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.normal_min >= self.normal_max or self.dt > 0.02:
            raise ValueError("invalid force range or WBC integration step")


@dataclass
class WBCResult:
    success: bool
    status: str
    torques: np.ndarray | None
    accelerations: np.ndarray | None
    forces: np.ndarray | None
    diagnostics: dict


class WholeBodyQP:
    def __init__(self, robot, foot_anchors, settings=None):
        self.robot = robot
        self.settings = WBCSettings() if settings is None else settings
        self.anchors = _array(foot_anchors, (4, 3), "foot anchors")
        if np.ptp(self.anchors[:, 2]) > 1e-8:
            raise ValueError("first WBC supports fixed horizontal coplanar feet")
        self.model_id = model_identity(robot)
        self.reset()

    def reset(self):
        self._last_solution = None

    def _quantities(self, q, v):
        import pinocchio as pin

        robot, model = self.robot, self.robot.model
        quantities = robot.evaluate(q, v)
        data = model.createData()
        qn, vn = robot.to_native_q(q), robot.to_native_v(v)
        pin.forwardKinematics(model, data, qn, vn, np.zeros(24))
        pin.computeJointJacobians(model, data, qn)
        pin.updateFramePlacements(model, data)
        base_id = model.getFrameId("base", pin.FrameType.BODY)

        def frame(frame_id):
            jac = pin.getFrameJacobian(model, data, frame_id, pin.LOCAL_WORLD_ALIGNED)[
                :, robot.v_indices
            ].copy()
            drift = pin.getFrameClassicalAcceleration(
                model, data, frame_id, pin.LOCAL_WORLD_ALIGNED
            ).vector.copy()
            return jac, drift, data.oMf[frame_id].homogeneous.copy()

        base_jac, base_drift, base_pose = frame(base_id)
        _, tcp_drift, _ = frame(robot.tcp_id)
        foot_drift = np.concatenate([frame(fid)[1][:3] for fid in robot.foot_ids])
        pin.centerOfMass(model, data, qn, vn, np.zeros(24))
        return (
            quantities,
            base_jac,
            base_drift,
            base_pose,
            tcp_drift,
            foot_drift,
            data.acom[0].copy(),
        )

    def build_problem(self, q, v, reference, previous_torque):
        import pinocchio as pin

        q = self.robot._q(q)
        v = _array(v, (24,), "velocity")
        previous = _array(previous_torque, (18,), "previous torque")
        if not isinstance(reference, GuidanceSample) or reference.model_id != self.model_id:
            raise ValueError("reference uses the wrong model")
        if not np.array_equal(reference.foot_anchors, self.anchors):
            raise ValueError("contact changes require a new fixed-support episode")
        s = self.settings
        value, base_jac, base_drift, base_pose, tcp_drift, foot_drift, com_drift = self._quantities(
            q, v
        )
        foot_jac = value.foot_jacobians.reshape(12, 24)
        contact_target = (
            -foot_drift
            - s.contact_kp * (value.foot_positions - self.anchors).ravel()
            - s.contact_kd * (foot_jac @ v)
        )
        eq = np.zeros((36, 54))
        eq[:24, :24], eq[:24, 24:36], eq[:24, 36:] = (
            value.mass_matrix,
            -foot_jac.T,
            -self.robot.actuation_matrix(),
        )
        eq[24:, :24] = foot_jac
        rhs = np.r_[-value.nonlinear_effects, contact_target]
        hessian, gradient = s.regularization * np.eye(54), np.zeros(54)
        tasks = {}

        def task(name, matrix, target, weight):
            nonlocal hessian, gradient
            hessian += weight * (matrix.T @ matrix)
            gradient -= weight * (matrix.T @ target)
            tasks[name] = (matrix, target)

        def acceleration_task(name, jac, drift, error, target_v, target_a, weight):
            desired = target_a + s.task_kp * error + s.task_kd * (target_v - jac @ v) - drift
            matrix = np.zeros((len(desired), 54))
            matrix[:, :24] = jac
            task(name, matrix, desired, weight)

        error = np.r_[
            reference.tcp_pose[:3, 3] - value.tcp_pose[:3, 3],
            pin.log3(reference.tcp_pose[:3, :3] @ value.tcp_pose[:3, :3].T),
        ]
        acceleration_task(
            "tcp",
            value.tcp_jacobian,
            tcp_drift,
            error,
            reference.tcp_velocity,
            reference.tcp_acceleration,
            s.tcp_weight,
        )
        acceleration_task(
            "com",
            value.com_jacobian,
            com_drift,
            reference.com_position - value.com,
            reference.com_velocity,
            reference.com_acceleration,
            s.com_weight,
        )
        acceleration_task(
            "base_position",
            base_jac[:3],
            base_drift[:3],
            reference.base_pose[:3, 3] - base_pose[:3, 3],
            reference.base_velocity[:3],
            reference.base_acceleration[:3],
            s.base_position_weight,
        )
        acceleration_task(
            "base_rotation",
            base_jac[3:],
            base_drift[3:],
            pin.log3(reference.base_pose[:3, :3] @ base_pose[:3, :3].T),
            reference.base_velocity[3:],
            reference.base_acceleration[3:],
            s.base_rotation_weight,
        )
        selector = np.zeros((18, 54))
        selector[:, 6:24] = np.eye(18)
        joint_target = (
            reference.joint_acceleration
            + s.joint_kp * (reference.joint_position - q[7:])
            + s.joint_kd * (reference.joint_velocity - v[6:])
        )
        task("posture", selector, joint_target, s.posture_weight)
        force_selector = np.zeros((12, 54))
        force_selector[:, 24:36] = np.eye(12)
        task("force", force_selector, reference.contact_forces.ravel(), s.force_weight)
        torque_selector = np.zeros((18, 54))
        torque_selector[:, 36:] = np.eye(18)
        task("torque_change", torque_selector, previous, s.torque_change_weight)
        inequalities, bounds, groups = [], [], {}

        def inequality(name, matrix, upper):
            start = sum(len(b) for b in bounds)
            inequalities.append(matrix)
            bounds.append(upper)
            groups[name] = slice(start, start + len(upper))

        torque_limits = self.robot.torque_limits
        inequality(
            "torque", np.vstack([torque_selector, -torque_selector]), np.tile(torque_limits, 2)
        )
        cone = np.array(
            [
                [1, 1, -s.friction],
                [1, -1, -s.friction],
                [-1, 1, -s.friction],
                [-1, -1, -s.friction],
                [0, 0, -1],
                [0, 0, 1],
            ],
            dtype=float,
        )
        cones = np.zeros((24, 54))
        for leg in range(4):
            cones[6 * leg : 6 * leg + 6, 24 + 3 * leg : 27 + 3 * leg] = cone
        inequality("contact_force", cones, np.tile([0, 0, 0, 0, -s.normal_min, s.normal_max], 4))
        velocity = self.robot.velocity_limits
        inequality(
            "joint_velocity",
            np.vstack([s.dt * selector, -s.dt * selector]),
            np.r_[velocity - v[6:], velocity + v[6:]],
        )
        lower, upper = self.robot.position_limits
        # Match the CPU plant's semi-implicit update, not a different .5*dt^2 rule.
        nominal_next = q[7:] + s.dt * v[6:]
        inequality(
            "joint_position",
            np.vstack([s.dt**2 * selector, -(s.dt**2) * selector]),
            np.r_[upper - nominal_next, nominal_next - lower],
        )
        return dict(
            H=hessian,
            g=gradient,
            A=eq,
            b=rhs,
            G=np.vstack(inequalities),
            upper=np.concatenate(bounds),
            groups=groups,
            tasks=tasks,
        )

    def solve(self, q, v, reference, previous_torque):
        import torch

        from ..qp import solve_inequality_qp

        started = time.perf_counter()
        problem = self.build_problem(q, v, reference, previous_torque)
        A, b, H, g = (problem[k] for k in ("A", "b", "H", "g"))
        u, singular, vt = np.linalg.svd(A, full_matrices=True)
        rank = int(np.sum(singular > 1e-10 * singular[0]))
        particular = vt[:rank].T @ ((u[:, :rank].T @ b) / singular[:rank])
        null = vt[rank:].T
        diagnostics = {"equality_rank": rank, "reduced_variables": null.shape[1]}
        if (
            np.max(np.abs(A @ particular - b)) > self.settings.constraint_tolerance
            or null.shape[1] == 0
        ):
            return WBCResult(False, "INVALID_EQUALITIES", None, None, None, diagnostics)
        reduced_h = null.T @ H @ null
        reduced_g = null.T @ (H @ particular + g)
        objective_scale = max(1.0, float(np.max(np.diag(reduced_h))))
        reduced_h, reduced_g = reduced_h / objective_scale, reduced_g / objective_scale
        reduced_G, reduced_b = problem["G"] @ null, problem["upper"] - problem["G"] @ particular
        # Include the bound: position limits divided by dt^2 otherwise create
        # huge slacks and a complementarity floor in the existing IPM solver.
        row_scale = np.maximum(
            np.maximum(np.linalg.norm(reduced_G, axis=1), np.abs(reduced_b)), 1e-8
        )
        initial = (
            None
            if self._last_solution is None
            else torch.tensor(
                (null.T @ (self._last_solution - particular))[None], dtype=torch.float64
            )
        )
        inputs = [
            torch.tensor(x, dtype=torch.float64)
            for x in (
                reduced_h,
                reduced_g[None],
                reduced_G / row_scale[:, None],
                (reduced_b / row_scale)[None],
            )
        ]
        result = solve_inequality_qp(
            *inputs,
            initial=initial,
            iterations=self.settings.qp_iterations,
            tolerance=self.settings.qp_tolerance,
        )
        solution = particular + null @ result.x[0].detach().numpy()
        finite = bool(np.isfinite(solution).all())
        equality_error = float(np.max(np.abs(A @ solution - b))) if finite else float("inf")
        violation = np.maximum(problem["G"] @ solution - problem["upper"], 0)
        violations = {
            name: float(np.max(violation[indices])) if finite else float("inf")
            for name, indices in problem["groups"].items()
        }
        success = (
            finite
            and bool(result.converged[0])
            and equality_error <= self.settings.constraint_tolerance
            and max(violations.values()) <= self.settings.constraint_tolerance
        )
        diagnostics.update(
            qp_converged=bool(result.converged[0]),
            iterations=int(result.iterations[0]),
            equality_residual=equality_error,
            violations=violations,
            reduced_dual_residual=float(result.dual_residual[0]),
            task_residuals={
                name: float(np.linalg.norm(matrix @ solution - target))
                for name, (matrix, target) in problem["tasks"].items()
            },
            solve_time_s=time.perf_counter() - started,
        )
        if not success:
            self.reset()
            return WBCResult(False, "NO_ACCEPTED_QP_CANDIDATE", None, None, None, diagnostics)
        self._last_solution = solution.copy()
        return WBCResult(
            True,
            "SUCCESS",
            solution[36:].copy(),
            solution[:24].copy(),
            solution[24:36].reshape(4, 3).copy(),
            diagnostics,
        )
