"""Independent dense KKT CPU plant for fixed Go2-Z1 contacts, not a physics engine."""

from dataclasses import dataclass

import numpy as np


@dataclass
class CPUStep:
    q: np.ndarray
    v: np.ndarray
    acceleration: np.ndarray
    forces: np.ndarray
    dynamics_residual: float
    contact_residual: float


class FixedSupportCPU:
    """Raw Pinocchio forward dynamics, independent of WBC's equality elimination.

    A supplied external world wrench acts at the base origin and is unknown to
    WBC. The plant never reads a Guidance or writes a desired pose into its state.
    """

    def __init__(self, robot, anchors, *, dt=0.005, kp=100.0, kd=20.0):
        from ..guidance import _array

        self.robot = robot
        self.anchors = _array(anchors, (4, 3), "anchors")
        if not np.isfinite([dt, kp, kd]).all() or min(dt, kp, kd) <= 0 or dt > 0.02:
            raise ValueError("invalid CPU integration/stabilization settings")
        self.dt, self.kp, self.kd = dt, kp, kd

    def step(self, q, v, torques, *, external_wrench=None):
        import pinocchio as pin

        from ..guidance import _array

        robot, model = self.robot, self.robot.model
        qn, vn = robot.to_native_q(q), robot.to_native_v(v)
        tau = _array(torques, (18,), "torques")
        if np.any(np.abs(tau) > robot.torque_limits + 1e-6):
            raise ValueError("plant refuses out-of-range commanded torques")
        d = model.createData()
        mass = pin.crba(model, d, qn).copy()
        mass = np.triu(mass) + np.triu(mass, 1).T
        nonlinear = pin.nonLinearEffects(model, d, qn, vn).copy()
        pin.forwardKinematics(model, d, qn, vn, np.zeros(24))
        pin.computeJointJacobians(model, d, qn)
        pin.updateFramePlacements(model, d)
        jacobians, drift = [], []
        for frame, anchor in zip(robot.foot_ids, self.anchors):
            jac = pin.getFrameJacobian(model, d, frame, pin.LOCAL_WORLD_ALIGNED)[:3].copy()
            bias = pin.getFrameClassicalAcceleration(
                model, d, frame, pin.LOCAL_WORLD_ALIGNED
            ).linear.copy()
            jacobians.append(jac)
            drift.append(
                bias + self.kp * (d.oMf[frame].translation - anchor) + self.kd * (jac @ vn)
            )
        jac, drift = np.vstack(jacobians), np.concatenate(drift)
        applied = robot.actuation_matrix(native=True) @ tau
        if external_wrench is not None:
            wrench = _array(external_wrench, (6,), "external world wrench")
            rotation = pin.Quaternion(qn[3:7]).toRotationMatrix()
            applied[:6] += np.r_[rotation.T @ wrench[:3], rotation.T @ wrench[3:]]
        system = np.block([[mass, -jac.T], [jac, np.zeros((12, 12))]])
        answer = np.linalg.solve(system, np.r_[applied - nonlinear, -drift])
        acceleration, forces = answer[:24], answer[24:]
        qnext = pin.integrate(model, qn, self.dt * vn + self.dt**2 * acceleration)
        vnext = vn + self.dt * acceleration
        return CPUStep(
            robot.from_native_q(qnext),
            robot.from_native_v(vnext),
            robot.from_native_v(acceleration),
            forces.reshape(4, 3),
            float(np.max(np.abs(mass @ acceleration + nonlinear - applied - jac.T @ forces))),
            float(np.max(np.abs(jac @ acceleration + drift))),
        )
