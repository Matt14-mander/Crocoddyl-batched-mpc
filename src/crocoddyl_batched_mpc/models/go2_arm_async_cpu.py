"""Private reusable KKT workspace for the asynchronous Go2-Z1 CPU path only."""

import numpy as np

from .go2_arm_cpu import CPUStep, FixedSupportCPU


class AsyncCPUPlant(FixedSupportCPU):
    """Same independent dense contact equations; no WBC/planner state replacement.

    Data and matrix storage belong to this plant. Observation computes only FK
    and frame velocity, not mass, gravity and Jacobians unused by diagnostics.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data = self.robot.model.createData()
        self.observation_data = self.robot.model.createData()
        self.system = np.zeros((36, 36))
        self.rhs = np.zeros(36)
        self.jac = np.zeros((12, 24))
        self.drift = np.zeros(12)
        self.actuation = self.robot.actuation_matrix(native=True)
        self.limits = self.robot.torque_limits
        self.zero = np.zeros(24)

    def observe(self, q, v):
        import pinocchio as pin

        robot, model, d = self.robot, self.robot.model, self.observation_data
        qn, vn = robot.to_native_q(q), robot.to_native_v(v)
        pin.forwardKinematics(model, d, qn, vn)
        pin.updateFramePlacements(model, d)
        return (
            d.oMf[robot.tcp_id].homogeneous.copy(),
            np.stack([d.oMf[f].translation.copy() for f in robot.foot_ids]),
            np.stack(
                [
                    pin.getFrameVelocity(model, d, f, pin.LOCAL_WORLD_ALIGNED).linear.copy()
                    for f in robot.foot_ids
                ]
            ),
        )

    def step(self, q, v, torques, *, external_wrench=None):
        import pinocchio as pin

        from ..guidance import _array

        robot, model, d = self.robot, self.robot.model, self.data
        qn, vn = robot.to_native_q(q), robot.to_native_v(v)
        tau = _array(torques, (18,), "torques")
        if np.any(np.abs(tau) > self.limits + 1e-6):
            raise ValueError("plant refuses out-of-range commanded torques")
        mass = pin.crba(model, d, qn).copy()
        mass = np.triu(mass) + np.triu(mass, 1).T
        nonlinear = pin.nonLinearEffects(model, d, qn, vn).copy()
        pin.forwardKinematics(model, d, qn, vn, self.zero)
        pin.computeJointJacobians(model, d, qn)
        pin.updateFramePlacements(model, d)
        for i, (frame, anchor) in enumerate(zip(robot.foot_ids, self.anchors)):
            rows = slice(3 * i, 3 * i + 3)
            self.jac[rows] = pin.getFrameJacobian(model, d, frame, pin.LOCAL_WORLD_ALIGNED)[:3]
            bias = pin.getFrameClassicalAcceleration(
                model, d, frame, pin.LOCAL_WORLD_ALIGNED
            ).linear
            self.drift[rows] = (
                bias
                + self.kp * (d.oMf[frame].translation - anchor)
                + self.kd * (self.jac[rows] @ vn)
            )
        applied = self.actuation @ tau
        if external_wrench is not None:
            wrench = _array(external_wrench, (6,), "external world wrench")
            rotation = pin.Quaternion(qn[3:7]).toRotationMatrix()
            applied[:3] += rotation.T @ wrench[:3]
            applied[3:6] += rotation.T @ wrench[3:]
        self.system[:24, :24] = mass
        self.system[:24, 24:] = -self.jac.T
        self.system[24:, :24] = self.jac
        self.rhs[:24], self.rhs[24:] = applied - nonlinear, -self.drift
        answer = np.linalg.solve(self.system, self.rhs)
        acceleration, forces = answer[:24], answer[24:]
        qnext = pin.integrate(model, qn, self.dt * vn + self.dt**2 * acceleration)
        vnext = vn + self.dt * acceleration
        return CPUStep(
            robot.from_native_q(qnext),
            robot.from_native_v(vnext),
            robot.from_native_v(acceleration),
            forces.reshape(4, 3).copy(),
            float(np.max(np.abs(mass @ acceleration + nonlinear - applied - self.jac.T @ forces))),
            float(np.max(np.abs(self.jac @ acceleration + self.drift))),
        )
