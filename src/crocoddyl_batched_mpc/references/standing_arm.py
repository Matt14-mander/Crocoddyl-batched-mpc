"""Shared fixed-support kinematic references and the W-to-Guidance adapter.

The common generator has no Crocoddyl dependency. It preserves four world foot
anchors and base orientation, allows base translation and all 18 joints, and
rejects unreachable targets. Kinematic validity is not an equilibrium certificate.
"""

import hashlib
import json
import time
from dataclasses import dataclass, replace

import numpy as np

from ..guidance import Guidance, GuidanceSample, _array, _pose, quintic


def model_identity(robot):
    # Gravity and tangent conventions are not encoded in URDF geometry/inertia.
    contract = dict(
        schema="go2_z1_guidance_v1",
        urdf=robot.urdf_xml,
        gravity_world=robot.model.gravity.linear.tolist(),
        canonical_joints=robot.joint_names,
        quaternion="xyzw",
        base_twist="local",
    )
    return hashlib.sha256(
        json.dumps(contract, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def reach_task(times, pose, offset, *, return_to_start=False):
    """C2 Cartesian reach (or reach-and-return), fixed world TCP orientation."""
    times = np.asarray(times, dtype=float)
    if (
        times.ndim != 1
        or len(times) < 3
        or not np.isfinite(times).all()
        or np.any(np.diff(times) <= 0)
    ):
        raise ValueError("task times must be finite, increasing, with at least three knots")
    pose = _pose(pose, (4, 4), "initial TCP pose")
    offset = _array(offset, (3,), "offset")
    poses = np.repeat(pose[None], len(times), axis=0)
    velocities = np.zeros((len(times), 6))
    accelerations = np.zeros_like(velocities)
    duration = times[-1] - times[0]
    for k, now in enumerate(times):
        t = now - times[0]
        if return_to_start:
            half = 0.5 * duration
            start, end = (np.zeros(3), offset) if t <= half else (offset, np.zeros(3))
            elapsed = t if t <= half else t - half
            h = half
        else:
            start, end, elapsed, h = np.zeros(3), offset, t, duration
        p, v, a = quintic(
            start, np.zeros(3), np.zeros(3), end, np.zeros(3), np.zeros(3), h, elapsed
        )
        poses[k, :3, 3] += p
        velocities[k, :3], accelerations[k, :3] = v, a
    return poses, velocities, accelerations


def _derivatives(values, times):
    velocity = np.gradient(values, times, axis=0, edge_order=2)
    velocity[[0, -1]] = 0
    acceleration = np.gradient(velocity, times, axis=0, edge_order=2)
    acceleration[[0, -1]] = 0
    return velocity, acceleration


@dataclass(frozen=True)
class CommonReference:
    guidance: Guidance
    nominal_states: np.ndarray
    diagnostics: dict

    def for_whole_body(self):
        from ..planners.standing_whole_body import StandingReference

        samples = self.guidance.samples
        return StandingReference(
            self.guidance.times - self.guidance.times[0],
            np.stack([s.tcp_pose for s in samples]),
            np.stack([s.tcp_velocity for s in samples]),
            np.stack([s.com_position for s in samples]),
            np.stack([s.base_pose for s in samples]),
            self.nominal_states.copy(),
        )


class StandingArmReferenceGenerator:
    def __init__(self, robot, *, max_iterations=100, tolerance=1e-7):
        if type(max_iterations) is not int or max_iterations < 1 or not 0 < tolerance < 1e-3:
            raise ValueError("invalid IK budget/tolerance")
        self.robot = robot
        self.max_iterations, self.tolerance = max_iterations, tolerance

    def generate(
        self,
        q0,
        times,
        tcp_poses,
        tcp_velocities,
        tcp_accelerations,
        *,
        sequence=0,
        valid_until=None,
    ):
        import pinocchio as pin

        started = time.perf_counter()
        q0 = self.robot._q(q0)
        times = np.asarray(times, dtype=float)
        if (
            times.ndim != 1
            or len(times) < 3
            or not np.isfinite(times).all()
            or np.any(np.diff(times) <= 0)
        ):
            raise ValueError("reference requires at least three increasing finite times")
        n = len(times)
        tcp_poses = _pose(tcp_poses, (n, 4, 4), "TCP poses")
        tcp_velocities = _array(tcp_velocities, (n, 6), "TCP velocities")
        tcp_accelerations = _array(tcp_accelerations, (n, 6), "TCP accelerations")
        if (
            np.max(np.abs(tcp_velocities[[0, -1]])) > 1e-10
            or np.max(np.abs(tcp_accelerations[[0, -1]])) > 1e-10
        ):
            raise ValueError("first reference generator requires stationary task endpoints")
        if not np.allclose(tcp_poses[:, :3, :3], tcp_poses[0, :3, :3], atol=1e-10, rtol=0):
            raise ValueError("first reference generator requires constant TCP orientation")
        if (
            np.max(np.abs(tcp_velocities[:, 3:])) > 1e-10
            or np.max(np.abs(tcp_accelerations[:, 3:])) > 1e-10
        ):
            raise ValueError("constant TCP orientation requires zero angular derivatives")
        original = self.robot.evaluate(q0)
        if not np.allclose(tcp_poses[0], original.tcp_pose, atol=1e-9, rtol=0):
            raise ValueError("reference must start at the measured initial TCP")
        anchors = original.foot_positions
        if np.ptp(anchors[:, 2]) > 1e-8:
            raise ValueError("fixed-support generator requires horizontal coplanar anchors")
        lower, upper = self.robot.position_limits
        if np.any(q0[7:] < lower) or np.any(q0[7:] > upper):
            raise ValueError("initial configuration exceeds joint limits")
        columns = np.r_[np.arange(3), np.arange(6, 24)]
        q, configurations, residuals = q0.copy(), [], []
        for target in tcp_poses:
            previous = q.copy()
            for _ in range(self.max_iterations):
                quantities = self.robot.evaluate(q)
                error = np.r_[
                    (anchors - quantities.foot_positions).ravel(),
                    target[:3, 3] - quantities.tcp_pose[:3, 3],
                    pin.log3(target[:3, :3] @ quantities.tcp_pose[:3, :3].T),
                ]
                if np.max(np.abs(error)) < self.tolerance:
                    break
                jac = np.vstack(
                    [quantities.foot_jacobians.reshape(12, 24), quantities.tcp_jacobian]
                )[:, columns]
                # Weak pose/temporal priors resolve redundancy without prescribing W's solution.
                toward = self.robot.from_native_v(
                    pin.difference(
                        self.robot.model,
                        self.robot.to_native_q(q),
                        self.robot.to_native_q(previous),
                    )
                )
                home = self.robot.from_native_v(
                    pin.difference(
                        self.robot.model, self.robot.to_native_q(q), self.robot.to_native_q(q0)
                    )
                )
                dq = np.linalg.solve(
                    jac.T @ jac + 1e-7 * np.eye(21),
                    jac.T @ error + 1e-8 * (0.8 * toward[columns] + 0.2 * home[columns]),
                )
                dq *= min(1.0, 0.1 / max(np.max(np.abs(dq)), 1e-12))
                full = np.zeros(24)
                full[columns] = dq
                q = self.robot.integrate(q, full)
                q[7:] = np.clip(q[7:], lower + 1e-7, upper - 1e-7)
                q[:3] = np.clip(q[:3], q0[:3] - [0.06, 0.06, 0.04], q0[:3] + [0.06, 0.06, 0.04])
            actual = self.robot.evaluate(q)
            residual = max(
                np.max(np.abs(actual.foot_positions - anchors)),
                np.max(np.abs(actual.tcp_pose[:3, 3] - target[:3, 3])),
                np.max(np.abs(pin.log3(target[:3, :3].T @ actual.tcp_pose[:3, :3]))),
            )
            if residual > self.tolerance:
                raise ValueError(
                    f"kinematic reference rejected at knot {len(configurations)}: {residual:g}"
                )
            configurations.append(q.copy())
            residuals.append(float(residual))
        configurations = np.stack(configurations)
        values = [self.robot.evaluate(q) for q in configurations]
        com = np.stack([value.com for value in values])
        com_v, com_a = _derivatives(com, times)
        joint_v, joint_a = _derivatives(configurations[:, 7:], times)
        base_v, base_a = _derivatives(configurations[:, :3], times)
        force = np.tile(-self.robot.total_mass * self.robot.model.gravity.linear / 4, (4, 1))
        samples, states = [], []
        identity = model_identity(self.robot)
        for k, q in enumerate(configurations):
            base = self.robot.frame_pose(q, "base")
            samples.append(
                GuidanceSample(
                    float(times[k]),
                    identity,
                    "common_kinematic",
                    anchors,
                    tcp_poses[k],
                    tcp_velocities[k],
                    tcp_accelerations[k],
                    com[k],
                    com_v[k],
                    com_a[k],
                    base,
                    np.r_[base_v[k], np.zeros(3)],
                    np.r_[base_a[k], np.zeros(3)],
                    q[7:],
                    joint_v[k],
                    joint_a[k],
                    force,
                )
            )
            states.append(np.r_[q, base[:3, :3].T @ base_v[k], np.zeros(3), joint_v[k]])
        packet = Guidance(
            times,
            tuple(samples),
            float(times[0]),
            float(times[-1]) if valid_until is None else valid_until,
            sequence,
        )
        return CommonReference(
            packet,
            np.stack(states),
            {
                "max_kinematic_residual": max(residuals),
                "generation_time_s": time.perf_counter() - started,
                "support_forces": "uniform gravity prior; not a feasibility certificate",
                "derivatives": "analytic TCP; finite-difference C2 knot data for kinematic posture",
            },
        )


def guidance_from_whole_body(robot, plan, common, *, sequence=1, use_optimized_joints=False):
    """Keep shared TCP and nominal joints; use W's COM/base/force predictions.

    W joint trajectories are an opt-in ablation, not extra supervision silently
    given to W in the planned C0/C1/W comparison. No optimized torques are executed.
    """
    import pinocchio as pin

    if not plan.diagnostics.get("accepted_for_task", False) or not bool(plan.mpc.usable[0]):
        raise ValueError("W plan did not pass feasibility and task gates")
    if common.guidance.samples[0].model_id != model_identity(robot):
        raise ValueError("common reference uses a different model")
    origin = common.guidance.times[0]
    if len(plan.times) != len(common.guidance.times) or not np.allclose(
        plan.times, common.guidance.times - origin, atol=1e-9
    ):
        raise ValueError("W and common reference knot times differ")
    states = plan.mpc.xs[0].detach().cpu().numpy()
    if states.shape != (len(plan.times), 49) or not np.isfinite(states).all():
        raise ValueError("invalid W state trajectory")
    if not np.allclose(states[0], common.nominal_states[0], atol=1e-9, rtol=0):
        raise ValueError("W and common reference initial states differ")
    torques = _array(plan.mpc.us[0].detach().cpu().numpy(), (len(states) - 1, 18), "W torques")
    forces = _array(plan.forces_world, (len(states), 4, 3), "W forces")
    acceleration = _array(plan.accelerations, (len(states), 24), "W accelerations")
    samples = []
    for k, state in enumerate(states):
        q, v = state[:25], state[25:]
        value = robot.evaluate(q, v)
        torque = torques[min(k, len(torques) - 1)]
        residual = (
            value.mass_matrix @ acceleration[k]
            + value.nonlinear_effects
            - robot.actuation_matrix() @ torque
            - value.foot_jacobians.reshape(12, 24).T @ forces[k].ravel()
        )
        if (
            np.max(np.abs(residual)) > 1e-6
            or np.max(
                np.linalg.norm(
                    value.foot_positions - common.guidance.samples[k].foot_anchors, axis=1
                )
            )
            > 1e-3
        ):
            raise ValueError("W plan does not match this model/contact episode")
        data = robot.model.createData()
        qn, vn = robot.to_native_q(q), robot.to_native_v(v)
        an = robot.to_native_v(acceleration[k])
        com = pin.centerOfMass(robot.model, data, qn, vn, an).copy()
        base = robot.frame_pose(q, "base")
        rotation = base[:3, :3]
        sample = common.guidance.samples[k]
        kwargs = dict(
            source="W_guidance",
            com_position=com,
            com_velocity=data.vcom[0].copy(),
            com_acceleration=data.acom[0].copy(),
            base_pose=base,
            base_velocity=np.r_[rotation @ v[:3], rotation @ v[3:6]],
            base_acceleration=np.r_[
                rotation @ (acceleration[k, :3] + np.cross(v[3:6], v[:3])),
                rotation @ acceleration[k, 3:6],
            ],
            contact_forces=forces[k],
        )
        if use_optimized_joints:
            kwargs.update(
                joint_position=q[7:], joint_velocity=v[6:], joint_acceleration=acceleration[k, 6:]
            )
        samples.append(replace(sample, **kwargs))
    return Guidance(
        common.guidance.times,
        tuple(samples),
        common.guidance.issued_at,
        common.guidance.valid_until,
        sequence,
        reason=(
            f"accepted W candidate; converged={plan.converged}; "
            f"optimized_joints={use_optimized_joints}"
        ),
    )
