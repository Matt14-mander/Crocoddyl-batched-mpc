"""Per-worker W export workspace, retaining independent per-knot dynamics checks."""

import numpy as np

from ..guidance import SAMPLE_SHAPES, _array
from ..guidance_batch import assemble_guidance
from .standing_arm import model_identity


class WholeBodyGuidanceExporter:
    """Compute only quantities consumed by Guidance; never reuse a gate verdict."""

    def __init__(self, robot):
        import pinocchio as pin

        self.robot = robot
        self.data = robot.model.createData()
        self.base_id = robot.model.getFrameId("base", pin.FrameType.BODY)

    def export(self, plan, common, *, sequence=1, use_optimized_joints=False, initial_state=None):
        import pinocchio as pin

        robot, model, data = self.robot, self.robot.model, self.data
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
        expected = (
            common.nominal_states[0]
            if initial_state is None
            else _array(initial_state, (49,), "measured initial state")
        )
        if not np.allclose(states[0], expected, atol=1e-9, rtol=0):
            raise ValueError("W and common reference initial states differ")
        torques = _array(plan.mpc.us[0].detach().cpu().numpy(), (len(states) - 1, 18), "W torques")
        forces = _array(plan.forces_world, (len(states), 4, 3), "W forces")
        acceleration = _array(plan.accelerations, (len(states), 24), "W accelerations")
        columns = {name: [] for name in SAMPLE_SHAPES}
        for k, state in enumerate(states):
            q, v = state[:25], state[25:]
            qn, vn = robot.to_native_q(q), robot.to_native_v(v)
            an = robot.to_native_v(acceleration[k])
            mass = pin.crba(model, data, qn).copy()
            mass = np.triu(mass) + np.triu(mass, 1).T
            nonlinear = pin.nonLinearEffects(model, data, qn, vn).copy()
            pin.computeJointJacobians(model, data, qn)
            pin.updateFramePlacements(model, data)
            jac = np.vstack(
                [
                    pin.getFrameJacobian(model, data, i, pin.LOCAL_WORLD_ALIGNED)[:3].copy()
                    for i in robot.foot_ids
                ]
            )
            feet = np.stack([data.oMf[i].translation.copy() for i in robot.foot_ids])
            # Preserve the original canonical summation order and gate tolerances.
            indices = robot.v_indices
            torque = torques[min(k, len(torques) - 1)]
            residual = (
                mass[np.ix_(indices, indices)] @ acceleration[k]
                + nonlinear[indices]
                - robot.actuation_matrix() @ torque
                - jac[:, indices].T @ forces[k].ravel()
            )
            sample = common.guidance.samples[k]
            if (
                np.max(np.abs(residual)) > 1e-6
                or np.max(np.linalg.norm(feet - sample.foot_anchors, axis=1)) > 1e-3
            ):
                raise ValueError("W plan does not match this model/contact episode")
            base = data.oMf[self.base_id].homogeneous.copy()
            rotation = base[:3, :3]
            com = pin.centerOfMass(model, data, qn, vn, an).copy()
            values = {name: getattr(sample, name) for name in SAMPLE_SHAPES}
            values.update(
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
                values.update(
                    joint_position=q[7:],
                    joint_velocity=v[6:],
                    joint_acceleration=acceleration[k, 6:],
                )
            for name, value in values.items():
                columns[name].append(value)
        return assemble_guidance(
            common.guidance.times,
            columns,
            model_id=common.guidance.samples[0].model_id,
            source="W_guidance",
            issued_at=common.guidance.issued_at,
            valid_until=common.guidance.valid_until,
            sequence=sequence,
            reason=f"accepted W candidate; converged={plan.converged}; "
            f"optimized_joints={use_optimized_joints}",
        )
