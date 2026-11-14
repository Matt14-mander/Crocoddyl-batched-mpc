"""Shared reference and measured-state adapters for C0; no future inertia map."""

import numpy as np

from ..guidance import SAMPLE_SHAPES, _array
from ..guidance_batch import assemble_guidance
from ..planners.standing_centroidal import CentroidalReference
from .standing_arm import model_identity


def centroidal_measurement(robot, state):
    """World momentum about measured COM. Uses only the current full-body state."""
    import pinocchio as pin

    state = _array(state, (49,), "measured state")
    data = robot.model.createData()
    momentum = pin.computeCentroidalMomentum(
        robot.model, data, robot.to_native_q(state[:25]), robot.to_native_v(state[25:])
    )
    return np.r_[data.com[0], momentum.linear, momentum.angular]


def centroidal_reference(robot, common):
    if common.guidance.samples[0].model_id != model_identity(robot):
        raise ValueError("common reference uses a different model")
    samples = common.guidance.samples
    return CentroidalReference(
        common.guidance.times - common.guidance.times[0],
        np.stack(
            [np.r_[s.com_position, robot.total_mass * s.com_velocity, np.zeros(3)] for s in samples]
        ),
        np.stack([s.contact_forces for s in samples[:-1]]),
    )


def guidance_from_centroidal(
    robot, planner, plan, common, previous_forces, *, initial_state=None, sequence=1
):
    """C0 supplies COM and forces; TCP, base and nominal joints remain shared priors.

    Angular momentum is a planner state/diagnostic; the existing Guidance has no
    momentum task. We do not invent a base-inertia mapping (which would be C1).
    """
    if not plan.diagnostics.get("accepted_for_task", False) or not bool(plan.mpc.usable[0]):
        raise ValueError("C0 candidate was rejected")
    identity = model_identity(robot)
    if common.guidance.samples[0].model_id != identity:
        raise ValueError("common reference uses a different model")
    if (
        planner.model.mass != robot.total_mass
        or not np.array_equal(planner.model.gravity, robot.model.gravity.linear)
        or not np.array_equal(planner.model.anchors, common.guidance.samples[0].foot_anchors)
    ):
        raise ValueError("C0 model/contact episode mismatch")
    times = common.guidance.times
    if len(times) != len(plan.times) or not np.allclose(
        plan.times, times - times[0], atol=1e-9, rtol=0
    ):
        raise ValueError("C0 and common reference times differ")
    x0 = centroidal_measurement(
        robot, common.nominal_states[0] if initial_state is None else initial_state
    )
    states = plan.mpc.xs[0].detach().cpu().numpy()
    forces = plan.mpc.us[0].detach().cpu().numpy().reshape(-1, 4, 3)
    audit = planner.validate_candidate(x0, states, forces, previous_forces)
    if (
        not audit["feasible"]
        or not np.isfinite(plan.linearization_defect)
        or plan.linearization_defect > planner.settings.linearization_tolerance
    ):
        raise ValueError("C0 candidate failed independent export validation")
    columns = {name: [] for name in SAMPLE_SHAPES}
    for k, sample in enumerate(common.guidance.samples):
        force = forces[min(k, len(forces) - 1)]
        values = {name: getattr(sample, name) for name in SAMPLE_SHAPES}
        values.update(
            com_position=states[k, :3],
            com_velocity=states[k, 3:6] / planner.model.mass,
            com_acceleration=planner.model.gravity + force.sum(axis=0) / planner.model.mass,
            contact_forces=force,
        )
        for name, value in values.items():
            columns[name].append(value)
    return assemble_guidance(
        times,
        columns,
        model_id=identity,
        source="C0_guidance",
        issued_at=common.guidance.issued_at,
        valid_until=common.guidance.valid_until,
        sequence=sequence,
        reason=(
            f"accepted C0 centroidal candidate; converged={plan.converged}; "
            "no whole-body certificate"
        ),
    )
