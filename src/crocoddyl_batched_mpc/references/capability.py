"""Optional, shared offline posture redistribution before a paired experiment.

All methods must receive the resulting CommonReference. A rolling C1 worker
rejects an unsupported preview instead of silently changing its joint prior.
"""

import time
from dataclasses import replace

import numpy as np

from ..guidance import Guidance, _array, _jacobian_derivative, _left_jacobian
from .configuration_centroidal import CapabilityError
from .standing_arm import CommonReference, _derivatives, model_identity


def redistribute_configuration(
    checker,
    q,
    home,
    tcp_pose,
    *,
    translation_bound=(0.08, 0.06, 0.075),
    rotation_bound=0.3,
    allow_base_rotation=True,
    iterations=12,
    step=0.04,
    payload_mass_kg=0.0,
):
    """Bounded nullspace search at fixed feet and TCP, with explicit failure.

    Deterministic coordinate search in the permitted task nullspace;
    every trial is projected back to the nonlinear task and checked by the
    original support QP. Search failure is not a global infeasibility proof.
    """
    import pinocchio as pin

    robot = checker.robot
    q, home = robot._q(q), robot._q(home)
    bounds = _array(translation_bound, (3,), "redistribution bounds")
    if np.any(bounds <= 0) or not 0 < rotation_bound < np.pi / 2 or not 0 < step <= 0.1:
        raise ValueError("invalid redistribution bounds/step")
    if type(iterations) is not int or iterations < 1:
        raise ValueError("invalid redistribution budget")
    if type(allow_base_rotation) is not bool:
        raise ValueError("allow_base_rotation must be boolean")
    columns = np.arange(24) if allow_base_rotation else np.r_[np.arange(3), np.arange(6, 24)]
    target = np.asarray(tcp_pose, dtype=float)
    if target.shape != (4, 4) or not np.isfinite(target).all():
        raise ValueError("invalid redistribution TCP")
    initial_rotation = robot.frame_pose(home, "base")[:3, :3]
    lower, upper = robot.position_limits
    kinematic_time, static_time, trials = 0.0, 0.0, 0

    def geometry(configuration):
        value = robot.evaluate(configuration)
        residual = np.r_[
            (checker.anchors - value.foot_positions).ravel(),
            target[:3, 3] - value.tcp_pose[:3, 3],
            pin.log3(target[:3, :3] @ value.tcp_pose[:3, :3].T),
        ]
        return residual, np.vstack([value.foot_jacobians.reshape(12, 24), value.tcp_jacobian])

    def project(candidate):
        if not allow_base_rotation:
            candidate[3:7] = home[3:7]
        for _ in range(40):
            residual, jac = geometry(candidate)
            if np.max(abs(residual)) <= 1e-7:
                rotation = robot.frame_pose(candidate, "base")[:3, :3]
                if (
                    np.max(abs(candidate[:3] - home[:3]) - bounds) <= 1e-9
                    and np.linalg.norm(pin.log3(rotation @ initial_rotation.T))
                    <= rotation_bound + 1e-9
                    and np.all(candidate[7:] >= lower)
                    and np.all(candidate[7:] <= upper)
                ):
                    return candidate
            delta = np.zeros(24)
            delta[columns] = np.linalg.lstsq(jac[:, columns], residual, rcond=1e-10)[0]
            delta *= min(1, 0.08 / max(np.max(abs(delta)), 1e-12))
            candidate = robot.integrate(candidate, delta)
            candidate[:3] = np.clip(candidate[:3], home[:3] - bounds, home[:3] + bounds)
            candidate[7:] = np.clip(candidate[7:], lower + 1e-8, upper - 1e-8)
            rotation = robot.frame_pose(candidate, "base")[:3, :3]
            delta_r = pin.log3(rotation @ initial_rotation.T)
            if np.linalg.norm(delta_r) > rotation_bound:
                candidate[3:7] = pin.Quaternion(
                    pin.exp3(delta_r * rotation_bound / np.linalg.norm(delta_r)) @ initial_rotation
                ).coeffs()
        return None

    def check(configuration):
        nonlocal static_time, trials
        start = time.perf_counter()
        result = checker.evaluate(configuration, payload_mass_kg=payload_mass_kg)
        static_time += time.perf_counter() - start
        trials += 1
        return result

    start = time.perf_counter()
    q = project(q)
    kinematic_time += time.perf_counter() - start
    if q is None:
        return None, dict(
            status="KINEMATIC_PROJECTION_FAILED",
            kinematic_time_s=kinematic_time,
            quasi_static_time_s=static_time,
            trials=trials,
        )
    best = check(q)
    before = best.diagnostics["peak_utilization"]
    for _ in range(iterations):
        if best.accepted:
            break
        start = time.perf_counter()
        _, jac = geometry(q)
        _, singular, vt = np.linalg.svd(jac[:, columns], full_matrices=True)
        reduced_null = vt[np.sum(singular > 1e-9) :].T
        null = np.zeros((24, reduced_null.shape[1]))
        null[columns] = reduced_null
        kinematic_time += time.perf_counter() - start
        choice = None
        for direction in null.T:
            for sign in (-1, 1):
                start = time.perf_counter()
                candidate = project(robot.integrate(q, sign * step * direction))
                kinematic_time += time.perf_counter() - start
                if candidate is None:
                    continue
                result = check(candidate)
                if (
                    result.diagnostics["qp_converged"]
                    and result.diagnostics["force_violation"] <= checker.settings.tolerance
                    and result.diagnostics["equilibrium_residual"] <= checker.settings.tolerance
                    and result.diagnostics["peak_utilization"]
                    < best.diagnostics["peak_utilization"] - 1e-8
                ):
                    choice, best = candidate, result
        if choice is None:
            step *= 0.5
        else:
            q = choice
    diagnostics = dict(
        status="SUCCESS" if best.accepted else "NO_CAPABLE_POSTURE_WITHIN_SEARCH_BUDGET",
        before_peak_utilization=before,
        after_peak_utilization=best.diagnostics["peak_utilization"],
        trials=trials,
        kinematic_time_s=kinematic_time,
        quasi_static_time_s=static_time,
        capability=best.diagnostics,
    )
    return (q if best.accepted else None), diagnostics


def prepare_capable_reference(robot, common, checker, *, allow_redistribution=True):
    """Return the unchanged prior if capable, otherwise a new shared prior.

    Initial measured posture is immutable. Invalid, out-of-budget or too-fast
    adjusted sequences raise CapabilityError with all attempted node results.
    """
    import pinocchio as pin

    if checker.robot is not robot or common.guidance.samples[0].model_id != model_identity(robot):
        raise ValueError("capability reference model mismatch")
    if not np.array_equal(checker.anchors, common.guidance.samples[0].foot_anchors):
        raise ValueError("capability reference contact mismatch")
    started = time.perf_counter()
    checks = [checker.evaluate(x[:25]) for x in common.nominal_states]
    rejected = [i for i, r in enumerate(checks) if not r.accepted]
    diagnostics = dict(
        initial_rejected_nodes=rejected,
        adjusted_nodes=[],
        attempts=[],
        kinematic_time_s=0.0,
        quasi_static_time_s=time.perf_counter() - started,
    )
    if not rejected:
        diagnostics.update(accepted=True, status="UNCHANGED_CAPABLE_COMMON")
        return common, diagnostics
    if 0 in rejected or not allow_redistribution:
        diagnostics.update(
            accepted=False,
            status="INITIAL_CONFIGURATION_UNSUPPORTED"
            if 0 in rejected
            else "REDISTRIBUTION_DISABLED",
        )
        raise CapabilityError(diagnostics["status"], diagnostics)
    configurations = common.nominal_states[:, :25].copy()
    home = configurations[0]
    for k in rejected:
        q, diag = redistribute_configuration(
            checker,
            configurations[k],
            home,
            common.guidance.samples[k].tcp_pose,
            translation_bound=common.diagnostics.get("base_translation_bound", (0.08, 0.06, 0.075)),
            rotation_bound=common.diagnostics.get("base_rotation_bound", 0.3),
            allow_base_rotation=common.diagnostics.get("allow_base_rotation", False),
        )
        diagnostics["attempts"].append(dict(node=k, **diag))
        diagnostics["kinematic_time_s"] += diag["kinematic_time_s"]
        diagnostics["quasi_static_time_s"] += diag["quasi_static_time_s"]
        if q is None:
            diagnostics.update(accepted=False, status="NO_CAPABLE_SHARED_REFERENCE")
            raise CapabilityError(diagnostics["status"], diagnostics)
        configurations[k] = q
        diagnostics["adjusted_nodes"].append(k)
    started = time.perf_counter()
    times = common.guidance.times
    values = [robot.evaluate(q) for q in configurations]
    com = np.stack([v.com for v in values])
    com_v, com_a = _derivatives(com, times)
    joint_v, joint_a = _derivatives(configurations[:, 7:], times)
    base_v, base_a = _derivatives(configurations[:, :3], times)
    bases = [robot.frame_pose(q, "base") for q in configurations]
    rot = np.stack([pin.log3(b[:3, :3] @ bases[0][:3, :3].T) for b in bases])
    rot_v, rot_a = _derivatives(rot, times)
    samples, states = [], []
    for k, sample in enumerate(common.guidance.samples):
        jac = _left_jacobian(rot[k])
        omega = jac @ rot_v[k]
        alpha = jac @ rot_a[k] + _jacobian_derivative(rot[k], rot_v[k]) @ rot_v[k]
        if (
            np.max(abs(joint_v[k]) - robot.velocity_limits) > 1e-6
            or np.max(abs(base_v[k])) > 0.5
            or np.max(abs(omega)) > 1.0
        ):
            diagnostics.update(accepted=False, status="REDISTRIBUTED_REFERENCE_TOO_FAST", node=k)
            raise CapabilityError(diagnostics["status"], diagnostics)
        samples.append(
            replace(
                sample,
                source="common_capability",
                com_position=com[k],
                com_velocity=com_v[k],
                com_acceleration=com_a[k],
                base_pose=bases[k],
                base_velocity=np.r_[base_v[k], omega],
                base_acceleration=np.r_[base_a[k], alpha],
                joint_position=configurations[k, 7:],
                joint_velocity=joint_v[k],
                joint_acceleration=joint_a[k],
            )
        )
        states.append(
            np.r_[
                configurations[k],
                bases[k][:3, :3].T @ base_v[k],
                bases[k][:3, :3].T @ omega,
                joint_v[k],
            ]
        )
    diagnostics["kinematic_time_s"] += time.perf_counter() - started
    diagnostics.update(accepted=True, status="REDISTRIBUTED_SHARED_REFERENCE")
    packet = Guidance(
        times,
        tuple(samples),
        common.guidance.issued_at,
        common.guidance.valid_until,
        common.guidance.sequence,
    )
    return CommonReference(
        packet, np.stack(states), dict(common.diagnostics, capability=diagnostics)
    ), diagnostics
