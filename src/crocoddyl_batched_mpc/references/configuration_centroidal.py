"""C1 adapters; preserve shared TCP/base/joint supervision at the WBC boundary."""

import time
from dataclasses import replace

import numpy as np

from ..models.configuration_centroidal import QuasiStaticCapability, momentum_map
from ..planners.standing_configuration_centroidal import ConfigurationReference
from .centroidal import guidance_from_centroidal
from .standing_arm import model_identity


class CapabilityError(ValueError):
    def __init__(self, message, diagnostics):
        super().__init__(message)
        self.diagnostics = diagnostics


def configuration_reference(robot, common, capability=None):
    identity = model_identity(robot)
    if common.guidance.samples[0].model_id != identity:
        raise ValueError("configuration reference uses a different model")
    checker = capability or QuasiStaticCapability(robot, common.guidance.samples[0].foot_anchors)
    if checker.robot is not robot or not np.array_equal(
        checker.anchors, common.guidance.samples[0].foot_anchors
    ):
        raise ValueError("capability checker model/contact mismatch")
    started = time.perf_counter()
    maps = [momentum_map(robot, x[:25]) for x in common.nominal_states]
    mapped_at = time.perf_counter()
    checks = [checker.evaluate(x[:25]) for x in common.nominal_states]
    static_at = time.perf_counter()
    quantities = [checker.quantities(x[:25]) for x in common.nominal_states]
    diagnostics = dict(
        accepted=all(c.accepted for c in checks),
        rejected_nodes=[k for k, c in enumerate(checks) if not c.accepted],
        static_nodes=[dict(status=c.status, **c.diagnostics) for c in checks],
        mapping_time_s=mapped_at - started,
        quasi_static_time_s=static_at - mapped_at,
        quantities_time_s=time.perf_counter() - static_at,
        max_map_condition=max(m.condition for m in maps),
        nominal_momentum_peak=float(
            max(np.linalg.norm(m.matrix @ x[25:]) for m, x in zip(maps, common.nominal_states))
        ),
        configuration_policy="common priors; no private joint/base redistribution in worker",
    )
    if not diagnostics["accepted"]:
        raise CapabilityError("C1 quasi-static reference rejected", diagnostics)
    return ConfigurationReference(
        common.guidance.times - common.guidance.times[0],
        np.stack([np.r_[m.com, m.matrix @ x[25:]] for m, x in zip(maps, common.nominal_states)]),
        np.stack([s.contact_forces for s in common.guidance.samples[:-1]]),
        common.nominal_states,
        np.stack([m.matrix for m in maps]),
        np.stack([m.base_inverse for m in maps]),
        np.stack([g for _, _, g in quantities]),
        np.stack([j for _, j, _ in quantities]),
        np.stack([c.forces for c in checks]),
        checker.torque_limits,
        identity,
        diagnostics,
    )


def guidance_from_configuration(robot, planner, plan, common, previous_forces, **kwargs):
    """Recompute maps and static balances, then reuse the frozen export contract.

    The mapped base twist constrains the C1 optimization; it is diagnostic only at
    the Guidance boundary. No additional joint/base supervision is granted to C1.
    """
    r = planner.configuration
    if r is None or r.model_id != model_identity(robot):
        raise ValueError("C1 model/context mismatch")
    if not np.array_equal(r.nominal_states, common.nominal_states) or not np.array_equal(
        r.times, common.guidance.times - common.guidance.times[0]
    ):
        raise ValueError("C1 common configuration preview changed")
    # Never trust a stale accepted flag or caller-supplied cached physics.
    checker = QuasiStaticCapability(robot, planner.model.anchors)
    if not np.array_equal(r.torque_limits, checker.torque_limits):
        raise ValueError("C1 export requires original model torque limits")
    s = planner.settings
    for k, x in enumerate(r.nominal_states):
        mapping = momentum_map(robot, x[:25])
        value, jac, gravity = checker.quantities(x[:25])
        for old, current in (
            (r.maps[k], mapping.matrix),
            (r.inverses[k], mapping.base_inverse),
            (r.gravities[k], gravity),
            (r.foot_jacobians[k], jac),
        ):
            if not np.allclose(old, current, atol=1e-12, rtol=0):
                raise ValueError("C1 cached configuration physics changed")
        forces = r.static_forces[k]
        tau = gravity[6:] - jac[:, 6:].T @ forces.ravel()
        equilibrium = gravity - jac.T @ forces.ravel() - robot.actuation_matrix() @ tau
        lower, upper = robot.position_limits
        if (
            np.max(abs(equilibrium)) > s.constraint_tolerance
            or np.max(abs(tau) - robot.torque_limits) > s.constraint_tolerance
            or np.max(abs(forces[:, 0]) + abs(forces[:, 1]) - s.friction * forces[:, 2])
            > s.constraint_tolerance
            or np.min(forces[:, 2]) < s.normal_min - s.constraint_tolerance
            or np.max(forces[:, 2]) > s.normal_max + s.constraint_tolerance
            or np.max(abs(value.foot_positions - checker.anchors)) > checker.settings.foot_tolerance
            or np.any(x[7:25] < lower - s.constraint_tolerance)
            or np.any(x[7:25] > upper + s.constraint_tolerance)
        ):
            raise ValueError("C1 static capability failed independent export audit")
    packet = guidance_from_centroidal(robot, planner, plan, common, previous_forces, **kwargs)
    return replace(
        packet,
        samples=tuple(replace(s, source="C1_guidance") for s in packet.samples),
        reason="accepted C1 candidate; configuration/static checks only, dynamic execution via WBC",
    )
