"""C1's extra constraints, original export audit, ACK lifecycle and shared WBC."""

import copy
from dataclasses import replace

import numpy as np
import pytest

from crocoddyl_batched_mpc.controllers.async_whole_body import (
    AsyncSettings,
    PlanRequest,
    StateSnapshot,
)
from crocoddyl_batched_mpc.controllers.configuration_centroidal_cpu import C1WorkerEngine
from crocoddyl_batched_mpc.controllers.rolling_whole_body_cpu import preview_window
from crocoddyl_batched_mpc.controllers.whole_body_qp import WBCSettings, WholeBodyQP
from crocoddyl_batched_mpc.models.centroidal import CentroidalModel
from crocoddyl_batched_mpc.models.configuration_centroidal import QuasiStaticCapability
from crocoddyl_batched_mpc.models.go2_arm import load_go2_arm
from crocoddyl_batched_mpc.planners.standing_centroidal import StandingCentroidalPlanner
from crocoddyl_batched_mpc.planners.standing_configuration_centroidal import (
    ConfigurationSettings,
    StandingConfigurationPlanner,
)
from crocoddyl_batched_mpc.references.centroidal import centroidal_measurement
from crocoddyl_batched_mpc.references.configuration_centroidal import (
    CapabilityError,
    configuration_reference,
    guidance_from_configuration,
)
from crocoddyl_batched_mpc.references.standing_arm import StandingArmReferenceGenerator, reach_task

pytest.importorskip("pinocchio")


@pytest.fixture(scope="module")
def task():
    robot = load_go2_arm()
    q = robot.standing_configuration()
    times = np.arange(41) * 0.05
    common = StandingArmReferenceGenerator(robot).generate(
        q, times, *reach_task(times, robot.evaluate(q).tcp_pose, [0.01, 0.01, 0])
    )
    wbc = WholeBodyQP(robot, common.guidance.samples[0].foot_anchors, WBCSettings(dt=0.02))
    result = wbc.solve(q, np.zeros(24), common.guidance.samples[0], np.zeros(18))
    assert result.success
    return robot, common, result.torques


def request(common, tau, seq=1, now=0.0, ack=0, recovering=False):
    return PlanRequest(
        "c1test",
        seq,
        now,
        0.0,
        0.0,
        ack,
        StateSnapshot(
            now,
            common.nominal_states[int(round(now / 0.05))],
            tau,
            common.guidance.sample(now),
            recovering,
        ),
    )


def test_c1_guidance_uses_measured_state_preserves_supervision_and_wbc(task):
    robot, common, tau = task
    e = C1WorkerEngine(robot.config, common, ConfigurationSettings(nodes=8), AsyncSettings())
    req = request(common, tau)
    packet, diag = e.solve(req)
    window = preview_window(robot, common, 0.0, e.planner.settings)
    for got, expected in zip(packet.samples, window.guidance.samples):
        for prefix in ("tcp", "base", "joint"):
            for suffix in (
                ("position", "velocity", "acceleration")
                if prefix == "joint"
                else ("pose", "velocity", "acceleration")
            ):
                np.testing.assert_array_equal(
                    getattr(got, prefix + "_" + suffix), getattr(expected, prefix + "_" + suffix)
                )
        assert got.source == "C1_guidance" and not got.com_position.flags.writeable
    np.testing.assert_array_equal(
        e.last_plan.mpc.xs[0, 0].numpy(), centroidal_measurement(robot, req.snapshot.state)
    )
    assert diag["candidate"]["accepted_for_task"]
    assert diag["candidate"]["metrics"]["quasi_static_torque_violation"] <= 1e-6
    assert diag["phase_timings"]["quasi_static_s"] > 0
    result = WholeBodyQP(robot, packet.samples[0].foot_anchors, WBCSettings(dt=0.02)).solve(
        req.snapshot.state[:25], req.snapshot.state[25:], packet.samples[0], tau
    )
    assert result.success and len(result.torques) == 18


def test_acknowledged_warm_start_failure_and_recovery(task):
    robot, common, tau = task
    e = C1WorkerEngine(robot.config, common, ConfigurationSettings(nodes=8), AsyncSettings())
    e.solve(request(common, tau))
    _, d = e.solve(request(common, tau, 2, 0.1, 0))
    assert d["cached_sequence"] is None and d["seed"] == "common_forces"
    _, d = e.solve(request(common, tau, 3, 0.2, 2))
    assert d["cached_sequence"] == 2 and d["seed"] == "shifted_acknowledged"
    _, d = e.solve(request(common, tau, 4, 0.7, 3, True))
    assert d["recovery"] and d["seed"] == "common_forces"


def test_capability_failure_does_not_publish_or_seed_a_rejected_packet(task):
    robot, common, tau = task
    e = C1WorkerEngine(robot.config, common, ConfigurationSettings(nodes=8), AsyncSettings())
    e.solve(request(common, tau))
    original = e.capability
    e.capability = QuasiStaticCapability(
        e.robot, original.anchors, torque_limits=e.robot.torque_limits * 0.1
    )
    with pytest.raises(CapabilityError):
        e.solve(request(common, tau, 2, 0.1, 1))
    assert e.pending is None and e.cached[0] == 1
    assert e.history[-1]["diagnostics"]["status"] == "REFERENCE_CAPABILITY_REJECTED"
    e.capability = original
    _, diag = e.solve(request(common, tau, 3, 0.2, 1, True))
    assert diag["cached_sequence"] == 1 and diag["recovery"]
    assert e.pending[0] == 3


@pytest.mark.parametrize(
    "damage",
    [
        "state",
        "terminal",
        "force",
        "map",
        "inverse",
        "gravity",
        "static_force",
        "context",
        "times",
        "model",
    ],
)
def test_export_rechecks_full_physics_and_context(task, damage):
    robot, common, tau = task
    e = C1WorkerEngine(robot.config, common, ConfigurationSettings(nodes=8), AsyncSettings())
    req = request(common, tau)
    e.solve(req)
    plan = copy.deepcopy(e.last_plan)
    window = preview_window(robot, common, 0.0, e.planner.settings)
    previous = np.array(e.history[-1]["diagnostics"]["previous_force_estimate"])
    if damage in ("state", "terminal"):
        plan.mpc.xs[0, 0 if damage == "state" else -1, 6] += 0.01
    elif damage == "force":
        plan.mpc.us[0, -1, 2] += 1
    elif damage == "times":
        plan.times = plan.times + 0.01
    elif damage == "model":
        e.robot.model.gravity.linear = np.array([0, 0, -8.0])
    else:
        field = {
            "map": "maps",
            "inverse": "inverses",
            "gravity": "gravities",
            "static_force": "static_forces",
            "context": "nominal_states",
        }[damage]
        value = getattr(e.planner.configuration, field).copy()
        value.reshape(-1)[-1] += 1
        e.planner.configuration = replace(e.planner.configuration, **{field: value})
    with pytest.raises(ValueError):
        guidance_from_configuration(
            e.robot, e.planner, plan, window, previous, initial_state=req.snapshot.state
        )


def test_extra_qp_rows_match_direct_mapped_speed_and_static_torque(task):
    robot, common, _ = task
    s = ConfigurationSettings(nodes=4)
    window = preview_window(robot, common, 0.1, s)
    ref = configuration_reference(robot, window)
    model = CentroidalModel(
        robot.total_mass, robot.model.gravity.linear, window.guidance.samples[0].foot_anchors
    )
    p = StandingConfigurationPlanner(model, ref.states[0, :3], s)
    p.configuration = ref
    initial, forces = ref.states[0], ref.forces
    states = model.rollout(initial, forces, s.dt)
    H, g, G, b, matrices, offsets = p._qp(initial, states, forces, ref, forces[0], None)
    old = StandingCentroidalPlanner(model, ref.states[0, :3], s)
    H0, g0, G0, _, _, _ = old._qp(initial, states, forces, ref, forces[0], None)
    np.testing.assert_array_equal(H, H0)
    np.testing.assert_array_equal(g, g0)
    trial = forces.ravel() + np.linspace(-1, 1, forces.size)
    linear = np.einsum("nij,j->ni", matrices, trial) + offsets
    velocities = p.mapped_velocities(linear)
    violations = []
    for v in velocities[1:]:
        violations.extend([v - p.velocity_limits, -v - p.velocity_limits])
    for k in range(s.nodes + 1):
        tau = (
            ref.gravities[k, 6:]
            - ref.foot_jacobians[k, :, 6:].T @ trial.reshape(-1, 12)[min(k, s.nodes - 1)]
        )
        violations.extend([tau - ref.torque_limits, -tau - ref.torque_limits])
    np.testing.assert_allclose((G @ trial - b)[len(G0) :], np.concatenate(violations), atol=1e-12)


def test_direct_candidate_audit_detects_unrealizable_speed_and_torque(task):
    robot, common, tau = task
    e = C1WorkerEngine(robot.config, common, ConfigurationSettings(nodes=8), AsyncSettings())
    e.solve(request(common, tau))
    p, plan = e.planner, e.last_plan
    states, forces = plan.mpc.xs[0].numpy().copy(), plan.mpc.us[0].numpy().reshape(-1, 4, 3).copy()
    previous = np.array(e.history[-1]["diagnostics"]["previous_force_estimate"])
    states[-1, 6:] += 100
    forces[-1] *= 100
    audit = p.validate_candidate(states[0], states, forces, previous)
    assert not audit["feasible"]
    assert {"mapped_base_velocity", "quasi_static_torque"}.issubset(audit["failures"])
