"""Equivalence and rejection contracts for cached W, export and IPC validation."""

import copy
import pickle
from dataclasses import replace

import numpy as np
import pytest

from crocoddyl_batched_mpc.controllers.async_whole_body import (
    AsyncSettings,
    PlanRequest,
    StateSnapshot,
    WWorkerEngine,
)
from crocoddyl_batched_mpc.controllers.rolling_whole_body_cpu import preview_window
from crocoddyl_batched_mpc.guidance import SAMPLE_SHAPES
from crocoddyl_batched_mpc.guidance_batch import copy_guidance
from crocoddyl_batched_mpc.models.go2_arm import load_go2_arm
from crocoddyl_batched_mpc.planners.cached_whole_body import CachedWholeBodyPlanner
from crocoddyl_batched_mpc.planners.standing_whole_body import (
    StandingWholeBodyPlanner,
    WholeBodySettings,
    smooth_tcp_reference,
)
from crocoddyl_batched_mpc.references.standing_arm import (
    StandingArmReferenceGenerator,
    guidance_from_whole_body,
    reach_task,
)
from crocoddyl_batched_mpc.references.whole_body_export import WholeBodyGuidanceExporter

pytest.importorskip("pinocchio")
pytest.importorskip("crocoddyl")


@pytest.fixture(scope="module")
def solved():
    robot = load_go2_arm()
    q = robot.standing_configuration()
    times = np.linspace(0, 2, 41)
    common = StandingArmReferenceGenerator(robot).generate(
        q, times, *reach_task(times, robot.evaluate(q).tcp_pose, [0.01, 0.01, 0.01])
    )
    settings = WholeBodySettings(nodes=16)
    engine = WWorkerEngine(robot.config, common, settings, AsyncSettings())
    snap = StateSnapshot(
        0.0,
        common.nominal_states[0],
        engine.planner._canonical_control(engine.planner.static_control),
        common.guidance.samples[0],
        False,
    )
    packet, _ = engine.solve(PlanRequest("test", 1, 0.0, 0.0, 0.0, 0, snap))
    window = preview_window(robot, common, 0, settings)
    return robot, window, engine.pending[2], packet


def assert_packet_equal(actual, expected):
    assert (actual.issued_at, actual.valid_until, actual.sequence, actual.valid, actual.reason) == (
        expected.issued_at,
        expected.valid_until,
        expected.sequence,
        expected.valid,
        expected.reason,
    )
    np.testing.assert_array_equal(actual.times, expected.times)
    for a, e in zip(actual.samples, expected.samples):
        assert (a.time, a.model_id, a.source) == (e.time, e.model_id, e.source)
        for name in SAMPLE_SHAPES:
            np.testing.assert_array_equal(getattr(a, name), getattr(e, name))


def test_cached_action_cost_dynamics_and_derivatives_with_changing_targets():
    import pinocchio as pin

    fresh = StandingWholeBodyPlanner(settings=WholeBodySettings(nodes=3))
    cached = CachedWholeBodyPlanner(fresh.robot, fresh.settings)
    rng = np.random.default_rng(4)
    for iteration in range(3):
        reference = smooth_tcp_reference(fresh.robot, fresh.settings, (0.01, 0.02, 0.01))
        reference.tcp_poses[:, :3, 3] += iteration * 0.003
        reference.tcp_poses[:, :3, :3] = pin.exp3(np.array([0.01, 0.02, -0.01]) * iteration)
        reference.base_poses[:, :3, 3] += iteration * 0.002
        reference.base_poses[:, :3, :3] = pin.exp3(np.array([0.02, -0.01, 0.01]) * iteration)
        reference.tcp_velocities[:] += iteration * 0.03
        reference.com_positions[:] += iteration * 0.004
        reference.nominal_states[:, 7:] += iteration * 0.001
        initial = fresh.initial_state.copy()
        initial[7:] += iteration * 0.002
        previous = fresh._canonical_control(fresh.static_control) + iteration * 0.01
        problems = [
            p.build_problem(reference, previous, initial_state=initial) for p in (fresh, cached)
        ]
        np.testing.assert_array_equal(problems[0].x0, problems[1].x0)
        x = fresh.state.integrate(problems[0].x0, rng.normal(size=66) * 0.03)
        if iteration == 2:
            x[7] = fresh.robot.model.upperPositionLimit[7] + 0.01
            x[25] = 2.0  # Activate the joint/velocity barriers as well as interior costs.
        u = fresh.static_control + rng.normal(size=18) * 0.1
        for k in range(4):
            actions = [(p.runningModels[k] if k < 3 else p.terminalModel) for p in problems]
            data = [a.createData() for a in actions]
            for action, d in zip(actions, data):
                args = (x, u) if k < 3 else (x,)
                action.calc(d, *args)
                action.calcDiff(d, *args)
            for field in ("cost", "xnext", "Fx", "Fu", "Lx", "Lu", "Lxx", "Lxu", "Luu"):
                np.testing.assert_array_equal(getattr(data[0], field), getattr(data[1], field))
    assert cached.model_builds == 4
    cached.settings = replace(cached.settings, torque_change_weight=0.02)
    cached.build_problem(reference)
    assert cached.model_builds == 8
    cached.anchors[0, 0] += 0.001
    cached.build_problem(reference)
    assert cached.model_builds == 12


@pytest.mark.parametrize("optimized_joints", [False, True])
def test_export_exact_and_workspace_has_no_previous_result_alias(solved, optimized_joints):
    robot, common, plan, _ = solved
    exporter = WholeBodyGuidanceExporter(robot)
    actual = exporter.export(plan, common, use_optimized_joints=optimized_joints)
    expected = guidance_from_whole_body(robot, plan, common, use_optimized_joints=optimized_joints)
    assert_packet_equal(actual, expected)
    exporter.export(plan, common)
    assert_packet_equal(actual, expected)
    assert not actual.samples[0].base_pose.flags.writeable
    assert not np.shares_memory(actual.samples[0].contact_forces, plan.forces_world)


@pytest.mark.parametrize(
    "field,k",
    [("forces_world", 0), ("forces_world", -1), ("accelerations", 2), ("accelerations", -1)],
)
def test_export_rechecks_dynamics_even_if_plan_claims_acceptance(solved, field, k):
    robot, common, plan, _ = solved
    damaged = copy.deepcopy(plan)
    getattr(damaged, field)[k] += 1
    for export in (
        lambda: guidance_from_whole_body(robot, damaged, common),
        lambda: WholeBodyGuidanceExporter(robot).export(damaged, common),
    ):
        with pytest.raises(ValueError, match="model/contact"):
            export()


def test_batch_copy_preserves_sample_time_tolerance_and_detaches_pickle(solved):
    packet = pickle.loads(pickle.dumps(solved[-1]))
    object.__setattr__(packet.samples[1], "time", packet.samples[1].time + 1e-10)
    actual = copy_guidance(packet)
    expected = replace(packet, samples=tuple(replace(s) for s in packet.samples))
    assert_packet_equal(actual, expected)
    for name in SAMPLE_SHAPES:
        a, p = getattr(actual.samples[1], name), getattr(packet.samples[1], name)
        assert not a.flags.writeable and not np.shares_memory(a, p)
        p.flat[0] += 1
        np.testing.assert_array_equal(a, getattr(expected.samples[1], name))


@pytest.mark.parametrize(
    "corruption",
    [
        "nan",
        "shape",
        "rotation",
        "reflection",
        "bottom",
        "jump",
        "time",
        "nan_time",
        "model",
        "source",
        "empty_source",
        "anchors",
        "times",
        "sequence",
        "valid",
        "issue",
        "expiry",
    ],
)
def test_batch_copy_rejects_same_malformed_packets_as_original(solved, corruption):
    packet = pickle.loads(pickle.dumps(solved[-1]))
    sample = packet.samples[-1]
    if corruption == "nan":
        sample.com_position[0] = np.nan
    elif corruption == "shape":
        object.__setattr__(sample, "joint_velocity", np.zeros(17))
    elif corruption == "rotation":
        sample.base_pose[0, 0] += 0.01
    elif corruption == "reflection":
        sample.tcp_pose[:3, 0] *= -1
    elif corruption == "bottom":
        sample.base_pose[3, 0] = 1e-8
    elif corruption == "jump":
        sample.base_pose[:3, :3] = np.diag([1.0, -1.0, -1.0]) @ packet.samples[-2].base_pose[:3, :3]
    elif corruption in ("time", "nan_time"):
        object.__setattr__(sample, "time", np.nan if corruption == "nan_time" else 3.0)
    elif corruption in ("model", "source", "empty_source"):
        object.__setattr__(
            sample,
            "model_id" if corruption == "model" else "source",
            "" if corruption == "empty_source" else "wrong",
        )
    elif corruption == "anchors":
        sample.foot_anchors[0, 0] += 0.001
    else:
        key, value = {
            "times": ("times", packet.times[::-1]),
            "sequence": ("sequence", 1.0),
            "valid": ("valid", 1),
            "issue": ("issued_at", np.nan),
            "expiry": ("valid_until", 99.0),
        }[corruption]
        object.__setattr__(packet, key, value)
    for validate in (
        copy_guidance,
        lambda p: replace(p, samples=tuple(replace(s) for s in p.samples)),
    ):
        with pytest.raises((ValueError, TypeError)):
            validate(packet)


@pytest.mark.parametrize("field", ["tcp_poses", "base_poses", "nominal_states", "times"])
def test_batch_reference_validation_keeps_all_knots_and_thresholds(field):
    planner = CachedWholeBodyPlanner(settings=WholeBodySettings(nodes=3))
    reference = smooth_tcp_reference(planner.robot, planner.settings)
    array = getattr(reference, field)
    if field.endswith("poses"):
        array[-1, 3, 0] = 2e-12
    elif field == "nominal_states":
        array[-1, 6] += 1e-7
    else:
        array[-1] += 2e-12
    with pytest.raises(ValueError):
        planner.build_problem(reference)


@pytest.mark.parametrize("damage", ["none", "gap", "quaternion", "torque"])
def test_cached_independent_gate_matches_fresh_for_every_node(solved, damage):
    robot, common, plan, _ = solved
    planners = [
        cls(robot, WholeBodySettings(nodes=16), common.nominal_states[0])
        for cls in (StandingWholeBodyPlanner, CachedWholeBodyPlanner)
    ]
    states, controls = plan.native_states.copy(), plan.native_controls.copy()
    if damage == "gap":
        states[-1, 0] += 0.01
    elif damage == "quaternion":
        states[3, 6] += 0.01
    elif damage == "torque":
        controls[-1, 0] += 100
    values = []
    for p in planners:
        reference = common.for_whole_body()
        problem = p.build_problem(reference)
        # Prime a reusable workspace with another configuration before checking.
        p._independent_node(states[0], controls[0])
        values.append(p.validate_candidate(problem, states, controls, reference))
    assert values[0][0] == values[1][0]
    for a, b in zip(values[0][1:], values[1][1:]):
        np.testing.assert_array_equal(a, b)
    if damage != "none":
        assert not values[1][0]["feasible"]
