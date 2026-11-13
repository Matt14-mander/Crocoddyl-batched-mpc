"""Inverse-dynamics QP gates, independent physical rollout and W integration."""

from dataclasses import replace

import numpy as np
import pytest

from crocoddyl_batched_mpc.controllers.standing_arm_cpu import run_cpu_episode
from crocoddyl_batched_mpc.controllers.whole_body_qp import WBCSettings, WholeBodyQP
from crocoddyl_batched_mpc.models.go2_arm import load_go2_arm
from crocoddyl_batched_mpc.models.go2_arm_cpu import FixedSupportCPU
from crocoddyl_batched_mpc.references.standing_arm import (
    StandingArmReferenceGenerator,
    guidance_from_whole_body,
    reach_task,
)


@pytest.fixture(scope="module")
def common():
    pytest.importorskip("pinocchio")
    robot = load_go2_arm()
    q = robot.standing_configuration()
    times = np.linspace(0, 0.5, 11)
    reference = StandingArmReferenceGenerator(robot).generate(
        q, times, *reach_task(times, robot.evaluate(q).tcp_pose, [0, 0.005, 0])
    )
    return robot, reference


def test_wbc_matches_independent_plant_and_constraints(common):
    robot, reference = common
    q, v = reference.nominal_states[0, :25], np.zeros(24)
    sample = reference.guidance.samples[0]
    wbc = WholeBodyQP(robot, sample.foot_anchors)
    result = wbc.solve(q, v, sample, np.zeros(18))
    assert result.success and result.diagnostics["qp_converged"]
    assert result.diagnostics["equality_rank"] == 36
    assert result.diagnostics["reduced_variables"] == 18
    plant = FixedSupportCPU(robot, wbc.anchors)
    step = plant.step(q, v, result.torques)
    np.testing.assert_allclose(step.acceleration, result.accelerations, atol=1e-7)
    np.testing.assert_allclose(step.forces, result.forces, atol=1e-7)
    model = robot.evaluate(q, v)
    residual = (
        model.mass_matrix @ result.accelerations
        + model.nonlinear_effects
        - robot.actuation_matrix() @ result.torques
        - model.foot_jacobians.reshape(12, 24).T @ result.forces.ravel()
    )
    np.testing.assert_allclose(residual, 0, atol=1e-7)
    assert np.max(np.abs(result.torques) / robot.torque_limits) < 1
    assert max(result.diagnostics["violations"].values()) <= 1e-6
    # Independent full equality-constrained optimum, without null-space reduction.
    problem = wbc.build_problem(q, v, sample, np.zeros(18))
    kkt = np.block([[problem["H"], problem["A"].T], [problem["A"], np.zeros((36, 36))]])
    optimum = np.linalg.solve(kkt, np.r_[-problem["g"], problem["b"]])[:54]
    assert np.max(problem["G"] @ optimum - problem["upper"]) < 0
    np.testing.assert_allclose(
        np.r_[result.accelerations, result.forces.ravel(), result.torques], optimum, atol=1e-6
    )
    assert wbc._last_solution is not None
    wbc.reset()
    assert wbc._last_solution is None


def test_active_hard_bounds_take_priority_over_task_tracking(common):
    robot, reference = common
    q = reference.nominal_states[0, :25]
    sample = reference.guidance.samples[0]
    pose = sample.tcp_pose.copy()
    pose[:3, 3] += 0.1
    sample = replace(sample, tcp_pose=pose)
    wbc = WholeBodyQP(robot, sample.foot_anchors)
    result = wbc.solve(q, np.zeros(24), sample, np.zeros(18))
    assert result.success
    problem = wbc.build_problem(q, np.zeros(24), sample, np.zeros(18))
    x = np.r_[result.accelerations, result.forces.ravel(), result.torques]
    slack = problem["upper"] - problem["G"] @ x
    assert np.min(slack) >= -1e-6 and np.min(np.abs(slack)) < 1e-5
    assert result.diagnostics["task_residuals"]["tcp"] > 0.1
    actual = FixedSupportCPU(robot, sample.foot_anchors).step(q, np.zeros(24), result.torques)
    np.testing.assert_allclose(actual.acceleration, result.accelerations, atol=1e-7)


def test_classical_acceleration_biases_match_derivatives(common):
    robot, reference = common
    q = reference.nominal_states[5, :25]
    v = np.random.default_rng(3).normal(0, 0.03, 24)
    wbc = WholeBodyQP(robot, reference.guidance.samples[0].foot_anchors)
    values = wbc._quantities(q, v)
    eps = 1e-6
    left = robot.evaluate(robot.integrate(q, -eps * v))
    right = robot.evaluate(robot.integrate(q, eps * v))
    for a, b, drift in (
        (left.tcp_jacobian, right.tcp_jacobian, values[4]),
        (left.foot_jacobians.reshape(12, 24), right.foot_jacobians.reshape(12, 24), values[5]),
        (left.com_jacobian, right.com_jacobian, values[6]),
    ):
        np.testing.assert_allclose((b - a) @ v / (2 * eps), drift, atol=1e-8)


def test_force_and_acceleration_mapping_under_rotated_base(common):
    pin = pytest.importorskip("pinocchio")
    robot, reference = common
    q = reference.nominal_states[0, :25].copy()
    q[3:7] = pin.Quaternion(pin.exp3(np.array([0.0, 0.0, 0.6]))).coeffs()
    q[:2] += [0.3, -0.2]
    times = np.linspace(0, 0.2, 3)
    shared = StandingArmReferenceGenerator(robot).generate(
        q, times, *reach_task(times, robot.evaluate(q).tcp_pose, [0, 0, 0])
    )
    sample = shared.guidance.samples[0]
    wbc = WholeBodyQP(robot, sample.foot_anchors)
    v = np.random.default_rng(9).normal(0, 0.001, 24)
    output = wbc.solve(q, v, sample, np.zeros(18))
    assert output.success
    step = FixedSupportCPU(robot, sample.foot_anchors).step(q, v, output.torques)
    np.testing.assert_allclose(step.acceleration, output.accelerations, atol=1e-7)
    np.testing.assert_allclose(step.forces, output.forces, atol=1e-7)


def test_no_command_when_hard_constraints_cannot_be_met(common):
    robot, reference = common
    sample = reference.guidance.samples[0]
    wbc = WholeBodyQP(robot, sample.foot_anchors)
    q = reference.nominal_states[0, :25].copy()
    # Even the maximum allowed negative next velocity cannot return this joint
    # within its upper position bound in one step: two hard inequalities conflict.
    q[7] = robot.position_limits[1][0] + 1.0
    result = wbc.solve(q, np.zeros(24), sample, np.zeros(18))
    assert not result.success
    assert result.torques is None and result.accelerations is None and result.forces is None


@pytest.mark.parametrize(
    "field,value",
    [("dt", 0), ("friction", -1), ("normal_max", 1), ("qp_iterations", True), ("task_kp", np.nan)],
)
def test_invalid_settings(field, value):
    with pytest.raises(ValueError):
        WBCSettings(**{field: value})


def test_reference_model_and_anchor_checks(common):
    robot, reference = common
    q = reference.nominal_states[0, :25]
    sample = reference.guidance.samples[0]
    wbc = WholeBodyQP(robot, sample.foot_anchors)
    for bad in (
        replace(sample, model_id="wrong"),
        replace(sample, foot_anchors=sample.foot_anchors + 0.01),
    ):
        with pytest.raises(ValueError):
            wbc.solve(q, np.zeros(24), bad, np.zeros(18))


def test_closed_loop_with_unmodelled_push_and_expired_reference(common):
    robot, reference = common
    packet = replace(reference.guidance, valid_until=0.3)

    def push(t):
        return np.array([5.0, 0, 0, 0, 0, 0]) if 0.1 <= t < 0.2 else np.zeros(6)

    report, trace, diagnostics = run_cpu_episode(
        robot, reference, packet, extra_time=1.0, external_wrench=push
    )
    assert report["passed"], report
    assert report["modes"]["return_to_standing"] > 0 and report["modes"]["standing"] > 0
    assert len(trace["states"]) == len(trace["torques"]) + 1
    assert report["metrics"]["nominal_acceleration_mismatch"] < 1e-6
    assert all(row["qp_converged"] for row in diagnostics)
    assert np.max(np.abs(trace["external_world_wrenches"])) == 5.0
    assert not report["realtime_accepted"]


def test_failed_wbc_terminates_before_applying_a_torque(common):
    robot, reference = common
    initial = reference.nominal_states[0].copy()
    initial[7] = robot.position_limits[1][0] + 1.0
    report, trace, _ = run_cpu_episode(robot, reference, initial_state=initial)
    assert not report["passed"] and report["termination"]["step"] == 0
    assert len(trace["states"]) == 1 and len(trace["torques"]) == 0


def test_numerical_exception_is_logged_without_executing_zero_torque(common, monkeypatch):
    robot, reference = common

    def failing_solve(*args, **kwargs):
        raise np.linalg.LinAlgError("injected numerical fault")

    monkeypatch.setattr(WholeBodyQP, "solve", failing_solve)
    report, trace, _ = run_cpu_episode(robot, reference)
    assert not report["passed"] and report["termination"]["status"] == "WBC_ERROR"
    assert len(trace["states"]) == 1 and len(trace["torques"]) == 0


def test_w_guidance_preserves_common_joint_and_tcp_references(common):
    pytest.importorskip("crocoddyl")
    from crocoddyl_batched_mpc.planners.standing_whole_body import StandingWholeBodyPlanner

    robot, reference = common
    plan = StandingWholeBodyPlanner(robot).solve(reference.for_whole_body())
    packet = guidance_from_whole_body(robot, plan, reference)
    for src, dst in zip(reference.guidance.samples, packet.samples):
        np.testing.assert_array_equal(src.joint_position, dst.joint_position)
        np.testing.assert_array_equal(src.tcp_pose, dst.tcp_pose)
    report, _, _ = run_cpu_episode(robot, reference, packet)
    assert report["passed"], report
    invalid = replace(plan, diagnostics={**plan.diagnostics, "accepted_for_task": False})
    with pytest.raises(ValueError, match="gates"):
        guidance_from_whole_body(robot, invalid, reference)
    invalid = replace(plan, accelerations=plan.accelerations + 1.0)
    with pytest.raises(ValueError, match="model/contact"):
        guidance_from_whole_body(robot, invalid, reference)
