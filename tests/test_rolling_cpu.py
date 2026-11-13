"""Receding-horizon feedback, manifold seeds and Guidance packet handoffs."""

import subprocess
import sys
from dataclasses import fields, replace

import numpy as np
import pytest

from crocoddyl_batched_mpc.controllers.rolling_whole_body_cpu import (
    RollingSettings,
    RollingWholeBodyCPU,
    preview_window,
    run_rolling_cpu_episode,
    shifted_seed,
)
from crocoddyl_batched_mpc.guidance import GuidanceBuffer, stationary
from crocoddyl_batched_mpc.models.go2_arm import load_go2_arm
from crocoddyl_batched_mpc.planners.standing_whole_body import (
    StandingWholeBodyPlanner,
    WholeBodySettings,
)
from crocoddyl_batched_mpc.references.standing_arm import StandingArmReferenceGenerator, reach_task

pin = pytest.importorskip("pinocchio")
pytest.importorskip("crocoddyl")


@pytest.fixture(scope="module")
def common():
    robot = load_go2_arm()
    q = robot.standing_configuration()
    times = np.linspace(3, 4, 21)  # Deliberately nonzero absolute origin.
    reference = StandingArmReferenceGenerator(robot).generate(
        q,
        times,
        *reach_task(times, robot.evaluate(q).tcp_pose, [0.005, 0.01, 0], return_to_start=True),
    )
    return robot, reference


def assert_samples_match(a, b, atol=1e-9):
    for field in fields(a):
        if isinstance(getattr(a, field.name), np.ndarray):
            np.testing.assert_allclose(
                getattr(a, field.name), getattr(b, field.name), atol=atol, rtol=0
            )


def test_optional_rolling_imports_remain_isolated():
    subprocess.run(
        [
            sys.executable,
            "-B",
            "-c",
            (
                "import sys; import crocoddyl_batched_mpc.controllers.rolling_whole_body_cpu; "
                "assert 'pinocchio' not in sys.modules and 'crocoddyl' not in sys.modules"
            ),
        ],
        check=True,
    )


def test_preview_uses_absolute_task_time_and_preserves_moving_endpoints(common):
    robot, task = common
    window = preview_window(robot, task, 3.175, WholeBodySettings())
    for t, sample in zip(window.guidance.times, window.guidance.samples):
        assert_samples_match(sample, task.guidance.sample(t))
    assert np.linalg.norm(window.guidance.samples[0].tcp_velocity) > 0.001
    assert np.linalg.norm(window.guidance.samples[-1].tcp_velocity) > 0.001
    assert np.linalg.norm(window.nominal_states[-1, 25:]) > 0.001
    tail = preview_window(robot, task, 3.9, WholeBodySettings())
    assert_samples_match(tail.guidance.samples[-1], stationary(task.guidance.samples[-1]))
    with pytest.raises(ValueError, match="origin"):
        preview_window(robot, task, 2.9, WholeBodySettings())


def test_handoff_is_c2_and_can_be_interrupted(common):
    _, task = common
    packet = task.guidance
    buffer = GuidanceBuffer(packet.samples[0], handoff_duration=0.1)
    assert buffer.submit(packet, 3.0)
    # A new reference may differ in pose and derivatives, not only sequence.
    samples = tuple(replace(s, com_position=s.com_position + [0.002, 0, 0]) for s in packet.samples)
    updated = replace(packet, samples=samples, sequence=1)
    outgoing, _ = buffer.sample(3.2)
    assert buffer.submit(updated, 3.2)
    assert_samples_match(buffer.sample(3.2)[0], outgoing)
    interrupted = buffer.sample(3.25)[0]
    assert buffer.submit(replace(packet, sequence=2), 3.25)
    assert_samples_match(buffer.sample(3.25)[0], interrupted)
    # Compare both sides against the exact target endpoint, including a and w.
    endpoint = packet.sample(3.35)
    assert_samples_match(buffer.sample(3.35)[0], endpoint, 1e-8)
    assert_samples_match(buffer.sample(3.35 + 1e-9)[0], endpoint, 1e-6)
    assert buffer.accepted_packets == 3


@pytest.mark.parametrize(
    "kind", ["invalid", "expired", "future", "sequence", "model", "anchors", "handoff_coverage"]
)
def test_rejected_new_packet_preserves_old_validity_and_handoff(common, kind):
    _, task = common
    packet = task.guidance
    buffer = GuidanceBuffer(packet.samples[0], handoff_duration=0.1)
    assert buffer.submit(packet, 3.0)
    candidate = replace(packet, sequence=1)
    if kind == "invalid":
        candidate = replace(candidate, valid=False)
    elif kind == "expired":
        candidate = replace(candidate, valid_until=3.01)
    elif kind == "future":
        candidate = replace(
            candidate,
            issued_at=3.03,
            times=packet.times + 0.03,
            samples=tuple(replace(s, time=s.time + 0.03) for s in packet.samples),
        )
    elif kind == "sequence":
        candidate = packet
    elif kind == "handoff_coverage":
        candidate = replace(candidate, valid_until=3.1)
    else:
        change = (
            dict(model_id="wrong")
            if kind == "model"
            else dict(foot_anchors=packet.samples[0].foot_anchors + 0.1)
        )
        candidate = replace(candidate, samples=tuple(replace(s, **change) for s in packet.samples))
    expected = buffer.sample(3.02)[0]
    assert not buffer.submit(candidate, 3.02)
    assert buffer.last_rejection == kind
    assert buffer.current is packet and buffer.current.valid_until == 4.0
    assert_samples_match(buffer.sample(3.02)[0], expected)
    assert buffer.rejected_packets == 1


def test_expiry_and_recovery_handoff_are_continuous(common):
    _, task = common
    packet = replace(task.guidance, valid_until=3.3)
    buffer = GuidanceBuffer(packet.samples[0], transition_duration=0.3, handoff_duration=0.05)
    assert buffer.submit(packet, 3.0)
    old = buffer.sample(3.3)[0]
    assert_samples_match(buffer.sample(3.3 + 1e-9)[0], old, 1e-6)
    outgoing, mode = buffer.sample(3.45)
    assert mode == "return_to_standing"
    assert buffer.submit(replace(task.guidance, sequence=1), 3.45)
    assert_samples_match(buffer.sample(3.45)[0], outgoing)
    assert_samples_match(buffer.sample(3.5)[0], task.guidance.sample(3.5), 1e-8)
    buffer.reset()
    assert buffer.current is None and buffer.handoff is None and buffer.accepted_packets == 0


def test_ambiguous_cross_packet_rotation_is_rejected(common):
    _, task = common
    buffer = GuidanceBuffer(task.guidance.samples[0], handoff_duration=0.1)
    assert buffer.submit(task.guidance, 3.0)
    samples = []
    for sample in task.guidance.samples:
        pose = sample.base_pose.copy()
        pose[:3, :3] = pin.exp3(np.array([np.pi, 0, 0])) @ pose[:3, :3]
        samples.append(replace(sample, base_pose=pose))
    packet = replace(task.guidance, samples=tuple(samples), sequence=1)
    assert not buffer.submit(packet, 3.0)
    assert buffer.last_rejection == "handoff_geometry"
    assert buffer.current is task.guidance


def test_no_accepted_w_packet_is_not_a_successful_rolling_episode(common, monkeypatch):
    robot, _ = common
    q = robot.standing_configuration()
    times = np.linspace(0, 0.1, 3)
    task = StandingArmReferenceGenerator(robot).generate(
        q, times, *reach_task(times, robot.evaluate(q).tcp_pose, [0, 0, 0])
    )
    rolling = RollingWholeBodyCPU(robot, task)

    def fail(*args, **kwargs):
        raise RuntimeError("no solver result")

    monkeypatch.setattr(rolling.planner, "solve", fail)
    report, trace, _ = run_rolling_cpu_episode(robot, task, rolling=rolling, extra_time=0)
    assert not report["passed"] and "no_accepted_W_packet" in report["failures"]
    assert report["termination"] is None
    assert np.all(trace["modes"] == "standing")
    assert np.max(np.abs(trace["torques"])) > 1  # WBC holds support, not zero torque.


def test_w_updates_measured_state_without_moving_support_or_base_bounds(common):
    robot, task = common
    planner = StandingWholeBodyPlanner(robot, initial_state=task.nominal_states[0])
    anchors, base = planner.anchors.copy(), planner.base_anchor.copy()
    measured = task.nominal_states[0].copy()
    measured[0] += 1e-6
    measured[25] = 1e-5
    previous = planner._canonical_control(planner.static_control) * 0.99
    window = preview_window(robot, task, 3.0, planner.settings)
    plan = planner.solve(window.for_whole_body(), previous, initial_state=measured)
    assert plan.diagnostics["accepted_for_task"]
    np.testing.assert_allclose(plan.mpc.xs[0, 0].numpy(), measured, atol=1e-12)
    np.testing.assert_array_equal(plan.native_states[0, 49:], planner._native_control(previous))
    np.testing.assert_array_equal(planner.anchors, anchors)
    np.testing.assert_array_equal(planner.base_anchor, base)
    # Fractional timestamp shift and terminal extension of a real accepted plan.
    seed = shifted_seed(robot, plan, 3.0, 3.075, measured, planner.settings)
    np.testing.assert_array_equal(seed[0][0], measured)
    np.testing.assert_allclose(np.linalg.norm(seed[0][:, 3:7], axis=1), 1, atol=1e-12)
    xs = plan.mpc.xs[0].numpy()
    expected_q = pin.interpolate(
        robot.model, robot.to_native_q(xs[2, :25]), robot.to_native_q(xs[3, :25]), 0.5
    )
    np.testing.assert_allclose(seed[0][1, :25], robot.from_native_q(expected_q), atol=1e-12)
    np.testing.assert_array_equal(seed[0][-1, :25], xs[-1, :25])
    np.testing.assert_array_equal(seed[0][-1, 25:], np.zeros(24))
    assert shifted_seed(robot, plan, 3.0, 3.5, measured, planner.settings) is None


@pytest.fixture(scope="module")
def executions(common):
    robot, task = common
    results = []
    for warm in (True, False):
        rolling = RollingWholeBodyCPU(robot, task, settings=RollingSettings(warm_start=warm))
        report, trace, _ = run_rolling_cpu_episode(robot, task, rolling=rolling)
        results.append((rolling, report, trace))
    return results


def test_continuous_native_w_replanning_uses_measured_feedback_and_wbc_torque(executions):
    rolling, report, trace = executions[0]
    assert report["passed"], report["failures"]
    assert report["rolling"]["accepted"] == 10
    assert report["rolling"]["shifted_attempts"] == 9
    for event, (origin, plan) in zip(rolling.events, rolling.plans):
        index = round((origin - 3.0) / 0.005)
        np.testing.assert_array_equal(event["initial_state"], trace["states"][index])
        np.testing.assert_allclose(plan.mpc.xs[0, 0].numpy(), trace["states"][index], atol=1e-12)
        previous = trace["torques"][index - 1] if index else np.zeros(18)
        np.testing.assert_array_equal(event["previous_applied_torque"], previous)
        np.testing.assert_array_equal(
            plan.native_states[0, 49:], rolling.planner._native_control(previous)
        )
    assert report["modes"]["handoff"] > 0
    assert report["modes"]["return_to_standing"] > 0
    assert report["modes"]["standing"] > 0


def test_warm_and_cold_feedback_have_equivalent_execution(executions):
    (_, warm, a), (_, cold, b) = executions
    assert warm["passed"] and cold["passed"]
    assert cold["rolling"]["accepted"] == 10 and cold["rolling"]["shifted_attempts"] == 0
    np.testing.assert_allclose(a["states"], b["states"], atol=2e-4, rtol=0)


def test_solver_failure_keeps_last_cache_and_packet_then_expires(common, monkeypatch):
    robot, task = common
    rolling = RollingWholeBodyCPU(robot, task)
    x = task.nominal_states[0]
    rolling.update(3.0, x, np.zeros(18), rolling.buffer)
    cached, current = rolling.cached_plan, rolling.buffer.current
    assert cached is not None

    def fail(*args, **kwargs):
        raise RuntimeError("injected solver failure")

    monkeypatch.setattr(rolling.planner, "solve", fail)
    rolling.update(3.1, x, np.zeros(18), rolling.buffer)
    assert rolling.cached_plan is cached and rolling.buffer.current is current
    assert not rolling.events[-1]["accepted"]
    assert rolling.buffer.current.valid_until == 3.5
    assert rolling.buffer.sample(3.4)[1] == "active"
    assert rolling.buffer.sample(3.6)[1] == "return_to_standing"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"replan_period": 0},
        {"warm_start": 1},
        {"handoff_duration": np.nan},
        {"recovery_duration": -1},
    ],
)
def test_invalid_rolling_settings(kwargs):
    with pytest.raises(ValueError):
        RollingSettings(**kwargs)
