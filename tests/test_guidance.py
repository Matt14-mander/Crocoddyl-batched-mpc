"""Common reference geometry, interpolation derivatives and reference lifecycle."""

import subprocess
import sys
from dataclasses import replace

import numpy as np
import pytest

from crocoddyl_batched_mpc.guidance import GuidanceBuffer, pose_interpolate
from crocoddyl_batched_mpc.models.go2_arm import load_go2_arm
from crocoddyl_batched_mpc.references.standing_arm import (
    StandingArmReferenceGenerator,
    model_identity,
    reach_task,
)


@pytest.fixture(scope="module")
def common():
    pytest.importorskip("pinocchio")
    robot = load_go2_arm()
    q0 = robot.standing_configuration()
    times = np.linspace(10, 11, 21)
    packet = StandingArmReferenceGenerator(robot).generate(
        q0, times, *reach_task(times, robot.evaluate(q0).tcp_pose, [0.01, 0.02, 0])
    )
    return robot, packet


def test_optional_imports_are_isolated():
    code = (
        "import sys; import crocoddyl_batched_mpc.guidance; "
        "import crocoddyl_batched_mpc.references.standing_arm; "
        "import crocoddyl_batched_mpc.controllers.whole_body_qp; "
        "assert 'pinocchio' not in sys.modules; assert 'crocoddyl' not in sys.modules"
    )
    subprocess.run([sys.executable, "-B", "-c", code], check=True)


def test_common_reference_fk_feet_and_mapping(common):
    robot, reference = common
    lower, upper = robot.position_limits
    for x, sample in zip(reference.nominal_states, reference.guidance.samples):
        value = robot.evaluate(x[:25])
        np.testing.assert_allclose(value.foot_positions, sample.foot_anchors, atol=1e-7)
        np.testing.assert_allclose(value.tcp_pose, sample.tcp_pose, atol=2e-7)
        np.testing.assert_allclose(value.com, sample.com_position, atol=1e-12)
        assert np.all(x[7:25] >= lower) and np.all(x[7:25] <= upper)
    np.testing.assert_array_equal(
        reference.for_whole_body().nominal_states, reference.nominal_states
    )
    np.testing.assert_allclose(reference.for_whole_body().times, np.arange(21) * 0.05)


def test_model_identity_includes_gravity_not_encoded_in_urdf(common):
    robot, reference = common
    config = {**robot.config, "gravity_world_m_s2": [0, 0, -9.5]}
    changed = load_go2_arm(config)
    assert changed.urdf_xml == robot.urdf_xml
    assert model_identity(changed) != reference.guidance.samples[0].model_id


def test_reference_rejects_unreachable_task(common):
    robot, reference = common
    times = np.linspace(0, 1, 3)
    q0 = reference.nominal_states[0, :25]
    with pytest.raises(ValueError, match="rejected"):
        StandingArmReferenceGenerator(robot).generate(
            q0, times, *reach_task(times, robot.evaluate(q0).tcp_pose, [2.0, 0, 0])
        )


def test_reference_rejects_inconsistent_angular_derivative(common):
    robot, reference = common
    times = np.linspace(0, 1, 3)
    q0 = reference.nominal_states[0, :25]
    pose, velocity, acceleration = reach_task(times, robot.evaluate(q0).tcp_pose, [0, 0.01, 0])
    velocity[1, 3] = 0.1
    with pytest.raises(ValueError, match="angular derivatives"):
        StandingArmReferenceGenerator(robot).generate(q0, times, pose, velocity, acceleration)


def test_interpolation_matches_task_derivatives(common):
    _, reference = common
    packet = reference.guidance
    t, h = 10.373, 1e-5
    before, actual, after = packet.sample(t - h), packet.sample(t), packet.sample(t + h)
    np.testing.assert_allclose(
        (after.tcp_pose[:3, 3] - before.tcp_pose[:3, 3]) / (2 * h),
        actual.tcp_velocity[:3],
        atol=2e-9,
    )
    np.testing.assert_allclose(
        (after.tcp_velocity - before.tcp_velocity) / (2 * h), actual.tcp_acceleration, atol=2e-8
    )
    for knot in packet.times[1:-1]:
        left, right = packet.sample(knot - 1e-8), packet.sample(knot + 1e-8)
        for field in ("tcp_velocity", "tcp_acceleration", "joint_velocity", "joint_acceleration"):
            np.testing.assert_allclose(getattr(left, field), getattr(right, field), atol=2e-6)


def test_rotation_interpolation_derivatives():
    pin = pytest.importorskip("pinocchio")
    a, b = np.eye(4), np.eye(4)
    a[:3, :3] = pin.exp3(np.array([0.1, -0.2, 0.05]))
    b[:3, :3] = pin.exp3(np.array([0.3, 0.1, -0.2])) @ a[:3, :3]
    v0, v1 = np.array([0, 0, 0, 0.02, -0.04, 0.01]), np.array([0, 0, 0, -0.03, 0.02, 0.01])
    a0, a1 = np.array([0, 0, 0, 0.01, 0.02, -0.01]), np.array([0, 0, 0, -0.01, 0.01, 0.03])
    t, h = 0.37, 1e-5
    pose, velocity, acceleration = pose_interpolate(a, v0, a0, b, v1, a1, 1.0, t)
    left = pose_interpolate(a, v0, a0, b, v1, a1, 1.0, t - h)
    right = pose_interpolate(a, v0, a0, b, v1, a1, 1.0, t + h)
    numerical = pin.log3(right[0][:3, :3] @ left[0][:3, :3].T) / (2 * h)
    np.testing.assert_allclose(numerical, velocity[3:], atol=1e-8)
    np.testing.assert_allclose((right[1] - left[1]) / (2 * h), acceleration, atol=2e-7)
    for elapsed, expected in ((0.0, (a, v0, a0)), (1.0, (b, v1, a1))):
        result = pose_interpolate(a, v0, a0, b, v1, a1, 1.0, elapsed)
        for x, y in zip(result, expected):
            np.testing.assert_allclose(x, y, atol=1e-10)
    assert np.isfinite(pose).all()


@pytest.mark.parametrize("change", ["times", "nonfinite", "pose", "anchors", "validity", "model"])
def test_packet_rejects_bad_contract(common, change):
    _, reference = common
    packet = reference.guidance
    with pytest.raises(ValueError):
        if change == "times":
            replace(packet, times=packet.times[::-1])
        elif change == "nonfinite":
            replace(packet.samples[0], com_position=np.full(3, np.nan))
        elif change == "pose":
            replace(packet.samples[0], tcp_pose=np.zeros((4, 4)))
        elif change == "validity":
            replace(packet, valid_until=packet.times[-1] + 1)
        else:
            field = (
                dict(foot_anchors=packet.samples[-1].foot_anchors + 0.01)
                if change == "anchors"
                else dict(model_id="another")
            )
            replace(packet, samples=(*packet.samples[:-1], replace(packet.samples[-1], **field)))


def test_packet_is_a_snapshot(common):
    _, reference = common
    sample = reference.guidance.samples[0]
    position = sample.com_position.copy()
    copy = replace(sample, com_position=position)
    position[:] = 100
    np.testing.assert_array_equal(copy.com_position, sample.com_position)
    with pytest.raises(ValueError):
        copy.com_position[0] = 3


def test_ambiguous_rotation_is_rejected_before_packet_acceptance(common):
    pin = pytest.importorskip("pinocchio")
    _, reference = common
    packet = reference.guidance
    pose = packet.samples[-1].base_pose.copy()
    pose[:3, :3] = pin.exp3(np.array([np.pi, 0, 0])) @ pose[:3, :3]
    with pytest.raises(ValueError, match="less than pi"):
        replace(packet, samples=(*packet.samples[:-1], replace(packet.samples[-1], base_pose=pose)))


def test_buffer_retains_valid_old_packet_and_c2_fallback(common):
    _, reference = common
    packet = replace(reference.guidance, valid_until=10.63)
    buffer = GuidanceBuffer(packet.samples[0], transition_duration=0.7)
    assert buffer.submit(packet, 10.0)
    assert not buffer.submit(replace(packet, valid=False, sequence=1, reason="solver failed"), 10.3)
    expected = packet.sample(10.5)
    actual, mode = buffer.sample(10.5)
    assert mode == "active"
    np.testing.assert_array_equal(actual.tcp_pose, expected.tcp_pose)
    start = packet.sample(packet.valid_until)
    sample, mode = buffer.sample(10.63000001)
    assert mode == "return_to_standing"
    np.testing.assert_allclose(sample.tcp_velocity, start.tcp_velocity, atol=1e-7)
    np.testing.assert_allclose(sample.tcp_acceleration, start.tcp_acceleration, atol=1e-6)
    final, mode = buffer.sample(11.5)
    assert mode == "standing"
    np.testing.assert_allclose(final.tcp_pose, packet.samples[0].tcp_pose)
    np.testing.assert_array_equal(final.tcp_velocity, np.zeros(6))
    with pytest.raises(ValueError, match="backwards"):
        buffer.sample(10)
    buffer.reset()
    assert buffer.submit(packet, 10.0) and buffer.rejected_packets == 0


def test_buffer_rejects_expired_or_wrong_support(common):
    _, reference = common
    packet = reference.guidance
    buffer = GuidanceBuffer(packet.samples[0])
    assert not buffer.submit(packet, 12.0)
    buffer.reset()
    shifted = tuple(
        replace(sample, foot_anchors=sample.foot_anchors + 0.1) for sample in packet.samples
    )
    assert not buffer.submit(replace(packet, samples=shifted), 10.0)
    assert buffer.sample(10.0)[1] == "standing"
    assert buffer.submit(packet, 10.0)
    assert not buffer.submit(packet, 10.1)
