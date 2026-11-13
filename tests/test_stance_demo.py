"""Rotating-base references, C2 high/low tasks, and gapped FDDP initialization."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from crocoddyl_batched_mpc.guidance import _left_jacobian
from crocoddyl_batched_mpc.models.go2_arm import load_go2_arm
from crocoddyl_batched_mpc.planners.standing_whole_body import (
    StandingWholeBodyPlanner,
    WholeBodySettings,
    smooth_tcp_reference,
)
from crocoddyl_batched_mpc.references.standing_arm import StandingArmReferenceGenerator

pin = pytest.importorskip("pinocchio")
pytest.importorskip("crocoddyl")

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "stance_demo", ROOT / "examples/go2_z1_stance_demo.py"
)
demo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(demo)


@pytest.fixture(scope="module")
def reference():
    robot = load_go2_arm()
    return robot, demo.build_reference(robot)


def test_high_low_reference_coordinates_and_support(reference):
    robot, common = reference
    rotations = []
    for state, sample in zip(common.nominal_states, common.guidance.samples):
        value = robot.evaluate(state[:25])
        np.testing.assert_allclose(value.foot_positions, sample.foot_anchors, atol=1e-7)
        np.testing.assert_allclose(value.tcp_pose, sample.tcp_pose, atol=2e-7)
        rotation = sample.base_pose[:3, :3]
        np.testing.assert_allclose(rotation @ state[25:28], sample.base_velocity[:3], atol=1e-12)
        np.testing.assert_allclose(rotation @ state[28:31], sample.base_velocity[3:], atol=1e-12)
        rotations.append(pin.log3(rotation))
    assert np.max(np.abs(np.array(rotations)[:, 1])) > 0.08
    assert np.ptp(common.nominal_states[:, 2]) > 0.05
    assert np.max(np.linalg.norm(rotations, axis=1)) <= 0.3
    np.testing.assert_allclose(common.nominal_states[[0, -1], 25:], 0, atol=1e-12)


def test_rotating_reference_world_angular_derivatives(reference):
    _, common = reference
    packet = common.guidance
    # Independent SO(3) central differences on interpolated poses/velocities.
    t, h = 2.321, 1e-5
    a, b, c = packet.sample(t - h), packet.sample(t), packet.sample(t + h)
    numerical = pin.log3(c.base_pose[:3, :3] @ a.base_pose[:3, :3].T) / (2 * h)
    np.testing.assert_allclose(numerical, b.base_velocity[3:], atol=2e-8)
    np.testing.assert_allclose(
        (c.base_velocity - a.base_velocity) / (2 * h), b.base_acceleration, atol=2e-7
    )
    # Knot velocity agrees with configuration-derived rotation derivatives.
    rotations = np.stack([pin.log3(x.base_pose[:3, :3]) for x in packet.samples])
    dr = np.gradient(rotations, packet.times, axis=0, edge_order=2)
    k = 43
    np.testing.assert_allclose(
        _left_jacobian(rotations[k]) @ dr[k], packet.samples[k].base_velocity[3:], atol=1e-12
    )


def test_high_low_task_is_c2_at_stops():
    pose = np.eye(4)
    for knot in demo.KEY_TIMES[1:-1]:
        p, v, a = demo.high_low_task([knot - 1e-7, knot, knot + 1e-7], pose)
        np.testing.assert_allclose(v, 0, atol=1e-10)
        np.testing.assert_allclose(a, 0, atol=1e-6)
        np.testing.assert_allclose(p[0], p[2], atol=1e-12)
    t, h = 2.31, 1e-5
    p, v, a = demo.high_low_task([t - h, t, t + h], pose)
    np.testing.assert_allclose((p[2, :3, 3] - p[0, :3, 3]) / (2 * h), v[1, :3], atol=1e-9)
    np.testing.assert_allclose((v[2] - v[0]) / (2 * h), a[1], atol=1e-8)


def test_gapped_seed_is_solved_and_independently_validated():
    robot = load_go2_arm()
    planner = StandingWholeBodyPlanner(robot, WholeBodySettings(nodes=10))
    reference = smooth_tcp_reference(robot, planner.settings, displacement=(0.005, 0, 0))
    seed = reference.nominal_states.copy()
    seed[1:-1, 7] += 0.003  # Deliberate dynamics/contact gaps, not a rollout.
    torque = np.tile(planner._canonical_control(planner.static_control), (10, 1))
    plan = planner.solve(reference, initial_guess=(seed, torque))
    assert plan.converged and plan.diagnostics["accepted_for_task"]
    assert plan.diagnostics["max_dynamics_gap"] < 1e-6
    assert plan.diagnostics["initial_guess"] == "canonical_trajectory"
    np.testing.assert_allclose(plan.native_states[1:, 49:], plan.native_controls, atol=1e-10)
    seed[0, 0] += 0.1
    with pytest.raises(ValueError, match="episode state"):
        planner.solve(reference, initial_guess=(seed, torque))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"motion_weights": np.zeros(24)},
        {"base_rotation_bound": np.pi},
        {"base_translation_bound": (0.1, -0.1, 0.1)},
        {"allow_base_rotation": 1},
    ],
)
def test_invalid_mobility_settings_are_rejected(kwargs):
    with pytest.raises(ValueError):
        StandingArmReferenceGenerator(load_go2_arm(), **kwargs)
