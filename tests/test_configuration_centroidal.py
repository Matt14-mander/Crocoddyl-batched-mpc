"""Physical C1 maps, static support/payload bounds and bounded redundancy search."""

import numpy as np
import pytest

from crocoddyl_batched_mpc.models.configuration_centroidal import (
    CapabilitySettings,
    QuasiStaticCapability,
    momentum_map,
)
from crocoddyl_batched_mpc.models.go2_arm import load_go2_arm
from crocoddyl_batched_mpc.references.capability import (
    prepare_capable_reference,
    redistribute_configuration,
)
from crocoddyl_batched_mpc.references.configuration_centroidal import (
    CapabilityError,
    configuration_reference,
)
from crocoddyl_batched_mpc.references.standing_arm import StandingArmReferenceGenerator, reach_task

pin = pytest.importorskip("pinocchio")


@pytest.fixture(scope="module")
def robot():
    return load_go2_arm()


@pytest.mark.parametrize("arm", [[0, 1.1, -1.1, 0, 0, 0], [0.2, 0.9, -1.0, 0.2, 0.1, 0.2]])
def test_momentum_map_native_order_world_frame_and_inverse(robot, arm):
    q = robot.standing_configuration()
    q[-6:] = arm
    q[3:7] = pin.Quaternion(pin.exp3(np.array([0.1, -0.15, 0.3]))).coeffs()
    v = np.linspace(-0.03, 0.04, 24)
    mapping = momentum_map(robot, q)
    data = robot.model.createData()
    h = pin.computeCentroidalMomentum(robot.model, data, robot.to_native_q(q), robot.to_native_v(v))
    np.testing.assert_allclose(mapping.matrix @ v, h.vector, atol=1e-13)
    np.testing.assert_allclose(mapping.base_velocity(h.vector, v[6:]), v[:6], atol=1e-13)
    np.testing.assert_allclose(mapping.com, robot.evaluate(q).com, atol=1e-14)
    np.testing.assert_allclose(
        mapping.matrix[:3], robot.total_mass * robot.evaluate(q).com_jacobian, atol=1e-13
    )
    epsilon = 1e-6
    derivative = (
        robot.evaluate(robot.integrate(q, epsilon * v)).com
        - robot.evaluate(robot.integrate(q, -epsilon * v)).com
    ) / (2 * epsilon)
    np.testing.assert_allclose(mapping.matrix[:3] @ v / robot.total_mass, derivative, atol=1e-9)
    assert not mapping.matrix.flags.writeable


def test_map_changes_with_arm_configuration(robot):
    q = robot.standing_configuration()
    other = q.copy()
    other[-5] += 0.3
    assert np.max(abs(momentum_map(robot, q).matrix - momentum_map(robot, other).matrix)) > 0.01


@pytest.mark.parametrize("payload", [0.0, 1.0, 5.0, 10.0])
def test_static_full_balance_and_payload_is_added_once(robot, payload):
    q = robot.standing_configuration()
    checker = QuasiStaticCapability(robot, robot.evaluate(q).foot_positions)
    result = checker.evaluate(q, payload_mass_kg=payload)
    value, jac, gravity = checker.quantities(q, payload_mass_kg=payload)
    np.testing.assert_allclose(
        gravity,
        robot.actuation_matrix() @ result.torques + jac.T @ result.forces.ravel(),
        atol=1e-9,
    )
    np.testing.assert_allclose(
        result.forces.sum(axis=0),
        -(robot.total_mass + payload) * robot.model.gravity.linear,
        atol=1e-9,
    )
    assert result.diagnostics["qp_converged"]
    assert result.accepted == (payload < 10)
    if not result.accepted:
        assert result.status == "QUASI_STATIC_LIMIT" and result.diagnostics["torque_violation"] > 0
    assert robot.config["payload_mass_kg"] == 0
    np.testing.assert_allclose(value.gravity, robot.evaluate(q).gravity)


def test_payload_offset_frame_and_support_limits(robot):
    q = robot.standing_configuration()
    checker = QuasiStaticCapability(robot, robot.evaluate(q).foot_positions)
    value, _, g = checker.quantities(q, payload_mass_kg=1.0, payload_offset_tcp=[0.1, 0.02, 0])
    force = robot.model.gravity.linear
    wrench = np.r_[force, np.cross(value.tcp_pose[:3, :3] @ [0.1, 0.02, 0], force)]
    np.testing.assert_allclose(g, value.gravity - value.tcp_jacobian.T @ wrench)
    limited = QuasiStaticCapability(robot, checker.anchors, CapabilitySettings(normal_max=10))
    assert not limited.evaluate(q).accepted


@pytest.mark.parametrize("payload", [-1, float("nan"), float("inf")])
def test_invalid_payload_rejected(robot, payload):
    q = robot.standing_configuration()
    checker = QuasiStaticCapability(robot, robot.evaluate(q).foot_positions)
    with pytest.raises(ValueError):
        checker.evaluate(q, payload_mass_kg=payload)


def test_derating_and_contacts_cannot_silently_relax_limits(robot):
    q = robot.standing_configuration()
    anchors = robot.evaluate(q).foot_positions
    with pytest.raises(ValueError):
        QuasiStaticCapability(robot, anchors, torque_limits=robot.torque_limits * 1.01)
    checker = QuasiStaticCapability(robot, anchors, torque_limits=robot.torque_limits * 0.1)
    assert not checker.evaluate(q).accepted
    shifted = q.copy()
    shifted[0] += 0.01
    assert not QuasiStaticCapability(robot, anchors).evaluate(shifted).accepted


def test_redistribution_holds_task_and_reduces_overload(robot):
    q = robot.standing_configuration()
    value = robot.evaluate(q)
    checker = QuasiStaticCapability(robot, value.foot_positions)
    assert not checker.evaluate(q, payload_mass_kg=6).accepted
    adjusted, diag = redistribute_configuration(checker, q, q, value.tcp_pose, payload_mass_kg=6)
    assert adjusted is not None, diag
    assert diag["after_peak_utilization"] < 1
    assert diag["after_peak_utilization"] < diag["before_peak_utilization"]
    actual = robot.evaluate(adjusted)
    np.testing.assert_allclose(actual.foot_positions, value.foot_positions, atol=1e-7)
    np.testing.assert_allclose(actual.tcp_pose, value.tcp_pose, atol=1e-7)
    assert np.max(abs(adjusted[:3] - q[:3]) - [0.08, 0.06, 0.075]) < 1e-8
    assert checker.evaluate(adjusted, payload_mass_kg=6).accepted


def test_redistribution_budget_failure_is_explicit(robot):
    q = robot.standing_configuration()
    value = robot.evaluate(q)
    checker = QuasiStaticCapability(robot, value.foot_positions)
    adjusted, diag = redistribute_configuration(
        checker, q, q, value.tcp_pose, payload_mass_kg=100, iterations=1
    )
    assert adjusted is None
    assert diag["status"] == "NO_CAPABLE_POSTURE_WITHIN_SEARCH_BUDGET"


def test_common_preparation_preserves_capable_prior_and_rejects_bad_initial(robot):
    q = robot.standing_configuration()
    times = np.arange(11) * 0.1
    common = StandingArmReferenceGenerator(robot).generate(
        q, times, *reach_task(times, robot.evaluate(q).tcp_pose, [0.01, 0, 0])
    )
    checker = QuasiStaticCapability(robot, common.guidance.samples[0].foot_anchors)
    prepared, diag = prepare_capable_reference(robot, common, checker)
    assert prepared is common and diag["adjusted_nodes"] == []
    reference = configuration_reference(robot, common, checker)
    for k, state in enumerate(common.nominal_states):
        np.testing.assert_allclose(reference.states[k, 3:], reference.maps[k] @ state[25:])
    assert np.max(abs(reference.states[:, 6:])) > 0.0001
    derated = QuasiStaticCapability(robot, checker.anchors, torque_limits=robot.torque_limits * 0.1)
    with pytest.raises(CapabilityError, match="INITIAL_CONFIGURATION_UNSUPPORTED"):
        prepare_capable_reference(robot, common, derated)
    with pytest.raises(CapabilityError) as exc:
        configuration_reference(robot, common, derated)
    assert 0 in exc.value.diagnostics["rejected_nodes"]


def test_shared_redistribution_respects_fixed_rotation_and_keeps_initial_state(robot):
    q = robot.standing_configuration()
    times = np.arange(11) * 0.2
    common = StandingArmReferenceGenerator(robot).generate(
        q, times, *reach_task(times, robot.evaluate(q).tcp_pose, [0.01, 0, 0])
    )
    # Artificial test-only derating puts later targets just over capacity;
    # neither URDF limits nor any executor/planner baseline is changed.
    checker = QuasiStaticCapability(
        robot, common.guidance.samples[0].foot_anchors, torque_limits=robot.torque_limits * 0.211
    )
    with pytest.raises(CapabilityError, match="REDISTRIBUTION_DISABLED"):
        prepare_capable_reference(robot, common, checker, allow_redistribution=False)
    prepared, diag = prepare_capable_reference(robot, common, checker)
    assert diag["status"] == "REDISTRIBUTED_SHARED_REFERENCE" and diag["adjusted_nodes"]
    np.testing.assert_array_equal(prepared.nominal_states[0], common.nominal_states[0])
    np.testing.assert_array_equal(prepared.nominal_states[:, 3:7], common.nominal_states[:, 3:7])
    for x, sample, original in zip(
        prepared.nominal_states, prepared.guidance.samples, common.guidance.samples
    ):
        assert checker.evaluate(x[:25]).accepted
        np.testing.assert_array_equal(sample.tcp_pose, original.tcp_pose)
        np.testing.assert_array_equal(sample.tcp_velocity, original.tcp_velocity)
        np.testing.assert_allclose(robot.evaluate(x[:25]).tcp_pose, original.tcp_pose, atol=1e-7)
    assert np.max(abs(prepared.nominal_states[:, :3] - q[:3]) - [0.06, 0.06, 0.04]) < 1e-8
