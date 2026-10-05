"""Official Go2 loading and independent fixed-contact KKT/Crocoddyl gates."""

import hashlib
import json
from importlib.resources import files

import numpy as np
import pytest
import torch

from crocoddyl_batched_mpc.manifolds import FloatingBaseManifold
from crocoddyl_batched_mpc.models.go2 import (
    GO2_JOINT_NAMES,
    Go2FixedContactReference,
    load_go2,
)


def test_pinned_go2_asset_integrity_and_license():
    assets = files("crocoddyl_batched_mpc").joinpath("assets/go2")
    metadata = json.loads(assets.joinpath("SOURCE.json").read_text())
    assert (
        hashlib.sha256(assets.joinpath("go2.urdf").read_bytes()).hexdigest() == metadata["sha256"]
    )
    assert "BSD 3-Clause" in assets.joinpath("LICENSE").read_text()
    assert len(metadata["commit"]) == 40


@pytest.fixture
def robot():
    pytest.importorskip("pinocchio")
    return load_go2()


@pytest.mark.crocoddyl
def test_go2_model_order_geometry_and_floating_state(robot):
    import pinocchio as pin

    assert (robot.nq, robot.nv, robot.nx, robot.ndx) == (19, 18, 37, 36)
    assert tuple(robot.model.names[2:]) == GO2_JOINT_NAMES
    assert (robot.torque_limits > 0).all()
    np.testing.assert_array_equal(robot.actuation_matrix()[:6], np.zeros((6, 12)))
    np.testing.assert_array_equal(robot.actuation_matrix()[6:], np.eye(12))
    q = robot.standing_configuration()
    assert np.all(q[7:] >= robot.model.lowerPositionLimit[7:])
    assert np.all(q[7:] <= robot.model.upperPositionLimit[7:])
    data = robot.model.createData()
    pin.framesForwardKinematics(robot.model, data, q)
    feet = np.stack([data.oMf[i].translation for i in robot.foot_ids])
    np.testing.assert_allclose(feet[:, 2], 0, atol=1e-12)
    assert feet[0, 0] > feet[2, 0]  # front/rear
    assert feet[0, 1] > feet[1, 1]  # left/right
    mass = pin.computeTotalMass(robot.model)
    assert abs(mass - 16.087) < 1e-9  # This pinned URDF's modeled mass.
    x = torch.from_numpy(np.r_[q, np.zeros(robot.nv)])[None]
    manifold = FloatingBaseManifold(12)
    delta = torch.linspace(-0.01, 0.01, 36, dtype=x.dtype)[None]
    result = manifold.integrate(x, delta).numpy()[0]
    np.testing.assert_allclose(
        result[:19], pin.integrate(robot.model, q, delta.numpy()[0, :18]), atol=1e-12
    )


@pytest.mark.crocoddyl
def test_go2_quasi_static_bilateral_equilibrium_and_weight_support(robot):
    import pinocchio as pin

    reference = Go2FixedContactReference(robot)
    q = reference.reference_configuration
    torque = reference.quasi_static_torques()
    result = reference.calc(q, np.zeros(robot.nv), torque)
    assert np.max(np.abs(result.acceleration)) < 1e-9
    assert np.max(np.abs(result.dynamics_residual)) < 1e-10
    assert np.max(np.abs(result.contact_residual)) < 1e-10
    assert (result.forces_world[:, 2] > 0).all()
    np.testing.assert_allclose(
        result.forces_world.sum(0), [0, 0, pin.computeTotalMass(robot.model) * 9.81], atol=1e-10
    )
    assert (np.abs(torque) < robot.torque_limits).all()
    # Zero joint torque does not hold the articulated robot stationary.
    assert np.linalg.norm(reference.calc(q, np.zeros(robot.nv), np.zeros(12)).acceleration) > 1


@pytest.mark.crocoddyl
@pytest.mark.parametrize("feet", [("FL", "FR", "RL", "RR"), ("RR", "FL")])
@pytest.mark.parametrize("gains", [(0.0, 0.0), (10.0, 3.0)])
def test_go2_contact_acceleration_forces_and_derivatives_match_crocoddyl(robot, feet, gains):
    pytest.importorskip("crocoddyl")
    import pinocchio as pin

    reference = Go2FixedContactReference(robot, feet=feet, stabilization=gains)
    rng = np.random.default_rng(3)
    q = pin.integrate(robot.model, reference.reference_configuration, rng.normal(size=18) * 0.015)
    v = rng.normal(size=18) * 0.05
    torque = rng.normal(size=12) * 0.3
    result = reference.calc(q, v, torque)
    model = reference.crocoddyl_model()
    data = model.createData()
    model.calc(data, np.r_[q, v], torque)
    model.calcDiff(data, np.r_[q, v], torque)
    np.testing.assert_allclose(result.acceleration, data.xout, atol=1e-9, rtol=1e-9)
    contacts = [data.multibody.contacts.contacts[name] for name in reference.feet]
    # ContactModel3D's f/df use its requested LOCAL_WORLD_ALIGNED force coordinates.
    forces = np.stack([d.f.linear for d in contacts])
    np.testing.assert_allclose(result.forces_world, forces, atol=1e-9, rtol=1e-9)
    assert np.max(np.abs(result.dynamics_residual)) < 1e-10
    assert np.max(np.abs(result.contact_residual)) < 1e-10
    da_dx, da_du, df_dx, df_du = reference.finite_difference_derivatives(q, v, torque)
    np.testing.assert_allclose(da_dx, data.Fx, atol=2e-5, rtol=2e-5)
    np.testing.assert_allclose(da_du, data.Fu, atol=2e-5, rtol=2e-5)
    np.testing.assert_allclose(df_dx, np.vstack([d.df_dx for d in contacts]), atol=2e-5, rtol=2e-5)
    np.testing.assert_allclose(df_du, np.vstack([d.df_du for d in contacts]), atol=2e-5, rtol=2e-5)
    # A later evaluation must not overwrite a previous result's arrays.
    acceleration = result.acceleration.copy()
    reference.calc(q, v, torque + 0.1)
    np.testing.assert_array_equal(result.acceleration, acceleration)


@pytest.mark.crocoddyl
def test_go2_reference_rejects_incompatible_native_versions(robot, monkeypatch):
    crocoddyl = pytest.importorskip("crocoddyl")
    import pinocchio as pin

    reference = Go2FixedContactReference(robot)
    monkeypatch.setattr(crocoddyl, "__version__", "3.2.1")
    monkeypatch.setattr(pin, "__version__", "4.1.0")
    with pytest.raises(ImportError, match=r"install crocoddyl==3\.2\.1 pin==4\.0\.0"):
        reference.crocoddyl_model()


@pytest.mark.crocoddyl
def test_go2_reference_rejects_invalid_inputs(robot):
    q, v, u = robot.standing_configuration(), np.zeros(18), np.zeros(12)
    reference = Go2FixedContactReference(robot)
    with pytest.raises(ValueError, match="unit xyzw"):
        bad = q.copy()
        bad[3:7] = 0
        reference.calc(bad, v, u)
    with pytest.raises(ValueError, match="finite vector"):
        reference.calc(q, v, np.full(12, np.nan))
    with pytest.raises(ValueError, match="epsilon"):
        reference.finite_difference_derivatives(q, v, u, epsilon=0)
    with pytest.raises(ValueError, match="duplicates"):
        Go2FixedContactReference(robot, feet=("FL", "FL"))
