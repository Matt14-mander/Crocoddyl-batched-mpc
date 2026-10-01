"""Lie group round trips, local Jacobians and optional independent robot oracle."""

import numpy as np
import pytest
import torch

from crocoddyl_batched_mpc.manifolds import (
    FloatingBaseManifold,
    SE3Manifold,
    SO3Manifold,
)

pytestmark = pytest.mark.ddp_acceptance


@pytest.mark.parametrize("manifold", [SO3Manifold(), SE3Manifold(), FloatingBaseManifold(2)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.cuda)])
def test_round_trip_identity_and_quaternion_sign(manifold, dtype, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    generator = torch.Generator(device=device).manual_seed(21)
    neutral = manifold.neutral(torch.zeros(4, manifold.nx, device=device, dtype=dtype))
    delta = torch.randn(4, manifold.ndx, device=device, dtype=dtype, generator=generator) * 0.3
    state = manifold.integrate(neutral, delta)
    tol = 3e-6 if dtype == torch.float32 else 1e-12
    torch.testing.assert_close(manifold.diff(state, neutral), delta, atol=tol, rtol=tol)
    torch.testing.assert_close(
        manifold.integrate(state, torch.zeros_like(delta)), state, atol=tol, rtol=tol
    )
    negated = state.clone()
    quaternion = slice(0, 4) if isinstance(manifold, SO3Manifold) else slice(3, 7)
    negated[:, quaternion] *= -1
    torch.testing.assert_close(
        manifold.diff(negated, state), torch.zeros_like(delta), atol=tol, rtol=tol
    )
    assert manifold.is_valid(state).all()
    assert manifold.nx == manifold.ndx + 1


@pytest.mark.parametrize("manifold", [SO3Manifold(), SE3Manifold(), FloatingBaseManifold(1)])
def test_difference_chart_jacobian_against_local_finite_differences(manifold):
    torch.manual_seed(17)
    neutral = manifold.neutral(torch.zeros(2, manifold.nx, dtype=torch.float64))
    base = manifold.integrate(neutral, torch.randn(2, manifold.ndx, dtype=torch.float64) * 0.3)
    delta = torch.randn(2, manifold.ndx, dtype=torch.float64) * 0.4
    target = manifold.integrate(base, delta)
    jacobian = manifold.diff_jacobian(target, base)
    columns = []
    for index in range(manifold.ndx):
        direction = torch.zeros_like(delta)
        direction[:, index] = 1e-6
        columns.append(
            (
                manifold.diff(manifold.integrate(target, direction), base)
                - manifold.diff(manifold.integrate(target, -direction), base)
            )
            / 2e-6
        )
    torch.testing.assert_close(jacobian, torch.stack(columns, -1), atol=5e-10, rtol=5e-9)
    torch.testing.assert_close(
        manifold.diff_jacobian(base, base),
        torch.eye(manifold.ndx, dtype=base.dtype).expand(2, -1, -1),
        atol=1e-12,
        rtol=1e-12,
    )


def test_so3_near_pi_and_finite_autograd_at_identity():
    manifold = SO3Manifold()
    identity = manifold.neutral(torch.zeros(2, 4, dtype=torch.float64))
    delta = torch.tensor([[torch.pi - 1e-7, 0, 0], [1e-10, -1e-10, 1e-10]], dtype=torch.float64)
    torch.testing.assert_close(
        manifold.diff(manifold.integrate(identity, delta), identity), delta, atol=1e-12, rtol=1e-12
    )
    zero = torch.zeros(2, 3, dtype=torch.float64, requires_grad=True)
    residual = manifold.diff(manifold.integrate(identity, zero), identity)
    residual.sum().backward()
    torch.testing.assert_close(zero.grad, torch.ones_like(zero), atol=1e-12, rtol=1e-12)
    assert not manifold.is_valid(torch.zeros_like(identity)).any()


@pytest.mark.parametrize("joint_count", [0, 2])
@pytest.mark.crocoddyl
def test_floating_base_matches_pinocchio_and_crocoddyl_state(joint_count):
    pin = pytest.importorskip("pinocchio")
    crocoddyl = pytest.importorskip("crocoddyl")
    model = pin.Model()
    parent = model.addJoint(0, pin.JointModelFreeFlyer(), pin.SE3.Identity(), "base")
    for index in range(joint_count):
        parent = model.addJoint(parent, pin.JointModelRY(), pin.SE3.Identity(), f"joint{index}")
    state_oracle = crocoddyl.StateMultibody(model)
    manifold = FloatingBaseManifold(joint_count)
    rng = np.random.default_rng(42)
    neutral = np.concatenate((pin.neutral(model), np.zeros(model.nv)))
    for scale in (0.0, 1e-8, 0.5, 1.0):
        first = rng.normal(size=manifold.ndx) * scale
        delta = rng.normal(size=manifold.ndx) * 0.3
        base = np.asarray(state_oracle.integrate(neutral, first))
        target = np.asarray(state_oracle.integrate(base, delta))
        tx = torch.from_numpy(base)[None]
        td = torch.from_numpy(delta)[None]
        actual = manifold.integrate(tx, td).numpy()[0]
        np.testing.assert_allclose(actual, target, atol=2e-12, rtol=2e-12)
        # Crocoddyl/Pinocchio diff takes (base,target), our interface (target,base).
        diff = manifold.diff(torch.from_numpy(target)[None], tx).numpy()[0]
        np.testing.assert_allclose(diff, state_oracle.diff(base, target), atol=2e-12)
        np.testing.assert_allclose(
            actual[: model.nq],
            pin.integrate(model, base[: model.nq], delta[: model.nv]),
            atol=2e-12,
        )
        jacobian = manifold.diff_jacobian(torch.from_numpy(target)[None], tx).numpy()[0]
        _, target_jacobian = state_oracle.Jdiff(base, target)
        np.testing.assert_allclose(jacobian, target_jacobian, atol=2e-11, rtol=2e-11)
