"""Torch Go2 physics, local derivatives, batching and independent native gates."""

import numpy as np
import pytest
import torch

from crocoddyl_batched_mpc.models import Go2FixedContactReference, Go2TorchDynamics


def _sample(model, batch=3):
    generator = torch.Generator(device=model.device).manual_seed(42)
    delta = 0.02 * torch.randn(
        batch, 36, generator=generator, device=model.device, dtype=model.dtype
    )
    x = model.manifold.integrate(model.standing_state.expand(batch, -1), delta)
    u = 0.3 * torch.randn(batch, 12, generator=generator, device=model.device, dtype=model.dtype)
    return x, u


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_torch_go2_equilibrium_and_batch_independence(dtype):
    m = Go2TorchDynamics(dtype=dtype)
    x = m.standing_state.expand(3, -1).clone()
    u = m.quasi_static_torques().expand(3, -1)
    result = m.contact_dynamics(x, u)
    tol = 3e-4 if dtype == torch.float32 else 1e-10
    assert result.acceleration.abs().max() < tol
    assert result.dynamics_residual.abs().max() < tol
    assert result.contact_residual.abs().max() < tol
    assert (result.forces_world[..., 2] > 0).all()
    torch.testing.assert_close(
        result.forces_world.sum(-2),
        x.new_tensor([0.0, 0.0, 16.087 * 9.81]).expand(3, -1),
        atol=tol,
        rtol=tol,
    )
    assert (u.abs() < m.torque_limits).all()
    torch.testing.assert_close(m.calc(x, u), x, atol=tol, rtol=tol)
    x, u = _sample(m)
    batched = m.contact_dynamics(x, u)
    for b in range(3):
        separate = m.contact_dynamics(x[b : b + 1], u[b : b + 1])
        torch.testing.assert_close(
            batched.acceleration[b], separate.acceleration[0], atol=tol, rtol=tol
        )
        torch.testing.assert_close(
            batched.forces_world[b], separate.forces_world[0], atol=tol, rtol=tol
        )
    torch.testing.assert_close(
        torch.linalg.vector_norm(m.calc(x, u)[:, 3:7], dim=-1), x.new_ones(3)
    )


@pytest.mark.crocoddyl
@pytest.mark.parametrize("feet", [("FL", "FR", "RL", "RR"), ("RR", "FL")])
@pytest.mark.parametrize("gains", [(0.0, 0.0), (10.0, 3.0)])
def test_torch_go2_quantities_forces_and_derivatives_against_native(feet, gains):
    pytest.importorskip("pinocchio")
    pytest.importorskip("crocoddyl")
    m = Go2TorchDynamics(feet=feet, stabilization=gains)
    reference = Go2FixedContactReference(feet=feet, stabilization=gains)
    x, u = _sample(m, 2)
    result = m.contact_dynamics(x, u)
    da_dx, da_du, df_dx, df_du = m.contact_derivatives(x, u)
    action = reference.crocoddyl_model()
    for b in range(2):
        q, v, torque = x[b, :19].numpy(), x[b, 19:].numpy(), u[b].numpy()
        oracle = reference.calc(q, v, torque)
        for field in (
            "mass_matrix",
            "nonlinear_effects",
            "contact_jacobian",
            "contact_drift",
            "acceleration",
            "forces_world",
        ):
            np.testing.assert_allclose(
                getattr(result, field)[b].numpy(), getattr(oracle, field), atol=1e-9, rtol=1e-9
            )
        data = action.createData()
        action.calc(data, x[b].numpy(), torque)
        action.calcDiff(data, x[b].numpy(), torque)
        contacts = [data.multibody.contacts.contacts[name] for name in feet]
        for actual, expected in (
            (da_dx[b], data.Fx),
            (da_du[b], data.Fu),
            (df_dx[b], np.vstack([d.df_dx for d in contacts])),
            (df_du[b], np.vstack([d.df_du for d in contacts])),
        ):
            np.testing.assert_allclose(actual.numpy(), expected, atol=2e-8, rtol=2e-8)


@pytest.mark.crocoddyl
def test_go2_discrete_local_derivatives_against_independent_pinocchio_plant():
    pin = pytest.importorskip("pinocchio")
    m = Go2TorchDynamics(stabilization=(10.0, 3.0))
    reference = Go2FixedContactReference(stabilization=(10.0, 3.0))
    x, u = _sample(m, 1)
    Fx, Fu = m.calc_diff(x, u)

    def plant(state, torque):
        acceleration = reference.calc(state[:19], state[19:], torque).acceleration
        velocity = state[19:] + m.dt * acceleration
        return np.r_[pin.integrate(reference.robot.model, state[:19], m.dt * velocity), velocity]

    state, control = x[0].numpy(), u[0].numpy()
    predicted = plant(state, control)
    np.testing.assert_allclose(m.calc(x, u)[0].numpy(), predicted, atol=1e-12, rtol=1e-12)
    eps = 1e-6
    columns = []
    for i in range(48):
        d = np.zeros(48)
        d[i] = eps
        plus = np.r_[
            pin.integrate(reference.robot.model, state[:19], d[:18]), state[19:] + d[18:36]
        ]
        minus = np.r_[
            pin.integrate(reference.robot.model, state[:19], -d[:18]), state[19:] - d[18:36]
        ]
        yp, ym = plant(plus, control + d[36:]), plant(minus, control - d[36:])
        dp = np.r_[
            pin.difference(reference.robot.model, predicted[:19], yp[:19]), yp[19:] - predicted[19:]
        ]
        dm = np.r_[
            pin.difference(reference.robot.model, predicted[:19], ym[:19]), ym[19:] - predicted[19:]
        ]
        columns.append((dp - dm) / (2 * eps))
    jacobian = np.stack(columns, -1)
    np.testing.assert_allclose(Fx[0].numpy(), jacobian[:, :36], atol=2e-7, rtol=2e-7)
    np.testing.assert_allclose(Fu[0].numpy(), jacobian[:, 36:], atol=2e-7, rtol=2e-7)


def test_go2_independent_world_anchors_and_input_contract():
    original = Go2TorchDynamics()
    q = original.reference_configuration.expand(2, -1).clone()
    q[1, 0] += 1.2
    m = Go2TorchDynamics(reference_configuration=q)
    x = m.standing_state
    u = m.quasi_static_torques()
    assert m.parameter_batch_size == 2
    assert m.contact_dynamics(x, u).acceleration.abs().max() < 1e-10
    # Translation changes world anchors but not robot-local dynamics.
    torch.testing.assert_close(
        m.contact_positions[1] - m.contact_positions[0], q.new_tensor([1.2, 0.0, 0.0]).expand(4, -1)
    )
    with pytest.raises(ValueError, match="matching"):
        m.calc(x, u[:, :1])
    with pytest.raises(ValueError, match="dtype"):
        m.calc(x.float(), u.float())
    with pytest.raises(ValueError, match="unique"):
        Go2TorchDynamics(feet=("FL", "FL"))
    with pytest.raises(ValueError, match="positive"):
        Go2TorchDynamics(dt=0)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_go2_cuda_nondefault_stream_and_local_derivatives(dtype):
    cpu = Go2TorchDynamics(dtype=dtype)
    gpu = Go2TorchDynamics(device="cuda", dtype=dtype)
    x, u = _sample(cpu, 2)
    expected = cpu.contact_dynamics(x, u)
    A, B = cpu.calc_diff(x, u)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        actual = gpu.contact_dynamics(x.cuda(), u.cuda())
        ga, gb = gpu.calc_diff(x.cuda(), u.cuda())
    stream.synchronize()
    tol = 3e-3 if dtype == torch.float32 else 2e-8
    torch.testing.assert_close(actual.acceleration.cpu(), expected.acceleration, atol=tol, rtol=tol)
    torch.testing.assert_close(actual.forces_world.cpu(), expected.forces_world, atol=tol, rtol=tol)
    torch.testing.assert_close(ga.cpu(), A, atol=tol, rtol=tol)
    torch.testing.assert_close(gb.cpu(), B, atol=tol, rtol=tol)
