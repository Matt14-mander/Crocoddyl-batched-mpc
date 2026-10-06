"""Hybrid physics, independent impact oracle and atomic mode changes."""

import numpy as np
import pytest
import torch

from crocoddyl_batched_mpc.models import (
    Go2ContactSwitchingMPC,
    Go2FixedContactReference,
    Go2HybridDynamics,
)


@pytest.mark.crocoddyl
def test_masked_kkt_matches_independent_support_subsets_and_flight():
    pin = pytest.importorskip("pinocchio")
    masks = torch.tensor(
        [[True] * 4, [False, True, True, True], [True, False, False, True], [False] * 4]
    )
    model = Go2HybridDynamics(contact_mask=masks, stabilization=(10.0, 3.0))
    delta = (
        torch.randn(4, 36, generator=torch.Generator().manual_seed(15), dtype=model.dtype) * 0.01
    )
    state = model.manifold.integrate(model.standing_state.expand(4, -1), delta)
    torque = torch.randn(4, 12, generator=torch.Generator().manual_seed(10), dtype=model.dtype)
    result = model.contact_dynamics(state, torque)
    assert result.forces_world[~masks].eq(0).all()
    assert result.dynamics_residual.abs().max() < 1e-10
    assert result.contact_residual.abs().max() < 1e-10
    for b in range(4):
        feet = tuple(f for f, active in zip(model.feet, masks[b]) if active)
        q, v, u = state[b, :19].numpy(), state[b, 19:].numpy(), torque[b].numpy()
        if feet:
            native = Go2FixedContactReference(
                feet=feet,
                stabilization=model.gains,
                reference_configuration=model.reference_configuration.numpy(),
            )
            expected = native.calc(q, v, u)
            np.testing.assert_allclose(
                result.forces_world[b, masks[b]].numpy(),
                expected.forces_world,
                atol=1e-9,
                rtol=1e-9,
            )
            acceleration = expected.acceleration
        else:
            native = Go2FixedContactReference()
            data = native.robot.model.createData()
            pin.computeAllTerms(native.robot.model, data, q, v)
            mass = np.triu(data.M) + np.triu(data.M, 1).T
            acceleration = np.linalg.solve(mass, native.actuation @ u - data.nle)
        np.testing.assert_allclose(
            result.acceleration[b].numpy(), acceleration, atol=1e-9, rtol=1e-9
        )


@pytest.mark.crocoddyl
def test_plastic_impact_matches_independent_mass_projection():
    pytest.importorskip("pinocchio")
    model = Go2HybridDynamics()
    state = model.standing_state[None].clone()
    state[:, 21] = -0.1
    event = model.transition(state, torch.zeros(4, dtype=torch.bool))
    assert event.accepted.all()
    native = Go2FixedContactReference(reference_configuration=state[0, :19].numpy())
    mass, _, J, _ = native._quantities(state[0, :19].numpy(), state[0, 19:].numpy())
    response = np.linalg.solve(mass, J.T)
    impulse = -np.linalg.solve(J @ response, J @ state[0, 19:].numpy())
    velocity = state[0, 19:].numpy() + response @ impulse
    np.testing.assert_allclose(event.state[0, 19:].numpy(), velocity, atol=1e-11, rtol=1e-11)
    np.testing.assert_allclose(
        event.impulses_world[0].numpy().flatten(), impulse, atol=1e-11, rtol=1e-11
    )
    torch.testing.assert_close(event.state[:, :19], state[:, :19], atol=0, rtol=0)
    assert (event.kinetic_energy_after < event.kinetic_energy_before).all()
    assert (event.impulses_world[..., 2] >= 0).all()


def test_touchdown_guards_and_environment_isolation():
    model = Go2HybridDynamics()
    state = model.standing_state.expand(6, -1).clone()
    state[:, 21] = -0.1
    state[1, 2] += 0.01  # above ground
    state[2, 21] = 0.1  # separating
    state[3, 19] = 1.0  # excessive sliding impulse
    state[4, 0] += 0.01  # anchor mismatch
    state[5, 3:7] = 0  # corrupt observation
    before = state.clone()
    event = model.transition(state, torch.zeros(4, dtype=torch.bool))
    assert event.accepted.tolist() == [True, False, False, False, False, False]
    torch.testing.assert_close(state, before, atol=0, rtol=0)
    torch.testing.assert_close(event.state[1:], before[1:], atol=0, rtol=0)
    assert event.impulses_world[1:].eq(0).all()
    # Conservative sticking impact is rejected rather than clipping its impulse.
    assert not model.transition(
        state[:1], torch.zeros(4, dtype=torch.bool), friction=0.1
    ).accepted.any()

    # Landing only the FL leg would require tensile impulses at FR/RL.
    single_leg = model.standing_state[None].clone()
    single_leg[:, 27] = 0.1
    mass, _, J, _ = model.quantities(single_leg[:, :19], single_leg[:, 19:])
    response = torch.linalg.solve(mass, J.transpose(-1, -2))
    impulse = -torch.linalg.solve(J @ response, J @ single_leg[:, 19:, None]).reshape(1, 4, 3)
    assert impulse[..., 2].min() < -1e-7
    assert not model.transition(
        single_leg, torch.tensor([False, True, True, True]), friction=2.0
    ).accepted.any()


def test_parameter_batch_and_fast_foot_geometry():
    model = Go2HybridDynamics(contact_mask=torch.ones(2, 4, dtype=torch.bool))
    assert model.parameter_batch_size == 2
    delta = torch.randn(2, 36, generator=torch.Generator().manual_seed(8), dtype=model.dtype) * 0.1
    state = model.manifold.integrate(model.standing_state.expand(2, -1), delta)
    torch.testing.assert_close(
        model.foot_positions(state[:, :19]),
        model._kinematics(state[:, :19], state[:, 19:])[3],
        atol=1e-12,
        rtol=1e-12,
    )
    with pytest.raises(ValueError, match="batch"):
        model.contact_dynamics(state[:1], state.new_zeros(1, 12))
    with pytest.raises(ValueError, match="batch"):
        Go2HybridDynamics(
            contact_mask=torch.ones(2, 4, dtype=torch.bool), contact_positions=torch.zeros(3, 4, 3)
        )


def test_retained_anchors_cannot_be_rebased_by_a_mode_request():
    controller = Go2ContactSwitchingMPC(compile_physics=False)
    state = controller.controllers[0].reference[None].clone()
    shifted = state[0, :19].clone()
    shifted[0] += 0.001
    controller.prepare_mode("shifted", [True] * 4, reference_configuration=shifted)
    event = controller.switch("shifted", state)
    assert not event.accepted.any() and controller.mode_indices.eq(0).all()
    torch.testing.assert_close(event.state, state, atol=0, rtol=0)


@pytest.mark.crocoddyl
def test_switching_swing_and_touchdown_closed_loop_with_independent_plant():
    pytest.importorskip("pinocchio")
    from examples.go2_contact_switching import run_switching

    report = run_switching(plant="pinocchio", compile_physics=False)
    assert report["all_actions_usable"]
    assert len(report["events"]) == 6
    assert all(event["accepted"] == [True, False] for event in report["events"])
    assert report["max_swing_height_m"] > 0.005
    assert report["max_inactive_force_N"] == 0
    assert report["max_torque_violation_Nm"] <= 1e-7
    assert report["max_force_violation_N"] <= 1e-7
    assert max(report["final_state_error"]) < 0.003
    landings = [event for event in report["events"] if event["mode"] == "stance"]
    assert all(event["energy_after_J"] < event["energy_before_J"] for event in landings)


def test_release_keeps_velocity_and_has_zero_inactive_force_and_derivatives():
    model = Go2HybridDynamics(contact_mask=[False, True, True, True])
    state = model.standing_state[None].clone()
    event = model.transition(state, torch.ones(4, dtype=torch.bool))
    assert event.accepted.all() and event.impulses_world.eq(0).all()
    torch.testing.assert_close(event.state, state, atol=0, rtol=0)
    u = model.quasi_static_torques()[None]
    da, du, df, fu = model.contact_derivatives(state, u)
    assert df[:, :3].eq(0).all() and fu[:, :3].eq(0).all()
    # Independent finite difference of acceleration and full fixed-slot force.
    d = state.new_zeros(1, 36)
    d[:, 8] = 1e-6
    plus = model.contact_dynamics(model.manifold.integrate(state, d), u)
    minus = model.contact_dynamics(model.manifold.integrate(state, -d), u)
    torch.testing.assert_close(
        da[..., 8], (plus.acceleration - minus.acceleration) / 2e-6, atol=2e-6, rtol=2e-6
    )
    torch.testing.assert_close(
        df[..., 8], (plus.forces_world - minus.forces_world).flatten(1) / 2e-6, atol=2e-6, rtol=2e-6
    )


def test_switch_is_per_environment_and_rejection_preserves_mode_and_caches():
    controller = Go2ContactSwitchingMPC(
        batch_size=2, minimum_normal_force=0.0, force_margin=1e-4, compile_physics=False
    )
    controller.prepare_mode("release_fl", [False, True, True, True])
    state = controller.controllers[0].reference.expand(2, -1).clone()
    controller.compute(state)
    old_cache = controller.controllers[0]._warm[1].clone()
    event = controller.switch("release_fl", state, mask=torch.tensor([True, False]))
    assert event.accepted.tolist() == [True, False]
    assert controller.mode_indices.tolist() == [1, 0]
    assert not controller.controllers[1]._valid[0]  # mode change clears old QP state
    torch.testing.assert_close(controller.controllers[0]._warm[1], old_cache, atol=0, rtol=0)
    action, result = controller.compute(event.state)
    assert result.usable.all()
    released = controller.controllers[1].dynamics.contact_dynamics(event.state, action)
    assert released.forces_world[0, 0].eq(0).all()
    old_modes = controller.mode_indices.clone()
    caches = [(c._warm.clone(), c._dual.clone(), c._valid.clone()) for c in controller.controllers]
    bad = state.clone()
    bad[0, 2] += 0.02
    rejected = controller.switch("stance", bad, mask=torch.tensor([True, False]))
    assert not rejected.accepted.any()
    torch.testing.assert_close(controller.mode_indices, old_modes)
    torch.testing.assert_close(rejected.state, bad)
    for c, saved in zip(controller.controllers, caches):
        for current, original in zip((c._warm, c._dual, c._valid), saved):
            torch.testing.assert_close(current, original, atol=0, rtol=0)
    landing = state.clone()
    landing[0, 21] = -0.03
    accepted = controller.switch("stance", landing, mask=torch.tensor([True, False]))
    assert accepted.accepted.tolist() == [True, False]
    assert controller.mode_indices.eq(0).all()
    assert accepted.impulses_world[0, :, 2].min() > 0
    assert controller.compute(accepted.state)[1].usable.all()
    with pytest.raises(ValueError, match="prepare"):
        controller.switch("missing", state)
    with pytest.raises(ValueError, match="unique"):
        controller.prepare_mode("stance", [True] * 4)
    with pytest.raises(ValueError, match="friction"):
        controller.switch("stance", state, friction=1.0)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_hybrid_touchdown_cuda_nondefault_stream():
    model = Go2HybridDynamics(device="cuda")
    state = model.standing_state[None].clone()
    state[:, 21] = -0.1
    torque = model.quasi_static_torques()[None]
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        event = model.transition(state, torch.zeros(4, device="cuda", dtype=torch.bool))
        result = model.contact_dynamics(event.state, torque)
    stream.synchronize()
    assert event.accepted.all() and result.contact_residual.abs().max() < 1e-9
