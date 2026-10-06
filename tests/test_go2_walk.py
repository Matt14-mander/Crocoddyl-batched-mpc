"""Slow crawl planning, live dynamics, gated execution and native closed loop."""

import numpy as np
import pytest
import torch

from crocoddyl_batched_mpc import SolveStatus
from crocoddyl_batched_mpc.models import Go2FixedContactReference, Go2HybridDynamics
from crocoddyl_batched_mpc.models.go2_walk import (
    Go2SlowWalkController,
    Go2WalkMPC,
    go2_center_of_mass,
    go2_leg_ik,
    go2_slow_walk_plan,
    go2_triangle_margin,
)


@pytest.fixture(scope="module")
def plan():
    return go2_slow_walk_plan(cycles=1)


def test_crawl_plan_order_fixed_stance_anchors_ik_and_support_margin(plan):
    model = Go2HybridDynamics()
    mask, anchors = plan.contact_masks, plan.anchors
    assert ((mask.sum(-1) == 3) | (mask.sum(-1) == 4)).all()
    releases = mask[:-1] & ~mask[1:]
    order = [model.feet[int(row.nonzero()[0])] for row in releases if row.any()]
    assert tuple(order) == ("RL", "FL", "RR", "FR")
    retained = mask[:-1] & mask[1:]
    assert (anchors[1:] - anchors[:-1])[retained].eq(0).all()
    torch.testing.assert_close(
        anchors[-1] - anchors[0], anchors.new_tensor([0.02, 0.0, 0.0]).expand(4, -1)
    )
    actual = model.foot_positions(plan.reference[:, :19])
    torch.testing.assert_close(actual, plan.foot_reference, atol=1e-9, rtol=1e-9)
    torch.testing.assert_close(actual[mask], anchors[mask], atol=1e-9, rtol=1e-9)
    assert (plan.reference[:, 7:19] >= model.lower_limits).all()
    assert (plan.reference[:, 7:19] <= model.upper_limits).all()
    for leg in range(4):
        assert abs(float(actual[~mask[:, leg], leg, 2].max()) - 0.015) < 1e-8
    three = mask.sum(-1) == 3
    assert (
        go2_triangle_margin(model, plan.reference[three], mask[three], anchors[three]).min() > 0.03
    )
    assert plan.reference[0, 19:].eq(0).all() and plan.reference[-1, 19:].eq(0).all()
    # Discrete endpoints approximate zero velocity; no discontinuous foot jump.
    assert (actual[1:] - actual[:-1]).norm(dim=-1).max() < 0.003
    torch.testing.assert_close(
        plan.reference[-1, :3] - plan.reference[0, :3], plan.reference.new_tensor([0.02, 0.0, 0.0])
    )


def test_plan_rejects_unsafe_parameters_and_unreachable_ik():
    with pytest.raises(ValueError, match="cycles"):
        go2_slow_walk_plan(cycles=0)
    with pytest.raises(ValueError, match="order"):
        go2_slow_walk_plan(order=("FL", "FR", "RL", "RL"))
    with pytest.raises(ValueError, match="slow walk"):
        go2_slow_walk_plan(step_length=0.2)
    model = Go2HybridDynamics()
    target = model.contact_positions[None].clone()
    target[:, 0, 0] += 10
    with pytest.raises(ValueError, match="unreachable"):
        go2_leg_ik(model, model.reference_configuration[None], target)
    with pytest.raises(ValueError, match="float64"):
        Go2WalkMPC(dtype=torch.float32, compile_physics=False)


@pytest.mark.crocoddyl
def test_urdf_com_against_pinocchio(plan):
    pin = pytest.importorskip("pinocchio")
    model = Go2HybridDynamics()
    native = Go2FixedContactReference()
    q = plan.reference[::73, :19]
    computed = go2_center_of_mass(model, q)
    for b in range(len(q)):
        expected = pin.centerOfMass(
            native.robot.model, native.robot.model.createData(), q[b].numpy()
        )
        np.testing.assert_allclose(computed[b].numpy(), expected, atol=1e-12, rtol=1e-12)


@pytest.mark.crocoddyl
def test_compiled_live_control_maps_for_mixed_masks_and_moved_anchors():
    pytest.importorskip("pinocchio")
    controller = Go2WalkMPC(batch_size=2)
    model = controller.model
    state = model.standing_state.expand(2, -1).clone()
    state[:, 0] = state.new_tensor([0.05, 0.12])
    anchors = model.contact_positions.expand(2, -1, -1).clone()
    anchors[:, :, 0] += state[:, 0, None]
    masks = torch.tensor([[True, True, False, True], [False, True, True, True]])
    torque = model.quasi_static_torques().expand(2, -1).clone()
    a0, Au, f0, Fu = controller._maps(state, contact_mask=masks, contact_positions=anchors)
    a = a0 + (Au @ torque[..., None]).squeeze(-1)
    f = (f0 + (Fu @ torque[..., None]).squeeze(-1)).reshape(2, 4, 3)
    assert f[~masks].eq(0).all()
    for b in range(2):
        native = Go2FixedContactReference(
            feet=tuple(name for name, active in zip(model.feet, masks[b]) if active),
            stabilization=model.gains,
        )
        native.positions = tuple(anchors[b, i].numpy() for i in range(4) if masks[b, i])
        expected = native.calc(state[b, :19].numpy(), state[b, 19:].numpy(), torque[b].numpy())
        np.testing.assert_allclose(a[b].numpy(), expected.acceleration, atol=1e-9, rtol=1e-9)
        np.testing.assert_allclose(
            f[b, masks[b]].numpy(), expected.forces_world, atol=1e-9, rtol=1e-9
        )
    output, forces, positions = controller._physics(state, torque, masks, anchors)
    torch.testing.assert_close(
        output,
        model.calc(state, torque, contact_mask=masks, contact_positions=anchors),
        atol=1e-11,
        rtol=1e-11,
    )
    torch.testing.assert_close(forces, f, atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(
        positions, model.foot_positions(output[:, :19]), atol=1e-12, rtol=1e-12
    )


def test_mpc_checks_complete_horizon_and_isolates_invalid_observations(plan):
    controller = Go2WalkMPC(batch_size=2, compile_physics=False)
    state = plan.reference[0].expand(2, -1).clone()
    reference = state[:, None].expand(-1, 3, -1).clone()
    masks = torch.ones(2, 4, dtype=torch.bool)
    anchors = plan.anchors[0].expand(2, -1, -1).clone()
    action, result = controller.compute(state, reference, state.new_zeros(2, 18), masks, anchors)
    assert result.usable.all()
    for t in range(2):
        physical = controller.model.contact_dynamics(
            result.xs[:, t], result.us[:, t], contact_mask=masks, contact_positions=anchors
        )
        assert controller.constraints.feasible(result.us[:, t], physical.forces_world).all()
        torch.testing.assert_close(
            controller.model.calc(
                result.xs[:, t], result.us[:, t], contact_mask=masks, contact_positions=anchors
            ),
            result.xs[:, t + 1],
        )
    state[0, 3:7] = 0
    action, result = controller.compute(state, reference, state.new_zeros(2, 18), masks, anchors)
    assert result.usable.tolist() == [False, True]
    assert result.status[0] == SolveStatus.NUMERICAL_FAILURE
    assert torch.isnan(action[0]).all()
    state[0] = plan.reference[0]
    anchors[0, 0, 0] = float("nan")
    action, result = controller.compute(state, reference, state.new_zeros(2, 18), masks, anchors)
    assert result.usable.tolist() == [False, True] and torch.isnan(action[0]).all()
    with pytest.raises(ValueError, match="reset"):
        controller.reset(torch.ones(3, dtype=torch.bool))


def test_touchdown_wait_timeout_and_selective_reset(plan):
    controller = Go2SlowWalkController(
        plan, batch_size=2, max_contact_wait=0.02, compile_physics=False
    )
    frame = plan.phase_names.index("land_RL")
    controller.indices[0] = frame
    controller.contact_mask[0] = plan.contact_masks[frame - 1]
    controller.anchors[0] = plan.anchors[frame - 1]
    state = plan.reference[0].expand(2, -1).clone()
    state[0] = plan.reference[frame]
    state[0, 19:] = 0
    state[0, 15] -= 0.06  # RL calf lifts only the free foot above the landing plane
    output = controller.compute(state)
    assert output.result.usable.all()
    assert output.waiting_for_contact.tolist() == [True, False]
    assert controller.indices.tolist() == [frame, 1]
    assert not output.contact_mask[0, 2] and not output.event_accepted.any()
    torch.testing.assert_close(output.state, state, atol=0, rtol=0)
    output = controller.compute(state)  # no plant advance: foot remains too high
    assert output.result.usable.tolist() == [False, True]
    assert torch.isnan(output.action[0]).all() and controller.indices[0] == frame
    old_index = controller.indices[1].clone()
    old_warm = controller.mpc._warm[1].clone()
    controller.reset(torch.tensor([True, False]))
    assert controller.indices[0] == 0 and not controller.failed[0]
    assert controller.contact_mask[0].all()
    torch.testing.assert_close(controller.indices[1], old_index)
    torch.testing.assert_close(controller.mpc._warm[1], old_warm, atol=0, rtol=0)


def test_liftoff_requires_measured_com_inside_support_triangle(plan):
    controller = Go2SlowWalkController(plan, max_contact_wait=0.02, compile_physics=False)
    frame = plan.phase_names.index("swing_RL")
    controller.indices[0] = frame
    state = plan.reference[0:1].clone()
    output = controller.compute(state)
    assert output.waiting_for_contact[0] and not output.event_accepted[0]
    assert output.contact_mask.all() and controller.indices[0] == frame
    torch.testing.assert_close(output.state, state, atol=0, rtol=0)


@pytest.mark.crocoddyl
def test_two_cycle_walk_against_independent_pinocchio_plant():
    pytest.importorskip("pinocchio")
    from examples.go2_slow_walk import run_walk

    report = run_walk(cycles=2, batch_size=2, plant="pinocchio", compile_physics=False)
    assert report["all_actions_usable"] and report["all_finished"]
    assert report["releases"] == [["RL", "FL", "RR", "FR"] * 2] * 2
    assert report["landings"] == report["releases"]
    assert min(report["forward_displacement_m"]) > 0.039
    assert max(report["final_state_error"]) < 0.002
    assert min(min(heights) for heights in report["peak_swing_height_m"]) > 0.01
    assert report["min_three_foot_com_margin_m"] >= 0.02
    assert report["max_stance_anchor_error_m"] < 0.002
    assert report["max_inactive_force_N"] == 0
    assert report["max_force_violation_N"] <= 1e-7
    assert report["max_torque_violation_Nm"] <= 1e-7
    assert report["max_impulse_violation_Ns"] <= 1e-7
    assert report["min_normal_force_N"] >= 0.5 - 1e-7


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_walk_cuda_nondefault_stream_live_masks():
    controller = Go2WalkMPC(batch_size=2, device="cuda")
    state = controller.model.standing_state.expand(2, -1).clone()
    reference = state[:, None].expand(-1, 3, -1).clone()
    mask = torch.ones(2, 4, device="cuda", dtype=torch.bool)
    anchors = controller.model.contact_positions.expand(2, -1, -1).clone()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        action, result = controller.compute(state, reference, state.new_zeros(2, 18), mask, anchors)
        consumer = action.square().sum()
    stream.synchronize()
    assert result.usable.all() and torch.isfinite(consumer)
