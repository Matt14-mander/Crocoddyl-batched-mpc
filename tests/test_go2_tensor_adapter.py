"""Simulator tensor mapping and end-to-end Go2 Torch adapter gates."""

import builtins

import pytest
import torch

from crocoddyl_batched_mpc.lie import quaternion_matrix
from crocoddyl_batched_mpc.models import (
    Go2TensorAdapter,
    go2_front_leg_tracking_plan,
)
from crocoddyl_batched_mpc.models.go2 import GO2_JOINT_NAMES


def _sim_tensors(adapter, state, origins):
    batch = len(state)
    pose = state.new_empty(batch, 7)
    pose[:, :3] = state[:, :3] + origins
    pose[:, 3:] = state[:, (6, 3, 4, 5)]  # xyzw -> wxyz
    rotation = quaternion_matrix(state[:, 3:7])
    velocity = torch.cat(
        (
            (rotation @ state[:, 19:22, None]).squeeze(-1),
            (rotation @ state[:, 22:25, None]).squeeze(-1),
        ),
        -1,
    )
    position = state.new_zeros(batch, adapter.sim_joint_count)
    joint_velocity = torch.zeros_like(position)
    position.index_copy_(1, adapter.mpc_to_sim, state[:, 7:19])
    joint_velocity.index_copy_(1, adapter.mpc_to_sim, state[:, 25:37])
    return pose, velocity, position, joint_velocity


def make_adapter(batch=2, device="cpu", **settings):
    names = ("camera_joint", *reversed(GO2_JOINT_NAMES))
    return Go2TensorAdapter(names, batch, device=device, compile_physics=False, **settings)


def test_pack_and_scatter_handle_origins_frames_and_permuted_joints():
    adapter = make_adapter()
    model = adapter.controller.model
    state = model.standing_state.expand(2, -1).clone()
    delta = state.new_zeros(2, 36)
    delta[:, 3:6] = state.new_tensor([[0.0, 0.0, 0.7], [0.2, -0.1, 0.3]])
    state = model.manifold.integrate(state, delta)
    state[:, 19:25] = state.new_tensor(
        [[0.2, -0.1, 0.05, 0.3, 0.1, -0.2], [-0.1, 0.3, 0.2, -0.2, 0.1, 0.4]]
    )
    state[:, 25:] = torch.arange(24, dtype=state.dtype).reshape(2, 12) * 0.01
    origins = state.new_tensor([[1.0, 2.0, 0.0], [-3.0, 1.5, 0.2]])
    values = _sim_tensors(adapter, state, origins)
    packed = adapter.pack_state(*values, origins)
    torch.testing.assert_close(packed, state, atol=1e-12, rtol=1e-12)
    body_adapter = make_adapter(quaternion_order="xyzw", velocity_frame="body")
    body_pose = torch.cat((state[:, :3] + origins, state[:, 3:7]), -1)
    body_position = state.new_zeros(2, body_adapter.sim_joint_count)
    body_velocity = torch.zeros_like(body_position)
    body_position.index_copy_(1, body_adapter.mpc_to_sim, state[:, 7:19])
    body_velocity.index_copy_(1, body_adapter.mpc_to_sim, state[:, 25:37])
    body_packed = body_adapter.pack_state(
        body_pose, state[:, 19:25], body_position, body_velocity, origins
    )
    torch.testing.assert_close(body_packed, state, atol=1e-12, rtol=1e-12)
    action = torch.arange(24, dtype=state.dtype).reshape(2, 12)
    scattered = adapter.scatter_action(action)
    torch.testing.assert_close(scattered.index_select(1, adapter.mpc_to_sim), action)
    assert scattered[:, 0].eq(0).all()


def test_adapter_compute_isolates_bad_environment_and_resets():
    adapter = make_adapter()
    plan = go2_front_leg_tracking_plan(adapter.controller.model, standing_only=True)
    state = plan.reference[0].expand(2, -1).clone()
    origins = state.new_tensor([[0.0, 0.0, 0.0], [2.0, -1.0, 0.0]])
    values = list(_sim_tensors(adapter, state, origins))
    output = adapter.compute(*values, plan.command(0, 2, adapter.controller.horizon), origins)
    assert output.tracking.usable.all()
    torch.testing.assert_close(output.mpc_state, state)
    torch.testing.assert_close(
        output.joint_effort.index_select(1, adapter.mpc_to_sim), output.tracking.action
    )
    values[0] = values[0].clone()
    values[0][0, 0] = float("nan")
    rejected = adapter.compute(*values, plan.command(0, 2, adapter.controller.horizon), origins)
    assert rejected.tracking.usable.tolist() == [False, True]
    assert rejected.joint_effort[0].eq(0).all()
    adapter.reset(torch.tensor([True, False]))
    values[0][0, 0] = state[0, 0] + origins[0, 0]
    assert adapter.compute(
        *values, plan.command(0, 2, adapter.controller.horizon), origins
    ).tracking.usable.all()


def test_adapter_uses_only_torch_and_validates_simulator_contract(monkeypatch):
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.split(".")[0] in {"numpy", "pinocchio", "crocoddyl", "isaaclab"}:
            raise AssertionError(f"unexpected optional dependency: {name}")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    adapter = make_adapter(batch=1)
    with pytest.raises(ValueError, match="missing Go2 joints"):
        Go2TensorAdapter(GO2_JOINT_NAMES[:-1])
    with pytest.raises(ValueError, match="quaternion_order"):
        Go2TensorAdapter(GO2_JOINT_NAMES, quaternion_order="scalar-last-ish")
    state = adapter.controller.model.standing_state[None]
    origins = state.new_zeros(1, 3)
    values = _sim_tensors(adapter, state, origins)
    with pytest.raises(ValueError, match="root_pose_world"):
        adapter.pack_state(values[0].float(), *values[1:], origins)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_adapter_cuda_nondefault_stream():
    adapter = make_adapter(device="cuda")
    plan = go2_front_leg_tracking_plan(adapter.controller.model, standing_only=True)
    state = plan.reference[0].expand(2, -1).clone()
    origins = state.new_tensor([[0.0, 0.0, 0.0], [2.0, -1.0, 0.0]])
    values = _sim_tensors(adapter, state, origins)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        output = adapter.compute(*values, plan.command(0, 2, adapter.controller.horizon), origins)
        consumed = output.joint_effort.square().sum(-1)
    stream.synchronize()
    assert output.tracking.usable.all() and torch.isfinite(consumed).all()
