"""End-to-end Go2 Torch example using simulator-shaped articulation tensors.

No Isaac Lab, Pinocchio or Crocoddyl import is required. The separate Torch
plant emits world-frame root state in wxyz order and reversed joint order,
matching the data conversion required by a vectorized simulator task.
"""

import argparse
import json

import torch

from crocoddyl_batched_mpc.lie import quaternion_matrix
from crocoddyl_batched_mpc.models import (
    Go2HybridDynamics,
    Go2TensorAdapter,
    go2_front_leg_tracking_plan,
)
from crocoddyl_batched_mpc.models.go2 import GO2_JOINT_NAMES


def _articulation_tensors(adapter, state, origins):
    pose = torch.cat((state[:, :3] + origins, state[:, (6, 3, 4, 5)]), -1)
    rotation = quaternion_matrix(state[:, 3:7])
    root_velocity_world = torch.cat(
        (
            (rotation @ state[:, 19:22, None]).squeeze(-1),
            (rotation @ state[:, 22:25, None]).squeeze(-1),
        ),
        -1,
    )
    joint_position = state.new_zeros(len(state), adapter.sim_joint_count)
    joint_velocity = torch.zeros_like(joint_position)
    joint_position.index_copy_(1, adapter.mpc_to_sim, state[:, 7:19])
    joint_velocity.index_copy_(1, adapter.mpc_to_sim, state[:, 25:37])
    return pose, root_velocity_world, joint_position, joint_velocity


@torch.no_grad()
def run_example(*, task="front-leg", batch_size=2, device="cpu"):
    if task not in ("stance", "front-leg"):
        raise ValueError("task must be stance or front-leg")
    if torch.device(device).type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    # Deliberately reverse simulator order to exercise name-based gather/scatter.
    adapter = Go2TensorAdapter(
        tuple(reversed(GO2_JOINT_NAMES)),
        batch_size,
        device=device,
        quaternion_order="wxyz",
        velocity_frame="world",
        compile_physics=False,
    )
    model = adapter.controller.model
    plant = Go2HybridDynamics(device=device, dt=model.dt, dtype=model.dtype)
    plan = go2_front_leg_tracking_plan(model, standing_only=task == "stance")
    state = plan.reference[0].expand(batch_size, -1).clone()
    origins = state.new_zeros(batch_size, 3)
    origins[:, 0] = torch.arange(batch_size, device=state.device, dtype=state.dtype) * 2.0
    all_usable = torch.ones(batch_size, device=state.device, dtype=torch.bool)
    maximum_error = state.new_zeros(())
    for step in range(len(plan.reference)):
        tensors = _articulation_tensors(adapter, state, origins)
        command = plan.command(step, batch_size, adapter.controller.horizon)
        output = adapter.compute(*tensors, command, origins)
        torque = output.joint_effort.index_select(1, adapter.mpc_to_sim)
        contact = plant.contact_dynamics(
            state,
            torque,
            contact_mask=command.support_mask,
            contact_positions=command.anchors,
        )
        velocity = state[:, 19:] + plan.dt * contact.acceleration
        state = plant.manifold.integrate(
            state, torch.cat((plan.dt * velocity, plan.dt * contact.acceleration), -1)
        )
        all_usable &= output.tracking.usable
        target = plan.feet[min(step + 1, len(plan.feet) - 1), 0]
        maximum_error = torch.maximum(
            maximum_error, (plant.foot_positions(state[:, :19])[:, 0] - target).norm(dim=-1).max()
        )
    final_target = plan.feet[-1, 0]
    final_error = (plant.foot_positions(state[:, :19])[:, 0] - final_target).norm(dim=-1).max()
    return {
        "task": task,
        "device": str(torch.device(device)),
        "batch_size": batch_size,
        "steps": len(plan.reference),
        "all_actions_usable": bool(all_usable.all()),
        "max_fl_error_m": float(maximum_error),
        "final_fl_error_m": float(final_error),
        "joint_order_permuted": True,
        "root_quaternion_order": "wxyz",
        "root_velocity_frame": "world",
        "passed": bool(all_usable.all())
        and float(maximum_error) < 0.01
        and float(final_error) < 0.003,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("stance", "front-leg"), default="front-leg")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(1)
    report = run_example(**vars(args))
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit("Go2 tensor adapter acceptance failed")


if __name__ == "__main__":
    main()
