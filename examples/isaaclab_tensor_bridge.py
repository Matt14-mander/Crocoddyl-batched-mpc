"""Runnable tensor contract demo; see docs/isaaclab.md for DirectRLEnv hooks.

This does not launch Isaac Sim or model a particular robot. Set --device cuda
to exercise the same direct tensor call path used by an Isaac Lab task.
"""

import argparse

import torch

from crocoddyl_batched_mpc import BatchedMPC, MPCController
from crocoddyl_batched_mpc.models import double_integrator


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-envs", type=int, default=128)
    args = parser.parse_args()
    problem = double_integrator(batch_size=args.num_envs, device=args.device)
    controller = MPCController(BatchedMPC(problem))
    # A task packs its simulator state in the model's specified order.
    state = torch.ones((args.num_envs, 2), device=args.device)
    actions, result = controller.compute(state)
    reset_mask = torch.arange(args.num_envs, device=args.device) % 3 == 0
    controller.reset(reset_mask)
    assert actions.device == state.device
    print(
        f"actions={tuple(actions.shape)} on {actions.device}; success={result.success.all().item()}"
    )


if __name__ == "__main__":
    main()
