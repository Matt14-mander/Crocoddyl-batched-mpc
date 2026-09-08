"""python examples/double_integrator.py --device cuda --batch-size 1024"""

import argparse

import torch

from crocoddyl_batched_mpc import BatchedMPC, MPCController
from crocoddyl_batched_mpc.models import double_integrator


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--backend", default="torch", choices=["torch", "crocoddyl"])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--steps", type=int, default=100)
    args = parser.parse_args()
    p = double_integrator(batch_size=args.batch_size, device=args.device)
    controller = MPCController(BatchedMPC(p, backend=args.backend))
    state = p.A.new_zeros(p.batch_size, p.nx)
    state[:, 0] = torch.linspace(-1, 1, p.batch_size, device=state.device)
    initial_error = state[:, 0].square().mean().sqrt().item()
    failures = torch.zeros((), dtype=torch.int64, device=state.device)
    for _ in range(args.steps):
        action, result = controller.compute(state)
        failures += (~result.success).sum()
        state = (p.A[:, 0] @ state[..., None] + p.B[:, 0] @ action[..., None]).squeeze(-1)
    print(f"device={state.device}, batch={p.batch_size}, failed_solves={failures.item()}")
    print(f"position RMS: {initial_error:.6f} -> {state[:, 0].square().mean().sqrt().item():.6f}")


if __name__ == "__main__":
    main()
