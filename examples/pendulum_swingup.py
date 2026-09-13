"""Pendulum swing-up example using DDP.

python examples/pendulum_swingup.py --device cpu --batch-size 16
python examples/pendulum_swingup.py --device cuda --batch-size 512
"""

import argparse

import torch

from crocoddyl_batched_mpc import BatchedMPC, DDPProblem, MPCController
from crocoddyl_batched_mpc.manifolds import EuclideanManifold
from crocoddyl_batched_mpc.models.pendulum import PendulumCost, PendulumDynamics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu", help="Device: cpu or cuda")
    parser.add_argument(
        "--batch-size", type=int, default=16, help="Number of parallel environments"
    )
    parser.add_argument("--horizon", type=int, default=50, help="MPC horizon")
    parser.add_argument("--steps", type=int, default=100, help="Simulation steps")
    parser.add_argument("--max-iters", type=int, default=50, help="Max DDP iterations per solve")
    args = parser.parse_args()

    device = args.device
    dtype = torch.float32
    batch_size = args.batch_size
    horizon = args.horizon

    # Create dynamics and cost
    dynamics = PendulumDynamics(device=device, dtype=dtype, dt=0.05)
    cost = PendulumCost(device=device, dtype=dtype)
    manifold = EuclideanManifold(n=2)

    # Initial guess: forward rollout with zero control
    x_init_single = torch.zeros(horizon + 1, 2, device=device, dtype=dtype)
    x_init_single[0] = torch.tensor([0.0, 0.0], device=device, dtype=dtype)  # Start at bottom
    u_zero = torch.zeros(1, device=device, dtype=dtype)

    for t in range(horizon):
        x_init_single[t + 1] = dynamics.calc(x_init_single[t : t + 1], u_zero.unsqueeze(0)).squeeze(
            0
        )

    x_init = x_init_single.unsqueeze(0).expand(batch_size, -1, -1).clone()
    u_init = torch.zeros(batch_size, horizon, 1, device=device, dtype=dtype)

    # Create DDP problem
    problem = DDPProblem(
        dynamics=dynamics,
        cost=cost,
        manifold=manifold,
        batch_size=batch_size,
        horizon=horizon,
        x_init=x_init,
        u_init=u_init,
        max_iterations=args.max_iters,
    )

    # Create controller
    solver = BatchedMPC(problem, backend="torch")
    controller = MPCController(solver)

    # Initial states: random angles near bottom
    torch.manual_seed(42)
    state = torch.zeros(batch_size, 2, device=device, dtype=dtype)
    state[:, 0] = torch.randn(batch_size, device=device, dtype=dtype) * 0.5  # Small random angles
    state[:, 1] = torch.randn(batch_size, device=device, dtype=dtype) * 0.2  # Small velocities

    print(f"Device: {device}, Batch: {batch_size}, Horizon: {horizon}")
    print(f"Initial position RMS: {state[:, 0].square().mean().sqrt().item():.4f} rad")
    print("Target: θ=π (upright)\n")

    # Simulation loop
    total_failures = 0
    total_iterations = 0

    for step in range(args.steps):
        action, result = controller.compute(state)

        failures = (~result.usable).sum().item()
        total_failures += failures
        total_iterations += result.iterations.float().mean().item()

        if step % 20 == 0:
            angle_error = ((state[:, 0] - torch.pi) % (2 * torch.pi)).abs()
            angle_error = torch.minimum(angle_error, 2 * torch.pi - angle_error)
            avg_angle_error = angle_error.mean().item()
            avg_vel = state[:, 1].abs().mean().item()

            print(
                f"Step {step:3d}: angle_err={avg_angle_error:.4f} rad, "
                f"vel={avg_vel:.4f} rad/s, failures={failures}/{batch_size}, "
                f"avg_iters={result.iterations.float().mean().item():.1f}"
            )

        # Apply dynamics
        state = dynamics.calc(state, action)

    # Final statistics
    final_angle = state[:, 0]
    final_vel = state[:, 1]

    # Compute angle error (wrap around)
    angle_error = ((final_angle - torch.pi) % (2 * torch.pi)).abs()
    angle_error = torch.minimum(angle_error, 2 * torch.pi - angle_error)

    print("\n=== Final Results ===")
    print(
        f"Final angle error: {angle_error.mean().item():.4f} ± {angle_error.std().item():.4f} rad"
    )
    mean_velocity = final_vel.abs().mean().item()
    std_velocity = final_vel.abs().std().item()
    print(f"Final velocity: {mean_velocity:.4f} ± {std_velocity:.4f} rad/s")
    print(f"Success rate: {(angle_error < 0.1).sum().item()}/{batch_size} envs stabilized")
    print(f"Total failures: {total_failures}")
    print(f"Avg iterations per solve: {total_iterations / args.steps:.1f}")


if __name__ == "__main__":
    main()
