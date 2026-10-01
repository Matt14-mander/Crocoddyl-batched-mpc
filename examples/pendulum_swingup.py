"""Unconstrained pendulum closed loop with sustained stabilization metrics.

python examples/pendulum_swingup.py --device cpu --batch-size 16
python examples/pendulum_swingup.py --device cuda --batch-size 512 --backend fddp
"""

import argparse

import torch

from crocoddyl_batched_mpc import BatchedMPC, DDPProblem, MPCController
from crocoddyl_batched_mpc.models.pendulum import PendulumCost, PendulumDynamics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--backend", choices=["torch", "fddp"], default="torch")
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--horizon", type=int, default=30)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--max-iters", type=int, default=6)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--settle-window", type=int, default=20)
    parser.add_argument("--angle-tolerance", type=float, default=0.1)
    parser.add_argument("--velocity-tolerance", type=float, default=0.1)
    args = parser.parse_args()
    for name in ("batch_size", "horizon", "steps", "max_iters", "threads", "settle_window"):
        if getattr(args, name) < 1:
            parser.error(f"{name} must be positive")
    if args.settle_window > args.steps:
        parser.error("settle-window must not exceed steps")
    for name in ("angle_tolerance", "velocity_tolerance"):
        value = getattr(args, name)
        if not 0 < value < float("inf"):
            parser.error(f"{name} must be finite and positive")
    torch.set_num_threads(args.threads)
    dtype = getattr(torch, args.dtype)
    dynamics = PendulumDynamics(device=args.device, dtype=dtype, dt=0.05)
    problem = DDPProblem.from_models(
        dynamics,
        PendulumCost(device=args.device, dtype=dtype),
        batch_size=args.batch_size,
        horizon=args.horizon,
        max_iterations=args.max_iters,
        cost_tolerance=1e-6,
    )
    controller = MPCController(BatchedMPC(problem, backend=args.backend))

    torch.manual_seed(42)
    state = torch.zeros(args.batch_size, 2, device=args.device, dtype=dtype)
    state[:, 0] = torch.randn(args.batch_size, device=args.device, dtype=dtype) * 0.5
    state[:, 1] = torch.randn(args.batch_size, device=args.device, dtype=dtype) * 0.2
    stabilized = torch.ones(args.batch_size, device=args.device, dtype=torch.bool)
    total_failures = state.new_zeros((), dtype=torch.int64)
    total_iterations = state.new_zeros(())

    print(f"Device: {args.device}, Batch: {args.batch_size}, Horizon: {args.horizon}")
    print("Target: θ=π (upright); unconstrained torque")
    for step in range(args.steps):
        action, result = controller.compute(state)
        total_failures += (~result.usable).sum()
        total_iterations += result.iterations.to(dtype).mean()
        state = dynamics.calc(state, action)
        angle_error = (torch.remainder(state[:, 0], 2 * torch.pi) - torch.pi).abs()
        velocity = state[:, 1].abs()
        if step >= args.steps - args.settle_window:
            stabilized &= (
                result.usable
                & torch.isfinite(state).all(-1)
                & (angle_error < args.angle_tolerance)
                & (velocity < args.velocity_tolerance)
            )
        if (step + 1) % 20 == 0:
            print(
                f"Step {step + 1:3d}: angle_err={angle_error.mean().item():.4f} rad, "
                f"vel={velocity.mean().item():.4f} rad/s, "
                f"failures={(~result.usable).sum().item()}/{args.batch_size}"
            )

    print(f"Final max angle error: {angle_error.max().item():.6f} rad")
    print(f"Final max velocity: {velocity.max().item():.6f} rad/s")
    print(
        f"Sustained stabilization ({args.settle_window} steps): "
        f"{stabilized.sum().item()}/{args.batch_size}"
    )
    print(f"Total unusable solves: {total_failures.item()}")
    print(f"Avg iterations per solve: {(total_iterations / args.steps).item():.1f}")
    if not stabilized.all().item() or total_failures.item():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
