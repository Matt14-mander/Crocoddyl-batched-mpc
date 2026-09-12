"""Test improved DDP solver with better initialization."""

import torch

from crocoddyl_batched_mpc import BatchedMPC, DDPProblem
from crocoddyl_batched_mpc.manifolds import EuclideanManifold
from crocoddyl_batched_mpc.models.pendulum import PendulumCost, PendulumDynamics

# Setup
device = "cpu"
dtype = torch.float64
batch_size = 1
horizon = 20

dynamics = PendulumDynamics(device=device, dtype=dtype, dt=0.05)
cost = PendulumCost(device=device, dtype=dtype)
manifold = EuclideanManifold(n=2)

# No initial trajectory - let solver initialize
problem = DDPProblem(
    dynamics=dynamics,
    cost=cost,
    manifold=manifold,
    batch_size=batch_size,
    horizon=horizon,
    max_iterations=50,
    cost_tolerance=1e-4,
)

# Initial state: near bottom
x0 = torch.tensor([[0.2, 0.0]], device=device, dtype=dtype)

print("Testing improved DDP with LQR initialization...")
print(f"Horizon: {horizon}, Max iterations: 50")
print(f"Initial state: θ={x0[0, 0].item():.3f} rad, θ̇={x0[0, 1].item():.3f} rad/s\n")

# Solve
solver = BatchedMPC(problem, backend="torch")
result = solver.solve(x0)

print(f"Status: {result.status.item()} (0=SUCCESS, 1=NUMERICAL_FAILURE, 2=MAX_ITERS)")
print(f"Iterations: {result.iterations.item()}")
print(f"Final cost: {result.cost.item():.4f}")
print(f"Final state: θ={result.xs[0, -1, 0].item():.4f} rad (target: π={torch.pi:.4f})")
print(f"Final velocity: θ̇={result.xs[0, -1, 1].item():.4f} rad/s")
print(f"\nFirst control: u[0]={result.us[0, 0].item():.4f}")
print(f"Control range: [{result.us[0].min().item():.4f}, {result.us[0].max().item():.4f}]")

# Check if converged
if result.status.item() == 0:
    print("\n✅ Successfully converged!")
else:
    print(f"\n⚠️ Did not converge in {result.iterations.item()} iterations")
