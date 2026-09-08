"""Test DDP solver with verbose output."""

import torch
from crocoddyl_batched_mpc import BatchedMPC, DDPProblem
from crocoddyl_batched_mpc.manifolds import EuclideanManifold
from crocoddyl_batched_mpc.models.pendulum import PendulumCost, PendulumDynamics

# Setup
device = "cpu"
dtype = torch.float64  # Use float64 for better numerics
batch_size = 1
horizon = 10

dynamics = PendulumDynamics(device=device, dtype=dtype, dt=0.05)
cost = PendulumCost(device=device, dtype=dtype)
manifold = EuclideanManifold(n=2)

# Initial trajectory: simple forward rollout
x_init = torch.zeros(batch_size, horizon + 1, 2, device=device, dtype=dtype)
x_init[:, 0, 0] = 0.1  # Start near bottom
u_init = torch.zeros(batch_size, horizon, 1, device=device, dtype=dtype)

# Forward rollout
for t in range(horizon):
    x_init[:, t + 1] = dynamics.calc(x_init[:, t], u_init[:, t])

print("Initial trajectory:")
print(f"x[0]: {x_init[0, 0]}")
print(f"x[T]: {x_init[0, -1]}")

problem = DDPProblem(
    dynamics=dynamics,
    cost=cost,
    manifold=manifold,
    batch_size=batch_size,
    horizon=horizon,
    x_init=x_init,
    u_init=u_init,
    max_iterations=20,
    cost_tolerance=1e-4,
)

# Compute initial cost
initial_cost = 0.0
for t in range(horizon):
    initial_cost += cost.calc(x_init[0:1, t], u_init[0:1, t]).item()
initial_cost += cost.calc(x_init[0:1, -1], None).item()

print(f"\nInitial cost: {initial_cost:.4f}")

# Solve
x0 = x_init[:, 0]
solver = BatchedMPC(problem, backend="torch")

print("\nSolving...")
result = solver.solve(x0)

print(f"\nResult:")
print(f"Status: {result.status.item()}")
print(f"Final cost: {result.cost.item():.4f}")
print(f"Iterations: {result.iterations.item()}")
print(f"x[0]: {result.xs[0, 0]}")
print(f"x[T]: {result.xs[0, -1]}")
print(f"u[0]: {result.us[0, 0]}")
print(f"Cost improvement: {initial_cost - result.cost.item():.4f}")
