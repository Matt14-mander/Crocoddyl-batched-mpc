"""Debug DDP backward pass."""

import torch
from crocoddyl_batched_mpc import DDPProblem
from crocoddyl_batched_mpc.manifolds import EuclideanManifold
from crocoddyl_batched_mpc.models.pendulum import PendulumCost, PendulumDynamics

# Setup
device = "cpu"
dtype = torch.float32
batch_size = 2
horizon = 5

dynamics = PendulumDynamics(device=device, dtype=dtype, dt=0.05)
cost = PendulumCost(device=device, dtype=dtype)
manifold = EuclideanManifold(n=2)

# Simple initial trajectory
x_init = torch.zeros(batch_size, horizon + 1, 2, device=device, dtype=dtype)
u_init = torch.zeros(batch_size, horizon, 1, device=device, dtype=dtype)

problem = DDPProblem(
    dynamics=dynamics,
    cost=cost,
    manifold=manifold,
    batch_size=batch_size,
    horizon=horizon,
    x_init=x_init,
    u_init=u_init,
    max_iterations=5,
)

# Test dynamics derivatives
x = torch.tensor([[0.5, 0.1], [0.3, -0.2]], device=device, dtype=dtype)
u = torch.tensor([[0.1], [0.2]], device=device, dtype=dtype)

print("Testing dynamics derivatives...")
with torch.enable_grad():
    Fx, Fu = dynamics.calc_diff(x, u)
    print(f"Fx shape: {Fx.shape}, finite: {torch.isfinite(Fx).all()}")
    print(f"Fu shape: {Fu.shape}, finite: {torch.isfinite(Fu).all()}")
    print(f"Fx[0]:\n{Fx[0]}")
    print(f"Fu[0]:\n{Fu[0]}")

# Test cost derivatives
print("\nTesting cost derivatives...")
lx, lu, lxx, luu, lxu = cost.calc_diff(x, u)
print(f"lx: {lx}")
print(f"lu: {lu}")
print(f"lxx[0]:\n{lxx[0]}")
print(f"luu[0]:\n{luu[0]}")

# Test Q-function computation (simplified backward pass)
print("\nTesting Q-function...")
Vx = lx
Vxx = lxx

Qx = lx + torch.matmul(Fx.transpose(-1, -2), Vx.unsqueeze(-1)).squeeze(-1)
Qu = lu + torch.matmul(Fu.transpose(-1, -2), Vx.unsqueeze(-1)).squeeze(-1)
Quu = luu + torch.matmul(torch.matmul(Fu.transpose(-1, -2), Vxx), Fu)

print(f"Qx: {Qx}")
print(f"Qu: {Qu}")
print(f"Quu[0]:\n{Quu[0]}")
print(f"Quu eigenvalues[0]: {torch.linalg.eigvalsh(Quu[0])}")

# Try Cholesky
print("\nTesting Cholesky...")
reg = 1e-6
eye = torch.eye(1, device=device, dtype=dtype)
Quu_reg = Quu + reg * eye
L, info = torch.linalg.cholesky_ex(Quu_reg, check_errors=False)
print(f"Cholesky info: {info}")
print(f"L[0]: {L[0]}")
