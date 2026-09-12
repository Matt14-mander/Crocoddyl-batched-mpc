import torch

from crocoddyl_batched_mpc.models.pendulum import PendulumCost

# Simple test
cost = PendulumCost(device="cpu", dtype=torch.float64)
x = torch.tensor([[1.0, 0.5]], dtype=torch.float64)
u = torch.tensor([[0.2]], dtype=torch.float64)

print("x_ref:", cost.x_ref)
print("Q:", cost.Q)
print("R:", cost.R)
print("x:", x)
print("u:", u)
print("error:", x - cost.x_ref)

# Analytical
value = cost.calc(x, u)
lx, lu, lxx, luu, lxu = cost.calc_diff(x, u)

print("\n=== Analytical ===")
print(f"cost: {value.item():.6f}")
print(f"lx: {lx}")
print(f"lu: {lu}")

# Finite difference for lx
eps = 1e-7
lx_fd = torch.zeros_like(x)
for i in range(2):
    x_plus = x.clone()
    x_plus[:, i] += eps
    l_plus = cost.calc(x_plus, u)
    lx_fd[:, i] = (l_plus - value) / eps

print("\n=== Finite Difference ===")
print(f"lx_fd: {lx_fd}")
print(f"lx error: {(lx - lx_fd).abs().max().item():.6e}")

# Manual calculation
error = x - cost.x_ref
print("\n=== Manual ===")
print(f"Q @ error^T = {torch.matmul(cost.Q, error.T).T}")
