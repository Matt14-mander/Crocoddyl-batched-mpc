"""Pendulum swing-up: single inverted pendulum with torque control.

State: x = [θ, θ̇] where θ is angle from downward vertical (θ=0 is down, θ=π is up)
Control: u = [τ] torque
Dynamics: θ̈ = (g/l)·sin(θ) - (b/(m·l²))·θ̇ + τ/(m·l²)

Default parameters correspond to a standard underactuated pendulum.
"""

import torch
from torch import Tensor


class PendulumDynamics:
    """Discrete-time pendulum dynamics using semi-implicit Euler integration.

    θ[t+1] = θ[t] + dt·θ̇[t+1]
    θ̇[t+1] = θ̇[t] + dt·θ̈
    """

    def __init__(
        self,
        mass: float = 1.0,
        length: float = 1.0,
        damping: float = 0.1,
        gravity: float = 9.81,
        dt: float = 0.05,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.m = mass
        self.l = length
        self.b = damping
        self.g = gravity
        self._dt = dt
        self.device = device
        self.dtype = dtype

        # Precompute coefficients
        self.inertia = mass * length**2
        self.gravity_coeff = gravity / length
        self.damping_coeff = damping / self.inertia
        self.control_coeff = 1.0 / self.inertia

    @property
    def nx(self) -> int:
        return 2

    @property
    def nu(self) -> int:
        return 1

    @property
    def dt(self) -> float:
        return self._dt

    def calc(self, x: Tensor, u: Tensor) -> Tensor:
        """Forward dynamics with semi-implicit Euler.

        Args:
            x: [batch, 2] where x[:, 0] = θ, x[:, 1] = θ̇
            u: [batch, 1] torque

        Returns:
            x_next: [batch, 2]
        """
        theta, theta_dot = x[:, 0:1], x[:, 1:2]
        tau = u[:, 0:1]

        # θ̈ = (g/l)·sin(θ) - (b/(m·l²))·θ̇ + τ/(m·l²)
        theta_ddot = (
            self.gravity_coeff * torch.sin(theta)
            - self.damping_coeff * theta_dot
            + self.control_coeff * tau
        )

        # Semi-implicit Euler: update velocity first, then position
        theta_dot_next = theta_dot + self.dt * theta_ddot
        theta_next = theta + self.dt * theta_dot_next

        return torch.cat([theta_next, theta_dot_next], dim=-1)

    def calc_diff(self, x: Tensor, u: Tensor) -> tuple[Tensor, Tensor]:
        """Compute dynamics Jacobians using automatic differentiation.

        Args:
            x: [batch, 2]
            u: [batch, 1]

        Returns:
            Fx: [batch, 2, 2] - ∂f/∂x
            Fu: [batch, 2, 1] - ∂f/∂u
        """
        batch = x.shape[0]
        x_copy = x.detach().clone().requires_grad_(True)
        u_copy = u.detach().clone().requires_grad_(True)

        x_next = self.calc(x_copy, u_copy)

        Fx = torch.zeros(batch, self.nx, self.nx, device=x.device, dtype=x.dtype)
        Fu = torch.zeros(batch, self.nx, self.nu, device=x.device, dtype=x.dtype)

        # Compute Jacobian for each output dimension
        for i in range(self.nx):
            grad_outputs = torch.zeros_like(x_next)
            grad_outputs[:, i] = 1.0

            # Compute gradients
            grads = torch.autograd.grad(
                x_next,
                [x_copy, u_copy],
                grad_outputs=grad_outputs,
                retain_graph=True,
                create_graph=False,
            )

            Fx[:, i, :] = grads[0]
            Fu[:, i, :] = grads[1]

        return Fx, Fu


class PendulumCost:
    """Quadratic tracking cost for pendulum swing-up.

    Running cost: l(x, u) = 0.5·(x - x_ref)^T·Q·(x - x_ref) + 0.5·u^T·R·u
    Terminal cost: l_T(x) = 0.5·(x - x_ref)^T·Q_T·(x - x_ref)

    Default target is upright position (θ=π, θ̇=0).
    """

    def __init__(
        self,
        x_ref: Tensor | None = None,
        Q: Tensor | None = None,
        R: Tensor | None = None,
        Q_terminal: Tensor | None = None,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.device = device
        self.dtype = dtype

        # Default: upright position (θ=π, θ̇=0)
        if x_ref is None:
            x_ref = torch.tensor([torch.pi, 0.0], device=device, dtype=dtype)
        self.x_ref = x_ref

        # Default weights
        if Q is None:
            Q = torch.diag(torch.tensor([10.0, 1.0], device=device, dtype=dtype))
        self.Q = Q

        if R is None:
            R = torch.tensor([[0.1]], device=device, dtype=dtype)
        self.R = R

        if Q_terminal is None:
            Q_terminal = torch.diag(torch.tensor([100.0, 10.0], device=device, dtype=dtype))
        self.Q_terminal = Q_terminal

    @property
    def nx(self) -> int:
        return 2

    @property
    def nu(self) -> int:
        return 1

    def calc(self, x: Tensor, u: Tensor | None = None) -> Tensor:
        """Evaluate cost.

        Args:
            x: [batch, 2]
            u: [batch, 1] or None for terminal cost

        Returns:
            cost: [batch]
        """
        error = x - self.x_ref
        Q = self.Q_terminal if u is None else self.Q
        cost = 0.5 * (error * (Q @ error.unsqueeze(-1)).squeeze(-1)).sum(-1)

        if u is not None:
            cost += 0.5 * (u * (self.R @ u.unsqueeze(-1)).squeeze(-1)).sum(-1)

        return cost

    def calc_diff(
        self, x: Tensor, u: Tensor | None = None
    ) -> tuple[Tensor, Tensor | None, Tensor, Tensor | None, Tensor | None]:
        """Compute cost derivatives.

        For quadratic cost, derivatives are analytical.

        Returns:
            lx: [batch, 2]
            lu: [batch, 1] or None
            lxx: [batch, 2, 2]
            luu: [batch, 1, 1] or None
            lxu: [batch, 2, 1] or None
        """
        batch = x.shape[0]
        error = x - self.x_ref.unsqueeze(0)  # Broadcast reference
        Q = self.Q_terminal if u is None else self.Q

        # First derivatives: lx = Q·(x - x_ref)
        lx = torch.matmul(Q.unsqueeze(0), error.unsqueeze(-1)).squeeze(-1)

        # Second derivatives (constant for quadratic cost)
        lxx = Q.unsqueeze(0).expand(batch, -1, -1).clone()

        if u is None:
            # Terminal cost
            return lx, None, lxx, None, None
        else:
            # Running cost: lu = R·u
            lu = torch.matmul(self.R.unsqueeze(0), u.unsqueeze(-1)).squeeze(-1)
            luu = self.R.unsqueeze(0).expand(batch, -1, -1).clone()
            lxu = torch.zeros(batch, self.nx, self.nu, device=x.device, dtype=x.dtype)
            return lx, lu, lxx, luu, lxu


def pendulum_problem(
    batch_size: int = 1,
    horizon: int = 50,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    **kwargs,
) -> tuple[PendulumDynamics, PendulumCost]:
    """Create a standard pendulum swing-up problem.

    Args:
        batch_size: Number of parallel environments
        horizon: MPC horizon length
        device: 'cpu' or 'cuda'
        dtype: torch.float32 or torch.float64
        **kwargs: Additional parameters passed to dynamics and cost

    Returns:
        dynamics: PendulumDynamics instance
        cost: PendulumCost instance
    """
    dynamics = PendulumDynamics(device=device, dtype=dtype, **kwargs)
    cost = PendulumCost(device=device, dtype=dtype, **kwargs)
    return dynamics, cost
