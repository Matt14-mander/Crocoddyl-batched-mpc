"""Pendulum swing-up: single inverted pendulum with torque control.

State: x = [θ, θ̇] where θ is angle from downward vertical (θ=0 is down, θ=π is up)
Control: u = [τ] torque
Dynamics: θ̈ = (g/l)·sin(θ) - (b/(m·l²))·θ̇ + τ/(m·l²)

Default parameters correspond to a standard underactuated pendulum.
"""

from dataclasses import dataclass, fields
from typing import Any

import torch
from torch import Tensor


def _batched_parameter(
    value: Any,
    *,
    batch_size: int,
    tail_shape: tuple[int, ...],
    device: torch.device | str,
    dtype: torch.dtype,
    name: str,
) -> Tensor:
    tensor = torch.as_tensor(value, device=device, dtype=dtype)
    if tuple(tensor.shape) == tail_shape:
        return tensor.expand(batch_size, *tail_shape).clone()
    expected = (batch_size, *tail_shape)
    if tuple(tensor.shape) != expected:
        raise ValueError(f"{name} must have shape {tail_shape} or {expected}")
    return tensor.clone()


def _validate_parameter_fields(instance: Any, shapes: dict[str, tuple[int, ...]]) -> None:
    values = [getattr(instance, field.name) for field in fields(instance)]
    first = values[0]
    if not isinstance(first, Tensor) or first.ndim < 1 or first.shape[0] < 1:
        raise ValueError("batched parameters must have a positive leading batch dimension")
    if first.dtype not in (torch.float32, torch.float64):
        raise ValueError("batched parameters require float32 or float64")
    batch_size = first.shape[0]
    for name, tail_shape in shapes.items():
        value = getattr(instance, name)
        expected = (batch_size, *tail_shape)
        if not isinstance(value, Tensor) or tuple(value.shape) != expected:
            raise ValueError(f"{name} must have shape {expected}")
        if value.device != first.device or value.dtype != first.dtype:
            raise ValueError("all batched parameters must share device and dtype")


def _masked_parameter_update(instance: Any, mask: Tensor, updates: dict[str, Any]) -> None:
    if not isinstance(mask, Tensor) or mask.dtype != torch.bool:
        raise TypeError("parameter update mask must be a bool tensor")
    if tuple(mask.shape) != (instance.batch_size,) or mask.device != instance.device:
        raise ValueError("parameter update mask must have shape [batch] on the parameter device")
    unknown = set(updates).difference(field.name for field in fields(instance))
    if unknown:
        raise ValueError(f"unknown parameters: {', '.join(sorted(unknown))}")

    prepared: list[tuple[Tensor, Tensor]] = []
    for name, value in updates.items():
        target = getattr(instance, name)
        if isinstance(value, Tensor):
            if value.device != target.device or value.dtype != target.dtype:
                raise ValueError(
                    f"{name} update must have device={target.device}, dtype={target.dtype}"
                )
            source = value
        else:
            source = target.new_tensor(value)
        candidate = target.clone()
        try:
            if tuple(source.shape) == tuple(target.shape):
                candidate[mask] = source[mask]
            else:
                candidate[mask] = source
        except RuntimeError as exc:
            raise ValueError(f"{name} update has incompatible shape {tuple(source.shape)}") from exc
        prepared.append((target, candidate))
    for target, candidate in prepared:
        target.copy_(candidate)


@dataclass(frozen=True)
class PendulumDynamicsParameters:
    """Per-environment physical parameters, each stored as a float tensor `[B]`."""

    mass: Tensor
    length: Tensor
    damping: Tensor
    gravity: Tensor

    def __post_init__(self) -> None:
        _validate_parameter_fields(self, {"mass": (), "length": (), "damping": (), "gravity": ()})

    @classmethod
    def create(
        cls,
        batch_size: int,
        *,
        mass: Any = 1.0,
        length: Any = 1.0,
        damping: Any = 0.1,
        gravity: Any = 9.81,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> "PendulumDynamicsParameters":
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        values = {
            name: _batched_parameter(
                value,
                batch_size=batch_size,
                tail_shape=(),
                device=device,
                dtype=dtype,
                name=name,
            )
            for name, value in {
                "mass": mass,
                "length": length,
                "damping": damping,
                "gravity": gravity,
            }.items()
        }
        return cls(**values)

    @property
    def batch_size(self) -> int:
        return self.mass.shape[0]

    @property
    def device(self) -> torch.device:
        return self.mass.device

    @property
    def dtype(self) -> torch.dtype:
        return self.mass.dtype

    def update_(self, mask: Tensor, **updates: Any) -> None:
        _masked_parameter_update(self, mask, updates)


@dataclass(frozen=True)
class PendulumCostParameters:
    """Per-environment references and weights with a leading batch dimension."""

    x_ref: Tensor
    Q: Tensor
    R: Tensor
    Q_terminal: Tensor

    def __post_init__(self) -> None:
        _validate_parameter_fields(
            self, {"x_ref": (2,), "Q": (2, 2), "R": (1, 1), "Q_terminal": (2, 2)}
        )

    @classmethod
    def create(
        cls,
        batch_size: int,
        *,
        x_ref: Any | None = None,
        Q: Any | None = None,
        R: Any | None = None,
        Q_terminal: Any | None = None,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> "PendulumCostParameters":
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        defaults = {
            "x_ref": [torch.pi, 0.0] if x_ref is None else x_ref,
            "Q": [[10.0, 0.0], [0.0, 1.0]] if Q is None else Q,
            "R": [[0.1]] if R is None else R,
            "Q_terminal": [[100.0, 0.0], [0.0, 10.0]] if Q_terminal is None else Q_terminal,
        }
        shapes = {"x_ref": (2,), "Q": (2, 2), "R": (1, 1), "Q_terminal": (2, 2)}
        values = {
            name: _batched_parameter(
                value,
                batch_size=batch_size,
                tail_shape=shapes[name],
                device=device,
                dtype=dtype,
                name=name,
            )
            for name, value in defaults.items()
        }
        return cls(**values)

    @property
    def batch_size(self) -> int:
        return self.x_ref.shape[0]

    @property
    def device(self) -> torch.device:
        return self.x_ref.device

    @property
    def dtype(self) -> torch.dtype:
        return self.x_ref.dtype

    def update_(self, mask: Tensor, **updates: Any) -> None:
        _masked_parameter_update(self, mask, updates)


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
        parameters: PendulumDynamicsParameters | None = None,
    ) -> None:
        self.parameters = parameters
        if parameters is not None:
            device, dtype = parameters.device, parameters.dtype
            self.m, self.l = parameters.mass, parameters.length
            self.b, self.g = parameters.damping, parameters.gravity
        else:
            self.m, self.l, self.b, self.g = mass, length, damping, gravity
        self._dt = dt
        self.device = torch.device(device)
        self.dtype = dtype

        if parameters is None:
            self.inertia = mass * length**2
            self.gravity_coeff = gravity / length
            self.damping_coeff = damping / self.inertia
            self.control_coeff = 1.0 / self.inertia

    @property
    def parameter_batch_size(self) -> int | None:
        return None if self.parameters is None else self.parameters.batch_size

    def update_parameters(self, mask: Tensor, **updates: Any) -> None:
        if self.parameters is None:
            raise RuntimeError(
                "Create PendulumDynamics with batched parameters before masked updates"
            )
        self.parameters.update_(mask, **updates)

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
        if self.parameters is None:
            gravity_coeff = self.gravity_coeff
            damping_coeff = self.damping_coeff
            control_coeff = self.control_coeff
        else:
            if x.shape[0] != self.parameters.batch_size:
                raise ValueError("state batch does not match dynamics parameter batch")
            inertia = (self.m * self.l.square())[:, None]
            gravity_coeff = (self.g / self.l)[:, None]
            damping_coeff = self.b[:, None] / inertia
            control_coeff = inertia.reciprocal()
        theta_ddot = (
            gravity_coeff * torch.sin(theta) - damping_coeff * theta_dot + control_coeff * tau
        )

        # Semi-implicit Euler: update velocity first, then position
        theta_dot_next = theta_dot + self.dt * theta_ddot
        theta_next = theta + self.dt * theta_dot_next

        return torch.cat([theta_next, theta_dot_next], dim=-1)

    def calc_diff(self, x: Tensor, u: Tensor) -> tuple[Tensor, Tensor]:
        """Compute analytical dynamics Jacobians.

        Args:
            x: [batch, 2]
            u: [batch, 1]

        Returns:
            Fx: [batch, 2, 2] - ∂f/∂x
            Fu: [batch, 2, 1] - ∂f/∂u
        """
        batch = x.shape[0]
        if self.parameters is None:
            gravity_coeff = x.new_full((batch,), self.gravity_coeff)
            damping_coeff = x.new_full((batch,), self.damping_coeff)
            control_coeff = x.new_full((batch,), self.control_coeff)
        else:
            if batch != self.parameters.batch_size:
                raise ValueError("state batch does not match dynamics parameter batch")
            inertia = self.m * self.l.square()
            gravity_coeff = self.g / self.l
            damping_coeff = self.b / inertia
            control_coeff = inertia.reciprocal()
        velocity_theta = self.dt * gravity_coeff * torch.cos(x[:, 0])
        velocity_velocity = 1.0 - self.dt * damping_coeff
        Fx = x.new_zeros(batch, 2, 2)
        Fx[:, 0, 0] = 1.0 + self.dt * velocity_theta
        Fx[:, 0, 1] = self.dt * velocity_velocity
        Fx[:, 1, 0] = velocity_theta
        Fx[:, 1, 1] = velocity_velocity
        Fu = x.new_empty(batch, 2, 1)
        Fu[:, 0, 0] = self.dt**2 * control_coeff
        Fu[:, 1, 0] = self.dt * control_coeff
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
        parameters: PendulumCostParameters | None = None,
    ) -> None:
        self.parameters = parameters
        if parameters is not None:
            if any(value is not None for value in (x_ref, Q, R, Q_terminal)):
                raise ValueError("Pass either batched cost parameters or individual tensors")
            self.device, self.dtype = parameters.device, parameters.dtype
            self.x_ref, self.Q = parameters.x_ref, parameters.Q
            self.R, self.Q_terminal = parameters.R, parameters.Q_terminal
            return
        self.device = torch.device(device)
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
    def parameter_batch_size(self) -> int | None:
        return None if self.parameters is None else self.parameters.batch_size

    def update_parameters(self, mask: Tensor, **updates: Any) -> None:
        if self.parameters is None:
            raise RuntimeError("Create PendulumCost with batched parameters before masked updates")
        self.parameters.update_(mask, **updates)

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
        if self.parameters is not None and x.shape[0] != self.parameters.batch_size:
            raise ValueError("state batch does not match cost parameter batch")
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
        if self.parameters is not None and batch != self.parameters.batch_size:
            raise ValueError("state batch does not match cost parameter batch")
        error = x - self.x_ref
        Q = self.Q_terminal if u is None else self.Q

        # First derivatives: lx = Q·(x - x_ref)
        lx = torch.matmul(Q, error.unsqueeze(-1)).squeeze(-1)

        # Second derivatives (constant for quadratic cost)
        lxx = Q.expand(batch, -1, -1) if Q.ndim == 2 else Q

        if u is None:
            # Terminal cost
            return lx, None, lxx, None, None
        else:
            # Running cost: lu = R·u
            lu = torch.matmul(self.R, u.unsqueeze(-1)).squeeze(-1)
            luu = self.R.expand(batch, -1, -1) if self.R.ndim == 2 else self.R
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
