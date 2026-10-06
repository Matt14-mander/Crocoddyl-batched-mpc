"""Cached local standing MPC, hard inequality QP and exact nonlinear gate.

This is a fixed-working-point local controller, not general nonlinear RTI SQP.
Online derivatives/line searches are eliminated. No hardware deadline guarantee.
"""

import math
import warnings

import torch

from ..qp import solve_inequality_qp
from ..result import MPCResult, SolveStatus
from .go2_constraints import Go2ContactConstraints
from .go2_torch import Go2StandingCost, Go2TorchDynamics


def _mv(a, x):
    return (a @ x[..., None]).squeeze(-1)


class Go2ConstrainedMPC:
    """Return (action,result), action=NaN if no exactly feasible trajectory exists.

    Fixed flat-ground support, shared reference. reset(mask) clears QP warm
    controls. Bounds/settings are setup-only: rebuild after changing the model.
    Failure must stop/replan at the caller, never blindly hold the last torque.
    """

    def __init__(
        self,
        batch_size=1,
        horizon=2,
        *,
        dt=0.02,
        device="cpu",
        dtype=torch.float64,
        friction=0.6,
        minimum_normal_force=1.0,
        maximum_normal_force=200.0,
        torque_limits=None,
        qp_iterations=12,
        force_margin=0.2,
        compile_physics=True,
    ):
        if batch_size < 1 or horizon < 1 or qp_iterations < 1:
            raise ValueError("batch_size, horizon and qp_iterations must be positive")
        if (
            not math.isfinite(force_margin)
            or force_margin < 0
            or force_margin >= (maximum_normal_force - minimum_normal_force) / 2
        ):
            raise ValueError("force_margin must fit strictly inside the normal force interval")
        self.batch_size, self.horizon, self.qp_iterations = batch_size, horizon, qp_iterations
        self.dynamics = Go2TorchDynamics(dt=dt, device=device, dtype=dtype)
        self.cost = Go2StandingCost(self.dynamics)
        self.constraints = Go2ContactConstraints(
            self.dynamics,
            friction=friction,
            minimum_normal_force=minimum_normal_force,
            maximum_normal_force=maximum_normal_force,
            torque_limits=torque_limits,
            tolerance=1e-7 if dtype == torch.float64 else 2e-4,
        )
        self.device, self.dtype = self.dynamics.device, dtype
        self.reference = self.dynamics.standing_state.detach()
        self.u_ref = self.cost.u_reference
        self._warm = self.reference.new_zeros(batch_size, horizon, 12)
        self._valid = torch.zeros(batch_size, device=self.device, dtype=torch.bool)
        x, u = self.reference[None], self.u_ref[None]
        # Preparation is deliberately outside the measured control cycle.
        with torch.enable_grad():
            A, B = self.dynamics.calc_diff(x, u)
            _, _, Fx, Fu = self.dynamics.contact_derivatives(x, u)
        A, B, Fx, Fu = A[0].detach(), B[0].detach(), Fx[0].detach(), Fu[0].detach()
        force = self.dynamics.contact_dynamics(x, u).forces_world[0].flatten()
        n = horizon * 12
        eye = torch.eye(n, device=self.device, dtype=dtype)
        P = torch.eye(36, device=self.device, dtype=dtype)
        S = x.new_zeros(36, n)
        H = torch.kron(torch.eye(horizon, device=self.device, dtype=dtype), self.cost.R)
        linear = x.new_zeros(n, 36)
        C, c = self.constraints.force_matrix, self.constraints.force_bound
        matrices, bounds, state_offsets = [], [], []
        for t in range(horizon):
            select = eye[t * 12 : (t + 1) * 12]
            matrices.append(C @ (Fx @ S + Fu @ select))
            bounds.append(c - C @ force - force_margin)
            state_offsets.append(C @ Fx @ P)
            P = A @ P
            S = A @ S + B @ select
            weight = self.cost.Q_terminal if t == horizon - 1 else self.cost.Q
            H = H + S.T @ weight @ S
            linear = linear + S.T @ weight @ P
        Gforce = torch.cat(matrices)
        limits = self.constraints.torque_limits.repeat(horizon)
        self.G = torch.cat((Gforce, eye, -eye))
        self.bound = torch.cat(
            (
                torch.cat(bounds),
                limits - self.u_ref.repeat(horizon) - 1e-5,
                limits + self.u_ref.repeat(horizon) - 1e-5,
            )
        )
        self.state_offsets = torch.cat((torch.cat(state_offsets), x.new_zeros(2 * n, 36)))
        scaling = self.G.norm(dim=-1).clamp_min(1e-8)
        self.G = self.G / scaling[:, None]
        self.bound = self.bound / scaling
        self.state_offsets = self.state_offsets / scaling[:, None]
        self.H, self.linear = H, linear
        self._dual = self.bound.new_ones(batch_size, self.bound.shape[-1])

        def physics(state, torque):
            contact = self.dynamics.contact_dynamics(state, torque)
            a = contact.acceleration
            velocity = state[:, 19:] + dt * a
            output = self.dynamics.manifold.integrate(state, torch.cat((dt * velocity, dt * a), -1))
            return output, contact.forces_world

        if compile_physics:

            class Physics(torch.nn.Module):
                def forward(self, state, torque):
                    return physics(state, torque)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", torch.jit.TracerWarning)
                traced = torch.jit.trace(
                    Physics().eval(),
                    (x.expand(2 * batch_size, -1), u.expand(2 * batch_size, -1)),
                    check_trace=False,
                )
                self._physics = torch.jit.freeze(traced)
        else:
            self._physics = physics
        # Warm compiler execution plans before entering the control loop.
        with torch.no_grad():
            for _ in range(5):
                self._physics(
                    x.expand(2 * batch_size, -1).contiguous(),
                    u.expand(2 * batch_size, -1).contiguous(),
                )
        for _ in range(2):
            self.compute(self.reference.expand(batch_size, -1))

    @torch.no_grad()
    def compute(self, state):
        if (
            state.shape != (self.batch_size, 37)
            or state.device != self.device
            or state.dtype != self.dtype
        ):
            raise ValueError("state must match controller [batch,37], device and dtype")
        valid = self.dynamics.manifold.is_valid(state)
        observation_valid = valid.clone()
        safe = torch.where(valid[:, None], state, self.reference)
        error = self.dynamics.manifold.diff(safe, self.reference)
        # The cached local model is only valid near its preparation pose.
        valid &= (error[:, :18].abs().amax(-1) <= 0.35) & (error[:, 18:].abs().amax(-1) <= 2.0)
        gradient = error @ self.linear.T
        bound = self.bound - error @ self.state_offsets.T
        qp = solve_inequality_qp(
            self.H,
            gradient,
            self.G,
            bound,
            initial=self._warm.flatten(1),
            initial_dual=self._dual,
            iterations=self.qp_iterations,
            tolerance=self.constraints.tolerance,
        )
        optimized = qp.x.reshape(self.batch_size, self.horizon, 12)
        fallback = torch.where(self._valid[:, None, None], self._warm, 0.0)
        if self.device.type == "cpu":
            controls = optimized[None] + self.u_ref
            candidates, feasible, costs = self._validate(safe, controls, valid)
            usable = feasible[0] & qp.feasible
            selected = torch.zeros(self.batch_size, device=self.device, dtype=torch.int64)
            us, xs, value = controls[0], candidates[0], costs[0]
            # CPU can branch without a device synchronization. The old sequence
            # is rechecked only when the optimized candidate is not feasible.
            if not bool(usable.all()):
                backup = fallback[None] + self.u_ref
                bx, bf, bc = self._validate(safe, backup, valid)
                use_backup = ~usable & bf[0]
                us = torch.where(use_backup[:, None, None], backup[0], us)
                xs = torch.where(use_backup[:, None, None], bx[0], xs)
                value = torch.where(use_backup, bc[0], value)
                selected = use_backup.to(torch.int64)
                usable |= use_backup
        else:
            controls = torch.stack((optimized, fallback)) + self.u_ref
            candidates, feasible, costs = self._validate(safe, controls, valid)
            feasible[0] &= qp.feasible
            selected = torch.where(feasible, costs, float("inf")).argmin(0)
            index = torch.arange(self.batch_size, device=self.device)
            usable = feasible.any(0)
            us, xs, value = (
                controls[selected, index],
                candidates[selected, index],
                costs[selected, index],
            )
        shifted = torch.cat((us[:, 1:], us[:, -1:]), 1) - self.u_ref
        self._warm.copy_(torch.where(usable[:, None, None], shifted, torch.zeros_like(shifted)))
        self._valid.copy_(usable)
        self._dual.copy_(
            torch.where((usable & qp.feasible)[:, None], qp.dual, torch.ones_like(qp.dual))
        )
        success = qp.converged & (selected == 0)
        failure = torch.where(
            ~observation_valid,
            SolveStatus.NUMERICAL_FAILURE,
            torch.where(~valid, SolveStatus.OUTSIDE_LOCAL_MODEL, SolveStatus.NO_FEASIBLE_CANDIDATE),
        )
        status = torch.where(
            ~usable, failure, torch.where(success, SolveStatus.SUCCESS, SolveStatus.MAX_ITERATIONS)
        )
        result = MPCResult(
            torch.where(usable[:, None, None], xs, float("nan")),
            torch.where(usable[:, None, None], us, float("nan")),
            torch.where(usable, value, float("nan")),
            status,
            qp.iterations,
            usable,
        )
        action = torch.where(usable[:, None], us[:, 0], float("nan"))
        return action, result

    def _validate(self, safe, controls, valid):
        count = controls.shape[0]
        states = safe[None].expand(count, -1, -1)
        trajectory = [states]
        feasible = valid[None].expand(count, -1).clone()
        cost = safe.new_zeros(count, self.batch_size)
        for t in range(self.horizon):
            action = controls[:, :, t]
            output, force = self._physics(states.reshape(-1, 37), action.reshape(-1, 12))
            forces = force.reshape(count, self.batch_size, len(self.dynamics.feet), 3)
            feasible &= self.constraints.feasible(action, forces)
            cost += self.cost.calc(states.reshape(-1, 37), action.reshape(-1, 12)).reshape(
                count, -1
            )
            states = output.reshape(count, self.batch_size, 37)
            feasible &= self.dynamics.manifold.is_valid(states)
            trajectory.append(states)
        cost += self.cost.calc(states.reshape(-1, 37)).reshape(count, -1)
        return torch.stack(trajectory, 2), feasible & torch.isfinite(cost), cost

    @torch.no_grad()
    def reset(self, mask=None):
        if mask is None:
            self._warm.zero_()
            self._valid.zero_()
            self._dual.fill_(1.0)
            return
        if (
            mask.shape != (self.batch_size,)
            or mask.device != self.device
            or mask.dtype != torch.bool
        ):
            raise ValueError("reset mask must be bool [batch] on controller device")
        self._warm.masked_fill_(mask[:, None, None], 0.0)
        self._valid.masked_fill_(mask, False)
        self._dual.masked_fill_(mask[:, None], 1.0)
