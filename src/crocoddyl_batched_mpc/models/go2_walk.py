"""Flat-ground quasi-static crawl references and online local tracking MPC."""

import math
import warnings
from dataclasses import dataclass

import torch

from ..lie import quaternion_matrix, skew
from ..qp import solve_inequality_qp
from ..result import MPCResult, SolveStatus
from .go2 import GO2_LEGS
from .go2_constraints import Go2ContactConstraints
from .go2_hybrid import Go2HybridDynamics
from .go2_torch import _mv


def smoothstep(s):
    return s**3 * (10 - 15 * s + 6 * s * s)


def go2_center_of_mass(model, q):
    """URDF mass-weighted world COM, including collapsed rotor/foot inertias."""
    leading = q.shape[:-1]
    R0 = quaternion_matrix(q[..., 3:7])
    rotations, positions = [R0[..., None, :, :]], [q[..., :3][..., None, :]]
    rotation = R0[..., None, :, :].expand(*leading, 4, 3, 3)
    position = q[..., :3][..., None, :].expand(*leading, 4, 3)
    for indices in model.depth_indices:
        A = skew(model.axes[indices])
        angle = q[..., 7 + indices, None, None]
        relative = model.rotations[indices] @ (
            model.eye3 + angle.sin() * A + (1 - angle.cos()) * (A @ A)
        )
        position = position + _mv(rotation, model.translations[indices])
        rotation = rotation @ relative
        rotations.append(rotation)
        positions.append(position)
    rotations, positions = torch.cat(rotations, -3), torch.cat(positions, -2)
    inertia = model.parallel_inertias
    masses = inertia[:, 0, 0]
    com_local = (
        torch.stack((inertia[:, 5, 1], inertia[:, 3, 2], inertia[:, 4, 0]), -1) / masses[:, None]
    )
    com_world = positions + _mv(rotations, com_local)
    return (com_world * masses[:, None]).sum(-2) / masses.sum()


def go2_triangle_margin(model, state, mask, anchors):
    """Signed COM-to-edge distances for a three-foot support triangle."""
    indices = torch.where(mask, torch.arange(4, device=mask.device), 4).sort(-1).values[:, :3]
    triangle = anchors.gather(1, indices.clamp_max(3)[..., None].expand(-1, -1, 3))[..., :2]
    edges = triangle.roll(-1, 1) - triangle

    def cross(a, b):
        return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]

    orientation = cross(edges[:, 0], edges[:, 1]).sign()
    com = go2_center_of_mass(model, state[:, :19])[:, None, :2]
    distance = (
        cross(edges, com - triangle) * orientation[:, None] / edges.norm(dim=-1).clamp_min(1e-12)
    )
    return distance.amin(-1)


def go2_leg_ik(model, base_configurations, foot_positions, *, iterations=12):
    """Offline batched Newton IK with fixed base; validates reachability and limits."""
    q = base_configurations.clone()
    for _ in range(iterations):
        _, _, _, position, J, _ = model._kinematics(q, q.new_zeros(q.shape[0], 18))
        J = J.reshape(-1, 4, 3, 18)
        leg_J = torch.stack([J[:, f, :, 6 + 3 * f : 9 + 3 * f] for f in range(4)], 1)
        update, info = torch.linalg.solve_ex(
            leg_J, (foot_positions - position)[..., None], check_errors=False
        )
        if (info != 0).any():
            raise ValueError("singular leg IK")
        q[:, 7:] += update.squeeze(-1).clamp(-0.2, 0.2).flatten(1)
    if (model.foot_positions(q) - foot_positions).abs().max() > 1e-8:
        raise ValueError("unreachable walk foot target")
    if ((q[:, 7:] < model.lower_limits) | (q[:, 7:] > model.upper_limits)).any():
        raise ValueError("walk IK exceeds URDF joint limits")
    return q


@dataclass
class Go2WalkPlan:
    reference: torch.Tensor
    acceleration: torch.Tensor
    contact_masks: torch.Tensor
    anchors: torch.Tensor
    foot_reference: torch.Tensor
    phase_names: tuple[str, ...]
    dt: float
    cycles: int
    step_length: float
    swing_height: float
    order: tuple[str, ...]

    @property
    def duration(self):
        return (len(self.reference) - 1) * self.dt


def go2_slow_walk_plan(
    *,
    cycles=2,
    step_length=0.02,
    swing_height=0.015,
    shift_time=1.0,
    swing_time=1.0,
    settle_time=0.4,
    dt=0.02,
    order=("RL", "FL", "RR", "FR"),
    device="cpu",
    dtype=torch.float64,
):
    """Finite straight crawl: one swinging leg, COM at support-triangle centroid.

    Quintic transfers/swing endpoints, fixed stance anchors, URDF-constrained
    leg IK. References are prepared outside the control loop, never applied
    directly to a plant. Repeated cycles translate all four feet forward.
    """
    if dtype != torch.float64:
        raise ValueError("slow walk requires float64 for strict contact checks")
    if not isinstance(cycles, int) or isinstance(cycles, bool) or cycles < 1:
        raise ValueError("cycles must be a positive integer")
    if tuple(sorted(order)) != tuple(sorted(GO2_LEGS)):
        raise ValueError("order must contain each Go2 foot exactly once")
    values = (step_length, swing_height, shift_time, swing_time, settle_time, dt)
    if not all(math.isfinite(v) and v > 0 for v in values):
        raise ValueError("walk distances and times must be finite and positive")
    if (
        step_length > 0.04
        or swing_height > 0.04
        or min(shift_time, swing_time) < 0.4
        or settle_time < 0.2
    ):
        raise ValueError("slow walk requires <=4 cm steps/height and conservative phase durations")
    model = Go2HybridDynamics(dt=dt, device=device, dtype=dtype)
    q0 = model.reference_configuration
    anchors = model.contact_positions.clone()
    body = q0[:3].clone()
    body_samples, feet_samples, mask_samples, anchor_samples, names = [], [], [], [], []

    def append(body_position, feet, mask, name):
        body_samples.append(body_position.clone())
        feet_samples.append(feet.clone())
        mask_samples.append(mask.clone())
        anchor_samples.append(anchors.clone())
        names.append(name)

    stance = torch.ones(4, dtype=torch.bool, device=model.device)
    append(body, anchors, stance, "start")
    for _ in range(cycles):
        for leg in order:
            foot = GO2_LEGS.index(leg)
            mask = stance.clone()
            mask[foot] = False
            centroid = anchors[mask, :2].mean(0)
            target = body.clone()
            target[:2] = centroid
            for _ in range(3):
                base = q0[None].clone()
                base[0, :3] = target
                pose = go2_leg_ik(model, base, anchors[None])
                target[:2] += centroid - go2_center_of_mass(model, pose)[0, :2]
            shift_ticks = math.ceil(shift_time / dt)
            for tick in range(1, shift_ticks + 1):
                append(
                    body + smoothstep(tick / shift_ticks) * (target - body),
                    anchors,
                    stance,
                    f"shift_{leg}",
                )
            body = target
            source = anchors[foot].clone()
            destination = source.clone()
            destination[0] += step_length
            swing_ticks = math.ceil(swing_time / dt)
            for tick in range(swing_ticks):
                s = tick / swing_ticks
                feet = anchors.clone()
                feet[foot] = source + smoothstep(s) * (destination - source)
                feet[foot, 2] += 64 * swing_height * s**3 * (1 - s) ** 3
                append(body, feet, mask, f"swing_{leg}")
            anchors[foot] = destination
            for _ in range(math.ceil(settle_time / dt)):
                append(body, anchors, stance, f"land_{leg}")
        target = body.clone()
        target[:2] = anchors[:, :2].mean(0)
        for tick in range(1, math.ceil(shift_time / dt) + 1):
            append(
                body + smoothstep(tick / math.ceil(shift_time / dt)) * (target - body),
                anchors,
                stance,
                "center",
            )
        body = target
    for _ in range(math.ceil(settle_time / dt)):
        append(body, anchors, stance, "finish")
    feet = torch.stack(feet_samples)
    bases = q0.expand(len(feet), -1).clone()
    bases[:, :3] = torch.stack(body_samples)
    q = go2_leg_ik(model, bases, feet)
    velocity = q.new_zeros(len(q), 18)
    velocity[1:-1, :3] = (q[2:, :3] - q[:-2, :3]) / (2 * dt)
    velocity[1:-1, 6:] = (q[2:, 7:] - q[:-2, 7:]) / (2 * dt)
    acceleration = torch.zeros_like(velocity)
    acceleration[1:-1] = (velocity[2:] - velocity[:-2]) / (2 * dt)
    return Go2WalkPlan(
        torch.cat((q, velocity), -1),
        acceleration,
        torch.stack(mask_samples),
        torch.stack(anchor_samples),
        feet,
        tuple(names),
        dt,
        cycles,
        step_length,
        swing_height,
        tuple(order),
    )


class Go2WalkMPC:
    """Online frozen-rigid-body, constant-torque local receding-horizon QP.

    Recompute exact control-affine acceleration/force maps at every measurement.
    Condense short local kinematic predictions; PD acceleration tracking is an
    auxiliary stage objective. All returned horizon nodes pass a full nonlinear
    force/torque/ground check. This is not a full nonlinear contact-sequence SQP.
    """

    def __init__(
        self,
        batch_size=1,
        *,
        horizon=2,
        dt=0.02,
        device="cpu",
        dtype=torch.float64,
        friction=0.6,
        minimum_normal_force=0.5,
        maximum_normal_force=200.0,
        force_margin=2.0,
        qp_iterations=16,
        compile_physics=True,
    ):
        if batch_size < 1 or horizon < 1 or qp_iterations < 1:
            raise ValueError("batch, horizon and QP budget must be positive")
        if dtype != torch.float64:
            raise ValueError("initial slow walk MPC requires float64")
        self.model = Go2HybridDynamics(dt=dt, device=device, dtype=dtype)
        self.constraints = Go2ContactConstraints(
            self.model,
            friction=friction,
            minimum_normal_force=minimum_normal_force,
            maximum_normal_force=maximum_normal_force,
        )
        self.batch_size, self.horizon, self.dt = batch_size, horizon, dt
        self.device, self.dtype = self.model.device, dtype
        self.force_margin, self.qp_iterations = force_margin, qp_iterations
        if (
            not math.isfinite(force_margin)
            or not 0 <= force_margin < (maximum_normal_force - minimum_normal_force) / 2
        ):
            raise ValueError("invalid force margin")
        self._warm = self.model.quasi_static_torques().expand(batch_size, -1).clone()
        self._dual = self._warm.new_ones(batch_size, 48)
        self._valid = torch.zeros(batch_size, device=self.device, dtype=torch.bool)
        self.acceleration_weight = self._warm.new_tensor([10.0] * 6 + [1.0] * 12)
        self.state_weight = self._warm.new_tensor(
            [500.0] * 6 + [100.0] * 12 + [10.0] * 6 + [1.0] * 12
        )
        self.eye = torch.eye(12, device=self.device, dtype=dtype)
        self._maps = self.model.control_affine_dynamics

        def physics(x, u, mask, anchors):
            result = self.model.contact_dynamics(x, u, contact_mask=mask, contact_positions=anchors)
            a = result.acceleration
            velocity = x[:, 19:] + dt * a
            output = self.model.manifold.integrate(x, torch.cat((dt * velocity, dt * a), -1))
            return output, result.forces_world, self.model.foot_positions(output[:, :19])

        self._physics = physics
        if compile_physics:

            class Maps(torch.nn.Module):
                def forward(_, x, mask, anchors):
                    return self.model.control_affine_dynamics(
                        x, contact_mask=mask, contact_positions=anchors
                    )

            class Physics(torch.nn.Module):
                def forward(_, x, u, mask, anchors):
                    return physics(x, u, mask, anchors)

            x = self.model.standing_state.expand(batch_size, -1).contiguous()
            mask = torch.ones(batch_size, 4, device=self.device, dtype=torch.bool)
            anchors = self.model.contact_positions.expand(batch_size, -1, -1).contiguous()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", torch.jit.TracerWarning)
                traced_maps = torch.jit.freeze(
                    torch.jit.trace(Maps().eval(), (x, mask, anchors), check_trace=False)
                )
                self._maps = lambda x, **kwargs: traced_maps(
                    x, kwargs["contact_mask"], kwargs["contact_positions"]
                )
                self._physics = torch.jit.freeze(
                    torch.jit.trace(
                        Physics().eval(), (x, self._warm, mask, anchors), check_trace=False
                    )
                )
            with torch.no_grad():
                for _ in range(5):
                    self._maps(x, contact_mask=mask, contact_positions=anchors)
                    self._physics(x, self._warm, mask, anchors)

        # Warm the full QP/validation path, not just the compiled physics kernels.
        x = self.model.standing_state.expand(batch_size, -1).contiguous()
        mask = torch.ones(batch_size, 4, device=self.device, dtype=torch.bool)
        anchors = self.model.contact_positions.expand(batch_size, -1, -1).contiguous()
        for _ in range(3):
            self.compute(
                x,
                x[:, None].expand(-1, horizon + 1, -1),
                x.new_zeros(batch_size, 18),
                mask,
                anchors,
            )

    @torch.no_grad()
    def compute(self, state, reference, reference_acceleration, contact_mask, anchors):
        B, T = self.batch_size, self.horizon
        if (
            state.shape != (B, 37)
            or reference.shape != (B, T + 1, 37)
            or reference_acceleration.shape != (B, 18)
        ):
            raise ValueError(
                "walk MPC expects state[B,37], reference[B,T+1,37], acceleration[B,18]"
            )
        if (
            contact_mask.shape != (B, 4)
            or contact_mask.dtype != torch.bool
            or contact_mask.device != self.device
            or anchors.shape != (B, 4, 3)
        ):
            raise ValueError("walk MPC expects bool contact_mask[B,4] and anchors[B,4,3]")
        for value in (state, reference, reference_acceleration, anchors):
            if value.device != self.device or value.dtype != self.dtype:
                raise ValueError("walk tensors must match MPC device/dtype")
        valid = self.model.manifold.is_valid(state) & self.model.manifold.is_valid(reference).all(
            -1
        )
        valid &= (
            (
                (reference[..., 7:19] >= self.model.lower_limits)
                & (reference[..., 7:19] <= self.model.upper_limits)
            )
            .flatten(1)
            .all(-1)
        )
        valid &= torch.isfinite(anchors).flatten(1).all(-1) & torch.isfinite(
            reference_acceleration
        ).all(-1)
        safe = torch.where(valid[:, None], state, self.model.standing_state)
        target_delta = self.model.manifold.diff(reference[:, 0], safe)[:, :18]
        a_target = (
            reference_acceleration + 200 * target_delta + 30 * (reference[:, 0, 19:] - safe[:, 19:])
        )
        a0, Au, f0, Fu = self._maps(safe, contact_mask=contact_mask, contact_positions=anchors)
        weighted = Au * self.acceleration_weight[:, None]
        H = T * (Au.transpose(-1, -2) @ weighted) + 1e-5 * self.eye
        gradient = T * _mv(Au.transpose(-1, -2), self.acceleration_weight * (a0 - a_target))
        for t in range(1, T + 1):
            position_coefficient = 0.5 * self.dt**2 * t * (t + 1)
            velocity_coefficient = self.dt * t
            mapping = torch.cat((position_coefficient * Au, velocity_coefficient * Au), -2)
            offset = torch.cat(
                (
                    self.dt * t * safe[:, 19:] + position_coefficient * a0,
                    safe[:, 19:] + velocity_coefficient * a0,
                ),
                -1,
            )
            goal = self.model.manifold.diff(reference[:, t], safe)
            goal = torch.cat((goal[:, :18], reference[:, t, 19:]), -1)
            H += mapping.transpose(-1, -2) @ (self.state_weight[:, None] * mapping)
            gradient += _mv(mapping.transpose(-1, -2), self.state_weight * (offset - goal))
        C, cb = self.constraints.force_matrix, self.constraints.force_bound
        force_rows = contact_mask.repeat_interleave(6, -1)
        force_G = C @ Fu
        force_b = cb - _mv(C, f0) - self.force_margin
        force_G = torch.where(force_rows[..., None], force_G, 0.0)
        force_b = torch.where(force_rows, force_b, 1.0)
        G = torch.cat((force_G, self.eye.expand(B, -1, -1), -self.eye.expand(B, -1, -1)), -2)
        bounds = torch.cat(
            (
                force_b,
                (self.constraints.torque_limits - 1e-5).expand(B, -1),
                (self.constraints.torque_limits - 1e-5).expand(B, -1),
            ),
            -1,
        )
        scaling = G.norm(dim=-1).clamp_min(1.0)
        qp = solve_inequality_qp(
            H,
            gradient,
            G / scaling[..., None],
            bounds / scaling,
            initial=self._warm,
            initial_dual=self._dual,
            iterations=self.qp_iterations,
            tolerance=1e-7,
        )
        candidates = [qp.x, self._warm]
        action = state.new_full((B, 12), float("nan"))
        xs = state.new_full((B, T + 1, 37), float("nan"))
        cost = state.new_full((B,), float("nan"))
        usable = torch.zeros(B, device=self.device, dtype=torch.bool)
        selected = torch.ones(B, device=self.device, dtype=torch.int64)
        for index, candidate in enumerate(candidates):
            if index == 1 and self.device.type == "cpu" and bool(usable.all()):
                break
            trajectory, feasible, value = self._validate(
                safe, candidate, contact_mask, anchors, reference
            )
            feasible &= valid & (qp.feasible if index == 0 else self._valid)
            choose = ~usable & feasible
            action = torch.where(choose[:, None], candidate, action)
            xs = torch.where(choose[:, None, None], trajectory, xs)
            cost = torch.where(choose, value, cost)
            selected = torch.where(choose, index, selected)
            usable |= feasible
        self._warm.copy_(torch.where(usable[:, None], action, torch.zeros_like(action)))
        self._valid.copy_(usable)
        self._dual.copy_(torch.where((usable & (selected == 0))[:, None], qp.dual, 1.0))
        status = torch.where(
            ~usable,
            torch.where(valid, SolveStatus.NO_FEASIBLE_CANDIDATE, SolveStatus.NUMERICAL_FAILURE),
            torch.where(
                qp.converged & (selected == 0), SolveStatus.SUCCESS, SolveStatus.MAX_ITERATIONS
            ),
        )
        us = action[:, None].expand(-1, T, -1).clone()
        return action, MPCResult(xs, us, cost, status, qp.iterations, usable)

    def _validate(self, state, action, mask, anchors, reference):
        values = [state]
        valid = torch.isfinite(action).all(-1)
        valid &= (
            (state[:, 7:19] >= self.model.lower_limits)
            & (state[:, 7:19] <= self.model.upper_limits)
        ).all(-1)
        valid &= (self.model.foot_positions(state[:, :19])[..., 2] >= -0.002).all(-1)
        bound = self.constraints.force_bound.expand(self.batch_size, -1) * mask.repeat_interleave(
            6, -1
        )
        cost = state.new_zeros(self.batch_size)
        for t in range(self.horizon):
            state, force, positions = self._physics(state, action, mask, anchors)
            violation = _mv(self.constraints.force_matrix, force.flatten(1)) - bound
            valid &= (violation.amax(-1) <= 1e-7) & (
                action.abs() <= self.constraints.torque_limits + 1e-7
            ).all(-1)
            valid &= self.model.manifold.is_valid(state) & torch.isfinite(force).flatten(1).all(-1)
            valid &= (
                (state[:, 7:19] >= self.model.lower_limits)
                & (state[:, 7:19] <= self.model.upper_limits)
            ).all(-1)
            valid &= (positions[..., 2] >= -0.002).all(-1)
            values.append(state)
            error = self.model.manifold.diff(state, reference[:, t + 1])
            cost += (error**2 * self.state_weight).sum(-1)
        return torch.stack(values, 1), valid, cost

    def reset(self, mask=None):
        if mask is None:
            self._warm.zero_()
            self._dual.fill_(1.0)
            self._valid.zero_()
        else:
            if (
                mask.shape != (self.batch_size,)
                or mask.device != self.device
                or mask.dtype != torch.bool
            ):
                raise ValueError("reset mask must be bool [batch] on MPC device")
            self._warm.masked_fill_(mask[:, None], 0.0)
            self._dual.masked_fill_(mask[:, None], 1.0)
            self._valid.masked_fill_(mask, False)


@dataclass
class Go2WalkControl:
    state: torch.Tensor  # post-impact state; apply to the simulation plant
    action: torch.Tensor
    result: MPCResult
    contact_mask: torch.Tensor
    anchors: torch.Tensor
    frame_indices: torch.Tensor
    event_accepted: torch.Tensor
    waiting_for_contact: torch.Tensor
    finished: torch.Tensor


class Go2SlowWalkController:
    """Execute a prepared crawl, pausing phase advancement for rejected contacts.

    No reference state is copied into the measured plant. Only a validated
    impact may change measured velocity; all other motion comes from torques.
    Each environment has its own clock, contacts, anchors and reset state.
    """

    def __init__(self, plan, batch_size=1, *, max_contact_wait=2.0, **settings):
        if not isinstance(plan, Go2WalkPlan):
            raise ValueError("a prepared Go2WalkPlan is required")
        if not math.isfinite(max_contact_wait) or max_contact_wait <= 0:
            raise ValueError("contact wait timeout must be finite and positive")
        self.plan = plan
        self.mpc = Go2WalkMPC(
            batch_size,
            dt=plan.dt,
            device=plan.reference.device,
            dtype=plan.reference.dtype,
            **settings,
        )
        self.batch_size = batch_size
        self.device, self.dtype = self.mpc.device, self.mpc.dtype
        self.indices = torch.zeros(batch_size, device=self.device, dtype=torch.int64)
        self.contact_mask = plan.contact_masks[0].expand(batch_size, -1).clone()
        self.anchors = plan.anchors[0].expand(batch_size, -1, -1).clone()
        self.wait_counts = torch.zeros_like(self.indices)
        self.failed = torch.zeros(batch_size, device=self.device, dtype=torch.bool)
        self.max_wait_ticks = math.ceil(max_contact_wait / plan.dt)

    def _references(self, waiting):
        frames = (
            self.indices[:, None] + torch.arange(self.mpc.horizon + 1, device=self.device)
        ).clamp_max(len(self.plan.reference) - 1)
        # The local prediction does not pretend an uncommitted mode has occurred.
        same_mode = (
            self.plan.contact_masks[frames] == self.plan.contact_masks[self.indices, None]
        ).all(-1)
        frames = torch.where(same_mode & ~waiting[:, None], frames, self.indices[:, None])
        reference = self.plan.reference[frames].clone()
        reference[:, :, 19:] = torch.where(waiting[:, None, None], 0.0, reference[:, :, 19:])
        acceleration = torch.where(waiting[:, None], 0.0, self.plan.acceleration[self.indices])
        return reference, acceleration

    @torch.no_grad()
    def compute(self, state):
        if (
            state.shape != (self.batch_size, 37)
            or state.device != self.device
            or state.dtype != self.dtype
        ):
            raise ValueError("walk state must match [batch,37], device and dtype")
        frame = self.indices.clone()
        requested_mask = self.plan.contact_masks[frame]
        requested_anchors = self.plan.anchors[frame]
        changed = (requested_mask != self.contact_mask).any(-1)
        retained = requested_mask & self.contact_mask
        anchor_match = (requested_anchors - self.anchors).abs().amax(-1) <= 1e-8
        accepted = ~changed
        projected = state
        if self.device.type != "cpu" or bool(changed.any()):
            event = self.mpc.model.transition(
                state,
                self.contact_mask,
                contact_mask=requested_mask,
                contact_positions=requested_anchors,
                friction=self.mpc.constraints.friction,
            )
            accepted = event.accepted & ((~retained) | anchor_match).all(-1) & ~self.failed
            releasing = (self.contact_mask & ~requested_mask).any(-1)
            if self.device.type != "cpu" or bool(releasing.any()):
                margin = go2_triangle_margin(
                    self.mpc.model, state, requested_mask, requested_anchors
                )
                accepted &= ~releasing | ((requested_mask.sum(-1) == 3) & (margin >= 0.02))
            projected = torch.where((changed & accepted)[:, None], event.state, state)
        target_mask = torch.where(accepted[:, None], requested_mask, self.contact_mask)
        target_anchors = torch.where(accepted[:, None, None], requested_anchors, self.anchors)
        waiting = changed & ~accepted
        saved = (self.mpc._warm.clone(), self.mpc._dual.clone(), self.mpc._valid.clone())
        self.mpc.reset(changed & accepted)
        reference, acceleration = self._references(waiting)
        action, result = self.mpc.compute(
            projected, reference, acceleration, target_mask, target_anchors
        )
        rejected = changed & accepted & ~result.usable
        if self.device.type != "cpu" or bool(rejected.any()):
            accepted &= ~rejected
            projected = torch.where(rejected[:, None], state, projected)
            target_mask = torch.where(rejected[:, None], self.contact_mask, target_mask)
            target_anchors = torch.where(rejected[:, None, None], self.anchors, target_anchors)
            waiting |= rejected
            self.mpc._warm.copy_(torch.where(rejected[:, None], saved[0], self.mpc._warm))
            self.mpc._dual.copy_(torch.where(rejected[:, None], saved[1], self.mpc._dual))
            self.mpc._valid.copy_(torch.where(rejected, saved[2], self.mpc._valid))
            reference, acceleration = self._references(waiting)
            action, result = self.mpc.compute(
                projected, reference, acceleration, target_mask, target_anchors
            )
        commit = changed & accepted & result.usable & ~self.failed
        self.contact_mask.copy_(torch.where(commit[:, None], requested_mask, self.contact_mask))
        self.anchors.copy_(torch.where(commit[:, None, None], requested_anchors, self.anchors))
        projected = torch.where(commit[:, None], projected, state)
        waiting = changed & ~commit
        self.wait_counts.copy_(torch.where(waiting, self.wait_counts + 1, 0))
        self.failed |= (self.wait_counts > self.max_wait_ticks) | ~result.usable
        result.feasible.logical_and_(~self.failed)
        usable_status = (result.status == SolveStatus.SUCCESS) | (
            result.status == SolveStatus.MAX_ITERATIONS
        )
        result.status.copy_(
            torch.where(
                self.failed & usable_status, SolveStatus.NO_FEASIBLE_CANDIDATE, result.status
            )
        )
        action = torch.where(self.failed[:, None], float("nan"), action)
        result.us.copy_(torch.where(self.failed[:, None, None], float("nan"), result.us))
        result.xs.copy_(torch.where(self.failed[:, None, None], float("nan"), result.xs))
        result.cost.copy_(torch.where(self.failed, float("nan"), result.cost))
        advance = result.usable & ~waiting
        self.indices.copy_(
            (self.indices + advance.to(torch.int64)).clamp_max(len(self.plan.reference) - 1)
        )
        return Go2WalkControl(
            projected,
            action,
            result,
            self.contact_mask.clone(),
            self.anchors.clone(),
            frame,
            changed & commit,
            waiting,
            (frame == len(self.plan.reference) - 1) & result.usable & ~waiting,
        )

    def reset(self, mask=None):
        """Caller must reset selected plants to plan.reference[0] at the same time."""
        if mask is None:
            mask = torch.ones(self.batch_size, device=self.device, dtype=torch.bool)
        if (
            mask.shape != (self.batch_size,)
            or mask.device != self.device
            or mask.dtype != torch.bool
        ):
            raise ValueError("reset mask must be bool [batch] on controller device")
        self.indices.masked_fill_(mask, 0)
        self.wait_counts.masked_fill_(mask, 0)
        self.failed.masked_fill_(mask, False)
        self.contact_mask.copy_(
            torch.where(mask[:, None], self.plan.contact_masks[0], self.contact_mask)
        )
        self.anchors.copy_(torch.where(mask[:, None, None], self.plan.anchors[0], self.anchors))
        self.mpc.reset(mask)
