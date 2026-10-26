"""Pure Torch standing / FL tracking interface for simulator integration.

Only flat-ground stance and release of FL are supported. Touchdown, object
contacts and state estimation belong to a future simulator adapter.
"""

from dataclasses import dataclass

import torch

from ..result import MPCResult, SolveStatus
from .go2_walk import Go2WalkMPC, go2_center_of_mass, go2_leg_ik, smoothstep


@dataclass(frozen=True)
class Go2TrackingCommand:
    reference: torch.Tensor  # [B,T+1,37], local-world pose and body-frame velocities
    acceleration: torch.Tensor  # [B,18], generalized acceleration target
    support_mask: torch.Tensor  # bool [B,4], FL/FR/RL/RR
    anchors: torch.Tensor  # [B,4,3], fixed local-world point contacts


@dataclass(frozen=True)
class Go2TrackingOutput:
    action: torch.Tensor  # [B,12], Nm; zero on failure, not a balance fallback
    usable: torch.Tensor  # bool [B]
    needs_reset: torch.Tensor  # bool [B], sticky until explicit reset
    support_mask: torch.Tensor
    result: MPCResult  # rejected trajectories remain NaN


class Go2TrackingMPC:
    """Unified measured-state -> torque API; no native callbacks or state edits.

    Commands may vary per environment. A requested FL release takes effect for
    usable outputs only. Re-adding FL is rejected: use a guarded touchdown
    controller, or reset after physically resetting the simulator episode.
    """

    def __init__(self, batch_size=1, *, horizon=2, **settings):
        self.mpc = Go2WalkMPC(batch_size, horizon=horizon, **settings)
        self.model = self.mpc.model
        self.batch_size, self.horizon = batch_size, horizon
        self.device, self.dtype = self.mpc.device, self.mpc.dtype
        self.failed = torch.zeros(batch_size, device=self.device, dtype=torch.bool)
        self.support_mask = torch.ones(batch_size, 4, device=self.device, dtype=torch.bool)
        self._anchors = self.model.contact_positions.expand(batch_size, -1, -1).clone()
        self._initialized = torch.zeros_like(self.failed)

    @torch.no_grad()
    def compute(self, state, command: Go2TrackingCommand):
        B, T = self.batch_size, self.horizon
        shapes = (
            (state, (B, 37)),
            (command.reference, (B, T + 1, 37)),
            (command.acceleration, (B, 18)),
            (command.anchors, (B, 4, 3)),
        )
        for value, shape in shapes:
            if value.shape != shape or value.device != self.device or value.dtype != self.dtype:
                raise ValueError(f"tracking tensor must match {shape}, device and float64")
        mask = command.support_mask
        if mask.shape != (B, 4) or mask.device != self.device or mask.dtype != torch.bool:
            raise ValueError("support_mask must be bool [B,4] on controller device")
        numerical = self.model.manifold.is_valid(state)
        numerical &= self.model.manifold.is_valid(command.reference).all(-1)
        numerical &= torch.isfinite(command.acceleration).all(-1)
        numerical &= torch.isfinite(command.anchors).flatten(1).all(-1)
        # No moving stance anchors, unsupported support sets, or implicit impacts.
        allowed = mask[:, 1:].all(-1) & (~mask | self.support_mask).all(-1)
        anchor_match = (command.anchors - self._anchors).abs().amax(-1) <= 1e-8
        allowed &= ~self._initialized | (~mask | anchor_match).all(-1)
        allowed &= (command.anchors[..., 2].abs() <= 1e-8).all(-1)
        valid = numerical & allowed & ~self.failed
        changed = (mask != self.support_mask).any(-1) & valid
        self.mpc.reset(changed)
        default = self.model.standing_state.expand(B, -1)
        safe = torch.where(valid[:, None], state, default)
        reference = torch.where(valid[:, None, None], command.reference, default[:, None])
        acceleration = torch.where(valid[:, None], command.acceleration, 0.0)
        supports = torch.where(valid[:, None], mask, True)
        anchors = torch.where(valid[:, None, None], command.anchors, self.model.contact_positions)
        action, raw = self.mpc.compute(safe, reference, acceleration, supports, anchors)
        usable = valid & raw.usable
        self.failed |= ~usable
        self.support_mask.copy_(torch.where(usable[:, None], mask, self.support_mask))
        self._anchors.copy_(torch.where(usable[:, None, None], command.anchors, self._anchors))
        self._initialized |= usable
        status = torch.where(
            ~numerical,
            SolveStatus.NUMERICAL_FAILURE,
            torch.where(~allowed, SolveStatus.OUTSIDE_LOCAL_MODEL, raw.status),
        )
        status = torch.where(
            ~usable & (status == SolveStatus.SUCCESS), SolveStatus.NO_FEASIBLE_CANDIDATE, status
        )
        status = torch.where(
            ~usable & (status == SolveStatus.MAX_ITERATIONS),
            SolveStatus.NO_FEASIBLE_CANDIDATE,
            status,
        )
        result = MPCResult(
            torch.where(usable[:, None, None], raw.xs, float("nan")),
            torch.where(usable[:, None, None], raw.us, float("nan")),
            torch.where(usable, raw.cost, float("nan")),
            status,
            raw.iterations,
            usable,
        )
        return Go2TrackingOutput(
            torch.where(usable[:, None], action, 0.0),
            usable,
            self.failed.clone(),
            self.support_mask.clone(),
            result,
        )

    @torch.no_grad()
    def reset(self, mask=None):
        """Call together with simulator episode reset; restores four-foot mode."""
        if mask is None:
            mask = torch.ones_like(self.failed)
        if (
            mask.shape != (self.batch_size,)
            or mask.device != self.device
            or mask.dtype != torch.bool
        ):
            raise ValueError("reset mask must be bool [B] on controller device")
        self.mpc.reset(mask)
        self.failed.masked_fill_(mask, False)
        self.support_mask.masked_fill_(mask[:, None], True)
        self._initialized.masked_fill_(mask, False)


@dataclass(frozen=True)
class Go2TrackingPlan:
    reference: torch.Tensor  # [N,37]
    acceleration: torch.Tensor  # [N,18]
    support_masks: torch.Tensor  # [N,4]
    feet: torch.Tensor  # [N,4,3]
    anchors: torch.Tensor  # [4,3]
    phases: tuple[str, ...]
    dt: float

    def command(self, step, batch_size, horizon):
        """Reference lookup only; no state propagation or online IK."""
        if not 0 <= step < len(self.reference):
            raise ValueError("plan exhausted; stop or reset the episode")
        indices = torch.arange(horizon + 1, device=self.reference.device)
        indices = (indices + step).clamp_max(len(self.reference) - 1)
        return Go2TrackingCommand(
            self.reference[indices][None].expand(batch_size, -1, -1),
            self.acceleration[step].expand(batch_size, -1),
            self.support_masks[step].expand(batch_size, -1),
            self.anchors.expand(batch_size, -1, -1),
        )


def go2_front_leg_tracking_plan(model, *, standing_only=False):
    """Setup-only IK: stance, COM transfer, FL lift, reach, return and hold.

    The last phase keeps FL airborne. No object force or touchdown is modeled.
    IK validation may synchronize CUDA during preparation, never during compute.
    """
    dt = model.dt
    if dt > 0.02:
        raise ValueError("tracking demo plan requires dt <= 0.02 seconds")
    q0 = model.reference_configuration
    anchors = model.contact_positions.clone()
    tripod = torch.tensor([False, True, True, True], device=model.device)
    stance = torch.ones(4, device=model.device, dtype=torch.bool)
    body = q0[:3].clone()
    target_body = body.clone()
    target_body[:2] = anchors[tripod, :2].mean(0)
    for _ in range(3):
        base = q0[None].clone()
        base[0, :3] = target_body
        pose = go2_leg_ik(model, base, anchors[None])
        target_body[:2] += anchors[tripod, :2].mean(0) - go2_center_of_mass(model, pose)[0, :2]
    bases, feet, masks, phases = [], [], [], []

    def append(position, foot, mask, phase):
        q = q0.clone()
        q[:3] = position
        bases.append(q)
        feet.append(foot.clone())
        masks.append(mask)
        phases.append(phase)

    for _ in range(round(0.4 / dt)):
        append(body, anchors, stance, "stance")
    if not standing_only:
        ticks = round(1.0 / dt)
        for i in range(1, ticks + 1):
            append(body + smoothstep(i / ticks) * (target_body - body), anchors, stance, "shift")
        for phase in ("lift", "reach", "return", "hold"):
            for i in range(1, ticks + 1):
                s = smoothstep(i / ticks)
                foot = anchors.clone()
                foot[0, 2] += 0.02 * (s if phase == "lift" else 1.0)
                progress = s if phase == "reach" else (1 - s if phase == "return" else 0.0)
                foot[0, 0] += 0.01 * progress
                foot[0, 1] += 0.005 * progress
                append(target_body, foot, tripod, phase)
    else:
        for _ in range(round(2.0 / dt)):
            append(body, anchors, stance, "stance")
    feet = torch.stack(feet)
    q = go2_leg_ik(model, torch.stack(bases), feet)
    v = q.new_zeros(len(q), 18)
    v[1:-1, :3] = (q[2:, :3] - q[:-2, :3]) / (2 * dt)
    v[1:-1, 6:] = (q[2:, 7:] - q[:-2, 7:]) / (2 * dt)
    a = torch.zeros_like(v)
    a[1:-1] = (v[2:] - v[:-2]) / (2 * dt)
    return Go2TrackingPlan(
        torch.cat((q, v), -1), a, torch.stack(masks), feet, anchors, tuple(phases), dt
    )
