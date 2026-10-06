"""Explicit Go2 support masks and guarded, perfectly inelastic touchdown.

Mode commands are external; this module does not infer contacts or plan a gait.
All four feet retain fixed tensor slots, including in flight (zero constraints).
"""

import math
from dataclasses import dataclass

import torch

from .go2 import GO2_LEGS, ContactDynamicsResult
from .go2_torch import Go2TorchDynamics, _mv


@dataclass
class ContactTransition:
    state: torch.Tensor
    accepted: torch.Tensor
    impulses_world: torch.Tensor
    kinetic_energy_before: torch.Tensor
    kinetic_energy_after: torch.Tensor


class Go2HybridDynamics(Go2TorchDynamics):
    """Four fixed slots; bool contact_mask [4] or [B,4], flat ground +z.

    Anchors are explicit and remain fixed while feet are in stance. Inactive
    feet have exactly zero force; no spring or bilateral ground attachment.
    Constructor parameters are immutable once used in a compiled controller.
    """

    def __init__(self, *, contact_mask=None, contact_positions=None, **kwargs):
        super().__init__(**kwargs)
        if self.feet != GO2_LEGS:
            raise ValueError("hybrid dynamics requires canonical FL/FR/RL/RR foot slots")
        if contact_mask is None:
            contact_mask = torch.ones(4, device=self.device, dtype=torch.bool)
        self.contact_mask = self._mask(contact_mask).clone()
        if contact_positions is not None:
            anchors = torch.as_tensor(contact_positions, device=self.device, dtype=self.dtype)
            if anchors.shape[-2:] != (4, 3) or anchors.ndim not in (2, 3):
                raise ValueError("contact_positions must be [4,3] or [B,4,3]")
            if not torch.isfinite(anchors).all():
                raise ValueError("contact_positions must be finite")
            self.contact_positions = anchors.clone()
        parameters = (
            (self.reference_configuration, 2),
            (self.contact_mask, 2),
            (self.contact_positions, 3),
        )
        batches = [value.shape[0] for value, batch_ndim in parameters if value.ndim == batch_ndim]
        if len(set(batches)) > 1:
            raise ValueError("reference, contact mask and anchors must share parameter batch size")

    @property
    def parameter_batch_size(self):
        mask = getattr(self, "contact_mask", None)
        if mask is not None and mask.ndim == 2:
            return self.contact_mask.shape[0]
        if self.contact_positions.ndim == 3:
            return self.contact_positions.shape[0]
        return super().parameter_batch_size

    def _mask(self, mask):
        if not isinstance(mask, torch.Tensor):
            mask = torch.as_tensor(mask, device=self.device)
        if (
            mask.dtype != torch.bool
            or mask.device != self.device
            or mask.ndim not in (1, 2)
            or mask.shape[-1] != 4
        ):
            raise ValueError("contact_mask must be bool [4] or [B,4] on model device")
        return mask

    def contact_dynamics(self, x, u):
        if (
            x.shape[-1] != 37
            or u.shape != (*x.shape[:-1], 12)
            or x.device != self.device
            or u.device != self.device
            or x.dtype != self.dtype
            or u.dtype != self.dtype
        ):
            raise ValueError("state/control must match hybrid Go2 shape, device and dtype")
        if self.parameter_batch_size is not None and x.shape[-2] != self.parameter_batch_size:
            raise ValueError("state batch must match hybrid parameters")
        mass, h, J, drift = self.quantities(x[..., :19], x[..., 19:])
        rows = self.contact_mask.repeat_interleave(3, -1).to(self.dtype)
        J = J * rows[..., None]
        drift = drift * rows
        # Inactive multiplier equations are f_i=0. Active equations are J_i a=-drift_i.
        inactive = torch.diag_embed(1 - rows).expand(*x.shape[:-1], 12, 12)
        kkt = torch.cat(
            (torch.cat((mass, -J.transpose(-1, -2)), -1), torch.cat((J, inactive), -1)), -2
        )
        rhs = torch.cat((_mv(self.actuation, u) - h, -drift), -1)
        solution, info = torch.linalg.solve_ex(kkt, rhs[..., None], check_errors=False)
        solution = torch.where((info == 0)[..., None], solution.squeeze(-1), float("nan"))
        a, f = solution[..., :18], solution[..., 18:]
        return ContactDynamicsResult(
            a,
            f.reshape(*x.shape[:-1], 4, 3),
            mass,
            h,
            J,
            drift,
            _mv(mass, a) + h - _mv(self.actuation, u) - _mv(J.transpose(-1, -2), f),
            _mv(J, a) + drift,
        )

    def quasi_static_torques(self):
        q = self.reference_configuration
        _, h, J, _ = self.quantities(q, q.new_zeros(*q.shape[:-1], 18))
        J = J * self.contact_mask.repeat_interleave(3, -1)[..., None]
        system = torch.cat((self.actuation.expand(*q.shape[:-1], 18, 12), J.transpose(-1, -2)), -1)
        seed = _mv(torch.linalg.pinv(system.double()), h.double()).to(self.dtype)
        if not torch.allclose(_mv(system, seed), h, atol=1e-5, rtol=1e-5):
            raise ValueError("selected contacts cannot balance the reference")
        return seed[..., :12]

    @torch.no_grad()
    def transition(
        self,
        state,
        previous_mask,
        *,
        friction=0.6,
        height_tolerance=0.002,
        approach_tolerance=1e-5,
        anchor_tolerance=0.002,
        ground_height=0.0,
        impulse_tolerance=1e-7,
    ):
        """Prepare a mode change without mutating model or input.

        Release preserves q/v. Added contacts require a nearby flat-ground
        anchor and a non-receding foot, then project v with a mass-metric
        plastic impact. Impulses at all retained/added feet must be unilateral
        and in the conservative friction cone. Rejected environments retain
        their original state; callers must also keep their original mode.
        """
        if state.ndim != 2 or state.shape[-1] != 37:
            raise ValueError("transition state must be [B,37]")
        if state.device != self.device or state.dtype != self.dtype:
            raise ValueError("transition state must match model device/dtype")
        if self.parameter_batch_size is not None and state.shape[0] != self.parameter_batch_size:
            raise ValueError("transition state batch must match hybrid parameters")
        scalars = (
            friction,
            height_tolerance,
            approach_tolerance,
            anchor_tolerance,
            ground_height,
            impulse_tolerance,
        )
        if (
            not all(math.isfinite(v) for v in scalars)
            or friction <= 0
            or min(height_tolerance, approach_tolerance, anchor_tolerance, impulse_tolerance) < 0
        ):
            raise ValueError("transition settings must be finite with nonnegative tolerances")
        old = self._mask(previous_mask).expand(state.shape[0], 4)
        new = self.contact_mask.expand(state.shape[0], 4)
        added = new & ~old
        impact = added.any(-1)
        valid = self.manifold.is_valid(state)
        safe = torch.where(valid[:, None], state, self.standing_state)
        q, velocity = safe[:, :19], safe[:, 19:]
        mass, _, J, _ = self.quantities(q, velocity)
        positions = self.foot_positions(q)
        foot_velocity = _mv(J, velocity).reshape(-1, 4, 3)
        anchors = self.contact_positions.expand(state.shape[0], 4, 3)
        near_ground = (positions[..., 2] - ground_height).abs() <= height_tolerance
        on_ground = (anchors[..., 2] - ground_height).abs() <= height_tolerance
        near_anchor = (positions - anchors).norm(dim=-1) <= anchor_tolerance
        approaching = foot_velocity[..., 2] <= approach_tolerance
        valid &= ((~added) | (near_ground & on_ground & near_anchor & approaching)).all(-1)
        # Retained feet stay near supplied anchors. The mode bank additionally
        # checks identity against their previous anchors.
        valid &= ((~(new & old)) | near_anchor).all(-1)
        rows = new.repeat_interleave(3, -1).to(self.dtype)
        active_J = J * rows[..., None]
        kkt = torch.cat(
            (
                torch.cat((mass, -active_J.transpose(-1, -2)), -1),
                torch.cat((active_J, torch.diag_embed(1 - rows)), -1),
            ),
            -2,
        )
        rhs = torch.cat((torch.zeros_like(velocity), -_mv(active_J, velocity)), -1)
        solution, info = torch.linalg.solve_ex(kkt, rhs[..., None], check_errors=False)
        delta, impulse = solution.squeeze(-1).split((18, 12), -1)
        impulse = impulse.reshape(-1, 4, 3)
        after = velocity + delta
        before_energy = 0.5 * (velocity * _mv(mass, velocity)).sum(-1)
        after_energy = 0.5 * (after * _mv(mass, after)).sum(-1)
        impulse_valid = (
            (impulse[..., 2] >= -impulse_tolerance)
            & (impulse[..., :2].abs().sum(-1) <= friction * impulse[..., 2] + impulse_tolerance)
        ).all(-1)
        residual = _mv(active_J, after).abs().amax(-1)
        impact_valid = (
            (info == 0)
            & torch.isfinite(solution).flatten(1).all(-1)
            & impulse_valid
            & (residual <= impulse_tolerance)
            & (after_energy <= before_energy + impulse_tolerance)
        )
        valid &= ~impact | impact_valid
        accepted_impact = valid & impact
        output = torch.cat((q, after), -1)
        return ContactTransition(
            torch.where(accepted_impact[:, None], output, state),
            valid,
            torch.where(accepted_impact[:, None, None], impulse, 0.0),
            before_energy,
            torch.where(accepted_impact, after_energy, before_energy),
        )
