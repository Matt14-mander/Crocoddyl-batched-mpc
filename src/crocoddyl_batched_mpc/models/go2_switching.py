"""Prepared support-mode bank with per-environment guarded contact events."""

import torch

from ..result import MPCResult
from .go2_hybrid import ContactTransition
from .go2_realtime import Go2ConstrainedMPC


class Go2ContactSwitchingMPC:
    """Prepare modes outside the loop, then request explicit release/touchdown.

    Each environment has its own current mode and warm state. A failed switch
    preserves its old mode/state. switch() returns the post-impact measurement
    that must be used by compute() and the plant. No mode prediction, swing
    objective or automatic ground detection is implicit.
    """

    def __init__(self, batch_size=1, horizon=2, **settings):
        if "contact_mask" in settings:
            raise ValueError("initial mode is four-foot support; use prepare_mode")
        self.settings = dict(settings)
        self.batch_size, self.horizon = batch_size, horizon
        initial = Go2ConstrainedMPC(batch_size, horizon, contact_mask=[True] * 4, **settings)
        self.controllers = [initial]
        self.names = {"stance": 0}
        self.device, self.dtype = initial.device, initial.dtype
        self.mode_indices = torch.zeros(batch_size, device=self.device, dtype=torch.int64)

    def prepare_mode(
        self, name, contact_mask, *, reference_configuration=None, contact_positions=None
    ):
        """Build derivatives, matrices and compiled graph; never in a 20 ms loop.

        A prepared landing pose supplies the new anchors. Retained anchors must
        remain identical at switch time. Flight dynamics exists, but this local
        standing MPC requires an equilibrium-capable support set.
        """
        if not isinstance(name, str) or not name or name in self.names:
            raise ValueError("mode name must be nonempty and unique")
        settings = self.settings.copy()
        if reference_configuration is not None:
            settings["reference_configuration"] = reference_configuration
        if contact_positions is not None:
            settings["contact_positions"] = contact_positions
        candidate = Go2ConstrainedMPC(
            self.batch_size, self.horizon, contact_mask=contact_mask, **settings
        )
        active = candidate.dynamics.contact_mask
        if (candidate.dynamics.contact_positions[active, 2].abs() > 0.002).any():
            raise ValueError("prepared support anchors must be on the flat ground")
        candidate.reset()
        self.names[name] = len(self.controllers)
        self.controllers.append(candidate)
        return self.names[name]

    @torch.no_grad()
    def switch(self, name, state, *, mask=None, **transition_settings):
        """Commit only guarded events with a feasible target-mode MPC trajectory."""
        if name not in self.names:
            raise ValueError("prepare the requested mode before switching")
        if (
            state.shape != (self.batch_size, 37)
            or state.device != self.device
            or state.dtype != self.dtype
        ):
            raise ValueError("state must match controller [batch,37], device and dtype")
        if mask is None:
            mask = torch.ones(self.batch_size, device=self.device, dtype=torch.bool)
        if (
            mask.shape != (self.batch_size,)
            or mask.device != self.device
            or mask.dtype != torch.bool
        ):
            raise ValueError("switch mask must be bool [batch] on controller device")
        index = self.names[name]
        target = self.controllers[index]
        old_masks = torch.stack([c.dynamics.contact_mask for c in self.controllers])[
            self.mode_indices
        ]
        old_anchors = torch.stack([c.dynamics.contact_positions for c in self.controllers])[
            self.mode_indices
        ]
        transition_settings.setdefault("friction", target.constraints.friction)
        if transition_settings["friction"] > target.constraints.friction:
            raise ValueError("impact friction cannot exceed target-mode friction")
        event = target.dynamics.transition(state, old_masks, **transition_settings)
        retained = old_masks & target.dynamics.contact_mask
        anchor_match = (old_anchors - target.dynamics.contact_positions).abs().amax(-1) <= 1e-8
        accepted = event.accepted & ((~retained) | anchor_match).all(-1) & mask
        changed = self.mode_indices != index
        # Old mode warm controls/duals cannot seed a differently sized force QP.
        saved = (target._warm.clone(), target._valid.clone(), target._dual.clone())
        target.reset(mask & changed)
        safe = torch.where(accepted[:, None], event.state, target.reference)
        _, candidate = target.compute(safe)
        accepted &= candidate.usable
        # A request (including rejection) must not perturb another environment's cache.
        target._warm.copy_(saved[0])
        target._valid.copy_(saved[1])
        target._dual.copy_(saved[2])
        committed = accepted & changed
        for c in self.controllers:
            c.reset(committed)
        self.mode_indices.copy_(torch.where(committed, index, self.mode_indices))
        return ContactTransition(
            torch.where(accepted[:, None], event.state, state),
            accepted,
            torch.where(accepted[:, None, None], event.impulses_world, 0.0),
            event.kinetic_energy_before,
            torch.where(accepted, event.kinetic_energy_after, event.kinetic_energy_before),
        )

    @torch.no_grad()
    def compute(self, state):
        if (
            state.shape != (self.batch_size, 37)
            or state.device != self.device
            or state.dtype != self.dtype
        ):
            raise ValueError("state must match controller [batch,37], device and dtype")
        action, output = None, None
        for index, c in enumerate(self.controllers):
            selected = self.mode_indices == index
            if self.device.type == "cpu" and not bool(selected.any()):
                continue
            safe = torch.where(selected[:, None], state, c.reference)
            u, result = c.compute(safe)
            # Unselected slots remain at this mode's reference, as warmed dummy
            # QPs. Resetting them every cycle forced extra Newton steps for the
            # entire batch. A real mode change clears all caches before use.
            if output is None:
                action, output = u, result
            else:
                action = torch.where(selected[:, None], u, action)
                output = MPCResult(
                    *[
                        torch.where(selected.reshape(-1, *([1] * (a.ndim - 1))), a, b)
                        for a, b in zip(
                            (
                                result.xs,
                                result.us,
                                result.cost,
                                result.status,
                                result.iterations,
                                result.feasible,
                            ),
                            (
                                output.xs,
                                output.us,
                                output.cost,
                                output.status,
                                output.iterations,
                                output.feasible,
                            ),
                        )
                    ]
                )
        return action, output

    def reset(self, mask=None):
        """Clear warm starts; contact modes persist (reset is not a contact event)."""
        for c in self.controllers:
            c.reset(mask)
