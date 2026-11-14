"""C1: C0 dynamics with future Ag(q), mapped speed and quasi-static torque limits.

Configuration is a checked common prior, not an additional optimization state.
The static torque approximation deliberately omits M a and Coriolis effects;
the unchanged inverse-dynamics WBC remains responsible for actual execution.
"""

from dataclasses import dataclass

import numpy as np

from ..guidance import _array
from .standing_centroidal import CentroidalReference, CentroidalSettings, StandingCentroidalPlanner


@dataclass(frozen=True)
class ConfigurationSettings(CentroidalSettings):
    base_linear_velocity_limit: float = 0.5
    base_angular_velocity_limit: float = 1.0


@dataclass(frozen=True)
class ConfigurationReference(CentroidalReference):
    nominal_states: np.ndarray
    maps: np.ndarray
    inverses: np.ndarray
    gravities: np.ndarray
    foot_jacobians: np.ndarray
    static_forces: np.ndarray
    torque_limits: np.ndarray
    model_id: str
    diagnostics: dict

    def __post_init__(self):
        super().__post_init__()
        n = len(self.times)
        for name, shape in (
            ("nominal_states", (n, 49)),
            ("maps", (n, 6, 24)),
            ("inverses", (n, 6, 6)),
            ("gravities", (n, 24)),
            ("foot_jacobians", (n, 12, 24)),
            ("static_forces", (n, 4, 3)),
            ("torque_limits", (18,)),
        ):
            object.__setattr__(self, name, _array(getattr(self, name), shape, name))
        if np.any(self.torque_limits <= 0):
            raise ValueError("invalid configuration torque limits")


class StandingConfigurationPlanner(StandingCentroidalPlanner):
    def __init__(self, model, com_anchor, settings=None):
        super().__init__(
            model, com_anchor, ConfigurationSettings() if settings is None else settings
        )
        self.configuration = None

    @property
    def velocity_limits(self):
        return np.r_[
            np.full(3, self.settings.base_linear_velocity_limit),
            np.full(3, self.settings.base_angular_velocity_limit),
        ]

    def mapped_velocities(self, states):
        r = self.configuration
        joint_momentum = np.einsum("nij,nj->ni", r.maps[:, :, 6:], r.nominal_states[:, 31:])
        return np.einsum("nij,nj->ni", r.inverses, states[:, 3:9] - joint_momentum)

    def _qp(self, initial, states, forces, reference, previous, radius):
        H, g, G, bound, matrices, offsets = super()._qp(
            initial, states, forces, reference, previous, radius
        )
        r, n = self.configuration, self.settings.nodes
        extra, limits = [], []
        # Future mapped base twist: Ag_b(q)^-1 (h - Ag_j(q) v_j).
        for k in range(1, n + 1):
            matrix = r.inverses[k] @ matrices[k, 3:9]
            offset = r.inverses[k] @ (offsets[k, 3:9] - r.maps[k, :, 6:] @ r.nominal_states[k, 31:])
            extra.extend([matrix, -matrix])
            limits.extend([self.velocity_limits - offset, self.velocity_limits + offset])
        # Predicted support forces must also fit the reference posture's static
        # joint torque envelope. Terminal posture uses the held last force.
        for k in range(n + 1):
            matrix = np.zeros((18, 12 * n))
            i = min(k, n - 1)
            matrix[:, i * 12 : (i + 1) * 12] = -r.foot_jacobians[k, :, 6:].T
            extra.extend([matrix, -matrix])
            limits.extend(
                [r.torque_limits - r.gravities[k, 6:], r.torque_limits + r.gravities[k, 6:]]
            )
        return H, g, np.vstack([G, *extra]), np.concatenate([bound, *limits]), matrices, offsets

    def validate_candidate(self, initial, states, forces, previous):
        audit = super().validate_candidate(initial, states, forces, previous)
        if self.configuration is None:
            raise ValueError("C1 requires a checked configuration preview")
        r = self.configuration
        velocities = self.mapped_velocities(np.asarray(states))
        torques = np.stack(
            [
                r.gravities[k, 6:]
                - r.foot_jacobians[k, :, 6:].T @ forces[min(k, len(forces) - 1)].ravel()
                for k in range(len(states))
            ]
        )
        speed_violation = float(max(0, np.max(abs(velocities) - self.velocity_limits)))
        torque_violation = float(max(0, np.max(abs(torques) - r.torque_limits)))
        audit["metrics"].update(
            mapped_base_velocity_violation=speed_violation,
            quasi_static_torque_violation=torque_violation,
            quasi_static_peak_utilization=float(np.max(abs(torques) / r.torque_limits)),
        )
        audit["limits"].update(
            mapped_base_velocity_violation=self.settings.constraint_tolerance,
            quasi_static_torque_violation=self.settings.constraint_tolerance,
        )
        for name, value in (
            ("mapped_base_velocity", speed_violation),
            ("quasi_static_torque", torque_violation),
        ):
            if value > self.settings.constraint_tolerance:
                audit["failures"].append(name)
        audit["feasible"] = not audit["failures"]
        return audit

    def solve(self, initial, reference, previous, *, initial_guess=None):
        if not isinstance(reference, ConfigurationReference) or not reference.diagnostics.get(
            "accepted"
        ):
            raise ValueError("C1 requires an accepted configuration/capability preview")
        self.configuration = reference
        plan = super().solve(initial, reference, previous, initial_guess=initial_guess)
        plan.diagnostics.update(
            configuration=reference.diagnostics,
            model_scope=(
                "C1 configuration-conditioned centroidal dynamics; Ag(q) speed mapping and "
                "static torque envelope, not a full-body dynamic feasibility certificate"
            ),
        )
        return plan
