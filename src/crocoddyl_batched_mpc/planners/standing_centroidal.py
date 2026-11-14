"""C0 sequential convex QP with hard force/COM/rate bounds and nonlinear audit."""

import math
import time
from dataclasses import dataclass

import numpy as np

from ..guidance import _array
from ..result import MPCResult, SolveStatus


@dataclass(frozen=True)
class CentroidalSettings:
    nodes: int = 24
    dt: float = 0.05
    max_iterations: int = 8
    qp_iterations: int = 60
    qp_tolerance: float = 1e-8
    constraint_tolerance: float = 1e-6
    dynamics_tolerance: float = 1e-8
    linearization_tolerance: float = 1e-6
    stopping_tolerance: float = 1e-7
    friction: float = 0.6
    normal_min: float = 2.0
    normal_max: float = 200.0
    force_rate_limit: float = 400.0  # N/s, each world force component
    com_translation_bound: tuple = (0.08, 0.08, 0.08)
    height_min: float = 0.15
    height_max: float = 0.65
    trust_radius: float = 0.02  # COM m; forces remain inside their hard cone/normal box
    trust_min: float = 1e-5
    com_weight: float = 10000.0
    velocity_weight: float = 100.0
    angular_momentum_weight: float = 100.0
    force_weight: float = 1e-4
    force_change_weight: float = 1e-3
    terminal_scale: float = 5.0

    def __post_init__(self):
        for name in ("nodes", "max_iterations", "qp_iterations"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name, value in vars(self).items():
            if name == "com_translation_bound":
                if len(value) != 3 or any(not math.isfinite(v) or v <= 0 for v in value):
                    raise ValueError("COM bounds require three positive finite values")
            elif not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.height_min >= self.height_max or self.normal_min >= self.normal_max:
            raise ValueError("invalid height or normal-force interval")
        if self.trust_min > self.trust_radius:
            raise ValueError("minimum trust radius exceeds initial radius")


@dataclass(frozen=True)
class CentroidalReference:
    times: np.ndarray
    states: np.ndarray
    forces: np.ndarray

    def __post_init__(self):
        times = np.asarray(self.times)
        if times.ndim != 1 or len(times) < 2:
            raise ValueError("reference needs at least two knots")
        times = _array(times, (len(times),), "times")
        if np.any(np.diff(times) <= 0):
            raise ValueError("reference times must increase")
        object.__setattr__(self, "times", times)
        object.__setattr__(self, "states", _array(self.states, (len(times), 9), "states"))
        object.__setattr__(self, "forces", _array(self.forces, (len(times) - 1, 4, 3), "forces"))


@dataclass
class CentroidalPlan:
    mpc: MPCResult
    times: np.ndarray
    diagnostics: dict
    converged: bool
    solve_time_s: float
    linearization_defect: float


class StandingCentroidalPlanner:
    def __init__(self, model, com_anchor, settings=None):
        self.model = model
        self.settings = CentroidalSettings() if settings is None else settings
        self.com_anchor = _array(com_anchor, (3,), "episode COM anchor")
        self.lower = self.com_anchor - self.settings.com_translation_bound
        self.upper = self.com_anchor + self.settings.com_translation_bound
        ground = float(model.anchors[0, 2])
        self.lower[2] = max(self.lower[2], ground + self.settings.height_min)
        self.upper[2] = min(self.upper[2], ground + self.settings.height_max)
        if np.any(self.lower >= self.upper):
            raise ValueError("COM box and height interval do not intersect")
        n = self.settings.nodes
        self.difference = np.eye(12 * n)
        self.difference[12:, :-12] -= np.eye(12 * (n - 1))
        cone = np.array(
            [[a, b, -self.settings.friction] for a, b in ((1, 1), (1, -1), (-1, 1), (-1, -1))]
        )
        cone = np.vstack([cone, [0, 0, -1], [0, 0, 1]])
        self.force_matrix = np.kron(np.eye(n * 4), cone)
        self.force_bound = np.tile(
            [0, 0, 0, 0, -self.settings.normal_min, self.settings.normal_max], n * 4
        )

    def _inputs(self, initial, reference, previous):
        s = self.settings
        initial = _array(initial, (9,), "initial centroidal state")
        previous = _array(previous, (4, 3), "previous contact forces")
        times = _array(reference.times, (s.nodes + 1,), "reference times")
        if not np.allclose(times, np.arange(s.nodes + 1) * s.dt, rtol=0, atol=1e-12):
            raise ValueError("reference times must be relative 0, dt, ..., N*dt")
        _array(reference.states, (s.nodes + 1, 9), "centroidal reference")
        _array(reference.forces, (s.nodes, 4, 3), "force reference")
        return initial, previous

    def _weights(self):
        s = self.settings
        weights = s.dt * np.tile(
            np.r_[
                np.full(3, s.com_weight),
                np.full(3, s.velocity_weight / self.model.mass**2),
                np.full(3, s.angular_momentum_weight),
            ],
            (s.nodes + 1, 1),
        )
        weights[-1] *= s.terminal_scale
        return weights

    def objective(self, states, forces, reference, previous):
        s = self.settings
        changes = np.diff(np.concatenate([previous[None], forces]), axis=0)
        return float(
            np.sum(self._weights() * (states - reference.states) ** 2)
            + s.dt * s.force_weight * np.sum((forces - reference.forces) ** 2)
            + s.dt * s.force_change_weight * np.sum(changes**2)
        )

    def _condense(self, initial, states, forces):
        s = self.settings
        matrices, offsets = [np.zeros((9, 12 * s.nodes))], [initial.copy()]
        for k in range(s.nodes):
            A, B = self.model.linearize(states[k], forces[k], s.dt)
            affine = (
                self.model.step(states[k], forces[k], s.dt) - A @ states[k] - B @ forces[k].ravel()
            )
            matrix = A @ matrices[-1]
            matrix[:, 12 * k : 12 * (k + 1)] += B
            matrices.append(matrix)
            offsets.append(A @ offsets[-1] + affine)
        return np.stack(matrices), np.stack(offsets)

    def _qp(self, initial, states, forces, reference, previous, radius):
        s = self.settings
        matrices, offsets = self._condense(initial, states, forces)
        L, d, weights = matrices.reshape(-1, s.nodes * 12), offsets.ravel(), self._weights().ravel()
        D = self.difference
        previous_vector = np.r_[previous.ravel(), np.zeros(12 * (s.nodes - 1))]
        H = 2 * (
            L.T @ (weights[:, None] * L)
            + s.dt * s.force_change_weight * D.T @ D
            + s.dt * s.force_weight * np.eye(12 * s.nodes)
        )
        g = 2 * (
            L.T @ (weights * (d - reference.states.ravel()))
            - s.dt * s.force_weight * reference.forces.ravel()
            - s.dt * s.force_change_weight * D.T @ previous_vector
        )
        # x0 is measured, fixed and separately checked. Only future COM has a trust box.
        C, c = matrices[1:, :3].reshape(-1, s.nodes * 12), offsets[1:, :3].ravel()
        # A separate initialization QP may omit the algorithmic trust box to
        # restore an infeasible shifted rollout. Physical bounds never change.
        lower = (
            np.broadcast_to(self.lower, (s.nodes, 3))
            if radius is None
            else np.maximum(self.lower, states[1:, :3] - radius)
        ).ravel()
        upper = (
            np.broadcast_to(self.upper, (s.nodes, 3))
            if radius is None
            else np.minimum(self.upper, states[1:, :3] + radius)
        ).ravel()
        G = np.vstack([self.force_matrix, D, -D, C, -C])
        bound = np.r_[
            self.force_bound,
            s.dt * s.force_rate_limit + previous_vector,
            s.dt * s.force_rate_limit - previous_vector,
            upper - c,
            c - lower,
        ]
        return (H + H.T) * 0.5, g, G, bound, matrices, offsets

    def validate_candidate(self, initial, states, forces, previous):
        """Independent nonlinear equations and every hard bound, including terminal COM."""
        s = self.settings
        initial = _array(initial, (9,), "initial state")
        previous = _array(previous, (4, 3), "previous forces")
        states = _array(states, (s.nodes + 1, 9), "candidate states")
        forces = _array(forces, (s.nodes, 4, 3), "candidate forces")
        # Explicit formulas here are intentionally independent of step/linearize/condensing.
        errors = []
        for k, force in enumerate(forces):
            c, linear, angular = np.split(states[k], 3)
            resultant = force.sum(axis=0) + self.model.mass * self.model.gravity
            moment = np.cross(self.model.anchors - c, force).sum(axis=0)
            predicted = np.r_[
                c + s.dt * linear / self.model.mass + 0.5 * s.dt**2 * resultant / self.model.mass,
                linear + s.dt * resultant,
                angular + s.dt * moment,
            ]
            errors.append(float(np.max(np.abs(states[k + 1] - predicted))))
        changes = np.diff(np.concatenate([previous[None], forces]), axis=0)
        metrics = dict(
            initial_error=float(np.max(np.abs(states[0] - initial))),
            dynamics_gap=max(errors),
            friction=float(
                max(
                    0.0, np.max(np.abs(forces[:, :, :2]).sum(axis=2) - s.friction * forces[:, :, 2])
                )
            ),
            normal_force=float(
                max(
                    0.0,
                    np.max(s.normal_min - forces[:, :, 2]),
                    np.max(forces[:, :, 2] - s.normal_max),
                )
            ),
            force_rate=float(max(0.0, np.max(np.abs(changes)) / s.dt - s.force_rate_limit)),
            com_box=float(
                max(0.0, np.max(self.lower - states[:, :3]), np.max(states[:, :3] - self.upper))
            ),
        )
        limits = {name: s.constraint_tolerance for name in metrics}
        limits.update(initial_error=s.dynamics_tolerance, dynamics_gap=s.dynamics_tolerance)
        failures = [name for name, value in metrics.items() if value > limits[name]]
        return dict(
            feasible=not failures,
            failures=failures,
            metrics=metrics,
            limits=limits,
            per_node_dynamics_gap=errors,
        )

    def solve(self, initial, reference, previous_forces, *, initial_guess=None):
        import torch

        from ..qp import solve_inequality_qp

        start = time.perf_counter()
        s = self.settings
        initial, previous = self._inputs(initial, reference, previous_forces)
        controls = np.array(
            reference.forces
            if initial_guess is None
            else _array(initial_guess, (s.nodes, 4, 3), "initial forces"),
            copy=True,
        )
        states = self.model.rollout(initial, controls, s.dt)
        radius, converged, defect = s.trust_radius, False, float("inf")
        history = []
        reason = None
        restore = not self.validate_candidate(initial, states, controls, previous)["feasible"]
        for iteration in range(s.max_iterations + int(restore)):
            restoring = restore and iteration == 0
            H, g, G, bound, matrices, offsets = self._qp(
                initial, states, controls, reference, previous, None if restoring else radius
            )
            qp_start = time.perf_counter()
            qp = solve_inequality_qp(
                torch.from_numpy(H),
                torch.from_numpy(g[None]),
                torch.from_numpy(G),
                torch.from_numpy(bound[None]),
                initial=torch.from_numpy(controls.reshape(1, -1)),
                iterations=s.qp_iterations,
                tolerance=s.qp_tolerance,
            )
            entry = dict(
                iteration=iteration,
                phase="feasibility_restoration" if restoring else "scp",
                trust_radius=None if restoring else radius,
                qp_time_s=time.perf_counter() - qp_start,
                qp_iterations=int(qp.iterations[0]),
                qp_converged=bool(qp.converged[0]),
                qp_feasible=bool(qp.feasible[0]),
                qp_primal_violation=float(qp.primal_violation[0]),
                qp_dual_residual=float(qp.dual_residual[0]),
            )
            history.append(entry)
            if not bool(qp.feasible[0]) or not bool(qp.converged[0]):
                reason = "qp_not_converged"
                break
            candidate = qp.x[0].numpy().reshape(s.nodes, 4, 3)
            linear = np.einsum("ijk,k->ij", matrices, candidate.ravel()) + offsets
            rollout = self.model.rollout(initial, candidate, s.dt)
            gap = max(
                float(
                    np.max(np.abs(linear[k + 1] - self.model.step(linear[k], candidate[k], s.dt)))
                )
                for k in range(s.nodes)
            )
            old_cost = self.objective(states, controls, reference, previous)
            predicted_cost = self.objective(linear, candidate, reference, previous)
            actual_cost = self.objective(rollout, candidate, reference, previous)
            predicted_reduction, actual_reduction = (
                old_cost - predicted_cost,
                old_cost - actual_cost,
            )
            ratio = actual_reduction / predicted_reduction if predicted_reduction > 1e-12 else 1.0
            entry.update(
                linearization_defect=gap,
                old_cost=old_cost,
                actual_cost=actual_cost,
                predicted_reduction=predicted_reduction,
                actual_reduction=actual_reduction,
                reduction_ratio=ratio,
            )
            if restoring:
                restored = self.validate_candidate(initial, rollout, candidate, previous)
                entry["accepted_step"] = restored["feasible"]
                entry["restoration_audit"] = restored
                if not restored["feasible"]:
                    reason = "infeasible_initialization"
                    break
                states, controls = rollout, candidate.copy()
                # Restoration establishes feasibility, not SCP convergence.
                # Keep defect=inf until a regular trust-region step is accepted.
                continue
            if actual_reduction < -s.stopping_tolerance * (1 + abs(old_cost)) or ratio < 0.1:
                radius *= 0.5
                entry["accepted_step"] = False
                if radius < s.trust_min:
                    reason = "trust_region_exhausted"
                    break
                continue
            step_norm = float(np.max(np.abs(candidate - controls)))
            states, controls, defect = rollout, candidate.copy(), gap
            entry["accepted_step"] = True
            if gap <= s.linearization_tolerance and (
                step_norm <= s.stopping_tolerance
                or abs(actual_reduction) <= s.stopping_tolerance * (1 + abs(old_cost))
            ):
                converged = True
                break
            if ratio > 0.75:
                radius = min(s.trust_radius, radius * 1.5)
        diagnostics = self.validate_candidate(initial, states, controls, previous)
        # A small QP residual alone cannot certify the original nonlinear dynamics.
        accepted = (
            diagnostics["feasible"] and np.isfinite(defect) and defect <= s.linearization_tolerance
        )
        diagnostics.update(
            accepted_for_task=accepted,
            linearization_defect=defect,
            linearization_tolerance=s.linearization_tolerance,
            history=history,
            scp_iterations=sum(row["phase"] == "scp" for row in history),
            restoration_attempts=sum(row["phase"] == "feasibility_restoration" for row in history),
            qp_total_iterations=sum(row["qp_iterations"] for row in history),
            termination_reason=reason,
            model_scope="centroidal feasibility only; whole-body feasibility is checked by WBC",
        )
        if not accepted:
            diagnostics["failures"].append("scp_candidate_rejected")
        status = (
            SolveStatus.SUCCESS
            if converged and accepted
            else (SolveStatus.MAX_ITERATIONS if accepted else SolveStatus.NO_FEASIBLE_CANDIDATE)
        )
        result = MPCResult(
            torch.from_numpy(states[None].copy()),
            torch.from_numpy(controls.reshape(1, s.nodes, 12).copy()),
            torch.tensor(
                [self.objective(states, controls, reference, previous)], dtype=torch.float64
            ),
            torch.tensor([int(status)]),
            torch.tensor([diagnostics["scp_iterations"]]),
            torch.tensor([accepted]),
        )
        elapsed = time.perf_counter() - start
        diagnostics["total_time_s"] = elapsed
        return CentroidalPlan(
            result, np.arange(s.nodes + 1) * s.dt, diagnostics, converged, elapsed, defect
        )
