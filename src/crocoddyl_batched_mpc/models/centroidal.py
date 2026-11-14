"""Fixed-contact centroidal C0 dynamics; world [COM, linear, angular momentum].

Mass includes the complete robot. There is no configuration/inertia prediction,
hand wrench or joint trajectory in this model. Forces are FL/FR/RL/RR world xyz.
"""

from dataclasses import dataclass

import numpy as np

from ..guidance import _array


def skew(v):
    x, y, z = v
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


@dataclass(frozen=True)
class CentroidalModel:
    mass: float
    gravity: np.ndarray
    anchors: np.ndarray

    def __post_init__(self):
        if not np.isfinite(self.mass) or self.mass <= 0:
            raise ValueError("mass must be finite and positive")
        object.__setattr__(self, "gravity", _array(self.gravity, (3,), "gravity"))
        object.__setattr__(self, "anchors", _array(self.anchors, (4, 3), "anchors"))
        if np.ptp(self.anchors[:, 2]) > 1e-8:
            raise ValueError("C0 requires fixed horizontal coplanar contacts")

    def derivative(self, x, forces):
        x = _array(x, (9,), "centroidal state")
        f = _array(forces, (4, 3), "contact forces")
        return np.r_[
            x[3:6] / self.mass,
            self.mass * self.gravity + f.sum(axis=0),
            np.cross(self.anchors - x[:3], f).sum(axis=0),
        ]

    def step(self, x, forces, dt):
        """Exact COM/linear-momentum ZOH step, explicit Euler angular momentum."""
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be positive")
        derivative = self.derivative(x, forces)
        result = np.asarray(x) + dt * derivative
        result[:3] += 0.5 * dt**2 * derivative[3:6] / self.mass
        return result

    def linearize(self, x, forces, dt):
        """Analytic A/B of step; the sole bilinear term is -COM cross sum(force)."""
        x = _array(x, (9,), "centroidal state")
        f = _array(forces, (4, 3), "contact forces")
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be positive")
        A, B = np.eye(9), np.zeros((9, 12))
        A[:3, 3:6] = dt / self.mass * np.eye(3)
        A[6:, :3] = dt * skew(f.sum(axis=0))
        for foot in range(4):
            cols = slice(3 * foot, 3 * foot + 3)
            B[:3, cols] = 0.5 * dt**2 / self.mass * np.eye(3)
            B[3:6, cols] = dt * np.eye(3)
            B[6:, cols] = dt * skew(self.anchors[foot] - x[:3])
        return A, B

    def rollout(self, initial, forces, dt):
        x = _array(initial, (9,), "centroidal initial state")
        controls = np.asarray(forces)
        if controls.ndim != 3 or controls.shape[1:] != (4, 3) or not len(controls):
            raise ValueError("forces must be a nonempty (N,4,3) trajectory")
        states = [x.copy()]
        for u in controls:
            states.append(self.step(states[-1], u, dt))
        return np.stack(states)
