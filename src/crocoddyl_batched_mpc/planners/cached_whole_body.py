"""Private per-worker W model cache; fresh shooting/solver data on every solve."""

import numpy as np

from .standing_whole_body import StandingWholeBodyPlanner


class CachedWholeBodyPlanner(StandingWholeBodyPlanner):
    """Reuse cost/contact models, update every changing target and nominal box.

    This workspace is owned by one worker. Previous shooting problems reference
    these mutable models and must not be evaluated after the next build. Returned
    plans own their arrays. Solvers and their data are freshly constructed, keeping
    numerical initialization identical to the uncached planner.
    """

    def __init__(self, *args, **kwargs):
        self._oracle_data = None
        self._actions = {}
        self._cache_key = None
        self.model_builds = 0
        super().__init__(*args, **kwargs)

    def _oracle_workspace(self):
        if self._oracle_data is None:
            self._oracle_data = self.robot.model.createData()
        return self._oracle_data

    def build_problem(self, *args, **kwargs):
        key = (
            self.settings,
            self.anchors.tobytes(),
            self.base_anchor.tobytes(),
            self.static_control.tobytes(),
            self.torque_limits.tobytes(),
        )
        if key != self._cache_key:
            self._actions.clear()
            self._cache_key = key
        return super().build_problem(*args, **kwargs)

    def _action(self, reference, knot, terminal):
        import crocoddyl as c
        import pinocchio as pin

        key = knot, terminal
        if key not in self._actions:
            self._actions[key] = super()._action(reference, knot, terminal)
            self.model_builds += 1
            return self._actions[key]
        action = self._actions[key]
        costs = action.physical_action.differential.costs.costs
        pose, base = reference.tcp_poses[knot], reference.base_poses[knot]
        for name, value in (
            ("tcp_position", pose[:3, 3]),
            ("tcp_rotation", pose[:3, :3]),
            ("tcp_velocity", pin.Motion(reference.tcp_velocities[knot])),
            ("com", reference.com_positions[knot]),
            ("base_position", base[:3, 3]),
            ("base_rotation", base[:3, :3]),
        ):
            costs[name].cost.residual.reference = value
        nominal = self._native_state(reference.nominal_states[knot])
        costs["posture_velocity"].cost.residual.reference = nominal
        cost = costs["joint_velocity_bounds"].cost
        cost.residual.reference = nominal
        lower, upper = np.full(48, -np.inf), np.full(48, np.inf)
        lower[6:24] = self.robot.model.lowerPositionLimit[7:] - nominal[7:25]
        upper[6:24] = self.robot.model.upperPositionLimit[7:] - nominal[7:25]
        lower[24:] = -self._velocity_limits() - nominal[25:]
        upper[24:] = self._velocity_limits() - nominal[25:]
        cost.activation.bounds = c.ActivationBounds(lower, upper)
        return action
