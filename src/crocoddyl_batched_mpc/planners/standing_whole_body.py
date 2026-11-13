"""Go2+Z1 fixed-support W planner: optional, offline Crocoddyl CPU reference.

Public states/controls use the canonical order of models.go2_arm. Native solver
states use Pinocchio order plus previous torque memory. No simulator, adapter,
WBC, execution fallback or modification of the legacy Go2 path is installed.
"""

import math
import time
from dataclasses import dataclass
from typing import Any

from ..models.go2_arm import load_go2_arm
from ..result import MPCResult, SolveStatus


def _native():
    import crocoddyl as c
    import numpy as np
    import pinocchio as pin

    if c.__version__.startswith("3.2.1") and tuple(
        int(v) for v in pin.__version__.split(".")[:2]
    ) >= (4, 1):
        raise RuntimeError("Crocoddyl 3.2.1 requires the validated Pinocchio 4.0 stack")
    return c, np, pin


@dataclass(frozen=True)
class WholeBodySettings:
    nodes: int = 10
    dt: float = 0.05
    max_iterations: int = 100
    stopping_threshold: float = 1e-7
    friction: float = 0.6
    normal_min: float = 2.0
    normal_max: float = 200.0
    contact_kp: float = 100.0
    contact_kd: float = 20.0
    tcp_position_weight: float = 20000.0
    tcp_rotation_weight: float = 500.0
    tcp_velocity_weight: float = 5.0
    com_weight: float = 100.0
    base_weight: float = 200.0
    posture_weight: float = 1.0
    torque_weight: float = 1e-4
    torque_change_weight: float = 0.01
    barrier_weight: float = 1e5
    base_translation_bound: tuple = (0.08, 0.08, 0.05)
    base_rotation_bound: float = 0.25
    base_linear_velocity_bound: float = 0.5
    base_angular_velocity_bound: float = 1.0
    gap_tolerance: float = 1e-6
    dynamics_tolerance: float = 1e-7
    bound_tolerance: float = 1e-6
    foot_position_tolerance: float = 1e-3
    foot_velocity_tolerance: float = 1e-2
    tcp_position_tolerance: float = 5e-3
    tcp_rotation_tolerance: float = 2e-2

    def __post_init__(self):
        for name in ("nodes", "max_iterations"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name, value in vars(self).items():
            if name in ("nodes", "max_iterations", "base_translation_bound"):
                continue
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        for name in (
            "dt",
            "stopping_threshold",
            "friction",
            "barrier_weight",
            "base_rotation_bound",
            "base_linear_velocity_bound",
            "base_angular_velocity_bound",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.normal_max <= self.normal_min:
            raise ValueError("normal_max must exceed normal_min")
        if len(self.base_translation_bound) != 3 or any(
            not math.isfinite(v) or v <= 0 for v in self.base_translation_bound
        ):
            raise ValueError("base_translation_bound requires three positive values")


@dataclass(frozen=True)
class StandingReference:
    """Discrete world-frame tasks at all N+1 knots; SI units, xyzw state.

    nominal_states are supplied task priors, never the W optimizer's solution.
    A future shared kinematic generator can provide these to C0/C1/W alike.
    tcp_velocities use world linear/angular order at the TCP origin.
    """

    times: Any
    tcp_poses: Any
    tcp_velocities: Any
    com_positions: Any
    base_poses: Any
    nominal_states: Any


def smooth_tcp_reference(robot, settings, displacement=(0.0, 0.0, 0.0), q=None):
    """Quintic translation over one horizon, fixed orientation, zero end speeds.

    COM/base/posture priors remain at the initial stance. This is a small-motion
    test task, not the future common whole-body inverse-kinematics generator.
    """
    _, np, _ = _native()
    q = robot.standing_configuration() if q is None else robot._q(q)
    offset = np.asarray(displacement, dtype=float)
    if offset.shape != (3,) or not np.isfinite(offset).all():
        raise ValueError("displacement must be a finite world vector [3]")
    quantities = robot.evaluate(q)
    n = settings.nodes + 1
    times = np.arange(n) * settings.dt
    s = times / times[-1]
    blend = 10 * s**3 - 15 * s**4 + 6 * s**5
    speed = (30 * s**2 - 60 * s**3 + 30 * s**4) / times[-1]
    poses = np.repeat(quantities.tcp_pose[None], n, axis=0)
    poses[:, :3, 3] += blend[:, None] * offset
    velocity = np.zeros((n, 6))
    velocity[:, :3] = speed[:, None] * offset
    return StandingReference(
        times,
        poses,
        velocity,
        np.repeat(quantities.com[None], n, axis=0),
        np.repeat(robot.frame_pose(q, "base")[None], n, axis=0),
        np.tile(np.r_[q, np.zeros(robot.nv)], (n, 1)),
    )


@dataclass
class WholeBodyPlan:
    """Offline candidate and explicit gate. mpc.action is NOT a WBC/fallback.

    mpc.xs/us are canonical physical states/torques. The raw native augmented
    trajectory is kept separately for checking the memory transition and gaps.
    Feasible MAX_ITERATIONS candidates are usable per MPCResult's existing
    convention; numerical convergence is reported independently.
    """

    mpc: MPCResult
    converged: bool
    solve_time_s: float
    diagnostics: dict
    native_states: Any
    native_controls: Any
    forces_world: Any
    accelerations: Any
    times: Any


class StandingWholeBodyPlanner:
    """One fixed, flat support episode with floating-base and all 18 actuators.

    Torque bounds are Box-FDDP boxes. Force and state barriers are soft; the
    independent post-solve gate decides feasibility, including the terminal
    state with the final torque held. No real-time claim is made.
    """

    def __init__(self, robot=None, settings=None, initial_state=None):
        c, np, pin = _native()
        from ._whole_body_native import TorqueMemoryState

        self.robot = load_go2_arm() if robot is None else robot
        self.settings = WholeBodySettings() if settings is None else settings
        self.physical_state = c.StateMultibody(self.robot.model)
        self.state = TorqueMemoryState(self.physical_state, self.robot.nu)
        x = (
            np.r_[self.robot.standing_configuration(), np.zeros(self.robot.nv)]
            if initial_state is None
            else np.asarray(initial_state, dtype=float).copy()
        )
        if x.shape != (49,) or not np.isfinite(x).all():
            raise ValueError("initial_state must be finite canonical (q25,v24)")
        self.initial_state = x
        self.x0_physical = self._native_state(x)
        self.anchors = self.robot.evaluate(x[:25]).foot_positions
        if np.ptp(self.anchors[:, 2]) > 1e-8:
            raise ValueError("first W reference requires coplanar horizontal foot anchors")
        self.base_anchor = self.robot.frame_pose(x[:25], "base")
        self.base_id = self.robot.model.getFrameId("base", pin.FrameType.BODY)
        self.actuation = c.ActuationModelFloatingBase(self.physical_state)
        self.control_indices = self.robot.v_indices[6:] - 6
        self.torque_limits = self._native_control(self.robot.torque_limits)
        self.contact_names = ("FL", "FR", "RL", "RR")
        # Compute a stationary seed at the episode configuration, then verify it.
        diff = self._differential(c.CostModelSum(self.physical_state, 18))
        stationary = self.x0_physical.copy()
        stationary[25:] = 0
        self.static_control = np.asarray(diff.quasiStatic(diff.createData(), stationary)).copy()
        static = self._independent_node(stationary, self.static_control)
        violations = self._node_violations(stationary, self.static_control, static)
        self.static_diagnostics = {
            "max_acceleration": float(np.max(np.abs(static["acceleration"]))),
            "forces_world": static["forces"].tolist(),
            "torques": self._canonical_control(self.static_control).tolist(),
            "violations": violations,
        }
        if self.static_diagnostics["max_acceleration"] > 1e-7 or any(
            violations[key] > self.settings.bound_tolerance
            for key in ("torque", "friction", "normal", "joint_position", "velocity")
        ):
            raise ValueError(
                "static seed failed support checks; this is not an infeasibility proof"
            )

    def _native_state(self, x):
        _, np, _ = _native()
        return np.r_[self.robot.to_native_q(x[:25]), self.robot.to_native_v(x[25:])]

    def _canonical_state(self, x):
        _, np, _ = _native()
        return np.r_[self.robot.from_native_q(x[:25]), self.robot.from_native_v(x[25:49])]

    def _native_control(self, u):
        _, np, _ = _native()
        out = np.empty(18)
        out[self.robot.v_indices[6:] - 6] = u
        return out

    def _canonical_control(self, u):
        return u[self.robot.v_indices[6:] - 6].copy()

    def _differential(self, costs):
        c, np, pin = _native()
        contacts = c.ContactModelMultiple(self.physical_state, 18)
        for name, frame, anchor in zip(self.contact_names, self.robot.foot_ids, self.anchors):
            contacts.addContact(
                name,
                c.ContactModel3D(
                    self.physical_state,
                    frame,
                    anchor,
                    pin.LOCAL_WORLD_ALIGNED,
                    18,
                    np.array([self.settings.contact_kp, self.settings.contact_kd]),
                ),
            )
        return c.DifferentialActionModelContactFwdDynamics(
            self.physical_state, self.actuation, contacts, costs, 0.0, True
        )

    def _validate_reference(self, reference):
        _, np, _ = _native()
        n = self.settings.nodes + 1
        for name, shape in (
            ("times", (n,)),
            ("tcp_poses", (n, 4, 4)),
            ("tcp_velocities", (n, 6)),
            ("com_positions", (n, 3)),
            ("base_poses", (n, 4, 4)),
            ("nominal_states", (n, 49)),
        ):
            value = np.asarray(getattr(reference, name))
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError(f"{name} must be finite with shape {shape}")
        if not np.allclose(reference.times, np.arange(n) * self.settings.dt, atol=1e-12, rtol=0):
            raise ValueError("reference times must be relative knots 0, dt, ..., N*dt")
        for poses in (reference.tcp_poses, reference.base_poses):
            for pose in poses:
                rotation = pose[:3, :3]
                if (
                    not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-8, rtol=0)
                    or abs(np.linalg.det(rotation) - 1) > 1e-8
                    or not np.allclose(pose[3], [0, 0, 0, 1], atol=1e-12, rtol=0)
                ):
                    raise ValueError("task poses must be rigid SE(3) transforms")
        for x in reference.nominal_states:
            self._native_state(x)

    def _velocity_limits(self):
        _, np, _ = _native()
        s = self.settings
        return np.r_[
            [s.base_linear_velocity_bound] * 3,
            [s.base_angular_velocity_bound] * 3,
            self.robot.model.velocityLimit[6:],
        ]

    def _action(self, reference, knot, terminal):
        c, np, pin = _native()
        from ._whole_body_native import TorqueMemoryAction, WorldForceConeResidual

        s, state = self.settings, self.physical_state
        costs = c.CostModelSum(state, 18)

        def add(name, residual, weight, activation=None):
            cost = (
                c.CostModelResidual(state, residual)
                if activation is None
                else c.CostModelResidual(state, activation, residual)
            )
            costs.addCost(name, cost, weight)

        def barrier(name, residual, lower, upper):
            activation = c.ActivationModelQuadraticBarrier(c.ActivationBounds(lower, upper))
            add(name, residual, s.barrier_weight, activation)

        pose = reference.tcp_poses[knot]
        add(
            "tcp_position",
            c.ResidualModelFrameTranslation(state, self.robot.tcp_id, pose[:3, 3], 18),
            s.tcp_position_weight,
        )
        add(
            "tcp_rotation",
            c.ResidualModelFrameRotation(state, self.robot.tcp_id, pose[:3, :3], 18),
            s.tcp_rotation_weight,
        )
        add(
            "tcp_velocity",
            c.ResidualModelFrameVelocity(
                state,
                self.robot.tcp_id,
                pin.Motion(reference.tcp_velocities[knot]),
                pin.LOCAL_WORLD_ALIGNED,
                18,
            ),
            s.tcp_velocity_weight,
        )
        add(
            "com",
            c.ResidualModelCoMPosition(state, reference.com_positions[knot], 18),
            s.com_weight,
        )
        base = reference.base_poses[knot]
        add(
            "base_position",
            c.ResidualModelFrameTranslation(state, self.base_id, base[:3, 3], 18),
            s.base_weight,
        )
        add(
            "base_rotation",
            c.ResidualModelFrameRotation(state, self.base_id, base[:3, :3], 18),
            s.base_weight,
        )
        nominal = self._native_state(reference.nominal_states[knot])
        weights = np.r_[
            np.zeros(6),
            self._native_control(np.r_[np.ones(12), np.full(6, 0.1)]),
            np.full(6, 0.1),
            np.full(18, 0.05),
        ]
        add(
            "posture_velocity",
            c.ResidualModelState(state, nominal, 18),
            s.posture_weight,
            c.ActivationModelWeightedQuad(weights),
        )
        # Native state difference is manifold-valued. Only scalar joints and
        # velocities use its box; base world translation/rotation get own residuals.
        lower, upper = np.full(48, -np.inf), np.full(48, np.inf)
        lower[6:24] = self.robot.model.lowerPositionLimit[7:] - nominal[7:25]
        upper[6:24] = self.robot.model.upperPositionLimit[7:] - nominal[7:25]
        lower[24:] = -self._velocity_limits() - nominal[25:]
        upper[24:] = self._velocity_limits() - nominal[25:]
        barrier("joint_velocity_bounds", c.ResidualModelState(state, nominal, 18), lower, upper)
        bounds = np.array(s.base_translation_bound)
        barrier(
            "base_translation_bounds",
            c.ResidualModelFrameTranslation(state, self.base_id, self.base_anchor[:3, 3], 18),
            -bounds,
            bounds,
        )
        angles = np.full(3, s.base_rotation_bound)
        barrier(
            "base_rotation_bounds",
            c.ResidualModelFrameRotation(state, self.base_id, self.base_anchor[:3, :3], 18),
            -angles,
            angles,
        )
        if not terminal:
            add("torque", c.ResidualModelControl(state, self.static_control), s.torque_weight)
            for name in self.contact_names:
                barrier(
                    name + "_force",
                    WorldForceConeResidual(state, name, 18, s.friction),
                    np.full(6, -np.inf),
                    np.array([0, 0, 0, 0, -s.normal_min, s.normal_max]),
                )
        action = c.IntegratedActionModelEuler(self._differential(costs), 0 if terminal else s.dt)
        action.u_lb, action.u_ub = -self.torque_limits, self.torque_limits
        return TorqueMemoryAction(self.state, action, s.torque_change_weight, s.dt)

    def build_problem(self, reference=None, previous_torque=None):
        """Return native augmented shooting problem; previous_torque is canonical."""
        c, np, _ = _native()
        reference = (
            smooth_tcp_reference(self.robot, self.settings, q=self.initial_state[:25])
            if reference is None
            else reference
        )
        self._validate_reference(reference)
        previous = self.static_control.copy()
        if previous_torque is not None:
            values = np.asarray(previous_torque, dtype=float)
            if values.shape != (18,) or not np.isfinite(values).all():
                raise ValueError("previous_torque must be finite canonical [18]")
            previous = self._native_control(values)
        running = [self._action(reference, i, False) for i in range(self.settings.nodes)]
        terminal = self._action(reference, self.settings.nodes, True)
        return c.ShootingProblem(np.r_[self.x0_physical, previous], running, terminal)

    def _independent_node(self, x, u):
        """Dense Pinocchio KKT oracle, independent of Crocoddyl contact data."""
        _, np, pin = _native()
        model, s = self.robot.model, self.settings
        q, v = x[:25], x[25:49]
        data = model.createData()
        mass = pin.crba(model, data, q).copy()
        mass = np.triu(mass) + np.triu(mass, 1).T
        h = pin.nonLinearEffects(model, data, q, v).copy()
        pin.forwardKinematics(model, data, q, v, np.zeros(24))
        pin.computeJointJacobians(model, data, q)
        pin.updateFramePlacements(model, data)
        jac, drift, positions, raw_bias = [], [], [], []
        for frame, anchor in zip(self.robot.foot_ids, self.anchors):
            j = pin.getFrameJacobian(model, data, frame, pin.LOCAL_WORLD_ALIGNED)[:3].copy()
            position = data.oMf[frame].translation.copy()
            bias = pin.getFrameClassicalAcceleration(
                model, data, frame, pin.LOCAL_WORLD_ALIGNED
            ).linear.copy()
            jac.append(j)
            raw_bias.append(bias)
            drift.append(bias + s.contact_kp * (position - anchor) + s.contact_kd * (j @ v))
            positions.append(position)
        jac, drift = np.vstack(jac), np.concatenate(drift)
        actuation = np.r_[np.zeros(6), u]
        matrix = np.block([[mass, -jac.T], [jac, np.zeros((12, 12))]])
        solution = np.linalg.solve(matrix, np.r_[actuation - h, -drift])
        a, force = solution[:24], solution[24:]
        return dict(
            acceleration=a,
            forces=force.reshape(4, 3),
            mass=mass,
            h=h,
            jacobian=jac,
            drift=drift,
            foot_positions=np.array(positions),
            foot_velocities=(jac @ v).reshape(4, 3),
            dynamics_residual=mass @ a + h - actuation - jac.T @ force,
            contact_residual=jac @ a + drift,
            unstabilized_contact_acceleration=jac @ a + np.concatenate(raw_bias),
        )

    def _node_violations(self, x, u, node):
        _, np, pin = _native()
        s, model = self.settings, self.robot.model

        def excess(value):
            return float(max(0, np.max(value)))

        force = node["forces"]
        base = self.robot.frame_pose(self.robot.from_native_q(x[:25]), "base")
        return {
            "torque": excess(np.abs(u) - self.torque_limits),
            "friction": excess(
                np.abs(force[:, 0]) + np.abs(force[:, 1]) - s.friction * force[:, 2]
            ),
            "normal": max(excess(s.normal_min - force[:, 2]), excess(force[:, 2] - s.normal_max)),
            "joint_position": max(
                excess(model.lowerPositionLimit[7:] - x[7:25]),
                excess(x[7:25] - model.upperPositionLimit[7:]),
            ),
            "velocity": excess(np.abs(x[25:49]) - self._velocity_limits()),
            "base_translation": excess(
                np.abs(base[:3, 3] - self.base_anchor[:3, 3]) - np.array(s.base_translation_bound)
            ),
            "base_rotation": excess(
                np.abs(pin.log3(self.base_anchor[:3, :3].T @ base[:3, :3])) - s.base_rotation_bound
            ),
        }

    def validate_candidate(self, problem, states, controls, reference):
        """Check every knot using independent KKT forces and manifold integration.

        Terminal support/force checks use the last torque held. Candidate gaps
        include memory and the initial condition. This checks sampled rigid-body
        predictions only; there is no collision, hardware or PhysX certificate.
        """
        _, np, pin = _native()
        s = self.settings
        xs, us = np.asarray(states), np.asarray(controls)
        if xs.shape != (s.nodes + 1, 67) or us.shape != (s.nodes, 18):
            raise ValueError("candidate dimensions must match the augmented problem")
        if not np.isfinite(xs).all() or not np.isfinite(us).all():
            return {"finite": False, "feasible": False, "failures": ["nonfinite"]}, None, None
        forces, accelerations, nodes, per_node, gaps, tcp_errors, rotation_errors = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )
        try:
            quaternion_error = float(np.max(np.abs(np.sum(xs[:, 3:7] ** 2, axis=1) - 1)))
            if quaternion_error > 1e-8:
                return {"finite": True, "feasible": False, "failures": ["quaternion"]}, None, None
            initial_gap = float(np.max(np.abs(self.state.diff(problem.x0, xs[0]))))
            for k, x in enumerate(xs):
                u = us[min(k, s.nodes - 1)]
                node = self._independent_node(x, u)
                if not all(np.isfinite(value).all() for value in node.values()):
                    return (
                        {"finite": False, "feasible": False, "failures": ["nonfinite_dynamics"]},
                        None,
                        None,
                    )
                nodes.append(node)
                forces.append(node["forces"])
                accelerations.append(self.robot.from_native_v(node["acceleration"]))
                per_node.append(self._node_violations(x, u, node))
                actual = self.robot.frame_pose(self.robot.from_native_q(x[:25]), "z1_tcp")
                target = reference.tcp_poses[k]
                tcp_errors.append(float(np.linalg.norm(actual[:3, 3] - target[:3, 3])))
                rotation_errors.append(
                    float(np.linalg.norm(pin.log3(target[:3, :3].T @ actual[:3, :3])))
                )
                if k < s.nodes:
                    a = node["acceleration"]
                    delta = np.r_[s.dt * x[25:49] + s.dt**2 * a, s.dt * a]
                    predicted = np.r_[self.physical_state.integrate(x[:49], delta), u]
                    gaps.append(float(np.max(np.abs(self.state.diff(predicted, xs[k + 1])))))
        except (ValueError, np.linalg.LinAlgError, RuntimeError) as exc:
            return (
                {
                    "finite": True,
                    "feasible": False,
                    "failures": ["validation_error"],
                    "error": str(exc),
                },
                None,
                None,
            )
        maxima = {name: max(item[name] for item in per_node) for name in per_node[0]}
        gap = max(initial_gap, max(gaps))
        dyn = max(float(np.max(np.abs(n["dynamics_residual"]))) for n in nodes)
        contact = max(float(np.max(np.abs(n["contact_residual"]))) for n in nodes)
        foot_pos = max(
            float(np.max(np.linalg.norm(n["foot_positions"] - self.anchors, axis=1))) for n in nodes
        )
        foot_vel = max(float(np.max(np.linalg.norm(n["foot_velocities"], axis=1))) for n in nodes)
        failures = [name for name, value in maxima.items() if value > s.bound_tolerance]
        for name, value, limit in (
            ("dynamics_gap", gap, s.gap_tolerance),
            ("dynamics_residual", dyn, s.dynamics_tolerance),
            ("contact_residual", contact, s.dynamics_tolerance),
            ("foot_position", foot_pos, s.foot_position_tolerance),
            ("foot_velocity", foot_vel, s.foot_velocity_tolerance),
        ):
            if value > limit:
                failures.append(name)
        tracking_passed = (
            max(tcp_errors) <= s.tcp_position_tolerance
            and max(rotation_errors) <= s.tcp_rotation_tolerance
        )
        return (
            {
                "finite": True,
                "feasible": not failures,
                "failures": failures,
                "tracking_passed": tracking_passed,
                "accepted_for_task": not failures and tracking_passed,
                "max_violations": maxima,
                "per_node_violations": per_node,
                "initial_gap": initial_gap,
                "max_dynamics_gap": gap,
                "node_gaps": gaps,
                "max_dynamics_residual": dyn,
                "max_contact_residual": contact,
                "max_unstabilized_contact_acceleration": max(
                    float(np.max(np.abs(n["unstabilized_contact_acceleration"]))) for n in nodes
                ),
                "max_foot_position_error_m": foot_pos,
                "max_foot_velocity_m_s": foot_vel,
                "tcp_position_rmse_m": float(np.sqrt(np.mean(np.square(tcp_errors)))),
                "tcp_position_peak_m": max(tcp_errors),
                "tcp_position_errors_m": tcp_errors,
                "tcp_rotation_peak_rad": max(rotation_errors),
                "min_normal_force_N": float(np.min(np.asarray(forces)[:, :, 2])),
                "max_normal_force_N": float(np.max(np.asarray(forces)[:, :, 2])),
                "max_torque_fraction": float(np.max(np.abs(us) / self.torque_limits)),
            },
            np.array(forces),
            np.array(accelerations),
        )

    def solve(self, reference=None, previous_torque=None, *, initial_guess=None):
        """Solve one offline horizon, optionally seeded by canonical (states, torques).

        The optional seed may have dynamics gaps; FDDP must close them and the
        same independent acceptance gates apply. The legacy rollout seed is unchanged.
        """
        c, np, _ = _native()
        import torch

        total_started = time.perf_counter()
        reference = (
            smooth_tcp_reference(self.robot, self.settings, q=self.initial_state[:25])
            if reference is None
            else reference
        )
        problem = self.build_problem(reference, previous_torque)
        solver = c.SolverBoxFDDP(problem)
        solver.th_stop = self.settings.stopping_threshold
        seed_u = [self.static_control.copy() for _ in range(self.settings.nodes)]
        seed_x = None
        if initial_guess is not None:
            if not isinstance(initial_guess, (tuple, list)) or len(initial_guess) != 2:
                raise ValueError("initial_guess requires canonical states and torques")
            guessed_x, guessed_u = (np.asarray(value, dtype=float) for value in initial_guess)
            if (
                guessed_x.shape != (self.settings.nodes + 1, 49)
                or guessed_u.shape != (self.settings.nodes, 18)
                or not np.isfinite(guessed_x).all()
                or not np.isfinite(guessed_u).all()
            ):
                raise ValueError("initial_guess has invalid dimensions or nonfinite values")
            if not np.allclose(guessed_x[0], self.initial_state, atol=1e-9, rtol=0):
                raise ValueError("initial_guess must start at the episode state")
            for x in guessed_x:
                self.robot._q(x[:25])
            seed_u = [self._native_control(u) for u in guessed_u]
            seed_x = [
                np.r_[self._native_state(x), problem.x0[49:] if k == 0 else seed_u[k - 1]]
                for k, x in enumerate(guessed_x)
            ]
        started = time.perf_counter()
        error = None
        try:
            if seed_x is None:
                seed_x = problem.rollout(seed_u)
            converged = bool(
                solver.solve(seed_x, seed_u, self.settings.max_iterations, initial_guess is None)
            )
            xs, us = np.array(solver.xs), np.array(solver.us)
            cost = float(solver.cost)
            iterations = min(int(solver.iter) + 1, self.settings.max_iterations)
        except (RuntimeError, ValueError, np.linalg.LinAlgError) as exc:
            error = str(exc)
            converged, cost, iterations = False, float("nan"), 0
            xs = np.full((self.settings.nodes + 1, 67), np.nan)
            us = np.full((self.settings.nodes, 18), np.nan)
        elapsed = time.perf_counter() - started
        diagnostics, forces, accelerations = self.validate_candidate(problem, xs, us, reference)
        if error is not None or not math.isfinite(cost) or not diagnostics["finite"]:
            status = SolveStatus.NUMERICAL_FAILURE
        elif not diagnostics["feasible"]:
            status = SolveStatus.NO_FEASIBLE_CANDIDATE
        else:
            status = SolveStatus.SUCCESS if converged else SolveStatus.MAX_ITERATIONS
        diagnostics["solver_error"] = error
        diagnostics["initial_guess"] = (
            "canonical_trajectory" if initial_guess is not None else "static_control_rollout"
        )
        diagnostics["static_seed"] = self.static_diagnostics
        diagnostics["problem_build_time_s"] = started - total_started
        diagnostics["validation_time_s"] = time.perf_counter() - started - elapsed
        # Invalid quaternions must never be sent through Pinocchio conversion.
        valid_states = np.isfinite(xs).all() and np.all(
            np.abs(np.sum(xs[:, 3:7] ** 2, axis=1) - 1) <= 1e-8
        )
        physical = (
            np.stack([self._canonical_state(x) for x in xs])
            if valid_states
            else np.full((self.settings.nodes + 1, 49), np.nan)
        )
        controls = np.stack([self._canonical_control(u) for u in us])
        if status == SolveStatus.NUMERICAL_FAILURE:
            physical[:] = np.nan
            controls[:] = np.nan
            cost = float("nan")
        result = MPCResult(
            torch.tensor(physical[None], dtype=torch.float64),
            torch.tensor(controls[None], dtype=torch.float64),
            torch.tensor([cost], dtype=torch.float64),
            torch.tensor([int(status)]),
            torch.tensor([iterations]),
            torch.tensor([diagnostics["feasible"]]),
        )
        diagnostics["total_time_s"] = time.perf_counter() - total_started
        return WholeBodyPlan(
            result,
            converged,
            elapsed,
            diagnostics,
            xs,
            us,
            forces,
            accelerations,
            reference.times.copy(),
        )
