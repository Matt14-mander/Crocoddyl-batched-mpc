"""Isolated CPU walking/curve demo; scheduled contacts, full Go2+Z1 dynamics.

This is an offline hybrid OCP and feedback replay, not a receding-horizon MPC or
PhysX simulation. The standing planner and legacy Go2 execution path are untouched.
"""

import math
import time
from dataclasses import dataclass
from types import SimpleNamespace

import crocoddyl as c
import numpy as np
import pinocchio as pin

from ..models.go2_arm import load_go2_arm
from .standing_whole_body import StandingWholeBodyPlanner, WholeBodySettings

LEGS = ("FL", "FR", "RL", "RR")


class MemoryImpulse(c.ActionModelAbstract):
    """Zero-time perfectly inelastic impact; previous torque memory is unchanged."""

    def __init__(self, state, physical):
        super().__init__(state, 0, 0)
        self.physical_action = physical

    def createData(self):
        data = c.ActionDataAbstract(self)
        data.physical = self.physical_action.createData()
        return data

    def calc(self, data, x, u=None):
        self.physical_action.calc(data.physical, x[:49])
        data.xnext[:] = np.r_[data.physical.xnext, x[49:]]
        data.cost = data.physical.cost

    def calcDiff(self, data, x, u=None):
        self.physical_action.calcDiff(data.physical, x[:49])
        data.Fx[:] = np.eye(66)
        data.Fx[:48, :48] = data.physical.Fx
        data.Lx[:] = 0
        data.Lxx[:] = 0
        data.Lx[:48] = data.physical.Lx
        data.Lxx[:48, :48] = data.physical.Lxx


class ImpulseCone(c.ResidualModelAbstract):
    def __init__(self, state, name, mu):
        super().__init__(state, 5, 0, True, True, False)
        self.name = name
        self.matrix = np.array([[1, 1, -mu], [1, -1, -mu], [-1, 1, -mu], [-1, -1, -mu], [0, 0, -1]])

    def createData(self, shared):
        data = c.ResidualDataAbstract(self, shared)
        data.impulse = shared.impulses.impulses[self.name]
        return data

    def calc(self, data, x, u=None):
        data.r[:] = self.matrix @ data.impulse.f.linear

    def calcDiff(self, data, x, u=None):
        data.Rx[:] = self.matrix @ data.impulse.df_dx[:3]


@dataclass(frozen=True)
class WalkingDemoSettings:
    cycles: int = 2
    dt: float = 0.025
    step_length: float = 0.08
    step_height: float = 0.04
    swing_nodes: int = 20
    transfer_nodes: int = 14
    hold_nodes: int = 20
    curve_y: float = 0.04
    curve_z: float = 0.06
    iterations: int = 250

    def __post_init__(self):
        for name in ("cycles", "swing_nodes", "transfer_nodes", "hold_nodes", "iterations"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("dt", "step_length", "step_height", "curve_y", "curve_z"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if abs(round(1 / self.dt) * self.dt - 1) > 1e-9:
            raise ValueError("dt must correspond to an integer Meshcat frame rate")


def _blend(s):
    return 10 * s**3 - 15 * s**4 + 6 * s**5


class WalkingWholeBodyDemo(StandingWholeBodyPlanner):
    def __init__(self, robot=None, config=None):
        self.config = WalkingDemoSettings() if config is None else config
        self.phase = None
        settings = WholeBodySettings(
            dt=self.config.dt,
            max_iterations=self.config.iterations,
            com_weight=2000,
            tcp_position_weight=40000,
            tcp_rotation_weight=5000,
            barrier_weight=1e6,
            normal_min=2.0,
            base_translation_bound=(0.1, 0.1, 0.07),
        )
        robot = load_go2_arm() if robot is None else robot
        initial_q = robot.standing_configuration()
        # Task-specific arm preparation keeps a world-fixed drawing plane reachable
        # throughout forward motion; the asset/default standing model is unchanged.
        initial_q[19:] = [0, 1.6, -1.6, 0, 0, 0]
        super().__init__(robot, settings, np.r_[initial_q, np.zeros(24)])
        self.initial_anchors = self.anchors.copy()
        self.initial_base = self.base_anchor.copy()
        self.schedule = self._schedule()
        self.reference, self.ik_errors = self._reference()

    def _differential(self, costs):
        if self.phase is None:
            return super()._differential(costs)
        contacts = c.ContactModelMultiple(self.physical_state, 18)
        for i in np.flatnonzero(self.phase["contacts"]):
            contacts.addContact(
                LEGS[i],
                c.ContactModel3D(
                    self.physical_state,
                    self.robot.foot_ids[i],
                    self.phase["anchors"][i],
                    pin.LOCAL_WORLD_ALIGNED,
                    18,
                    np.array([self.settings.contact_kp, self.settings.contact_kd]),
                ),
            )
        return c.DifferentialActionModelContactFwdDynamics(
            self.physical_state, self.actuation, contacts, costs, 0.0, True
        )

    def _velocity_limits(self):
        # Guide the soft barrier inside the physical acceptance bounds.
        return super()._velocity_limits() - 0.02

    def _schedule(self):
        cfg = self.config
        anchors = self.initial_anchors.copy()
        initial_com = self.robot.evaluate(self.initial_state[:25]).com
        rows, clock = [], 0.0
        com = initial_com.copy()

        def append(kind, mask, feet, target_com, swing=-1):
            nonlocal clock
            rows.append(
                dict(
                    kind=kind,
                    time=clock,
                    contacts=np.array(mask, dtype=bool),
                    anchors=anchors.copy(),
                    feet=feet.copy(),
                    com=target_com.copy(),
                    swing=swing,
                    dt=0.0 if kind == "impact" else cfg.dt,
                )
            )
            if kind != "impact":
                clock += cfg.dt

        for _ in range(cfg.hold_nodes):
            append("flow", [True] * 4, anchors, com)
        for _ in range(cfg.cycles):
            for leg in (3, 1, 2, 0):
                mask = np.ones(4, dtype=bool)
                mask[leg] = False
                mean = anchors.mean(axis=0)
                support = anchors[mask].mean(axis=0)
                target = initial_com.copy()
                target[:2] = 0.45 * mean[:2] + 0.55 * support[:2]
                previous = com.copy()
                for k in range(cfg.transfer_nodes):
                    com = previous + _blend((k + 1) / cfg.transfer_nodes) * (target - previous)
                    append("flow", [True] * 4, anchors, com)
                start = anchors[leg].copy()
                for k in range(cfg.swing_nodes):
                    phase = k / cfg.swing_nodes
                    feet = anchors.copy()
                    feet[leg, 0] += cfg.step_length * _blend(phase)
                    feet[leg, 2] += cfg.step_height * 64 * phase**3 * (1 - phase) ** 3
                    append("flow", mask, feet, com, swing=leg)
                anchors[leg] = start + [cfg.step_length, 0, 0]
                append("impact", [True] * 4, anchors, com, swing=leg)
        final_com = initial_com + [cfg.cycles * cfg.step_length, 0, 0]
        previous = com.copy()
        for k in range(cfg.hold_nodes):
            com = previous + _blend((k + 1) / cfg.hold_nodes) * (final_com - previous)
            append("flow", [True] * 4, anchors, com)
        append("terminal", [True] * 4, anchors, final_com)
        rows[-1]["dt"] = 0.0
        return rows

    def _ik(self, q, base_position, feet, tcp_pose):
        model = self.robot.model
        q = q.copy()
        q[:3] = base_position
        q[3:7] = [0, 0, 0, 1]
        data = model.createData()
        error = np.zeros(18)
        for _ in range(60):
            pin.computeJointJacobians(model, data, q)
            pin.updateFramePlacements(model, data)
            errors, jacobians = [], []
            for frame, target in zip(self.robot.foot_ids, feet):
                errors.append(target - data.oMf[frame].translation)
                jacobians.append(
                    pin.getFrameJacobian(model, data, frame, pin.LOCAL_WORLD_ALIGNED)[:3, 6:]
                )
            actual = data.oMf[self.robot.tcp_id]
            errors.extend(
                [
                    tcp_pose[:3, 3] - actual.translation,
                    -pin.log3(actual.rotation @ tcp_pose[:3, :3].T),
                ]
            )
            jacobians.append(
                pin.getFrameJacobian(model, data, self.robot.tcp_id, pin.LOCAL_WORLD_ALIGNED)[:, 6:]
            )
            error = np.concatenate(errors)
            if np.max(np.abs(error)) < 1e-8:
                break
            jac = np.vstack(jacobians)
            dq = np.linalg.solve(jac.T @ jac + 1e-6 * np.eye(18), jac.T @ error)
            dq *= min(1, 0.15 / max(np.max(np.abs(dq)), 1e-12))
            q[7:] = np.clip(
                q[7:] + dq, model.lowerPositionLimit[7:] + 1e-6, model.upperPositionLimit[7:] - 1e-6
            )
        return q, float(np.max(np.abs(error)))

    def _reference(self):
        rows = self.schedule
        times = np.array([r["time"] for r in rows])
        count = len(rows)
        original = self.robot.evaluate(self.initial_state[:25])
        poses = np.repeat(original.tcp_pose[None], count, axis=0)
        hold = self.config.hold_nodes * self.config.dt
        duration = times[-1] - 2 * hold
        progress = np.clip((times - hold) / duration, 0, 1)
        theta = 2 * np.pi * _blend(progress)
        theta_dot = 2 * np.pi * (30 * progress**2 - 60 * progress**3 + 30 * progress**4) / duration
        poses[:, 1, 3] += self.config.curve_y * np.sin(2 * theta)
        poses[:, 2, 3] += self.config.curve_z * np.sin(theta)
        velocity = np.zeros((count, 6))
        velocity[:, 1] = 2 * self.config.curve_y * np.cos(2 * theta) * theta_dot
        velocity[:, 2] = self.config.curve_z * np.cos(theta) * theta_dot
        base = np.repeat(self.initial_base[None], count, axis=0)
        coms = np.stack([r["com"] for r in rows])
        base[:, :3, 3] += coms - original.com
        q = self.x0_physical[:25].copy()
        nominal, errors = [], []
        for row, pose, base_pose in zip(rows, poses, base):
            q, error = self._ik(q, base_pose[:3, 3], row["feet"], pose)
            nominal.append(self._canonical_state(np.r_[q, np.zeros(24)]))
            errors.append(error)
        # Local finite differences are only a warm start, never the executed trajectory.
        nominal = np.array(nominal)
        for k in range(1, count - 1):
            h = times[k + 1] - times[k - 1]
            if h > 0:
                v = (
                    pin.difference(
                        self.robot.model,
                        self.robot.to_native_q(nominal[k - 1, :25]),
                        self.robot.to_native_q(nominal[k + 1, :25]),
                    )
                    / h
                )
                nominal[k, 25:] = self.robot.from_native_v(v)
        nominal[0] = self.initial_state
        return SimpleNamespace(
            times=times,
            tcp_poses=poses,
            tcp_velocities=velocity,
            base_poses=base,
            com_positions=coms,
            nominal_states=nominal,
        ), errors

    def _impact(self, row):
        state = self.physical_state
        impulses = c.ImpulseModelMultiple(state)
        costs = c.CostModelSum(state, 0)
        for i, frame in enumerate(self.robot.foot_ids):
            impulses.addImpulse(LEGS[i], c.ImpulseModel3D(state, frame, pin.LOCAL_WORLD_ALIGNED))
            translation = c.ResidualModelFrameTranslation(state, frame, row["anchors"][i], 0)
            costs.addCost(LEGS[i] + "_landing", c.CostModelResidual(state, translation), 1e9)
            velocity = c.ResidualModelFrameVelocity(
                state, frame, pin.Motion.Zero(), pin.LOCAL_WORLD_ALIGNED, 0
            )
            costs.addCost(LEGS[i] + "_landing_velocity", c.CostModelResidual(state, velocity), 1e3)
            barrier = c.ActivationModelQuadraticBarrier(
                c.ActivationBounds(np.full(5, -np.inf), np.zeros(5))
            )
            costs.addCost(
                LEGS[i] + "_impulse_cone",
                c.CostModelResidual(
                    state, barrier, ImpulseCone(state, LEGS[i], self.settings.friction)
                ),
                1e7,
            )
        physical = c.ActionModelImpulseFwdDynamics(state, impulses, costs, 0.0, 1e-12, True)
        return MemoryImpulse(self.state, physical)

    def build_walking_problem(self):
        models, controls, guesses = [], [], []
        previous = self.static_control.copy()
        for k, row in enumerate(self.schedule):
            self.phase = row
            self.contact_names = tuple(LEGS[i] for i in np.flatnonzero(row["contacts"]))
            self.base_anchor = self.reference.base_poses[k]
            x = self._native_state(self.reference.nominal_states[k])
            guesses.append(np.r_[x, previous])
            if row["kind"] == "impact":
                action = self._impact(row)
                control = np.zeros(0)
            else:
                action = self._action(self.reference, k, row["kind"] == "terminal")
                costs = action.physical_action.differential.costs
                for i, foot in enumerate(self.robot.foot_ids):
                    # Semi-implicit rigid point contacts have O(dt^2) geometric drift.
                    # Permit 0.5 mm in this soft cost; independently gate at 1 mm.
                    floor = c.ActivationModelQuadraticBarrier(
                        c.ActivationBounds(
                            np.array([-np.inf, -np.inf, -0.0005]), np.full(3, np.inf)
                        )
                    )
                    costs.addCost(
                        LEGS[i] + "_floor",
                        c.CostModelResidual(
                            self.physical_state,
                            floor,
                            c.ResidualModelFrameTranslation(
                                self.physical_state, foot, np.zeros(3), 18
                            ),
                        ),
                        1e8,
                    )
                if row["swing"] >= 0:
                    i = row["swing"]
                    costs.addCost(
                        "swing_position",
                        c.CostModelResidual(
                            self.physical_state,
                            c.ResidualModelFrameTranslation(
                                self.physical_state, self.robot.foot_ids[i], row["feet"][i], 18
                            ),
                        ),
                        1e7,
                    )
                stationary = x.copy()
                stationary[25:] = 0
                diff = action.physical_action.differential
                control = np.clip(
                    diff.quasiStatic(diff.createData(), stationary),
                    -self.torque_limits,
                    self.torque_limits,
                )
                previous = control.copy()
            models.append(action)
            if row["kind"] != "terminal":
                controls.append(control)
        self.phase = None
        self.contact_names = LEGS
        self.base_anchor = self.initial_base.copy()
        problem = c.ShootingProblem(
            np.r_[self.x0_physical, self.static_control], models[:-1], models[-1]
        )
        guesses[0] = problem.x0.copy()
        return problem, guesses, controls

    def solve_walking(self, verbose=False):
        problem, xs, us = self.build_walking_problem()
        solver = c.SolverBoxFDDP(problem)
        solver.th_stop = 1e-6
        if verbose:
            solver.setCallbacks([c.CallbackVerbose()])
        started = time.perf_counter()
        converged = solver.solve(xs, us, self.config.iterations, False)
        return dict(
            solver=solver,
            problem=problem,
            converged=bool(converged),
            solve_time_s=time.perf_counter() - started,
        )

    def oracle(self, x, u, row):
        """Independent dense Pinocchio KKT, including zero-time impact momentum."""
        model = self.robot.model
        q, v = x[:25], x[25:49]
        d = model.createData()
        mass = pin.crba(model, d, q).copy()
        mass = np.triu(mass) + np.triu(mass, 1).T
        h = pin.nonLinearEffects(model, d, q, v).copy()
        pin.forwardKinematics(model, d, q, v, np.zeros(24))
        pin.computeJointJacobians(model, d, q)
        pin.updateFramePlacements(model, d)
        feet = np.stack([d.oMf[f].translation.copy() for f in self.robot.foot_ids])
        jac_all = np.stack(
            [
                pin.getFrameJacobian(model, d, f, pin.LOCAL_WORLD_ALIGNED)[:3].copy()
                for f in self.robot.foot_ids
            ]
        )
        active = np.flatnonzero(row["contacts"])
        jac = jac_all[active].reshape(-1, 24)
        mat = np.block([[mass, -jac.T], [jac, np.zeros((3 * len(active), 3 * len(active)))]])
        if row["kind"] == "impact":
            sol = np.linalg.solve(mat, np.r_[mass @ v, np.zeros(jac.shape[0])])
            xnext = np.r_[q, sol[:24], x[49:]]
            residual = mass @ (sol[:24] - v) - jac.T @ sol[24:]
            contact = jac @ sol[:24]
        else:
            drift = np.concatenate(
                [
                    pin.getFrameClassicalAcceleration(
                        model, d, self.robot.foot_ids[i], pin.LOCAL_WORLD_ALIGNED
                    ).linear.copy()
                    + self.settings.contact_kp * (feet[i] - row["anchors"][i])
                    + self.settings.contact_kd * (jac_all[i] @ v)
                    for i in active
                ]
            )
            act = np.r_[np.zeros(6), u]
            sol = np.linalg.solve(mat, np.r_[act - h, -drift])
            dt = row["dt"]
            xnext = np.r_[
                self.physical_state.integrate(
                    x[:49], np.r_[dt * v + dt**2 * sol[:24], dt * sol[:24]]
                ),
                u,
            ]
            residual = mass @ sol[:24] + h - act - jac.T @ sol[24:]
            contact = jac @ sol[:24] + drift
        forces = np.zeros((4, 3))
        forces[active] = sol[24:].reshape(-1, 3)
        return dict(
            xnext=xnext,
            forces=forces,
            feet=feet,
            foot_velocities=jac_all @ v,
            dynamics_residual=residual,
            contact_residual=contact,
            tcp=d.oMf[self.robot.tcp_id].homogeneous.copy(),
            com=pin.centerOfMass(model, d, q).copy(),
        )

    def audit(self, xs, us):
        if len(xs) != len(self.schedule) or len(us) != len(xs) - 1:
            raise ValueError("Trace length does not match contact schedule")
        if any(np.shape(x) != (67,) or not np.isfinite(x).all() for x in xs):
            raise ValueError("Expected finite augmented native states of shape (67,)")
        for row, u in zip(self.schedule, us):
            expected = (0,) if row["kind"] == "impact" else (18,)
            if np.shape(u) != expected or not np.isfinite(u).all():
                raise ValueError("Invalid control dimensions or nonfinite controls")
        metrics = dict(
            gap=0.0,
            dynamics_residual=0.0,
            contact_residual=0.0,
            torque=0.0,
            friction=0.0,
            normal=0.0,
            impulse_friction=0.0,
            impulse_normal=0.0,
            joint=0.0,
            velocity=0.0,
            stance_position=0.0,
            stance_velocity=0.0,
            penetration=0.0,
            tcp_peak=0.0,
            rotation_peak=0.0,
            quaternion=0.0,
            base_position=0.0,
            base_rotation=0.0,
        )
        expected = np.r_[self.x0_physical, self.static_control]
        metrics["initial_gap"] = float(np.max(np.abs(self.state.diff(expected, xs[0]))))
        tcp_errors, records = [], []
        for k, row in enumerate(self.schedule):
            x = np.asarray(xs[k])
            u = np.asarray(us[k] if k < len(us) else us[-1])
            node = self.oracle(x, u, row)
            forces = node["forces"][row["contacts"]]
            if k < len(us):
                metrics["gap"] = max(
                    metrics["gap"], float(np.max(np.abs(self.state.diff(node["xnext"], xs[k + 1]))))
                )
            metrics["dynamics_residual"] = max(
                metrics["dynamics_residual"], float(np.max(np.abs(node["dynamics_residual"])))
            )
            metrics["contact_residual"] = max(
                metrics["contact_residual"], float(np.max(np.abs(node["contact_residual"])))
            )
            friction = float(
                np.max(np.abs(forces[:, :2]).sum(axis=1) - self.settings.friction * forces[:, 2])
            )
            prefix = "impulse_" if row["kind"] == "impact" else ""
            normal = (
                float(np.max(-forces[:, 2]))
                if prefix
                else float(max(np.max(2 - forces[:, 2]), np.max(forces[:, 2] - 200)))
            )
            metrics[prefix + "friction"] = max(metrics[prefix + "friction"], friction)
            metrics[prefix + "normal"] = max(metrics[prefix + "normal"], normal)
            if len(u):
                metrics["torque"] = max(
                    metrics["torque"], float(np.max(np.abs(u) - self.torque_limits))
                )
            metrics["joint"] = max(
                metrics["joint"],
                float(np.max(self.robot.model.lowerPositionLimit[7:] - x[7:25])),
                float(np.max(x[7:25] - self.robot.model.upperPositionLimit[7:])),
            )
            metrics["velocity"] = max(
                metrics["velocity"], float(np.max(np.abs(x[25:49]) - super()._velocity_limits()))
            )
            metrics["quaternion"] = max(
                metrics["quaternion"], abs(float(np.linalg.norm(x[3:7])) - 1)
            )
            base = self.robot.frame_pose(self.robot.from_native_q(x[:25]), "base")
            target_base = self.reference.base_poses[k]
            metrics["base_position"] = max(
                metrics["base_position"],
                float(
                    np.max(
                        np.abs(base[:3, 3] - target_base[:3, 3])
                        - self.settings.base_translation_bound
                    )
                ),
            )
            metrics["base_rotation"] = max(
                metrics["base_rotation"],
                float(
                    np.max(
                        np.abs(pin.log3(target_base[:3, :3].T @ base[:3, :3]))
                        - self.settings.base_rotation_bound
                    )
                ),
            )
            position = np.linalg.norm(
                node["feet"][row["contacts"]] - row["anchors"][row["contacts"]], axis=1
            )
            metrics["stance_position"] = max(metrics["stance_position"], float(np.max(position)))
            if not prefix:
                metrics["stance_velocity"] = max(
                    metrics["stance_velocity"],
                    float(np.max(np.linalg.norm(node["foot_velocities"][row["contacts"]], axis=1))),
                )
            metrics["penetration"] = max(metrics["penetration"], float(np.max(-node["feet"][:, 2])))
            target = self.reference.tcp_poses[k]
            error = float(np.linalg.norm(node["tcp"][:3, 3] - target[:3, 3]))
            tcp_errors.append(error)
            metrics["tcp_peak"] = max(metrics["tcp_peak"], error)
            metrics["rotation_peak"] = max(
                metrics["rotation_peak"],
                float(np.linalg.norm(pin.log3(target[:3, :3].T @ node["tcp"][:3, :3]))),
            )
            records.append(node)
        metrics["tcp_rmse"] = float(np.sqrt(np.mean(np.square(tcp_errors))))
        metrics["forward_displacement"] = float(xs[-1][0] - xs[0][0])
        metrics["max_ik_error"] = max(self.ik_errors)
        metrics["minimum_swing_peak"] = min(
            max(
                record["feet"][leg, 2]
                for row, record in zip(self.schedule, records)
                if row["kind"] == "flow" and row["swing"] == leg
            )
            for leg in range(4)
        )
        metrics["terminal_speed"] = float(np.max(np.abs(xs[-1][25:49])))
        return metrics, records

    def acceptance(self, metrics):
        """Numerical replay gates, not collision/real-time/hardware acceptance.

        Contact drift is bounded at 1 mm/1 cm/s. Cone tolerances are 1 mN for
        forces and 0.1 mN.s for impacts; all reported violations remain visible.
        """
        limits = dict(
            initial_gap=1e-10,
            gap=1e-6,
            dynamics_residual=1e-7,
            contact_residual=1e-7,
            torque=1e-6,
            friction=1e-3,
            normal=1e-3,
            impulse_friction=1e-4,
            impulse_normal=1e-4,
            joint=1e-6,
            velocity=1e-6,
            quaternion=1e-10,
            base_position=1e-6,
            base_rotation=1e-6,
            stance_position=1e-3,
            stance_velocity=1e-2,
            penetration=1e-3,
            tcp_peak=5e-3,
            rotation_peak=2e-2,
            max_ik_error=1e-6,
        )
        failures = [
            key
            for key, limit in limits.items()
            if not math.isfinite(metrics[key]) or metrics[key] > limit
        ]
        minimums = dict(
            forward_displacement=0.6 * self.config.cycles * self.config.step_length,
            minimum_swing_peak=0.6 * self.config.step_height,
        )
        failures.extend(
            key
            for key, limit in minimums.items()
            if not math.isfinite(metrics[key]) or metrics[key] < limit
        )
        return dict(
            passed=not failures, failures=failures, upper_limits=limits, lower_limits=minimums
        )

    def feedback_replay(self, xs, us, gains):
        """Roll independently from x0 with u = u_plan - K * diff(x_plan, x).

        Uses dense Pinocchio KKT only, never Crocoddyl rollout or IK state updates.
        This nominal initial-state replay checks model consistency, not robustness.
        """
        self.audit(xs, us)
        if len(gains) != len(us):
            raise ValueError("One feedback gain is required for each action")
        actual = [np.r_[self.x0_physical, self.static_control]]
        controls, clipped = [], 0
        for k, row in enumerate(self.schedule[:-1]):
            gain = np.asarray(gains[k])
            if gain.shape != (len(us[k]), 66) or not np.isfinite(gain).all():
                raise ValueError("Invalid feedback gain")
            control = np.asarray(us[k]) - gain @ self.state.diff(xs[k], actual[-1])
            if len(control):
                bounded = np.clip(control, -self.torque_limits, self.torque_limits)
                clipped += int(np.any(control != bounded))
                control = bounded
            controls.append(control)
            actual.append(self.oracle(actual[-1], control, row)["xnext"])
        deviation = max(float(np.max(np.abs(self.state.diff(a, b)))) for a, b in zip(xs, actual))
        return actual, controls, dict(clipped_steps=clipped, maximum_plan_deviation=deviation)
