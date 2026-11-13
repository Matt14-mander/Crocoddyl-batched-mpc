"""Measured-state receding-horizon W -> Guidance -> WBC on the CPU plant.

Scheduling uses simulation time. Native solves are synchronous and wall time is
reported, not hidden as a real-time or asynchronous MPC implementation. Episode
foot anchors and base bounds never move with measured tracking error.
"""

import time
from dataclasses import dataclass, replace

import numpy as np

from ..guidance import Guidance, GuidanceBuffer, stationary
from ..planners.standing_whole_body import StandingWholeBodyPlanner, WholeBodySettings
from ..references.standing_arm import CommonReference, guidance_from_whole_body, model_identity
from .standing_arm_cpu import run_cpu_episode
from .whole_body_qp import WBCSettings


@dataclass(frozen=True)
class RollingSettings:
    replan_period: float = 0.1
    handoff_duration: float = 0.05
    return_duration: float = 1.0
    recovery_duration: float = 0.4
    warm_start: bool = True

    def __post_init__(self):
        for name in ("replan_period", "handoff_duration", "return_duration", "recovery_duration"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if type(self.warm_start) is not bool:
            raise ValueError("warm_start must be boolean")


def preview_window(robot, common, now, settings, *, recovery=None):
    """Sample the shared task in absolute time; hold its stationary final pose.

    These are task priors, not measured states or W-optimized joints. No endpoint
    velocity is reset at a rolling window boundary inside the task.
    """
    import pinocchio as pin

    if not np.isfinite(now) or now < common.guidance.times[0]:
        raise ValueError("preview time precedes task origin")
    times = now + np.arange(settings.nodes + 1) * settings.dt
    samples, states = [], []
    for t in times:
        sample = (
            common.guidance.sample(t)
            if t <= common.guidance.valid_until
            else stationary(common.guidance.samples[-1], time=t)
        )
        if recovery is not None and recovery.times[0] <= t <= recovery.valid_until:
            sample = recovery.sample(t)
        sample = replace(sample, time=float(t), source="rolling_common")
        rotation = sample.base_pose[:3, :3]
        q = np.r_[sample.base_pose[:3, 3], pin.Quaternion(rotation).coeffs(), sample.joint_position]
        states.append(
            np.r_[
                q,
                rotation.T @ sample.base_velocity[:3],
                rotation.T @ sample.base_velocity[3:],
                sample.joint_velocity,
            ]
        )
        samples.append(sample)
    return CommonReference(
        Guidance(times, tuple(samples), float(now), float(times[-1])),
        np.stack(states),
        {"source": "shared task preview", "absolute_origin": float(now)},
    )


def shifted_seed(robot, plan, plan_origin, now, state, settings):
    """Shift an accepted canonical trajectory; interpolate q on its manifold.

    Controls are piecewise constant. Uncovered tail knots hold the last pose
    with zero speed and last torque; they are only an infeasible FDDP seed.
    Replace the first state by feedback. Torque memory is rebuilt by W.solve
    from the actual previously applied WBC torque, never a stale MPC action.
    """
    import pinocchio as pin

    age = now - plan_origin
    if not np.isfinite(age) or age < 0 or age >= plan.times[-1]:
        return None
    xs = plan.mpc.xs[0].detach().cpu().numpy()
    us = plan.mpc.us[0].detach().cpu().numpy()
    states, controls = [], []
    for k in range(settings.nodes + 1):
        t = age + k * settings.dt
        i = int(np.clip(np.searchsorted(plan.times, t, side="right") - 1, 0, len(us) - 1))
        if t >= plan.times[-1]:
            x = xs[-1].copy()
            if t > plan.times[-1]:
                x[25:] = 0
        else:
            fraction = (t - plan.times[i]) / (plan.times[i + 1] - plan.times[i])
            q = pin.interpolate(
                robot.model,
                robot.to_native_q(xs[i, :25]),
                robot.to_native_q(xs[i + 1, :25]),
                float(fraction),
            )
            x = np.r_[
                robot.from_native_q(q), (1 - fraction) * xs[i, 25:] + fraction * xs[i + 1, 25:]
            ]
        states.append(x)
        if k < settings.nodes:
            controls.append(us[i].copy())
    states[0] = np.array(state, dtype=float, copy=True)
    return np.stack(states), np.stack(controls)


def common_seed(planner, window, state):
    """Fresh shared-kinematic seed with per-node quasi-static torque priors.

    A cold solve does not reuse a previous optimizer result. In particular, it
    need not extrapolate moving feedback under one fixed standing torque, whose
    rollout can leave the local region over the rest of the horizon.
    """
    import crocoddyl as c

    diff = planner._differential(c.CostModelSum(planner.physical_state, 18))
    data = diff.createData()
    states = window.nominal_states.copy()
    controls = []
    for x in states[:-1]:
        native = planner._native_state(x)
        native[25:] = 0
        controls.append(planner._canonical_control(np.asarray(diff.quasiStatic(data, native))))
    states[0] = np.array(state, dtype=float, copy=True)
    return states, np.stack(controls)


def latency_summary(values, budget):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return {}
    return dict(
        samples=len(values),
        p50_s=float(np.percentile(values, 50)),
        p95_s=float(np.percentile(values, 95)),
        max_s=float(np.max(values)),
        budget_s=float(budget),
        deadline_misses=int(np.sum(values > budget)),
    )


class RollingWholeBodyCPU:
    """Per-episode planner and packet scheduler; WBC/plant own execution state."""

    def __init__(self, robot, common, *, planner_settings=None, settings=None):
        self.robot, self.common = robot, common
        self.settings = RollingSettings() if settings is None else settings
        ps = WholeBodySettings() if planner_settings is None else planner_settings
        if max(self.settings.replan_period, self.settings.handoff_duration) >= ps.nodes * ps.dt:
            raise ValueError("planning horizon must cover the replan period and handoff")
        if self.settings.recovery_duration > ps.nodes * ps.dt:
            raise ValueError("planning horizon must cover the recovery transition")
        packet = common.guidance
        if packet.samples[0].model_id != model_identity(robot):
            raise ValueError("task and robot model differ")
        if not packet.valid or packet.valid_until != packet.times[-1]:
            raise ValueError("rolling task must cover its full preview")
        for prefix in ("tcp", "base", "com", "joint"):
            for derivative in ("velocity", "acceleration"):
                if np.max(np.abs(getattr(packet.samples[-1], prefix + "_" + derivative))) > 1e-8:
                    raise ValueError("task must end at a stationary hold")
        self.planner = StandingWholeBodyPlanner(robot, ps, common.nominal_states[0])
        if not np.allclose(self.planner.anchors, packet.samples[0].foot_anchors, atol=1e-9, rtol=0):
            raise ValueError("task and planner support anchors differ")
        self.buffer = GuidanceBuffer(
            packet.samples[0],
            self.settings.return_duration,
            handoff_duration=self.settings.handoff_duration,
        )
        self.origin = float(packet.times[0])
        self.next_slot = 0
        self.cached_plan = None
        self.cached_origin = None
        self.recovery = None
        self.events = []
        self.plans = []

    def update(self, now, state, previous_torque, manager):
        if manager is not self.buffer:
            raise ValueError("rolling episode requires its own Guidance buffer")
        due = self.origin + self.next_slot * self.settings.replan_period
        if now < due - 1e-9 or now >= self.common.guidance.times[-1] - 1e-9:
            return
        slot = max(
            self.next_slot, int(np.floor((now - self.origin + 1e-9) / self.settings.replan_period))
        )
        self.next_slot = slot + 1
        sequence = slot + 1
        started = time.perf_counter()
        event = dict(
            sequence=sequence,
            requested_at=float(now),
            accepted=False,
            converged=False,
            seed="common_trajectory",
            initial_state=np.asarray(state).tolist(),
            previous_applied_torque=np.asarray(previous_torque).tolist(),
        )
        try:
            recovery = self.recovery
            if (manager.current is None and now > self.origin) or (
                manager.current is not None and now > manager.current.valid_until
            ):
                # Rejoin a task after expiry from the currently executed fallback
                # reference, not an unreachable jump back to the missed target.
                # Persist the bridge in absolute time across subsequent replans.
                outgoing, _ = manager.sample(now)
                end = now + self.settings.recovery_duration
                task = self.common.guidance
                target = (
                    task.sample(end)
                    if end <= task.valid_until
                    else stationary(task.samples[-1], time=end)
                )
                recovery = Guidance(
                    np.array([now, end]),
                    (replace(outgoing, source="recovery"), replace(target, source="recovery")),
                    now,
                    end,
                )
                event["recovery_started"] = True
            window = preview_window(
                self.robot, self.common, now, self.planner.settings, recovery=recovery
            )
            seed = None
            if self.settings.warm_start and self.cached_plan is not None:
                seed = shifted_seed(
                    self.robot,
                    self.cached_plan,
                    self.cached_origin,
                    now,
                    state,
                    self.planner.settings,
                )
                if seed is not None:
                    event.update(seed="shifted", seed_age_s=float(now - self.cached_origin))
            if seed is None:
                seed = common_seed(self.planner, window, state)
            plan = self.planner.solve(
                window.for_whole_body(),
                previous_torque,
                initial_guess=seed,
                initial_state=state,
            )
            event.update(
                converged=plan.converged,
                status=int(plan.mpc.status[0]),
                iterations=int(plan.mpc.iterations[0]),
                solve_time_s=plan.solve_time_s,
                plan_accepted=bool(plan.diagnostics.get("accepted_for_task", False)),
                plan_failures=plan.diagnostics.get("failures", []),
                max_dynamics_gap=plan.diagnostics.get("max_dynamics_gap"),
                build_time_s=plan.diagnostics.get("problem_build_time_s"),
                validation_time_s=plan.diagnostics.get("validation_time_s"),
            )
            packet = guidance_from_whole_body(
                self.robot, plan, window, sequence=sequence, initial_state=state
            )
            event.update(
                valid_until=float(packet.valid_until), accepted=manager.submit(packet, now)
            )
            if event["accepted"]:
                self.cached_plan, self.cached_origin = plan, now
                self.recovery = recovery
                self.plans.append((now, plan))
            else:
                event["reason"] = manager.last_rejection
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
            # No packet submission and no cache replacement. Old validity is not
            # extended; the buffer follows its existing expiry -> standing path.
            event["reason"] = str(exc)
        event["total_time_s"] = time.perf_counter() - started
        self.events.append(event)


def run_rolling_cpu_episode(
    robot, common, *, rolling=None, wbc_settings=None, extra_time=1.5, external_wrench=None
):
    """Run a synchronous receding-horizon simulation with independent plant feedback."""
    rolling = RollingWholeBodyCPU(robot, common) if rolling is None else rolling
    if rolling.robot is not robot or rolling.common is not common:
        raise ValueError("rolling planner must belong to this robot and common-reference episode")
    settings = WBCSettings() if wbc_settings is None else wbc_settings
    ratio = rolling.settings.replan_period / settings.dt
    if abs(ratio - round(ratio)) > 1e-9:
        raise ValueError("replan period must be an integer number of CPU steps")
    if rolling.events or rolling.buffer.current is not None:
        raise ValueError("use a fresh rolling episode, not a previous episode's warm cache")
    report, trace, qp = run_cpu_episode(
        robot,
        common,
        settings=settings,
        extra_time=extra_time,
        external_wrench=external_wrench,
        guidance_buffer=rolling.buffer,
        packet_update=rolling.update,
    )
    events = rolling.events
    original_errors = []
    for t, state in zip(trace["times"], trace["states"]):
        target = common.guidance.sample(min(float(t), common.guidance.valid_until))
        actual = robot.evaluate(state[:25]).tcp_pose[:3, 3]
        original_errors.append(float(np.linalg.norm(actual - target.tcp_pose[:3, 3])))
    report["rolling"] = dict(
        attempts=len(events),
        accepted=sum(e["accepted"] for e in events),
        failed=sum(not e["accepted"] for e in events),
        converged=sum(e["converged"] for e in events),
        shifted_attempts=sum(e["seed"] == "shifted" for e in events),
        events=events,
        original_task_tcp_peak_m=max(original_errors),
        original_task_tcp_rmse_m=float(np.sqrt(np.mean(np.square(original_errors)))),
        timing=latency_summary([e["total_time_s"] for e in events], rolling.settings.replan_period),
        schedule="synchronous simulation clock; wall time not used as a delivery delay",
    )
    if not any(e["accepted"] for e in events):
        report["failures"].append("no_accepted_W_packet")
        report["passed"] = False
    return report, trace, qp
