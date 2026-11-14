"""Wall-paced independent CPU plant with a spawned W planner.

The old applied torque is held during WBC computation and IPC. The plant advances
over elapsed wall time in <=5 ms substeps; it never replaces state with a desired
or predicted state. Numerical catch-up cost and missed control slots are reported.
This is a soft real-time CPU experiment, not a 200 Hz guarantee or physics engine.
"""

import time
from copy import copy
from dataclasses import asdict, dataclass

import numpy as np

from ..models.go2_arm_async_cpu import AsyncCPUPlant
from ..models.go2_arm_cpu import FixedSupportCPU
from ..planners.standing_whole_body import WholeBodySettings
from .async_timing import ExecutionClock
from .async_whole_body import AsyncSettings, AsyncWReceiver, SpawnWWorker, StateSnapshot
from .rolling_whole_body_cpu import latency_summary
from .whole_body_qp import WBCSettings, WholeBodyQP


@dataclass(frozen=True)
class WallCPUSettings:
    control_period: float = 0.02
    integration_step: float = 0.005
    max_catchup: float = 0.2
    max_command_age: float = 0.08
    extra_time: float = 2.5
    startup_timeout: float = 15.0

    def __post_init__(self):
        if any(not np.isfinite(x) or x <= 0 for x in vars(self).values()):
            raise ValueError("wall CPU durations must be positive and finite")
        if self.control_period > 0.02 or self.integration_step > 0.005:
            raise ValueError("WBC dt must be <=20 ms and plant step <=5 ms")


def run_async_cpu_episode(
    robot,
    common,
    *,
    settings=None,
    planner_settings=None,
    wall_settings=None,
    faults=None,
    external_wrench=None,
    execution=None,
    worker_factory=SpawnWWorker,
    wbc_settings=None,
    optimized=True,
):
    """Wall-clock experiment by default; explicit virtual execution is for regression only."""
    import pinocchio as pin

    execution = ExecutionClock() if execution is None else execution
    if execution.virtual and worker_factory is SpawnWWorker:
        raise ValueError("virtual clock requires an explicitly scheduled test worker")
    settings = AsyncSettings() if settings is None else settings
    ps = WholeBodySettings(nodes=24) if planner_settings is None else planner_settings
    ws = WallCPUSettings() if wall_settings is None else wall_settings
    horizon = ps.nodes * ps.dt
    if (
        max(settings.handoff_duration, settings.replan_period, settings.recovery_duration)
        >= horizon
    ):
        raise ValueError("horizon must exceed handoff, replan and recovery durations")
    if not common.guidance.valid or common.guidance.valid_until != common.guidance.times[-1]:
        raise ValueError("complete valid task preview required")
    wbc_settings = WBCSettings(dt=ws.control_period) if wbc_settings is None else wbc_settings
    if wbc_settings.dt != ws.control_period:
        raise ValueError("WBC and control period must match")
    wbc = WholeBodyQP(robot, common.guidance.samples[0].foot_anchors, wbc_settings)
    plant = (AsyncCPUPlant if optimized else FixedSupportCPU)(
        robot,
        wbc.anchors,
        dt=ws.integration_step,
        kp=wbc_settings.contact_kp,
        kd=wbc_settings.contact_kd,
    )
    state = common.nominal_states[0].copy()
    q, v = state[:25].copy(), state[25:].copy()
    torque = np.zeros(18)
    # Load/cold-start the QP before beginning the wall-clock task, while stationary.
    for _ in range(3):
        output = wbc.solve(q, v, common.guidance.samples[0], torque)
        if not output.success:
            raise RuntimeError("initial WBC support failed")
        torque = output.torques.copy()
    worker = worker_factory(robot.config, common, ps, settings, faults=faults)
    startup = []
    started = execution.now()
    try:
        while not worker.ready:
            startup.extend(worker.poll())
            if not worker.alive() or any(e["kind"] == "startup_error" for e in startup):
                raise RuntimeError(f"planner startup failed: {startup}")
            if execution.now() - started > ws.startup_timeout:
                raise RuntimeError("planner startup timed out")
            if not worker.ready:
                execution.wait(0.01)
        startup_seconds = execution.now() - started
        origin = float(common.guidance.times[0])
        execution.begin()
        zero = execution.now()

        def clock():
            return origin + execution.now() - zero

        receiver = AsyncWReceiver(
            common, worker, settings=settings, clock=clock, clock_zero=zero, horizon=horizon
        )
        stamp = origin
        end = float(common.guidance.times[-1]) + ws.extra_time
        states, times, torques, forces = [np.r_[q, v]], [stamp], [], []
        controls, reference_poses, errors, original_errors = [], [], [], []
        wbc_states, diagnostic_buffers = [], []
        modes, cycle_times, wbc_times, ipc_times, qp_rows = [], [], [], [], []
        bounds = dict(
            tcp_peak=0.005,
            rotation_peak=0.02,
            foot_position=0.001,
            foot_velocity=0.01,
            torque_violation=1e-6,
            joint_position_violation=1e-6,
            joint_velocity_violation=1e-6,
            force_violation=1e-5,
            dynamics_residual=1e-7,
            contact_residual=1e-7,
            quaternion_error=1e-10,
            qp_equality_residual=1e-6,
            qp_inequality_violation=1e-6,
            terminal_speed=0.05,
        )
        metrics = dict.fromkeys(bounds, 0.0)
        lower, upper = robot.position_limits
        next_slot, missed_slots, discarded = 0, 0, 0
        max_state_lag = 0.0
        failure = None
        timings = {
            k: []
            for k in (
                "plant_step",
                "plant_diagnostics",
                "request_sample",
                "ipc",
                "reference_sample",
                "wbc",
            )
        }
        scheduler_lateness, command_ages, cycle_observations = [], [], []
        torque_limits, velocity_limits = robot.torque_limits, robot.velocity_limits

        def advance(target):
            nonlocal q, v, stamp
            if target - stamp > ws.max_catchup:
                raise RuntimeError("wall-clock plant catch-up bound exceeded")
            while target - stamp > 1e-10:
                plant.dt = min(ws.integration_step, target - stamp)
                wrench = np.zeros(6) if external_wrench is None else external_wrench(stamp - origin)
                step_start = time.perf_counter()
                step = plant.step(q, v, torque, external_wrench=wrench)
                timings["plant_step"].append(time.perf_counter() - step_start)
                diagnostic_start = time.perf_counter()
                q, v = step.q, step.v
                stamp += plant.dt
                states.append(np.r_[q, v])
                times.append(stamp)
                torques.append(torque.copy())
                forces.append(step.forces)
                if optimized:
                    _, foot_positions, foot_velocities = plant.observe(q, v)
                else:
                    value = robot.evaluate(q, v)
                    foot_positions, foot_velocities = value.foot_positions, value.foot_jacobians @ v
                quantities = dict(
                    foot_position=np.max(np.linalg.norm(foot_positions - wbc.anchors, axis=1)),
                    foot_velocity=np.max(np.linalg.norm(foot_velocities, axis=1)),
                    torque_violation=np.max(np.abs(torque) - torque_limits),
                    joint_position_violation=max(np.max(q[7:] - upper), np.max(lower - q[7:])),
                    joint_velocity_violation=np.max(np.abs(v[6:]) - velocity_limits),
                    force_violation=np.max(
                        np.r_[
                            np.abs(step.forces[:, :2]).sum(axis=1)
                            - wbc_settings.friction * step.forces[:, 2],
                            wbc_settings.normal_min - step.forces[:, 2],
                            step.forces[:, 2] - wbc_settings.normal_max,
                        ]
                    ),
                    dynamics_residual=step.dynamics_residual,
                    contact_residual=step.contact_residual,
                    quaternion_error=abs(np.linalg.norm(q[3:7]) - 1),
                )
                for key, value in quantities.items():
                    if not np.isfinite(value):
                        raise RuntimeError("nonfinite CPU dynamics")
                    metrics[key] = max(metrics[key], float(value))
                timings["plant_diagnostics"].append(time.perf_counter() - diagnostic_start)

        while stamp < end - 1e-9:
            delay = origin + next_slot * ws.control_period - clock()
            if delay > 0:
                execution.wait(delay)
            execution.phase("wake")
            scheduler_lateness.append(max(0.0, clock() - origin - next_slot * ws.control_period))
            cycle_start = time.perf_counter()
            cycle_cpu_start = time.thread_time()
            delay_count = len(execution.events)
            stage = "advance_sensed_state"
            sensed = min(clock(), end)
            try:
                advance(sensed)
                if stamp >= end - 1e-9:
                    break
                stage = "request_snapshot"
                snapshot = None
                request_start = time.perf_counter()
                if receiver.wants_snapshot(sensed):
                    outgoing, mode = receiver.buffer.sample(sensed)
                    snapshot = StateSnapshot(
                        sensed,
                        np.r_[q, v],
                        torque,
                        outgoing,
                        mode in ("standing", "return_to_standing"),
                    )
                timings["request_sample"].append(time.perf_counter() - request_start)
                ipc_start = time.perf_counter()
                stage = "receive_and_submit"
                receiver.tick(snapshot)
                ipc_times.append(time.perf_counter() - ipc_start)
                timings["ipc"].append(ipc_times[-1])
                execution.phase("before_sample")
                sampled_at = min(clock(), end)
                # IPC/validation takes real time too: obtain a fresh plant state
                # at the WBC reference timestamp, rather than using the request snapshot.
                stage = "advance_sampled_state"
                advance(sampled_at)
                if stamp >= end - 1e-9:
                    break
                wbc_state = np.r_[q, v]
                stage = "reference_and_wbc"
                reference_start = time.perf_counter()
                reference, mode = receiver.buffer.sample(sampled_at)
                timings["reference_sample"].append(time.perf_counter() - reference_start)
                packet = receiver.buffer.current
                valid_until = packet.valid_until if packet is not None else None
                solve_start = time.perf_counter()
                output = wbc.solve(q, v, reference, torque)
                timings["wbc"].append(time.perf_counter() - solve_start)
                execution.phase("after_solve")
                wbc_times.append(output.diagnostics.get("solve_time_s", 0.0))
                qp_rows.append(output.diagnostics)
                if not output.success:
                    raise RuntimeError("WBC rejected: " + output.status)
                completed = min(clock(), end)
                age = completed - sampled_at
                command_ages.append(age)
                stale = age > ws.max_command_age or (
                    mode in ("active", "handoff") and completed > valid_until
                )
                # Old torque remains applied throughout this command's computation.
                stage = "advance_command_state"
                advance(completed)
                if stale:
                    discarded += 1
                else:
                    torque = output.torques.copy()
                max_state_lag = max(max_state_lag, clock() - stamp)
                # Immutable packets plus shallow buffer state retain the exact executed
                # reference. Expensive tracking diagnostics run after the timed episode.
                stage = "record_control"
                diagnostic_buffers.append(copy(receiver.buffer))
                metrics["qp_equality_residual"] = max(
                    metrics["qp_equality_residual"], output.diagnostics["equality_residual"]
                )
                metrics["qp_inequality_violation"] = max(
                    metrics["qp_inequality_violation"],
                    max(output.diagnostics["violations"].values()),
                )
                controls.append(
                    [
                        sensed,
                        sampled_at,
                        stamp,
                        not stale,
                        receiver.buffer.current.sequence if receiver.buffer.current else 0,
                    ]
                )
                wbc_states.append(wbc_state)
            except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
                failure = dict(
                    time=clock(),
                    state_time=stamp,
                    reason=str(exc),
                    stage=stage,
                    cycle_wall_s=time.perf_counter() - cycle_start,
                    cycle_thread_cpu_s=time.thread_time() - cycle_cpu_start,
                )
                break
            execution.phase("control_complete")
            cycle_wall = time.perf_counter() - cycle_start
            cycle_cpu = time.thread_time() - cycle_cpu_start
            cycle_times.append(cycle_wall)
            cycle_observations.append(
                dict(
                    at=sensed,
                    wall_s=cycle_wall,
                    thread_cpu_s=cycle_cpu,
                    non_thread_cpu_s=max(0.0, cycle_wall - cycle_cpu),
                    injected_delays=len(execution.events) - delay_count,
                )
            )
            following = max(
                next_slot + 1, int(np.floor((clock() - origin) / ws.control_period)) + 1
            )
            missed_slots += following - next_slot - 1
            next_slot = following
        elapsed = execution.now() - zero
        diagnostic_start = time.perf_counter()
        state_times = np.array(times)
        for control, buffer in zip(controls, diagnostic_buffers):
            applied_at = control[2]
            index = int(np.searchsorted(state_times, applied_at))
            actual_state = states[index]
            current, current_mode = buffer.sample(applied_at)
            actual_pose = (
                plant.observe(actual_state[:25], actual_state[25:])[0]
                if optimized
                else robot.evaluate(actual_state[:25], actual_state[25:]).tcp_pose
            )
            error = float(np.linalg.norm(actual_pose[:3, 3] - current.tcp_pose[:3, 3]))
            original = common.guidance.sample(min(applied_at, common.guidance.valid_until))
            original_errors.append(
                float(np.linalg.norm(actual_pose[:3, 3] - original.tcp_pose[:3, 3]))
            )
            errors.append(error)
            metrics["tcp_peak"] = max(metrics["tcp_peak"], error)
            metrics["rotation_peak"] = max(
                metrics["rotation_peak"],
                float(np.linalg.norm(pin.log3(current.tcp_pose[:3, :3].T @ actual_pose[:3, :3]))),
            )
            reference_poses.append(current.tcp_pose)
            modes.append(current_mode)
        offline_diagnostics_s = time.perf_counter() - diagnostic_start
        metrics.update(
            tcp_rmse=float(np.sqrt(np.mean(np.square(errors)))) if errors else float("inf"),
            terminal_speed=float(np.max(np.abs(v))),
            plant_steps=len(torques),
            control_updates=len(controls),
        )
        deliveries = [e for e in receiver.events if e["kind"] == "delivery"]
        accepted = [e for e in deliveries if e["accepted"]]
        failed = [
            key
            for key, bound in bounds.items()
            if not np.isfinite(metrics[key]) or metrics[key] > bound
        ]
        if failure:
            failed.append("terminated")
        if not accepted:
            failed.append("no_accepted_W_packet")
        report = dict(
            passed=not failed,
            failures=failed,
            termination=failure,
            metrics=metrics,
            limits=bounds,
            settings=dict(
                async_planner=asdict(settings),
                planner=asdict(ps),
                wall=asdict(ws),
                wbc=asdict(wbc_settings),
            ),
            wall_elapsed_s=elapsed,
            integrated_duration_s=stamp - origin,
            missed_control_slots=missed_slots,
            discarded_commands=discarded,
            max_numerical_state_lag_s=max_state_lag,
            startup_time_s=startup_seconds,
            worker_startup=startup,
            requests=len(receiver.requests),
            accepted=len(accepted),
            pending_cancelled=None
            if receiver.pending is None
            else receiver.pending["request"].sequence,
            events=receiver.events,
            modes={m: modes.count(m) for m in sorted(set(modes))},
            delivery_age=latency_summary([e["age_s"] for e in accepted], settings.max_state_age),
            wbc_timing=latency_summary(wbc_times, ws.control_period),
            cycle_timing=latency_summary(cycle_times, ws.control_period),
            ipc_timing=latency_summary(ipc_times, ws.control_period),
            original_task_tcp_peak_m=max(original_errors, default=float("inf")),
            original_task_tcp_rmse_m=float(np.sqrt(np.mean(np.square(original_errors))))
            if original_errors
            else float("inf"),
            offline_diagnostics_s=offline_diagnostics_s,
            cycle_thread_cpu=latency_summary(
                [c["thread_cpu_s"] for c in cycle_observations], ws.control_period
            ),
            cycle_non_thread_cpu=latency_summary(
                [c["non_thread_cpu_s"] for c in cycle_observations], ws.control_period
            ),
            cycle_outliers=sorted(cycle_observations, key=lambda c: c["wall_s"], reverse=True)[:10],
            realtime_accepted=False,
            timing_mode="deterministic_virtual" if execution.virtual else "wall",
            injected_delays=execution.events,
            execution_settings=dict(
                modeled_phase_costs_s=execution.costs if execution.virtual else {},
                delays=[asdict(d) for d in execution.delays],
                scheduled_worker_latency_s=getattr(worker, "latency", None),
            ),
            optimized_cpu=optimized,
            scheduler_lateness=latency_summary(scheduler_lateness, ws.control_period),
            command_age=latency_summary(command_ages, ws.max_command_age),
            phase_timings={
                k: {**latency_summary(v, ws.control_period), "total_s": sum(v)}
                for k, v in timings.items()
            },
            schedule=(
                "prescribed virtual timing; native solves execute synchronously in test transport"
                if execution.virtual
                else "monotonic wall clock; held-torque integration during computation; "
                "no solver waits in control"
            ),
        )
        trace = dict(
            times=np.array(times),
            states=np.array(states),
            torques=np.array(torques),
            forces=np.array(forces),
            controls=np.array(controls),
            wbc_states=np.array(wbc_states),
            modes=np.array(modes),
            tcp_reference=np.array(reference_poses),
            request_times=np.array([r.snapshot.time for r in receiver.requests]),
            request_states=np.array([r.snapshot.state for r in receiver.requests]),
            request_previous_torques=np.array(
                [r.snapshot.previous_torque for r in receiver.requests]
            ),
            request_sent_at=np.array([r.sent_at for r in receiver.requests]),
        )
    finally:
        worker.close()
    report["worker_closed"] = worker.process is None
    return report, trace, qp_rows
