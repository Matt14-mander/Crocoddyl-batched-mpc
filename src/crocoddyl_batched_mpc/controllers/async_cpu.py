"""Wall-paced independent CPU plant with a spawned W planner.

The old applied torque is held during WBC computation and IPC. The plant advances
over elapsed wall time in <=5 ms substeps; it never replaces state with a desired
or predicted state. Numerical catch-up cost and missed control slots are reported.
This is a soft real-time CPU experiment, not a 200 Hz guarantee or physics engine.
"""

import time
from dataclasses import asdict, dataclass

import numpy as np

from ..models.go2_arm_cpu import FixedSupportCPU
from ..planners.standing_whole_body import WholeBodySettings
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
):
    """Real monotonic-clock experiment; worker startup occurs before task time zero."""
    import pinocchio as pin

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
    wbc_settings = WBCSettings(dt=ws.control_period)
    wbc = WholeBodyQP(robot, common.guidance.samples[0].foot_anchors, wbc_settings)
    plant = FixedSupportCPU(
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
    worker = SpawnWWorker(robot.config, common, ps, settings, faults=faults)
    startup = []
    started = time.monotonic()
    try:
        while not worker.ready:
            startup.extend(worker.poll())
            if not worker.alive() or any(e["kind"] == "startup_error" for e in startup):
                raise RuntimeError(f"planner startup failed: {startup}")
            if time.monotonic() - started > ws.startup_timeout:
                raise RuntimeError("planner startup timed out")
            if not worker.ready:
                time.sleep(0.01)
        startup_seconds = time.monotonic() - started
        origin = float(common.guidance.times[0])
        zero = time.monotonic()

        def clock():
            return origin + time.monotonic() - zero

        receiver = AsyncWReceiver(
            common, worker, settings=settings, clock=clock, clock_zero=zero, horizon=horizon
        )
        stamp = origin
        end = float(common.guidance.times[-1]) + ws.extra_time
        states, times, torques, forces = [np.r_[q, v]], [stamp], [], []
        controls, reference_poses, errors, original_errors = [], [], [], []
        wbc_states = []
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

        def advance(target):
            nonlocal q, v, stamp
            if target - stamp > ws.max_catchup:
                raise RuntimeError("wall-clock plant catch-up bound exceeded")
            while target - stamp > 1e-10:
                plant.dt = min(ws.integration_step, target - stamp)
                wrench = np.zeros(6) if external_wrench is None else external_wrench(stamp - origin)
                step = plant.step(q, v, torque, external_wrench=wrench)
                q, v = step.q, step.v
                stamp += plant.dt
                states.append(np.r_[q, v])
                times.append(stamp)
                torques.append(torque.copy())
                forces.append(step.forces)
                value = robot.evaluate(q, v)
                quantities = dict(
                    foot_position=np.max(
                        np.linalg.norm(value.foot_positions - wbc.anchors, axis=1)
                    ),
                    foot_velocity=np.max(np.linalg.norm(value.foot_jacobians @ v, axis=1)),
                    torque_violation=np.max(np.abs(torque) - robot.torque_limits),
                    joint_position_violation=max(np.max(q[7:] - upper), np.max(lower - q[7:])),
                    joint_velocity_violation=np.max(np.abs(v[6:]) - robot.velocity_limits),
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

        while stamp < end - 1e-9:
            delay = origin + next_slot * ws.control_period - clock()
            if delay > 0:
                time.sleep(delay)
            cycle_start = time.monotonic()
            sensed = min(clock(), end)
            try:
                advance(sensed)
                if stamp >= end - 1e-9:
                    break
                outgoing, mode = receiver.buffer.sample(sensed)
                snapshot = StateSnapshot(
                    sensed,
                    np.r_[q, v],
                    torque,
                    outgoing,
                    mode in ("standing", "return_to_standing"),
                )
                ipc_start = time.monotonic()
                receiver.tick(snapshot)
                ipc_times.append(time.monotonic() - ipc_start)
                sampled_at = min(clock(), end)
                # IPC/validation takes real time too: obtain a fresh plant state
                # at the WBC reference timestamp, rather than using the request snapshot.
                advance(sampled_at)
                if stamp >= end - 1e-9:
                    break
                wbc_state = np.r_[q, v]
                reference, mode = receiver.buffer.sample(sampled_at)
                packet = receiver.buffer.current
                valid_until = packet.valid_until if packet is not None else None
                output = wbc.solve(q, v, reference, torque)
                wbc_times.append(output.diagnostics.get("solve_time_s", 0.0))
                qp_rows.append(output.diagnostics)
                if not output.success:
                    raise RuntimeError("WBC rejected: " + output.status)
                completed = min(clock(), end)
                age = completed - sampled_at
                stale = age > ws.max_command_age or (
                    mode in ("active", "handoff") and completed > valid_until
                )
                # Old torque remains applied throughout this command's computation.
                advance(completed)
                if stale:
                    discarded += 1
                else:
                    torque = output.torques.copy()
                max_state_lag = max(max_state_lag, clock() - stamp)
                current, current_mode = receiver.buffer.sample(stamp)
                actual = robot.evaluate(q, v)
                error = float(np.linalg.norm(actual.tcp_pose[:3, 3] - current.tcp_pose[:3, 3]))
                original = common.guidance.sample(min(stamp, common.guidance.valid_until))
                original_errors.append(
                    float(np.linalg.norm(actual.tcp_pose[:3, 3] - original.tcp_pose[:3, 3]))
                )
                errors.append(error)
                metrics["tcp_peak"] = max(metrics["tcp_peak"], error)
                metrics["rotation_peak"] = max(
                    metrics["rotation_peak"],
                    float(
                        np.linalg.norm(
                            pin.log3(current.tcp_pose[:3, :3].T @ actual.tcp_pose[:3, :3])
                        )
                    ),
                )
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
                reference_poses.append(current.tcp_pose)
                modes.append(current_mode)
            except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
                failure = dict(time=clock(), state_time=stamp, reason=str(exc))
                break
            cycle_times.append(time.monotonic() - cycle_start)
            following = max(
                next_slot + 1, int(np.floor((clock() - origin) / ws.control_period)) + 1
            )
            missed_slots += following - next_slot - 1
            next_slot = following
        elapsed = time.monotonic() - zero
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
            realtime_accepted=False,
            schedule=(
                "monotonic wall clock; held-torque integration during computation; "
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
