"""Guidance -> WBC -> independent fixed-contact CPU dynamics execution loop."""

import time

import numpy as np

from ..guidance import GuidanceBuffer
from ..models.go2_arm_cpu import FixedSupportCPU
from .whole_body_qp import WBCSettings, WholeBodyQP


def run_cpu_episode(
    robot,
    common,
    packet=None,
    *,
    settings=None,
    extra_time=0.5,
    external_wrench=None,
    initial_state=None,
):
    """Execute feedback using only measured state, returning actual samples.

    A rejected WBC result latches termination; no plant step or replacement
    torque is applied. Computation is offline at fixed simulation dt, with
    measured latency reported independently of nominal integration frequency.
    """
    import pinocchio as pin

    settings = WBCSettings() if settings is None else settings
    packet = common.guidance if packet is None else packet
    if not np.isfinite(extra_time) or extra_time < 0:
        raise ValueError("extra_time must be finite and nonnegative")
    manager = GuidanceBuffer(common.guidance.samples[0])
    origin = float(common.guidance.times[0])
    if not manager.submit(packet, origin):
        raise ValueError("initial Guidance was rejected")
    controller = WholeBodyQP(robot, common.guidance.samples[0].foot_anchors, settings)
    plant = FixedSupportCPU(
        robot, controller.anchors, dt=settings.dt, kp=settings.contact_kp, kd=settings.contact_kd
    )
    state = (
        common.nominal_states[0].copy()
        if initial_state is None
        else np.array(initial_state, dtype=float)
    )
    if state.shape != (49,) or not np.isfinite(state).all():
        raise ValueError("initial state must be finite canonical (q25,v24)")
    q, v = robot._q(state[:25]), state[25:].copy()
    previous = np.zeros(18)
    duration = common.guidance.times[-1] - origin + extra_time
    steps = round(duration / settings.dt)
    if abs(steps * settings.dt - duration) > 1e-8:
        raise ValueError("episode duration must be a multiple of dt")
    rows, modes, solve_times, cycle_times, task_errors = [], [], [], [], []
    control_rows, force_rows, external_rows, qp_rows = [], [], [], []
    references = []
    metrics = dict(
        tcp_peak=0.0,
        rotation_peak=0.0,
        foot_position=0.0,
        foot_velocity=0.0,
        torque_violation=0.0,
        joint_position_violation=0.0,
        joint_velocity_violation=0.0,
        force_violation=0.0,
        dynamics_residual=0.0,
        contact_residual=0.0,
        quaternion_error=0.0,
        nominal_acceleration_mismatch=0.0,
        qp_equality_residual=0.0,
        qp_inequality_violation=0.0,
    )
    failure = None
    lower, upper = robot.position_limits
    for k in range(steps + 1):
        cycle_start = time.perf_counter()
        now = origin + k * settings.dt
        reference, mode = manager.sample(now)
        actual = robot.evaluate(q, v)
        error = float(np.linalg.norm(actual.tcp_pose[:3, 3] - reference.tcp_pose[:3, 3]))
        task_errors.append(error)
        metrics["tcp_peak"] = max(metrics["tcp_peak"], error)
        metrics["rotation_peak"] = max(
            metrics["rotation_peak"],
            float(np.linalg.norm(pin.log3(reference.tcp_pose[:3, :3].T @ actual.tcp_pose[:3, :3]))),
        )
        metrics["foot_position"] = max(
            metrics["foot_position"],
            float(np.max(np.linalg.norm(actual.foot_positions - controller.anchors, axis=1))),
        )
        metrics["foot_velocity"] = max(
            metrics["foot_velocity"],
            float(np.max(np.linalg.norm(actual.foot_jacobians @ v, axis=1))),
        )
        metrics["joint_position_violation"] = max(
            metrics["joint_position_violation"],
            float(np.max(q[7:] - upper)),
            float(np.max(lower - q[7:])),
        )
        metrics["joint_velocity_violation"] = max(
            metrics["joint_velocity_violation"],
            float(np.max(np.abs(v[6:]) - robot.velocity_limits)),
        )
        metrics["quaternion_error"] = max(
            metrics["quaternion_error"], abs(float(np.linalg.norm(q[3:7])) - 1)
        )
        rows.append(np.r_[q, v])
        references.append(reference)
        modes.append(mode)
        if k == steps:
            break
        try:
            output = controller.solve(q, v, reference, previous)
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
            failure = dict(step=k, time=now, status="WBC_ERROR", reason=str(exc))
            break
        solve_times.append(output.diagnostics.get("solve_time_s", float("nan")))
        qp_rows.append(output.diagnostics)
        if not output.success:
            failure = dict(step=k, time=now, status=output.status, diagnostics=output.diagnostics)
            break
        wrench = np.zeros(6) if external_wrench is None else external_wrench(now - origin)
        try:
            step = plant.step(q, v, output.torques, external_wrench=wrench)
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
            failure = dict(step=k, time=now, status="PLANT_ERROR", reason=str(exc))
            break
        force = step.forces
        violations = np.r_[
            np.abs(force[:, :2]).sum(axis=1) - settings.friction * force[:, 2],
            settings.normal_min - force[:, 2],
            force[:, 2] - settings.normal_max,
        ]
        metrics["force_violation"] = max(metrics["force_violation"], float(np.max(violations)))
        metrics["torque_violation"] = max(
            metrics["torque_violation"], float(np.max(np.abs(output.torques) - robot.torque_limits))
        )
        metrics["dynamics_residual"] = max(metrics["dynamics_residual"], step.dynamics_residual)
        metrics["contact_residual"] = max(metrics["contact_residual"], step.contact_residual)
        if np.max(np.abs(wrench)) == 0:
            metrics["nominal_acceleration_mismatch"] = max(
                metrics["nominal_acceleration_mismatch"],
                float(np.max(np.abs(step.acceleration - output.accelerations))),
            )
        metrics["qp_equality_residual"] = max(
            metrics["qp_equality_residual"], output.diagnostics["equality_residual"]
        )
        metrics["qp_inequality_violation"] = max(
            metrics["qp_inequality_violation"], max(output.diagnostics["violations"].values())
        )
        control_rows.append(output.torques)
        force_rows.append(force)
        external_rows.append(wrench)
        previous = output.torques.copy()
        q, v = step.q, step.v
        cycle_times.append(time.perf_counter() - cycle_start)
    metrics.update(
        tcp_rmse=float(np.sqrt(np.mean(np.square(task_errors)))),
        terminal_speed=float(np.max(np.abs(v))),
        executed_steps=len(control_rows),
    )
    limits = dict(
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
        nominal_acceleration_mismatch=1e-6,
        qp_equality_residual=1e-6,
        qp_inequality_violation=1e-6,
        terminal_speed=0.05,
    )
    failed_gates = [
        name
        for name, bound in limits.items()
        if not np.isfinite(metrics[name]) or metrics[name] > bound
    ]
    if failure is not None:
        failed_gates.append("WBC_terminated")

    def timing(values):
        # Keep the first cycle separately; report both cold and steady latency.
        values = np.asarray(values)
        if not len(values):
            return {}
        warmed = values[1:] if len(values) > 1 else values
        return dict(
            first_s=float(values[0]),
            p50_s=float(np.percentile(warmed, 50)),
            p95_s=float(np.percentile(warmed, 95)),
            p99_s=float(np.percentile(warmed, 99)),
            max_s=float(np.max(values)),
            deadline_misses=int(np.sum(values > settings.dt)),
            samples=len(values),
        )

    report = dict(
        passed=not failed_gates,
        failures=failed_gates,
        limits=limits,
        metrics=metrics,
        termination=failure,
        solver_timing=timing(solve_times),
        cycle_timing=timing(cycle_times),
        modes={mode: modes.count(mode) for mode in sorted(set(modes))},
        rejected_packets=manager.rejected_packets,
        integration_frequency_hz=1 / settings.dt,
        realtime_accepted=False,
        realtime_note="Fixed simulation step; measured latency is not real-time certification",
    )
    trace = dict(
        times=origin + np.arange(len(rows)) * settings.dt,
        states=np.stack(rows),
        torques=np.asarray(control_rows),
        forces=np.asarray(force_rows),
        external_world_wrenches=np.asarray(external_rows),
        modes=np.array(modes),
        tcp_reference=np.stack([r.tcp_pose for r in references]),
    )
    return report, trace, qp_rows
