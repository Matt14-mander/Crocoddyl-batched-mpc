"""Torque-driven flat-ground slow walk; optional independent Pinocchio plant."""

import argparse
import json
import platform
import sys
import time

import torch

from crocoddyl_batched_mpc.models import Go2FixedContactReference, load_go2
from crocoddyl_batched_mpc.models.go2_walk import (
    Go2SlowWalkController,
    go2_slow_walk_plan,
    go2_triangle_margin,
)


def run_walk(
    *,
    cycles=2,
    batch_size=2,
    plant="torch",
    compile_physics=True,
    step_length=0.02,
    swing_height=0.015,
    progress=False,
):
    if plant not in ("torch", "pinocchio"):
        raise ValueError("plant must be torch/pinocchio")
    started = time.perf_counter()
    plan = go2_slow_walk_plan(cycles=cycles, step_length=step_length, swing_height=swing_height)
    controller = Go2SlowWalkController(plan, batch_size, compile_physics=compile_physics)
    model = controller.mpc.model
    state = plan.reference[0].expand(batch_size, -1).clone()
    if batch_size > 1:
        state[1, 19:21] = state.new_tensor([0.002, -0.001])
    native, robot = {}, None
    if plant == "pinocchio":
        import numpy as np
        import pinocchio as pin

        robot = load_go2()

    def reference_for(mask, anchors):
        key = tuple(mask.tolist())
        if key not in native:
            native[key] = Go2FixedContactReference(
                robot,
                feet=tuple(f for f, active in zip(model.feet, key) if active),
                stabilization=model.gains,
            )
        reference = native[key]
        reference.positions = tuple(
            anchors[i].numpy().copy() for i, active in enumerate(key) if active
        )
        return reference

    prepare_seconds = time.perf_counter() - started
    timings, errors, events = [], [], []
    releases = [[] for _ in range(batch_size)]
    landings = [[] for _ in range(batch_size)]
    peak_swing = torch.zeros(batch_size, 4, dtype=state.dtype)
    min_margin = float("inf")
    min_normal = float("inf")
    max_force_violation = max_torque_violation = max_inactive_force = max_impulse_violation = 0.0
    max_rotation_error = max_anchor_error = 0.0
    max_wait_count = 0
    finished_hold = 0
    initial_x = state[:, 0].clone()
    all_usable = True
    budget = len(plan.reference) + cycles * 4 * controller.max_wait_ticks + 100
    for step in range(budget):
        old_mask = controller.contact_mask.clone()
        before = time.perf_counter_ns()
        control = controller.compute(state)
        timings.append((time.perf_counter_ns() - before) / 1e6)
        if not control.result.usable.all():
            all_usable = False
            break
        new = control.contact_mask & ~old_mask
        gone = old_mask & ~control.contact_mask
        if control.event_accepted.any():
            for b in range(batch_size):
                if gone[b].any():
                    releases[b].append(model.feet[int(gone[b].nonzero()[0])])
                if new[b].any():
                    landings[b].append(model.feet[int(new[b].nonzero()[0])])
            events.append(
                {
                    "step": step,
                    "frames": control.frame_indices.tolist(),
                    "accepted": control.event_accepted.tolist(),
                    "contacts": control.contact_mask.tolist(),
                }
            )
        outputs, forces = [], []
        for b in range(batch_size):
            x, u = control.state[b : b + 1], control.action[b : b + 1]
            mask, anchors = control.contact_mask[b : b + 1], control.anchors[b : b + 1]
            if plant == "torch":
                physical = model.contact_dynamics(
                    x, u, contact_mask=mask, contact_positions=anchors
                )
                force = physical.forces_world[0]
                output = model.calc(x, u, contact_mask=mask, contact_positions=anchors)[0]
            else:
                reference = reference_for(mask[0], anchors[0])
                if new[b].any():
                    q, v = state[b, :19].numpy(), state[b, 19:].numpy()
                    M, _, J, _ = reference._quantities(q, v)
                    response = np.linalg.solve(M, J.T)
                    impulse = -np.linalg.solve(J @ response, J @ v)
                    expected_velocity = v + response @ impulse
                    np.testing.assert_allclose(
                        x[0, 19:].numpy(), expected_velocity, atol=1e-9, rtol=1e-9
                    )
                    np.testing.assert_allclose(x[0, :19].numpy(), q, atol=0, rtol=0)
                    impulses = impulse.reshape(-1, 3)
                    max_impulse_violation = max(
                        max_impulse_violation,
                        float(np.maximum(-impulses[:, 2], 0).max()),
                        float(
                            np.maximum(
                                np.abs(impulses[:, :2]).sum(-1) - 0.6 * impulses[:, 2], 0
                            ).max()
                        ),
                    )
                q, v, torque = x[0, :19].numpy(), x[0, 19:].numpy(), u[0].numpy()
                physical = reference.calc(q, v, torque)
                velocity = v + plan.dt * physical.acceleration
                output = torch.from_numpy(
                    np.r_[pin.integrate(robot.model, q, plan.dt * velocity), velocity]
                )
                force = state.new_zeros(4, 3)
                force[mask[0]] = torch.from_numpy(physical.forces_world)
            outputs.append(output)
            forces.append(force)
        state = torch.stack(outputs)
        force = torch.stack(forces)
        torch.testing.assert_close(control.result.xs[:, 1], state, atol=1e-9, rtol=1e-9)
        bound = controller.mpc.constraints.force_bound[
            None
        ] * control.contact_mask.repeat_interleave(6, -1)
        fv = force.flatten(1) @ controller.mpc.constraints.force_matrix.T - bound
        tv = control.action.abs() - controller.mpc.constraints.torque_limits
        max_force_violation = max(max_force_violation, float(fv.clamp_min(0).max()))
        max_torque_violation = max(max_torque_violation, float(tv.clamp_min(0).max()))
        min_normal = min(min_normal, float(force[..., 2][control.contact_mask].min()))
        if (~control.contact_mask).any():
            max_inactive_force = max(
                max_inactive_force, float(force[~control.contact_mask].abs().max())
            )
        positions = model.foot_positions(state[:, :19])
        peak_swing = torch.maximum(
            peak_swing, torch.where(~control.contact_mask, positions[..., 2], 0.0)
        )
        stance_error = torch.where(
            control.contact_mask, (positions - control.anchors).norm(dim=-1), 0.0
        )
        max_anchor_error = max(max_anchor_error, float(stance_error.max()))
        three = control.contact_mask.sum(-1) == 3
        if three.any():
            margin = go2_triangle_margin(model, state, control.contact_mask, control.anchors)
            min_margin = min(min_margin, float(margin[three].min()))
        error = model.manifold.diff(state, plan.reference[control.frame_indices])
        errors.append(error.norm(dim=-1))
        max_rotation_error = max(max_rotation_error, float(error[:, 3:6].norm(dim=-1).max()))
        max_wait_count = max(max_wait_count, int(controller.wait_counts.max()))
        if progress and (step % 100 == 0 or control.event_accepted.any()):
            print(
                f"step={step} frames={control.frame_indices.tolist()} "
                f"x={state[:, 0].tolist()} wait={controller.wait_counts.tolist()}",
                file=sys.stderr,
                flush=True,
            )
        if control.finished.all():
            finished_hold += 1
            if finished_hold == 20:
                break
    latency = torch.tensor(timings, dtype=torch.float64)
    return {
        "plant": plant,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "platform": platform.platform(),
        "cpu_threads": torch.get_num_threads(),
        "dtype": str(state.dtype),
        "batch_size": batch_size,
        "dt_s": plan.dt,
        "horizon": controller.mpc.horizon,
        "force_margin_N": controller.mpc.force_margin,
        "friction": controller.mpc.constraints.friction,
        "cycles": cycles,
        "order": list(plan.order),
        "step_length_m": step_length,
        "swing_height_m": swing_height,
        "planned_duration_s": plan.duration,
        "simulated_duration_s": len(errors) * plan.dt,
        "prepare_seconds": prepare_seconds,
        "steps_completed": len(errors),
        "all_actions_usable": all_usable,
        "all_finished": bool(control.finished.all()) and finished_hold == 20,
        "releases": releases,
        "landings": landings,
        "events": events,
        "forward_displacement_m": (state[:, 0] - initial_x).tolist(),
        "final_state_error": model.manifold.diff(state, plan.reference[-1]).norm(dim=-1).tolist(),
        "peak_swing_height_m": peak_swing.tolist(),
        "min_three_foot_com_margin_m": min_margin,
        "min_normal_force_N": min_normal,
        "max_inactive_force_N": max_inactive_force,
        "max_force_violation_N": max_force_violation,
        "max_torque_violation_Nm": max_torque_violation,
        "max_impulse_violation_Ns": max_impulse_violation,
        "max_stance_anchor_error_m": max_anchor_error,
        "max_base_rotation_error_rad": max_rotation_error,
        "max_contact_wait_ticks": max_wait_count,
        "cycle_median_ms": float(latency.median()),
        "cycle_p99_ms": float(latency.quantile(0.99)),
        "cycle_max_ms": float(latency.max()),
        "deadline_misses": int((latency > 20).sum()),
        "latency_scope": (
            "scheduler/contact event/online QP/nonlinear horizon gate; "
            "excludes prepare/plant/sensors/IO"
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cycles", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--plant", choices=("torch", "pinocchio"), default="torch")
    parser.add_argument("--step-length", type=float, default=0.02)
    parser.add_argument("--swing-height", type=float, default=0.015)
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--require-deadline", action="store_true")
    args = vars(parser.parse_args())
    require = args.pop("require_deadline")
    torch.set_num_threads(1)
    report = run_walk(**args)
    print(json.dumps(report, indent=2))
    if not report["all_actions_usable"] or not report["all_finished"]:
        raise SystemExit("walk stopped: no feasible action or contact timeout")
    if require and report["deadline_misses"]:
        raise SystemExit("50 Hz deadline audit failed")


if __name__ == "__main__":
    main()
