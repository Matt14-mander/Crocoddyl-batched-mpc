"""Pure Torch closed loop: stance or three-foot FL target tracking.

The plant is a separate Torch model instance, not result.xs or the reference.
It shares controller equations: this is a numerical integration smoke test,
not independent simulator/real-robot validation.
"""

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import torch

from crocoddyl_batched_mpc.models import (
    Go2HybridDynamics,
    Go2TrackingMPC,
    go2_front_leg_tracking_plan,
)


@torch.no_grad()
def run_tracking(*, device="cpu", batch_size=2, task="front-leg", compile_physics=True):
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; CPU fallback is not permitted")
    if task not in ("stance", "front-leg"):
        raise ValueError("task must be stance or front-leg")
    started = time.perf_counter()
    controller = Go2TrackingMPC(batch_size, device=device, compile_physics=compile_physics)
    model = controller.model
    plant = Go2HybridDynamics(device=device, dt=model.dt, dtype=model.dtype)
    plan = go2_front_leg_tracking_plan(model, standing_only=task == "stance")
    state = plan.reference[0].expand(batch_size, -1).clone()
    # Small distinct initial joint perturbations, not reference-state playback.
    state[:, 8] += torch.linspace(-1e-4, 1e-4, batch_size, device=device, dtype=model.dtype)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    preparation = time.perf_counter() - started
    timings, errors, support_errors, violations, usable, inactive_forces = [], [], [], [], [], []
    events = []
    loop_started = time.perf_counter()
    for step in range(len(plan.reference)):
        command = plan.command(step, batch_size, controller.horizon)
        if device.type == "cuda":
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            begin.record(torch.cuda.current_stream(device))
        else:
            before = time.perf_counter()
        output = controller.compute(state, command)
        if device.type == "cuda":
            end.record(torch.cuda.current_stream(device))
            events.append((begin, end))
        else:
            timings.append((time.perf_counter() - before) * 1000)
        contact = plant.contact_dynamics(
            state,
            output.action,
            contact_mask=command.support_mask,
            contact_positions=command.anchors,
        )
        velocity = state[:, 19:] + plan.dt * contact.acceleration
        next_state = plant.manifold.integrate(
            state, torch.cat((plan.dt * velocity, plan.dt * contact.acceleration), -1)
        )
        # Failed environments stop integrating; the final report rejects the run.
        state = torch.where(output.usable[:, None], next_state, state)
        feet = plant.foot_positions(state[:, :19])
        target = plan.feet[min(step + 1, len(plan.feet) - 1)]
        errors.append((feet[:, 0] - target[0]).norm(dim=-1))
        support_errors.append(
            torch.where(command.support_mask, (feet - command.anchors).norm(dim=-1), 0.0).amax(-1)
        )
        constraints = controller.mpc.constraints
        tv, _ = constraints.violations(output.action, contact.forces_world)
        force_residual = (
            contact.forces_world.flatten(1) @ constraints.force_matrix.T - constraints.force_bound
        ).reshape(batch_size, 4, 6)
        fv = torch.where(command.support_mask[..., None], force_residual, 0.0)
        fv = fv.clamp_min(0).flatten(1).amax(-1)
        violations.append(torch.stack((tv, fv), -1))
        inactive_forces.append(
            torch.where(~command.support_mask[..., None], contact.forces_world, 0.0)
            .abs()
            .flatten(1)
            .amax(-1)
        )
        usable.append(output.usable)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        timings = [a.elapsed_time(b) for a, b in events]
    elapsed = time.perf_counter() - loop_started
    latency = torch.tensor(timings, dtype=torch.float64)
    errors = torch.stack(errors)
    # Task-level thresholds, not just equation agreement.
    usable = torch.stack(usable)
    maximum_support_error = float(torch.stack(support_errors).max())
    violation = torch.stack(violations).amax((0, 1))
    final_error = float(errors[-1].max())
    max_error = float(errors.max())
    inactive = float(torch.stack(inactive_forces).max())
    passed = bool(usable.all()) and bool(torch.isfinite(state).all())
    passed &= max_error < 0.01 and final_error < 0.003 and maximum_support_error < 0.003
    passed &= float(violation.max()) <= 1e-7 and inactive <= 1e-10
    return {
        "task": task,
        "device": str(device),
        "batch_size": batch_size,
        "dtype": str(model.dtype),
        "horizon": controller.horizon,
        "dt_s": plan.dt,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "platform": platform.platform(),
        "cpu_threads": torch.get_num_threads(),
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "steps": len(plan.reference),
        "preparation_s": preparation,
        "all_actions_usable": bool(usable.all()),
        "failed_env_steps": int((~usable).sum()),
        "max_fl_error_m": max_error,
        "final_fl_error_m": final_error,
        "max_support_error_m": maximum_support_error,
        "max_inactive_force_N": inactive,
        "max_torque_violation_Nm": float(violation[0]),
        "max_support_force_violation_N": float(violation[1]),
        "controller_median_ms": float(latency.median()),
        "controller_p99_ms": float(latency.quantile(0.99)),
        "controller_max_ms": float(latency.max()),
        "loop_wall_s": elapsed,
        "environment_steps_per_s": len(usable) * batch_size / elapsed,
        "peak_allocated_bytes": (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        ),
        "passed": passed,
        "timing_scope": "compute only; CUDA events include stream time; loop includes Torch plant",
        "validation_scope": "shared-equation Torch plant; no Isaac Lab or native oracle",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--task", choices=("stance", "front-leg"), default="front-leg")
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    report = run_tracking(
        device=args.device,
        batch_size=args.batch_size,
        task=args.task,
        compile_physics=not args.no_compile,
    )
    payload = json.dumps(report, indent=2, allow_nan=False)
    print(payload)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n")
    if not report["passed"]:
        sys.exit("tracking acceptance failed; inspect the report")


if __name__ == "__main__":
    main()
