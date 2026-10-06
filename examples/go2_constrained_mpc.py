"""50 Hz target: local hard-constrained MPC, exact horizon gate and timing.

Measure controller-only wall time; plant, sensors, transport and actuator IO
are excluded. Timing is observational; use --require-deadline to audit a run.
"""

import argparse
import json
import platform
import time

import torch

from crocoddyl_batched_mpc.models import Go2ConstrainedMPC, Go2FixedContactReference


def run_constrained(
    *,
    batch_size=2,
    steps=120,
    horizon=2,
    device="cpu",
    plant="torch",
    friction=0.6,
    qp_iterations=12,
):
    if steps < 10 or plant not in ("torch", "pinocchio"):
        raise ValueError("steps>=10 and a torch/pinocchio plant are required")
    if plant == "pinocchio" and torch.device(device).type != "cpu":
        raise ValueError("independent Pinocchio plant requires CPU")
    started = time.perf_counter()
    c = Go2ConstrainedMPC(
        batch_size, horizon, device=device, friction=friction, qp_iterations=qp_iterations
    )
    m = c.dynamics
    delta = c.reference.new_zeros(batch_size, 36)
    delta[:, 3] = 0.02 * torch.where(torch.arange(batch_size, device=device) % 2 == 0, 1.0, -1.0)
    delta[:, 6:18] = 0.015
    delta[:, 18:] = 0.02
    state = m.manifold.integrate(c.reference.expand(batch_size, -1), delta)
    for _ in range(3):
        c.compute(state)
    if c.device.type == "cuda":
        torch.cuda.synchronize(c.device)
    prepare_seconds = time.perf_counter() - started
    native = None
    if plant == "pinocchio":
        import numpy as np
        import pinocchio as pin

        native = Go2FixedContactReference(stabilization=m.gains)

    times, errors, forces, torques, qp_counts = [], [], [], [], []
    usable = True
    disturbance_step = min(32, steps // 2)
    for step in range(steps):
        if step == disturbance_step:
            state[:, 22:25] += state.new_tensor([0.08, -0.06, 0.04])
        # CUDA sync belongs to measurement, never inside the controller.
        if c.device.type == "cuda":
            torch.cuda.synchronize(c.device)
        before = time.perf_counter_ns()
        action, result = c.compute(state)
        if c.device.type == "cuda":
            torch.cuda.synchronize(c.device)
        times.append((time.perf_counter_ns() - before) / 1e6)
        if not bool(result.usable.all()):
            usable = False
            break  # NaN actions are never sent to the plant.
        if native is None:
            contact = m.contact_dynamics(state, action)
            force = contact.forces_world
            state = m.calc(state, action)
        else:
            outputs, physical_forces = [], []
            for b in range(batch_size):
                q, v, u = state[b, :19].numpy(), state[b, 19:].numpy(), action[b].numpy()
                physical = native.calc(q, v, u)
                velocity = v + m.dt * physical.acceleration
                outputs.append(
                    np.r_[pin.integrate(native.robot.model, q, m.dt * velocity), velocity]
                )
                physical_forces.append(physical.forces_world)
            state = torch.from_numpy(np.stack(outputs))
            force = torch.from_numpy(np.stack(physical_forces))
        errors.append(m.manifold.diff(state, c.reference))
        forces.append(force)
        torques.append(action)
        qp_counts.append(result.iterations)
    latency = torch.tensor(times, dtype=torch.float64)
    report = {
        "batch_size": batch_size,
        "horizon": horizon,
        "dt": m.dt,
        "steps_requested": steps,
        "steps_completed": len(errors),
        "plant": plant,
        "device": str(c.device),
        "dtype": str(c.dtype),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "platform": platform.platform(),
        "cpu_threads": torch.get_num_threads(),
        "prepare_seconds": prepare_seconds,
        "all_actions_usable": usable,
        "latency_scope": "compute/QP/nonlinear horizon gate; excludes plant/sensors/IO",
        "median_ms": float(latency.median()),
        "p95_ms": float(latency.quantile(0.95)),
        "p99_ms": float(latency.quantile(0.99)),
        "max_ms": float(latency.max()),
        "deadline_ms": m.dt * 1000,
        "deadline_misses": int((latency > m.dt * 1000).sum()),
    }
    if errors:
        trajectory = torch.stack(errors)
        f, u = torch.stack(forces), torch.stack(torques)
        tv, fv = c.constraints.violations(u, f)
        report.update(
            final_state_error=trajectory[-1].norm(dim=-1).tolist(),
            last_five_error_max=trajectory[-5:].norm(dim=-1).amax(0).tolist(),
            max_torque_violation_Nm=float(tv.max()),
            max_force_violation_N=float(fv.max()),
            min_normal_force_N=float(f[..., 2].min()),
            max_diamond_ratio=float((f[..., :2].abs().sum(-1) / f[..., 2]).max()),
            max_torque_fraction=float((u.abs() / c.constraints.torque_limits).max()),
            max_qp_iterations=int(torch.stack(qp_counts).max()),
        )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--horizon", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--plant", choices=("torch", "pinocchio"), default="torch")
    parser.add_argument("--friction", type=float, default=0.6)
    parser.add_argument("--qp-iterations", type=int, default=12)
    parser.add_argument("--require-deadline", action="store_true")
    args = vars(parser.parse_args())
    require = args.pop("require_deadline")
    torch.set_num_threads(1)
    report = run_constrained(**args)
    print(json.dumps(report, indent=2))
    if not report["all_actions_usable"]:
        raise SystemExit("no feasible trajectory; plant stopped")
    if require and report["deadline_misses"]:
        raise SystemExit("50 Hz deadline audit failed")


if __name__ == "__main__":
    main()
