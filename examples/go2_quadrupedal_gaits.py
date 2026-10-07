"""Go2 walk/trot/pace/bound: offline Crocoddyl planning, Torch batch feedback.

The native plant is independent of the Torch execution dynamics. No reference
pose is applied to the plant. Zero-time impacts do not advance simulation time.
"""

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch

from crocoddyl_batched_mpc.models.go2_gaits import (
    GO2_GAIT_SWINGS,
    Go2GaitController,
    solve_go2_gait,
)


def run_gait(
    gait,
    *,
    cycles=1,
    batch_size=2,
    iterations=150,
    plant="pinocchio",
    step_length=0.04,
    step_height=0.02,
    dt=0.02,
    step_knots=20,
    support_knots=10,
    save_path=None,
    perturbation=1e-5,
    prepared_plan=None,
):
    if plant not in ("torch", "pinocchio"):
        raise ValueError("plant must be torch or pinocchio")
    started = time.perf_counter()
    plan = (
        prepared_plan
        if prepared_plan is not None
        else solve_go2_gait(
            gait,
            cycles=cycles,
            iterations=iterations,
            step_length=step_length,
            step_height=step_height,
            dt=dt,
            step_knots=step_knots,
            support_knots=support_knots,
        )
    )
    preparation = time.perf_counter() - started
    if save_path is not None:
        plan.save(save_path)
    controller = Go2GaitController(plan, batch_size)
    native_models = [m for solver in plan.native_solvers for m in solver.problem.runningModels]
    native_data = [[m.createData() for m in native_models] for _ in range(batch_size)]
    state = plan.states[0].expand(batch_size, -1).clone()
    if batch_size > 1:
        state[1, 19] += perturbation
    initial = state.clone()
    model = controller.dynamics.model
    timings = []
    force_violation = torque_violation = impulse_violation = oracle_error = 0.0
    minimum_normal = float("inf")
    minimum_height = 0.0
    peak_height = torch.zeros(batch_size, 4, dtype=torch.float64)
    landing_groups = []
    failed_node = None
    for t in range(len(plan.torques)):
        before = time.perf_counter_ns()
        control = controller.compute(state)
        timings.append((time.perf_counter_ns() - before) / 1e6)
        if not control.result.usable.all():
            failed_node = t
            break
        outputs = []
        for b in range(batch_size):
            native = native_models[t]
            data = native_data[b][t]
            x, u = state[b].numpy(), control.action[b].numpy()
            if control.durations[b] == 0:
                native.calc(data, x)
                collection = data.multibody.impulses.impulses.todict()
            else:
                native.calc(data, x, u)
                collection = data.differential.multibody.contacts.contacts.todict()
            expected = torch.from_numpy(data.xnext.copy())
            oracle_error = max(
                oracle_error, float((expected - control.result.xs[b, 1]).abs().max())
            )
            torch.testing.assert_close(expected, control.result.xs[b, 1], atol=1e-9, rtol=1e-9)
            outputs.append(expected if plant == "pinocchio" else control.result.xs[b, 1])
            force = np.stack([v.f.linear.copy() for v in collection.values()])
            violation = max(
                0.0,
                float((np.abs(force[:, :2]).sum(-1) - 0.6 * force[:, 2]).max()),
                float(-force[:, 2].min()),
            )
            if control.durations[b] == 0:
                impulse_violation = max(impulse_violation, violation)
                np.testing.assert_array_equal(expected[:19].numpy(), x[:19])
            else:
                force_violation = max(
                    force_violation, violation, float((force[:, 2] - 200).clip(0).max())
                )
                minimum_normal = min(minimum_normal, float(force[:, 2].min()))
                torque_violation = max(
                    torque_violation, float((np.abs(u) - model.torque_limits.numpy()).clip(0).max())
                )
        state = torch.stack(outputs)
        positions = model.foot_positions(state[:, :19])
        minimum_height = min(minimum_height, float(positions[..., 2].min()))
        if plan.durations[t] == 0:
            landing_groups.append(
                [leg for leg, active in zip(model.feet, plan.landings[t]) if active]
            )
        else:
            peak_height = torch.maximum(
                peak_height, torch.where(~plan.contacts[t], positions[..., 2], 0.0)
            )
    latency = torch.tensor(timings, dtype=torch.float64)
    return dict(
        gait=gait,
        cycles=cycles,
        batch_size=batch_size,
        plant=plant,
        python=platform.python_version(),
        torch=torch.__version__,
        platform=platform.platform(),
        cpu_threads=torch.get_num_threads(),
        dt_s=dt,
        step_length_m=step_length,
        step_height_m=step_height,
        step_knots=step_knots,
        support_knots=support_knots,
        preparation_seconds=preparation,
        optimizer_converged=list(plan.converged),
        optimizer_iterations=[s.iter for s in plan.native_solvers],
        planned_nodes=len(plan.torques),
        completed_nodes=len(timings) - (failed_node is not None),
        simulated_duration_s=float(
            plan.durations[: len(timings) - (failed_node is not None)].sum()
        ),
        all_actions_usable=failed_node is None,
        all_finished=bool(control.finished.all()),
        failed_node=failed_node,
        landing_groups=landing_groups,
        forward_displacement_m=(state[:, 0] - initial[:, 0]).tolist(),
        final_tracking_error=model.manifold.diff(state, plan.states[-1]).norm(dim=-1).tolist(),
        peak_swing_height_m=peak_height.tolist(),
        minimum_foot_height_m=minimum_height,
        minimum_normal_force_N=minimum_normal,
        max_force_violation_N=force_violation,
        max_torque_violation_Nm=torque_violation,
        max_impulse_violation_Ns=impulse_violation,
        max_native_dynamics_error=oracle_error,
        control_median_ms=float(latency.median()),
        control_p99_ms=float(latency.quantile(0.99)),
        control_max_ms=float(latency.max()),
        deadline_misses=int((latency > 20).sum()),
        latency_scope=(
            "Torch feedback/constraint QP when required/physics gates; "
            "excludes offline planner/native plant/IO"
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gait", choices=(*GO2_GAIT_SWINGS, "all"), default="all")
    parser.add_argument("--cycles", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=150)
    parser.add_argument("--plant", choices=("torch", "pinocchio"), default="pinocchio")
    parser.add_argument("--step-length", type=float, default=0.04)
    parser.add_argument("--step-height", type=float, default=0.02)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--step-knots", type=int, default=20)
    parser.add_argument("--support-knots", type=int, default=10)
    parser.add_argument("--save-dir", type=Path)
    args = vars(parser.parse_args())
    gait = args.pop("gait")
    output = args.pop("save_dir")
    if output is not None:
        output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    reports = []
    for name in GO2_GAIT_SWINGS if gait == "all" else (gait,):
        report = run_gait(
            name, save_path=output / f"{name}.pt" if output is not None else None, **args
        )
        reports.append(report)
        print(json.dumps(report, indent=2), flush=True)
    if not all(r["all_actions_usable"] and r["all_finished"] for r in reports):
        raise SystemExit("gait execution rejected: constraints/contact/dynamics check failed")


if __name__ == "__main__":
    main()
