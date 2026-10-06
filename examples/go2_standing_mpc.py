"""Fixed-support standing MPC with a disturbance and an optional native plant.

This is a simulation correctness example, not a hardware controller. The Torch
plant is convenient; --plant pinocchio uses the independent CPU KKT reference.
"""

import argparse
import json
import time

import torch

from crocoddyl_batched_mpc import BatchedMPC, MPCController
from crocoddyl_batched_mpc.models import Go2FixedContactReference, go2_standing_problem


def run_standing(
    *,
    batch_size=2,
    steps=65,
    horizon=8,
    dt=0.02,
    max_iterations=1,
    backend="torch",
    plant="torch",
    device="cpu",
):
    if steps < 4 or batch_size < 1:
        raise ValueError("steps >=4 and batch_size >=1 are required")
    if plant not in ("torch", "pinocchio"):
        raise ValueError("plant must be torch or pinocchio")
    if plant == "pinocchio" and torch.device(device).type != "cpu":
        raise ValueError("the independent Pinocchio plant is CPU-only")
    problem = go2_standing_problem(
        batch_size, horizon, dt=dt, max_iterations=max_iterations, device=device
    )
    dynamics = problem.dynamics
    controller = MPCController(BatchedMPC(problem, backend=backend))
    reference = dynamics.standing_state.expand(batch_size, -1)
    delta = reference.new_zeros(batch_size, 36)
    delta[:, 3] = 0.02 * torch.where(torch.arange(batch_size, device=device) % 2 == 0, 1.0, -1.0)
    delta[:, 6:18] = 0.015
    delta[:, 18:] = 0.02
    state = dynamics.manifold.integrate(reference, delta)
    initial_error = dynamics.manifold.diff(state, reference).norm(dim=-1)
    native = None
    if plant == "pinocchio":
        import numpy as np
        import pinocchio as pin

        native = Go2FixedContactReference(stabilization=dynamics.gains)

    def advance(x, u):
        if native is None:
            result = dynamics.contact_dynamics(x, u)
            return dynamics.calc(x, u), result
        outputs, results = [], []
        for b in range(batch_size):
            q, v, torque = x[b, :19].numpy(), x[b, 19:].numpy(), u[b].numpy()
            result = native.calc(q, v, torque)
            velocity = v + dt * result.acceleration
            outputs.append(np.r_[pin.integrate(native.robot.model, q, dt * velocity), velocity])
            results.append(result)
        # CPU plant transfer is deliberate and confined to this oracle example.
        return torch.from_numpy(np.stack(outputs)).to(dtype=x.dtype), results

    histories, torques, normals, residuals, friction = [], [], [], [], []
    usable = torch.ones(batch_size, dtype=torch.bool, device=device)
    disturbance_step = steps // 2
    started = time.perf_counter()
    for step in range(steps):
        if step == disturbance_step:
            state[:, 22:25] += state.new_tensor([0.08, -0.06, 0.04])
        action, result = controller.compute(state)
        usable &= result.usable & torch.isfinite(action).all(-1)
        state, physical = advance(state, action)
        if native is None:
            forces = physical.forces_world
            residual = physical.contact_residual.abs().amax(-1)
        else:
            forces = state.new_tensor(np.stack([r.forces_world for r in physical]))
            residual = state.new_tensor([np.max(np.abs(r.contact_residual)) for r in physical])
        histories.append(dynamics.manifold.diff(state, reference))
        torques.append((action.abs() / dynamics.torque_limits).amax(-1))
        normals.append(forces[..., 2].amin(-1))
        residuals.append(residual)
        friction.append((forces[..., :2].norm(dim=-1) / forces[..., 2]).amax(-1))
    elapsed = time.perf_counter() - started
    history = torch.stack(histories)
    positions = dynamics.foot_positions(state[:, :19])
    report = {
        "plant": plant,
        "backend": backend,
        "device": str(dynamics.device),
        "batch_size": batch_size,
        "steps": steps,
        "dt": dt,
        "horizon": horizon,
        "iterations_per_solve": max_iterations,
        "disturbance_step": disturbance_step,
        "wall_seconds": elapsed,
        "all_actions_usable": bool(usable.all()),
        "initial_state_error": initial_error.tolist(),
        "final_state_error": history[-1].norm(dim=-1).tolist(),
        "final_configuration_error": history[-1, :, :18].norm(dim=-1).tolist(),
        "final_velocity_norm": state[:, 19:].norm(dim=-1).tolist(),
        "last_five_state_error_max": history[-5:].norm(dim=-1).amax(0).tolist(),
        "max_torque_limit_fraction": torch.stack(torques).amax(0).tolist(),
        "min_normal_force_N": torch.stack(normals).amin(0).tolist(),
        "max_tangential_normal_ratio": torch.stack(friction).amax(0).tolist(),
        "max_contact_acceleration_residual": torch.stack(residuals).amax(0).tolist(),
        "final_foot_position_error_m": (positions - dynamics.contact_positions)
        .norm(dim=-1)
        .amax(-1)
        .tolist(),
    }
    return report, state, history, controller


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--steps", type=int, default=65)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--max-iterations", type=int, default=1)
    parser.add_argument("--backend", choices=("torch", "fddp"), default="torch")
    parser.add_argument("--plant", choices=("torch", "pinocchio"), default="torch")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(1)
    report, _, _, _ = run_standing(**vars(args))
    print(json.dumps(report, indent=2))
    if not report["all_actions_usable"]:
        raise SystemExit("standing simulation produced an unusable MPC action")


if __name__ == "__main__":
    main()
