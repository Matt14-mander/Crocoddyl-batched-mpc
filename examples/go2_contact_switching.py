"""Explicit four/three-foot modes, guarded touchdown and independent plant.

This is a contact-event acceptance example with two swing posture targets, not a gait.
A downward velocity disturbance at touchdown exercises nonzero impact impulses.
"""

import argparse
import json
import platform
import time

import torch

from crocoddyl_batched_mpc.models import Go2ContactSwitchingMPC, Go2FixedContactReference


def run_switching(*, steps=160, plant="torch", compile_physics=True):
    if steps < 130 or plant not in ("torch", "pinocchio"):
        raise ValueError("steps>=130 and a torch/pinocchio plant are required")
    started = time.perf_counter()
    controller = Go2ContactSwitchingMPC(
        batch_size=2,
        horizon=2,
        minimum_normal_force=0.0,
        force_margin=1e-4,
        compile_physics=compile_physics,
    )
    swing_pose = controller.controllers[0].reference[:19].clone()
    swing_pose[9] -= 0.05
    controller.prepare_mode(
        "release_fl", [False, True, True, True], reference_configuration=swing_pose
    )
    controller.prepare_mode("lower_fl", [False, True, True, True])
    state = controller.controllers[0].reference.expand(2, -1).clone()
    native = None
    if plant == "pinocchio":
        import numpy as np
        import pinocchio as pin

        native = [
            Go2FixedContactReference(
                feet=tuple(
                    f for f, active in zip(c.dynamics.feet, c.dynamics.contact_mask) if active
                ),
                reference_configuration=c.reference[:19].numpy(),
                stabilization=c.dynamics.gains,
            )
            for c in controller.controllers
        ]
    prepare_seconds = time.perf_counter() - started
    # Environment 0 releases FL; environment 1 remains in four-foot stance.
    select = torch.tensor([True, False])
    events, latencies, errors = [], [], []
    torque_violation = force_violation = inactive_force = max_swing_height = 0.0
    for step in range(steps):
        cycle_started = time.perf_counter_ns()
        if step in (20, 80):
            name = "release_fl"
        elif step in (35, 95):
            name = "lower_fl"
        elif step in (60, 120):
            name = "stance"
            state[0, 21] -= 0.03  # external downward velocity kick, not an actuator command
        else:
            name = None
        if name is not None:
            before = time.perf_counter_ns()
            event = controller.switch(name, state, mask=select)
            events.append(
                {
                    "step": step,
                    "mode": name,
                    "accepted": event.accepted.tolist(),
                    "event_ms": (time.perf_counter_ns() - before) / 1e6,
                    "impulse_Ns": event.impulses_world[0].tolist(),
                    "energy_before_J": float(event.kinetic_energy_before[0]),
                    "energy_after_J": float(event.kinetic_energy_after[0]),
                }
            )
            if not event.accepted[0]:
                raise RuntimeError(f"contact event rejected at step {step}; old mode retained")
            state = event.state
        action, result = controller.compute(state)
        latencies.append((time.perf_counter_ns() - cycle_started) / 1e6)
        if not result.usable.all():
            raise RuntimeError(f"no feasible mode trajectory at step {step}; plant stopped")
        outputs = []
        for b in range(2):
            index = int(controller.mode_indices[b])
            c = controller.controllers[index]
            if native is None:
                contact = c.dynamics.contact_dynamics(state[b : b + 1], action[b : b + 1])
                force = contact.forces_world
                output = c.dynamics.calc(state[b : b + 1], action[b : b + 1])[0]
            else:
                reference = native[index]
                q, v, u = state[b, :19].numpy(), state[b, 19:].numpy(), action[b].numpy()
                contact = reference.calc(q, v, u)
                velocity = v + 0.02 * contact.acceleration
                output = torch.from_numpy(
                    np.r_[pin.integrate(reference.robot.model, q, 0.02 * velocity), velocity]
                )
                torch.testing.assert_close(result.xs[b, 1], output, atol=1e-9, rtol=1e-9)
                force = torch.zeros(1, 4, 3, dtype=state.dtype)
                force[0, c.dynamics.contact_mask] = torch.from_numpy(contact.forces_world)
            tv, fv = c.constraints.violations(action[b : b + 1], force)
            torque_violation = max(torque_violation, float(tv.max()))
            force_violation = max(force_violation, float(fv.max()))
            if (~c.dynamics.contact_mask).any():
                inactive_force = max(
                    inactive_force, float(force[:, ~c.dynamics.contact_mask].abs().max())
                )
            outputs.append(output)
        state = torch.stack(outputs)
        max_swing_height = max(
            max_swing_height,
            float(controller.controllers[0].dynamics.foot_positions(state[:1, :19])[0, 0, 2]),
        )
        errors.append(
            controller.controllers[0]
            .dynamics.manifold.diff(state, controller.controllers[0].reference)
            .norm(dim=-1)
        )
    latency = torch.tensor(latencies, dtype=torch.float64)
    return {
        "plant": plant,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "platform": platform.platform(),
        "cpu_threads": torch.get_num_threads(),
        "dtype": str(controller.dtype),
        "steps": steps,
        "batch_size": 2,
        "horizon": 2,
        "prepare_seconds": prepare_seconds,
        "events": events,
        "all_actions_usable": True,
        "max_inactive_force_N": inactive_force,
        "max_swing_height_m": max_swing_height,
        "max_torque_violation_Nm": torque_violation,
        "max_force_violation_N": force_violation,
        "final_state_error": errors[-1].tolist(),
        "cycle_median_ms": float(latency.median()),
        "cycle_p99_ms": float(latency.quantile(0.99)),
        "cycle_max_ms": float(latency.max()),
        "deadline_misses": int((latency > 20).sum()),
        "latency_scope": "switch (on event cycles) and compute; excludes plant/sensors/IO",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=160)
    parser.add_argument("--plant", choices=("torch", "pinocchio"), default="torch")
    args = parser.parse_args()
    torch.set_num_threads(1)
    print(json.dumps(run_switching(**vars(args)), indent=2))


if __name__ == "__main__":
    main()
