"""Go2 fixed-foot CPU oracle; does not run a simulator or a robot controller."""

import json

from crocoddyl_batched_mpc.models import Go2FixedContactReference, load_go2


def main():
    import numpy as np
    import pinocchio as pin

    robot = load_go2()
    reference = Go2FixedContactReference(robot, stabilization=(10.0, 3.0))
    q, v = reference.reference_configuration, np.zeros(robot.nv)
    torque = reference.quasi_static_torques()
    result = reference.calc(q, v, torque)
    oracle = reference.crocoddyl_model()
    data = oracle.createData()
    oracle.calc(data, np.r_[q, v], torque)
    oracle.calcDiff(data, np.r_[q, v], torque)
    contacts = [data.multibody.contacts.contacts[name] for name in reference.feet]
    da_dx, da_du, df_dx, df_du = reference.finite_difference_derivatives(q, v, torque)
    errors = {
        "acceleration": float(np.max(np.abs(result.acceleration - data.xout))),
        "force": float(
            np.max(np.abs(result.forces_world - np.stack([d.f.linear for d in contacts])))
        ),
        "da_dx": float(np.max(np.abs(da_dx - data.Fx))),
        "da_du": float(np.max(np.abs(da_du - data.Fu))),
        "df_dx": float(np.max(np.abs(df_dx - np.vstack([d.df_dx for d in contacts])))),
        "df_du": float(np.max(np.abs(df_du - np.vstack([d.df_du for d in contacts])))),
    }
    report = {
        "nq": robot.nq,
        "nv": robot.nv,
        "nx": robot.nx,
        "ndx": robot.ndx,
        "model_mass_kg": float(pin.computeTotalMass(robot.model)),
        "base_height_m": float(q[2]),
        "torque_Nm": torque.tolist(),
        "contact_forces_world_N": result.forces_world.tolist(),
        "max_acceleration": float(np.max(np.abs(result.acceleration))),
        "max_dynamics_residual": float(np.max(np.abs(result.dynamics_residual))),
        "max_contact_residual": float(np.max(np.abs(result.contact_residual))),
        "max_abs_errors_vs_crocoddyl": errors,
        "scope": "CPU bilateral point contacts; no friction/torque-constrained MPC or simulation",
    }
    print(json.dumps(report, indent=2))
    if report["max_acceleration"] > 1e-9 or errors["acceleration"] > 1e-9:
        raise SystemExit(1)
    if max(errors[k] for k in ("da_dx", "da_du", "df_dx", "df_du")) > 2e-5:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
