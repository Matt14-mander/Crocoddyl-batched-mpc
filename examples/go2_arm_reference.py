"""Inspect/export the independent CPU Go2+Z1 model; no simulator or controller."""

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pinocchio as pin

from crocoddyl_batched_mpc.models.go2_arm import load_go2_arm


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--urdf-output", type=Path)
    args = parser.parse_args()
    robot = load_go2_arm(args.config)
    q = robot.standing_configuration()
    states = {"nominal": q}
    tilted = q.copy()
    tilted[:3] += [0.2, -0.15, 0.1]
    tilted[3:7] = np.array([0.12, -0.08, 0.17, 1]) / np.linalg.norm([0.12, -0.08, 0.17, 1])
    tilted[19:] = [0.3, 1.0, -1.6, 0.4, 0.2, -0.3]
    states["tilted"] = tilted
    cases = []
    for name, state in states.items():
        quantities = robot.evaluate(state)
        cases.append(
            {
                "name": name,
                "q": state.tolist(),
                "tcp_pose_world": quantities.tcp_pose.tolist(),
                "foot_positions_world_m": quantities.foot_positions.tolist(),
                "com_world_m": quantities.com.tolist(),
                "generalized_gravity": quantities.gravity.tolist(),
                "potential_energy_J": robot.potential_energy(state),
                "mass_matrix_min_eigenvalue": float(
                    np.linalg.eigvalsh(quantities.mass_matrix).min()
                ),
            }
        )
    lower, upper = robot.position_limits
    report = {
        "scope": "CPU rigid-body model inspection, not a standing/control/PhysX acceptance",
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "pinocchio": pin.__version__,
        "platform": platform.platform(),
        "config": robot.config,
        "urdf_sha256": hashlib.sha256(robot.urdf_xml.encode()).hexdigest(),
        "dimensions": {name: getattr(robot, name) for name in ("nq", "nv", "nx", "ndx", "nu")},
        "total_mass_kg": robot.total_mass,
        "joint_mapping": [
            {
                "name": name,
                "canonical_q_index": 7 + i,
                "canonical_v_index": 6 + i,
                "torque_index": i,
                "native_joint_id": robot.joint_ids[i],
                "native_q_index": int(robot.q_indices[7 + i]),
                "native_v_index": int(robot.v_indices[6 + i]),
                "lower_rad": float(lower[i]),
                "upper_rad": float(upper[i]),
                "effort_Nm": float(robot.torque_limits[i]),
                "velocity_rad_s": float(robot.velocity_limits[i]),
            }
            for i, name in enumerate(robot.joint_names)
        ],
        "cases": cases,
        "gravity_interpretation": (
            "First six components are unactuated base wrench. Joint components alone "
            "do not define a support-consistent standing torque; contact forces/WBC are pending."
        ),
    }
    payload = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
    if args.urdf_output:
        args.urdf_output.parent.mkdir(parents=True, exist_ok=True)
        args.urdf_output.write_text(robot.urdf_xml)
    print(payload, end="")


if __name__ == "__main__":
    main()
