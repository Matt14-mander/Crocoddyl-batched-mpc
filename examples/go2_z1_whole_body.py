"""Reproduce the isolated Go2+Z1 W CPU baseline; no simulator or torque execution."""

import argparse
import hashlib
import json
import math
import platform
import sys
import time
from dataclasses import asdict
from pathlib import Path

import crocoddyl
import numpy as np
import pinocchio
import torch

from crocoddyl_batched_mpc.planners.standing_whole_body import (
    StandingWholeBodyPlanner,
    WholeBodySettings,
    smooth_tcp_reference,
)
from crocoddyl_batched_mpc.result import SolveStatus

CASES = {
    "hold": (0, 0, 0),
    "reach": (0.01, 0, 0),
    "side": (0, 0.02, 0),
    "combined": (0.01, 0.02, 0),
}


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return None if isinstance(value, float) and not math.isfinite(value) else value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--nodes", type=int, default=10)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--cases", choices=tuple(CASES), nargs="+", default=list(CASES))
    args = parser.parse_args()
    settings = WholeBodySettings(nodes=args.nodes, dt=args.dt, max_iterations=args.iterations)
    started = time.perf_counter()
    planner = StandingWholeBodyPlanner(settings=settings)
    initialization_s = time.perf_counter() - started
    root = Path(__file__).resolve().parents[1]
    paths = [
        "src/crocoddyl_batched_mpc/planners/standing_whole_body.py",
        "src/crocoddyl_batched_mpc/planners/_whole_body_native.py",
        "src/crocoddyl_batched_mpc/models/go2_arm.py",
        "examples/go2_z1_whole_body.py",
    ]
    report = {
        "scope": "offline CPU fixed-support W prediction; not WBC, PhysX or real-time acceptance",
        "versions": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "pinocchio": pinocchio.__version__,
            "crocoddyl": crocoddyl.__version__,
            "torch": torch.__version__,
            "platform": platform.platform(),
        },
        "source_sha256": {p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in paths},
        "urdf_sha256": hashlib.sha256(planner.robot.urdf_xml.encode()).hexdigest(),
        "simulation_config": planner.robot.config,
        "settings": asdict(settings),
        "initialization_time_s": initialization_s,
        "dimensions": {
            "physical_nx": 49,
            "physical_ndx": 48,
            "augmented_nx": 67,
            "augmented_ndx": 66,
            "nu": 18,
        },
        "joint_names": planner.robot.joint_names,
        "native_q_indices": planner.robot.q_indices.tolist(),
        "native_v_indices": planner.robot.v_indices.tolist(),
        "contact_order": planner.contact_names,
        "fixed_foot_anchors_world_m": planner.anchors.tolist(),
        "cases": [],
    }
    passed = True
    for name in args.cases:
        reference = smooth_tcp_reference(planner.robot, settings, CASES[name])
        result = planner.solve(reference)
        accepted = result.converged and result.diagnostics.get("accepted_for_task", False)
        passed = passed and accepted
        case = {
            "name": name,
            "displacement_world_m": CASES[name],
            "passed": accepted,
            "converged": result.converged,
            "status": SolveStatus(result.mpc.status.item()).name,
            "iterations": result.mpc.iterations.item(),
            "cost": result.mpc.cost.item(),
            "solve_time_s": result.solve_time_s,
            "diagnostics": result.diagnostics,
            "times_s": result.times.tolist(),
            "canonical_states": result.mpc.xs[0].tolist(),
            "canonical_torques_Nm": result.mpc.us[0].tolist(),
            "native_augmented_states": result.native_states.tolist(),
            "native_controls": result.native_controls.tolist(),
            "forces_world_N": None if result.forces_world is None else result.forces_world.tolist(),
            "canonical_accelerations": (
                None if result.accelerations is None else result.accelerations.tolist()
            ),
            "reference": {key: value.tolist() for key, value in vars(reference).items()},
        }
        report["cases"].append(case)
        compact = {
            "name": name,
            "passed": accepted,
            "status": case["status"],
            "iterations": case["iterations"],
            "solve_time_s": result.solve_time_s,
            "total_time_s": result.diagnostics["total_time_s"],
            "tcp_rmse_m": result.diagnostics.get("tcp_position_rmse_m"),
            "failures": result.diagnostics["failures"],
        }
        print(json.dumps(_json_safe(compact), allow_nan=False), flush=True)
    report["passed"] = passed
    payload = json.dumps(_json_safe(report), indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
