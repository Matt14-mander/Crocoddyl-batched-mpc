"""Isolated common-reference / W-Guidance / WBC CPU closed-loop acceptance."""

import argparse
import hashlib
import json
import math
import platform
import sys
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pinocchio
import torch

from crocoddyl_batched_mpc.controllers.standing_arm_cpu import run_cpu_episode
from crocoddyl_batched_mpc.controllers.whole_body_qp import WBCSettings
from crocoddyl_batched_mpc.models.go2_arm import load_go2_arm
from crocoddyl_batched_mpc.references.standing_arm import (
    StandingArmReferenceGenerator,
    guidance_from_whole_body,
    model_identity,
    reach_task,
)

ROOT = Path(__file__).resolve().parents[1]


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return None if isinstance(value, float) and not math.isfinite(value) else value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/go2-z1-wbc")
    parser.add_argument(
        "--cases",
        nargs="+",
        choices=("hold", "reference", "w_guidance", "push", "expired"),
        default=["hold", "reference", "w_guidance", "push", "expired"],
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    robot = load_go2_arm()
    q0 = robot.standing_configuration()
    times = np.linspace(0.0, 2.0, 41)
    generator = StandingArmReferenceGenerator(robot)
    tcp0 = robot.evaluate(q0).tcp_pose
    common = generator.generate(
        q0, times, *reach_task(times, tcp0, [0.01, 0.02, 0], return_to_start=True)
    )
    hold = generator.generate(q0, times, *reach_task(times, tcp0, [0, 0, 0]))
    settings = WBCSettings()
    sources = [
        "guidance.py",
        "references/standing_arm.py",
        "controllers/whole_body_qp.py",
        "controllers/standing_arm_cpu.py",
        "models/go2_arm_cpu.py",
        "models/go2_arm.py",
        "qp.py",
    ]
    source_paths = [ROOT / "src/crocoddyl_batched_mpc" / name for name in sources] + [
        Path(__file__)
    ]
    report = dict(
        scope="Fixed four-foot CPU feedback control, not PhysX or receding-horizon MPC",
        versions=dict(
            python=sys.version.split()[0],
            numpy=np.__version__,
            pinocchio=pinocchio.__version__,
            torch=torch.__version__,
            platform=platform.platform(),
        ),
        model_id=model_identity(robot),
        model_config=robot.config,
        wbc_settings=asdict(settings),
        reference_diagnostics=common.diagnostics,
        source_sha256={
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in source_paths
        },
        joint_names=robot.joint_names,
        cases={},
    )
    for name in args.cases:
        reference = hold if name == "hold" else common
        packet = reference.guidance
        planner_info = None
        wrench = None
        if name == "w_guidance":
            from crocoddyl_batched_mpc.planners.standing_whole_body import (
                StandingWholeBodyPlanner,
                WholeBodySettings,
            )

            planner = StandingWholeBodyPlanner(robot, WholeBodySettings(nodes=40, dt=0.05))
            result = planner.solve(common.for_whole_body())
            planner_info = dict(
                converged=result.converged,
                diagnostics=result.diagnostics,
                solve_time_s=result.solve_time_s,
            )
            try:
                packet = guidance_from_whole_body(robot, result, common)
            except ValueError as exc:
                report["cases"][name] = dict(
                    passed=False,
                    failures=["W_reference_rejected"],
                    reason=str(exc),
                    planner=planner_info,
                )
                print(json.dumps(_json_safe(report["cases"][name])), flush=True)
                continue
        elif name == "push":

            def wrench(t):
                return np.array([5.0, 0, 0, 0, 0, 0]) if 0.8 <= t < 0.9 else np.zeros(6)
        elif name == "expired":
            packet = replace(packet, valid_until=0.65)
        case, trace, diagnostics = run_cpu_episode(
            robot, reference, packet, settings=settings, external_wrench=wrench
        )
        np.savez_compressed(args.output / f"{name}.npz", **trace)
        case["planner"] = planner_info
        case["qp_diagnostics"] = diagnostics
        case["trace_sha256"] = hashlib.sha256(
            (args.output / f"{name}.npz").read_bytes()
        ).hexdigest()
        report["cases"][name] = case
        print(
            json.dumps(
                {
                    "case": name,
                    "passed": case["passed"],
                    "failures": case["failures"],
                    "metrics": case["metrics"],
                    "cycle_timing": case["cycle_timing"],
                }
            ),
            flush=True,
        )
    report["passed"] = all(case["passed"] for case in report["cases"].values())
    (args.output / "report.json").write_text(
        json.dumps(_json_safe(report), indent=2, allow_nan=False) + "\n"
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
