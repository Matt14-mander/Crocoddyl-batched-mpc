"""Receding-horizon Go2-Z1 CPU acceptance; no dependency installation or training."""

import argparse
import hashlib
import json
import platform
import sys
from dataclasses import asdict
from pathlib import Path

import crocoddyl
import numpy as np
import pinocchio
import torch

from crocoddyl_batched_mpc.controllers.rolling_whole_body_cpu import (
    RollingSettings,
    RollingWholeBodyCPU,
    run_rolling_cpu_episode,
)
from crocoddyl_batched_mpc.models.go2_arm import load_go2_arm
from crocoddyl_batched_mpc.references.standing_arm import (
    StandingArmReferenceGenerator,
    model_identity,
    reach_task,
)

ROOT = Path(__file__).resolve().parents[1]


def task_reference(robot, duration=2.0):
    q = robot.standing_configuration()
    times = np.arange(round(duration / 0.05) + 1) * 0.05
    return StandingArmReferenceGenerator(robot).generate(
        q,
        times,
        *reach_task(times, robot.evaluate(q).tcp_pose, [0.01, 0.02, 0], return_to_start=True),
    )


def inject_failures(rolling, name):
    """Deterministic test faults around the real solver; never a fake accepted plan."""
    solve = rolling.planner.solve
    calls = 0

    def wrapped(*args, **kwargs):
        nonlocal calls
        calls += 1
        if (name == "transient" and calls == 4) or (name == "outage" and 5 <= calls <= 24):
            raise RuntimeError("injected solver outage")
        plan = solve(*args, **kwargs)
        if name == "transient" and calls == 5:
            plan.diagnostics = {
                **plan.diagnostics,
                "accepted_for_task": False,
                "failures": ["injected_candidate_rejection"],
            }
        return plan

    rolling.planner.solve = wrapped


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/go2-z1-rolling-cpu")
    parser.add_argument(
        "--cases",
        nargs="+",
        choices=("warm", "cold", "transient", "outage", "push", "high_low"),
        default=["warm", "cold", "transient", "outage", "push", "high_low"],
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    robot = load_go2_arm()
    paths = [
        "src/crocoddyl_batched_mpc/" + p
        for p in (
            "guidance.py",
            "references/standing_arm.py",
            "planners/standing_whole_body.py",
            "planners/_whole_body_native.py",
            "controllers/rolling_whole_body_cpu.py",
            "controllers/standing_arm_cpu.py",
            "controllers/whole_body_qp.py",
            "models/go2_arm.py",
            "models/go2_arm_cpu.py",
            "qp.py",
        )
    ] + ["examples/go2_z1_rolling_cpu.py", "examples/go2_z1_stance_demo.py"]
    report = dict(
        scope="Synchronous receding-horizon CPU feedback; not wall-clock real-time or PhysX",
        versions=dict(
            python=sys.version.split()[0],
            numpy=np.__version__,
            pinocchio=pinocchio.__version__,
            crocoddyl=crocoddyl.__version__,
            torch=torch.__version__,
            platform=platform.platform(),
        ),
        model_id=model_identity(robot),
        source_sha256={p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in paths},
        cases={},
    )
    traces = {}
    for name in args.cases:
        if name == "high_low":
            from go2_z1_stance_demo import build_reference

            common = build_reference(robot)
        else:
            common = task_reference(robot, 4.0 if name == "outage" else 2.0)
        rolling = RollingWholeBodyCPU(
            robot, common, settings=RollingSettings(warm_start=name != "cold")
        )
        if name in ("transient", "outage"):
            inject_failures(rolling, name)
        wrench = None
        if name == "push":

            def wrench(t):
                return np.array([5.0, 0, 0, 0, 0, 0]) if 0.8 <= t < 0.9 else np.zeros(6)

        case, trace, qp = run_rolling_cpu_episode(
            robot, common, rolling=rolling, external_wrench=wrench
        )
        events = case["rolling"]["events"]
        expected_failures = {"transient": 2, "outage": 20}.get(name, 0)
        checks = {
            "expected_packet_failures": case["rolling"]["failed"] == expected_failures,
            "continued_after_failure": events[-1]["accepted"],
            "fixed_support": np.array_equal(
                rolling.planner.anchors, common.guidance.samples[0].foot_anchors
            ),
            "shifted_seed_used": name == "cold" or case["rolling"]["shifted_attempts"] > 0,
        }
        if name == "outage":
            recovery = events[24]
            checks.update(
                outage_recovered=recovery["accepted"] and recovery.get("recovery_started", False),
                expired_seed_discarded=recovery["seed"] == "common_trajectory",
                stood_before_recovery=bool(
                    np.any(
                        (trace["times"] > 1.81)
                        & (trace["times"] < 2.4)
                        & (trace["modes"] == "standing")
                    )
                ),
            )
        for key, passed in checks.items():
            if not passed:
                case["failures"].append(key)
        case["passed"] = not case["failures"]
        case["scenario_checks"] = checks
        case["settings"] = dict(
            rolling=asdict(rolling.settings), planner=asdict(rolling.planner.settings)
        )
        trace["plan_origins"] = np.array([t for t, _ in rolling.plans])
        if rolling.plans:
            trace["plan_states"] = np.stack([p.mpc.xs[0].numpy() for _, p in rolling.plans])
            trace["plan_torques"] = np.stack([p.mpc.us[0].numpy() for _, p in rolling.plans])
        np.savez_compressed(args.output / f"{name}.npz", **trace)
        case["trace_sha256"] = hashlib.sha256(
            (args.output / f"{name}.npz").read_bytes()
        ).hexdigest()
        (args.output / f"{name}_qp.json").write_text(json.dumps(qp, allow_nan=False) + "\n")
        traces[name] = trace
        report["cases"][name] = case
        print(
            json.dumps(
                {
                    "case": name,
                    "passed": case["passed"],
                    "failures": case["failures"],
                    "metrics": case["metrics"],
                    "rolling": {k: v for k, v in case["rolling"].items() if k != "events"},
                }
            ),
            flush=True,
        )
        (args.output / "report.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n"
        )
    if "warm" in traces and "cold" in traces:
        report["warm_cold_comparison"] = dict(
            max_state_difference=float(
                np.max(np.abs(traces["warm"]["states"] - traces["cold"]["states"]))
            ),
            warm_total_iterations=sum(
                e.get("iterations", 0) for e in report["cases"]["warm"]["rolling"]["events"]
            ),
            cold_total_iterations=sum(
                e.get("iterations", 0) for e in report["cases"]["cold"]["rolling"]["events"]
            ),
            note="Numerical equivalence and timing are reported; no guaranteed warm-start speedup",
        )
    report["passed"] = all(c["passed"] for c in report["cases"].values())
    (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
