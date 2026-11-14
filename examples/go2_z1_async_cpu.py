"""Real spawned-planner and monotonic-clock Go2-Z1 CPU acceptance."""

import argparse
import hashlib
import json
import platform
import sys
from dataclasses import replace
from pathlib import Path

import crocoddyl
import numpy as np
import pinocchio
import torch
from go2_z1_rolling_cpu import task_reference

from crocoddyl_batched_mpc.controllers.async_cpu import run_async_cpu_episode
from crocoddyl_batched_mpc.controllers.async_whole_body import AsyncSettings, WorkerFault
from crocoddyl_batched_mpc.models.go2_arm import load_go2_arm
from crocoddyl_batched_mpc.references.standing_arm import model_identity

ROOT = Path(__file__).resolve().parents[1]


def json_safe(value):
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/go2-z1-async-cpu")
    parser.add_argument(
        "--cases",
        nargs="+",
        choices=("normal", "delay", "failure", "timeout", "expired", "restart", "high_low"),
        default=["normal", "delay", "failure", "timeout", "expired", "restart", "high_low"],
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    robot = load_go2_arm()
    paths = [
        "src/crocoddyl_batched_mpc/" + p
        for p in (
            "controllers/async_whole_body.py",
            "controllers/async_cpu.py",
            "controllers/async_timing.py",
            "models/go2_arm_async_cpu.py",
            "controllers/rolling_whole_body_cpu.py",
            "controllers/whole_body_qp.py",
            "guidance.py",
            "references/standing_arm.py",
            "planners/standing_whole_body.py",
            "planners/_whole_body_native.py",
            "models/go2_arm.py",
            "models/go2_arm_cpu.py",
            "qp.py",
        )
    ] + [
        "examples/go2_z1_async_cpu.py",
        "examples/go2_z1_rolling_cpu.py",
        "examples/go2_z1_stance_demo.py",
    ]
    report = dict(
        scope="Wall-clock asynchronous fixed-support CPU; no 200 Hz certification",
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
    for name in args.cases:
        common = task_reference(robot, 6.0)
        settings, faults = AsyncSettings(), {}
        if name == "high_low":
            from go2_z1_stance_demo import build_reference

            common = build_reference(robot)
        elif name == "delay":
            faults = {3: WorkerFault(delay=0.15)}
        elif name == "failure":
            faults = {i: WorkerFault(fail=True) for i in (3, 4, 5, 6)}
        elif name == "timeout":
            faults = {3: WorkerFault(delay=1.2)}
        elif name == "expired":
            settings = replace(settings, result_timeout=2.0, max_state_age=2.0, hard_timeout=2.5)
            faults = {3: WorkerFault(delay=1.35)}
        elif name == "restart":
            settings = replace(settings, result_timeout=0.5, hard_timeout=0.9)
            faults = {3: WorkerFault(delay=5.0)}
        case, trace, qp = run_async_cpu_episode(robot, common, settings=settings, faults=faults)
        deliveries = [e for e in case["events"] if e["kind"] == "delivery"]
        reasons = [e["reason"] for e in deliveries if not e["accepted"]]
        checks = dict(
            worker_closed=case["worker_closed"],
            multiple_packets=case["accepted"] >= 3,
            plant_wall_duration=abs(
                case["integrated_duration_s"]
                - (
                    common.guidance.times[-1]
                    - common.guidance.times[0]
                    + case["settings"]["wall"]["extra_time"]
                )
            )
            < 1e-6,
        )
        if name in ("normal", "high_low"):
            checks["no_rejected_delivery"] = not reasons
        if name == "delay":
            checks["delayed_packet_accepted"] = any(
                e["sequence"] == 3 and e["accepted"] and e["age_s"] >= 0.15 for e in deliveries
            )
        if name in ("failure", "timeout", "expired"):
            expected = {"failure": "worker_error", "timeout": "timeout", "expired": "expired"}[name]
            checks["expected_rejection"] = expected in reasons
            checks["fallback_executed"] = case["modes"].get("return_to_standing", 0) > 0
            rejected = [e["at"] for e in deliveries if e["reason"] == expected]
            checks["recovered"] = bool(rejected) and any(
                e["accepted"] and e["at"] > min(rejected) for e in deliveries
            )
        if name == "restart":
            restarts = [e["at"] for e in case["events"] if e["kind"] == "worker_restart"]
            checks["worker_restarted"] = bool(restarts)
            checks["recovered"] = bool(restarts) and any(
                e["accepted"] and e["at"] > restarts[0] for e in deliveries
            )
            checks["fallback_executed"] = case["modes"].get("return_to_standing", 0) > 0
        for key, ok in checks.items():
            if not ok:
                case["failures"].append(key)
        case["passed"] = not case["failures"]
        case["scenario_checks"] = checks
        np.savez_compressed(args.output / f"{name}.npz", **trace)
        case["trace_sha256"] = hashlib.sha256(
            (args.output / f"{name}.npz").read_bytes()
        ).hexdigest()
        (args.output / f"{name}_qp.json").write_text(json.dumps(qp, allow_nan=False) + "\n")
        report["cases"][name] = case
        report = json_safe(report)
        (args.output / "report.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n"
        )
        print(
            json.dumps(
                json_safe(
                    dict(
                        case=name,
                        passed=case["passed"],
                        failures=case["failures"],
                        termination=case["termination"],
                        requests=case["requests"],
                        accepted=case["accepted"],
                        rejections=reasons,
                        metrics=case["metrics"],
                        delivery_age=case["delivery_age"],
                        ipc_timing=case["ipc_timing"],
                        missed_slots=case["missed_control_slots"],
                    )
                )
            ),
            flush=True,
        )
    report["passed"] = all(c["passed"] for c in report["cases"].values())
    (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
