"""Initial C0/frozen-W CPU model comparison, common prescribed delivery schedule.

Native solvers really execute. Their wall cost is reported separately; virtual
latency grants both enough compute budget and is not a real-time speed claim.
"""

import argparse
import hashlib
import json
import platform
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pinocchio as pin
import torch
from go2_z1_async_cpu import json_safe
from go2_z1_rolling_cpu import task_reference
from go2_z1_stance_demo import build_reference

from crocoddyl_batched_mpc.controllers.async_cpu import run_async_cpu_episode
from crocoddyl_batched_mpc.controllers.async_timing import ExecutionClock, ScheduledWWorker
from crocoddyl_batched_mpc.controllers.async_whole_body import AsyncSettings, WorkerFault
from crocoddyl_batched_mpc.controllers.centroidal_cpu import ScheduledC0Worker
from crocoddyl_batched_mpc.guidance import SAMPLE_SHAPES
from crocoddyl_batched_mpc.models.go2_arm import load_go2_arm
from crocoddyl_batched_mpc.planners.standing_centroidal import CentroidalSettings
from crocoddyl_batched_mpc.planners.standing_whole_body import WholeBodySettings
from crocoddyl_batched_mpc.references.standing_arm import (
    StandingArmReferenceGenerator,
    model_identity,
    reach_task,
)

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "docs/benchmarks/go2_z1_w_frozen_2026-11-14/manifest.json"


def verify_w_baseline():
    manifest = json.loads(FROZEN.read_text())
    for group in ("source_sha256", "asset_sha256", "dependency_metadata_sha256", "evidence_sha256"):
        for name, expected in manifest[group].items():
            if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
                raise ValueError(f"frozen W drift: {name}")
    return manifest


def hold_reference(robot):
    q = robot.standing_configuration()
    times = np.arange(81) * 0.05
    return StandingArmReferenceGenerator(robot).generate(
        q, times, *reach_task(times, robot.evaluate(q).tcp_pose, [0.0, 0.0, 0.0])
    )


def execution_audit(robot, common, trace):
    """Actual COM/momenta, common task errors; no planner state replaces feedback."""
    com_errors, angular = [], []
    data = robot.model.createData()
    for t, x in zip(trace["times"], trace["states"]):
        pin.computeCentroidalMomentum(
            robot.model, data, robot.to_native_q(x[:25]), robot.to_native_v(x[25:])
        )
        reference = common.guidance.sample(float(min(t, common.guidance.valid_until)))
        com_errors.append(float(np.linalg.norm(data.com[0] - reference.com_position)))
        angular.append(float(np.linalg.norm(data.hg.angular)))
    return dict(
        common_com_peak_m=max(com_errors),
        common_com_rmse_m=float(np.sqrt(np.mean(np.square(com_errors)))),
        actual_angular_momentum_peak_kg_m2_s=max(angular),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--cases",
        nargs="+",
        default=["hold", "reach", "high_low", "failure", "expired"],
        choices=["hold", "reach", "high_low", "failure", "expired"],
    )
    parser.add_argument("--methods", nargs="+", choices=["C0", "W"], default=["W", "C0"])
    args = parser.parse_args()
    if (args.output / "report.json").exists():
        parser.error("use a new directory to preserve earlier results")
    args.output.mkdir(parents=True, exist_ok=True)
    frozen = verify_w_baseline()
    robot = load_go2_arm()
    paths = list((ROOT / "src/crocoddyl_batched_mpc").rglob("*.py")) + [Path(__file__)]
    report = dict(
        scope=(
            "Initial model-quality pairing with ample solve budget; "
            "prescribed delivery, not wall realtime"
        ),
        complete=False,
        requested_cases=args.cases,
        requested_methods=args.methods,
        frozen_w_id=frozen["baseline_id"],
        frozen_w_commit="cd00a27ecf736bf0856ad21d299aa5e0ef929ade",
        frozen_w_manifest_sha256=hashlib.sha256(FROZEN.read_bytes()).hexdigest(),
        frozen_hashes_verified=True,
        model_id=model_identity(robot),
        versions=dict(
            python=sys.version.split()[0],
            numpy=np.__version__,
            torch=torch.__version__,
            pinocchio=pin.__version__,
            platform=platform.platform(),
        ),
        source_sha256={
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths
        },
        cases={},
    )
    for name in args.cases:
        common = (
            build_reference(robot)
            if name == "high_low"
            else (hold_reference(robot) if name == "hold" else task_reference(robot, 6.0))
        )
        settings, faults = AsyncSettings(), {}
        if name == "failure":
            faults = {k: WorkerFault(fail=True) for k in (3, 4, 5, 6)}
        elif name == "expired":
            settings = replace(settings, result_timeout=2.0, max_state_age=2.0, hard_timeout=2.5)
            faults = {3: WorkerFault(delay=1.35)}
        pair = {}
        for method in args.methods:
            clock = ExecutionClock(
                virtual=True, phase_costs={"before_sample": 0.002, "after_solve": 0.008}
            )
            workers = []
            worker_type = ScheduledC0Worker if method == "C0" else ScheduledWWorker

            def factory(*a, **kw):
                worker = worker_type(*a, execution=clock, latency=0.25, **kw)
                workers.append(worker)
                return worker

            ps = CentroidalSettings() if method == "C0" else WholeBodySettings(nodes=24)
            started = time.perf_counter()
            case, trace, qp = run_async_cpu_episode(
                robot,
                common,
                settings=settings,
                planner_settings=ps,
                execution=clock,
                worker_factory=factory,
                faults=faults,
            )
            case["native_experiment_elapsed_s"] = time.perf_counter() - started
            case["method"] = method
            reference_hash = hashlib.sha256(common.nominal_states.tobytes())
            reference_hash.update(common.guidance.times.tobytes())
            for sample in common.guidance.samples:
                for field in SAMPLE_SHAPES:
                    reference_hash.update(getattr(sample, field).tobytes())
            case["shared_reference_sha256"] = reference_hash.hexdigest()
            com_reference = np.stack([sample.com_position for sample in common.guidance.samples])
            case["shared_com_reference_extent"] = dict(
                initial=com_reference[0].tolist(),
                minimum_delta=(com_reference - com_reference[0]).min(axis=0).tolist(),
                maximum_delta=(com_reference - com_reference[0]).max(axis=0).tolist(),
            )
            case["execution_audit"] = execution_audit(robot, common, trace)
            events = [e for e in case["events"] if e["kind"] == "delivery"]
            accepted = [e for e in events if e["accepted"]]
            rejected = [e for e in events if not e["accepted"]]
            checks = dict(
                cpu_gates=case["passed"],
                multiple_packets=len(accepted) >= 3,
                worker_closed=case["worker_closed"],
                full_duration=abs(case["integrated_duration_s"] - (common.guidance.times[-1] + 2.5))
                < 1e-8,
                shifted_seed_used=any(
                    e["diagnostics"].get("seed") == "shifted_acknowledged" for e in accepted
                ),
            )
            if name in ("failure", "expired"):
                reason = "worker_error" if name == "failure" else "expired"
                expected = [e for e in rejected if e["reason"] == reason]
                checks.update(
                    expected_rejection=bool(expected),
                    no_unexpected_rejections=all(e["reason"] == reason for e in rejected),
                    fallback=case["modes"].get("return_to_standing", 0) > 0,
                    recovery=bool(expected) and any(e["at"] > expected[-1]["at"] for e in accepted),
                )
            else:
                checks["no_rejected_delivery"] = not rejected
            if method == "C0":
                case["planner_attempts"] = workers[0].engine.history
            case["scenario_checks"], case["scenario_passed"] = checks, all(checks.values())
            key = f"{name}_{method}"
            np.savez_compressed(args.output / f"{key}.npz", **trace)
            case["trace_sha256"] = hashlib.sha256(
                (args.output / f"{key}.npz").read_bytes()
            ).hexdigest()
            (args.output / f"{key}_qp.json").write_text(
                json.dumps(json_safe(qp), allow_nan=False) + "\n"
            )
            pair[method] = case
            report["cases"][name] = pair
            report["passed"] = all(
                c["scenario_passed"] for pair in report["cases"].values() for c in pair.values()
            )
            (args.output / "report.json").write_text(
                json.dumps(json_safe(report), indent=2, allow_nan=False) + "\n"
            )
            print(
                json.dumps(
                    json_safe(
                        dict(
                            case=key,
                            passed=case["scenario_passed"],
                            failures=case["failures"],
                            checks=checks,
                            accepted=case["accepted"],
                            requests=case["requests"],
                            tcp_peak=case["metrics"]["tcp_peak"],
                            rotation_peak=case["metrics"]["rotation_peak"],
                            native_elapsed_s=case["native_experiment_elapsed_s"],
                        )
                    )
                ),
                flush=True,
            )
        if len(pair) == 2:
            assert pair["C0"]["shared_reference_sha256"] == pair["W"]["shared_reference_sha256"]
            for field in ("async_planner", "wall", "wbc"):
                assert pair["C0"]["settings"][field] == pair["W"]["settings"][field], field
            assert pair["C0"]["limits"] == pair["W"]["limits"]
            assert pair["C0"]["execution_settings"] == pair["W"]["execution_settings"]
    report["complete"] = True
    report["paired_contract_checked"] = len(args.methods) == 2
    (args.output / "report.json").write_text(
        json.dumps(json_safe(report), indent=2, allow_nan=False) + "\n"
    )
    verify_w_baseline()
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
