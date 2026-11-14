"""Repeatable control-delay CPU regression; virtual replay and actual wall sleep."""

import argparse
import hashlib
import json
from dataclasses import replace
from functools import partial
from pathlib import Path

import numpy as np
from go2_z1_async_cpu import json_safe
from go2_z1_rolling_cpu import task_reference

from crocoddyl_batched_mpc.controllers.async_cpu import run_async_cpu_episode
from crocoddyl_batched_mpc.controllers.async_timing import (
    ControlDelay,
    ExecutionClock,
    ScheduledWWorker,
)
from crocoddyl_batched_mpc.controllers.async_whole_body import AsyncSettings, WorkerFault
from crocoddyl_batched_mpc.models.go2_arm import load_go2_arm

ROOT = Path(__file__).resolve().parents[1]


def delays_for(name):
    if name == "burst":
        return [
            ControlDelay(4.9, 0.15),
            ControlDelay(4.9, 0.11, "after_solve"),
            ControlDelay(5.38, 0.09),
            ControlDelay(5.5, 0.025, "after_solve"),
        ]
    if name == "boundary":
        return [ControlDelay(1.65, 0.12), ControlDelay(1.65, 0.09, "after_solve")]
    if name == "overrun":
        return [ControlDelay(0.5, 0.25)]
    return []


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["virtual", "wall"], default="virtual")
    p.add_argument(
        "--cases",
        nargs="+",
        choices=["expired", "burst", "boundary", "overrun"],
        default=["expired", "burst", "boundary", "overrun"],
    )
    p.add_argument("--repeat", type=int, default=2)
    p.add_argument("--output", type=Path, default=ROOT / "artifacts/go2-z1-delay-regression")
    args = p.parse_args()
    if args.repeat < 1:
        p.error("--repeat must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    robot = load_go2_arm()
    common = task_reference(robot, 6.0)
    paths = [p for p in (ROOT / "src/crocoddyl_batched_mpc").rglob("*.py")]
    paths += [
        Path(__file__),
        ROOT / "examples/go2_z1_async_cpu.py",
        ROOT / "examples/go2_z1_rolling_cpu.py",
    ]
    report = dict(
        mode=args.mode,
        source_sha256={
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths
        },
        cases={},
    )
    for name in args.cases:
        previous = None
        for repeat in range(args.repeat):
            clock = ExecutionClock(
                virtual=args.mode == "virtual",
                delays=delays_for(name),
                phase_costs={
                    "before_sample": 0.009,
                    "after_solve": 0.018,
                    "control_complete": 0.006,
                },
            )
            kw = dict(
                execution=clock,
                settings=replace(
                    AsyncSettings(), result_timeout=2.0, max_state_age=2.0, hard_timeout=2.5
                ),
                faults={3: WorkerFault(delay=1.35)},
            )
            if clock.virtual:
                kw["worker_factory"] = partial(ScheduledWWorker, execution=clock)
            case, trace, qp = run_async_cpu_episode(robot, common, **kw)
            checks = dict(
                worker_closed=case["worker_closed"],
                delays_executed=len(case["injected_delays"]) == len(delays_for(name)),
            )
            if name == "overrun":
                checks["expected_stop"] = (
                    bool(case["termination"]) and "catch-up bound" in case["termination"]["reason"]
                )
            else:
                checks["cpu_gates"] = case["passed"]
                expired = [e for e in case["events"] if e.get("reason") == "expired"]
                checks["expired_rejected"] = bool(expired)
                checks["recovered"] = bool(expired) and any(
                    e.get("accepted") and e["at"] > expired[0]["at"] for e in case["events"]
                )
                checks["fallback_executed"] = case["modes"].get("return_to_standing", 0) > 0
                checks["full_duration"] = abs(case["integrated_duration_s"] - 8.5) < 1e-8
                if name in ("burst", "boundary"):
                    checks["stale_command_discarded"] = case["discarded_commands"] > 0
            if clock.virtual and previous is not None:
                checks["repeat_identical"] = all(
                    np.array_equal(trace[k], previous[k]) for k in trace
                )
            previous = trace
            key = f"{name}_{repeat + 1}"
            np.savez_compressed(args.output / f"{key}.npz", **trace)
            case["checks"] = checks
            case["scenario_passed"] = all(checks.values())
            case["trace_sha256"] = hashlib.sha256(
                (args.output / f"{key}.npz").read_bytes()
            ).hexdigest()
            (args.output / f"{key}_qp.json").write_text(json.dumps(json_safe(qp)) + "\n")
            report["cases"][key] = json_safe(case)
            (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
            print(
                json.dumps(
                    json_safe(
                        dict(
                            case=key,
                            passed=case["scenario_passed"],
                            checks=checks,
                            failures=case["failures"],
                            tcp_peak=case["metrics"]["tcp_peak"],
                            rotation_peak=case["metrics"]["rotation_peak"],
                            termination=case["termination"],
                        )
                    )
                ),
                flush=True,
            )
    report["passed"] = all(c["scenario_passed"] for c in report["cases"].values())
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
