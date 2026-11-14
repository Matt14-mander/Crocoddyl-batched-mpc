"""Paired W numerical/performance audit against an explicit local Git revision."""

import argparse
import hashlib
import json
import subprocess
import sys
import time
import types
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
from go2_z1_async_cpu import json_safe
from go2_z1_rolling_cpu import task_reference
from go2_z1_stance_demo import build_reference

from crocoddyl_batched_mpc.controllers.async_whole_body import (
    AsyncSettings,
    PlanRequest,
    StateSnapshot,
    WWorkerEngine,
)
from crocoddyl_batched_mpc.controllers.rolling_whole_body_cpu import preview_window
from crocoddyl_batched_mpc.guidance import SAMPLE_SHAPES
from crocoddyl_batched_mpc.guidance_batch import copy_guidance
from crocoddyl_batched_mpc.models.go2_arm import load_go2_arm
from crocoddyl_batched_mpc.planners.standing_whole_body import WholeBodySettings
from crocoddyl_batched_mpc.references.standing_arm import guidance_from_whole_body, model_identity

ROOT = Path(__file__).resolve().parents[1]


def reference_planner(revision):
    """Read the trusted repository's old planner; no checkout or dependency changes."""
    commit = subprocess.check_output(
        ["git", "rev-parse", "--verify", revision + "^{commit}"], cwd=ROOT, text=True
    ).strip()
    path = "src/crocoddyl_batched_mpc/planners/standing_whole_body.py"
    source = subprocess.check_output(["git", "show", f"{commit}:{path}"], cwd=ROOT)
    name = "crocoddyl_batched_mpc.planners._audit_reference"
    module = types.ModuleType(name)
    module.__package__ = "crocoddyl_batched_mpc.planners"
    sys.modules[name] = module
    exec(compile(source, f"{commit}:{path}", "exec"), module.__dict__)
    return module.StandingWholeBodyPlanner, commit, hashlib.sha256(source).hexdigest()


def measure_pair(functions, repeats=5, calls=5):
    for fn in functions:
        fn()
    timings = [[], []]
    for repeat in range(repeats):
        for index in (0, 1) if repeat % 2 == 0 else (1, 0):
            for _ in range(calls):
                start = time.perf_counter()
                functions[index]()
                timings[index].append(time.perf_counter() - start)
    return {
        name: dict(
            median_ms=float(np.median(values) * 1000),
            p95_ms=float(np.percentile(values, 95) * 1000),
            samples_s=values,
        )
        for name, values in zip(("before", "after"), timings)
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-commit", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--saved-before",
        type=Path,
        help="Optional trajectories.npz captured before editing the workspace",
    )
    args = parser.parse_args()
    if (args.output / "report.json").exists():
        parser.error("use a new output directory to preserve previous evidence")
    args.output.mkdir(parents=True, exist_ok=True)
    old_type, commit, old_hash = reference_planner(args.reference_commit)
    robot = load_go2_arm()
    settings = WholeBodySettings(nodes=24)
    arrays, cases = {}, []
    for name, common, times in (
        ("reach", task_reference(robot, 6.0), [0.0, 0.5, 1.0, 1.5]),
        ("high_low", build_reference(robot), [0.0, 2.0, 4.0, 6.0]),
    ):
        engines = [
            WWorkerEngine(robot.config, common, settings, AsyncSettings(), optimized=o)
            for o in (False, True)
        ]
        engines[0].planner = old_type(engines[0].robot, settings, common.nominal_states[0])
        previous = [e.planner._canonical_control(e.planner.static_control) for e in engines]
        for seq, now in enumerate(times, 1):
            pair = []
            for index, engine in enumerate(engines):
                state = common.nominal_states[int(round(now / 0.05))]
                snap = StateSnapshot(
                    now, state, previous[index], common.guidance.sample(now), False
                )
                request = PlanRequest("audit", seq, now, 0.0, 0.0, seq - 1, snap)
                start = time.perf_counter()
                packet, diagnostics = engine.solve(request)
                elapsed = time.perf_counter() - start
                plan = engine.pending[2]
                key = f"{name}_{seq}"
                values = dict(
                    states=plan.native_states,
                    controls=plan.native_controls,
                    forces=plan.forces_world,
                    acceleration=plan.accelerations,
                )
                values.update(
                    {
                        "guidance_" + f: np.stack([getattr(s, f) for s in packet.samples])
                        for f in SAMPLE_SHAPES
                    }
                )
                for field, value in values.items():
                    arrays[f"{index}_{key}_{field}"] = value.copy()
                pair.append(
                    dict(
                        total_s=elapsed,
                        diag=diagnostics,
                        diagnostics=plan.diagnostics,
                        cost=float(plan.mpc.cost[0]),
                        converged=plan.converged,
                    )
                )
                previous[index] = plan.mpc.us[0, 0].numpy().copy()
            errors = {
                f: float(np.max(np.abs(arrays[f"0_{key}_{f}"] - arrays[f"1_{key}_{f}"])))
                for f in values
            }
            # All gate metrics, including per-node values and failure lists, must match.
            gates = [
                {k: v for k, v in r["diagnostics"].items() if not k.endswith("time_s")}
                for r in pair
            ]
            passed = (
                max(errors.values()) == 0
                and gates[0] == gates[1]
                and pair[0]["cost"] == pair[1]["cost"]
                and pair[0]["converged"] == pair[1]["converged"]
                and pair[0]["diag"]["iterations"] == pair[1]["diag"]["iterations"]
            )
            cases.append(
                dict(name=key, passed=passed, errors=errors, before=pair[0], after=pair[1])
            )
        # Measure the same final input; reset/rebuild private problem data as appropriate.
        window = preview_window(robot, common, now, settings)
        reference = window.for_whole_body()
        plans = [e.pending[2] for e in engines]
        problems = [
            e.planner.build_problem(reference, previous[i], initial_state=state)
            for i, e in enumerate(engines)
        ]
        packets = [guidance_from_whole_body(robot, p, window, initial_state=state) for p in plans]
        measurements = {}
        measurements["construct"] = measure_pair(
            [
                lambda e=e, i=i: e.planner.build_problem(
                    reference, previous[i], initial_state=state
                )
                for i, e in enumerate(engines)
            ]
        )
        measurements["validate"] = measure_pair(
            [
                lambda e=e, p=p, problem=problem: e.planner.validate_candidate(
                    problem, p.native_states, p.native_controls, reference
                )
                for e, p, problem in zip(engines, plans, problems)
            ]
        )
        measurements["export"] = measure_pair(
            [
                lambda: guidance_from_whole_body(robot, plans[0], window, initial_state=state),
                lambda: engines[1].exporter.export(plans[1], window, initial_state=state),
            ]
        )
        measurements["receive_validate"] = measure_pair(
            [
                lambda: replace(packets[0], samples=tuple(replace(s) for s in packets[0].samples)),
                lambda: copy_guidance(packets[1]),
            ],
            calls=20,
        )
        for case in cases:
            if case["name"] == key:
                case["microbenchmark"] = measurements
    saved_errors = {}
    if args.saved_before:
        with np.load(args.saved_before) as saved:
            for key in saved:
                saved_errors[key] = float(np.max(np.abs(saved[key] - arrays["1_" + key])))
    paths = list((ROOT / "src/crocoddyl_batched_mpc").rglob("*.py")) + [Path(__file__)]
    report = dict(
        passed=all(c["passed"] for c in cases) and all(v == 0 for v in saved_errors.values()),
        reference_commit=commit,
        reference_planner_sha256=old_hash,
        settings=asdict(settings),
        model_id=model_identity(robot),
        cases=cases,
        saved_before_max_errors=saved_errors,
        source_sha256={
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths
        },
    )
    np.savez_compressed(args.output / "trajectories.npz", **arrays)
    report["trace_sha256"] = hashlib.sha256(
        (args.output / "trajectories.npz").read_bytes()
    ).hexdigest()
    (args.output / "report.json").write_text(json.dumps(json_safe(report), indent=2) + "\n")
    print(
        json.dumps(
            dict(
                passed=report["passed"],
                cases=len(cases),
                max_saved_error=max(saved_errors.values(), default=0),
            )
        )
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
