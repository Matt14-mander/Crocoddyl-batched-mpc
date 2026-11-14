"""Audit C1 pairing/recheck traces, feedback and standalone static capability.

Run after benchmarks, serially: this performs independent native calculations.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from go2_z1_async_cpu import json_safe
from go2_z1_c0_compare import verify_w_baseline

from crocoddyl_batched_mpc.models.configuration_centroidal import QuasiStaticCapability
from crocoddyl_batched_mpc.models.go2_arm import load_go2_arm
from crocoddyl_batched_mpc.models.go2_arm_cpu import FixedSupportCPU
from crocoddyl_batched_mpc.references.capability import (
    prepare_capable_reference,
    redistribute_configuration,
)
from crocoddyl_batched_mpc.references.centroidal import centroidal_measurement
from crocoddyl_batched_mpc.references.standing_arm import StandingArmReferenceGenerator, reach_task


def identical_trace(left, right):
    with np.load(left) as a, np.load(right) as b:
        assert set(a.files) == set(b.files), (left, right)
        for key in a.files:
            assert np.array_equal(a[key], b[key]), (left, right, key)
        return dict(equal=True, arrays=len(a.files))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paired", type=Path, required=True)
    parser.add_argument("--recheck", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("use a new output file")
    first = json.loads((args.paired / "report.json").read_text())
    final = json.loads((args.recheck / "report.json").read_text())
    assert first["complete"] and first["passed"] and first["paired_contract_checked"]
    assert final["complete"] and final["passed"] and final["requested_methods"] == ["C1"]
    verify_w_baseline()
    robot = load_go2_arm()
    summary, feedback, traces = {}, [], {}
    for name, pair in first["cases"].items():
        recheck = final["cases"][name]["C1"]
        assert pair["C1"]["shared_reference_sha256"] == recheck["shared_reference_sha256"]
        summary[name] = {}
        traces[name] = dict(
            C1_recheck=identical_trace(
                args.paired / f"{name}_C1.npz", args.recheck / f"{name}_C1.npz"
            ),
            W_frozen=identical_trace(
                args.paired / f"{name}_W.npz", args.baseline / f"{name}_W.npz"
            ),
            C0_baseline=identical_trace(
                args.paired / f"{name}_C0.npz", args.baseline / f"{name}_C0.npz"
            ),
        )
        for method, case in pair.items():
            attempts = case.get("planner_attempts")
            diagnostics = (
                [a["diagnostics"] for a in attempts]
                if attempts is not None
                else [
                    e["diagnostics"]
                    for e in case["events"]
                    if e["kind"] == "delivery" and e.get("diagnostics")
                ]
            )
            costs = [sum(d["phase_timings"].values()) * 1000 for d in diagnostics]
            summary[name][method] = dict(
                passed=case["scenario_passed"],
                accepted=case["accepted"],
                requests=case["requests"],
                tcp_peak_mm=1000 * case["metrics"]["tcp_peak"],
                rotation_peak_rad=case["metrics"]["rotation_peak"],
                original_task_tcp_peak_mm=1000 * case["original_task_tcp_peak_m"],
                execution=case["execution_audit"],
                native_request_count=len(costs),
                native_request_p50_ms=float(np.median(costs)),
                native_request_p95_ms=float(np.percentile(costs, 95)),
                phase_median_ms={
                    k: 1000 * float(np.median([d["phase_timings"][k] for d in diagnostics]))
                    for k in diagnostics[0]["phase_timings"]
                },
            )
            if method == "C1":
                static = [
                    node
                    for d in diagnostics
                    for node in d["candidate"]["configuration"]["static_nodes"]
                ]
                summary[name][method].update(
                    static_nodes=len(static),
                    static_peak_utilization=max(n["peak_utilization"] for n in static),
                    static_equilibrium_residual=max(n["equilibrium_residual"] for n in static),
                    predicted_static_peak_utilization=max(
                        d["candidate"]["metrics"]["quasi_static_peak_utilization"]
                        for d in diagnostics
                    ),
                    all_candidates_converged=all(d["converged"] for d in diagnostics),
                    recheck_native_request_p50_ms=float(
                        np.median(
                            [
                                sum(a["diagnostics"]["phase_timings"].values()) * 1000
                                for a in recheck["planner_attempts"]
                            ]
                        )
                    ),
                )
                with np.load(args.recheck / f"{name}_C1.npz") as trace:
                    plant = FixedSupportCPU(
                        robot, robot.evaluate(trace["states"][0, :25]).foot_positions
                    )
                    for attempt in recheck["planner_attempts"]:
                        state, tau = (
                            np.array(attempt["state"]),
                            np.array(attempt["previous_torque"]),
                        )
                        index = int(np.argmin(abs(trace["request_times"] - attempt["state_time"])))
                        assert abs(trace["request_times"][index] - attempt["state_time"]) < 1e-12
                        d = attempt["diagnostics"]
                        errors = dict(
                            state=float(np.max(abs(state - trace["request_states"][index]))),
                            previous_torque=float(
                                np.max(abs(tau - trace["request_previous_torques"][index]))
                            ),
                            momentum=float(
                                np.max(
                                    abs(
                                        centroidal_measurement(robot, state)
                                        - d["measured_centroidal_state"]
                                    )
                                )
                            ),
                            previous_force=float(
                                np.max(
                                    abs(
                                        plant.step(state[:25], state[25:], tau).forces
                                        - d["previous_force_estimate"]
                                    )
                                )
                            ),
                        )
                        assert max(errors.values()) <= 1e-12, (name, attempt["sequence"], errors)
                        feedback.append(
                            dict(case=name, sequence=attempt["sequence"], errors=errors)
                        )
    q = robot.standing_configuration()
    value = robot.evaluate(q)
    checker = QuasiStaticCapability(robot, value.foot_positions)
    probes = []
    for payload in (0, 1, 5, 6, 10):
        result = checker.evaluate(q, payload_mass_kg=payload)
        probes.append(
            dict(
                payload_mass_kg=payload,
                accepted=result.accepted,
                status=result.status,
                diagnostics=result.diagnostics,
                forces=result.forces.tolist(),
                torques=result.torques.tolist(),
            )
        )
    adjusted, redistribution = redistribute_configuration(
        checker, q, q, value.tcp_pose, payload_mass_kg=6
    )
    assert adjusted is not None
    actual = robot.evaluate(adjusted)
    redistribution.update(
        payload_mass_kg=6,
        initial_q=q.tolist(),
        adjusted_q=adjusted.tolist(),
        foot_error=float(np.max(abs(actual.foot_positions - value.foot_positions))),
        tcp_pose_error=float(np.max(abs(actual.tcp_pose - value.tcp_pose))),
    )
    times = np.arange(11) * 0.2
    common = StandingArmReferenceGenerator(robot).generate(
        q, times, *reach_task(times, value.tcp_pose, [0.01, 0, 0])
    )
    derated = QuasiStaticCapability(
        robot, checker.anchors, torque_limits=robot.torque_limits * 0.211
    )
    prepared, shared = prepare_capable_reference(robot, common, derated)
    shared.update(
        test_only_torque_scale=0.211,
        fixed_rotation_error=float(
            np.max(abs(prepared.nominal_states[:, 3:7] - common.nominal_states[:, 3:7]))
        ),
    )
    result = dict(
        scope=(
            "first three-way pairing and final C1 recheck; "
            "static payload probes are not dynamic payload acceptance"
        ),
        summary=summary,
        traces=traces,
        feedback=feedback,
        feedback_requests=len(feedback),
        feedback_max_errors={
            k: max(row["errors"][k] for row in feedback) for k in feedback[0]["errors"]
        },
        payload_probes=probes,
        redistribution=redistribution,
        shared_redistribution=shared,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(json_safe(result), indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            json_safe(
                dict(
                    feedback_requests=len(feedback),
                    feedback_max_errors=result["feedback_max_errors"],
                    all_traces_identical=True,
                )
            )
        )
    )


if __name__ == "__main__":
    main()
