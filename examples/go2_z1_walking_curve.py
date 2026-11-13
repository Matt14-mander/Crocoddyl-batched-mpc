"""Solve and independently replay the isolated Go2-Z1 walking figure-eight task."""

import argparse
import hashlib
import json
import platform
import sys
import time
from dataclasses import asdict
from pathlib import Path

import crocoddyl
import numpy as np
import pinocchio

from crocoddyl_batched_mpc.planners.walking_whole_body_demo import (
    WalkingDemoSettings,
    WalkingWholeBodyDemo,
)

ROOT = Path(__file__).resolve().parents[1]


def save_result(planner, xs, us, gains, solver_info, output):
    plan_metrics, _ = planner.audit(xs, us)
    actual, controls, feedback = planner.feedback_replay(xs, us, gains)
    metrics, nodes = planner.audit(actual, controls)
    plan_gate, replay_gate = planner.acceptance(plan_metrics), planner.acceptance(metrics)
    paths = [
        "src/crocoddyl_batched_mpc/planners/walking_whole_body_demo.py",
        "src/crocoddyl_batched_mpc/planners/standing_whole_body.py",
        "src/crocoddyl_batched_mpc/planners/_whole_body_native.py",
        "src/crocoddyl_batched_mpc/models/go2_arm.py",
        "examples/go2_z1_walking_curve.py",
    ]
    model_hash = hashlib.sha256(planner.robot.urdf_xml.encode()).hexdigest()
    report = {
        "scope": (
            "Offline hybrid whole-body OCP with scheduled rigid point contacts; "
            "nominal CPU feedback replay"
        ),
        "settings": asdict(planner.config),
        "ocp_settings": asdict(planner.settings),
        "versions": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "pinocchio": pinocchio.__version__,
            "crocoddyl": crocoddyl.__version__,
            "platform": platform.platform(),
        },
        "source_sha256": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in paths},
        "urdf_sha256": model_hash,
        "model_configuration": planner.robot.config,
        "initial_canonical_state": planner.initial_state.tolist(),
        "joint_names": planner.robot.joint_names,
        "dimensions": {
            "q": 25,
            "v": 24,
            "physical_x": 49,
            "augmented_x": 67,
            "augmented_dx": 66,
            "flow_u": 18,
            "impact_u": 0,
        },
        "solver": solver_info,
        "plan_metrics": plan_metrics,
        "plan_acceptance": plan_gate,
        "replay_metrics": metrics,
        "replay_acceptance": replay_gate,
        "feedback": feedback,
        "disposition": {
            "passed": ["18-actuator canonical mapping"]
            + (
                ["independent Pinocchio dynamics replay", "walking and TCP tracking gates"]
                if replay_gate["passed"]
                else []
            ),
            "skipped": ["Isaac Lab / PhysX", "CUDA", "existing single-Go2 training execution"],
            "pending": [
                "receding-horizon MPC and WBC",
                "contact detection and collision checks",
                "disturbance robustness",
                "real-time and hardware validation",
            ]
            + ([] if solver_info["converged"] else ["solver convergence threshold"]),
        },
        "units": {
            "positions": "m",
            "angles": "rad",
            "time": "s",
            "controls": "Nm",
            "forces": "N on flow rows, Ns on zero-time impact rows",
        },
        "assumptions": [
            "Go2-Z1 mount and TCP use the existing simulation-only CPU model configuration.",
            "Flat ground, four point feet, crawl RR-FR-RL-FL; collisions are not checked.",
            "Perfectly inelastic zero-duration impulses; semi-implicit flow at configured dt.",
            "Mu=0.6 inscribed diamond cones; force 2..200 N per active support foot.",
            "0.5 mm floor soft-cost allowance; 1 mm independent numerical geometry gate.",
            "World-fixed vertical yz figure eight, fixed orientation; IK is warm start only.",
            "Replay starts at the same initial state; it checks model consistency, not robustness.",
        ],
    }
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    sizes = np.array([len(u) for u in us])

    def padded(sequence, width):
        result = np.zeros((len(sequence), *width))
        for i, value in enumerate(sequence):
            result[i, : len(value)] = value
        return result

    np.savez_compressed(
        output / "trace.npz",
        times=planner.reference.times,
        kinds=np.array([r["kind"] for r in planner.schedule]),
        contacts=np.stack([r["contacts"] for r in planner.schedule]),
        anchors=np.stack([r["anchors"] for r in planner.schedule]),
        canonical_states=np.stack([planner._canonical_state(x[:49]) for x in actual]),
        canonical_controls=np.stack(
            [planner._canonical_control(u) if len(u) else np.zeros(18) for u in controls]
        ),
        native_augmented_states=np.stack(actual),
        forces=np.stack([n["forces"] for n in nodes]),
        feet=np.stack([n["feet"] for n in nodes]),
        tcp_actual=np.stack([n["tcp"] for n in nodes]),
        tcp_reference=planner.reference.tcp_poses,
        com=np.stack([n["com"] for n in nodes]),
        plan_states=np.stack(xs),
        plan_controls=padded(us, (18,)),
        plan_feedback=padded(gains, (18, 66)),
        control_sizes=sizes,
        urdf_sha256=np.array(model_hash),
        config_json=np.array(json.dumps(asdict(planner.config))),
    )
    report["trace_sha256"] = hashlib.sha256((output / "trace.npz").read_bytes()).hexdigest()
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/go2-z1-meshcat")
    parser.add_argument("--iterations", type=int, default=250)
    parser.add_argument(
        "--warm-start", type=Path, help="Optional trace.npz from this task; never a pickle"
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    planner = WalkingWholeBodyDemo(config=WalkingDemoSettings(iterations=args.iterations))
    problem, xs, us = planner.build_walking_problem()
    seed = None
    if args.warm_start:
        checkpoint = np.load(args.warm_start, allow_pickle=False)
        if (
            checkpoint["urdf_sha256"].item()
            != hashlib.sha256(planner.robot.urdf_xml.encode()).hexdigest()
        ):
            raise ValueError("Warm start uses a different robot model")
        settings = json.loads(checkpoint["config_json"].item())
        if any(settings[k] != v for k, v in asdict(planner.config).items() if k != "iterations"):
            raise ValueError("Warm start uses a different task schedule")
        xs = list(checkpoint["plan_states"])
        us = [u[:n] for u, n in zip(checkpoint["plan_controls"], checkpoint["control_sizes"])]
        planner.audit(xs, us)
        seed = {
            "path": str(args.warm_start),
            "sha256": hashlib.sha256(args.warm_start.read_bytes()).hexdigest(),
        }
    solver = crocoddyl.SolverBoxFDDP(problem)
    solver.th_stop = 1e-6
    if args.verbose:
        solver.setCallbacks([crocoddyl.CallbackVerbose()])
    started = time.perf_counter()
    # Recompute defects even for checkpoints; feasibility is independently audited.
    converged = solver.solve(xs, us, args.iterations, False)
    info = dict(
        converged=bool(converged),
        iterations=int(solver.iter),
        cost=float(solver.cost),
        stop=float(solver.stop),
        stopping_threshold=solver.th_stop,
        solve_time_s=time.perf_counter() - started,
        warm_start=seed,
    )
    report = save_result(planner, solver.xs, solver.us, solver.K, info, args.output)
    print(
        json.dumps(
            {
                "solver": info,
                "replay_acceptance": report["replay_acceptance"],
                "replay_metrics": report["replay_metrics"],
            },
            indent=2,
        )
    )
    return 0 if report["replay_acceptance"]["passed"] and report["plan_acceptance"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
