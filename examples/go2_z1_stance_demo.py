"""Fig. 3-inspired high/low reaching: measured W -> Guidance -> WBC CPU trace."""

import argparse
import hashlib
import json
import platform
import sys
from dataclasses import asdict
from pathlib import Path

import crocoddyl
import numpy as np
import pinocchio as pin
import torch

from crocoddyl_batched_mpc.controllers.standing_arm_cpu import run_cpu_episode
from crocoddyl_batched_mpc.controllers.whole_body_qp import WBCSettings
from crocoddyl_batched_mpc.guidance import quintic
from crocoddyl_batched_mpc.models.go2_arm import load_go2_arm
from crocoddyl_batched_mpc.planners.standing_whole_body import (
    StandingWholeBodyPlanner,
    WholeBodySettings,
)
from crocoddyl_batched_mpc.references.standing_arm import (
    StandingArmReferenceGenerator,
    guidance_from_whole_body,
    model_identity,
)

ROOT = Path(__file__).resolve().parents[1]
KEY_TIMES = np.array([0, 1, 3.5, 4.5, 7.5, 8.5, 11, 12])
OFFSETS = np.array(
    [
        [0, 0, 0],
        [0, 0, 0],
        [0.06, 0, -0.20],
        [0.06, 0, -0.20],
        [0, 0, 0.12],
        [0, 0, 0.12],
        [0, 0, 0],
        [0, 0, 0],
    ]
)


def high_low_task(times, initial_pose):
    """Fixed orientation, C2 quintic segments with stationary dwell boundaries."""
    times = np.asarray(times, dtype=float)
    if (
        times.ndim != 1
        or not len(times)
        or not np.isfinite(times).all()
        or np.any(np.diff(times) <= 0)
    ):
        raise ValueError("increasing finite task times required")
    if times[0] < KEY_TIMES[0] or times[-1] > KEY_TIMES[-1]:
        raise ValueError("task time outside high/low script")
    poses = np.repeat(initial_pose[None], len(times), axis=0)
    velocities, accelerations = np.zeros((len(times), 6)), np.zeros((len(times), 6))
    for k, t in enumerate(times):
        j = min(np.searchsorted(KEY_TIMES, t, side="right") - 1, len(KEY_TIMES) - 2)
        p, v, a = quintic(
            OFFSETS[j],
            np.zeros(3),
            np.zeros(3),
            OFFSETS[j + 1],
            np.zeros(3),
            np.zeros(3),
            KEY_TIMES[j + 1] - KEY_TIMES[j],
            t - KEY_TIMES[j],
        )
        poses[k, :3, 3] += p
        velocities[k, :3], accelerations[k, :3] = v, a
    return poses, velocities, accelerations


def build_reference(robot):
    q0 = robot.standing_configuration()
    # Demo-only home posture, with room above and below; model defaults unchanged.
    q0[19:] = [0, 1.1, -1.1, 0, 0, 0]
    times = np.arange(241) * 0.05
    generator = StandingArmReferenceGenerator(
        robot,
        allow_base_rotation=True,
        base_translation_bound=(0.08, 0.06, 0.075),
        base_rotation_bound=0.3,
        max_iterations=200,
    )
    return generator.generate(q0, times, *high_low_task(times, robot.evaluate(q0).tcp_pose))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/go2-z1-stance-meshcat")
    parser.add_argument("--iterations", type=int, default=150)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    robot = load_go2_arm()
    common = build_reference(robot)
    print(json.dumps({"stage": "reference", **common.diagnostics}), flush=True)
    settings = WholeBodySettings(
        nodes=240,
        dt=0.05,
        max_iterations=args.iterations,
        base_translation_bound=(0.09, 0.07, 0.085),
        base_rotation_bound=0.35,
    )
    planner = StandingWholeBodyPlanner(robot, settings, initial_state=common.nominal_states[0])
    differential = planner._differential(crocoddyl.CostModelSum(planner.physical_state, 18))
    seed_controls = []
    for x in common.nominal_states[:-1]:
        stationary = planner._native_state(x)
        stationary[25:] = 0
        torque = differential.quasiStatic(differential.createData(), stationary)
        seed_controls.append(planner._canonical_control(torque))
    plan = planner.solve(
        common.for_whole_body(), initial_guess=(common.nominal_states, seed_controls)
    )
    report = dict(
        scope="Fixed-foot W-Guidance-WBC CPU feedback execution; offline playback, not online MPC",
        paper="https://arxiv.org/html/2511.19709v1#S6.SS2",
        paper_relationship=(
            "Fig. 3 high/low whole-body reaching adapted to fixed Go2-Z1 support; "
            "not B2 trotting or its inverse-dynamics MPC algorithm"
        ),
        versions=dict(
            python=sys.version.split()[0],
            numpy=np.__version__,
            torch=torch.__version__,
            pinocchio=pin.__version__,
            crocoddyl=crocoddyl.__version__,
            platform=platform.platform(),
        ),
        model_id=model_identity(robot),
        model_config=robot.config,
        urdf_sha256=hashlib.sha256(robot.urdf_xml.encode()).hexdigest(),
        joint_names=robot.joint_names,
        reference_diagnostics=common.diagnostics,
        task=dict(
            key_times=KEY_TIMES.tolist(),
            offsets_m=OFFSETS.tolist(),
            initial_q=common.nominal_states[0, :25].tolist(),
        ),
        planner=dict(
            settings=asdict(settings),
            converged=plan.converged,
            solve_time_s=plan.solve_time_s,
            diagnostics=plan.diagnostics,
        ),
        passed=False,
    )
    print(
        json.dumps(
            {
                "stage": "W",
                "converged": plan.converged,
                "accepted": plan.diagnostics.get("accepted_for_task", False),
                "failures": plan.diagnostics["failures"],
                "solver_error": plan.diagnostics.get("solver_error"),
            }
        ),
        flush=True,
    )
    try:
        packet = guidance_from_whole_body(robot, plan, common)
    except ValueError as exc:
        report["failure"] = str(exc)
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        return 1
    wbc = WBCSettings()
    case, trace, diagnostics = run_cpu_episode(robot, common, packet, settings=wbc, extra_time=1.5)
    # FK below reads measured plant states only; no substitution with planned configurations.
    values = [robot.evaluate(x[:25], x[25:]) for x in trace["states"]]
    trace.update(
        tcp_actual=np.stack([x.tcp_pose for x in values]),
        com_actual=np.stack([x.com for x in values]),
        feet_actual=np.stack([x.foot_positions for x in values]),
        foot_anchors=packet.samples[0].foot_anchors,
        plan_times=plan.times,
        plan_states=plan.mpc.xs[0].detach().cpu().numpy(),
        plan_torques=plan.mpc.us[0].detach().cpu().numpy(),
        common_states=common.nominal_states,
    )
    rpy = np.stack(
        [pin.rpy.matrixToRpy(pin.Quaternion(x[3:7]).toRotationMatrix()) for x in trace["states"]]
    )
    trace["base_rpy"] = rpy
    case["metrics"].update(
        base_height_min_m=float(trace["states"][:, 2].min()),
        base_height_max_m=float(trace["states"][:, 2].max()),
        pitch_min_deg=float(np.rad2deg(rpy[:, 1].min())),
        pitch_max_deg=float(np.rad2deg(rpy[:, 1].max())),
        tcp_height_min_m=float(trace["tcp_actual"][:, 2, 3].min()),
        tcp_height_max_m=float(trace["tcp_actual"][:, 2, 3].max()),
        max_torque_fraction=float(np.max(np.abs(trace["torques"]) / robot.torque_limits)),
    )
    np.savez_compressed(args.output / "trace.npz", **trace)
    report.update(
        passed=case["passed"],
        execution=case,
        wbc_settings=asdict(wbc),
        qp_diagnostics=diagnostics,
        trace_sha256=hashlib.sha256((args.output / "trace.npz").read_bytes()).hexdigest(),
    )
    sources = [
        Path(__file__),
        *[
            ROOT / "src/crocoddyl_batched_mpc" / p
            for p in (
                "references/standing_arm.py",
                "guidance.py",
                "controllers/standing_arm_cpu.py",
                "controllers/whole_body_qp.py",
                "models/go2_arm_cpu.py",
                "models/go2_arm.py",
                "planners/standing_whole_body.py",
                "planners/_whole_body_native.py",
                "qp.py",
            )
        ],
    ]
    report["source_sha256"] = {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "stage": "CPU",
                "passed": case["passed"],
                "failures": case["failures"],
                "metrics": case["metrics"],
                "cycle_timing": case["cycle_timing"],
            }
        ),
        flush=True,
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
