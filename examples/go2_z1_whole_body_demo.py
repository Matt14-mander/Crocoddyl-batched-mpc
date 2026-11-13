"""Export an independent Go2+Z1 W prediction replay. No renderer dependencies.

Uses the existing native CPU stack only while solving/FK exporting. The resulting
HTML is self-contained and does not execute the controller or connect to a server.
"""

import argparse
import hashlib
import json
import platform
import xml.etree.ElementTree as ET
from dataclasses import asdict
from pathlib import Path

import crocoddyl
import numpy as np
import pinocchio as pin

from crocoddyl_batched_mpc.planners.standing_whole_body import (
    StandingWholeBodyPlanner,
    WholeBodySettings,
    smooth_tcp_reference,
)


def round_trip_reference(planner, displacement=(0.06, 0.12, -0.03)):
    """A single out-and-back trajectory, solved jointly (never reversed playback)."""
    reference = smooth_tcp_reference(planner.robot, planner.settings)
    duration = reference.times[-1]
    phase = np.where(
        reference.times <= duration / 2,
        reference.times * 2 / duration,
        2 - reference.times * 2 / duration,
    )
    blend = 10 * phase**3 - 15 * phase**4 + 6 * phase**5
    speed = (30 * phase**2 - 60 * phase**3 + 30 * phase**4) * 2 / duration
    speed *= np.where(reference.times <= duration / 2, 1, -1)
    offset = np.asarray(displacement, dtype=float)
    if offset.shape != (3,) or not np.isfinite(offset).all():
        raise ValueError("displacement must be a finite vector [3]")
    reference.tcp_poses[:, :3, 3] += blend[:, None] * offset
    reference.tcp_velocities[:, :3] = speed[:, None] * offset
    return reference


def _origin(element):
    origin = element.find("origin")
    attrs = {} if origin is None else origin.attrib
    return pin.SE3(
        pin.rpy.rpyToMatrix(np.fromstring(attrs.get("rpy", "0 0 0"), sep=" ")),
        np.fromstring(attrs.get("xyz", "0 0 0"), sep=" "),
    )


def _pose(transform):
    return np.r_[transform.translation, pin.Quaternion(transform.rotation).coeffs()]


def _bone_pose(start, end):
    direction = end - start
    length = np.linalg.norm(direction)
    if length < 1e-10:
        raise ValueError("visual bone has zero length")
    z = direction / length
    helper = np.array([1, 0, 0]) if abs(z[0]) < 0.8 else np.array([0, 1, 0])
    y = np.cross(z, helper)
    y /= np.linalg.norm(y)
    x = np.cross(y, z)
    return pin.SE3(np.column_stack([x, y, z]), (start + end) / 2), length


def export_payload(planner, reference, plan):
    """Primitive geometry and true Pinocchio link placements at solved knots.

    Go2/bracket primitives come from the derived URDF. Z1 uses joint-centre bones
    and illustrative joint/gripper primitives, not a collision/visual mesh model.
    Browser interpolation between knots is only for display, not dynamics validation.
    """
    if not plan.converged or not plan.diagnostics.get("accepted_for_task", False):
        raise ValueError("refusing to label an unaccepted candidate as a validated demo")
    robot = planner.robot
    model = robot.model
    geometries, attachment = [], []
    tree = ET.fromstring(robot.urdf_xml)

    def frame(name):
        return model.getFrameId(name, pin.FrameType.BODY)

    for link in tree.findall("link"):
        name = link.attrib["name"]
        if name.startswith("z1_"):
            continue
        for index, collision in enumerate(link.findall("collision")):
            geom = collision.find("geometry")[0]
            if geom.tag == "box":
                size = np.fromstring(geom.attrib["size"], sep=" ").tolist()
            elif geom.tag == "sphere":
                size = [float(geom.attrib["radius"])]
            elif geom.tag == "cylinder":
                size = [float(geom.attrib["radius"]), float(geom.attrib["length"])]
            else:
                raise ValueError("demo requires primitive Go2 collision geometry")
            material = "mount" if name == "go2_z1_bracket" else "body"
            geometries.append(
                dict(name=f"{name}_{index}", kind=geom.tag, size=size, material=material)
            )
            attachment.append(("frame", frame(name), _origin(collision)))
    seed = model.createData()
    pin.framesForwardKinematics(model, seed, robot.to_native_q(planner.initial_state[:25]))
    for index in range(6):
        parent, child = frame(f"z1_link{index:02d}"), frame(f"z1_link{index + 1:02d}")
        _, length = _bone_pose(seed.oMf[parent].translation, seed.oMf[child].translation)
        geometries.append(
            dict(
                name=f"Z1_link_{index + 1}",
                kind="cylinder",
                size=[0.0225 if index in (1, 2) else 0.027, float(length)],
                material="arm",
            )
        )
        attachment.append(("bone", parent, child))
        geometries.append(
            dict(name=f"Z1_joint_{index + 1}", kind="sphere", size=[0.032], material="joint")
        )
        attachment.append(("frame", child, pin.SE3.Identity()))
    parent, child = frame("z1_link06"), frame("z1_gripperStator")
    _, length = _bone_pose(seed.oMf[parent].translation, seed.oMf[child].translation)
    geometries.append(
        dict(name="Z1_gripper_mount", kind="cylinder", size=[0.023, float(length)], material="arm")
    )
    attachment.append(("bone", parent, child))
    for name, xyz, size in (
        ("gripper_body", [0.018, 0, 0], [0.046, 0.05, 0.036]),
        ("gripper_finger_left", [0.055, 0.025, 0], [0.05, 0.012, 0.022]),
        ("gripper_finger_right", [0.055, -0.025, 0], [0.05, 0.012, 0.022]),
    ):
        geometries.append(dict(name=name, kind="box", size=size, material="arm"))
        attachment.append(("frame", frame("z1_gripperStator"), pin.SE3(np.eye(3), np.array(xyz))))

    frames = []
    states = plan.mpc.xs[0].numpy()
    for k, state in enumerate(states):
        q = robot.to_native_q(state[:25])
        data = model.createData()
        pin.framesForwardKinematics(model, data, q)
        placements = []
        for kind, parent, child in attachment:
            transform = (
                data.oMf[parent] * child
                if kind == "frame"
                else _bone_pose(data.oMf[parent].translation, data.oMf[child].translation)[0]
            )
            placements.extend(_pose(transform))
        tcp = data.oMf[robot.tcp_id].translation.copy()
        target = reference.tcp_poses[k, :3, 3]
        com = pin.centerOfMass(model, data, q).copy()
        feet = np.stack([data.oMf[i].translation for i in robot.foot_ids])
        torque = plan.mpc.us[0, min(k, len(states) - 2)].numpy()
        frames.append(
            {
                "t": round(float(plan.times[k]), 8),
                "poses": np.round(placements, 7).tolist(),
                "tcp": np.round(tcp, 8).tolist(),
                "target": np.round(target, 8).tolist(),
                "com": np.round(com, 8).tolist(),
                "gripper": np.round(data.oMf[frame("z1_gripperStator")].translation, 8).tolist(),
                "feet": np.round(feet, 8).tolist(),
                "forces": np.round(plan.forces_world[k], 6).tolist(),
                "error_mm": float(np.linalg.norm(tcp - target) * 1000),
                "torque_percent": float(np.max(np.abs(torque) / robot.torque_limits) * 100),
                "foot_drift_um": float(
                    np.max(np.linalg.norm(feet - planner.anchors, axis=1)) * 1e6
                ),
            }
        )
    return dict(
        version=1,
        duration=float(plan.times[-1]),
        geometries=geometries,
        frames=frames,
        anchors=planner.anchors.tolist(),
        source="Go2-Z1 W / offline CPU prediction / simplified geometry",
        accepted=True,
        diagnostics={
            key: plan.diagnostics[key]
            for key in (
                "tcp_position_rmse_m",
                "tcp_position_peak_m",
                "max_foot_position_error_m",
                "max_torque_fraction",
                "max_dynamics_gap",
            )
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/go2-z1-demo"))
    parser.add_argument("--fragment", type=Path, help="optional inline replay destination")
    args = parser.parse_args()
    planner = StandingWholeBodyPlanner(settings=WholeBodySettings(nodes=60, dt=0.05))
    reference = round_trip_reference(planner)
    plan = planner.solve(reference)
    args.output.mkdir(parents=True, exist_ok=True)
    report = dict(
        scope="Offline W prediction; no WBC, PhysX or torque execution",
        settings=asdict(planner.settings),
        converged=plan.converged,
        status=int(plan.mpc.status.item()),
        iterations=int(plan.mpc.iterations.item()),
        solve_time_s=plan.solve_time_s,
        diagnostics=plan.diagnostics,
        versions=dict(
            crocoddyl=crocoddyl.__version__,
            pinocchio=pin.__version__,
            numpy=np.__version__,
            platform=platform.platform(),
        ),
        urdf_sha256=hashlib.sha256(planner.robot.urdf_xml.encode()).hexdigest(),
        displacement_world_m=[0.06, 0.12, -0.03],
        duration_s=3.0,
    )
    (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    payload = export_payload(planner, reference, plan)
    (args.output / "scene.json").write_text(json.dumps(payload, separators=(",", ":")))
    trace = dict(
        times=plan.times.tolist(),
        states=plan.mpc.xs[0].tolist(),
        torques=plan.mpc.us[0].tolist(),
        forces_world=plan.forces_world.tolist(),
        reference={key: value.tolist() for key, value in vars(reference).items()},
    )
    (args.output / "trace.json").write_text(json.dumps(trace, indent=2, allow_nan=False) + "\n")
    here = Path(__file__).resolve().parent
    template = (here / "go2_z1_demo_fragment.html").read_text()
    serialized = json.dumps(payload, separators=(",", ":"), allow_nan=False).replace("<", "\\u003c")
    fragment = template.replace("__GO2_Z1_DATA__", serialized)
    if len(fragment.encode()) >= 1_000_000:
        raise ValueError("inline replay must remain below 1 MB")
    (args.output / "replay.html").write_text(fragment)
    shell = (here / "go2_z1_demo_shell.html").read_text()
    (args.output / "index.html").write_text(shell.replace("<!--REPLAY-->", fragment))
    if args.fragment:
        args.fragment.parent.mkdir(parents=True, exist_ok=True)
        args.fragment.write_text(fragment)
    print(
        json.dumps(
            dict(
                output=str((args.output / "index.html").resolve()),
                frames=len(payload["frames"]),
                geometries=len(payload["geometries"]),
                fragment_bytes=len(fragment.encode()),
                converged=plan.converged,
                accepted_for_task=plan.diagnostics["accepted_for_task"],
                tcp_rmse_mm=plan.diagnostics["tcp_position_rmse_m"] * 1000,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
