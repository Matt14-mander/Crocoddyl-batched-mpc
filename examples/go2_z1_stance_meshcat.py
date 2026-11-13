"""Official mesh playback of validated, measured W-Guidance-WBC CPU states."""

import argparse
import hashlib
import json
import warnings
from pathlib import Path

import meshcat
import meshcat.geometry as geometry
import numpy as np
import pinocchio as pin
from go2_z1_meshcat import RecordedWindow, translate
from meshcat.animation import Animation
from pinocchio.visualize import MeshcatVisualizer

from crocoddyl_batched_mpc.models.go2_arm import load_go2_arm
from crocoddyl_batched_mpc.references.standing_arm import model_identity

ROOT = Path(__file__).resolve().parents[1]


def validate_trace(report, trace, robot):
    if (
        not report.get("passed")
        or not report["execution"]["passed"]
        or not report["planner"]["diagnostics"]["accepted_for_task"]
    ):
        raise ValueError("Refusing to export a failed W/CPU closed loop")
    if report["model_id"] != model_identity(robot):
        raise ValueError("Trace and viewer model identities differ")
    states, times = trace["states"], trace["times"]
    if (
        states.shape != (len(times), 49)
        or not np.isfinite(states).all()
        or np.any(np.diff(times) <= 0)
        or abs(times[0]) > 1e-12
    ):
        raise ValueError("Invalid measured state/time arrays")
    if (
        len(trace["torques"]) != len(times) - 1
        or report["execution"]["metrics"]["executed_steps"] != len(times) - 1
    ):
        raise ValueError("Measured trace is not a complete execution")
    if not np.isfinite(trace["tcp_actual"]).all():
        raise ValueError("Invalid measured TCP trace")
    # Check FK from executed configurations, never from IK or W predictions.
    actual = np.stack([robot.evaluate(x[:25]).tcp_pose for x in states])
    if not np.allclose(actual, trace["tcp_actual"], atol=1e-10, rtol=0):
        raise ValueError("Displayed TCP differs from executed-state FK")


def force_transform(origin, force):
    length = np.linalg.norm(force) * 0.002
    transform = translate(origin + 0.5 * 0.002 * force)
    if length > 0:
        direction = force / np.linalg.norm(force)
        x = np.cross([0, 1, 0], direction)
        if np.linalg.norm(x) < 1e-8:
            x = np.cross([1, 0, 0], direction)
        x /= np.linalg.norm(x)
        transform[:3, :3] = np.column_stack([x, direction, np.cross(x, direction)])
    return transform


def export(report_path, trace_path, output, assets):
    report = json.loads(Path(report_path).read_text())
    if hashlib.sha256(Path(trace_path).read_bytes()).hexdigest() != report["trace_sha256"]:
        raise ValueError("Trace hash differs from validation report")
    trace = np.load(trace_path, allow_pickle=False)
    robot = load_go2_arm()
    validate_trace(report, trace, robot)
    manifest = json.loads((assets / "MANIFEST.json").read_text())
    for item in manifest["files"]:
        if hashlib.sha256((assets / item["path"]).read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError(f"Visual asset hash mismatch: {item['path']}")
    model = robot.model
    urdf = ROOT / "src/crocoddyl_batched_mpc/assets/go2_arm/go2_z1_fixed_gripper.urdf"
    visual = pin.buildGeomFromUrdf(
        model, str(urdf), pin.GeometryType.VISUAL, package_dirs=[str(assets / "robots")]
    )
    window = RecordedWindow()
    viewer = meshcat.Visualizer(window=window)
    viz = MeshcatVisualizer(model, pin.GeometryModel(), visual)
    viz.initViewer(viewer=viewer)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        viz.loadViewerModel(rootNodeName="go2_z1")
    expected = {
        viewer[viz.getViewerNodeName(g, pin.GeometryType.VISUAL)].path.lower()
        for g in visual.geometryObjects
    }
    if not expected.issubset(set(window.objects)):
        raise RuntimeError("Some official robot meshes did not load")
    viz.displayCollisions(False)
    for name in ("/Axes", "/Grid"):
        viewer[name].set_property("visible", False)
    viewer["/Background"].set_property("top_color", [0.94, 0.96, 0.97])
    viewer["/Background"].set_property("bottom_color", [0.88, 0.92, 0.94])
    viewer["ground"].set_object(
        geometry.Box([2.4, 1.8, 0.012]), geometry.MeshLambertMaterial(color=0xE1E7EB)
    )
    ground = float(trace["foot_anchors"][:, 2].mean())
    viewer["ground"].set_transform(translate([0.15, 0, ground - 0.006]))
    for axis in (0, 1):
        for j in range(-8, 9):
            points = np.zeros((3, 2))
            points[axis] = j * 0.1
            points[1 - axis] = [-0.8, 0.8]
            points[2] = ground + 0.0002
            viewer[f"grid/{axis}_{j}"].set_object(
                geometry.Line(
                    geometry.PointsGeometry(points), geometry.LineBasicMaterial(color=0xC8D3D9)
                )
            )
    target, actual = trace["tcp_reference"][:, :3, 3], trace["tcp_actual"][:, :3, 3]
    viewer["reference"].set_object(
        geometry.Line(
            geometry.PointsGeometry(target.T),
            geometry.LineBasicMaterial(color=0xDB8B29, linewidth=3),
        )
    )
    viewer["tcp"].set_object(geometry.Sphere(0.008), geometry.MeshLambertMaterial(color=0x087F8C))
    for leg, anchor in enumerate(trace["foot_anchors"]):
        viewer[f"anchors/{leg}"].set_object(
            geometry.Sphere(0.009), geometry.MeshLambertMaterial(color=0x4C9A78)
        )
        viewer[f"anchors/{leg}"].set_transform(translate(anchor))
        viewer[f"forces/{leg}"].set_object(
            geometry.Cylinder(1, 0.003),
            geometry.MeshLambertMaterial(color=0x4C9A78, opacity=0.8, transparent=True),
        )
    # 200 Hz physics -> 40 Hz display, retaining actual configurations only.
    times = trace["times"]
    if not np.allclose(np.diff(times), 0.005, atol=1e-10, rtol=0):
        raise ValueError("This exporter requires the validated 200 Hz trace")
    indices = np.arange(0, len(times), 5)
    if indices[-1] != len(times) - 1:
        raise ValueError("Trace duration must align with the 40 Hz display grid")
    animation = Animation(default_framerate=40)
    for j in range(1, len(indices)):
        a, b = indices[j - 1], indices[j]
        node = viewer[f"actual_trace/{j:04d}"]
        node.set_object(
            geometry.Line(
                geometry.PointsGeometry(actual[[a, b]].T),
                geometry.LineBasicMaterial(color=0x087F8C, linewidth=3),
            )
        )
        animation.at_frame(node, 0).set_property("visible", "boolean", False)
        animation.at_frame(node, j).set_property("visible", "boolean", True)
        if j < len(indices) - 1:
            animation.at_frame(node, len(indices) - 1).set_property("visible", "boolean", True)
    for frame, k in enumerate(indices):
        q = robot.to_native_q(trace["states"][k, :25])
        pin.forwardKinematics(model, viz.data, q)
        pin.updateGeometryPlacements(model, viz.data, visual, viz.visual_data)
        for i, obj in enumerate(visual.geometryObjects):
            transform = viz.visual_data.oMg[i].homogeneous.copy()
            transform[:3, :3] *= np.asarray(obj.meshScale)[None, :]
            animation.at_frame(
                viewer[viz.getViewerNodeName(obj, pin.GeometryType.VISUAL)], frame
            ).set_transform(transform)
        animation.at_frame(viewer["tcp"], frame).set_transform(translate(actual[k]))
        for leg in range(4):
            force = trace["forces"][min(k, len(times) - 2), leg]
            force_frame = animation.at_frame(viewer[f"forces/{leg}"], frame)
            force_frame.set_transform(force_transform(trace["feet_actual"][k, leg], force))
            # Meshcat 0.3.2 set_transform records position/quaternion only.
            force_frame.set_property(
                "scale", "vector3", [1, float(0.002 * np.linalg.norm(force)), 1]
            )
    viewer.set_animation(animation, play=False, repetitions=1000000)
    slim = dict(
        duration=float(times[-1]),
        times=times[indices].tolist(),
        target_xz=target[indices][:, [0, 2]].tolist(),
        actual_xz=actual[indices][:, [0, 2]].tolist(),
        tcp_error_mm=(1000 * np.linalg.norm(target[indices] - actual[indices], axis=1)).tolist(),
        base_height_cm=(100 * trace["states"][indices, 2]).tolist(),
        pitch_deg=np.rad2deg(trace["base_rpy"][indices, 1]).tolist(),
        normal_forces=trace["forces"][np.minimum(indices, len(times) - 2), :, 2].tolist(),
        torque_fraction=(
            np.max(np.abs(trace["torques"]) / robot.torque_limits, axis=1)[
                np.minimum(indices, len(times) - 2)
            ]
        ).tolist(),
        metrics=report["execution"]["metrics"],
        converged=report["planner"]["converged"],
        cycle_timing=report["execution"]["cycle_timing"],
        modes=trace["modes"][indices].tolist(),
        visual_objects=len(visual.geometryObjects),
        paper=report["paper"],
    )
    template = Path(__file__).with_name("go2_z1_stance_shell.html")
    html = template.read_text().replace(
        "/*__MESHCAT_JS__*/",
        (Path(meshcat.__file__).parent / "viewer/dist/main.min.js").read_text(),
    )
    html = html.replace("/*__COMMANDS__*/", json.dumps(window.commands)).replace(
        "/*__REPORT__*/", json.dumps(slim, allow_nan=False)
    )
    licenses = (
        (assets / "LICENSE").read_text()
        + "\n\n"
        + (Path(meshcat.__file__).parent / "viewer/LICENSE").read_text()
    )
    html = html.replace(
        "<head>",
        "<!-- Official asset and viewer licenses:\n"
        + licenses.replace("--", "—")
        + "\n-->\n<head>",
        1,
    )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html)
    provenance = dict(
        assets=manifest,
        meshcat_version="0.3.2",
        robot_visual_objects=len(visual.geometryObjects),
        source="executed CPU states",
        physics_hz=200,
        display_hz=40,
        trace_sha256=report["trace_sha256"],
        html_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
        source_sha256={
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (Path(__file__), template, Path(__file__).with_name("go2_z1_meshcat.py"))
        },
    )
    output.with_name("viewer.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(
        json.dumps(
            dict(
                output=str(output),
                bytes=output.stat().st_size,
                visual_objects=len(visual.geometryObjects),
            )
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--assets", type=Path, default=ROOT / ".cache/go2-z1-visual-assets")
    args = parser.parse_args()
    export(args.report, args.trace, args.output, args.assets)
