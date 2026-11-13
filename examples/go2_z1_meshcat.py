"""Export native Meshcat commands, official meshes and a CPU trace as standalone HTML.

Only the transport is replaced: Meshcat's own geometry, animation, MessagePack
protocol and JavaScript viewer are used. Gzip compresses bytes, never mesh topology.
"""

import argparse
import base64
import gzip
import hashlib
import json
import warnings
from pathlib import Path

import meshcat
import meshcat.geometry as geometry
import numpy as np
import pinocchio as pin
import umsgpack
from meshcat.animation import Animation
from pinocchio.visualize import MeshcatVisualizer

from crocoddyl_batched_mpc.models.go2_arm import load_go2_arm

ROOT = Path(__file__).resolve().parents[1]


class RecordedWindow:
    """In-memory Meshcat transport: no daemon, ports or environment changes."""

    def __init__(self):
        self.commands = []
        self.objects = []

    def send(self, command):
        value = command.lower()
        if value["type"] == "set_object":
            self.objects.append(value["path"])
        packed = umsgpack.packb(value)
        self.commands.append(base64.b64encode(gzip.compress(packed, mtime=0)).decode())


def translate(position):
    transform = np.eye(4)
    transform[:3, 3] = position
    return transform


def export(report_path, trace_path, output, assets):
    report = json.loads(Path(report_path).read_text())
    if hashlib.sha256(Path(trace_path).read_bytes()).hexdigest() != report["trace_sha256"]:
        raise ValueError("Trace hash differs from its validation report")
    trace = np.load(trace_path, allow_pickle=False)
    if not report["replay_acceptance"]["passed"]:
        raise ValueError("Refusing to publish a replay that failed the dynamics gates")
    robot = load_go2_arm()
    if report["urdf_sha256"] != hashlib.sha256(robot.urdf_xml.encode()).hexdigest():
        raise ValueError("Trace and viewer model hashes differ")
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
    # Missing meshes are fatal; do not silently fall back to substitute geometry.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        viz.loadViewerModel(rootNodeName="go2_z1")
    expected = {
        viewer[viz.getViewerNodeName(g, pin.GeometryType.VISUAL)].path.lower()
        for g in visual.geometryObjects
    }
    if not expected.issubset(set(window.objects)):
        raise RuntimeError("Some robot visual objects were not loaded")
    viz.displayCollisions(False)
    viewer["/Axes"].set_property("visible", False)
    viewer["/Grid"].set_property("visible", False)
    viewer["/Background"].set_property("top_color", [0.94, 0.96, 0.98])
    viewer["/Background"].set_property("bottom_color", [0.87, 0.91, 0.94])
    viewer["ground"].set_object(
        geometry.Box([2.8, 2.0, 0.012]), geometry.MeshLambertMaterial(color=0xE1E7EB)
    )
    viewer["ground"].set_transform(translate([0.25, 0, -0.012]))
    for axis in (0, 1):
        for i in range(-7, 8):
            points = np.zeros((3, 2))
            points[axis] = i * 0.1
            points[1 - axis] = [-0.7, 0.9]
            points[2] = -0.0055
            viewer[f"grid/{axis}_{i}"].set_object(
                geometry.Line(
                    geometry.PointsGeometry(points),
                    geometry.LineBasicMaterial(color=0xC1CDD5, linewidth=1),
                )
            )
    target = trace["tcp_reference"][:, :3, 3]
    actual = trace["tcp_actual"][:, :3, 3]
    viewer["reference"].set_object(
        geometry.Line(
            geometry.PointsGeometry(target.T),
            geometry.LineBasicMaterial(color=0xD58421, linewidth=4),
        )
    )
    viewer["tcp"].set_object(geometry.Sphere(0.008), geometry.MeshLambertMaterial(color=0x138CA0))
    times = trace["times"]
    fps = round(1 / report["settings"]["dt"])
    animation = Animation(default_framerate=fps)
    # Impacts have zero duration and unchanged q: retain the last state at that time.
    index_by_frame = {round(t * fps): k for k, t in enumerate(times)}
    indices = list(index_by_frame.values())
    for j in range(1, len(indices)):
        a, b = indices[j - 1], indices[j]
        node = viewer[f"trace/{j:04d}"]
        node.set_object(
            geometry.Line(
                geometry.PointsGeometry(actual[[a, b]].T),
                geometry.LineBasicMaterial(color=0x138CA0, linewidth=4),
            )
        )
        animation.at_frame(node, 0).set_property("visible", "boolean", False)
        animation.at_frame(node, round(times[b] * fps)).set_property("visible", "boolean", True)
        if round(times[b] * fps) < round(times[-1] * fps):
            animation.at_frame(node, round(times[-1] * fps)).set_property(
                "visible", "boolean", True
            )
    for frame, k in index_by_frame.items():
        q = robot.to_native_q(trace["canonical_states"][k, :25])
        pin.forwardKinematics(model, viz.data, q)
        pin.updateGeometryPlacements(model, viz.data, visual, viz.visual_data)
        for i, obj in enumerate(visual.geometryObjects):
            transform = viz.visual_data.oMg[i].homogeneous.copy()
            transform[:3, :3] *= np.asarray(obj.meshScale)[None, :]
            node = viewer[viz.getViewerNodeName(obj, pin.GeometryType.VISUAL)]
            animation.at_frame(node, frame).set_transform(transform)
        animation.at_frame(viewer["tcp"], frame).set_transform(translate(actual[k]))
    viewer.set_animation(animation, play=False, repetitions=1000000)
    slim = {
        "duration": float(times[-1]),
        "times": times[indices].tolist(),
        "contacts": trace["contacts"][indices].astype(int).tolist(),
        "tcp_error_mm": (1000 * np.linalg.norm(actual[indices] - target[indices], axis=1)).tolist(),
        "forward_cm": (
            100 * (trace["canonical_states"][indices, 0] - trace["canonical_states"][0, 0])
        ).tolist(),
        "metrics": report["replay_metrics"],
        "converged": report["solver"]["converged"],
        "steps": int(sum(kind == "impact" for kind in trace["kinds"])),
        "visual_objects": len(visual.geometryObjects),
        "target_yz": target[indices, 1:].tolist(),
        "actual_yz": actual[indices, 1:].tolist(),
    }
    viewer_js = (Path(meshcat.__file__).parent / "viewer/dist/main.min.js").read_text()
    template = Path(__file__).with_name("go2_z1_meshcat_shell.html").read_text()
    html = template.replace("/*__MESHCAT_JS__*/", viewer_js)
    html = html.replace("/*__COMMANDS__*/", json.dumps(window.commands))
    html = html.replace("/*__REPORT__*/", json.dumps(slim, allow_nan=False))
    licenses = (
        (assets / "LICENSE").read_text()
        + "\n\n"
        + (Path(meshcat.__file__).parent / "viewer/LICENSE").read_text()
    )
    html = html.replace(
        "<head>",
        "<!-- Embedded visual asset / viewer licenses:\n"
        + licenses.replace("--", "—")
        + "\n-->\n<head>",
        1,
    )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html)
    provenance = {
        "assets": manifest,
        "meshcat_version": "0.3.2",
        "robot_visual_objects": len(visual.geometryObjects),
        "trace_sha256": report["trace_sha256"],
        "html_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "source_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (Path(__file__), Path(__file__).with_name("go2_z1_meshcat_shell.html"))
        },
    }
    output.with_name("viewer.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return {
        "output": str(output),
        "bytes": output.stat().st_size,
        "visual_objects": len(visual.geometryObjects),
        "unique_official_meshes": manifest["visual_meshes"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--assets", type=Path, default=ROOT / ".cache/go2-z1-visual-assets")
    args = parser.parse_args()
    print(json.dumps(export(args.report, args.trace, args.output, args.assets), indent=2))
