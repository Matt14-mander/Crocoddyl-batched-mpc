"""Optional Meshcat playback of measured Go2 plant configurations.

Recording needs only Torch. Rendering imports the optional viewer/native stack
after control has finished, so it is excluded from control latency measurements.
"""

import html
import json
import math
from pathlib import Path

import torch


class Go2Trace:
    """One selected batch environment, including zero-time contact events."""

    def __init__(self, name, environment=0):
        if environment < 0:
            raise ValueError("environment must be nonnegative")
        self.name = name
        self.environment = environment
        self.configurations = []
        self.times = []
        self.contacts = []
        self.report = {}

    def append(self, state, duration, contacts):
        if state.ndim != 2 or state.shape[1] != 37:
            raise ValueError("expected batched Go2 states [B,37]")
        if self.environment >= state.shape[0]:
            raise ValueError("display environment exceeds batch size")
        q = state[self.environment, :19].detach().cpu()
        mask = contacts[self.environment].detach().cpu()
        if not torch.isfinite(q).all() or mask.shape != (4,):
            raise ValueError("invalid configuration/contact mask")
        duration = float(duration)
        if not math.isfinite(duration) or duration < 0 or (not self.times and duration != 0):
            raise ValueError("trace starts at zero and requires finite nonnegative durations")
        self.configurations.append(q.tolist())
        self.contacts.append(mask.bool().tolist())
        self.times.append((self.times[-1] if self.times else 0.0) + duration)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                dict(
                    format="go2-playback-v1",
                    name=self.name,
                    environment=self.environment,
                    configurations=self.configurations,
                    times=self.times,
                    contacts=self.contacts,
                    report=self.report,
                )
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path):
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        if value.get("format") != "go2-playback-v1":
            raise ValueError("unsupported Go2 playback format")
        trace = cls(value["name"], value["environment"])
        configurations, times, contacts = (
            value["configurations"],
            value["times"],
            value["contacts"],
        )
        if not configurations or not len(configurations) == len(times) == len(contacts):
            raise ValueError("empty or inconsistent playback")
        previous = 0.0
        for q, t, mask in zip(configurations, times, contacts):
            state = torch.zeros(1, 37, dtype=torch.float64)
            if len(q) != 19:
                raise ValueError("Go2 configuration must have 19 entries")
            state[0, :19] = torch.tensor(q, dtype=torch.float64)
            duration = float(t) - previous
            # Validate through the same recorder, independently of original batch index.
            original = trace.environment
            trace.environment = 0
            trace.append(state, duration, torch.tensor([mask]))
            trace.environment = original
            previous = float(t)
        trace.report = value.get("report", {})
        return trace


def animation_samples(times, fps=50):
    """Last pose wins at zero-time impacts; no extra simulation time is inserted."""
    if fps <= 0 or not math.isfinite(fps):
        raise ValueError("fps must be positive and finite")
    samples = {}
    for index, t in enumerate(times):
        samples[round(t * fps)] = index
    return sorted(samples.items())


def export_go2_html(trace, path, *, mesh_dir=None):
    """Export a standalone Meshcat scene with browser-native animation controls.

    mesh_dir is a ROS package search root containing go2_description/dae/.
    Without meshes, display the URDF's articulated collision primitives.
    """
    import contextlib
    import io

    import meshcat
    import meshcat.geometry as geometry
    import numpy as np
    import pinocchio as pin
    from meshcat.animation import Animation
    from pinocchio.visualize import MeshcatVisualizer

    from crocoddyl_batched_mpc.models.go2 import GO2_LEGS, load_go2

    if not trace.times:
        raise ValueError("cannot display an empty trace")
    robot = load_go2()
    collision = pin.buildGeomFromUrdf(robot.model, str(robot.urdf_path), pin.GeometryType.COLLISION)
    visual = collision
    if mesh_dir is not None:
        visual = pin.buildGeomFromUrdf(
            robot.model, str(robot.urdf_path), pin.GeometryType.VISUAL, [str(Path(mesh_dir))]
        )
    # Meshcat prints its transient server URL; this exporter returns a persistent file.
    with contextlib.redirect_stdout(io.StringIO()):
        viewer = meshcat.Visualizer()
    try:
        viz = MeshcatVisualizer(robot.model, collision, visual)
        viz.initViewer(viewer=viewer)
        viz.loadViewerModel(
            rootNodeName="go2", visual_color=None if mesh_dir else [0.65, 0.7, 0.8, 1]
        )
        viz.displayCollisions(False)
        from meshcat import transformations

        camera = transformations.rotation_matrix(-0.7, [0, 0, 1])
        camera[2, 3] = 0.15
        viewer["/Cameras/default"].set_transform(camera)
        viewer["/Cameras/default/rotated/<object>"].set_property("zoom", 3.5)
        viewer["/Grid"].set_property("visible", False)
        viewer["/Axes"].set_property("visible", False)
        viewer["/Background"].set_property("top_color", [0.94, 0.96, 0.99])
        viewer["/Background"].set_property("bottom_color", [0.8, 0.85, 0.9])
        floor = np.eye(4)
        floor[2, 3] = -0.016
        viewer["floor"].set_object(
            geometry.Box([3, 2, 0.03]), geometry.MeshLambertMaterial(color=0xDFE6ED)
        )
        viewer["floor"].set_transform(floor)
        for leg in GO2_LEGS:
            for phase, color in (("stance", 0x00A896), ("swing", 0xF59E0B)):
                viewer[f"feet/{leg}/{phase}"].set_object(
                    geometry.Sphere(0.025), geometry.MeshLambertMaterial(color=color)
                )
        frames = animation_samples(trace.times)
        animation = Animation(default_framerate=50)
        paths = [[] for _ in GO2_LEGS]
        for frame, i in frames:
            q = np.asarray(trace.configurations[i])
            pin.forwardKinematics(robot.model, viz.data, q)
            pin.updateFramePlacements(robot.model, viz.data)
            pin.updateGeometryPlacements(robot.model, viz.data, visual, viz.visual_data)
            with animation.at_frame(viewer, frame) as scene:
                for k, obj in enumerate(visual.geometryObjects):
                    transform = viz.visual_data.oMg[k].homogeneous.copy()
                    transform[:3, :3] *= np.asarray(obj.meshScale)[None, :]
                    scene[viz.getViewerNodeName(obj, pin.GeometryType.VISUAL)].set_transform(
                        transform
                    )
                for k, leg in enumerate(GO2_LEGS):
                    position = viz.data.oMf[robot.foot_ids[k]].translation.copy()
                    paths[k].append(position)
                    transform = np.eye(4)
                    transform[:3, 3] = position
                    for phase in ("stance", "swing"):
                        node = scene[f"feet/{leg}/{phase}"]
                        node.set_transform(transform)
                        visible = trace.contacts[i][k] == (phase == "stance")
                        node.set_property("visible", "boolean", visible)
        for leg, points in zip(GO2_LEGS, paths):
            viewer[f"paths/{leg}"].set_object(
                geometry.Line(
                    geometry.PointsGeometry(np.asarray(points).T),
                    geometry.LineBasicMaterial(color=0x437EA8, linewidth=2),
                )
            )
        viz.display(np.asarray(trace.configurations[0]))
        viewer.set_animation(animation, play=True, repetitions=1000000)
        document = viewer.static_html()
    finally:
        # meshcat-python 0.3.2's Visualizer.close calls a missing window.close.
        viewer.window.zmq_socket.close(linger=0)
        process = viewer.window.server_proc
        if process is not None:
            process.terminate()
            process.wait(timeout=5)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    label = html.escape(str(trace.name))
    shape = "Go2 外观网格" if mesh_dir else "Go2 URDF 碰撞几何"
    status = "完成" if trace.report.get("all_finished") else "部分轨迹 / 未完成"
    badge = (
        '<div style="position:fixed;left:16px;top:16px;z-index:10;background:#fffE;'
        'padding:12px;border-radius:8px;font:14px sans-serif;pointer-events:none">'
        f"<b>{label}</b> · 环境 {trace.environment} · {trace.times[-1]:.2f} s · {status}<br>"
        f"{shape} · 绿色：支撑足 · 橙色：摆动足<br>"
        "实际闭环轨迹 · 拖动旋转，滚轮缩放 · 右侧 Animation 控制回放</div>"
    )
    body_end = document.index(">", document.index("<body")) + 1
    document = document[:body_end] + badge + document[body_end:]
    path.write_text(document, encoding="utf-8")
    return path.resolve()


def export_gallery(pages, path):
    """Small local landing page; each gait keeps its independent playback timeline."""
    import os

    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    entries = [
        (str(name), os.path.relpath(Path(page).resolve(), path.parent))
        for name, page in pages.items()
    ]
    if not entries:
        raise ValueError("gallery requires at least one playback")
    buttons = "".join(
        f'<button data-page="{html.escape(page, quote=True)}">{html.escape(name)}</button>'
        for name, page in entries
    )
    path.write_text(
        '<!doctype html><html lang="zh-CN"><meta charset="utf-8">'
        "<title>Go2 步态 · Meshcat</title><style>"
        "body{margin:0;font:15px system-ui;background:#edf2f7}header{height:64px;"
        "display:flex;align-items:center;gap:12px;padding:0 20px}button{cursor:pointer;"
        "padding:9px 16px;border:1px solid #b9c8d8;border-radius:6px;background:white}"
        "button.active{background:#155e75;color:white}iframe{border:0;width:100%;"
        "height:calc(100vh - 64px)}</style><header><b>Go2 闭环步态</b>"
        + buttons
        + '</header><iframe title="Go2 Meshcat 动画"></iframe><script>'
        'const buttons=document.querySelectorAll("button");'
        'buttons.forEach(b=>b.onclick=()=>{buttons.forEach(x=>x.classList.remove("active"));'
        'b.classList.add("active");document.querySelector("iframe").src=b.dataset.page;});'
        "buttons[0].click();</script></html>",
        encoding="utf-8",
    )
    return path


def add_viewer_arguments(parser):
    parser.add_argument("--display", action="store_true", help="open exported Meshcat playback")
    parser.add_argument("--html", type=Path, help="HTML file (slow walk) or directory (four gaits)")
    parser.add_argument(
        "--trace", type=Path, help="trace file (slow walk) or directory (four gaits)"
    )
    parser.add_argument("--display-env", type=int, default=0, help="batch environment to record")
    parser.add_argument("--mesh-dir", type=Path, help="root containing go2_description/dae/")


def take_viewer_arguments(args):
    options = {
        key: args.pop(key) for key in ("display", "html", "trace", "display_env", "mesh_dir")
    }
    if not 0 <= options["display_env"] < args["batch_size"]:
        raise ValueError("--display-env must be within the batch")
    if options["display"] or options["html"]:
        import importlib.util

        if any(importlib.util.find_spec(name) is None for name in ("meshcat", "pinocchio")):
            raise RuntimeError('Meshcat playback requires pip install -e ".[viewer]"')
    return options
