"""Audit the isolated Z1 asset with an existing Pinocchio installation.

No simulator, network request or dependency installation is performed.
"""

import hashlib
import json
import platform
import sys
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pinocchio as pin

directory = Path(__file__).resolve().parent
path = directory / "z1_fixed_gripper.urdf"
root = ET.parse(path).getroot()
links = []
for link in root.findall("link"):
    inertial = link.find("inertial")
    inertia = inertial.find("inertia").attrib
    matrix = np.array(
        [[float(inertia["i" + a + b if a <= b else "i" + b + a]) for b in "xyz"] for a in "xyz"]
    )
    eigenvalues = np.linalg.eigvalsh(matrix)
    assert np.isfinite(eigenvalues).all() and eigenvalues.min() > 0
    assert eigenvalues[-1] <= eigenvalues[0] + eigenvalues[1] + 1e-12
    mass = float(inertial.find("mass").attrib["value"])
    assert mass > 0
    links.append(
        {
            "name": link.attrib["name"],
            "mass_kg": mass,
            "com_in_link_m": [float(x) for x in inertial.find("origin").attrib["xyz"].split()],
            "inertia_eigenvalues_kg_m2": eigenvalues.tolist(),
        }
    )
joints = []
for joint in root.findall("joint"):
    if joint.attrib["type"] != "revolute":
        continue
    joints.append(
        {
            "name": joint.attrib["name"],
            "axis": [float(x) for x in joint.find("axis").attrib["xyz"].split()],
            "origin_xyz_m": [float(x) for x in joint.find("origin").attrib["xyz"].split()],
            "limits": {k: float(v) for k, v in joint.find("limit").attrib.items()},
            "upstream_dynamics": {k: float(v) for k, v in joint.find("dynamics").attrib.items()},
        }
    )
model = pin.buildModelFromUrdf(str(path), pin.JointModelFreeFlyer())
assert (model.nq, model.nv) == (13, 12)
assert list(model.names)[2:] == [f"z1_joint{i}" for i in range(1, 7)]
mass = sum(link["mass_kg"] for link in links)
assert abs(pin.computeTotalMass(model) - mass) < 1e-12
frame_id = model.getFrameId("z1_gripperStator", pin.FrameType.BODY)
assert frame_id < model.nframes
cases = []
configurations = {
    "zero": [0, 0, 0, 0, 0, 0],
    "folded": [0, 1.2, -1.8, 0.6, 0, 0],
    "asymmetric": [0.3, 1.0, -1.6, 0.4, 0.2, -0.3],
}
for name, arm in configurations.items():
    q = pin.neutral(model)
    q[7:] = arm
    assert (q[7:] >= model.lowerPositionLimit[7:]).all() and (
        q[7:] <= model.upperPositionLimit[7:]
    ).all()
    data = model.createData()
    pin.computeJointJacobians(model, data, q)
    pin.updateFramePlacements(model, data)
    position = data.oMf[frame_id].translation.copy()
    jacobian = pin.getFrameJacobian(
        model, data, frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
    ).copy()
    gravity = pin.computeGeneralizedGravity(model, data, q).copy()
    dynamics = pin.rnea(model, data, q, np.zeros(model.nv), np.zeros(model.nv)).copy()
    assert np.max(np.abs(gravity - dynamics)) < 1e-12
    matrix = pin.crba(model, data, q).copy()
    eig = np.linalg.eigvalsh(matrix)
    assert eig.min() > 0
    fd = np.empty_like(jacobian)
    grad = np.empty(model.nv)
    eps = 1e-6
    for col in range(model.nv):
        delta = np.zeros(model.nv)
        delta[col] = eps
        qp = pin.integrate(model, q, delta)
        qm = pin.integrate(model, q, -delta)
        dp = model.createData()
        dm = model.createData()
        pin.framesForwardKinematics(model, dp, qp)
        pin.framesForwardKinematics(model, dm, qm)
        plus = dp.oMf[frame_id]
        minus = dm.oMf[frame_id]
        fd[:3, col] = (plus.translation - minus.translation) / (2 * eps)
        fd[3:, col] = pin.log3(plus.rotation @ minus.rotation.T) / (2 * eps)
        grad[col] = (
            pin.computePotentialEnergy(model, dp, qp) - pin.computePotentialEnergy(model, dm, qm)
        ) / (2 * eps)
    je = float(np.max(np.abs(jacobian - fd)))
    ge = float(np.max(np.abs(gravity - grad)))
    assert je < 1e-7, (name, je)
    assert ge < 1e-6, (name, ge)
    cases.append(
        {
            "name": name,
            "arm_configuration_rad": arm,
            "gripper_mount_position_in_base_m": position.tolist(),
            "com_in_base_m": pin.centerOfMass(model, data, q).tolist(),
            "arm_gravity_torque_Nm": gravity[6:].tolist(),
            "max_gravity_torque_over_urdf_effort": float(
                np.max(np.abs(gravity[6:]) / model.effortLimit[6:])
            ),
            "min_mass_matrix_eigenvalue": float(eig.min()),
            "frame_jacobian_fd_max_abs_error": je,
            "gravity_potential_gradient_fd_max_abs_error": ge,
        }
    )
report = {
    "date": datetime.now(ZoneInfo("Asia/Shanghai")).date().isoformat(),
    "scope": (
        "Standalone floating-base Z1 with fixed gripper; "
        "no Go2 assembly or support/contact verification"
    ),
    "audit_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "python": sys.version.split()[0],
    "platform": platform.platform(),
    "numpy": np.__version__,
    "pinocchio": pin.__version__,
    "urdf_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    "nq": model.nq,
    "nv": model.nv,
    "actuated_joints": 6,
    "locked_gripper_angle_rad": 0.0,
    "links": links,
    "joints": joints,
    "mass_kg": mass,
    "arm_without_gripper_mass_kg": sum(
        x["mass_kg"] for x in links if "gripper" not in x["name"].lower()
    ),
    "gripper_mass_kg": sum(x["mass_kg"] for x in links if "gripper" in x["name"].lower()),
    "frame_checked": "z1_gripperStator",
    "finite_difference_step": 1e-6,
    "jacobian_threshold": 1e-7,
    "gravity_gradient_threshold": 1e-6,
    "cases": cases,
    "passed": True,
    "not_validated": [
        "Go2 mount transform and bracket inertia",
        "hardware AIR/PRO identity, continuous torque and payload rating",
        "mesh collision and visual loading",
        "self collision or environment collision",
        "combined Go2 dynamics, fixed-support feasibility, WBC and PhysX",
    ],
}
(directory / "ASSET_AUDIT.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
print(
    json.dumps(
        {
            k: report[k]
            for k in [
                "mass_kg",
                "arm_without_gripper_mass_kg",
                "gripper_mass_kg",
                "cases",
                "passed",
            ]
        },
        indent=2,
    )
)
