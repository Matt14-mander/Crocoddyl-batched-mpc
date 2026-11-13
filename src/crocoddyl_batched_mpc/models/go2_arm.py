"""Isolated CPU Go2+Z1 rigid-body model with a fixed gripper and explicit mounting.

Import this module directly; the legacy models/__init__.py remains unchanged.
Public q/v/tau use legs FL/FR/RL/RR followed by Z1 J1..J6. Pinocchio's internal
ordering is mapped by name. Base quaternion is xyzw; base velocities are local
linear/angular twists. No controller, contact solve, PD or passive torque is added.
"""

import copy
import hashlib
import json
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any

from .go2 import GO2_JOINT_NAMES, GO2_LEGS

GO2_ARM_JOINT_NAMES = GO2_JOINT_NAMES + tuple(f"z1_joint{i}" for i in range(1, 7))


def _dependencies():
    try:
        import numpy as np
        import pinocchio as pin
    except ImportError as exc:
        raise ImportError("Go2+Z1 CPU model requires the optional NumPy/Pinocchio stack") from exc
    return np, pin


def _asset_directory():
    return Path(str(files("crocoddyl_batched_mpc").joinpath("assets/go2_arm")))


def _vector(value, length, name):
    np, _ = _dependencies()
    array = np.asarray(value, dtype=float)
    if array.shape != (length,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite vector of length {length}")
    return array.copy()


def _config(config):
    np, _ = _dependencies()
    if config is None:
        config = _asset_directory() / "SIMULATION_CONFIG.json"
    result = (
        json.loads(Path(config).read_text())
        if isinstance(config, (str, Path))
        else copy.deepcopy(config)
    )
    if result["schema_version"] != 1:
        raise ValueError("unsupported Go2+Z1 simulation schema")
    expected = (
        (result["mount"]["parent_frame"], "base"),
        (result["mount"]["child_frame"], "z1_link00"),
        (result["bracket"]["frame"], "go2_z1_bracket"),
        (result["tcp"]["parent_frame"], "z1_gripperStator"),
        (result["tcp"]["frame"], "z1_tcp"),
        (result["bracket"]["inertia_model"], "uniform_box_at_com_aligned_with_base"),
    )
    if any(actual != required for actual, required in expected):
        raise ValueError("unsupported mount, bracket or TCP frame convention")
    for section in ("mount", "tcp"):
        for key in ("xyz_m", "rpy_rad"):
            result[section][key] = _vector(result[section][key], 3, section + "." + key).tolist()
    bracket = result["bracket"]
    size = _vector(bracket["box_size_m"], 3, "bracket.box_size_m")
    center = _vector(bracket["com_xyz_in_base_m"], 3, "bracket.com_xyz_in_base_m")
    mass = float(bracket["mass_kg"])
    if not math.isfinite(mass) or mass <= 0 or (size <= 0).any():
        raise ValueError("bracket mass and box dimensions must be positive and finite")
    bracket.update(mass_kg=mass, box_size_m=size.tolist(), com_xyz_in_base_m=center.tolist())
    result["nominal_arm_configuration_rad"] = _vector(
        result["nominal_arm_configuration_rad"], 6, "nominal arm configuration"
    ).tolist()
    result["gravity_world_m_s2"] = _vector(result["gravity_world_m_s2"], 3, "gravity").tolist()
    if result["payload_mass_kg"] != 0:
        raise ValueError("this initial model supports zero additional payload only")
    passive = result["passive_joint_dynamics"]
    if not np.array_equal([passive["damping"], passive["friction"]], [0.0, 0.0]):
        raise ValueError("this rigid-body baseline requires zero passive damping/friction")
    return result


def bracket_inertia(config):
    """Diagonal inertia at the uniform box COM, in base-aligned axes (kg m^2)."""
    np, _ = _dependencies()
    box = config["bracket"]
    x, y, z = box["box_size_m"]
    return box["mass_kg"] / 12 * np.array([y * y + z * z, x * x + z * z, x * x + y * y])


def _numbers(values):
    return " ".join(format(float(value), ".17g") for value in values)


def build_go2_arm_urdf(config=None):
    """Return deterministic combined XML; never modify either source URDF.

    Visual/mesh references are retained for future import, but meshes are not
    loaded by the CPU model. A single free-flyer is added by the native loader.
    """
    np, _ = _dependencies()
    config = _config(config)
    directory = _asset_directory()
    spec = json.loads((directory / "MODEL_SPEC.json").read_text())
    trees = []
    for key in ("go2", "z1"):
        content = (directory / spec[key]["path"]).read_bytes()
        if hashlib.sha256(content).hexdigest() != spec[key]["sha256"]:
            raise ValueError(f"{key} source asset checksum mismatch")
        trees.append(ET.fromstring(content))
    output = ET.Element("robot", {"name": "go2_z1_fixed_gripper"})
    for tree in trees:
        for node in tree:
            output.append(copy.deepcopy(node))

    bracket = config["bracket"]
    link = ET.SubElement(output, "link", {"name": bracket["frame"]})
    inertial = ET.SubElement(link, "inertial")
    ET.SubElement(inertial, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
    ET.SubElement(inertial, "mass", {"value": format(bracket["mass_kg"], ".17g")})
    diagonal = bracket_inertia(config)
    ET.SubElement(
        inertial,
        "inertia",
        {
            "ixx": str(diagonal[0]),
            "iyy": str(diagonal[1]),
            "izz": str(diagonal[2]),
            "ixy": "0",
            "ixz": "0",
            "iyz": "0",
        },
    )
    for tag in ("visual", "collision"):
        geometry = ET.SubElement(ET.SubElement(link, tag), "geometry")
        ET.SubElement(geometry, "box", {"size": _numbers(bracket["box_size_m"])})

    def fixed_joint(name, parent, child, xyz, rpy):
        joint = ET.SubElement(output, "joint", {"name": name, "type": "fixed"})
        ET.SubElement(joint, "parent", {"link": parent})
        ET.SubElement(joint, "child", {"link": child})
        ET.SubElement(joint, "origin", {"xyz": _numbers(xyz), "rpy": _numbers(rpy)})

    fixed_joint(
        "go2_z1_bracket_joint", "base", bracket["frame"], bracket["com_xyz_in_base_m"], [0, 0, 0]
    )
    fixed_joint(
        "go2_z1_mount_joint",
        bracket["frame"],
        "z1_link00",
        np.array(config["mount"]["xyz_m"]) - bracket["com_xyz_in_base_m"],
        config["mount"]["rpy_rad"],
    )
    ET.SubElement(output, "link", {"name": "z1_tcp"})
    fixed_joint(
        "z1_tcp_joint",
        "z1_gripperStator",
        "z1_tcp",
        config["tcp"]["xyz_m"],
        config["tcp"]["rpy_rad"],
    )
    for joint in output.findall("joint"):
        if joint.attrib["type"] != "fixed":
            dynamics = joint.find("dynamics")
            if dynamics is None:
                dynamics = ET.SubElement(joint, "dynamics")
            dynamics.set("damping", "0")
            dynamics.set("friction", "0")
    ET.indent(output, space="  ")
    return ET.tostring(output, encoding="unicode", xml_declaration=True) + "\n"


@dataclass
class Go2ArmQuantities:
    """World FK/Jacobians and rigid-body quantities in the public tangent order."""

    tcp_pose: Any  # 4x4 world transform
    foot_positions: Any  # [4,3], FL/FR/RL/RR
    com: Any  # [3], world
    tcp_jacobian: Any  # [6,24], world-aligned linear/angular at TCP
    foot_jacobians: Any  # [4,3,24], world
    com_jacobian: Any  # [3,24], world
    mass_matrix: Any  # [24,24]
    gravity: Any  # [24], includes unactuated floating-base wrench
    nonlinear_effects: Any  # [24], Coriolis/centrifugal + gravity; no passive terms


@dataclass
class Go2ArmRobot:
    model: Any
    config: dict
    urdf_xml: str
    joint_ids: tuple[int, ...]
    q_indices: Any
    v_indices: Any
    foot_ids: tuple[int, ...]
    tcp_id: int

    nq, nv, nx, ndx, nu = 25, 24, 49, 48, 18
    joint_names = GO2_ARM_JOINT_NAMES

    def _q(self, q):
        np, _ = _dependencies()
        q = _vector(q, self.nq, "q")
        if abs(np.dot(q[3:7], q[3:7]) - 1) > 1e-8:
            raise ValueError("q requires a unit xyzw base quaternion")
        return q

    def to_native_q(self, q):
        np, _ = _dependencies()
        result = np.empty(self.nq)
        result[self.q_indices] = self._q(q)
        return result

    def from_native_q(self, q):
        return self._q(_vector(q, self.nq, "native q")[self.q_indices])

    def to_native_v(self, v):
        np, _ = _dependencies()
        result = np.empty(self.nv)
        result[self.v_indices] = _vector(v, self.nv, "v")
        return result

    def from_native_v(self, v):
        return _vector(v, self.nv, "native v")[self.v_indices]

    @property
    def total_mass(self):
        _, pin = _dependencies()
        return float(pin.computeTotalMass(self.model))

    @property
    def torque_limits(self):
        return self.model.effortLimit[self.v_indices[6:]].copy()

    @property
    def velocity_limits(self):
        return self.model.velocityLimit[self.v_indices[6:]].copy()

    @property
    def position_limits(self):
        indices = self.q_indices[7:]
        return self.model.lowerPositionLimit[indices].copy(), self.model.upperPositionLimit[
            indices
        ].copy()

    def actuation_matrix(self, *, native=False):
        """Map 18 torques to public generalized force, or native rows on request.

        The default matches evaluate().mass_matrix/gravity. Use native=True only
        alongside direct Pinocchio calls.
        """
        np, _ = _dependencies()
        matrix = np.zeros((self.nv, self.nu))
        rows = self.v_indices[6:] if native else np.arange(6, self.nv)
        matrix[rows, np.arange(self.nu)] = 1
        return matrix

    def _joint_map(self, names):
        names = tuple(names)
        if len(names) != len(set(names)):
            raise ValueError("joint names must be unique")
        if not set(self.joint_names).issubset(names):
            raise ValueError("joint names must include all 18 Go2+Z1 joints")
        return [names.index(name) for name in self.joint_names]

    def pack_configuration(self, base_pose, joint_positions, joint_names):
        """Pack arbitrary named joint order; base_pose is world xyz + xyzw."""
        np, _ = _dependencies()
        names = tuple(joint_names)
        joint_positions = _vector(joint_positions, len(names), "joint positions")
        return self._q(
            np.r_[_vector(base_pose, 7, "base pose"), joint_positions[self._joint_map(names)]]
        )

    def pack_velocity(self, base_velocity, joint_velocities, joint_names):
        """Base linear/angular velocity is in body axes, not world axes."""
        np, _ = _dependencies()
        names = tuple(joint_names)
        joint_velocities = _vector(joint_velocities, len(names), "joint velocities")
        return np.r_[
            _vector(base_velocity, 6, "base velocity"), joint_velocities[self._joint_map(names)]
        ]

    def scatter_torques(self, torques, joint_names):
        np, _ = _dependencies()
        names = tuple(joint_names)
        result = np.zeros(len(names))
        result[self._joint_map(names)] = _vector(torques, self.nu, "torques")
        return result

    def integrate(self, q, tangent):
        _, pin = _dependencies()
        return self.from_native_q(
            pin.integrate(self.model, self.to_native_q(q), self.to_native_v(tangent))
        )

    def standing_configuration(self):
        """Geometric flat-ground seed only; no support-force equilibrium solve."""
        np, pin = _dependencies()
        q = self.from_native_q(pin.neutral(self.model))
        q[7:19] = [0.0, 0.8, -1.6] * 4
        q[19:] = self.config["nominal_arm_configuration_rad"]
        q[2] -= self.evaluate(q).foot_positions[:, 2].mean()
        lower, upper = self.position_limits
        if np.any(q[7:] < lower) or np.any(q[7:] > upper):
            raise ValueError("nominal standing configuration exceeds joint bounds")
        return q

    def potential_energy(self, q):
        _, pin = _dependencies()
        return float(
            pin.computePotentialEnergy(self.model, self.model.createData(), self.to_native_q(q))
        )

    def frame_pose(self, q, frame):
        _, pin = _dependencies()
        if not self.model.existFrame(frame, pin.FrameType.BODY):
            raise ValueError(f"unknown body frame: {frame}")
        data = self.model.createData()
        pin.framesForwardKinematics(self.model, data, self.to_native_q(q))
        return data.oMf[self.model.getFrameId(frame, pin.FrameType.BODY)].homogeneous.copy()

    def evaluate(self, q, v=None):
        np, pin = _dependencies()
        qn = self.to_native_q(q)
        vn = self.to_native_v(np.zeros(self.nv) if v is None else v)
        data = self.model.createData()
        mass = pin.crba(self.model, data, qn).copy()
        mass = np.triu(mass) + np.triu(mass, 1).T
        gravity = pin.computeGeneralizedGravity(self.model, data, qn).copy()
        nonlinear = pin.nonLinearEffects(self.model, data, qn, vn).copy()
        com = pin.centerOfMass(self.model, data, qn).copy()
        com_jacobian = pin.jacobianCenterOfMass(self.model, data, qn).copy()
        pin.computeJointJacobians(self.model, data, qn)
        pin.updateFramePlacements(self.model, data)

        def jacobian(frame):
            return pin.getFrameJacobian(self.model, data, frame, pin.LOCAL_WORLD_ALIGNED)[
                :, self.v_indices
            ].copy()

        return Go2ArmQuantities(
            tcp_pose=data.oMf[self.tcp_id].homogeneous.copy(),
            foot_positions=np.stack([data.oMf[i].translation.copy() for i in self.foot_ids]),
            com=com,
            tcp_jacobian=jacobian(self.tcp_id),
            foot_jacobians=np.stack([jacobian(i)[:3] for i in self.foot_ids]),
            com_jacobian=com_jacobian[:, self.v_indices].copy(),
            mass_matrix=mass[np.ix_(self.v_indices, self.v_indices)].copy(),
            gravity=gravity[self.v_indices].copy(),
            nonlinear_effects=nonlinear[self.v_indices].copy(),
        )


def load_go2_arm(config=None):
    """Load only this independent CPU model, using explicit simulation assumptions."""
    np, pin = _dependencies()
    config = _config(config)
    xml = build_go2_arm_urdf(config)
    model = pin.buildModelFromXML(xml, pin.JointModelFreeFlyer())
    if (model.nq, model.nv) != (25, 24):
        raise ValueError("Go2+Z1 requires one free flyer and 18 scalar joints")
    model.gravity.linear = np.array(config["gravity_world_m_s2"])
    if not all(model.existJointName(name) for name in GO2_ARM_JOINT_NAMES):
        raise ValueError("combined model is missing an actuated joint")
    joints = tuple(model.getJointId(name) for name in GO2_ARM_JOINT_NAMES)
    if any(model.joints[j].nq != 1 or model.joints[j].nv != 1 for j in joints):
        raise ValueError("Go2+Z1 requires scalar actuated joints")
    qi = np.array([*range(7), *(model.joints[j].idx_q for j in joints)], dtype=int)
    vi = np.array([*range(6), *(model.joints[j].idx_v for j in joints)], dtype=int)
    if sorted(qi) != list(range(25)) or sorted(vi) != list(range(24)):
        raise ValueError("native state mapping is not a bijection")
    qi.setflags(write=False)
    vi.setflags(write=False)
    frames = [f"{leg}_foot" for leg in GO2_LEGS] + ["z1_tcp"]
    if not all(model.existFrame(name, pin.FrameType.BODY) for name in frames):
        raise ValueError("combined model is missing a foot/TCP frame")
    ids = [model.getFrameId(name, pin.FrameType.BODY) for name in frames]
    return Go2ArmRobot(model, config, xml, joints, qi, vi, tuple(ids[:4]), ids[4])
