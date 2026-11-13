"""Independent XML kinematics/mass oracle and public Go2+Z1 CPU contracts."""

import copy
import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

from crocoddyl_batched_mpc.models.go2_arm import GO2_ARM_JOINT_NAMES, load_go2_arm

ASSETS = Path(__file__).resolve().parents[1] / "src/crocoddyl_batched_mpc/assets"


@pytest.fixture
def robot():
    pytest.importorskip("pinocchio")
    return load_go2_arm()


def _rotation(axis, angle):
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    x, y, z = axis
    skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + np.sin(angle) * skew + (1 - np.cos(angle)) * (skew @ skew)


def _transform(xyz, rpy):
    roll, pitch, yaw = rpy
    result = np.eye(4)
    result[:3, :3] = (
        _rotation([0, 0, 1], yaw) @ _rotation([0, 1, 0], pitch) @ _rotation([1, 0, 0], roll)
    )
    result[:3, 3] = xyz
    return result


def _origin(node):
    origin = node.find("origin")
    attributes = {} if origin is None else origin.attrib
    return _transform(
        np.fromstring(attributes.get("xyz", "0 0 0"), sep=" "),
        np.fromstring(attributes.get("rpy", "0 0 0"), sep=" "),
    )


def _base(q):
    x, y, z, w = q[3:7]
    result = np.eye(4)
    result[:3, :3] = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
    result[:3, 3] = q[:3]
    return result


def _tree(path, root_name, root_pose, angles):
    """Use only separate source XML, not generated combined XML or model methods."""
    tree = ET.parse(path).getroot()
    poses = {root_name: root_pose}
    pending = tree.findall("joint")
    while pending:
        ready = [joint for joint in pending if joint.find("parent").attrib["link"] in poses]
        assert ready, "source XML is disconnected"
        for joint in ready:
            transform = _origin(joint)
            if joint.attrib["type"] == "revolute":
                rotation = np.eye(4)
                rotation[:3, :3] = _rotation(
                    np.fromstring(joint.find("axis").attrib["xyz"], sep=" "),
                    angles[joint.attrib["name"]],
                )
                transform = transform @ rotation
            else:
                assert joint.attrib["type"] == "fixed"
            poses[joint.find("child").attrib["link"]] = (
                poses[joint.find("parent").attrib["link"]] @ transform
            )
            pending.remove(joint)
    mass, first_moment = 0.0, np.zeros(3)
    for link in tree.findall("link"):
        inertial = link.find("inertial")
        if inertial is not None:
            weight = float(inertial.find("mass").attrib["value"])
            center = (poses[link.attrib["name"]] @ _origin(inertial))[:3, 3]
            mass += weight
            first_moment += weight * center
    return poses, mass, first_moment


def _reference(q, config):
    angles = dict(zip(GO2_ARM_JOINT_NAMES, q[7:]))
    base = _base(q)
    go2, m_go2, h_go2 = _tree(ASSETS / "go2/go2.urdf", "base", base, angles)
    mount = config["mount"]
    arm, m_arm, h_arm = _tree(
        ASSETS / "go2_arm/z1_fixed_gripper.urdf",
        "z1_link00",
        base @ _transform(mount["xyz_m"], mount["rpy_rad"]),
        angles,
    )
    bracket = config["bracket"]
    center = base @ np.r_[bracket["com_xyz_in_base_m"], 1]
    mass = m_go2 + m_arm + bracket["mass_kg"]
    com = (h_go2 + h_arm + bracket["mass_kg"] * center[:3]) / mass
    tcp = arm["z1_gripperStator"] @ _transform(config["tcp"]["xyz_m"], config["tcp"]["rpy_rad"])
    feet = np.stack([go2[f"{leg}_foot"][:3, 3] for leg in ("FL", "FR", "RL", "RR")])
    potential = -mass * np.dot(config["gravity_world_m_s2"], com)
    return tcp, feet, mass, com, potential


def _case(robot, variant):
    if variant == "rotated_mount":
        config = copy.deepcopy(robot.config)
        config["mount"].update(xyz_m=[-0.02, 0.03, 0.12], rpy_rad=[0.15, -0.2, 0.35])
        config["tcp"].update(xyz_m=[0.09, 0.015, -0.01], rpy_rad=[0.1, 0.2, -0.3])
        config["bracket"]["mass_kg"] = 0.47
        config["gravity_world_m_s2"] = [0.2, -0.1, -9.7]
        robot = load_go2_arm(config)
    q = robot.standing_configuration()
    if variant != "nominal":
        q[:3] += [0.2, -0.15, 0.1]
        q[3:7] = np.array([0.12, -0.08, 0.17, 1]) / np.linalg.norm([0.12, -0.08, 0.17, 1])
        q[7:19] += np.linspace(-0.03, 0.03, 12)
        q[19:] = [0.3, 1.0, -1.6, 0.4, 0.2, -0.3]
    return robot, q


@pytest.mark.parametrize("variant", ["nominal", "tilted", "rotated_mount"])
def test_fk_com_and_mass_match_separate_xml_oracle(robot, variant):
    robot, q = _case(robot, variant)
    actual = robot.evaluate(q)
    tcp, feet, mass, com, potential = _reference(q, robot.config)
    np.testing.assert_allclose(actual.tcp_pose, tcp, atol=2e-12, rtol=0)
    np.testing.assert_allclose(actual.foot_positions, feet, atol=2e-12, rtol=0)
    np.testing.assert_allclose(actual.com, com, atol=2e-12, rtol=0)
    assert abs(robot.total_mass - mass) < 1e-12
    assert abs(robot.potential_energy(q) - potential) < 1e-10
    assert np.linalg.eigvalsh(actual.mass_matrix).min() > 0
    np.testing.assert_allclose(actual.mass_matrix, actual.mass_matrix.T, atol=1e-13, rtol=0)
    if variant == "nominal":
        assert abs(robot.total_mass - 21.60796983) < 1e-12
        assert np.abs(feet[:, 2]).max() < 1e-12
        np.testing.assert_allclose(
            robot.frame_pose(q, "z1_link00"),
            _base(q) @ _transform([0.05, 0, 0.10], [0, 0, 0]),
            atol=1e-12,
        )
    print(
        json.dumps(
            {
                "case": variant,
                "fk_max_abs_error": float(np.abs(actual.tcp_pose - tcp).max()),
                "com_max_abs_error": float(np.abs(actual.com - com).max()),
                "mass_kg": mass,
            }
        )
    )


@pytest.mark.parametrize("variant", ["nominal", "tilted", "rotated_mount"])
def test_full_tangent_jacobians_and_gravity_against_independent_differences(robot, variant):
    robot, q = _case(robot, variant)
    actual = robot.evaluate(q)
    tcp_fd = np.empty((6, 24))
    foot_fd = np.empty((4, 3, 24))
    com_fd = np.empty((3, 24))
    gravity_fd = np.empty(24)
    epsilon = 1e-6
    for column in range(24):
        delta = np.zeros(24)
        delta[column] = epsilon
        tp, fp, _, cp, up = _reference(robot.integrate(q, delta), robot.config)
        tm, fm, _, cm, um = _reference(robot.integrate(q, -delta), robot.config)
        tcp_fd[:3, column] = (tp[:3, 3] - tm[:3, 3]) / (2 * epsilon)
        rotation = tp[:3, :3] @ tm[:3, :3].T
        tcp_fd[3:, column] = np.array(
            [
                rotation[2, 1] - rotation[1, 2],
                rotation[0, 2] - rotation[2, 0],
                rotation[1, 0] - rotation[0, 1],
            ]
        ) / (4 * epsilon)
        foot_fd[:, :, column] = (fp - fm) / (2 * epsilon)
        com_fd[:, column] = (cp - cm) / (2 * epsilon)
        gravity_fd[column] = (up - um) / (2 * epsilon)
    np.testing.assert_allclose(actual.tcp_jacobian, tcp_fd, atol=2e-8, rtol=0)
    np.testing.assert_allclose(actual.foot_jacobians, foot_fd, atol=2e-8, rtol=0)
    np.testing.assert_allclose(actual.com_jacobian, com_fd, atol=2e-8, rtol=0)
    np.testing.assert_allclose(actual.gravity, gravity_fd, atol=2e-6, rtol=0)
    np.testing.assert_allclose(actual.nonlinear_effects, actual.gravity, atol=1e-12, rtol=0)
    print(
        json.dumps(
            {
                "case": variant,
                "tcp_jacobian_max_abs_error": float(np.abs(actual.tcp_jacobian - tcp_fd).max()),
                "foot_jacobian_max_abs_error": float(np.abs(actual.foot_jacobians - foot_fd).max()),
                "com_jacobian_max_abs_error": float(np.abs(actual.com_jacobian - com_fd).max()),
                "gravity_max_abs_error": float(np.abs(actual.gravity - gravity_fd).max()),
            }
        )
    )


def test_all_eighteen_joint_state_torque_mapping_and_virtual_work(robot):
    import pinocchio as pin

    assert (robot.nq, robot.nv, robot.nx, robot.ndx, robot.nu) == (25, 24, 49, 48, 18)
    _, q = _case(robot, "tilted")
    v = np.linspace(-0.3, 0.5, 24)
    tau = np.linspace(-2.0, 3.0, 18)
    qn, vn = robot.to_native_q(q), robot.to_native_v(v)
    for index, name in enumerate(GO2_ARM_JOINT_NAMES):
        joint = robot.model.joints[robot.model.getJointId(name)]
        assert qn[joint.idx_q] == q[7 + index]
        assert vn[joint.idx_v] == v[6 + index]
    np.testing.assert_array_equal(robot.from_native_q(qn), q)
    np.testing.assert_array_equal(robot.from_native_v(vn), v)
    names = ("camera_joint", *reversed(GO2_ARM_JOINT_NAMES), "spare_joint")
    positions, velocities = np.ones(20) * 99, np.ones(20) * 88
    for index, name in enumerate(GO2_ARM_JOINT_NAMES):
        positions[names.index(name)] = q[7 + index]
        velocities[names.index(name)] = v[6 + index]
    np.testing.assert_array_equal(robot.pack_configuration(q[:7], positions, names), q)
    np.testing.assert_array_equal(robot.pack_velocity(v[:6], velocities, names), v)
    effort = robot.scatter_torques(tau, names)
    assert effort[0] == effort[-1] == 0
    for index, name in enumerate(GO2_ARM_JOINT_NAMES):
        assert effort[names.index(name)] == tau[index]
    matrix = robot.actuation_matrix(native=True)
    np.testing.assert_array_equal(matrix[:6], np.zeros((6, 18)))
    np.testing.assert_array_equal(matrix.T @ matrix, np.eye(18))
    for index, joint_id in enumerate(robot.joint_ids):
        expected = np.zeros(24)
        expected[robot.model.joints[joint_id].idx_v] = 1
        np.testing.assert_array_equal(matrix[:, index], expected)
    assert abs(vn @ (matrix @ tau) - v[6:] @ tau) < 1e-12
    np.testing.assert_array_equal(robot.actuation_matrix() @ tau, np.r_[np.zeros(6), tau])
    native = pin.nonLinearEffects(robot.model, robot.model.createData(), qn, vn)
    np.testing.assert_allclose(
        robot.evaluate(q, v).nonlinear_effects, native[robot.v_indices], atol=1e-12
    )
    # Explicitly validate URDF arm effort/velocity limits in canonical order.
    np.testing.assert_array_equal(robot.torque_limits[12:], [30, 60, 30, 30, 30, 30])
    np.testing.assert_array_equal(robot.velocity_limits[12:], [3.1415] * 6)


def test_invalid_state_names_and_configuration_rejected(robot):
    q = robot.standing_configuration()
    q[3:7] = 0
    with pytest.raises(ValueError, match="quaternion"):
        robot.evaluate(q)
    with pytest.raises(ValueError, match="finite"):
        robot.to_native_v(np.full(24, np.nan))
    with pytest.raises(ValueError, match="unique"):
        robot.scatter_torques(np.zeros(18), GO2_ARM_JOINT_NAMES + (GO2_ARM_JOINT_NAMES[0],))
    with pytest.raises(ValueError, match="all 18"):
        robot.scatter_torques(np.zeros(18), GO2_ARM_JOINT_NAMES[:-1])
    with pytest.raises(ValueError, match="finite"):
        robot.scatter_torques(np.zeros(12), GO2_ARM_JOINT_NAMES)
    with pytest.raises(ValueError, match="unknown"):
        robot.frame_pose(robot.standing_configuration(), "missing_tcp")
    for section, key, value in [
        ("bracket", "mass_kg", -0.3),
        ("bracket", "box_size_m", [0.1, 0, 0.1]),
        ("mount", "xyz_m", None),
        ("tcp", "rpy_rad", [0, float("nan"), 0]),
        ("passive_joint_dynamics", "damping", 1.0),
    ]:
        config = copy.deepcopy(robot.config)
        config[section][key] = value
        with pytest.raises(ValueError):
            load_go2_arm(config)
    config = copy.deepcopy(robot.config)
    config["payload_mass_kg"] = 1.0
    with pytest.raises(ValueError, match="payload"):
        load_go2_arm(config)


def test_import_does_not_require_native_dependencies_or_change_legacy_api():
    code = """
import builtins
import sys
import torch
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in {'numpy', 'pinocchio', 'crocoddyl'}:
        raise AssertionError('unexpected native import: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
from crocoddyl_batched_mpc.models.go2_arm import GO2_ARM_JOINT_NAMES
from crocoddyl_batched_mpc.models import Go2TensorAdapter, Go2TrackingMPC
assert len(GO2_ARM_JOINT_NAMES) == 18
assert 'pinocchio' not in sys.modules
"""
    result = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_zero_passive_dynamics_and_bracket_inertia_in_combined_asset(robot):
    xml = ET.fromstring(robot.urdf_xml)
    moving = [j for j in xml.findall("joint") if j.attrib["type"] != "fixed"]
    assert len(moving) == 18
    assert {j.attrib["name"] for j in moving} == set(GO2_ARM_JOINT_NAMES)
    for joint in moving:
        assert joint.find("dynamics").attrib == {"damping": "0", "friction": "0"}
    bracket = next(link for link in xml.findall("link") if link.attrib["name"] == "go2_z1_bracket")
    assert float(bracket.find("inertial/mass").attrib["value"]) == 0.3
    inertia = bracket.find("inertial/inertia").attrib
    np.testing.assert_allclose(
        [float(inertia[key]) for key in ["ixx", "iyy", "izz"]],
        [0.0002425, 0.00068, 0.0008425],
        atol=1e-16,
    )
    assert all(link.attrib["name"] != "world" for link in xml.findall("link"))
    # Source arm retains its original passive dynamics; only the derived model changes.
    original = ET.parse(ASSETS / "go2_arm/z1_fixed_gripper.urdf").getroot()
    assert original.find("joint/dynamics").attrib["damping"] == "1.0"
