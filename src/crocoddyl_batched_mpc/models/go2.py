"""Unitree Go2 description and an optional CPU fixed-contact correctness reference.

Pinocchio/NumPy are imported only when the reference is used. This module is not
an asynchronous Torch dynamics model and must not be used as a CUDA hot path.
"""

from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any

GO2_LEGS = ("FL", "FR", "RL", "RR")
GO2_JOINT_NAMES = tuple(
    f"{leg}_{joint}_joint" for leg in GO2_LEGS for joint in ("hip", "thigh", "calf")
)


def _dependencies():
    try:
        import numpy as np
        import pinocchio as pin
    except ImportError as exc:
        raise ImportError(
            "Go2 CPU reference requires Pinocchio and NumPy; use the robotics extra"
        ) from exc
    return np, pin


@dataclass
class Go2Robot:
    """Pinocchio model with explicit joint/foot order and URDF-derived limits."""

    model: Any
    urdf_path: Path
    joint_ids: tuple[int, ...]
    foot_ids: tuple[int, ...]

    @property
    def nq(self):
        return self.model.nq

    @property
    def nv(self):
        return self.model.nv

    @property
    def nx(self):
        return self.nq + self.nv

    @property
    def ndx(self):
        return 2 * self.nv

    @property
    def torque_limits(self):
        np, _ = _dependencies()
        return np.array(
            [self.model.effortLimit[self.model.joints[j].idx_v] for j in self.joint_ids]
        )

    def actuation_matrix(self):
        np, _ = _dependencies()
        matrix = np.zeros((self.nv, len(self.joint_ids)))
        for column, joint_id in enumerate(self.joint_ids):
            matrix[self.model.joints[joint_id].idx_v, column] = 1
        return matrix

    def standing_configuration(self):
        """Crouched configuration, ideal foot-frame point contacts on z=0.

        This geometric seed is not a hardware standing controller. Meshes and
        collision sphere radii are not included in this point-contact reference.
        """
        np, pin = _dependencies()
        q = pin.neutral(self.model)
        for name, joint_id in zip(GO2_JOINT_NAMES, self.joint_ids):
            value = 0.0 if "hip" in name else (0.8 if "thigh" in name else -1.6)
            q[self.model.joints[joint_id].idx_q] = value
        data = self.model.createData()
        pin.framesForwardKinematics(self.model, data, q)
        q[2] = -np.mean([data.oMf[i].translation[2] for i in self.foot_ids])
        return q


def load_go2(urdf_path: str | Path | None = None) -> Go2Robot:
    """Load the pinned official URDF without fetching meshes or changing ROS paths."""
    _, pin = _dependencies()
    path = (
        Path(str(files("crocoddyl_batched_mpc").joinpath("assets/go2/go2.urdf")))
        if urdf_path is None
        else Path(urdf_path)
    )
    model = pin.buildModelFromUrdf(str(path), pin.JointModelFreeFlyer())
    if model.nq != 19 or model.nv != 18:
        raise ValueError("Go2 reference requires a free flyer and 12 scalar joints")
    if not all(model.existJointName(name) for name in GO2_JOINT_NAMES):
        raise ValueError("Go2 URDF is missing expected joint names")
    foot_names = tuple(f"{leg}_foot" for leg in GO2_LEGS)
    if not all(model.existFrame(name) for name in foot_names):
        raise ValueError("Go2 URDF is missing expected foot frames")
    joints = tuple(model.getJointId(name) for name in GO2_JOINT_NAMES)
    if any(model.joints[j].nq != 1 or model.joints[j].nv != 1 for j in joints):
        raise ValueError("Go2 leg joints must use scalar coordinates")
    if tuple(model.joints[j].idx_v for j in joints) != tuple(range(6, 18)):
        raise ValueError("Go2 joint order must be FL/FR/RL/RR, each hip/thigh/calf")
    return Go2Robot(model, path, joints, tuple(model.getFrameId(n) for n in foot_names))


@dataclass
class ContactDynamicsResult:
    acceleration: Any
    forces_world: Any
    mass_matrix: Any
    nonlinear_effects: Any
    contact_jacobian: Any
    contact_drift: Any
    dynamics_residual: Any
    contact_residual: Any


class Go2FixedContactReference:
    """Independent dense KKT reference using Pinocchio rigid-body quantities.

    All selected foot frames have bilateral 3D point constraints, with world
    force axes. No friction cone, unilateral test, torque clipping, contact
    switching, or integration is implicit. Mutable Pinocchio data is local to
    each calc, making stored results independent of later calls.
    """

    def __init__(
        self,
        robot: Go2Robot | None = None,
        *,
        feet=GO2_LEGS,
        reference_configuration=None,
        stabilization=(0.0, 0.0),
    ):
        np, pin = _dependencies()
        self.robot = load_go2() if robot is None else robot
        self.feet = tuple(feet)
        if not self.feet or len(set(self.feet)) != len(self.feet):
            raise ValueError("feet must be a nonempty sequence without duplicates")
        if any(name not in GO2_LEGS for name in self.feet):
            raise ValueError("feet must use FL/FR/RL/RR")
        gains = np.asarray(stabilization, dtype=float)
        if gains.shape != (2,) or not np.isfinite(gains).all() or (gains < 0).any():
            raise ValueError(
                "stabilization must contain finite nonnegative position/velocity gains"
            )
        self.gains = gains.copy()
        q = (
            self.robot.standing_configuration()
            if reference_configuration is None
            else np.asarray(reference_configuration, dtype=float).copy()
        )
        self._validate(q, np.zeros(self.robot.nv), np.zeros(12))
        self.reference_configuration = q
        self.frame_ids = tuple(self.robot.foot_ids[GO2_LEGS.index(name)] for name in self.feet)
        data = self.robot.model.createData()
        pin.framesForwardKinematics(self.robot.model, data, q)
        self.positions = tuple(data.oMf[i].translation.copy() for i in self.frame_ids)
        self.actuation = self.robot.actuation_matrix()

    def _validate(self, q, v, u):
        np, _ = _dependencies()
        for name, value, size in (("q", q, self.robot.nq), ("v", v, self.robot.nv), ("u", u, 12)):
            if np.shape(value) != (size,) or not np.isfinite(value).all():
                raise ValueError(f"{name} must be a finite vector of length {size}")
        if abs(np.dot(q[3:7], q[3:7]) - 1) >= 1e-5:
            raise ValueError("q requires a unit xyzw base quaternion")

    def _quantities(self, q, v):
        np, pin = _dependencies()
        model = self.robot.model
        data = model.createData()
        pin.computeAllTerms(model, data, q, v)
        mass = np.triu(data.M) + np.triu(data.M, 1).T
        nonlinear = data.nle.copy()
        pin.forwardKinematics(model, data, q, v, np.zeros(model.nv))
        pin.updateFramePlacements(model, data)
        jacobians, drifts = [], []
        for frame_id, reference in zip(self.frame_ids, self.positions):
            jacobian = pin.getFrameJacobian(model, data, frame_id, pin.LOCAL_WORLD_ALIGNED)[:3]
            velocity = pin.getFrameVelocity(model, data, frame_id, pin.LOCAL_WORLD_ALIGNED).linear
            acceleration = pin.getFrameClassicalAcceleration(
                model, data, frame_id, pin.LOCAL_WORLD_ALIGNED
            ).linear
            drift = (
                acceleration
                + self.gains[0] * (data.oMf[frame_id].translation - reference)
                + self.gains[1] * velocity
            )
            jacobians.append(jacobian.copy())
            drifts.append(drift.copy())
        return mass, nonlinear, np.vstack(jacobians), np.concatenate(drifts)

    def calc(self, q, v, u) -> ContactDynamicsResult:
        np, _ = _dependencies()
        q, v, u = (np.asarray(value, dtype=float) for value in (q, v, u))
        self._validate(q, v, u)
        mass, nonlinear, jacobian, drift = self._quantities(q, v)
        nc = jacobian.shape[0]
        kkt = np.block([[mass, -jacobian.T], [jacobian, np.zeros((nc, nc))]])
        rhs = np.concatenate((self.actuation @ u - nonlinear, -drift))
        solution = np.linalg.solve(kkt, rhs)
        acceleration, force = solution[: self.robot.nv], solution[self.robot.nv :]
        return ContactDynamicsResult(
            acceleration,
            force.reshape(-1, 3),
            mass,
            nonlinear,
            jacobian,
            drift,
            mass @ acceleration + nonlinear - self.actuation @ u - jacobian.T @ force,
            jacobian @ acceleration + drift,
        )

    def quasi_static_torques(self):
        """Minimum-norm bilateral equilibrium seed, without inequality constraints."""
        np, _ = _dependencies()
        _, nonlinear, jacobian, _ = self._quantities(
            self.reference_configuration, np.zeros(self.robot.nv)
        )
        solution, _, _, _ = np.linalg.lstsq(
            np.column_stack((self.actuation, jacobian.T)), nonlinear, rcond=None
        )
        residual = np.column_stack((self.actuation, jacobian.T)) @ solution - nonlinear
        if np.max(np.abs(residual)) > 1e-8 * (1 + np.max(np.abs(nonlinear))):
            raise ValueError("selected contacts cannot balance this reference configuration")
        return solution[:12].copy()

    def finite_difference_derivatives(self, q, v, u, *, epsilon=1e-6):
        """Local state/control derivatives of acceleration and world contact force.

        Return (da_dx, da_du, df_dx, df_du). Uses Pinocchio configuration integrate
        and additive velocity increments, matching StateMultibody's tangent chart.
        This offline oracle deliberately does not claim optimized derivatives.
        """
        np, pin = _dependencies()
        q, v, u = (np.asarray(value, dtype=float) for value in (q, v, u))
        self._validate(q, v, u)
        if not np.isfinite(epsilon) or epsilon <= 0:
            raise ValueError("epsilon must be finite and positive")
        state_a, state_f, control_a, control_f = [], [], [], []
        for i in range(self.robot.ndx):
            dq, dv = np.zeros(self.robot.nv), np.zeros(self.robot.nv)
            (dq if i < self.robot.nv else dv)[i % self.robot.nv] = epsilon
            plus = self.calc(pin.integrate(self.robot.model, q, dq), v + dv, u)
            minus = self.calc(pin.integrate(self.robot.model, q, -dq), v - dv, u)
            state_a.append((plus.acceleration - minus.acceleration) / (2 * epsilon))
            state_f.append((plus.forces_world.ravel() - minus.forces_world.ravel()) / (2 * epsilon))
        for i in range(12):
            du = np.zeros(12)
            du[i] = epsilon
            plus, minus = self.calc(q, v, u + du), self.calc(q, v, u - du)
            control_a.append((plus.acceleration - minus.acceleration) / (2 * epsilon))
            control_f.append(
                (plus.forces_world.ravel() - minus.forces_world.ravel()) / (2 * epsilon)
            )
        return tuple(
            np.stack(columns, axis=-1) for columns in (state_a, control_a, state_f, control_f)
        )

    def crocoddyl_model(self):
        """Construct the separate Crocoddyl contact action with identical conventions."""
        np, pin = _dependencies()
        import crocoddyl

        # This wheel combination can segfault in the first native calc(), so
        # reject it before creating an action rather than risking the process.
        if crocoddyl.__version__ == "3.2.1" and tuple(
            map(int, pin.__version__.split(".")[:2])
        ) >= (4, 1):
            raise ImportError(
                "Go2 contact reference: Crocoddyl 3.2.1 is incompatible with "
                f"Pinocchio {pin.__version__}; install crocoddyl==3.2.1 pin==4.0.0 "
                "together in a fresh environment (or use the project's crocoddyl extra)"
            )

        state = crocoddyl.StateMultibody(self.robot.model)
        actuation = crocoddyl.ActuationModelFloatingBase(state)
        contacts = crocoddyl.ContactModelMultiple(state, actuation.nu)
        for name, frame_id, position in zip(self.feet, self.frame_ids, self.positions):
            contacts.addContact(
                name,
                crocoddyl.ContactModel3D(
                    state, frame_id, position, pin.LOCAL_WORLD_ALIGNED, actuation.nu, self.gains
                ),
            )
        return crocoddyl.DifferentialActionModelContactFwdDynamics(
            state, actuation, contacts, crocoddyl.CostModelSum(state, actuation.nu), 0.0, True
        )
