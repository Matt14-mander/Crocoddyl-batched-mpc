"""Go2 rigid-body dynamics in Torch, with bilateral fixed point contacts.

URDF parsing happens once on CPU. Evaluation, KKT solves and automatic local
derivatives use only Torch on the chosen device. No Pinocchio callback is used.
"""

import math
import xml.etree.ElementTree as ET
from importlib.resources import files

import torch
from torch import Tensor

from ..costs import ManifoldQuadraticCost
from ..ddp_problem import DDPProblem
from ..lie import quaternion_matrix, skew
from ..manifolds import FloatingBaseManifold
from .go2 import GO2_JOINT_NAMES, GO2_LEGS, ContactDynamicsResult


def _mv(matrix, vector):
    return (matrix @ vector[..., None]).squeeze(-1)


def _motion_cross(v, w):
    return torch.cat(
        (
            torch.linalg.cross(v[..., 3:], w[..., :3]) + torch.linalg.cross(v[..., :3], w[..., 3:]),
            torch.linalg.cross(v[..., 3:], w[..., 3:]),
        ),
        -1,
    )


def _force_cross(v, f):
    return torch.cat(
        (
            torch.linalg.cross(v[..., 3:], f[..., :3]),
            torch.linalg.cross(v[..., :3], f[..., :3]) + torch.linalg.cross(v[..., 3:], f[..., 3:]),
        ),
        -1,
    )


def _origin(element):
    origin = element.find("origin")
    attributes = {} if origin is None else origin.attrib
    p = torch.tensor(
        [float(x) for x in attributes.get("xyz", "0 0 0").split()], dtype=torch.float64
    )
    r, pitch, y = [float(x) for x in attributes.get("rpy", "0 0 0").split()]
    cr, sr, cp, sp, cy, sy = (
        math.cos(r),
        math.sin(r),
        math.cos(pitch),
        math.sin(pitch),
        math.cos(y),
        math.sin(y),
    )
    R = torch.tensor(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=torch.float64,
    )
    return R, p


def _spatial_transform(R, p):
    Rt = R.transpose(-1, -2)
    zero = torch.zeros_like(Rt)
    return torch.cat((torch.cat((Rt, -Rt @ skew(p)), -1), torch.cat((zero, Rt), -1)), -2)


def _urdf_parameters():
    """Collapse fixed links including rotors/feet into their moving parent."""
    root = ET.fromstring(files("crocoddyl_batched_mpc").joinpath("assets/go2/go2.urdf").read_text())
    links = {link.attrib["name"]: link for link in root.findall("link")}
    children = {}
    for joint in root.findall("joint"):
        children.setdefault(joint.find("parent").attrib["link"], []).append(joint)
    inertias = [torch.zeros(6, 6, dtype=torch.float64) for _ in range(13)]
    placements, joint_data = {}, {}

    def visit(link_name, body, R, p):
        placements[link_name] = (body, p)
        inertial = links[link_name].find("inertial")
        if inertial is not None:
            Ri, pi = _origin(inertial)
            mass = float(inertial.find("mass").attrib["value"])
            a = inertial.find("inertia").attrib
            inertia = torch.tensor(
                [
                    [float(a["ixx"]), float(a["ixy"]), float(a["ixz"])],
                    [float(a["ixy"]), float(a["iyy"]), float(a["iyz"])],
                    [float(a["ixz"]), float(a["iyz"]), float(a["izz"])],
                ],
                dtype=torch.float64,
            )
            rotation, com = R @ Ri, p + R @ pi
            C = skew(com)
            inertias[body] += torch.cat(
                (
                    torch.cat((mass * torch.eye(3, dtype=torch.float64), -mass * C), -1),
                    torch.cat((mass * C, rotation @ inertia @ rotation.T - mass * C @ C), -1),
                ),
                -2,
            )
        for joint in children.get(link_name, []):
            Rj, pj = _origin(joint)
            rotation, position = R @ Rj, p + R @ pj
            child = joint.find("child").attrib["link"]
            if joint.attrib["type"] == "fixed":
                visit(child, body, rotation, position)
            else:
                index = GO2_JOINT_NAMES.index(joint.attrib["name"]) + 1
                if joint.attrib["type"] != "revolute":
                    raise ValueError("Go2 requires scalar revolute joints")
                axis = torch.tensor(
                    [float(x) for x in joint.find("axis").attrib["xyz"].split()],
                    dtype=torch.float64,
                )
                limit = joint.find("limit").attrib
                joint_data[index] = (
                    body,
                    rotation,
                    position,
                    axis,
                    float(limit["effort"]),
                    float(limit["lower"]),
                    float(limit["upper"]),
                )
                visit(
                    child,
                    index,
                    torch.eye(3, dtype=torch.float64),
                    torch.zeros(3, dtype=torch.float64),
                )

    visit("base", 0, torch.eye(3, dtype=torch.float64), torch.zeros(3, dtype=torch.float64))
    return torch.stack(inertias), joint_data, placements


class Go2TorchDynamics:
    """Full floating-base Go2, fixed bilateral point contacts, semi-implicit Euler.

    x=[q(19),v(18)], u=12 joint torques. Base velocities/increments are body
    [linear,angular]. Forces are world aligned in the requested feet order.
    The fixed URDF topology is shared; reference poses may be per environment.
    This correctness baseline has no inequality constraints or contact switching.
    """

    nx, ndx, nu = 37, 36, 12

    def __init__(
        self,
        *,
        dt=0.01,
        feet=GO2_LEGS,
        stabilization=(100.0, 20.0),
        reference_configuration=None,
        device="cpu",
        dtype=torch.float64,
    ):
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be finite and positive")
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("Go2 requires float32 or float64")
        self.device, self.dtype, self.dt = torch.device(device), dtype, dt
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.feet = tuple(feet)
        if (
            not self.feet
            or len(set(self.feet)) != len(self.feet)
            or any(f not in GO2_LEGS for f in self.feet)
        ):
            raise ValueError("feet must be a nonempty unique sequence of FL/FR/RL/RR")
        if len(stabilization) != 2 or any(not math.isfinite(g) or g < 0 for g in stabilization):
            raise ValueError("stabilization requires two finite nonnegative gains")
        self.gains = tuple(stabilization)
        self.manifold = FloatingBaseManifold(12)
        inertia, joints, placements = _urdf_parameters()
        def convert(t):
            return t.to(device=self.device, dtype=dtype)
        self.inertias = convert(inertia)
        self.parents = tuple(joints[i][0] for i in range(1, 13))
        self.rotations = convert(torch.stack([joints[i][1] for i in range(1, 13)]))
        self.translations = convert(torch.stack([joints[i][2] for i in range(1, 13)]))
        self.axes = convert(torch.stack([joints[i][3] for i in range(1, 13)]))
        self.torque_limits = torch.tensor(
            [joints[i][4] for i in range(1, 13)], device=self.device, dtype=dtype
        )
        self.lower_limits = torch.tensor(
            [joints[i][5] for i in range(1, 13)], device=self.device, dtype=dtype
        )
        self.upper_limits = torch.tensor(
            [joints[i][6] for i in range(1, 13)], device=self.device, dtype=dtype
        )
        self.foot_bodies = tuple(placements[f"{f}_foot"][0] for f in self.feet)
        self.foot_offsets = convert(torch.stack([placements[f"{f}_foot"][1] for f in self.feet]))
        self.eye3 = torch.eye(3, device=self.device, dtype=dtype)
        self.base_jacobian = torch.eye(18, device=self.device, dtype=dtype)[:6]
        self.joint_subspaces = torch.cat((torch.zeros_like(self.axes), self.axes), -1)
        self.joint_jacobians = (
            self.joint_subspaces[:, :, None]
            * torch.eye(18, device=self.device, dtype=dtype)[6:, None, :]
        )
        self.gravity = torch.tensor([0.0, 0.0, -9.81], device=self.device, dtype=dtype)
        self.actuation = torch.eye(18, device=self.device, dtype=dtype)[:, 6:]
        if reference_configuration is None:
            q = torch.zeros(1, 19, device=self.device, dtype=dtype)
            q[:, 6] = 1
            q[:, 7:] = q.new_tensor([0.0, 0.8, -1.6] * 4)
            positions = self._kinematics(q, q.new_zeros(1, 18))[3]
            q[:, 2] = -positions[..., 2].mean(-1)
            self.reference_configuration = q[0]
        else:
            q = torch.as_tensor(reference_configuration, device=self.device, dtype=dtype).clone()
            if q.ndim not in (1, 2) or q.shape[-1] != 19:
                raise ValueError("reference_configuration must be [19] or [B,19]")
            state = torch.cat((q, q.new_zeros(*q.shape[:-1], 18)), -1)
            if not self.manifold.is_valid(state).all():
                raise ValueError("reference_configuration must contain valid unit quaternions")
            self.reference_configuration = q
        self.contact_positions = self._kinematics(
            self.reference_configuration,
            self.reference_configuration.new_zeros(*self.reference_configuration.shape[:-1], 18),
        )[3].detach()

    @property
    def parameter_batch_size(self):
        return (
            self.reference_configuration.shape[0]
            if self.reference_configuration.ndim == 2
            else None
        )

    @property
    def standing_state(self):
        q = self.reference_configuration
        return torch.cat((q, q.new_zeros(*q.shape[:-1], 18)), -1)

    def _kinematics(self, q, v):
        R0 = quaternion_matrix(q[..., 3:7])
        rotations, positions = [R0], [q[..., :3]]
        jacobians = [self.base_jacobian.expand(*q.shape[:-1], 6, 18)]
        velocities, accelerations = [v[..., :6]], [torch.zeros_like(v[..., :6])]
        for i, parent in enumerate(self.parents):
            A = skew(self.axes[i])
            angle = q[..., 7 + i, None, None]
            R = self.rotations[i] @ (self.eye3 + angle.sin() * A + (1 - angle.cos()) * (A @ A))
            p = self.translations[i].expand(*q.shape[:-1], 3)
            X = _spatial_transform(R, p)
            J = X @ jacobians[parent] + self.joint_jacobians[i]
            vj = self.joint_subspaces[i] * v[..., 6 + i, None]
            vi = _mv(X, velocities[parent]) + vj
            ai = _mv(X, accelerations[parent]) + _motion_cross(vi, vj)
            rotations.append(rotations[parent] @ R)
            positions.append(positions[parent] + _mv(rotations[parent], p))
            jacobians.append(J)
            velocities.append(vi)
            accelerations.append(ai)
        foot_positions, foot_jacobians, foot_drifts = [], [], []
        for i, body in enumerate(self.foot_bodies):
            offset, R = self.foot_offsets[i], rotations[body]
            C = torch.cat((self.eye3, -skew(offset)), -1)
            velocity = _mv(C, velocities[body])
            acceleration = _mv(C, accelerations[body]) + torch.linalg.cross(
                velocities[body][..., 3:], velocity
            )
            foot_positions.append(positions[body] + _mv(R, offset.expand(*q.shape[:-1], 3)))
            foot_jacobians.append(R @ C @ jacobians[body])
            foot_drifts.append(_mv(R, acceleration))
        return (
            torch.stack(rotations, -3),
            torch.stack(jacobians, -3),
            (torch.stack(velocities, -2), torch.stack(accelerations, -2)),
            torch.stack(foot_positions, -2),
            torch.cat(foot_jacobians, -2),
            torch.cat(foot_drifts, -1),
        )

    def quantities(self, q: Tensor, v: Tensor):
        """Return M, h, world foot J, and stabilized Jdot*v."""
        rotations, J, (velocity, acceleration), positions, contact_J, drift = self._kinematics(q, v)
        inertia_J = self.inertias @ J
        mass = (J.transpose(-1, -2) @ inertia_J).sum(-3)
        gravity_local = _mv(
            rotations.transpose(-1, -2), self.gravity.expand(*rotations.shape[:-2], 3)
        )
        gravity_spatial = torch.cat((gravity_local, torch.zeros_like(gravity_local)), -1)
        force = _mv(self.inertias, acceleration - gravity_spatial) + _force_cross(
            velocity, _mv(self.inertias, velocity)
        )
        nonlinear = _mv(J.transpose(-1, -2), force).sum(-2)
        drift = (
            drift
            + self.gains[0] * (positions - self.contact_positions).flatten(-2)
            + self.gains[1] * _mv(contact_J, v)
        )
        return mass, nonlinear, contact_J, drift

    def foot_positions(self, q: Tensor) -> Tensor:
        """World foot-frame origins, ordered by feet, with shape [...,nf,3]."""
        return self._kinematics(q, q.new_zeros(*q.shape[:-1], 18))[3]

    def contact_dynamics(self, x: Tensor, u: Tensor) -> ContactDynamicsResult:
        """Batched continuous accelerations/forces and residuals; no host transfer."""
        if x.shape[-1] != self.nx or u.shape != (*x.shape[:-1], self.nu):
            raise ValueError("Go2 expects matching [...,37] states and [...,12] torques")
        if (
            x.device != self.device
            or u.device != self.device
            or x.dtype != self.dtype
            or u.dtype != self.dtype
        ):
            raise ValueError("state/control must match Go2 device and dtype")
        mass, nonlinear, J, drift = self.quantities(x[..., :19], x[..., 19:])
        nc = J.shape[-2]
        zero = x.new_zeros(*x.shape[:-1], nc, nc)
        kkt = torch.cat((torch.cat((mass, -J.transpose(-1, -2)), -1), torch.cat((J, zero), -1)), -2)
        rhs = torch.cat((_mv(self.actuation, u) - nonlinear, -drift), -1)
        solution, info = torch.linalg.solve_ex(kkt, rhs[..., None], check_errors=False)
        solution = torch.where(
            (info == 0)[..., None], solution.squeeze(-1), torch.full_like(rhs, float("nan"))
        )
        a, f = solution[..., :18], solution[..., 18:]
        return ContactDynamicsResult(
            a,
            f.reshape(*x.shape[:-1], len(self.feet), 3),
            mass,
            nonlinear,
            J,
            drift,
            _mv(mass, a) + nonlinear - _mv(self.actuation, u) - _mv(J.transpose(-1, -2), f),
            _mv(J, a) + drift,
        )

    def calc(self, x: Tensor, u: Tensor) -> Tensor:
        a = self.contact_dynamics(x, u).acceleration
        v = x[..., 19:] + self.dt * a
        dx = torch.cat((self.dt * v, self.dt * a), -1)
        return self.manifold.integrate(x, dx)

    def _local_jacobians(self, function, x, u):
        """Vectorized forward AD of independent batch environments."""
        delta = x.new_zeros(*x.shape[:-1], self.ndx + self.nu)
        basis = torch.eye(self.ndx + self.nu, device=x.device, dtype=x.dtype)
        directions = basis[:, None, :].expand(-1, x.shape[0], -1)

        def local(d):
            return function(self.manifold.integrate(x, d[..., :36]), u + d[..., 36:])

        def column(direction):
            return torch.func.jvp(local, (delta,), (direction,))[1]

        jac = torch.func.vmap(column)(directions).permute(1, 2, 0)
        return jac[..., :36], jac[..., 36:]

    def calc_diff(self, x: Tensor, u: Tensor):
        predicted = self.calc(x, u).detach()
        return self._local_jacobians(
            lambda state, control: self.manifold.diff(self.calc(state, control), predicted), x, u
        )

    def contact_derivatives(self, x: Tensor, u: Tensor):
        def values(state, control):
            result = self.contact_dynamics(state, control)
            return torch.cat((result.acceleration, result.forces_world.flatten(-2)), -1)

        dx, du = self._local_jacobians(values, x, u)
        return dx[..., :18, :], du[..., :18, :], dx[..., 18:, :], du[..., 18:, :]

    def quasi_static_torques(self):
        """Minimum norm equilibrium seed, computed once outside the solve loop."""
        q = self.reference_configuration
        _, h, J, _ = self.quantities(q, q.new_zeros(*q.shape[:-1], 18))
        system = torch.cat((self.actuation.expand(*q.shape[:-1], 18, 12), J.transpose(-1, -2)), -1)
        # Initialization only: avoid float32 SVD error in the equilibrium seed.
        seed = _mv(torch.linalg.pinv(system.double()), h.double()).to(self.dtype)
        if not torch.allclose(_mv(system, seed), h, atol=1e-5, rtol=1e-5):
            raise ValueError("selected contacts cannot balance the reference")
        return seed[..., :12]


class Go2StandingCost(ManifoldQuadraticCost):
    """Track the standing pose and quasi-static torque, including feedforward."""

    def __init__(self, dynamics: Go2TorchDynamics):
        reference = dynamics.standing_state
        weights = reference.new_tensor(
            [500.0, 500.0, 1500.0, 300.0, 300.0, 300.0] + [30.0] * 12 + [10.0] * 6 + [1.0] * 12
        )
        Q = torch.diag(weights)
        super().__init__(
            dynamics.manifold,
            reference,
            Q,
            0.01 * torch.eye(12, device=reference.device, dtype=reference.dtype),
            10 * Q,
        )
        self.u_reference = dynamics.quasi_static_torques().detach()

    def calc(self, x, u=None):
        return super().calc(x, None if u is None else u - self.u_reference)

    def calc_diff(self, x, u=None):
        return super().calc_diff(x, None if u is None else u - self.u_reference)


def go2_standing_problem(
    batch_size=1,
    horizon=10,
    *,
    dt=0.01,
    device="cpu",
    dtype=torch.float64,
    max_iterations=3,
    **dynamics_kwargs,
):
    dynamics = Go2TorchDynamics(dt=dt, device=device, dtype=dtype, **dynamics_kwargs)
    cost = Go2StandingCost(dynamics)
    seed = cost.u_reference.expand(batch_size, 12)[:, None].expand(-1, horizon, -1).clone()
    return DDPProblem.from_models(
        dynamics,
        cost,
        batch_size,
        horizon,
        manifold=dynamics.manifold,
        u_init=seed,
        max_iterations=max_iterations,
        cost_tolerance=1e-6,
        gradient_tolerance=1e-5,
        regularization_init=1e-5,
    )
