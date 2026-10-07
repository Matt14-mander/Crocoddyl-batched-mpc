"""Go2 versions of Crocoddyl's four example gaits and Torch feedback execution.

Crocoddyl is used only by the offline CPU planner. Execution uses Torch only.
The example's force barriers guide optimization; explicit execution gates, not
optimizer convergence, determine whether a trajectory is physically usable.
"""

import math
from dataclasses import dataclass

import torch

from ..qp import solve_inequality_qp
from ..result import MPCResult, SolveStatus
from .go2 import GO2_LEGS, load_go2
from .go2_hybrid import Go2HybridDynamics
from .go2_torch import _mv

GO2_GAIT_SWINGS = {
    "walk": (("RR",), ("FR",), ("RL",), ("FL",)),
    "trot": (("FR", "RL"), ("FL", "RR")),
    "pace": (("FR", "RR"), ("FL", "RL")),
    "bound": (("FL", "FR"), ("RL", "RR")),
}
_METHODS = dict(
    zip(
        GO2_GAIT_SWINGS,
        (
            "createWalkingProblem",
            "createTrottingProblem",
            "createPacingProblem",
            "createBoundingProblem",
        ),
    )
)


def _settings(gait, step_length, step_height, dt, step_knots, support_knots):
    if gait not in GO2_GAIT_SWINGS:
        raise ValueError("gait must be walk, trot, pace or bound")
    if not all(math.isfinite(v) and v > 0 for v in (step_length, step_height, dt)):
        raise ValueError("step length/height and dt must be finite and positive")
    for value in (step_knots, support_knots):
        if not isinstance(value, int) or isinstance(value, bool) or value < 2:
            raise ValueError("step/support knots must be integers >=2")


def _native_builder(robot, guided):
    # Keep the optional dependency out of imports and all Torch execution paths.
    import crocoddyl as c
    import numpy as np
    from crocoddyl.utils.quadruped import SimpleQuadrupedalGaitProblem

    class ForceResidual(c.ResidualModelAbstract):
        """World-frame diamond cone and normal bounds, including impulse Jacobians."""

        def __init__(self, state, name, nu, impact):
            super().__init__(state, 6, nu, True, True, not impact)
            self.name, self.impact = name, impact
            self.matrix = np.array(
                [
                    [1.0, 1.0, -0.6],
                    [1.0, -1.0, -0.6],
                    [-1.0, 1.0, -0.6],
                    [-1.0, -1.0, -0.6],
                    [0.0, 0.0, -1.0],
                    [0.0, 0.0, 1.0],
                ]
            )

        def createData(self, shared):
            data = c.ResidualDataAbstract(self, shared)
            collection = shared.impulses.impulses if self.impact else shared.contacts.contacts
            data.force = collection[self.name]
            return data

        def calc(self, data, x, u=None):
            data.r[:] = self.matrix @ data.force.f.linear

        def calcDiff(self, data, x, u=None):
            data.Rx[:] = self.matrix @ data.force.df_dx[:3]
            if not self.impact:
                data.Ru[:] = self.matrix @ data.force.df_du[:3]

    class Builder(SimpleQuadrupedalGaitProblem):
        def guide(self, model, impact=False):
            costs = model.costs if impact else model.differential.costs
            if guided:
                # Original example's very strong orientation prior is unsuitable
                # for Go2's dynamic two-foot phases; allow body roll/pitch motion.
                weights = np.array([0.0] * 3 + [20.0] * 3 + [0.01] * 12 + [2.0] * 6 + [1.0] * 12)
                costs.costs["stateReg"].cost.activation.weights = weights**2
                items = (
                    model.impulses.impulses if impact else model.differential.contacts.contacts
                ).todict()
                bound = np.array([0.0] * 5 + [200.0])
                bound -= np.array([0.001] * 5 + [0.0]) if impact else 2.0
                activation = c.ActivationModelQuadraticBarrier(
                    c.ActivationBounds(np.full(6, -np.inf), bound)
                )
                for name in items:
                    residual = ForceResidual(self.state, name, 0 if impact else 12, impact)
                    costs.addCost(
                        name + "_forceGuide",
                        c.CostModelResidual(self.state, activation, residual),
                        1e6 if impact else 1e2,
                    )
            if not impact:
                model.u_lb = -robot.torque_limits
                model.u_ub = robot.torque_limits
            return model

        def createSwingFootModel(self, *args, **kwargs):
            return self.guide(super().createSwingFootModel(*args, **kwargs))

        def createImpulseModel(self, *args, **kwargs):
            args = (list(robot.foot_ids), *args[1:])
            model = super().createImpulseModel(*args, **kwargs)
            # Land on the actual plane, rather than upstream's last swing knot
            # (which is still above the ground for its triangular height curve).
            for name, item in model.costs.costs.todict().items():
                if name.endswith("_footTrack"):
                    goal = item.cost.residual.reference.copy()
                    goal[2] = 0.0
                    item.cost.residual.reference = goal
                    item.weight = 1e10
            # Impulse-based costs require force Jacobians; the upstream example
            # leaves this disabled because it has no impulse-cone costs.
            model = c.ActionModelImpulseFwdDynamics(
                model.state, model.impulses, model.costs, 0.0, 1e-12, True
            )
            return self.guide(model, True)

    robot.model.referenceConfigurations["standing"] = robot.standing_configuration()
    builder = Builder(robot.model, "FL_foot", "FR_foot", "RL_foot", "RR_foot")
    builder.mu = 0.6
    return builder


def build_go2_gait_problem(
    gait,
    *,
    step_length=0.04,
    step_height=0.02,
    dt=0.02,
    step_knots=20,
    support_knots=10,
    initial_state=None,
    first_step=True,
    guided=True,
):
    """Build one exact example contact sequence on the pinned Go2 URDF (CPU)."""
    _settings(gait, step_length, step_height, dt, step_knots, support_knots)
    import numpy as np

    robot = load_go2()
    builder = _native_builder(robot, guided)
    builder.firstStep = bool(first_step)
    x0 = (
        np.r_[robot.standing_configuration(), np.zeros(18)]
        if initial_state is None
        else np.asarray(initial_state).copy()
    )
    if x0.shape != (37,) or not np.isfinite(x0).all() or abs(np.linalg.norm(x0[3:7]) - 1) > 1e-8:
        raise ValueError("initial state must be finite Go2 [37] with unit quaternion")
    problem = getattr(builder, _METHODS[gait])(
        x0, step_length, step_height, dt, step_knots, support_knots
    )
    import crocoddyl as c
    import pinocchio as pin

    advance = step_length * (0.75 if first_step and gait != "bound" else 1.0)
    goal = pin.centerOfMass(robot.model, robot.model.createData(), x0[:19]).copy()
    goal[0] += advance
    final = builder.createSwingFootModel(dt, list(robot.foot_ids), comTask=goal)
    # Give all-contact dynamics time to settle the body after the final pair.
    running = list(problem.runningModels) + [final] * (2 * support_knots)
    data = robot.model.createData()
    pin.framesForwardKinematics(robot.model, data, x0[:19])
    anchors = np.stack([data.oMf[f].translation.copy() for f in robot.foot_ids])
    adapted = []
    # Upstream reuses four-support models across phases. Rebuild contacts in
    # chronological order so each node has its actual fixed world anchors.
    for old in [*running, final]:
        if old.nu == 0:
            for name, item in old.costs.costs.todict().items():
                if name.endswith("_footTrack"):
                    leg = GO2_LEGS.index(name[:2])
                    anchors[leg] = item.cost.residual.reference.copy()
            model = old
        else:
            contacts = c.ContactModelMultiple(builder.state, 12)
            for name, item in old.differential.contacts.contacts.todict().items():
                leg = GO2_LEGS.index(name[:2])
                contacts.addContact(
                    name,
                    c.ContactModel3D(
                        builder.state,
                        item.contact.id,
                        anchors[leg],
                        pin.LOCAL_WORLD_ALIGNED,
                        12,
                        np.array([100.0, 50.0]),
                    ),
                )
            diff = c.DifferentialActionModelContactFwdDynamics(
                builder.state, builder.actuation, contacts, old.differential.costs, 0.0, True
            )
            for foot in robot.foot_ids:
                name = robot.model.frames[foot].name + "_groundGuide"
                if name not in diff.costs.costs.todict():
                    activation = c.ActivationModelQuadraticBarrier(
                        c.ActivationBounds(np.array([-np.inf, -np.inf, -0.001]), np.full(3, np.inf))
                    )
                    residual = c.ResidualModelFrameTranslation(builder.state, foot, np.zeros(3), 12)
                    diff.costs.addCost(
                        name, c.CostModelResidual(builder.state, activation, residual), 1e9
                    )
            model = c.IntegratedActionModelEuler(diff, dt)
            model.u_lb = -robot.torque_limits
            model.u_ub = robot.torque_limits
        model.go2_anchors = anchors.copy()
        adapted.append(model)
    problem = c.ShootingProblem(x0, adapted[:-1], adapted[-1])
    return problem


@dataclass
class Go2GaitPlan:
    gait: str
    states: torch.Tensor  # [N+1,37]
    torques: torch.Tensor  # [N,12]; zero on zero-time impulses
    gains: torch.Tensor  # [N,12,36]; Crocoddyl feedback convention
    contacts: torch.Tensor  # [N,4]; contacts used by each flow or impulse node
    landings: torch.Tensor  # [N,4]; newly landing group, excluding retained feet
    anchors: torch.Tensor  # [N,4,3]; chronological world contact anchors
    durations: torch.Tensor  # [N]; impulses have exactly zero duration
    native_solvers: tuple  # offline CPU oracle, never accessed during execution
    converged: tuple[bool, ...]
    cycles: int
    dt: float

    @property
    def duration(self):
        return float(self.durations.sum())

    def to(self, device):
        return Go2GaitPlan(
            self.gait,
            self.states.to(device),
            self.torques.to(device),
            self.gains.to(device),
            self.contacts.to(device),
            self.landings.to(device),
            self.anchors.to(device),
            self.durations.to(device),
            (),
            self.converged,
            self.cycles,
            self.dt,
        )

    def save(self, path):
        """Export tensors/metadata only; never pickle native solver instances."""
        torch.save(
            dict(
                format="go2-gaits-v1",
                gait=self.gait,
                cycles=self.cycles,
                dt=self.dt,
                converged=self.converged,
                **{
                    name: getattr(self, name).detach().cpu()
                    for name in (
                        "states",
                        "torques",
                        "gains",
                        "contacts",
                        "landings",
                        "anchors",
                        "durations",
                    )
                },
            ),
            path,
        )


def load_go2_gait_plan(path, *, device="cpu"):
    """Load an exported plan without NumPy, Pinocchio or Crocoddyl."""
    data = torch.load(path, map_location=device, weights_only=True)
    if data.pop("format", None) != "go2-gaits-v1":
        raise ValueError("unsupported Go2 gait plan format")
    plan = Go2GaitPlan(native_solvers=(), **data)
    _validate_plan(plan)
    return plan


def _validate_plan(plan):
    if plan.gait not in GO2_GAIT_SWINGS or not math.isfinite(plan.dt) or plan.dt <= 0:
        raise ValueError("invalid gait plan metadata")
    count = len(plan.torques)
    device = plan.states.device
    shapes = dict(
        states=(count + 1, 37),
        torques=(count, 12),
        gains=(count, 12, 36),
        contacts=(count, 4),
        landings=(count, 4),
        anchors=(count, 4, 3),
        durations=(count,),
    )
    if count < 1:
        raise ValueError("gait plan must have at least one node")
    for name, shape in shapes.items():
        value = getattr(plan, name)
        dtype = torch.bool if name in ("contacts", "landings") else torch.float64
        if (
            value.shape != shape
            or value.device != device
            or value.dtype != dtype
            or not torch.isfinite(value).all()
        ):
            raise ValueError("gait plan tensors must match shape/device and float64/bool metadata")
    if not ((plan.durations == 0) | (plan.durations == plan.dt)).all():
        raise ValueError("gait nodes must have dt duration or zero-time impact")
    if not (plan.contacts.sum(-1) >= 2).all() or (plan.landings & ~plan.contacts).any():
        raise ValueError("invalid gait support/landing mask")


def solve_go2_gait(gait, *, cycles=1, iterations=150, guided=True, **settings):
    """Offline Box-FDDP, continuation force guides, then export padded Torch data.

    A budget-exhausted result is exported for audit, never silently certified.
    Each cycle starts from the actual optimized terminal state of its predecessor.
    """
    if not isinstance(cycles, int) or isinstance(cycles, bool) or cycles < 1:
        raise ValueError("cycles must be a positive integer")
    if not isinstance(iterations, int) or isinstance(iterations, bool) or iterations < 1:
        raise ValueError("iterations must be a positive integer")
    import crocoddyl as c
    import numpy as np

    solvers, convergence, states, torques, gains, masks, landings, durations = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    x0 = None
    anchor_samples = []
    for cycle in range(cycles):
        problem = build_go2_gait_problem(
            gait, initial_state=x0, first_step=cycle == 0, guided=guided, **settings
        )
        solver = c.SolverBoxFDDP(problem)
        solver.th_stop = 1e-7
        seed = [problem.x0.copy() for _ in range(problem.T + 1)]
        controls = problem.quasiStatic(seed[:-1])
        ok = solver.solve(seed, controls, iterations, False)
        if guided:
            for penalty in (1e4, 1e6):
                for model in [*problem.runningModels, problem.terminalModel]:
                    costs = model.costs if model.nu == 0 else model.differential.costs
                    for name, item in costs.costs.todict().items():
                        if name.endswith("_forceGuide"):
                            item.weight = penalty * (1e4 if model.nu == 0 else 1.0)
                controls = [np.asarray(u).copy() for u in solver.us]
                ok = solver.solve(problem.rollout(controls), controls, iterations, True)
        # Export actual dynamics, never a budget-exhausted trajectory with gaps.
        solver.setCandidate(problem.rollout(solver.us), solver.us, True)
        # Refresh K at the exported states/controls, rather than the previous
        # accepted backward-pass trajectory. Its sign is u = uff - K*dx.
        problem.calc(solver.xs, solver.us)
        problem.calcDiff(solver.xs, solver.us)
        solver.computeDirection(False)
        solvers.append(solver)
        convergence.append(bool(ok))
        states.extend(np.asarray(x).copy() for x in solver.xs[:-1])
        for t, model in enumerate(problem.runningModels):
            anchor_samples.append(model.go2_anchors.copy())
            impact = model.nu == 0
            torques.append(np.zeros(12) if impact else np.asarray(solver.us[t]).copy())
            gains.append(np.zeros((12, 36)) if impact else np.asarray(solver.K[t]).copy())
            items = model.impulses.impulses if impact else model.differential.contacts.contacts
            names = items.todict()
            mask = [
                f"{leg}_foot_" + ("impulse" if impact else "contact") in names for leg in GO2_LEGS
            ]
            landings.append(
                [active and not previous for active, previous in zip(mask, masks[-1])]
                if impact
                else [False] * 4
            )
            masks.append(mask)
            durations.append(0.0 if impact else model.dt)
        x0 = np.asarray(solver.xs[-1]).copy()
    states.append(x0)
    return Go2GaitPlan(
        gait,
        torch.tensor(np.asarray(states), dtype=torch.float64),
        torch.tensor(np.asarray(torques), dtype=torch.float64),
        torch.tensor(np.asarray(gains), dtype=torch.float64),
        torch.tensor(masks, dtype=torch.bool),
        torch.tensor(landings, dtype=torch.bool),
        torch.tensor(np.asarray(anchor_samples), dtype=torch.float64),
        torch.tensor(durations, dtype=torch.float64),
        tuple(solvers),
        tuple(convergence),
        cycles,
        settings.get("dt", 0.02),
    )


class Go2GaitDynamics:
    """Torch equivalent of the adapted flow and zero-time sticking impulses.

    Go2 landings project all four target contacts, including retained feet.
    Physical checks belong to the executor below.
    """

    def __init__(self, *, dt=0.02, device="cpu"):
        self.model = Go2HybridDynamics(
            dt=dt, device=device, dtype=torch.float64, stabilization=(100.0, 50.0)
        )

    def evaluate(self, state, torque, contacts, durations, anchors):
        if state.ndim != 2 or state.shape[-1] != 37 or torque.shape != (len(state), 12):
            raise ValueError("gait dynamics expects state[B,37] and torque[B,12]")
        if contacts.shape != (len(state), 4) or durations.shape != (len(state),):
            raise ValueError("gait dynamics expects contacts[B,4] and duration[B]")
        if (
            contacts.dtype != torch.bool
            or any(v.device != self.model.device for v in (state, torque, contacts, durations))
            or any(v.dtype != torch.float64 for v in (state, torque, durations))
        ):
            raise ValueError("gait tensors must match device and float64/bool metadata")
        mass, h, J, drift, kkt = self.model._contact_system(state, contacts, anchors)
        flow_rhs = torch.cat((_mv(self.model.actuation, torque) - h, -drift), -1)
        impulse_rhs = torch.cat((torch.zeros_like(h), -_mv(J, state[:, 19:])), -1)
        impact = durations == 0
        rhs = torch.where(impact[:, None], impulse_rhs, flow_rhs)
        solution, info = torch.linalg.solve_ex(kkt, rhs[..., None], check_errors=False)
        solution = solution.squeeze(-1)
        solution = torch.where((info == 0)[:, None], solution, float("nan"))
        velocity = state[:, 19:] + torch.where(
            impact[:, None], solution[:, :18], durations[:, None] * solution[:, :18]
        )
        delta = torch.cat((durations[:, None] * velocity, velocity - state[:, 19:]), -1)
        next_state = self.model.manifold.integrate(state, delta)
        return next_state, solution[:, 18:].reshape(-1, 4, 3), J, mass


@dataclass
class Go2GaitControl:
    action: torch.Tensor
    result: MPCResult
    contacts: torch.Tensor
    durations: torch.Tensor
    node_indices: torch.Tensor
    finished: torch.Tensor


class Go2GaitController:
    """Torch batch feedback for an offline gait, with strict physical rejection.

    No native planner callback, state teleportation or action clipping. Indices
    advance only for usable actions; failure is sticky until selective reset.
    The finite plan must be stopped when finished, not treated as a standing MPC.
    """

    def __init__(self, plan, batch_size=1):
        if not isinstance(plan, Go2GaitPlan) or batch_size < 1:
            raise ValueError("a Go2GaitPlan and positive batch size are required")
        _validate_plan(plan)
        self.plan, self.batch_size = plan, batch_size
        self.dynamics = Go2GaitDynamics(dt=plan.dt, device=plan.states.device)
        self.indices = torch.zeros(batch_size, device=plan.states.device, dtype=torch.int64)
        self.failed = torch.zeros(batch_size, device=plan.states.device, dtype=torch.bool)
        self.device = plan.states.device
        self.matrix = plan.states.new_tensor(
            [[1, 1, -0.6], [1, -1, -0.6], [-1, 1, -0.6], [-1, -1, -0.6], [0, 0, -1], [0, 0, 1]]
        )
        self._warm = plan.torques[0].expand(batch_size, -1).clone()
        self._dual = plan.states.new_ones(batch_size, 48)

    def _project(self, state, action, contacts, impact, anchors):
        model = self.dynamics.model
        _, _, f0, Fu = model.control_affine_dynamics(
            state, contact_mask=contacts, contact_positions=anchors
        )
        force_map = Fu.reshape(self.batch_size, 4, 3, 12)
        rows = (contacts & ~impact[:, None]).repeat_interleave(6, -1)
        force_G = (self.matrix @ force_map).flatten(1, 2)
        force_b = (
            action.new_tensor([0, 0, 0, 0, 0, 200]) - (f0.reshape(-1, 4, 3) @ self.matrix.T)
        ).flatten(1)
        eye = torch.eye(12, device=self.device, dtype=torch.float64).expand(self.batch_size, -1, -1)
        G = torch.cat((torch.where(rows[..., None], force_G, 0.0), eye, -eye), -2)
        b = torch.cat(
            (
                torch.where(rows, force_b, 1.0),
                model.torque_limits.expand(self.batch_size, -1),
                model.torque_limits.expand(self.batch_size, -1),
            ),
            -1,
        )
        scale = G.norm(dim=-1).clamp_min(1.0)
        qp = solve_inequality_qp(
            eye,
            -action,
            G / scale[..., None],
            b / scale,
            initial=self._warm,
            initial_dual=self._dual,
            iterations=24,
            tolerance=1e-9,
        )
        self._warm.copy_(qp.x)
        self._dual.copy_(qp.dual)
        return qp

    @torch.no_grad()
    def compute(self, state):
        if (
            state.shape != (self.batch_size, 37)
            or state.device != self.device
            or state.dtype != torch.float64
        ):
            raise ValueError("gait state must match [batch,37], device and float64")
        model = self.dynamics.model
        live = self.indices < len(self.plan.torques)
        nodes = self.indices.clamp_max(len(self.plan.torques) - 1)
        valid = model.manifold.is_valid(state)
        safe = torch.where(valid[:, None], state, model.standing_state)
        error = model.manifold.diff(safe, self.plan.states[nodes])
        action = self.plan.torques[nodes] - _mv(self.plan.gains[nodes], error)
        contacts = self.plan.contacts[nodes]
        durations = self.plan.durations[nodes]
        anchors = self.plan.anchors[nodes]
        output, force, J, mass = self.dynamics.evaluate(safe, action, contacts, durations, anchors)
        impact = durations == 0
        raw_violation = (
            torch.where(
                contacts[..., None],
                force @ self.matrix.T - force.new_tensor([0, 0, 0, 0, 0, 200]),
                0.0,
            )
            .flatten(1)
            .amax(-1)
        )
        project = ~impact & (
            (raw_violation > 1e-7) | (action.abs() > model.torque_limits + 1e-7).any(-1)
        )
        qp_valid = torch.ones_like(project)
        qp_converged = torch.ones_like(project)
        iterations = torch.zeros_like(nodes)
        if self.device.type != "cpu" or bool(project.any()):
            qp = self._project(safe, action, contacts, impact, anchors)
            action = torch.where(project[:, None], qp.x, action)
            qp_valid = ~project | qp.feasible
            qp_converged = ~project | qp.converged
            iterations = torch.where(project, qp.iterations, iterations)
            output, force, J, mass = self.dynamics.evaluate(
                safe, action, contacts, durations, anchors
            )
        rows = contacts[..., None]
        inequalities = force @ self.matrix.T
        bound = force.new_tensor([0, 0, 0, 0, 0, 200])
        violation = (
            torch.where(rows, inequalities - bound, torch.zeros_like(inequalities))
            .flatten(1)
            .amax(-1)
        )
        positions = model.foot_positions(safe[:, :19])
        landed = (positions[..., 2].abs() <= 0.002) | ~contacts
        landed &= ((positions - anchors).norm(dim=-1) <= 0.002) | ~contacts
        approaching = (
            _mv(J, safe[:, 19:]).reshape(-1, 4, 3)[..., 2] <= 1e-5
        ) | ~self.plan.landings[nodes]
        velocity_residual = _mv(J, output[:, 19:]).abs().amax(-1)
        old_energy = 0.5 * (safe[:, 19:] * _mv(mass, safe[:, 19:])).sum(-1)
        new_energy = 0.5 * (output[:, 19:] * _mv(mass, output[:, 19:])).sum(-1)
        impact_valid = (
            landed.all(-1)
            & approaching.all(-1)
            & (velocity_residual <= 1e-7)
            & (new_energy <= old_energy + 1e-7)
        )
        feasible = (
            valid
            & live
            & ~self.failed
            & qp_valid
            & model.manifold.is_valid(output)
            & (violation <= 1e-7)
        )
        feasible &= ~impact | impact_valid
        feasible &= (action.abs() <= model.torque_limits + 1e-7).all(-1)
        for x in (safe, output):
            feasible &= (
                (x[:, 7:19] >= model.lower_limits) & (x[:, 7:19] <= model.upper_limits)
            ).all(-1)
            feasible &= (model.foot_positions(x[:, :19])[..., 2] >= -0.002).all(-1)
        self.failed |= live & ~feasible
        self.indices += feasible.to(torch.int64)
        action = torch.where(feasible[:, None], action, float("nan"))
        status = torch.where(
            feasible,
            torch.where(qp_converged, SolveStatus.SUCCESS, SolveStatus.MAX_ITERATIONS),
            torch.where(valid, SolveStatus.NO_FEASIBLE_CANDIDATE, SolveStatus.NUMERICAL_FAILURE),
        )
        xs = torch.stack((state, output), 1)
        xs = torch.where(feasible[:, None, None], xs, float("nan"))
        result = MPCResult(
            xs,
            action[:, None],
            torch.where(feasible, error.square().sum(-1), float("nan")),
            status,
            iterations,
            feasible,
        )
        return Go2GaitControl(
            action,
            result,
            contacts.clone(),
            durations.clone(),
            nodes,
            self.indices >= len(self.plan.torques),
        )

    def reset(self, mask=None):
        if mask is None:
            mask = torch.ones(self.batch_size, device=self.device, dtype=torch.bool)
        if (
            mask.shape != (self.batch_size,)
            or mask.dtype != torch.bool
            or mask.device != self.device
        ):
            raise ValueError("reset mask must be bool [batch] on controller device")
        self.indices.masked_fill_(mask, 0)
        self.failed.masked_fill_(mask, False)
        self._warm.copy_(torch.where(mask[:, None], self.plan.torques[0], self._warm))
        self._dual.copy_(torch.where(mask[:, None], 1.0, self._dual))
