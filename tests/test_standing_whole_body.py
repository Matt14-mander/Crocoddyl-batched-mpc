"""Independent KKT, manifold/adjacent-control derivatives, and offline W gates."""

import copy
import json
import subprocess
import sys
from dataclasses import replace

import numpy as np
import pytest

from crocoddyl_batched_mpc.planners.standing_whole_body import (
    StandingWholeBodyPlanner,
    WholeBodySettings,
    smooth_tcp_reference,
)
from crocoddyl_batched_mpc.result import SolveStatus


@pytest.fixture(scope="module")
def planner():
    pytest.importorskip("pinocchio")
    pytest.importorskip("crocoddyl")
    return StandingWholeBodyPlanner()


@pytest.fixture(scope="module")
def moving_case(planner):
    reference = smooth_tcp_reference(planner.robot, planner.settings, (0.01, 0.02, 0))
    return reference, planner.solve(reference)


def test_import_does_not_load_native_stack():
    script = (
        "import sys; import crocoddyl_batched_mpc.planners.standing_whole_body; "
        "assert 'crocoddyl' not in sys.modules; assert 'pinocchio' not in sys.modules"
    )
    subprocess.run([sys.executable, "-B", "-c", script], check=True)


@pytest.mark.parametrize(
    "values",
    [
        {"nodes": 0},
        {"nodes": True},
        {"dt": 0},
        {"dt": np.nan},
        {"normal_min": 10, "normal_max": 9},
        {"friction": 0},
        {"base_translation_bound": (1, -1, 1)},
        {"torque_change_weight": -1},
    ],
)
def test_invalid_settings(values):
    with pytest.raises(ValueError):
        WholeBodySettings(**values)


def test_static_support_balances_weight_and_moment(planner):
    p = planner
    node = p._independent_node(p.x0_physical, p.static_control)
    q = p.initial_state[:25]
    com = p.robot.evaluate(q).com
    forces = node["forces"]
    np.testing.assert_allclose(
        forces.sum(axis=0), -p.robot.total_mass * p.robot.model.gravity.linear, atol=1e-10
    )
    np.testing.assert_allclose(np.cross(p.anchors - com, forces).sum(axis=0), 0, atol=1e-10)
    assert np.max(np.abs(node["acceleration"])) < 1e-9
    assert all(v == 0 for v in p.static_diagnostics["violations"].values())
    # Joint gravity alone is not the support-consistent torque.
    assert (
        np.linalg.norm(p.robot.evaluate(q).gravity[6:] - p._canonical_control(p.static_control)) > 1
    )


def test_product_manifold_derivatives(planner):
    c = pytest.importorskip("crocoddyl")
    p = planner
    rng = np.random.default_rng(4)
    x = np.r_[p.x0_physical, p.static_control]
    x = p.state.integrate(x, rng.normal(size=66) * 0.02)
    dx = rng.normal(size=66) * 0.015
    y = p.state.integrate(x, dx)
    np.testing.assert_allclose(p.state.diff(x, y), dx, atol=2e-14)
    epsilon = 1e-6
    ji = p.state.Jintegrate(x, dx)
    jd = p.state.Jdiff(x, y)
    for column in range(66):
        delta = np.eye(66)[column] * epsilon
        xp, xm = p.state.integrate(x, delta), p.state.integrate(x, -delta)
        yp, ym = p.state.integrate(y, delta), p.state.integrate(y, -delta)
        actual = [
            (
                p.state.diff(y, p.state.integrate(xp, dx))
                - p.state.diff(y, p.state.integrate(xm, dx))
            )
            / (2 * epsilon),
            (
                p.state.diff(y, p.state.integrate(x, dx + delta))
                - p.state.diff(y, p.state.integrate(x, dx - delta))
            )
            / (2 * epsilon),
            (p.state.diff(xp, y) - p.state.diff(xm, y)) / (2 * epsilon),
            (p.state.diff(x, yp) - p.state.diff(x, ym)) / (2 * epsilon),
        ]
        for matrix, difference in zip([*ji, *jd], actual):
            np.testing.assert_allclose(matrix[:, column], difference, atol=3e-9, rtol=0)
    transport = rng.normal(size=(66, 5))
    for component, index in ((c.Jcomponent.first, 0), (c.Jcomponent.second, 1)):
        np.testing.assert_allclose(
            p.state.JintegrateTransport(x, dx, transport.copy(), component), ji[index] @ transport
        )


def test_native_contact_and_augmented_action_derivatives(planner):
    p = planner
    rng = np.random.default_rng(7)
    problem = p.build_problem()
    action = problem.runningModels[0]
    x = p.state.integrate(problem.x0, rng.normal(size=66) * 0.002)
    u = p.static_control + rng.normal(size=18) * 0.5
    data = action.createData()
    action.calc(data, x, u)
    action.calcDiff(data, x, u)
    independent = p._independent_node(x, u)
    np.testing.assert_allclose(
        data.physical.differential.xout, independent["acceleration"], atol=2e-10, rtol=0
    )
    for index, name in enumerate(p.contact_names):
        native = data.physical.differential.multibody.contacts.contacts[name]
        np.testing.assert_allclose(native.f.linear, independent["forces"][index], atol=2e-10)
    epsilon = 1e-6
    max_errors = {"Fx": 0, "Fu": 0, "Lx": 0, "Lu": 0, "force_x": 0, "force_u": 0}

    def evaluate(state, control):
        result = action.createData()
        action.calc(result, state, control)
        force = result.physical.differential.multibody.contacts.contacts["FL"].f.linear.copy()
        return result.xnext.copy(), result.cost, force

    contact = data.physical.differential.multibody.contacts.contacts["FL"]
    for kind, count in (("x", 66), ("u", 18)):
        for column in range(count):
            delta = np.eye(count)[column] * epsilon
            if kind == "x":
                plus = evaluate(p.state.integrate(x, delta), u)
                minus = evaluate(p.state.integrate(x, -delta), u)
                fdyn, gradient = data.Fx[:, column], data.Lx[column]
                fforce = contact.df_dx[:3, column] if column < 48 else np.zeros(3)
            else:
                plus, minus = evaluate(x, u + delta), evaluate(x, u - delta)
                fdyn, gradient, fforce = (
                    data.Fu[:, column],
                    data.Lu[column],
                    contact.df_du[:3, column],
                )
            dynamic_fd = (
                p.state.diff(data.xnext, plus[0]) - p.state.diff(data.xnext, minus[0])
            ) / (2 * epsilon)
            gradient_fd = (plus[1] - minus[1]) / (2 * epsilon)
            force_fd = (plus[2] - minus[2]) / (2 * epsilon)
            max_errors["F" + kind] = max(
                max_errors["F" + kind], float(np.max(np.abs(fdyn - dynamic_fd)))
            )
            max_errors["L" + kind] = max(max_errors["L" + kind], float(abs(gradient - gradient_fd)))
            max_errors["force_" + kind] = max(
                max_errors["force_" + kind], float(np.max(np.abs(fforce - force_fd)))
            )
            np.testing.assert_allclose(fdyn, dynamic_fd, atol=5e-6, rtol=1e-6)
            np.testing.assert_allclose(gradient, gradient_fd, atol=2e-5, rtol=1e-5)
            np.testing.assert_allclose(fforce, force_fd, atol=2e-5, rtol=2e-6)
    print("derivative_errors", json.dumps(max_errors))


def test_true_adjacent_torque_cost_and_cross_derivatives(planner):
    p = planner
    problem = p.build_problem(previous_torque=np.ones(18))
    action = problem.runningModels[0]
    x, u = problem.x0.copy(), p.static_control + np.linspace(-0.02, 0.02, 18)
    data = action.createData()
    action.calc(data, x, u)
    action.calcDiff(data, x, u)
    scale = p.settings.dt * p.settings.torque_change_weight
    np.testing.assert_allclose(data.xnext[49:], u)
    np.testing.assert_allclose(
        data.cost - data.physical.cost, scale / 2 * np.sum((u - x[49:]) ** 2)
    )
    np.testing.assert_allclose(data.Lxu[48:], -scale * np.eye(18))
    np.testing.assert_allclose(data.Lxx[48:, 48:], scale * np.eye(18))
    np.testing.assert_allclose(data.Fu[48:], np.eye(18))
    # The second node's penalty uses the first optimized torque, not a frozen seed.
    next_data = problem.runningModels[1].createData()
    next_u = u + 0.02
    problem.runningModels[1].calc(next_data, data.xnext, next_u)
    np.testing.assert_allclose(
        next_data.cost - next_data.physical.cost,
        scale / 2 * np.sum((next_u - u) ** 2),
        atol=1e-12,
    )
    terminal = problem.terminalModel
    terminal_data = terminal.createData()
    terminal.calc(terminal_data, data.xnext)
    terminal.calcDiff(terminal_data, data.xnext)
    assert terminal_data.cost == terminal_data.physical.cost
    np.testing.assert_array_equal(terminal_data.Lx[48:], 0)


def test_active_force_barrier_cost_gradient(planner):
    p = copy.copy(planner)
    # Make the force barrier active without changing the dynamics or warm seed.
    p.settings = replace(planner.settings, normal_max=50)
    problem = p.build_problem()
    action, x, u = problem.runningModels[0], problem.x0.copy(), p.static_control.copy()
    data = action.createData()
    action.calc(data, x, u)
    action.calcDiff(data, x, u)
    assert data.cost > 1e4
    epsilon = 1e-6
    for kind, columns in (("x", range(48)), ("u", range(18))):
        for column in columns:
            costs = []
            for sign in (1, -1):
                delta = np.eye(66 if kind == "x" else 18)[column] * epsilon * sign
                state = p.state.integrate(x, delta) if kind == "x" else x
                control = u + delta if kind == "u" else u
                trial = action.createData()
                action.calc(trial, state, control)
                costs.append(trial.cost)
            derivative = data.Lx[column] if kind == "x" else data.Lu[column]
            np.testing.assert_allclose(
                derivative, (costs[0] - costs[1]) / (2 * epsilon), atol=0.03, rtol=2e-6
            )


def test_static_hold_solution(planner):
    result = planner.solve()
    assert result.converged and result.mpc.status.item() == SolveStatus.SUCCESS
    assert result.mpc.usable.item() and result.diagnostics["accepted_for_task"]
    assert result.diagnostics["tcp_position_peak_m"] < 1e-10
    np.testing.assert_allclose(
        result.mpc.xs[0].numpy(), np.tile(planner.initial_state, (11, 1)), atol=1e-10
    )
    assert result.forces_world.shape == (11, 4, 3)
    assert result.accelerations.shape == (11, 24)


def test_low_speed_task_and_independent_rollout(planner, moving_case):
    p = planner
    reference, result = moving_case
    assert result.converged and result.mpc.status.item() == SolveStatus.SUCCESS
    assert result.mpc.usable.item() and result.diagnostics["accepted_for_task"]
    assert result.diagnostics["tcp_position_rmse_m"] < 5e-4
    assert result.diagnostics["tcp_rotation_peak_rad"] < 2e-3
    assert result.diagnostics["max_dynamics_gap"] < 1e-9
    assert result.diagnostics["max_foot_position_error_m"] < 1e-5
    assert np.linalg.norm(result.native_controls[-1] - result.native_controls[0]) > 0.01
    # Reintegrate the complete horizon from x0 with independent KKT at each step.
    state = p.x0_physical.copy()
    for k, u in enumerate(result.native_controls):
        node = p._independent_node(state, u)
        a, dt = node["acceleration"], p.settings.dt
        state = p.physical_state.integrate(state, np.r_[dt * state[25:] + dt**2 * a, dt * a])
        np.testing.assert_allclose(state, result.native_states[k + 1, :49], atol=2e-11)
    np.testing.assert_allclose(reference.tcp_velocities[[0, -1]], 0, atol=1e-12)
    print(
        "moving_case",
        json.dumps(
            {
                k: result.diagnostics[k]
                for k in (
                    "tcp_position_rmse_m",
                    "tcp_rotation_peak_rad",
                    "max_dynamics_gap",
                    "max_foot_position_error_m",
                    "max_torque_fraction",
                )
            }
        ),
    )


@pytest.mark.parametrize(
    "fault,expected",
    [
        ("nan", "nonfinite"),
        ("quaternion", "quaternion"),
        ("memory", "dynamics_gap"),
        ("torque", "torque"),
        ("joint", "joint_position"),
        ("velocity", "velocity"),
        ("foot", "foot_position"),
        ("initial", "dynamics_gap"),
    ],
)
def test_bad_candidates_are_rejected(planner, moving_case, fault, expected):
    reference, result = moving_case
    xs, us = result.native_states.copy(), result.native_controls.copy()
    if fault == "nan":
        xs[2, 0] = np.nan
    elif fault == "quaternion":
        xs[2, 6] = 2
    elif fault == "memory":
        xs[2, -1] += 1
    elif fault == "torque":
        us[2, 0] = 2 * planner.torque_limits[0]
    elif fault == "joint":
        xs[-1, 7] = planner.robot.model.upperPositionLimit[7] + 0.1
    elif fault == "velocity":
        xs[-1, 31] = planner.robot.model.velocityLimit[6] + 0.1
    elif fault == "foot":
        xs[-1, 2] += 0.03
    else:
        xs[0, -1] += 0.1
    report, _, _ = planner.validate_candidate(planner.build_problem(reference), xs, us, reference)
    assert not report["feasible"] and expected in report["failures"]


def test_force_gate_checks_terminal_and_tracking_is_separate(planner, moving_case):
    reference, result = moving_case
    # Tighten acceptance only on a copy: force violation can coexist with zero shooting gaps.
    strict = copy.copy(planner)
    strict.settings = replace(planner.settings, normal_min=55)
    report, _, _ = strict.validate_candidate(
        planner.build_problem(reference), result.native_states, result.native_controls, reference
    )
    assert not report["feasible"] and "normal" in report["failures"]
    assert len(report["per_node_violations"]) == 11
    assert report["per_node_violations"][-1]["normal"] > 0
    strict.settings = replace(planner.settings, friction=0.0001)
    report, _, _ = strict.validate_candidate(
        planner.build_problem(reference), result.native_states, result.native_controls, reference
    )
    assert not report["feasible"] and "friction" in report["failures"]
    wrong_target = copy.deepcopy(reference)
    wrong_target.tcp_poses[:, 0, 3] += 1
    report, _, _ = planner.validate_candidate(
        planner.build_problem(reference), result.native_states, result.native_controls, wrong_target
    )
    assert report["feasible"] and not report["tracking_passed"]
    assert not report["accepted_for_task"]


def test_iteration_limit_is_not_reported_as_convergence(planner):
    p = StandingWholeBodyPlanner(planner.robot, replace(planner.settings, max_iterations=1))
    result = p.solve(smooth_tcp_reference(p.robot, p.settings, (0.01, 0.02, 0)))
    assert not result.converged
    assert result.mpc.iterations.item() == 1
    assert result.mpc.status.item() == SolveStatus.MAX_ITERATIONS
    assert result.mpc.feasible.item()


def test_numeric_failure_does_not_yield_usable_action(planner, monkeypatch):
    c = pytest.importorskip("crocoddyl")

    class FailingSolver:
        def __init__(self, problem):
            pass

        def solve(self, *args):
            raise RuntimeError("injected numerical failure")

    monkeypatch.setattr(c, "SolverBoxFDDP", FailingSolver)
    result = planner.solve()
    assert result.mpc.status.item() == SolveStatus.NUMERICAL_FAILURE
    assert not result.mpc.usable.item()
    assert np.isnan(result.mpc.xs.numpy()).all()
    assert np.isnan(result.mpc.us.numpy()).all()
    assert "injected" in result.diagnostics["solver_error"]


def test_converged_invalid_candidate_is_not_usable(planner, moving_case, monkeypatch):
    c = pytest.importorskip("crocoddyl")
    reference, plan = moving_case

    class InvalidSolver:
        def __init__(self, problem):
            self.xs, self.us = plan.native_states.copy(), plan.native_controls.copy()
            self.us[-1, 0] = 2 * planner.torque_limits[0]
            self.cost, self.iter = 1.0, 0

        def solve(self, *args):
            return True

    monkeypatch.setattr(c, "SolverBoxFDDP", InvalidSolver)
    result = planner.solve(reference)
    assert result.converged
    assert result.mpc.status.item() == SolveStatus.NO_FEASIBLE_CANDIDATE
    assert not result.mpc.usable.item()
    assert not result.diagnostics["accepted_for_task"]


def test_malformed_inputs_rejected(planner):
    reference = smooth_tcp_reference(planner.robot, planner.settings)
    bad = replace(reference, times=reference.times + 0.01)
    with pytest.raises(ValueError, match="times"):
        planner.build_problem(bad)
    bad = copy.deepcopy(reference)
    bad.tcp_poses[1, 0, 0] = 2
    with pytest.raises(ValueError, match=r"SE\(3\)"):
        planner.build_problem(bad)
    with pytest.raises(ValueError, match="previous_torque"):
        planner.build_problem(reference, np.ones(12))


def test_inadequate_static_seed_is_explicit(planner):
    with pytest.raises(ValueError, match="static seed"):
        StandingWholeBodyPlanner(planner.robot, replace(planner.settings, normal_min=60))
