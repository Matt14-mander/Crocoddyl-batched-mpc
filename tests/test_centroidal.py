"""C0 dynamics derivatives, SCP condensation, physical gates and solve status."""

import subprocess
import sys
from dataclasses import replace

import numpy as np
import pytest

from crocoddyl_batched_mpc.models.centroidal import CentroidalModel
from crocoddyl_batched_mpc.planners.standing_centroidal import (
    CentroidalReference,
    CentroidalSettings,
    StandingCentroidalPlanner,
)


@pytest.fixture
def problem():
    model = CentroidalModel(
        20.0, [0, 0, -9.81], [[0.2, 0.15, 0], [0.2, -0.15, 0], [-0.2, 0.15, 0], [-0.2, -0.15, 0]]
    )
    settings = CentroidalSettings(nodes=6)
    initial = np.r_[[0.0, 0.0, 0.4], np.zeros(6)]
    force = np.tile([0.0, 0.0, 49.05], (4, 1))
    planner = StandingCentroidalPlanner(model, initial[:3], settings)
    reference = CentroidalReference(
        np.arange(7) * settings.dt, np.tile(initial, (7, 1)), np.tile(force, (6, 1, 1))
    )
    return planner, initial, reference, force


def test_import_c0_does_not_import_native_stack():
    subprocess.run(
        [
            sys.executable,
            "-B",
            "-c",
            "import sys; import crocoddyl_batched_mpc.controllers.centroidal_cpu; "
            'assert "pinocchio" not in sys.modules and "crocoddyl" not in sys.modules',
        ],
        check=True,
    )


@pytest.mark.parametrize(
    "values",
    [
        {"mass": 0},
        {"mass": np.nan},
        {"gravity": [0, 0, np.inf]},
        {"anchors": np.zeros((3, 3))},
        {"anchors": [[0, 0, 0], [1, 0, 0.1], [0, 1, 0], [1, 1, 0]]},
    ],
)
def test_invalid_model(values, problem):
    model = problem[0].model
    with pytest.raises(ValueError):
        replace(model, **values)


@pytest.mark.parametrize(
    "values",
    [
        {"nodes": 0},
        {"nodes": True},
        {"dt": 0},
        {"friction": np.nan},
        {"trust_min": 0.1},
        {"height_min": 0.8},
        {"normal_min": 300},
        {"force_rate_limit": -1},
        {"com_translation_bound": (0.1, 0, 0.1)},
        {"qp_iterations": 0},
    ],
)
def test_invalid_settings(values):
    with pytest.raises(ValueError):
        CentroidalSettings(**values)


def test_full_mass_gravity_balance_translation_invariance_and_derivatives(problem):
    planner, x, _, force = problem
    model = planner.model
    np.testing.assert_allclose(model.derivative(x, force), 0, atol=1e-13)
    rng = np.random.default_rng(12)
    x = x + rng.normal(size=9) * 0.03
    force = force + rng.normal(size=(4, 3)) * 2
    A, B = model.linearize(x, force, 0.05)
    epsilon = 1e-5
    for k in range(9):
        delta = np.eye(9)[k] * epsilon
        finite = (model.step(x + delta, force, 0.05) - model.step(x - delta, force, 0.05)) / (
            2 * epsilon
        )
        np.testing.assert_allclose(A[:, k], finite, atol=2e-10, rtol=0)
    for k in range(12):
        delta = (np.eye(12)[k] * epsilon).reshape(4, 3)
        finite = (model.step(x, force + delta, 0.05) - model.step(x, force - delta, 0.05)) / (
            2 * epsilon
        )
        np.testing.assert_allclose(B[:, k], finite, atol=2e-10, rtol=0)
    shift = np.array([0.4, -0.2, 0.1])
    moved = replace(model, anchors=model.anchors + shift)
    shifted = x.copy()
    shifted[:3] += shift
    np.testing.assert_allclose(
        model.derivative(x, force), moved.derivative(shifted, force), atol=1e-13
    )


def test_condensed_affine_dynamics_and_quadratic_objective(problem):
    p, x, reference, previous = problem
    rng = np.random.default_rng(7)
    controls = reference.forces + rng.normal(size=reference.forces.shape) * 0.5
    states = p.model.rollout(x, controls, p.settings.dt)
    H, g, _, _, matrices, offsets = p._qp(x, states, controls, reference, previous, 0.02)
    np.testing.assert_allclose(
        np.einsum("ijk,k->ij", matrices, controls.ravel()) + offsets, states, atol=1e-13
    )
    vector = controls.ravel() + rng.normal(size=controls.size) * 0.1

    def obj(u):
        linear = np.einsum("ijk,k->ij", matrices, u) + offsets
        return p.objective(linear, u.reshape(controls.shape), reference, previous)

    for k in range(0, len(vector), 7):
        delta = np.eye(len(vector))[k] * 1e-5
        actual = (obj(vector + delta) - obj(vector - delta)) / (2e-5)
        assert abs(actual - (H @ vector + g)[k]) < 1e-7
    assert np.linalg.eigvalsh(H).min() > 0


def test_static_scp_support_balance_and_no_state_reset(problem):
    p, x, reference, previous = problem
    plan = p.solve(x, reference, previous)
    assert plan.converged and plan.mpc.usable[0] and plan.diagnostics["accepted_for_task"]
    np.testing.assert_allclose(plan.mpc.xs[0].numpy(), reference.states, atol=2e-8)
    np.testing.assert_allclose(
        plan.mpc.us[0].numpy().reshape(-1, 4, 3), reference.forces, atol=2e-7
    )
    perturbed = x.copy()
    perturbed[3] = 0.03
    second = p.solve(
        perturbed, reference, previous, initial_guess=plan.mpc.us[0].numpy().reshape(-1, 4, 3)
    )
    assert second.mpc.usable[0]
    np.testing.assert_array_equal(second.mpc.xs[0, 0].numpy(), perturbed)


@pytest.mark.parametrize(
    "damage,expected",
    [
        ("initial", "initial_error"),
        ("gap", "dynamics_gap"),
        ("terminal", "com_box"),
        ("cone", "friction"),
        ("normal", "normal_force"),
        ("rate", "force_rate"),
    ],
)
def test_full_nonlinear_gate_rejects_each_constraint(problem, damage, expected):
    p, x, ref, prev = problem
    states, forces = ref.states.copy(), ref.forces.copy()
    if damage == "initial":
        states[0, 0] += 0.001
    elif damage == "gap":
        states[2, 6] += 0.001
    elif damage == "terminal":
        states[-1, 2] = p.upper[2] + 0.01
    elif damage == "cone":
        forces[2, 0, :2] = [25, 25]
    elif damage == "normal":
        forces[-1, 0, 2] = 0
    elif damage == "rate":
        forces[1, 0, 2] += 25
    result = p.validate_candidate(x, states, forces, prev)
    assert not result["feasible"] and expected in result["failures"]


def test_infeasible_qp_has_no_usable_action(problem):
    old, x, ref, prev = problem
    p = StandingCentroidalPlanner(
        old.model, x[:3], replace(old.settings, normal_max=20.0, qp_iterations=20)
    )
    plan = p.solve(x, ref, prev)
    assert not plan.mpc.usable[0] and not plan.diagnostics["accepted_for_task"]
    assert plan.diagnostics["termination_reason"] == "qp_not_converged"
    np.testing.assert_array_equal(plan.mpc.action.numpy(), np.zeros((1, 12)))


def test_one_scp_iteration_does_not_accept_linearized_feasibility_alone(problem):
    old, x, ref, prev = problem
    states = ref.states.copy()
    states[:, 0] = np.linspace(0, 0.03, len(states))
    ref = replace(ref, states=states)
    p = StandingCentroidalPlanner(
        old.model, x[:3], replace(old.settings, max_iterations=1, linearization_tolerance=1e-14)
    )
    plan = p.solve(x, ref, prev)
    assert plan.diagnostics["history"][0]["qp_feasible"]
    assert plan.linearization_defect > p.settings.linearization_tolerance
    assert not plan.mpc.usable[0]


def test_infeasible_shifted_seed_is_restored_before_local_trust_steps(problem):
    p, initial, reference, previous = problem
    seed = reference.forces.copy()
    seed[:, :, 2] = 100.0
    outside = p.model.rollout(initial, seed, p.settings.dt)
    assert outside[-1, 2] > p.upper[2] + p.settings.trust_radius
    plan = p.solve(initial, reference, previous, initial_guess=seed)
    first = plan.diagnostics["history"][0]
    assert first["phase"] == "feasibility_restoration" and first["accepted_step"]
    assert first["trust_radius"] is None and first["restoration_audit"]["feasible"]
    assert plan.mpc.usable[0] and plan.converged
    assert plan.diagnostics["history"][1]["phase"] == "scp"
    assert plan.diagnostics["history"][1]["trust_radius"] == p.settings.trust_radius
    assert plan.diagnostics["metrics"]["com_box"] == 0.0
    assert plan.diagnostics["metrics"]["force_rate"] == 0.0
