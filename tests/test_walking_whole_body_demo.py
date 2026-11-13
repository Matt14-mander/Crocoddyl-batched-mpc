"""Hybrid dynamics and its independent replay, without changing legacy Go2 tests."""

import numpy as np
import pytest

pytest.importorskip("pinocchio")
pytest.importorskip("crocoddyl")

from crocoddyl_batched_mpc.planners.walking_whole_body_demo import (  # noqa: E402
    WalkingDemoSettings,
    WalkingWholeBodyDemo,
)


@pytest.fixture(scope="module")
def task():
    planner = WalkingWholeBodyDemo(config=WalkingDemoSettings(cycles=1))
    problem, xs, us = planner.build_walking_problem()
    return planner, problem, xs, us


@pytest.mark.parametrize(
    "kwargs",
    [
        {"cycles": 0},
        {"cycles": True},
        {"dt": np.nan},
        {"dt": 0.03},
        {"step_length": -1},
        {"iterations": 0},
    ],
)
def test_invalid_settings(kwargs):
    with pytest.raises(ValueError):
        WalkingDemoSettings(**kwargs)


def test_schedule_and_world_reference(task):
    p, _, _, _ = task
    impacts = [r for r in p.schedule if r["kind"] == "impact"]
    assert [r["swing"] for r in impacts] == [3, 1, 2, 0]
    assert all(r["dt"] == 0 and r["contacts"].sum() == 4 for r in impacts)
    assert all(
        r["contacts"].sum() == 3 for r in p.schedule if r["kind"] == "flow" and r["swing"] >= 0
    )
    np.testing.assert_allclose(
        p.schedule[-1]["anchors"] - p.initial_anchors, np.tile([p.config.step_length, 0, 0], (4, 1))
    )
    poses = p.reference.tcp_poses
    np.testing.assert_allclose(poses[:, 0, 3], poses[0, 0, 3])
    np.testing.assert_allclose(poses[-1], poses[0], atol=1e-14)
    assert np.ptp(poses[:, 1, 3]) > 1.9 * p.config.curve_y
    assert np.ptp(poses[:, 2, 3]) > 1.9 * p.config.curve_z
    # Vertically stacked lobes: horizontal offset is near zero at maximum height.
    top = np.argmax(poses[:, 2, 3])
    assert abs(poses[top, 1, 3] - poses[0, 1, 3]) < 0.01
    assert max(p.ik_errors) < 1e-6
    x = p._native_state(p.initial_state)
    np.testing.assert_array_equal(p._canonical_state(x), p.initial_state)
    u = np.arange(18, dtype=float)
    np.testing.assert_array_equal(p._canonical_control(p._native_control(u)), u)


@pytest.mark.parametrize("kind", ["flow", "impact"])
def test_hybrid_action_matches_independent_kkt(task, kind):
    p, problem, xs, us = task
    k = next(k for k, r in enumerate(p.schedule) if r["kind"] == kind and r["swing"] == 3)
    action = problem.runningModels[k]
    x, u = xs[k].copy(), us[k]
    x[25:49] = np.random.default_rng(8).normal(0, 0.003, 24)
    data = action.createData()
    action.calc(data, x, u)
    node = p.oracle(x, u, p.schedule[k])
    np.testing.assert_allclose(p.state.diff(data.xnext, node["xnext"]), 0, atol=1e-10)
    assert np.max(np.abs(node["dynamics_residual"])) < 1e-9
    assert np.max(np.abs(node["contact_residual"])) < 1e-9
    if kind == "impact":
        np.testing.assert_array_equal(data.xnext[:25], x[:25])
        np.testing.assert_array_equal(data.xnext[49:], x[49:])
        for leg, name in enumerate(("FL", "FR", "RL", "RR")):
            np.testing.assert_allclose(
                data.physical.multibody.impulses.impulses[name].f.linear,
                node["forces"][leg],
                atol=1e-9,
            )


def test_impulse_manifold_derivatives_and_memory(task):
    p, problem, xs, us = task
    k = next(k for k, row in enumerate(p.schedule) if row["kind"] == "impact")
    action, x = problem.runningModels[k], xs[k].copy()
    x[25:49] = np.random.default_rng(6).normal(0, 0.002, 24)
    data = action.createData()
    action.calc(data, x, us[k])
    action.calcDiff(data, x, us[k])
    epsilon = 2e-7
    for j in range(66):
        delta = np.eye(66)[j] * epsilon
        a, b = action.createData(), action.createData()
        action.calc(a, p.state.integrate(x, delta), us[k])
        action.calc(b, p.state.integrate(x, -delta), us[k])
        finite = p.state.diff(b.xnext, a.xnext) / (2 * epsilon)
        np.testing.assert_allclose(data.Fx[:, j], finite, atol=2e-6, rtol=2e-6)
    np.testing.assert_array_equal(data.Fx[48:, 48:], np.eye(18))
    np.testing.assert_array_equal(data.Lx[48:], np.zeros(18))


def test_audit_checks_terminal_initial_and_nonfinite(task):
    p, _, xs, us = task
    bad = [x.copy() for x in xs]
    bad[-1][7] = p.robot.model.upperPositionLimit[7] + 0.1
    metrics, records = p.audit(bad, us)
    assert len(records) == len(xs)
    assert metrics["joint"] >= 0.099
    assert "joint" in p.acceptance(metrics)["failures"]
    bad[0][0] += 0.01
    assert p.audit(bad, us)[0]["initial_gap"] >= 0.0099
    bad[-1][0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        p.audit(bad, us)
