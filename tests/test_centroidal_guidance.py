"""Measured C0 mapping, common Guidance, ACK warm starts and independent export."""

import copy

import numpy as np
import pytest

from crocoddyl_batched_mpc.controllers.async_whole_body import (
    AsyncSettings,
    PlanRequest,
    StateSnapshot,
)
from crocoddyl_batched_mpc.controllers.centroidal_cpu import C0WorkerEngine
from crocoddyl_batched_mpc.controllers.rolling_whole_body_cpu import preview_window
from crocoddyl_batched_mpc.controllers.whole_body_qp import WBCSettings, WholeBodyQP
from crocoddyl_batched_mpc.models.go2_arm import load_go2_arm
from crocoddyl_batched_mpc.planners.standing_centroidal import CentroidalSettings
from crocoddyl_batched_mpc.references.centroidal import (
    centroidal_measurement,
    guidance_from_centroidal,
)
from crocoddyl_batched_mpc.references.standing_arm import StandingArmReferenceGenerator, reach_task

pytest.importorskip("pinocchio")


@pytest.fixture(scope="module")
def task():
    robot = load_go2_arm()
    q = robot.standing_configuration()
    t = np.arange(41) * 0.05
    common = StandingArmReferenceGenerator(robot).generate(
        q, t, *reach_task(t, robot.evaluate(q).tcp_pose, [0.01, 0.01, 0])
    )
    wbc = WholeBodyQP(robot, common.guidance.samples[0].foot_anchors, WBCSettings(dt=0.02))
    result = wbc.solve(q, np.zeros(24), common.guidance.samples[0], np.zeros(18))
    assert result.success
    return robot, common, result.torques


def request(common, tau, seq=1, now=0.0, ack=0, recovering=False):
    state = common.nominal_states[int(round(now / 0.05))]
    return PlanRequest(
        "c0test",
        seq,
        now,
        0.0,
        0.0,
        ack,
        StateSnapshot(now, state, tau, common.guidance.sample(now), recovering),
    )


def test_measurement_momentum_world_order_and_current_state_only(task):
    import pinocchio as pin

    robot, common, _ = task
    x = common.nominal_states[8].copy()
    x[25:] += np.linspace(-0.01, 0.02, 24)
    measured = centroidal_measurement(robot, x)
    q, v = x[:25], x[25:]
    data = robot.model.createData()
    jac = pin.jacobianCenterOfMass(robot.model, data, robot.to_native_q(q))
    np.testing.assert_allclose(
        measured[3:6], robot.total_mass * jac @ robot.to_native_v(v), atol=1e-13
    )
    np.testing.assert_allclose(measured[:3], robot.evaluate(q).com, atol=1e-13)
    assert np.linalg.norm(measured[6:]) > 0


def test_c0_guidance_preserves_shared_task_and_uses_feedback(task):
    robot, common, tau = task
    e = C0WorkerEngine(robot.config, common, CentroidalSettings(nodes=16), AsyncSettings())
    req = request(common, tau)
    packet, diag = e.solve(req)
    window = preview_window(robot, common, 0.0, e.planner.settings)
    for got, expected in zip(packet.samples, window.guidance.samples):
        for prefix in ("tcp", "base", "joint"):
            suffixes = (
                ("position", "velocity", "acceleration")
                if prefix == "joint"
                else ("pose", "velocity", "acceleration")
            )
            for suffix in suffixes:
                np.testing.assert_array_equal(
                    getattr(got, prefix + "_" + suffix), getattr(expected, prefix + "_" + suffix)
                )
        assert got.source == "C0_guidance" and not got.com_position.flags.writeable
    np.testing.assert_array_equal(
        e.pending[2].mpc.xs[0, 0].numpy(), centroidal_measurement(robot, req.snapshot.state)
    )
    assert diag["initial_state_error"] == 0 and diag["cached_sequence"] is None
    assert diag["candidate"]["accepted_for_task"]
    assert e.planner.model.mass == robot.total_mass


def test_only_acknowledged_force_plan_seeds_next_solve_and_recovery(task):
    robot, common, tau = task
    e = C0WorkerEngine(robot.config, common, CentroidalSettings(nodes=16), AsyncSettings())
    e.solve(request(common, tau))
    _, d = e.solve(request(common, tau, seq=2, now=0.1, ack=0))
    assert d["cached_sequence"] is None and d["seed"] == "common_forces"
    _, d = e.solve(request(common, tau, seq=3, now=0.2, ack=2))
    assert d["cached_sequence"] == 2 and d["seed"] == "shifted_acknowledged"
    _, d = e.solve(request(common, tau, seq=4, now=1.1, ack=3, recovering=True))
    assert d["seed"] == "common_forces" and d["recovery"]


@pytest.mark.parametrize("damage", ["initial", "terminal", "force", "metadata", "gravity"])
def test_export_rechecks_candidate_and_episode(task, damage):
    robot, common, tau = task
    e = C0WorkerEngine(robot.config, common, CentroidalSettings(nodes=16), AsyncSettings())
    req = request(common, tau)
    e.solve(req)
    window = preview_window(robot, common, 0.0, e.planner.settings)
    plan = copy.deepcopy(e.pending[2])
    previous = np.array(e.history[-1]["diagnostics"]["previous_force_estimate"])
    if damage == "initial":
        plan.mpc.xs[0, 0, 0] += 0.01
    elif damage == "terminal":
        plan.mpc.xs[0, -1, 6] += 0.01
    elif damage == "force":
        plan.mpc.us[0, -1, 2] += 1
    elif damage == "metadata":
        plan.times = plan.times + 0.01
    elif damage == "gravity":
        e.robot.model.gravity.linear = np.array([0, 0, -8.0])
    with pytest.raises(ValueError):
        guidance_from_centroidal(
            e.robot, e.planner, plan, window, previous, initial_state=req.snapshot.state
        )
