"""Independent dynamics parity and repeatable latency-injection contracts."""

from dataclasses import replace
from functools import partial

import numpy as np
import pytest

from crocoddyl_batched_mpc.controllers.async_cpu import run_async_cpu_episode
from crocoddyl_batched_mpc.controllers.async_timing import (
    ControlDelay,
    ExecutionClock,
    ScheduledWWorker,
)
from crocoddyl_batched_mpc.controllers.async_whole_body import AsyncSettings, WorkerFault
from crocoddyl_batched_mpc.models.go2_arm import load_go2_arm
from crocoddyl_batched_mpc.models.go2_arm_async_cpu import AsyncCPUPlant
from crocoddyl_batched_mpc.models.go2_arm_cpu import FixedSupportCPU
from crocoddyl_batched_mpc.references.standing_arm import StandingArmReferenceGenerator, reach_task

pytest.importorskip("pinocchio")
pytest.importorskip("crocoddyl")


@pytest.fixture(scope="module")
def robot():
    return load_go2_arm()


def task_reference(robot, duration):
    q = robot.standing_configuration()
    t = np.arange(round(duration / 0.05) + 1) * 0.05
    return StandingArmReferenceGenerator(robot).generate(
        q, t, *reach_task(t, robot.evaluate(q).tcp_pose, [0.005, 0.01, 0], return_to_start=True)
    )


def test_cached_plant_matches_original_kkt_and_fk(robot):
    q = robot.standing_configuration()
    v = np.linspace(-0.003, 0.003, 24)
    anchors = robot.evaluate(q).foot_positions
    fast, oracle = AsyncCPUPlant(robot, anchors), FixedSupportCPU(robot, anchors)
    other = AsyncCPUPlant(robot, anchors)
    assert fast.data is not other.data and fast.observation_data is not fast.data
    data = fast.data
    for i in range(20):
        fast.dt = oracle.dt = [0.005, 0.002, 0.001][i % 3]
        tau = 0.4 * np.sin(np.arange(18) + i)
        wrench = np.array([1.0, -2.0, 0.5, 0.02, 0.04, -0.01])
        a = fast.step(q, v, tau, external_wrench=wrench)
        b = oracle.step(q, v, tau, external_wrench=wrench)
        for name in ("q", "v", "acceleration", "forces", "dynamics_residual", "contact_residual"):
            np.testing.assert_allclose(getattr(a, name), getattr(b, name), atol=2e-11, rtol=1e-10)
        pose, feet, velocity = fast.observe(a.q, a.v)
        expected = robot.evaluate(a.q, a.v)
        np.testing.assert_allclose(pose, expected.tcp_pose, atol=1e-12)
        np.testing.assert_allclose(feet, expected.foot_positions, atol=1e-12)
        np.testing.assert_allclose(velocity, expected.foot_jacobians @ a.v, atol=1e-12)
        assert fast.data is data
        q, v = a.q, a.v
    with pytest.raises(ValueError, match="out-of-range"):
        fast.step(q, v, robot.torque_limits * 2)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"at": -1, "duration": 0.1},
        {"at": 0, "duration": float("nan")},
        {"at": 0, "duration": 0},
        {"at": 0, "duration": 0.1, "phase": "unknown"},
    ],
)
def test_invalid_delay(kwargs):
    with pytest.raises(ValueError):
        ControlDelay(**kwargs)


def test_virtual_delays_repeat_and_never_sleep_on_host(monkeypatch):
    def forbidden(*args):
        raise AssertionError("virtual replay must not sleep")

    monkeypatch.setattr("time.sleep", forbidden)
    traces = []
    for _ in range(2):
        clock = ExecutionClock(
            virtual=True, delays=[ControlDelay(0.1, 0.15), ControlDelay(0.1, 0.11, "after_solve")]
        )
        clock.begin()
        clock.wait(0.1)
        clock.phase("before_sample")
        clock.phase("after_solve")
        clock.phase("before_sample")
        traces.append((clock.now(), clock.events))
    assert traces[0] == traces[1]
    assert len(traces[0][1]) == 2


def test_virtual_time_cannot_use_real_worker(robot):
    common = task_reference(robot, 0.2)
    with pytest.raises(ValueError, match="scheduled test worker"):
        run_async_cpu_episode(robot, common, execution=ExecutionClock(virtual=True))


def run_replay(robot, delays):
    common = task_reference(robot, 3.0)
    clock = ExecutionClock(
        virtual=True,
        delays=delays,
        phase_costs={"before_sample": 0.009, "after_solve": 0.018, "control_complete": 0.006},
    )
    return run_async_cpu_episode(
        robot,
        common,
        execution=clock,
        worker_factory=partial(ScheduledWWorker, execution=clock),
        settings=replace(AsyncSettings(), result_timeout=2.0, max_state_age=2.0, hard_timeout=2.5),
        faults={3: WorkerFault(delay=1.35)},
    )


def test_cpu_expiry_and_delayed_command_are_reproducible(robot):
    delays = [ControlDelay(1.65, 0.12), ControlDelay(1.65, 0.09, "after_solve")]
    a, x, _ = run_replay(robot, delays)
    b, y, _ = run_replay(robot, delays)
    assert a["passed"] and b["passed"], (a["failures"], b["failures"])
    assert a["timing_mode"] == "deterministic_virtual"
    assert a["injected_delays"] == b["injected_delays"] and len(a["injected_delays"]) == 2
    assert a["discarded_commands"] > 0
    assert any(e.get("reason") == "expired" for e in a["events"])
    assert a["modes"]["return_to_standing"] > 0 and a["accepted"] >= 3
    for k in x:
        np.testing.assert_array_equal(x[k], y[k])
    assert a["metrics"] == b["metrics"]
    times = x["times"]
    indices = np.searchsorted(times, x["controls"][:, 1])
    np.testing.assert_array_equal(x["wbc_states"], x["states"][indices])
    assert np.max(np.diff(times)) <= 0.005 + 1e-12


def test_excessive_pause_stops_without_truncating_elapsed_time(robot):
    report, trace, _ = run_replay(robot, [ControlDelay(0.5, 0.25)])
    assert not report["passed"] and "terminated" in report["failures"]
    assert "catch-up bound" in report["termination"]["reason"]
    assert report["worker_closed"]
    assert report["termination"]["time"] - report["termination"]["state_time"] > 0.2
    assert trace["times"][-1] == report["termination"]["state_time"]
