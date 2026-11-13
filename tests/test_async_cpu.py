"""Deterministic wall-clock/IPC policies and native acknowledged warm-start cache."""

import subprocess
import sys
import threading
from dataclasses import replace

import numpy as np
import pytest

from crocoddyl_batched_mpc.controllers.async_cpu import WallCPUSettings
from crocoddyl_batched_mpc.controllers.async_whole_body import (
    AsyncSettings,
    AsyncWReceiver,
    PlanRequest,
    SpawnWWorker,
    StateSnapshot,
    WWorkerEngine,
)
from crocoddyl_batched_mpc.controllers.rolling_whole_body_cpu import preview_window
from crocoddyl_batched_mpc.models.go2_arm import load_go2_arm
from crocoddyl_batched_mpc.planners.standing_whole_body import WholeBodySettings
from crocoddyl_batched_mpc.references.standing_arm import StandingArmReferenceGenerator, reach_task

pytest.importorskip("pinocchio")
pytest.importorskip("crocoddyl")


class Clock:
    value = 10.0

    def __call__(self):
        return self.value


class Transport:
    def __init__(self):
        self.ready, self.living = True, True
        self.epoch = "epoch-0"
        self.messages, self.jobs = [], []
        self.restarts = 0

    def poll(self):
        messages, self.messages = self.messages, []
        return messages

    def submit(self, request):
        self.jobs.append(request)
        return True

    def alive(self):
        return self.living

    def terminate(self):
        self.ready, self.living = False, False

    def start(self):
        self.restarts += 1
        self.epoch = f"epoch-{self.restarts}"
        self.ready, self.living = False, True


@pytest.fixture(scope="module")
def task():
    robot = load_go2_arm()
    q = robot.standing_configuration()
    t = np.linspace(10, 12, 41)
    common = StandingArmReferenceGenerator(robot).generate(
        q, t, *reach_task(t, robot.evaluate(q).tcp_pose, [0, 0, 0])
    )
    return robot, common


def setup(task, **settings):
    robot, common = task
    clock, transport = Clock(), Transport()
    receiver = AsyncWReceiver(
        common,
        transport,
        clock=clock,
        clock_zero=0.0,
        horizon=0.8,
        settings=AsyncSettings(**{"handoff_duration": 0.1, "replan_period": 0.3, **settings}),
    )
    return robot, common, clock, transport, receiver


def submit(common, clock, receiver):
    outgoing, mode = receiver.buffer.sample(clock())
    snap = StateSnapshot(
        clock(),
        common.nominal_states[0],
        np.zeros(18),
        outgoing,
        mode in ("standing", "return_to_standing"),
    )
    receiver.tick(snap)
    return receiver.requests[-1]


def result(robot, common, request, *, duration=0.1):
    packet = preview_window(
        robot, common, request.snapshot.time, WholeBodySettings(nodes=16)
    ).guidance
    return dict(
        kind="result",
        epoch=request.epoch,
        sequence=request.sequence,
        state_time=request.snapshot.time,
        sent_at=request.sent_at,
        started_at=request.sent_at + 0.001,
        completed_at=request.sent_at + duration,
        packet=replace(packet, sequence=request.sequence),
        diagnostics={},
    )


def test_async_imports_do_not_load_native_modules():
    subprocess.run(
        [
            sys.executable,
            "-B",
            "-c",
            "import sys; import crocoddyl_batched_mpc.controllers.async_cpu; "
            "assert 'crocoddyl' not in sys.modules and 'pinocchio' not in sys.modules",
        ],
        check=True,
    )


def test_spawn_bootstrap_never_runs_in_control_thread(task, monkeypatch):
    entered, release = threading.Event(), threading.Event()

    def blocked_start(self):
        entered.set()
        if not release.wait(timeout=2.0):
            raise RuntimeError("bootstrap was not released by control thread")

    monkeypatch.setattr(SpawnWWorker, "_spawn", blocked_start)
    robot, common = task
    worker = SpawnWWorker(robot.config, common, WholeBodySettings(nodes=16), AsyncSettings())
    try:
        assert entered.wait(timeout=1.0)
        assert worker._starting and worker.alive() and not worker.ready
        assert worker.poll() == []
        assert worker._supervisor.ident != threading.get_ident()
    finally:
        release.set()
        worker.close()
    assert worker.process is None and not worker._supervisor.is_alive()


def test_delivery_uses_original_clock_and_consumes_latency(task):
    robot, common, clock, transport, receiver = setup(task)
    request = submit(common, clock, receiver)
    transport.messages.append(result(robot, common, request))
    clock.value = 10.2
    receiver.tick()
    assert receiver.events[-1]["accepted"]
    assert receiver.events[-1]["age_s"] == pytest.approx(0.2)
    assert receiver.buffer.current.times[0] == 10.0
    assert receiver.buffer.current.valid_until == 10.8
    assert receiver.buffer.handoff.valid_until == pytest.approx(10.3)
    endpoint = receiver.buffer.sample(10.3)[0]
    np.testing.assert_allclose(endpoint.tcp_pose, receiver.buffer.current.sample(10.3).tcp_pose)
    clock.value = 10.31
    second = submit(common, clock, receiver)
    assert second.acknowledged_sequence == 1


@pytest.mark.parametrize("age,expected", [(0.6, None), (0.60001, "timeout")])
def test_deadline_uses_receipt_even_if_compute_finished_early(task, age, expected):
    robot, common, clock, transport, receiver = setup(task)
    request = submit(common, clock, receiver)
    transport.messages.append(result(robot, common, request, duration=0.05))
    clock.value = 10 + age
    receiver.tick()
    assert receiver.events[-1]["reason"] == expected
    assert receiver.events[-1]["accepted"] == (expected is None)


@pytest.mark.parametrize("age,expected", [(0.81, "expired"), (0.71, "handoff_coverage")])
def test_expired_and_short_coverage_are_not_retimed(task, age, expected):
    robot, common, clock, transport, receiver = setup(task, result_timeout=1.5, max_state_age=1.5)
    request = submit(common, clock, receiver)
    transport.messages.append(result(robot, common, request))
    clock.value = 10 + age
    receiver.tick()
    assert receiver.events[-1]["reason"] == expected
    assert receiver.buffer.current is None and receiver.last_accepted[1] == 0


def test_old_valid_packet_survives_error_and_timeout_then_expires(task):
    robot, common, clock, transport, receiver = setup(task)
    first = submit(common, clock, receiver)
    transport.messages.append(result(robot, common, first))
    clock.value = 10.1
    receiver.tick()
    old = receiver.buffer.current
    clock.value = 10.31
    second = submit(common, clock, receiver)
    bad = result(robot, common, second)
    bad.update(error="solver failed", packet=None)
    transport.messages.append(bad)
    clock.value = 10.45
    receiver.tick()
    assert receiver.events[-1]["reason"] == "worker_error" and receiver.buffer.current is old
    clock.value = 10.62
    third = submit(common, clock, receiver)
    assert third.acknowledged_sequence == 1
    clock.value = 10.7
    receiver.tick()
    assert receiver.buffer.sample(10.7)[1] == "active"
    clock.value = 11.23
    receiver.tick()
    assert receiver.pending["timed_out"] and receiver.buffer.current.valid_until == 10.8
    assert receiver.buffer.sample(11.23)[1] == "return_to_standing"
    transport.messages.append(result(robot, common, third))
    receiver.tick()
    assert receiver.events[-1]["reason"] == "timeout" and receiver.buffer.current is old
    clock.value = 11.81
    assert receiver.buffer.sample(clock())[1] == "standing"


@pytest.mark.parametrize(
    "change,expected",
    [
        ("epoch", "old_epoch"),
        ("sequence", "unexpected_sequence"),
        ("timestamp", "timestamp_contract"),
        ("retimed", "packet_contract"),
        ("extended", "packet_contract"),
        ("state_age", "state_age"),
        ("model", "model"),
        ("invalid", "invalid"),
    ],
)
def test_reject_wrong_generation_sequence_and_time_contract(task, change, expected):
    robot, common, clock, transport, receiver = setup(
        task, max_state_age=0.15 if change == "state_age" else 0.65
    )
    req = submit(common, clock, receiver)
    msg = result(robot, common, req)
    if change == "epoch":
        msg["epoch"] = "old-worker"
    elif change == "sequence":
        msg["sequence"] = 99
    elif change == "timestamp":
        msg["completed_at"] = np.nan
    elif change == "retimed":
        packet = msg["packet"]
        msg["packet"] = replace(
            packet,
            times=packet.times + 0.1,
            issued_at=10.1,
            valid_until=10.9,
            samples=tuple(replace(s, time=s.time + 0.1) for s in packet.samples),
        )
    elif change == "extended":
        packet = preview_window(robot, common, 10.0, WholeBodySettings(nodes=20)).guidance
        msg["packet"] = replace(packet, sequence=1)
    elif change == "model":
        msg["packet"] = replace(
            msg["packet"],
            samples=tuple(replace(s, model_id="wrong") for s in msg["packet"].samples),
        )
    elif change == "invalid":
        msg["packet"] = replace(msg["packet"], valid=False)
    transport.messages.append(msg)
    clock.value = 10.2
    receiver.tick()
    assert receiver.events[-1]["reason"] == expected
    assert receiver.buffer.current is None and receiver.last_accepted[1] == 0
    if change in ("epoch", "sequence"):
        assert receiver.pending is not None


def test_one_inflight_job_and_hard_timeout_restart_without_old_ack(task):
    robot, common, clock, transport, receiver = setup(task, result_timeout=0.2, hard_timeout=0.5)
    old = submit(common, clock, receiver)
    clock.value = 10.31
    submit(common, clock, receiver)
    assert len(transport.jobs) == 1 and receiver.pending["timed_out"]
    clock.value = 10.51
    receiver.tick()
    assert transport.restarts == 1 and receiver.pending is None
    transport.ready = True
    clock.value = 10.6
    fresh = submit(common, clock, receiver)
    assert fresh.epoch != old.epoch and fresh.sequence == 2 and fresh.acknowledged_sequence == 0
    transport.messages.append(result(robot, common, old))
    clock.value = 10.61
    receiver.tick()
    assert receiver.events[-1]["reason"] == "old_epoch"
    assert receiver.pending["request"] is fresh


def test_monotonic_time_and_snapshot_age(task):
    _, common, clock, _, receiver = setup(task)
    outgoing = receiver.buffer.sample(10.0)[0]
    snap = StateSnapshot(10.0, common.nominal_states[0], np.zeros(18), outgoing, True)
    clock.value = 10.7
    receiver.tick(snap)
    assert receiver.events[-1]["kind"] == "snapshot_rejected" and not receiver.requests
    clock.value = 10.6
    with pytest.raises(ValueError, match="monotonic"):
        receiver.tick()
    with pytest.raises(ValueError, match="timestamp"):
        replace(snap, time=11.0)


def test_native_worker_uses_only_acknowledged_plan_for_warm_start(task):
    robot, common = task
    engine = WWorkerEngine(robot.config, common, WholeBodySettings(nodes=16), AsyncSettings())
    diagnostics = []
    for sequence, ack in [(1, 0), (2, 0), (3, 2), (4, 2)]:
        t = 10 + (sequence - 1) * 0.1
        outgoing = common.guidance.sample(t)
        snap = StateSnapshot(t, common.nominal_states[0], np.zeros(18), outgoing, False)
        request = PlanRequest("test-epoch", sequence, t, 0.0, 10.0, ack, snap)
        packet, diag = engine.solve(request)
        assert packet.sequence == sequence and packet.issued_at == t
        assert diag["initial_state_error"] == 0 and diag["torque_memory_error"] == 0
        diagnostics.append(diag)
    assert diagnostics[1]["seed"] == "common_trajectory"  # Unacknowledged plan 1 discarded.
    assert (
        diagnostics[2]["seed"] == "shifted_acknowledged" and diagnostics[2]["cached_sequence"] == 2
    )
    assert diagnostics[3]["cached_sequence"] == 2  # Unacknowledged plan 3 not promoted.


@pytest.mark.parametrize(
    "kwargs", [{"result_timeout": 0}, {"max_state_age": np.nan}, {"hard_timeout": 0.1}]
)
def test_invalid_async_settings(kwargs):
    with pytest.raises(ValueError):
        AsyncSettings(**kwargs)


@pytest.mark.parametrize(
    "kwargs", [{"control_period": 0.03}, {"integration_step": 0.01}, {"max_catchup": -1}]
)
def test_invalid_wall_settings(kwargs):
    with pytest.raises(ValueError):
        WallCPUSettings(**kwargs)
