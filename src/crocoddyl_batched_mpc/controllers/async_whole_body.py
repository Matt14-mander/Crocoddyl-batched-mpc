"""Spawned W planner and nonblocking wall-clock Guidance receiver.

No native model is shared between processes. Packet time starts at the measured
state timestamp, not at delivery. Only receiver-acknowledged candidates seed
later solves. One in-flight request bounds queue age and memory.
"""

import multiprocessing as mp
import os
import queue
import threading
import time
import uuid
from dataclasses import dataclass, replace

import numpy as np

from ..guidance import Guidance, GuidanceBuffer, _array, stationary
from ..models.go2_arm import load_go2_arm
from ..planners.standing_whole_body import StandingWholeBodyPlanner
from ..references.standing_arm import guidance_from_whole_body, model_identity
from .rolling_whole_body_cpu import common_seed, preview_window, shifted_seed


@dataclass(frozen=True)
class AsyncSettings:
    replan_period: float = 0.5
    handoff_duration: float = 0.4
    return_duration: float = 1.0
    recovery_duration: float = 0.8
    result_timeout: float = 0.6
    max_state_age: float = 0.65
    hard_timeout: float = 2.0

    def __post_init__(self):
        if any(not np.isfinite(v) or v <= 0 for v in vars(self).values()):
            raise ValueError("asynchronous durations must be finite and positive")
        if self.hard_timeout <= self.result_timeout:
            raise ValueError("hard_timeout must exceed result_timeout")


@dataclass(frozen=True)
class StateSnapshot:
    time: float
    state: np.ndarray
    previous_torque: np.ndarray
    outgoing: object
    recovering: bool

    def __post_init__(self):
        if not np.isfinite(self.time) or abs(self.outgoing.time - self.time) > 1e-8:
            raise ValueError("snapshot and outgoing reference require the same finite timestamp")
        object.__setattr__(self, "state", _array(self.state, (49,), "measured state"))
        object.__setattr__(
            self, "previous_torque", _array(self.previous_torque, (18,), "applied torque")
        )


@dataclass(frozen=True)
class PlanRequest:
    epoch: str
    sequence: int
    sent_at: float
    clock_zero: float
    origin: float
    acknowledged_sequence: int
    snapshot: StateSnapshot


@dataclass(frozen=True)
class WorkerFault:
    """Explicit acceptance-test fault, never enabled by default."""

    delay: float = 0.0
    fail: bool = False

    def __post_init__(self):
        if not np.isfinite(self.delay) or self.delay < 0 or type(self.fail) is not bool:
            raise ValueError("invalid worker test fault")


class WWorkerEngine:
    def __init__(self, config, common, planner_settings, settings):
        self.robot = load_go2_arm(config)
        if model_identity(self.robot) != common.guidance.samples[0].model_id:
            raise ValueError("worker model differs from task")
        self.common, self.settings = common, settings
        self.planner = StandingWholeBodyPlanner(
            self.robot, planner_settings, common.nominal_states[0]
        )
        self.cached = None
        self.pending = None

    def solve(self, request):
        # A candidate produced in this worker is not automatically an accepted
        # packet. A rejected/late candidate must never poison the warm cache.
        if self.pending is not None:
            if self.pending[0] == request.acknowledged_sequence:
                self.cached = self.pending
            self.pending = None
        timing_start = time.perf_counter()
        snap, now = request.snapshot, request.snapshot.time
        recovery = self.cached[3] if self.cached is not None else None
        if snap.recovering:
            end = now + self.settings.recovery_duration
            task = self.common.guidance
            target = (
                task.sample(end)
                if end <= task.valid_until
                else stationary(task.samples[-1], time=end)
            )
            recovery = Guidance(
                np.array([now, end]),
                (
                    replace(snap.outgoing, source="async_recovery"),
                    replace(target, source="async_recovery"),
                ),
                now,
                end,
            )
        window = preview_window(
            self.robot, self.common, now, self.planner.settings, recovery=recovery
        )
        preview_done = time.perf_counter()
        seed = None
        if self.cached is not None:
            seed = shifted_seed(
                self.robot, self.cached[2], self.cached[1], now, snap.state, self.planner.settings
            )
        seed_kind = "shifted_acknowledged" if seed is not None else "common_trajectory"
        if seed is None:
            seed = common_seed(self.planner, window, snap.state)
        seed_done = time.perf_counter()
        plan = self.planner.solve(
            window.for_whole_body(),
            snap.previous_torque,
            initial_guess=seed,
            initial_state=snap.state,
        )
        solve_done = time.perf_counter()
        packet = guidance_from_whole_body(
            self.robot, plan, window, sequence=request.sequence, initial_state=snap.state
        )
        self.pending = (request.sequence, now, plan, recovery)
        return packet, dict(
            phase_timings=dict(
                preview_s=preview_done - timing_start,
                seed_s=seed_done - preview_done,
                solve_and_accept_s=solve_done - seed_done,
                guidance_s=time.perf_counter() - solve_done,
            ),
            converged=plan.converged,
            seed=seed_kind,
            cached_sequence=None if self.cached is None else self.cached[0],
            recovery=bool(snap.recovering),
            iterations=int(plan.mpc.iterations[0]),
            solve_time_s=plan.solve_time_s,
            max_dynamics_gap=plan.diagnostics["max_dynamics_gap"],
            initial_state_error=float(np.max(np.abs(plan.mpc.xs[0, 0].numpy() - snap.state))),
            torque_memory_error=float(
                np.max(
                    np.abs(
                        plan.native_states[0, 49:]
                        - self.planner._native_control(snap.previous_torque)
                    )
                )
            ),
        )


def _worker_main(inbox, outbox, config, common, planner_settings, settings, epoch, faults):
    try:
        engine = WWorkerEngine(config, common, planner_settings, settings)
        outbox.put(
            dict(kind="ready", epoch=epoch, pid=os.getpid(), model_id=model_identity(engine.robot))
        )
    except Exception as exc:
        outbox.put(dict(kind="startup_error", epoch=epoch, reason=f"{type(exc).__name__}: {exc}"))
        return
    while True:
        request = inbox.get()
        if request is None:
            return

        def wall():
            return request.origin + time.monotonic() - request.clock_zero

        result = dict(
            kind="result",
            epoch=epoch,
            sequence=request.sequence,
            state_time=request.snapshot.time,
            sent_at=request.sent_at,
            started_at=wall(),
            packet=None,
            diagnostics={},
        )
        try:
            fault = faults.get(request.sequence, WorkerFault())
            if fault.delay:
                time.sleep(fault.delay)
            if fault.fail:
                raise RuntimeError("injected worker failure")
            result["packet"], result["diagnostics"] = engine.solve(request)
        except Exception as exc:
            result["error"] = f"{type(exc).__name__}: {exc}"
        result["completed_at"] = wall()
        outbox.put(result)


class SpawnWWorker:
    """Bounded IPC, nonblocking polling/submission; blocking cleanup is outside control."""

    def __init__(self, config, common, planner_settings, settings, *, faults=None):
        self.args = config, common, planner_settings, settings, dict(faults or {})
        self.process = None
        self.ready = False
        self.retiring = False
        self._starting = False
        self._closing = False
        self._launch_error = None
        self.start()

    def start(self):
        if self._starting or (self.process is not None and self.process.is_alive()):
            raise RuntimeError("cannot replace a live or starting worker")
        self.ready, self.retiring = False, False
        self.epoch = uuid.uuid4().hex
        self._starting = True
        self._launch_error = None
        self._supervisor = threading.Thread(
            target=self._bootstrap, daemon=True, name="go2-z1-w-supervisor"
        )
        self._supervisor.start()

    def _bootstrap(self):
        # spawn's argument pipe can block until the child imports its modules.
        # Never perform that operation in the control thread, including restarts.
        try:
            self._spawn()
        except Exception as exc:
            self._launch_error = dict(
                kind="startup_error", epoch=self.epoch, reason=f"{type(exc).__name__}: {exc}"
            )
        finally:
            if self._closing and self.process is not None and self.process.is_alive():
                self.process.terminate()
            self._starting = False

    def _spawn(self):
        if self.process is not None:
            if self.process.is_alive():
                raise RuntimeError("cannot replace a live worker")
            self._release()
        context = mp.get_context("spawn")
        self.inbox, self.outbox = context.Queue(maxsize=1), context.Queue(maxsize=2)
        config, common, ps, settings, faults = self.args
        self.process = context.Process(
            target=_worker_main,
            args=(self.inbox, self.outbox, config, common, ps, settings, self.epoch, faults),
            name="go2-z1-w-planner",
            daemon=True,
        )
        self.ready, self.retiring = False, False
        self.process.start()

    def poll(self):
        if self._starting:
            return []
        if self._launch_error is not None:
            error, self._launch_error = self._launch_error, None
            return [error]
        result = []
        for _ in range(2):
            try:
                message = self.outbox.get_nowait()
            except queue.Empty:
                break
            if message["epoch"] == self.epoch and message["kind"] == "ready":
                self.ready = True
            result.append(message)
        return result

    def submit(self, request):
        if not self.ready or self.retiring or not self.process.is_alive():
            return False
        try:
            self.inbox.put_nowait(request)
        except queue.Full:
            return False
        return True

    def terminate(self):
        self.ready, self.retiring = False, True
        if self.process.is_alive():
            self.process.terminate()

    def alive(self):
        return self._starting or (self.process is not None and self.process.is_alive())

    def _release(self):
        self.process.join(timeout=0)
        self.process.close()
        for q in (self.inbox, self.outbox):
            q.cancel_join_thread()
            q.close()
        self.process = None

    def close(self):
        self._closing = True
        self._supervisor.join(timeout=15.0)
        if self._supervisor.is_alive():
            raise RuntimeError("worker supervisor did not finish shutdown")
        if self.process is None:
            return
        self.terminate()
        self.process.join(timeout=0.5)
        if self.process.is_alive():
            self.process.kill()
            self.process.join(timeout=0.5)
        if self.process.is_alive():
            raise RuntimeError("worker did not terminate")
        self._release()


class AsyncWReceiver:
    """All buffer mutations occur in the control process, never in the worker."""

    def __init__(self, common, worker, *, clock, clock_zero, settings=None, horizon=1.2):
        self.common, self.worker = common, worker
        self.settings = AsyncSettings() if settings is None else settings
        self.clock, self.clock_zero = clock, clock_zero
        if not np.isfinite(horizon) or horizon <= self.settings.handoff_duration:
            raise ValueError("horizon must exceed handoff duration")
        self.horizon = horizon
        self.origin = float(common.guidance.times[0])
        self.buffer = GuidanceBuffer(
            common.guidance.samples[0],
            self.settings.return_duration,
            handoff_duration=self.settings.handoff_duration,
        )
        self.pending = None
        self.sequence = 0
        self.last_accepted = (None, 0)
        self.next_due = self.origin
        self.last_time = -np.inf
        self.events = []
        self.requests = []

    def _now(self):
        now = float(self.clock())
        if not np.isfinite(now) or now < self.last_time:
            raise ValueError("wall clock must be finite and monotonic")
        self.last_time = now
        return now

    def receive(self, result, now):
        event = dict(
            kind="delivery",
            at=now,
            epoch=result.get("epoch"),
            sequence=result.get("sequence"),
            accepted=False,
        )
        reason = None
        request = self.pending["request"] if self.pending else None
        if result.get("epoch") != self.worker.epoch:
            reason = "old_epoch"
        elif request is None or result.get("sequence") != request.sequence:
            reason = "unexpected_sequence"
        else:
            times = [
                request.snapshot.time,
                request.sent_at,
                result.get("started_at", np.nan),
                result.get("completed_at", np.nan),
                now,
            ]
            event.update(
                age_s=now - request.snapshot.time,
                completed_at=result.get("completed_at"),
                sent_at=request.sent_at,
                diagnostics=result.get("diagnostics", {}),
            )
            if (
                not np.isfinite(times).all()
                or np.any(np.diff(times) < -1e-7)
                or result.get("state_time") != request.snapshot.time
                or result.get("sent_at") != request.sent_at
            ):
                reason = "timestamp_contract"
            elif self.pending["timed_out"] or now > request.sent_at + self.settings.result_timeout:
                reason = "timeout"
            elif now - request.snapshot.time > self.settings.max_state_age:
                reason = "state_age"
            elif result.get("error"):
                reason = "worker_error"
                event["error"] = result["error"]
            else:
                packet = result.get("packet")
                if (
                    not isinstance(packet, Guidance)
                    or packet.sequence != request.sequence
                    or packet.issued_at != request.snapshot.time
                    or packet.times[0] != request.snapshot.time
                    or packet.valid_until > request.snapshot.time + self.horizon + 1e-9
                ):
                    reason = "packet_contract"
                else:
                    # Pickle does not re-run dataclass validation/read-only copying.
                    try:
                        packet = replace(packet, samples=tuple(replace(s) for s in packet.samples))
                        event["valid_until"] = float(packet.valid_until)
                        event["accepted"] = self.buffer.submit(packet, now)
                        reason = None if event["accepted"] else self.buffer.last_rejection
                    except (ValueError, TypeError):
                        reason = "packet_contract"
                    if event["accepted"]:
                        self.last_accepted = (self.worker.epoch, packet.sequence)
            self.pending = None
        event["reason"] = reason
        self.events.append(event)
        return event["accepted"]

    def wants_snapshot(self, now):
        return (
            self.pending is None
            and self.worker.ready
            and now >= self.next_due
            and now < self.common.guidance.times[-1]
        )

    def tick(self, snapshot=None):
        for result in self.worker.poll():
            now = self._now()  # Includes deserialization/delivery time.
            if result["kind"] == "result":
                self.receive(result, now)
            else:
                self.events.append({**result, "at": now})
        now = self._now()
        if self.pending:
            age = now - self.pending["request"].sent_at
            if age > self.settings.result_timeout and not self.pending["timed_out"]:
                self.pending["timed_out"] = True
                self.events.append(
                    dict(kind="timeout", at=now, sequence=self.pending["request"].sequence)
                )
            if age > self.settings.hard_timeout:
                self.events.append(
                    dict(
                        kind="worker_terminated", at=now, sequence=self.pending["request"].sequence
                    )
                )
                self.worker.terminate()
                self.pending = None
        if not self.worker.alive():
            self.events.append(
                dict(kind="worker_restart", at=now, abandoned=self.pending is not None)
            )
            self.pending = None
            self.worker.start()
            return
        if (
            snapshot is None
            or self.pending is not None
            or not self.worker.ready
            or now < self.next_due
            or now >= self.common.guidance.times[-1]
        ):
            return
        if snapshot.time > now or now - snapshot.time > self.settings.max_state_age:
            self.events.append(dict(kind="snapshot_rejected", at=now, reason="state_age"))
            return
        sequence = self.sequence + 1
        acknowledged = self.last_accepted[1] if self.last_accepted[0] == self.worker.epoch else 0
        request = PlanRequest(
            self.worker.epoch, sequence, now, self.clock_zero, self.origin, acknowledged, snapshot
        )
        if self.worker.submit(request):
            self.sequence = sequence
            self.pending = dict(request=request, timed_out=False)
            self.requests.append(request)
            self.next_due = now + self.settings.replan_period
