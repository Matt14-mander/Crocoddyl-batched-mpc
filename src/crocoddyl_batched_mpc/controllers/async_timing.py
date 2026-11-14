"""Explicit wall-clock fault injection and deterministic timing for CPU experiments."""

import time
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ControlDelay:
    at: float
    duration: float
    phase: str = "before_sample"

    def __post_init__(self):
        if not np.isfinite([self.at, self.duration]).all() or self.at < 0 or self.duration <= 0:
            raise ValueError("delay time must be nonnegative and duration positive")
        if self.phase not in ("before_sample", "after_solve", "wake"):
            raise ValueError("unknown control delay phase")


class ExecutionClock:
    """Actual time by default. Virtual time is explicitly labelled, never a speed claim.

    Virtual phase durations model CPU work independently of host load. Native W,
    WBC and contact dynamics still execute; only their execution time is prescribed.
    """

    def __init__(self, *, delays=(), virtual=False, phase_costs=None):
        self.virtual = virtual
        self.value = 0.0
        self.delays = tuple(delays)
        if any(not isinstance(d, ControlDelay) for d in self.delays):
            raise ValueError("delays must be ControlDelay instances")
        self.costs = dict(phase_costs or {"before_sample": 0.002, "after_solve": 0.008})
        if any(not np.isfinite(x) or x < 0 for x in self.costs.values()):
            raise ValueError("phase costs must be nonnegative and finite")
        self.events = []
        self.used = set()
        self.zero = None

    def now(self):
        return self.value if self.virtual else time.monotonic()

    def wait(self, duration):
        if duration > 0:
            if self.virtual:
                self.value += duration
            else:
                time.sleep(duration)

    def begin(self):
        self.zero = self.now()

    def phase(self, name):
        if self.virtual:
            self.wait(self.costs.get(name, 0.0))
        for i, delay in enumerate(self.delays):
            if i in self.used or delay.phase != name or self.now() - self.zero < delay.at:
                continue
            started = self.now() - self.zero
            self.wait(delay.duration)
            self.used.add(i)
            self.events.append(
                dict(
                    index=i,
                    phase=name,
                    scheduled_at=delay.at,
                    started_at=started,
                    duration_s=self.now() - self.zero - started,
                )
            )


class ScheduledWWorker:
    """Deterministic test transport with real native solves and prescribed latency.

    This is not a background process and must never be used for wall-clock or
    performance acceptance. It exercises the same receiver and CPU control loop.
    """

    def __init__(
        self, config, common, planner_settings, settings, *, execution, latency=0.25, faults=None
    ):
        if not execution.virtual or not np.isfinite(latency) or latency <= 0:
            raise ValueError("scheduled worker requires virtual time and positive latency")
        self.args = config, common, planner_settings, settings
        self.execution, self.latency = execution, latency
        self.faults = dict(faults or {})
        self.generation = 0
        self.process = object()
        self.start()

    def start(self):
        from .async_whole_body import WWorkerEngine

        self.engine = WWorkerEngine(*self.args)
        self.generation += 1
        self.epoch = f"scheduled-{self.generation}"
        self.ready = True
        self.running = True
        self.pending = None

    def alive(self):
        return self.running

    def submit(self, request):
        if not self.ready or self.pending is not None:
            return False
        fault = self.faults.get(request.sequence)
        duration = self.latency + (fault.delay if fault else 0.0)
        result = dict(
            kind="result",
            epoch=self.epoch,
            sequence=request.sequence,
            state_time=request.snapshot.time,
            sent_at=request.sent_at,
            started_at=request.sent_at,
            completed_at=request.sent_at + duration,
            packet=None,
            diagnostics={},
        )
        try:
            if fault and fault.fail:
                raise RuntimeError("injected worker failure")
            result["packet"], result["diagnostics"] = self.engine.solve(request)
        except Exception as exc:
            result["error"] = f"{type(exc).__name__}: {exc}"
        self.pending = (request, result)
        return True

    def poll(self):
        if self.pending is None:
            return []
        request, result = self.pending
        now = request.origin + self.execution.now() - request.clock_zero
        if now < result["completed_at"]:
            return []
        self.pending = None
        return [result]

    def terminate(self):
        self.running, self.ready, self.pending = False, False, None

    def close(self):
        self.terminate()
        self.process = None
