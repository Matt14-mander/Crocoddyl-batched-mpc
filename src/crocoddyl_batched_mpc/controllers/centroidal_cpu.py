"""C0 rolling engine on the frozen Guidance/receiver/WBC CPU execution contract."""

import time
from dataclasses import replace

import numpy as np

from ..guidance import Guidance, stationary
from ..models.centroidal import CentroidalModel
from ..models.go2_arm import load_go2_arm
from ..models.go2_arm_async_cpu import AsyncCPUPlant
from ..planners.standing_centroidal import StandingCentroidalPlanner
from ..references.centroidal import (
    centroidal_measurement,
    centroidal_reference,
    guidance_from_centroidal,
)
from ..references.standing_arm import model_identity
from .async_timing import ScheduledWWorker
from .rolling_whole_body_cpu import preview_window


class C0WorkerEngine:
    """One mutable episode workspace; only acknowledged plans seed another solve."""

    def __init__(self, config, common, planner_settings, settings):
        self.robot = load_go2_arm(config)
        if model_identity(self.robot) != common.guidance.samples[0].model_id:
            raise ValueError("worker model differs from task")
        self.common, self.settings = common, settings
        anchors = common.guidance.samples[0].foot_anchors
        model = CentroidalModel(self.robot.total_mass, self.robot.model.gravity.linear, anchors)
        initial = centroidal_measurement(self.robot, common.nominal_states[0])
        self.planner = StandingCentroidalPlanner(model, initial[:3], planner_settings)
        # Current contact-force estimate from measured q,v and actual previous tau.
        # This is feedback adaptation, not future full-body feasibility supervision.
        self.estimator = AsyncCPUPlant(self.robot, anchors)
        self.cached = self.pending = None
        self.history = []
        self.last_plan = None

    def solve(self, request):
        if self.pending is not None:
            if self.pending[0] == request.acknowledged_sequence:
                self.cached = self.pending
            self.pending = None
        started = time.perf_counter()
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
        reference = centroidal_reference(self.robot, window)
        initial = centroidal_measurement(self.robot, snap.state)
        previous = self.estimator.step(
            snap.state[:25], snap.state[25:], snap.previous_torque
        ).forces
        preview_done = time.perf_counter()
        seed, seed_kind = None, "common_forces"
        if self.cached is not None:
            age = now - self.cached[1]
            old = self.cached[2]
            if 0 <= age < old.times[-1]:
                indices = np.clip(
                    np.searchsorted(old.times, age + reference.times[:-1], side="right") - 1,
                    0,
                    len(old.times) - 2,
                )
                seed = old.mpc.us[0].numpy()[indices].reshape(-1, 4, 3).copy()
                seed_kind = "shifted_acknowledged"
        plan = self.planner.solve(initial, reference, previous, initial_guess=seed)
        solve_done = time.perf_counter()
        self.last_plan = plan
        diagnostics = dict(
            backend="C0",
            converged=plan.converged,
            seed=seed_kind,
            cached_sequence=None if self.cached is None else self.cached[0],
            recovery=bool(snap.recovering),
            iterations=int(plan.mpc.iterations[0]),
            solve_time_s=plan.solve_time_s,
            max_dynamics_gap=plan.diagnostics["metrics"]["dynamics_gap"],
            initial_state_error=float(np.max(np.abs(plan.mpc.xs[0, 0].numpy() - initial))),
            measured_centroidal_state=initial.tolist(),
            previous_force_estimate=previous.tolist(),
            phase_timings=dict(
                preview_and_measurement_s=preview_done - started,
                solve_and_accept_s=solve_done - preview_done,
                guidance_s=time.perf_counter() - solve_done,
            ),
            candidate=plan.diagnostics,
        )
        self.history.append(
            dict(
                sequence=request.sequence,
                state_time=now,
                state=snap.state.tolist(),
                previous_torque=snap.previous_torque.tolist(),
                diagnostics=diagnostics,
            )
        )
        try:
            packet = guidance_from_centroidal(
                self.robot,
                self.planner,
                plan,
                window,
                previous,
                initial_state=snap.state,
                sequence=request.sequence,
            )
        except ValueError as exc:
            diagnostics["export_error"] = str(exc)
            raise ValueError(
                f"C0 export rejected: {plan.diagnostics['failures']}; "
                f"{plan.diagnostics['termination_reason']}; {exc}"
            ) from exc
        self.pending = (request.sequence, now, plan, recovery)
        diagnostics["phase_timings"]["guidance_s"] = time.perf_counter() - solve_done
        return packet, diagnostics


class ScheduledC0Worker(ScheduledWWorker):
    """Real C0 solve with prescribed delivery time for paired CPU model comparison.

    Inherits transport/error/restart handling unchanged. No wall-clock or spawn
    performance claims: this is deliberately the same virtual test transport as W.
    """

    def start(self):
        self.engine = C0WorkerEngine(*self.args)
        self.generation += 1
        self.epoch = f"scheduled-{self.generation}"
        self.ready = self.running = True
        self.pending = None
