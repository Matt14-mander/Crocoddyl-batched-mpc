"""M2 acceptance tests for stateful, per-environment warm starts."""

from dataclasses import replace

import pytest
import torch

from crocoddyl_batched_mpc import BatchedMPC, MPCController, MPCResult, SolveStatus
from tests.test_ddp import linear_problems

pytestmark = [pytest.mark.ddp_acceptance, pytest.mark.m2_acceptance]


class RecordingBackend:
    def __init__(self, statuses: list[torch.Tensor] | None = None) -> None:
        self.initial_controls: list[torch.Tensor] = []
        self.statuses = statuses or []
        self.calls = 0

    def solve(self, problem, x0):
        assert problem.u_init is not None
        self.initial_controls.append(problem.u_init.clone())
        value = self.calls + 1
        us = torch.full_like(problem.u_init, float(value))
        xs = x0[:, None].expand(-1, problem.horizon + 1, -1).clone()
        if self.calls < len(self.statuses):
            status = self.statuses[self.calls].to(device=x0.device)
        else:
            status = torch.full(
                (problem.batch_size,),
                SolveStatus.SUCCESS,
                device=x0.device,
                dtype=torch.int64,
            )
        failed = status == SolveStatus.NUMERICAL_FAILURE
        us = torch.where(failed[:, None, None], float("nan"), us)
        xs = torch.where(failed[:, None, None], float("nan"), xs)
        cost = torch.where(failed, float("nan"), x0.new_full((problem.batch_size,), value))
        self.calls += 1
        return MPCResult(
            us=us,
            xs=xs,
            cost=cost,
            status=status,
            iterations=status.new_ones(status.shape),
        )


def make_controller(batch: int = 3, statuses=None):
    _, problem = linear_problems(batch=batch, horizon=3)
    base = torch.arange(batch * 3, dtype=torch.float64).reshape(batch, 3, 1)
    problem = replace(problem, u_init=base)
    backend = RecordingBackend(statuses)
    return MPCController(BatchedMPC(problem, backend=backend)), backend, base


def test_horizon_shift_becomes_next_initial_guess():
    controller, backend, base = make_controller()
    state = torch.ones(3, 2, dtype=torch.float64)
    controller.compute(state)
    controller.compute(state)
    torch.testing.assert_close(backend.initial_controls[0], base)
    torch.testing.assert_close(backend.initial_controls[1], torch.ones_like(base))


def test_selective_reset_only_discards_selected_warm_start():
    controller, backend, base = make_controller()
    state = torch.ones(3, 2, dtype=torch.float64)
    controller.compute(state)
    controller.reset(torch.tensor([False, True, False]))
    controller.compute(state)
    expected = torch.ones_like(base)
    expected[1] = base[1]
    torch.testing.assert_close(backend.initial_controls[1], expected)


def test_failed_environment_does_not_overwrite_its_previous_warm_start():
    statuses = [
        torch.tensor([0, 0, 0]),
        torch.tensor([0, SolveStatus.NUMERICAL_FAILURE, 0]),
        torch.tensor([0, 0, 0]),
    ]
    controller, backend, _ = make_controller(statuses=statuses)
    state = torch.ones(3, 2, dtype=torch.float64)
    controller.compute(state)
    controller.compute(state)
    controller.compute(state)
    assert backend.initial_controls[2][0, 0, 0] == 2
    assert backend.initial_controls[2][1, 0, 0] == 1
    assert backend.initial_controls[2][2, 0, 0] == 2


def test_replacing_problem_invalidates_all_controller_caches():
    controller, backend, base = make_controller()
    state = torch.ones(3, 2, dtype=torch.float64)
    controller.compute(state)
    controller.solver.problem = replace(controller.solver.problem)
    controller.compute(state)
    torch.testing.assert_close(backend.initial_controls[1], base)


def test_problem_dimension_change_requires_a_new_controller():
    controller, _, _ = make_controller()
    _, other = linear_problems(batch=2, horizon=3)
    controller.solver.problem = other
    with pytest.raises(ValueError, match="new MPCController"):
        controller.compute(torch.ones(2, 2, dtype=torch.float64))
