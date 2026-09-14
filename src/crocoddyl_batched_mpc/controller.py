"""Stateful tensor adapter for vectorized environments, including Isaac Lab."""

from collections.abc import Mapping
from dataclasses import replace
from typing import Any

import torch
from torch import Tensor

from .ddp_problem import DDPProblem
from .problem import LQRProblem
from .result import MPCResult
from .solver import BatchedMPC


class MPCController:
    """Compute the first action and retain per-environment MPC state.

    DDP solutions are horizon-shifted for the next solve. Reset terminated
    environments to clear both the warm start and fallback action. The fallback
    policy is generic; the robot task defines physical action limits.
    """

    def __init__(self, solver: BatchedMPC, *, warm_start: bool = True) -> None:
        if not isinstance(warm_start, bool):
            raise TypeError("warm_start must be a bool")
        self.solver = solver
        p = solver.problem

        # Get dimensions based on problem type
        if isinstance(p, LQRProblem):
            batch_size, nu = p.batch_size, p.nu
            device, dtype = p.A.device, p.A.dtype
        elif isinstance(p, DDPProblem):
            batch_size, nu = p.batch_size, p.nu
            device, dtype = p.device, p.dtype
        else:
            raise TypeError(f"Unsupported problem type: {type(p)}")

        self._last_action = torch.zeros(batch_size, nu, device=device, dtype=dtype)
        self._problem_identity = p
        self._warm_start_enabled = warm_start and isinstance(p, DDPProblem)
        if isinstance(p, DDPProblem):
            self._warm_controls = torch.zeros(batch_size, p.horizon, nu, device=device, dtype=dtype)
            self._warm_states = torch.zeros(
                batch_size, p.horizon + 1, p.nx, device=device, dtype=dtype
            )
            self._warm_valid = torch.zeros(batch_size, device=device, dtype=torch.bool)
        else:
            self._warm_controls = None
            self._warm_states = None
            self._warm_valid = None

    @torch.no_grad()
    def compute(self, state: Tensor) -> tuple[Tensor, MPCResult]:
        problem = self.solver.problem
        self._handle_problem_replacement(problem)
        if self._warm_start_enabled:
            assert isinstance(problem, DDPProblem)
            assert self._warm_controls is not None and self._warm_valid is not None
            initial = (
                torch.zeros_like(self._warm_controls) if problem.u_init is None else problem.u_init
            )
            initial = torch.where(self._warm_valid[:, None, None], self._warm_controls, initial)
            solve_problem = replace(problem, u_init=initial)
            if getattr(self.solver.backend, "uses_state_guess", False):
                assert self._warm_states is not None
                if problem.x_init is None:
                    initial_states = torch.empty_like(self._warm_states)
                    initial_states[:, 0] = state
                    for t in range(problem.horizon):
                        initial_states[:, t + 1] = problem.dynamics.calc(
                            initial_states[:, t], initial[:, t]
                        )
                else:
                    initial_states = problem.x_init.clone()
                    initial_states[:, 0] = state
                initial_states = torch.where(
                    self._warm_valid[:, None, None], self._warm_states, initial_states
                )
                initial_states[:, 0] = state
                solve_problem = replace(solve_problem, x_init=initial_states)
            result = self.solver.backend.solve(solve_problem, state)
            shifted = torch.cat((result.us[:, 1:], result.us[:, -1:]), dim=1)
            self._warm_controls.copy_(
                torch.where(result.usable[:, None, None], shifted, self._warm_controls)
            )
            if getattr(self.solver.backend, "uses_state_guess", False):
                assert self._warm_states is not None
                tail = problem.dynamics.calc(result.xs[:, -1], result.us[:, -1])
                shifted_states = torch.cat((result.xs[:, 1:], tail[:, None]), dim=1)
                self._warm_states.copy_(
                    torch.where(result.usable[:, None, None], shifted_states, self._warm_states)
                )
            self._warm_valid |= result.usable
        else:
            result = self.solver.solve(state)
        action = torch.where(result.usable[:, None], result.action, self._last_action)
        self._last_action.copy_(action)
        return action, result

    @torch.no_grad()
    def reset(self, mask: Tensor | None = None) -> None:
        """Clear fallback and warm-start state for all or selected environments.

        Call this after mutating model/reference tensors in place so the next
        solve cannot consume a trajectory produced for the old problem.
        """
        if mask is None:
            self._last_action.zero_()
            if self._warm_controls is not None and self._warm_valid is not None:
                self._warm_controls.zero_()
                assert self._warm_states is not None
                self._warm_states.zero_()
                self._warm_valid.zero_()
            return
        self._validate_mask(mask)
        self._last_action.masked_fill_(mask[:, None], 0.0)
        if self._warm_controls is not None and self._warm_valid is not None:
            self._warm_controls.masked_fill_(mask[:, None, None], 0.0)
            assert self._warm_states is not None
            self._warm_states.masked_fill_(mask[:, None, None], 0.0)
            self._warm_valid.masked_fill_(mask, False)

    @torch.no_grad()
    def update_parameters(
        self,
        mask: Tensor,
        *,
        dynamics: Mapping[str, Any] | None = None,
        cost: Mapping[str, Any] | None = None,
    ) -> None:
        """Update batched model parameters and invalidate the same environments.

        Models opt into this API by implementing ``update_parameters(mask, **values)``.
        Cache invalidation also runs when an update raises, so a partial model update
        can never leave the previous warm start marked valid.
        """
        self._validate_mask(mask)
        if not isinstance(self.solver.problem, DDPProblem):
            raise TypeError("parameter updates require a DDPProblem")
        if dynamics is None and cost is None:
            raise ValueError("provide dynamics and/or cost parameter updates")
        try:
            for name, values in (("dynamics", dynamics), ("cost", cost)):
                if values is None:
                    continue
                model = getattr(self.solver.problem, name)
                update = getattr(model, "update_parameters", None)
                if update is None:
                    raise TypeError(f"{name} model does not support parameter updates")
                update(mask, **dict(values))
        finally:
            self.reset(mask)

    def _validate_mask(self, mask: Tensor) -> None:
        if not isinstance(mask, Tensor) or mask.dtype != torch.bool:
            raise TypeError("reset mask must be a bool tensor")
        if mask.shape != self._last_action.shape[:1] or mask.device != self._last_action.device:
            raise ValueError("reset mask must have shape [batch] on the problem device")

    def _handle_problem_replacement(self, problem: LQRProblem | DDPProblem) -> None:
        if problem is self._problem_identity:
            return
        old = self._problem_identity
        compatible = (
            type(problem) is type(old)
            and problem.batch_size == old.batch_size
            and problem.nu == old.nu
            and (
                not isinstance(problem, DDPProblem)
                or (problem.horizon == old.horizon and problem.nx == old.nx)
            )
        )
        if not compatible:
            raise ValueError("Create a new MPCController after changing problem type or dimensions")
        if isinstance(problem, LQRProblem):
            same_metadata = (
                problem.A.device == self._last_action.device
                and problem.A.dtype == self._last_action.dtype
            )
        else:
            same_metadata = (
                problem.device == self._last_action.device
                and problem.dtype == self._last_action.dtype
            )
        if not same_metadata:
            raise ValueError("Create a new MPCController after changing problem device or dtype")
        self.reset()
        self._problem_identity = problem
