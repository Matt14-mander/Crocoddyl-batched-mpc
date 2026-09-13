"""Batched single-shooting DDP backend for nonlinear MPC."""

import torch
from torch import Tensor

from ..ddp_problem import DDPProblem
from ..result import MPCResult, SolveStatus


def _mv(matrix: Tensor, vector: Tensor) -> Tensor:
    return (matrix @ vector.unsqueeze(-1)).squeeze(-1)


class TorchDDPBackend:
    """Fixed-budget batched DDP with per-environment solver state.

    Environments remain independent. A rejected line-search step increases
    regularization and can be retried; it is not a numerical failure. The
    implementation deliberately runs the configured iteration count without
    tensor-to-host convergence branches, which keeps the CUDA path asynchronous.
    """

    @torch.no_grad()
    def solve(self, problem: DDPProblem, x0: Tensor) -> MPCResult:
        problem.validate_state(x0)
        batch, horizon, nu = problem.batch_size, problem.horizon, problem.nu
        device, dtype = x0.device, x0.dtype

        if problem.u_init is None:
            us = torch.zeros(batch, horizon, nu, device=device, dtype=dtype)
        else:
            us = problem.u_init.clone()

        initial_finite = torch.isfinite(x0).all(-1) & torch.isfinite(us).flatten(1).all(-1)
        safe_x0 = torch.where(initial_finite[:, None], x0, 0.0)
        us = torch.where(initial_finite[:, None, None], us, 0.0)
        xs = self._rollout(problem, safe_x0, us)
        cost = self._compute_cost(problem, xs, us)
        initial_finite &= torch.isfinite(xs).flatten(1).all(-1) & torch.isfinite(cost)

        failed = ~initial_finite
        converged = torch.zeros(batch, dtype=torch.bool, device=device)
        iterations = torch.zeros(batch, dtype=torch.int64, device=device)
        reg = torch.full((batch,), problem.regularization_init, device=device, dtype=dtype)
        reg_factor = problem.regularization_factor

        for _ in range(problem.max_iterations):
            active = ~(failed | converged)
            iterations += active.to(torch.int64)

            with torch.enable_grad():
                gains, offsets, expected, backward_ok, derivatives_finite = self._backward_pass(
                    problem, xs, us, reg
                )

            failed |= active & ~derivatives_finite
            active &= derivatives_finite

            factorization_failed = active & ~backward_ok
            failed |= factorization_failed & (reg >= problem.regularization_max)
            reg = torch.where(
                factorization_failed,
                torch.clamp_max(reg * reg_factor, problem.regularization_max),
                reg,
            )

            can_step = active & backward_ok
            feedforward_norm = offsets.abs().flatten(1).amax(-1)
            stationary = can_step & (feedforward_norm <= problem.gradient_tolerance)
            converged |= stationary
            attempted = can_step & ~stationary

            xs_new, us_new, cost_new, accepted, rollout_finite = self._forward_pass(
                problem,
                safe_x0,
                xs,
                us,
                gains,
                offsets,
                cost,
                expected,
                attempted,
            )
            failed |= attempted & ~rollout_finite
            accepted &= ~failed

            improvement = cost - cost_new
            relative_improvement = improvement / cost.abs().clamp_min(1.0)
            xs = torch.where(accepted[:, None, None], xs_new, xs)
            us = torch.where(accepted[:, None, None], us_new, us)
            cost = torch.where(accepted, cost_new, cost)

            converged |= accepted & (relative_improvement <= problem.cost_tolerance)
            reg = self._update_regularization(
                problem, reg, attempted & ~failed, accepted, improvement, expected
            )

        finite_result = (
            torch.isfinite(xs).flatten(1).all(-1)
            & torch.isfinite(us).flatten(1).all(-1)
            & torch.isfinite(cost)
        )
        failed |= ~finite_result
        status = torch.where(
            failed,
            SolveStatus.NUMERICAL_FAILURE,
            torch.where(converged, SolveStatus.SUCCESS, SolveStatus.MAX_ITERATIONS),
        )
        return MPCResult(
            torch.where(failed[:, None, None], float("nan"), xs),
            torch.where(failed[:, None, None], float("nan"), us),
            torch.where(failed, float("nan"), cost),
            status,
            iterations,
        )

    def _rollout(self, problem: DDPProblem, x0: Tensor, us: Tensor) -> Tensor:
        xs = torch.empty(
            problem.batch_size,
            problem.horizon + 1,
            problem.nx,
            device=x0.device,
            dtype=x0.dtype,
        )
        xs[:, 0] = x0
        for t in range(problem.horizon):
            xs[:, t + 1] = problem.dynamics.calc(xs[:, t], us[:, t])
        return xs

    def _compute_cost(self, problem: DDPProblem, xs: Tensor, us: Tensor) -> Tensor:
        cost = torch.zeros(problem.batch_size, device=xs.device, dtype=xs.dtype)
        for t in range(problem.horizon):
            cost += problem.cost.calc(xs[:, t], us[:, t])
        return cost + problem.cost.calc(xs[:, problem.horizon], None)

    def _backward_pass(
        self, problem: DDPProblem, xs: Tensor, us: Tensor, reg: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        batch, horizon, nu, ndx = (
            problem.batch_size,
            problem.horizon,
            problem.nu,
            problem.ndx,
        )
        device, dtype = xs.device, xs.dtype
        gains = torch.zeros(batch, horizon, nu, ndx, device=device, dtype=dtype)
        offsets = torch.zeros(batch, horizon, nu, device=device, dtype=dtype)
        eye_nu = torch.eye(nu, device=device, dtype=dtype)

        Vx, _, Vxx, _, _ = problem.cost.calc_diff(xs[:, horizon], None)
        derivatives_finite = torch.isfinite(Vx).all(-1) & torch.isfinite(Vxx).flatten(1).all(-1)
        factorization_ok = torch.ones(batch, dtype=torch.bool, device=device)
        expected = torch.zeros(batch, device=device, dtype=dtype)

        for t in reversed(range(horizon)):
            Fx, Fu = problem.dynamics.calc_diff(xs[:, t], us[:, t])
            lx, lu, lxx, luu, lxu = problem.cost.calc_diff(xs[:, t], us[:, t])
            assert lu is not None and luu is not None and lxu is not None

            stage_finite = torch.ones(batch, dtype=torch.bool, device=device)
            for value in (Fx, Fu, lx, lu, lxx, luu, lxu):
                stage_finite &= torch.isfinite(value).flatten(1).all(-1)
            derivatives_finite &= stage_finite

            FxT, FuT = Fx.transpose(-1, -2), Fu.transpose(-1, -2)
            Qx = lx + _mv(FxT, Vx)
            Qu = lu + _mv(FuT, Vx)
            Qxx = lxx + FxT @ Vxx @ Fx
            Quu = luu + FuT @ Vxx @ Fu
            Qux = lxu.transpose(-1, -2) + FuT @ Vxx @ Fx

            Quu_reg = Quu + reg[:, None, None] * eye_nu
            L, info = torch.linalg.cholesky_ex(Quu_reg, check_errors=False)
            good = stage_finite & (info == 0) & torch.isfinite(L).flatten(1).all(-1)
            factorization_ok &= good
            L_safe = torch.where(good[:, None, None], L, eye_nu)
            rhs = torch.cat((-Qux, -Qu.unsqueeze(-1)), dim=-1)
            intermediate = torch.linalg.solve_triangular(L_safe, rhs, upper=False)
            solution = torch.linalg.solve_triangular(
                L_safe.transpose(-1, -2), intermediate, upper=True
            )
            K = torch.where(good[:, None, None], solution[..., :ndx], 0.0)
            k = torch.where(good[:, None], solution[..., ndx], 0.0)
            gains[:, t], offsets[:, t] = K, k

            linear = (k * Qu).sum(-1)
            quadratic = 0.5 * (k * _mv(Quu, k)).sum(-1)
            expected += torch.where(good, -(linear + quadratic), 0.0)

            KT, QuxT = K.transpose(-1, -2), Qux.transpose(-1, -2)
            Vx = Qx + _mv(KT, Qu) + _mv(QuxT, k) + _mv(KT, _mv(Quu, k))
            Vxx = Qxx + QuxT @ K + KT @ Qux + KT @ Quu @ K
            Vxx = 0.5 * (Vxx + Vxx.transpose(-1, -2))

        return gains, offsets, expected.clamp_min(0.0), factorization_ok, derivatives_finite

    def _forward_pass(
        self,
        problem: DDPProblem,
        x0: Tensor,
        xs_nominal: Tensor,
        us_nominal: Tensor,
        gains: Tensor,
        offsets: Tensor,
        cost_nominal: Tensor,
        expected: Tensor,
        eligible: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        batch, horizon, nx, nu = (
            problem.batch_size,
            problem.horizon,
            problem.nx,
            problem.nu,
        )
        device, dtype = x0.device, x0.dtype
        alphas = (1.0, 0.8, 0.6, 0.4, 0.2, 0.1, 0.05, 0.01)
        best_xs, best_us, best_cost = xs_nominal.clone(), us_nominal.clone(), cost_nominal.clone()
        accepted = torch.zeros(batch, dtype=torch.bool, device=device)
        any_finite = torch.zeros(batch, dtype=torch.bool, device=device)

        for alpha in alphas:
            xs_candidate = torch.empty(batch, horizon + 1, nx, device=device, dtype=dtype)
            us_candidate = torch.empty(batch, horizon, nu, device=device, dtype=dtype)
            xs_candidate[:, 0] = x0
            rollout_finite = torch.isfinite(x0).all(-1)
            for t in range(horizon):
                dx = problem.manifold.diff(xs_candidate[:, t], xs_nominal[:, t])
                us_candidate[:, t] = us_nominal[:, t] + alpha * offsets[:, t] + _mv(gains[:, t], dx)
                xs_candidate[:, t + 1] = problem.dynamics.calc(
                    xs_candidate[:, t], us_candidate[:, t]
                )
                rollout_finite &= torch.isfinite(us_candidate[:, t]).all(-1)
                rollout_finite &= torch.isfinite(xs_candidate[:, t + 1]).all(-1)

            candidate_cost = self._compute_cost(problem, xs_candidate, us_candidate)
            rollout_finite &= torch.isfinite(candidate_cost)
            any_finite |= eligible & rollout_finite
            improvement = cost_nominal - candidate_cost
            required = 1e-4 * alpha * expected
            acceptable = (
                eligible
                & rollout_finite
                & ~accepted
                & (improvement > 0.0)
                & (improvement >= required)
            )
            best_xs = torch.where(acceptable[:, None, None], xs_candidate, best_xs)
            best_us = torch.where(acceptable[:, None, None], us_candidate, best_us)
            best_cost = torch.where(acceptable, candidate_cost, best_cost)
            accepted |= acceptable

        return best_xs, best_us, best_cost, accepted, any_finite | ~eligible

    def _update_regularization(
        self,
        problem: DDPProblem,
        reg: Tensor,
        attempted: Tensor,
        accepted: Tensor,
        improvement: Tensor,
        expected: Tensor,
    ) -> Tensor:
        ratio = improvement / expected.clamp_min(torch.finfo(reg.dtype).eps)
        factor = problem.regularization_factor
        decrease = accepted & (ratio >= 0.75)
        increase = attempted & (~accepted | (ratio < 0.25))
        reg = torch.where(decrease, torch.clamp_min(reg / factor, problem.regularization_min), reg)
        return torch.where(increase, torch.clamp_max(reg * factor, problem.regularization_max), reg)
