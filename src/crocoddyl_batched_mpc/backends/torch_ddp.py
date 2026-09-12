"""Differential Dynamic Programming (DDP) backend for nonlinear MPC.

Implements batched DDP with:
- Backward Riccati-like recursion for computing optimal feedback gains
- Forward pass with line search for trajectory update
- Adaptive regularization for numerical stability
- Per-environment failure tracking and isolation
"""

import torch
from torch import Tensor

from ..ddp_problem import DDPProblem
from ..result import MPCResult, SolveStatus


class TorchDDPBackend:
    """Batched DDP solver using PyTorch on CPU or CUDA.

    Each environment in the batch is solved independently with shared dynamics/cost models.
    Failed environments (non-PD Hessian, divergence) are isolated and don't affect others.
    """

    def __init__(self) -> None:
        pass

    def solve(self, problem: DDPProblem, x0: Tensor) -> MPCResult:
        """Solve nonlinear optimal control problem using DDP.

        Args:
            problem: DDPProblem with dynamics, cost, and parameters
            x0: Initial state [batch, nx]

        Returns:
            MPCResult with optimal trajectory and status
        """
        problem.validate_state(x0)

        batch = problem.batch_size
        horizon = problem.horizon
        nu = problem.nu
        device, dtype = x0.device, x0.dtype

        # Initialize trajectory
        if problem.x_init is not None:
            xs = problem.x_init.clone()
        else:
            # Simple forward rollout with zero control
            xs = self._initialize_trajectory(problem, x0)

        if problem.u_init is not None:
            us = problem.u_init.clone()
        else:
            us = torch.zeros(batch, horizon, nu, device=device, dtype=dtype)

        # Track which environments are still valid
        valid = torch.ones(batch, dtype=torch.bool, device=device)

        # Initial cost
        cost = self._compute_cost(problem, xs, us)
        best_cost = cost.clone()

        # Regularization
        reg = torch.full((batch,), problem.regularization_init, device=device, dtype=dtype)

        for iteration in range(problem.max_iterations):
            # Backward pass: compute gains and expected cost reduction
            with torch.enable_grad():
                gains, offsets, expected_reduction, bp_success = self._backward_pass(
                    problem, xs, us, reg
                )

            valid = valid & bp_success

            if not valid.any():
                # All environments failed
                break

            # Forward pass with line search
            xs_new, us_new, cost_new, fp_success = self._forward_pass(
                problem, x0, xs, us, gains, offsets, cost, expected_reduction
            )

            valid = valid & fp_success

            # Update trajectories for successful environments
            improvement = cost - cost_new
            accept = (improvement > 0) & valid

            xs = torch.where(accept[:, None, None], xs_new, xs)
            us = torch.where(accept[:, None, None], us_new, us)
            cost = torch.where(accept, cost_new, cost)

            # Update regularization
            reg = self._update_regularization(problem, reg, accept, improvement, expected_reduction)

            # Check convergence
            relative_improvement = improvement / (torch.abs(best_cost) + 1e-8)
            gradient_norm = torch.zeros(batch, device=device, dtype=dtype)

            # Compute gradient norm for convergence check
            for t in range(horizon):
                gradient_norm += offsets[:, t].abs().sum(-1)

            converged = (
                (relative_improvement < problem.cost_tolerance)
                | (gradient_norm < problem.gradient_tolerance)
            ) & valid

            if converged.all():
                break

        # Final status
        status = torch.where(
            valid,
            torch.where(
                converged,
                SolveStatus.SUCCESS,
                SolveStatus.MAX_ITERATIONS,
            ),
            SolveStatus.NUMERICAL_FAILURE,
        )

        # Mark failed trajectories as NaN
        xs = torch.where(valid[:, None, None], xs, float("nan"))
        us = torch.where(valid[:, None, None], us, float("nan"))
        cost = torch.where(valid, cost, float("nan"))

        iterations = torch.full((batch,), iteration + 1, dtype=torch.int64, device=device)

        return MPCResult(xs, us, cost, status, iterations)

    def _initialize_trajectory(self, problem: DDPProblem, x0: Tensor) -> Tensor:
        """Forward rollout with zero control to initialize state trajectory."""
        batch, horizon, nx = problem.batch_size, problem.horizon, problem.nx
        device, dtype = x0.device, x0.dtype

        xs = torch.zeros(batch, horizon + 1, nx, device=device, dtype=dtype)
        xs[:, 0] = x0

        u_zero = torch.zeros(batch, problem.nu, device=device, dtype=dtype)

        for t in range(horizon):
            xs[:, t + 1] = problem.dynamics.calc(xs[:, t], u_zero)

        return xs

    def _compute_cost(self, problem: DDPProblem, xs: Tensor, us: Tensor) -> Tensor:
        """Compute total cost for each environment."""
        batch, horizon = problem.batch_size, problem.horizon
        cost = torch.zeros(batch, device=xs.device, dtype=xs.dtype)

        # Running cost
        for t in range(horizon):
            cost += problem.cost.calc(xs[:, t], us[:, t])

        # Terminal cost
        cost += problem.cost.calc(xs[:, horizon], None)

        return cost

    def _backward_pass(
        self, problem: DDPProblem, xs: Tensor, us: Tensor, reg: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Compute optimal feedback gains via backward Riccati recursion.

        Returns:
            K: Feedback gains [batch, T, nu, ndx]
            k: Feedforward terms [batch, T, nu]
            expected_reduction: Expected cost improvement [batch]
            success: Whether backward pass succeeded for each env [batch]
        """
        batch, horizon, nu, ndx = (
            problem.batch_size,
            problem.horizon,
            problem.nu,
            problem.ndx,
        )
        device, dtype = xs.device, xs.dtype

        K = torch.zeros(batch, horizon, nu, ndx, device=device, dtype=dtype)
        k = torch.zeros(batch, horizon, nu, device=device, dtype=dtype)

        # Value function derivatives at terminal time
        lx_T, _, lxx_T, _, _ = problem.cost.calc_diff(xs[:, horizon], None)
        Vx = lx_T
        Vxx = lxx_T

        expected_reduction = torch.zeros(batch, device=device, dtype=dtype)
        success = torch.ones(batch, dtype=torch.bool, device=device)

        eye_nu = torch.eye(nu, device=device, dtype=dtype)

        for t in reversed(range(horizon)):
            # Dynamics derivatives
            Fx, Fu = problem.dynamics.calc_diff(xs[:, t], us[:, t])

            # Cost derivatives
            lx, lu, lxx, luu, lxu = problem.cost.calc_diff(xs[:, t], us[:, t])

            # Q-function derivatives
            Qx = lx + torch.matmul(Fx.transpose(-1, -2), Vx.unsqueeze(-1)).squeeze(-1)
            Qu = lu + torch.matmul(Fu.transpose(-1, -2), Vx.unsqueeze(-1)).squeeze(-1)

            Qxx = lxx + torch.matmul(torch.matmul(Fx.transpose(-1, -2), Vxx), Fx)
            Quu = luu + torch.matmul(torch.matmul(Fu.transpose(-1, -2), Vxx), Fu)
            Qux = lxu.transpose(-1, -2) + torch.matmul(torch.matmul(Fu.transpose(-1, -2), Vxx), Fx)

            # Add regularization: Quu += λ·I
            Quu_reg = Quu + reg[:, None, None] * eye_nu

            # Cholesky decomposition
            L, info = torch.linalg.cholesky_ex(Quu_reg, check_errors=False)
            good = (info == 0) & torch.isfinite(L).flatten(1).all(-1)
            success = success & good

            # Compute gains (set to zero for failed environments)
            L_safe = torch.where(good[:, None, None], L, eye_nu)

            # Solve: Quu_reg·K = -Qux and Quu_reg·k = -Qu
            rhs = torch.cat((-Qux, -Qu.unsqueeze(-1)), dim=-1)
            intermediate = torch.linalg.solve_triangular(L_safe, rhs, upper=False)
            solution = torch.linalg.solve_triangular(
                L_safe.transpose(-1, -2), intermediate, upper=True
            )

            K[:, t] = torch.where(good[:, None, None], solution[..., :ndx], 0.0)
            k[:, t] = torch.where(good[:, None], solution[..., ndx], 0.0)

            # Expected cost reduction (for line search)
            expected_reduction += torch.where(
                good,
                -0.5 * torch.matmul(k[:, t].unsqueeze(-2), Qu.unsqueeze(-1)).squeeze(),
                0.0,
            )

            # Update value function
            Vx = Qx + torch.matmul(K[:, t].transpose(-1, -2), Qu.unsqueeze(-1)).squeeze(-1)
            Vxx = Qxx + torch.matmul(torch.matmul(K[:, t].transpose(-1, -2), Quu), K[:, t])
            Vxx = 0.5 * (Vxx + Vxx.transpose(-1, -2))  # Enforce symmetry

        return K, k, expected_reduction, success

    def _forward_pass(
        self,
        problem: DDPProblem,
        x0: Tensor,
        xs_nom: Tensor,
        us_nom: Tensor,
        K: Tensor,
        k: Tensor,
        cost_nom: Tensor,
        expected_reduction: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Forward rollout with line search.

        Returns:
            xs_new: New state trajectory [batch, T+1, nx]
            us_new: New control trajectory [batch, T, nu]
            cost_new: New cost [batch]
            success: Line search succeeded [batch]
        """
        batch, horizon, nx, nu = problem.batch_size, problem.horizon, problem.nx, problem.nu
        device, dtype = x0.device, x0.dtype

        # Line search parameters - more aggressive
        alphas = torch.tensor(
            [1.0, 0.8, 0.6, 0.4, 0.2, 0.1, 0.05, 0.01], device=device, dtype=dtype
        )

        best_xs = xs_nom.clone()
        best_us = us_nom.clone()
        best_cost = cost_nom.clone()
        success = torch.zeros(batch, dtype=torch.bool, device=device)

        for alpha in alphas:
            xs_new = torch.zeros(batch, horizon + 1, nx, device=device, dtype=dtype)
            us_new = torch.zeros(batch, horizon, nu, device=device, dtype=dtype)
            xs_new[:, 0] = x0

            rollout_valid = torch.ones(batch, dtype=torch.bool, device=device)

            # Rollout with updated controls
            for t in range(horizon):
                # State deviation (in tangent space for manifolds)
                dx = problem.manifold.diff(xs_new[:, t], xs_nom[:, t])

                # Control update: u = u_nom + α·k + K·dx
                us_new[:, t] = (
                    us_nom[:, t]
                    + alpha * k[:, t]
                    + torch.matmul(K[:, t], dx.unsqueeze(-1)).squeeze(-1)
                )

                # Forward dynamics
                xs_new[:, t + 1] = problem.dynamics.calc(xs_new[:, t], us_new[:, t])

                # Check for NaN/Inf in rollout
                rollout_valid = rollout_valid & torch.isfinite(xs_new[:, t + 1]).all(-1)

            # Compute new cost only for valid rollouts
            cost_new = torch.where(
                rollout_valid,
                self._compute_cost(problem, xs_new, us_new),
                torch.full((batch,), float("inf"), device=device, dtype=dtype),
            )

            # Check improvement
            improvement = cost_nom - cost_new
            improved = (improvement > 0) & torch.isfinite(cost_new) & rollout_valid

            # Update best for environments that improved
            update = improved & ~success
            best_xs = torch.where(update[:, None, None], xs_new, best_xs)
            best_us = torch.where(update[:, None, None], us_new, best_us)
            best_cost = torch.where(update, cost_new, best_cost)
            success = success | improved

        return best_xs, best_us, best_cost, success

    def _update_regularization(
        self,
        problem: DDPProblem,
        reg: Tensor,
        accept: Tensor,
        improvement: Tensor,
        expected_improvement: Tensor,
    ) -> Tensor:
        """Adaptive regularization update with more aggressive tuning."""
        # Increase reg if step was rejected or improvement was poor
        # Decrease reg if step was good

        ratio = improvement / (expected_improvement.abs() + 1e-8)

        # More aggressive decrease for very good steps (ratio > 0.9)
        very_good = accept & (ratio > 0.9)
        reg = torch.where(
            very_good,
            torch.maximum(
                reg / (problem.regularization_factor * 2),  # Faster decrease
                torch.tensor(problem.regularization_min, device=reg.device),
            ),
            reg,
        )

        # Normal decrease for good steps (0.5 < ratio <= 0.9)
        good = accept & (ratio > 0.5) & (ratio <= 0.9) & ~very_good
        reg = torch.where(
            good,
            torch.maximum(
                reg / problem.regularization_factor,
                torch.tensor(problem.regularization_min, device=reg.device),
            ),
            reg,
        )

        # Increase for rejected or poor steps
        increase = ~accept | (ratio <= 0.25)
        reg = torch.where(
            increase,
            torch.minimum(
                reg * problem.regularization_factor,
                torch.tensor(problem.regularization_max, device=reg.device),
            ),
            reg,
        )

        return reg
