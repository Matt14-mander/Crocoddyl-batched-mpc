"""Optional serial Crocoddyl oracle; deliberately rejects CUDA inputs."""

import math

import torch
from torch import Tensor

from ..problem import LQRProblem
from ..result import MPCResult, SolveStatus
from .torch_lqr import trajectory_cost


class CrocoddylCPUBackend:
    """Convert each affine LQR into a Crocoddyl ShootingProblem + SolverDDP.

    Models are rebuilt on each solve so tensor updates are visible. This slow
    CPU reference is for correctness, not RL throughput. NumPy conversion is
    restricted to this explicit backend. Crocoddyl internally uses float64.
    """

    def __init__(
        self,
        *,
        regularization: float = 0.0,
        max_iterations: int = 100,
        tolerance: float = 1e-9,
    ) -> None:
        if not math.isfinite(regularization) or regularization < 0:
            raise ValueError("regularization must be finite and nonnegative")
        if max_iterations < 1 or not math.isfinite(tolerance) or tolerance <= 0:
            raise ValueError("max_iterations and tolerance must be positive")
        try:
            import crocoddyl
            import numpy as np
        except ImportError as exc:
            raise ImportError(
                "Crocoddyl CPU reference requires optional bindings. Install the "
                "'crocoddyl' extra in a supported environment; Torch needs no Crocoddyl."
            ) from exc
        self.crocoddyl, self.np = crocoddyl, np
        self.regularization = regularization
        self.max_iterations = max_iterations
        self.tolerance = tolerance

    @torch.no_grad()
    def solve(self, problem: LQRProblem, x0: Tensor) -> MPCResult:
        p = problem
        p.validate_state(x0)
        if p.A.device.type != "cpu":
            raise ValueError("Crocoddyl backend is CPU-only; use backend='torch' for CUDA")
        c, np = self.crocoddyl, self.np
        arrays = {
            name: getattr(p, name).detach().double().numpy()
            for name in ("A", "B", "Q", "R", "Qf", "f", "q", "r", "qf")
        }
        initial = x0.detach().double().numpy()
        xs = x0.new_full((p.batch_size, p.horizon + 1, p.nx), float("nan"))
        us = x0.new_full((p.batch_size, p.horizon, p.nu), float("nan"))
        status = torch.full((p.batch_size,), SolveStatus.NUMERICAL_FAILURE, dtype=torch.int64)
        iterations = torch.zeros(p.batch_size, dtype=torch.int64)
        cross = np.zeros((p.nx, p.nu))
        for b in range(p.batch_size):
            if not np.isfinite(initial[b]).all() or not all(
                np.isfinite(a[b]).all() for a in arrays.values()
            ):
                continue
            try:
                running = [
                    c.ActionModelLQR(
                        arrays["A"][b, t],
                        arrays["B"][b, t],
                        arrays["Q"][b, t],
                        arrays["R"][b, t] + self.regularization * np.eye(p.nu),
                        cross,
                        arrays["f"][b, t],
                        arrays["q"][b, t],
                        arrays["r"][b, t],
                    )
                    for t in range(p.horizon)
                ]
                terminal = c.ActionModelLQR(
                    np.eye(p.nx),
                    np.zeros((p.nx, p.nu)),
                    arrays["Qf"][b],
                    np.eye(p.nu),
                    cross,
                    np.zeros(p.nx),
                    arrays["qf"][b],
                    np.zeros(p.nu),
                )
                ddp = c.SolverDDP(c.ShootingProblem(initial[b], running, terminal))
                ddp.th_stop = self.tolerance
                converged = ddp.solve([], [], self.max_iterations)
                iterations[b] = ddp.iter + 1
                if not converged:
                    status[b] = SolveStatus.MAX_ITERATIONS
                    continue
                xs[b] = torch.as_tensor(np.asarray(ddp.xs), dtype=x0.dtype)
                us[b] = torch.as_tensor(np.asarray(ddp.us), dtype=x0.dtype)
                status[b] = SolveStatus.SUCCESS
            except (ArithmeticError, RuntimeError):
                # Numerical/model failure is isolated to this environment.
                continue
        cost = trajectory_cost(p, xs, us, self.regularization)
        finite = torch.isfinite(cost) & torch.isfinite(xs).flatten(1).all(-1)
        finite = finite & torch.isfinite(us).flatten(1).all(-1)
        status = torch.where(
            (status == SolveStatus.SUCCESS) & ~finite, SolveStatus.NUMERICAL_FAILURE, status
        )
        valid = status == SolveStatus.SUCCESS
        return MPCResult(
            torch.where(valid[:, None, None], xs, float("nan")),
            torch.where(valid[:, None, None], us, float("nan")),
            torch.where(valid, cost, float("nan")),
            status,
            iterations,
        )
