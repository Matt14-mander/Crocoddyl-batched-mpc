from torch import Tensor

from .backends.base import MPCBackend
from .backends.torch_lqr import TorchLQRBackend
from .ddp_problem import DDPProblem
from .problem import LQRProblem
from .result import MPCResult


class BatchedMPC:
    """Unified solve API. Device is determined by the problem, never auto-migrated."""

    def __init__(
        self,
        problem: LQRProblem | DDPProblem,
        *,
        backend: str | MPCBackend = "torch",
        regularization: float = 0.0,
    ) -> None:
        self.problem = problem

        # Auto-select backend based on problem type
        if isinstance(backend, str):
            if backend == "torch":
                if isinstance(problem, LQRProblem):
                    self.backend = TorchLQRBackend(regularization=regularization)
                elif isinstance(problem, DDPProblem):
                    from .backends.torch_ddp import TorchDDPBackend
                    self.backend = TorchDDPBackend()
                else:
                    raise TypeError(f"Unsupported problem type: {type(problem)}")
            elif backend == "crocoddyl":
                from .backends.crocoddyl_cpu import CrocoddylCPUBackend
                self.backend = CrocoddylCPUBackend(regularization=regularization)
            else:
                raise ValueError(f"Unknown backend: {backend!r}; choose 'torch' or 'crocoddyl'")
        else:
            if regularization != 0.0:
                raise ValueError("Configure regularization directly on a custom backend")
            self.backend = backend

    def solve(self, x0: Tensor) -> MPCResult:
        return self.backend.solve(self.problem, x0)
