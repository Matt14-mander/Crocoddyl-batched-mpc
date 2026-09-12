"""Tensor-first batched MPC with validated LQR and an experimental nonlinear DDP backend."""

from .controller import MPCController
from .ddp_problem import DDPProblem
from .problem import LQRProblem
from .result import MPCResult, SolveStatus
from .solver import BatchedMPC

__version__ = "0.1.0"
__all__ = ["BatchedMPC", "DDPProblem", "LQRProblem", "MPCController", "MPCResult", "SolveStatus"]
