from typing import Protocol

from torch import Tensor

from ..problem import LQRProblem
from ..result import MPCResult


class MPCBackend(Protocol):
    """Backend boundary; version 0.1 accepts only Euclidean, unconstrained LQR."""

    def solve(self, problem: LQRProblem, x0: Tensor) -> MPCResult: ...
