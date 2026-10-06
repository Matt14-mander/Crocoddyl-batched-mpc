"""Independent active-set enumeration gates for the dense inequality QP."""

import itertools

import numpy as np
import torch

from crocoddyl_batched_mpc.qp import solve_inequality_qp


def _enumerate(H, g, G, b):
    n = len(g)
    best, value = None, float("inf")
    for count in range(n + 1):
        for active in itertools.combinations(range(len(b)), count):
            A = G[list(active)]
            K = np.block([[H, A.T], [A, np.zeros((count, count))]])
            try:
                solved = np.linalg.solve(K, np.r_[-g, b[list(active)]])
            except np.linalg.LinAlgError:
                continue
            x, dual = solved[:n], solved[n:]
            if np.max(G @ x - b) > 1e-8 or (dual < -1e-8).any():
                continue
            objective = 0.5 * x @ H @ x + g @ x
            if objective < value:
                best, value = x, objective
    return best


def test_qp_matches_enumerated_optimum_with_active_general_inequalities():
    rng = np.random.default_rng(12)
    A = rng.normal(size=(3, 3))
    H = A.T @ A + np.eye(3)
    G = np.r_[np.eye(3), -np.eye(3), [[1.0, 1.0, 1.0], [-1.0, 0.5, 1.0]]]
    b = np.array([0.5, 0.6, 0.4, 0.5, 0.6, 0.4, 0.3, 0.1])
    g = rng.normal(size=(5, 3)) * 4
    inputs = [
        torch.tensor(x, dtype=torch.float64) for x in (H, g, G, np.broadcast_to(b, (5, 8)).copy())
    ]
    result = solve_inequality_qp(*inputs, iterations=20, tolerance=1e-11)
    assert result.feasible.all() and result.converged.all()
    for i in range(5):
        np.testing.assert_allclose(
            result.x[i].numpy(), _enumerate(H, g[i], G, b), atol=2e-6, rtol=2e-6
        )
    warm = solve_inequality_qp(
        *inputs, initial=result.x, initial_dual=result.dual, iterations=10, tolerance=1e-11
    )
    torch.testing.assert_close(warm.x, result.x, atol=2e-6, rtol=2e-6)


def test_qp_infeasible_environment_does_not_contaminate_feasible_neighbor():
    H = torch.eye(2, dtype=torch.float64)
    G = torch.cat((H, -H))
    b = torch.tensor([[-1.0, 1.0, -1.0, 1.0], [1.0, 1.0, 1.0, 1.0]], dtype=torch.float64)
    g = torch.tensor([[-2.0, -0.4], [-2.0, -0.4]], dtype=torch.float64)
    result = solve_inequality_qp(H, g, G, b, iterations=20)
    assert result.feasible.tolist() == [False, True]
    torch.testing.assert_close(
        result.x[1], torch.tensor([1.0, 0.4], dtype=g.dtype), atol=2e-6, rtol=2e-6
    )


def test_qp_expired_budget_is_not_a_false_convergence_certificate():
    H = torch.eye(2, dtype=torch.float64)
    g = torch.tensor([[-10.0, 10.0]], dtype=H.dtype)
    G = torch.cat((H, -H))
    b = torch.ones(1, 4, dtype=H.dtype)
    result = solve_inequality_qp(H, g, G, b, iterations=1)
    assert not result.converged.any()
