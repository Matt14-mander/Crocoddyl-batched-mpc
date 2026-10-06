"""Dense batched inequality QP, primal-dual predictor/corrector baseline."""

from dataclasses import dataclass

import torch
from torch import Tensor


def _mv(a, x):
    return (a @ x[..., None]).squeeze(-1)


@dataclass
class QPSolution:
    x: Tensor
    feasible: Tensor
    converged: Tensor
    primal_violation: Tensor
    dual_residual: Tensor
    dual: Tensor
    iterations: Tensor


def solve_inequality_qp(
    H, g, G, b, *, initial=None, initial_dual=None, iterations=12, tolerance=1e-7
):
    """min .5 x'Hx+g'x, Gx<=b. No soft constraints.

    H/G may be shared or batch-major. Maximum budget, per-environment masks;
    CPU stops when all environments finish; CUDA never inspects a host scalar.
    feasibility and KKT convergence are separate. No infeasibility certificate.
    """
    batch, n = g.shape
    H = H.expand(batch, n, n)
    G = G.expand(batch, b.shape[-1], n)
    x = torch.zeros_like(g) if initial is None else initial.clone()
    s = (b - _mv(G, x)).clamp_min(1.0)
    z = torch.ones_like(s) if initial_dual is None else initial_dual.clone().clamp_min(1e-12)
    floor = 1e-12 if g.dtype == torch.float64 else 1e-7
    valid = torch.ones(batch, dtype=torch.bool, device=g.device)
    converged = torch.zeros_like(valid)
    counts = torch.zeros(batch, device=g.device, dtype=torch.int64)

    def step_length(value, direction, fraction):
        limit = torch.where(direction < 0, -value / direction.clamp_max(-floor), float("inf"))
        return (fraction * limit.amin(-1)).clamp_max(1.0)

    for _ in range(iterations):
        rd = _mv(H, x) + g + _mv(G.transpose(-1, -2), z)
        rp = _mv(G, x) + s - b
        mu = (s * z).mean(-1)
        converged |= (
            (rd.abs().amax(-1) <= tolerance) & (rp.abs().amax(-1) <= tolerance) & (mu <= tolerance)
        )
        active = valid & ~converged
        # CPU scalar inspection is cheap; CUDA keeps its fixed asynchronous budget.
        if g.device.type == "cpu" and bool((converged | ~valid).all()):
            break
        counts += active.to(torch.int64)
        W = z / s.clamp_min(floor)
        matrix = H + G.transpose(-1, -2) @ (W[..., None] * G)
        L, info = torch.linalg.cholesky_ex(matrix, check_errors=False)
        valid &= info == 0

        def direction(rc):
            rhs = -rd + _mv(G.transpose(-1, -2), (rc - z * rp) / s.clamp_min(floor))
            if g.device.type == "cpu":
                dx = torch.cholesky_solve(rhs[..., None], L).squeeze(-1)
            else:
                # cholesky_solve synchronizes on some supported CUDA versions.
                forward = torch.linalg.solve_triangular(L, rhs[..., None], upper=False)
                dx = torch.linalg.solve_triangular(
                    L.transpose(-1, -2), forward, upper=True
                ).squeeze(-1)
            ds = -rp - _mv(G, dx)
            dz = (-rc - z * ds) / s.clamp_min(floor)
            return dx, ds, dz

        _, ds_aff, dz_aff = direction(s * z)
        ap = step_length(s, ds_aff, 1.0)
        ad = step_length(z, dz_aff, 1.0)
        mu_aff = ((s + ap[:, None] * ds_aff) * (z + ad[:, None] * dz_aff)).mean(-1)
        sigma = (mu_aff / mu.clamp_min(floor)).clamp(0, 1).pow(3)
        dx, ds, dz = direction(s * z + ds_aff * dz_aff - sigma[:, None] * mu[:, None])
        ap = step_length(s, ds, 0.995)
        ad = step_length(z, dz, 0.995)
        active &= valid
        x = torch.where(active[:, None], x + ap[:, None] * dx, x)
        s = torch.where(active[:, None], (s + ap[:, None] * ds).clamp_min(floor), s)
        z = torch.where(active[:, None], (z + ad[:, None] * dz).clamp_min(floor), z)

    violation = (_mv(G, x) - b).clamp_min(0).amax(-1)
    dual = (_mv(H, x) + g + _mv(G.transpose(-1, -2), z)).abs().amax(-1)
    mu = (s * z).mean(-1)
    finite = torch.isfinite(x).all(-1) & valid
    feasible = finite & (violation <= tolerance)
    converged = feasible & (dual <= tolerance) & (mu <= tolerance)
    return QPSolution(x, feasible, converged, violation, dual, z, counts)
