"""Floating-base state geometry; no rigid-body/contact dynamics are simulated."""

import argparse

import torch

from crocoddyl_batched_mpc.manifolds import FloatingBaseManifold


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--joints", type=int, default=12)
    args = parser.parse_args()
    if args.batch_size < 1 or args.joints < 0:
        parser.error("batch-size must be positive; joints must be nonnegative")
    manifold = FloatingBaseManifold(args.joints)
    state = manifold.neutral(
        torch.empty(args.batch_size, manifold.nx, device=args.device, dtype=torch.float64)
    )
    delta = state.new_zeros(args.batch_size, manifold.ndx)
    delta[:, 0] = 0.1  # Body-forward displacement.
    delta[:, 5] = 0.2  # Body-yaw rotation, coupled to translation by SE(3) exp.
    candidate = manifold.integrate(state, delta)
    recovered = manifold.diff(candidate, state)
    torch.testing.assert_close(recovered, delta, atol=1e-12, rtol=1e-12)
    print(f"nq={manifold.nq}, nv={manifold.nv}, nx={manifold.nx}, ndx={manifold.ndx}")
    print(f"State shape: {tuple(candidate.shape)}; tangent shape: {tuple(recovered.shape)}")
    print(f"Max round-trip error: {(recovered - delta).abs().max().item():.3g}")
    print("First pose [world xyz, quaternion xyzw]:", candidate[0, :7].tolist())


if __name__ == "__main__":
    main()
