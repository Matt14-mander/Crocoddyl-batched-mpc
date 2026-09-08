"""Host-observed solve latency; CUDA synchronization is only at timing boundaries."""

import argparse
import json
import statistics
import time

import torch

from crocoddyl_batched_mpc import BatchedMPC
from crocoddyl_batched_mpc.models import double_integrator


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--backend", default="torch", choices=["torch", "crocoddyl"])
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    if args.repeats < 1 or args.warmup < 0 or args.threads < 1:
        parser.error("repeats/threads must be positive; warmup must be nonnegative")
    torch.set_num_threads(args.threads)
    p = double_integrator(batch_size=args.batch_size, horizon=args.horizon, device=args.device)
    solver = BatchedMPC(p, backend=args.backend)
    x0 = p.A.new_ones(p.batch_size, p.nx)

    def synchronize() -> None:
        if x0.is_cuda:
            torch.cuda.synchronize(x0.device)

    for _ in range(args.warmup):
        solver.solve(x0)
    synchronize()
    samples = []
    for _ in range(args.repeats):
        start = time.perf_counter()
        result = solver.solve(x0)
        synchronize()
        samples.append((time.perf_counter() - start) * 1000)
    median = statistics.median(samples)
    report = {
        **vars(args),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device_name": torch.cuda.get_device_name(x0.device) if x0.is_cuda else "CPU",
        "dtype": str(x0.dtype),
        "nx": p.nx,
        "nu": p.nu,
        "median_ms": median,
        "min_ms": min(samples),
        "max_ms": max(samples),
        "env_solves_per_second": 1000 * p.batch_size / median,
        "all_success": result.success.all().item(),
        "scope": "solve incl. validation/allocation; excludes model setup and simulation",
    }
    print(json.dumps(report, indent=2))
    if not report["all_success"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
