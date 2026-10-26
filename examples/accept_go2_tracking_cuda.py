"""Fail-closed CUDA acceptance: contract tests, closed loops and batch timings."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch
from go2_front_leg_tracking import run_tracking


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 32, 128])
    parser.add_argument("--output", type=Path, default=Path("artifacts/go2-tracking-cuda.json"))
    parser.add_argument("--no-compile", action="store_true")
    args = parser.parse_args()
    if any(batch < 1 for batch in args.batch_sizes):
        parser.error("batch sizes must be positive")
    torch.set_num_threads(1)
    report = {"status": "unavailable", "cuda_available": torch.cuda.is_available(), "runs": []}
    exit_code = 2
    if report["cuda_available"]:
        root = Path(__file__).resolve().parents[1]
        tests = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "tests/test_go2_tracking.py",
                "tests/test_go2_torch.py",
                "tests/test_go2_switching.py",
                "-m",
                "cuda",
            ],
            cwd=root,
            capture_output=True,
            text=True,
        )
        report["contract_tests"] = {
            "exit_code": tests.returncode,
            "stdout": tests.stdout,
            "stderr": tests.stderr,
        }
        report["status"] = "failed"
        exit_code = 1
        if tests.returncode == 0:
            try:
                for batch in args.batch_sizes:
                    for task in ("stance", "front-leg"):
                        print(f"CUDA acceptance: B={batch}, task={task}", flush=True)
                        report["runs"].append(
                            run_tracking(
                                device="cuda",
                                batch_size=batch,
                                task=task,
                                compile_physics=not args.no_compile,
                            )
                        )
                if all(run["passed"] for run in report["runs"]):
                    report["status"] = "passed"
                    exit_code = 0
            except Exception as exc:
                report["error"] = f"{type(exc).__name__}: {exc}"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"{report['status']}: {args.output}")
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
