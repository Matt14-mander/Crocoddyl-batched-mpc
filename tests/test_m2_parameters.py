"""M2.1 acceptance for per-environment model parameters and updates."""

import pytest
import torch

from crocoddyl_batched_mpc import BatchedMPC, DDPProblem, MPCController
from crocoddyl_batched_mpc.models import (
    PendulumCost,
    PendulumCostParameters,
    PendulumDynamics,
    PendulumDynamicsParameters,
)
from tests.test_m2_warm_start import RecordingBackend

pytestmark = [pytest.mark.ddp_acceptance, pytest.mark.m2_acceptance]


def parameterized_problem(
    batch: int = 3, *, device: str = "cpu"
) -> tuple[DDPProblem, PendulumDynamicsParameters, PendulumCostParameters]:
    dtype = torch.float64
    dynamics_parameters = PendulumDynamicsParameters.create(
        batch,
        mass=torch.linspace(0.8, 1.2, batch, device=device, dtype=dtype),
        length=torch.linspace(0.7, 1.1, batch, device=device, dtype=dtype),
        damping=torch.linspace(0.05, 0.15, batch, device=device, dtype=dtype),
        device=device,
        dtype=dtype,
    )
    references = torch.tensor(
        [[torch.pi, 0.0], [2.8, 0.1], [3.3, -0.1]], device=device, dtype=dtype
    )[:batch]
    cost_parameters = PendulumCostParameters.create(
        batch, x_ref=references, device=device, dtype=dtype
    )
    problem = DDPProblem.from_models(
        PendulumDynamics(parameters=dynamics_parameters),
        PendulumCost(parameters=cost_parameters),
        batch_size=batch,
        horizon=15,
        max_iterations=12,
        cost_tolerance=1e-7,
        gradient_tolerance=1e-7,
    )
    return problem, dynamics_parameters, cost_parameters


def test_parameterized_model_batch_matches_single_environment_models():
    problem, dynamics_parameters, cost_parameters = parameterized_problem()
    x = torch.tensor([[0.2, 0.1], [-0.4, 0.3], [0.8, -0.2]], dtype=torch.float64)
    u = torch.tensor([[0.1], [-0.2], [0.3]], dtype=torch.float64)
    next_batch = problem.dynamics.calc(x, u)
    derivatives_batch = problem.dynamics.calc_diff(x, u)
    cost_batch = problem.cost.calc(x, u)
    cost_derivatives_batch = problem.cost.calc_diff(x, u)

    for index in range(problem.batch_size):
        dynamics = PendulumDynamics(
            parameters=PendulumDynamicsParameters(
                **{
                    name: getattr(dynamics_parameters, name)[index : index + 1]
                    for name in ("mass", "length", "damping", "gravity")
                }
            )
        )
        cost = PendulumCost(
            parameters=PendulumCostParameters(
                **{
                    name: getattr(cost_parameters, name)[index : index + 1]
                    for name in ("x_ref", "Q", "R", "Q_terminal")
                }
            )
        )
        torch.testing.assert_close(
            next_batch[index], dynamics.calc(x[index : index + 1], u[index : index + 1])[0]
        )
        for batched, single in zip(
            derivatives_batch, dynamics.calc_diff(x[index : index + 1], u[index : index + 1])
        ):
            torch.testing.assert_close(batched[index], single[0])
        torch.testing.assert_close(
            cost_batch[index], cost.calc(x[index : index + 1], u[index : index + 1])[0]
        )
        for batched, single in zip(
            cost_derivatives_batch, cost.calc_diff(x[index : index + 1], u[index : index + 1])
        ):
            if batched is not None and single is not None:
                torch.testing.assert_close(batched[index], single[0])


def test_parameterized_ddp_batch_matches_separate_solves():
    problem, dynamics_parameters, cost_parameters = parameterized_problem()
    x0 = torch.tensor([[0.2, 0.0], [0.5, -0.1], [-0.4, 0.2]], dtype=torch.float64)
    batched = BatchedMPC(problem).solve(x0)
    for index in range(problem.batch_size):
        dynamics = PendulumDynamics(
            parameters=PendulumDynamicsParameters(
                **{
                    name: getattr(dynamics_parameters, name)[index : index + 1].clone()
                    for name in ("mass", "length", "damping", "gravity")
                }
            )
        )
        cost = PendulumCost(
            parameters=PendulumCostParameters(
                **{
                    name: getattr(cost_parameters, name)[index : index + 1].clone()
                    for name in ("x_ref", "Q", "R", "Q_terminal")
                }
            )
        )
        single_problem = DDPProblem.from_models(
            dynamics,
            cost,
            batch_size=1,
            horizon=problem.horizon,
            max_iterations=problem.max_iterations,
            cost_tolerance=problem.cost_tolerance,
            gradient_tolerance=problem.gradient_tolerance,
        )
        single = BatchedMPC(single_problem).solve(x0[index : index + 1])
        assert batched.status[index] == single.status[0]
        assert batched.iterations[index] == single.iterations[0]
        torch.testing.assert_close(batched.xs[index], single.xs[0], atol=1e-9, rtol=1e-9)
        torch.testing.assert_close(batched.us[index], single.us[0], atol=1e-9, rtol=1e-9)
        torch.testing.assert_close(batched.cost[index], single.cost[0], atol=1e-9, rtol=1e-9)


def test_masked_update_changes_parameters_and_invalidates_only_selected_cache():
    problem, dynamics_parameters, cost_parameters = parameterized_problem()
    base = torch.zeros(problem.batch_size, problem.horizon, problem.nu, dtype=problem.dtype)
    problem = DDPProblem.from_models(
        problem.dynamics,
        problem.cost,
        batch_size=problem.batch_size,
        horizon=problem.horizon,
        max_iterations=problem.max_iterations,
        u_init=base,
    )
    backend = RecordingBackend()
    controller = MPCController(BatchedMPC(problem, backend=backend))
    state = torch.zeros(problem.batch_size, problem.nx, dtype=problem.dtype)
    controller.compute(state)
    before_mass = dynamics_parameters.mass.clone()
    before_reference = cost_parameters.x_ref.clone()
    mask = torch.tensor([False, True, False])
    controller.update_parameters(
        mask,
        dynamics={"mass": torch.tensor([1.7], dtype=problem.dtype)},
        cost={"x_ref": torch.tensor([2.5, 0.2], dtype=problem.dtype)},
    )
    controller.compute(state)

    torch.testing.assert_close(dynamics_parameters.mass[[0, 2]], before_mass[[0, 2]])
    torch.testing.assert_close(cost_parameters.x_ref[[0, 2]], before_reference[[0, 2]])
    torch.testing.assert_close(dynamics_parameters.mass[1], torch.tensor(1.7, dtype=problem.dtype))
    torch.testing.assert_close(
        cost_parameters.x_ref[1], torch.tensor([2.5, 0.2], dtype=problem.dtype)
    )
    expected_initial = torch.ones_like(base)
    expected_initial[1] = base[1]
    torch.testing.assert_close(backend.initial_controls[1], expected_initial)


def test_failed_parameter_update_still_invalidates_affected_cache():
    problem, _, _ = parameterized_problem()
    backend = RecordingBackend()
    controller = MPCController(BatchedMPC(problem, backend=backend))
    state = torch.zeros(problem.batch_size, problem.nx, dtype=problem.dtype)
    controller.compute(state)
    mask = torch.tensor([True, False, False])
    with pytest.raises(ValueError, match="unknown parameters"):
        controller.update_parameters(mask, dynamics={"unknown": 1.0})
    controller.compute(state)
    assert backend.initial_controls[1][0].eq(0).all()
    assert backend.initial_controls[1][1:].eq(1).all()


def test_problem_rejects_parameter_batch_mismatch():
    problem, _, _ = parameterized_problem(batch=2)
    with pytest.raises(ValueError, match="parameter batch"):
        DDPProblem.from_models(problem.dynamics, problem.cost, batch_size=3, horizon=4)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_parameterized_models_match_cpu_and_cuda():
    cpu, _, _ = parameterized_problem(device="cpu")
    gpu, _, _ = parameterized_problem(device="cuda")
    x = torch.tensor([[0.2, 0.1], [-0.4, 0.3], [0.8, -0.2]], dtype=torch.float64)
    u = torch.tensor([[0.1], [-0.2], [0.3]], dtype=torch.float64)
    torch.testing.assert_close(cpu.dynamics.calc(x, u), gpu.dynamics.calc(x.cuda(), u.cuda()).cpu())
    torch.testing.assert_close(cpu.cost.calc(x, u), gpu.cost.calc(x.cuda(), u.cuda()).cpu())
