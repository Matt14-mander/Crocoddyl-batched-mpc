"""Four Go2 contact sequences, native optimization, hard-gated Torch execution."""

import numpy as np
import pytest
import torch

from crocoddyl_batched_mpc import SolveStatus
from crocoddyl_batched_mpc.models.go2_gaits import (
    GO2_GAIT_SWINGS,
    Go2GaitController,
    Go2GaitDynamics,
    Go2GaitPlan,
    build_go2_gait_problem,
    load_go2_gait_plan,
    solve_go2_gait,
)


def standing_plan():
    model = Go2GaitDynamics().model
    state = model.standing_state
    return Go2GaitPlan(
        "walk",
        state.expand(3, -1).clone(),
        model.quasi_static_torques().expand(2, -1).clone(),
        state.new_zeros(2, 12, 36),
        torch.ones(2, 4, dtype=torch.bool),
        torch.zeros(2, 4, dtype=torch.bool),
        model.contact_positions.expand(2, -1, -1).clone(),
        state.new_full((2,), 0.02),
        (),
        (True,),
        1,
        0.02,
    )


def test_export_load_and_torch_only_batch_execution(tmp_path):
    plan = standing_plan()
    path = tmp_path / "gait.pt"
    plan.save(path)
    imported = load_go2_gait_plan(path)
    assert imported.native_solvers == () and imported.gait == "walk"
    controller = Go2GaitController(imported, batch_size=2)
    state = imported.states[0].expand(2, -1).clone()
    for _ in range(2):
        out = controller.compute(state)
        assert out.result.usable.all()
        state = out.result.xs[:, 1].clone()
    assert out.finished.all()
    assert not controller.compute(state).result.usable.any()
    torch.testing.assert_close(state, imported.states[-1].expand(2, -1), atol=1e-10, rtol=1e-10)


def test_invalid_observation_failure_is_sticky_and_selective_reset():
    plan = standing_plan()
    controller = Go2GaitController(plan, batch_size=2)
    state = plan.states[:1].expand(2, -1).clone()
    state[0, 3:7] = 0
    out = controller.compute(state)
    assert out.result.status.tolist() == [SolveStatus.NUMERICAL_FAILURE, SolveStatus.SUCCESS]
    assert torch.isnan(out.action[0]).all() and out.result.usable[1]
    assert controller.indices.tolist() == [0, 1]
    good = plan.states[:1].expand(2, -1).clone()
    out = controller.compute(good)
    assert out.result.usable.tolist() == [False, True]
    cache = controller._warm[1].clone()
    controller.reset(torch.tensor([True, False]))
    assert controller.indices.tolist() == [0, 2] and not controller.failed.any()
    assert torch.equal(cache, controller._warm[1])
    assert controller.compute(good).result.usable.tolist() == [True, False]


def test_invalid_plan_metadata_and_masks_are_rejected():
    plan = standing_plan()
    plan.durations[0] = -0.01
    with pytest.raises(ValueError, match="duration"):
        Go2GaitController(plan)
    plan = standing_plan()
    plan.contacts[0] = False
    with pytest.raises(ValueError, match="support"):
        Go2GaitController(plan)
    plan = standing_plan()
    plan.states[0, 0] = float("nan")
    with pytest.raises(ValueError, match="tensors"):
        Go2GaitController(plan)
    with pytest.raises(ValueError, match="gait"):
        build_go2_gait_problem("gallop")
    with pytest.raises(ValueError, match="positive"):
        solve_go2_gait("walk", cycles=0)


def test_torque_feedback_is_constrained_by_qp_or_rejected():
    plan = standing_plan()
    plan.torques[0] += 100.0
    controller = Go2GaitController(plan)
    state = plan.states[:1].clone()
    out = controller.compute(state)
    assert out.result.iterations.item() > 0
    if out.result.usable.item():
        model = controller.dynamics.model
        assert (out.action.abs() <= model.torque_limits + 1e-7).all()
        _, forces, _, _ = controller.dynamics.evaluate(
            state, out.action, out.contacts, out.durations, plan.anchors[:1]
        )
        assert (
            (forces @ controller.matrix.T) - forces.new_tensor([0, 0, 0, 0, 0, 200])
        ).max() <= 1e-7
    else:
        assert torch.isnan(out.action).all()


@pytest.fixture(scope="module", params=tuple(GO2_GAIT_SWINGS))
def native_plan(request):
    pytest.importorskip("crocoddyl")
    pytest.importorskip("pinocchio")
    return solve_go2_gait(request.param, iterations=150)


@pytest.mark.crocoddyl
def test_example_sequence_and_native_torch_contact_impulse_oracle(native_plan):
    plan = native_plan
    groups = [
        tuple(leg for leg, active in zip(("FL", "FR", "RL", "RR"), row) if active)
        for row in plan.landings
        if row.any()
    ]
    assert [set(group) for group in groups] == [set(group) for group in GO2_GAIT_SWINGS[plan.gait]]
    assert (plan.contacts[plan.durations == 0].sum(-1) == 4).all()
    assert (plan.torques[plan.durations == 0] == 0).all()
    flow_count = 3 if plan.gait == "walk" else 2
    assert flow_count in plan.contacts[plan.durations > 0].sum(-1)
    dynamics = Go2GaitDynamics(dt=plan.dt)
    native_models = [m for solver in plan.native_solvers for m in solver.problem.runningModels]
    for t, native in enumerate(native_models):
        x = plan.states[t : t + 1]
        u = plan.torques[t : t + 1]
        out, forces, _, _ = dynamics.evaluate(
            x, u, plan.contacts[t : t + 1], plan.durations[t : t + 1], plan.anchors[t : t + 1]
        )
        data = native.createData()
        if native.nu == 0:
            native.calc(data, x[0].numpy())
            entries = data.multibody.impulses.impulses.todict()
            suffix = "_impulse"
            np.testing.assert_array_equal(data.xnext[:19], x[0, :19].numpy())
        else:
            native.calc(data, x[0].numpy(), u[0].numpy())
            entries = data.differential.multibody.contacts.contacts.todict()
            suffix = "_contact"
        np.testing.assert_allclose(out[0].numpy(), data.xnext, atol=1e-9, rtol=1e-9)
        np.testing.assert_allclose(out[0].numpy(), plan.states[t + 1].numpy(), atol=1e-9, rtol=1e-9)
        for i, leg in enumerate(("FL", "FR", "RL", "RR")):
            name = leg + "_foot" + suffix
            expected = entries[name].f.linear if name in entries else np.zeros(3)
            np.testing.assert_allclose(forces[0, i].numpy(), expected, atol=1e-9, rtol=1e-9)


@pytest.mark.crocoddyl
def test_all_gaits_native_batch_feedback_forward_motion_and_hard_gates(native_plan):
    from examples.go2_quadrupedal_gaits import run_gait

    report = run_gait(native_plan.gait, prepared_plan=native_plan)
    assert report["all_actions_usable"] and report["all_finished"]
    assert min(report["forward_displacement_m"]) > 0.015
    assert max(report["final_tracking_error"]) < 0.005
    assert min(min(row) for row in report["peak_swing_height_m"]) > 0.005
    assert report["minimum_foot_height_m"] >= -0.002
    assert report["max_force_violation_N"] <= 1e-7
    assert report["max_torque_violation_Nm"] <= 1e-7
    assert report["max_impulse_violation_Ns"] <= 1e-7
    assert report["max_native_dynamics_error"] < 1e-9


@pytest.mark.crocoddyl
def test_impulse_force_guide_jacobian_against_finite_differences():
    pytest.importorskip("crocoddyl")
    pytest.importorskip("pinocchio")
    problem = build_go2_gait_problem("trot", step_knots=4, support_knots=2)
    model = next(m for m in problem.runningModels if m.nu == 0)
    x = problem.x0.copy()
    x[19:] = np.linspace(-0.1, 0.1, 18)
    data = model.createData()
    model.calc(data, x)
    model.calcDiff(data, x)
    name = next(name for name in model.costs.costs.todict() if name.endswith("_forceGuide"))
    jacobian = data.costs.costs[name].residual.Rx.copy()
    finite = []
    for j in range(36):
        delta = np.zeros(36)
        delta[j] = 1e-6
        plus, minus = model.createData(), model.createData()
        model.calc(plus, model.state.integrate(x, delta))
        model.calc(minus, model.state.integrate(x, -delta))
        finite.append(
            (plus.costs.costs[name].residual.r.copy() - minus.costs.costs[name].residual.r.copy())
            / 2e-6
        )
    np.testing.assert_allclose(jacobian, np.asarray(finite).T, atol=1e-8, rtol=1e-8)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_exported_gait_on_cuda_nondefault_stream():
    plan = standing_plan().to("cuda")
    controller = Go2GaitController(plan, batch_size=2)
    state = plan.states[:1].expand(2, -1).clone()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        out = controller.compute(state)
        consumer = out.action.square().sum()
    stream.synchronize()
    assert out.result.usable.all() and torch.isfinite(consumer)
