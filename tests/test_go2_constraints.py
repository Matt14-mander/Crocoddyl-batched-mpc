import pytest
import torch

from crocoddyl_batched_mpc import SolveStatus
from crocoddyl_batched_mpc.models import Go2ConstrainedMPC, Go2ContactConstraints, Go2TorchDynamics
from examples.go2_constrained_mpc import run_constrained


def test_diamond_implies_circular_coulomb_and_positive_normal():
    m = Go2TorchDynamics()
    limits = Go2ContactConstraints(m, friction=0.4, minimum_normal_force=0.0)
    forces = torch.tensor(
        [[[4.0, 0.0, 10.0]] * 4, [[3.0, 3.0, 10.0]] * 4, [[0.0, 0.0, -1.0]] * 4], dtype=m.dtype
    )
    torque = torch.zeros(3, 12, dtype=m.dtype)
    assert limits.feasible(torque, forces).tolist() == [True, False, False]
    torque[0, 0] = m.torque_limits[0] + 1e-3
    assert not limits.feasible(torque, forces)[0]
    with pytest.raises(ValueError, match="URDF"):
        Go2ContactConstraints(m, torque_limits=m.torque_limits * 1.1)


def test_controller_exact_horizon_gate_and_failure_never_holds_unsafe_torque():
    c = Go2ConstrainedMPC(batch_size=3, horizon=2, compile_physics=False)
    x = c.reference.expand(3, -1).clone()
    u, result = c.compute(x)
    assert result.usable.all()
    for t in range(c.horizon):
        contact = c.dynamics.contact_dynamics(result.xs[:, t], result.us[:, t])
        assert c.constraints.feasible(result.us[:, t], contact.forces_world).all()
        torch.testing.assert_close(
            c.dynamics.calc(result.xs[:, t], result.us[:, t]), result.xs[:, t + 1]
        )
    # A valid prior action does not become a fallback for an invalid observation.
    old = c._warm[0].clone()
    x[1, 3:7] = 0
    u, result = c.compute(x)
    assert result.usable.tolist() == [True, False, True]
    assert torch.isnan(u[1]).all() and torch.isfinite(u[[0, 2]]).all()
    assert not c._valid[1]
    c.reset(torch.tensor([False, True, False]))
    torch.testing.assert_close(c._warm[0], old, atol=1e-5, rtol=1e-5)
    assert c._dual[1].eq(1).all()


def test_controller_rejects_infeasible_load_capacity():
    c = Go2ConstrainedMPC(
        batch_size=2, horizon=2, maximum_normal_force=5.0, force_margin=0.1, compile_physics=False
    )
    action, result = c.compute(c.reference.expand(2, -1))
    assert not result.usable.any()
    assert result.status.eq(SolveStatus.NO_FEASIBLE_CANDIDATE).all()
    assert torch.isnan(action).all()
    assert not c._valid.any()


def test_tight_torque_constraints_and_compiled_kernel_match_physics():
    m = Go2TorchDynamics()
    limits = (m.quasi_static_torques().abs() + 0.15).clamp_max(m.torque_limits)
    c = Go2ConstrainedMPC(batch_size=2, horizon=2, torque_limits=limits)
    delta = torch.zeros(2, 36, dtype=c.dtype)
    delta[:, 6:18] = 0.01
    delta[:, 18:] = 0.01
    x = c.dynamics.manifold.integrate(c.reference.expand(2, -1), delta)
    u, result = c.compute(x)
    assert result.usable.all()
    assert (result.us.abs() <= limits + 1e-7).all()
    expected = c.dynamics.contact_dynamics(x, u)
    compiled, forces = c._physics(x, u)
    torch.testing.assert_close(compiled, c.dynamics.calc(x, u), atol=1e-11, rtol=1e-11)
    torch.testing.assert_close(forces, expected.forces_world, atol=1e-10, rtol=1e-10)
    assert (limits - result.us.abs()).min() < 1e-4, "exercise an active torque face"


def test_active_friction_and_normal_force_faces():
    c = Go2ConstrainedMPC(horizon=2, friction=0.005, force_margin=1e-4, compile_physics=False)
    action, result = c.compute(c.reference[None])
    assert result.usable.all()
    force = c.dynamics.contact_dynamics(result.xs[:, 0], action).forces_world
    assert c.constraints.feasible(action, force).all()
    ratio = force[..., :2].abs().sum(-1) / force[..., 2]
    assert ratio.max() > 0.0049 and ratio.max() <= 0.005 + 1e-7
    c = Go2ConstrainedMPC(
        horizon=2, minimum_normal_force=39.5, force_margin=0.01, compile_physics=False
    )
    action, result = c.compute(c.reference[None])
    assert result.usable.all()
    forces = c.dynamics.contact_dynamics(result.xs[:, 0], action).forces_world
    assert c.constraints.feasible(action, forces).all()
    assert forces[..., 2].min() >= 39.5
    nominal = c.dynamics.contact_dynamics(c.reference[None], c.u_ref[None]).forces_world
    assert nominal[..., 2].min() < 39.5
    assert (forces[..., 2] - 39.5).min() < 0.011


def test_outside_local_model_is_reported_explicitly():
    c = Go2ConstrainedMPC(horizon=1, compile_physics=False)
    delta = c.reference.new_zeros(1, 36)
    delta[:, 0] = 0.4
    action, result = c.compute(c.dynamics.manifold.integrate(c.reference[None], delta))
    assert result.status[0] == SolveStatus.OUTSIDE_LOCAL_MODEL
    assert not result.usable.any() and torch.isnan(action).all()


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_constrained_cuda_stream_feasibility():
    c = Go2ConstrainedMPC(batch_size=2, device="cuda", horizon=2)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        action, result = c.compute(c.reference.expand(2, -1))
    stream.synchronize()
    assert result.usable.all() and torch.isfinite(action).all()


@pytest.mark.crocoddyl
def test_constrained_standing_against_independent_pinocchio_plant():
    pytest.importorskip("pinocchio")
    report = run_constrained(steps=120, plant="pinocchio")
    assert report["all_actions_usable"] and report["steps_completed"] == 120
    assert max(report["last_five_error_max"]) < 0.003
    assert report["max_torque_violation_Nm"] <= 1e-7
    assert report["max_force_violation_N"] <= 1e-7
    assert report["min_normal_force_N"] >= 1.0 - 1e-7
    assert report["max_diamond_ratio"] <= 0.6 + 1e-7
    print(report)
