"""Integration contract and task-level acceptance for the Torch-only Go2 API."""

import subprocess
import sys
from dataclasses import replace

import pytest
import torch

from crocoddyl_batched_mpc.models import (
    Go2TrackingMPC,
    go2_front_leg_tracking_plan,
)


def make_case(device="cpu", batch=2):
    controller = Go2TrackingMPC(batch, device=device, compile_physics=False)
    plan = go2_front_leg_tracking_plan(controller.model)
    return controller, plan


def test_no_optional_native_dependency_is_imported():
    code = """
import builtins
import torch
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in {'numpy', 'pinocchio', 'crocoddyl', 'meshcat'}:
        raise AssertionError('unexpected optional dependency: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
from crocoddyl_batched_mpc.models import Go2TrackingMPC, go2_front_leg_tracking_plan
torch.set_num_threads(1)
c = Go2TrackingMPC(compile_physics=False)
p = go2_front_leg_tracking_plan(c.model)
for step in (0, p.phases.index('reach')):
    out = c.compute(p.reference[step:step+1], p.command(step, 1, 2))
    assert out.usable.all()
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_compiled_physics_support_masks_remain_dynamic():
    eager, plan = make_case()
    compiled = Go2TrackingMPC(2, compile_physics=True)
    for step in (0, plan.phases.index("reach")):
        state = plan.reference[step].expand(2, -1).clone()
        command = plan.command(step, 2, 2)
        expected = eager.compute(state, command)
        actual = compiled.compute(state, command)
        assert actual.usable.all()
        # QP solutions near zero torque vary slightly with JIT BLAS fusion.
        torch.testing.assert_close(actual.action, expected.action, atol=2e-6, rtol=2e-6)
        torch.testing.assert_close(actual.result.xs, expected.result.xs, atol=2e-8, rtol=2e-8)


def test_stance_and_front_leg_closed_loop():
    from examples.go2_front_leg_tracking import run_tracking

    for task in ("stance", "front-leg"):
        report = run_tracking(task=task, compile_physics=False)
        assert report["passed"], report
        assert report["all_actions_usable"]
        assert report["max_support_force_violation_N"] <= 1e-7


@pytest.mark.parametrize("field", ["state", "reference", "acceleration", "anchors"])
def test_invalid_slot_is_finite_and_sticky_until_masked_reset(field):
    controller, plan = make_case()
    command = plan.command(0, 2, 2)
    state = plan.reference[0].expand(2, -1).clone()
    if field == "state":
        state[0, 0] = float("nan")
    else:
        bad = getattr(command, field).clone()
        bad[0] = float("nan")
        command = replace(command, **{field: bad})
    output = controller.compute(state, command)
    assert output.usable.tolist() == [False, True]
    assert output.needs_reset.tolist() == [True, False]
    assert output.action[0].eq(0).all() and torch.isfinite(output.action).all()
    assert not output.result.usable[0] and output.result.xs[0].isnan().all()
    command = plan.command(0, 2, 2)
    state = plan.reference[0].expand(2, -1).clone()
    assert controller.compute(state, command).usable.tolist() == [False, True]
    saved = tuple(x[1].clone() for x in (controller.mpc._warm, controller.mpc._dual))
    controller.reset(torch.tensor([True, False]))
    for value, before in zip((controller.mpc._warm, controller.mpc._dual), saved):
        torch.testing.assert_close(value[1], before, atol=0, rtol=0)
    assert controller.compute(state, command).usable.all()


def test_per_environment_support_release_and_no_implicit_touchdown():
    controller, plan = make_case()
    step = plan.phases.index("reach")
    command = plan.command(step, 2, 2)
    reference = command.reference.clone()
    reference[1] = plan.reference[0]
    supports = command.support_mask.clone()
    supports[1] = True
    command = replace(command, reference=reference, support_mask=supports)
    state = reference[:, 0].clone()
    output = controller.compute(state, command)
    assert output.usable.all()
    assert output.support_mask.tolist() == [[False, True, True, True], [True] * 4]
    force = controller.model.contact_dynamics(
        state, output.action, contact_mask=output.support_mask, contact_positions=command.anchors
    ).forces_world
    assert force[0, 0].eq(0).all()
    reattach = replace(command, support_mask=torch.ones_like(supports))
    output = controller.compute(state, reattach)
    assert output.usable.tolist() == [False, True]
    assert not output.support_mask[0, 0]
    controller.reset(torch.tensor([True, False]))
    assert controller.support_mask.all()


def test_invalid_support_anchor_change_and_dtype_rejected():
    controller, plan = make_case()
    command = plan.command(0, 2, 2)
    state = command.reference[:, 0].clone()
    controller.compute(state, command)
    anchors = command.anchors.clone()
    anchors[0, 1, 0] += 0.01
    output = controller.compute(state, replace(command, anchors=anchors))
    assert output.usable.tolist() == [False, True]
    controller.reset()
    mask = command.support_mask.clone()
    mask[0, 1] = False
    assert controller.compute(state, replace(command, support_mask=mask)).usable.tolist() == [
        False,
        True,
    ]
    with pytest.raises(ValueError, match="float64"):
        controller.compute(state.float(), command)
    with pytest.raises(ValueError, match="reset mask"):
        controller.reset(torch.ones(2))


@pytest.mark.skipif(torch.cuda.is_available(), reason="tests unavailable-CUDA path")
def test_cuda_acceptance_refuses_to_report_success_without_cuda(tmp_path):
    import json

    path = tmp_path / "acceptance.json"
    result = subprocess.run(
        [sys.executable, "examples/accept_go2_tracking_cuda.py", "--output", str(path)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2, result.stdout + result.stderr
    report = json.loads(path.read_text())
    assert report["status"] == "unavailable" and not report["cuda_available"]
    assert report["runs"] == []


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_tracking_cuda_parity_stream_and_no_scalar_sync():
    cpu, cpu_plan = make_case()
    gpu, gpu_plan = make_case("cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    for step in (0, gpu_plan.phases.index("reach")):
        reference = cpu_plan.reference[step].expand(2, -1).clone()
        expected = cpu.compute(reference, cpu_plan.command(step, 2, 2))
        command = gpu_plan.command(step, 2, 2)
        state = reference.to("cuda")
        stream.wait_stream(torch.cuda.current_stream())
        previous = torch.cuda.get_sync_debug_mode()
        try:
            torch.cuda.set_sync_debug_mode("error")
            with torch.cuda.stream(stream):
                actual = gpu.compute(state, command)
                consumed = actual.action.square().sum(-1)
        finally:
            torch.cuda.set_sync_debug_mode(previous)
        stream.synchronize()
        assert actual.action.is_cuda and actual.result.status.is_cuda
        torch.testing.assert_close(actual.usable.cpu(), expected.usable)
        torch.testing.assert_close(actual.action.cpu(), expected.action, atol=2e-6, rtol=2e-6)
        assert torch.isfinite(consumed).all()
    # Corrupt one observation on GPU: reject only that slot and keep finite output.
    bad = state.clone()
    bad[0, 0] = float("nan")
    stream.wait_stream(torch.cuda.current_stream())
    previous = torch.cuda.get_sync_debug_mode()
    try:
        torch.cuda.set_sync_debug_mode("error")
        with torch.cuda.stream(stream):
            rejected = gpu.compute(bad, command)
    finally:
        torch.cuda.set_sync_debug_mode(previous)
    stream.synchronize()
    assert rejected.usable.tolist() == [False, True]
    assert rejected.action[0].eq(0).all() and torch.isfinite(rejected.action).all()
    # Selective reset on the same non-default stream must leave slot 1 intact.
    torch.cuda.current_stream().wait_stream(stream)
    saved = gpu.mpc._warm[1].clone()
    reset = torch.tensor([True, False], device="cuda")
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        gpu.reset(reset)
    stream.synchronize()
    torch.testing.assert_close(gpu.mpc._warm[1], saved, atol=0, rtol=0)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_tracking_cuda_closed_loop():
    from examples.go2_front_leg_tracking import run_tracking

    for task in ("stance", "front-leg"):
        report = run_tracking(device="cuda", task=task, compile_physics=False)
        assert report["passed"], report
