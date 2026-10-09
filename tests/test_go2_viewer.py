"""Playback records actual states and preserves contact-event simulation time."""

import json

import pytest
import torch

from crocoddyl_batched_mpc.go2_viewer import Go2Trace, animation_samples, export_gallery


def test_selected_environment_snapshot_and_zero_time_impact_roundtrip(tmp_path):
    state = torch.zeros(2, 37, dtype=torch.float64)
    state[:, 6] = 1
    state[1, 0] = 0.25
    mask = torch.ones(2, 4, dtype=torch.bool)
    trace = Go2Trace("slow walk", environment=1)
    trace.append(state, 0, mask)
    state[1, 0] += 0.01
    mask[1, 2] = False
    trace.append(state, 0.02, mask)
    # Impact changes velocity/contact status at the very same configuration/time.
    state[1, 19] = 5
    mask[1, 2] = True
    trace.append(state, 0, mask)
    state[1, 0] += 0.01
    trace.append(state, 0.02, mask)
    state.fill_(99)
    mask.zero_()
    assert trace.times == [0, 0.02, 0.02, 0.04]
    assert trace.configurations[0][0] == 0.25
    assert not trace.contacts[1][2] and trace.contacts[2][2]
    assert animation_samples(trace.times) == [(0, 0), (1, 2), (2, 3)]
    path = tmp_path / "trace.json"
    trace.report = {"all_finished": True}
    trace.save(path)
    loaded = Go2Trace.load(path)
    assert loaded.environment == 1 and loaded.report == trace.report
    assert loaded.configurations == trace.configurations
    assert loaded.times == trace.times and loaded.contacts == trace.contacts


def test_trace_rejects_invalid_time_environment_and_payload(tmp_path):
    state = torch.zeros(1, 37)
    mask = torch.ones(1, 4, dtype=torch.bool)
    with pytest.raises(ValueError, match="environment"):
        Go2Trace("walk", 1).append(state, 0, mask)
    for duration in (0.02, -0.01, float("nan")):
        with pytest.raises(ValueError, match="duration"):
            Go2Trace("walk").append(state, duration, mask)
    trace = Go2Trace("walk")
    trace.append(state, 0, mask)
    with pytest.raises(ValueError, match="configuration"):
        trace.append(state.fill_(float("nan")), 0.02, mask)
    path = tmp_path / "trace.json"
    trace.save(path)
    value = json.loads(path.read_text())
    value["times"].append(1)
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="inconsistent"):
        Go2Trace.load(path)


def test_gallery_relative_pages_and_escaped_names(tmp_path):
    path = export_gallery({"walk <test>": tmp_path / "walk.html"}, tmp_path / "index.html")
    text = path.read_text()
    assert 'data-page="walk.html"' in text
    assert "walk &lt;test&gt;" in text
    assert "buttons[0].click()" in text
    with pytest.raises(ValueError, match="at least one"):
        export_gallery({}, tmp_path / "empty.html")
