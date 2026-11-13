"""Shared, timestamped CPU task references. No planner or simulator imports.

SI units; poses and task linear/angular derivatives are world-aligned at the
named frame origin. Joint arrays use FL/FR/RL/RR then Z1 J1..J6. Reference forces
are world vectors in FL/FR/RL/RR order. This first contract fixes all four feet.
"""

from dataclasses import dataclass, fields, replace

import numpy as np


def _array(value, shape, name):
    value = np.array(value, dtype=float, copy=True)
    if value.shape != shape or not np.isfinite(value).all():
        raise ValueError(f"{name} must be finite with shape {shape}")
    value.setflags(write=False)
    return value


def _pose(value, shape, name):
    value = _array(value, shape, name)
    rotation = value[..., :3, :3]
    if (
        not np.allclose(value[..., 3, :], [0, 0, 0, 1], atol=1e-10, rtol=0)
        or not np.allclose(rotation @ rotation.swapaxes(-1, -2), np.eye(3), atol=1e-8, rtol=0)
        or not np.allclose(np.linalg.det(rotation), 1, atol=1e-8, rtol=0)
    ):
        raise ValueError(f"{name} requires proper SE(3) poses")
    return value


def quintic(p0, v0, a0, p1, v1, a1, duration, elapsed):
    """C2 Hermite interpolation with analytic first and second derivatives."""
    s = np.clip(elapsed / duration, 0, 1)
    c0, c1, c2 = p0, duration * v0, 0.5 * duration**2 * a0
    d = p1 - c0 - c1 - c2
    dv = duration * v1 - c1 - 2 * c2
    da = duration**2 * a1 - 2 * c2
    c3, c4, c5 = 10 * d - 4 * dv + 0.5 * da, -15 * d + 7 * dv - da, 6 * d - 3 * dv + 0.5 * da
    p = c0 + s * (c1 + s * (c2 + s * (c3 + s * (c4 + s * c5))))
    v = (c1 + s * (2 * c2 + s * (3 * c3 + s * (4 * c4 + s * 5 * c5)))) / duration
    a = (2 * c2 + s * (6 * c3 + s * (12 * c4 + s * 20 * c5))) / duration**2
    return p, v, a


def _left_jacobian(r):
    x, y, z = r
    skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    angle = np.linalg.norm(r)
    if angle < 1e-4:
        a = 0.5 - angle**2 / 24 + angle**4 / 720
        b = 1 / 6 - angle**2 / 120 + angle**4 / 5040
    else:
        a, b = (1 - np.cos(angle)) / angle**2, (angle - np.sin(angle)) / angle**3
    return np.eye(3) + a * skew + b * skew @ skew


def _jacobian_derivative(r, velocity):
    h = 1e-5 / max(1.0, np.linalg.norm(velocity))
    return (_left_jacobian(r + h * velocity) - _left_jacobian(r - h * velocity)) / (2 * h)


def pose_interpolate(p0, v0, a0, p1, v1, a1, duration, elapsed):
    """World-frame C2 pose interpolation, including SO(3) angular derivatives."""
    import pinocchio as pin

    position, linear_v, linear_a = quintic(
        p0[:3, 3], v0[:3], a0[:3], p1[:3, 3], v1[:3], a1[:3], duration, elapsed
    )
    r1 = pin.log3(p1[:3, :3] @ p0[:3, :3].T)
    if np.linalg.norm(r1) >= np.pi - 1e-5:
        raise ValueError("adjacent rotations must be less than pi apart")
    j1 = _left_jacobian(r1)
    d1 = np.linalg.solve(j1, v1[3:])
    dd1 = np.linalg.solve(j1, a1[3:] - _jacobian_derivative(r1, d1) @ d1)
    r, dr, ddr = quintic(np.zeros(3), v0[3:], a0[3:], r1, d1, dd1, duration, elapsed)
    jac = _left_jacobian(r)
    pose = np.eye(4)
    pose[:3, :3], pose[:3, 3] = pin.exp3(r) @ p0[:3, :3], position
    return (
        pose,
        np.r_[linear_v, jac @ dr],
        np.r_[linear_a, jac @ ddr + _jacobian_derivative(r, dr) @ dr],
    )


@dataclass(frozen=True)
class GuidanceSample:
    time: float
    model_id: str
    source: str
    foot_anchors: np.ndarray
    tcp_pose: np.ndarray
    tcp_velocity: np.ndarray
    tcp_acceleration: np.ndarray
    com_position: np.ndarray
    com_velocity: np.ndarray
    com_acceleration: np.ndarray
    base_pose: np.ndarray
    base_velocity: np.ndarray
    base_acceleration: np.ndarray
    joint_position: np.ndarray
    joint_velocity: np.ndarray
    joint_acceleration: np.ndarray
    contact_forces: np.ndarray

    def __post_init__(self):
        if not np.isfinite(self.time) or not self.model_id or not self.source:
            raise ValueError("sample requires finite time, model_id and source")
        for name, shape in SAMPLE_SHAPES.items():
            fn = _pose if name.endswith("pose") else _array
            object.__setattr__(self, name, fn(getattr(self, name), shape, name))


SAMPLE_SHAPES = {
    "foot_anchors": (4, 3),
    "tcp_pose": (4, 4),
    "tcp_velocity": (6,),
    "tcp_acceleration": (6,),
    "com_position": (3,),
    "com_velocity": (3,),
    "com_acceleration": (3,),
    "base_pose": (4, 4),
    "base_velocity": (6,),
    "base_acceleration": (6,),
    "joint_position": (18,),
    "joint_velocity": (18,),
    "joint_acceleration": (18,),
    "contact_forces": (4, 3),
}


@dataclass(frozen=True)
class Guidance:
    """Immutable packet; absolute monotonic timestamps and explicit validity."""

    times: np.ndarray
    samples: tuple[GuidanceSample, ...]
    issued_at: float
    valid_until: float
    sequence: int = 0
    valid: bool = True
    reason: str = ""

    def __post_init__(self):
        samples = tuple(self.samples)
        n = len(samples)
        times = _array(self.times, (n,), "times")
        if n < 2 or np.any(np.diff(times) <= 0):
            raise ValueError("Guidance requires at least two strictly increasing timestamps")
        if (
            not np.isfinite([self.issued_at, self.valid_until]).all()
            or self.issued_at > times[0]
            or not times[0] <= self.valid_until <= times[-1]
        ):
            raise ValueError("invalid issue time or validity interval")
        if type(self.sequence) is not int or self.sequence < 0 or type(self.valid) is not bool:
            raise ValueError("invalid sequence or validity flag")
        first = samples[0]
        for t, sample in zip(times, samples):
            if not isinstance(sample, GuidanceSample) or abs(t - sample.time) > 1e-9:
                raise ValueError("sample timestamp does not match knot")
            if sample.model_id != first.model_id or sample.source != first.source:
                raise ValueError("all knots require the same model and source")
            if not np.array_equal(sample.foot_anchors, first.foot_anchors):
                raise ValueError("fixed-support Guidance cannot change foot anchors")
        for previous, following in zip(samples[:-1], samples[1:]):
            for name in ("tcp_pose", "base_pose"):
                relative = getattr(following, name)[:3, :3] @ getattr(previous, name)[:3, :3].T
                if (np.trace(relative) - 1) / 2 <= np.cos(np.pi - 1e-5):
                    raise ValueError("adjacent rotations must be less than pi apart")
        object.__setattr__(self, "times", times)
        object.__setattr__(self, "samples", samples)

    def sample(self, now):
        if not self.valid:
            raise ValueError("invalid Guidance: " + self.reason)
        if not np.isfinite(now) or not self.times[0] <= now <= self.valid_until:
            raise ValueError("Guidance is not valid at requested time")
        k = min(int(np.searchsorted(self.times, now, side="right")) - 1, len(self.times) - 2)
        left, right = self.samples[k : k + 2]
        h, t = self.times[k + 1] - self.times[k], now - self.times[k]
        values = dict(
            time=float(now),
            model_id=left.model_id,
            source=left.source,
            foot_anchors=left.foot_anchors,
        )
        for prefix in ("tcp", "base"):
            p, v, a = pose_interpolate(
                *(getattr(left, prefix + "_" + s) for s in ("pose", "velocity", "acceleration")),
                *(getattr(right, prefix + "_" + s) for s in ("pose", "velocity", "acceleration")),
                h,
                t,
            )
            values.update(
                {prefix + "_pose": p, prefix + "_velocity": v, prefix + "_acceleration": a}
            )
        for prefix in ("com", "joint"):
            p, v, a = quintic(
                *(
                    getattr(left, prefix + "_" + s)
                    for s in ("position", "velocity", "acceleration")
                ),
                *(
                    getattr(right, prefix + "_" + s)
                    for s in ("position", "velocity", "acceleration")
                ),
                h,
                t,
            )
            values.update(
                {prefix + "_position": p, prefix + "_velocity": v, prefix + "_acceleration": a}
            )
        values["contact_forces"] = (1 - t / h) * left.contact_forces + t / h * right.contact_forces
        return GuidanceSample(**values)


def stationary(sample, *, time=None, source="standing"):
    """A stationary target; callers choose a previously validated support pose."""
    values = {field.name: getattr(sample, field.name) for field in fields(sample)}
    values.update(time=sample.time if time is None else time, source=source)
    for key in values:
        if key.endswith(("velocity", "acceleration")):
            values[key] = np.zeros_like(values[key])
    return GuidanceSample(**values)


class GuidanceBuffer:
    """Retain old valid packets; smoothly return to a configured standing target.

    Rejected packets never extend the old validity period. WBC failure handling
    remains with the execution loop, which must stop rather than apply zero torque.
    Optional handoffs join the outgoing sample to the new packet at now + duration,
    preserving C2 task derivatives and C0 forces, including interrupted handoffs.
    The default zero duration retains the original direct replacement behavior.
    """

    def __init__(self, standing, transition_duration=1.0, *, handoff_duration=0.0):
        if not np.isfinite(transition_duration) or transition_duration <= 0:
            raise ValueError("transition_duration must be positive")
        self.standing = stationary(standing)
        self.transition_duration = transition_duration
        if not np.isfinite(handoff_duration) or handoff_duration < 0:
            raise ValueError("handoff_duration must be finite and nonnegative")
        self.handoff_duration = handoff_duration
        self.reset()

    def reset(self):
        self.current = None
        self.transition = None
        self.handoff = None
        self.last_time = -np.inf
        self.rejected_packets = 0
        self.accepted_packets = 0
        self.last_rejection = None

    def submit(self, packet, now):
        if not np.isfinite(now) or now < self.last_time:
            raise ValueError("time cannot move backwards without reset")
        self.last_time = now
        first = packet.samples[0]
        checks = (
            (packet.valid, "invalid"),
            (packet.issued_at <= now and packet.times[0] <= now, "future"),
            (now <= packet.valid_until, "expired"),
            (first.model_id == self.standing.model_id, "model"),
            (np.array_equal(first.foot_anchors, self.standing.foot_anchors), "anchors"),
            (self.current is None or packet.sequence > self.current.sequence, "sequence"),
            (now + self.handoff_duration <= packet.valid_until, "handoff_coverage"),
        )
        reason = next((reason for passed, reason in checks if not passed), None)
        bridge = None
        if reason is None and self.handoff_duration > 0:
            outgoing, _ = self.sample(now)
            end = now + self.handoff_duration
            try:
                incoming = packet.sample(end)
                bridge = Guidance(
                    np.array([now, end]),
                    (replace(outgoing, source="handoff"), replace(incoming, source="handoff")),
                    now,
                    end,
                    packet.sequence,
                )
            except ValueError:
                reason = "handoff_geometry"
        self.last_rejection = reason
        if reason is not None:
            self.rejected_packets += 1
            return False
        self.current, self.transition, self.handoff = packet, None, bridge
        self.accepted_packets += 1
        return True

    def sample(self, now):
        if not np.isfinite(now) or now < self.last_time:
            raise ValueError("time cannot move backwards without reset")
        self.last_time = now
        if self.current is None:
            return replace(self.standing, time=now), "standing"
        if self.handoff is not None and now <= self.handoff.valid_until:
            return self.handoff.sample(now), "handoff"
        if now <= self.current.valid_until:
            return self.current.sample(now), "active"
        if self.transition is None:
            start = self.current.valid_until
            first = replace(self.current.sample(start), source="return_to_standing")
            end = start + self.transition_duration
            last = replace(self.standing, time=end, source=first.source)
            self.transition = Guidance(np.array([start, end]), (first, last), start, end)
        if now <= self.transition.valid_until:
            return self.transition.sample(now), "return_to_standing"
        return replace(self.standing, time=now), "standing"
