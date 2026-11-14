"""Batch validation with the same GuidanceSample contract and detached storage.

Dataclass allocation bypasses __init__ only after every field has passed the
original shape, finite and SE(3) checks. Guidance's packet-level validator still
runs. No checksum, type tag or sender assertion substitutes for validation.
"""

import numpy as np

from .guidance import SAMPLE_SHAPES, Guidance, GuidanceSample, _pose


def assemble_guidance(
    times,
    columns,
    *,
    model_id,
    source,
    issued_at,
    valid_until,
    sequence=0,
    valid=True,
    reason="",
    sample_times=None,
):
    times = np.array(times, dtype=float, copy=True)
    if times.ndim != 1 or not np.isfinite(times).all() or not model_id or not source:
        raise ValueError("samples require finite times, model_id and source")
    if set(columns) != set(SAMPLE_SHAPES):
        raise ValueError("Guidance requires every sample field")
    n = len(times)
    sample_times = np.array(times if sample_times is None else sample_times, dtype=float)
    if sample_times.shape != (n,) or not np.isfinite(sample_times).all():
        raise ValueError("invalid sample times")
    arrays = {}
    for name, shape in SAMPLE_SHAPES.items():
        array = np.array(columns[name], dtype=float, copy=True)
        if array.shape != (n, *shape) or not np.isfinite(array).all():
            raise ValueError(f"invalid Guidance field: {name}")
        if name.endswith("pose"):
            # Original pose validator already accepts leading batch dimensions.
            array = _pose(array, (n, *shape), name)
        array.setflags(write=False)
        arrays[name] = array
    samples = []
    for k, t in enumerate(sample_times):
        sample = object.__new__(GuidanceSample)
        for name, value in dict(
            time=float(t),
            model_id=model_id,
            source=source,
            **{name: array[k] for name, array in arrays.items()},
        ).items():
            object.__setattr__(sample, name, value)
        samples.append(sample)
    return Guidance(times, tuple(samples), issued_at, valid_until, sequence, valid, reason)


def copy_guidance(packet):
    """Revalidate an IPC packet, including per-knot metadata before aggregation."""
    if not isinstance(packet, Guidance) or len(packet.samples) < 2:
        raise ValueError("expected a complete Guidance packet")
    first = packet.samples[0]
    if not isinstance(first, GuidanceSample):
        raise ValueError("invalid Guidance sample type")
    times = np.asarray(packet.times)
    if times.shape != (len(packet.samples),):
        raise ValueError("sample count differs from times")
    for t, s in zip(times, packet.samples):
        if (
            not isinstance(s, GuidanceSample)
            or not np.isfinite(s.time)
            or abs(t - s.time) > 1e-9
            or s.model_id != first.model_id
            or s.source != first.source
        ):
            raise ValueError("inconsistent sample metadata")
    return assemble_guidance(
        times,
        {name: np.stack([getattr(s, name) for s in packet.samples]) for name in SAMPLE_SHAPES},
        model_id=first.model_id,
        source=first.source,
        issued_at=packet.issued_at,
        valid_until=packet.valid_until,
        sequence=packet.sequence,
        valid=packet.valid,
        reason=packet.reason,
        sample_times=[s.time for s in packet.samples],
    )
