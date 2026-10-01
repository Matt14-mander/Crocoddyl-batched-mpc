"""Tensor-only Lie group operations, with xyzw quaternions and body increments."""

import torch
from torch import Tensor


def skew(v: Tensor) -> Tensor:
    x, y, z = v.unbind(-1)
    zero = torch.zeros_like(x)
    return torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), -1).reshape(*v.shape[:-1], 3, 3)


def quaternion_multiply(a: Tensor, b: Tensor) -> Tensor:
    a, b = torch.broadcast_tensors(a, b)
    av, aw, bv, bw = a[..., :3], a[..., 3:], b[..., :3], b[..., 3:]
    return torch.cat(
        (aw * bv + bw * av + torch.linalg.cross(av, bv), aw * bw - (av * bv).sum(-1, keepdim=True)),
        -1,
    )


def quaternion_conjugate(q: Tensor) -> Tensor:
    return torch.cat((-q[..., :3], q[..., 3:]), -1)


def quaternion_normalize(q: Tensor) -> Tensor:
    return q / torch.linalg.vector_norm(q, dim=-1, keepdim=True)


def quaternion_exp(v: Tensor) -> Tensor:
    squared = v.square().sum(-1, keepdim=True)
    angle = squared.clamp_min(1e-16).sqrt()
    scale = torch.where(
        squared < 1e-6, 0.5 - squared / 48 + squared.square() / 3840, torch.sin(angle / 2) / angle
    )
    scalar = torch.where(
        squared < 1e-6, 1 - squared / 8 + squared.square() / 384, torch.cos(angle / 2)
    )
    return torch.cat((scale * v, scalar), -1)


def quaternion_log(q: Tensor) -> Tensor:
    q = quaternion_normalize(q)
    q = torch.where(q[..., 3:] < 0, -q, q)
    vector, scalar = q[..., :3], q[..., 3:]
    squared = vector.square().sum(-1, keepdim=True)
    length = squared.clamp_min(1e-16).sqrt()
    scale = torch.where(
        squared < 1e-6,
        2 + squared / 3 + 3 * squared.square() / 20,
        2 * torch.atan2(length, scalar) / length,
    )
    return scale * vector


def quaternion_matrix(q: Tensor) -> Tensor:
    q = quaternion_normalize(q)
    v, w = q[..., :3], q[..., 3:]
    hat = skew(v)
    eye = torch.eye(3, device=q.device, dtype=q.dtype)
    return eye + 2 * w[..., None] * hat + 2 * hat @ hat


def so3_left_jacobian(v: Tensor) -> Tensor:
    squared = v.square().sum(-1)[..., None, None]
    angle = squared.clamp_min(1e-16).sqrt()
    a = torch.where(
        squared < 1e-6,
        0.5 - squared / 24 + squared.square() / 720,
        (1 - angle.cos()) / squared.clamp_min(1e-16),
    )
    b = torch.where(
        squared < 1e-6,
        1 / 6 - squared / 120 + squared.square() / 5040,
        (angle - angle.sin()) / (squared.clamp_min(1e-16) * angle),
    )
    hat = skew(v)
    return torch.eye(3, device=v.device, dtype=v.dtype) + a * hat + b * hat @ hat


def so3_left_jacobian_inverse(v: Tensor) -> Tensor:
    squared = v.square().sum(-1)[..., None, None]
    angle = squared.clamp_min(1e-16).sqrt()
    c = torch.where(
        squared < 1e-6,
        1 / 12 + squared / 720 + squared.square() / 30240,
        (1 - (angle / 2) / torch.tan(angle / 2)) / squared.clamp_min(1e-16),
    )
    hat = skew(v)
    return torch.eye(3, device=v.device, dtype=v.dtype) - 0.5 * hat + c * hat @ hat


def se3_right_jacobian_inverse(v: Tensor) -> Tensor:
    """Inverse dexp for [linear, angular]; principal log angle is at most pi.

    Expand the right Jacobian then invert it with an asynchronous Torch operation.
    The 24-term series covers the principal log chart in float32/float64.
    """
    linear, angular = skew(v[..., :3]), skew(v[..., 3:])
    zero = torch.zeros_like(linear)
    ad = torch.cat((torch.cat((angular, linear), -1), torch.cat((zero, angular), -1)), -2)
    term = torch.eye(6, device=v.device, dtype=v.dtype).expand(*v.shape[:-1], 6, 6)
    jacobian = term.clone()
    for n in range(1, 25):
        term = term @ (-ad) / (n + 1)
        jacobian = jacobian + term
    return torch.linalg.inv_ex(jacobian, check_errors=False)[0]
