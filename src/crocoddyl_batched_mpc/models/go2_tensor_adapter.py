"""Pure Torch adapter from simulator articulation tensors to Go2 tracking MPC."""

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from ..lie import quaternion_matrix
from .go2 import GO2_JOINT_NAMES
from .go2_tracking import Go2TrackingCommand, Go2TrackingMPC, Go2TrackingOutput


@dataclass(frozen=True)
class Go2TensorOutput:
    """MPC output in both simulator and internal layouts."""

    joint_effort: torch.Tensor  # [B,Nsim], simulator joint order
    mpc_state: torch.Tensor  # [B,37], Go2/Pinocchio order
    tracking: Go2TrackingOutput


class Go2TensorAdapter:
    """Map vectorized simulator tensors to the pure Torch Go2 controller.

    The adapter imports no Isaac Lab modules. It accepts the tensors exposed by
    an articulation and returns effort targets in the articulation's joint order.
    Root pose is world-frame `[position, quaternion]`; velocities may be world or
    body frame. Environment origins are removed from root position.
    """

    def __init__(
        self,
        joint_names: Sequence[str],
        batch_size: int = 1,
        *,
        quaternion_order: str = "wxyz",
        velocity_frame: str = "world",
        **controller_settings,
    ) -> None:
        names = tuple(joint_names)
        if len(names) == 0 or len(set(names)) != len(names):
            raise ValueError("joint_names must be a nonempty unique sequence")
        missing = [name for name in GO2_JOINT_NAMES if name not in names]
        if missing:
            raise ValueError(f"simulator is missing Go2 joints: {', '.join(missing)}")
        if quaternion_order not in ("xyzw", "wxyz"):
            raise ValueError("quaternion_order must be 'xyzw' or 'wxyz'")
        if velocity_frame not in ("body", "world"):
            raise ValueError("velocity_frame must be 'body' or 'world'")
        self.controller = Go2TrackingMPC(batch_size, **controller_settings)
        self.batch_size = batch_size
        self.sim_joint_count = len(names)
        self.quaternion_order = quaternion_order
        self.velocity_frame = velocity_frame
        self.device, self.dtype = self.controller.device, self.controller.dtype
        self.mpc_to_sim = torch.tensor(
            [names.index(name) for name in GO2_JOINT_NAMES],
            dtype=torch.int64,
            device=self.device,
        )

    def _validate_float(self, name: str, value: torch.Tensor, shape: tuple[int, ...]) -> None:
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if value.shape != shape or value.device != self.device or value.dtype != self.dtype:
            raise ValueError(
                f"{name} must have shape {shape}, device={self.device}, dtype={self.dtype}"
            )

    @torch.no_grad()
    def pack_state(
        self,
        root_pose_world: torch.Tensor,
        root_velocity: torch.Tensor,
        joint_position: torch.Tensor,
        joint_velocity: torch.Tensor,
        env_origins: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Pack articulation tensors into `[q(19), v_body(18)]`."""
        batch, joints = self.batch_size, self.sim_joint_count
        self._validate_float("root_pose_world", root_pose_world, (batch, 7))
        self._validate_float("root_velocity", root_velocity, (batch, 6))
        self._validate_float("joint_position", joint_position, (batch, joints))
        self._validate_float("joint_velocity", joint_velocity, (batch, joints))
        if env_origins is None:
            origins = root_pose_world.new_zeros(batch, 3)
        else:
            self._validate_float("env_origins", env_origins, (batch, 3))
            origins = env_origins
        quaternion = root_pose_world[:, 3:7]
        if self.quaternion_order == "wxyz":
            quaternion = quaternion[:, (1, 2, 3, 0)]
        base_velocity = root_velocity
        if self.velocity_frame == "world":
            world_to_body = quaternion_matrix(quaternion).transpose(-1, -2)
            base_velocity = torch.cat(
                (
                    (world_to_body @ root_velocity[:, :3, None]).squeeze(-1),
                    (world_to_body @ root_velocity[:, 3:, None]).squeeze(-1),
                ),
                -1,
            )
        return torch.cat(
            (
                root_pose_world[:, :3] - origins,
                quaternion,
                joint_position.index_select(1, self.mpc_to_sim),
                base_velocity,
                joint_velocity.index_select(1, self.mpc_to_sim),
            ),
            -1,
        )

    @torch.no_grad()
    def scatter_action(self, action: torch.Tensor) -> torch.Tensor:
        """Scatter `[B,12]` torque from MPC order into simulator joint order."""
        self._validate_float("action", action, (self.batch_size, 12))
        effort = action.new_zeros(self.batch_size, self.sim_joint_count)
        effort.index_copy_(1, self.mpc_to_sim, action)
        return effort

    @torch.no_grad()
    def compute(
        self,
        root_pose_world: torch.Tensor,
        root_velocity: torch.Tensor,
        joint_position: torch.Tensor,
        joint_velocity: torch.Tensor,
        command: Go2TrackingCommand,
        env_origins: torch.Tensor | None = None,
    ) -> Go2TensorOutput:
        state = self.pack_state(
            root_pose_world,
            root_velocity,
            joint_position,
            joint_velocity,
            env_origins,
        )
        tracking = self.controller.compute(state, command)
        return Go2TensorOutput(self.scatter_action(tracking.action), state, tracking)

    @torch.no_grad()
    def reset(self, mask: torch.Tensor | None = None) -> None:
        self.controller.reset(mask)
