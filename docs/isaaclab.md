# Isaac Lab tensor 接入

核心包不导入 `isaaclab`，可在普通 Torch 环境测试。在任务的配套 Python 环境安装后，
直接传 CUDA tensor。任务负责模型、状态提取和单位转换。

Go2 站立/三足前腿跟踪使用 `Go2TrackingMPC` 统一入口，`Go2TensorAdapter` 已实现状态与扭矩映射，
命令形状、失败锁定/reset 语义及 CUDA 验收命令见 [接口文档](go2_tracking.md)。
该路径的失败输出为零扭矩并要求外部恢复/重置，不沿用下文通用控制器的保持动作策略。
目前未提供完整 Isaac Lab Go2 环境，adapter 的独立 Torch 闭环不代替 PhysX 验证。

## Go2 articulation tensor adapter

Adapter 不导入 Isaac Lab，可直接接收 articulation tensor。构造时传入模拟器实际关节名，
内部按名称 gather 到 FL/FR/RL/RR 顺序，输出再 scatter 回模拟器顺序。额外的非驱动槽输出零 effort。

```python
from crocoddyl_batched_mpc.models import Go2TensorAdapter

self.go2_mpc = Go2TensorAdapter(
    sim_joint_names,
    self.num_envs,
    device=self.device,
    quaternion_order="wxyz",  # 与任务实际 tensor 约定一致
    velocity_frame="world",
)

# root_pose_w: [B,7] position + quaternion
# root_velocity_w: [B,6] world linear/angular velocity
# joint_pos/joint_vel: [B,N] simulator joint order
output = self.go2_mpc.compute(
    root_pose_w,
    root_velocity_w,
    joint_pos,
    joint_vel,
    command,
    self.scene.env_origins,
)
self.robot.set_joint_effort_target(output.joint_effort)
reset_mask = output.tracking.needs_reset
```

Adapter 完成环境原点扣除、wxyz/xyzw 转换、世界速度到 body velocity 的旋转、关节重排和
逐环境失败隔离。输入必须与控制器位于同一设备且为 float64；不进行隐式搬运或精度转换。
任务仍需从自身 observation/command 构造 `Go2TrackingCommand`，并在 episode reset 后用相同
mask 调用 `self.go2_mpc.reset(mask)`。

## DirectRLEnv 接入片段

以下不是完整环境。`problem` 必须对应机器人在 MPC 周期内的离散动力学，
`state` 顺序和 `nu` 控制通道必须与模型一致。

```python
import torch
from crocoddyl_batched_mpc import BatchedMPC, MPCController

# 在任务 __init__ 中，problem 已建立在 self.device 上：
self.mpc = MPCController(BatchedMPC(problem, backend="torch"))
self.reset_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)


def _pre_physics_step(self, actions: torch.Tensor):
    state = self.pack_mpc_state()  # 由任务实现，核对状态顺序/单位。
    self.mpc_action, self.mpc_result = self.mpc.compute(state)
    # 这是 MPC-only；residual RL / reference policy 需另行实现 actions 映射。


def _apply_action(self):
    # 仅当模型 u 定义为对应关节力矩/力时才可如此调用。
    self.robot.set_joint_effort_target(self.mpc_action, joint_ids=self.control_joint_ids)


def _reset_idx(self, env_ids):
    super()._reset_idx(env_ids)
    # 同时执行任务自己的模拟器状态重置。
    if env_ids is None:
        self.mpc.reset()
    else:
        self.reset_mask.zero_()
        self.reset_mask[env_ids] = True
        self.mpc.reset(self.reset_mask)
```

逐环境 domain randomization 应通过控制器更新，使参数与 warm-start 失效使用同一个 mask：

```python
self.mpc.update_parameters(
    reset_mask,
    dynamics={"mass": randomized_mass, "length": randomized_length},
    cost={"x_ref": randomized_reference},
)
```

更新 tensor 可以是完整 `[B,...]`、被选环境数量 `[N,...]`，或不带 batch 维的共享值；
tensor 必须与 problem 位于同一 device 且 dtype 相同。

通用非 Go2 路径的 `pack_mpc_state`、`problem`、`robot`、`control_joint_ids` 仍由任务提供。
如果基类构造过程触发 reset，应确认控制器已初始化后再使用。

## 时序和 reset

- `_pre_physics_step` 解一次并取首步动作，`_apply_action` 在 decimation 中保持动作。
  此时模型 `dt = sim.dt * decimation`；更改 MPC 频率需重新核对离散化。
- 世界坐标状态需减去环境原点；旋转/四元数遵守模型约定。
- reset 清除对应 fallback；后续非线性 warm start 同样需要环境隔离。
- status 保持在设备，用 tensor mask 选择动作，周期性日志才读取 host。
- 控制器失败时保持动作缓存；机器人任务负责 actuator limits 和连续失败策略。
- double integrator 示例的 `u` 是加速度，不能直接作为关节力矩。

```bash
python examples/isaaclab_tensor_bridge.py --device cuda --num-envs 4096
```

该命令仅验证 tensor 调用契约，不启动 Isaac Sim。
参考：[官方动作和重置生命周期](https://isaac-sim.github.io/IsaacLab/main/source/tutorials/03_envs/create_direct_rl_env.html)。
