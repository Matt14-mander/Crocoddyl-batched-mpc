# Isaac Lab tensor 接入

核心包不导入 `isaaclab`，可在普通 Torch 环境测试。在任务的配套 Python 环境安装后，
直接传 CUDA tensor。任务负责模型、状态提取和单位转换。

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

`pack_mpc_state`、`problem`、`robot`、`control_joint_ids` 由任务提供。
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
