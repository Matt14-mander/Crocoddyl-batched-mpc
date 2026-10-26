# Go2 站立与三足前腿跟踪统一接口

此入口面向后续 Isaac Lab 接入：纯 Torch 在线计算，固定 12 关节 Go2，
不调用 NumPy、Pinocchio、Crocoddyl、Meshcat，不依赖离线 Crocoddyl 轨迹。
复用 `Go2WalkMPC` 的逐周期动力学映射、短时域常扭矩 QP 和非线性约束检查。
它是局部跟踪控制器，不是完整非线性接触序列 MPC，不是物体操作控制器。

## 运行

从项目根目录执行，基础 Torch 环境即可：

```bash
PYTHONPATH=src .venv/bin/python examples/go2_front_leg_tracking.py --task stance
PYTHONPATH=src .venv/bin/python examples/go2_front_leg_tracking.py --task front-leg
```

其他机器先在其配套 Torch 环境执行 `python -m pip install -e .`，然后直接运行
`python examples/go2_front_leg_tracking.py ...`。不要为了本示例安装 robotics/viewer extra。

本机 `croco_env` 的 Torch 2.2.2 需要兼容 NumPy 路径（这是 Torch 安装的兼容处理，
控制器本身不使用 NumPy）：

```bash
PYTHONPATH=.dev-tools/croco-numpy126:src OPENBLAS_NUM_THREADS=1 \
  VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1 \
  /Users/zhengyuanhao/anaconda3/envs/croco_env/bin/python \
  examples/go2_front_leg_tracking.py --task front-leg
```

`front-leg` 共 270 个 20 ms 控制周期：站立 0.4 s、重心转移 1 s、左前腿抬高
2 cm 用时 1 s、前移 1 cm/侧移 5 mm 用时 1 s、返回用时 1 s、空中保持 1 s。
最后左前足仍在空中，不隐式触地。`stance` 共 120 周期。
以上周期数针对默认 dt=0.02 s；示例计划要求 dt<=0.02 s。
参考 IK 仅在准备阶段执行，参考姿态从未直接写入 plant。
示例使用另一实例的 Torch 动力学独立积分实际扭矩，不使用 `result.xs` 推进状态；
但两者共享方程，因此只证明数值闭环，不能代替 PhysX 或真机验证。

可加 `--batch-size 32 --device cuda --output artifacts/tracking.json`；
`--no-compile` 使用 eager 物理路径。所有配置目前仅支持 float64。
没有 CUDA 时指定 `--device cuda` 会报错，不会降级为 CPU。

## 在线调用契约

```python
from crocoddyl_batched_mpc.models import (
    Go2TrackingCommand, Go2TrackingMPC, go2_front_leg_tracking_plan,
)

controller = Go2TrackingMPC(batch_size=32, horizon=2, dt=0.02, device="cuda")
plan = go2_front_leg_tracking_plan(controller.model)  # 准备阶段，可发生 host 同步

# 每个周期：state 必须来自模拟器测量，不能使用参考姿态或预测姿态代替。
command = plan.command(step, controller.batch_size, controller.horizon)
output = controller.compute(state, command)
torques = output.action
reset_mask = output.needs_reset  # 交给任务的终止/重置逻辑处理
# 重置模拟器相同环境后：controller.reset(reset_mask)
```

不使用示例计划时，直接构造 `Go2TrackingCommand`。每个环境可传不同参考与支撑模式，
参考可逐周期更新；Cartesian 目标到参考姿态的转换由调用方负责。
当前提供的 IK 是准备工具，不承诺适合作为 GPU 训练的在线 IK。

| Tensor | 形状 | 约定 |
| --- | --- | --- |
| `state` | `[B,37]` | `[q(19),v(18)]`，来自测量 |
| `command.reference` | `[B,T+1,37]` | 当前及后续 T 步参考，时间间隔 dt |
| `command.acceleration` | `[B,18]` | 当前广义加速度参考，可用零值作低速目标 |
| `command.support_mask` | bool `[B,4]` | FL、FR、RL、RR；仅全支撑或释放 FL |
| `command.anchors` | `[B,4,3]` | 环境局部世界系点接触锚点，地面 z=0 |
| `output.action` | `[B,12]` | Nm；FL/FR/RL/RR，各 hip/thigh/calf |
| `output.usable` | bool `[B]` | 是否有通过约束门禁的候选 |
| `output.needs_reset` | bool `[B]` | 锁定失败；必须显式重置 |
| `output.result` | `MPCResult` | 预测、代价、逐环境求解状态和迭代数 |

除 bool/status 外均为同设备 float64，不自动迁移、不自动换精度。
`q=[xyz,xyzw,joints(12)]`；xyz 为减去环境原点后的世界位置，四元数表示机身朝向。
`v=[机身坐标系线速度(3),机身坐标系角速度(3),关节速度(12)]`，单位为 m/s、rad/s；
关节角 rad，长度 m。加速度遵守相同自由浮基广义速度约定。

模拟器若提供 wxyz 四元数，需变换为 xyzw；若提供世界系浮基速度，需旋转到机身系。
关节按名称建立映射，不能假设 Isaac Lab 的 tensor 排序相同。例如初始化阶段：

```python
from crocoddyl_batched_mpc.models.go2 import GO2_JOINT_NAMES

assert len(set(sim_joint_names)) == len(sim_joint_names)
mpc_to_sim = [sim_joint_names.index(name) for name in GO2_JOINT_NAMES]
# 打包关节状态时按 mpc_to_sim gather；输出扭矩时按同一映射 scatter。
```

## 模式、失败与 reset

- 支撑锚点在一个 episode 内不能移动；初次成功命令绑定本环境锚点。
- FL 可从支撑释放；重新增加 FL 支撑会被拒绝，避免隐式碰撞投影或状态瞬移。
  需要触地时，应另接显式事件控制器与仿真接触检测；这里不提供空中 reset 来触地。
- 请求释放只有候选可用才提交；`output.support_mask` 是提交后的模式。
- NaN/Inf 输入、不可行轨迹或不支持的模式使该环境失败锁定，其他环境继续。
  形状、设备或 dtype 错误则直接抛出 `ValueError`，这是调用错误。
- 失败 `action` 为有限零值，预测和代价保留 NaN，`usable=False`。
  **零扭矩只是防止 NaN 进入模拟器，不是安全站立回退**；任务应终止/重置或调用
  经验证的外部恢复控制器，不能继续把该环境计作正常训练轨迹。
- `MAX_ITERATIONS` 可以仍然 `usable=True`，表示候选可行，不表示优化收敛。
- 模拟器 episode 重置后调用 `reset(bool[B])`，清除选中环境缓存、失败锁与模式，
  恢复四足支撑；未选环境不受影响。`reset()` 重置全部环境。
- 在线不要用 `.item()`、`.cpu()`、`.numpy()` 或 Python `if output.usable.all()`
  判断 GPU tensor；以同设备 mask 执行动作和 reset。示例的报告拷贝只在循环后发生。

## CUDA 验收

在有 NVIDIA GPU 的目标机器安装项目与 pytest，然后从项目根目录执行：

```bash
python -m pip install -e '.[dev]'
PYTHONPATH=src python examples/accept_go2_tracking_cuda.py \
  --batch-sizes 1 32 128 --output artifacts/go2-tracking-cuda.json
```

先执行 CUDA 专项测试：Go2 动力学/导数与 CPU 的一致性、混合接触事件、
CPU/GPU 动作一致性、非默认 stream 的即时消费者、异常环境隔离与逐环境 reset，
并在 `compute` 中启用 Torch CUDA 同步调试错误检测。然后运行不同 batch 的站立和前腿
闭环，输出约束违反、跟踪误差、失败数、控制延迟中位/P99/最大、循环吞吐与峰值显存。
同步调试不是所有隐式同步的完整 profiler 证明；Isaac Lab 联调仍需检查实际热路径。

退出码：0=全部通过；1=测试/任务失败；2=CUDA 不可用。无 CUDA 的 JSON 明确写
`status=unavailable`，不能作为 CUDA 验收记录。本机只能执行 CPU 回归。

任务验收要求：全部周期动作可用、状态有限；FL 最大位置误差 <1 cm、最终误差 <3 mm，
支撑足最大锚点误差 <3 mm；扭矩/支撑力约束违反 <=1e-7，非支撑足力 <=1e-10。
这些是示例门槛，不是硬件精度保证。性能仅报告，不以 20 ms 作为仿真训练通过条件。
CUDA event 计时包含 compute 的 stream 执行时间，循环墙钟还包含 Torch plant 和指标计算；
不含 RL、PhysX、传感器或通信。原生 CUDA 内核、Isaac Lab 联调、物体接触与力控制未完成。

CPU 回归：

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/test_go2_tracking.py -q
```
