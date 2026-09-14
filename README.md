# Crocoddyl-batched-mpc

为批量 RL 训练构建统一 CPU/GPU MPC 后端，面向 Isaac Lab 的 Torch CUDA tensor 调用。
这是独立的 downstream 包，不修改 Crocoddyl 源码。

## 当前阶段：M2.2 FDDP gap handling

| 能力 | 当前实现 |
| --- | --- |
| 统一接口 | `LQRProblem` → `BatchedMPC.solve(x0)` → `MPCResult` |
| Torch CPU/CUDA | 批量 Riccati 递推，环境维并行、时间维顺序 |
| 数学问题 | 有限时域、时变仿射动力学、二次及线性代价、无约束 LQR |
| 非线性求解 | 批量 dynamics/cost/manifold、逐环境参数、Torch DDP/FDDP |
| Crocoddyl CPU | 可选 `ActionModelLQR` + `ShootingProblem` + `SolverDDP` 参考后端 |
| RL 控制器 | 首步动作、DDP horizon-shift warm start、失败回退、按 mask 重置 |
| Isaac Lab | Tensor bridge 示例及 DirectRLEnv 接入说明 |
| 验证 | 独立稠密解、闭环、批量隔离、CPU/CUDA 一致性、CUDA stream 测试 |

LQR 后端已经通过独立稠密解和 CPU/CUDA 测试。非线性 DDP 的导数、状态机、
线性 LQR 对照和独立数值门禁已经通过；M2 已完成跨周期 warm start、逐环境物理参数、
参考和代价权重更新，并实现了 FDDP 动态 gap 的 modified Riccati sweep 与逐环境收缩。
Crocoddyl 外部 FDDP 对照仍需在提供 bindings 的环境中执行。
通用 Crocoddyl action model 的 GPU 执行、完整 Crocoddyl FDDP 数值规则、接触动力学、控制约束、可微求解、
原生 C++/CUDA 内核和 CUDA Graph 均在后续计划中。Isaac Sim 真实任务尚未联调。

## 安装与运行

在已有 PyTorch 环境内安装。Isaac Lab 用户优先使用其配套 Python/Torch 环境，
保留与模拟器匹配的 CUDA 版本。基础包只依赖 Torch，无需 Crocoddyl、Pinocchio 或 Isaac Sim。

```bash
python -m pip install -e .
python examples/double_integrator.py --device cpu
python examples/double_integrator.py --device cuda --batch-size 1024
python examples/isaaclab_tensor_bridge.py --device cuda
```

```python
import torch
from crocoddyl_batched_mpc import BatchedMPC, MPCController
from crocoddyl_batched_mpc.models import double_integrator

problem = double_integrator(batch_size=4096, horizon=20, device="cuda:0")
controller = MPCController(BatchedMPC(problem, backend="torch"))
x0 = torch.zeros(4096, 2, device="cuda:0")
actions, result = controller.compute(x0)  # [4096, 1]，留在 CUDA 上
controller.reset(torch.zeros(4096, dtype=torch.bool, device="cuda:0"))
```

不连续状态轨迹可作为 FDDP 初值：

```python
from dataclasses import replace

problem = replace(problem, x_init=state_guess, u_init=control_guess)
solver = BatchedMPC(problem, backend="fddp")
result = solver.solve(x0)
```

输入必须与模型 device/dtype 一致，不自动迁移、不自动降级到 CPU。`result.status`
是同设备的逐环境 tensor；训练热路径不要调用 `.item()` / `.cpu()` 读取状态。
保持动作是基础回退机制，机器人任务需自行定义动作单位、限幅和失败策略。

可选 Crocoddyl CPU 对照（在提供 Crocoddyl bindings 的平台/环境运行）：

```bash
python -m pip install -e '.[crocoddyl]'
python examples/double_integrator.py --device cpu --backend crocoddyl
```

## 开发与验证

```bash
python -m pip install -e '.[dev]'
python -m pytest -q
python -m pytest -m m2_acceptance -q
ruff check .
python benchmarks/bench_lqr.py --device cpu --batch-size 1024
python benchmarks/bench_lqr.py --device cuda --batch-size 1024
```

CUDA/Crocoddyl 不可用时对应测试明确跳过。CI 有独立 Crocoddyl 对照任务；
CPU CI 不能替代 GPU 验证。基准记录求解耗时，不代表完整 RL 训练吞吐。

## 设计文档

- [接口与架构](docs/architecture.md)：数学定义、tensor 布局、后端和失败语义。
- [阶段计划](docs/roadmap.md)：从 LQR 到非线性批量 MPC 的验收条件。
- [Isaac Lab 接入](docs/isaaclab.md)：控制时序、状态映射和环境重置。
- [验证记录](docs/validation.md)：本机实际执行结果和未验证项。
- [DDP 正确性验收](docs/ddp_acceptance.md)：数值门槛、运行命令和覆盖边界。
- [M2 验收](docs/m2_acceptance.md)：warm start、逐环境参数和更新失效协议。
- [FDDP gap 设计](docs/fddp.md)：gap 定义、modified Riccati sweep、merit 和返回语义。

`src/crocoddyl_batched_mpc/` 包含 problem、result、solver、controller、backends；
`tests/` 数值测试；`examples/` 调用示例；`benchmarks/` 性能入口。
