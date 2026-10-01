# 项目验证记录

## 2026-09-30 物理摆修正与持续闭环验收

### 进度复核与本次开发

- 当前阶段仍为 M2.2：LQR、DDP/FDDP、horizon-shift warm start、逐环境参数更新已实现。
- 修正 Pendulum 的重力方向：角度从向下垂直测量时，重力项为 `-(g/l)*sin(theta)`。
  同步修正解析 Jacobian 和 Crocoddyl oracle 的独立 action 公式。旧模型会使向上平衡点
  自然稳定，旧数值与闭环记录不能用于证明物理倒立摆稳定性。
- 新增 3 项物理门禁：共享/逐环境参数的平衡点恢复方向，以及非单位质量、长度、阻尼的
  扭矩平衡。有限差分只能验证代码和导数一致，不能独立发现共同的物理符号错误。
- 新增 2 项闭环门禁：CPU DDP 与 FDDP，各运行 B=3、100 周期，独立物理 plant、
  horizon=30、每周期最多 6 次迭代。第 60 周期施加单环境状态扰动，验证新测量、warm start
  和持续直立。详见 `ddp_acceptance.md`。
- 摆示例新增 dtype/backend/线程数及持续稳定窗口选项，稳定判据同时检查角度、角速度、
  有限性和动作可用性；未满足验收时返回非零退出码。

### 实际执行结果

环境：macOS x86_64，Python 3.11.4，Torch 2.2.2，NumPy 1.26.4；项目独立 `.venv`。
本机没有 CUDA，该测试环境未安装 Crocoddyl。未更改已有 Anaconda 环境。

- 新闭环测试加入前的完整回归（含新增 3 项物理测试）：**50 passed, 18 skipped**。
- `python -m pytest tests/test_pendulum_closed_loop.py -q`：**2 passed**，耗时约 160 秒。
  两次运行合计覆盖当前收集的全部 70 项：52 项通过，18 项因 CUDA/Crocoddyl 不可用跳过。
- `ruff check .` 与 `git diff --check`：通过。
- `PYTHONPATH=src .venv/bin/python examples/pendulum_swingup.py --batch-size 3`：
  CPU float32，100 周期，最后 20 步持续稳定 **3/3**，不可用求解 **0**，
  最终最大角度误差约 **1e-6 rad**、最大速度约 **4e-6 rad/s**。
- 标准 T=40 的 float64 单次求解：零控制 cost=2419.973216，优化后 cost=509.286965，
  独立 adjoint 最大控制残差=3.47e-5，终端状态=[3.125814, 0.026646]，5 次迭代收敛。

### 下一阶段仍需完成

- 在同时提供 Torch/Crocoddyl 的环境执行修正后的外部 FDDP oracle，记录版本和误差。
- 在 CUDA 环境复验修正后的非线性模型、持续闭环和无 host sync 调用路径。
- 本次闭环使用无约束扭矩及同参数 plant，尚未验收控制约束、模型失配和真实 Isaac Lab。
- M3 尚需先进行非线性求解器 profiling，再决定原生内核与 CUDA Graph 的实现范围。

以下为历史执行记录；本次没有重新验证其中的 GPU 与跨平台性能结论。

## 2026-09-20 FDDP 外部 oracle 准备

- 新增非线性 Crocoddyl `SolverFDDP` 对照：相同半隐式摆模型的运行/终端值与导数、
  不连续初值首轮反馈增益、1/2/5/25 次预算的 gap 与代价、最终轨迹。
- 本机完整测试：**60 passed, 5 skipped**；FDDP 专项为 **8 passed, 3 skipped**。
  新增三个跳过项均为 Crocoddyl 外部 oracle；其余两个是原有 Crocoddyl 对照。
- 本机 `base` 未安装 Crocoddyl；`croco_env` 亦未安装 Crocoddyl/Torch。
  因此目前只有 oracle 模型公式自检通过，**尚无跨库误差或迭代数据**。
- 外部执行命令：`python -m pytest -m fddp_oracle -q -s`。CI 的
  `crocoddyl-reference` 任务运行 `-m crocoddyl`，会纳入新增三个测试。

## 2026-09-12 工程基线复核

- 解决 `models.py` 与 `models/` 同名造成的导入冲突，线性模型现位于 `models/linear.py`。
- 根目录调试和手工运行脚本已归入 `examples/`。
- 完整测试：**59 passed, 2 skipped**；两个跳过项均为未安装的 Crocoddyl bindings。
- DDP 正确性验收：**42 passed, 1 skipped**，包含模型导数、状态机、独立数值门槛
  和 FDDP gap handling。
- M2.2 验收：**18 passed**，覆盖 horizon shift、局部 reset、逐环境参数、FDDP gap 收缩、
  批量隔离及 CPU/CUDA 一致性。
- Ruff 检查通过，M0 LQR 和 M1 模型导数测试可以在同一次测试运行中完成。
- DDP 新增 8 项状态机测试：与精确 LQR 对齐、当前 `x0` 轨迹一致性、逐环境失败、
  Cholesky 失败的正则化重试、线搜索拒绝、固定预算可用解、模型 dtype 校验和
  CUDA 无 host sync。
- Pendulum 单次求解（B=1、T=20、最多 50 次迭代）在 5 次迭代收敛，
  cost=519.4505，终态 θ=3.2642 rad、角速度 1.2589 rad/s。它证明单次求解明显改善，
  尚不能替代长时闭环稳定验收。

日期：2026-09-03。下列为本机执行结果，CI 配置尚未在远程运行。

## 环境

- Windows，Python 3.12.3：`E:\anaconda3\python.exe`。
- PyTorch 2.7.0+cu128，CUDA runtime 12.8。
- NVIDIA GeForce RTX 4070 Laptop GPU，驱动 555.97，显存约 8 GB。
- 系统默认 `python` 指向另一套 MSYS Python，未安装 Torch；本次使用上述 Anaconda 解释器。
- 没有修改已有 Python 环境；Ruff/build 辅助工具置于被 Git 忽略的 `.dev-tools/`。

## 数值与调用路径

`E:\anaconda3\python.exe -m pytest -q`：**17 passed, 1 skipped**。

通过项包括：

- T=1/5、含仿射动力学和线性代价的随机 LQR，与消元后完整控制 Hessian 的独立稠密解对齐。
- 正则项 λ=0/0.2 的控制解和目标值；动力学可行性；逐环境独立求解一致性。
- float32/float64、非连续状态 view、输入不变和无梯度输出。
- 非有限输入/模型与非正定控制 Hessian 的逐环境失败隔离。
- 控制器保持上次有效动作、选择性 reset、返回值不暴露内部缓存。
- 闭环状态误差下降；错误 shape/dtype/参数明确拒绝。
- 真实 CUDA 上 float32/float64 与 CPU 对齐，非默认 stream 及同 stream 即时消费者。
- 预热后 profiler 中无 `cudaStreamSynchronize`、`cudaDeviceSynchronize`、
  `aten::item`、`aten::_local_scalar_dense`。该检查针对当前 Torch 环境，并非所有平台保证。

Crocoddyl 对照测试因未安装 bindings 跳过。参考适配器已编写，运行兼容性及数值对照仍待验证。
独立 CI 任务会先明确导入 Crocoddyl，避免缺依赖导致 CI 静默跳过。

真实 CUDA 示例：

- double integrator，B=1024，100 个闭环步骤：0 次失败，位置 RMS 从 0.577914 降至 0.006807。
- tensor bridge，B=4096：返回 `[4096, 1]` CUDA 动作，全部求解成功。
- 此处的 bridge 没有启动 Isaac Sim；真实机器人环境尚未联调。

## 初步性能数据

最终三角求解版本，B=1024、T=20、nx=2、nu=1、float32，CPU threads=1，
warmup=2、repeats=5。使用 `benchmarks/bench_lqr.py`；含有限性检查、分配、完整轨迹与代价，
CUDA 在计时边界同步，排除模型构建和模拟器。

| 设备 | 中位延迟 ms | 最小/最大 ms | 环境求解数/秒 |
| --- | ---: | ---: | ---: |
| CPU | 25.362 | 23.564 / 25.472 | 40,375 |
| CUDA | 36.758 | 32.644 / 56.126 | 27,858 |

这是小规模烟雾基准，采样数量少，不作硬件或训练性能结论。此配置未显示 GPU 加速优势。
后续需测量更多维度/批量，并减少 Python 调度和临时分配。初次 profiler 发现
`torch.cholesky_solve` 内部同步，现已用两次 `solve_triangular` 替代并加入回归测试。

## 工程检查与未覆盖项

- Ruff 检查通过；sdist 与 wheel 可构建。
- 尚未验证：Crocoddyl bindings、真实 Isaac Lab/Isaac Sim、原生 CUDA 内核、CUDA Graph、
  非线性/接触/约束 MPC、多 GPU、可微求解以及最小 Torch 2.2 版本兼容性。
