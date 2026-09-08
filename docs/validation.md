# 初始框架验证记录

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
