# M2.0 验收：跨周期 warm start

M2 的首个增量将 DDP 从无状态单次求解扩展为可用于向量化环境控制循环的逐环境状态机。
`MPCController` 缓存每个可用解，将 `u[1:]` 左移并用最后一个控制补齐 horizon，作为
下一控制周期的初始控制序列。

```bash
python -m pytest -m m2_acceptance -q
```

2026-09-13 本机结果：**5 passed**。

验收覆盖：

- 首次求解使用 `DDPProblem.u_init`，后续求解使用上一解的 horizon shift。
- `reset(mask)` 只清除指定环境的 warm start 和失败回退动作。
- 数值失败不覆盖该环境此前的有效 warm start。
- 替换 `solver.problem` 对象时自动清除全部控制器缓存。
- problem 类型、batch、horizon、控制维度、device 或 dtype 改变时要求重建控制器。

模型和参考 tensor 允许原地更新，因此控制器无法在不读取或复制 tensor 的情况下自动识别变化。
调用方必须在更新后、下一次 `compute` 前调用 `reset(affected_mask)`。这一显式失效协议避免
CUDA 热路径加入版本读取或 host synchronization。

M2 的后续验收将覆盖逐环境随机化参数及其更新接口、长时非线性闭环，以及 FDDP gap 处理。
