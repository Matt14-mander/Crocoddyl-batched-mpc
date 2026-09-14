# M2.1 验收：跨周期状态与逐环境参数

M2 的首个增量将 DDP 从无状态单次求解扩展为可用于向量化环境控制循环的逐环境状态机。
`MPCController` 缓存每个可用解，将 `u[1:]` 左移并用最后一个控制补齐 horizon，作为
下一控制周期的初始控制序列。

```bash
python -m pytest -m m2_acceptance -q
```

2026-09-14 本机结果：**11 passed**。

验收覆盖：

- 首次求解使用 `DDPProblem.u_init`，后续求解使用上一解的 horizon shift。
- `reset(mask)` 只清除指定环境的 warm start 和失败回退动作。
- 数值失败不覆盖该环境此前的有效 warm start。
- 替换 `solver.problem` 对象时自动清除全部控制器缓存。
- problem 类型、batch、horizon、控制维度、device 或 dtype 改变时要求重建控制器。
- `[B]` 物理参数和 `[B,…]` 参考/权重在批量模型中与独立模型计算一致。
- 不同物理参数与目标下，批量 DDP 和逐环境独立 DDP 的轨迹、控制及状态一致。
- 参数 batch 与 problem batch 不一致时在构造阶段拒绝。
- CPU/CUDA 上的参数化动力学和代价在 float64 容差内一致。

`MPCController.update_parameters(mask, dynamics=..., cost=...)` 是推荐更新入口。它调用模型的
参数更新方法，并始终失效同一批环境的 warm start 和回退动作，即使更新因参数名或形状错误
而抛出异常。直接原地修改参数 tensor 时，调用方仍必须在下一次 `compute` 前调用
`reset(affected_mask)`。

M2 的后续验收将覆盖长时非线性闭环以及 FDDP gap 处理。
