# FDDP 动态 gap 设计

## Gap 定义与初值

对候选轨迹 `(x, u)`，第 `t` 个动态 gap 定义为：

```text
g[t] = f(x[t], u[t]) ⊖ x[t + 1]
```

其中 `⊖` 是状态流形的局部差分；欧氏状态下即
`f(x[t], u[t]) - x[t + 1]`。`max(abs(g)) <= gap_tolerance` 时，该环境的轨迹
被视为动态可行。

`backend="fddp"` 使用 `DDPProblem.x_init` 传入可能不连续的状态轨迹，并使用
`u_init` 传入控制初值。求解器始终用本次 `solve(x0)` 的 `x0` 覆盖状态初值的首项，
避免跨控制周期使用过期测量。未提供 `x_init` 时，求解器先从 `x0` 和 `u_init`
单次 rollout，得到零 gap 的可行初值。

## Modified Riccati sweep 与 forward pass

Backward pass 在每个时刻先将 gap 注入价值函数梯度：

```text
Vx_gap = Vx + Vxx @ g[t]
Qx = lx + Fx.T @ Vx_gap
Qu = lu + Fu.T @ Vx_gap
```

其余二阶项和正则化沿用批量 DDP 的 Gauss-Newton/iLQR 递推。Forward pass 对每个
候选步长 `alpha` 使用：

```text
x_candidate[t + 1] = integrate(
    f(x_candidate[t], u_candidate[t]),
    (alpha - 1) * g_nominal[t],
)
```

因此 `alpha=0` 保留名义 gap，`alpha=1` 完全闭合名义 gap，中间步长按比例收缩。
这一符号和更新方式与 Crocoddyl `SolverFDDP` 的 gap forward pass 一致。

## 接受准则与返回语义

当前批量后端用显式 L1 merit 逐环境接受 line-search 候选：

```text
merit = trajectory_cost + gap_penalty * sum(abs(g))
```

这允许在动态 gap 明显缩小时暂时接受更高的轨迹代价。`gap_penalty` 必须为有限正数，
并应按任务代价尺度调节。该准则是面向批量 tensor 执行的第一版工程选择；它没有复刻
Crocoddyl 基于 `dg`、`dq` 和 `dv` 的完整 expected-improvement 数值规则，因此两者的
逐次 line-search 决策不保证完全一致。

`MPCResult.feasible` 是逐环境 bool tensor。FDDP 只有在数值有限且 gap 达到容差时，
`result.usable` 才为真：

- 已收敛且可行：`SUCCESS`，动作可用。
- 固定迭代预算耗尽但可行：`MAX_ITERATIONS`，best-so-far 动作仍可用。
- 预算耗尽且不可行：`MAX_ITERATIONS`，保留轨迹用于诊断，动作不可用。
- 导数、分解或 rollout 出现非有限值：`NUMERICAL_FAILURE`，轨迹和代价置为 NaN。

`MPCController` 仅缓存可用环境的结果。下一控制周期同时 shift 状态和控制轨迹，末端
状态由末端控制再推进一步，并用新测量覆盖首状态。环境 reset 会同时清除两类缓存。

## 当前边界

当前实现使用 dynamics 一阶导数和 cost 二阶导数，没有加入 dynamics 二阶项、控制盒约束、
接触约束或 Crocoddyl action-model 适配。线性问题、批量隔离、部分 gap 收缩、状态缓存和
CPU/CUDA 一致性已有本地验收；与 Crocoddyl 非线性 `SolverFDDP` 的轨迹、反馈增益和迭代
行为对照，需要在安装 Crocoddyl bindings 的环境中完成。

参考：

- [Crocoddyl SolverFDDP 类文档](https://gepettoweb.laas.fr/doc/loco-3d/crocoddyl/master/doxygen-html/classcrocoddyl_1_1SolverFDDP.html)
- [Crocoddyl fddp.cpp 源码](https://gepettoweb.laas.fr/doc/loco-3d/crocoddyl/master/doxygen-html/fddp_8cpp_source.html)
