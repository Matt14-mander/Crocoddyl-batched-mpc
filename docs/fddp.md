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
CPU/CUDA 一致性已有本地验收。外部 oracle 测试位于 `tests/test_fddp_oracle.py`：
使用相同的半隐式摆动力学、代价和不连续初值，比较 action 的值与导数、首轮
backward pass 的反馈增益、多个迭代预算下的 gap 收缩，以及最终轨迹和代价。
Crocoddyl 的 `K` 约定在控制律中作减法，Torch 的 `K` 已带负号，因此测试比较
`K_torch` 与 `-K_crocoddyl`。两者 line search 和正则化规则不同，不要求逐次步长
或收敛迭代数完全相同。

在安装 Crocoddyl 与 Torch 的 CPU 环境执行：

```bash
python -m pip install -e '.[dev,crocoddyl]'
python -m pytest -m fddp_oracle -q -s
```

2026-10-01 已在本机 `croco_env`（Python 3.10.18、Crocoddyl 3.0.1、Torch 2.2.2）
执行外部对照，3 项 `fddp_oracle` 均通过；包含线性参考的全部 Crocoddyl 对照为 5 项通过。
本机 EigenPy 会将单列 `Fu/Lxu/K` squeeze 为向量，测试适配层恢复其数学形状，
没有放宽数值容差。首轮反馈增益最大绝对误差为 8.71e-5，最终状态、控制及代价
最大绝对误差为 6.40e-6、7.05e-5、1.13e-9。初始最大 gap=0.08，双方第一轮均闭合。
Torch 在 3 次有效迭代后冻结，Crocoddyl 返回 iteration_index=4；两者的停止规则不同。

该 Intel macOS 环境原有 NumPy 2.2.6 与 Torch 2.2 的 NumPy 桥接不兼容。
环境原有版本已保留，测试通过项目内独立 NumPy 1.26.4 路径执行：

```bash
PYTHONPATH=.dev-tools/croco-numpy126:src \
  /Users/zhengyuanhao/anaconda3/envs/croco_env/bin/python -m pytest -m fddp_oracle -q -s
```

该路径在 `.gitignore` 内；重新准备方式见 `validation.md`。此配置只验证本机测试中的
模型与预算序列，不代表完整复刻 Crocoddyl 的 line search 或正则化规则。

参考：

- [Crocoddyl SolverFDDP 类文档](https://gepettoweb.laas.fr/doc/loco-3d/crocoddyl/master/doxygen-html/classcrocoddyl_1_1SolverFDDP.html)
- [Crocoddyl fddp.cpp 源码](https://gepettoweb.laas.fr/doc/loco-3d/crocoddyl/master/doxygen-html/fddp_8cpp_source.html)
