# DDP 正确性验收

本验收将“能运行”和“某次 cost 下降”提升为可重复的数值门槛。正式测试位于
`tests/test_ddp_acceptance.py`，状态机边界测试位于 `tests/test_ddp.py`，导数测试位于
`tests/test_derivatives.py`。

## 执行命令

```bash
# 本机完整验收；缺少 Crocoddyl 时对应项目明确跳过
python -m pytest -m ddp_acceptance -q

# CI 的纯 CPU 门槛
python -m pytest -m "ddp_acceptance and not cuda and not crocoddyl" -q

# 全部回归
python -m pytest -q
```

## 验收矩阵

| 门槛 | 判据 | 当前状态 |
| --- | --- | --- |
| 线性 oracle | DDP 的 `xs/us/cost` 与精确 Riccati LQR 误差 ≤ 1e-9 | 通过 |
| 非线性可行性 | 独立 rollout 与返回轨迹误差 ≤ 1e-12 | 通过 |
| 代价一致性 | 独立重算 Pendulum objective，误差 ≤ 1e-10 | 通过 |
| 非线性改进 | 标准 Pendulum 的 cost 不超过零控制基线的 65% | 通过 |
| 一阶最优性 | 独立离散 adjoint 的最大控制残差 < 3e-3 | 通过 |
| 终端任务 | T=40 时角度误差 < 0.05 rad、速度绝对值 < 0.25 rad/s | 通过 |
| 批量隔离 | B=3 与三个独立 B=1 的状态、控制、代价和状态码一致 | 通过 |
| CPU/CUDA | float64 非线性解及状态码在 1e-8 内一致 | 通过 |
| 异常语义 | 导数非有限、分解重试、线搜索拒绝、预算耗尽均返回规定状态 | 通过 |
| CUDA 热路径 | profiler 不出现 host synchronization 或 tensor scalar extraction | 通过 |
| Crocoddyl oracle | 线性问题直接与 Crocoddyl DDP/LQR 参考对齐，误差 ≤ 1e-6 | 待依赖 |

2026-09-13 本机结果：**24 passed, 1 skipped**；跳过项为未安装的 Crocoddyl bindings。
标准非线性场景实测 cost 由零控制的 1418.6707 降至 519.0529，一阶控制残差
1.68e-3，终端角度误差 9.07e-3 rad，终端速度绝对值 0.1555 rad/s。

## 标准非线性场景

- 模型：半隐式 Euler Pendulum，`dt=0.05 s`，float64。
- 初态：`θ=0.2 rad, θ̇=0`；目标：`θ=π, θ̇=0`。
- horizon：40；最多 25 次 DDP 迭代；cost/gradient tolerance：1e-6。
- 轨迹和代价由测试中的独立代码重算，不调用 backend 私有 helper。
- 一阶残差通过离散 costate backward sweep 重算，不使用 solver 的 feedforward norm。

这些阈值验证当前模型与算法的数学一致性，不代表真实机器人稳定性、安全性或实时性能。
尚未纳入本门槛的内容包括 horizon-shift warm start、控制约束、FDDP、多体接触、
真实 Isaac Lab 动力学以及逐环境随机化参数；它们需要各自新增验收场景。
