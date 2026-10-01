# DDP 正确性验收

本验收将“能运行”和“某次 cost 下降”提升为可重复的数值门槛。正式测试位于
`tests/test_ddp_acceptance.py`，状态机边界测试位于 `tests/test_ddp.py`，导数测试位于
`tests/test_derivatives.py`；物理约定与持续闭环分别见
`tests/test_pendulum_physics.py` 和 `tests/test_pendulum_closed_loop.py`。

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
| 物理方向 | 向下重力恢复、向上重力发散，非单位参数的扭矩平衡 | CPU 通过 |
| 持续闭环 | B=3、100 控制周期；直立保持 20 步，局部扰动后再保持 20 步 | CPU DDP/FDDP 通过 |
| 批量隔离 | B=3 与三个独立 B=1 的状态、控制、代价和状态码一致 | 通过 |
| CPU/CUDA | float64 非线性解及状态码在 1e-8 内一致 | 通过 |
| 异常语义 | 导数非有限、分解重试、线搜索拒绝、预算耗尽均返回规定状态 | 通过 |
| CUDA 热路径 | profiler 不出现 host synchronization 或 tensor scalar extraction | 通过 |
| Crocoddyl oracle | 线性问题直接与 Crocoddyl DDP/LQR 参考对齐，误差 ≤ 1e-6 | 待依赖 |

2026-09-14 本机结果：**42 passed, 1 skipped**；跳过项为未安装的 Crocoddyl bindings。
上述历史记录使用修正前的重力方向，不能作为物理摆的验收数据。
2026-09-30 在 macOS CPU / Torch 2.2.2 上修正重力方向后，标准场景 cost 由
2419.9732 降至 509.2870，独立一阶控制残差 3.47e-5，终端角度误差
0.01578 rad、终端速度 0.02665 rad/s，5 次迭代收敛。
本次未执行 CUDA；表中的 CPU/CUDA 和 CUDA 热路径通过状态为此前历史记录。

## 标准非线性场景

- 模型：半隐式 Euler Pendulum，`dt=0.05 s`，float64。
- 初态：`θ=0.2 rad, θ̇=0`；目标：`θ=π, θ̇=0`。
- horizon：40；最多 25 次 DDP 迭代；cost/gradient tolerance：1e-6。
- 轨迹和代价由测试中的独立代码重算，不调用 backend 私有 helper。
- 一阶残差通过离散 costate backward sweep 重算，不使用 solver 的 feedforward norm。

这些阈值验证当前模型与算法的数学一致性，不代表真实机器人稳定性、安全性或实时性能。
本门槛已经纳入 horizon-shift warm start 和 FDDP gap 测试。尚未覆盖控制约束、多体接触、
真实 Isaac Lab 动力学以及逐环境随机化参数；它们需要各自新增验收场景。

## 持续闭环场景

使用独立写出的半隐式物理 plant（重力为 `-9.81*sin(theta)`），不调用模型的
`calc` 推进实际状态。初态为 `[0,0]`、`[0.4,-0.2]`、`[-0.4,0.2]`，
horizon=30，每周期预算 6 次迭代，float64，DDP 与 FDDP 各执行 100 周期。
第 60 周期对第一个环境施加 `[0.2 rad, -0.3 rad/s]` 状态扰动，继续消费新测量和 warm start。
扰动前和恢复后分别连续 20 步满足角度误差 <0.05 rad、速度 <0.1 rad/s，
每周期动作可用；最终误差 <0.002 rad、速度 <0.01 rad/s。
该场景使用无约束扭矩、同参数模型与 plant，尚不覆盖模型失配或扭矩限幅。
