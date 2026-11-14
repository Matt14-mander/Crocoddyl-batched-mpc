# Go2–Z1 异步 CPU 耗时与可重复延迟验收

2026-11-14。开发基线为 `e9a9b5e`。本轮只扩展带臂异步执行路径，不改动现有单 Go2
训练代码、共享 WBC/求解器的默认控制律或依赖环境。此前首次失败与复测记录保留在
[2026-11-13 归档](benchmarks/go2_z1_async_cpu_2026-11-13/README.md)，不覆盖历史结果。

## 定位与修改

首次失败的姿态峰值 0.020445 rad 出现在 5.661 s，之前存在约 282 ms 的控制间隔，
随后又出现约 182 ms 间隔；过期包已经正确拒收。这说明需要检查长时间保持力矩、
状态/参考采样延迟和控制更新，而不能只修改包有效期。

对修改前版本的过期场景使用 cProfile：`advance` 累计约 3.98 s，WBC 求解约
1.44 s，Guidance 插值约 1.23 s。积分循环每个子步调用完整 `robot.evaluate`，
重复计算了诊断并不需要的质量矩阵、重力、非线性项和完整雅可比。profiling 本身
有开销，此结果只用于定位，不能当作无观察器的实时性能验收。

本轮改动：

- 新增异步专用 `AsyncCPUPlant`，复用私有 Pinocchio Data、KKT 矩阵和常量。
  足端与 TCP 观测只计算 FK 和 frame velocity；与原独立 KKT plant 对照，
  检查不同子步、非零速度、外力和 18 关节映射，仍由实际力矩积分状态。
- 只在规划器可以接受请求时生成快照及 outgoing Guidance，忙时继续轮询。
- 每次控制保留当时的不可变 Guidance 与 buffer 快照，回合结束后根据实际轨迹
  计算完整 TCP/姿态误差及原始任务偏差。没有减少检查样本；每个积分子步的足端、
  关节、力矩、接触力与动力学残差检查仍在前台执行。
- 记录 plant、逐步诊断、参考、IPC、WBC 分项耗时，调度唤醒延迟、命令年龄、
  错过的控制槽和离线诊断耗时。后台另记录预览、初值、求解/验收、Guidance 导出耗时。
  同时记录控制线程 CPU 时间和墙钟时间、最慢十个周期以及异常发生的阶段。
  二者之差是非线程 CPU 时间，包含主动 sleep、IPC 等待及被调度出去的时间；
  不能把全部差值直接归因于操作系统调度。

控制增益、误差和约束门槛保持不变。1.2 s 规划时域、0.5 s 请求周期、0.4 s 换包、
0.6 s 默认接收超时和 0.08 s 命令最大年龄沿用基线。

这些优化由 `run_async_cpu_episode` 默认启用，实际作用于异步 CPU 闭环；单元测试
用于核对与原实现的数值一致性。接入 Isaac Lab 时状态推进由 PhysX 完成，实机状态
变化来自机器人本体，本地 plant 的加速比例不能直接作为这两种执行路径的收益。

## 可重复扰动与真实墙钟的区别

`ExecutionClock` 的默认模式使用真实单调时钟。显式 `ControlDelay` 在指定任务时刻
后的第一个对应阶段注入 sleep，记录实际开始时刻及经过时间，原控制循环继续按
真实经过时间积分旧力矩；不会把延迟从任务时间里扣除。

`virtual` 模式使用可控时钟和 `ScheduledWWorker`，实际调用同一个 W、Guidance、
WBC 和独立 plant，但人为指定计算及交付时间。这个测试 transport 不创建后台进程，
不能用于并发或实时性能结论；API 禁止虚拟时钟搭配真实 spawn worker。
相同输入重复运行后逐项比较完整轨迹数组，不只比较通过标志。

确定性测试使用后台交付延迟 0.25 s，控制阶段预算分别为采样前 9 ms、求解后 18 ms、
周期末 6 ms，使普通更新间隔约 40 ms；这是明确给定的扰动模型，不是假定机器能达到
这些耗时。所有过期场景额外注入第 3 个 W 请求 1.35 s 延迟，同时沿用之前过期测试的
2 s 接收超时/状态年龄，确保走到预测时域过期分支。

| 场景 | 额外前台延迟 | 判据 |
| --- | --- | --- |
| expired | 无 | 过期拒收、回退、恢复及原 CPU 门槛 |
| burst | 4.9 s 后采样前 150 ms、求解后 110 ms；5.38 s 后采样前 90 ms；5.5 s 后求解后 25 ms | 类似历史连续长间隔，丢弃过龄命令，恢复后仍满足原门槛 |
| boundary | 1.65 s 后采样前 120 ms、求解后 90 ms | 有效期边界附近阻塞，旧包到期回退、拒包并恢复 |
| overrun | 0.5 s 后采样前 250 ms | 超过 0.2 s 补算上限，明确终止并清理 worker；不截短 elapsed time 来伪造成功 |

`overrun` 的 `scenario_passed` 表示预期终止行为通过，其闭环 `passed` 仍为 false，
不会混入成功跟踪统计。延迟注入是历史模式的受控变体，不声称复现了所有操作系统
调度事件或当时完全相同的 W 轨迹。

## 复现

以下命令均使用现有环境，不安装依赖。输出目录与历史归档分开：

```bash
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1
export PYTHONPATH=.dev-tools/crocoddyl321-pin400/cmeel.prefix/lib/python3.11/site-packages:src
.venv/bin/python -B examples/go2_z1_delay_regression.py --mode virtual --repeat 2 \
  --output artifacts/go2-z1-jitter-virtual
.venv/bin/python -B examples/go2_z1_delay_regression.py --mode wall --repeat 2 \
  --output artifacts/go2-z1-jitter-wall
.venv/bin/python -B examples/go2_z1_async_cpu.py \
  --output artifacts/go2-z1-jitter-seven-cases
```

完整回归：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -B -m pytest -p no:cacheprovider -q \
  tests/test_async_timing.py tests/test_async_cpu.py tests/test_rolling_cpu.py \
  tests/test_stance_demo.py tests/test_guidance.py tests/test_whole_body_qp.py \
  tests/test_standing_whole_body.py tests/test_go2_tensor_adapter.py \
  tests/test_walking_whole_body_demo.py
```

## 实测记录

本轮记录与结论见下述验收归档；跟踪成功、预期终止和实时预算分别统计。

最终代码的原有七场景批次 **7/7 通过**：共 86 次请求、79 包接收、2706 次 WBC 更新。
控制增益与原验收门槛完全一致，所有接收/拒收与闭环约束均按原规则检查。

| 场景 | 接收 / 请求 | TCP 峰值 mm | 姿态峰值 rad |
| --- | --- | --- | --- |
| normal | 12 / 12 | 0.0387 | 0.000085 |
| delay | 12 / 12 | 0.0324 | 0.000083 |
| failure | 8 / 12 | 1.0703 | 0.000398 |
| timeout | 9 / 10 | 1.0234 | 0.000350 |
| expired | 9 / 10 | 1.0986 | 0.000363 |
| restart | 6 / 7 | 0.2909 | 0.000131 |
| high_low | 23 / 23 | 1.0476 | 0.009538 |

七例前台周期中位数 12.28–13.69 ms，实际控制间隔中位数 20.72–22.33 ms；历史基线
分别约 26.7–31.1 ms 和 39.0–45.3 ms。两次运行主机负载不同，不能把全部差值归因于
某一项代码修改。配对微基准采用同一进程、预热后交替五组各 100 次调用：独立 KKT
单步中位数从 0.594 ms 降至 0.285 ms（约 2.09 倍），诊断观测从 0.336 ms 降至
0.0540 ms（约 6.23 倍）。逐步约束核对仍执行，跟踪误差统计只是移出计时回合。

普通场景 W 分项中位数：预览约 50.2 ms、初值约 1.24 ms、求解与验收约 131.4 ms、
Guidance 导出约 38.8 ms。后续重点是问题构造/验收与导出、前台包校验，不能把规划
总耗时等同于 Box-FDDP 的迭代时间。

**保留的本轮失败：** 增加线程 CPU 时间遥测之前的一次完整七场景运行仍为 6/7。
过期回合姿态峰值只有 0.000314 rad，没有重现原姿态超限，但状态落后墙钟约 474 ms，
触发 200 ms 补算保护，未完整执行，因此仍记失败。该次最长周期约 337 ms；当时
没有线程 CPU 时间，不能事后断言全部属于操作系统调度。本次只是增加遥测后重新
验证，没有放宽控制增益、验收或补算上限。此失败和最终批次分开保存。

当前证据支持原姿态超限在本轮标准回合和指定延迟范围内通过回归，不支持任意主机
停顿下的稳定性保证。每例仍有 51–141 个错过的控制槽，50 Hz 无漏槽、
200 Hz 与硬实时能力均未验收。CPU 积分的数值时间滞后继续明确报告；这些结果仍限于
相同刚体模型与固定支撑，未进行 Isaac Lab/PhysX、载荷、碰撞或实机验收。

最终代码的真实墙钟延迟实验，每类重复两次：

| 场景 | 结果 | TCP 峰值范围 mm | 姿态峰值范围 rad | 说明 |
| --- | --- | --- | --- | --- |
| expired | 2/2 完整通过 | 0.8753–0.9081 | 0.000350–0.000514 | 正确拒收过期包并恢复 |
| burst | 2/2 完整通过 | 1.3499–1.8186 | 0.005208–0.012103 | 最大控制间隔分别约 247/323 ms，过龄命令丢弃 |
| boundary | 2/2 完整通过 | 1.9919–2.8315 | 0.002094–0.003605 | 有效期边界附近连续停顿、回退与恢复 |
| overrun | 2/2 预期终止 | 不计入跟踪通过率 | 不计入跟踪通过率 | 250 ms 单次暂停越过补算上限，停止并清理后台 |

两次 overrun 被捕获于 `advance_sampled_state`；所在周期墙钟约 267/257 ms，而线程
CPU 只用约 8.0/2.1 ms，差值与主动注入等待一致。它验证了等待与实际计算的计时分离，
不能用它反推上文未记录线程时间的自然长停顿原因。

虚拟时钟实验对上述四类也各重复两次，六个跟踪回合和两个预期终止均符合判据；
每类两次完整状态、力矩、请求与参考轨迹数组逐项完全一致。该结论是确定性功能验收，
不参与墙钟性能统计。新增测试覆盖独立 plant 数值对照、FK/足速度、工作区隔离、
重复延迟、过龄命令、过期回退、反馈时间一致性和超限停止。

相关回归 **139 passed / 1 skipped**，含 9 项新增测试；CUDA adapter stream 因本机无
CUDA 跳过。136 个既有代码等文件中只扩展两个异步控制模块与异步示例的源文件哈希
清单，其余 133 个文件、88 个依赖元数据和 9 个历史证据文件哈希不变。
未安装依赖或修改单 Go2 训练代码。

最终三批的 213 次请求与 6014 次 WBC 输入均和实际积分轨迹/上一力矩核对一致；
全部离线 TCP/姿态统计再用独立原始 FK 重算，峰值一致。
证据见 [本轮归档](benchmarks/go2_z1_async_jitter_2026-11-14/README.md)、
[反馈核对](benchmarks/go2_z1_async_jitter_2026-11-14/feedback_audit.json)、
[离线诊断核对](benchmarks/go2_z1_async_jitter_2026-11-14/offline_diagnostics_audit.json) 和
[隔离记录](benchmarks/go2_z1_async_jitter_2026-11-14/preservation.json)。

字段说明：虚拟模式沿用的 `wall_elapsed_s` 取自规定时钟，不是宿主机实际耗时。
`timing_mode`、`execution_settings` 和 `schedule` 明确区分该模式；性能分析使用真实
墙钟批次，虚拟模式仅用于轨迹与时间契约回归。
