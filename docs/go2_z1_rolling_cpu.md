# Go2–Z1 W 滚动 CPU 闭环与 Guidance 换包

2026-11-13。在 `9fa2671` 固定支撑 Meshcat 基线上增加实测状态驱动的滚动求解：
共同任务预览 → W 短时域规划 → Guidance 平滑换包 → 同一 WBC → 18 关节力矩 →
独立 Pinocchio 接触 plant → 下一周期实测状态。旧离线演示仍可按原入口复现。

这里的“在线”指在 CPU 仿真回合内不断读取实际状态并重新规划。求解是同步的，
执行期间仿真时钟暂停；没有后台规划线程，也没有把实际计算耗时折算为包延迟。
因此这是 receding-horizon 闭环功能验收，不是墙钟实时、异步通信或 Isaac Lab 验收。

## 实现与默认配置

新增 `controllers/rolling_whole_body_cpu.py`，不注册到旧单 Go2 的导入入口。
`RollingWholeBodyCPU` 持有同一回合的 W 规划器、Guidance buffer 和热启动缓存；
`run_rolling_cpu_episode` 复用现有 WBC/plant 与全部逐步验收门槛。

| 参数 | 默认值与含义 |
| --- | --- |
| W 时域 | 10 节点 × 0.05 s = 0.5 s |
| 重规划周期 | 仿真时间 0.1 s，名义 10 Hz；不是已达到的实时频率 |
| WBC/plant | dt=0.005 s，200 Hz 数值积分；真实耗时另报 |
| 新包交接 | 0.05 s，从正在执行的参考接到新包相同绝对时刻的参考 |
| 过期回退 | 原包到期后 1 s 平滑回初始站姿，不延长旧包有效期 |
| 失联恢复 | 0.4 s 从当前回退参考重接原任务，并跨后续规划窗口保留该桥接轨迹 |
| 规划停止 | 共同任务结束后不再请求新包；最后一包到期后仍执行既有回退 |

W 每次求解接收实测 `(q25,v24)` 和**上一控制步实际施加的 WBC 力矩**。
只更新 shooting problem 的初值，不重置足端锚点或身体活动边界；足端数值漂移
不会被吸收到新的支撑点中。W 输出仍只形成参考，不直接给 plant 施加规划力矩。
原 `solve()` 不传新参数时保持原行为。

## 热启动与共同预览

共同任务按绝对时间截取窗口，保持窗口首尾原有速度和加速度；只有超过任务末端
才使用已验证的静止末姿。名义关节参考继续来自共同生成器，不用 W 优化关节替换。

第一包或缓存过期时，以共同姿态和逐节点准静态力矩构造新的 FDDP 初值。
随后按实际经过时间移位最近一条已接收、已通过验收的 W 轨迹：构型在 Pinocchio
流形上插值，速度插值，力矩按分段常数取值。未覆盖的尾部保持最后姿态、零速度及
最后力矩；首状态替换为实测状态，力矩记忆从实际上一 WBC 力矩重新建立。
这些只是允许含动力学 gap 的初值，仍须由 FDDP 闭合 gap 并通过独立 KKT 验收。
缓存年龄达到原时域长度即丢弃；失败候选和被拒绝的新包不替换缓存。

关闭 `RollingSettings(warm_start=False)` 后，每次都从共同参考重新初始化，
用于同场景冷/热启动对照。报告迭代数、状态差异和耗时，不预设热启动一定更快。

## Guidance 新包交接和失败路径

`GuidanceBuffer` 新增可选 `handoff_duration`，默认 0，保持原直接换包行为。
滚动路径开启交接，任务位姿、速度和加速度保持 C2 连续；接触力为 C0 连续。
交接尚未结束又收到新包时，从当前正在执行的交接采样重新桥接。

无效、过期、未来时间、重复/倒序 sequence、错误模型/足锚点、剩余有效期不足及
跨包旋转歧义均明确拒绝。拒绝不刷新旧包有效期。规划异常或验收失败时继续执行
旧有效包，过期后回站立；恢复规划后经平滑任务重接和新包交接再进入正常跟踪。
该重接会暂时偏离原始任务，报告分别记录执行参考误差和原始任务误差。

WBC 无可行解时沿用已有终止策略，不执行下一 plant 步，不用全零力矩代替站立。
如果整回合没有一包 W 规划被接受，即使 WBC 一直稳定站立，也判定滚动验收失败。
每个新回合必须创建新的滚动控制器，避免带入旧状态、序号或热启动缓存。

## 本机复现

只读取已有 `.venv` 与隔离 native 栈，不安装或升级依赖。在仓库根目录运行：

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 \
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1 \
PYTHONPATH=.dev-tools/crocoddyl321-pin400/cmeel.prefix/lib/python3.11/site-packages:src \
  .venv/bin/python -B examples/go2_z1_rolling_cpu.py
```

支持 `--cases warm cold transient outage push high_low` 选择场景：冷/热启动配对、
一次异常和一次候选拒绝、2 s 持续失联后恢复、5 N/0.1 s 外力，以及原 12 s 高低任务。
故障通过示例中的显式测试包装器注入，成功候选均由真实 Crocoddyl 求解。
产物在 `artifacts/go2-z1-rolling-cpu/`：实际状态、力矩、参考和独立规划轨迹 NPZ，
每次请求/接收/收敛状态及时间的报告，逐步 QP 诊断和来源哈希。

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1 \
PYTHONPATH=.dev-tools/crocoddyl321-pin400/cmeel.prefix/lib/python3.11/site-packages:src \
  .venv/bin/python -B -m pytest -p no:cacheprovider -q \
  tests/test_rolling_cpu.py tests/test_stance_demo.py tests/test_guidance.py \
  tests/test_whole_body_qp.py tests/test_standing_whole_body.py \
  tests/test_go2_tensor_adapter.py tests/test_walking_whole_body_demo.py
```

## 本机验收结果

六组全部通过 CPU 验收，共 6600 个实际力矩控制步、240 次规划请求。
218 个包被接收，22 次失败全部来自显式注入；没有额外规划失败或 WBC 终止。

| 场景 | 接包/请求 | 执行 TCP RMSE / 峰值（mm） | 结果 |
| --- | --- | --- | --- |
| 热启动 | 20/20 | 0.0325 / 0.1110 | 19 次移位热启动，连续换包通过 |
| 冷启动 | 20/20 | 0.0325 / 0.1110 | 每次重新用共同参考初始化 |
| 短暂失败 | 18/20 | 0.0328 / 0.1110 | 1 次异常、1 次候选拒绝；旧包维持有效，后续换包恢复 |
| 持续失联 | 20/40 | 0.0329 / 0.1169 | 2 s 失联，经历旧包、回站立、站立保持、任务重接；过期缓存丢弃 |
| 5 N 外力 | 20/20 | 0.0349 / 0.1110 | 0.1 s 躯干外力，反馈执行通过 |
| 高低目标 | 120/120 | 0.0482 / 0.0966 | 120 次均收敛且通过独立门槛，完整 13.5 s / 2700 步 |

表中误差相对**正在执行的参考**。持续失联时，机器人按约定回站立，因此相对
**原始任务**的 TCP 峰值偏差达到 22.35 mm（RMSE 9.88 mm）；该偏差单列，
不把成功回退算作失联期间仍在跟踪原目标。最大足漂移为高低场景的约 0.00409 mm，
各场景力矩、关节位置/速度和接触力约束违反量均为零。

冷/热启动配对实际状态各分量最大差约 `1.42e-7`；热启动总迭代 50 次、冷启动 46 次。
本轮没有显示热启动加速，移位机制的正确性与性能收益分别记录。
回合后逐条核对全部 240 次实测初值/上一实际 WBC 力矩，以及 218 条已接收计划的
首状态，均与独立 CPU 轨迹对应记录完全一致。

高低场景规划完整路径耗时 p50/p95 为 123/189 ms，120 次均超过名义 100 ms 预算；
控制整周期（含遇到重规划时的同步求解）p50/p95 为 12.5/31.6 ms。
当前既未达到 10 Hz 规划墙钟预算，也未达到 200 Hz WBC/plant 预算。
这些是本次桌面 CPU 运行的实测时间，不是专用实时主机或隔离性能基准。

相关回归 **105 passed / 1 skipped**，其中 21 项新测试覆盖绝对时间预览、流形移位、
实测初值和实际力矩记忆、冷/热启动闭环、C2 换包及被打断的交接、过期/乱序/未来包、
跨包姿态歧义、失败缓存保留及完全无 W 包时的失败判定。CUDA stream 测试因本机无
CUDA 跳过。Ruff、格式检查和 `git diff --check` 通过。

以 `9fa2671` 为基线，129 个既有代码/配置/测试/示例文件中仅 4 个带臂路径模块按计划
扩展，其余 125 个文件与 88 个依赖元数据文件哈希保持不变。未安装或升级依赖，
未修改单 Go2 训练入口或配置，也未操作集成仓库和训练服务器。
证据见 [汇总及逐次规划记录](benchmarks/go2_z1_rolling_cpu_2026-11-13/summary.json)、
[反馈核对](benchmarks/go2_z1_rolling_cpu_2026-11-13/feedback_audit.json) 和
[隔离记录](benchmarks/go2_z1_rolling_cpu_2026-11-13/preservation.json)。

## 当前边界

本轮限定为相同刚体模型内的固定支撑 CPU 闭环。C0/C1、带臂 Tensor adapter、
Isaac Lab/PhysX、碰撞和带载验收、接触切换、行走 WBC 及实机均不在本轮实现。
后续已新增独立的 [后台异步与墙钟 Guidance 路径](go2_z1_async_cpu.md)，本页同步入口
保持不变。问题构造/求解复用与实时性能优化仍待完成。
连续包的平滑性不等同于全时域可行性证明，实际执行仍由 WBC 约束和独立 plant 验收。
