# Go2–Z1 后台 W 规划与墙钟 Guidance 生命周期

2026-11-13。在 `0318ee8` 同步滚动基线上新增独立进程规划和单调墙钟执行路径。
原单包、同步滚动、Meshcat 入口和现有单 Go2 训练代码均保留。

## 进程与时间契约

`controllers/async_whole_body.py` 提供 `SpawnWWorker`、`WWorkerEngine` 和
`AsyncWReceiver`。后台用 `multiprocessing` 的 spawn 模式创建自己的 Go2–Z1 模型和
Crocoddyl 规划器，前台独占 Guidance buffer、WBC 和独立 CPU plant。
两边不共享可变的 Pinocchio/Crocoddyl 对象，也不在前台等待求解 future 或工作进程退出。
初次启动和故障重启均由管理线程完成；spawn 的参数传输与子进程导入不会阻塞控制线程。

所有任务时刻映射自同一个 `time.monotonic()` 时钟；UTC/系统日期调整不参与控制。
每次请求记录状态采样时刻、发送时刻、实际状态、上一实际 WBC 力矩、正在执行的参考、
回退状态、worker epoch 与递增 sequence。结果另含后台开始/完成时刻，前台在反序列化
完成后记录接收时刻。计划时间轴仍从**状态采样时刻**开始，接收时直接采样已过去相应
时间的新包；不把旧预测整体平移到接收时刻，也不重置有效期。

只允许一条在途请求，队列容量有界。忙时跳过请求，不堆积旧状态。前台接收后依次检查：

1. epoch、sequence 及采样/发送/开始/完成/接收时序是否匹配当前请求。
2. 接收时间是否超过请求超时或最大状态年龄，即使后台早已计算完成也不能忽略交付延迟。
3. 包序号、起始时间、issued_at、模型与足锚点、有效性和时域是否匹配。
4. 包是否已经过期，或剩余时间不足以完成 Guidance 平滑交接。

不合格结果不会替换旧包，也不会延长旧包有效期。成功交接继续使用既有 C2 位姿/导数
桥接和 C0 力参考。前台只把**实际接收成功**的序号作为下一请求的 ACK；后台仅据此
提升候选为热启动缓存，迟到或被拒绝的结果不会污染后续移位初值。

## 超时、过期和恢复

| 配置 | 默认值 |
| --- | --- |
| 规划时域 | 24 × 0.05 s = 1.2 s |
| 请求周期 | 0.5 s（名义 2 Hz）；后台忙时不排队 |
| 接收超时 / 最大状态年龄 | 0.6 / 0.65 s |
| 后台硬超时 | 2 s |
| Guidance 交接 / 回站立 / 任务重接 | 0.4 / 1.0 / 0.8 s |
| 名义 WBC 周期 / 最大积分子步 | 0.02 / 0.005 s |
| plant 最大补算时间 / 命令最大年龄 | 0.2 / 0.08 s |
| 任务末尾继续执行 | 2.5 s，用于观察到期回退和稳定 |

普通求解失败或软超时期间，前台继续执行旧有效包；旧包到期后按墙钟进入回站立，
再保持站立。软超时不启动第二个并发求解；原工作未结束前继续非阻塞轮询，迟到结果
明确丢弃。超过硬超时则终止该进程，确认退出后以新 epoch 后台重启，期间 WBC 继续运行。
旧 epoch 的结果不能被新进程回合接收，新进程也不继承旧进程的热启动缓存。

恢复规划从当时实际状态和回退参考出发，沿有限时长的恢复参考重接共同任务。
恢复桥接在已接收的计划之间保留，仍须通过 W 的独立动力学和任务门槛。
WBC 无可行解或 CPU 动力学异常时终止回合；没有全零力矩替代站立的策略。
进程清理和有界 join 在控制回合结束后执行。

## 墙钟 CPU 执行的含义

`controllers/async_cpu.py` 的 `run_async_cpu_episode` 在后台就绪且 WBC 完成静态预热后
启动任务时钟，预热时间单列。之后不因 W 求解暂停时钟：每次按实际经过的时间，用
不大于 5 ms 的子步推进独立接触动力学；在 WBC 计算期间保持上一条实际力矩。
超过名义控制周期会跳过已经错过的控制槽，不压缩时间或补发一串过时的控制请求。
IPC 接收与验收后再次推进 plant，WBC 使用与采样参考同时刻的实际状态；规划请求
仍保留先前的状态快照。两套状态均记录，可逐条与积分轨迹核对。

WBC 计算完成后，先积分旧力矩在计算期间的作用，再应用新命令；过龄或在计算中
对应 Guidance 已过期的命令被丢弃。数值积分自身的耗时会产生小的状态滞后，报告
`max_numerical_state_lag_s`；下一次继续补算已知力矩历史，绝不写入规划/IK 状态。
若补算跨度超过上限，明确终止，而非偷偷截断 elapsed time。

这是受实际墙钟驱动的 CPU 模型实验，仍不是硬实时系统或 PhysX 验证。
WBC 的 20 ms 设置是预测约束的名义控制步长；实际控制间隔、错过的槽数、IPC 开销、
包年龄和完整 CPU 约束结果全部单列，不宣称已达到 50 Hz 或 200 Hz 实时频率。

## 复现

使用现有独立 native 环境，不安装或升级任何依赖。在仓库根目录执行：

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 \
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1 \
PYTHONPATH=.dev-tools/crocoddyl321-pin400/cmeel.prefix/lib/python3.11/site-packages:src \
  .venv/bin/python -B examples/go2_z1_async_cpu.py
```

默认执行 `normal delay failure timeout expired restart high_low` 七组场景。
故障由显式 `WorkerFault` 配置注入：真实后台 sleep 造成交付延迟，后台异常造成失败，
长时间阻塞触发实际进程终止/重启；成功候选均经过真实 Crocoddyl 求解。
完整状态、力矩、请求快照、时间戳、接包事件与 QP 诊断位于
`artifacts/go2-z1-async-cpu/`。每次墙钟运行的包数量和耗时会随主机负载变化。

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1 \
PYTHONPATH=.dev-tools/crocoddyl321-pin400/cmeel.prefix/lib/python3.11/site-packages:src \
  .venv/bin/python -B -m pytest -p no:cacheprovider -q \
  tests/test_async_cpu.py tests/test_rolling_cpu.py tests/test_stance_demo.py \
  tests/test_guidance.py tests/test_whole_body_qp.py tests/test_standing_whole_body.py \
  tests/test_go2_tensor_adapter.py tests/test_walking_whole_body_demo.py
```

## 本机验收结果与未通过项

在 `0318ee8` 基线上，Python 3.11.4、NumPy 1.26.4、Torch 2.2.2、
Pinocchio 4.0.0、Crocoddyl 3.2.1；沿用现有独立环境。完整七场景批次共
84 次请求、77 包接收、1513 次 WBC 更新，全部生命周期断言通过。
**CPU 误差及约束门槛为 6/7 通过，过期场景姿态误差未通过；不能记为全批次通过。**

| 场景 | 接收 / 请求 | TCP 峰值 mm | 姿态峰值 rad | CPU 门槛与生命周期结果 |
| --- | --- | --- | --- | --- |
| normal | 12 / 12 | 0.0597 | 0.000170 | 通过，连续换包 |
| delay | 12 / 12 | 1.6853 | 0.007317 | 通过，额外 0.15 s 延迟仍在有效期内接收 |
| failure | 8 / 12 | 1.1914 | 0.000571 | 通过，4 次注入失败、旧包沿用、回退与恢复 |
| timeout | 9 / 10 | 2.2395 | 0.001268 | 通过，0.6 s 超时，迟到结果拒收后恢复 |
| expired | 8 / 9 | 3.5834 | **0.020445** | **姿态未通过**（上限 0.02）；过期拒包、回退与恢复通过 |
| restart | 5 / 6 | 0.2397 | 0.000187 | 通过，硬超时终止、换 epoch、后台重启与恢复 |
| high_low | 23 / 23 | 1.2146 | 0.011270 | 通过，12 s 高低任务与末尾 2.5 s 回退 |

小幅任务为 6 s，另执行 2.5 s 观察回退。`expired` 专门将接收超时/状态年龄放宽至
2 s、硬超时至 2.5 s，并注入 1.35 s 后台延迟，使结果因 1.2 s 预测时域耗尽而拒收，
不是因默认软超时先被丢弃。`restart` 使用 0.5/0.9 s 软/硬超时和 5 s 注入阻塞。
所有成功包均真实求解；没有用故障标签代替运行和验收。

过期场景首次运行出现约 282 ms 的相邻控制间隔、一次过龄命令丢弃，姿态超限发生于
这个运行中。相同源文件哈希的独立复测通过（8/10 接收，TCP 峰值 2.3780 mm、
姿态 0.001515 rad），其中还出现一次非注入的 W 可行性/任务门槛拒绝，随后正常恢复。
复测不能抹除首次失败；目前证据支持生命周期功能，但不足以证明对主机调度抖动
稳定或满足硬实时要求。首次记录与复测分别保留。

逐条核对的 84 次请求状态、上一实际力矩、1513 次 WBC 输入状态均与独立积分轨迹
完全一致；接收计划的初值/力矩记忆误差为 0，有效期未因交付而平移。
延迟、超时、过期请求计算期间分别有 10、34、37 次前台控制更新；旧包均在有效期内
继续执行。硬超时终止至新后台 ready 约 2.53 s，期间有 63 次控制更新。
全部工作进程在回合后清理。完整轨迹、held torque、请求状态与 QP 诊断在本地 artifacts，
文档内保存摘要、哈希和核对结果。

七例实际控制间隔中位数为 39.0–45.3 ms，整周期中位数 26.7–31.1 ms；
20 ms 控制预算未通过。接收包年龄 p95 为 0.300–0.542 s，前台 IPC 最大约 29 ms。
失败回退期间相对原始任务的 TCP 峰值约 22.34 mm，与表中相对实际执行参考的误差
分开记录。力矩/关节/接触力约束违反为零，不代表姿态、实时性或物理模型均已验收。

相关测试 **130 passed / 1 skipped**，含 25 项新测试；跳过项为本机无 CUDA 的 adapter
stream 测试。确定性时钟测试覆盖接收截止边界、交付延迟、过期与覆盖不足、旧包沿用、
epoch/序号/时戳契约、单请求在途与硬超时恢复、ACK 热启动；管理线程测试验证启动
阻塞不会发生在控制线程。已有 132 个代码/配置/测试/示例文件和 88 个依赖元数据文件
哈希全部不变，旧单 Go2 路径未改动，也未安装或升级依赖。

开发初期的短交接和直接 spawn 曾分别导致恢复误差/约束超限、重启时控制线程约
2.17 s 阻塞，记录保存在 `initial_failures.json`。本轮改为管理线程启动，并依据测量
采用 0.4 s 换包、0.8 s 任务重接、0.5 s 请求周期与 1.2 s 时域，避免频繁打断尚未结束的
恢复交接。验收阈值保持不变；当前剩余的墙钟抖动问题继续作为未通过项。

证据：[完整批次](benchmarks/go2_z1_async_cpu_2026-11-13/summary.json)、
[过期复测](benchmarks/go2_z1_async_cpu_2026-11-13/expiry_recheck.json)、
[状态与时间核对](benchmarks/go2_z1_async_cpu_2026-11-13/feedback_audit.json)、
[隔离记录](benchmarks/go2_z1_async_cpu_2026-11-13/preservation.json)、
[早期失败记录](benchmarks/go2_z1_async_cpu_2026-11-13/initial_failures.json)。

## 后续范围

2026-11-14 已新增 [耗时剖析、异步 CPU 优化与可重复延迟扰动](go2_z1_async_jitter.md)。
本页 2026-11-13 数值和失败状态保留为历史基线；新结果另行归档。

求解器对象/问题复用、IPC 校验开销优化、调度抖动和硬实时能力仍需继续测量。
C0/C1、独立带臂 Tensor adapter 与 Isaac Lab 场景、碰撞/载荷/PhysX/实机验收及行走
WBC 仍是后续阶段；本轮不接入或改变现有单 Go2 训练生命周期。
