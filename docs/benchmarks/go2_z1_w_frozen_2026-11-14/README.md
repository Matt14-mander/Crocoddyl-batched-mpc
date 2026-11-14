# W 固定支撑 CPU 冻结验收归档

2026-11-14，优化前提交 `5a16a96510bb9dd126583725a626787a2bebc01e`。
[实现与复现](../../go2_z1_w_frozen_baseline.md)；冻结以 [manifest.json](manifest.json)
中的源码、模型、设置及证据哈希为准。本归档随 W 性能优化一起纳入 Git 提交，未创建
release tag；提交可通过清单的 Git 历史定位。单 Go2 训练代码和依赖环境未改动，
原有失败和复测归档完整保留。

| 文件 | 结果与范围 |
| --- | --- |
| [before.json](before.json) | 修改前额外保存的 8 组 W 请求；全部收敛。原始数组另存本地 artifacts |
| [paired.json](paired.json)、[日志](paired.log) | 与提交 `5a16a96` 的 planner 同进程配对：8/8，144 组数组、代价、迭代数、全部非计时验收指标完全一致；另含预热交替微基准 |
| [paired_trajectories.npz](paired_trajectories.npz) | 约 686 KiB 原始配对数组；键前缀 `0_`/`1_` 分别为优化前/本轮，校验值见 paired.json |
| [seven_cases.json](seven_cases.json)、[日志](seven_cases.log) | 真实 spawn 后台七场景 7/7；86 请求、79 包接收、2909 次控制更新 |
| [wall_delays.json](wall_delays.json)、[日志](wall_delays.log) | 真实墙钟每类两次：6/6 完整跟踪、2/2 按预期超限停止 |
| [virtual_delays.json](virtual_delays.json)、[日志](virtual_delays.log) | 确定性时间模型每类两次：6/6 完整跟踪、2/2 按预期停止；重复数组一致，且与上一轮八份确定性闭环轨迹完全一致；不能用于并发/实时速度结论 |
| [tests.log](tests.log) | 完整相关回归 175 passed / 1 CUDA skipped；本轮新增 36 项 |
| [tests_initial.log](tests_initial.log) | 最初工作区/异步专项 57 passed，保留执行记录 |
| [preservation.json](preservation.json) | 140 个既有文件中 4 个异步/W 相关文件扩展，其他 136 个不变；88 个依赖元数据、25 个历史证据文件不变；设置及原数学 action 的 AST 一致 |
| [verification.json](verification.json) | 当前源码与实验记录哈希核对；七场景全部设置/门槛与上轮一致；确定性回放的历史轨迹核对 |
| [manifest.json](manifest.json) | W 冻结清单：模型身份、环境版本、完整设置/门槛、相关源码/资产/依赖及证据哈希 |

## 七场景结果

| 场景 | 接收 / 请求 | TCP 峰值 mm | 姿态峰值 rad |
| --- | ---: | ---: | ---: |
| normal | 12 / 12 | 0.03836 | 0.0000802 |
| delay | 12 / 12 | 0.02703 | 0.0000752 |
| failure | 8 / 12 | 0.59685 | 0.0003377 |
| timeout | 9 / 10 | 0.73426 | 0.0002809 |
| expired | 9 / 10 | 0.56328 | 0.0002822 |
| restart | 6 / 7 | 0.20296 | 0.0001014 |
| high_low | 23 / 23 | 0.57309 | 0.0061692 |

普通场景交付年龄中位数约 182 ms，上轮约 246 ms；前台 IPC 最大约 9.27 ms，
上轮约 21–23 ms。不同批次主机调度不受控制，性能归因以配对微基准为主。
本批次每例仍错过 27–119 个名义控制槽，周期中位数约 12.40–14.13 ms。
`realtime_accepted=false` 保留；不视为 50 Hz 无漏槽或 200 Hz 验收。

## 重复墙钟扰动

| 场景 | 结果 | TCP 峰值范围 mm | 姿态峰值范围 rad |
| --- | --- | ---: | ---: |
| expired | 2/2 完整通过 | 0.6321–0.7608 | 0.000293–0.000297 |
| burst | 2/2 完整通过 | 1.2449–1.2736 | 0.005419–0.006901 |
| boundary | 2/2 完整通过 | 1.7977–2.2566 | 0.002064–0.002819 |
| overrun | 2/2 按预期停止 | 不计入跟踪通过 | 不计入跟踪通过 |

250 ms 注入越过原有 200 ms 补算界，两例在 `advance_sampled_state` 停止并清理 worker。
这些例子的 `passed=false`、`failures=["terminated"]` 和终止原因完整保留；
`scenario_passed=true` 仅表示测试预期的拒绝/终止正确，不能表述为闭环跟踪成功。
其余三类均检查实际过期拒收、回站立、恢复；burst/boundary 还检查过龄命令丢弃。
本轮没有非预期闭环失败，不覆盖或删除前两轮已有的失败记录。

历史姿态超限及复测见 [11-13 归档](../go2_z1_async_cpu_2026-11-13/README.md)；
历史 474 ms 状态滞后导致保护停止及复测见
[上一轮耗时/扰动归档](../go2_z1_async_jitter_2026-11-14/README.md)。

除体积较小的配对数组随冻结清单归档外，完整闭环 NPZ 和逐次 QP 日志保持本地忽略规则，
分别位于 `artifacts/go2-z1-w-freeze-seven/`、`go2-z1-w-freeze-wall/`、
`go2-z1-w-freeze-virtual/`（均位于 artifacts 下）。JSON 中保存各回合轨迹 SHA-256。
修改前独立快照在 `artifacts/go2-z1-w-freeze-before/`，首轮缓存对照在
`artifacts/go2-z1-w-freeze-after/`，最终配对在 `artifacts/go2-z1-w-freeze-paired/`。
