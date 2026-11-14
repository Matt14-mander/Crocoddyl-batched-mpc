# Go2–Z1 异步 CPU 耗时与延迟验收归档

2026-11-14，开发基线 `e9a9b5e`。本轮不改变单 Go2 训练与依赖，不放宽控制增益或
验收门槛。实现与完整复现命令见 [说明](../../go2_z1_async_jitter.md)。
提交前核对了当前源码与最终三批实验的源文件哈希一致；既有失败与复测内容保持原样。

| 文件 | 证据与结论 |
| --- | --- |
| [baseline_profile.json](baseline_profile.json)、[baseline_profile_run.json](baseline_profile_run.json) | 修改前 cProfile 定位；含观察器开销，不是实时验收 |
| [microbenchmark.json](microbenchmark.json) | 同进程交替微基准：KKT 单步约 2.09 倍、诊断观测约 6.23 倍 |
| [first_seven_cases.json](first_seven_cases.json)、[日志](first_seven_cases.log) | 本轮增加线程遥测前的 6/7 批次；过期回合因约 474 ms 状态滞后触发补算保护，失败状态保留 |
| [seven_cases.json](seven_cases.json)、[日志](seven_cases.log) | 最终代码七场景 7/7，86 次请求、2706 次控制；仍有调度漏槽，不是硬实时验收 |
| [wall_delays.json](wall_delays.json)、[日志](wall_delays.log) | 真实后台进程和前台 sleep：6/6 完整跟踪，2/2 预期超限停止；原始闭环失败标志在停止例中仍保留 |
| [virtual_delays.json](virtual_delays.json)、[日志](virtual_delays.log) | 实际 W/WBC/plant 加规定计算时间：六个跟踪回合、两个预期终止；重复轨迹完全一致，不作为并发/速度证据 |
| [feedback_audit.json](feedback_audit.json) | 最终三批共 213 次请求、6014 次 WBC 输入与实际状态/力矩逐条核对；原门槛和增益一致 |
| [offline_diagnostics_audit.json](offline_diagnostics_audit.json) | 用独立原始 FK 对全部控制采样重新计算 TCP/姿态误差，确认离线诊断未遗漏样本 |
| [tests.log](tests.log) | 相关回归 139 passed / 1 skipped（无 CUDA），含 9 项新增测试 |
| [preservation.json](preservation.json) | 136 个既有文件中仅 3 个异步相关文件扩展，其余 133 个、88 个依赖元数据及 9 个历史证据文件不变 |

`scenario_passed` 与 `passed` 不同：前者允许 overrun 场景按预期停止；该场景的
`passed=false` 和终止原因完整保留。历史姿态超限与复测记录仍在
[2026-11-13 目录](../go2_z1_async_cpu_2026-11-13/README.md)，不覆盖。

完整 NPZ 和逐次 QP 日志位于本地 `artifacts/go2-z1-jitter-seven-cases/`、
`artifacts/go2-z1-jitter-wall/`、`artifacts/go2-z1-jitter-virtual-final/`；
本轮首批失败轨迹在 `artifacts/go2-z1-jitter-seven-first/`。
这里保存摘要、日志与校验值，二进制轨迹沿用忽略规则留在本地。
