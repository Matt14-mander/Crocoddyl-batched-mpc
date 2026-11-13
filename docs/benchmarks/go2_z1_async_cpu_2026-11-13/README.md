# Go2–Z1 后台异步 CPU 验收归档

开发基线为 `0318ee8`。本目录随后台 W、墙钟 Guidance、相关测试和开发计划一起提交。
提交前已核对代码与运行记录中的 SHA-256 一致；没有重新运行并覆盖首次结果。

| 文件 | 内容与结论 |
| --- | --- |
| [summary.json](summary.json)、[closed_loop.log](closed_loop.log) | 当前代码完整七场景批次；生命周期断言全部通过，CPU 门槛 6/7 通过。过期场景姿态峰值 0.020445 rad 超过 0.02，顶层 `passed=false` 保留 |
| [expiry_recheck.json](expiry_recheck.json)、[expiry_recheck.log](expiry_recheck.log) | 相同代码的独立过期场景复测及轨迹核对；CPU 门槛通过，仍记录一次非注入 W 可行性拒绝及随后恢复 |
| [initial_failures.json](initial_failures.json) | 更早开发版本的失败摘要、参数及源文件哈希；包含过快交接和控制线程直接 spawn 的问题，不能当作当前代码结果 |
| [feedback_audit.json](feedback_audit.json) | 完整批次 84 次请求、1513 次 WBC 状态、力矩记忆与时间契约核对；其中 `passed` 仅指核对通过，CPU 门槛另见 `closed_loop_gates_passed` |
| [tests.log](tests.log) | 相关回归：130 passed / 1 skipped，跳过项为无 CUDA 的 adapter stream 测试 |
| [preservation.json](preservation.json) | 132 个既有代码等文件、88 个依赖元数据文件哈希及新源码哈希；既有文件全部不变 |

首次完整批次与复测分别保留，不合并为“7/7 全部通过”。墙钟控制预算与调度抖动
容忍仍未验收；复测成功不替代首次失败，也不代表 PhysX 或实机结果。

完整 NPZ 轨迹与逐步 QP 数据仍在被 Git 忽略的本地
`artifacts/go2-z1-async-cpu/`、`artifacts/go2-z1-async-cpu-expiry-recheck/`；
其轨迹哈希保存在上述记录中。本次提交保存日志、摘要和核对证据，不包含这些二进制轨迹。
复现入口、环境与默认参数见 [异步 CPU 说明](../../go2_z1_async_cpu.md)。

下一步先测量并减少构造/求解、IPC 与积分诊断开销，增加可重复的控制延迟扰动验收；
保持原门槛并保留所有失败回合。C0/C1、独立带臂仿真接口和物理验收按
[开发计划](../../standing_arm_phase_a_plan.md) 后续推进。
