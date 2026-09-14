# 开发计划与验收条件

## M0：契约与基线（本次）

- [x] Python 包、统一 problem/result/backend/controller、可选依赖。
- [x] Torch CPU/CUDA affine LQR、逐环境失败状态和重置。
- [x] 稠密最优解、闭环、CPU/CUDA 和 stream 测试。
- [x] Crocoddyl CPU 参考适配器、对照测试及 CI 任务。
- [x] Tensor bridge、Isaac Lab 接入说明、可复现基准入口。
- [ ] 在安装 Crocoddyl 的环境执行参考对照并记录版本与误差。

M0 不等于通用 GPU Crocoddyl 完成；实际执行结果见 validation.md。

## M1：模型抽象与 CPU 非线性 MPC

- [x] 定义批量 dynamics/cost/derivatives 和 state `integrate/diff` 接口。
- [x] 实现 Pendulum 模型，并用有限差分验证 CPU/CUDA 导数。
- [x] 实现 Torch DDP 的 backward/forward pass 原型。
- [x] 修正逐环境收敛、失败和线搜索状态机。
- [x] 验证 DDP 在线性问题上与 LQR 后端一致。
- [ ] 完成稳定的 Pendulum 闭环及 Crocoddyl 数值对照。
- [ ] 分离全局模型参数与逐环境随机化参数。

当前状态：**M1 正确性基线已冻结**。DDP 数值验收见 `ddp_acceptance.md`；未安装
Crocoddyl 的环境仍会明确跳过外部 oracle。

验收：导数通过有限差分；相同模型、初值、horizon 和容差下对齐轨迹、代价、反馈策略；
覆盖不可行初始轨迹、非正定 Hessian 和预算耗尽。

## M2：批量非线性求解

当前状态：**M2.2 FDDP gap handling**。

- [x] backward/forward、逐环境线搜索/自适应正则化、固定迭代预算、失败和冻结 mask。
- [x] controller horizon-shift warm start；局部 reset 同时清除动作与轨迹缓存。
- [x] 分离共享模型与逐环境参数；按 mask 更新物理参数、参考和权重并同步失效缓存。
- [x] FDDP 动态 gap、modified Riccati sweep、逐环境 line search 和可行性语义。
- [ ] 在 Crocoddyl 环境执行非线性 FDDP 轨迹、反馈增益和迭代行为对照。

控制 box 约束需要独立 QP/Box-DDP，最终动作裁剪不等价于受约束最优解。

验收：批量元素与独立求解一致，局部重置不污染其他环境；参考/模型更新不读取陈旧缓存；
CPU/GPU 在明确容差内一致。

## M3：原生 C++/CUDA 与性能

先 profile 模型导数、线性代数、Python 调度、分配和同步，再确定内核边界。
通过 Torch dispatcher 注册 CPU/CUDA 算子，遵守 device guard/current stream；
规划 workspace/静态尺寸，之后验证 CUDA Graph。

验收：B=1/32/256/1024/4096、T=10/20/40 和多个 nx/nu/dtype 的延迟分布、吞吐、显存、误差；
记录 CPU 线程数、GPU/驱动及软件版本。保持稠密 oracle 和 Crocoddyl 对照，不预先承诺加速倍数。

## M4：Isaac Lab 真实任务与 RL

先集成 DirectRLEnv，再封装 ManagerBased ActionTerm。明确状态顺序、单位、局部坐标、
控制通道、模型 dt、policy decimation 和 MPC 频率。依次评估 MPC-only、residual RL 和参考策略。

验收：真实 Isaac Sim 上数百环境闭环，reset/timeout 正确；profile 确认无热路径 host round-trip；
记录 episode return、失败率、控制耗时与整体训练吞吐。浮基/接触模型单独核对流形和动力学约定。

## 近期决策

使用 Python/Torch 建立正确性基线，Crocoddyl 保持可选参考依赖。M1/M4 前确定首个机器人/任务、
Isaac Lab 版本和是否需要可微 MPC。安装优先兼容模拟器配套环境，不强行升级 Isaac Sim。
