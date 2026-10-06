# 开发计划与验收条件

## M0：契约与基线（本次）

- [x] Python 包、统一 problem/result/backend/controller、可选依赖。
- [x] Torch CPU/CUDA affine LQR、逐环境失败状态和重置。
- [x] 稠密最优解、闭环、CPU/CUDA 和 stream 测试。
- [x] Crocoddyl CPU 参考适配器、对照测试及 CI 任务。
- [x] Tensor bridge、Isaac Lab 接入说明、可复现基准入口。
- [x] 在本机 croco_env 执行参考对照并记录版本与误差（Crocoddyl 3.0.1）。

M0 不等于通用 GPU Crocoddyl 完成；实际执行结果见 validation.md。

## M1：模型抽象与 CPU 非线性 MPC

- [x] 定义批量 dynamics/cost/derivatives 和 state `integrate/diff` 接口。
- [x] 实现 Pendulum 模型，并用有限差分验证 CPU/CUDA 导数。
- [x] 实现 Torch DDP 的 backward/forward pass 原型。
- [x] 修正逐环境收敛、失败和线搜索状态机。
- [x] 验证 DDP 在线性问题上与 LQR 后端一致。
- [x] CPU Pendulum 持续闭环及局部扰动恢复（DDP/FDDP，独立物理 plant）。
- [x] Crocoddyl 非线性 FDDP 数值对照（摆 action、反馈增益、轨迹和迭代预算）。
- [ ] 修正重力符号后的 CUDA 闭环验收。
- [x] 分离全局模型参数与逐环境随机化参数。

当前状态：**M1 正确性基线已冻结**。DDP 数值验收见 `ddp_acceptance.md`；未安装
Crocoddyl 的环境仍会明确跳过外部 oracle。

验收：导数通过有限差分；相同模型、初值、horizon 和容差下对齐轨迹、代价、反馈策略；
覆盖不可行初始轨迹、非正定 Hessian 和预算耗尽。

## M2：批量非线性求解

当前状态：**M2.7 Go2 显式接触切换与局部约束 MPC**。

- [x] backward/forward、逐环境线搜索/自适应正则化、固定迭代预算、失败和冻结 mask。
- [x] controller horizon-shift warm start；局部 reset 同时清除动作与轨迹缓存。
- [x] 分离共享模型与逐环境参数；按 mask 更新物理参数、参考和权重并同步失效缓存。
- [x] FDDP 动态 gap、modified Riccati sweep、逐环境 line search 和可行性语义。
- [x] 建立非线性 FDDP 外部 oracle 测试：相同摆模型、反馈增益、轨迹与预算序列。
- [x] 在本机 croco_env 执行 oracle 并记录版本、数值误差及迭代行为（见 validation.md）。

控制 box 约束需要独立 QP/Box-DDP，最终动作裁剪不等价于受约束最优解。

验收：批量元素与独立求解一致，局部重置不污染其他环境；参考/模型更新不读取陈旧缓存；
CPU/GPU 在明确容差内一致。

## 浮基与足式接触推进

- [x] 四元数 SO(3)、SE(3) 和浮基状态；分离 nx/ndx。
- [x] 切空间 dynamics/cost 契约、自动局部导数基线和流形 residual 代价。
- [x] FDDP 非零 gap 的输出 chart Jacobian；几何与 Pinocchio/Crocoddyl 对照。
- [x] 非欧氏求解、warm start、reset、异常四元数隔离与旧欧氏模型回归。
- [ ] 在 CUDA 环境执行浮基/切空间回归与性能测量。
- [x] 选定宇树 Go2，固定官方 URDF、驱动顺序和足端 frame。
- [x] Go2 CPU 固定接触 KKT 与 Crocoddyl 加速度/接触力/导数 oracle。
- [x] Go2 Torch 批量固定接触离散动力学，并核对积分和导数（CPU）。
- [x] 固定支撑站立 MPC，覆盖扰动恢复与接触残差（独立 CPU plant）。
- [ ] Go2 在 CUDA 实机执行接触/导数/闭环测试与性能测量。
- [x] 固定工作点局部 MPC 的扭矩、法向力与内接摩擦棱锥硬约束；完整非线性可行性检查。
- [x] CPU B=2/T=2、200 周期、20 ms 的预热后测量（不含传感器/plant/IO）。
- [x] 四足固定槽的逐环境支撑掩码，非支撑力为零；腾空动力学与独立 CPU 对照。
- [x] 显式释放/触地事件、质量度量塑性碰撞、单边/摩擦冲量检查及能量门禁。
- [x] 预构建模式控制器、逐环境切换与缓存清理；摆腿姿态目标及独立 plant 切换闭环。
- [ ] 混合模式/切换周期的 20 ms 频率验收，批量模式融合与消除重复候选求解。
- [ ] 在线重新线性化的一般约束非线性 MPC、跨预测节点接触序列、步态/摆腿轨迹规划及离地互补。

已提供 Go2 Torch 批量刚体动力学和四足固定点接触站立 MPC 正确性基线。
局部约束控制器提供平地各支撑阶段的硬接触不等式门禁；新接口支持外部显式接触切换。
一般足式行走、更多 batch/horizon
的延迟分布及硬实时系统保证仍待验证；完整 DDP/FDDP 路径仍为无约束正确性基线。
接口及验收边界见 `floating_base.md` 与 `go2.md`。

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

使用 Python/Torch 建立正确性基线，Crocoddyl 保持可选参考依赖。首个机器人已选定宇树 Go2，
已完成固定支撑站立与显式接触事件，下一步推进接触序列、摆腿轨迹和混合模式性能。
仍需确定 Isaac Lab 版本、任务接口及是否需要可微 MPC。
安装优先兼容模拟器配套环境，不强行升级 Isaac Sim。
